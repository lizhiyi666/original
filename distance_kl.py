"""Train-only cumulative-route prior and a finite-candidate differentiable KL.

The forward MC paths use actual POIs; gradients are straight-through estimates.
References are process-cached by file content and bin count, never checkpointed.
"""
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import math
from pathlib import Path

import torch
import torch.nn.functional as F

DISTANCE_REVISION = 'distance-kl-v2'
DISTANCE_IMPLEMENTATIONS = {'legacy': 'distance-kl-legacy-v1', 'batched': 'distance-kl-batched-v1'}
EPS = 1e-8


def distance_metadata(backend='legacy'):
    if backend not in DISTANCE_IMPLEMENTATIONS:
        raise ValueError(f'Unknown distance backend: {backend}')
    return dict(distance_backend=backend, distance_implementation_version=DISTANCE_IMPLEMENTATIONS[backend])


def validate_distance_metadata(record, *, expected=None, allow_historical=False):
    """Historical unlabelled traces may be audited, but not silently resumed."""
    actual = distance_metadata(record.get('distance_backend', 'legacy'))
    version = record.get('distance_implementation_version')
    if version is None and allow_historical and 'distance_backend' not in record:
        version = DISTANCE_IMPLEMENTATIONS['legacy']
    if version != actual['distance_implementation_version']:
        raise RuntimeError('Distance implementation version mismatch or missing version')
    if expected is not None and actual != expected:
        raise RuntimeError('Distance backend/implementation mismatch')
    return actual


def distance_objective_class(backend='legacy'):
    distance_metadata(backend)  # Fail closed on misspelled/unsupported backends.
    return DistanceObjective if backend == 'legacy' else BatchedDistanceObjective


def distance_output_directory(data_directory, backend='legacy', output_tag=None):
    implementation = distance_metadata(backend)
    root = Path(data_directory)
    if backend == 'legacy':
        return root
    if not output_tag:
        raise ValueError('Batched distance sampling requires an explicit independent output_tag')
    from experiment_io import safe_tag
    return root / 'distance-backends' / implementation['distance_implementation_version'] / safe_tag(output_tag)


def haversine(left, right):
    """Broadcastable [..., latitude/longitude] coordinates, output kilometres."""
    left, right = torch.deg2rad(left), torch.deg2rad(right)
    delta = right - left
    a = (delta[..., 0] / 2).sin().square() + left[..., 0].cos() * right[..., 0].cos() * (delta[..., 1] / 2).sin().square()
    return 12742.0 * a.clamp(0, 1).sqrt().asin()


def soft_histogram(km, centers, bandwidth):
    weights = F.softmax(-0.5 * ((km.clamp_min(0).log1p()[..., None] - centers) / bandwidth).square(), dim=-1)
    histogram = weights.reshape(-1, centers.numel()).mean(0) + EPS
    return histogram / histogram.sum()


@dataclass(frozen=True)
class DistanceReference:
    poi_ids: torch.Tensor
    coordinates: torch.Tensor
    centers: torch.Tensor
    bandwidth: float
    target: torch.Tensor
    fingerprint: str
    route_count: int


def reference_from_data(raw, bins=32, fingerprint='in-memory'):
    if bins < 2:
        raise ValueError('Distance bins must be at least 2')
    mapping = {int(k): v for k, v in raw['poi_gps'].items()}
    ids = sorted(mapping)
    if not ids or ids[0] < 0:
        raise ValueError('Distance prior requires valid POI tokens')
    coords = torch.tensor([list(map(float, mapping[i].split(','))) if isinstance(mapping[i], str)
                           else mapping[i] for i in ids], dtype=torch.float64)
    if (coords.shape != (len(ids), 2) or not torch.isfinite(coords).all()
            or (coords[:, 0].abs() > 90).any() or (coords[:, 1].abs() > 180).any()):
        raise ValueError('Invalid training POI coordinates')
    lookup = {token: i for i, token in enumerate(ids)}
    distances = []
    for seq in raw['sequences']:
        tokens = [int(p) for p in seq['checkins']]
        if any(p not in lookup for p in tokens):
            raise ValueError('Training route contains an unmapped POI')
        if len(tokens) >= 2:
            positions = coords[[lookup[p] for p in tokens]]
            distances.append(haversine(positions[:-1], positions[1:]).sum())
    if not distances:
        raise ValueError('Distance prior requires training routes with at least two POIs')
    distances = torch.stack(distances)
    maximum = 1.25 * math.log1p(max(float(distances.max()), 1.0))
    centers = torch.linspace(0, maximum, bins, dtype=torch.float64)
    bandwidth = maximum / (bins - 1)
    return DistanceReference(torch.tensor(ids), coords, centers, bandwidth,
                             soft_histogram(distances, centers, bandwidth), fingerprint, len(distances))


@lru_cache(maxsize=8)
def _load_reference(path, fingerprint, bins):
    return reference_from_data(torch.load(path, map_location='cpu', weights_only=False), bins, fingerprint)


def load_distance_reference(train_path, bins=32):
    path = Path(train_path).resolve()
    if not path.name.endswith('_train.pkl'):
        raise ValueError('Distance reference must come from a *_train.pkl split, never test')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    fingerprint = hashlib.sha256(f'{DISTANCE_REVISION}:{digest.hexdigest()}:{bins}'.encode()).hexdigest()
    return _load_reference(str(path), fingerprint, bins)


def distance_generator(seed, device):
    digest = hashlib.sha256(f'{DISTANCE_REVISION}:{int(seed)}:distance'.encode()).digest()
    return torch.Generator(device=device).manual_seed(int.from_bytes(digest[:8], 'big') & ((1 << 63) - 1))


def attach_distance_reference(projector, datamodule, seed=0):
    if projector.projection_distance_kl_weight <= 0:
        return None
    path = Path(datamodule.root) / datamodule.name / f'{datamodule.name}_train.pkl'
    projector.distance_reference = load_distance_reference(path, projector.distance_bins)
    projector.distance_seed = int(seed)
    projector.distance_generator = None  # Created lazily on the actual logits device.
    return projector.distance_reference.fingerprint


class DistanceObjective:
    """Fixed candidates/geometry for one projection invocation, ragged POI rows."""
    def __init__(self, reference, y_model, poi_mask, active_rows, *, topk=32, paths=8, temperature=1.0):
        if poi_mask is None or tuple(poi_mask.shape) != tuple(y_model.shape[:2]):
            raise ValueError('Distance KL requires an aligned poi_mask')
        if topk < 1 or paths < 1 or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('Invalid distance sampling parameters')
        ids = reference.poi_ids.to(y_model.device)
        if ids.max() >= y_model.shape[-1] - 2:
            raise ValueError('Distance POI token mapping is incompatible with the model vocabulary')
        coords = reference.coordinates.to(device=y_model.device, dtype=y_model.dtype)
        self.centers = reference.centers.to(device=y_model.device, dtype=y_model.dtype)
        self.target = reference.target.to(device=y_model.device, dtype=y_model.dtype)
        self.bandwidth = reference.bandwidth
        self.paths, self.temperature = paths, temperature
        self.rows, coverage = [], []
        with torch.no_grad():
            for b in torch.where(active_rows & (poi_mask.bool().sum(1) >= 2))[0].tolist():
                positions = torch.where(poi_mask[b].bool())[0]
                probs = y_model[b, positions][:, ids].softmax(-1)
                values, candidates = probs.topk(min(topk, len(ids)), dim=-1)
                geometry = coords[candidates]
                edges = haversine(geometry[:-1, :, None, :], geometry[1:, None, :, :])
                self.rows.append((b, positions, ids[candidates], edges))
                coverage.append(values.sum(-1))
        self.coverage = torch.cat(coverage) if coverage else y_model.new_empty(0)
        if self.rows:
            # One packed gather in loss(): per-row indexing of the full y would
            # create B dense [B,L,V] scatter-backward operations on large batches.
            self.batch_indices = torch.cat([torch.full_like(ids, b) for b, _, ids, _ in self.rows])
            self.position_indices = torch.cat([positions[:, None].expand_as(ids) for _, positions, ids, _ in self.rows])
            self.candidate_ids = torch.cat([ids for _, _, ids, _ in self.rows])

    def noise(self, generator, stochastic=True):
        if not stochastic:
            return [None] * len(self.rows)
        result = []
        for _, _, ids, _ in self.rows:
            u = torch.rand((self.paths, *ids.shape), generator=generator, device=ids.device, dtype=self.centers.dtype)
            result.append(-torch.log(-torch.log(u.clamp(min=1e-8, max=1-1e-7))))
        return result

    def loss(self, y, noise):
        if not self.rows:
            return y.sum() * 0
        if len(noise) != len(self.rows):
            raise ValueError('Distance noise must cover every eligible row')
        packed_logits = y[self.batch_indices, self.position_indices, self.candidate_ids]
        lengths = []
        offset = 0
        for (_, positions, _, edges), gumbel in zip(self.rows, noise):
            logits = packed_logits[offset:offset + len(positions)]
            offset += len(positions)
            if gumbel is None:
                weights = logits.softmax(-1).unsqueeze(0)
            else:
                soft = ((logits.unsqueeze(0) + gumbel) / self.temperature).softmax(-1)
                hard = F.one_hot(soft.argmax(-1), soft.shape[-1]).to(soft.dtype)
                weights = hard - soft.detach() + soft
            # Forward: exact distance between selected POIs. Backward: ST bilinear surrogate.
            route = torch.einsum('mli,lij,mlj->m', weights[:, :-1], edges, weights[:, 1:])
            lengths.append(route)
        histogram = soft_histogram(torch.stack(lengths), self.centers, self.bandwidth)
        return (self.target * (self.target.log() - histogram.log())).sum()


class BatchedDistanceObjective(DistanceObjective):
    """Same finite-candidate objective; padded rows and edges, no inner row loop.

    Candidate selection/geometry and row-wise random draws deliberately reuse the
    reference implementation. Only deterministic tensor algebra is batched.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.rows:
            return
        count = len(self.rows)
        length = max(len(row[1]) for row in self.rows)
        topk = self.rows[0][2].shape[-1]
        ids = self.rows[0][2]
        self.padded_batch = ids.new_zeros((count, 1, 1))
        self.padded_positions = ids.new_zeros((count, length, 1))
        self.padded_ids = ids.new_zeros((count, length, topk))
        self.valid_positions = torch.zeros((count, length), device=ids.device, dtype=torch.bool)
        self.valid_edges = torch.zeros((count, length - 1), device=ids.device, dtype=torch.bool)
        self.padded_edges = self.centers.new_zeros((count, length - 1, topk, topk))
        for i, (b, positions, candidates, edges) in enumerate(self.rows):
            n = len(positions)
            self.padded_batch[i] = b
            self.padded_positions[i, :n, 0] = positions
            self.padded_ids[i, :n] = candidates
            self.valid_positions[i, :n] = True
            self.valid_edges[i, :n-1] = True
            self.padded_edges[i, :n-1] = edges

    def noise(self, generator, stochastic=True):
        # Do NOT replace these draws by a single padded torch.rand: CUDA consumes
        # a different random stream for different launch shapes/call counts.
        ragged = super().noise(generator, stochastic)
        if not self.rows or not stochastic:
            return None
        count, length, topk = self.padded_ids.shape
        result = self.centers.new_zeros((count, self.paths, length, topk))
        for i, noise in enumerate(ragged):
            result[i, :, :noise.shape[1]] = noise
        return result

    def loss(self, y, noise):
        if not self.rows:
            return y.sum() * 0
        logits = y[self.padded_batch, self.padded_positions, self.padded_ids]
        if noise is None:
            weights = logits.softmax(-1).unsqueeze(1)
        else:
            expected = (len(self.rows), self.paths, *self.padded_ids.shape[1:])
            if tuple(noise.shape) != expected:
                raise ValueError('Distance noise must cover every eligible row')
            soft = ((logits.unsqueeze(1) + noise) / self.temperature).softmax(-1)
            hard = F.one_hot(soft.argmax(-1), soft.shape[-1]).to(soft.dtype)
            weights = hard - soft.detach() + soft
        weights = weights * self.valid_positions[:, None, :, None]
        # [row, edge, path, candidate] @ [row, edge, candidate, candidate].
        # Exact candidate-to-candidate distances, never mean-coordinate geometry.
        by_position = weights.transpose(1, 2)
        weighted_edges = torch.matmul(by_position[:, :-1], self.padded_edges)
        lengths = ((weighted_edges * by_position[:, 1:]).sum(-1)
                   * self.valid_edges[:, :, None]).sum(1)
        # There are no padded trajectories here: every eligible row gets exactly
        # the same number of paths and the same histogram weight as the reference.
        histogram = soft_histogram(lengths, self.centers, self.bandwidth)
        return (self.target * (self.target.log() - histogram.log())).sum()
