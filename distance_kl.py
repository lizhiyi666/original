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
EPS = 1e-8


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
