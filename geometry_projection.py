"""PCDG-Geo: train-only, category-preserving terminal POI refinement.

No test reference trajectories are accepted by the refinement API. Model scores
are detached, only candidate logits are optimized, and the off path draws no RNG.
"""
import copy
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

VERSION = 'same-category-geometry-v1'
EPS = 1e-8


def geometry_output_directory(base, mode, output_tag):
    if mode == 'off':
        return Path(base)
    if mode != 'same_category_v1' or not output_tag:
        raise ValueError('Enabled geometry needs its own explicit output tag')
    from experiment_io import safe_tag
    return Path(base) / 'geometry-refinement' / VERSION / safe_tag(output_tag)


@dataclass(frozen=True)
class GeometryConfig:
    geometry_refinement: str = 'off'
    geometry_steps: int = 50
    geometry_distance_weight: float = 1.
    geometry_radius_weight: float = 1.
    geometry_prior_weight: float = .01
    geometry_paths: int = 8
    geometry_topk: int = 32
    geometry_temperature: float = 1.
    geometry_learning_rate: float = .05
    geometry_noise_interval: int = 20

    def __post_init__(self):
        if self.geometry_refinement not in ('off', 'same_category_v1'):
            raise ValueError('Unknown geometry refinement')
        if self.geometry_steps not in (50, 100, 200):
            raise ValueError('Geometry steps must be a prequalified fixed budget')
        if (self.geometry_paths != 8 or self.geometry_topk != 32 or self.geometry_temperature != 1.
                or self.geometry_learning_rate != .05 or self.geometry_noise_interval != 20):
            raise ValueError('Geometry protocol constants may not change silently')
        if any(not math.isfinite(v) or v < 0 for v in (self.geometry_distance_weight,
                    self.geometry_radius_weight, self.geometry_prior_weight)):
            raise ValueError('Geometry weights must be finite and nonnegative')

    def metadata(self):
        return dict(asdict(self), geometry_implementation_version=VERSION)


def geometry_generator(seed, global_start, device):
    encoded = f'{VERSION}:{int(seed)}:{int(global_start)}'.encode()
    value = int.from_bytes(hashlib.sha256(encoded).digest()[:8], 'big') & ((1 << 63) - 1)
    return torch.Generator(device=device).manual_seed(value)


def safe_sqrt(value):
    # Zero subgradient at coincidence, without evaluating sqrt'(0).
    return torch.where(value > 0, value.clamp_min(torch.finfo(value.dtype).tiny).sqrt(),
                       torch.zeros_like(value))


def offset_distance(left, right, origin_latitude):
    """Haversine on centered radians: avoids loss of nearby-point precision in FP32."""
    delta = right - left
    a = ((delta[..., 0] / 2).sin().square()
         + (left[..., 0] + origin_latitude).cos() * (right[..., 0] + origin_latitude).cos()
         * (delta[..., 1] / 2).sin().square()).clamp(0, 1)
    root = safe_sqrt(a)
    surrogate = root.clamp(max=1-torch.finfo(root.dtype).eps).asin()
    with torch.no_grad():
        exact = a.sqrt().asin()
    return 12742. * (exact + (surrogate - surrogate.detach()))


def legacy_radius(points, mask, origin_latitude):
    """Exactly the historical sqrt(mean(distance-to-arithmetic-centroid)) observable."""
    count = mask.sum(-1).clamp_min(1)
    center = (points * mask[..., None]).sum(-2) / count[..., None]
    distances = offset_distance(points, center[..., None, :], origin_latitude)
    return safe_sqrt((distances * mask).sum(-1) / count)


def histogram_weights(values, centers, width):
    return (-.5 * ((values.clamp_min(0).log1p()[..., None]-centers)/width).square()).softmax(-1)


def normalize_histogram(histogram):
    histogram = histogram + EPS
    return histogram / histogram.sum(-1, keepdim=True)


def js_divergence(left, right):
    mid = .5 * (left + right)
    return .5 * ((left * (left.log()-mid.log())).sum(-1)
                 + (right * (right.log()-mid.log())).sum(-1))


class GeometryReference:
    def __init__(self, raw, fit_indices, fingerprint='in-memory'):
        self.fingerprint = fingerprint
        self.fit_indices = tuple(map(int, fit_indices))
        if len(set(self.fit_indices)) != len(self.fit_indices) or not self.fit_indices:
            raise ValueError('Geometry reference indices must be unique and nonempty')
        gps = {int(k): v for k, v in raw['poi_gps'].items()}
        cats = {int(k): int(v) for k, v in raw['poi_category'].items()}
        if set(gps) != set(cats) or min(gps, default=-1) < 0:
            raise ValueError('POI coordinate/category catalogs disagree')
        self.ids = torch.tensor(sorted(gps), dtype=torch.long)
        self.categories = torch.tensor([cats[i] for i in self.ids.tolist()])
        self.coordinates = torch.tensor([list(map(float, gps[i].split(','))) if isinstance(gps[i], str)
                                         else gps[i] for i in self.ids.tolist()], dtype=torch.float64)
        if (self.coordinates.shape != (len(gps), 2) or not torch.isfinite(self.coordinates).all()
                or (self.coordinates[:, 0].abs() > 90).any() or (self.coordinates[:, 1].abs() > 180).any()):
            raise ValueError('Invalid geometry reference coordinates')
        self.lookup = {token: i for i, token in enumerate(self.ids.tolist())}
        self.poi_category = cats
        self.category_counts = {c: int((self.categories == c).sum()) for c in set(cats.values())}
        radians = torch.deg2rad(self.coordinates)
        self.origin = radians.mean(0)
        self.offsets = radians - self.origin
        routes, radii = [], []
        for index in self.fit_indices:
            if not 0 <= index < len(raw['sequences']):
                raise ValueError('Invalid training reference index')
            ids = [self.lookup[int(i)] for i in raw['sequences'][index]['checkins']]
            if len(ids) < 2:
                continue
            points = self.offsets[ids]
            routes.append(offset_distance(points[:-1], points[1:], self.origin[0]).sum())
            radii.append(legacy_radius(points, torch.ones(len(ids), dtype=torch.bool), self.origin[0]))
        if not routes:
            raise ValueError('Geometry reference needs multipoint training trajectories')
        self.route_count = len(routes)
        self.histograms = {}
        for name, values in (('distance', torch.stack(routes)), ('radius', torch.stack(radii))):
            maximum = 1.25 * math.log1p(max(float(values.max()), 1.))
            centers = torch.linspace(0, maximum, 32, dtype=torch.float64)
            width = maximum / 31
            target = normalize_histogram(histogram_weights(values, centers, width).mean(0))
            self.histograms[name] = (centers, width, target)
        self._devices = {}

    def on_device(self, device):
        key = str(torch.device(device))
        if key not in self._devices:
            self._devices[key] = dict(ids=self.ids.to(device), categories=self.categories.to(device),
                offsets=self.offsets.to(device=device, dtype=torch.float32),
                origin_latitude=self.origin[0].to(device=device, dtype=torch.float32),
                histograms={name: (c.to(device=device, dtype=torch.float32), w,
                    target.to(device=device, dtype=torch.float32)) for name, (c, w, target) in self.histograms.items()})
        return self._devices[key]


@lru_cache(maxsize=8)
def _load_geometry_reference(path, content_sha256, fit_indices):
    raw = torch.load(path, map_location='cpu', weights_only=False)
    fp = hashlib.sha256(json.dumps([VERSION, content_sha256, list(fit_indices)], separators=(',', ':')).encode()).hexdigest()
    return GeometryReference(raw, fit_indices, fp)


def load_geometry_reference(path, fit_indices):
    from experiment_io import sha256_file
    path = Path(path).resolve()
    if not path.name.endswith('_train.pkl'):
        raise ValueError('Geometry reference must use the training split')
    return _load_geometry_reference(str(path), sha256_file(path), tuple(map(int, fit_indices)))


def generated_batch(records, device):
    """Re-encode cleaned generated records, never their paired test GPS/POIs."""
    from datamodule import Batch, Sequence
    sequences = [Sequence(time=np.asarray(r['arrival_times'], dtype=np.float32),
        checkins=np.asarray(r['checkins'], dtype=np.int64), category=np.asarray(r['marks'], dtype=np.int64),
        tmax=24., **{f'condition{i}{suffix}': np.asarray(r[f'condition{i}{suffix}'], dtype=np.int64)
                    for i in range(1, 7) for suffix in ('', '_indicator')}) for r in records]
    return Batch.from_sequence_list(sequences).to(device)


def untruncated_scores(model, records):
    from discrete_diffusion.diffusion_transformer import index_to_log_onehot
    if model.training:
        raise ValueError('Geometry scoring requires an evaluation-mode frozen model')
    device = next(model.parameters()).device
    batch = generated_batch(records, device)
    with torch.no_grad():
        embedding = model.condition_encoder(batch)
        logits = model.predict_start(index_to_log_onehot(batch.checkin_sequences, model.num_classes),
            embedding, torch.zeros(len(records), dtype=torch.long, device=device), batch)
    if logits.dtype != torch.float32 or not torch.isfinite(logits).all():
        raise FloatingPointError('Geometry scoring must return finite FP32 logits')
    return logits.detach()


class GeometryObjective:
    def __init__(self, reference, records, model_logits):
        if model_logits.dtype != torch.float32:
            raise ValueError('Geometry optimization is FP32 only')
        self.reference = reference
        self.rows = [i for i, r in enumerate(records) if len(r['checkins']) >= 2]
        self.data = reference.on_device(model_logits.device)
        self.lengths = [len(records[i]['checkins']) for i in self.rows]
        count, length = len(self.rows), max(self.lengths)
        device = model_logits.device
        self.mask = torch.arange(length, device=device)[None, :] < torch.tensor(self.lengths, device=device)[:, None]
        original = torch.zeros(count, length, dtype=torch.long, device=device)
        for row, index in enumerate(self.rows):
            tokens = [int(x) for x in records[index]['checkins']]
            mapped = [reference.lookup[token] for token in tokens]
            if not np.allclose(records[index]['gps'], reference.coordinates[mapped].numpy(), rtol=0, atol=1e-8):
                raise ValueError('Generated GPS does not match the immutable POI catalog')
            original[row, :len(mapped)] = torch.tensor(mapped, device=device)
        row_ids = torch.tensor(self.rows, device=device)[:, None, None]
        positions = torch.tensor(self.lengths, device=device)[:, None, None] + 2 + torch.arange(length, device=device)[None, :, None]
        scores = model_logits[row_ids, self.data['ids'][None, None, :], positions].detach()
        same_category = self.data['categories'][None, None, :] == self.data['categories'][original][..., None]
        rank = scores.masked_fill(~same_category, -torch.inf).argsort(dim=-1, descending=True, stable=True)
        original_points = self.data['offsets'][original]
        center = (original_points * self.mask[..., None]).sum(1) / self.mask.sum(1)[:, None]
        nearby = offset_distance(center[:, None, :], self.data['offsets'][None, :, :], self.data['origin_latitude'])
        nearest = nearby[:, None, :].expand_as(scores).masked_fill(~same_category, torch.inf).argsort(dim=-1, stable=True)
        priority = torch.empty_like(rank)
        indices = torch.arange(rank.shape[-1], device=device)[None, None, :].expand_as(rank)
        priority.scatter_(-1, rank, indices + 64)
        for chosen, start in ((rank[..., :15], 1), (nearest[..., :16], 16)):
            values = torch.arange(start, start + chosen.shape[-1], device=device).expand_as(chosen)
            priority.scatter_reduce_(-1, chosen, values, reduce='amin', include_self=True)
        priority.scatter_(-1, original[..., None], 0)
        self.candidates = priority.argsort(dim=-1, stable=True)[..., :min(32, rank.shape[-1])]
        self.valid = same_category.gather(-1, self.candidates)
        self.original = original
        candidate_scores = scores.gather(-1, self.candidates).masked_fill(~self.valid, -1e9)
        self.q = .5 * candidate_scores.softmax(-1)
        self.q[..., 0] += .5
        self.log_q = self.q.clamp_min(1e-30).log().detach()
        self.points = self.data['offsets'][self.candidates]
        self.edges = offset_distance(self.points[:, :-1, :, None, :], self.points[:, 1:, None, :, :],
                                     self.data['origin_latitude']).detach()
        self.edge_mask = self.mask[:, :-1] & self.mask[:, 1:]

    def noise(self, generator, paths=8):
        shape = (len(self.rows), paths, *self.candidates.shape[1:])
        uniform = torch.rand(shape, generator=generator, device=self.q.device, dtype=self.q.dtype)
        return -torch.log(-torch.log(uniform.clamp(min=1e-8, max=1-1e-7)))

    def soft(self, logits):
        return logits.masked_fill(~self.valid, -1e9).softmax(-1)

    def weights(self, logits, noise):
        soft = (logits[:, None] + noise).masked_fill(~self.valid[:, None], -1e9).softmax(-1)
        hard = F.one_hot(soft.argmax(-1), soft.shape[-1]).to(soft.dtype)
        return (hard - soft.detach() + soft) * self.mask[:, None, :, None]

    def observables(self, weights):
        by_position = weights.transpose(1, 2)
        distances = ((torch.matmul(by_position[:, :-1], self.edges) * by_position[:, 1:]).sum(-1)
                     * self.edge_mask[:, :, None]).sum(1)
        points = (weights[..., None] * self.points[:, None]).sum(-2)
        radii = legacy_radius(points, self.mask[:, None].expand(points.shape[:-1]), self.data['origin_latitude'])
        return distances, radii

    def geometric_losses(self, weights, separate_paths=False):
        values = self.observables(weights)
        losses = []
        for name, value in zip(('distance', 'radius'), values):
            centers, width, target = self.data['histograms'][name]
            bins = histogram_weights(value, centers, width)
            histogram = bins.mean(0) if separate_paths else bins.mean((0, 1))
            losses.append(js_divergence(normalize_histogram(histogram), target))
        return losses

    def loss(self, logits, noise, config):
        distance, radius = self.geometric_losses(self.weights(logits, noise))
        p = self.soft(logits)
        kl = (p * (p.clamp_min(1e-30).log() - self.log_q)).sum(-1)
        kl = (kl * self.mask).sum() / self.mask.sum()
        return config.geometry_distance_weight * distance + config.geometry_radius_weight * radius + config.geometry_prior_weight * kl

    def select(self, logits, noise, config):
        choices = (logits[:, None] + noise).masked_fill(~self.valid[:, None], -1e9).argmax(-1)
        choices = torch.cat((torch.zeros_like(choices[:, :1]), choices), 1)
        weights = F.one_hot(choices, self.q.shape[-1]).to(self.q.dtype) * self.mask[:, None, :, None]
        distance, radius = self.geometric_losses(weights, separate_paths=True)
        cost = -(weights * self.log_q[:, None]).sum((0, 2, 3)) / self.mask.sum()
        scores = config.geometry_distance_weight * distance + config.geometry_radius_weight * radius + config.geometry_prior_weight * cost
        if not torch.isfinite(scores).all():
            raise FloatingPointError('Non-finite discrete geometry score')
        winner = int(scores.argmin())  # Original batch is first: deterministic ties retain it.
        chosen = self.candidates.gather(-1, choices[:, winner, :, None]).squeeze(-1)
        return chosen, dict(selected_path=winner, original_objective=float(scores[0]),
            selected_objective=float(scores[winner]), distance_js_before=float(distance[0]),
            distance_js_after=float(distance[winner]), radius_js_before=float(radius[0]), radius_js_after=float(radius[winner]))


def assert_invariants(before, after, reference):
    if len(before) != len(after):
        raise RuntimeError('Geometry refinement changed batch size')
    for a, b in zip(before, after):
        if a.keys() != b.keys() or any(not np.array_equal(a[k], b[k]) for k in a if k not in ('checkins', 'gps')):
            raise RuntimeError('Geometry refinement changed a frozen field')
        ca = [reference.poi_category[int(i)] for i in a['checkins']]
        cb = [reference.poi_category[int(i)] for i in b['checkins']]
        if ca != cb:
            raise RuntimeError('Geometry refinement changed actual category sequence')
        expected = [reference.coordinates[reference.lookup[int(i)]].tolist() for i in b['checkins']]
        if not np.array_equal(np.asarray(b['gps']), np.asarray(expected)):
            raise RuntimeError('Geometry refinement produced non-catalog GPS')


def refine_records(records, model, reference, config=GeometryConfig(), *, seed=0, global_start=0,
                   projection_executed=True, scored_logits=None):
    started = time.perf_counter()
    stats = dict(config.metadata(), optimizer_steps=0, replaced_pois=0)
    if config.geometry_refinement == 'off' or not projection_executed or not records:
        return records, dict(stats, skipped=True)
    if reference is None:
        raise ValueError('Enabled geometry refinement needs a training reference')
    movable = any(len(r['checkins']) >= 2 and any(reference.category_counts[reference.poi_category[int(p)]] > 1
                  for p in r['checkins']) for r in records)
    if not movable:
        return records, dict(stats, skipped=True)
    logits = untruncated_scores(model, records) if scored_logits is None else scored_logits.detach()
    objective = GeometryObjective(reference, records, logits)
    rng = geometry_generator(seed, global_start, logits.device)
    parameters = objective.log_q.clone().requires_grad_()
    optimizer = torch.optim.Adam([parameters], lr=config.geometry_learning_rate)
    with torch.enable_grad():
        for step in range(config.geometry_steps):
            if step % config.geometry_noise_interval == 0:
                noise = objective.noise(rng, config.geometry_paths)
            optimizer.zero_grad(set_to_none=True)
            loss = objective.loss(parameters, noise, config)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite geometry objective')
            loss.backward()
            if parameters.grad is None:
                raise FloatingPointError('Missing geometry gradient')
            try:
                torch.nn.utils.clip_grad_norm_([parameters], 10., error_if_nonfinite=True)
            except RuntimeError as exc:
                raise FloatingPointError('Non-finite geometry gradient') from exc
            optimizer.step()
            if not torch.isfinite(parameters).all():
                raise FloatingPointError('Non-finite geometry logits')
    with torch.no_grad():
        chosen, diagnostics = objective.select(parameters, noise, config)
    chosen = chosen.cpu()
    result = copy.deepcopy(records)
    replaced = 0
    for row, index in enumerate(objective.rows):
        ids = chosen[row, :objective.lengths[row]]
        tokens = reference.ids[ids].numpy()
        replaced += int(np.count_nonzero(np.asarray(records[index]['checkins']) != tokens))
        result[index]['checkins'] = tokens.astype(np.asarray(records[index]['checkins']).dtype, copy=False)
        result[index]['gps'] = reference.coordinates[ids].tolist()
    assert_invariants(records, result, reference)
    stats.update(diagnostics, optimizer_steps=config.geometry_steps, replaced_pois=replaced,
                 geometry_reference_sha256=reference.fingerprint, elapsed_seconds=time.perf_counter()-started,
                 eligible_rows=len(objective.rows), candidate_count=int(objective.valid.sum()))
    return result, stats
