"""Default-off train-reference distribution balancing; no test targets or new base RNG.

The old geometry module and its public configuration remain unchanged. This
module owns a separately versioned objective and terminal refinement path.
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

from experiment_io import sha256_file
from geometry_projection import (GeometryConfig, GeometryObjective, GeometryReference,
    assert_invariants, histogram_weights, js_divergence, normalize_histogram,
    offset_distance, untruncated_scores)

VERSION = 'train-reference-distribution-balance-v1'
GEOMETRY_VERSION = 'same-category-distribution-geometry-v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class BalanceConfig:
    distribution_balance: str = 'off'
    constraint_scale: float = 1.
    category_distribution_weight: float = 0.
    poi_distribution_weight: float = 0.

    def __post_init__(self):
        if self.distribution_balance not in ('off', 'train_ref_v1'):
            raise ValueError('Unknown distribution balance mode')
        if (not math.isfinite(self.constraint_scale) or not 0 < self.constraint_scale <= 1
                or any(not math.isfinite(v) or v < 0 for v in
                       (self.category_distribution_weight, self.poi_distribution_weight))):
            raise ValueError('Invalid distribution balance coefficients')

    @property
    def enabled(self):
        return self.distribution_balance != 'off'

    def metadata(self):
        return dict(asdict(self), distribution_balance_version=VERSION,
                    distribution_geometry_version=GEOMETRY_VERSION)


class DistributionReference(GeometryReference):
    def __init__(self, raw, fit_indices, fingerprint='in-memory'):
        super().__init__(raw, fit_indices, fingerprint)
        self.semantic_ids = torch.tensor(sorted(set(self.poi_category.values())), dtype=torch.long)
        lookup = {c: i for i, c in enumerate(self.semantic_ids.tolist())}
        size = len(lookup)
        self.hour_counts = torch.zeros(24, size, dtype=torch.float64)
        self.transition_counts = torch.zeros(size, size, dtype=torch.float64)
        self.visit_counts = torch.zeros(len(self.ids), dtype=torch.float64)
        distinct = []
        for index in self.fit_indices:
            record = raw['sequences'][index]
            marks, times, pois = record['marks'], np.asarray(record['arrival_times']), record['checkins']
            if (len(marks) != len(times) or len(pois) != len(times) or times.ndim != 1
                    or not np.isfinite(times).all() or np.any(times < 0) or np.any(times >= 24)
                    or np.any(np.diff(times) < 0)):
                raise ValueError('Invalid aligned training category reference')
            if any(c not in lookup for c in marks):
                raise ValueError('Non-semantic category in training reference')
            categories = [lookup[int(c)] for c in marks]
            for hour, cat in zip(times.astype(int), categories):
                self.hour_counts[hour, cat] += 1
            for left, right in zip(categories[:-1], categories[1:]):
                self.transition_counts[left, right] += 1
            # Match evaluation-v2 eligibility for visitation and DailyLoc.
            if len(pois) > 1:
                for token in pois:
                    self.visit_counts[self.lookup[int(token)]] += 1
                distinct.append(len(set(map(int, pois))))
        if not distinct:
            raise ValueError('Distribution reference needs multipoint trajectories')
        values = torch.tensor(distinct, dtype=torch.float64)
        maximum = 1.25 * math.log1p(max(float(values.max()), 1.))
        centers = torch.linspace(0, maximum, 32, dtype=torch.float64)
        width = maximum / 31
        self.histograms['daily'] = (centers, width,
            normalize_histogram(histogram_weights(values, centers, width).mean(0)))

    def on_device(self, device):
        result = super().on_device(device)
        if 'semantic_ids' not in result:
            result.update(semantic_ids=self.semantic_ids.to(device),
                hour_counts=self.hour_counts.to(device=device, dtype=torch.float32),
                transition_counts=self.transition_counts.to(device=device, dtype=torch.float32),
                visit_counts=self.visit_counts.to(device=device, dtype=torch.float32))
        return result


@lru_cache(maxsize=8)
def _load_reference(path, file_hash, indices):
    raw = torch.load(path, map_location='cpu', weights_only=False)
    return DistributionReference(raw, indices, digest([VERSION, file_hash, list(indices)]))


def load_reference(path, indices):
    path = Path(path).resolve()
    if not path.name.endswith('_train.pkl'):
        raise ValueError('Distribution reference must be the training split')
    if any(isinstance(i, bool) or not isinstance(i, (int, np.integer)) for i in indices):
        raise ValueError('Reference indices must be integers')
    return _load_reference(str(path), sha256_file(path), tuple(map(int, indices)))


def load_reference_manifest(manifest_path, dataset, train_path):
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    split = manifest['splits'][dataset]
    pools = [set(split[k]) for k in ('screen', 'confirm', 'reference')]
    if any(pools[i] & pools[j] for i in range(3) for j in range(i)):
        raise ValueError('Training reference overlaps calibration pools')
    if sha256_file(train_path) != manifest['profiles'][dataset]['data_sha256']['train']:
        raise RuntimeError('Distribution training data fingerprint mismatch')
    return load_reference(train_path, split['reference'])


def aligned_category_times(batch):
    mask = batch.category_mask.bool()
    if not torch.equal(mask.sum(1), batch.mask.bool().sum(1)):
        raise ValueError('Category positions do not align with cached times')
    times = torch.zeros(mask.shape, device=batch.time.device, dtype=batch.time.dtype)
    times[mask] = batch.time[batch.mask.bool()]
    return times


class CategoryDistributionObjective:
    """Expected counts, no Monte Carlo draws. Input logits are [B,L,V]."""
    def __init__(self, reference, category_mask, category_times, active_rows, vocab_size):
        if category_times is None or category_times.shape != category_mask.shape:
            raise ValueError('Distribution projection needs aligned category times')
        data = reference.on_device(category_times.device)
        self.ids = data['semantic_ids']
        if int(self.ids.min()) < 0 or int(self.ids.max()) >= vocab_size:
            raise ValueError('Semantic reference IDs outside model vocabulary')
        self.mask = category_mask.bool() & active_rows[:, None]
        times = category_times[self.mask]
        if not torch.isfinite(times).all() or (times < 0).any() or (times >= 24).any():
            raise ValueError('Invalid cached category times')
        hours = category_times.masked_fill(~self.mask, 0).long()
        self.hours = F.one_hot(hours, 24).float() * self.mask[..., None]
        self.edges = self.mask[:, :-1] & self.mask[:, 1:]
        self.hour_support = data['hour_counts'].sum(-1) > 0
        self.transition_support = data['transition_counts'].sum(-1) > 0
        self.hour_target = normalize_histogram(data['hour_counts'])
        self.transition_target = normalize_histogram(data['transition_counts'])
        self.diagnostics = dict(category_reference_sha256=reference.fingerprint,
            unsupported_reference_hours=(~self.hour_support).nonzero().flatten().tolist(),
            unsupported_reference_sources=reference.semantic_ids[~self.transition_support.cpu()].tolist())

    def loss(self, logits):
        p = logits.index_select(-1, self.ids).softmax(-1)
        hours = torch.einsum('blh,blc->hc', self.hours, p)
        transitions = torch.einsum('blc,bld,bl->cd', p[:, :-1], p[:, 1:], self.edges.float())
        zero = p.sum() * 0
        # Retain the 24-hour denominator; unsupported reference hours contribute zero.
        hourly = (js_divergence(normalize_histogram(hours[self.hour_support]),
                              self.hour_target[self.hour_support]).sum() / 24
                  if self.hour_support.any() else zero)
        transition = (js_divergence(normalize_histogram(transitions[self.transition_support]),
                                   self.transition_target[self.transition_support]).mean()
                      if self.transition_support.any() else zero)
        return .5 * (hourly + transition)


def top_visit_jsd(counts, reference, top=1000):
    """Identity-aligned top-k union, not a sorted-frequency-shape comparison."""
    def truncate(values):
        order = values.argsort(descending=True, stable=True)[:min(top, values.numel())]
        return torch.zeros_like(values).scatter(0, order, values[order])
    left, right = truncate(counts), truncate(reference)
    support = (left.detach() > 0) | (right > 0)
    if not support.any():
        return counts.sum() * 0
    if not (left.sum() > 0) or not (right.sum() > 0):
        return counts.sum() * 0 + math.log(2)
    # Same epsilon convention as evaluation-v2; top-k IDs are piecewise constant.
    left, right = left[support], right[support]
    left = left / left.sum() + 1e-14
    right = right / right.sum() + 1e-14
    return js_divergence(left, right)


class BalancedGeometryObjective(GeometryObjective):
    def __init__(self, reference, records, model_logits):
        super().__init__(reference, records, model_logits)
        device = model_logits.device
        length = self.mask.shape[1]
        rows = torch.tensor(self.rows, device=device)[:, None, None]
        positions = torch.tensor(self.lengths, device=device)[:, None, None] + 2 + torch.arange(length, device=device)[None, :, None]
        scores = model_logits[rows, self.data['ids'][None, None, :], positions].detach()
        same = self.data['categories'][None, None, :] == self.data['categories'][self.original][..., None]
        rank = scores.masked_fill(~same, -torch.inf).argsort(dim=-1, descending=True, stable=True)
        points = self.data['offsets'][self.original]
        center = (points * self.mask[..., None]).sum(1) / self.mask.sum(1)[:, None]
        near = offset_distance(center[:, None, :], self.data['offsets'][None, :, :], self.data['origin_latitude'])
        nearest = near[:, None].expand_as(scores).masked_fill(~same, torch.inf).argsort(dim=-1, stable=True)
        frequent = self.data['visit_counts'][None, None].expand_as(scores).masked_fill(~same, -torch.inf).argsort(dim=-1, descending=True, stable=True)
        priority = torch.empty_like(rank)
        priority.scatter_(-1, rank, torch.arange(rank.shape[-1], device=device).expand_as(rank) + 64)
        for chosen, start in ((rank[..., :11], 1), (nearest[..., :12], 12), (frequent[..., :8], 24)):
            priority.scatter_reduce_(-1, chosen, torch.arange(start, start+chosen.shape[-1], device=device).expand_as(chosen), reduce='amin', include_self=True)
        priority.scatter_(-1, self.original[..., None], 0)
        self.candidates = priority.argsort(dim=-1, stable=True)[..., :min(32, rank.shape[-1])]
        self.valid = same.gather(-1, self.candidates)
        self.q = .5 * scores.gather(-1, self.candidates).masked_fill(~self.valid, -1e9).softmax(-1)
        self.q[..., 0] += .5
        self.log_q = self.q.clamp_min(1e-30).log().detach()
        self.points = self.data['offsets'][self.candidates]
        self.edges = offset_distance(self.points[:, :-1, :, None, :], self.points[:, 1:, None, :, :], self.data['origin_latitude']).detach()

    def distribution_losses(self, weights, hard=False):
        weights = weights * self.mask[..., None]
        row_visits = weights.new_zeros(len(self.rows), len(self.reference.ids)).scatter_add(
            1, self.candidates.flatten(1), weights.flatten(1))
        rank = top_visit_jsd(row_visits.sum(0), self.data['visit_counts'])
        if hard:
            distinct = (row_visits > 0).to(weights.dtype).sum(1)
        else:
            log_absent = torch.log1p(-weights.clamp(max=1-1e-7))
            summed = weights.new_zeros(row_visits.shape).scatter_add(1, self.candidates.flatten(1), log_absent.flatten(1))
            distinct = (-torch.expm1(summed)).sum(1)
        centers, width, target = self.data['histograms']['daily']
        daily = js_divergence(normalize_histogram(histogram_weights(distinct, centers, width).mean(0)), target)
        return rank, daily

    def loss(self, logits, noise, config):
        old = super().loss(logits, noise, FIXED_GEOMETRY)
        if not config.poi_distribution_weight:
            return old
        rank, daily = self.distribution_losses(self.soft(logits))
        return old + config.poi_distribution_weight * .5 * (rank + daily)

    def select(self, logits, noise, config):
        choices = (logits[:, None] + noise).masked_fill(~self.valid[:, None], -1e9).argmax(-1)
        choices = torch.cat((torch.zeros_like(choices[:, :1]), choices), 1)
        weights = F.one_hot(choices, self.q.shape[-1]).to(self.q.dtype) * self.mask[:, None, :, None]
        distance, radius = self.geometric_losses(weights, separate_paths=True)
        cost = -(weights * self.log_q[:, None]).sum((0, 2, 3)) / self.mask.sum()
        rank, daily = torch.stack([torch.stack(self.distribution_losses(weights[:, i], hard=True))
                                   for i in range(weights.shape[1])]).unbind(-1)
        scores = distance + 2*radius + .1*cost + config.poi_distribution_weight*.5*(rank+daily)
        if not torch.isfinite(scores).all():
            raise FloatingPointError('Non-finite balanced discrete selection')
        winner = int(scores.argmin())
        chosen = self.candidates.gather(-1, choices[:, winner, :, None]).squeeze(-1)
        return chosen, dict(selected_path=winner, original_objective=float(scores[0]),
            selected_objective=float(scores[winner]), rank_proxy_before=float(rank[0]),
            rank_proxy_after=float(rank[winner]), daily_proxy_before=float(daily[0]), daily_proxy_after=float(daily[winner]))


FIXED_GEOMETRY = GeometryConfig('same_category_v1', geometry_steps=200,
    geometry_distance_weight=1., geometry_radius_weight=2., geometry_prior_weight=.1)


def refine_balanced_records(records, model, reference, config, *, seed=0, global_start=0,
                            projection_executed=True, scored_logits=None):
    started = time.perf_counter()
    stats = dict(config.metadata(), optimizer_steps=0, replaced_pois=0)
    if not config.enabled or not projection_executed or not records:
        return records, dict(stats, skipped=True)
    if reference is None:
        raise ValueError('Balanced refinement needs a training reference')
    if not any(len(r['checkins']) > 1 and any(reference.category_counts[reference.poi_category[int(p)]] > 1
               for p in r['checkins']) for r in records):
        return records, dict(stats, skipped=True)
    logits = untruncated_scores(model, records) if scored_logits is None else scored_logits.detach()
    objective = BalancedGeometryObjective(reference, records, logits)
    value = int.from_bytes(hashlib.sha256(f'{GEOMETRY_VERSION}:{int(seed)}:{int(global_start)}'.encode()).digest()[:8], 'big') & ((1 << 63)-1)
    rng = torch.Generator(device=logits.device).manual_seed(value)
    parameters = objective.log_q.clone().requires_grad_()
    optimizer = torch.optim.Adam([parameters], lr=.05)
    with torch.enable_grad():
        for step in range(200):
            if step % 20 == 0:
                noise = objective.noise(rng, 8)
            optimizer.zero_grad()
            loss = objective.loss(parameters, noise, config)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite balanced geometry objective')
            loss.backward()
            if parameters.grad is None:
                raise FloatingPointError('Missing balanced geometry gradient')
            torch.nn.utils.clip_grad_norm_([parameters], 10., error_if_nonfinite=True)
            optimizer.step()
            if not torch.isfinite(parameters).all():
                raise FloatingPointError('Non-finite balanced geometry parameters')
    with torch.no_grad():
        chosen, diagnostics = objective.select(parameters, noise, config)
    chosen = chosen.cpu()
    result, replaced = copy.deepcopy(records), 0
    for row, index in enumerate(objective.rows):
        ids = chosen[row, :objective.lengths[row]]
        tokens = reference.ids[ids].numpy()
        replaced += int(np.count_nonzero(np.asarray(records[index]['checkins']) != tokens))
        result[index]['checkins'] = tokens.astype(np.asarray(records[index]['checkins']).dtype, copy=False)
        result[index]['gps'] = reference.coordinates[ids].tolist()
    assert_invariants(records, result, reference)
    stats.update(diagnostics, optimizer_steps=200, replaced_pois=replaced,
                 distribution_reference_sha256=reference.fingerprint, elapsed_seconds=time.perf_counter()-started)
    return result, stats


def configure_balance(dd, config, reference, *, terminal_refinement=True):
    dd.balance_config = config
    dd.balance_reference = reference
    dd.balance_terminal_refinement = terminal_refinement
    if not config.enabled:
        if getattr(dd, 'constraint_projector', None) is not None:
            dd.constraint_projector.balance_config = None
        return
    if reference is None or not dd.use_constraint_projection or dd.constraint_projector is None:
        raise ValueError('Distribution balancing requires PCDG and its training reference')
    if getattr(getattr(dd, 'geometry_config', None), 'geometry_refinement', 'off') != 'off':
        raise ValueError('Cannot combine old and distribution-balanced geometry')
    if any(c < dd.num_spectial or c >= dd.num_spectial+dd.type_classes for c in reference.semantic_ids.tolist()):
        raise ValueError('Semantic reference does not match category vocabulary')
    dd.enable_category_consistent_decoding(reference.poi_category, enabled=True)
    dd.constraint_projector.balance_config = config
    dd.constraint_projector.category_reference = reference
