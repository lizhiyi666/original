"""Frozen selection rules for PCDG-Geo; no test-driven parameter selection."""
from dataclasses import asdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics

import numpy as np

from geometry_projection import GeometryConfig
from evaluations.statistical_metrics import evaluation, radius, travel_distance

REVISION = 'pcdg-geo-v1'
CITIES = ('NewYork_PO1_OOD', 'Istanbul_PO1_OOD')
SEEDS = (135398, 135399, 135400)
SPLIT_SEED = 20261010
STEP_CHOICES = (50, 100, 200)
RADIUS_WEIGHTS = (.5, 1., 2., 4.)
PRIOR_WEIGHTS = (.001, .01, .1)
SPEED_CEILING = 1.10
INVARIANT_METRICS = ('strict_ovr', 'ovr_skip', 'pair_coverage', 'category_coverage', 'Unsat',
                     'Category', 'CategoryTransition', 'empty_count', 'empty_rate', 'category_poi_mismatch_rate')


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def split_indices(count):
    if count <= 1024:
        raise ValueError('Training split must leave a separate geometry prior pool')
    indices = np.random.default_rng(SPLIT_SEED).permutation(count).tolist()
    return dict(screen=indices[:512], confirm=indices[512:1024], reference=sorted(indices[1024:]))


def configurations(steps):
    return [GeometryConfig('same_category_v1', geometry_steps=steps, geometry_radius_weight=r,
                           geometry_prior_weight=b) for r, b in itertools.product(RADIUS_WEIGHTS, PRIOR_WEIGHTS)]


def config_label(config):
    return f'r{config.geometry_radius_weight:g}-b{config.geometry_prior_weight:g}'


def metric_means(rows):
    keys = sorted(rows[0])
    return {k: (statistics.mean(row[k] for row in rows) if all(row.get(k) is not None for row in rows) else None)
            for k in keys if k != 'evaluation_version'}


def check_invariant_metrics(actual, original):
    for metric in INVARIANT_METRICS:
        a, b = actual.get(metric), original.get(metric)
        if (a is None) != (b is None) or (a is not None and abs(a-b) > 1e-12):
            raise RuntimeError(f'Category-preserving refinement changed invariant metric: {metric}')


def quality_failures(actual, full, require_geometry_nonworse=False):
    required = ('strict_ovr', 'pair_coverage', 'category_coverage', 'DailyLoc', 'G-RANK', 'Distance', 'Radius')
    if any(actual.get(k) is None or full.get(k) is None or not math.isfinite(actual[k]) for k in required):
        return ['undefined-metric']
    failures = []
    if actual['strict_ovr'] > full['strict_ovr'] + .01 + 1e-12: failures.append('strict-ovr')
    for k in ('pair_coverage', 'category_coverage'):
        if actual[k] < full[k] - .01 - 1e-12: failures.append(k)
    for k in ('DailyLoc', 'G-RANK'):
        if actual[k] > full[k] * 1.05 + 1e-12: failures.append(k)
    if require_geometry_nonworse:
        for k in ('Distance', 'Radius'):
            if actual[k] > full[k] + 1e-12: failures.append(k)
    return failures


def target_metrics(city, full, jointgen):
    return {'Distance': full['Distance'] if city == CITIES[0] else jointgen['Distance'],
            'Radius': jointgen['Radius']}


def rank_candidates(results, baselines, *, confirmation=False):
    """Each city value is a seed-list; tie-break by objective, cost, then coefficients."""
    ranked = []
    for label, candidate in results.items():
        ratios, failed, cost = [], [], 0.
        for city in CITIES:
            rows = candidate['cities'].get(city, [])
            if not rows:
                failed.append(f'{city}:missing'); continue
            actual = metric_means([r['metrics'] for r in rows])
            full = metric_means(baselines[city]['full'])
            joint = metric_means(baselines[city]['no_projection'])
            failed.extend(f'{city}:{f}' for f in quality_failures(actual, full, confirmation))
            for metric, target in target_metrics(city, full, joint).items():
                if actual.get(metric) is None or target is None:
                    failed.append(f'{city}:{metric}:undefined')
                else:
                    ratios.append(actual[metric] / max(target, 1e-12))
            cost += statistics.mean(r['offline_seconds'] for r in rows)
        if failed:
            continue
        config = candidate['config']
        ranked.append(dict(label=label, config=config, rank=[max(ratios), statistics.mean(ratios), cost,
            config['geometry_radius_weight'], config['geometry_prior_weight']], ratios=ratios))
    return sorted(ranked, key=lambda r:r['rank'])


def choose_steps(speed_results):
    eligible = []
    for steps in STEP_CHOICES:
        cases = speed_results.get(str(steps), {})
        if set(cases) != {city+'/'+case for city in CITIES for case in ('ordinary', 'long')}:
            continue
        if all(case['ratio'] <= SPEED_CEILING and case['state'] == 'complete' for case in cases.values()):
            eligible.append(steps)
    # The plan forbids proceeding if 50 steps cannot meet the gate.
    return max(eligible) if 50 in eligible else None


def geometric_diagnostics(refs, before, after, mapping):
    result = {}
    def selected(records):
        return [i for i,(r,s) in enumerate(zip(refs, records)) if len(r['marks']) and len(s['checkins'])
            and int(r['marks'][0]) == mapping[int(s['checkins'][0])]
            and int(r['marks'][-1]) == mapping[int(s['checkins'][-1])]]
    first, second = selected(before), selected(after)
    if first != second:
        raise RuntimeError('Distance endpoint eligibility changed')
    result['endpoint_match_count'] = len(first)
    result['endpoint_indices_sha256'] = content_hash(first)
    generated_mask = [i for i in first if len(before[i]['gps']) > 1]
    real_mask = [i for i in first if len(refs[i]['gps']) > 1]
    result.update(distance_generated_count=len(generated_mask),distance_real_count=len(real_mask),
                  distance_generated_indices_sha256=content_hash(generated_mask),distance_real_indices_sha256=content_hash(real_mask))
    real = [np.asarray(r['gps']) for r in refs if len(r['gps']) > 1]
    for label, records in (('before', before), ('after', after)):
        points = [np.asarray(r['gps']) for r in records if len(r['gps']) > 1]
        distances = [float(travel_distance(p)) for p in points]
        radii = [float(radius(p)) for p in points]
        result[label] = dict(eligible_multipoint=len(points),
            unfiltered_distance_jsd=evaluation(distances, [float(travel_distance(p)) for p in real]) if distances and real else None,
            distance_quantiles=np.quantile(distances,[.1,.5,.9,.99]).tolist() if distances else None,
            radius_quantiles=np.quantile(radii,[.1,.5,.9,.99]).tolist() if radii else None)
    replaced = sum(int(np.count_nonzero(np.asarray(a['checkins']) != np.asarray(b['checkins']))) for a,b in zip(before,after))
    total = sum(len(a['checkins']) for a in before)
    result.update(replaced_pois=replaced, poi_replacement_rate=replaced/total if total else 0.)
    return result
