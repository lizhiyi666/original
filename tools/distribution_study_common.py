"""Frozen, bounded selection and immutable artifacts for distribution-balance v1."""
from dataclasses import asdict
import itertools
import json
import math
from pathlib import Path
import statistics

import numpy as np
import torch

from distribution_balance import BalanceConfig
from experiment_io import atomic_json, publish_torch, sha256_file
from evaluations.statistical_metrics import STATISTICAL_NAMES, evaluation, travel_distance

CITIES = ('NewYork_PO1_OOD', 'Istanbul_PO1_OOD')
SEEDS = (135398, 135399, 135400)
RUN_ID = 'ny-ist-distribution-balanced-v1-20261012'
LOWER = ('strict_ovr', 'ovr_skip', 'Unsat')
HIGHER = ('pair_coverage', 'category_coverage')
METRICS = STATISTICAL_NAMES + LOWER + HIGHER
TOLERANCE = 1e-12


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def configurations():
    return [BalanceConfig('train_ref_v1', r, c, p)
            for r, c, p in itertools.product((.25, .5, 1.), (0., 1.), (0., .1, 1.))]


def label(config, spatial=False):
    base = f'r{config.constraint_scale:g}-c{config.category_distribution_weight:g}'
    return base if spatial else base + f'-p{config.poi_distribution_weight:g}'


def aggregate(rows):
    keys = sorted(set().union(*(r.keys() for r in rows))) if rows else []
    result = {}
    for key in keys:
        values = [r.get(key) for r in rows]
        good = bool(values) and all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)
        result[key] = dict(mean=statistics.mean(values) if good else None,
            sample_sd=statistics.stdev(values) if good and len(values) > 1 else None, n=len(values))
    return result


def means(rows):
    return {k: v['mean'] for k, v in aggregate(rows).items()}


def failures(actual, pcdg, jointgen, *, distributions):
    required = METRICS if distributions else LOWER + HIGHER
    if any(not isinstance(actual.get(k), (int, float)) or not math.isfinite(actual[k])
           for k in required):
        return ['undefined-candidate-metric']
    failed = []
    for key in required:
        reference = jointgen if key in STATISTICAL_NAMES else pcdg
        target = reference.get(key)
        if not isinstance(target, (int, float)) or not math.isfinite(target):
            failed.append(key+':undefined-baseline')
        elif (actual[key] < target-TOLERANCE if key in HIGHER else actual[key] > target+TOLERANCE):
            failed.append(key)
    return failed


def select(results, baselines, *, confirmation=False):
    """All cities/seeds required; screen constraints first, confirm all 11 metrics."""
    ranked, diagnostics = [], {}
    seeds = SEEDS if confirmation else SEEDS[:1]
    for name, candidate in sorted(results.items()):
        ratios, errors = [], []
        for city in CITIES:
            rows = candidate.get('cities', {}).get(city, [])
            base = baselines.get(city, {})
            if (sorted(r['seed'] for r in rows) != list(seeds)
                    or sorted(r['seed'] for r in base.get('rows', [])) != list(seeds)):
                errors.append(city+':missing-or-duplicate-seed'); continue
            actual = means([r['metrics'] for r in rows])
            pcdg = means([r['full'] for r in base['rows']])
            joint = means([r['no_projection'] for r in base['rows']])
            errors.extend(city+':'+k for k in failures(actual, pcdg, joint, distributions=confirmation))
            for metric in STATISTICAL_NAMES:
                value, target = actual.get(metric), joint.get(metric)
                if value is None or target is None or not math.isfinite(value) or not math.isfinite(target):
                    errors.append(city+':'+metric+':undefined'); continue
                ratios.append(value/max(target, TOLERANCE))
        diagnostics[name] = errors
        if not errors:
            cfg = BalanceConfig(**candidate['config'])
            rank = [max(ratios), statistics.mean(ratios), cfg.constraint_scale,
                    cfg.category_distribution_weight, cfg.poi_distribution_weight]
            ranked.append(dict(label=name, config=asdict(cfg), rank=rank))
    return dict(ranking=sorted(ranked, key=lambda r: r['rank']), failures=diagnostics,
                confirmation=confirmation, tolerance=TOLERANCE)


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise RuntimeError('Conflicting immutable JSON: '+str(path))
        return
    atomic_json(path, value)


def saved_payload(path, identity):
    path = Path(path); receipt = path.with_suffix('.receipt.json')
    if not path.exists() and not receipt.exists():
        return None
    if not path.exists() or not receipt.exists():
        raise RuntimeError('Half-published batch; explicit recovery required: '+str(path))
    if read(receipt) != dict(identity=identity, sha256=sha256_file(path)):
        raise RuntimeError('Receipt identity/hash conflict: '+str(path))
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload.get('identity') != identity:
        raise RuntimeError('Payload identity conflict: '+str(path))
    return payload


def save_payload(path, identity, **payload):
    if saved_payload(path, identity) is not None:
        raise RuntimeError('Attempt to republish completed result: '+str(path))
    value = dict(identity=identity, **payload)
    publish_torch(path, value)
    immutable_json(Path(path).with_suffix('.receipt.json'), dict(identity=identity, sha256=sha256_file(path)))
    return value


def aligned_indices(indices, expected):
    if (any(isinstance(i, bool) or not isinstance(i, int) for i in indices)
            or len(indices) != len(set(indices)) or indices != list(expected)):
        raise RuntimeError('Duplicate, missing, reordered, or invalid indices')


def distance_diagnostics(refs, generated, mapping):
    full = [[], []]; counts = dict(generated=0, reference=0, endpoint_pairs=0)
    for ref, gen in zip(refs, generated):
        cats = [mapping[int(p)] for p in gen['checkins']]
        match = bool(cats and len(ref['marks']) and cats[0] == ref['marks'][0] and cats[-1] == ref['marks'][-1])
        counts['endpoint_pairs'] += int(match)
        for index, record in enumerate((gen, ref)):
            if len(record['gps']) > 1:
                full[index].append(travel_distance(np.asarray(record['gps'])))
                if match:counts['generated' if index == 0 else 'reference'] += 1
    value = evaluation(*full) if all(full) else None
    return dict(distance_unfiltered=value if value is not None and math.isfinite(value) else None,
                distance_endpoint_selection=counts)
