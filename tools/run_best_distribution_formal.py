"""Explicitly authorized formal sampling of one frozen, best confirmed candidate.

Never repeats calibration or baseline/temporal sampling. Legacy v2 table values
are kept alongside a deterministic v3 audit; a quality failure does not delete
results or trigger a new search. This is separate from the bounded study.
"""
import argparse
import copy
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from distribution_balance import BalanceConfig, digest
from evaluations.grank_v3 import evaluate as evaluate_v3, REVISION as V3_REVISION
from experiment_io import atomic_json, safe_tag, sha256_file
from tools.category_cache_audit import metrics_equal
from tools.distribution_study_common import (CITIES, SEEDS, STATISTICAL_NAMES, LOWER, HIGHER, METRICS,
    aggregate, aligned_indices, failures, immutable_json, label, means, read, select)
from tools.geometry_inheritance import checked_path
from tools.run_category_geometry import fingerprint
from tools.run_distribution_study import Sources, copy_input, phase_run, verify_source_manifest, verify as verify_legacy

RUN_ID = 'ny-ist-distribution-best-formal-v1-20261013'
BEST_LABEL = 'r0.5-c0-p1'
EXPECTED_SAMPLES = {'NewYork_PO1_OOD': 2108, 'Istanbul_PO1_OOD': 4914}
EXPECTED_RECORDS = 21066
UNCHANGED_EXCEPT = {'G-RANK', 'totalJSD', 'evaluation_version'}


def rank_confirmed(summary):
    """Existing shared-city rank, without requiring the failed distribution gate."""
    confirm = summary['confirm']
    ranked = []
    for name, candidate in confirm['results'].items():
        if set(candidate['cities']) != set(CITIES):
            raise RuntimeError('Confirmed candidate city inventory differs')
        ratios, errors = [], []
        for city in CITIES:
            rows, base = candidate['cities'][city], confirm['baselines'][city]['rows']
            if sorted(r['seed'] for r in rows) != list(SEEDS) or sorted(r['seed'] for r in base) != list(SEEDS):
                raise RuntimeError('Confirmed candidate requires all three fixed seeds')
            actual, joint, pcdg = means([r['metrics'] for r in rows]), means([r['no_projection'] for r in base]), means([r['full'] for r in base])
            errors += [city + ':' + e for e in failures(actual, pcdg, joint, distributions=False)]
            for key in STATISTICAL_NAMES:
                a, b = actual.get(key), joint.get(key)
                if a is None or b is None or not math.isfinite(a) or not math.isfinite(b):
                    raise RuntimeError('Undefined candidate or baseline distribution metric')
                ratios.append(a / max(b, 1e-12))
        cfg = BalanceConfig(**candidate['config'])
        if label(cfg) != name:
            raise RuntimeError('Candidate label/config mismatch')
        if not errors:
            ranked.append(dict(label=name, config=asdict(cfg), rank=[max(ratios), statistics.mean(ratios),
                cfg.constraint_scale, cfg.category_distribution_weight, cfg.poi_distribution_weight]))
    if not ranked:
        raise RuntimeError('No confirmed candidate preserves the constraint gates')
    return sorted(ranked, key=lambda r: r['rank'])


class SelectionEvidence:
    def __init__(self, root):
        self.root = Path(root).resolve()
        audit = read(checked_path(self.root, 'audit.json'))
        if audit.get('state') != 'passed' or audit.get('evaluation_version') != 3 or audit.get('metrics_recomputed') != 64:
            raise RuntimeError('Expected the audited 64-record v3 calibration re-evaluation')
        if read(checked_path(self.root, 'status.json')).get('state') != 'complete':
            raise RuntimeError('Selection re-evaluation is not complete')
        self.files = dict(audit['files'])
        self.files.update({n: sha256_file(self.root / n) for n in ('audit.json', 'status.json')})
        self.verify()
        self.manifest = read(self.root / 'manifest.json')
        self.summary = read(self.root / 'summary.json')
        if self.manifest.get('revision') != V3_REVISION or self.summary.get('original_terminal_state') != 'stopped':
            raise RuntimeError('Unexpected selection provenance')
        self.ranking = rank_confirmed(self.summary)
        if self.ranking[0]['label'] != BEST_LABEL:
            raise RuntimeError('Best confirmed candidate changed; user decision required')

    def verify(self):
        for name, expected in self.files.items():
            if sha256_file(checked_path(self.root, name)) != expected:
                raise RuntimeError('Selection evidence changed: ' + name)


def check_formal_counts(formal):
    if set(formal['results']) != {BEST_LABEL}:
        raise RuntimeError('Formal results must contain exactly the frozen candidate')
    candidate = formal['results'][BEST_LABEL]
    if set(candidate['cities']) != set(CITIES) or set(formal['baselines']) != set(CITIES):
        raise RuntimeError('Formal city inventory differs')
    total = 0
    for city in CITIES:
        rows = candidate['cities'][city]
        if sorted(r['seed'] for r in rows) != list(SEEDS):
            raise RuntimeError('Formal result seed inventory differs')
        if sorted(r['seed'] for r in formal['baselines'][city]['rows']) != list(SEEDS):
            raise RuntimeError('Formal baseline seed inventory differs')
        for row in rows:
            if row['samples'] != EXPECTED_SAMPLES[city] or label(BalanceConfig(**row['config'])) != BEST_LABEL:
                raise RuntimeError('Formal samples/config differ')
            total += row['samples']
    if total != EXPECTED_RECORDS:
        raise RuntimeError('Formal delivery must contain 21066 new trajectory records')


def summarize(formal):
    formal['selection'] = select(formal['results'], formal['baselines'], confirmation=True)
    for candidate in formal['results'].values():
        candidate['aggregate'] = {c: aggregate([r['metrics'] for r in candidate['cities'][c]]) for c in CITIES}
        candidate['paired_deltas'] = {}
        for city in CITIES:
            paired = {r['seed']: r for r in formal['baselines'][city]['rows']}
            candidate['paired_deltas'][city] = [dict(seed=r['seed'], deltas={k:
                (r['metrics'][k] - paired[r['seed']]['full' if k in LOWER + HIGHER else 'no_projection'][k]
                 if r['metrics'].get(k) is not None and paired[r['seed']]['full' if k in LOWER + HIGHER else 'no_projection'].get(k) is not None else None)
                for k in METRICS}) for r in candidate['cities'][city]]
    return formal


def recompute_v3(out):
    """18 CPU metric sets: six new results plus two reused baselines per seed/city."""
    out = Path(out)
    manifest = read(out / 'manifest.json')
    manifest_sha = sha256_file(out / 'manifest.json')
    records = []
    for city in CITIES:
        path = out / 'evaluation_inputs' / city / f'{city}_test.pkl'
        if sha256_file(path) != manifest['profiles'][city]['data_sha256']['test']:
            raise RuntimeError('Formal evaluation input hash differs')
        raw = torch.load(path, map_location='cpu', weights_only=False)
        indices = list(range(EXPECTED_SAMPLES[city]))
        if len(raw['sequences']) != len(indices):
            raise RuntimeError('Formal reference count differs')
        for seed in SEEDS:
            for relative in (f'baselines/formal/{city}/seed-{seed}.pkl',
                             f'results/formal/{city}/seed-{seed}/{BEST_LABEL}.pkl'):
                p = checked_path(out, relative)
                payload = torch.load(p, map_location='cpu', weights_only=False)
                ident = payload['identity']
                if (ident['study_manifest_sha256'] != manifest_sha or ident['city'] != city
                        or ident['seed'] != seed or ident['phase'] != 'formal' or payload['split'] != 'test'):
                    raise RuntimeError('Formal payload identity differs')
                aligned_indices(payload['indices'], indices)
                aligned_indices(ident['indices'], indices)
                receipt = read(p.with_suffix('.receipt.json'))
                if receipt != dict(identity=ident, sha256=sha256_file(p)):
                    raise RuntimeError('Formal payload receipt differs')
                methods = payload['sequences_by_method'] if relative.startswith('baselines/') else {'metrics': payload['sequences']}
                if relative.startswith('baselines/') and set(methods) != {'full', 'no_projection'}:
                    raise RuntimeError('Reused baseline method inventory differs')
                for method, generated in sorted(methods.items()):
                    detail = {}
                    metrics, _ = evaluate_v3(raw['sequences'], generated, raw['poi_category'], indices, diagnostics=detail)
                    legacy = payload['result'][method]
                    if legacy.get('evaluation_version') != 2:
                        raise RuntimeError('Existing table requires v2 legacy metrics')
                    metrics_equal({k: v for k, v in metrics.items() if k not in UNCHANGED_EXCEPT},
                                  {k: v for k, v in legacy.items() if k not in UNCHANGED_EXCEPT})
                    records.append(dict(path=relative, source_sha256=sha256_file(p), city=city, seed=seed,
                        method=method, metrics=metrics, legacy_metrics=legacy, grank_diagnostics=detail))
    if len(records) != 18:
        raise RuntimeError('Expected 18 independent formal metric sets')
    return dict(evaluation_version=3, revision=V3_REVISION, source_manifest_sha256=manifest_sha, records=records)


def v3_summary(formal, dual):
    result = copy.deepcopy(formal)
    for row in dual['records']:
        if row['method'] == 'metrics':
            rows = result['results'][BEST_LABEL]['cities'][row['city']]
            next(r for r in rows if r['seed'] == row['seed'])['metrics'] = row['metrics']
        else:
            rows = result['baselines'][row['city']]['rows']
            next(r for r in rows if r['seed'] == row['seed'])[row['method']] = row['metrics']
    return summarize(result)


def report(summary):
    lines = ['# 最佳候选正式测试结果', '',
        '用户于 2026-10-13 明确要求：按已选最佳候选正式采样，即使校准未达标，也将实际测试结果插入现有方法表。',
        '原校准失败记录保留；本次只用冻结的 r0.5-c0-p1，不追加搜索。两城、三种子、6 份结果、21066 条新轨迹。',
        '已有 JointGen/PCDG、模型和时间缓存复用，不重训、不重新采样基线或时间。', '',
        '现有文档表格沿用服务器封存 evaluation v2；确定性 evaluation v3 是独立复核，不混用版本阈值。', '',
        '| 城市 | Distance | Radius | CategoryTransition | DailyLoc | Category | G-RANK |',
        '|---|---:|---:|---:|---:|---:|---:|']
    candidate = summary['formal']['results'][BEST_LABEL]
    for city in CITIES:
        vals = candidate['aggregate'][city]
        cells = ['undefined' if vals[k]['mean'] is None else f"{vals[k]['mean']:.6f} ± {vals[k]['sample_sd']:.6f}" for k in STATISTICAL_NAMES]
        lines.append('| ' + city + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', '| 城市 | strict_ovr (%) | ovr_skip (%) | pair_coverage (%) | category_coverage (%) | Unsat (%) |',
              '|---|---:|---:|---:|---:|---:|']
    for city in CITIES:
        vals = candidate['aggregate'][city]
        cells = ['undefined' if vals[k]['mean'] is None else f"{100*vals[k]['mean']:.2f} ± {100*vals[k]['sample_sd']:.2f}" for k in ('strict_ovr','ovr_skip','pair_coverage','category_coverage','Unsat')]
        lines.append('| ' + city + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', 'v2 严格门槛失败项：' + json.dumps(summary['formal']['selection']['failures'], ensure_ascii=False),
              'v3 严格门槛失败项：' + json.dumps(summary['formal_v3']['selection']['failures'], ensure_ascii=False),
              '以上为三种子均值 ± 样本标准差，不是置信区间，也不表示统计显著。', '']
    return '\n'.join(lines)


def execute(args, out, source, evidence):
    if not args.allow_confirmation_gate_failure:
        raise RuntimeError('Explicit user-authorized gate override is required')
    selected = evidence.ranking[0]
    cfg = BalanceConfig(**selected['config'])
    if {c: source.profiles[c]['test_count'] for c in CITIES} != EXPECTED_SAMPLES:
        raise RuntimeError('Original formal sample counts changed')
    frozen = dict(selected=selected, ranking=evidence.ranking, selection_source=str(evidence.root),
        selection_files=evidence.files, original_confirmation_passed=False,
        override='User explicitly requested formal sampling of the best confirmed candidate despite unmet calibration gates on 2026-10-13.',
        frozen_before_formal_sampling=True)
    immutable_json(out / 'selected-candidate.json', frozen)
    for city in CITIES:
        copy_input(source, out, city, 'test')
    manifest = dict(revision='distribution-best-formal-v1', run_id=args.run_id,
        source_commit=args.source_commit, source_sha256=fingerprint(), source_inputs=source.inputs,
        geometry_source=str(source.geometry), category_source=str(source.category), profiles=source.profiles,
        splits=source.geo['splits'], seeds=list(SEEDS), configs=[asdict(cfg)],
        distribution_references={c: source.refs[c].fingerprint for c in CITIES},
        selected_candidate_sha256=sha256_file(out / 'selected-candidate.json'), gpus=args.gpus,
        batch_size=64, precision='FP32; TF32 disabled', memory_ceiling=.8, geometry_steps=200,
        user_authorized_gate_override=True, no_training=True, no_temporal_resampling=True,
        no_baseline_resampling=True, table_evaluation_version=2, audit_evaluation_version=3)
    immutable_json(out / 'manifest.json', manifest)
    atomic_json(out / 'status.json', dict(state='running', phase='formal', pid=os.getpid(), updated_at=time.time()))
    results, baselines = phase_run(args, out, 'formal', [cfg])
    formal = summarize(dict(results=results, baselines=baselines))
    check_formal_counts(formal)
    dual = recompute_v3(out)
    formal3 = v3_summary(formal, dual)
    summary = dict(terminal_state='complete', reason='user-authorized-best-candidate-formal-complete',
        selected_candidate=selected, calibration_gate_passed=False, user_authorized_gate_override=True,
        quality_goals_met=bool(formal['selection']['ranking']), quality_goals_met_v3=bool(formal3['selection']['ranking']),
        formal=formal, formal_v3=formal3)
    evidence.verify()
    source.verify()
    verify_source_manifest(source, manifest)
    registry = {}
    for city in CITIES:
        for row in results[BEST_LABEL]['cities'][city]:
            path = f'results/formal/{city}/seed-{row["seed"]}/{BEST_LABEL}.pkl'
            registry[f'{city}/{row["seed"]}'] = dict(path=path, sha256=sha256_file(out / path), **row)
    atomic_json(out / 'dual-evaluation-v3.json', dual)
    atomic_json(out / 'summary.json', summary)
    atomic_json(out / 'registry.json', registry)
    (out / 'report.md').write_text(report(summary), encoding='utf-8')
    files = {p.relative_to(out).as_posix(): sha256_file(p) for p in out.rglob('*') if p.is_file()
             and p.name not in ('audit.json', 'status.json', 'pipeline.lock') and p.suffix not in ('.log', '.lock')}
    atomic_json(out / 'audit.json', dict(state='passed', terminal_state='complete', results=6,
        trajectory_records=EXPECTED_RECORDS, files=files, originals_unchanged=True, no_training=True,
        no_temporal_resampling=True, no_baseline_resampling=True, selection_evidence_unchanged=True,
        configuration_frozen_before_test=True, user_authorized_gate_override=True,
        quality_goals_met=summary['quality_goals_met'], quality_goals_met_v3=summary['quality_goals_met_v3'], v3_metric_sets=18))
    atomic_json(out / 'status.json', dict(state='complete', phase='sealed', pid=os.getpid(),
        results=6, trajectory_records=EXPECTED_RECORDS, quality_goals_met=summary['quality_goals_met']))
    return verify_delivery(out)


def verify_delivery(out, recompute=False):
    out = Path(out)
    base = verify_legacy(out, recompute=False)
    audit, manifest, summary = read(out / 'audit.json'), read(out / 'manifest.json'), read(out / 'summary.json')
    if (base['results'] != 6 or audit.get('trajectory_records') != EXPECTED_RECORDS
            or manifest.get('revision') != 'distribution-best-formal-v1'
            or not audit.get('user_authorized_gate_override')):
        raise RuntimeError('Not the explicit six-result formal delivery')
    check_formal_counts(summary['formal'])
    if sha256_file(out / 'selected-candidate.json') != manifest['selected_candidate_sha256']:
        raise RuntimeError('Frozen candidate changed')
    stored = read(out / 'dual-evaluation-v3.json')
    if stored.get('evaluation_version') != 3 or len(stored['records']) != 18:
        raise RuntimeError('Missing complete v3 audit')
    maximum = 0.0
    if recompute:
        actual = recompute_v3(out)
        if actual['source_manifest_sha256'] != stored['source_manifest_sha256']:
            raise RuntimeError('Recompute manifest differs')
        for a, b in zip(actual['records'], stored['records']):
            for key in ('path', 'source_sha256', 'city', 'seed', 'method', 'legacy_metrics', 'grank_diagnostics'):
                if a[key] != b[key]:
                    raise RuntimeError('Recompute identity/top-1000 mismatch: ' + key)
            metrics_equal(a['metrics'], b['metrics'])
            maximum = max([maximum] + [abs(v - b['metrics'][k]) for k, v in a['metrics'].items() if v is not None])
    base.update(trajectory_records=EXPECTED_RECORDS, v3_metric_sets_recomputed=18 if recompute else 0,
                cpu_recomputed=recompute, table_evaluation_version=2, audit_evaluation_version=3,
                v3_max_abs_difference=maximum, quality_goals_met_v3=summary['quality_goals_met_v3'])
    return base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry-source', type=Path)
    parser.add_argument('--category-source', type=Path)
    parser.add_argument('--selection-source', type=Path)
    parser.add_argument('--run-id', default=RUN_ID)
    parser.add_argument('--source-commit')
    parser.add_argument('--gpus', default='0,1')
    parser.add_argument('--allow-confirmation-gate-failure', action='store_true')
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--verify-only', type=Path)
    parser.add_argument('--recompute', action='store_true')
    args = parser.parse_args()
    if args.verify_only:
        print(json.dumps(verify_delivery(args.verify_only, args.recompute))); return
    if not all((args.geometry_source, args.category_source, args.selection_source)):
        parser.error('All three sealed source paths are required')
    if not args.preflight_only and (not args.allow_confirmation_gate_failure or not re.fullmatch('[0-9a-f]{40}', args.source_commit or '')):
        parser.error('Verified source commit and explicit gate-override flag are required')
    if len(set(args.gpus.split(','))) != 2:
        parser.error('Exactly two distinct GPUs are required')
    out = ROOT / 'experiment_runs' / safe_tag(args.run_id)
    if out.exists() and not args.preflight_only:
        raise RuntimeError('Existing formal run; no duplicate launch, overwrite or automatic resume')
    evidence = SelectionEvidence(args.selection_source)
    source = Sources(args.geometry_source, args.category_source)
    if args.preflight_only:
        print(json.dumps(dict(state='passed', source_files=len(source.inputs), selected=evidence.ranking[0],
                             test_samples={c: source.profiles[c]['test_count'] for c in CITIES}))); return
    out.mkdir(parents=True, exist_ok=False)
    import fcntl
    with (out / 'pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            result = execute(args, out, source, evidence)
        except BaseException as exc:
            atomic_json(out / 'status.json', dict(state='failed', pid=os.getpid(), error_type=type(exc).__name__, error=str(exc)))
            raise
    print(json.dumps(result))


if __name__ == '__main__':
    main()
