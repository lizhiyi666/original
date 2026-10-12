"""Category-preserving geometry refinement on top of frozen 方案A (category-consistent) outputs.

Reuses the frozen same-category geometry core (geometry_projection) to refine only the
POI/GPS of each already-generated 方案A trajectory, keeping category tokens, actual POI
categories, time, length, order and all conditions fixed. No training, no re-sampling of
方案A or Off, no parameter search. Two city workers (one GPU each) refine A's original
batches in place; the controller then merges, evaluates Off/A/A+Geo (Off and A reused
from the sealed 方案A run), checks invariants and the 5% DailyLoc/G-RANK ceiling, and
audits. The base ALM, Distance KL, category decoding, training, global exporter and
evaluation definitions are untouched.
"""
import argparse
import copy
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from experiment_io import atomic_json, publish_torch, safe_tag, sha256_file
from geometry_projection import (GeometryConfig, assert_invariants,
                                 load_geometry_reference, refine_records)
from tools.ablation_common import evaluate
from tools.city_common import load_city

RUN_ID = 'ny-ist-category-geo-fixed-v1-20261012'
CITIES = ('NewYork_PO1_OOD', 'Istanbul_PO1_OOD')
SEEDS = (135398, 135399, 135400)
EXPECTED_CONFIG_SHA = 'eded18df16922bdc618f3328119bc9101022d61ced9373d72919188f201ed3a2'
SAMPLES = {'NewYork_PO1_OOD': 2108, 'Istanbul_PO1_OOD': 4914}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def fingerprint():
    paths = list(ROOT.glob('*.py'))
    for folder in ('tools', 'tests', 'config', 'add_thin', 'discrete_diffusion', 'evaluations'):
        paths += [p for p in (ROOT / folder).rglob('*') if p.is_file() and p.suffix in ('.py', '.yaml', '.yml')]
    return {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in sorted(paths)}


def mean_sd(values):
    clean = [v for v in values if v is not None and math.isfinite(v)]
    if len(clean) != len(values) or not clean:
        return dict(mean=None, sample_sd=None, n=len(clean))
    m = sum(clean) / len(clean)
    sd = math.sqrt(sum((v - m) ** 2 for v in clean) / (len(clean) - 1)) if len(clean) > 1 else 0.0
    return dict(mean=m, sample_sd=sd, n=len(clean))


def a_shard_path(category_source, city, seed, rank):
    return Path(category_source) / city / f'seed-{seed}' / f'rank-{rank}' / 'payload.pkl'


def verify_a_source(category_source):
    """A must be complete+passed; every listed file hash must match its audit."""
    cs = Path(category_source)
    if read(cs / 'status.json').get('state') != 'complete' or read(cs / 'audit.json').get('state') != 'passed':
        raise RuntimeError('方案A source is not complete/passed')
    audit = read(cs / 'audit.json')
    for name, expected in audit['files'].items():
        if sha256_file(cs / name) != expected:
            raise RuntimeError('方案A source file changed: ' + name)
    return audit


def city_profile(category_source, city):
    """Read the city profile (checkpoint/data/config) from one A shard job — never guessed."""
    payload = torch.load(a_shard_path(category_source, city, SEEDS[0], 0), map_location='cpu', weights_only=False)
    return payload['job']['city']


def make_reference(profile, geo_manifest, city):
    train = Path(profile['data_root']) / profile['dataset'] / f"{profile['dataset']}_train.pkl"
    if sha256_file(train) != profile['data_sha256']['train']:
        raise RuntimeError('Geometry training reference changed: ' + str(train))
    reference = load_geometry_reference(train, geo_manifest['splits'][city]['reference'])
    expected = geo_manifest['geometry_references'][city]
    if reference.fingerprint != expected:
        raise RuntimeError(f'Geometry reference fingerprint mismatch for {city}: {reference.fingerprint} != {expected}')
    return reference


def make_config(recommendation):
    config = GeometryConfig(**recommendation['config'])
    if recommendation.get('configuration_sha256') != EXPECTED_CONFIG_SHA:
        raise RuntimeError('Frozen geometry configuration sha256 changed')
    if config.geometry_refinement != 'same_category_v1' or config.geometry_steps != 200:
        raise RuntimeError('Unexpected frozen geometry configuration')
    return config


def source_packets(payload):
    """Slice A sequences into per-batch packets in batch_traces order (replicating source_batches)."""
    packets, offset = [], 0
    for trace in payload['batch_traces']:
        n = len(trace['indices'])
        records = payload['sequences'][offset:offset + n]
        offset += n
        packets.append(dict(records=records, indices=[int(i) for i in trace['indices']],
                            global_start=int(trace['global_start']),
                            projection_executed=any(c['optimizer_steps'] > 0 for c in trace['projection_stats'])))
    if offset != len(payload['sequences']):
        raise RuntimeError('Source batch layout is incomplete')
    return packets


def resume_or_conflict(bpath, receipt_path, ident):
    """True → an identical completed batch exists (skip); False → write fresh.
    Raise on a half-done batch whose identity or bytes do not match (never overwrite)."""
    if bpath.exists() and receipt_path.exists():
        r = read(receipt_path)
        if r.get('identity') == ident and r.get('sha256') == sha256_file(bpath):
            return True
        raise RuntimeError(f'Conflicting half-done batch: {bpath}')
    return False


def _forbid(*_a, **_k):
    raise RuntimeError('Temporal/diffusion sampling is forbidden in geometry refinement')


def refine_city(category_source, geometry_source, out, city, manifest_sha, device='cuda'):
    """One city worker: load the frozen model once, refine every seed/rank/batch of A in place."""
    from tools.baseline_common import precision, state_hash
    precision()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no CPU/smaller-batch substitution')
    geo_manifest = read(Path(geometry_source) / 'manifest.json')
    recommendation = read(Path(geometry_source) / 'recommendation.json')
    profile = city_profile(category_source, city)
    config = make_config(recommendation)
    task, _ = load_city(profile, device=device)
    dd = task.discrete_diffusion
    if dd.training:
        raise RuntimeError('Model must be in eval mode')
    task.tpp_model.sample = _forbid
    dd.sample_fast = _forbid
    reference = make_reference(profile, geo_manifest, city)
    before = state_hash(dd)
    for seed in SEEDS:
        for rank in (0, 1):
            spath = a_shard_path(category_source, city, seed, rank)
            payload = torch.load(spath, map_location='cpu', weights_only=False)
            source_sha = sha256_file(spath)
            if payload['job']['seed'] != seed:
                raise RuntimeError('Source shard seed mismatch')
            for packet in source_packets(payload):
                rank_dir = out / city / f'seed-{seed}' / f'rank-{rank}'
                rank_dir.mkdir(parents=True, exist_ok=True)
                bpath = rank_dir / f"batch-{packet['global_start']}.pkl"
                receipt = rank_dir / f"batch-{packet['global_start']}.receipt.json"
                ident = dict(city=city, seed=seed, rank=rank, global_start=packet['global_start'],
                             indices=packet['indices'], manifest_sha256=manifest_sha, source_sha256=source_sha)
                if resume_or_conflict(bpath, receipt, ident):
                    continue
                rng_before = (torch.random.get_rng_state().clone(),
                              torch.cuda.get_rng_state().clone() if device.startswith('cuda') else None)
                # Mirror the frozen formal replay path (run_geometry_study.source_batches/formal):
                # refine_records scores internally only after its own movable/skip guard, so empty,
                # single-point and non-projected batches are never fed to the model.
                values, diag = refine_records(packet['records'], dd, reference, config,
                                              seed=seed, global_start=packet['global_start'],
                                              projection_executed=packet['projection_executed'])
                assert_invariants(packet['records'], values, reference)
                rng_after = (torch.random.get_rng_state(),
                             torch.cuda.get_rng_state() if device.startswith('cuda') else None)
                if not torch.equal(rng_before[0], rng_after[0]) or (
                        rng_before[1] is not None and not torch.equal(rng_before[1], rng_after[1])):
                    raise RuntimeError('Geometry refinement leaked into global RNG')
                publish_torch(bpath, dict(identity=ident, sequences=values, diagnostics=diag))
                atomic_json(receipt, dict(identity=ident, sha256=sha256_file(bpath)))
    if state_hash(dd) != before:
        raise RuntimeError('Model tensor state changed during refinement')
    atomic_json(out / f'worker-{city}.done.json', dict(city=city, state='complete',
                geometry_reference_sha256=reference.fingerprint))


INVARIANT_KEYS =('strict_ovr', 'ovr_skip', 'pair_coverage', 'category_coverage', 'Unsat', 'Category',
                  'CategoryTransition', 'category_poi_mismatch_rate', 'empty_count', 'empty_rate')


def merge_city_seed(out, category_source, city, seed, raw, reference):
    """Merge both ranks' A and A+Geo into full ordered test sets; evaluate both."""
    a_idx, a_rec, geo_rec, replaced = [], [], [], 0
    for rank in (0, 1):
        payload = torch.load(a_shard_path(category_source, city, seed, rank), map_location='cpu', weights_only=False)
        a_idx += list(payload['indices']); a_rec += payload['sequences']
        offset = 0
        for trace in payload['batch_traces']:
            n = len(trace['indices'])
            bp = torch.load(out / city / f'seed-{seed}' / f'rank-{rank}' / f"batch-{trace['global_start']}.pkl",
                            map_location='cpu', weights_only=False)
            if bp['identity']['indices'] != list(trace['indices']):
                raise RuntimeError('Refined batch indices do not match the source batch')
            geo_rec += bp['sequences']; replaced += bp['diagnostics'].get('replaced_pois', 0)
            offset += n
        if offset != len(payload['sequences']):
            raise RuntimeError('Source/refined batch layout mismatch')
    order = sorted(range(len(a_idx)), key=lambda i: a_idx[i])
    idx = [a_idx[i] for i in order]
    if idx != list(range(SAMPLES[city])):
        raise RuntimeError(f'{city}/{seed}: merged indices are not the full ordered test set')
    a_sorted = [a_rec[i] for i in order]; geo_sorted = [geo_rec[i] for i in order]
    assert_invariants(a_sorted, geo_sorted, reference)
    refs = [raw['sequences'][i] for i in idx]
    a_metrics, _ = evaluate(refs, a_sorted, raw['poi_category'], idx)
    geo_metrics, _ = evaluate(refs, geo_sorted, raw['poi_category'], idx)
    for k in INVARIANT_KEYS:
        av, gv = a_metrics.get(k), geo_metrics.get(k)
        if av is None or gv is None or abs(float(av) - float(gv)) > 1e-12:
            raise RuntimeError(f'{city}/{seed}: invariant metric {k} changed under geometry refinement')
    return dict(city=city, seed=seed, samples=len(idx), replaced_pois=replaced,
                a=a_metrics, geo=geo_metrics, sequences=geo_sorted, test_indices=idx)


def quality_gate(a_agg, geo_agg):
    """Both cities: A+Geo Distance & Radius below A; DailyLoc & G-RANK <= A*1.05 (unrounded means)."""
    def val(agg, k):
        v = agg[k]['mean']
        return v if v is not None and math.isfinite(v) else None
    ad, gd = val(a_agg, 'Distance'), val(geo_agg, 'Distance')
    ar, gr = val(a_agg, 'Radius'), val(geo_agg, 'Radius')
    adl, gdl = val(a_agg, 'DailyLoc'), val(geo_agg, 'DailyLoc')
    ag, gg = val(a_agg, 'G-RANK'), val(geo_agg, 'G-RANK')
    if None in (ad, gd, ar, gr, adl, gdl, ag, gg):
        return False, dict(reason='missing or non-finite metric')
    checks = dict(distance_below=gd < ad, radius_below=gr < ar,
                  dailyloc_within=gdl <= adl * 1.05, grank_within=gg <= ag * 1.05)
    return all(checks.values()), checks


def aggregate(per_seed):
    keys = sorted({k for m in per_seed for k in m})
    return {k: mean_sd([m.get(k) for m in per_seed]) for k in keys}


def build_report(summary):
    L = ['# 方案A + 同类别几何细化（固定配置迁移，测试侧三种子）', '',
         '在冻结方案A输出上追加 same_category_v1 几何细化（steps=200, dw=1.0, rw=2.0, pw=0.1）。',
         'Off/A 复用方案A原始输出（重算指标，非重采）；A+Geo 仅更换同类别 POI 与对应 catalog GPS，',
         '类别/实际类别/时间/长度/顺序/条件逐事件不变。三种子均值±样本标准差（固定检查点采样波动）。',
         '这只是“旧固定几何配置迁移到方案A”的实验，不是未接触测试集的全新泛化验证。', '']
    cols = ['Distance', 'Radius', 'DailyLoc', 'G-RANK', 'CategoryTransition', 'Category']
    for city in CITIES:
        s = summary[city]
        L += [f'## {city}  质量通过={s["quality_goals_met"]}  POI替换率(逐种子)={s["replaced_pois_by_seed"]}',
              '', '| 指标 | Off | A | A+Geo |', '|---|---:|---:|---:|']
        for k in cols:
            def cell(grp):
                d = s['aggregate'][grp].get(k, {})
                m, sd = d.get('mean'), d.get('sample_sd')
                return '—' if m is None else f'{m:.4f} ± {sd:.4f}'
            L.append(f'| {k} | {cell("off")} | {cell("a")} | {cell("geo")} |')
        L += ['', f'质量门槛检查(A+Geo vs A，两城均值)：{s["gate_checks"]}', '']
    L += ['约束/类别指标（strict_ovr、Category、CategoryTransition、覆盖率、Unsat、mismatch 等）在 A+Geo 下与 A 逐数值一致（1e-12），本节不再重复列出。']
    return '\n'.join(L) + '\n'


def dispatch(out, category_source, geometry_source, gpus, env_base):
    procs = []
    for city, gpu in zip(CITIES, gpus):
        done = out / f'worker-{city}.done.json'
        if done.exists():
            procs.append((city, None, None)); continue
        env = dict(env_base, CUDA_VISIBLE_DEVICES=str(gpu))
        log = (out / f'worker-{city}.log').open('w')
        p = subprocess.Popen([sys.executable, '-u', '-B', 'tools/run_category_geometry.py',
                              '--refine-city', city, '--category-source', str(category_source),
                              '--geometry-source', str(geometry_source), '--run-id', RUN_ID],
                             cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
        procs.append((city, p, log))
    failed = []
    for city, p, log in procs:
        if p is None:
            continue
        rc = p.wait(); log.close()
        if rc != 0 or not (out / f'worker-{city}.done.json').exists():
            failed.append(city)
    if failed:
        raise RuntimeError('Geometry worker failed for: ' + ', '.join(failed) + ' (see worker-<city>.log)')


def execute(args, out):
    cs = Path(args.category_source).resolve()
    gs = Path(args.geometry_source).resolve()
    if ROOT.resolve() in (cs, gs) or out.resolve().is_relative_to(cs) or out.resolve().is_relative_to(gs):
        raise RuntimeError('Separate code and output deployment required')
    verify_a_source(cs)
    if read(gs / 'status.json').get('state') != 'complete' or read(gs / 'audit.json').get('state') != 'passed':
        raise RuntimeError('Geometry source is not complete/passed')
    geo_manifest = read(gs / 'manifest.json')
    config = make_config(read(gs / 'recommendation.json'))
    a_summary = read(cs / 'summary.json')
    raw_by_city, ref_by_city = {}, {}
    for city in CITIES:
        profile = city_profile(cs, city)
        test = Path(profile['data_root']) / profile['dataset'] / f"{profile['dataset']}_test.pkl"
        if sha256_file(test) != profile['data_sha256']['test']:
            raise RuntimeError('Reference test data changed: ' + str(test))
        raw_by_city[city] = torch.load(test, map_location='cpu', weights_only=False)
        ref_by_city[city] = make_reference(profile, geo_manifest, city)
    manifest = dict(revision='ny-ist-category-geo-fixed-v1', run_id=RUN_ID, seeds=list(SEEDS), cities=list(CITIES),
                    category_source=str(cs), category_source_manifest_sha256=sha256_file(cs / 'manifest.json'),
                    geometry_source=str(gs), geometry_source_manifest_sha256=sha256_file(gs / 'manifest.json'),
                    geometry_config=config.metadata(), configuration_sha256=EXPECTED_CONFIG_SHA,
                    geometry_references={c: ref_by_city[c].fingerprint for c in CITIES},
                    source_sha256=fingerprint(), batch_size=64, precision='FP32; TF32 disabled',
                    dailyloc_grank_ceiling=1.05, expected_results=6,
                    no_training=True, no_resampling=True, no_parameter_search=True)
    if (out / 'manifest.json').exists():
        if not args.resume or read(out / 'manifest.json') != manifest:
            raise RuntimeError('Resume requires an identical manifest')
    else:
        atomic_json(out / 'manifest.json', manifest)
    manifest_sha = sha256_file(out / 'manifest.json')
    gpus = [int(g) for g in str(args.gpus).split(',') if g != '']
    dispatch(out, cs, gs, gpus, os.environ.copy())
    summary, registry = {}, {}
    for city in CITIES:
        per_seed = []
        for seed in SEEDS:
            r = merge_city_seed(out, cs, city, seed, raw_by_city[city], ref_by_city[city])
            mp = out / city / f'seed-{seed}' / 'a_geo_generated.pkl'
            publish_torch(mp, dict(sequences=r['sequences'], test_indices=r['test_indices'], t_max=24.0, city=city,
                          seed=seed, manifest_sha256=manifest_sha, geometry_reference_sha256=ref_by_city[city].fingerprint))
            atomic_json(mp.with_suffix('.receipt.json'), dict(manifest_sha256=manifest_sha, sha256=sha256_file(mp),
                        samples=r['samples'], replaced_pois=r['replaced_pois']))
            registry[f'{city}/{seed}/a_geo'] = dict(path=str(mp), sha256=sha256_file(mp), samples=r['samples'], metrics=r['geo'])
            per_seed.append(r)
        a_agg = aggregate([r['a'] for r in per_seed]); geo_agg = aggregate([r['geo'] for r in per_seed])
        ok, checks = quality_gate(a_agg, geo_agg)
        summary[city] = dict(aggregate=dict(off=a_summary[city]['aggregate']['off'], a=a_agg, geo=geo_agg),
                             per_seed=[dict(seed=r['seed'], a=r['a'], geo=r['geo']) for r in per_seed],
                             replaced_pois_by_seed={r['seed']: r['replaced_pois'] for r in per_seed},
                             quality_goals_met=ok, gate_checks=checks)
    quality = all(summary[c]['quality_goals_met'] for c in CITIES)
    atomic_json(out / 'summary.json', summary)
    atomic_json(out / 'registry.json', registry)
    (out / 'report.md').write_text(build_report(summary), encoding='utf-8')
    verify_a_source(cs)
    if fingerprint() != manifest['source_sha256']:
        raise RuntimeError('Running code changed during the run')
    skip = ('audit.json', 'status.json', 'pipeline.lock')
    files = {p.relative_to(out).as_posix(): sha256_file(p) for p in out.rglob('*')
             if p.is_file() and p.name not in skip and p.suffix != '.log'}
    atomic_json(out / 'audit.json', dict(state='passed', cities=list(CITIES), seeds=list(SEEDS), results=6,
                parent_unchanged=True, no_training=True, no_resampling=True, quality_goals_met=quality, files=files))
    atomic_json(out / 'status.json', dict(state='complete', phase='complete', pid=os.getpid(), quality_goals_met=quality))
    print(json.dumps(dict(state='complete', quality_goals_met=quality,
          summary_sha256=sha256_file(out / 'summary.json'), audit_sha256=sha256_file(out / 'audit.json'))), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--category-source')
    ap.add_argument('--geometry-source')
    ap.add_argument('--run-id', default=RUN_ID)
    ap.add_argument('--gpus', default='0,1')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--verify-only', type=Path, help='Read-only verification of a finished output directory')
    ap.add_argument('--refine-city', help='Internal single-city worker mode (launched per GPU by the controller)')
    args = ap.parse_args()
    if args.verify_only:
        out = args.verify_only
        audit = read(out / 'audit.json')
        if audit['state'] != 'passed':
            raise RuntimeError('Audit not passed')
        for name, expected in audit['files'].items():
            if sha256_file(out / name) != expected:
                raise RuntimeError('Local hash mismatch: ' + name)
        print(json.dumps(dict(state='passed', verified_files=len(audit['files']), results=audit.get('results'),
              quality_goals_met=audit.get('quality_goals_met'), summary_sha256=sha256_file(out / 'summary.json'))))
        return
    out = ROOT / 'experiment_runs' / safe_tag(args.run_id)
    if args.refine_city:
        if not args.category_source or not args.geometry_source:
            ap.error('worker mode requires --category-source and --geometry-source')
        manifest_sha = sha256_file(out / 'manifest.json')
        refine_city(args.category_source, args.geometry_source, out, args.refine_city, manifest_sha, device='cuda')
        return
    if not args.category_source or not args.geometry_source:
        ap.error('--category-source and --geometry-source are required')
    if out.exists() and not args.resume:
        raise RuntimeError('Existing run directory; use --resume or a new id, never overwrite')
    out.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (out / 'pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            execute(args, out)
        except BaseException as exc:
            atomic_json(out / 'status.json', dict(state='failed', pid=os.getpid(),
                        error_type=type(exc).__name__, error=str(exc)))
            raise


if __name__ == '__main__':
    main()





