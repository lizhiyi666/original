"""Formal three-seed category-consistent decoding comparison across two cities.

Off = the sealed three-seed Full results from a completed two-city delivery
(reused verbatim, never resampled). On = category-consistent decoding
(sampled-category-poi-v2) regenerated on each sealed Full shard's own job and
time cache. On vs Off is compared on the shared-time-cache aligned basis
(On covers every cached slot; Off is an ordered subset); results aggregate to
per-city three-seed mean / sample standard deviation.

On sampling for the twelve (city, seed, rank) shards is dispatched across the
available GPUs (default two) by launching tools/ablation_worker.py subprocesses,
so both GPUs are used; the single controller then aligns, aggregates and audits.
The global exporter is unchanged.
"""
import argparse
import copy
import json
import math
import os
import subprocess
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from experiment_io import atomic_json, publish_torch, safe_tag, sha256_file
from discrete_diffusion.diffusion_transformer import CATEGORY_DECODING_VERSION
from tools import ablation_worker as worker
from tools.ablation_common import evaluate
from tools.baseline_common import precision

SEEDS = (135398, 135399, 135400)
CITIES = ('NewYork_PO1_OOD', 'Istanbul_PO1_OOD')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def fingerprint():
    paths = list(ROOT.glob('*.py'))
    for folder in ('tools', 'tests', 'config', 'add_thin', 'discrete_diffusion', 'evaluations'):
        paths += [p for p in (ROOT / folder).rglob('*') if p.is_file() and p.suffix in ('.py', '.yaml', '.yml')]
    return {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in sorted(paths)}


def mean_sd(values):
    """Three-seed mean and sample standard deviation; None-safe per metric."""
    clean = [v for v in values if v is not None and math.isfinite(v)]
    if len(clean) != len(values) or not clean:
        return dict(mean=None, sample_sd=None, n=len(clean))
    m = sum(clean) / len(clean)
    sd = math.sqrt(sum((v - m) ** 2 for v in clean) / (len(clean) - 1)) if len(clean) > 1 else 0.0
    return dict(mean=m, sample_sd=sd, n=len(clean))


def aggregate(per_seed_metrics):
    keys = sorted({k for m in per_seed_metrics for k in m})
    return {k: mean_sd([m.get(k) for m in per_seed_metrics]) for k in keys}


def build_report(summary, manifest):
    lines = ['# Formal three-seed category-consistent decoding comparison', '',
             'Off: reused sealed three-seed Full (not resampled). On: sampled-category-poi-v2, regenerated',
             'on each sealed shard job/cache. Paired on the shared time cache (On covers every cached slot;',
             'Off is an ordered subset). Means are three fixed sampling seeds (135398/135399/135400); the',
             'sample standard deviation reflects sampling variation under a fixed checkpoint, not independent',
             'training runs. Istanbul base training used batch=512 vs NewYork batch=64. Global exporter unchanged.', '']
    cols = ['category_poi_mismatch_rate', 'strict_ovr', 'pair_coverage', 'category_coverage', 'Unsat',
            'Distance', 'Radius', 'CategoryTransition', 'DailyLoc', 'Category', 'G-RANK']
    for city in CITIES:
        agg = summary[city]['aggregate']
        lines += [f'## {city}', '', '| Metric | Off (mean±sd) | On (mean±sd) | On-matched (mean±sd) |', '|---|---:|---:|---:|']
        for key in cols:
            def cell(group):
                s = agg[group].get(key, {})
                m, sd = s.get('mean'), s.get('sample_sd')
                return '—' if m is None else f'{m:.4f} ± {sd:.4f}'
            lines.append(f'| {key} | {cell("off")} | {cell("on")} | {cell("on_matched")} |')
        lines += ['', f'Restored events (On resolved slots Off dropped) by seed: {summary[city]["restored_events_by_seed"]}.', '']
    lines += ['On changes trajectory length, so On-matched gives the length-aligned comparison.',
              'Test-side formal comparison on fixed checkpoints; detail in summary.json.']
    return '\n'.join(lines) + '\n'


def shard_jobs(registry, out, manifest_sha):
    """Build the twelve (city,seed,rank) On jobs; verify sealed Off shards + caches."""
    jobs = []
    for city in CITIES:
        for seed in SEEDS:
            entry = registry[f'{city}/{seed}/full']
            for rank, shard_ref in enumerate(entry['source_shards']):
                shard_path = Path(shard_ref['path'])
                if sha256_file(shard_path) != shard_ref['sha256']:
                    raise RuntimeError(f'Sealed Off shard changed: {shard_path}')
                off_shard = torch.load(shard_path, map_location='cpu', weights_only=False)
                job = copy.deepcopy(off_shard['job'])
                if sha256_file(Path(job['cache'])) != job['cache_sha256']:
                    raise RuntimeError(f"Original time cache changed: {job['cache']}")
                rank_dir = out / city / f'seed-{seed}' / f'rank-{rank}'
                rank_dir.mkdir(parents=True, exist_ok=True)
                job.update(output_dir=str(rank_dir), manifest_sha256=manifest_sha,
                           category_consistent_decoding=True,
                           category_consistent_decoding_version=CATEGORY_DECODING_VERSION)
                atomic_json(rank_dir / 'on-job.json', job)
                jobs.append(dict(city=city, seed=seed, rank=rank, rank_dir=rank_dir, job=job,
                                 cost=float(entry['timing']['spatial_wall_seconds'])))
    return jobs


def shard_complete(spec):
    """True if the shard's On payload already exists, complete, and bound to this job."""
    payload, result = spec['rank_dir'] / 'payload.pkl', spec['rank_dir'] / 'result.json'
    if not (payload.exists() and result.exists()):
        return False
    r = read(result)
    if r.get('state') != 'complete' or r.get('samples') != len(spec['job']['indices']):
        return False
    return torch.load(payload, map_location='cpu', weights_only=False).get('job') == spec['job']


def dispatch(jobs, gpus, env_base):
    """Run ablation_worker.py per shard across GPUs; longest-first greedy, one slot per GPU."""
    pending = sorted([j for j in jobs if not shard_complete(j)], key=lambda j: -j['cost'])
    running, free = {}, list(gpus)
    while pending or running:
        while free and pending:
            spec, gpu = pending.pop(0), free.pop(0)
            env = dict(env_base, CUDA_VISIBLE_DEVICES=str(gpu))
            log = (spec['rank_dir'] / 'worker.log').open('w')
            proc = subprocess.Popen([sys.executable, '-u', '-B', 'tools/ablation_worker.py',
                                     '--job', str(spec['rank_dir'] / 'on-job.json')],
                                    cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
            running[gpu] = (spec, proc, log)
        time.sleep(5)
        for gpu, (spec, proc, log) in list(running.items()):
            if proc.poll() is not None:
                log.close(); del running[gpu]; free.append(gpu)
                if proc.returncode != 0 or not shard_complete(spec):
                    raise RuntimeError(f"On shard failed: {spec['city']}/{spec['seed']}/rank-{spec['rank']} "
                                       f"(see {spec['rank_dir'] / 'worker.log'})")


def align_city_seed(city, seed, registry, out, raw):
    """Load both ranks' On payloads, align each against sealed Off, merge, evaluate."""
    from tools.category_cache_audit import audit_alignment, audit_traces
    entry = registry[f'{city}/{seed}/full']
    mapping, gps = raw['poi_category'], raw['poi_gps']
    off_all, on_all, matched_all, indices_all, restored = [], [], [], [], 0
    for rank, shard_ref in enumerate(entry['source_shards']):
        off_shard = torch.load(Path(shard_ref['path']), map_location='cpu', weights_only=False)
        rank_dir = out / city / f'seed-{seed}' / f'rank-{rank}'
        payload = torch.load(rank_dir / 'payload.pkl', map_location='cpu', weights_only=False)
        job = payload['job']
        cache = torch.load(Path(job['cache']), map_location='cpu', weights_only=False)
        worker.check_cache(cache, job)
        alignment, matched = audit_alignment(cache['batches'], off_shard['sequences'],
                                             payload['sequences'], mapping, gps)
        audit_traces(cache['batches'], off_shard['batch_traces'], payload['batch_traces'])
        off_all += off_shard['sequences']; on_all += payload['sequences']; matched_all += matched
        indices_all += job['indices']; restored += alignment['restored_events']
    order = sorted(range(len(indices_all)), key=lambda i: indices_all[i])
    idx = [indices_all[i] for i in order]
    if idx != sorted(idx) or len(set(idx)) != len(idx):
        raise RuntimeError('Merged indices are not a unique ordered cover')
    refs = [raw['sequences'][i] for i in idx]
    off = [off_all[i] for i in order]; on = [on_all[i] for i in order]; mt = [matched_all[i] for i in order]
    off_m, _ = evaluate(refs, off, mapping, idx)
    on_m, _ = evaluate(refs, on, mapping, idx)
    mt_m, _ = evaluate(refs, mt, mapping, idx)
    return dict(city=city, seed=seed, samples=len(idx), restored_events=restored,
                off=off_m, on=on_m, on_matched=mt_m)


def execute(args, out):
    precision()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no CPU/smaller-batch substitution')
    parent = Path(args.parent).resolve()
    if ROOT.resolve() == parent.parent.parent or out.resolve().is_relative_to(parent):
        raise RuntimeError('Separate code and output deployment required')
    if read(parent / 'status.json')['state'] != 'server-complete' or read(parent / 'audit.json')['state'] != 'passed':
        raise RuntimeError('Parent is not complete and audited')
    registry = read(parent / 'registry.json')
    parent_manifest = read(parent / 'manifest.json')
    seal = read(parent / 'delivery-files.json')
    for city in CITIES:
        for seed in SEEDS:
            if f'{city}/{seed}/full' not in registry:
                raise RuntimeError(f'Parent missing sealed Full: {city}/{seed}')
    raw_by_city = {}
    for city in CITIES:
        profile = parent_manifest['profiles'][city]
        data_path = Path(profile['data_root']) / city / f'{city}_test.pkl'
        if sha256_file(data_path) != profile['data_sha256']['test']:
            raise RuntimeError(f'Reference data changed: {data_path}')
        raw_by_city[city] = torch.load(data_path, map_location='cpu', weights_only=False)
    manifest = dict(revision='ny-ist-category-formal-v1', run_id=args.run_id, seeds=list(SEEDS), cities=list(CITIES),
                    parent_run=str(parent), parent_manifest_sha256=seal['manifest_sha256'],
                    category_consistent_decoding_version=CATEGORY_DECODING_VERSION, source_sha256=fingerprint(),
                    no_training=True, off_reused=True, precision='FP32; TF32 disabled', batch_size=64, gpus=str(args.gpus),
                    interpretation='Test-side formal three-seed On/Off category-decoding comparison; Off reused, not resampled.')
    if (out / 'manifest.json').exists():
        if not args.resume or read(out / 'manifest.json') != manifest:
            raise RuntimeError('Resume requires an identical manifest')
    else:
        atomic_json(out / 'manifest.json', manifest)
    manifest_sha = sha256_file(out / 'manifest.json')
    jobs = shard_jobs(registry, out, manifest_sha)
    gpus = [int(g) for g in str(args.gpus).split(',') if g != '']
    if not gpus:
        raise RuntimeError('At least one GPU index is required')
    dispatch(jobs, gpus, os.environ.copy())
    summary = {}
    for city in CITIES:
        per_seed = [align_city_seed(city, seed, registry, out, raw_by_city[city]) for seed in SEEDS]
        summary[city] = dict(per_seed=per_seed,
                             restored_events_by_seed={r['seed']: r['restored_events'] for r in per_seed},
                             aggregate=dict(off=aggregate([r['off'] for r in per_seed]),
                                            on=aggregate([r['on'] for r in per_seed]),
                                            on_matched=aggregate([r['on_matched'] for r in per_seed])))
    atomic_json(out / 'summary.json', summary)
    (out / 'report.md').write_text(build_report(summary, manifest), encoding='utf-8')
    if read(parent / 'status.json')['state'] != 'server-complete' or fingerprint() != manifest['source_sha256']:
        raise RuntimeError('Parent or running code changed during the run')
    skip = ('audit.json', 'status.json', 'pipeline.lock', 'worker.log', 'progress.json')
    files = {p.relative_to(out).as_posix(): sha256_file(p) for p in out.rglob('*')
             if p.is_file() and p.name not in skip}
    atomic_json(out / 'audit.json', dict(state='passed', cities=list(CITIES), seeds=list(SEEDS),
                off_reused=True, no_training=True, parent_unchanged=True, gpus=str(args.gpus), files=files))
    atomic_json(out / 'status.json', dict(state='complete', phase='complete', pid=os.getpid()))
    print(json.dumps(dict(state='complete', summary_sha256=sha256_file(out / 'summary.json'),
          audit_sha256=sha256_file(out / 'audit.json'))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', help='Sealed two-city delivery run dir with 3-seed Full in registry')
    parser.add_argument('--run-id', default='ny-ist-category-formal-v1-20261011')
    parser.add_argument('--gpus', default='0,1', help='Comma-separated GPU indices for On sampling')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--verify-only', type=Path, help='Read-only verification of a downloaded directory')
    args = parser.parse_args()
    if args.verify_only:
        out = args.verify_only
        audit = read(out / 'audit.json')
        if audit['state'] != 'passed':
            raise RuntimeError('Audit not passed')
        for name, expected in audit['files'].items():
            if sha256_file(out / name) != expected:
                raise RuntimeError('Local hash mismatch: ' + name)
        print(json.dumps(dict(state='passed', verified_files=len(audit['files']),
              summary_sha256=sha256_file(out / 'summary.json'))))
        return
    if not args.parent:
        parser.error('--parent is required')
    out = ROOT / 'experiment_runs' / safe_tag(args.run_id)
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
