"""Fresh two-city distance-v2 sampling: 66 results, no training or historical samples."""
import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import wandb
from experiment_io import atomic_json, publish_torch, sha256_file, safe_tag
from distance_kl import load_distance_reference, distance_metadata, validate_distance_metadata
from evaluations.statistical_metrics import EVALUATION_VERSION
from tools.ablation_common import DISTANCE_VERSION, DISTANCE_VARIANTS, VERSION, SEEDS, STEPS, evaluate, same_records
from tools.baseline_common import precision
from tools.city_common import profile, load_city, verify_tracking
from tools.run_two_city_study import Study, NY, IST, NY_CFG_SHA
from tools.resume_two_city_emptyfix import corrected_fixture
from tools.unified_tables import key

REVISION = 'two-city-distance-v2'
IST_CFG_SHA = '8ef158ed4d853bd2f7a3fe57f6278b4f3cef8d739c7de58cbaf8e26a6e8c3fb2'
SAMPLING_VARIANTS = ('no_projection', 'full', 'no_existence', 'no_order', 'no_kl',
                     'fixed_multipliers', 'no_gumbel', 'no_distance_kl', 'energy', 'cfg')
ALL_VARIANTS = (*SAMPLING_VARIANTS, 'postswap')


def expected_keys():
    return {key(d, s, v) for d in (NY, IST) for s in SEEDS for v in ALL_VARIANTS}


def validate_trace(payload, reference_sha):
    """Audit the actual worker trace, including disabled objectives and independent RNG."""
    job, result = payload['job'], payload['result']
    implementation = validate_distance_metadata(job, allow_historical=True)
    historical = 'distance_backend' not in job and 'distance_implementation_version' not in job
    validate_distance_metadata(result, expected=implementation, allow_historical=historical)
    if job['projection_revision'] != DISTANCE_VERSION or result['projection_revision'] != DISTANCE_VERSION:
        raise RuntimeError('Wrong projection revision')
    variant = job['variant']
    setting = DISTANCE_VARIANTS.get(variant)
    actual = result['projection_settings']
    if setting is None:
        if actual is not None or result['optimizer_steps'] != 0:
            raise RuntimeError('A baseline unexpectedly used projection')
    else:
        expected = dict(token_kl_weight=setting['kl'], distance_kl_weight=setting['distance'],
            order_weight=setting['order'], existence_weight=setting['existence'],
            gumbel=setting['gumbel'], update_multipliers=setting['update'], distance_paths=8,
            distance_topk=32, distance_bins=32, distance_temperature=1, outer=10, inner=50, early_stop=False)
        if actual != expected:
            raise RuntimeError('Actual projector configuration differs from declared protocol')
    gradients = []
    for trace in payload['batch_traces']:
        if not trace['global_rng_unchanged']:
            raise RuntimeError('Global RNG was modified')
        calls = [c for c in trace['projection_stats'] if c['optimizer_steps']]
        expected = STEPS if setting and trace['effective_constraints'] else []
        if [c['diffusion_step'] for c in calls] != expected or any(c['optimizer_steps'] != 500 for c in calls):
            raise RuntimeError('Actual projection update budget differs')
        enabled = bool(setting and setting['distance'])
        if bool(trace.get('distance_rng_before')) != enabled:
            raise RuntimeError('Distance random stream enabled for wrong variant')
        for call in calls:
            if enabled:
                validate_distance_metadata(call, expected=implementation, allow_historical=historical)
                if call.get('distance_reference_sha256') != reference_sha:
                    raise RuntimeError('Distance reference fingerprint mismatch')
                estimator = 'straight-through-mc' if setting['gumbel'] else 'expected-route'
                if call['distance_estimator'] != estimator:
                    raise RuntimeError('Incorrect distance estimator')
                if not all(math.isfinite(call[k]) for k in ('distance_kl', 'distance_poi_gradient_norm')):
                    raise FloatingPointError('Non-finite distance diagnostics')
                if call['distance_active_rows']:
                    gradients.append(call['distance_poi_gradient_norm'])
            elif 'distance_reference_sha256' in call:
                raise RuntimeError('Disabled distance objective was evaluated')
        if enabled and calls and any(c.get('distance_active_rows') for c in calls):
            changed = trace['distance_rng_before'] != trace['distance_rng_after']
            if changed != setting['gumbel']:
                raise RuntimeError('Distance RNG does not match stochastic/deterministic estimator')
    return gradients


class DistanceStudy(Study):
    def __init__(self, args):
        self.args = args
        self.distance_implementation = distance_metadata(getattr(args, 'distance_backend', 'legacy'))
        if self.distance_implementation['distance_backend'] == 'batched' and args.run_id == REVISION:
            raise ValueError('Batched backend requires a new independent --run-id')
        self.out = ROOT / 'experiment_runs' / safe_tag(args.run_id)
        manifest_path = self.out / 'manifest.json'
        if manifest_path.exists():
            validate_distance_metadata(json.loads(manifest_path.read_text()), expected=self.distance_implementation)
        self.out.mkdir(parents=True, exist_ok=True)
        self.phase, self.registry, self.cfg_cost = 'precheck', {}, {}
        self.delivery = args.delivery_root.rstrip('/').replace('\\', '/')
        self.projection_revision = DISTANCE_VERSION

    def status(self, state, **extra):
        record = dict(state=state, phase=self.phase, pid=os.getpid(), updated_at=time.time(),
            completed_unique_results=len(self.registry), target_unique_results=66, **extra)
        atomic_json(self.out / 'status.json', record)
        print(json.dumps(record), flush=True)

    def fingerprints(self):
        paths = list(ROOT.glob('*.py'))
        for name in ('add_thin', 'discrete_diffusion', 'evaluations', 'tools', 'tests', 'config'):
            paths += [p for p in (ROOT / name).rglob('*') if p.is_file() and p.suffix in ('.py', '.yaml')]
        return {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in sorted(paths)}

    def precheck(self):
        precision()
        if torch.cuda.device_count() != 2 or shutil.disk_usage(ROOT).free < 5 * 1024**3:
            raise RuntimeError('Require two GPUs and at least 5 GiB free')
        source_path = Path(self.args.input_manifest).resolve()
        source = json.loads(source_path.read_text())
        self.profiles = source['profiles']
        for dataset, p in self.profiles.items():
            if profile(dataset, p['checkpoint'], p['config_path'], p['data_root']) != p:
                raise RuntimeError('Original city profile identity changed')
        if set(self.profiles) != {NY, IST}:
            raise RuntimeError('Expected the two approved cities')
        self.source = Path(self.profiles[NY]['checkpoint']).parent
        self.ny_ablation = Path(source['source_ny_ablation'])  # Identity only; never read samples/caches.
        self.cfg = {NY: Path(source['newyork_cfg']), IST: source_path.parent / IST / 'cfg-training/last.ckpt'}
        for dataset, expected in ((NY, NY_CFG_SHA), (IST, IST_CFG_SHA)):
            if sha256_file(self.cfg[dataset]) != expected:
                raise RuntimeError('Approved CFG checkpoint hash mismatch')
        self.references = {d: load_distance_reference(Path(p['data_root']) / d / f'{d}_train.pkl')
                           for d, p in self.profiles.items()}
        source_training = json.loads((self.source / 'manifest.json').read_text())
        self.entity = source_training['entity']
        packages = {name: importlib.metadata.version(name) for name in source_training['packages']}
        if packages != source_training['packages']:
            raise RuntimeError('Frozen experiment package versions changed')
        self.manifest = dict(version=REVISION, projection_revision=DISTANCE_VERSION,
            **self.distance_implementation,
            evaluation_version=EVALUATION_VERSION, run_id=self.args.run_id, profiles=self.profiles,
            input_manifest=str(source_path), input_manifest_sha256=sha256_file(source_path),
            cfg_checkpoints={d: dict(path=str(p), sha256=sha256_file(p)) for d, p in self.cfg.items()},
            distance_references={d: dict(sha256=r.fingerprint, route_count=r.route_count,
                centers=r.centers.tolist(), target=r.target.tolist(), bandwidth=r.bandwidth)
                for d, r in self.references.items()}, code_sha256=self.fingerprints(), packages=packages,
            seeds=list(SEEDS), variants=DISTANCE_VARIANTS, sampling_variants=list(SAMPLING_VARIANTS),
            methods=['no_projection', 'postswap', 'energy', 'cfg', 'full'],
            total_unique_results=66, total_trajectory_records=231726, temporal_cache_shards=12,
            batch_size=64, world_size=2, projection_steps=STEPS,
            fixed_budget=dict(outer=10, inner=50, early_stop=False, temperature=3),
            distance=dict(weight=1, paths=8, topk=32, bins=32, temperature=1),
            energy=dict(temperature=3, scale=100, existence_weight=5, last_k=40, frequency=4),
            cfg_scale=1, rng_version=VERSION, paired_temporal_inputs=True,
            fresh_temporal_caches=True, reused_spatial_results=False, retrained=False,
            hyperparameter_search=False, downstream_tasks=False, max_gpu_memory_fraction=.8,
            precision='FP32 model; legacy normalization retained; no autocast or TF32',
            entity=self.entity, project='Marionette', delivery_root=self.delivery)
        path = self.out / 'manifest.json'
        if path.exists():
            if not self.args.resume or json.loads(path.read_text()) != self.manifest:
                raise RuntimeError('Resume requires identical code/inputs and explicit --resume')
        else:
            if self.args.resume or any(p.name != 'pipeline.lock' for p in self.out.iterdir()):
                raise RuntimeError('Cannot initialize over existing files or resume without manifest')
            atomic_json(path, self.manifest)
        self.manifest_sha = sha256_file(path)
        if (self.out / 'registry.json').exists():
            self.registry = json.loads((self.out / 'registry.json').read_text())
            if not set(self.registry).issubset(expected_keys()):
                raise RuntimeError('Foreign results in registry')
            for r in self.registry.values():
                if r['origin_manifest_sha256'] != self.manifest_sha or sha256_file(r['path']) != r['sha256']:
                    raise RuntimeError('Registered result changed')
        self.raw = {d: torch.load(Path(p['data_root']) / d / f'{d}_test.pkl', map_location='cpu', weights_only=False)
                    for d, p in self.profiles.items()}
        self.status('running')
        self.tables()
        if self.args.sync_only:
            return
        self.phase = 'server-regression-tests'; self.status('running')
        with (self.out / f'unit-tests-{time.time_ns()}.log').open('w') as log:
            subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
                           stdout=log, stderr=subprocess.STDOUT, check=True, cwd=ROOT)
        self.phase = 'strict-checkpoint-loads'; self.status('running')
        for d in (NY, IST):
            for cfg in (None, self.cfg[d]):
                task, dm = load_city(self.profiles[d], cfg, device='cpu')
                if any(not torch.isfinite(p).all() for p in task.parameters()):
                    raise FloatingPointError('Non-finite checkpoint parameter')
                del task, dm
        atomic_json(self.out / 'strict-loads.json', dict(state='passed', checkpoints=4))

    def city_job(self, dataset, *args, **kwargs):
        job = super().city_job(dataset, *args, **kwargs)
        job.update(projection_revision=DISTANCE_VERSION, max_gpu_memory_fraction=.8,
                   **self.distance_implementation,
                   distance_reference_sha256=self.references[dataset].fingerprint)
        return job

    def cache_paths(self, dataset, seed):
        n = self.profiles[dataset]['test_count']; half = (n + 1) // 2
        jobs = [self.city_job(dataset, self.out / dataset / 'time-cache' / f'seed-{seed}-rank-{r}',
                    'cache', list(range(a, b)), seed, r, start=a)
                for r, (a, b) in enumerate(((0, half), (half, n)))]
        self.run_jobs(jobs)
        return [Path(j['output_dir']) / 'payload.pkl' for j in jobs], self.manifest_sha

    def reference_trace_paths(self, dataset, seed, rank):
        return self.out / dataset / f'seed-{seed}' / 'no_projection' / f'rank-{rank}' / 'payload.pkl'

    def publish_result(self, dataset, seed, variant, data, timing, shards):
        folder = self.out / dataset / f'seed-{seed}' / variant
        folder.mkdir(parents=True, exist_ok=True)
        indices = list(range(self.profiles[dataset]['test_count']))
        if data['test_indices'] != indices or len(data['sequences']) != len(indices):
            raise RuntimeError('Incomplete city output')
        diagnostics = {}
        raw = self.raw[dataset]
        metrics, rows = evaluate(raw['sequences'], data['sequences'], raw['poi_category'], indices, diagnostics=diagnostics)
        path = folder / 'generated.pkl'
        if path.exists():
            old = torch.load(path, map_location='cpu', weights_only=False)
            if (old['manifest_sha256'] != self.manifest_sha or old['source_shards'] != shards
                    or not same_records(old['sequences'], data['sequences'])):
                raise RuntimeError('Existing merged output differs')
        else:
            publish_torch(path, dict(data, manifest_sha256=self.manifest_sha, dataset=dataset,
                seed=seed, variant=variant, source_shards=shards, t_max=24.0))
        digest = sha256_file(path)
        if (folder / 'metrics.json').exists():
            old = json.loads((folder / 'metrics.json').read_text())
            if old['metrics'] != metrics or old['output_sha256'] != digest:
                raise RuntimeError('Existing merged evaluation differs')
            timing = old['timing']
        atomic_json(folder / 'metrics.json', dict(metrics=metrics, metric_diagnostics=diagnostics,
                    timing=timing, output_sha256=digest))
        atomic_json(folder / 'per_condition.json', rows)
        self.register(dataset, seed, variant, path, metrics, timing, 'new', self.manifest_sha, shards)
        self.tables()
        # Tracking errors never roll back a numerical result or restart a worker.
        self.sync_record(self.registry[key(dataset, seed, variant)])

    def sync_record(self, record):
        folder = Path(record['path']).parent
        receipt = folder / 'wandb-sync.json'
        expected = dict(complete=True, result_sha256=record['sha256'])
        if receipt.exists():
            proof = json.loads(receipt.read_text())
            if proof['state'] != 'verified' or proof['expected'] != expected:
                raise RuntimeError('Tracking receipt mismatch')
            return True
        dataset, seed, variant = record['dataset'], record['seed'], record['variant']
        run_id = f"{self.args.run_id}-{'ny' if dataset == NY else 'ist'}-{seed}-{variant}"
        run = None
        try:
            run = wandb.init(project='Marionette', entity=self.entity, id=run_id, name=run_id,
                group=self.args.run_id, job_type=REVISION, mode='online', resume='allow',
                dir=str(self.out), save_code=False, settings=wandb.Settings(init_timeout=60),
                config=dict(dataset=dataset, seed=seed, variant=variant,
                            study_manifest_sha256=self.manifest_sha, projection_revision=DISTANCE_VERSION))
            values = dict(record['metrics'], **record['timing'])
            run.log(values); run.summary.update(dict(values, **expected))
            url = run.url; run.finish(); run = None
            verify_tracking(self.entity, run_id, expected, receipt, url)
            return True
        except Exception as exc:
            atomic_json(folder / 'wandb-pending.json', dict(state='retry-sync-only', run_id=run_id,
                        error=str(exc), updated_at=time.time(), result_sha256=record['sha256']))
            print(f'Tracking deferred for {run_id}: {type(exc).__name__}', flush=True)
            return False
        finally:
            if run is not None:
                run.finish(exit_code=1)

    def preflight_city(self, dataset):
        root = self.out / 'preflight' / dataset
        self.phase = f'{dataset}-preflight-cache'; self.status('running')
        cache = root / 'time-cache' / 'payload.pkl'
        self.run_jobs([self.city_job(dataset, cache.parent, 'cache', list(range(64)), SEEDS[0], 0, split='train')])
        fixture = corrected_fixture(torch.load(cache, map_location='cpu', weights_only=False), sha256_file(cache))
        empty = root / 'empty-fixture' / 'payload.pkl'; proof_path = empty.parent / 'receipt.json'
        if empty.exists():
            proof = json.loads(proof_path.read_text())
            if proof['source_sha256'] != sha256_file(cache) or proof['sha256'] != sha256_file(empty):
                raise RuntimeError('Synthetic empty fixture changed')
        else:
            publish_torch(empty, fixture)
            atomic_json(proof_path, dict(source_sha256=sha256_file(cache), sha256=sha256_file(empty)))
        row = fixture['test_only_synthetic_empty_row']
        timing = {}
        for variant in SAMPLING_VARIANTS:
            self.phase = f'{dataset}-preflight-{variant}'; self.status('running')
            jobs = [self.city_job(dataset, root / variant / str(gpu), 'sample', list(range(64)), SEEDS[0], gpu,
                        split='train', variant=variant, cache=source, cfg=self.cfg[dataset] if variant == 'cfg' else None,
                        empty_row=row if gpu else None) for gpu, source in enumerate((cache, empty))]
            results = self.run_jobs(jobs)
            timing[variant] = results[0]['spatial_seconds']
            for gpu, job in enumerate(jobs):
                path = Path(job['output_dir']) / 'payload.pkl'
                payload = torch.load(path, map_location='cpu', weights_only=False)
                gradients = validate_trace(payload, self.references[dataset].fingerprint)
                if DISTANCE_VARIANTS.get(variant, {}).get('distance') if DISTANCE_VARIANTS.get(variant) else False:
                    if not gradients or max(gradients) <= 0:
                        raise RuntimeError('Preflight distance term has no effective POI gradient')
                if variant in DISTANCE_VARIANTS and variant != 'no_projection' and results[gpu]['projection_calls'] != 10:
                    raise RuntimeError('Preflight did not exercise projection')
                self.verify_pairing([root / 'no_projection' / str(gpu) / 'payload.pkl', path])
        job = self.city_job(dataset, root / 'full-gpu1', 'sample', list(range(64)), SEEDS[0], 1,
                            split='train', variant='full', cache=cache)
        self.run_jobs([job])
        first = torch.load(root / 'full/0/payload.pkl', map_location='cpu', weights_only=False)
        second = torch.load(root / 'full-gpu1/payload.pkl', map_location='cpu', weights_only=False)
        if not same_records(first['sequences'], second['sequences']):
            raise RuntimeError('Full output differs across the two GPUs')
        self.verify_pairing([root / 'full/0/payload.pkl', root / 'full-gpu1/payload.pkl'])
        # Also exercise deterministic PostSwap on train-only pilot outputs.
        from baseline_posthoc_swap import apply_posthoc_swap
        train = torch.load(Path(self.profiles[dataset]['data_root']) / dataset / f'{dataset}_train.pkl',
                           map_location='cpu', weights_only=False)
        base = torch.load(root / 'no_projection/0/payload.pkl', map_location='cpu', weights_only=False)
        fixed, _ = apply_posthoc_swap(base['sequences'], train['sequences'][:64], train['poi_category'], verbose=False)
        evaluate(train['sequences'][:64], fixed, train['poi_category'], list(range(64)))
        batches_per_rank = math.ceil(math.ceil(self.profiles[dataset]['test_count'] / 2) / 64)
        atomic_json(root / 'receipt.json', dict(state='passed', manifest_sha256=self.manifest_sha,
            all_variants=list(ALL_VARIANTS), pilot_indices=list(range(64)), synthetic_empty_row=row,
            independent_rng_verified=True, full_two_gpu_equal=True, seconds_per_batch=timing,
            estimated_formal_spatial_seconds=3 * batches_per_rank * sum(timing.values()),
            estimate_note='Train-pilot estimate; test lengths and temporal sampling costs can differ.'))

    def audit(self):
        if set(self.registry) != expected_keys():
            raise RuntimeError('Expected exactly 66 results')
        count = 0
        caches = []
        for dataset in (NY, IST):
            for seed in SEEDS:
                self.verify_city_pairing(dataset, seed, list(SAMPLING_VARIANTS))
                for rank in (0, 1):
                    path = self.out / dataset / 'time-cache' / f'seed-{seed}-rank-{rank}' / 'payload.pkl'
                    result = json.loads((path.parent / 'result.json').read_text())
                    if sha256_file(path) != result['output_sha256']:
                        raise RuntimeError('Temporal cache hash mismatch')
                    cache = torch.load(path, map_location='cpu', weights_only=False)
                    if cache['manifest_sha256'] != self.manifest_sha:
                        raise RuntimeError('Historical temporal cache was reused')
                    caches.append(dict(path=str(path), sha256=result['output_sha256']))
        for record in self.registry.values():
            path = Path(record['path']); dataset = record['dataset']
            if sha256_file(path) != record['sha256']:
                raise RuntimeError('Merged output changed')
            data = torch.load(path, map_location='cpu', weights_only=False)
            indices = list(range(self.profiles[dataset]['test_count']))
            if data['test_indices'] != indices or data['manifest_sha256'] != self.manifest_sha:
                raise RuntimeError('Merged identity mismatch')
            sequences, shard_indices = [], []
            for shard in record['source_shards']:
                if sha256_file(shard['path']) != shard['sha256']:
                    raise RuntimeError('Source shard hash mismatch')
                part = torch.load(shard['path'], map_location='cpu', weights_only=False)
                sequences += part['sequences']; shard_indices += part['indices']
                if record['variant'] != 'postswap':
                    job = part['job']
                    if job['manifest_sha256'] != self.manifest_sha or sha256_file(job['cache']) != job['cache_sha256']:
                        raise RuntimeError('Time pairing or shard manifest mismatch')
                    validate_trace(part, self.references[dataset].fingerprint)
            if indices != shard_indices or not same_records(sequences, data['sequences']):
                raise RuntimeError('Merged records differ from shards')
            raw, diagnostics = self.raw[dataset], {}
            metrics, rows = evaluate(raw['sequences'], sequences, raw['poi_category'], indices, diagnostics=diagnostics)
            reported = json.loads((path.parent / 'metrics.json').read_text())
            if (metrics != record['metrics'] or metrics != reported['metrics']
                    or diagnostics != reported['metric_diagnostics']
                    or rows != json.loads((path.parent / 'per_condition.json').read_text())):
                raise RuntimeError('Whole-dataset evaluation does not reproduce')
            if record['variant'] == 'postswap':
                parent = self.registry[key(dataset, record['seed'], 'no_projection')]
                if data['parent_sha256'] != parent['sha256']:
                    raise RuntimeError('PostSwap not derived from this run')
            count += len(sequences)
        if count != 231726 or len(caches) != 12 or self.fingerprints() != self.manifest['code_sha256']:
            raise RuntimeError('Final count or code fingerprint mismatch')
        for dataset, p in self.profiles.items():
            if profile(dataset, p['checkpoint'], p['config_path'], p['data_root']) != p:
                raise RuntimeError('Original data/base checkpoint changed')
            if sha256_file(self.cfg[dataset]) != self.manifest['cfg_checkpoints'][dataset]['sha256']:
                raise RuntimeError('CFG checkpoint changed')
        atomic_json(self.out / 'audit.json', dict(state='passed', unique_results=66,
            trajectory_records=count, temporal_caches=caches, projection_revision=DISTANCE_VERSION,
            all_hashes_valid=True, original_inputs_unchanged=True, whole_dataset_metrics_recomputed=True,
            paired_spatial_rng=True, distance_switches_verified=True, full_ranking_not_an_acceptance_condition=True))

    def execute(self):
        self.precheck()
        if not self.args.sync_only:
            for dataset in (NY, IST):
                self.preflight_city(dataset)
            estimates = {d: json.loads((self.out / 'preflight' / d / 'receipt.json').read_text())
                         ['estimated_formal_spatial_seconds'] for d in (NY, IST)}
            atomic_json(self.out / 'runtime-estimate.json', dict(seconds_by_city=estimates,
                estimated_total_spatial_hours=sum(estimates.values()) / 3600,
                note='Includes three seeds and two-GPU sharding; excludes temporal generation and reporting.'))
            for dataset in (NY, IST):
                for seed in SEEDS:
                    for variant in SAMPLING_VARIANTS:
                        self.sampling(dataset, seed, variant, self.cfg[dataset] if variant == 'cfg' else None)
                        if variant == 'no_projection':
                            self.postswap(dataset, seed)
                    self.verify_city_pairing(dataset, seed, list(SAMPLING_VARIANTS))
        self.phase = 'final-audit'; self.status('running'); self.audit()
        self.phase = 'tracking-sync'; self.status('running')
        pending = [k for k, r in self.registry.items() if not self.sync_record(r)]
        if pending:
            self.status('tracking-pending', pending=pending)
            return
        self.tables(complete=True)
        self.phase = 'server-complete-awaiting-local-verification'
        self.status('server-complete', local_delivery_verified=False)
        files = {p.relative_to(self.out).as_posix(): sha256_file(p) for p in sorted(self.out.rglob('*'))
                 if p.is_file() and p.name not in ('pipeline.lock', 'delivery-files.json')
                 and 'wandb' not in p.relative_to(self.out).parts}
        atomic_json(self.out / 'delivery-files.json', dict(state='sealed', manifest_sha256=self.manifest_sha, files=files))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-manifest', default='/root/experiments/pcdg/two-city-v1/experiment_runs/two-city-v1/manifest.json')
    parser.add_argument('--run-id', default=REVISION)
    parser.add_argument('--distance-backend', choices=('legacy', 'batched'), default='legacy')
    parser.add_argument('--delivery-root', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--sync-only', action='store_true')
    args = parser.parse_args()
    if args.sync_only and not args.resume:
        parser.error('--sync-only requires --resume')
    os.chdir(ROOT)
    study = DistanceStudy(args)
    import fcntl
    with (study.out / 'pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            study.execute()
        except BaseException as exc:
            study.status('failed', error_type=type(exc).__name__, error=str(exc))
            raise


if __name__ == '__main__':
    main()
