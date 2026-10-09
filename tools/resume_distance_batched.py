"""Audited successor run: retain completed legacy results, batch unfinished jobs.

Never edit the parent run or relabel its bytes as batched results. The successor
contains a verified byte-copy of the frozen parent, with per-result provenance.
"""
import argparse
import copy
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from distance_kl import distance_metadata, load_distance_reference, validate_distance_metadata
from experiment_io import atomic_json, sha256_file, safe_tag
from tools.ablation_common import DISTANCE_VERSION, SEEDS, evaluate, same_records
from tools.ablation_worker import check_cache
from tools.baseline_common import precision
from tools.city_common import profile
from tools.run_two_city_distance_v2 import (DistanceStudy, ALL_VARIANTS, SAMPLING_VARIANTS,
    expected_keys, validate_trace)
from tools.run_two_city_study import NY, IST
from tools.unified_tables import key

REVISION = 'distance-batched-successor-v1'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def child(root, relative):
    root = Path(root).resolve()
    path = root / relative
    if not path.resolve().is_relative_to(root) or path.is_symlink():
        raise RuntimeError('Snapshot path escapes its root')
    return path


def inventory(root):
    root = Path(root)
    files = {}
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root)
        if 'wandb' in relative.parts or path.name == 'pipeline.lock':
            continue
        if path.is_symlink():
            raise RuntimeError('Snapshot must contain regular files, not symlinks')
        if path.is_file():
            files[relative.as_posix()] = sha256_file(path)
    return files


def copy_verified(source, destination, expected):
    source, destination = Path(source), Path(destination)
    if sha256_file(source) != expected:
        raise RuntimeError('Parent artifact changed before copying')
    if destination.exists():
        if sha256_file(destination) != expected:
            raise RuntimeError('Different successor snapshot file already exists')
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False, suffix='.copying') as stream:
        temporary = Path(stream.name)
        with source.open('rb') as original:
            shutil.copyfileobj(original, stream, 1024 * 1024)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        if sha256_file(temporary) != expected or sha256_file(source) != expected:
            raise RuntimeError('Parent changed or snapshot copy failed verification')
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def require_parent_stopped(parent):
    """A live parent/worker may be stopped, but must not be writing this run."""
    proc = Path('/proc')
    if not proc.is_dir():
        raise RuntimeError('Successor execution requires Linux process-state verification')
    code_root = Path(parent).resolve().parent.parent
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            relevant = ('run_two_city_distance_v2.py' in command or 'ablation_worker.py' in command)
            if relevant and (entry / 'cwd').resolve() == code_root:
                state = next(line for line in (entry / 'status').read_text().splitlines() if line.startswith('State:'))
                if '\tT' not in state and '\tt' not in state:
                    raise RuntimeError(f'Parent process is still running: {entry.name}')
        except (FileNotFoundError, ProcessLookupError):
            continue


def snapshot_parent(parent, destination, receipt):
    parent, destination, receipt = map(Path, (parent, destination, receipt))
    if destination.resolve().is_relative_to(parent.resolve()):
        raise RuntimeError('Successor must be outside the immutable parent run')
    files = inventory(parent)
    proof = dict(revision=REVISION, parent_run=str(parent.resolve()), files=files,
                 excluded=['wandb', 'pipeline.lock'])
    if receipt.exists() and read_json(receipt) != proof:
        raise RuntimeError('Parent snapshot inventory changed')
    for relative, digest in files.items():
        copy_verified(child(parent, relative), child(destination, relative), digest)
    if inventory(parent) != files or inventory(destination) != files:
        raise RuntimeError('Snapshot inventory verification failed')
    if not receipt.exists():
        atomic_json(receipt, proof)
    return proof


def inherited_record(record, parent, snapshot, delivery):
    result = copy.deepcopy(record)
    original = Path(record['path'])
    relative = original.relative_to(parent)
    result.update(path=str(child(snapshot, relative)), origin='inherited-legacy',
        parent_result_path=str(original),
        delivery_path=delivery + '/inherited/legacy/' + relative.as_posix(),
        **distance_metadata('legacy'))
    for shard in result['source_shards']:
        shard['path'] = str(child(snapshot, Path(shard['path']).relative_to(parent)))
    return result


class SuccessorStudy(DistanceStudy):
    def __init__(self, args):
        args.distance_backend = 'batched'
        super().__init__(args)
        self.parent = Path(args.parent_run).resolve()
        if self.out.resolve() == self.parent or self.out.resolve().is_relative_to(self.parent):
            raise RuntimeError('A successor cannot overwrite/nest in its parent run')
        self.snapshot = self.out / 'inherited' / 'legacy'
        self.snapshot_receipt = self.out / 'parent-snapshot.json'

    def canonical_artifact(self, path):
        path = Path(path)
        if path.is_relative_to(self.parent):
            return child(self.snapshot, path.relative_to(self.parent))
        if not path.resolve().is_relative_to(self.out.resolve()):
            raise RuntimeError('Unregistered external result/cache path')
        return path

    def status(self, state, **extra):
        super().status(state, implementation_transition=REVISION,
            inherited_legacy_results=sum(r.get('origin') == 'inherited-legacy' for r in self.registry.values()),
            new_batched_results=sum(r.get('origin') == 'new' for r in self.registry.values()), **extra)

    def precheck(self):
        precision()
        if torch.cuda.device_count() != 2 or shutil.disk_usage(ROOT).free < 5 * 1024**3:
            raise RuntimeError('Require two GPUs and at least 5 GiB free')
        require_parent_stopped(self.parent)
        parent_manifest = read_json(self.parent / 'manifest.json')
        if (parent_manifest['projection_revision'] != DISTANCE_VERSION
                or parent_manifest['seeds'] != list(SEEDS) or parent_manifest['batch_size'] != 64
                or parent_manifest['fixed_budget'] != dict(outer=10, inner=50, early_stop=False, temperature=3)
                or parent_manifest.get('distance_backend', 'legacy') != 'legacy'):
            raise RuntimeError('Not the approved legacy distance protocol')
        for relative, digest in parent_manifest['code_sha256'].items():
            if sha256_file(child(self.parent.parent.parent, relative)) != digest:
                raise RuntimeError('Frozen parent source changed')
        snapshot_parent(self.parent, self.snapshot, self.snapshot_receipt)
        self.parent_manifest_sha = sha256_file(self.snapshot / 'manifest.json')
        self.parent_registry = read_json(self.snapshot / 'registry.json')
        if not self.parent_registry or not set(self.parent_registry).issubset(expected_keys()):
            raise RuntimeError('Invalid parent result registry')
        self.profiles = parent_manifest['profiles']
        for dataset, p in self.profiles.items():
            if profile(dataset, p['checkpoint'], p['config_path'], p['data_root']) != p:
                raise RuntimeError('Original input profile changed')
        self.cfg = {d: Path(p['path']) for d, p in parent_manifest['cfg_checkpoints'].items()}
        for d, path in self.cfg.items():
            if sha256_file(path) != parent_manifest['cfg_checkpoints'][d]['sha256']:
                raise RuntimeError('Original CFG checkpoint changed')
        self.source = Path(self.profiles[NY]['checkpoint']).parent
        self.ny_ablation = self.parent
        self.references = {d: load_distance_reference(Path(p['data_root']) / d / f'{d}_train.pkl')
                           for d, p in self.profiles.items()}
        for d, reference in self.references.items():
            if reference.fingerprint != parent_manifest['distance_references'][d]['sha256']:
                raise RuntimeError('Distance reference changed')
        if {n: importlib.metadata.version(n) for n in parent_manifest['packages']} != parent_manifest['packages']:
            raise RuntimeError('Frozen package versions changed')
        self.entity = parent_manifest['entity']
        self.manifest = dict(parent_manifest, version=REVISION, run_id=self.args.run_id,
            **self.distance_implementation, code_sha256=self.fingerprints(),
            parent_run=str(self.parent), parent_manifest_sha256=self.parent_manifest_sha,
            parent_snapshot_sha256=sha256_file(self.snapshot_receipt),
            inherited_result_sha256={k: r['sha256'] for k, r in self.parent_registry.items()},
            implementation_transition=REVISION, mixed_implementations=True,
            implementation_policy='Completed legacy results retained byte-for-byte; unfinished results use batched backend. Not a homogeneous-backend experiment.',
            fresh_temporal_caches=False, reused_spatial_results=True, delivery_root=self.delivery)
        manifest_path = self.out / 'manifest.json'
        if manifest_path.exists():
            if not self.args.resume or read_json(manifest_path) != self.manifest:
                raise RuntimeError('Successor resume requires identical manifest/code and explicit --resume')
        elif self.args.resume:
            raise RuntimeError('No successor manifest to resume')
        else:
            atomic_json(manifest_path, self.manifest)
        self.manifest_sha = sha256_file(manifest_path)
        self.registry = read_json(self.out / 'registry.json') if (self.out / 'registry.json').exists() else {}
        if not set(self.registry).issubset(expected_keys()):
            raise RuntimeError('Foreign results in successor registry')
        self.raw = {d: torch.load(Path(p['data_root']) / d / f'{d}_test.pkl', map_location='cpu', weights_only=False)
                    for d, p in self.profiles.items()}
        for result_key, record in self.parent_registry.items():
            if result_key != key(record['dataset'], record['seed'], record['variant']):
                raise RuntimeError('Parent registry key mismatch')
            expected = inherited_record(record, self.parent, self.snapshot, self.delivery)
            if result_key in self.registry and self.registry[result_key] != expected:
                raise RuntimeError('Inherited record was relabelled or changed')
            self.verify_record(expected)
            self.registry[result_key] = expected
        for record in self.registry.values():
            if record['origin'] != 'inherited-legacy':
                self.verify_record(record)
        self.tables()
        self.phase = 'server-regression-tests'; self.status('running')
        with (self.out / f'unit-tests-{time.time_ns()}.log').open('w') as log:
            subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        atomic_json(self.out / 'inheritance-audit.json', dict(state='passed',
            inherited_results=len(self.parent_registry), parent_snapshot_sha256=sha256_file(self.snapshot_receipt),
            original_files_unchanged=True, whole_dataset_metrics_recomputed=True))

    def register(self, dataset, seed, variant, path, metrics, timing, origin, origin_manifest, shards):
        if origin != 'new' or origin_manifest != self.manifest_sha:
            raise RuntimeError('New results require the successor manifest')
        result_key = key(dataset, seed, variant)
        if result_key in self.parent_registry:
            raise RuntimeError('Recomputing/replacing an inherited result is forbidden')
        self.registry[result_key] = dict(dataset=dataset, seed=seed, variant=variant, path=str(path),
            sha256=sha256_file(path), metrics=metrics, timing=timing, origin=origin,
            origin_manifest_sha256=origin_manifest, source_shards=shards,
            delivery_path=self.delivery + '/' + Path(path).relative_to(self.out).as_posix(),
            **self.distance_implementation)
        atomic_json(self.out / 'registry.json', self.registry)

    def cache_paths(self, dataset, seed):
        paths = [self.snapshot / dataset / 'time-cache' / f'seed-{seed}-rank-{rank}' / 'payload.pkl'
                 for rank in (0, 1)]
        if any(path.exists() for path in paths):
            if not all(path.exists() for path in paths):
                raise RuntimeError('Incomplete parent time-cache pair; no silent resampling')
            for path in paths:
                self.verify_cache(path, self.parent_manifest_sha)
            return paths, self.parent_manifest_sha
        return super().cache_paths(dataset, seed)

    def verify_cache(self, path, origin):
        result = read_json(path.parent / 'result.json')
        if result['state'] != 'complete' or sha256_file(path) != result['output_sha256']:
            raise RuntimeError('Time cache has no verified completion')
        cache = torch.load(path, map_location='cpu', weights_only=False)
        if cache['manifest_sha256'] != origin:
            raise RuntimeError('Unexpected temporal-cache origin')
        check_cache(cache, cache['job'])

    def verify_city_pairing(self, dataset, seed, variants):
        for rank in (0, 1):
            paths = [self.registry[key(dataset, seed, variant)]['source_shards'][rank]['path']
                     for variant in ('no_projection', *variants)]
            self.verify_pairing(paths)

    def preflight_city(self, dataset):
        root = self.out / 'batched-preflight' / dataset
        receipt = root / 'receipt.json'
        if receipt.exists():
            saved = read_json(receipt)
            if saved['manifest_sha256'] != self.manifest_sha:
                raise RuntimeError('Preflight belongs to different code/manifest')
            for relative, digest in saved['artifacts'].items():
                if sha256_file(child(self.out, relative)) != digest:
                    raise RuntimeError('Preflight artifact changed')
            return
        old = self.snapshot / 'preflight' / dataset
        normal = old / 'time-cache' / 'payload.pkl'
        empty = old / 'empty-fixture' / 'payload.pkl'
        fixture = torch.load(empty, map_location='cpu', weights_only=False)
        row = fixture['test_only_synthetic_empty_row']
        artifacts = {}
        for variant in ('full', 'no_gumbel'):
            self.phase = f'{dataset}-batched-preflight-{variant}'; self.status('running')
            jobs = [self.city_job(dataset, root / variant / str(gpu), 'sample', list(range(64)), SEEDS[0], gpu,
                split='train', variant=variant, cache=cache, cache_origin=self.parent_manifest_sha,
                empty_row=row if gpu else None) for gpu, cache in enumerate((normal, empty))]
            self.run_jobs(jobs)
            for gpu, job in enumerate(jobs):
                path = Path(job['output_dir']) / 'payload.pkl'
                payload = torch.load(path, map_location='cpu', weights_only=False)
                gradients = validate_trace(payload, self.references[dataset].fingerprint)
                if not gradients or max(gradients) <= 0:
                    raise RuntimeError('Real-data preflight lacks effective distance gradients')
                self.verify_pairing([old / 'no_projection' / str(gpu) / 'payload.pkl', path])
                artifacts[path.relative_to(self.out).as_posix()] = sha256_file(path)
        job = self.city_job(dataset, root / 'full-gpu1', 'sample', list(range(64)), SEEDS[0], 1,
            split='train', variant='full', cache=normal, cache_origin=self.parent_manifest_sha)
        self.run_jobs([job])
        second_path = Path(job['output_dir']) / 'payload.pkl'
        first = torch.load(root / 'full/0/payload.pkl', map_location='cpu', weights_only=False)
        second = torch.load(second_path, map_location='cpu', weights_only=False)
        if not same_records(first['sequences'], second['sequences']):
            raise RuntimeError('Batched real-data Full differs across the two GPUs')
        validate_trace(second, self.references[dataset].fingerprint)
        self.verify_pairing([root / 'full/0/payload.pkl', second_path])
        artifacts[second_path.relative_to(self.out).as_posix()] = sha256_file(second_path)
        atomic_json(receipt, dict(state='passed', manifest_sha256=self.manifest_sha,
            batch_size=64, normal_and_empty=True, deterministic_and_gumbel=True,
            full_two_gpu_equal=True, independent_rng_verified=True, artifacts=artifacts))

    def verify_record(self, record):
        result_key = key(record['dataset'], record['seed'], record['variant'])
        inherited = result_key in self.parent_registry
        if record['origin'] != ('inherited-legacy' if inherited else 'new'):
            raise RuntimeError('Result origin label does not match the transition plan')
        origin = self.parent_manifest_sha if inherited else self.manifest_sha
        implementation = distance_metadata('legacy' if inherited else 'batched')
        validate_distance_metadata(record, expected=implementation)
        if record['origin_manifest_sha256'] != origin:
            raise RuntimeError('Result manifest provenance mismatch')
        path = self.canonical_artifact(record['path'])
        if sha256_file(path) != record['sha256']:
            raise RuntimeError('Result hash mismatch')
        data = torch.load(path, map_location='cpu', weights_only=False)
        indices = list(range(self.profiles[record['dataset']]['test_count']))
        if data['test_indices'] != indices or data['manifest_sha256'] != origin:
            raise RuntimeError('Result indices/manifest mismatch')
        sequences, shard_indices = [], []
        for shard in record['source_shards']:
            source = self.canonical_artifact(shard['path'])
            if sha256_file(source) != shard['sha256']:
                raise RuntimeError('Source shard changed')
            part = torch.load(source, map_location='cpu', weights_only=False)
            sequences.extend(part['sequences']); shard_indices.extend(part['indices'])
            if part['manifest_sha256'] != origin:
                raise RuntimeError('Shard manifest provenance mismatch')
            if record['variant'] != 'postswap':
                job = part['job']
                if job['manifest_sha256'] != origin:
                    raise RuntimeError('Worker manifest provenance mismatch')
                validate_distance_metadata(job, expected=implementation, allow_historical=inherited)
                if sha256_file(self.canonical_artifact(job['cache'])) != job['cache_sha256']:
                    raise RuntimeError('Paired time-cache fingerprint changed')
                validate_trace(part, self.references[record['dataset']].fingerprint)
        if shard_indices != indices or not same_records(sequences, data['sequences']):
            raise RuntimeError('Shard/merged output mismatch')
        raw = self.raw[record['dataset']]
        diagnostics = {}
        metrics, rows = evaluate(raw['sequences'], sequences, raw['poi_category'], indices, diagnostics=diagnostics)
        reported = read_json(path.parent / 'metrics.json')
        if (metrics != record['metrics'] or metrics != reported['metrics']
                or diagnostics != reported['metric_diagnostics']
                or rows != read_json(path.parent / 'per_condition.json')):
            raise RuntimeError('Whole-dataset metrics failed to reproduce')
        if record['variant'] == 'postswap':
            source_registry = self.parent_registry if inherited else self.registry
            if data['parent_sha256'] != source_registry[key(record['dataset'], record['seed'], 'no_projection')]['sha256']:
                raise RuntimeError('PostSwap parent changed')
        return len(sequences)

    def audit(self):
        if set(self.registry) != expected_keys():
            raise RuntimeError('Expected exactly 66 successor results')
        require_parent_stopped(self.parent)
        proof = read_json(self.snapshot_receipt)
        if inventory(self.parent) != proof['files'] or inventory(self.snapshot) != proof['files']:
            raise RuntimeError('Original parent/snapshot artifacts changed')
        count = sum(self.verify_record(record) for record in self.registry.values())
        caches = []
        for dataset in (NY, IST):
            for seed in SEEDS:
                self.verify_city_pairing(dataset, seed, SAMPLING_VARIANTS)
                for rank in (0, 1):
                    relative = Path(dataset) / 'time-cache' / f'seed-{seed}-rank-{rank}' / 'payload.pkl'
                    inherited = (self.snapshot / relative).exists()
                    path = (self.snapshot if inherited else self.out) / relative
                    self.verify_cache(path, self.parent_manifest_sha if inherited else self.manifest_sha)
                    caches.append(dict(path=str(path), sha256=sha256_file(path), inherited=inherited))
            p = self.profiles[dataset]
            if profile(dataset, p['checkpoint'], p['config_path'], p['data_root']) != p:
                raise RuntimeError('Original input profile changed')
            if sha256_file(self.cfg[dataset]) != self.manifest['cfg_checkpoints'][dataset]['sha256']:
                raise RuntimeError('Original CFG weights changed')
        if count != 231726 or self.fingerprints() != self.manifest['code_sha256']:
            raise RuntimeError('Final count or successor source fingerprint mismatch')
        atomic_json(self.out / 'audit.json', dict(state='passed', unique_results=66,
            trajectory_records=count, temporal_caches=caches, projection_revision=DISTANCE_VERSION,
            mixed_implementations=True, inherited_legacy_results=len(self.parent_registry),
            new_batched_results=66-len(self.parent_registry), original_inputs_unchanged=True,
            original_parent_unchanged=True, all_hashes_valid=True, whole_dataset_metrics_recomputed=True,
            paired_spatial_rng=True, distance_switches_verified=True))

    def execute(self):
        self.precheck()
        if not self.args.sync_only:
            for dataset in (NY, IST):
                self.preflight_city(dataset)
            if self.args.preflight_only:
                self.phase = 'ready-for-batched-continuation'; self.status('preflight-passed')
                return
            for dataset in (NY, IST):
                for seed in SEEDS:
                    for variant in SAMPLING_VARIANTS:
                        if key(dataset, seed, variant) not in self.registry:
                            self.sampling(dataset, seed, variant, self.cfg[dataset] if variant == 'cfg' else None)
                        if variant == 'no_projection' and key(dataset, seed, 'postswap') not in self.registry:
                            self.postswap(dataset, seed)
                    self.verify_city_pairing(dataset, seed, SAMPLING_VARIANTS)
        self.phase = 'final-audit'; self.status('running'); self.audit()
        self.phase = 'tracking-sync'; self.status('running')
        pending = [k for k, r in self.registry.items() if not self.sync_record(r)]
        if pending:
            self.status('tracking-pending', pending=pending)
            return
        self.tables(complete=True)
        self.phase = 'server-complete-awaiting-local-verification'; self.status('server-complete')
        files = inventory(self.out)
        files.pop('delivery-files.json', None)
        atomic_json(self.out / 'delivery-files.json', dict(state='sealed', manifest_sha256=self.manifest_sha, files=files))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-run', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--delivery-root', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--sync-only', action='store_true')
    args = parser.parse_args()
    safe_tag(args.run_id)
    if args.sync_only and not args.resume:
        parser.error('--sync-only requires --resume')
    os.chdir(ROOT)
    study = SuccessorStudy(args)
    import fcntl
    with (study.out / 'pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            study.execute()
        except BaseException as exc:
            if not study.registry and (study.out / 'registry.json').exists():
                study.registry = read_json(study.out / 'registry.json')
            study.status('failed', error_type=type(exc).__name__, error=str(exc))
            raise


if __name__ == '__main__':
    main()
