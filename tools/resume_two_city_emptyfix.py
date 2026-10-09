"""Resume the sealed two-city study with a separately audited empty test fixture.

The original controller, worker, manifest, failed fixture and completed outputs
remain byte-for-byte intact. Only synthetic preflight jobs use the new fixture.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from experiment_io import atomic_json, publish_torch, sha256_file
from tools.ablation_common import SEEDS, VARIANTS, evaluate, same_records
from tools.run_two_city_study import Study, NY, IST
from tools.unified_tables import key

REVISION = 'empty-fixture-v1'


def corrected_fixture(source, source_sha):
    """Derive one empty row without changing the source or its reference PO edges."""
    data = copy.deepcopy(source)
    batch = data['batches'][0]['batch']
    batch._validate()
    active = ((batch.unpadded_length > 0) & batch.po_matrix.bool().flatten(1).any(1)).nonzero().flatten()
    if not len(active):
        raise RuntimeError('Expected a nonempty constrained row in the test fixture')
    row = int(active[0])
    batch.unpadded_length[row] = 0
    for name in ('mask', 'time', 'tau', 'category_mask', 'poi_mask', 'kept'):
        value = getattr(batch, name, None)
        if value is not None:
            value[row].zero_()
    for i in range(1, 7):
        getattr(batch, f'condition{i}')[row].zero_()
    # Keep indicators and reference constraints: empty output must still count
    # as a strict constraint violation. Other rows and RNG layout are unchanged.
    batch._validate()
    data.update(test_only_synthetic_empty_row=row, derived_from_sha256=source_sha,
                test_fixture_revision=REVISION)
    return data


def verify_hashes(hashes):
    for name, expected in hashes.items():
        if sha256_file(name) != expected:
            raise RuntimeError(f'Preserved artifact changed: {name}')


class RecoveryStudy(Study):
    @property
    def recovery_dir(self):
        return self.out / 'recovery' / REVISION

    def seal_recovery(self):
        path = self.recovery_dir / 'manifest.json'
        recovery_code = {str(ROOT / name): sha256_file(ROOT / name) for name in (
            'tools/resume_two_city_emptyfix.py', 'tests/test_two_city_recovery.py')}
        if path.exists():
            proof = json.loads(path.read_text())
            if (proof['source_manifest_sha256'] != self.manifest_sha or
                    proof['recovery_code_sha256'] != recovery_code):
                raise RuntimeError('Recovery provenance differs; refusing a silent revision')
            verify_hashes(proof['preserved_sha256'])
            return
        state = json.loads((self.out / 'status.json').read_text())
        failed = json.loads((self.out / 'preflight-ist/no_projection/1/status.json').read_text())
        if (state['state'] != 'failed' or state['phase'] != 'Istanbul-layout-and-ablation-preflight'
                or failed != dict(state='failed', error_type='AssertionError', error='wrong mask')
                or len(self.registry) != 30):
            raise RuntimeError('Not the reviewed 30-result empty-fixture failure')
        paths = {self.out / 'manifest.json', self.ny_ablation / 'manifest.json',
                 self.ny_ablation / 'audit.json', self.ny_cfg}
        for record in self.registry.values():
            p = Path(record['path'])
            paths.add(p)
            paths.update(Path(s['path']) for s in record['source_shards'])
            for folder in [p.parent] + [Path(s['path']).parent for s in record['source_shards']]:
                paths.update(q for q in folder.glob('*.json') if q.is_file())
        for folder in (self.ny_ablation / 'time-cache', self.out / 'preflight-ny', self.out / 'preflight-ist'):
            paths.update(p for p in folder.rglob('*') if p.is_file())
        proof = dict(revision=REVISION, source_manifest_sha256=self.manifest_sha,
                     recovery_code_sha256=recovery_code,
                     preserved_sha256={str(p): sha256_file(p) for p in sorted(paths)},
                     original_failure=state, original_registry=self.registry,
                     scope='Synthetic preflight row only; production code, inputs, seeds and budgets unchanged')
        atomic_json(path, proof)

    def restore_registry(self):
        registry = json.loads((self.out / 'registry.json').read_text())
        expected_ny = {key(NY, s, v) for s in SEEDS for v in (*VARIANTS, 'postswap', 'energy', 'cfg')}
        if {k for k, r in registry.items() if r['dataset'] == NY} != expected_ny:
            raise RuntimeError('Expected exactly 30 completed NewYork anchors')
        for k, record in registry.items():
            if k != key(record['dataset'], record['seed'], record['variant']):
                raise RuntimeError('Registry identity mismatch')
            path = Path(record['path'])
            if sha256_file(path) != record['sha256']:
                raise RuntimeError('Registered result hash mismatch')
            data = torch.load(path, map_location='cpu', weights_only=False)
            indices = list(range(self.profiles[record['dataset']]['test_count']))
            if data.get('test_indices', data.get('indices')) != indices:
                raise RuntimeError('Registered result indices differ')
            shards, shard_indices = [], []
            for shard in record['source_shards']:
                if sha256_file(shard['path']) != shard['sha256']:
                    raise RuntimeError('Registered source shard changed')
                part = torch.load(shard['path'], map_location='cpu', weights_only=False)
                shards.extend(part['sequences'])
                shard_indices.extend(part.get('indices', part.get('test_indices')))
            if shard_indices != indices or not same_records(shards, data['sequences']):
                raise RuntimeError('Registered source shard contents differ')
            raw = self.raw[record['dataset']]
            metrics, rows = evaluate(raw['sequences'], data['sequences'], raw['poi_category'], indices)
            if (metrics != record['metrics'] or
                    rows != json.loads((path.parent / 'per_condition.json').read_text())):
                raise RuntimeError('Registered metrics do not reproduce')
            if record['origin'] == 'new':
                receipt = json.loads((path.parent / 'wandb-sync.json').read_text())
                if (receipt['state'] != 'verified' or
                        receipt['expected']['result_sha256'] != record['sha256']):
                    raise RuntimeError('Missing verified tracking receipt')
        self.registry = registry
        for seed in SEEDS:
            self.verify_city_pairing(NY, seed, ['energy', 'cfg'])
        self.cfg_cost[NY] = dict(json.loads((self.ny_cfg.parent / 'result.json').read_text()), reused=True)

    def ist_pilot_caches(self):
        source = self.out / 'preflight-ist/time-cache/payload.pkl'
        job = self.city_job(IST, source.parent, 'cache', list(range(64)), 135398, 0, split='train')
        self.run_jobs([job])  # Checks and reuses the already sealed temporal cache.
        source_sha = sha256_file(source)
        expected = corrected_fixture(torch.load(source, map_location='cpu', weights_only=False), source_sha)
        target = self.recovery_dir / 'empty-fixture/payload.pkl'
        receipt = target.parent / 'receipt.json'
        if target.exists():
            proof = json.loads(receipt.read_text())
            if (proof['source_sha256'] != source_sha or
                    proof['output_sha256'] != sha256_file(target)):
                raise RuntimeError('Corrected fixture changed')
            torch.load(target, map_location='cpu', weights_only=False)['batches'][0]['batch']._validate()
        else:
            publish_torch(target, expected)
            atomic_json(receipt, dict(state='validated', source_sha256=source_sha,
                        output_sha256=sha256_file(target), revision=REVISION,
                        row=expected['test_only_synthetic_empty_row']))
        return source, target, expected['test_only_synthetic_empty_row']

    def pilot_folder(self, variant, gpu):
        # Normal jobs retain their original identity; failed synthetic jobs live
        # in a new namespace. The entire old failure directory is preserved.
        root = self.recovery_dir / 'preflight-ist' if gpu else self.out / 'preflight-ist'
        return root / variant / str(gpu)

    def preflight_ist(self, cfg=None):
        self.phase = 'Istanbul-CFG-preflight' if cfg else 'Istanbul-layout-and-ablation-preflight'
        self.status('running', recovery_revision=REVISION)
        normal, empty, row = self.ist_pilot_caches()
        variants = ['cfg'] if cfg else ['no_projection'] + [v for v in VARIANTS if v != 'no_projection'] + ['energy']
        for variant in variants:
            jobs = [self.city_job(IST, self.pilot_folder(variant, gpu), 'sample', list(range(64)),
                        135398, gpu, split='train', variant=variant, cache=cache, cfg=cfg,
                        empty_row=row if gpu else None) for gpu, cache in enumerate((normal, empty))]
            self.run_jobs(jobs)
            for gpu in (0, 1):
                self.verify_pairing([self.pilot_folder('no_projection', gpu) / 'payload.pkl',
                                     self.pilot_folder(variant, gpu) / 'payload.pkl'])
        if not cfg:
            job = self.city_job(IST, self.out / 'preflight-ist/full-gpu1', 'sample', list(range(64)),
                                135398, 1, split='train', variant='full', cache=normal)
            self.run_jobs([job])
            first = torch.load(self.pilot_folder('full', 0) / 'payload.pkl', map_location='cpu', weights_only=False)
            second = torch.load(Path(job['output_dir']) / 'payload.pkl', map_location='cpu', weights_only=False)
            if not same_records(first['sequences'], second['sequences']):
                raise RuntimeError('Istanbul Full differs across GPUs')
        atomic_json(self.recovery_dir / ('cfg-preflight-receipt.json' if cfg else 'preflight-receipt.json'),
                    dict(state='passed', semantic_classes=9, model_slots=10, synthetic_empty_row=row,
                         corrected_fixture_sha256=sha256_file(empty), revision=REVISION))

    def audit(self):
        proof = json.loads((self.recovery_dir / 'manifest.json').read_text())
        verify_hashes(proof['preserved_sha256'])
        verify_hashes(proof['recovery_code_sha256'])
        if sha256_file(self.out / 'manifest.json') != proof['source_manifest_sha256']:
            raise RuntimeError('Original study manifest changed')
        super().audit()
        atomic_json(self.recovery_dir / 'audit.json', dict(state='passed',
                    preserved_files=len(proof['preserved_sha256']), original_manifest_unchanged=True,
                    production_code_unchanged=True, old_failed_fixture_preserved=True))

    def execute(self):
        self.precheck()  # Original strict manifest, data, checkpoint and code gates.
        self.restore_registry()  # No rewriting, sampling or training of NewYork.
        self.seal_recovery()
        print(json.dumps(dict(recovery=REVISION, verified_results=len(self.registry),
                              newyork_skipped=True)), flush=True)
        self.preflight_ist()
        if self.args.preflight_only:
            self.status('preflight-passed', recovery_revision=REVISION)
            return
        for seed in SEEDS:
            for variant in VARIANTS:
                if key(IST, seed, variant) not in self.registry:
                    self.sampling(IST, seed, variant)
            if key(IST, seed, 'postswap') not in self.registry:
                self.postswap(IST, seed)
            if key(IST, seed, 'energy') not in self.registry:
                self.sampling(IST, seed, 'energy')
            self.tables()
        self.training(2)
        checkpoint = self.training(1000)
        self.preflight_ist(checkpoint)
        for seed in SEEDS:
            if key(IST, seed, 'cfg') not in self.registry:
                self.sampling(IST, seed, 'cfg', checkpoint)
        self.phase = 'final-audit'
        self.status('running', recovery_revision=REVISION)
        self.audit()
        self.tables(complete=True)
        self.phase = 'complete'
        self.status('complete', total_unique_results=60, base_models_retrained=False, recovery_revision=REVISION)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('newyork-source', 'newyork-ablation', 'newyork-cfg', 'istanbul-inputs', 'delivery-root'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--run-id', default='two-city-v1')
    parser.add_argument('--resume', action='store_true', required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    study = RecoveryStudy(args)
    import fcntl
    with (study.out / 'pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            study.execute()
        except BaseException as exc:
            # Do not erase known progress if an early read-only gate fails.
            if not study.registry and (study.out / 'registry.json').exists():
                study.registry = json.loads((study.out / 'registry.json').read_text())
            study.status('failed', error_type=type(exc).__name__, error=str(exc), recovery_revision=REVISION)
            raise


if __name__ == '__main__':
    main()
