"""User-approved full OOD resampling from frozen train-only calibration. No training."""
import argparse
from copy import deepcopy
import importlib.metadata
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_newyork_ood import Experiment, code_fingerprint, disk_guard, DATASET, SEED
from tools.perfcal_common import temperature_choice, batch_choice, calibration_metrics
from experiment_io import atomic_json, sha256_file, validate_sequences
from merge_results import merge_parts
import numpy as np
import torch
import wandb
from omegaconf import OmegaConf

REVISION = 'perfcal-v1-ood-r2'


def selected_settings(calibration, results, receipt):
    if receipt.get('second_gpu_verified') is not True:
        raise RuntimeError('Calibration has no second-GPU approval')
    temperature = temperature_choice([r for r in results if r['kind'] == 'temperature'])
    batches = [r for r in results if r['kind'] == 'batch']
    selected, _ = batch_choice(batches, calibration['profile'])
    if not temperature or not selected:
        raise RuntimeError('Calibration has no eligible configuration')
    expected = dict(batch_size=selected['batch_size'], temperature=temperature['temperature'],
                    projection_outer_iters=10, projection_inner_iters=50)
    if receipt.get('recommendation') != expected:
        raise RuntimeError('Recommendation does not match recorded calibration selection')
    if selected['metrics']['strict_ovr'] != min(
            r['metrics']['strict_ovr'] for r in batches if r['state'] == 'complete'):
        raise RuntimeError('Recommended batch is not the lowest-violation measured setting')
    verification = next(r for r in results if r['kind'] == 'verification')
    if (verification['state'] != 'complete' or verification['batch_size'] != expected['batch_size']
            or verification['temperature'] != expected['temperature']
            or verification['physical_gpu'] == selected['physical_gpu']):
        raise RuntimeError('Second-GPU receipt does not match the selected configuration')
    projection = dict(calibration['profile']['projection'])
    if (projection['projection_outer_iters'], projection['projection_inner_iters']) != (10, 50):
        raise RuntimeError('Calibration projection budget changed')
    if projection.pop('use_gumbel_softmax') is not True:
        raise RuntimeError('Expected calibrated Gumbel projection')
    projection['gumbel_temperature'] = expected['temperature']
    return expected['batch_size'], projection


class CalibratedSampling(Experiment):
    def __init__(self, args):
        self.source = Path(args.source_experiment).resolve()
        self.calibration = Path(args.calibration).resolve()
        self.source_manifest = json.loads((self.source/'manifest.json').read_text())
        args.run_id = self.source_manifest['run_id']
        args.stage = 'sample'
        args.preflight_id = None
        args.sampling_revision = REVISION
        super().__init__(args)

    def train(self):
        raise RuntimeError('This entry point never trains')

    def fingerprints(self):
        hashes = code_fingerprint()
        for name in ('tools/run_calibrated_ood.py', 'tools/sample_perfcal.py', 'tools/perfcal_common.py',
                     'tools/validate_calibrated_empty.py'):
            hashes[name] = sha256_file(ROOT/name)
        return hashes

    def precheck(self):
        self.phase = 'calibration-precheck'
        self.status('running')
        disk_guard()
        if ROOT in (self.source.parent.parent, self.calibration.parent.parent):
            raise RuntimeError('Use a separate sampling deployment, not training/calibration directories')
        source = self.source_manifest
        if (source['dataset'] != DATASET or source['epochs'] != 1000 or source['seed'] != SEED
                or source['train_batch_size'] != 64 or source['sample_count'] != 2108
                or source['po_loss_weight'] != 0):
            raise RuntimeError('Unexpected source training profile')
        if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
            raise RuntimeError('Exactly two visible GPUs are required')
        calibration = json.loads((self.calibration/'manifest.json').read_text())
        receipt = json.loads((self.calibration/'recommendation.json').read_text())
        results = json.loads((self.calibration/'results.json').read_text())
        state = json.loads((self.calibration/'status.json').read_text())
        audit = json.loads((self.calibration/'audit.json').read_text())
        if (state['state'] != 'complete' or audit['state'] != 'complete'
                or calibration['reference_split'] != 'train' or calibration['independent_validation']
                or not audit['inputs_unchanged'] or not audit['second_gpu_exact_times_marks_pois']):
            raise RuntimeError('Calibration was not completed and audited')
        if (calibration['source_run_id'] != self.run_id
                or calibration['source_manifest_sha256'] != sha256_file(self.source/'manifest.json')
                or calibration['checkpoint_sha256'] != sha256_file(self.source/'final.ckpt')):
            raise RuntimeError('Calibration used different training inputs/checkpoint')
        for name, expected in calibration['code_sha256'].items():
            if sha256_file(ROOT/name) != expected:
                raise RuntimeError(f'Calibrated implementation changed: {name}')
        for split, expected in source['data_sha256'].items():
            if sha256_file(ROOT/f'data/{DATASET}/{DATASET}_{split}.pkl') != expected:
                raise RuntimeError(f'Dataset changed: {split}')
        if {name: importlib.metadata.version(name) for name in source['packages']} != source['packages']:
            raise RuntimeError('Environment changed since training')
        batch, projection = selected_settings(calibration, results, receipt)
        self.entity = wandb.Api(timeout=30).default_entity
        if self.entity != source['entity']:
            raise RuntimeError('W&B account differs from the training account')
        if wandb.Api(timeout=30).run(f'{self.entity}/Marionette/{self.run_id}').state != 'finished':
            raise RuntimeError('Source training did not finish')
        self.manifest = deepcopy(source)
        self.manifest.update(sampling_revision=REVISION, result_id=self.result_id,
            sample_batch_size=batch, projection=projection, empty_policy='keep', inference_only=True,
            code_sha256=self.fingerprints(), training_code_sha256=source['code_sha256'],
            source_experiment=str(self.source), source_manifest_sha256=sha256_file(self.source/'manifest.json'),
            source_config_sha256=sha256_file(self.source/'config_hydra.yaml'),
            source_checkpoint_sha256=calibration['checkpoint_sha256'],
            calibration_directory=str(self.calibration),
            calibration_manifest_sha256=sha256_file(self.calibration/'manifest.json'),
            recommendation_sha256=sha256_file(self.calibration/'recommendation.json'),
            calibration_results_sha256=sha256_file(self.calibration/'results.json'),
            calibration_audit_sha256=sha256_file(self.calibration/'audit.json'),
            calibration_reference_split='train', precision='FP32; no autocast; TF32 disabled',
            selection='Lowest measured calibration strict OVR; no OOD tuning')
        path = self.directory/'manifest.json'
        if path.exists():
            if not self.args.resume or json.loads(path.read_text()) != self.manifest:
                raise RuntimeError('Existing sampling requires identical inputs and explicit --resume')
        elif self.args.resume:
            raise RuntimeError('Cannot resume without a manifest')
        else:
            atomic_json(path, self.manifest)
        checkpoint, run_path = self.checkpoint(require_complete=True)
        if sha256_file(checkpoint) != self.manifest['source_checkpoint_sha256']:
            raise RuntimeError('Training checkpoint differs from calibrated checkpoint')
        if sha256_file(run_path/'config_hydra.yaml') != self.manifest['source_config_sha256']:
            raise RuntimeError('W&B checkpoint configuration differs from frozen training config')
        config = OmegaConf.load(run_path/'config_hydra.yaml')
        if bool(config.model.get('use_constraint_projection', False)):
            raise RuntimeError('Expected projection-disabled training config before applying sampling settings')
        self.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'tests', '-v'], 'unit-tests')

    def sample_command(self, method, checkpoint, tag, rank, *, benchmark=False):
        if method not in ('native', 'projection'):
            raise ValueError('Unknown sampling method')
        command = [sys.executable, '-u', '-B', 'tools/sample_perfcal.py', '--run_id', self.run_id,
            '--checkpoint', str(checkpoint), '--output_tag', tag, '--rank', str(rank),
            '--world_size', '1' if benchmark else '2', '--seed', str(SEED),
            '--batch_size', str(self.manifest['sample_batch_size']), '--constraint_source', 'strict_test',
            '--sampling_revision', REVISION]
        if benchmark:
            command += ['--start_index', '64', '--max_samples', '64']
        if self.args.resume:
            command += ['--resume']
        if method == 'projection':
            command += ['--use_constraint_projection', '--use_gumbel_softmax']
            for key, value in self.manifest['projection'].items():
                command += [f'--{key}', str(value)]
        return command

    def benchmark(self, checkpoint):
        self.phase = 'calibrated-empty-batch-regression'
        self.status('running')
        proofs = {}
        for method in ('native', 'projection'):
            tag = self.output_tag(f'regression_{method}')
            self.run(self.sample_command(method, checkpoint, tag, 0, benchmark=True),
                     f'regression-{method}', dict(os.environ, CUDA_VISIBLE_DEVICES='0'))
            path = merge_parts(DATASET, self.run_id, 1, tag, expected_count=64)
            data = torch.load(path, map_location='cpu', weights_only=False)
            if data['test_indices'] != list(range(64,128)) or len(data['sequences']) != 64:
                raise RuntimeError('Calibrated end-to-end regression lost test indices')
            for index in data['temporal_empty_test_indices']:
                if len(data['sequences'][index-64]['checkins']) != 0:
                    raise RuntimeError('Temporal empty record was filled')
            if (data['projection_calls'] > 0) != (method == 'projection'):
                raise RuntimeError('Regression projection switch mismatch')
            proofs[method] = dict(output_sha256=sha256_file(path), projection_calls=data['projection_calls'],
                                 temporal_empty_test_indices=data['temporal_empty_test_indices'])
        if proofs['native']['temporal_empty_test_indices'] != proofs['projection']['temporal_empty_test_indices']:
            raise RuntimeError('Paired regression has different temporal-empty indices')
        fixture_command=[sys.executable,'-u','-B','tools/validate_calibrated_empty.py',
            '--source-experiment',str(self.source),'--calibration',str(self.calibration),
            '--output-dir',str(self.directory/'frozen-empty-regression')]
        if self.args.resume:
            fixture_command+=['--resume']
        self.run(fixture_command,'frozen-empty-regression',dict(os.environ,CUDA_VISIBLE_DEVICES='0'))
        frozen=json.loads((self.directory/'frozen-empty-regression/receipt.json').read_text())
        if frozen['state']!='passed' or not frozen['empty_index_99_preserved']:
            raise RuntimeError('Frozen empty-input regression did not pass')
        atomic_json(self.directory/'regression.json', dict(state='passed', indices=list(range(64,128)),
                    empty_index_99_preserved_on_frozen_input=True, projection=self.manifest['projection'],
                    methods=proofs,frozen_input_receipt_sha256=sha256_file(
                        self.directory/'frozen-empty-regression/receipt.json')))

    def evaluate(self, method, generated_path, timing):
        raw = torch.load(ROOT/f'data/{DATASET}/{DATASET}_test.pkl', map_location='cpu', weights_only=False)
        generated = torch.load(generated_path, map_location='cpu', weights_only=False)
        validate_sequences(generated['sequences'], raw['poi_category'])
        for reference, sample in zip(raw['sequences'], generated['sequences']):
            for i in range(1,7):
                key = f'condition{i}_indicator'
                if not np.array_equal(reference[key], sample[key]):
                    raise RuntimeError('Generated conditions are not aligned with OOD test indices')
        config = generated['metadata']['sampling_config']
        if config['batch_size'] != self.manifest['sample_batch_size']:
            raise RuntimeError('Output has an unexpected batch size')
        if method == 'projection' and any(config[k] != v for k,v in self.manifest['projection'].items()):
            raise RuntimeError('Output has unapproved projection parameters')
        extra = calibration_metrics(raw['sequences'], generated['sequences'], raw['poi_category'])
        return super().evaluate(method, generated_path, dict(timing, pair_coverage=extra['pair_coverage']))

    def execute(self):
        self.precheck()
        checkpoint = self.freeze_checkpoint()
        self.benchmark(checkpoint)
        metrics = {}
        for method in ('native', 'projection'):
            generated, timing = self.sample(method, checkpoint)
            metrics[method] = self.evaluate(method, generated, timing)
        if self.fingerprints() != self.manifest['code_sha256']:
            raise RuntimeError('Sampling code changed during the experiment')
        if (sha256_file(self.source/'final.ckpt') != self.manifest['source_checkpoint_sha256']
                or sha256_file(self.source/'manifest.json') != self.manifest['source_manifest_sha256']):
            raise RuntimeError('Source training artifacts changed')
        for split, expected in self.manifest['data_sha256'].items():
            if sha256_file(ROOT/f'data/{DATASET}/{DATASET}_{split}.pkl') != expected:
                raise RuntimeError('Input data changed during sampling')
        atomic_json(self.directory/'comparison.json', metrics)
        self.phase = 'complete'
        self.status('complete', samples_per_method=self.manifest['sample_count'],
                    projection=self.manifest['projection'], retrained=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-experiment', required=True)
    parser.add_argument('--calibration', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    experiment = CalibratedSampling(args)
    import fcntl
    with (experiment.directory/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            experiment.execute()
        except BaseException as exc:
            experiment.status('failed', error_type=type(exc).__name__, error=str(exc))
            raise


if __name__ == '__main__':
    main()
