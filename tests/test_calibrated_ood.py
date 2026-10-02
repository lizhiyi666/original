from pathlib import Path
from types import SimpleNamespace
import unittest

from tools.run_calibrated_ood import CalibratedSampling, selected_settings


class CalibratedCommandTests(unittest.TestCase):
    def experiment(self):
        experiment = object.__new__(CalibratedSampling)
        experiment.args = SimpleNamespace(resume=False)
        experiment.run_id = 'training-id'
        experiment.manifest = dict(sample_batch_size=64, projection=dict(
            projection_outer_iters=10, projection_inner_iters=50, gumbel_temperature=3.0,
            projection_last_k_steps=40, projection_frequency=4))
        return experiment

    def test_projection_uses_selected_budget_not_legacy_defaults(self):
        command = self.experiment().sample_command('projection', Path('frozen.ckpt'), 'unique', 1)
        for flag, value in [('--projection_outer_iters','10'), ('--projection_inner_iters','50'),
                            ('--gumbel_temperature','3.0'), ('--batch_size','64'), ('--world_size','2')]:
            self.assertEqual(command[command.index(flag)+1], value)
        self.assertIn('tools/sample_perfcal.py', command)

    def test_native_never_enables_projection(self):
        command = self.experiment().sample_command('native', Path('frozen.ckpt'), 'unique', 0)
        self.assertNotIn('--use_constraint_projection', command)

    def test_regression_retains_global_slice_and_full_batch(self):
        command = self.experiment().sample_command('projection', Path('frozen.ckpt'), 'unique', 0, benchmark=True)
        for flag in ('--start_index', '--max_samples', '--batch_size'):
            self.assertEqual(command[command.index(flag)+1], '64')
        self.assertEqual(command[command.index('--world_size')+1], '1')

    def test_training_is_forbidden(self):
        with self.assertRaises(RuntimeError):
            self.experiment().train()

    def test_unverified_calibration_is_rejected(self):
        with self.assertRaises(RuntimeError):
            selected_settings({}, [], {'second_gpu_verified': False})

    def test_selection_and_temperature_are_pinned_to_receipt(self):
        calibration = dict(profile=dict(memory_fraction_limit=.8, strict_ovr_tolerance=.01,
            pair_coverage_tolerance=.01, throughput_tie_fraction=.05,
            projection=dict(projection_outer_iters=10, projection_inner_iters=50, use_gumbel_softmax=True)))
        common = dict(state='complete', batch_size=64, temperature=3.0, physical_gpu=0,
            metrics=dict(strict_ovr=.4, pair_coverage=.65), sampling_seconds=100,
            samples_per_second=2, memory_fraction=.1, violating_probes_all_zero=False)
        results = [dict(common, kind='temperature', label='temperature-3p0'),
                   dict(common, kind='batch', label='batch-64'),
                   dict(common, kind='verification', label='second-gpu', physical_gpu=1)]
        receipt = dict(second_gpu_verified=True, recommendation=dict(
            batch_size=64, temperature=3.0, projection_outer_iters=10, projection_inner_iters=50))
        batch, projection = selected_settings(calibration, results, receipt)
        self.assertEqual((batch, projection['gumbel_temperature']), (64, 3.0))
        receipt['recommendation']['temperature'] = .1
        with self.assertRaises(RuntimeError):
            selected_settings(calibration, results, receipt)


if __name__ == '__main__':
    unittest.main()
