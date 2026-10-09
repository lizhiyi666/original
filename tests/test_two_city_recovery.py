import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from datamodule import Batch
from tools.resume_two_city_emptyfix import corrected_fixture, RecoveryStudy, REVISION


class RecoveryTests(unittest.TestCase):
    def fixture(self):
        times = torch.tensor([[1., 2., 0.], [1., 2., 3.]])
        mask = times > 0
        args = dict(mask=mask, time=times, tau=torch.tensor([[1., 1., 0.], [1., 1., 1.]]),
                    tmax=torch.tensor(24.), unpadded_length=torch.tensor([2, 3]),
                    category_mask=torch.tensor([[0, 1, 1, 0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 0, 0, 0, 0, 0]]),
                    poi_mask=torch.tensor([[0, 0, 0, 0, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 1, 1, 0]]),
                    po_matrix=torch.zeros(2, 10, 10))
        args['po_matrix'][:, 0, 1] = 1
        for i in range(1, 7):
            args[f'condition{i}'] = mask.long()
            args[f'condition{i}_indicator'] = torch.ones(2, 1, dtype=torch.long)
        return dict(batches=[dict(batch=Batch(**args), indices=[0, 1])], indices=[0, 1], manifest_sha256='sealed')

    def test_reproduces_failure_and_corrected_fixture_validates(self):
        source = self.fixture()
        old = copy.deepcopy(source['batches'][0]['batch'])
        old.unpadded_length[0] = 0
        old.mask[0].zero_()
        old.category_mask[0].zero_()
        old.poi_mask[0].zero_()
        with self.assertRaisesRegex(AssertionError, 'wrong mask'):
            old._validate()
        fixed = corrected_fixture(source, 'original-sha')
        batch = fixed['batches'][0]['batch']
        batch._validate()
        self.assertEqual(fixed['test_only_synthetic_empty_row'], 0)
        self.assertEqual(fixed['derived_from_sha256'], 'original-sha')
        self.assertEqual(fixed['test_fixture_revision'], REVISION)
        self.assertEqual(int(batch.time[0].count_nonzero()), 0)
        self.assertEqual(int(batch.tau[0].count_nonzero()), 0)

    def test_original_other_rows_constraints_and_context_unchanged(self):
        source = self.fixture()
        original = copy.deepcopy(source['batches'][0]['batch'])
        fixed = corrected_fixture(source, 'original-sha')['batches'][0]['batch']
        for name, value in vars(original).items():
            if isinstance(value, torch.Tensor):
                self.assertTrue(torch.equal(value, getattr(source['batches'][0]['batch'], name)), name)
                if value.ndim and value.shape[0] == 2:
                    self.assertTrue(torch.equal(value[1], getattr(fixed, name)[1]), name)
        self.assertTrue(torch.equal(fixed.po_matrix, original.po_matrix))
        for i in range(1, 7):
            self.assertTrue(torch.equal(getattr(fixed, f'condition{i}_indicator'),
                                        getattr(original, f'condition{i}_indicator')))
        self.assertTrue(torch.equal(fixed.unpadded_length, torch.tensor([0, 3])))

    def test_missing_constrained_row_rejected(self):
        source = self.fixture()
        source['batches'][0]['batch'].po_matrix.zero_()
        with self.assertRaisesRegex(RuntimeError, 'nonempty constrained row'):
            corrected_fixture(source, 'original-sha')

    def test_new_namespace_only_for_synthetic_jobs(self):
        study = object.__new__(RecoveryStudy)
        study.out = Path('/test/study')
        self.assertEqual(study.pilot_folder('full', 0), study.out / 'preflight-ist/full/0')
        self.assertEqual(study.pilot_folder('full', 1), study.recovery_dir / 'preflight-ist/full/1')

    def test_preflight_resume_never_calls_newyork_generation(self):
        study = object.__new__(RecoveryStudy)
        study.args = SimpleNamespace(preflight_only=True)
        study.registry = {}
        with mock.patch.multiple(study, precheck=mock.DEFAULT, restore_registry=mock.DEFAULT,
                                 seal_recovery=mock.DEFAULT, preflight_ist=mock.DEFAULT,
                                 status=mock.DEFAULT, sampling=mock.DEFAULT, postswap=mock.DEFAULT,
                                 training=mock.DEFAULT, import_newyork=mock.DEFAULT) as mocks:
            study.execute()
            for name in ('sampling', 'postswap', 'training', 'import_newyork'):
                mocks[name].assert_not_called()
            mocks['preflight_ist'].assert_called_once_with()


if __name__ == '__main__':
    unittest.main()
