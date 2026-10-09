import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from evaluations.statistical_metrics import Get_Statistical_Metrics, STATISTICAL_NAMES
from tools.ablation_common import (DISTANCE_VARIANTS, DISTANCE_VERSION, VARIANTS,
                                   evaluate, projector)
from tools.baseline_common import full_metrics
from tools.unified_tables import JSD, METHODS, render, table


class UpgradeIntegrationTests(unittest.TestCase):
    def records(self):
        mapping = {6: 4, 7: 5, 8: 4}
        coords = {6: [0., 0.], 7: [0., .01], 8: [0., .03]}
        result = []
        for pois in ([6, 7], [6, 8, 7]):
            result.append(dict(checkins=np.array(pois), marks=np.array([mapping[p] for p in pois]),
                arrival_times=np.arange(len(pois), dtype=float), gps=[coords[p] for p in pois],
                **{f'condition{i}_indicator': np.ones(24) for i in range(1, 7)}))
        return result, mapping

    def test_direct_baseline_ablation_and_cli_agree(self):
        from evaluation import run_Statistical
        refs, mapping = self.records()
        generated = copy.deepcopy(refs)
        generated[0]['marks'][:] = 9  # Evaluators must use POI-derived categories.
        direct = Get_Statistical_Metrics(refs, refs)
        details = {}
        ablation, _ = evaluate(refs, generated, mapping, [0, 1], diagnostics=details)
        baseline = full_metrics(refs, generated, mapping)
        with mock.patch('evaluation.torch.load', side_effect=[dict(sequences=refs, poi_category=mapping),
                                                               dict(sequences=copy.deepcopy(generated))]):
            standalone = run_Statistical('toy', '')[0]
        for name in (*STATISTICAL_NAMES, 'totalJSD', 'evaluation_version'):
            self.assertEqual(direct[name], ablation[name], name)
            self.assertEqual(direct[name], baseline[name], name)
            self.assertEqual(direct[name], standalone[name], name)
        self.assertEqual(len(details['category_hourly']), 24)
        self.assertEqual(generated[0]['marks'].tolist(), [9, 9])

    def test_tables_reject_legacy_before_writing(self):
        record = {'toy': dict(metrics=dict(Category=0, Interval=0))}
        with self.assertRaisesRegex(ValueError, 'version'):
            table(record, METHODS, JSD)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'must-not-exist'
            with self.assertRaisesRegex(ValueError, 'version'):
                render(output, record, {})
            self.assertFalse(output.exists())

    def test_ablation_switches_and_legacy_defaults(self):
        self.assertEqual(DISTANCE_VARIANTS['no_kl']['distance'], 0)
        self.assertEqual(DISTANCE_VARIANTS['no_kl']['kl'], 0)
        self.assertEqual(DISTANCE_VARIANTS['no_distance_kl']['distance'], 0)
        self.assertEqual(DISTANCE_VARIANTS['no_distance_kl']['kl'], 1)
        self.assertFalse(DISTANCE_VARIANTS['no_gumbel']['gumbel'])
        self.assertNotIn('no_distance_kl', VARIANTS)
        dd = SimpleNamespace(num_classes=11, type_classes=2, num_spectial=4)
        self.assertEqual(projector(dd, 'full').projection_distance_kl_weight, 0)
        self.assertEqual(projector(dd, 'no_kl', revision=DISTANCE_VERSION).projection_distance_kl_weight, 0)
        with self.assertRaisesRegex(ValueError, 'datamodule'):
            projector(dd, 'full', revision=DISTANCE_VERSION)

    def test_sampler_defaults_and_explicit_deterministic_flag(self):
        import sample
        args = sample.parser.parse_args([])
        self.assertEqual(args.sampling_revision, 'distance-kl-v2')
        self.assertIsNone(args.use_gumbel_softmax)
        self.assertFalse(sample.parser.parse_args(['--no_gumbel_softmax']).use_gumbel_softmax)
        self.assertTrue(sample.parser.parse_args(['--use_gumbel_softmax']).use_gumbel_softmax)

    def test_distance_poi_mask_reaches_projector(self):
        from discrete_diffusion.diffusion_transformer import DiffusionTransformer
        dd = object.__new__(DiffusionTransformer)
        torch.nn.Module.__init__(dd)
        dd.use_constraint_projection = True
        dd.projection_last_k_steps = 40
        dd.projection_frequency = 4
        dd.projection_call_count = 0
        logits = torch.zeros(1, 11, 4)
        dd.p_pred = lambda *args: (logits, logits)
        dd.log_sample_categorical = lambda value: value
        poi = torch.tensor([[0, 0, 1, 1]], dtype=torch.bool)
        batch = SimpleNamespace(category_mask=~poi, poi_mask=poi)
        p = mock.Mock(projection_distance_kl_weight=1, last_projection_stats={'optimizer_steps': 1})
        p.compile_batched_constraints.return_value = (torch.ones(1, 2, 1), torch.ones(1, 2, 1), torch.ones(1, 1))
        p.project_with_matrices.return_value = logits
        dd.constraint_projector = p
        dd.p_sample(logits, None, torch.tensor([0]), batch, po_constraints=[[([0], [1])]], diffusion_index=0)
        self.assertIs(p.project_with_matrices.call_args.kwargs['poi_mask'], poi)


if __name__ == '__main__':
    unittest.main()
