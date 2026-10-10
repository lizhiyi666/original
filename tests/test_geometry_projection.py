import copy
import os
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from evaluations.statistical_metrics import radius, travel_distance
from geometry_projection import (GeometryConfig, GeometryObjective, GeometryReference, assert_invariants,
    geometry_generator, legacy_radius, load_geometry_reference, offset_distance, refine_records,
    untruncated_scores)


def fixture(count=40):
    gps = {6+i: [40.7 + (i % 8) * .002, -74 + (i // 8) * .002] for i in range(count)}
    categories = {token: 4 + (token % 2) for token in gps}
    records = []
    for tokens in ([], [6], [6, 8], [7, 6, 9, 12], [6, 6, 6]):
        records.append(dict(checkins=np.array(tokens, dtype=np.int64), marks=np.full(len(tokens), 5, dtype=np.int64),
            gps=[gps[p] for p in tokens], arrival_times=np.arange(1, len(tokens)+1, dtype=float),
            **{f'condition{i}':np.ones(len(tokens), dtype=np.int64) for i in range(1,7)},
            **{f'condition{i}_indicator':np.ones(24, dtype=np.int64) for i in range(1,7)}))
    raw = dict(poi_gps=gps, poi_category=categories, sequences=records)
    reference = GeometryReference(raw, range(len(records)))
    rng = torch.Generator().manual_seed(31)
    logits = torch.randn(len(records), max(gps)+3, 11, generator=rng)
    return raw, records, reference, logits


class GeometryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.device = os.environ.get('GEOMETRY_TEST_DEVICE', 'cpu')

    def test_exact_legacy_observables_and_coincidence_gradients(self):
        _, records, reference, _ = fixture()
        for record in records[2:]:
            coordinates = torch.tensor(record['gps'], dtype=torch.float64)
            origin = torch.deg2rad(coordinates.mean(0))
            offset = (torch.deg2rad(coordinates)-origin).to(self.device).requires_grad_()
            latitude = origin[0].to(self.device)
            length = offset_distance(offset[:-1], offset[1:], latitude).sum()
            spread = legacy_radius(offset, torch.ones(len(offset), dtype=torch.bool, device=self.device), latitude)
            self.assertAlmostEqual(float(length), travel_distance(np.asarray(record['gps'])), places=8)
            self.assertAlmostEqual(float(spread), radius(np.asarray(record['gps'])), places=8)
            (length + spread).backward()
            self.assertTrue(torch.isfinite(offset.grad).all())

    def test_candidates_have_original_and_same_actual_category(self):
        _, records, ref, logits = fixture(80)
        logits.zero_()  # Stable tie policy should be POI-id order.
        objective = GeometryObjective(ref, records, logits.to(self.device))
        self.assertTrue(torch.equal(objective.candidates[..., 0], objective.original))
        for row, index in enumerate(objective.rows):
            for position, token in enumerate(records[index]['checkins']):
                candidates = objective.candidates[row, position][objective.valid[row, position]].cpu().tolist()
                self.assertEqual(len(candidates), 32)
                self.assertEqual(len(candidates), len(set(candidates)))
                self.assertTrue(all(ref.poi_category[int(ref.ids[c])] == ref.poi_category[int(token)] for c in candidates))
                self.assertGreaterEqual(float(objective.q[row, position, 0]), .5)
        repeated = GeometryObjective(ref, records, logits.to(self.device))
        self.assertTrue(torch.equal(objective.candidates, repeated.candidates))

    def test_losses_gradients_padding_and_reproducibility(self):
        _, records, ref, logits = fixture(8)
        objective = GeometryObjective(ref, records, logits.to(self.device))
        theta = objective.log_q.clone().requires_grad_()
        noise = objective.noise(geometry_generator(17, 0, self.device))
        loss = objective.loss(theta, noise, GeometryConfig('same_category_v1'))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(theta.grad).all())
        self.assertEqual(theta.grad[~objective.valid].count_nonzero().item(), 0)
        self.assertEqual(theta.grad[~objective.mask].count_nonzero().item(), 0)
        before = torch.get_rng_state()
        cuda_before = torch.cuda.get_rng_state() if self.device == 'cuda' else None
        original = copy.deepcopy(records)
        first, stats = refine_records(records, None, ref, GeometryConfig('same_category_v1'),
                                     seed=17, scored_logits=logits.to(self.device))
        second, repeated = refine_records(records, None, ref, GeometryConfig('same_category_v1'),
                                         seed=17, scored_logits=logits.to(self.device))
        assert_invariants(original, first, ref)
        self.assertEqual(stats['optimizer_steps'], 50)
        self.assertLessEqual(stats['selected_objective'], stats['original_objective'])
        self.assertEqual(stats['selected_path'], repeated['selected_path'])
        for a, b, c in zip(records, original, first):
            self.assertTrue(all(np.array_equal(a[k], b[k]) for k in a))
        for a, b in zip(first, second):
            self.assertTrue(all(np.array_equal(a[k], b[k]) for k in a))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        if cuda_before is not None:self.assertTrue(torch.equal(cuda_before, torch.cuda.get_rng_state()))

    def test_off_no_projection_empty_singleton_and_one_candidate_skip(self):
        _, records, ref, logits = fixture()
        with patch('geometry_projection.untruncated_scores', side_effect=AssertionError('Must not score')):
            self.assertIs(refine_records(records, None, None)[0], records)
            self.assertIs(refine_records(records, None, ref, GeometryConfig('same_category_v1'), projection_executed=False)[0], records)
            tiny = records[:2]
            self.assertIs(refine_records(tiny, None, ref, GeometryConfig('same_category_v1'))[0], tiny)
            one = dict(poi_gps={6:[0., 0.]}, poi_category={6:4}, sequences=[dict(checkins=[6,6])])
            one_ref = GeometryReference(one, [0])
            value = [dict(checkins=[6,6])]
            self.assertIs(refine_records(value, None, one_ref, GeometryConfig('same_category_v1'))[0], value)

    def test_actual_category_invariant_detects_wrong_poi(self):
        _, records, ref, _ = fixture()
        changed = copy.deepcopy(records)
        changed[2]['checkins'][0] = 7
        changed[2]['gps'][0] = ref.coordinates[ref.lookup[7]].tolist()
        with self.assertRaisesRegex(RuntimeError, 'actual category'):
            assert_invariants(records, changed, ref)

    def test_nonfinite_loss_gradient_and_updates_fail(self):
        _, records, ref, logits = fixture(8)
        config = GeometryConfig('same_category_v1')
        for stage in ('loss', 'gradient', 'update'):
            if stage == 'loss':
                context = patch.object(GeometryObjective, 'loss', lambda obj, x, n, c: x.sum()*float('nan'))
            elif stage == 'gradient':
                context = patch('torch.nn.utils.clip_grad_norm_', side_effect=RuntimeError('nonfinite'))
            else:
                def invalid(opt, *args, **kwargs):
                    with torch.no_grad():opt.param_groups[0]['params'][0].fill_(float('nan'))
                context = patch.object(torch.optim.Adam, 'step', invalid)
            with context, self.assertRaises(FloatingPointError):
                refine_records(records, None, ref, config, scored_logits=logits.to(self.device))

    def test_train_only_reference_and_index_content_fingerprint(self):
        raw, _, _, _ = fixture()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'toy_train.pkl'; torch.save(raw, path)
            a = load_geometry_reference(path, [2,3])
            self.assertIs(a, load_geometry_reference(path, [2,3]))
            self.assertNotEqual(a.fingerprint, load_geometry_reference(path, [2,4]).fingerprint)
            with self.assertRaisesRegex(ValueError, 'training'):
                load_geometry_reference(Path(folder)/'toy_test.pkl', [2])

    def test_untruncated_scorer_one_forward_and_no_model_gradients(self):
        _, records, _, logits = fixture()
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.parameter=torch.nn.Parameter(torch.zeros(1));self.num_classes=logits.shape[1]
            def condition_encoder(self, batch):return torch.zeros(batch.batch_size,1,device=self.parameter.device)
            def predict_start(self, x, emb, t, batch):
                self.seen_t=t
                return logits.to(self.parameter.device).clone()
            def predict_start_with_truncate(self, *args):raise AssertionError('Truncation forbidden')
        model = Model().to(self.device).eval()
        before = model.parameter.detach().clone()
        with patch.object(model, 'predict_start', wraps=model.predict_start) as score:
            result = untruncated_scores(model, records)
            self.assertEqual(score.call_count,1)
        self.assertTrue((model.seen_t==0).all())
        self.assertFalse(result.requires_grad)
        self.assertIsNone(model.parameter.grad)
        self.assertTrue(torch.equal(before,model.parameter))


if __name__ == '__main__':
    unittest.main()
