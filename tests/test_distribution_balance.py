import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from constraint_projection import ConstraintProjection
from distribution_balance import (BalanceConfig, BalancedGeometryObjective, CategoryDistributionObjective,
    DistributionReference, load_reference, top_visit_jsd, refine_balanced_records, aligned_category_times)
from evaluations.statistical_metrics import grank_jsd
from geometry_projection import assert_invariants
from tests.test_geometry_projection import fixture


class DistributionBalanceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.device = os.environ.get('GEOMETRY_TEST_DEVICE', 'cpu')
        self.raw, self.records, _, logits = fixture()
        self.ref = DistributionReference(self.raw, range(len(self.records)))
        self.logits = logits.to(self.device)

    def test_reference_self_edges_hour_and_boundaries(self):
        self.assertEqual(int(self.ref.transition_counts.sum()), 6)
        self.assertEqual(int(self.ref.hour_counts.sum()), 10)
        self.assertEqual(int(self.ref.transition_counts[1, 1]), 6)
        self.assertEqual(int(self.ref.hour_counts[1, 1]), 4)
        self.assertEqual(int(self.ref.visit_counts.sum()), 9)

    def test_train_only_and_fingerprints(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'toy_train.pkl'
            torch.save(self.raw, path)
            first = load_reference(path, list(range(5)))
            self.assertIs(first, load_reference(path, list(range(5))))
            self.assertNotEqual(first.fingerprint, load_reference(path, [2, 3]).fingerprint)
            with self.assertRaises(ValueError):load_reference(path, [True, 2])
            with self.assertRaises(ValueError):load_reference(path, [2, 2])
            with self.assertRaises(ValueError):load_reference(Path(folder)/'toy_test.pkl', [2])

    def test_expected_category_gradient_and_inactive_rows(self):
        mask = torch.tensor([[0, 1, 1, 0]]*2, device=self.device).bool()
        times = torch.tensor([[0., 1., 2., 0.]]*2, device=self.device)
        objective = CategoryDistributionObjective(self.ref, mask, times,
            torch.tensor([True, False], device=self.device), 12)
        logits = torch.zeros(2, 4, 12, device=self.device, requires_grad=True)
        before = torch.get_rng_state()
        loss = objective.loss(logits)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad[0, 1:3, 4:6].abs().sum()), 0)
        self.assertEqual(float(logits.grad[1].abs().sum()), 0)
        self.assertEqual(float(logits.grad[:, (0, 3)].abs().sum()), 0)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertIn(0, objective.diagnostics['unsupported_reference_hours'])

    def test_aligned_time_context_and_rejection(self):
        from types import SimpleNamespace
        batch = SimpleNamespace(time=torch.tensor([[2., 3., 0.], [4., 0., 0.]]),
            mask=torch.tensor([[1, 1, 0], [1, 0, 0]]),
            category_mask=torch.tensor([[0, 1, 1, 0, 0], [0, 1, 0, 0, 0]]))
        self.assertEqual(aligned_category_times(batch).tolist(), [[0, 2, 3, 0, 0], [0, 4, 0, 0, 0]])
        batch.category_mask[1, 2] = 1
        with self.assertRaises(ValueError):aligned_category_times(batch)

    def test_off_projection_exact_rng_and_identity_config(self):
        states = []
        for config in (None, BalanceConfig(), BalanceConfig('train_ref_v1')):
            p = ConstraintProjection(11, 2, 4, device=self.device, verbose=False,
                outer_iterations=2, inner_iterations=3, generator=torch.Generator(device=self.device).manual_seed(18), early_stop=False)
            p.balance_config = config
            a, b, mask = p.compile_batched_constraints([[([0], [1])], []], self.device)
            y = torch.zeros(2, 11, 4, device=self.device)
            cat = torch.tensor([[1, 1, 0, 0]]*2, device=self.device)
            out = p.project_with_matrices(y, a, b, cat, mask)
            states.append((out, p.generator.get_state()))
            self.assertTrue(torch.equal(out[1], y[1]))
        for a, b in zip(states, states[1:]):
            for x, y in zip(a, b):self.assertTrue(torch.equal(x, y))

    def test_enabled_projector_budget_and_missing_context(self):
        p = ConstraintProjection(11, 2, 4, device=self.device, verbose=False,
            outer_iterations=2, inner_iterations=3, early_stop=False, use_gumbel_softmax=False)
        p.balance_config = BalanceConfig('train_ref_v1', .5, 1., 0.)
        p.category_reference = self.ref
        a, b, mask = p.compile_batched_constraints([[([0], [1])]], self.device)
        y = torch.zeros(1, 11, 4, device=self.device)
        cat = torch.tensor([[1, 1, 0, 0]], device=self.device)
        with self.assertRaises(ValueError):p.project_with_matrices(y, a, b, cat, mask)
        out = p.project_with_matrices(y, a, b, cat, mask, category_times=torch.tensor([[1., 2., 0., 0.]], device=self.device))
        self.assertTrue(torch.isfinite(out).all())
        self.assertEqual(p.last_projection_stats['optimizer_steps'], 6)

    def test_identity_rank_not_shape_and_evaluator_agreement(self):
        p = torch.tensor([6., 3., 1., 0.], device=self.device)
        q = torch.tensor([0., 1., 3., 6.], device=self.device)
        self.assertGreater(float(top_visit_jsd(p, q)), .1)
        self.assertEqual(float(top_visit_jsd(p, p)), 0.)
        gen, real = np.repeat(np.arange(4), [6, 3, 1, 0]), np.repeat(np.arange(4), [0, 1, 3, 6])
        self.assertAlmostEqual(float(top_visit_jsd(p, q)), grank_jsd(gen, real), places=6)
        p.requires_grad_();top_visit_jsd(p, q, top=2).backward()
        self.assertTrue(torch.isfinite(p.grad).all())

    def test_candidates_original_valid_and_losses(self):
        objective = BalancedGeometryObjective(self.ref, self.records, self.logits)
        self.assertTrue(torch.equal(objective.candidates[..., 0], objective.original))
        self.assertLessEqual(objective.candidates.shape[-1], 32)
        p = objective.log_q.clone().requires_grad_()
        rank, daily = objective.distribution_losses(objective.soft(p))
        (rank+daily).backward()
        self.assertTrue(torch.isfinite(p.grad).all())
        cats = objective.data['categories'][objective.candidates]
        expected = objective.data['categories'][objective.original][..., None].expand_as(cats)
        self.assertTrue(torch.equal(cats[objective.valid], expected[objective.valid]))

    def test_hard_distinct_repeated_pois(self):
        objective = BalancedGeometryObjective(self.ref, self.records, self.logits)
        weights = torch.zeros_like(objective.q);weights[..., 0] = 1
        rank, daily = objective.distribution_losses(weights, hard=True)
        # Reference and these eligible original records are exactly the same pool.
        self.assertAlmostEqual(float(rank), 0., places=6)
        self.assertAlmostEqual(float(daily), 0., places=6)

    def test_refinement_reproducible_and_frozen_fields(self):
        config = BalanceConfig('train_ref_v1', .5, 1., .1)
        before = torch.get_rng_state()
        first, stats = refine_balanced_records(self.records, None, self.ref, config, seed=77, scored_logits=self.logits)
        second, again = refine_balanced_records(self.records, None, self.ref, config, seed=77, scored_logits=self.logits)
        assert_invariants(self.records, first, self.ref)
        self.assertEqual(stats['optimizer_steps'], 200)
        self.assertLessEqual(stats['selected_objective'], stats['original_objective'])
        self.assertEqual(stats['selected_path'], again['selected_path'])
        for a, b in zip(first, second):
            self.assertTrue(all(np.array_equal(a[k], b[k]) for k in a))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_empty_singleton_off_skip_without_scoring(self):
        with patch('distribution_balance.untruncated_scores', side_effect=AssertionError('unexpected scoring')):
            for records, config, active in ((self.records, BalanceConfig(), True),
                    (self.records[:2], BalanceConfig('train_ref_v1'), True),
                    (self.records, BalanceConfig('train_ref_v1'), False)):
                result, stats = refine_balanced_records(records, None, self.ref, config, projection_executed=active)
                self.assertIs(result, records)
                self.assertTrue(stats['skipped'])

    def test_bad_config(self):
        for kwargs in ({'constraint_scale': 0}, {'constraint_scale': float('nan')},
                       {'poi_distribution_weight': -1}, {'distribution_balance': 'wrong'}):
            with self.assertRaises(ValueError):BalanceConfig(**kwargs)


if __name__ == '__main__':unittest.main()
