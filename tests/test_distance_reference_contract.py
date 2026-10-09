"""Lock reference semantics before introducing an explicitly selected backend."""
import unittest

import torch

from distance_kl import DistanceObjective, distance_generator, soft_histogram
from tools.distance_benchmark_fixture import fixture, make_projector


class ReferenceContractTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_ragged_rows_equal_weight_and_exact_rng_calls(self):
        d = fixture(batch=18, max_pois=6, candidates=5)
        objective = DistanceObjective(d['reference'], d['y'], d['poi'], d['active'])
        rng, expected_rng = distance_generator(8, 'cpu'), distance_generator(8, 'cpu')
        actual = objective.noise(rng)
        for row, noise in zip(objective.rows, actual):
            expected = torch.rand((8, *row[2].shape), generator=expected_rng)
            expected = -torch.log(-torch.log(expected.clamp(min=1e-8, max=1-1e-7)))
            self.assertTrue(torch.equal(noise, expected))
        self.assertTrue(torch.equal(rng.get_state(), expected_rng.get_state()))
        y = d['y'].clone().requires_grad_()
        routes = []
        for b, positions, ids, edges in objective.rows:
            weights = y[b, positions[:, None], ids].softmax(-1)
            routes.append((weights[:-1, :, None] * edges * weights[1:, None, :]).sum())
        hist = soft_histogram(torch.stack(routes), objective.centers, objective.bandwidth)
        expected = (objective.target * (objective.target.log() - hist.log())).sum()
        actual = objective.loss(y, objective.noise(None, False))
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        actual.backward()
        eligible = d['poi'] & d['active'][:, None] & (d['poi'].sum(1) >= 2)[:, None]
        self.assertEqual(y.grad[~eligible].count_nonzero().item(), 0)

    def test_full_fixed_budget_and_independent_streams(self):
        d = fixture(batch=4, max_pois=3, candidates=5)
        p, args = make_projector(d)
        global_before = torch.get_rng_state()
        output = p.project_with_matrices(*args, poi_mask=d['poi'])
        self.assertEqual(p.last_projection_stats['optimizer_steps'], 500)
        self.assertEqual(p.last_projection_stats['outer_iterations'], 10)
        self.assertTrue(torch.equal(global_before, torch.get_rng_state()))
        self.assertTrue(torch.equal(output[~d['active']], args[0][~d['active']]))
        self.assertTrue(torch.isfinite(output).all())


if __name__ == '__main__':
    unittest.main()
