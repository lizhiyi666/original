import tempfile
from pathlib import Path
import unittest

import torch

from constraint_projection import ConstraintProjection
from distance_kl import (DistanceObjective, distance_generator, haversine,
                         load_distance_reference, reference_from_data)


class DistanceKLTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.raw = dict(poi_gps={6: '0,0', 7: '0,0.01', 8: '0,1'},
                        sequences=[dict(checkins=[6, 7]), dict(checkins=[7, 6])])
        self.ref = reference_from_data(self.raw, bins=16)
        self.y = torch.full((3, 4, 11), -8.)
        self.y[:, :2, 4:6] = 0
        self.y[:, 2:, 6:9] = 0
        self.y[:, 2, 6] = 2
        self.y[:, 3, 8] = 2
        self.poi = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 0]], dtype=torch.bool)
        self.cat = torch.tensor([[1, 1, 0, 0]] * 3, dtype=torch.bool)

    def test_haversine_and_repeated_poi(self):
        coords = torch.tensor([[0., 0.], [0., 1.]], dtype=torch.float64)
        self.assertAlmostEqual(float(haversine(coords[0], coords[1])), 111.1949266, places=5)
        self.assertEqual(float(haversine(coords[0], coords[0])), 0)
        zero = reference_from_data(dict(self.raw, sequences=[dict(checkins=[6, 6])]))
        self.assertTrue(torch.isfinite(zero.target).all())
        self.assertAlmostEqual(float(zero.target.sum()), 1)

    def test_train_only_content_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'toy_train.pkl'
            torch.save(self.raw, path)
            first = load_distance_reference(path, 16)
            self.assertIs(first, load_distance_reference(path, 16))
            torch.save(dict(self.raw, sequences=[dict(checkins=[6, 8])]), path)
            self.assertNotEqual(first.fingerprint, load_distance_reference(path, 16).fingerprint)
            with self.assertRaisesRegex(ValueError, 'train'):
                load_distance_reference(Path(folder) / 'toy_test.pkl')

    def test_mc_gradient_and_fixed_rng(self):
        y = self.y.clone().requires_grad_()
        objective = DistanceObjective(self.ref, y.detach(), self.poi, torch.tensor([1, 0, 1], dtype=torch.bool), paths=8)
        self.assertEqual(len(objective.rows), 1)
        before = torch.get_rng_state()
        noise = objective.noise(distance_generator(4, 'cpu'))
        torch.testing.assert_close(noise[0], objective.noise(distance_generator(4, 'cpu'))[0])
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        loss = objective.loss(y, noise)
        loss.backward()
        self.assertTrue(torch.isfinite(y.grad).all())
        self.assertGreater(float(y.grad[0, 2:].norm()), 0)
        self.assertEqual(float(y.grad[1:].norm()), 0)
        self.assertEqual(float(y.grad[0, :2].norm()), 0)
        self.assertTrue(torch.allclose(objective.coverage, torch.ones_like(objective.coverage)))

    def test_deterministic_objective_can_decrease(self):
        y = self.y[:1].clone().requires_grad_()
        # Moderate mismatch: avoid testing optimisation of an already saturated tail.
        y.data[:, 3, 8] = -2
        nearby = reference_from_data(dict(self.raw, poi_gps={6: '0,0', 7: '0,0.01', 8: '0,0.03'}), bins=16)
        objective = DistanceObjective(nearby, y.detach(), self.poi[:1], torch.tensor([True]))
        opt = torch.optim.SGD([y], lr=.01)
        noise = objective.noise(None, stochastic=False)
        initial = float(objective.loss(y, noise))
        for _ in range(30):
            opt.zero_grad(); objective.loss(y, noise).backward(); opt.step()
        self.assertLess(float(objective.loss(y, noise)), initial)

    def test_fixed_mc_objective_can_decrease(self):
        nearby = reference_from_data(dict(self.raw, poi_gps={6: '0,0', 7: '0,0.01', 8: '0,0.03'}), bins=16)
        y = torch.zeros(1, 2, 11, requires_grad=True)
        objective = DistanceObjective(nearby, y.detach(), torch.ones(1, 2, dtype=torch.bool),
                                      torch.tensor([True]), paths=32)
        noise = objective.noise(distance_generator(42, 'cpu'))
        initial = float(objective.loss(y, noise))
        opt = torch.optim.SGD([y], lr=.01)
        for _ in range(150):
            opt.zero_grad(); objective.loss(y, noise).backward(); opt.step()
        self.assertLess(float(objective.loss(y, noise)), initial)

    def test_hard_forward_uses_real_route_not_mean_coordinates(self):
        y = self.y[:1].clone().requires_grad_()
        objective = DistanceObjective(self.ref, y.detach(), self.poi[:1], torch.tensor([True]), paths=8)
        ids = objective.rows[0][2]
        noise = torch.full((8, *ids.shape), -10000.)
        for position, token in enumerate((6, 7)):
            noise[:, position, torch.where(ids[position] == token)[0]] = 10000.
        self.assertAlmostEqual(float(objective.loss(y, [noise])), 0, places=6)

    def test_empty_and_singleton_do_not_draw_distance_noise(self):
        for mask in (torch.zeros_like(self.poi), self.poi & torch.tensor([1, 1, 1, 0], dtype=torch.bool)):
            objective = DistanceObjective(self.ref, self.y, mask, torch.ones(3, dtype=torch.bool))
            rng = distance_generator(12, 'cpu')
            before = rng.get_state()
            self.assertEqual(objective.noise(rng), [])
            self.assertTrue(torch.equal(before, rng.get_state()))
        p = self.projector(projection_distance_kl_weight=1, distance_reference=self.ref)
        a, b, mask = p.compile_batched_constraints([[([0], [1])]] * 3, 'cpu')
        before = torch.get_rng_state()
        original = self.y.transpose(1, 2)
        result = p.project_with_matrices(original, a, b, torch.zeros_like(self.cat), mask, poi_mask=self.poi)
        self.assertTrue(torch.equal(original, result))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def projector(self, **kwargs):
        return ConstraintProjection(11, 2, 4, device='cpu', verbose=False,
            outer_iterations=2, inner_iterations=3, use_gumbel_softmax=False, **kwargs)

    def test_enabled_projection_updates_poi_only_on_eligible_rows(self):
        p = self.projector(projection_distance_kl_weight=1, distance_reference=self.ref)
        a, b, mask = p.compile_batched_constraints([[([0], [1])], [], [([0], [1])]], 'cpu')
        original = self.y.transpose(1, 2)
        output = p.project_with_matrices(original, a, b, self.cat, mask, poi_mask=self.poi)
        self.assertGreater(float((output[0, :, 2:] - original[0, :, 2:]).norm()), 0)
        torch.testing.assert_close(output[1], original[1], rtol=0, atol=0)
        torch.testing.assert_close(output[2, :, 2:], original[2, :, 2:], rtol=0, atol=1e-7)
        self.assertEqual(p.last_projection_stats['distance_active_rows'], 1)
        self.assertEqual(p.last_projection_stats['optimizer_steps'], 6)
        self.assertGreater(p.last_projection_stats['distance_poi_gradient_norm'], 0)

    def test_disabled_identical_and_missing_reference_fails(self):
        states = []
        for weight in (None, 0):
            p = self.projector(**({} if weight is None else dict(projection_distance_kl_weight=0, distance_reference=self.ref)))
            a, b, mask = p.compile_batched_constraints([[([0], [1])]] * 3, 'cpu')
            torch.manual_seed(19)
            states.append((p.project_with_matrices(self.y.transpose(1, 2), a, b, self.cat, mask), torch.get_rng_state()))
        for left, right in zip(*states):
            self.assertTrue(torch.equal(left, right))
        p = self.projector(projection_distance_kl_weight=1)
        with self.assertRaisesRegex(ValueError, 'reference'):
            p.project_with_matrices(self.y.transpose(1, 2), a, b, self.cat)


if __name__ == '__main__':
    unittest.main()
