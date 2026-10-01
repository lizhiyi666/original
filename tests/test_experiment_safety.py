import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from experiment_io import strict_test_matrix, publish_torch, sha256_file, validate_sequences
from merge_results import merge_parts
from constraint_projection import ConstraintProjection
from discrete_diffusion.diffusion_transformer import DiffusionTransformer, ConditionEmbeddingModel


class ConstraintTests(unittest.TestCase):
    def test_position_overflow_rejected_before_allocating_embeddings(self):
        model = SimpleNamespace(max_position_embeddings=3000)
        with self.assertRaises(ValueError):
            ConditionEmbeddingModel.forward(model, SimpleNamespace(time=torch.zeros(1, 3001)))

    def matrix(self, cats):
        return strict_test_matrix({"checkins": cats}, {4: 4, 5: 5, 6: 6}, {4: 0, 5: 1, 6: 2})

    def test_absent_categories_have_no_edges(self):
        matrix = self.matrix([4, 4, 5])
        self.assertEqual(matrix.sum().item(), 1)
        self.assertEqual(matrix[0, 1].item(), 1)
        self.assertEqual(matrix[2].sum().item(), 0)
        self.assertEqual(matrix[:, 2].sum().item(), 0)

    def test_interleaved_and_empty(self):
        for cats in ([], [4], [4, 4], [4, 5, 4, 5]):
            self.assertEqual(self.matrix(cats).sum().item(), 0)

    def test_reversed_and_repeated(self):
        self.assertEqual(self.matrix([5, 5, 4, 4])[1, 0].item(), 1)

    def test_unknown_poi_fails(self):
        with self.assertRaises(ValueError):
            self.matrix([99])

    def test_real_projection_gradient_and_empty_batch(self):
        projector = ConstraintProjection(num_classes=9, type_classes=3, num_spectial=4,
                                         outer_iterations=2, inner_iterations=2, device="cpu")
        self.assertEqual(projector.compile_batched_constraints([[]], "cpu"), (None, None, None))
        w_a, w_b, mask = projector.compile_batched_constraints([[([0], [1])]], "cpu")
        logits = torch.log_softmax(torch.randn(1, 9, 7), dim=1)
        positions = torch.tensor([[0, 1, 1, 0, 0, 0, 0]])
        with torch.no_grad():
            projected = projector.project_with_matrices(logits, w_a, w_b, positions, mask)
        self.assertEqual(projected.shape, logits.shape)
        self.assertTrue(torch.isfinite(projected).all())

    def test_native_and_projection_switches(self):
        class Projector:
            calls = 0
            def compile_batched_constraints(self, constraints, device):
                return torch.ones(1), torch.ones(1), None
            def project_with_matrices(self, x, *args, **kwargs):
                self.calls += 1
                return x
        projector = Projector()
        logits = torch.zeros(1, 9, 7)
        model = SimpleNamespace(projection_last_k_steps=40, projection_frequency=4,
                                use_constraint_projection=False, constraint_projector=projector,
                                p_pred=lambda *args: (logits, logits), log_sample_categorical=lambda x: x)
        batch = SimpleNamespace(category_mask=torch.ones(1, 7))
        DiffusionTransformer.p_sample(model, logits, None, None, batch, [[([0], [1])]], 0)
        self.assertEqual(projector.calls, 0)
        model.use_constraint_projection = True
        DiffusionTransformer.p_sample(model, logits, None, None, batch, [[([0], [1])]], 0)
        self.assertEqual(projector.calls, 1)
        self.assertEqual(model.projection_call_count, 1)
        DiffusionTransformer.p_sample(model, logits, None, None, batch, [[([0], [1])]], 41)
        self.assertEqual(projector.calls, 1)


class MergeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.base = self.root / "D"
        self.base.mkdir()
        self.sequence = {"arrival_times": [1.0, 2.0], "checkins": [11, 12]}
        torch.save({"sequences": [self.sequence] * 3}, self.base / "D_test.pkl")
        self.metadata = dict(total_samples=3, world_size=2, run_id="r", output_tag="r_native",
                             data_name="D", dataset_sha256=sha256_file(self.base / "D_test.pkl"))
        self.parts = []
        for rank, indices in enumerate(([0, 1], [2])):
            part = dict(metadata=copy.deepcopy(self.metadata), rank=rank, test_indices=indices,
                        sequences=[self.sequence] * len(indices), t_max=24.0, projection_calls=0)
            self.parts.append(part)

    def tearDown(self):
        self.tmp.cleanup()

    def save(self, rank):
        torch.save(self.parts[rank], self.base / f"D_r_native_generated_part{rank}.pkl")

    def merge(self):
        return merge_parts("D", "r", 2, "r_native", self.root, expected_count=3)

    def test_complete_merge_retains_shards_and_can_reverify(self):
        self.save(0)
        self.save(1)
        path = self.merge()
        self.assertEqual(torch.load(path, weights_only=False)["test_indices"], [0, 1, 2])
        self.assertTrue((self.base / "D_r_native_generated_part0.pkl").exists())
        self.assertEqual(self.merge(), path)

    def test_missing_shard_never_publishes(self):
        self.save(0)
        with self.assertRaises(FileNotFoundError):
            self.merge()
        self.assertFalse((self.base / "D_r_native_generated.pkl").exists())

    def test_duplicate_indices_rejected(self):
        self.parts[1]["test_indices"] = [1]
        self.save(0)
        self.save(1)
        with self.assertRaises(ValueError):
            self.merge()

    def test_mixed_parameters_rejected(self):
        self.parts[1]["metadata"]["output_tag"] = "r_projection"
        self.save(0)
        self.save(1)
        with self.assertRaises(ValueError):
            self.merge()

    def test_changed_dataset_rejected(self):
        self.save(0)
        self.save(1)
        torch.save({"changed": True}, self.base / "D_test.pkl")
        with self.assertRaises(ValueError):
            self.merge()

    def test_atomic_publish_refuses_overwrite(self):
        path = self.base / "result.pkl"
        publish_torch(path, {"a": 1})
        with self.assertRaises(FileExistsError):
            publish_torch(path, {"a": 2})
        self.assertEqual(torch.load(path, weights_only=False), {"a": 1})

    def test_nan_generation_rejected(self):
        with self.assertRaises(ValueError):
            validate_sequences([dict(arrival_times=[float("nan")], checkins=[11])])


if __name__ == "__main__":
    unittest.main()
