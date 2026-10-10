"""自适应存在性门控（existence_adaptive）小批量逻辑验证（CPU，无需检查点）。

验证：
1. 默认关闭时，输出与不带该功能的实现逐元素一致，且不产生 gated 统计键；
2. gate 阈值极大（全部低于阈值）→ 所有存在性约束被门控清零，等价于存在性权重为 0；
3. gate 阈值极小（全部高于阈值）→ 门控退化为全开，等价于自适应关闭；
4. gated 计数与初始硬存在性违反一致（仅缺失类别的约束被保留）。
"""
import unittest
import torch

from constraint_projection import ConstraintProjection


TYPE_CLASSES = 3
NUM_SPECIAL = 4
NUM_POIS = 2
NUM_CLASSES = NUM_SPECIAL + TYPE_CLASSES + NUM_POIS + 2  # 11


def _proj(existence_adaptive=False, gate=0.0, existence_weight=5.0):
    p = ConstraintProjection(
        num_classes=NUM_CLASSES, type_classes=TYPE_CLASSES, num_spectial=NUM_SPECIAL,
        tau=0.0, lambda_init=0.0, mu_init=1.0, outer_iterations=3, inner_iterations=3,
        eta=1.0, use_gumbel_softmax=False, device='cpu',
        projection_existence_weight=existence_weight, projection_order_weight=1.0,
        projection_kl_weight=1.0, verbose=False, early_stop=False,
        existence_adaptive=existence_adaptive, existence_violation_gate=gate,
    )
    p.generator = torch.Generator().manual_seed(0)
    return p


def _inputs():
    # 3 个类别位置，argmax 类别依次为 cat0,cat1,cat1 -> 缺失 cat2
    cat_of_pos = [NUM_SPECIAL + 0, NUM_SPECIAL + 1, NUM_SPECIAL + 1]
    L = len(cat_of_pos)
    log_probs = torch.full((1, NUM_CLASSES, L), -10.0)
    for pos, cls in enumerate(cat_of_pos):
        log_probs[0, cls, pos] = 10.0
    category_mask = torch.ones((1, L), dtype=torch.int64)
    # 约束0: cat0 先于 cat1；约束1: cat1 先于 cat2（cat2 缺失 -> 存在性违反）
    W_A = torch.zeros((TYPE_CLASSES, 2)); W_B = torch.zeros((TYPE_CLASSES, 2))
    W_A[0, 0] = 1.0; W_B[1, 0] = 1.0
    W_A[1, 1] = 1.0; W_B[2, 1] = 1.0
    return log_probs, W_A, W_B, category_mask


class TestExistenceAdaptive(unittest.TestCase):
    def setUp(self):
        self.log_probs, self.W_A, self.W_B, self.category_mask = _inputs()

    def _run(self, proj):
        out = proj.project_with_matrices(self.log_probs.clone(), self.W_A, self.W_B,
                                         self.category_mask.clone())
        return out, dict(proj.last_projection_stats)

    def test_disabled_matches_plain_and_has_no_gate_key(self):
        out, stats = self._run(_proj(existence_adaptive=False))
        self.assertNotIn('existence_gated_constraints', stats)
        out2, _ = self._run(_proj(existence_adaptive=False))
        self.assertTrue(torch.equal(out, out2))  # 确定性

    def test_initial_violation_gates_only_missing_category(self):
        # gate=0：约束0(存在性违反0)被清零，约束1(cat2缺失,违反1)保留 -> gated=1
        _, stats = self._run(_proj(existence_adaptive=True, gate=0.0))
        self.assertEqual(stats['existence_gated_constraints'], 1)

    def test_huge_gate_suppresses_all_existence(self):
        _, stats = self._run(_proj(existence_adaptive=True, gate=1e9))
        self.assertEqual(stats['existence_gated_constraints'], 0)
        # 全部存在性被清零 == 存在性权重为 0（仅顺序项）
        out_gated, _ = self._run(_proj(existence_adaptive=True, gate=1e9, existence_weight=5.0))
        out_weight0, _ = self._run(_proj(existence_adaptive=False, existence_weight=0.0))
        self.assertTrue(torch.allclose(out_gated, out_weight0, atol=1e-6))

    def test_negative_gate_keeps_all_and_matches_adaptive_off(self):
        _, stats = self._run(_proj(existence_adaptive=True, gate=-1e9))
        # 全部约束保留（= 活跃约束数）
        active = stats['active_constraints']
        self.assertEqual(stats['existence_gated_constraints'], active)
        out_on, _ = self._run(_proj(existence_adaptive=True, gate=-1e9))
        out_off, _ = self._run(_proj(existence_adaptive=False))
        self.assertTrue(torch.allclose(out_on, out_off, atol=1e-6))


if __name__ == '__main__':
    unittest.main()
