"""方案A：类别一致的 POI 解码——小批量逻辑验证。

不依赖检查点或 GPU：构造与真实 content 布局一致的小批次
([start, cat_1..cat_L, sep, poi_1..poi_L, end]) 和合成 logits，验证：

1. enable_category_consistent_decoding 正确构造 class->category 查表；
2. 开启后，每个 POI 位置仅保留与对齐类别一致的 POI 类，其余 POI 类被置 -70；
3. 类别位置、特殊符与非 POI 类不受影响；
4. 目标类别无可用 POI 时跳过该位置（不产生整列 -70 的空分布）；
5. 关闭时输出与输入逐元素一致（保持既有默认行为）；
6. 掩码后按 gumbel 采样，生成 POI 与其对齐类别的不一致率为 0。
"""
import types
import unittest

import torch

from discrete_diffusion.diffusion_transformer import (
    DiffusionTransformer,
    index_to_log_onehot,
    log_onehot_to_index,
)


def _dummy(num_classes):
    d = types.SimpleNamespace()
    d.num_classes = num_classes
    d.category_consistent_decoding = False
    d.poi_class_to_category = None
    d.sampling_generator = None
    return d


def _batch(unpadded_lengths, content_len):
    B = len(unpadded_lengths)
    cat = torch.zeros(B, content_len, dtype=torch.int64)
    poi = torch.zeros(B, content_len, dtype=torch.int64)
    for i, L in enumerate(unpadded_lengths):
        cat[i, 1:1 + L] = 1
        poi[i, L + 2:L + 2 + L] = 1
    return types.SimpleNamespace(
        category_mask=cat, poi_mask=poi,
        unpadded_length=torch.tensor(unpadded_lengths, dtype=torch.int64),
    )


class TestCategoryConsistentDecoding(unittest.TestCase):
    def setUp(self):
        # 词表：4 特殊符(0..3)，3 类别(4,5,6)，4 POI(7,8,9,10)，+2 mask -> 13 类
        self.num_classes = 3 + 4 + 4 + 2
        self.poi_category = {7: 4, 8: 4, 9: 5, 10: 6}
        self.dummy = _dummy(self.num_classes)

    def _logits_one_seq(self, L, cat_tokens, poi_pref):
        """构造 (1,C,content_len) logits（log-prob 量级，<=0）：
        类别位置 argmax=cat_tokens[k]，POI 位置 argmax=poi_pref[k]（可能是错类别）。"""
        content_len = 2 * L + 3
        logits = torch.full((1, self.num_classes, content_len), -10.0)
        for k in range(L):
            logits[0, cat_tokens[k], 1 + k] = -0.5          # 类别位置最高
            logits[0, poi_pref[k], L + 2 + k] = -0.5         # POI 位置：偏好 token
        return logits, content_len

    def test_table_construction(self):
        DiffusionTransformer.enable_category_consistent_decoding(self.dummy, self.poi_category)
        t = self.dummy.poi_class_to_category
        self.assertTrue(self.dummy.category_consistent_decoding)
        self.assertEqual(t.shape[0], self.num_classes)
        self.assertEqual(int(t[7]), 4)
        self.assertEqual(int(t[8]), 4)
        self.assertEqual(int(t[9]), 5)
        self.assertEqual(int(t[10]), 6)
        for non_poi in [0, 1, 2, 3, 4, 5, 6, 11, 12]:
            self.assertEqual(int(t[non_poi]), -1)

    def test_mask_restricts_to_intended_category(self):
        DiffusionTransformer.enable_category_consistent_decoding(self.dummy, self.poi_category)
        # L=2，类别计划 [4,5]；POI 位置故意偏好错类别 POI(9->cat5, 10->cat6)
        logits, cl = self._logits_one_seq(2, cat_tokens=[4, 5], poi_pref=[9, 10])
        batch = _batch([2], cl)
        out = DiffusionTransformer._apply_category_consistent_poi_mask(self.dummy, logits.clone(), batch)

        NEG = -70.0
        # POI 位置0 (绝对 pos=4) 目标类别=4 -> 仅 7,8 允许
        p0 = 4
        self.assertGreater(out[0, 7, p0].item(), NEG + 1)
        self.assertGreater(out[0, 8, p0].item(), NEG + 1)
        self.assertAlmostEqual(out[0, 9, p0].item(), NEG, places=5)
        self.assertAlmostEqual(out[0, 10, p0].item(), NEG, places=5)
        # POI 位置1 (绝对 pos=5) 目标类别=5 -> 仅 9 允许
        p1 = 5
        self.assertGreater(out[0, 9, p1].item(), NEG + 1)
        for wrong in [7, 8, 10]:
            self.assertAlmostEqual(out[0, wrong, p1].item(), NEG, places=5)
        # 类别位置与特殊符不受影响
        for cat_pos in [1, 2]:
            self.assertTrue(torch.equal(out[0, :, cat_pos], logits[0, :, cat_pos]))
        # 非 POI 类在 POI 位置也不被改动
        for non_poi in [0, 1, 2, 3, 4, 5, 6, 11, 12]:
            self.assertAlmostEqual(out[0, non_poi, p0].item(), logits[0, non_poi, p0].item(), places=5)

    def test_skip_when_no_poi_for_intended_category(self):
        # 查表中去掉类别6的 POI(10)，令 token 10 变为非 POI
        DiffusionTransformer.enable_category_consistent_decoding(self.dummy, {7: 4, 8: 4, 9: 5})
        # 类别计划 [6,4]：pos0 目标类别6无可用 POI -> 跳过；pos1 目标类别4 -> 仅7,8
        logits, cl = self._logits_one_seq(2, cat_tokens=[6, 4], poi_pref=[7, 9])
        batch = _batch([2], cl)
        out = DiffusionTransformer._apply_category_consistent_poi_mask(self.dummy, logits.clone(), batch)
        p0 = 4  # 目标类别6无 POI -> 整列保持不变
        self.assertTrue(torch.equal(out[0, :, p0], logits[0, :, p0]))
        p1 = 5  # 目标类别4 -> 9 被屏蔽
        self.assertAlmostEqual(out[0, 9, p1].item(), -70.0, places=5)
        self.assertGreater(out[0, 7, p1].item(), -69.0)

    def test_disabled_is_noop(self):
        self.dummy.category_consistent_decoding = False
        logits, cl = self._logits_one_seq(2, [4, 5], [9, 10])
        batch = _batch([2], cl)
        out = DiffusionTransformer._apply_category_consistent_poi_mask(self.dummy, logits.clone(), batch)
        self.assertTrue(torch.equal(out, logits))

    def test_sampled_pois_have_zero_category_mismatch(self):
        torch.manual_seed(0)
        DiffusionTransformer.enable_category_consistent_decoding(self.dummy, self.poi_category)
        # 每个 POI 位置：错类别 POI 的 logit 更高(-0.5)，但同类别也有候选(-1.5)。
        # 不掩码会选错类别；掩码后应回退到同类别 POI。
        plans = [
            # (类别计划, 错类别偏好POI, 同类别候选POI)
            ([4, 5, 4], [10, 7, 9], [7, 9, 8]),
            ([6, 4], [8, 9], [10, 7]),
        ]
        content_len = max(2 * len(c) + 3 for c, _, _ in plans)
        B = len(plans)
        logits = torch.full((B, self.num_classes, content_len), -10.0)
        lengths = []
        for i, (cats, wrong, right) in enumerate(plans):
            L = len(cats); lengths.append(L)
            for k in range(L):
                logits[i, cats[k], 1 + k] = -0.5
                logits[i, wrong[k], L + 2 + k] = -0.5   # 错类别偏好更高
                logits[i, right[k], L + 2 + k] = -1.5   # 同类别候选次之
        batch = _batch(lengths, content_len)
        masked = DiffusionTransformer._apply_category_consistent_poi_mask(self.dummy, logits.clone(), batch)
        sample = DiffusionTransformer.log_sample_categorical(self.dummy, masked)
        tokens = log_onehot_to_index(sample)  # (B, content_len)

        mismatches = 0
        total = 0
        table = self.dummy.poi_class_to_category
        for i, (cats, _, _) in enumerate(plans):
            L = len(cats)
            for k in range(L):
                poi_token = int(tokens[i, L + 2 + k])
                total += 1
                self.assertGreaterEqual(int(table[poi_token]), 0, f"non-POI sampled at i={i},k={k}: {poi_token}")
                if int(table[poi_token]) != cats[k]:
                    mismatches += 1
        self.assertEqual(total, 5)
        self.assertEqual(mismatches, 0)


if __name__ == "__main__":
    unittest.main()
