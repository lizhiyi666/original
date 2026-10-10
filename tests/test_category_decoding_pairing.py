"""诊断层配对比较（修正后）小批量验证。

run_category_decoding_check.compare_pair 不再要求 off/on 逐条件等长，而是基于
共享时间缓存对齐：On 必须覆盖每个缓存 POI 槽位，Off 为其有序子集（旧导出器把
非目录 token 槽位静默删除）。全局导出器 to_seq_list 不改动。
"""
import copy
import unittest

import numpy as np
import torch  # noqa: F401  (fixture 依赖)

from tests.test_category_cache_audit import fixture
from tools.run_category_decoding_check import compare_pair


def _trace(indices):
    return dict(indices=list(indices), global_start=0,
                spatial_rng_before='a', spatial_rng_after='b', projection_rng_after='c',
                distance_rng_before=None, distance_rng_after=None,
                global_rng_unchanged=True, effective_constraints=False, projection_stats=[])


class ComparePairAlignedBasisTests(unittest.TestCase):
    def test_restored_events_pass_under_aligned_basis(self):
        items, off, on, mapping, gps = fixture()
        tr = _trace(items[0]['indices'])
        alignment, matched = compare_pair(items, off, on, [tr], [copy.deepcopy(tr)], mapping, gps)
        # fixture: 2 个被旧导出器删除、被 On 还原的事件
        self.assertEqual(alignment['restored_events'], 2)
        self.assertEqual([len(r['checkins']) for r in matched], [1, 2, 0])

    def test_count_guard_rejects_misaligned_batches(self):
        items, off, on, mapping, gps = fixture()
        tr = _trace(items[0]['indices'])
        with self.assertRaisesRegex(RuntimeError, 'align with the time cache'):
            compare_pair(items, off[:2], on, [tr], [copy.deepcopy(tr)], mapping, gps)

    def test_on_must_cover_every_cached_slot(self):
        items, off, on, mapping, gps = fixture()
        on = copy.deepcopy(on)
        on[0]['checkins'] = on[0]['checkins'][:1]  # On 丢了一个缓存槽位
        tr = _trace(items[0]['indices'])
        with self.assertRaisesRegex(RuntimeError, 'retain every cached event'):
            compare_pair(items, off, on, [tr], [copy.deepcopy(tr)], mapping, gps)

    def test_trace_rng_mismatch_still_rejected(self):
        items, off, on, mapping, gps = fixture()
        a = _trace(items[0]['indices']); b = _trace(items[0]['indices']); b['spatial_rng_after'] = 'different'
        with self.assertRaises(RuntimeError):
            compare_pair(items, off, on, [a], [b], mapping, gps)


if __name__ == '__main__':
    unittest.main()
