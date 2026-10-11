"""Unit tests for the formal three-seed category-decoding orchestrator helpers.

Covers the pure aggregation/report/verify-only surface (no GPU/checkpoint). The
end-to-end On sampling is exercised on the server; here we validate the
statistics, None-safety, report rendering, and the read-only delivery check.
"""
import json
import unittest
from pathlib import Path
import tempfile

from tools import run_category_formal as rcf
from experiment_io import sha256_file


class FormalHelperTests(unittest.TestCase):
    def test_mean_sd_basic_and_single(self):
        self.assertEqual(rcf.mean_sd([1.0, 2.0, 3.0]), {'mean': 2.0, 'sample_sd': 1.0, 'n': 3})
        s = rcf.mean_sd([5.0])
        self.assertEqual(s['mean'], 5.0)
        self.assertEqual(s['sample_sd'], 0.0)

    def test_mean_sd_none_contaminated_is_undefined(self):
        s = rcf.mean_sd([1.0, None, 3.0])
        self.assertIsNone(s['mean'])
        self.assertIsNone(s['sample_sd'])
        self.assertEqual(s['n'], 2)

    def test_aggregate_per_key(self):
        agg = rcf.aggregate([{'a': 0.1, 'b': 0.2}, {'a': 0.3, 'b': 0.4}, {'a': 0.5, 'b': 0.6}])
        self.assertAlmostEqual(agg['a']['mean'], 0.3)
        self.assertAlmostEqual(agg['b']['mean'], 0.4)
        self.assertEqual(agg['a']['n'], 3)

    def test_build_report_covers_both_cities_and_columns(self):
        summary = {c: dict(
            aggregate=dict(off=rcf.aggregate([{'Distance': 0.2, 'strict_ovr': 0.3, 'G-RANK': 0.19}]),
                           on=rcf.aggregate([{'Distance': 0.1, 'strict_ovr': 0.02, 'G-RANK': 0.20}]),
                           on_matched=rcf.aggregate([{'Distance': 0.1, 'strict_ovr': 0.03, 'G-RANK': 0.20}])),
            restored_events_by_seed={135398: 3, 135399: 2, 135400: 4}) for c in rcf.CITIES}
        report = rcf.build_report(summary, {})
        for city in rcf.CITIES:
            self.assertIn(city, report)
        self.assertIn('Distance', report)
        self.assertIn('On-matched', report)
        self.assertIn('batch=512', report)  # Istanbul caveat disclosed

    def test_verify_only_passes_and_detects_tamper(self):
        import sys
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / 'run'
            out.mkdir()
            (out / 'summary.json').write_text('{"ok": true}', encoding='utf-8')
            files = {'summary.json': sha256_file(out / 'summary.json')}
            (out / 'audit.json').write_text(json.dumps(dict(state='passed', files=files)), encoding='utf-8')
            argv = sys.argv
            try:
                sys.argv = ['x', '--verify-only', str(out)]
                rcf.main()  # should not raise
            finally:
                sys.argv = argv
            # tamper
            (out / 'summary.json').write_text('{"ok": false}', encoding='utf-8')
            try:
                sys.argv = ['x', '--verify-only', str(out)]
                with self.assertRaisesRegex(RuntimeError, 'Local hash mismatch'):
                    rcf.main()
            finally:
                sys.argv = argv


if __name__ == '__main__':
    unittest.main()
