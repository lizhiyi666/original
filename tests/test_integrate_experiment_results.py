"""Delivery safety tests; no network or experiment dependencies."""
import hashlib
import sys
import tempfile
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from integrate_experiment_results import safe_child, copy_no_overwrite, digest, local_path, RUNS


class DeliverySafety(unittest.TestCase):
    def test_relative_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError): safe_child(Path(d), '../outside')

    def test_known_remote_mapping(self):
        p=local_path('/root/experiments/pcdg/ablation-v1/experiment_runs/pcdg-ablation-v1/seed-1/generated.pkl')
        self.assertEqual(p, RUNS/'pcdg-ablation-v1/seed-1/generated.pkl')

    def test_unrecognized_remote_mapping_is_rejected(self):
        with self.assertRaises(ValueError): local_path('/unknown/generated.pkl')

    def test_copy_is_verified_and_idempotent(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); src=p/'src'; dst=p/'dst'
            src.write_bytes(b'known experiment data')
            h=hashlib.sha256(src.read_bytes()).hexdigest()
            self.assertTrue(copy_no_overwrite(src,dst,h))
            self.assertFalse(copy_no_overwrite(src,dst,h))
            self.assertEqual(digest(dst),h)

    def test_conflicting_destination_is_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); src=p/'src'; dst=p/'dst'
            src.write_bytes(b'new'); dst.write_bytes(b'old')
            with self.assertRaises(RuntimeError): copy_no_overwrite(src,dst,digest(src))
            self.assertEqual(dst.read_bytes(),b'old')


if __name__=='__main__': unittest.main()
