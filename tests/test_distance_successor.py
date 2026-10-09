import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from distance_kl import distance_metadata
from experiment_io import atomic_json, sha256_file
from tools.resume_distance_batched import (SuccessorStudy, child, copy_verified, inherited_record,
    inventory, snapshot_parent)
from tools.run_two_city_study import NY, IST
from tools.ablation_common import SEEDS
from tools.run_two_city_distance_v2 import expected_keys
from tools.unified_tables import key, render


class SuccessorTests(unittest.TestCase):
    def test_snapshot_preserves_bytes_and_rejects_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            parent, snapshot = root / 'parent', root / 'successor' / 'inherited'
            atomic_json(parent / 'manifest.json', {'version': 'legacy'})
            atomic_json(parent / 'registry.json', {'count': 35})
            atomic_json(parent / 'wandb' / 'ignored.json', {})
            atomic_json(parent / 'pipeline.lock', {})
            before = inventory(parent)
            receipt = root / 'successor' / 'snapshot.json'
            proof = snapshot_parent(parent, snapshot, receipt)
            self.assertEqual(proof['files'], before)
            self.assertEqual(inventory(snapshot), before)
            self.assertEqual(inventory(parent), before)
            self.assertEqual(snapshot_parent(parent, snapshot, receipt), proof)
            atomic_json(parent / 'registry.json', {'count': 36})
            with self.assertRaisesRegex(RuntimeError, 'inventory changed'):
                snapshot_parent(parent, snapshot, receipt)
            self.assertEqual(inventory(snapshot), before)

    def test_snapshot_collision_and_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            atomic_json(root / 'source.json', {'a': 1})
            atomic_json(root / 'target.json', {'a': 2})
            with self.assertRaisesRegex(RuntimeError, 'Different'):
                copy_verified(root / 'source.json', root / 'target.json', sha256_file(root / 'source.json'))
            with self.assertRaises(RuntimeError):
                child(root, '../outside')

    def test_inherited_record_is_not_relabelled_or_reserialized(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            parent, snapshot = root / 'parent', root / 'next' / 'inherited' / 'legacy'
            record = dict(path=str(parent / 'city' / 'generated.pkl'), sha256='sealed', origin='new',
                origin_manifest_sha256='parent-manifest', source_shards=[dict(path=str(parent / 'city' / 'part.pkl'), sha256='part')],
                metrics={'metric': .25}, timing={'seconds': 2.})
            before = copy.deepcopy(record)
            result = inherited_record(record, parent, snapshot, 'D:/delivery')
            self.assertEqual(record, before)
            self.assertEqual(result['origin'], 'inherited-legacy')
            self.assertEqual(result['distance_backend'], 'legacy')
            self.assertEqual(result['origin_manifest_sha256'], 'parent-manifest')
            self.assertEqual(result['sha256'], 'sealed')
            self.assertEqual(result['metrics'], record['metrics'])
            self.assertEqual(Path(result['path']), snapshot / 'city' / 'generated.pkl')

    def test_inherited_cache_pair_never_launches_new_job(self):
        with tempfile.TemporaryDirectory() as folder:
            study = object.__new__(SuccessorStudy)
            study.snapshot = Path(folder)
            study.parent_manifest_sha = 'sealed-parent'
            for rank in (0, 1):
                atomic_json(study.snapshot / NY / 'time-cache' / f'seed-{SEEDS[0]}-rank-{rank}' / 'payload.pkl', {})
            with patch.object(study, 'verify_cache') as verify, patch('tools.run_two_city_study.subprocess.Popen') as spawn:
                paths, origin = study.cache_paths(NY, SEEDS[0])
                self.assertEqual(len(paths), 2)
                self.assertEqual(origin, 'sealed-parent')
                self.assertEqual(verify.call_count, 2)
                spawn.assert_not_called()

    def test_partial_old_cache_pair_fails_without_resampling(self):
        with tempfile.TemporaryDirectory() as folder:
            study = object.__new__(SuccessorStudy)
            study.snapshot = Path(folder)
            atomic_json(study.snapshot / NY / 'time-cache' / f'seed-{SEEDS[0]}-rank-0' / 'payload.pkl', {})
            with self.assertRaisesRegex(RuntimeError, 'Incomplete parent'):
                study.cache_paths(NY, SEEDS[0])

    def test_completed_results_are_skipped_in_successor_execution(self):
        study = object.__new__(SuccessorStudy)
        study.args = SimpleNamespace(sync_only=False, preflight_only=False)
        study.registry = {k: {} for k in expected_keys()}
        missing = key(IST, SEEDS[0], 'full')
        del study.registry[missing]
        study.cfg = {NY: 'ny-cfg', IST: 'ist-cfg'}
        with patch.object(study, 'precheck'), patch.object(study, 'preflight_city'), \
             patch.object(study, 'verify_city_pairing'), patch.object(study, 'status'), \
             patch.object(study, 'sampling') as sample, patch.object(study, 'postswap') as postswap, \
             patch.object(study, 'audit', side_effect=RuntimeError('stop-after-scheduling')):
            with self.assertRaisesRegex(RuntimeError, 'stop-after-scheduling'):
                study.execute()
            sample.assert_called_once_with(IST, SEEDS[0], 'full', None)
            postswap.assert_not_called()

    def test_register_refuses_replacing_inherited_result(self):
        study = object.__new__(SuccessorStudy)
        study.manifest_sha = 'new'
        study.parent_registry = {key(NY, SEEDS[0], 'full'): {}}
        with self.assertRaisesRegex(RuntimeError, 'forbidden'):
            study.register(NY, SEEDS[0], 'full', 'unused', {}, {}, 'new', 'new', [])

    def test_mixed_implementation_report_is_explicit(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            render(root, {}, dict(projection_revision='pcdg-distance-v2', total_unique_results=66,
                                 implementation_transition='distance-batched-successor-v1'))
            self.assertIn('不是全程单一后端', (root / 'report.md').read_text(encoding='utf-8'))
            self.assertIn('Mixed implementation', (root / 'methods-jsd.md').read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
