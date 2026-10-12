"""Orchestration tests for category-preserving geometry refinement.

The frozen geometry core and evaluation are already covered by test_geometry_projection and the
ablation suite; these tests exercise only the new glue in tools.run_category_geometry and
tools.fetch_category_geometry: source-batch slicing, A-source verification, frozen-config/reference
gating, merge/index/invariant logic, the quality gate, resume/conflict handling, and the
independent 6-result downloader (never the old 18-result study acceptance).
"""
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import tools.fetch_category_geometry as fcg
import tools.run_category_geometry as rcg
from experiment_io import atomic_json, publish_torch, sha256_file

CITY = 'NewYork_PO1_OOD'


def record(index):
    """A trivial single-event A record; geometry content is irrelevant to the glue under test."""
    return dict(checkins=np.array([index + 6], dtype=np.int64), marks=np.array([4], dtype=np.int64),
                gps=[[0., 0.]], arrival_times=np.array([1.], dtype=float),
                **{f'condition{i}': np.ones(1, dtype=np.int64) for i in range(1, 7)},
                **{f'condition{i}_indicator': np.ones(24, dtype=np.int64) for i in range(1, 7)})


def write_shard(cs, city, seed, rank, batches):
    """batches: list of (global_start, [index, ...]) in storage order."""
    sequences, traces, indices = [], [], []
    for global_start, batch in batches:
        sequences += [record(i) for i in batch]
        indices += list(batch)
        traces.append(dict(global_start=global_start, indices=list(batch),
                           projection_stats=[dict(optimizer_steps=500)]))
    payload = dict(job=dict(city=dict(dataset=city), seed=seed), indices=indices,
                   sequences=sequences, batch_traces=traces)
    path = rcg.a_shard_path(cs, city, seed, rank)
    path.parent.mkdir(parents=True, exist_ok=True)
    publish_torch(path, payload)


def write_refined(out, city, seed, rank, batches, manifest='m', source='s'):
    for global_start, batch in batches:
        rank_dir = out / city / f'seed-{seed}' / f'rank-{rank}'
        rank_dir.mkdir(parents=True, exist_ok=True)
        bpath = rank_dir / f'batch-{global_start}.pkl'
        ident = dict(city=city, seed=seed, rank=rank, global_start=global_start,
                     indices=list(batch), manifest_sha256=manifest, source_sha256=source)
        publish_torch(bpath, dict(identity=ident, sequences=[record(i) for i in batch],
                                  diagnostics=dict(replaced_pois=len(batch))))
        atomic_json(bpath.with_name(f'batch-{global_start}.receipt.json'),
                    dict(identity=ident, sha256=sha256_file(bpath)))

def frozen_config():
    return dict(geometry_refinement='same_category_v1', geometry_steps=200,
                geometry_distance_weight=1.0, geometry_radius_weight=2.0, geometry_prior_weight=0.1,
                geometry_paths=8, geometry_topk=32, geometry_temperature=1.0,
                geometry_learning_rate=0.05, geometry_noise_interval=20)


class SourceAndConfigTests(unittest.TestCase):
    def test_source_packets_slice_order_and_projection_flag(self):
        payload = dict(sequences=[record(i) for i in range(5)], batch_traces=[
            dict(global_start=0, indices=[0, 1, 2], projection_stats=[dict(optimizer_steps=500)]),
            dict(global_start=3, indices=[3, 4], projection_stats=[dict(optimizer_steps=0)])])
        packets = rcg.source_packets(payload)
        self.assertEqual([p['indices'] for p in packets], [[0, 1, 2], [3, 4]])
        self.assertEqual([p['global_start'] for p in packets], [0, 3])
        self.assertEqual([p['projection_executed'] for p in packets], [True, False])
        self.assertEqual([len(p['records']) for p in packets], [3, 2])

    def test_source_packets_rejects_incomplete_layout(self):
        payload = dict(sequences=[record(i) for i in range(6)], batch_traces=[
            dict(global_start=0, indices=[0, 1, 2], projection_stats=[dict(optimizer_steps=500)]),
            dict(global_start=3, indices=[3, 4], projection_stats=[dict(optimizer_steps=500)])])
        with self.assertRaisesRegex(RuntimeError, 'incomplete'):
            rcg.source_packets(payload)

    def test_make_config_accepts_frozen_and_rejects_drift(self):
        rec = dict(config=frozen_config(), configuration_sha256=rcg.EXPECTED_CONFIG_SHA)
        config = rcg.make_config(rec)
        self.assertEqual(config.geometry_steps, 200)
        self.assertEqual(config.geometry_radius_weight, 2.0)
        with self.assertRaisesRegex(RuntimeError, 'sha256'):
            rcg.make_config(dict(rec, configuration_sha256='0' * 64))
        with self.assertRaisesRegex(RuntimeError, 'Unexpected'):
            rcg.make_config(dict(config=dict(frozen_config(), geometry_steps=100),
                                 configuration_sha256=rcg.EXPECTED_CONFIG_SHA))
        with self.assertRaisesRegex(RuntimeError, 'Unexpected'):
            rcg.make_config(dict(config=dict(frozen_config(), geometry_refinement='off'),
                                 configuration_sha256=rcg.EXPECTED_CONFIG_SHA))

class VerifyAndResumeTests(unittest.TestCase):
    def test_verify_a_source_requires_complete_passed_and_matching_hashes(self):
        with tempfile.TemporaryDirectory() as folder:
            cs = Path(folder)
            (cs / 'summary.json').write_text('{"x":1}', encoding='utf-8')
            atomic_json(cs / 'status.json', dict(state='complete'))
            files = {'summary.json': sha256_file(cs / 'summary.json')}
            atomic_json(cs / 'audit.json', dict(state='passed', files=files))
            self.assertEqual(rcg.verify_a_source(cs)['state'], 'passed')
            atomic_json(cs / 'status.json', dict(state='running'))
            with self.assertRaisesRegex(RuntimeError, 'complete'):
                rcg.verify_a_source(cs)
            atomic_json(cs / 'status.json', dict(state='complete'))
            atomic_json(cs / 'audit.json', dict(state='failed', files=files))
            with self.assertRaisesRegex(RuntimeError, 'complete'):
                rcg.verify_a_source(cs)
            atomic_json(cs / 'audit.json', dict(state='passed', files=files))
            (cs / 'summary.json').write_text('{"x":2}', encoding='utf-8')
            with self.assertRaisesRegex(RuntimeError, 'changed'):
                rcg.verify_a_source(cs)

    def test_resume_skips_identical_batch_and_rejects_conflict(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            write_refined(out, CITY, rcg.SEEDS[0], 0, [(0, [0, 1])])
            rank_dir = out / CITY / f'seed-{rcg.SEEDS[0]}' / 'rank-0'
            bpath = rank_dir / 'batch-0.pkl'
            receipt = rank_dir / 'batch-0.receipt.json'
            ident = dict(city=CITY, seed=rcg.SEEDS[0], rank=0, global_start=0,
                         indices=[0, 1], manifest_sha256='m', source_sha256='s')
            self.assertTrue(rcg.resume_or_conflict(bpath, receipt, ident))
            with self.assertRaisesRegex(RuntimeError, 'Conflicting'):
                rcg.resume_or_conflict(bpath, receipt, dict(ident, source_sha256='other'))
            fresh = rank_dir / 'batch-99.pkl'
            self.assertFalse(rcg.resume_or_conflict(fresh, rank_dir / 'batch-99.receipt.json', ident))

def stub_metrics(**over):
    base = {k: 0.5 for k in rcg.INVARIANT_KEYS}
    base.update(Distance=0.1, Radius=0.1, DailyLoc=0.1, **{'G-RANK': 0.1})
    base.update(over)
    return base


class MergeTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.cs = Path(self.folder.name) / 'A'
        self.out = Path(self.folder.name) / 'out'
        self.seed = rcg.SEEDS[0]
        self.raw = dict(sequences=[record(i) for i in range(6)], poi_category={i + 6: 4 for i in range(6)})
        self.enter = patch.dict(rcg.SAMPLES, {CITY: 6}, clear=False)
        self.enter.start(); self.addCleanup(self.enter.stop)
        self.inv = patch.object(rcg, 'assert_invariants', lambda *a, **k: None)
        self.inv.start(); self.addCleanup(self.inv.stop)

    def _shards(self, rank0, rank1, rrank0=None, rrank1=None):
        write_shard(self.cs, CITY, self.seed, 0, rank0)
        write_shard(self.cs, CITY, self.seed, 1, rank1)
        write_refined(self.out, CITY, self.seed, 0, rrank0 or rank0)
        write_refined(self.out, CITY, self.seed, 1, rrank1 or rank1)

    def test_merge_recovers_full_ordered_test_set_out_of_order(self):
        self._shards([(0, [0, 2]), (2, [4])], [(0, [1, 3]), (2, [5])])
        with patch.object(rcg, 'evaluate', lambda *a, **k: (stub_metrics(), [])):
            r = rcg.merge_city_seed(self.out, self.cs, CITY, self.seed, self.raw, object())
        self.assertEqual(r['samples'], 6)
        self.assertEqual(r['test_indices'], list(range(6)))
        self.assertEqual(r['replaced_pois'], 6)
        self.assertEqual(len(r['sequences']), 6)

    def test_merge_rejects_missing_or_duplicate_indices(self):
        with patch.object(rcg, 'evaluate', lambda *a, **k: (stub_metrics(), [])):
            self._shards([(0, [0, 2]), (2, [4])], [(0, [1, 3])])
            with self.assertRaisesRegex(RuntimeError, 'full ordered test set'):
                rcg.merge_city_seed(self.out, self.cs, CITY, self.seed, self.raw, object())

    def test_merge_rejects_refined_batch_whose_indices_differ_from_source(self):
        with patch.object(rcg, 'evaluate', lambda *a, **k: (stub_metrics(), [])):
            self._shards([(0, [0, 2]), (2, [4])], [(0, [1, 3]), (2, [5])],
                         rrank0=[(0, [0, 9]), (2, [4])])
            with self.assertRaisesRegex(RuntimeError, 'do not match the source batch'):
                rcg.merge_city_seed(self.out, self.cs, CITY, self.seed, self.raw, object())

    def test_merge_rejects_changed_invariant_metric(self):
        calls = {'n': 0}

        def ev(*a, **k):
            calls['n'] += 1
            return (stub_metrics(Category=0.9) if calls['n'] == 2 else stub_metrics(), [])

        self._shards([(0, [0, 2]), (2, [4])], [(0, [1, 3]), (2, [5])])
        with patch.object(rcg, 'evaluate', ev):
            with self.assertRaisesRegex(RuntimeError, 'invariant metric Category'):
                rcg.merge_city_seed(self.out, self.cs, CITY, self.seed, self.raw, object())


class QualityAndAggregateTests(unittest.TestCase):
    def _agg(self, distance, radius, dailyloc, grank):
        return {'Distance': dict(mean=distance), 'Radius': dict(mean=radius),
                'DailyLoc': dict(mean=dailyloc), 'G-RANK': dict(mean=grank)}

    def test_quality_gate_passes_only_when_all_four_hold(self):
        a = self._agg(0.2, 0.2, 0.1, 0.1)
        ok, checks = rcg.quality_gate(a, self._agg(0.15, 0.18, 0.104, 0.105))
        self.assertTrue(ok)
        self.assertTrue(all(checks.values()))
        self.assertFalse(rcg.quality_gate(a, self._agg(0.25, 0.18, 0.1, 0.1))[0])
        self.assertFalse(rcg.quality_gate(a, self._agg(0.15, 0.18, 0.11, 0.1))[0])
        self.assertFalse(rcg.quality_gate(a, self._agg(0.15, 0.18, 0.1, 0.106))[0])
        ok, info = rcg.quality_gate(a, self._agg(None, 0.18, 0.1, 0.1))
        self.assertFalse(ok)
        self.assertIn('reason', info)

    def test_aggregate_and_mean_sd(self):
        agg = rcg.aggregate([dict(Distance=0.1, R=1.0), dict(Distance=0.2, R=3.0), dict(Distance=0.3, R=5.0)])
        self.assertAlmostEqual(agg['Distance']['mean'], 0.2)
        self.assertEqual(agg['Distance']['n'], 3)
        self.assertAlmostEqual(agg['R']['sample_sd'], 2.0)
        self.assertIsNone(rcg.mean_sd([0.1, None, 0.3])['mean'])

class DownloaderTests(unittest.TestCase):
    def _good(self, out):
        (out / 'report.md').write_text('r', encoding='utf-8')
        atomic_json(out / 'summary.json', {CITY: {}, 'Istanbul_PO1_OOD': {}})
        files = {'report.md': sha256_file(out / 'report.md'),
                 'summary.json': sha256_file(out / 'summary.json')}
        atomic_json(out / 'audit.json', dict(state='passed', results=6, quality_goals_met=True, files=files))

    def test_verify_accepts_six_result_pass(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            self._good(out)
            r = fcg.verify(out)
            self.assertEqual(r['results'], 6)
            self.assertEqual(r['cities'], sorted([CITY, 'Istanbul_PO1_OOD']))

    def test_verify_rejects_non_six_results_and_bad_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            self._good(out)
            audit = json.loads((out / 'audit.json').read_text(encoding='utf-8'))
            atomic_json(out / 'audit.json', dict(audit, results=18))
            with self.assertRaisesRegex(RuntimeError, '6-result'):
                fcg.verify(out)
            atomic_json(out / 'audit.json', audit)
            (out / 'report.md').write_text('tampered', encoding='utf-8')
            with self.assertRaisesRegex(RuntimeError, 'hash mismatch'):
                fcg.verify(out)

    def _archive(self, folder, members):
        archive = Path(folder) / f'd{len(list(Path(folder).glob("*.tar.gz")))}.tar.gz'
        with tarfile.open(archive, 'w:gz') as tar:
            for name, data, isfile in members:
                info = tarfile.TarInfo(name)
                if isfile:
                    info.size = len(data)
                    tar.addfile(info, io.BytesIO(data))
                else:
                    info.type = tarfile.DIRTYPE
                    tar.addfile(info)
        return archive

    def test_safe_extract_writes_regular_files_and_blocks_traversal(self):
        with tempfile.TemporaryDirectory() as folder:
            archive = self._archive(folder, [('summary.json', b'{}', True)])
            out = Path(folder) / 'o1'
            out.mkdir()
            fcg.safe_extract(archive, out)
            self.assertEqual((out / 'summary.json').read_bytes(), b'{}')
            evil = self._archive(folder, [('../evil.json', b'x', True)])
            out2 = Path(folder) / 'o2'
            out2.mkdir()
            with self.assertRaises(RuntimeError):
                fcg.safe_extract(evil, out2)

    def test_safe_extract_rejects_duplicates_and_non_files(self):
        with tempfile.TemporaryDirectory() as folder:
            dup = self._archive(folder, [('a.json', b'1', True), ('a.json', b'2', True)])
            out = Path(folder) / 'o3'
            out.mkdir()
            with self.assertRaisesRegex(RuntimeError, 'Duplicate'):
                fcg.safe_extract(dup, out)
            directory = self._archive(folder, [('sub', b'', False)])
            out2 = Path(folder) / 'o4'
            out2.mkdir()
            with self.assertRaisesRegex(RuntimeError, 'regular files'):
                fcg.safe_extract(directory, out2)


if __name__ == '__main__':
    unittest.main()
