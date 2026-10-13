import copy
from dataclasses import asdict
import hashlib
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from evaluations import statistical_metrics as v2
from evaluations.grank_v3 import grank_jsd, ranked_counts, evaluate
from experiment_io import atomic_json, publish_torch, sha256_file
from tools.ablation_common import evaluate as evaluate_v2
from tools.distribution_study_common import CITIES, SEEDS, configurations, label, save_payload, select
from tools.reevaluate_distribution_v3 import (
    Source, run, verify, compare, compare_records, disjoint, records_for, summary_for, code_fingerprint, read)


def fixture(root):
    from tests.test_geometry_projection import fixture as geometry_fixture
    raw, records, _, _ = geometry_fixture()
    for r in records:
        r['marks'] = np.array([raw['poi_category'][int(p)] for p in r['checkins']], dtype=np.int64)
    indices = list(range(len(records)))
    metrics = evaluate_v2(records, records, raw['poi_category'], indices)[0]
    profiles = {}
    for city in CITIES:
        path = root / 'evaluation_inputs' / city / f'{city}_train.pkl'
        publish_torch(path, raw)
        profiles[city] = dict(data_sha256={'train': sha256_file(path)}, test_count=len(records))
    manifest = dict(source_commit='0' * 40, source_sha256={}, profiles=profiles,
                    splits={c: dict(screen=indices, confirm=indices) for c in CITIES})
    atomic_json(root / 'manifest.json', manifest)
    sha = sha256_file(root / 'manifest.json')
    cfg = configurations()[0]
    name = label(cfg)
    summary = dict(terminal_state='stopped', reason='test-fixture', quality_goals_met=False)
    for phase in ('screen', 'confirm'):
        results = {name: dict(config=asdict(cfg), cities={c: [] for c in CITIES})}
        baselines = {c: dict(rows=[]) for c in CITIES}
        for city in CITIES:
            for seed in (SEEDS[:1] if phase == 'screen' else SEEDS):
                identity = dict(study_manifest_sha256=sha, phase=phase, city=city, seed=seed, indices=indices)
                base = dict(seed=seed, full=copy.deepcopy(metrics), no_projection=copy.deepcopy(metrics))
                save_payload(root / f'baselines/{phase}/{city}/seed-{seed}.pkl', dict(identity, kind='baseline'),
                             result=base, sequences_by_method={'full': records, 'no_projection': records},
                             city=city, split='train', indices=indices)
                baselines[city]['rows'].append(base)
                row = dict(seed=seed, metrics=copy.deepcopy(metrics), config=asdict(cfg), samples=len(indices))
                save_payload(root / f'results/{phase}/{city}/seed-{seed}/{name}.pkl', dict(identity, kind='result'),
                             result=row, sequences=records, city=city, split='train', indices=indices)
                results[name]['cities'][city].append(row)
        summary[phase] = dict(results=results, baselines=baselines,
                              selection=select(results, baselines, confirmation=phase == 'confirm'))
    atomic_json(root / 'summary.json', summary)
    atomic_json(root / 'registry.json', {})
    atomic_json(root / 'status.json', dict(state='stopped', pid=1))
    files = {p.relative_to(root).as_posix(): sha256_file(p) for p in root.rglob('*')
             if p.is_file() and p.name != 'status.json'}
    atomic_json(root / 'audit.json', dict(state='passed', terminal_state='stopped', results=0,
                                        quality_goals_met=False, files=files))
    return raw, records


class GrankV3Tests(unittest.TestCase):
    def test_tie_boundary_exact_poi_id_order(self):
        selected, counts = ranked_counts(list(reversed(range(1500))))
        self.assertEqual(selected, [(i, 1) for i in range(1000)])
        self.assertEqual(len(counts), 1500)
        detail = {}
        self.assertEqual(grank_jsd(range(1500), reversed(range(1500)), diagnostics=detail), 0)
        self.assertEqual(detail['generated']['boundary_tied'], 1500)
        self.assertEqual(detail['generated']['boundary_selected'], 1000)

    def test_counts_before_id_and_large_integer_ids(self):
        big = 2**60
        selected, _ = ranked_counts([big + 2, big, big + 1, big + 2], 2)
        self.assertEqual(selected, [(big + 2, 2), (big, 1)])

    def test_identity_alignment_not_sorted_popularity(self):
        self.assertAlmostEqual(grank_jsd([1, 1, 2], [3, 3, 4]), math.log(2), places=11)
        self.assertEqual(grank_jsd([1, 2, 1], [2, 1, 1]), 0)

    def test_order_and_argsort_independence(self):
        values = [v for v in range(1200) for _ in range(v % 4 + 1)]
        with patch('numpy.argsort', side_effect=AssertionError('No platform-dependent argsort')):
            self.assertEqual(grank_jsd(values, values[::-1]), 0)

    def test_no_boundary_tie_matches_legacy(self):
        left, right = [1] * 5 + [2] * 3 + [3], [2] * 5 + [1] * 3 + [3]
        self.assertAlmostEqual(grank_jsd(left, right, 2), v2.grank_jsd(left, right, 2), places=14)

    def test_empty_and_invalid_arguments(self):
        self.assertTrue(math.isnan(grank_jsd([], [])))
        self.assertTrue(math.isnan(grank_jsd([1], [])))
        for top in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                ranked_counts([1], top)
        for values in ([True], [1.0], [float('nan')], ['1']):
            with self.assertRaises(ValueError):
                ranked_counts(values)

    def test_full_evaluation_preserves_other_metrics_rng_and_v2(self):
        from tests.test_geometry_projection import fixture as geometry_fixture
        raw, records, _, _ = geometry_fixture()
        original = copy.deepcopy(records)
        indices = list(range(len(records)))
        before = torch.get_rng_state().clone()
        old, old_rows = evaluate_v2(records, records, raw['poi_category'], indices)
        new, new_rows = evaluate(records, records, raw['poi_category'], indices)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(v2.EVALUATION_VERSION, 2)
        self.assertEqual(old['evaluation_version'], 2)
        self.assertEqual(new['evaluation_version'], 3)
        self.assertEqual(old_rows, new_rows)
        for k in old:
            if k not in ('G-RANK', 'totalJSD', 'evaluation_version'):
                self.assertEqual(old[k], new[k])
        for a, b in zip(original, records):
            for key in a:
                np.testing.assert_array_equal(a[key], b[key])


class ReevaluateV3Tests(unittest.TestCase):
    def test_end_to_end_preserves_source_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder) / 'source', Path(folder) / 'v3'
            fixture(source)
            before = Source(source).snapshot
            with patch('tools.reevaluate_distribution_v3.evaluate', wraps=evaluate) as called:
                receipt = run(source, output, '1' * 40)
                self.assertEqual(called.call_count, 24)
            self.assertEqual(receipt['records'], 24)
            self.assertEqual(Source(source).snapshot, before)
            self.assertEqual(verify(output, source, True)['metrics_recomputed'], 24)
            self.assertEqual(read(output / 'audit.json')['new_formal_results'], 0)
            self.assertFalse((source / 'local-delivery-audit.json').exists())
            self.assertEqual(read(source / 'summary.json')['screen']['baselines'][CITIES[0]]['rows'][0]['full']['evaluation_version'], 2)
            self.assertEqual(read(output / 'summary.json')['screen']['baselines'][CITIES[0]]['rows'][0]['full']['evaluation_version'], 3)
            with self.assertRaisesRegex(RuntimeError, 'exists'):
                run(source, output, '1' * 40)
            with self.assertRaisesRegex(RuntimeError, 'Windows and Linux'):
                compare(output, output)

    def test_rejects_nested_output_and_bad_commit(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            for source, output in [(p, p), (p, p / 'out'), (p / 'in', p)]:
                with self.assertRaisesRegex(RuntimeError, 'disjoint'):
                    disjoint(source, output)
            with self.assertRaises(ValueError):
                run(p / 'source', p / 'out', 'not-a-commit')

    def test_source_tampering_fails_and_no_output_is_created(self):
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder) / 'source', Path(folder) / 'v3'
            fixture(source)
            (source / 'summary.json').write_text('{}')
            with self.assertRaisesRegex(RuntimeError, 'hash mismatch'):
                run(source, output, '1' * 40)
            self.assertFalse(output.exists())

    def test_identity_indices_and_missing_inventory_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'source'
            fixture(root)
            source = Source(root)
            item = source.items[0]
            changed = dict(item, seed=1)
            with self.assertRaisesRegex(RuntimeError, 'identity'):
                source.packet(changed)
            payload = torch.load(root / item['path'], weights_only=False)
            payload['indices'] = [0] * len(payload['indices'])
            with patch('tools.reevaluate_distribution_v3.torch.load', return_value=payload):
                with self.assertRaisesRegex(RuntimeError, 'indices'):
                    source.packet(item)
            source.audit['files']['results/extra.pkl'] = 'bad'
            with self.assertRaisesRegex(RuntimeError, 'inventory'):
                source.inventory()

    def test_changed_legacy_non_grank_metric_stops(self):
        with tempfile.TemporaryDirectory() as folder:
            source_path = Path(folder) / 'source'
            fixture(source_path)
            source = Source(source_path)
            real_evaluate = evaluate
            def bad(*a, **kw):
                values, rows = real_evaluate(*a, **kw)
                values['Category'] += 1e-6
                return values, rows
            with patch('tools.reevaluate_distribution_v3.evaluate', side_effect=bad):
                with self.assertRaisesRegex(RuntimeError, 'Category'):
                    records_for(source)

    def test_comparison_detects_metric_and_membership_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            source_path = Path(folder) / 'source'
            fixture(source_path)
            rows = records_for(Source(source_path))
            self.assertEqual(compare_records(rows, copy.deepcopy(rows)), 0)
            other = copy.deepcopy(rows)
            other[0]['metrics']['G-RANK'] += 1e-8
            with self.assertRaisesRegex(RuntimeError, 'G-RANK'):
                compare_records(rows, other)
            other = copy.deepcopy(rows)
            other[0]['grank_diagnostics']['generated']['ranked_counts_sha256'] = 'wrong'
            with self.assertRaisesRegex(RuntimeError, 'selection'):
                compare_records(rows, other)

    def test_fingerprint_permits_only_line_endings_for_legacy_code(self):
        with tempfile.TemporaryDirectory() as folder:
            source_path = Path(folder) / 'source'
            fixture(source_path)
            source = Source(source_path)
            name = 'evaluations/statistical_metrics.py'
            data = Path(name).read_bytes().replace(b'\r\n', b'\n')
            source.manifest['source_sha256'] = {name: hashlib.sha256(data).hexdigest()}
            code_fingerprint(source)
            source.manifest['source_sha256'][name] = '0' * 64
            with self.assertRaisesRegex(RuntimeError, 'beyond line endings'):
                code_fingerprint(source)


if __name__ == '__main__':
    unittest.main()
