import copy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from distribution_balance import BalanceConfig
from experiment_io import atomic_json, publish_torch, sha256_file
from tools.ablation_common import evaluate
from tools.distribution_study_common import CITIES, SEEDS, METRICS, HIGHER, save_payload
from tools.run_best_distribution_formal import (
    BEST_LABEL, EXPECTED_SAMPLES, rank_confirmed, check_formal_counts,
    execute, recompute_v3, summarize, v3_summary)
from tools.fetch_best_distribution_formal import PACK


def metrics(value=.1):
    return dict({k: (.8 if k in HIGHER else value) for k in METRICS}, evaluation_version=2)


def formal_fixture():
    config = asdict(BalanceConfig('train_ref_v1', .5, 0., 1.))
    return dict(results={BEST_LABEL: dict(config=config, cities={city: [dict(seed=seed,
        config=config, samples=EXPECTED_SAMPLES[city], metrics=metrics(.09)) for seed in SEEDS] for city in CITIES})},
        baselines={city: dict(rows=[dict(seed=seed, full=metrics(), no_projection=metrics()) for seed in SEEDS]) for city in CITIES})


class BestFormalTests(unittest.TestCase):
    def test_best_joint_rank_can_be_reported_despite_distribution_gate_failure(self):
        formal = formal_fixture()
        for city in CITIES:
            for row in formal['results'][BEST_LABEL]['cities'][city]:
                row['metrics']['Category'] = .15
        other = copy.deepcopy(formal['results'][BEST_LABEL])
        other['config']['poi_distribution_weight'] = .1
        for city in CITIES:
            for row in other['cities'][city]:
                row['metrics']['G-RANK'] = .14
        formal['results']['r0.5-c0-p0.1'] = other
        ranking = rank_confirmed({'confirm': formal})
        self.assertEqual(ranking[0]['label'], BEST_LABEL)
        self.assertGreater(ranking[0]['rank'][0], 1)

    def test_missing_seed_undefined_and_worse_constraints_fail_closed(self):
        formal = formal_fixture()
        formal['results'][BEST_LABEL]['cities'][CITIES[0]].pop()
        with self.assertRaisesRegex(RuntimeError, 'three fixed seeds'):
            rank_confirmed({'confirm': formal})
        formal = formal_fixture()
        formal['results'][BEST_LABEL]['cities'][CITIES[0]][0]['metrics']['Category'] = None
        with self.assertRaisesRegex(RuntimeError, 'Undefined'):
            rank_confirmed({'confirm': formal})
        formal = formal_fixture()
        for city in CITIES:
            for row in formal['results'][BEST_LABEL]['cities'][city]:
                row['metrics']['strict_ovr'] = .11
        with self.assertRaisesRegex(RuntimeError, 'constraint gates'):
            rank_confirmed({'confirm': formal})

    def test_exact_six_results_and_21066_records(self):
        formal = formal_fixture()
        check_formal_counts(formal)
        formal['results'][BEST_LABEL]['cities'][CITIES[0]][0]['samples'] -= 1
        with self.assertRaisesRegex(RuntimeError, 'samples/config'):
            check_formal_counts(formal)
        formal = formal_fixture()
        formal['results'][BEST_LABEL]['cities'][CITIES[0]][0]['seed'] = 1
        with self.assertRaisesRegex(RuntimeError, 'seed inventory'):
            check_formal_counts(formal)

    def test_no_formal_without_explicit_user_override(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            with self.assertRaisesRegex(RuntimeError, 'override'):
                execute(SimpleNamespace(allow_confirmation_gate_failure=False), out, None, None)
            self.assertEqual(list(out.iterdir()), [])

    def test_config_is_frozen_before_only_formal_phase(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            cfg = asdict(BalanceConfig('train_ref_v1', .5, 0., 1.))
            evidence = SimpleNamespace(root=out/'evidence', files={'summary.json':'sha'},
                ranking=[dict(label=BEST_LABEL, config=cfg, rank=[1.5, .9, .5, 0., 1.])])
            source = SimpleNamespace(profiles={c: {'test_count': EXPECTED_SAMPLES[c]} for c in CITIES},
                inputs={}, geometry=out/'geo', category=out/'cat', geo={'splits':{}},
                refs={c: SimpleNamespace(fingerprint='ref') for c in CITIES})
            args = SimpleNamespace(allow_confirmation_gate_failure=True, run_id='toy', source_commit='a'*40, gpus='0,1')
            def phase(args, where, name, configs):
                self.assertEqual(name, 'formal')
                self.assertEqual([asdict(c) for c in configs], [cfg])
                self.assertTrue((where/'selected-candidate.json').is_file())
                self.assertTrue((where/'manifest.json').is_file())
                raise RuntimeError('test stops before sampling')
            with patch('tools.run_best_distribution_formal.copy_input'), patch('tools.run_best_distribution_formal.fingerprint',return_value={}),\
                    patch('tools.run_best_distribution_formal.phase_run',side_effect=phase) as called:
                with self.assertRaisesRegex(RuntimeError, 'test stops'):
                    execute(args, out, source, evidence)
                self.assertEqual(called.call_count, 1)

    def test_v3_cpu_audit_does_not_relabel_v2_table_metrics(self):
        from tests.test_geometry_projection import fixture
        raw, records, _, _ = fixture()
        count = len(records)
        indices = list(range(count))
        metric = evaluate(records, records, raw['poi_category'], indices)[0]
        cfg = asdict(BalanceConfig('train_ref_v1', .5, 0., 1.))
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            profiles = {}
            for city in CITIES:
                p = out/'evaluation_inputs'/city/f'{city}_test.pkl'
                publish_torch(p, raw)
                profiles[city] = {'data_sha256': {'test': sha256_file(p)}}
            atomic_json(out/'manifest.json', {'profiles': profiles})
            manifest_sha = sha256_file(out/'manifest.json')
            formal = dict(results={BEST_LABEL: dict(config=cfg,cities={c: [] for c in CITIES})},
                          baselines={c: dict(rows=[]) for c in CITIES})
            for city in CITIES:
                for seed in SEEDS:
                    ident = dict(study_manifest_sha256=manifest_sha, city=city, phase='formal', seed=seed, indices=indices)
                    base = dict(seed=seed, full=metric, no_projection=metric)
                    row = dict(seed=seed, config=cfg, samples=count, metrics=metric)
                    save_payload(out/f'baselines/formal/{city}/seed-{seed}.pkl', dict(ident,kind='baseline'),
                        city=city, split='test', indices=indices, result=base, sequences_by_method={'full':records,'no_projection':records})
                    save_payload(out/f'results/formal/{city}/seed-{seed}/{BEST_LABEL}.pkl', dict(ident,kind='result'),
                        city=city, split='test', indices=indices, result=row, sequences=records)
                    formal['baselines'][city]['rows'].append(base)
                    formal['results'][BEST_LABEL]['cities'][city].append(row)
            before = copy.deepcopy(formal)
            with patch('tools.run_best_distribution_formal.EXPECTED_SAMPLES', {c:count for c in CITIES}):
                dual = recompute_v3(out)
            result = v3_summary(formal, dual)
            self.assertEqual(len(dual['records']), 18)
            self.assertEqual(formal, before)
            for row in dual['records']:
                self.assertEqual(row['metrics']['evaluation_version'], 3)
                self.assertEqual(row['legacy_metrics']['evaluation_version'], 2)
            self.assertEqual(result['results'][BEST_LABEL]['aggregate'][CITIES[0]]['evaluation_version']['mean'],3)

    def test_remote_pack_rejects_live_workers_and_wrong_record_count(self):
        compile(PACK.replace('ROOT_VALUE',repr('/tmp/test')), '<pack>', 'exec')
        self.assertIn("audit['trajectory_records']==21066", PACK)
        self.assertIn('formal worker still running', PACK)


if __name__ == '__main__':
    unittest.main()
