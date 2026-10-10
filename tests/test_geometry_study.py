from dataclasses import asdict
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from geometry_projection import GeometryConfig
from tools.geometry_study_common import (CITIES, STEP_CHOICES, choose_steps, configurations, config_label,
    split_indices, quality_failures, rank_candidates, check_invariant_metrics, geometric_diagnostics)
from tools.run_geometry_study import GeometryStudy


def metric():
    return dict(strict_ovr=.3,pair_coverage=.7,category_coverage=.85,DailyLoc=.04,
                **{'G-RANK':.18},Distance=.15,Radius=.05)


class GeometryStudyTests(unittest.TestCase):
    def test_worker_resume_requires_geometry_version_configuration_and_reference(self):
        from tools.ablation_worker import validate_geometry_result
        config=GeometryConfig('same_category_v1',geometry_steps=100)
        job=dict(geometry_config=asdict(config),geometry_reference_sha256='train-only')
        result=dict(config.metadata(),geometry_reference_sha256='train-only')
        validate_geometry_result(job,result)
        for mutation in ({'geometry_implementation_version':'old'}, {'geometry_steps':50},
                         {'geometry_reference_sha256':'test'}, {'geometry_refinement':'off'}):
            with self.assertRaises(RuntimeError):validate_geometry_result(job,dict(result,**mutation))
        with self.assertRaises(RuntimeError):validate_geometry_result({},result)
        validate_geometry_result({}, {})

    def test_split_is_reproducible_disjoint_and_leaves_reference(self):
        for count in (3160,7035):
            split=split_indices(count)
            self.assertEqual(split,split_indices(count))
            self.assertEqual(len(split['screen']),512)
            self.assertEqual(len(split['confirm']),512)
            all_indices=sum(split.values(),[])
            self.assertEqual(len(all_indices),count)
            self.assertEqual(len(set(all_indices)),count)

    def test_exactly_twelve_weight_configs_with_shared_fixed_steps(self):
        configs=configurations(100)
        self.assertEqual(len(configs),12)
        self.assertEqual(len({config_label(c) for c in configs}),12)
        self.assertTrue(all(c.geometry_steps==100 and c.geometry_paths==8 and c.geometry_topk==32 for c in configs))

    def test_speed_gate_requires_every_case_and_fifty_steps(self):
        cases={city+'/'+case:dict(state='complete',ratio=1.09) for city in CITIES for case in ('ordinary','long')}
        rows={str(s):dict(cases) for s in STEP_CHOICES}
        self.assertEqual(choose_steps(rows),200)
        rows['200'][CITIES[1]+'/long']=dict(state='complete',ratio=1.100001)
        self.assertEqual(choose_steps(rows),100)
        rows['50'][CITIES[0]+'/ordinary']=dict(state='complete',ratio=1.2)
        self.assertIsNone(choose_steps(rows))
        self.assertIsNone(choose_steps({'50':{}}))

    def test_percentage_points_and_relative_jsd_gates(self):
        full=metric()
        within=dict(full,strict_ovr=.31,pair_coverage=.69,category_coverage=.84,DailyLoc=.042,**{'G-RANK':.189})
        self.assertEqual(quality_failures(within,full),[])
        for field,value in [('strict_ovr',.311),('pair_coverage',.689),('DailyLoc',.043),('G-RANK',.190)]:
            self.assertTrue(quality_failures(dict(full,**{field:value}),full))
        self.assertIn('Radius',quality_failures(dict(full,Radius=.051),full,True))

    def test_rank_uses_both_cities_and_frozen_order(self):
        base={city:dict(full=[metric()],no_projection=[dict(metric(),Distance=.12,Radius=.025)]) for city in CITIES}
        candidates={}
        for i,config in enumerate(configurations(50)[:2]):
            m=dict(metric(),Distance=.11+i*.01,Radius=.02+i*.005)
            candidates[config_label(config)]=dict(config=asdict(config),cities={c:[dict(metrics=m,offline_seconds=1.)] for c in CITIES})
        ranking=rank_candidates(candidates,base,confirmation=True)
        self.assertEqual(ranking[0]['label'],config_label(configurations(50)[0]))
        candidates[ranking[0]['label']]['cities'][CITIES[1]][0]['metrics']['G-RANK']=.4
        self.assertEqual(len(rank_candidates(candidates,base,confirmation=True)),1)

    def test_invariant_metrics_fail_even_inside_quality_allowance(self):
        full=metric()
        with self.assertRaisesRegex(RuntimeError,'invariant'):
            check_invariant_metrics(dict(full,strict_ovr=.301),full)

    def test_payload_resume_checks_identity_and_bytes(self):
        study=object.__new__(GeometryStudy)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'cache.pkl'; identity=dict(study='test',seed=1)
            study.save_payload(path,identity,value=[1,2])
            self.assertEqual(study.load_payload(path,identity)['value'],[1,2])
            with self.assertRaises(RuntimeError):study.load_payload(path,dict(identity,seed=2))
            path.write_bytes(b'changed')
            with self.assertRaises(RuntimeError):study.load_payload(path,identity)

    def test_speed_failure_does_not_start_quality_or_formal_work(self):
        study=object.__new__(GeometryStudy)
        study.args=type('Args',(),{'stage':'all'})()
        study.manifest_sha='test'
        with tempfile.TemporaryDirectory() as folder:
            study.out=Path(folder)
            with patch.object(study,'precheck'),patch.object(study,'prepare_pool'), \
                 patch.object(study,'speed_gate',return_value=(None,{})),patch.object(study,'profile_tail'), \
                 patch.object(study,'finish_model'),patch.object(study,'verify_parent'), \
                 patch.object(study,'calibrate') as calibrate,patch.object(study,'formal') as formal:
                study.execute()
                calibrate.assert_not_called();formal.assert_not_called()
                self.assertTrue((study.out/'report.md').exists())


if __name__=='__main__':unittest.main()
