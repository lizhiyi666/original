from dataclasses import asdict
import copy
import json
from pathlib import Path
import statistics
import tempfile
import unittest
from unittest.mock import patch

from tests.test_geometry_study import metric
from tools.geometry_study_common import (CITIES, configurations, config_label, quality_failures,
    rank_candidates, validate_other_jsd_tolerance, check_invariant_metrics)
from tools.run_geometry_study import GeometryStudy


class GeometryToleranceTests(unittest.TestCase):
    def test_relative_boundaries_and_legacy_default(self):
        full=metric()
        for key in ('DailyLoc','G-RANK'):
            changed=dict(full,**{key:full[key]*1.0914})
            self.assertIn(key,quality_failures(changed,full))
            self.assertEqual(quality_failures(changed,full,other_jsd_relative_tolerance=.10),[])
            self.assertEqual(quality_failures(dict(full,**{key:full[key]*1.1}),full,
                other_jsd_relative_tolerance=.1),[])
            self.assertIn(key,quality_failures(dict(full,**{key:full[key]*1.10001}),full,
                other_jsd_relative_tolerance=.1))
        for x in (-.1,1.1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):validate_other_jsd_tolerance(x)

    def test_undefined_metrics_and_invariants_still_fail(self):
        full=metric()
        for x in (None,float('nan'),float('inf')):
            for actual,base in ((dict(full,DailyLoc=x),full),(full,dict(full,DailyLoc=x))):
                self.assertEqual(quality_failures(actual,base,other_jsd_relative_tolerance=.1),['undefined-metric'])
        self.assertIn('Radius',quality_failures(dict(full,Radius=.06),full,True,other_jsd_relative_tolerance=.1))
        with self.assertRaises(RuntimeError):check_invariant_metrics(dict(full,strict_ovr=.301),full)

    def test_selection_routes_same_tolerance_and_records_old_rule(self):
        full=metric();actual=dict(full,DailyLoc=full['DailyLoc']*1.0914)
        base={city:dict(full=[full]*3,no_projection=[full]*3) for city in CITIES}
        cfg=configurations(200)[2]
        candidates={config_label(cfg):dict(config=asdict(cfg),cities={city:[dict(metrics=actual,offline_seconds=1)]*3
                                                                    for city in CITIES})}
        study=object.__new__(GeometryStudy);study.other_jsd_relative_tolerance=.1
        for confirmation in (False,True):
            record=study.selection_record(candidates,base,confirmation)
            self.assertEqual(len(record['ranking']),1)
            self.assertEqual(record['ranking_at_5_percent'],[])
            self.assertEqual(record['other_jsd_relative_tolerance'],.1)

    def test_final_report_uses_ten_percent_but_keeps_original_verdict(self):
        study=object.__new__(GeometryStudy);study.other_jsd_relative_tolerance=.1
        full=metric();actual=dict(full,DailyLoc=full['DailyLoc']*1.0914)
        study.parent_registry={};registry={}
        for city in CITIES:
            for seed in (135398,135399,135400):
                for variant in ('full','no_projection'):
                    study.parent_registry[f'{city}/{seed}/{variant}']=dict(metrics=full)
                for variant in ('full','distance_only','radius_only'):
                    registry[f'{city}/{seed}/{variant}']=dict(metrics=actual)
        with tempfile.TemporaryDirectory() as folder:
            study.out=Path(folder)
            acceptance=study.final_report(registry)
        for city in CITIES:
            self.assertEqual(acceptance[city]['quality_failures'],[])
            self.assertEqual(acceptance[city]['failures_at_5_percent'],['DailyLoc'])

    def test_nonfinite_jointgen_target_cannot_enter_ranking(self):
        full=metric();cfg=configurations(200)[0]
        base={city:dict(full=[full],no_projection=[dict(full,Radius=float('nan'))]) for city in CITIES}
        candidates={config_label(cfg):dict(config=asdict(cfg),cities={city:[dict(metrics=full,offline_seconds=1)]
                                                                    for city in CITIES})}
        self.assertEqual(rank_candidates(candidates,base,other_jsd_relative_tolerance=.1),[])

    def test_inherited_qualification_and_quality_failure_do_not_replay_or_start_formal(self):
        study=object.__new__(GeometryStudy)
        study.args=type('Args',(),{'stage':'all'})();study.manifest_sha='new'
        study.manifest={'inheritance':{'source':'old'}}
        with tempfile.TemporaryDirectory() as folder:
            study.out=Path(folder)
            perf=study.out/'source-performance.json'
            perf.write_text(json.dumps(dict(selected_steps=200,speed={})),encoding='utf-8')
            study.inherited=type('Inherited',(),{'file':lambda self,name:perf})()
            with patch.object(study,'precheck'),patch.object(study,'prepare_pool'), \
                 patch.object(study,'speed_gate') as speed,patch.object(study,'profile_tail') as profile, \
                 patch.object(study,'finish_model'),patch.object(study,'verify_parent'), \
                 patch.object(study,'stop_report'),patch.object(study,'status'), \
                 patch.object(study,'calibrate',return_value=(None,'no-confirmed-quality-and-speed-candidate')), \
                 patch.object(study,'formal') as formal:
                study.execute()
            speed.assert_not_called();profile.assert_not_called();formal.assert_not_called()

    def test_all_config_specific_speed_failures_stop_selection(self):
        study=object.__new__(GeometryStudy);study.other_jsd_relative_tolerance=.1;study.manifest_sha='new'
        full=metric();base=dict(full=full,no_projection=full)
        with tempfile.TemporaryDirectory() as folder:
            study.out=Path(folder)
            with patch.object(study,'read_pool',return_value=([],base)), \
                 patch.object(study,'prepare_pool',return_value=([],base)),patch.object(study,'status'), \
                 patch.object(study,'refine_pool',side_effect=lambda city,pool,seed,config:
                     dict(state='complete',metrics=full,offline_seconds=1,config=asdict(config))), \
                 patch.object(study,'speed_gate',return_value=(None,{})) as speed:
                config,reason=study.calibrate(200)
            self.assertIsNone(config);self.assertEqual(speed.call_count,3)
            self.assertFalse((study.out/'recommendation.json').exists())

    def test_new_top_three_follow_ranking_not_old_finalist(self):
        # Small portable fixture with the actual observed ordering, no artifacts.
        full=metric();base={city:dict(full=[full],no_projection=[full]) for city in CITIES}
        candidates={}
        for config in configurations(200):
            r,b=config.geometry_radius_weight,config.geometry_prior_weight
            displacement={1.:.90,2.:.92,4.:.94,.5:.96}[r]
            actual=dict(full,Distance=full['Distance']*displacement,Radius=full['Radius']*displacement,
                        DailyLoc=full['DailyLoc']*(1.04 if r==.5 and b==.1 else 1.09 if b==.1 else 1.2))
            candidates[config_label(config)]=dict(config=asdict(config),cities={city:[dict(metrics=actual,offline_seconds=1.)]
                                                                                  for city in CITIES})
        self.assertEqual([r['label'] for r in rank_candidates(candidates,base)],['r0.5-b0.1'])
        ranked=rank_candidates(candidates,base,other_jsd_relative_tolerance=.1)
        self.assertEqual([r['label'] for r in ranked[:3]],['r1-b0.1','r2-b0.1','r4-b0.1'])

    def test_no_fourth_place_fallback_when_all_confirmations_fail(self):
        study=object.__new__(GeometryStudy);study.other_jsd_relative_tolerance=.1
        study.manifest_sha='new';full=metric()
        base=dict(full=full,no_projection=full)
        confirmed_labels=[]
        def refine(city,pool,seed,config):
            if pool=='confirm':confirmed_labels.append(config_label(config))
            m=dict(full,DailyLoc=.06) if pool=='confirm' else dict(full)
            return dict(state='complete',metrics=m,offline_seconds=1,config=asdict(config))
        with tempfile.TemporaryDirectory() as folder:
            study.out=Path(folder)
            with patch.object(study,'read_pool',return_value=([],base)), \
                 patch.object(study,'prepare_pool',return_value=([],base)), \
                 patch.object(study,'refine_pool',side_effect=refine),patch.object(study,'status'), \
                 patch.object(study,'speed_gate') as speed:
                config,reason=study.calibrate(200)
            self.assertIsNone(config);self.assertEqual(reason,'no-confirmed-quality-and-speed-candidate')
            self.assertEqual(len(set(confirmed_labels)),3)
            self.assertEqual(len(confirmed_labels),18)
            speed.assert_not_called()


if __name__=='__main__':unittest.main()
