from dataclasses import asdict
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import torch

from distribution_balance import BalanceConfig
from experiment_io import atomic_json, sha256_file
from tools.distribution_study_common import (CITIES, SEEDS, METRICS, HIGHER,
    configurations, label, failures, select, save_payload, saved_payload, aligned_indices)
from tools.run_distribution_study import verify, execute, Worker


def metric(value=.1):
    return {k:(.8 if k in HIGHER else value) for k in METRICS}


def fixture(confirmation=False):
    seeds=SEEDS if confirmation else SEEDS[:1]
    cfg=configurations()[0]
    results={label(cfg):dict(config=asdict(cfg),cities={c:[dict(seed=s,metrics=metric(.09)) for s in seeds] for c in CITIES})}
    baselines={c:dict(rows=[dict(seed=s,full=metric(),no_projection=metric()) for s in seeds]) for c in CITIES}
    return results,baselines


class DistributionStudyTests(unittest.TestCase):
    def test_fixed_grid_and_reusable_category_groups(self):
        configs=configurations()
        self.assertEqual(len(configs),18)
        self.assertEqual(len(set(label(c,spatial=True) for c in configs)),6)
        self.assertEqual(len(set(label(c) for c in configs)),18)

    def test_strict_constraints_no_old_one_percentage_point_allowance(self):
        base=metric();bad=metric();bad['strict_ovr']+=.0001
        self.assertIn('strict_ovr',failures(bad,base,base,distributions=False))
        bad=metric();bad['pair_coverage']-=.0001
        self.assertIn('pair_coverage',failures(bad,base,base,distributions=False))
        bad=metric();bad['strict_ovr']+=5e-13
        self.assertEqual(failures(bad,base,base,distributions=True),[])

    def test_all_six_distribution_gates(self):
        for name in ('Distance','Radius','CategoryTransition','DailyLoc','Category','G-RANK'):
            bad=metric();bad[name]=.10001
            self.assertIn(name,failures(bad,metric(),metric(),distributions=True))

    def test_missing_and_nonfinite_fail(self):
        for value in (None,float('nan'),float('inf')):
            bad=metric();bad['Unsat']=value
            self.assertTrue(failures(bad,metric(),metric(),distributions=True))
        bad=metric();del bad['Category']
        self.assertTrue(failures(bad,metric(),metric(),distributions=True))

    def test_shared_configuration_and_all_seeds(self):
        results,base=fixture(True)
        self.assertEqual(len(select(results,base,confirmation=True)['ranking']),1)
        name=next(iter(results));results[name]['cities'][CITIES[1]][0]['metrics']['G-RANK']=.5
        self.assertEqual(select(results,base,confirmation=True)['ranking'],[])
        results,base=fixture(True);results[next(iter(results))]['cities'][CITIES[1]].pop()
        self.assertEqual(select(results,base,confirmation=True)['ranking'],[])

    def test_screen_does_not_require_distribution_gate_but_confirmation_does(self):
        results,base=fixture()
        for c in CITIES:results[next(iter(results))]['cities'][c][0]['metrics']['CategoryTransition']=.2
        self.assertEqual(len(select(results,base)['ranking']),1)

    def test_ties_use_parameter_tuple(self):
        results,base=fixture()
        original=next(iter(results.values()))
        for cfg in configurations():results[label(cfg)]=dict(copy.deepcopy(original),config=asdict(cfg))
        self.assertEqual([r['label'] for r in select(results,base)['ranking']],[label(c) for c in configurations()])

    def test_result_reuse_and_conflict_refusal(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'batch.pkl';ident={'source':'same','indices':[1,2]}
            self.assertIsNone(saved_payload(path,ident))
            save_payload(path,ident,result={'value':2})
            self.assertEqual(saved_payload(path,ident)['result']['value'],2)
            with self.assertRaises(RuntimeError):save_payload(path,ident,result={'value':2})
            with self.assertRaises(RuntimeError):saved_payload(path,dict(ident,source='changed'))
            path.with_suffix('.receipt.json').unlink()
            with self.assertRaises(RuntimeError):saved_payload(path,ident)

    def test_index_validation(self):
        aligned_indices([0,1,2],range(3))
        for values in ([0,1,1],[1,0,2],[0,1],[0,True,2],[0,1.,2]):
            with self.assertRaises(RuntimeError):aligned_indices(values,range(3))

    def test_stopped_delivery_not_confused_with_six_result_completion(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);atomic_json(root/'summary.json',{'terminal_state':'stopped'})
            atomic_json(root/'status.json',{'state':'stopped'})
            audit=dict(state='passed',terminal_state='stopped',results=0,quality_goals_met=False,
                       files={'summary.json':sha256_file(root/'summary.json')})
            atomic_json(root/'audit.json',audit)
            self.assertEqual(verify(root)['results'],0)
            atomic_json(root/'audit.json',dict(audit,results=6))
            with self.assertRaises(RuntimeError):verify(root)
            atomic_json(root/'audit.json',audit);atomic_json(root/'summary.json',{'tampered':True})
            with self.assertRaises(RuntimeError):verify(root)

    def test_controller_stops_without_formal_when_confirmation_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)
            src=SimpleNamespace(geometry=out,category=out,inputs={},profiles={},geo={'splits':{}},
                                refs={c:SimpleNamespace(fingerprint=c) for c in CITIES})
            args=SimpleNamespace(run_id='test',source_commit='sha',gpus='0,1')
            phases=[];sealed=[]
            def run(args,out,phase,configs):
                phases.append(phase);results,base=fixture(phase=='confirm')
                if phase=='confirm':
                    for c in CITIES:
                        for row in results[next(iter(results))]['cities'][c]:row['metrics']['CategoryTransition']=.2
                return results,base
            with patch('tools.run_distribution_study.copy_input'),patch('tools.run_distribution_study.fingerprint',return_value={}),\
                    patch('tools.run_distribution_study.phase_run',side_effect=run),\
                    patch('tools.run_distribution_study.seal_run',side_effect=lambda o,s,m,v:sealed.append(v)):
                execute(args,out,src)
            self.assertEqual(phases,['screen','confirm'])
            self.assertEqual(sealed[0]['terminal_state'],'stopped')
            self.assertFalse((out/'recommendation.json').exists())

    def test_cpu_recomputes_delivered_metrics_from_identical_input(self):
        from tests.test_geometry_projection import fixture as geo_fixture
        from tools.ablation_common import evaluate
        from experiment_io import publish_torch
        raw,records,_,_=geo_fixture()
        city=CITIES[0];indices=list(range(len(records)))
        metrics=evaluate(records,records,raw['poi_category'],indices)[0]
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);input_path=out/'evaluation_inputs'/city/f'{city}_train.pkl'
            publish_torch(input_path,raw)
            atomic_json(out/'manifest.json',{'profiles':{city:{'data_sha256':{'train':sha256_file(input_path)}}}})
            result_path=out/'results'/'screen'/city/'seed-135398'/'test.pkl'
            publish_torch(result_path,dict(city=city,split='train',indices=indices,sequences=records,result={'metrics':metrics}))
            atomic_json(out/'summary.json',{'terminal_state':'stopped'});atomic_json(out/'status.json',{'state':'stopped'})
            files={p.relative_to(out).as_posix():sha256_file(p) for p in out.rglob('*') if p.is_file() and p.name!='status.json'}
            atomic_json(out/'audit.json',dict(state='passed',terminal_state='stopped',results=0,quality_goals_met=False,files=files))
            self.assertEqual(verify(out,recompute=True)['metrics_recomputed'],1)

    def test_formal_identity_category_configuration_reuses_archived_a(self):
        from tests.test_category_cache_audit import fixture as cache_fixture
        items,off,on,mapping,gps=cache_fixture()
        with tempfile.TemporaryDirectory() as folder:
            worker=Worker.__new__(Worker)
            worker.out=Path(folder);worker.sha='manifest';worker.city=CITIES[0]
            worker.args=SimpleNamespace(phase='formal');worker.raw={'poi_category':mapping,'poi_gps':gps}
            packet=dict(temporal=items[0]['batch'],indices=items[0]['indices'],global_start=0,
                source_sha256='time',distance_backend='legacy',source_a_records=on,source_a_sha256='a',
                full_stats={'projection_stats':[{'optimizer_steps':500}]})
            with patch('tools.run_distribution_study.decode_preserving_empty',side_effect=AssertionError('Must reuse A')):
                result=worker.sample(packet,SEEDS[0],BalanceConfig('train_ref_v1',1.,0.,.1))
            self.assertTrue(result['stats']['reused_archived_a'])
            self.assertEqual(result['stats']['spatial_seconds'],0.)


if __name__=='__main__':unittest.main()
