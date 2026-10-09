import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import torch
from tools.city_common import cfg_steps,expanded_matrix
from tools.ablation_worker import check_cache
from tools.unified_tables import (METHODS,ABLATIONS,JSD,CONSTRAINTS,SEEDS,key,table,render,aggregate)
from tools.run_two_city_study import Study
from experiment_io import atomic_json,publish_torch,sha256_file


class DatasetAdapterTests(unittest.TestCase):
    def test_cfg_budget_is_dataset_specific(self):
        self.assertEqual(cfg_steps({'train_count':3160}),50000)
        self.assertEqual(cfg_steps({'train_count':7035}),110000)
        self.assertEqual(cfg_steps({'train_count':7035},2),220)

    def test_legacy_slots_only_pad_constraints_without_relabeling(self):
        matrix=torch.zeros(9,9); matrix[0,8]=1
        original=matrix.clone()
        padded=expanded_matrix(matrix,9,10)
        self.assertEqual(padded.shape,(10,10))
        self.assertTrue(torch.equal(matrix,original))
        self.assertTrue(torch.equal(padded[:9,:9],matrix))
        self.assertEqual(float(padded[9].sum()+padded[:,9].sum()),0)
        self.assertIs(expanded_matrix(matrix,9,9),matrix)
        with self.assertRaises(ValueError): expanded_matrix(matrix,9,8)

    def cache(self):
        matrix=torch.zeros(2,10,10); matrix[:,0,1]=1
        batch=SimpleNamespace(batch_size=2,time=torch.tensor([[1.,2.],[1.,2.]]),
            mask=torch.ones(2,2,dtype=torch.bool),unpadded_length=torch.tensor([2,2]),po_matrix=matrix)
        cache=dict(indices=[0,1],manifest_sha256='old',batches=[dict(batch=batch,indices=[0,1])],
                   job=dict(checkpoint_sha256='weights',seed=135398,split='test'))
        job=dict(indices=[0,1],manifest_sha256='new',cache_manifest_sha256='old',
                 city=dict(semantic_classes=9,model_category_slots=10),
                 checkpoint_sha256='weights',seed=135398,split='test')
        return cache,job

    def test_reference_origin_is_separate_from_new_manifest(self):
        cache,job=self.cache(); check_cache(cache,job)
        self.assertEqual(cache['manifest_sha256'],'old')
        wrong=dict(job,cache_manifest_sha256='new')
        with self.assertRaises(RuntimeError): check_cache(cache,wrong)

    def test_foreign_weights_seed_and_unused_edges_are_rejected(self):
        for update in ({'checkpoint_sha256':'other'},{'seed':135399}):
            cache,job=self.cache()
            with self.assertRaises(RuntimeError): check_cache(cache,dict(job,**update))
        cache,job=self.cache(); cache['batches'][0]['batch'].po_matrix[:,9,0]=1
        with self.assertRaises(RuntimeError): check_cache(cache,job)


class TableTests(unittest.TestCase):
    def records(self):
        result={}
        variants={v for v,_ in (*METHODS,*ABLATIONS)}
        for dataset in ('NewYork_PO1_OOD','Istanbul_PO1_OOD'):
            for seed in SEEDS:
                for variant in variants:
                    m={k:.1+(seed-SEEDS[0])*.01 for k,_,_ in (*JSD,*CONSTRAINTS)}
                    m.update(totalJSD=.6,empty_count=1,evaluation_version=2)
                    result[key(dataset,seed,variant)]=dict(dataset=dataset,seed=seed,variant=variant,
                        metrics=m,timing={},path='/test/fixture.pkl',sha256='test-only',origin='test-only')
        return result

    def test_method_and_ablation_headers_and_anchor_values_match(self):
        records=self.records()
        for metrics,percent in ((JSD,False),(CONSTRAINTS,True)):
            _,_,methods=table(records,METHODS,metrics,percent)
            _,_,ablations=table(records,ABLATIONS,metrics,percent)
            for dataset in methods:
                for anchor in ('full','no_projection'):
                    self.assertEqual(methods[dataset][anchor],ablations[dataset][anchor])
        self.assertEqual([k for k,_,_ in JSD],['Distance','Radius','CategoryTransition','DailyLoc','Category','G-RANK'])
        self.assertEqual(len(CONSTRAINTS),5)

    def test_best_uses_unrounded_mean_not_full_preference(self):
        records=self.records()
        for dataset in ('NewYork_PO1_OOD','Istanbul_PO1_OOD'):
            for seed in SEEDS:
                records[key(dataset,seed,'full')]['metrics']['Distance']=.10004
                records[key(dataset,seed,'energy')]['metrics']['Distance']=.10003
        md,tex,_=table(records,METHODS,JSD)
        energy=next(line for line in md.splitlines() if line.startswith('| EnergyGuide'))
        full=next(line for line in md.splitlines() if line.startswith('| PCDG'))
        self.assertTrue(energy.startswith('| EnergyGuide | **0.1000'))
        self.assertTrue(full.startswith('| PCDG | 0.1000'))
        self.assertIn(r'\toprule',tex); self.assertIn(r'\bottomrule',tex)
        self.assertIn(r'Istanbul\_PO1\_OOD',tex)

    def test_percent_precision_undefined_and_sample_sd(self):
        records=self.records()
        for seed in SEEDS: records[key('NewYork_PO1_OOD',seed,'cfg')]['metrics']['ovr_skip']=None
        md,_,_=table(records,METHODS,CONSTRAINTS,True)
        self.assertIn('11.00 ± 1.00',md)
        self.assertIn('—',md)
        value=aggregate(records,'NewYork_PO1_OOD','full','strict_ovr')
        self.assertAlmostEqual(value['sd'],.01)

    def test_incomplete_city_is_not_reported_as_final(self):
        records=self.records()
        records={k:v for k,v in records.items() if v['dataset']=='NewYork_PO1_OOD' and v['variant'] in dict(ABLATIONS)}
        with tempfile.TemporaryDirectory() as directory:
            render(directory,records,{},complete=False)
            text=(Path(directory)/'report.md').read_text(encoding='utf-8')
            self.assertIn('阶段性',text)
            self.assertIn('21/60',text)
            self.assertNotIn('### Istanbul_PO1_OOD',text)
            self.assertIn('Table not ready',(Path(directory)/'methods-jsd.tex').read_text())


class JobReuseTests(unittest.TestCase):
    def test_same_run_cache_reuse_never_launches_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            job=dict(output_dir=str(folder),kind='cache',manifest_sha256='sealed')
            atomic_json(folder/'job.json',job)
            publish_torch(folder/'payload.pkl',dict(example='cache'))
            value=dict(state='complete',output_sha256=sha256_file(folder/'payload.pkl'))
            atomic_json(folder/'result.json',value)
            study=object.__new__(Study); study.args=SimpleNamespace(resume=False)
            with mock.patch('tools.run_pcdg_ablation.subprocess.Popen') as process:
                self.assertEqual(study.run_jobs([job]),[value]); process.assert_not_called()
            with self.assertRaises(RuntimeError): study.run_jobs([dict(job,manifest_sha256='changed')])


if __name__=='__main__': unittest.main()
