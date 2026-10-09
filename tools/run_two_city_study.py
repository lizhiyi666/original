"""Two-city reporting study: reference immutable NY ablations, generate 39 new results."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import wandb
from experiment_io import atomic_json,publish_torch,sha256_file,safe_tag
from tools.run_pcdg_ablation import Ablation
from tools.run_newyork_ood import code_fingerprint
from tools.ablation_common import VARIANTS,SEEDS,VERSION,evaluate,same_records
from tools.city_common import profile,cfg_steps,verify_tracking
from tools.unified_tables import key,render
from baseline_posthoc_swap import apply_posthoc_swap

NY='NewYork_PO1_OOD'; IST='Istanbul_PO1_OOD'
NY_CFG_SHA='a3edb330d4e3f506680d2fae906faa5a9c287929ff2eb3995d9b8cbaa538e5c0'


class Study(Ablation):
    def __init__(self,args):
        self.args=args
        self.out=ROOT/'experiment_runs'/safe_tag(args.run_id)
        self.out.mkdir(parents=True,exist_ok=True)
        self.phase='precheck'; self.registry={}; self.cfg_cost={}
        self.ny_ablation=Path(args.newyork_ablation).resolve()
        self.source=Path(args.newyork_source).resolve()
        self.delivery=args.delivery_root.rstrip('/').replace('\\','/')

    def status(self,state,**extra):
        record=dict(state=state,phase=self.phase,pid=os.getpid(),updated_at=time.time(),
                    completed_unique_results=len(self.registry),target_unique_results=60,
                    completed_new_results=sum(r['origin']=='new' for r in self.registry.values()),**extra)
        atomic_json(self.out/'status.json',record)
        print(json.dumps(record),flush=True)

    def fingerprints(self):
        hashes=code_fingerprint()
        for name in ('tools/city_common.py','tools/unified_tables.py','tools/run_two_city_study.py',
                     'tools/ablation_worker.py','tools/ablation_common.py','tools/run_pcdg_ablation.py',
                     'tools/ablation_report.py','tools/train_baseline_cfg.py','tools/baseline_common.py',
                     'tools/perfcal_common.py','tests/test_two_city.py'):
            hashes[name]=sha256_file(ROOT/name)
        return hashes

    def precheck(self):
        if torch.cuda.device_count()!=2 or shutil.disk_usage(ROOT).free<5*1024**3:
            raise RuntimeError('Require two GPUs and 5 GiB disk space')
        ist=Path(self.args.istanbul_inputs).resolve()
        self.profiles={NY:profile(NY,self.source/'final.ckpt',self.source/'config_hydra.yaml',ROOT/'data'),
                       IST:profile(IST,ist/'base.ckpt',ist/'config_hydra.yaml',ROOT/'data')}
        self.ny_cfg=Path(self.args.newyork_cfg).resolve()
        if sha256_file(self.ny_cfg)!=NY_CFG_SHA:
            raise RuntimeError('NewYork CFG differs from the completed approved checkpoint')
        self.ny_manifest=json.loads((self.ny_ablation/'manifest.json').read_text())
        self.ny_manifest_sha=sha256_file(self.ny_ablation/'manifest.json')
        audit=json.loads((self.ny_ablation/'audit.json').read_text())
        if (audit['state']!='passed' or len(audit['runs'])!=21 or
                self.ny_manifest['checkpoint_sha256']!=self.profiles[NY]['checkpoint_sha256'] or
                self.ny_manifest['data_sha256']!=self.profiles[NY]['data_sha256'] or
                self.ny_manifest['seeds']!=list(SEEDS) or self.ny_manifest['variants']!=list(VARIANTS)):
            raise RuntimeError('NY ablations are not the approved completed 21-run study')
        self.entity=wandb.Api(timeout=30).default_entity
        expected=json.loads((self.source/'manifest.json').read_text())['entity']
        if self.entity!=expected:
            raise RuntimeError('W&B account changed')
        self.manifest=dict(version='two-city-v1',run_id=self.args.run_id,profiles=self.profiles,
            code_sha256=self.fingerprints(),source_ny_ablation=str(self.ny_ablation),
            source_ny_manifest_sha256=self.ny_manifest_sha,newyork_cfg=str(self.ny_cfg),newyork_cfg_sha256=NY_CFG_SHA,
            seeds=list(SEEDS),variants=VARIANTS,methods=['no_projection','postswap','energy','cfg','full'],
            rng_version=VERSION,paired_temporal_inputs=True,spatial_batch=64,
            energy=dict(temperature=3,scale=100,existence_weight=5,last_k=40,frequency=4),cfg_scale=1,
            fixed_budget=dict(outer=10,inner=50,early_stop=False,temperature=3),
            retrain_newyork=False,retrain_istanbul_base=False,train_istanbul_cfg=True,
            new_results=39,total_unique_results=60,downstream_tasks=False,
            precision='FP32 model; legacy normalization unchanged; no TF32',delivery_root=self.delivery)
        path=self.out/'manifest.json'
        if path.exists():
            if not self.args.resume or json.loads(path.read_text())!=self.manifest:
                raise RuntimeError('Existing study requires identical manifest and explicit --resume')
        elif self.args.resume:
            raise RuntimeError('Cannot resume without study manifest')
        else:
            if any(p.name!='pipeline.lock' for p in self.out.iterdir()):
                raise RuntimeError('Refusing to initialize over outputs without their manifest')
            atomic_json(path,self.manifest)
        self.manifest_sha=sha256_file(path)
        with (self.out/f'unit-tests-{time.time_ns()}.log').open('w') as stream:
            subprocess.run([sys.executable,'-B','-m','unittest','discover','-s','tests','-v'],
                           stdout=stream,stderr=subprocess.STDOUT,check=True)
        self.raw={d:torch.load(ROOT/f'data/{d}/{d}_test.pkl',map_location='cpu',weights_only=False) for d in (NY,IST)}

    def run_jobs(self,jobs):
        # A sealed cache is intentionally reused by multiple variants within one run.
        # The parent manifest has already enforced explicit cross-run --resume.
        known={}; pending=[]
        for job in jobs:
            folder=Path(job['output_dir']); path=folder/'result.json'
            if path.exists():
                result=json.loads(path.read_text())
                if result.get('state')=='complete':
                    if json.loads((folder/'job.json').read_text())!=job or sha256_file(folder/'payload.pkl')!=result['output_sha256']:
                        raise RuntimeError('Sealed job/cache differs from original inputs')
                    known[job['output_dir']]=result
                    continue
            pending.append(job)
        if pending:
            for job,result in zip(pending,super().run_jobs(pending)):
                known[job['output_dir']]=result
        return [known[job['output_dir']] for job in jobs]

    def register(self,dataset,seed,variant,path,metrics,timing,origin,origin_manifest,shards):
        p=Path(path)
        delivery=(str(Path(self.delivery).parent/'pcdg-ablation-v1'/p.relative_to(self.ny_ablation))
                  if origin=='reused' else self.delivery+'/'+str(p.relative_to(self.out)))
        self.registry[key(dataset,seed,variant)]=dict(dataset=dataset,seed=seed,variant=variant,path=str(p),
            sha256=sha256_file(p),metrics=metrics,timing=timing,origin=origin,
            origin_manifest_sha256=origin_manifest,source_shards=shards,delivery_path=delivery.replace('\\','/'))
        atomic_json(self.out/'registry.json',self.registry)

    def import_newyork(self):
        self.phase='import-newyork-21'; self.status('running')
        for seed in SEEDS:
            for variant in VARIANTS:
                folder=self.ny_ablation/f'seed-{seed}'/variant
                result=json.loads((folder/'metrics.json').read_text())
                artifact=folder/'generated.pkl'
                if sha256_file(artifact)!=result['output_sha256']:
                    raise RuntimeError('Old NY result fingerprint changed')
                data=torch.load(artifact,map_location='cpu',weights_only=False)
                if data['test_indices']!=list(range(self.profiles[NY]['test_count'])):
                    raise RuntimeError('Old NY result indices differ')
                metrics,rows=evaluate(self.raw[NY]['sequences'],data['sequences'],self.raw[NY]['poi_category'],data['test_indices'])
                if metrics!=result['metrics'] or rows!=json.loads((folder/'per_condition.json').read_text()):
                    raise RuntimeError('Recomputed NY metrics differ from original source')
                for shard in data['source_shards']:
                    if sha256_file(shard['path'])!=shard['sha256']:
                        raise RuntimeError('Reused NY shard changed')
                self.register(NY,seed,variant,artifact,metrics,result['timing'],'reused',self.ny_manifest_sha,data['source_shards'])
        receipt=self.ny_cfg.parent/'result.json'
        self.cfg_cost[NY]=dict(json.loads(receipt.read_text()),reused=True)
        self.tables()

    def tables(self,complete=False):
        render(self.out,self.registry,dict(self.manifest,cfg_training=self.cfg_cost),complete=complete)

    def city_job(self,dataset,folder,kind,indices,seed,gpu,split='test',start=0,variant=None,cache=None,cfg=None,
                 cache_origin=None,empty_row=None):
        folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
        p=self.profiles[dataset]
        return dict(city=p,kind=kind,split=split,indices=indices,seed=seed,gpu=gpu,global_start=start,
            variant=variant,cache=str(cache) if cache else None,cache_sha256=sha256_file(cache) if cache else None,
            cache_manifest_sha256=cache_origin or self.manifest_sha,
            cfg_checkpoint=str(cfg) if cfg else None,cfg_checkpoint_sha256=sha256_file(cfg) if cfg else None,
            source=str(self.source) if dataset==NY else str(Path(p['config_path']).parent),
            checkpoint=p['checkpoint'],checkpoint_sha256=p['checkpoint_sha256'],
            output_dir=str(folder),manifest_sha256=self.manifest_sha,test_empty_row=empty_row)

    def cache_paths(self,dataset,seed):
        if dataset==NY:
            paths=[self.ny_ablation/'time-cache'/f'seed-{seed}-rank-{r}'/'payload.pkl' for r in (0,1)]
            for path in paths:
                result=json.loads((path.parent/'result.json').read_text())
                if sha256_file(path)!=result['output_sha256']:
                    raise RuntimeError('Old NY time cache changed')
            return paths,self.ny_manifest_sha
        folders=[self.out/dataset/'time-cache'/f'seed-{seed}-rank-{r}' for r in (0,1)]
        n=self.profiles[dataset]['test_count']; half=(n+1)//2
        jobs=[self.city_job(dataset,folders[r],'cache',list(range(a,b)),seed,r,start=a)
              for r,(a,b) in enumerate(((0,half),(half,n)))]
        self.run_jobs(jobs)
        return [f/'payload.pkl' for f in folders],self.manifest_sha

    def publish_result(self,dataset,seed,variant,data,timing,shards):
        folder=self.out/dataset/f'seed-{seed}'/variant; folder.mkdir(parents=True,exist_ok=True)
        indices=list(range(self.profiles[dataset]['test_count']))
        if data['test_indices']!=indices or len(data['sequences'])!=len(indices):
            raise RuntimeError('Incomplete city output')
        metric_diagnostics = {}
        metrics,rows=evaluate(self.raw[dataset]['sequences'],data['sequences'],self.raw[dataset]['poi_category'],indices,diagnostics=metric_diagnostics)
        path=folder/'generated.pkl'
        if path.exists():
            old=torch.load(path,map_location='cpu',weights_only=False)
            if old['manifest_sha256']!=self.manifest_sha or not same_records(old['sequences'],data['sequences']):
                raise RuntimeError('Existing generated data differs')
        else: publish_torch(path,dict(data,manifest_sha256=self.manifest_sha,dataset=dataset,seed=seed,
                                     variant=variant,source_shards=shards,t_max=24.0))
        reported=folder/'metrics.json'
        if reported.exists():
            old=json.loads(reported.read_text())
            if old['metrics']!=metrics or old['output_sha256']!=sha256_file(path):
                raise RuntimeError('Existing metrics differ')
            timing=old['timing']
        result=dict(metrics=metrics,metric_diagnostics=metric_diagnostics,timing=timing,output_sha256=sha256_file(path))
        atomic_json(reported,result); atomic_json(folder/'per_condition.json',rows)
        receipt=folder/'wandb-sync.json'
        if not receipt.exists():
            run_id=f"{self.args.run_id}-{'ny' if dataset==NY else 'ist'}-{seed}-{variant}"
            run=wandb.init(project='Marionette',entity=self.entity,id=run_id,name=run_id,group=self.args.run_id,
                job_type='two-city-comparison',mode='online',resume='allow',dir=str(self.out),save_code=False,
                config=dict(dataset=dataset,seed=seed,variant=variant,study_manifest_sha256=self.manifest_sha,
                            base_checkpoint_sha256=self.profiles[dataset]['checkpoint_sha256']))
            run.log(dict(metrics,**timing)); run.summary.update(dict(metrics,**timing,complete=True,result_sha256=result['output_sha256']))
            run.finish()
            verify_tracking(self.entity,run_id,dict(complete=True,result_sha256=result['output_sha256']),receipt,run.url)
        else:
            proof=json.loads(receipt.read_text())
            if proof['expected']['result_sha256']!=result['output_sha256']:
                raise RuntimeError('Tracking receipt differs')
        self.register(dataset,seed,variant,path,metrics,timing,'new',self.manifest_sha,shards)

    def sampling(self,dataset,seed,variant,cfg=None):
        self.phase=f'{dataset}-{seed}-{variant}'; self.status('running')
        paths,origin=self.cache_paths(dataset,seed)
        n=self.profiles[dataset]['test_count']; half=(n+1)//2
        jobs=[self.city_job(dataset,self.out/dataset/f'seed-{seed}'/variant/f'rank-{r}','sample',
              list(range(a,b)),seed,r,start=a,variant=variant,cache=paths[r],cfg=cfg,cache_origin=origin)
              for r,(a,b) in enumerate(((0,half),(half,n)))]
        results=self.run_jobs(jobs); records=[]; indices=[]; shards=[]
        for job,result in zip(jobs,results):
            path=Path(job['output_dir'])/'payload.pkl'
            part=torch.load(path,map_location='cpu',weights_only=False)
            if part['job']!=job or sha256_file(path)!=result['output_sha256']:
                raise RuntimeError('Shard provenance mismatch')
            records+=part['sequences']; indices+=part['indices']
            shards.append(dict(path=str(path),sha256=result['output_sha256']))
        cache_cost=max(json.loads((p.parent/'result.json').read_text())['temporal_generation_seconds'] for p in paths)
        timing=dict(spatial_wall_seconds=max(r['spatial_seconds'] for r in results),
                    shared_temporal_cache_seconds=cache_cost,
                    peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in results),
                    actual_optimizer_steps=sum(r['optimizer_steps'] for r in results),
                    actual_projection_calls=sum(r['projection_calls'] for r in results))
        self.publish_result(dataset,seed,variant,dict(test_indices=indices,sequences=records),timing,shards)

    def postswap(self,dataset,seed):
        self.phase=f'{dataset}-{seed}-postswap'; self.status('running')
        source=self.registry[key(dataset,seed,'no_projection')]
        if sha256_file(source['path'])!=source['sha256']:
            raise RuntimeError('JointGen source changed')
        existing=self.out/dataset/f'seed-{seed}'/'postswap'
        if (existing/'metrics.json').exists() and (existing/'generated.pkl').exists():
            data=torch.load(existing/'generated.pkl',map_location='cpu',weights_only=False)
            prior=json.loads((existing/'metrics.json').read_text())
            if data['parent_sha256']!=source['sha256'] or sha256_file(existing/'generated.pkl')!=prior['output_sha256']:
                raise RuntimeError('Existing PostSwap derivation changed')
            self.publish_result(dataset,seed,'postswap',data,prior['timing'],data['source_shards'])
            return
        old=torch.load(source['path'],map_location='cpu',weights_only=False)
        started=time.perf_counter()
        fixed,_=apply_posthoc_swap(old['sequences'],self.raw[dataset]['sequences'],self.raw[dataset]['poi_category'],
                                  po_matrices=None,verbose=False)
        elapsed=time.perf_counter()-started
        for a,b in zip(old['sequences'],fixed):
            for field in ['arrival_times']+[f'condition{i}{suffix}' for i in range(1,7) for suffix in ('','_indicator')]:
                if not np.array_equal(a[field],b[field]): raise RuntimeError('PostSwap changed time/context')
            def events(s): return Counter((int(p),int(m),tuple(g)) for p,m,g in zip(s['checkins'],s['marks'],s['gps']))
            if len(a['checkins'])!=len(b['checkins']) or events(a)!=events(b): raise RuntimeError('PostSwap changed events')
        folder=self.out/dataset/f'seed-{seed}'/'postswap'; folder.mkdir(parents=True,exist_ok=True)
        n=self.profiles[dataset]['test_count']; half=(n+1)//2; shards=[]
        for rank,(a,b) in enumerate(((0,half),(half,n))):
            path=folder/f'part-{rank}.pkl'
            part=dict(indices=list(range(a,b)),sequences=fixed[a:b],parent_sha256=source['sha256'],manifest_sha256=self.manifest_sha)
            if not path.exists(): publish_torch(path,part)
            elif not same_records(torch.load(path,map_location='cpu',weights_only=False)['sequences'],part['sequences']):
                raise RuntimeError('Existing PostSwap shard differs')
            shards.append(dict(path=str(path),sha256=sha256_file(path)))
        metrics,_=evaluate(self.raw[dataset]['sequences'],fixed,self.raw[dataset]['poi_category'],list(range(n)))
        if metrics['pair_coverage']!=source['metrics']['pair_coverage']: raise RuntimeError('PostSwap changed coverage')
        base=source['timing']['spatial_wall_seconds']
        self.publish_result(dataset,seed,'postswap',dict(test_indices=list(range(n)),sequences=fixed,parent_sha256=source['sha256']),
            dict(spatial_wall_seconds=base+elapsed,base_spatial_seconds=base,posthoc_seconds=elapsed),shards)

    def reference_trace_paths(self,dataset,seed,rank):
        if dataset==NY:
            return self.ny_ablation/f'seed-{seed}'/'no_projection'/f'rank-{rank}'/'payload.pkl'
        return self.out/dataset/f'seed-{seed}'/'no_projection'/f'rank-{rank}'/'payload.pkl'

    def verify_city_pairing(self,dataset,seed,variants):
        for rank in (0,1):
            paths=[self.reference_trace_paths(dataset,seed,rank)]
            paths += [self.out/dataset/f'seed-{seed}'/v/f'rank-{rank}/payload.pkl' for v in variants]
            self.verify_pairing(paths)

    def preflight_ny(self):
        self.phase='NewYork-loader-and-baseline-preflight'; self.status('running')
        pilot=self.ny_ablation/'preflight'
        for variant in ('no_projection','full','energy','cfg'):
            paths=[pilot/'train-cache/payload.pkl',pilot/'history-cache/payload.pkl']
            jobs=[]
            for gpu,(cache,split) in enumerate(zip(paths,('train','test'))):
                source=torch.load(cache,map_location='cpu',weights_only=False)
                jobs.append(self.city_job(NY,self.out/'preflight-ny'/variant/str(gpu),'sample',source['indices'],135398,gpu,
                    split=split,start=source['batches'][0]['global_start'],variant=variant,cache=cache,
                    cfg=self.ny_cfg if variant=='cfg' else None,cache_origin=self.ny_manifest_sha,
                    empty_row=35 if split=='test' else None))
            self.run_jobs(jobs)
            for gpu,part in enumerate(('train','history')):
                path=self.out/'preflight-ny'/variant/str(gpu)/'payload.pkl'
                reference=pilot/(variant if variant in ('full','no_projection') else 'no_projection')/part/'payload.pkl'
                self.verify_pairing([reference,path])
                if variant in ('full','no_projection'):
                    a=torch.load(reference,map_location='cpu',weights_only=False); b=torch.load(path,map_location='cpu',weights_only=False)
                    if not same_records(a['sequences'],b['sequences']): raise RuntimeError('NY base-loader migration changed outputs')
        atomic_json(self.out/'preflight-ny/receipt.json',dict(state='passed',old_full_and_jointgen_equal=True))

    def ist_pilot_caches(self):
        folder=self.out/'preflight-ist'
        normal=folder/'time-cache'
        job=self.city_job(IST,normal,'cache',list(range(64)),135398,0,split='train')
        self.run_jobs([job])
        source=normal/'payload.pkl'
        data=torch.load(source,map_location='cpu',weights_only=False)
        empty=folder/'empty-fixture'; empty.mkdir(exist_ok=True)
        target=empty/'payload.pkl'
        if not target.exists():
            # A derived test-only input; no original trajectory or formal cache is altered.
            batch=data['batches'][0]['batch']
            active=((batch.unpadded_length>0)&batch.po_matrix.bool().flatten(1).any(1)).nonzero().flatten()
            row=int(active[0]) if len(active) else 0
            batch.unpadded_length[row]=0; batch.mask[row].zero_()
            batch.category_mask[row].zero_(); batch.poi_mask[row].zero_()
            data['test_only_synthetic_empty_row']=row
            data['derived_from_sha256']=sha256_file(source)
            publish_torch(target,data)
        row=torch.load(target,map_location='cpu',weights_only=False)['test_only_synthetic_empty_row']
        return source,target,row

    def preflight_ist(self,cfg=None):
        self.phase='Istanbul-CFG-preflight' if cfg else 'Istanbul-layout-and-ablation-preflight'; self.status('running')
        normal,empty,row=self.ist_pilot_caches()
        variants=['cfg'] if cfg else ['no_projection']+[v for v in VARIANTS if v!='no_projection']+['energy']
        for variant in variants:
            jobs=[self.city_job(IST,self.out/'preflight-ist'/variant/str(gpu),'sample',list(range(64)),135398,gpu,
                    split='train',variant=variant,cache=cache,cfg=cfg,empty_row=row if gpu else None)
                  for gpu,cache in enumerate((normal,empty))]
            self.run_jobs(jobs)
            for gpu in (0,1):
                self.verify_pairing([self.out/'preflight-ist'/'no_projection'/str(gpu)/'payload.pkl',
                                     self.out/'preflight-ist'/variant/str(gpu)/'payload.pkl'])
        if not cfg:
            job=self.city_job(IST,self.out/'preflight-ist/full-gpu1','sample',list(range(64)),135398,1,
                              split='train',variant='full',cache=normal)
            self.run_jobs([job])
            first=torch.load(self.out/'preflight-ist/full/0/payload.pkl',map_location='cpu',weights_only=False)
            second=torch.load(Path(job['output_dir'])/'payload.pkl',map_location='cpu',weights_only=False)
            if not same_records(first['sequences'],second['sequences']): raise RuntimeError('Istanbul Full differs across GPUs')
        atomic_json(self.out/'preflight-ist'/('cfg-receipt.json' if cfg else 'receipt.json'),
                    dict(state='passed',semantic_classes=9,model_slots=10,synthetic_empty_row=row))

    def training(self,epochs):
        self.phase='Istanbul-CFG-preflight-training' if epochs==2 else 'Istanbul-CFG-training'; self.status('running')
        folder=self.out/IST/('cfg-preflight' if epochs==2 else 'cfg-training'); folder.mkdir(parents=True,exist_ok=True)
        job=dict(city=self.profiles[IST],epochs=epochs,output_dir=str(folder),entity=self.entity,
            wandb_id=f"{self.args.run_id}-ist-cfg-{'preflight' if epochs==2 else 'training'}",group=self.args.run_id,
            study_manifest_sha256=self.manifest_sha,optimizer='AdamW',learning_rate=.001,weight_decay=0,
            batch_size=64,po_drop_probability=.1,temporal_frozen=True,expected_updates=cfg_steps(self.profiles[IST],epochs))
        path=folder/'job.json'
        if path.exists() and json.loads(path.read_text())!=job: raise RuntimeError('Training job changed')
        if not path.exists(): atomic_json(path,job)
        result_path=folder/'result.json'
        ready=result_path.exists() and json.loads(result_path.read_text()).get('state')=='complete'
        if not ready:
            command=[sys.executable,'-u','-B','tools/train_baseline_cfg.py','--job',str(path)]
            if self.args.resume: command+=['--resume']
            with (folder/f'train-{time.time_ns()}.log').open('w') as stream:
                subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True,
                               env=dict(os.environ,CUDA_VISIBLE_DEVICES='0'))
        result=json.loads(result_path.read_text()); checkpoint=folder/'last.ckpt'
        if (result['state']!='complete' or result['global_step']!=job['expected_updates'] or
                sha256_file(checkpoint)!=result['checkpoint_sha256']): raise RuntimeError('CFG training proof mismatch')
        if not (folder/'wandb-sync.json').exists():
            run=wandb.init(project='Marionette',entity=self.entity,id=job['wandb_id'],group=self.args.run_id,
                job_type='cfg-sync-recovery',resume='allow',mode='online',dir=str(folder),config=job)
            run.summary.update(dict(result,complete=True)); run.finish()
            verify_tracking(self.entity,job['wandb_id'],dict(complete=True,checkpoint_sha256=result['checkpoint_sha256'],
                            global_step=result['global_step']),folder/'wandb-sync.json',run.url)
        if epochs==1000: self.cfg_cost[IST]=dict(result,reused=False)
        return checkpoint

    def audit(self):
        if len(self.registry)!=60: raise RuntimeError('Expected 60 unique dataset/seed/result records')
        for dataset in (NY,IST):
            for seed in SEEDS:
                self.verify_city_pairing(dataset,seed,['energy','cfg']+(list(VARIANTS) if dataset==IST else []))
            p=self.profiles[dataset]
            if sha256_file(p['checkpoint'])!=p['checkpoint_sha256']: raise RuntimeError('Base checkpoint changed')
            for split,digest in p['data_sha256'].items():
                if sha256_file(Path(p['data_root'])/dataset/f'{dataset}_{split}.pkl')!=digest: raise RuntimeError('Dataset changed')
        for record in self.registry.values():
            if sha256_file(record['path'])!=record['sha256']: raise RuntimeError('Registered output changed')
            data=torch.load(record['path'],map_location='cpu',weights_only=False)
            indices=data.get('test_indices',data.get('indices'))
            if indices!=list(range(self.profiles[record['dataset']]['test_count'])): raise RuntimeError('Audit indices differ')
            rows=[]; records=[]; shard_indices=[]
            for shard in record['source_shards']:
                if sha256_file(shard['path'])!=shard['sha256']: raise RuntimeError('Source shard changed')
                part=torch.load(shard['path'],map_location='cpu',weights_only=False)
                records+=part['sequences']; shard_indices+=part.get('indices',part.get('test_indices'))
            if shard_indices!=indices or not same_records(records,data['sequences']): raise RuntimeError('Shard/merge mismatch')
            metrics,_=evaluate(self.raw[record['dataset']]['sequences'],records,self.raw[record['dataset']]['poi_category'],indices)
            if metrics!=record['metrics']: raise RuntimeError('Recomputed shared metrics differ')
        if self.fingerprints()!=self.manifest['code_sha256']: raise RuntimeError('Code changed during study')
        atomic_json(self.out/'audit.json',dict(state='passed',unique_results=60,new_results=39,reused_results=21,
            samples={NY:2108,IST:4914},shared_metrics=True,paired_spatial_rng=True,all_hashes_valid=True,
            identical_method_ablation_anchors=True,originals_unchanged=True))

    def execute(self):
        self.precheck(); self.import_newyork(); self.preflight_ny()
        for seed in SEEDS:
            self.postswap(NY,seed); self.sampling(NY,seed,'energy'); self.sampling(NY,seed,'cfg',self.ny_cfg)
            self.verify_city_pairing(NY,seed,['energy','cfg']); self.tables()
        self.preflight_ist()
        for seed in SEEDS:
            for variant in VARIANTS: self.sampling(IST,seed,variant)
            self.postswap(IST,seed); self.sampling(IST,seed,'energy'); self.tables()
        self.training(2); checkpoint=self.training(1000); self.preflight_ist(checkpoint)
        for seed in SEEDS: self.sampling(IST,seed,'cfg',checkpoint)
        self.phase='final-audit'; self.status('running'); self.audit(); self.tables(complete=True)
        self.phase='complete'; self.status('complete',total_unique_results=60,base_models_retrained=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--newyork-source',required=True); parser.add_argument('--newyork-ablation',required=True)
    parser.add_argument('--newyork-cfg',required=True); parser.add_argument('--istanbul-inputs',required=True)
    parser.add_argument('--delivery-root',required=True); parser.add_argument('--run-id',default='two-city-v1')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(); os.chdir(ROOT); study=Study(args)
    import fcntl
    with (study.out/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try: study.execute()
        except BaseException as exc:
            study.status('failed',error_type=type(exc).__name__,error=str(exc)); raise


if __name__=='__main__': main()
