"""Versioned baseline2/3/4 suite. Formal OOD settings are selected on TRAIN only."""
import argparse
from collections import Counter
import importlib.metadata
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
from baseline_posthoc_swap import apply_posthoc_swap
from experiment_io import atomic_json,publish_torch,sha256_file,safe_tag
from tools.baseline_common import DATASET,SEED,validate_alignment,full_metrics,select_candidate
from tools.run_newyork_ood import code_fingerprint,DATA_HASHES


class Suite:
    def __init__(self,args):
        self.args=args
        self.id=safe_tag(args.suite_id)
        self.out=ROOT/'experiment_runs'/self.id
        self.out.mkdir(parents=True,exist_ok=True)
        self.source=Path(args.source_experiment).resolve()
        self.reference=Path(args.reference_run).resolve()
        self.baseline1=Path(args.baseline1).resolve()
        self.indices=json.loads(Path(args.indices).read_text())
        self.phase='precheck'

    def status(self,state,**extra):
        if state=='running' and shutil.disk_usage(ROOT).free < 5*1024**3:
            raise RuntimeError('Less than 5 GiB free before next phase')
        atomic_json(self.out/'status.json',dict(state=state,phase=self.phase,pid=os.getpid(),
                    updated_at=time.time(),**extra))
        print(json.dumps(dict(state=state,phase=self.phase,**extra)),flush=True)

    def fingerprints(self):
        hashes=code_fingerprint()
        for name in ('tools/baseline_common.py','tools/baseline_worker.py','tools/train_baseline_cfg.py',
                     'tools/run_baseline_suite.py','tests/test_baseline_suite.py',
                     'tests/fixtures/diffusion_reference_6e4a5dd.py'):
            hashes[name]=sha256_file(ROOT/name)
        return hashes

    def precheck(self):
        self.original=json.loads((self.source/'manifest.json').read_text())
        self.entity=wandb.Api(timeout=30).default_entity
        if self.entity!=self.original['entity'] or torch.cuda.device_count()!=2:
            raise RuntimeError('Expected original W&B account and two visible GPUs')
        if shutil.disk_usage(ROOT).free < 5*1024**3:
            raise RuntimeError('Less than 5 GiB disk space')
        if {k:importlib.metadata.version(k) for k in self.original['packages']}!=self.original['packages']:
            raise RuntimeError('Training environment changed')
        for split,digest in DATA_HASHES.items():
            if sha256_file(ROOT/f'data/{DATASET}/{DATASET}_{split}.pkl')!=digest:
                raise RuntimeError('Dataset fingerprint changed')
        if len(self.indices)!=512 or len(set(self.indices))!=512 or min(self.indices)<0 or max(self.indices)>=3160:
            raise RuntimeError('Expected original 512 unique training calibration indices')
        old_native=json.loads((self.reference/'metrics-native.json').read_text())
        old_projection=json.loads((self.reference/'metrics-projection.json').read_text())
        if sha256_file(self.baseline1)!=old_native['output_sha256']:
            raise RuntimeError('baseline1 differs from verified perfcal-v1-ood-r2 output')
        saved=torch.load(self.source/'final.ckpt',map_location='cpu',weights_only=False)
        if saved['epoch']!=999:
            raise RuntimeError('Original checkpoint did not complete 1000 epochs')
        del saved
        self.manifest=dict(revision='baseline-suite-v1',suite_id=self.id,methods=self.args.methods,
            source=str(self.source),source_manifest_sha256=sha256_file(self.source/'manifest.json'),
            checkpoint_sha256=sha256_file(self.source/'final.ckpt'),data_sha256=DATA_HASHES,
            baseline1=str(self.baseline1),baseline1_sha256=old_native['output_sha256'],
            reference_run=str(self.reference),reference_projection_sha256=old_projection['output_sha256'],
            code_sha256=self.fingerprints(),seed=SEED,batch_size=64,world_size=2,indices=self.indices,
            precision='FP32; no autocast; TF32 disabled',
            energy=dict(temperatures=[1,2,3],scales=[1,10,100],existence_weight=5,last_k=40,frequency=4),
            cfg=dict(scales=[0,1,1.5,2,3,5],po_drop_probability=.1,epochs=1000,steps=50000,
                     preflight_epochs=2,temporal_frozen=True,spatial_initialization='fresh'),
            entity=self.entity,selection_split='train',independent_validation=False)
        path=self.out/'manifest.json'
        if path.exists():
            if not self.args.resume or json.loads(path.read_text())!=self.manifest:
                raise RuntimeError('Existing suite requires identical inputs/code and explicit --resume')
        elif self.args.resume:
            raise RuntimeError('Cannot resume without suite manifest')
        else:
            atomic_json(path,self.manifest)
        self.manifest_sha=sha256_file(path)
        log=self.out/f'unit-tests-{time.time_ns()}.log'
        with log.open('w') as stream:
            subprocess.run([sys.executable,'-B','-m','unittest','discover','-s','tests','-v'],
                           stdout=stream,stderr=subprocess.STDOUT,check=True)
        self.metrics={'baseline1':old_native,'projection':old_projection}
        self.status('running')

    def job(self,label,method,split,indices,gpu,scale=1,temperature=0,cfg=None,fixture=None):
        folder=self.out/label
        folder.mkdir(exist_ok=True)
        return dict(method=method,split=split,indices=indices,gpu=gpu,scale=scale,temperature=temperature,
            cfg_checkpoint=str(cfg) if cfg else None,cfg_sha256=sha256_file(cfg) if cfg else None,
            source=str(self.source),output_dir=str(folder),suite_manifest_sha256=self.manifest_sha,
            fixture=str(fixture) if fixture else None,fixture_sha256=sha256_file(fixture) if fixture else None)

    def run_jobs(self,jobs):
        processes=[]
        results={}
        try:
            for job in jobs:
                folder=Path(job['output_dir'])
                path=folder/'job.json'
                if path.exists() and json.loads(path.read_text())!=job:
                    raise RuntimeError('Job parameters changed')
                if not path.exists():
                    atomic_json(path,job)
                result_path=folder/'result.json'
                if result_path.exists():
                    previous=json.loads(result_path.read_text())
                    if not self.args.resume:
                        raise RuntimeError('Existing job needs --resume')
                    if previous['state']=='complete':
                        if sha256_file(folder/'generated.pkl')!=previous['output_sha256']:
                            raise RuntimeError('Completed worker output hash changed')
                        results[str(folder)]=previous
                        continue
                    atomic_json(folder/f'result-history-{time.time_ns()}.json',previous)
                    if (folder/'generated.pkl').exists():
                        raise RuntimeError('Nonterminal worker already published output; manual audit required')
                log=(folder/f'worker-{time.time_ns()}.log').open('w')
                proc=subprocess.Popen([sys.executable,'-u','-B','tools/baseline_worker.py','--job',str(path)],
                    stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(job['gpu'])))
                processes.append((proc,log,folder))
            while any(p.poll() is None for p,_,_ in processes):
                if any(p.poll() not in (None,0) for p,_,_ in processes):
                    # Record candidate failures; sibling jobs are still awaited to preserve their outputs.
                    pass
                time.sleep(2)
            for proc,log,folder in processes:
                log.close()
                if not (folder/'result.json').exists():
                    raise RuntimeError(f'Worker exited without result: {folder}')
                result=json.loads((folder/'result.json').read_text())
                if proc.returncode and result['state']=='complete':
                    raise RuntimeError('Contradictory worker exit/result')
                results[str(folder)]=result
            return [results[j['output_dir']] for j in jobs]
        finally:
            for proc,log,_ in processes:
                if proc.poll() is None:
                    proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                log.close()

    def publish_metrics(self,method,path,timing):
        generated=torch.load(path,map_location='cpu',weights_only=False)
        raw=torch.load(ROOT/f'data/{DATASET}/{DATASET}_test.pkl',map_location='cpu',weights_only=False)
        if generated['indices']!=list(range(2108)):
            raise RuntimeError('Full OOD indices are incomplete')
        validate_alignment(generated['sequences'],raw['sequences'],raw['poi_category'])
        metrics=full_metrics(raw['sequences'],generated['sequences'],raw['poi_category'])
        run=wandb.init(project='Marionette',entity=self.entity,id=f'{self.id}-{method}-ood',
            name=f'{self.id}-{method}-ood',group=self.original['run_id'],job_type='baseline-ood-evaluation',
            mode='online',resume='allow',dir=str(self.out),save_code=False,
            config=dict(self.manifest,method=method))
        run.log(dict(metrics,**timing))
        run.summary.update(dict(metrics,**timing,complete=True,output_sha256=sha256_file(path)))
        run.finish()
        if wandb.Api(timeout=30).run(f'{self.entity}/Marionette/{run.id}').summary.get('complete') is not True:
            raise RuntimeError('W&B evaluation readback failed')
        result=dict(metrics=metrics,timing=timing,output=str(path),output_sha256=sha256_file(path),wandb_url=run.url)
        atomic_json(self.out/f'metrics-{method}.json',result)
        self.metrics[method]=result
        return result

    def baseline2(self):
        self.phase='baseline2-posthoc'
        self.status('running')
        out=self.out/'baseline2.pkl'
        if not out.exists():
            old=torch.load(self.baseline1,map_location='cpu',weights_only=False)
            raw=torch.load(ROOT/f'data/{DATASET}/{DATASET}_test.pkl',map_location='cpu',weights_only=False)
            started=time.perf_counter()
            fixed,summary=apply_posthoc_swap(old['sequences'],raw['sequences'],raw['poi_category'],
                                            po_matrices=None,verbose=False)
            elapsed=time.perf_counter()-started
            for before,after in zip(old['sequences'],fixed):
                for key in ['arrival_times']+[f'condition{i}' for i in range(1,7)]+[f'condition{i}_indicator' for i in range(1,7)]:
                    if not np.array_equal(before[key],after[key]):
                        raise RuntimeError('Posthoc changed times/context')
                def events(s):
                    return Counter((int(p),int(m),tuple(g)) for p,m,g in zip(s['checkins'],s['marks'],s['gps']))
                if len(before['checkins'])!=len(after['checkins']) or events(before)!=events(after):
                    raise RuntimeError('Posthoc changed event multiset')
            before_metrics=full_metrics(raw['sequences'],old['sequences'],raw['poi_category'])
            after_metrics=full_metrics(raw['sequences'],fixed,raw['poi_category'])
            if before_metrics['pair_coverage']!=after_metrics['pair_coverage']:
                raise RuntimeError('Posthoc changed pair coverage')
            payload=dict(sequences=fixed,indices=list(range(2108)),method='baseline2',
                         suite_manifest_sha256=self.manifest_sha,parent_sha256=sha256_file(self.baseline1),
                         posthoc_seconds=elapsed,summary=summary)
            for rank,(a,b) in enumerate(((0,1054),(1054,2108))):
                publish_torch(self.out/f'baseline2-part{rank}.pkl',dict(payload,sequences=fixed[a:b],indices=list(range(a,b))))
            publish_torch(out,payload)
        saved=torch.load(out,map_location='cpu',weights_only=False)
        if saved['suite_manifest_sha256']!=self.manifest_sha:
            raise RuntimeError('Posthoc result belongs to another suite')
        self.publish_metrics('baseline2',out,dict(posthoc_seconds=saved['posthoc_seconds'],
            base_sampling_seconds=self.metrics['baseline1']['metrics']['wall_seconds']))

    def calibrate(self,method,cfg=None):
        self.phase=f'{method}-train-calibration'
        self.status('running')
        run=wandb.init(project='Marionette',entity=self.entity,id=f'{self.id}-{method}-cal',
            group=self.original['run_id'],job_type='train-engineering-calibration',mode='online',resume='allow',
            dir=str(self.out),save_code=False,config=dict(method=method,indices=self.indices,independent_validation=False))
        rows=[]
        grid=[(t,s) for t in (1,2,3) for s in (1,10,100)] if method=='baseline3' else [(0,s) for s in (0,1,1.5,2,3,5)]
        try:
            for temperature,scale in grid:
                label=f'{method}-cal-t{temperature}-s{scale}'
                job=self.job(label,method,'train',self.indices,0,scale,temperature,cfg)
                result=self.run_jobs([job])[0]
                rows.append(result)
                atomic_json(self.out/f'calibration-{method}.json',rows)
                run.log(dict(temperature=temperature,scale=scale,state=result['state'],**result.get('metrics',{})))
                print(json.dumps(dict(candidate=label,state=result['state'],metrics=result.get('metrics'))),flush=True)
            chosen=select_candidate(rows,method)
            selection=dict(method=method,scale=chosen['scale'],temperature=chosen['temperature'],
                           metrics=chosen['metrics'],reference_split='train',independent_validation=False)
            atomic_json(self.out/f'selection-{method}.json',selection)
            run.summary.update(dict(selected=selection,complete=True))
            run.finish()
            if wandb.Api(timeout=30).run(f'{self.entity}/Marionette/{run.id}').summary.get('complete') is not True:
                raise RuntimeError('Calibration W&B readback failed')
            return selection
        except BaseException:
            run.finish(exit_code=1)
            raise

    def guided_ood(self,method,selection,cfg=None):
        self.phase=f'{method}-safety-regression'
        self.status('running')
        scale,temperature=selection['scale'],selection['temperature']
        first=self.job(f'{method}-gpu1-check',method,'train',self.indices[:64],1,scale,temperature,cfg)
        fixture=self.reference/'frozen-empty-regression/legacy-temporal-input.pkl'
        empty=self.job(f'{method}-empty-check',method,'test',list(range(64,128)),0,scale,temperature,cfg,fixture)
        if any(r['state']!='complete' for r in self.run_jobs([first,empty])):
            raise RuntimeError('Second-GPU/fixed-empty regression failed')
        self.phase=f'{method}-full-ood'
        self.status('running')
        started=time.perf_counter()
        jobs=[self.job(f'{method}-rank{rank}',method,'test',list(range(a,b)),rank,scale,temperature,cfg)
              for rank,(a,b) in enumerate(((0,1054),(1054,2108)))]
        results=self.run_jobs(jobs)
        if any(r['state']!='complete' for r in results):
            raise RuntimeError('Full OOD shard failed; merge refused')
        indices,sequences=[],[]
        for job,result in zip(jobs,results):
            path=Path(job['output_dir'])/'generated.pkl'
            if sha256_file(path)!=result['output_sha256']:
                raise RuntimeError('Shard hash changed')
            payload=torch.load(path,map_location='cpu',weights_only=False)
            if payload['job']!=job or payload['indices']!=job['indices']:
                raise RuntimeError('Shard parameters/index mismatch')
            indices+=payload['indices']
            sequences+=payload['sequences']
        if indices!=list(range(2108)) or len(sequences)!=2108:
            raise RuntimeError('Incomplete full OOD output')
        path=self.out/f'{method}.pkl'
        if not path.exists():
            publish_torch(path,dict(indices=indices,sequences=sequences,method=method,
                                   selection=selection,suite_manifest_sha256=self.manifest_sha))
        else:
            old=torch.load(path,map_location='cpu',weights_only=False)
            if old['suite_manifest_sha256']!=self.manifest_sha or old['selection']!=selection:
                raise RuntimeError('Existing merged result mismatch')
        timing=dict(wall_seconds=time.perf_counter()-started,
            worker_seconds=max(r['sampling_seconds'] for r in results),
            peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in results))
        if self.args.resume and (self.out/f'metrics-{method}.json').exists():
            timing=json.loads((self.out/f'metrics-{method}.json').read_text())['timing']
        self.publish_metrics(method,path,timing)

    def train_cfg(self,epochs):
        self.phase='cfg-preflight' if epochs==2 else 'cfg-training'
        self.status('running')
        folder=self.out/self.phase
        folder.mkdir(exist_ok=True)
        job=dict(source=str(self.source),epochs=epochs,output_dir=str(folder),entity=self.entity,
                 wandb_id=f'{self.id}-{self.phase}',group=self.original['run_id'],
                 suite_manifest_sha256=self.manifest_sha,optimizer='AdamW',learning_rate=.001,
                 weight_decay=0,lr_factor=.95,lr_patience=1000,batch_size=64,
                 po_drop_probability=.1,temporal_frozen=True,spatial_initialization='fresh')
        path=folder/'job.json'
        if path.exists() and json.loads(path.read_text())!=job:
            raise RuntimeError('Training job changed')
        atomic_json(path,job)
        if (folder/'result.json').exists():
            result=json.loads((folder/'result.json').read_text())
            if result['state']=='complete':
                if not self.args.resume or sha256_file(folder/'last.ckpt')!=result['checkpoint_sha256']:
                    raise RuntimeError('Existing CFG training requires verified --resume')
                return folder/'last.ckpt'
        command=[sys.executable,'-u','-B','tools/train_baseline_cfg.py','--job',str(path)]
        if self.args.resume:
            command+=['--resume']
        with (folder/f'train-{time.time_ns()}.log').open('w') as log:
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,
                           env=dict(os.environ,CUDA_VISIBLE_DEVICES='0'))
        result=json.loads((folder/'result.json').read_text())
        if result['state']!='complete' or result['global_step']!=epochs*50:
            raise RuntimeError('CFG training did not complete its exact budget')
        return folder/'last.ckpt'

    def audit_and_report(self):
        raw=torch.load(ROOT/f'data/{DATASET}/{DATASET}_test.pkl',map_location='cpu',weights_only=False)
        audits={}
        for method in self.args.methods:
            path=self.out/f'{method}.pkl'
            reported=json.loads((self.out/f'metrics-{method}.json').read_text())
            if sha256_file(path)!=reported['output_sha256']:
                raise RuntimeError('Published method result changed')
            data=torch.load(path,map_location='cpu',weights_only=False)
            if data['indices']!=list(range(2108)):
                raise RuntimeError('Published OOD index mismatch')
            validate_alignment(data['sequences'],raw['sequences'],raw['poi_category'])
            recomputed=full_metrics(raw['sequences'],data['sequences'],raw['poi_category'])
            if any(abs(v-reported['metrics'][k])>1e-12 for k,v in recomputed.items()):
                raise RuntimeError('Independent metric recomputation differs')
            parts=[]
            for rank in (0,1):
                shard=(self.out/f'baseline2-part{rank}.pkl' if method=='baseline2' else
                       self.out/f'{method}-rank{rank}/generated.pkl')
                part=torch.load(shard,map_location='cpu',weights_only=False)
                if method!='baseline2':
                    proof=json.loads((shard.parent/'result.json').read_text())
                    if sha256_file(shard)!=proof['output_sha256'] or proof['projection_calls']!=0:
                        raise RuntimeError('Shard fingerprint/ALM invariant failed')
                parts+=part['indices']
            if parts!=list(range(2108)):
                raise RuntimeError('Source shards missing indices')
            audits[method]=dict(state='passed',samples=2108,indices_and_conditions=True,
                               metrics_recomputed=True,source_shards_retained=True,sha256=sha256_file(path))
        atomic_json(self.out/'audit.json',dict(state='passed',methods=audits))
        lines=['# Baseline suite v1 — 完整 OOD 对照','',
            '每组均为 NewYork_PO1_OOD 的完整 2108 条条件。引导参数仅在固定训练集工程校准池中选择。','',
            '| 方法 | 严格违反率 | 类别对覆盖率 | 类别覆盖率 | 空输出 |','|---|---:|---:|---:|---:|']
        for method in ('baseline1','baseline2','baseline3','baseline4','projection'):
            if method not in self.metrics:
                continue
            m=self.metrics[method]['metrics']
            strict=m.get('strict_ovr',m.get('OVR_ref_strict'))
            coverage=m.get('category_coverage',m.get('coverage'))
            lines.append(f"| {method} | {strict:.2%} | {m['pair_coverage']:.2%} | {coverage:.2%} | {m['empty_count']} |")
        lines+=['','## 统计指标','',
                '| 方法 | Distance | Radius | DailyLoc | Interval | Category | G-RANK | totalJSD |',
                '|---|---:|---:|---:|---:|---:|---:|---:|']
        for method,record in self.metrics.items():
            m=record['metrics']
            lines.append('| '+method+' | '+' | '.join(f'{m[k]:.6f}' for k in
                ('Distance','Radius','DailyLoc','Interval','Category','G-RANK','totalJSD'))+' |')
        lines+=['','## 设置和时间','']
        for method in self.args.methods:
            lines.append(f"- {method}: {json.dumps(self.metrics[method]['timing'],ensure_ascii=False)}")
            selection=self.out/f'selection-{method}.json'
            if selection.exists():
                lines.append(f"- {method} 训练集选定设置: {selection.read_text()}")
            lines.append(f"- [{method} W&B]({self.metrics[method]['wandb_url']})")
        if 'baseline4' in self.args.methods:
            training=json.loads((self.out/'cfg-training/result.json').read_text())
            lines.append(f"- CFG 独立空间训练：1000 轮、50000 次更新，耗时 {training['training_seconds']:.1f} 秒；时间模型权重指纹 {training['temporal_sha256']} 保持不变。")
            lines.append('- CFG scale=1 是有条件模型极限；若选中该值，不表示外推引导取得额外收益。')
        lines+=['','基线训练/采样成本不同，不能宣称所有方法使用同一空间检查点。',
                'baseline2 只排序已有事件，不能补回缺失类别。完整 JSON 保留 Unsat、跳过缺失后的违反率等全部指标。',
                '所有方法通过索引、条件、文件指纹与指标重算检查；原 baseline1/投影和原训练检查点未覆盖。','']
        (self.out/'report.md').write_text('\n'.join(lines),encoding='utf-8')

    def execute(self):
        self.precheck()
        if 'baseline2' in self.args.methods:
            self.baseline2()
        if 'baseline3' in self.args.methods:
            selection=self.calibrate('baseline3')
            self.guided_ood('baseline3',selection)
        if 'baseline4' in self.args.methods:
            self.train_cfg(2)
            checkpoint=self.train_cfg(1000)
            selection=self.calibrate('baseline4',checkpoint)
            self.guided_ood('baseline4',selection,checkpoint)
        if self.fingerprints()!=self.manifest['code_sha256']:
            raise RuntimeError('Code changed during suite')
        if sha256_file(self.source/'final.ckpt')!=self.manifest['checkpoint_sha256']:
            raise RuntimeError('Original checkpoint changed')
        if sha256_file(self.baseline1)!=self.manifest['baseline1_sha256']:
            raise RuntimeError('Original baseline1 changed')
        for split,digest in DATA_HASHES.items():
            if sha256_file(ROOT/f'data/{DATASET}/{DATASET}_{split}.pkl')!=digest:
                raise RuntimeError('Input data changed')
        atomic_json(self.out/'comparison.json',self.metrics)
        self.audit_and_report()
        self.phase='complete'
        self.status('complete',methods=self.args.methods,ood_samples_per_method=2108)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-experiment',required=True)
    parser.add_argument('--baseline1',required=True)
    parser.add_argument('--reference-run',required=True)
    parser.add_argument('--indices',required=True)
    parser.add_argument('--suite-id',default='nyood-baselines-v1-20261002')
    parser.add_argument('--methods',nargs='+',choices=['baseline2','baseline3','baseline4'],
                        default=['baseline2','baseline3','baseline4'])
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    os.chdir(ROOT)
    suite=Suite(args)
    import fcntl
    with (suite.out/'suite.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            suite.execute()
        except BaseException as exc:
            suite.status('failed',error_type=type(exc).__name__,error=str(exc))
            raise


if __name__=='__main__':
    main()
