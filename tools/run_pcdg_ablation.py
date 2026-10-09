"""Paired 7 x 3 PCDG ablations. No training, tuning, or legacy-result substitution."""
import argparse
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
import torch
import wandb
from omegaconf import OmegaConf
from experiment_io import atomic_json,publish_torch,sha256_file,safe_tag
from evaluate_utils import get_run_data
from tools.ablation_common import VERSION,VARIANTS,DISTANCE_VERSION,DISTANCE_VARIANTS,SEEDS,STEPS,evaluate,same_records
from evaluations.statistical_metrics import EVALUATION_VERSION
from tools.ablation_report import make_report
from tools.run_newyork_ood import code_fingerprint,DATA_HASHES


class Ablation:
    def __init__(self,args):
        self.args=args
        self.projection_revision = getattr(args, 'projection_revision', VERSION)
        self.variant_settings = DISTANCE_VARIANTS if self.projection_revision == DISTANCE_VERSION else VARIANTS
        self.out=ROOT/'experiment_runs'/safe_tag(args.run_id)
        self.out.mkdir(parents=True,exist_ok=True)
        self.cache=Path(args.cache_dir).resolve() if args.cache_dir else self.out/'time-cache'
        self.cache.mkdir(parents=True,exist_ok=True)
        self.source=Path(args.source_experiment).resolve()
        self.checkpoint=Path(args.checkpoint).resolve() if args.checkpoint else self.source/'final.ckpt'
        self.phase='precheck'

    def fingerprints(self):
        paths=code_fingerprint()
        for name in ('tools/ablation_common.py','tools/ablation_worker.py','tools/ablation_report.py',
                     'tools/run_pcdg_ablation.py','tools/baseline_common.py','tools/perfcal_common.py',
                     'tests/test_ablation.py','tests/fixtures/projection_reference_d5d7d01.py'):
            paths[name]=sha256_file(ROOT/name)
        return paths

    def status(self,state,**extra):
        atomic_json(self.out/'status.json',dict(state=state,phase=self.phase,pid=os.getpid(),
                    updated_at=time.time(),**extra))
        print(json.dumps(dict(state=state,phase=self.phase,**extra)),flush=True)

    def precheck(self):
        if len(set(self.args.seeds))!=len(self.args.seeds) or len(set(self.args.variants))!=len(self.args.variants):
            raise ValueError('Duplicate seed/variant')
        if 'full' not in self.args.variants:
            raise ValueError('Full is required for paired differences')
        source=json.loads((self.source/'manifest.json').read_text())
        if (source['dataset']!='NewYork_PO1_OOD' or source['epochs']!=1000 or
                source['train_batch_size']!=64 or source['sample_count']!=2108):
            raise RuntimeError('Wrong source experiment')
        if torch.cuda.device_count()!=2 or shutil.disk_usage(ROOT).free < 5*1024**3:
            raise RuntimeError('Require two GPUs and at least 5 GiB free')
        versions={k:importlib.metadata.version(k) for k in source['packages']}
        if versions!=source['packages']:
            raise RuntimeError('Package versions changed')
        for split,digest in DATA_HASHES.items():
            if sha256_file(ROOT/f'data/NewYork_PO1_OOD/NewYork_PO1_OOD_{split}.pkl')!=digest:
                raise RuntimeError('Dataset changed')
        saved=torch.load(self.checkpoint,map_location='cpu',weights_only=False)
        if (saved.get('epoch')!=999 or 'state_dict' not in saved or
                any('po_encoder' in key for key in saved.get('state_dict',{})) or
                sha256_file(self.checkpoint)!=sha256_file(self.source/'final.ckpt')):
            raise RuntimeError('Only the original completed non-CFG checkpoint is allowed')
        del saved
        _,_,run_path=get_run_data(source['run_id'],ROOT/'wandb')
        if sha256_file(Path(run_path)/'config_hydra.yaml')!=sha256_file(self.source/'config_hydra.yaml'):
            raise RuntimeError('Source checkpoint config changed')
        config=OmegaConf.load(Path(run_path)/'config_hydra.yaml')
        if config.model.get('use_constraint_projection',False) or config.model.get('po_cfg_enabled',False):
            raise RuntimeError('Expected original unprojected base model configuration')
        all_indices=json.loads(Path(self.args.preflight_indices).read_text())
        if len(all_indices)!=512 or len(set(all_indices))!=512 or min(all_indices)<0 or max(all_indices)>=3160:
            raise RuntimeError('Invalid original calibration pool')
        self.pilot_indices=all_indices[:64]
        self.entity=wandb.Api(timeout=30).default_entity
        if self.entity!=source['entity']:
            raise RuntimeError('W&B account changed')
        self.manifest=dict(version=self.projection_revision,evaluation_version=EVALUATION_VERSION,run_id=self.args.run_id,source=str(self.source),
            source_run_id=source['run_id'],checkpoint=str(self.checkpoint),checkpoint_sha256=sha256_file(self.checkpoint),
            source_manifest_sha256=sha256_file(self.source/'manifest.json'),
            source_config_sha256=sha256_file(self.source/'config_hydra.yaml'),data_sha256=DATA_HASHES,
            code_sha256=self.fingerprints(),packages=versions,seeds=self.args.seeds,variants=self.args.variants,
            variant_settings=self.variant_settings,cache_dir=str(self.cache),batch_size=64,world_size=2,sample_count=2108,
            projection_steps=STEPS,outer=10,inner=50,early_stop=False,temperature=3,learning_rate=1,
            lambda_init=1,mu_init=1,mu_max=1000,mu_alpha=2,gradient_clip=10,
            precision='model FP32; legacy softmax normalization retained; no autocast or TF32',
            rng_protocol='SHA256 compact JSON [version, seed, global_start, stream]; first8 big-endian & 2^63-1',
            temporal_seed_protocol='seed + global batch start; retain adaptive sampler state across batches',
            preflight_indices=self.pilot_indices,fixture=str(Path(self.args.historical_fixture).resolve()),
            fixture_sha256=sha256_file(self.args.historical_fixture),retrained=False,hyperparameter_search=False)
        path=self.out/'manifest.json'
        if path.exists():
            if not self.args.resume or json.loads(path.read_text())!=self.manifest:
                raise RuntimeError('Existing study requires identical inputs/code and --resume')
        elif self.args.resume:
            raise RuntimeError('Cannot resume without original manifest')
        else:
            atomic_json(path,self.manifest)
        self.manifest_sha=sha256_file(path)
        self.status('running')
        with (self.out/f'unit-tests-{time.time_ns()}.log').open('w') as stream:
            subprocess.run([sys.executable,'-B','-m','unittest','discover','-s','tests','-v'],
                           stdout=stream,stderr=subprocess.STDOUT,check=True)

    def job(self,folder,kind,split,indices,seed,gpu,global_start=0,variant=None,cache=None,fixture=None,historical_empty=False):
        folder.mkdir(parents=True,exist_ok=True)
        return dict(kind=kind,split=split,indices=indices,seed=seed,gpu=gpu,global_start=global_start,
            projection_revision=getattr(self, 'projection_revision', VERSION),
            variant=variant,cache=str(cache) if cache else None,cache_sha256=sha256_file(cache) if cache else None,
            fixture=str(fixture) if fixture else None,fixture_sha256=sha256_file(fixture) if fixture else None,
            historical_empty=bool(fixture) if kind=='cache' else historical_empty,
            source=str(self.source),checkpoint=str(self.checkpoint),checkpoint_sha256=self.manifest['checkpoint_sha256'],
            output_dir=str(folder),manifest_sha256=self.manifest_sha)

    def run_jobs(self,jobs):
        if shutil.disk_usage(ROOT).free < 5*1024**3:
            raise RuntimeError('Less than 5 GiB free')
        pending=[]
        results={}
        try:
            for job in jobs:
                folder=Path(job['output_dir'])
                path=folder/'job.json'
                if path.exists() and json.loads(path.read_text())!=job:
                    raise RuntimeError('Job inputs differ from immutable original')
                if not path.exists():
                    atomic_json(path,job)
                result=folder/'result.json'
                if result.exists():
                    if not self.args.resume:
                        raise RuntimeError('Existing worker result requires --resume')
                    value=json.loads(result.read_text())
                    if value['state']=='complete':
                        if sha256_file(folder/'payload.pkl')!=value['output_sha256']:
                            raise RuntimeError('Worker artifact checksum mismatch')
                        results[str(folder)]=value
                        continue
                elif (folder/'payload.pkl').exists() and not self.args.resume:
                    raise RuntimeError('Published artifact needs explicit resume reconciliation')
                log=(folder/f'worker-{time.time_ns()}.log').open('w')
                process=subprocess.Popen([sys.executable,'-u','-B','tools/ablation_worker.py','--job',str(path)],
                    cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(job['gpu'])))
                pending.append((process,log,folder))
            while any(p.poll() is None for p,_,_ in pending):
                if any(p.poll() not in (None,0) for p,_,_ in pending):
                    raise RuntimeError('Worker failed; peer stopped and merge refused')
                time.sleep(2)
            for process,log,folder in pending:
                if process.returncode:
                    raise RuntimeError(f'Worker failed: {folder}')
                value=json.loads((folder/'result.json').read_text())
                if value['state']!='complete' or sha256_file(folder/'payload.pkl')!=value['output_sha256']:
                    raise RuntimeError('Worker completion/sha mismatch')
                results[str(folder)]=value
            return [results[j['output_dir']] for j in jobs]
        finally:
            for process,log,_ in pending:
                if process.poll() is None:
                    process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()
                log.close()

    def verify_pairing(self,paths):
        payloads=[torch.load(p,map_location='cpu',weights_only=False) for p in paths]
        expected=[(b['indices'],b['spatial_rng_before'],b['spatial_rng_after']) for b in payloads[0]['batch_traces']]
        for payload in payloads:
            actual=[(b['indices'],b['spatial_rng_before'],b['spatial_rng_after']) for b in payload['batch_traces']]
            if actual!=expected:
                raise RuntimeError('Variants did not use identical discrete-sampling random streams')

    def preflight(self):
        self.phase='real-data-preflight'; self.status('running')
        root=self.out/'preflight'
        train=root/'train-cache'
        history=root/'history-cache'
        self.run_jobs([self.job(train,'cache','train',self.pilot_indices,135398,0),
            self.job(history,'cache','test',list(range(64,128)),135398,1,64,
                     fixture=Path(self.args.historical_fixture).resolve())])
        for variant in self.args.variants:
            self.run_jobs([self.job(root/variant/'train','sample','train',self.pilot_indices,135398,0,
                                   variant=variant,cache=train/'payload.pkl'),
                           self.job(root/variant/'history','sample','test',list(range(64,128)),135398,1,64,
                                   variant=variant,cache=history/'payload.pkl',historical_empty=True)])
        self.verify_pairing([root/v/'train/payload.pkl' for v in self.args.variants])
        self.verify_pairing([root/v/'history/payload.pkl' for v in self.args.variants])
        self.run_jobs([self.job(root/'full-gpu1','sample','train',self.pilot_indices,135398,1,
                               variant='full',cache=train/'payload.pkl')])
        a=torch.load(root/'full/train/payload.pkl',map_location='cpu',weights_only=False)
        b=torch.load(root/'full-gpu1/payload.pkl',map_location='cpu',weights_only=False)
        if not same_records(a['sequences'],b['sequences']):
            raise RuntimeError('Fixed Full preflight differs across the two GPUs')
        atomic_json(root/'receipt.json',dict(state='passed',manifest_sha256=self.manifest_sha,
            variants=self.args.variants,global_rng_unchanged=True,discrete_streams_paired=True,
            frozen_empty_kept=True,full_two_gpu_records_equal=True))

    def merge_evaluate(self,seed,variant,jobs,results):
        folder=self.out/f'seed-{seed}'/variant
        payloads=[torch.load(Path(j['output_dir'])/'payload.pkl',map_location='cpu',weights_only=False) for j in jobs]
        indices=[]; generated=[]; condition_rows=[]
        for job,part in zip(jobs,payloads):
            if part['job']!=job:
                raise RuntimeError('Shard job mismatch')
            indices+=part['indices']; generated+=part['sequences']; condition_rows+=part['per_condition']
        if indices!=list(range(2108)) or len(generated)!=2108:
            raise RuntimeError('Full OOD count/index mismatch')
        raw=torch.load(ROOT/'data/NewYork_PO1_OOD/NewYork_PO1_OOD_test.pkl',map_location='cpu',weights_only=False)
        metric_diagnostics = {}
        metrics,rows=evaluate(raw['sequences'],generated,raw['poi_category'],indices,diagnostics=metric_diagnostics)
        if rows!=condition_rows:
            raise RuntimeError('Merged per-condition diagnostics differ from shard diagnostics')
        path=folder/'generated.pkl'
        artifact=dict(format=self.projection_revision,seed=seed,variant=variant,manifest_sha256=self.manifest_sha,
            test_indices=indices,t_max=24.0,sequences=generated,
            source_shards=[dict(path=str(Path(j['output_dir'])/'payload.pkl'),sha256=r['output_sha256'])
                           for j,r in zip(jobs,results)])
        if path.exists():
            previous=torch.load(path,map_location='cpu',weights_only=False)
            if previous['manifest_sha256']!=self.manifest_sha or not same_records(previous['sequences'],generated):
                raise RuntimeError('Existing merged artifact differs')
        else:
            publish_torch(path,artifact)
        atomic_json(folder/'per_condition.json',rows)
        timing=dict(spatial_wall_seconds=max(r['spatial_seconds'] for r in results),
            summed_projection_seconds=sum(r['projection_seconds'] for r in results),
            actual_optimizer_steps=sum(r['optimizer_steps'] for r in results),
            actual_projection_calls=sum(r['projection_calls'] for r in results),
            peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in results))
        value=dict(seed=seed,variant=variant,metrics=metrics,metric_diagnostics=metric_diagnostics,timing=timing,output_sha256=sha256_file(path),
                   worker_diagnostics=results)
        existing=folder/'metrics.json'
        if existing.exists() and json.loads(existing.read_text())!=value:
            raise RuntimeError('Recomputed numerical result differs on resume')
        atomic_json(existing,value)
        return value

    def sync(self,result):
        seed,variant=result['seed'],result['variant']
        folder=self.out/f'seed-{seed}'/variant
        receipt=folder/'wandb-sync.json'
        if receipt.exists():
            previous=json.loads(receipt.read_text())
            if previous['output_sha256']!=result['output_sha256']:
                raise RuntimeError('Tracking receipt points to another result')
            return
        self.phase=f'sync-{seed}-{variant}'; self.status('running')
        run_id=f'{self.args.run_id}-{seed}-{variant}'
        run=wandb.init(project='Marionette',entity=self.entity,id=run_id,name=run_id,
            group=self.args.run_id,job_type='pcdg-ablation',mode='online',resume='allow',dir=str(self.out),
            save_code=False,config=dict(self.manifest,variant=variant,sampling_seed=seed))
        run.log(dict(result['metrics'],**result['timing']))
        run.summary.update(dict(result['metrics'],**result['timing'],complete=True,
                                result_sha256=result['output_sha256']))
        run.finish()
        last=None
        for delay in (0,2,5):
            if delay:
                time.sleep(delay)
            try:
                remote=wandb.Api(timeout=30).run(f'{self.entity}/Marionette/{run_id}')
                if remote.summary.get('complete') is True and remote.summary.get('result_sha256')==result['output_sha256']:
                    atomic_json(receipt,dict(state='verified',output_sha256=result['output_sha256'],url=run.url))
                    return
                last='Summary not yet consistent'
            except Exception as exc:
                last=str(exc)
        raise RuntimeError(f'Tracking readback failed; numerical files retained; resume sync only: {last}')

    def audit(self):
        audits=[]
        raw=torch.load(ROOT/'data/NewYork_PO1_OOD/NewYork_PO1_OOD_test.pkl',map_location='cpu',weights_only=False)
        for seed in self.args.seeds:
            for rank in (0,1):
                self.verify_pairing([self.out/f'seed-{seed}'/v/f'rank-{rank}/payload.pkl' for v in self.args.variants])
                cache=self.cache/f'seed-{seed}-rank-{rank}'
                result=json.loads((cache/'result.json').read_text())
                if sha256_file(cache/'payload.pkl')!=result['output_sha256']:
                    raise RuntimeError('Sealed time cache changed')
            for variant in self.args.variants:
                folder=self.out/f'seed-{seed}'/variant
                reported=json.loads((folder/'metrics.json').read_text())
                data=torch.load(folder/'generated.pkl',map_location='cpu',weights_only=False)
                if sha256_file(folder/'generated.pkl')!=reported['output_sha256'] or data['test_indices']!=list(range(2108)):
                    raise RuntimeError('Audit merged hash/index mismatch')
                rows=[]; sequences=[]; indices=[]
                for record in data['source_shards']:
                    if sha256_file(record['path'])!=record['sha256']:
                        raise RuntimeError('Audit source shard changed')
                    part=torch.load(record['path'],map_location='cpu',weights_only=False)
                    sequences+=part['sequences']; indices+=part['indices']; rows+=part['per_condition']
                if indices!=data['test_indices'] or not same_records(sequences,data['sequences']):
                    raise RuntimeError('Audit source shards and merged records differ')
                metrics,recomputed=evaluate(raw['sequences'],sequences,raw['poi_category'],indices)
                if metrics!=reported['metrics'] or rows!=recomputed:
                    raise RuntimeError('Audit metric recomputation differs')
                audits.append(dict(seed=seed,variant=variant,samples=2108,state='passed'))
        if self.fingerprints()!=self.manifest['code_sha256'] or sha256_file(self.checkpoint)!=self.manifest['checkpoint_sha256']:
            raise RuntimeError('Code/checkpoint changed during experiment')
        for split,digest in DATA_HASHES.items():
            if sha256_file(ROOT/f'data/NewYork_PO1_OOD/NewYork_PO1_OOD_{split}.pkl')!=digest:
                raise RuntimeError('Dataset changed during experiment')
        atomic_json(self.out/'audit.json',dict(state='passed',runs=audits,time_caches_unchanged=True,
                                             paired_discrete_rng=True,code_and_inputs_unchanged=True))

    def execute(self):
        self.precheck(); self.preflight()
        for seed in self.args.seeds:
            self.phase=f'time-cache-{seed}'; self.status('running')
            caches=[self.cache/f'seed-{seed}-rank-{rank}' for rank in (0,1)]
            self.run_jobs([self.job(caches[r],'cache','test',list(range(a,b)),seed,r,a)
                           for r,(a,b) in enumerate(((0,1054),(1054,2108)))])
            for variant in self.args.variants:
                self.phase=f'spatial-{seed}-{variant}'; self.status('running')
                jobs=[self.job(self.out/f'seed-{seed}'/variant/f'rank-{r}','sample','test',
                    list(range(a,b)),seed,r,a,variant,caches[r]/'payload.pkl')
                    for r,(a,b) in enumerate(((0,1054),(1054,2108)))]
                results=self.run_jobs(jobs)
                result=self.merge_evaluate(seed,variant,jobs,results)
                self.sync(result)
        self.phase='audit'; self.status('running'); self.audit()
        make_report(self.out,self.args.seeds,self.args.variants)
        self.phase='complete'; self.status('complete',runs=len(self.args.seeds)*len(self.args.variants),retrained=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-experiment',required=True)
    parser.add_argument('--checkpoint')
    parser.add_argument('--preflight-indices',required=True)
    parser.add_argument('--historical-fixture',required=True)
    parser.add_argument('--cache-dir')
    parser.add_argument('--projection-revision',choices=[VERSION,DISTANCE_VERSION],default=VERSION)
    parser.add_argument('--run-id')
    parser.add_argument('--seeds',type=int,nargs='+',choices=SEEDS,default=list(SEEDS))
    parser.add_argument('--variants',nargs='+',choices=list(DISTANCE_VARIANTS))
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    variants = DISTANCE_VARIANTS if args.projection_revision == DISTANCE_VERSION else VARIANTS
    args.variants = args.variants or list(variants)
    if any(v not in variants for v in args.variants):
        parser.error('no_distance_kl requires --projection-revision pcdg-distance-v2')
    args.run_id = args.run_id or args.projection_revision
    os.chdir(ROOT)
    experiment=Ablation(args)
    import fcntl
    with (experiment.out/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            experiment.execute()
        except BaseException as exc:
            experiment.status('failed',error_type=type(exc).__name__,error=str(exc))
            raise


if __name__=='__main__':
    main()
