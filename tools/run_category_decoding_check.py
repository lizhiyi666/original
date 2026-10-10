"""128-row paired NewYork diagnostic; reuse sealed off outputs and time inputs."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('OMP_NUM_THREADS','4')
import numpy as np
import torch
from experiment_io import atomic_json, publish_torch, safe_tag, sha256_file
from discrete_diffusion.diffusion_transformer import CATEGORY_DECODING_VERSION
from tools import ablation_worker as worker
from tools.ablation_common import evaluate
from tools.baseline_common import precision, state_hash
from tools.geometry_inheritance import checked_path
from evaluations.statistical_metrics import evaluation, travel_distance

CITY='NewYork_PO1_OOD'
SEED=135398


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def fingerprint():
    paths=list(ROOT.glob('*.py'))
    for folder in ('tools','tests','config','add_thin','discrete_diffusion','evaluations'):
        paths += [p for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in ('.py','.yaml','.yml')]
    return {p.relative_to(ROOT).as_posix():sha256_file(p) for p in sorted(paths)}


def subset_inputs(cache, shard, indices):
    wanted=set(indices);batches=[];records=[];traces=[];covered=[];offset=0
    for item in cache['batches']:
        intersection=wanted.intersection(item['indices'])
        if intersection and intersection!=set(item['indices']):
            raise RuntimeError('Diagnostic must use complete original batches')
        if intersection:
            if len(item['indices'])!=64 or item['batch'].batch_size!=64:
                raise RuntimeError('Fixed batch64 is required')
            batches.append(copy.deepcopy(item));covered.extend(item['indices'])
    if covered!=indices:raise RuntimeError('Missing/duplicate/reordered time-cache rows')
    covered=[]
    for trace in shard['batch_traces']:
        n=len(trace['indices']);values=shard['sequences'][offset:offset+n];offset+=n
        if wanted.intersection(trace['indices']):
            if not set(trace['indices']).issubset(wanted):raise RuntimeError('Partial original output batch')
            covered.extend(trace['indices']);records.extend(values);traces.append(copy.deepcopy(trace))
    if covered!=indices or offset!=len(shard['sequences']):raise RuntimeError('Original output alignment mismatch')
    if [(b['indices'],b['global_start']) for b in batches] != [(t['indices'],t['global_start']) for t in traces]:
        raise RuntimeError('Time inputs and original output traces do not align')
    derived=copy.deepcopy(cache)
    derived.update(format='category-decoding-cache-subset-v1',indices=indices,batches=batches)
    derived['source_result']=copy.deepcopy(cache.get('result'))
    derived['result']=dict(state='complete',kind='cache-subset',samples=len(indices),resampled=False)
    if 'adaptive_states' in derived:
        starts={b['global_start'] for b in batches}
        derived['adaptive_states']=[a for a in derived['adaptive_states'] if a['global_start'] in starts]
    return derived,records,traces


def compare_pair(before,after,old_traces,new_traces):
    if len(before)!=128 or len(after)!=128 or len(old_traces)!=2 or len(new_traces)!=2:
        raise RuntimeError('Expected exactly two paired batches / 128 records')
    for a,b in zip(before,after):
        if a.keys()!=b.keys() or len(a['checkins'])!=len(b['checkins']):
            raise RuntimeError('Generated length or record fields changed')
        if any(not np.array_equal(a[k],b[k]) for k in a if k not in ('marks','checkins','gps')):
            raise RuntimeError('Time, condition or empty-record metadata changed')
    for a,b in zip(old_traces,new_traces):
        for key in ('indices','global_start','spatial_rng_before','spatial_rng_after',
                    'projection_rng_after','distance_rng_before','distance_rng_after'):
            if a.get(key)!=b.get(key):raise RuntimeError('Paired trace mismatch: '+key)


def distribution_counts(refs,records,mapping):
    eligible=[i for i,(r,s) in enumerate(zip(refs,records)) if len(r['marks']) and len(s['checkins'])
              and int(r['marks'][0])==mapping[int(s['checkins'][0])]
              and int(r['marks'][-1])==mapping[int(s['checkins'][-1])]]
    real=[float(travel_distance(np.asarray(r['gps']))) for r in refs if len(r['gps'])>1]
    generated=[float(travel_distance(np.asarray(s['gps']))) for s in records if len(s['gps'])>1]
    return dict(endpoint_indices=eligible,
        distance_real_count=sum(len(refs[i]['gps'])>1 for i in eligible),
        distance_generated_count=sum(len(records[i]['gps'])>1 for i in eligible),
        unfiltered_distance_jsd=evaluation(generated,real) if generated and real else None)


def verify_delivery(folder):
    folder=Path(folder);audit=read(folder/'audit.json');status=read(folder/'status.json')
    if (status['state']!='complete' or audit['state']!='passed' or audit['new_samples']!=128
            or audit['baseline_samples']!=128 or not audit['originals_unchanged'] or not audit['paired_rng']):
        raise RuntimeError('Diagnostic is not a complete audited pair')
    for name,expected in audit['files'].items():
        if sha256_file(checked_path(folder,name))!=expected:raise RuntimeError('Artifact hash mismatch: '+name)
    summary=read(folder/'comparison.json')
    if summary['on']['category_poi_mismatch_rate']!=0:
        raise RuntimeError('Final POI/category mismatch remains')
    return dict(state='passed',new_samples=128,reused_baseline_samples=128,
                verified_files=len(audit['files'])+2,audit_sha256=sha256_file(folder/'audit.json'),
                manifest_sha256=sha256_file(folder/'manifest.json'),actual_mismatch_rate=0.)


def execute(args,out):
    precision()
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; no smaller-batch/CPU substitution')
    parent=Path(args.source_run).resolve()
    if ROOT.resolve()==parent.parent.parent or out.resolve().is_relative_to(parent):
        raise RuntimeError('Separate code and output deployment required')
    seal=read(parent/'delivery-files.json');parent_manifest=read(parent/'manifest.json')
    if read(parent/'status.json')['state']!='server-complete' or read(parent/'audit.json')['state']!='passed':
        raise RuntimeError('Parent is not complete and audited')
    registry=read(parent/'registry.json')
    if len(registry)!=66:raise RuntimeError('Expected sealed 66-result parent')
    record=registry[f'{CITY}/{SEED}/full'];source=record['source_shards'][0]
    if sha256_file(source['path'])!=source['sha256']:raise RuntimeError('Original Full shard changed')
    shard=torch.load(source['path'],map_location='cpu',weights_only=False);old_job=shard['job']
    profile=parent_manifest['profiles'][CITY]
    if old_job['city']!=profile or old_job['seed']!=SEED or old_job['variant']!='full' or old_job['split']!='test':
        raise RuntimeError('Original sampling profile differs')
    original_cache=Path(old_job['cache'])
    if sha256_file(original_cache)!=old_job['cache_sha256']:raise RuntimeError('Original time cache changed')
    cache=torch.load(original_cache,map_location='cpu',weights_only=False)
    worker.check_cache(cache,old_job)
    indices=list(range(128));derived,off_records,off_traces=subset_inputs(cache,shard,indices)
    inputs={str(parent/'manifest.json'):seal['manifest_sha256'],str(parent/'delivery-files.json'):sha256_file(parent/'delivery-files.json'),
        source['path']:source['sha256'],str(original_cache):old_job['cache_sha256'],
        profile['checkpoint']:profile['checkpoint_sha256'],profile['config_path']:profile['config_sha256']}
    for split in ('train','test'):
        inputs[str(Path(profile['data_root'])/CITY/f'{CITY}_{split}.pkl')]=profile['data_sha256'][split]
    def verify_inputs():
        for path,expected in inputs.items():
            if sha256_file(path)!=expected:raise RuntimeError('Read-only input changed: '+path)
        for name,expected in seal['files'].items():
            if sha256_file(checked_path(parent,name))!=expected:raise RuntimeError('Original artifact changed: '+name)
    verify_inputs()
    manifest=dict(revision='ny-category-sampled-v2',run_id=args.run_id,seed=SEED,indices=indices,batch_size=64,
        source_run=str(parent),source_inputs=inputs,source_sha256=fingerprint(),
        category_consistent_decoding_version=CATEGORY_DECODING_VERSION,distance_backend=record['distance_backend'],
        projection_settings=shard['result']['projection_settings'],geometry_refinement='off',
        no_training=True,baseline_reused=True,precision='FP32; TF32 disabled',memory_limit=.8,
        interpretation='Single-seed 128-row test-side diagnostic; not tuning or a full three-seed experiment.')
    if (out/'manifest.json').exists():
        if not args.resume or read(out/'manifest.json')!=manifest:raise RuntimeError('Resume requires identical manifest')
    else:atomic_json(out/'manifest.json',manifest)
    manifest_sha=sha256_file(out/'manifest.json')
    raw=torch.load(Path(profile['data_root'])/CITY/f'{CITY}_test.pkl',map_location='cpu',weights_only=False)
    refs=[raw['sequences'][i] for i in indices];mapping=raw['poi_category']
    derived['derivation']=dict(manifest_sha256=manifest_sha,source_cache=str(original_cache),source_sha256=old_job['cache_sha256'],indices=indices)
    cache_path=out/'derived-time-cache.pkl'
    if not cache_path.exists():
        publish_torch(cache_path,derived)
        atomic_json(out/'derived-time-cache.receipt.json',dict(manifest_sha256=manifest_sha,sha256=sha256_file(cache_path)))
    if read(out/'derived-time-cache.receipt.json')!=dict(manifest_sha256=manifest_sha,sha256=sha256_file(cache_path)):
        raise RuntimeError('Derived subset cache identity changed')
    off_metrics,off_conditions=evaluate(refs,off_records,mapping,indices)
    if not (out/'off.pkl').exists():
        publish_torch(out/'off.pkl',dict(sequences=off_records,test_indices=indices,batch_traces=off_traces,
            metrics=off_metrics,per_condition=off_conditions,origin='subset of sealed original Full; not resampled',source=source))
        atomic_json(out/'off.receipt.json',dict(manifest_sha256=manifest_sha,sha256=sha256_file(out/'off.pkl')))
    if read(out/'off.receipt.json')!=dict(manifest_sha256=manifest_sha,sha256=sha256_file(out/'off.pkl')):
        raise RuntimeError('Published off subset changed')
    job=copy.deepcopy(old_job)
    job.update(indices=indices,global_start=0,output_dir=str(out/'on'),manifest_sha256=manifest_sha,
        cache=str(cache_path),cache_sha256=sha256_file(cache_path),cache_manifest_sha256=cache['manifest_sha256'],
        category_consistent_decoding=True,category_consistent_decoding_version=CATEGORY_DECODING_VERSION,
        distance_backend=record['distance_backend'],distance_implementation_version=record['distance_implementation_version'])
    atomic_json(out/'on-job.json',job)
    atomic_json(out/'status.json',dict(state='running',phase='on-sampling',pid=os.getpid()))
    path=out/'on'/'payload.pkl'
    if path.exists():
        if read(path.parent/'receipt.json')!=dict(manifest_sha256=manifest_sha,sha256=sha256_file(path)):
            raise RuntimeError('Published on output changed')
        payload=torch.load(path,map_location='cpu',weights_only=False)
        if payload['job']!=job:raise RuntimeError('Existing on output belongs to another job')
        result=payload['result'];worker.validate_category_decoding_result(job,result)
    else:
        original_task=worker.original_task;models=[]
        def checked_task(value):
            task,dm=original_task(value);models.append((task,state_hash(task)));return task,dm
        worker.original_task=checked_task
        try:payload,result=worker.sample(job)
        finally:worker.original_task=original_task
        if any(state_hash(task)!=before for task,before in models):raise RuntimeError('Model tensor state changed')
        publish_torch(path,payload)
        atomic_json(path.parent/'receipt.json',dict(manifest_sha256=manifest_sha,sha256=sha256_file(path)))
    compare_pair(off_records,payload['sequences'],off_traces,payload['batch_traces'])
    if result['projection_settings']!=manifest['projection_settings']:raise RuntimeError('Projection budget/settings changed')
    if result['metrics']['category_poi_mismatch_rate']!=0:raise RuntimeError('Actual category mismatch remains')
    comparison=dict(off=off_metrics,on=result['metrics'],seed=SEED,samples=128,
        off_diagnostics=distribution_counts(refs,off_records,mapping),
        on_diagnostics=distribution_counts(refs,payload['sequences'],mapping),
        on_spatial_seconds=result['spatial_seconds'],off_spatial_seconds=None,
        timing_note='Off output was reused; no paired complete timing ratio is claimed.',
        paired_rng=True,model_weights_unchanged=True)
    atomic_json(out/'comparison.json',comparison)
    lines=['# NewYork category-consistent decoding: 128-row diagnostic','',manifest['interpretation'],'',
        'Off: original sealed Full subset. On: sampled-category-poi-v2. Same time inputs/seed/legacy backend; no Geo refinement.',
        'Category/POI consistency is checked against the actual generated category tokens, not the logit argmax.','',
        '| Metric (raw units) | Off | On |','|---|---:|---:|']
    for key in ('category_poi_mismatch_rate','token_strict_ovr','strict_ovr','pair_coverage','category_coverage',
                'Distance','Radius','DailyLoc','G-RANK','empty_rate'):
        show=lambda v:'undefined' if v is None else f'{v:.6f}'
        lines.append(f'| {key} | {show(off_metrics[key])} | {show(result["metrics"][key])} |')
    lines += ['', 'Distance endpoint-selected sets may change; inspect both counts and unfiltered Distance in comparison.json.',
              comparison['timing_note'],f'On spatial sampling: {result["spatial_seconds"]:.3f} seconds.',
              'No new hyperparameter search, no base retraining, and no claims of three-seed significance.']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    verify_inputs()
    if fingerprint()!=manifest['source_sha256']:raise RuntimeError('Running code changed')
    files={p.relative_to(out).as_posix():sha256_file(p) for p in out.rglob('*')
           if p.is_file() and p.name not in ('audit.json','status.json','pipeline.lock')}
    atomic_json(out/'audit.json',dict(state='passed',new_samples=128,baseline_samples=128,originals_unchanged=True,
        paired_rng=True,model_weights_unchanged=True,files=files))
    atomic_json(out/'status.json',dict(state='complete',phase='complete',pid=os.getpid(),new_samples=128,baseline_reused=True))
    print(json.dumps(verify_delivery(out)),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run')
    parser.add_argument('--run-id',default='ny-category-sampled-v2-20261010')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--verify-only',type=Path,help='Read-only verification of a downloaded directory')
    args=parser.parse_args()
    if args.verify_only:
        print(json.dumps(verify_delivery(args.verify_only)));return
    if not args.source_run:parser.error('--source-run is required')
    out=ROOT/'experiment_runs'/safe_tag(args.run_id)
    if out.exists() and not args.resume:raise RuntimeError('Existing diagnostic directory; no overwrite')
    out.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (out/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:execute(args,out)
        except BaseException as exc:
            atomic_json(out/'status.json',dict(state='failed',pid=os.getpid(),error_type=type(exc).__name__,error=str(exc)))
            raise


if __name__=='__main__':main()
