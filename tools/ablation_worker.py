"""Atomic time-cache creation and spatial-only paired ablation sampling."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from datamodule import Batch
from evaluate_utils import get_task,get_run_data
from experiment_io import atomic_json,publish_torch,sha256_file,seed_sampling,decode_preserving_empty
from tools.baseline_common import precision,references,validate_alignment
from tools.ablation_common import VERSION,VARIANTS,DISTANCE_VERSION,DISTANCE_VARIANTS,STEPS,generator,rng_digest,projector,evaluate
from distance_kl import validate_distance_metadata


def validate_geometry_result(job, result):
    configuration = job.get('geometry_config', {})
    if configuration.get('geometry_refinement', 'off') == 'off':
        if result.get('geometry_refinement', 'off') != 'off':
            raise RuntimeError('Geometry result cannot be resumed as an unrefined job')
        return
    from geometry_projection import GeometryConfig
    expected = GeometryConfig(**configuration).metadata()
    if any(result.get(k) != v for k, v in expected.items()):
        raise RuntimeError('Geometry result implementation/configuration changed')
    reference = job.get('geometry_reference_sha256')
    if not reference or result.get('geometry_reference_sha256') != reference:
        raise RuntimeError('Geometry result reference changed')


def memory_guard(job):
    """An explicit safety ceiling; never silently reduce batch size or precision."""
    limit = job.get('max_gpu_memory_fraction')
    if limit is not None:
        total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
        used = torch.cuda.max_memory_reserved()
        if used > total * limit:
            raise RuntimeError(f'GPU reserved-memory safety ceiling exceeded: {used}/{total} > {limit}')


def original_task(job):
    if job.get('city'):
        from tools.city_common import load_city
        task,dm=load_city(job['city'],job.get('cfg_checkpoint'))
        if task.device.type!='cuda':
            raise RuntimeError('CUDA required for the declared experiment')
        return task,dm
    source=Path(job['source'])
    manifest=json.loads((source/'manifest.json').read_text())
    _,_,run_path=get_run_data(manifest['run_id'],ROOT/'wandb')
    task,dm=get_task(run_path,str(ROOT),job['checkpoint'])
    if getattr(task.discrete_diffusion,'cfg_trained',False):
        raise RuntimeError('PCDG ablations must not use CFG weights')
    if task.device.type!='cuda':
        raise RuntimeError('CUDA required for the declared experiment')
    return task,dm


def job_references(job,dm):
    if job.get('city'):
        from tools.city_common import references as city_references
        return city_references(dm,job['city'],job['split'],job['indices'])
    return references(dm,job['split'],job['indices'])


def check_cache(cache,job):
    expected=job['indices']
    origin=job.get('cache_manifest_sha256',job['manifest_sha256'])
    if cache['indices']!=expected or cache['manifest_sha256']!=origin:
        raise RuntimeError('Time cache indices/protocol mismatch')
    if job.get('city'):
        source_job=cache['job']
        if (source_job['checkpoint_sha256']!=job['checkpoint_sha256'] or
                source_job['seed']!=job['seed'] or source_job['split']!=job['split']):
            raise RuntimeError('Foreign cache has a different checkpoint/seed/split')
    indices=[]
    for item in cache['batches']:
        batch=item['batch']
        indices+=item['indices']
        if batch.batch_size!=len(item['indices']) or batch.time.device.type!='cpu':
            raise RuntimeError('Cache batch shape/device mismatch')
        if not torch.isfinite(batch.time).all() or not torch.equal(batch.mask.sum(1),batch.unpadded_length):
            raise RuntimeError('Invalid cached time/mask')
        if job.get('city'):
            p=job['city']; semantic=p['semantic_classes']; slots=p['model_category_slots']
            if batch.po_matrix.shape!=(batch.batch_size,slots,slots):
                raise RuntimeError('Cached constraint matrix has wrong model-slot dimensions')
            if batch.po_matrix[:,semantic:,:].any() or batch.po_matrix[:,:,semantic:].any():
                raise RuntimeError('Unused legacy slots must never receive constraint edges')
        for i in range(batch.batch_size):
            t=batch.time[i][batch.mask[i]]
            if ((t<0)|(t>=24)).any() or (t.diff()<=0).any():
                raise RuntimeError('Invalid cached event times')
    if indices!=expected:
        raise RuntimeError('Missing/duplicate cache batch indices')


def create_cache(job):
    started=time.perf_counter()
    if job.get('fixture'):
        payload=torch.load(job['fixture'],map_location='cpu',weights_only=False)
        if payload['inputs']['checkpoint_sha256']!=sha256_file(job['checkpoint']):
            raise RuntimeError('Historical fixture used different weights')
        batch=payload['batch']
        if job['indices']!=list(range(64,128)) or int(batch.unpadded_length[35])!=0:
            raise RuntimeError('Not the validated empty index-99 fixture')
        items=[dict(indices=job['indices'],global_start=64,batch=batch)]
        adaptation=[]
    else:
        task,dm=original_task(job)
        raw,selected,_=job_references(job,dm)
        items=[]
        adaptation=[]
        generation_started=time.perf_counter()
        for offset in range(0,len(selected),64):
            global_start=job['global_start']+offset
            # Reset randomness but retain the temporal sampler's adaptive state.
            seed_sampling(job['seed']+global_start)
            before=float(task.tpp_model.intensity_model.rejections_sample_multiple)
            batch=Batch.from_sequence_list(selected[offset:offset+64]).to(task.device)
            with torch.no_grad():
                temporal=task.tpp_model.sample(batch.batch_size,x_n=batch,tmax=batch.tmax)
            memory_guard(job)
            maximum=min(task.discrete_diffusion.condition_encoder.max_position_embeddings,
                        (task.discrete_diffusion.transformer.positional_encoding.num_embeddings-3)//2)
            if int(temporal.unpadded_length.max())>maximum:
                raise RuntimeError('Temporal length exceeds unchanged model capacity')
            temporal=temporal.mask_check()
            temporal.po_matrix=batch.po_matrix
            items.append(dict(indices=job['indices'][offset:offset+64],global_start=global_start,
                              batch=temporal.to('cpu')))
            adaptation.append(dict(global_start=global_start,before=before,
                after=float(task.tpp_model.intensity_model.rejections_sample_multiple)))
            atomic_json(Path(job['output_dir'])/'progress.json',dict(completed_samples=offset+batch.batch_size,
                       total_samples=len(selected),phase='time-cache'))
        torch.cuda.synchronize()
        generation_seconds=time.perf_counter()-generation_started
    result=dict(state='complete',kind='cache',samples=len(job['indices']),
                wall_seconds=time.perf_counter()-started,
                temporal_generation_seconds=generation_seconds if not job.get('fixture') else 0.0)
    cache=dict(format=VERSION,job=job,manifest_sha256=job['manifest_sha256'],indices=job['indices'],
               batches=items,adaptive_states=adaptation,result=result)
    check_cache(cache,job)
    return cache,result


def sample(job):
    cache_path=Path(job['cache'])
    if sha256_file(cache_path)!=job['cache_sha256']:
        raise RuntimeError('Time cache fingerprint changed')
    cache=torch.load(cache_path,map_location='cpu',weights_only=False)
    check_cache(cache,job)
    task,dm=original_task(job)
    raw,_,refs=job_references(job,dm)
    def forbidden(*args,**kwargs):
        raise RuntimeError('Temporal resampling is forbidden in spatial ablations')
    task.tpp_model.sample=forbidden
    dd=task.discrete_diffusion
    revision = job.get('projection_revision', VERSION)
    variants = DISTANCE_VARIANTS if revision == DISTANCE_VERSION else VARIANTS
    dd.use_guidance_baseline=False
    dd.use_constraint_projection=job['variant'] in variants and job['variant']!='no_projection'
    dd.projection_frequency=4
    dd.projection_last_k_steps=40
    dd.projection_call_count=0
    implementation = validate_distance_metadata(job, allow_historical=True)
    p=projector(dd,job['variant'],revision=revision,datamodule=dm,
                distance_backend=implementation['distance_backend']) if job['variant'] in variants else None
    dd.constraint_projector=p
    geometry_on = job.get('geometry_config', {}).get('geometry_refinement', 'off') != 'off'
    if geometry_on:
        from geometry_projection import GeometryConfig, load_geometry_reference
        dd.geometry_config = GeometryConfig(**job['geometry_config'])
        dd.geometry_reference = load_geometry_reference(Path(dm.root)/dm.name/f'{dm.name}_train.pkl', job['geometry_fit_indices'])
        if dd.geometry_reference.fingerprint != job['geometry_reference_sha256'] or p is None:
            raise RuntimeError('Geometry reference/projection configuration mismatch')
    else:
        dd.geometry_config = None
    if job.get('distance_reference_sha256') and p is not None and p.projection_distance_kl_weight:
        if p.distance_reference.fingerprint != job['distance_reference_sha256']:
            raise RuntimeError('Distance training reference changed')
    if job['variant']=='energy':
        from baseline_models import configure_energy
        configure_energy(dd,temperature=3,scale=100,last_k=40,frequency=4)
    elif job['variant']=='cfg':
        if not getattr(dd,'cfg_trained',False):
            raise RuntimeError('PO-CFG requires a validated trained checkpoint')
        dd.cfg_scale=1.0; dd.reset_cfg_stats()
    elif job['variant'] not in variants:
        raise ValueError('Unknown sampling variant')
    calls=[]
    current={}
    old_sample=dd.p_sample
    def step(*args,**kwargs):
        current['step']=kwargs.get('diffusion_index')
        return old_sample(*args,**kwargs)
    dd.p_sample=step
    if p is not None:
        old_project=p.project_with_matrices
        def traced(*args,**kwargs):
            t=time.perf_counter()
            output=old_project(*args,**kwargs)
            memory_guard(job)
            calls.append(dict(p.last_projection_stats,diffusion_step=current['step'],
                              projection_wall_seconds=time.perf_counter()-t))
            return output
        p.project_with_matrices=traced
    generated=[]
    batch_traces=[]
    temporal_empty=[]
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter()
    for item in cache['batches']:
        batch=copy.deepcopy(item['batch']).to(task.device)
        spatial=generator(job['seed'],item['global_start'],'spatial',task.device)
        projection=generator(job['seed'],item['global_start'],'projection',task.device)
        dd.sampling_generator=spatial
        if p is not None:
            p.generator=projection
            if p.projection_distance_kl_weight > 0:
                from distance_kl import distance_generator
                p.distance_generator = distance_generator(job['seed'] + item['global_start'], task.device)
        distance_rng_before = rng_digest(p.distance_generator) if p is not None and p.distance_generator is not None else None
        before_spatial=rng_digest(spatial)
        before_global=(rng_digest(torch.get_rng_state()),rng_digest(torch.cuda.get_rng_state()))
        begin=len(calls)
        dd.geometry_seed=job['seed']
        dd.geometry_global_start=item['global_start']
        dd.last_geometry_stats=None
        energy_before=dd.energy_guidance.stats['calls'] if job['variant']=='energy' else 0
        cfg_before=dd.cfg_stats['calls'] if job['variant']=='cfg' else 0
        output=decode_preserving_empty(task,batch,raw['poi_gps'])
        memory_guard(job)
        after_global=(rng_digest(torch.get_rng_state()),rng_digest(torch.cuda.get_rng_state()))
        if before_global!=after_global:
            raise RuntimeError('Spatial/projection sampling leaked into global RNG')
        trace=calls[begin:]
        effective=bool(((batch.unpadded_length>0)&batch.po_matrix.bool().flatten(1).any(1)).any())
        active=[c for c in trace if c['optimizer_steps']>0]
        expected=STEPS if effective and p is not None else []
        if [c['diffusion_step'] for c in active]!=expected or any(c['optimizer_steps']!=500 for c in active):
            raise RuntimeError('Ablation did not execute the fixed 10 x 500 budget')
        if job['variant']=='energy' and dd.energy_guidance.stats['calls']-energy_before!=(10 if effective else 0):
            raise RuntimeError('Energy guidance did not execute its fixed schedule')
        if job['variant']=='cfg' and dd.cfg_stats['calls']-cfg_before!=(dd.num_timesteps if bool((batch.unpadded_length>0).any()) else 0):
            raise RuntimeError('PO-CFG did not execute both prediction branches')
        temporal_empty.extend(index for index,n in zip(item['indices'],batch.unpadded_length.tolist()) if n==0)
        generated.extend(output)
        batch_traces.append(dict(global_start=item['global_start'],indices=item['indices'],
            spatial_rng_before=before_spatial,spatial_rng_after=rng_digest(spatial),
            distance_rng_before=distance_rng_before,
            distance_rng_after=rng_digest(p.distance_generator) if distance_rng_before else None,
            projection_rng_after=rng_digest(projection),effective_constraints=effective,
            projection_stats=trace,global_rng_unchanged=True))
        if geometry_on:
            batch_traces[-1]['geometry_projection_stats'] = dd.last_geometry_stats
        atomic_json(Path(job['output_dir'])/'progress.json',dict(completed_samples=len(generated),
            total_samples=len(job['indices']),phase='spatial',elapsed_seconds=time.perf_counter()-started))
    torch.cuda.synchronize()
    spatial_seconds=time.perf_counter()-started
    metric_diagnostics = {}
    metrics,rows=evaluate(refs,generated,raw['poi_category'],job['indices'],diagnostics=metric_diagnostics)
    if job.get('historical_empty') and len(generated[35]['checkins'])!=0:
        raise RuntimeError('Historical empty index 99 was filled')
    if job.get('test_empty_row') is not None and len(generated[job['test_empty_row']]['checkins'])!=0:
        raise RuntimeError('Test-only fixed empty row was filled')
    if sha256_file(cache_path)!=job['cache_sha256']:
        raise RuntimeError('Spatial worker modified the sealed time cache')
    active=[c for c in calls if c['optimizer_steps']>0]
    probes=sum(c.get('unmet_probe_count',0) for c in calls)
    active_rows=sum(c.get('active_rows',0) for c in active)
    elements=sum(c.get('logit_element_count',0) for c in calls)
    result=dict(state='complete',kind='sample',samples=len(generated),metrics=metrics,
        metric_diagnostics=metric_diagnostics, projection_revision=revision, **implementation,
        spatial_seconds=spatial_seconds,samples_per_second=len(generated)/spatial_seconds,
        projection_calls=len(active),projection_invocations=len(calls),
        optimizer_steps=sum(c['optimizer_steps'] for c in calls),
        projection_seconds=sum(c['projection_wall_seconds'] for c in calls),
        zero_gradient_probes=sum(c.get('zero_gradient_probe_count',0) for c in calls),gradient_probes=probes,
        zero_gradient_ratio=sum(c.get('zero_gradient_probe_count',0) for c in calls)/probes if probes else None,
        kl_model_to_projected_per_active_row=sum(c.get('kl_model_to_projected_sum',0) for c in calls)/active_rows if active_rows else 0.0,
        logit_rms_change=(sum(c.get('logit_squared_change_sum',0) for c in calls)/elements)**.5 if elements else 0.0,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
    result['projection_settings'] = None if p is None else dict(
        token_kl_weight=p.projection_kl_weight, distance_kl_weight=p.projection_distance_kl_weight,
        order_weight=p.projection_order_weight, existence_weight=p.projection_existence_weight,
        gumbel=p.use_gumbel_softmax, update_multipliers=p.update_multipliers,
        distance_paths=p.distance_paths, distance_topk=p.distance_topk,
        distance_bins=p.distance_bins, distance_temperature=p.distance_temperature,
        outer=p.outer_iterations, inner=p.inner_iterations, early_stop=p.early_stop)
    if geometry_on:
        result.update(dd.geometry_config.metadata(), geometry_reference_sha256=dd.geometry_reference.fingerprint,
                      geometry_optimizer_steps=sum((b.get('geometry_projection_stats') or {}).get('optimizer_steps',0) for b in batch_traces))
    if job['variant']=='energy':
        result['energy_stats']=dict(dd.energy_guidance.stats)
    if job['variant']=='cfg':
        result['cfg_stats']=dict(dd.cfg_stats)
    if job['variant'] in ('energy','cfg') and dd.projection_call_count!=0:
        raise RuntimeError('Baseline unexpectedly used ALM')
    payload=dict(format=VERSION,job=job,manifest_sha256=job['manifest_sha256'],indices=job['indices'],
                 test_indices=job['indices'] if job['split']=='test' else None,t_max=24.0,
                 sequences=generated,per_condition=rows,batch_traces=batch_traces,
                 temporal_empty_indices=temporal_empty,result=result)
    return payload,result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--job',required=True)
    args=parser.parse_args()
    os.chdir(ROOT)
    job=json.loads(Path(args.job).read_text())
    implementation = validate_distance_metadata(job, allow_historical=True)
    folder=Path(job['output_dir'])
    precision()
    if sha256_file(job['checkpoint'])!=job['checkpoint_sha256']:
        raise RuntimeError('Checkpoint changed')
    if job.get('cfg_checkpoint') and sha256_file(job['cfg_checkpoint'])!=job['cfg_checkpoint_sha256']:
        raise RuntimeError('CFG checkpoint changed')
    atomic_json(folder/'status.json',dict(state='running',pid=os.getpid()))
    try:
        path=folder/'payload.pkl'
        if path.exists():
            payload=torch.load(path,map_location='cpu',weights_only=False)
            if payload['job']!=job:
                raise RuntimeError('Published artifact belongs to another job')
            result=payload['result']
            if job['kind'] != 'cache':
                validate_distance_metadata(result, expected=implementation)
                validate_geometry_result(job, result)
            if result['state']!='complete' or payload['indices']!=job['indices']:
                raise RuntimeError('Published artifact is not a complete matching job')
            if job['kind']=='cache':
                check_cache(payload,job)
            elif len(payload['sequences'])!=len(job['indices']):
                raise RuntimeError('Published sample count mismatch')
        else:
            payload,result=create_cache(job) if job['kind']=='cache' else sample(job)
            publish_torch(path,payload)
        result=dict(result,output_sha256=sha256_file(path))
        atomic_json(folder/'result.json',result)
        atomic_json(folder/'status.json',dict(state='complete'))
        print(json.dumps({k:result[k] for k in ('state','kind','samples')}),flush=True)
    except BaseException as exc:
        atomic_json(folder/'status.json',dict(state='failed',error_type=type(exc).__name__,error=str(exc)))
        raise


if __name__=='__main__':
    main()
