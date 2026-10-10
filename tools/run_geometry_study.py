"""Bounded PCDG-Geo study: speed gate, train-only calibration, then sealed refinement.

Use a new deployment/output directory. This never trains or edits its parent run.
Every completed base batch is immutable and reused; online timing recomputes all
per-batch geometry work and excludes temporal generation/loading/network time.
"""
import argparse
import copy
from dataclasses import asdict, replace
import gc
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '4')
import numpy as np
import torch
from datamodule import Batch
from distance_kl import distance_generator
from experiment_io import atomic_json, decode_preserving_empty, publish_torch, safe_tag, seed_sampling, sha256_file
from geometry_projection import (GeometryConfig, VERSION as GEOMETRY_VERSION, assert_invariants,
    load_geometry_reference, refine_records, untruncated_scores)
from tools.ablation_common import DISTANCE_VERSION, evaluate, generator, projector, rng_digest, same_records
from tools.baseline_common import precision, state_hash
from tools.city_common import load_city, references as city_references
from tools.geometry_study_common import (CITIES, REVISION, SEEDS, SPEED_CEILING, STEP_CHOICES,
    check_invariant_metrics, choose_steps, configurations, config_label, content_hash,
    geometric_diagnostics, metric_means, quality_failures, rank_candidates, split_indices, target_metrics)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


class GeometryStudy:
    def __init__(self, args):
        self.args = args
        self.out = ROOT / 'experiment_runs' / safe_tag(args.run_id)
        self.parent = Path(args.source_run).resolve()
        if self.out.resolve().is_relative_to(self.parent) or ROOT.resolve() == self.parent.parent.parent:
            raise RuntimeError('Use a separate code/output deployment, never the sealed source checkout')
        self.out.mkdir(parents=True, exist_ok=True)
        self.phase = 'precheck'
        self.task = self.dm = self.loaded_city = None
        self.score_cache = {}
        self.cold_reference_seconds = {}
        self.cold_model_records = []
        self.device = torch.device(args.device)

    def status(self, state='running', **fields):
        value = dict(state=state, phase=self.phase, pid=os.getpid(), updated_at=time.time(), **fields)
        atomic_json(self.out/'status.json', value)
        print(json.dumps(value), flush=True)

    def fingerprint(self):
        paths = list(ROOT.glob('*.py'))
        for directory in ('tools','tests','config','add_thin','discrete_diffusion','evaluations'):
            paths += [p for p in (ROOT/directory).rglob('*') if p.is_file() and p.suffix in ('.py','.yaml','.yml')]
        return {p.relative_to(ROOT).as_posix():sha256_file(p) for p in sorted(paths)}

    def precheck(self):
        precision()
        if self.device.type != 'cuda' or not torch.cuda.is_available():
            raise RuntimeError('Full study requires CUDA; no CPU/smaller-batch substitution')
        self.parent_manifest = read(self.parent/'manifest.json')
        self.parent_registry = read(self.parent/'registry.json')
        audit = read(self.parent/'audit.json')
        state = read(self.parent/'status.json')
        if state['state'] != 'server-complete' or audit['state'] != 'passed' or len(self.parent_registry) != 66:
            raise RuntimeError('Parent must be a completed audited 66-result study')
        self.parent_seal = read(self.parent/'delivery-files.json')
        if sha256_file(self.parent/'manifest.json') != self.parent_seal['manifest_sha256']:
            raise RuntimeError('Parent manifest does not match its seal')
        self.verify_parent()
        self.profiles = self.parent_manifest['profiles']
        self.splits = {city:split_indices(self.profiles[city]['train_count']) for city in CITIES}
        self.refs = {}
        self.train_raw = {}
        for city in CITIES:
            profile = self.profiles[city]
            for path, digest in ((profile['checkpoint'],profile['checkpoint_sha256']),
                                 (profile['config_path'],profile['config_sha256'])):
                if sha256_file(path) != digest:raise RuntimeError('Frozen model/config changed')
            path = Path(profile['data_root'])/city/f'{city}_train.pkl'
            if sha256_file(path) != profile['data_sha256']['train']:raise RuntimeError('Training bytes changed')
            started=time.perf_counter()
            self.refs[city]=load_geometry_reference(path,self.splits[city]['reference'])
            self.train_raw[city]=torch.load(path,map_location='cpu',weights_only=False)
            self.cold_reference_seconds[city]=time.perf_counter()-started
            print(json.dumps(dict(cold_reference_city=city,seconds=time.perf_counter()-started)),flush=True)
        manifest = dict(revision=REVISION,geometry_implementation_version=GEOMETRY_VERSION,
            run_id=self.args.run_id,parent_run=str(self.parent),parent_manifest_sha256=sha256_file(self.parent/'manifest.json'),
            parent_seal_sha256=sha256_file(self.parent/'delivery-files.json'),profiles=self.profiles,
            source_sha256=self.fingerprint(),splits=self.splits,split_seed=20261010,
            geometry_references={c:self.refs[c].fingerprint for c in CITIES},seeds=list(SEEDS),
            quality_configs=[dict(radius_weight=c.geometry_radius_weight,prior_weight=c.geometry_prior_weight)
                             for c in configurations(50)],step_choices=list(STEP_CHOICES),
            batch_size=64,outer=10,inner=50,projection_calls=10,base_backend='batched',
            warmups=2,repeats=5,speed_ratio_ceiling=SPEED_CEILING,device=str(self.device),
            geometry_probe=dict(radius_weight=1.,prior_weight=.01),
            calibration='train-side engineering calibration; base model saw training records',
            test_policy='One frozen-configuration test evaluation; no test-driven search',
            constraints_tolerance=.01,other_jsd_relative_tolerance=.05,
            planned_new_formal_results=18,no_training=True,precision='FP32; TF32 disabled',memory_limit=.8)
        path=self.out/'manifest.json'
        if path.exists():
            if not self.args.resume or read(path)!=manifest:raise RuntimeError('Resume requires identical manifest/code and --resume')
        elif self.args.resume:
            raise RuntimeError('Cannot resume without a manifest')
        else:
            if any(p.name!='pipeline.lock' for p in self.out.iterdir()):raise RuntimeError('Output directory is not empty')
            atomic_json(path,manifest)
        self.manifest=manifest;self.manifest_sha=sha256_file(path)
        atomic_json(self.out/'cold-start.json',dict(reference_seconds=self.cold_reference_seconds,model_loads=[]))
        self.status()

    def verify_parent(self):
        for name,digest in self.parent_seal['files'].items():
            path=self.parent/name
            if not path.resolve().is_relative_to(self.parent) or path.is_symlink() or sha256_file(path)!=digest:
                raise RuntimeError(f'Sealed parent artifact changed: {name}')

    def identity(self, **fields):
        return dict(study_manifest_sha256=self.manifest_sha,**fields)

    def load_payload(self,path,identity):
        path=Path(path)
        if not path.exists():return None
        receipt=path.with_suffix('.receipt.json')
        if not receipt.exists() or read(receipt)!=dict(identity=identity,sha256=sha256_file(path)):
            raise RuntimeError(f'Payload/receipt identity mismatch; retain artifact for inspection: {path}')
        payload=torch.load(path,map_location='cpu',weights_only=False)
        if payload.get('identity')!=identity:raise RuntimeError('Foreign payload identity')
        return payload

    def save_payload(self,path,identity,**payload):
        value=dict(identity=identity,**payload)
        publish_torch(path,value)
        atomic_json(Path(path).with_suffix('.receipt.json'),dict(identity=identity,sha256=sha256_file(path)))
        return value

    def load_model(self,city,force=False):
        if self.loaded_city==city and not force:return self.task,self.dm
        self.finish_model()
        started=time.perf_counter()
        self.task,self.dm=load_city(self.profiles[city],device=str(self.device))
        self.loaded_city=city
        self.model_hash=state_hash(self.task)
        self.refs[city].on_device(self.device)
        torch.cuda.synchronize(self.device)
        self.cold_model_records.append(dict(city=city,seconds=time.perf_counter()-started))
        atomic_json(self.out/'cold-start.json',dict(reference_seconds=self.cold_reference_seconds,model_loads=self.cold_model_records))
        print(json.dumps(dict(cold_model_city=city,seconds=time.perf_counter()-started)),flush=True)
        return self.task,self.dm

    def finish_model(self):
        if self.task is not None:
            if state_hash(self.task)!=self.model_hash:raise RuntimeError('Frozen model state was modified')
            self.task=self.dm=None;self.loaded_city=None;self.score_cache.clear()
            gc.collect()
            if torch.cuda.is_available():torch.cuda.empty_cache()

    def memory_guard(self):
        peak=torch.cuda.max_memory_reserved(self.device)
        if peak>torch.cuda.get_device_properties(self.device).total_memory*.8:
            raise RuntimeError('GPU memory ceiling exceeded; no batch reduction is permitted')

    def configure_sampler(self,city,variant,seed,start,geometry=None):
        task,dm=self.load_model(city)
        dd=task.discrete_diffusion
        dd.use_guidance_baseline=False
        dd.use_constraint_projection=variant=='full'
        dd.projection_frequency=4;dd.projection_last_k_steps=40;dd.projection_call_count=0
        spatial=generator(seed,start,'spatial',self.device)
        projection=generator(seed,start,'projection',self.device)
        dd.sampling_generator=spatial
        p=projector(dd,'full',projection,revision=DISTANCE_VERSION,datamodule=dm,distance_backend='batched') if variant=='full' else None
        dd.constraint_projector=p
        calls=[]
        if p is not None:
            p.distance_generator=distance_generator(seed+start,self.device)
            original=p.project_with_matrices
            def traced(*args,**kwargs):
                value=original(*args,**kwargs)
                calls.append(dict(p.last_projection_stats))
                return value
            p.project_with_matrices=traced
        dd.geometry_config=geometry
        dd.geometry_reference=self.refs[city]
        dd.geometry_seed=seed;dd.geometry_global_start=start;dd.last_geometry_stats=None
        return task,dd,calls,spatial,projection

    def spatial_sample(self,city,item,variant,seed,geometry=None):
        task,dd,calls,spatial,projection=self.configure_sampler(city,variant,seed,item['global_start'],geometry)
        batch=copy.deepcopy(item['temporal']).to(self.device)
        global_before=(rng_digest(torch.get_rng_state()),rng_digest(torch.cuda.get_rng_state(self.device)))
        torch.cuda.synchronize(self.device);torch.cuda.reset_peak_memory_stats(self.device)
        started=time.perf_counter()
        records=decode_preserving_empty(task,batch,self.train_raw[city]['poi_gps'])
        torch.cuda.synchronize(self.device)
        seconds=time.perf_counter()-started
        self.memory_guard()
        after=(rng_digest(torch.get_rng_state()),rng_digest(torch.cuda.get_rng_state(self.device)))
        if global_before!=after:raise RuntimeError('Sampling/refinement changed global RNG')
        active=any(c['optimizer_steps']>0 for c in calls)
        if variant=='full' and active and (len(calls)!=10 or any(c['optimizer_steps']!=500 for c in calls)):
            raise RuntimeError('Base PCDG projection budget changed')
        stats=dict(spatial_seconds=seconds,projection_executed=active,
            projection_calls=sum(c['optimizer_steps']>0 for c in calls),optimizer_steps=sum(c['optimizer_steps'] for c in calls),
            spatial_rng_after=rng_digest(spatial),projection_rng_after=rng_digest(projection),
            distance_rng_after=rng_digest(dd.constraint_projector.distance_generator) if dd.constraint_projector else None,
            global_rng_unchanged=True,geometry=dd.last_geometry_stats,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device),peak_reserved_bytes=torch.cuda.max_memory_reserved(self.device))
        return records,stats

    def prepare_pool(self,city,pool,seed):
        indices=self.splits[city][pool]
        task,dm=self.load_model(city,force=True)
        raw,selected,_=city_references(dm,self.profiles[city],'train',indices)
        packets=[]
        offset_base=0 if pool=='screen' else 512
        for offset in range(0,len(indices),64):
            self.phase=f'prepare-{city}-{pool}-{seed}-{offset}';self.status()
            path=self.out/'calibration'/city/pool/f'seed-{seed}'/f'batch-{offset}.pkl'
            identity=self.identity(kind='base-cache',city=city,pool=pool,seed=seed,indices=indices[offset:offset+64],global_start=offset_base+offset)
            saved=self.load_payload(path,identity)
            if saved is not None:
                if int(saved['adaptive_after']) != saved['adaptive_after']:
                    raise RuntimeError('Temporal adaptive multiplier is not an integer')
                task.tpp_model.intensity_model.rejections_sample_multiple=int(saved['adaptive_after'])
                packets.append(saved);continue
            seed_sampling(seed+offset_base+offset)
            source=Batch.from_sequence_list(selected[offset:offset+64]).to(self.device)
            adaptive_before=float(task.tpp_model.intensity_model.rejections_sample_multiple)
            with torch.no_grad():temporal=task.tpp_model.sample(source.batch_size,x_n=source,tmax=source.tmax)
            self.memory_guard()
            maximum=min(task.discrete_diffusion.condition_encoder.max_position_embeddings,
                        (task.discrete_diffusion.transformer.positional_encoding.num_embeddings-3)//2)
            if int(temporal.unpadded_length.max())>maximum:raise RuntimeError('Temporal capacity overflow; no truncation')
            temporal=temporal.mask_check();temporal.po_matrix=source.po_matrix
            item=dict(temporal=temporal.to('cpu'),global_start=offset_base+offset)
            native,native_stats=self.spatial_sample(city,item,'no_projection',seed)
            full,full_stats=self.spatial_sample(city,item,'full',seed)
            if native_stats['spatial_rng_after']!=full_stats['spatial_rng_after']:raise RuntimeError('Base streams are not paired')
            packet=self.save_payload(path,identity,**item,indices=indices[offset:offset+64],
                adaptive_before=adaptive_before,adaptive_after=float(task.tpp_model.intensity_model.rejections_sample_multiple),
                no_projection=native,full=full,native_stats=native_stats,full_stats=full_stats)
            packets.append(packet)
        baseline={}
        refs=[raw['sequences'][i] for i in indices]
        for variant in ('full','no_projection'):
            records=[r for p in packets for r in p[variant]]
            baseline[variant]=evaluate(refs,records,raw['poi_category'],indices)[0]
        atomic_json(self.out/'calibration'/city/pool/f'seed-{seed}'/'baseline-metrics.json',baseline)
        return packets,baseline

    def read_pool(self,city,pool,seed):
        packets=[];indices=self.splits[city][pool];offset_base=0 if pool=='screen' else 512
        for offset in range(0,len(indices),64):
            path=self.out/'calibration'/city/pool/f'seed-{seed}'/f'batch-{offset}.pkl'
            identity=self.identity(kind='base-cache',city=city,pool=pool,seed=seed,indices=indices[offset:offset+64],global_start=offset_base+offset)
            packet=self.load_payload(path,identity)
            if packet is None:raise RuntimeError('Missing completed base cache')
            packets.append(packet)
        return packets,read(path.parent/'baseline-metrics.json')

    def speed_cases(self,city):
        packets,_=self.read_pool(city,'screen',SEEDS[0])
        eligible=[(int(p['temporal'].unpadded_length.max()),i,p) for i,p in enumerate(packets)
                  if p['full_stats']['projection_executed'] and p['temporal'].batch_size==64]
        if not eligible:raise RuntimeError('No full-batch projected speed cases')
        eligible.sort(key=lambda r:(r[0],r[1]))
        return {'ordinary':eligible[(len(eligible)-1)//2][2],'long':eligible[-1][2]}

    def measure_case(self,city,case,item,config,round_name):
        label='base' if config is None else f's{config.geometry_steps}-{config_label(config)}'
        path=self.out/'speed'/round_name/city/case/f'{label}.json'
        identity=self.identity(city=city,case=case,config=asdict(config) if config else None,
                               base_identity=item['identity'],round=round_name)
        if path.exists():
            result=read(path)
            if result['identity']!=identity or result['state']!='complete':raise RuntimeError('Invalid completed speed record')
            return result
        samples=[];allocated=[];reserved=[]
        for trial in range(7):
            self.phase=f'speed-{round_name}-{city}-{case}-{label}-{trial}';self.status()
            records,stats=self.spatial_sample(city,item,'full',SEEDS[0],config)
            if config is None:
                if not same_records(records,item['full']):raise RuntimeError('Replayed base Full records changed')
            else:
                assert_invariants(item['full'],records,self.refs[city])
                if not stats['geometry'] or stats['geometry']['optimizer_steps']!=config.geometry_steps:
                    raise RuntimeError('Speed case did not execute the full geometry budget')
            for key in ('spatial_rng_after','projection_rng_after','distance_rng_after'):
                if stats[key]!=item['full_stats'][key]:raise RuntimeError('Geometry changed an original random stream')
            if trial>=2:
                samples.append(stats['spatial_seconds']);allocated.append(stats['peak_allocated_bytes']);reserved.append(stats['peak_reserved_bytes'])
        result=dict(identity=identity,state='complete',seconds=samples,median_seconds=statistics.median(samples),
            peak_allocated_bytes=max(allocated),peak_reserved_bytes=max(reserved),batch_size=64,
            optimizer_steps=5000,geometry_steps=config.geometry_steps if config else 0,
            includes='complete spatial sampling, decoding, fresh model score, candidate construction, optimization and discrete selection',
            excludes='temporal generation, model loading, network; no per-batch scoring cache in online timing')
        atomic_json(path,result)
        return result

    def speed_gate(self,chosen_config=None,round_name='qualification'):
        steps=STEP_CHOICES if chosen_config is None else (chosen_config.geometry_steps,)
        summary={}
        bases={}
        for city in CITIES:
            self.load_model(city)
            for case,item in self.speed_cases(city).items():
                bases[city+'/'+case]=self.measure_case(city,case,item,None,round_name)
        for step in steps:
            summary[str(step)]={}
            config=chosen_config or GeometryConfig('same_category_v1',geometry_steps=step)
            for city in CITIES:
                self.load_model(city)
                for case,item in self.speed_cases(city).items():
                    measured=self.measure_case(city,case,item,config,round_name)
                    base=bases[city+'/'+case]
                    summary[str(step)][city+'/'+case]=dict(state='complete',ratio=measured['median_seconds']/base['median_seconds'],
                        base=base['median_seconds'],augmented=measured['median_seconds'])
            atomic_json(self.out/'speed'/f'{round_name}-summary.json',summary)
            if chosen_config is None and step==50 and choose_steps(summary) is None:
                return None,summary
        selected=choose_steps(summary) if chosen_config is None else (
            chosen_config.geometry_steps if all(r['ratio']<=SPEED_CEILING for r in summary[str(chosen_config.geometry_steps)].values()) else None)
        return selected,summary

    def profile_tail(self,steps):
        profiles={}
        for city in CITIES:
            self.load_model(city)
            for case,item in self.speed_cases(city).items():
                _,stats=refine_records(item['full'],self.task.discrete_diffusion,self.refs[city],
                    GeometryConfig('same_category_v1',geometry_steps=steps),seed=SEEDS[0],global_start=item['global_start'],profile=True)
                self.memory_guard()
                profiles[city+'/'+case]=stats
        atomic_json(self.out/'speed'/'separate-stage-profiles.json',profiles)

    def stop_report(self,reason):
        lines=['# PCDG-Geo: study stopped at a declared gate','',reason,'',
               'No test-set tuning, no base retraining, no changes to source artifacts.','',
               'See performance-qualification.json, speed/, screening.json and confirmation.json when present.']
        path=self.out/'performance-qualification.json'
        if path.exists():lines+=['','## Complete spatial-sampling timing',json.dumps(read(path),indent=2)]
        (self.out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

    def refine_pool(self,city,pool,seed,config):
        label=config_label(config)
        path=self.out/'candidates'/label/city/pool/f'seed-{seed}.pkl'
        identity=self.identity(kind='candidate',city=city,pool=pool,seed=seed,config=asdict(config))
        saved=self.load_payload(path,identity)
        if saved is not None:return saved['result']
        task,_=self.load_model(city)
        packets,baselines=self.read_pool(city,pool,seed)
        generated=[];diags=[];seconds=0.
        for packet in packets:
            cache_key=(city,pool,seed,packet['global_start'])
            can_refine=packet['full_stats']['projection_executed'] and any(len(r['checkins'])>=2 for r in packet['full'])
            if can_refine and cache_key not in self.score_cache:
                self.score_cache[cache_key]=untruncated_scores(task.discrete_diffusion,packet['full']).cpu()
            scores=self.score_cache.get(cache_key)
            torch.cuda.synchronize(self.device);started=time.perf_counter()
            values,diag=refine_records(packet['full'],task.discrete_diffusion,self.refs[city],config,seed=seed,
                global_start=packet['global_start'],projection_executed=packet['full_stats']['projection_executed'],
                scored_logits=scores.to(self.device) if scores is not None else None)
            torch.cuda.synchronize(self.device);seconds+=time.perf_counter()-started
            self.memory_guard();generated.extend(values);diags.append(diag)
        indices=self.splits[city][pool]
        metrics,_=evaluate([self.train_raw[city]['sequences'][i] for i in indices],generated,self.train_raw[city]['poi_category'],indices)
        check_invariant_metrics(metrics,baselines['full'])
        result=dict(state='complete',metrics=metrics,offline_seconds=seconds,scoring_cache_reused=True,config=asdict(config))
        self.save_payload(path,identity,sequences=generated,geometry_diagnostics=diags,result=result)
        return result

    def calibrate(self,steps):
        configs=configurations(steps)
        results={config_label(c):dict(config=asdict(c),cities={}) for c in configs}
        baselines={}
        for city in CITIES:
            _,base=self.read_pool(city,'screen',SEEDS[0]);baselines[city]={k:[v] for k,v in base.items()}
            for config in configs:
                self.phase=f'screen-{city}-{config_label(config)}';self.status()
                results[config_label(config)]['cities'][city]=[self.refine_pool(city,'screen',SEEDS[0],config)]
        ranked=rank_candidates(results,baselines)
        atomic_json(self.out/'screening.json',dict(results=results,baselines=baselines,ranking=ranked))
        finalists=ranked[:3]
        if not finalists:return None,'no-screening-candidate'
        confirmed={r['label']:dict(config=r['config'],cities={}) for r in finalists}
        confirm_base={}
        for city in CITIES:
            confirm_base[city]={'full':[],'no_projection':[]}
            for seed in SEEDS:
                _,base=self.prepare_pool(city,'confirm',seed)
                for k,v in base.items():confirm_base[city][k].append(v)
                for r in finalists:
                    config=GeometryConfig(**r['config'])
                    self.phase=f'confirm-{city}-{seed}-{r["label"]}';self.status()
                    confirmed[r['label']]['cities'].setdefault(city,[]).append(self.refine_pool(city,'confirm',seed,config))
        ranking=rank_candidates(confirmed,confirm_base,confirmation=True)
        atomic_json(self.out/'confirmation.json',dict(results=confirmed,baselines=confirm_base,ranking=ranking))
        for r in ranking:
            config=GeometryConfig(**r['config'])
            passed,speed=self.speed_gate(config,round_name='confirm-'+r['label'])
            if passed is not None:
                recommendation=dict(state='selected',config=asdict(config),configuration_sha256=content_hash(asdict(config)),
                    study_manifest_sha256=self.manifest_sha,screening_sha256=sha256_file(self.out/'screening.json'),
                    confirmation_sha256=sha256_file(self.out/'confirmation.json'),speed=speed)
                atomic_json(self.out/'recommendation.json',recommendation)
                return config,None
        return None,'no-confirmed-quality-and-speed-candidate'

    def source_batches(self,record):
        packets=[]
        for shard in record['source_shards']:
            path=Path(shard['path'])
            if sha256_file(path)!=shard['sha256']:raise RuntimeError('Sealed Full shard changed')
            value=torch.load(path,map_location='cpu',weights_only=False)
            offset=0
            for trace in value['batch_traces']:
                n=len(trace['indices']);records=value['sequences'][offset:offset+n];offset+=n
                packets.append(dict(records=records,indices=trace['indices'],global_start=trace['global_start'],
                    projection_executed=any(c['optimizer_steps']>0 for c in trace['projection_stats'])))
            if offset!=len(value['sequences']):raise RuntimeError('Source batch layout is incomplete')
        return packets

    def formal(self,config):
        from tools.unified_tables import key
        registry={}
        for city in CITIES:
            self.load_model(city)
            raw_path=Path(self.profiles[city]['data_root'])/city/f'{city}_test.pkl'
            if sha256_file(raw_path)!=self.profiles[city]['data_sha256']['test']:raise RuntimeError('Test input changed')
            raw=torch.load(raw_path,map_location='cpu',weights_only=False)
            for seed in SEEDS:
                source=self.parent_registry[key(city,seed,'full')]
                if sha256_file(source['path'])!=source['sha256']:raise RuntimeError('Sealed Full result changed')
                parent=torch.load(source['path'],map_location='cpu',weights_only=False)
                packets=self.source_batches(source)
                for variant,cfg in (('full',config),('distance_only',replace(config,geometry_radius_weight=0.)),
                                    ('radius_only',replace(config,geometry_distance_weight=0.))):
                    self.phase=f'formal-{city}-{seed}-{variant}';self.status()
                    folder=self.out/'formal'/city/f'seed-{seed}'/variant
                    identity=self.identity(kind='formal',city=city,seed=seed,variant=variant,config=asdict(cfg),
                                           parent_result_sha256=source['sha256'])
                    path=folder/'generated.pkl'
                    saved=self.load_payload(path,identity)
                    if saved is not None:
                        registry[key(city,seed,variant)]=saved['result'];continue
                    generated=[];stats=[];seconds=0.
                    for packet in packets:
                        part_path=folder/f'batch-{packet["global_start"]}.pkl'
                        part_id=dict(identity,global_start=packet['global_start'],indices=packet['indices'])
                        part=self.load_payload(part_path,part_id)
                        if part is None:
                            torch.cuda.synchronize(self.device);started=time.perf_counter()
                            values,diag=refine_records(packet['records'],self.task.discrete_diffusion,self.refs[city],cfg,
                                seed=seed,global_start=packet['global_start'],projection_executed=packet['projection_executed'])
                            torch.cuda.synchronize(self.device);elapsed=time.perf_counter()-started;self.memory_guard()
                            part=self.save_payload(part_path,part_id,sequences=values,diagnostics=diag,seconds=elapsed)
                        generated.extend(part['sequences']);stats.append(part['diagnostics']);seconds+=part['seconds']
                    assert_invariants(parent['sequences'],generated,self.refs[city])
                    metrics,conditions=evaluate(raw['sequences'],generated,raw['poi_category'],parent['test_indices'])
                    check_invariant_metrics(metrics,source['metrics'])
                    diagnostics=geometric_diagnostics(raw['sequences'],parent['sequences'],generated,raw['poi_category'])
                    result=dict(state='complete',dataset=city,seed=seed,variant=variant,metrics=metrics,
                        config=asdict(cfg),parent_result_sha256=source['sha256'],base_backend=source.get('distance_backend','legacy'),
                        geometry_implementation_version=GEOMETRY_VERSION,geometry_reference_sha256=self.refs[city].fingerprint,
                        offline_refinement_seconds=seconds,source_full_spatial_seconds=source['timing']['spatial_wall_seconds'],
                        cost_note='Offline refinement is not total online sampling cost; parent costs retain historical backend.',
                        diagnostics=diagnostics,path=str(path))
                    atomic_json(folder/'per_condition.json',conditions)
                    self.save_payload(path,identity,sequences=generated,test_indices=parent['test_indices'],t_max=24.,
                                      geometry_diagnostics=stats,result=result)
                    registry[key(city,seed,variant)]=result
                    atomic_json(self.out/'registry.json',registry)
        atomic_json(self.out/'registry.json',registry)
        if len(registry)!=18:raise RuntimeError('Formal registry must contain exactly 18 new results')
        return registry

    def final_report(self,registry):
        from tools.unified_tables import key
        rows=[];acceptance={}
        for city in CITIES:
            original_full=metric_means([self.parent_registry[key(city,s,'full')]['metrics'] for s in SEEDS])
            joint=metric_means([self.parent_registry[key(city,s,'no_projection')]['metrics'] for s in SEEDS])
            for variant in ('full','distance_only','radius_only'):
                values=[registry[key(city,s,variant)]['metrics'] for s in SEEDS]
                means=metric_means(values)
                sd={k:statistics.stdev(v[k] for v in values) if all(v[k] is not None for v in values) else None
                    for k in means}
                rows.append(dict(city=city,variant=variant,mean=means,sample_sd=sd))
                if variant=='full':
                    failures=quality_failures(means,original_full,True)
                    targets=target_metrics(city,original_full,joint)
                    acceptance[city]=dict(quality_failures=failures,targets=targets,
                        geometry_goals_pass=all(means[k] is not None and means[k]<=v for k,v in targets.items()))
        atomic_json(self.out/'results-summary.json',dict(rows=rows,acceptance=acceptance))
        lines=['# PCDG-Geo final results','',
            'One frozen configuration; 18 new refinements of sealed Full outputs. No test-driven search.',
            'The calibration data were used in base training; this is engineering calibration, not independent validation.',
            'The baseline was previously inspected. Reported benchmark iteration is not a new untouched test set.','',
            '| City | Variant | Distance | Radius | DailyLoc | G-RANK |','|---|---|---:|---:|---:|---:|']
        for r in rows:
            cells=[f"{r['mean'][m]:.6f} ± {r['sample_sd'][m]:.6f}" if r['mean'][m] is not None else 'undefined'
                   for m in ('Distance','Radius','DailyLoc','G-RANK')]
            lines.append('| '+' | '.join([r['city'],r['variant'],*cells])+' |')
        lines+=['','## Acceptance','',json.dumps(acceptance,indent=2),'',
            'Speed: see speed/ and recommendation.json. All costs are complete spatial sampling on the same GPU/input; temporal generation, loading and network excluded.',
            'Offline refinement costs in registry.json are NOT complete online sampling costs. Historical legacy/batched Full provenance is retained.',
            'All frozen category/time/order/length/condition fields and endpoint-selection masks were checked. Geometry goals are not guaranteed.']
        (self.out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        return acceptance

    def execute(self):
        self.precheck()
        for city in CITIES:self.prepare_pool(city,'screen',SEEDS[0])
        steps,speed=self.speed_gate()
        atomic_json(self.out/'performance-qualification.json',dict(selected_steps=steps,speed=speed,
                    ceiling=SPEED_CEILING,study_manifest_sha256=self.manifest_sha))
        self.profile_tail(steps or 50)
        if steps is None:
            self.finish_model();self.verify_parent();self.phase='stopped-at-speed-gate'
            self.stop_report('50 steps exceed the 10% complete spatial-sampling overhead ceiling. Quality search and formal refinement were not started.')
            self.status('stopped',reason='50 steps exceed 10 percent complete spatial-sampling overhead; no quality search or formal sampling launched')
            return
        if self.args.stage=='speed':
            self.finish_model();self.phase='speed-qualified';self.status('ready',selected_steps=steps);return
        config,reason=self.calibrate(steps)
        if config is None:
            self.finish_model();self.verify_parent();self.phase='stopped-at-quality-gate';self.stop_report(reason);self.status('stopped',reason=reason);return
        registry=self.formal(config)
        self.finish_model();self.verify_parent()
        if self.fingerprint()!=self.manifest['source_sha256']:raise RuntimeError('Study source changed')
        acceptance=self.final_report(registry)
        files={p.relative_to(self.out).as_posix():sha256_file(p) for p in self.out.rglob('*')
               if p.is_file() and p.name not in ('pipeline.lock','status.json','audit.json')}
        atomic_json(self.out/'audit.json',dict(state='passed',new_results=18,parent_unchanged=True,
            no_training=True,config_frozen_before_test=True,files=files,acceptance=acceptance))
        self.phase='complete'
        self.status('complete',new_results=18,quality_goals_met=all(not v['quality_failures'] and v['geometry_goals_pass'] for v in acceptance.values()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run',required=True)
    parser.add_argument('--run-id',default='pcdg-geo-v1-20261010')
    parser.add_argument('--stage',choices=('speed','all'),default='all')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    os.chdir(ROOT)
    study=GeometryStudy(args)
    import fcntl
    with (study.out/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:study.execute()
        except BaseException as exc:
            study.status('failed',error_type=type(exc).__name__,error=str(exc));raise


if __name__=='__main__':main()
