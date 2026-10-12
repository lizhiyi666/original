"""Bounded two-city distribution study. Never trains, creates time caches, or restarts workers."""
import argparse
import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '4')
import torch

from distribution_balance import (BalanceConfig, VERSION, configure_balance, digest,
    load_reference_manifest, refine_balanced_records)
from distance_kl import distance_generator
from experiment_io import atomic_json, decode_preserving_empty, safe_tag, sha256_file
from geometry_projection import assert_invariants, untruncated_scores
from tools.ablation_common import DISTANCE_VERSION, evaluate, generator, projector, rng_digest
from tools.baseline_common import precision, state_hash
from tools.category_cache_audit import audit_alignment, metrics_equal
from tools.city_common import load_city
from tools.distribution_study_common import (CITIES, SEEDS, RUN_ID, METRICS, LOWER, HIGHER,
    aggregate, aligned_indices, configurations, distance_diagnostics, failures,
    immutable_json, label, means, read, saved_payload, save_payload, select)
from tools.geometry_inheritance import checked_path
from tools.run_category_geometry import fingerprint


class Sources:
    def __init__(self, geometry, category):
        self.geometry, self.category = Path(geometry).resolve(), Path(category).resolve()
        self.inputs = {}
        self.geo = self.sealed(self.geometry)
        self.cat = self.sealed(self.category)
        self.parent = Path(self.cat['parent_run'])
        seal_path = self.parent/'delivery-files.json'
        self.add(seal_path)
        seal = read(seal_path)
        if read(self.parent/'audit.json')['state'] != 'passed':
            raise RuntimeError('PCDG/JointGen parent audit failed')
        self.add(self.parent/'audit.json')
        for name, expected in seal['files'].items():self.add(checked_path(self.parent, name), expected)
        self.add(self.parent/'manifest.json', seal['manifest_sha256'])
        self.add(self.parent/'registry.json')
        self.registry = read(self.parent/'registry.json')
        if len(self.registry) != 66:
            raise RuntimeError('Expected the sealed 66-result baseline')
        self.profiles = self.geo['profiles']
        self.prefix = self.geo.get('inheritance', {}).get('snapshot', '')
        self.cache_root = self.geometry/self.prefix
        self.cache_manifest = read(self.cache_root/'manifest.json')
        self.cache_manifest_sha = self.add(self.cache_root/'manifest.json')
        if self.cache_manifest['splits'] != self.geo['splits']:
            raise RuntimeError('Inherited calibration split changed')
        self.refs = {}
        for city in CITIES:
            profile, split = self.profiles[city], self.geo['splits'][city]
            if len(split['screen']) != 512 or len(split['confirm']) != 512:
                raise RuntimeError('Calibration pool budget changed')
            joined = split['screen']+split['confirm']+split['reference']
            aligned_indices(sorted(joined), range(profile['train_count']))
            for name, expected in ((profile['checkpoint'], profile['checkpoint_sha256']),
                                   (profile['config_path'], profile['config_sha256'])):
                self.add(name, expected)
            for part in ('train', 'test'):
                self.add(self.data_path(city, part), profile['data_sha256'][part])
            self.refs[city] = load_reference_manifest(self.geometry/'manifest.json', city, self.data_path(city, 'train'))
            for pool, seeds in (('screen', SEEDS[:1]), ('confirm', SEEDS)):
                for seed in seeds:
                    self.training_packets(city, pool, seed)
            for seed in SEEDS:
                for rank in range(2):
                    path = self.category/city/f'seed-{seed}'/f'rank-{rank}'/'payload.pkl'
                    # Already part of category audit; do not accept an unsigned shard.
                    if str(path) not in self.inputs:raise RuntimeError('Unsigned category source shard')
                    shard = torch.load(path, map_location='cpu', weights_only=False)
                    job = shard['job']
                    if job['city'] != profile or job['seed'] != seed:
                        raise RuntimeError('Category and geometry profiles differ')
                    self.add(job['cache'], job['cache_sha256'])

    def data_path(self, city, split):
        p = self.profiles[city]
        return Path(p['data_root'])/p['dataset']/f'{p["dataset"]}_{split}.pkl'

    def add(self, path, expected=None):
        path = Path(path).resolve(); actual = sha256_file(path)
        if expected is not None and actual != expected:
            raise RuntimeError('Read-only input fingerprint mismatch: '+str(path))
        if str(path) in self.inputs and self.inputs[str(path)] != actual:
            raise RuntimeError('Source changed during preflight: '+str(path))
        self.inputs[str(path)] = actual
        return actual

    def sealed(self, root):
        if read(root/'status.json').get('state') != 'complete':
            raise RuntimeError('Source is not complete: '+str(root))
        audit = read(root/'audit.json')
        if audit.get('state') != 'passed':raise RuntimeError('Source audit failed')
        self.add(root/'audit.json');self.add(root/'status.json')
        for name, expected in audit['files'].items():self.add(checked_path(root, name), expected)
        if str(root/'manifest.json') not in self.inputs:raise RuntimeError('Source manifest is not sealed')
        return read(root/'manifest.json')

    def verify(self):
        for path, expected in self.inputs.items():
            if sha256_file(path) != expected:raise RuntimeError('Read-only source changed: '+path)

    def training_packets(self, city, pool, seed):
        indices = self.geo['splits'][city][pool]; packets = []
        for offset in range(0, 512, 64):
            path = self.cache_root/'calibration'/city/pool/f'seed-{seed}'/f'batch-{offset}.pkl'
            if str(path) not in self.inputs:raise RuntimeError('Missing sealed time cache: '+str(path))
            identity = dict(study_manifest_sha256=self.cache_manifest_sha, kind='base-cache', city=city,
                pool=pool, seed=seed, indices=indices[offset:offset+64], global_start=offset+(512 if pool=='confirm' else 0))
            packet = saved_payload(path, identity)
            if packet is None or packet['indices'] != identity['indices'] or packet['global_start'] != identity['global_start']:
                raise RuntimeError('Calibration source identity mismatch')
            if packet['temporal'].batch_size != 64:
                raise RuntimeError('Original calibration batch changed')
            packets.append(dict(packet, source_sha256=self.inputs[str(path)], distance_backend=self.geo['base_backend']))
        return packets

    def formal_packets(self, city, seed):
        from tools.ablation_worker import check_cache
        packets = []
        for rank in range(2):
            path = self.category/city/f'seed-{seed}'/f'rank-{rank}'/'payload.pkl'
            shard = torch.load(path, map_location='cpu', weights_only=False)
            job = shard['job']
            if sha256_file(job['cache']) != self.inputs[str(Path(job['cache']).resolve())]:
                raise RuntimeError('Formal temporal cache changed')
            cache = torch.load(job['cache'], map_location='cpu', weights_only=False)
            check_cache(cache, job)
            if len(cache['batches']) != len(shard['batch_traces']):raise RuntimeError('Formal batch trace mismatch')
            offset=0
            for item, trace in zip(cache['batches'], shard['batch_traces']):
                if item['indices'] != trace['indices'] or item['global_start'] != trace['global_start']:
                    raise RuntimeError('Formal cached batch boundaries changed')
                packets.append(dict(temporal=item['batch'], indices=item['indices'], global_start=item['global_start'],
                    full_stats=trace, source_sha256=job['cache_sha256'], distance_backend=job.get('distance_backend', 'legacy'),
                    source_a_records=shard['sequences'][offset:offset+len(item['indices'])],source_a_sha256=self.inputs[str(path)]))
                offset+=len(item['indices'])
            if offset!=len(shard['sequences']):raise RuntimeError('Source A records do not match cache layout')
        aligned_indices([i for p in packets for i in p['indices']], range(self.profiles[city]['test_count']))
        return packets


def identity(manifest_sha, **fields):
    return dict(study_manifest_sha256=manifest_sha, **fields)


def verify_source_manifest(source, manifest):
    if source.inputs != manifest['source_inputs'] or fingerprint() != manifest['source_sha256']:
        raise RuntimeError('Study source/code identity changed; no implicit resume')


def copy_input(source, out, city, split):
    src = source.data_path(city, split); dst = out/'evaluation_inputs'/city/f'{city}_{split}.pkl'
    expected = source.profiles[city]['data_sha256'][split]
    if dst.exists():
        if sha256_file(dst) != expected:raise RuntimeError('Local evaluation input conflict')
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open('rb') as inp, dst.open('xb') as target:shutil.copyfileobj(inp, target)
    if sha256_file(dst) != expected:raise RuntimeError('Evaluation input copy failed')


class Worker:
    def __init__(self, args, out, source):
        self.args, self.out, self.source, self.city = args, out, source, args.worker_city
        self.manifest = read(out/'manifest.json');self.sha = sha256_file(out/'manifest.json')
        verify_source_manifest(source, self.manifest)
        precision();torch.set_num_threads(4)
        self.device = torch.device('cuda:0')
        self.task, self.dm = load_city(source.profiles[self.city], device=str(self.device))
        self.task.requires_grad_(False)
        self.model_hash = state_hash(self.task)
        self.ref = source.refs[self.city]
        self.raw = torch.load(source.data_path(self.city, 'train' if args.phase != 'formal' else 'test'),
                              map_location='cpu', weights_only=False)
        def forbidden(*a, **kw):raise RuntimeError('TPP sampling/training forbidden in this study')
        self.task.tpp_model.sample = forbidden
        self.progress_path = out/f'worker-{args.phase}-{self.city}.json'

    def progress(self, **values):
        atomic_json(self.progress_path, dict(state='running', pid=os.getpid(), phase=self.args.phase,
                    city=self.city, updated_at=time.time(), **values))

    def guard(self):
        if torch.cuda.max_memory_reserved(self.device) > torch.cuda.get_device_properties(self.device).total_memory*.8:
            raise RuntimeError('80-percent memory ceiling exceeded; do not shrink batch')

    def sample(self, packet, seed, config):
        city, phase = self.city, self.args.phase
        path = self.out/'spatial'/phase/city/f'seed-{seed}'/label(config, spatial=True)/f'batch-{packet["global_start"]}.pkl'
        ident = identity(self.sha, kind='spatial', phase=phase, city=city, seed=seed,
            category_config=[config.constraint_scale, config.category_distribution_weight],
            indices=packet['indices'], global_start=packet['global_start'],
            source_sha256=packet['source_sha256'], distance_backend=packet['distance_backend'])
        saved = saved_payload(path, ident)
        if saved is not None:return saved
        if phase=='formal' and config.constraint_scale==1. and config.category_distribution_weight==0.:
            # This category configuration IS archived A: reuse it, never resample it.
            records=packet['source_a_records'];trace=packet['full_stats']
            audit_alignment([dict(batch=packet['temporal'],indices=packet['indices'],global_start=packet['global_start'])],
                            records,records,self.raw['poi_category'],self.raw['poi_gps'])
            stats=dict(trace,projection_executed=any(c['optimizer_steps']>0 for c in trace['projection_stats']),
                       spatial_seconds=0.,reused_archived_a=True,source_a_sha256=packet['source_a_sha256'])
            return save_payload(path,ident,sequences=records,stats=stats)
        self.progress(operation='spatial', seed=seed, config=label(config, spatial=True), global_start=packet['global_start'])
        dd = self.task.discrete_diffusion
        dd.geometry_config = None;dd.use_guidance_baseline = False;dd.use_constraint_projection = True
        dd.projection_frequency = 4;dd.projection_last_k_steps = 40;dd.projection_call_count = 0
        start = packet['global_start']
        spatial = generator(seed, start, 'spatial', self.device)
        projection = generator(seed, start, 'projection', self.device)
        p = projector(dd, 'full', projection, revision=DISTANCE_VERSION, datamodule=self.dm,
                      distance_backend=packet['distance_backend'])
        p.distance_generator = distance_generator(seed+start, self.device)
        dd.constraint_projector = p;dd.sampling_generator = spatial
        configure_balance(dd, BalanceConfig('train_ref_v1', config.constraint_scale, config.category_distribution_weight, 0.),
                          self.ref, terminal_refinement=False)
        calls = []; original = p.project_with_matrices
        def traced(*a, **kw):
            result = original(*a, **kw);self.guard();calls.append(dict(p.last_projection_stats));return result
        p.project_with_matrices = traced
        batch = copy.deepcopy(packet['temporal']).to(self.device)
        before = (rng_digest(torch.get_rng_state()), rng_digest(torch.cuda.get_rng_state(self.device)))
        torch.cuda.synchronize(self.device);torch.cuda.reset_peak_memory_stats(self.device);started=time.perf_counter()
        records = decode_preserving_empty(self.task, batch, self.raw['poi_gps'])
        torch.cuda.synchronize(self.device);seconds=time.perf_counter()-started;self.guard()
        after = (rng_digest(torch.get_rng_state()), rng_digest(torch.cuda.get_rng_state(self.device)))
        if before != after:raise RuntimeError('Spatial sampling consumed global RNG')
        active = [c for c in calls if c['optimizer_steps'] > 0]
        if active and (len(active) != 10 or any(c['optimizer_steps'] != 500 for c in active)):
            raise RuntimeError('Original ALM budget changed')
        stats = dict(spatial_rng_after=rng_digest(spatial), projection_rng_after=rng_digest(projection),
            distance_rng_after=rng_digest(p.distance_generator), global_rng_unchanged=True,
            projection_executed=bool(active), projection_calls=len(active), optimizer_steps=sum(c['optimizer_steps'] for c in active),
            spatial_seconds=seconds, projection_stats=calls, distance_backend=packet['distance_backend'])
        for key in ('spatial_rng_after', 'projection_rng_after', 'distance_rng_after'):
            if stats[key] != packet['full_stats'][key]:raise RuntimeError('Paired original RNG changed: '+key)
        audit_alignment([dict(batch=packet['temporal'], indices=packet['indices'], global_start=start)],
                        records, records, self.raw['poi_category'], self.raw['poi_gps'])
        if state_hash(self.task) != self.model_hash:raise RuntimeError('Frozen model changed')
        return save_payload(path, ident, sequences=records, stats=stats)

    def baseline(self, packets, seed):
        path = self.out/'baselines'/self.args.phase/self.city/f'seed-{seed}.pkl'
        indices = [i for p in packets for i in p['indices']]
        ident = identity(self.sha, kind='baseline', phase=self.args.phase, city=self.city, seed=seed, indices=indices)
        old = saved_payload(path, ident)
        if old is not None:return old['result']
        references = [self.raw['sequences'][i] for i in indices]
        if self.args.phase == 'formal':
            records = {}
            for variant in ('full', 'no_projection'):
                entry = self.source.registry[f'{self.city}/{seed}/{variant}']
                if sha256_file(entry['path']) != entry['sha256']:raise RuntimeError('Formal baseline changed')
                payload = torch.load(entry['path'], map_location='cpu', weights_only=False)
                records[variant] = payload['sequences']
        else:
            records = {k: [r for p in packets for r in p[k]] for k in ('full', 'no_projection')}
        result = dict(seed=seed)
        for name, values in records.items():
            result[name] = evaluate(references, values, self.raw['poi_category'], indices)[0]
            expected = (self.source.registry[f'{self.city}/{seed}/{name}']['metrics'] if self.args.phase == 'formal' else
                read(self.source.cache_root/'calibration'/self.city/self.args.phase/f'seed-{seed}'/'baseline-metrics.json')[name])
            metrics_equal(result[name], expected)
        save_payload(path, ident, result=result, sequences_by_method=records, indices=indices,
                     city=self.city, split='test' if self.args.phase == 'formal' else 'train')
        return result

    def run(self, configs):
        rows = {label(c): [] for c in configs}; baselines=[]
        seeds = SEEDS[:1] if self.args.phase == 'screen' else SEEDS
        groups = {}
        for cfg in configs:groups.setdefault(label(cfg, spatial=True), []).append(cfg)
        for seed in seeds:
            packets = (self.source.formal_packets(self.city, seed) if self.args.phase == 'formal'
                       else self.source.training_packets(self.city, self.args.phase, seed))
            indices = [i for p in packets for i in p['indices']]
            expected = (range(self.source.profiles[self.city]['test_count']) if self.args.phase == 'formal'
                        else self.source.geo['splits'][self.city][self.args.phase])
            aligned_indices(indices, expected)
            baselines.append(self.baseline(packets, seed))
            for group in groups.values():
                collected = {label(c): [] for c in group}; traces={label(c): [] for c in group}
                for packet in packets:
                    sampled = self.sample(packet, seed, group[0])
                    scored = None
                    for cfg in group:
                        name=label(cfg);start=packet['global_start']
                        path=self.out/'refined'/self.args.phase/self.city/f'seed-{seed}'/name/f'batch-{start}.pkl'
                        ident=identity(self.sha, kind='refined', phase=self.args.phase, city=self.city, seed=seed,
                            config=asdict(cfg), indices=packet['indices'], global_start=start,
                            spatial_identity=sampled['identity'])
                        part=saved_payload(path, ident)
                        if part is None:
                            self.progress(operation='refinement', seed=seed, config=name, global_start=start)
                            records=sampled['sequences'];active=sampled['stats']['projection_executed']
                            movable=active and any(len(r['checkins'])>1 and any(self.ref.category_counts[self.ref.poi_category[int(p)]]>1
                                         for p in r['checkins']) for r in records)
                            if movable and scored is None:scored=untruncated_scores(self.task.discrete_diffusion, records)
                            before=(rng_digest(torch.get_rng_state()),rng_digest(torch.cuda.get_rng_state(self.device)))
                            output, stats=refine_balanced_records(records,self.task.discrete_diffusion,self.ref,cfg,
                                seed=seed,global_start=start,projection_executed=active,scored_logits=scored)
                            self.guard()
                            if before!=(rng_digest(torch.get_rng_state()),rng_digest(torch.cuda.get_rng_state(self.device))):
                                raise RuntimeError('Refinement consumed global RNG')
                            assert_invariants(records,output,self.ref)
                            part=save_payload(path,ident,sequences=output,stats=stats)
                        collected[name].extend(part['sequences']);traces[name].append(part['stats'])
                for cfg in group:
                    name=label(cfg);path=self.out/'results'/self.args.phase/self.city/f'seed-{seed}'/(name+'.pkl')
                    ident=identity(self.sha,kind='result',phase=self.args.phase,city=self.city,seed=seed,config=asdict(cfg),indices=indices)
                    result=saved_payload(path,ident)
                    if result is None:
                        references=[self.raw['sequences'][i] for i in indices]
                        metrics, _=evaluate(references,collected[name],self.raw['poi_category'],indices)
                        row=dict(seed=seed,metrics=metrics,config=asdict(cfg),samples=len(indices),
                            diagnostics=distance_diagnostics(references,collected[name],self.raw['poi_category']))
                        result=save_payload(path,ident,result=row,sequences=collected[name],indices=indices,
                            city=self.city,split='test' if self.args.phase=='formal' else 'train',refinement_stats=traces[name])
                    rows[name].append(result['result'])
        if state_hash(self.task)!=self.model_hash:raise RuntimeError('Frozen model changed')
        verify_source_manifest(self.source,self.manifest);self.source.verify()
        result=dict(state='complete',study_manifest_sha256=self.sha,city=self.city,phase=self.args.phase,results=rows,baselines=baselines,
                    frozen_model_sha256=self.model_hash,model_unchanged=True)
        immutable_json(self.out/f'worker-{self.args.phase}-{self.city}.done.json',result)
        atomic_json(self.progress_path,dict(state='complete',pid=os.getpid(),city=self.city,phase=self.args.phase))


def phase_run(args, out, phase, configs):
    selection=out/f'{phase}-configs.json';immutable_json(selection,[asdict(c) for c in configs])
    workers=[]
    try:
        for city,gpu in zip(CITIES,args.gpus.split(',')):
            done=out/f'worker-{phase}-{city}.done.json'
            if done.exists():
                if read(done).get('state')!='complete':raise RuntimeError('Invalid completed worker')
                continue
            command=[sys.executable,'-u','-B',str(Path(__file__).resolve()),'--geometry-source',str(args.geometry_source),
                '--category-source',str(args.category_source),'--run-id',args.run_id,'--worker-city',city,'--phase',phase]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='4')
            stream=(out/f'worker-{phase}-{city}.log').open('ab')
            proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            workers.append((city,proc,stream))
        while any(p.poll() is None for _,p,_ in workers):
            if any(p.poll() not in (None,0) for _,p,_ in workers):
                raise RuntimeError('Worker failed; no automatic restart')
            time.sleep(2)
        if any(p.returncode for _,p,_ in workers):raise RuntimeError('Worker abnormal exit')
    finally:
        # Only our own live workers are stopped on failure; other experiments are untouched.
        for _,proc,stream in workers:
            if proc.poll() is None:proc.terminate();proc.wait(timeout=30)
            stream.close()
    rows={label(c):dict(config=asdict(c),cities={}) for c in configs};baselines={}
    for city in CITIES:
        value=read(out/f'worker-{phase}-{city}.done.json')
        if (value['phase']!=phase or set(value['results'])!=set(rows)
                or value.get('study_manifest_sha256')!=sha256_file(out/'manifest.json')):
            raise RuntimeError('Worker result/config mismatch')
        baselines[city]=dict(rows=value['baselines'])
        for name in rows:
            for row in value['results'][name]:
                path=out/'results'/phase/city/f'seed-{row["seed"]}'/(name+'.pkl')
                payload=torch.load(path,map_location='cpu',weights_only=False)
                checked=saved_payload(path,payload['identity'])
                if (checked['identity']['study_manifest_sha256']!=value['study_manifest_sha256']
                        or checked['result']!=row):raise RuntimeError('Worker summary differs from immutable result')
            rows[name]['cities'][city]=value['results'][name]
    return rows,baselines


def report_text(summary):
    lines=['# 两城分布校正：有限校准与严格验收','',
        '模型未重训；训练侧工程校准，基础模型见过训练数据。不是独立泛化验证。',
        '六项分布指标不高于 JointGen；五项约束指标不差于 PCDG；容差仅 1e-12。',
        '两城共用参数，按各城三种子未舍入均值验收。不把均值通过解释为统计显著。','',
        '终态：'+summary['terminal_state'], '原因：'+summary['reason'],'']
    for phase in ('screen','confirm','formal'):
        if phase not in summary:continue
        result=summary[phase]
        lines += [f'## {phase}', '', '| 配置 | 城市 | Distance | Radius | CategoryTransition | DailyLoc | Category | G-RANK | 约束失败 |',
                  '|---|---|---:|---:|---:|---:|---:|---:|---|']
        for name,candidate in result['results'].items():
            for city in CITIES:
                actual=means([r['metrics'] for r in candidate['cities'][city]])
                pcdg=means([r['full'] for r in result['baselines'][city]['rows']])
                errors=failures(actual,pcdg,{},distributions=False)
                values=[actual.get(k) for k in ('Distance','Radius','CategoryTransition','DailyLoc','Category','G-RANK')]
                lines.append('| '+name+' | '+city+' | '+' | '.join('—' if v is None else f'{v:.6f}' for v in values)+' | '+(', '.join(errors) or '无')+' |')
        lines.append('')
    lines += ['未通过只说明本次预注册候选范围未达标；不证明目标不可能。',
        '原评价版本与非标准 Radius 公式保持不变；无首尾筛选 Distance 和入选数见各结果 diagnostics。',
        '各批次记录空间采样与细化耗时；不得将离线细化耗时当成端到端速度。','']
    return '\n'.join(lines)


def seal_run(out, source, manifest, summary):
    source.verify();verify_source_manifest(source,manifest)
    atomic_json(out/'summary.json',summary)
    (out/'report.md').write_text(report_text(summary),encoding='utf-8')
    formal=summary.get('formal',{}).get('results',{})
    registry={}
    for name,candidate in formal.items():
        for city in CITIES:
            for row in candidate['cities'][city]:
                path=out/'results'/'formal'/city/f'seed-{row["seed"]}'/(name+'.pkl')
                registry[f'{city}/{row["seed"]}']=dict(path=path.relative_to(out).as_posix(),sha256=sha256_file(path),**row)
    if formal and (len(registry)!=6 or sum(r['samples'] for r in registry.values())!=21066):
        raise RuntimeError('Formal delivery must be 6/6 and 21066 records')
    atomic_json(out/'registry.json',registry)
    files={p.relative_to(out).as_posix():sha256_file(p) for p in out.rglob('*') if p.is_file()
           and p.name not in ('audit.json','status.json','pipeline.lock') and p.suffix not in ('.log','.lock')}
    audit=dict(state='passed',terminal_state=summary['terminal_state'],results=len(registry),files=files,
        originals_unchanged=True,no_training=True,no_temporal_resampling=True,
        quality_goals_met=summary.get('quality_goals_met',False),configuration_frozen_before_test=bool(formal))
    atomic_json(out/'audit.json',audit)
    atomic_json(out/'status.json',dict(state=summary['terminal_state'],phase='sealed',pid=os.getpid(),
        reason=summary['reason'],quality_goals_met=audit['quality_goals_met']))


def execute(args,out,source):
    for city in CITIES:copy_input(source,out,city,'train')
    manifest=dict(revision=VERSION,run_id=args.run_id,source_commit=args.source_commit,source_sha256=fingerprint(),
        source_inputs=source.inputs,geometry_source=str(source.geometry),category_source=str(source.category),
        profiles=source.profiles,splits=source.geo['splits'],seeds=list(SEEDS),configs=[asdict(c) for c in configurations()],
        distribution_references={c:source.refs[c].fingerprint for c in CITIES},gpus=args.gpus,batch_size=64,
        precision='FP32; TF32 disabled',memory_ceiling=.8,geometry_steps=200,calibration='train-side engineering calibration',
        policy='shared parameters; strict PCDG constraints; JointGen six-metric ceiling; no test-driven selection',
        tolerance=1e-12,max_screen_configs=18,max_confirm_configs=2,max_formal_results=6)
    immutable_json(out/'manifest.json',manifest)
    summary=dict(terminal_state='stopped',reason='not-yet-calibrated',quality_goals_met=False)
    for phase in ('screen','confirm','formal'):
        if phase=='screen':configs=configurations()
        elif phase=='confirm':configs=[BalanceConfig(**r['config']) for r in summary['screen']['selection']['ranking'][:2]]
        else:configs=[BalanceConfig(**read(out/'recommendation.json')['config'])]
        if not configs:
            summary['reason']='no-constraint-feasible-screening-candidate';break
        if phase=='formal':
            for city in CITIES:copy_input(source,out,city,'test')
        atomic_json(out/'status.json',dict(state='running',phase=phase,pid=os.getpid(),updated_at=time.time()))
        results,baselines=phase_run(args,out,phase,configs)
        selection=select(results,baselines,confirmation=phase!='screen')
        summary[phase]=dict(results=results,baselines=baselines,selection=selection)
        for candidate in results.values():
            candidate['aggregate']={c:aggregate([r['metrics'] for r in candidate['cities'][c]]) for c in CITIES}
            candidate['paired_deltas']={}
            for city in CITIES:
                paired={b['seed']:b for b in baselines[city]['rows']}
                candidate['paired_deltas'][city]=[dict(seed=r['seed'],deltas={k:
                    (r['metrics'][k]-paired[r['seed']]['full' if k in LOWER+HIGHER else 'no_projection'][k]
                     if r['metrics'].get(k) is not None and paired[r['seed']]['full' if k in LOWER+HIGHER else 'no_projection'].get(k) is not None else None)
                    for k in METRICS}) for r in candidate['cities'][city]]
        immutable_json(out/f'{phase}-selection.json',summary[phase])
        if phase=='confirm':
            if not selection['ranking']:
                summary['reason']='no-confirmed-candidate-meets-all-gates';break
            chosen=selection['ranking'][0]
            immutable_json(out/'recommendation.json',dict(config=chosen['config'],configuration_sha256=digest(chosen['config']),
                confirmation_sha256=sha256_file(out/'confirm-selection.json'),frozen_before_test=True))
        if phase=='formal':
            summary.update(terminal_state='complete',reason='formal-evaluation-complete',
                           quality_goals_met=bool(selection['ranking']))
    seal_run(out,source,manifest,summary)
    print(json.dumps(dict(state=summary['terminal_state'],reason=summary['reason'],quality_goals_met=summary['quality_goals_met'])),flush=True)


def verify(out, recompute=False):
    out=Path(out);audit=read(out/'audit.json');status=read(out/'status.json')
    if (audit.get('state')!='passed' or status.get('state') not in ('stopped','complete')
            or audit['terminal_state']!=status['state'] or audit['results']!=(6 if status['state']=='complete' else 0)):
        raise RuntimeError('Invalid terminal audit')
    for name,expected in audit['files'].items():
        if sha256_file(checked_path(out,name))!=expected:raise RuntimeError('Delivery hash mismatch: '+name)
    count=0;inputs={}
    if recompute:
        manifest=read(out/'manifest.json')
        for name in audit['files']:
            if not name.endswith('.pkl') or not name.startswith(('results/','baselines/')):continue
            payload=torch.load(checked_path(out,name),map_location='cpu',weights_only=False)
            city,split=payload['city'],payload['split'];key=(city,split)
            if key not in inputs:
                path=out/'evaluation_inputs'/city/f'{city}_{split}.pkl'
                if sha256_file(path)!=manifest['profiles'][city]['data_sha256'][split]:raise RuntimeError('CPU evaluation input changed')
                inputs[key]=torch.load(path,map_location='cpu',weights_only=False)
            raw=inputs[key];indices=payload['indices'];refs=[raw['sequences'][i] for i in indices]
            values=(payload['sequences_by_method'] if name.startswith('baselines/') else {'metrics':payload['sequences']})
            for method,records in values.items():
                actual=evaluate(refs,records,raw['poi_category'],indices)[0]
                expected=payload['result'][method]
                metrics_equal(actual,expected);count+=1
    return dict(state='passed',results=audit['results'],terminal_state=status['state'],verified_files=len(audit['files'])+2,
        metrics_recomputed=count,cpu_recomputed=recompute,quality_goals_met=audit['quality_goals_met'],summary_sha256=sha256_file(out/'summary.json'))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--geometry-source',type=Path);ap.add_argument('--category-source',type=Path)
    ap.add_argument('--run-id',default=RUN_ID);ap.add_argument('--gpus',default='0,1')
    ap.add_argument('--source-commit');ap.add_argument('--resume',action='store_true')
    ap.add_argument('--preflight-only',action='store_true');ap.add_argument('--worker-city',choices=CITIES)
    ap.add_argument('--phase',choices=('screen','confirm','formal'))
    ap.add_argument('--verify-only',type=Path);ap.add_argument('--recompute',action='store_true')
    args=ap.parse_args()
    if args.verify_only:print(json.dumps(verify(args.verify_only,args.recompute)));return
    if not args.geometry_source or not args.category_source:ap.error('Both sealed source directories are required')
    source=Sources(args.geometry_source,args.category_source)
    if args.preflight_only:
        print(json.dumps(dict(state='passed',source_files=len(source.inputs),references={c:r.fingerprint for c,r in source.refs.items()})));return
    out=ROOT/'experiment_runs'/safe_tag(args.run_id)
    if args.worker_city:
        if not args.phase:ap.error('Worker phase is required')
        import fcntl
        with (out/f'worker-{args.phase}-{args.worker_city}.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            Worker(args,out,source).run([BalanceConfig(**c) for c in read(out/f'{args.phase}-configs.json')])
        return
    if not args.source_commit:ap.error('--source-commit is required for deployment provenance')
    if len(set(args.gpus.split(',')))!=2:ap.error('Exactly two distinct GPUs are required')
    if out.exists() and not args.resume:raise RuntimeError('Existing run; no overwrite or duplicate start')
    out.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (out/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'audit.json').exists():print(json.dumps(verify(out)));return
        try:execute(args,out,source)
        except BaseException as exc:
            atomic_json(out/'status.json',dict(state='failed',pid=os.getpid(),error_type=type(exc).__name__,error=str(exc)))
            raise


if __name__=='__main__':main()
