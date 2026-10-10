"""Independent, CPU-only alignment audit. Never invokes a sampler or model."""
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch

from experiment_io import sha256_file
from tools.ablation_common import evaluate, same_records
from tools.geometry_inheritance import checked_path
from tools.run_category_decoding_check import distribution_counts
from evaluations.statistical_metrics import evaluation, travel_distance

REVISION='category-time-cache-audit-v1'
EVENT_FIELDS=('arrival_times','marks','checkins','gps')+tuple(f'condition{i}' for i in range(1,7))
EXPECTED_ERROR='Generated length or record fields changed'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def array(value):
    return value.detach().cpu().numpy() if isinstance(value,torch.Tensor) else np.asarray(value)


def equivalent(a,b):
    if isinstance(a,torch.Tensor) or isinstance(b,torch.Tensor):
        return isinstance(a,torch.Tensor) and isinstance(b,torch.Tensor) and a.dtype==b.dtype and torch.equal(a,b)
    if isinstance(a,np.ndarray) or isinstance(b,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equivalent(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return isinstance(b,type(a)) and len(a)==len(b) and all(equivalent(x,y) for x,y in zip(a,b))
    return a==b


def cache_matches_original(derived_items,original_items):
    if len(derived_items)!=len(original_items):raise RuntimeError('Derived/original cache batch count differs')
    for a,b in zip(derived_items,original_items):
        if (a['indices']!=b['indices'] or a['global_start']!=b['global_start']
                or not equivalent(vars(a['batch']),vars(b['batch']))):
            raise RuntimeError('Derived time cache differs from the original cache')


def validate_gps(record,mapping,gps,require_category_match):
    pois=np.asarray(record['checkins']);marks=np.asarray(record['marks'])
    if len(pois)!=len(marks) or len(pois)!=len(record['gps']):raise RuntimeError('POI/category/GPS length differs')
    expected=[]
    for poi,mark in zip(pois,marks):
        if int(poi) not in mapping or int(poi) not in gps:raise RuntimeError('Non-catalog POI in saved output')
        if require_category_match and int(mark)!=int(mapping[int(poi)]):raise RuntimeError('Actual sampled category mismatch')
        value=gps[int(poi)]
        expected.append([float(v) for v in value.split(',')] if isinstance(value,str) else value)
    if not np.array_equal(np.asarray(record['gps']).reshape(-1,2),np.asarray(expected).reshape(-1,2)):
        raise RuntimeError('Saved GPS differs from catalog coordinates')


def audit_alignment(items,off,on,mapping,gps):
    if len(off)!=len(on):raise RuntimeError('Off/on row counts differ')
    matched=[];rows=[];offset=0
    for item in items:
        batch=item['batch']
        if batch.batch_size!=len(item['indices']):raise RuntimeError('Time-cache index count differs')
        for i,index in enumerate(item['indices']):
            a,b=off[offset],on[offset];offset+=1
            mask=batch.mask[i].bool()
            times=array(batch.time[i][mask])
            if len(times)!=int(batch.unpadded_length[i]) or not np.isfinite(times).all() or (np.diff(times)<=0).any():
                raise RuntimeError('Invalid cached temporal sequence')
            expected={'arrival_times':times}
            expected.update({f'condition{n}':array(getattr(batch,f'condition{n}')[i][mask]) for n in range(1,7)})
            if len(b['checkins'])!=len(times):raise RuntimeError('On output does not retain every cached event')
            for key,value in expected.items():
                if not np.array_equal(value,b[key]):raise RuntimeError(f'On output changed cached {key} at {index}')
            old_times=np.asarray(a['arrival_times'])
            positions=np.searchsorted(times,old_times)
            if (len(a['checkins'])!=len(old_times) or (positions>=len(times)).any()
                    or (np.diff(positions)<=0).any() or not np.array_equal(times[positions],old_times)):
                raise RuntimeError('Off times are not an ordered subset of cached events')
            for key,value in expected.items():
                if not np.array_equal(value[positions],a[key]):raise RuntimeError('Off conditions do not align with cached events')
            for n in range(1,7):
                key=f'condition{n}_indicator';value=array(getattr(batch,key)[i])
                if not np.array_equal(value,a[key]) or not np.array_equal(value,b[key]):
                    raise RuntimeError('Trajectory-level condition changed')
            if a.keys()!=b.keys():raise RuntimeError('Record field sets differ')
            for key in a.keys()-set(EVENT_FIELDS):
                if not np.array_equal(a[key],b[key]):raise RuntimeError('Non-event record metadata changed')
            validate_gps(a,mapping,gps,False);validate_gps(b,mapping,gps,True)
            keep=set(positions.tolist());restored=[j for j in range(len(times)) if j not in keep]
            aligned=copy.deepcopy(b)
            for key in EVENT_FIELDS:
                values=np.asarray(b[key])[positions]
                aligned[key]=values.tolist() if key=='gps' else values.copy()
            matched.append(aligned)
            rows.append(dict(index=int(index),cache_events=len(times),off_events=len(old_times),on_events=len(b['checkins']),
                off_cache_positions=positions.tolist(),restored_cache_positions=restored))
    if offset!=len(on):raise RuntimeError('Cache rows do not cover all outputs')
    restored_rows=[r for r in rows if r['restored_cache_positions']]
    return dict(rows=rows,restored_rows=restored_rows,restored_events=sum(len(r['restored_cache_positions']) for r in rows),
                off_events=sum(len(r['checkins']) for r in off),on_events=sum(len(r['checkins']) for r in on),
                on_matches_full_time_cache=True,off_is_ordered_cache_subset=True),matched


def audit_traces(items,old,new):
    if len(items)!=len(old) or len(old)!=len(new):raise RuntimeError('Trace batch count differs')
    keys=('indices','global_start','spatial_rng_before','spatial_rng_after',
          'projection_rng_after','distance_rng_before','distance_rng_after')
    for item,a,b in zip(items,old,new):
        if a['indices']!=item['indices'] or a['global_start']!=item['global_start']:raise RuntimeError('Trace/cache indices differ')
        for key in keys:
            if key not in a or key not in b or a[key]!=b[key]:raise RuntimeError('RNG/index trace differs: '+key)
        if a.get('global_rng_unchanged') is not True or b.get('global_rng_unchanged') is not True:
            raise RuntimeError('Missing global RNG preservation evidence')
        expected=list(range(36,-1,-4)) if a['effective_constraints'] else []
        if a['effective_constraints']!=b['effective_constraints']:raise RuntimeError('Effective constraints changed')
        for trace in (a,b):
            active=[c for c in trace['projection_stats'] if c['optimizer_steps']>0]
            if [c['diffusion_step'] for c in active]!=expected or any(c['optimizer_steps']!=500 for c in active):
                raise RuntimeError('Projection budget changed')
    return dict(paired_rng=True,batches=len(items),fields_checked=list(keys),projection_budget_preserved=True)


def metrics_equal(actual,stored):
    if actual.keys()!=stored.keys():raise RuntimeError('Stored/recomputed metric keys differ')
    for key,value in actual.items():
        expected=stored[key]
        if value is None and expected is None:continue
        if value is None or expected is None or not math.isfinite(value) or not math.isfinite(expected) or abs(value-expected)>1e-12:
            raise RuntimeError('Stored/recomputed metric differs: '+key)


def compare_metrics(refs,off,on,matched,mapping,indices):
    results={};conditions={};diagnostics={}
    for name,records in (('off',off),('on',on),('on_matched_events',matched)):
        results[name],conditions[name]=evaluate(refs,records,mapping,indices)
        diagnostics[name]=distribution_counts(refs,records,mapping)
    common=sorted(set(diagnostics['off']['endpoint_indices'])&set(diagnostics['on']['endpoint_indices']))
    fixed_pool=[i for i in common if min(len(refs[i]['gps']),len(off[i]['gps']),len(on[i]['gps']))>1]
    fixed={}
    for name,records in (('off',off),('on',on)):
        a=[float(travel_distance(np.asarray(records[i]['gps']))) for i in fixed_pool]
        b=[float(travel_distance(np.asarray(refs[i]['gps']))) for i in fixed_pool]
        fixed[name]=evaluation(a,b) if fixed_pool else None
    return dict(metrics=results,diagnostics=diagnostics,common_endpoint_distance=dict(
        indices=[indices[i] for i in fixed_pool],count=len(fixed_pool),jsd=fixed),
        sensitivity_note='Matched-event and common-endpoint analyses are additional diagnostics, not replacements or tuning criteria.'),conditions


def verify_snapshot(root):
    root=Path(root);seal=read(root/'snapshot-inventory.json')
    if seal.get('state')!='sealed' or seal.get('source_state')!='failed':raise RuntimeError('Unexpected source snapshot state')
    for name,expected in seal['files'].items():
        if sha256_file(checked_path(root,name))!=expected:raise RuntimeError('Snapshot file hash differs: '+name)
    status=read(root/'source/run/status.json')
    if status['state']!='failed' or status.get('error')!=EXPECTED_ERROR:raise RuntimeError('Original failure state was not preserved')
    return seal


def analyze_snapshot(root,workspace):
    root=Path(root);seal=verify_snapshot(root)
    provenance=read(root/'source/provenance.json')
    for name,expected in provenance['evaluation_code_sha256'].items():
        import hashlib
        data=checked_path(workspace,name).read_bytes().replace(b'\r\n',b'\n')
        if hashlib.sha256(data).hexdigest()!=expected:raise RuntimeError('Evaluation implementation differs: '+name)
    source=root/'source/run';manifest=read(source/'manifest.json');manifest_sha=sha256_file(source/'manifest.json')
    for name,receipt in (('off.pkl','off.receipt.json'),('on/payload.pkl','on/receipt.json'),
                         ('derived-time-cache.pkl','derived-time-cache.receipt.json')):
        if read(source/receipt)!=dict(manifest_sha256=manifest_sha,sha256=sha256_file(source/name)):
            raise RuntimeError('Saved source receipt differs: '+name)
    off=torch.load(source/'off.pkl',map_location='cpu',weights_only=False)
    on=torch.load(source/'on/payload.pkl',map_location='cpu',weights_only=False)
    cache=torch.load(source/'derived-time-cache.pkl',map_location='cpu',weights_only=False)
    bundle=torch.load(root/'source/original-inputs.pkl',map_location='cpu',weights_only=False)
    indices=manifest['indices']
    if indices!=list(range(128)) or manifest['seed']!=135398 or manifest['batch_size']!=64:
        raise RuntimeError('Unexpected diagnostic selection')
    if len(cache['batches'])!=2 or any(b['batch'].batch_size!=64 for b in cache['batches']):
        raise RuntimeError('Expected two original batches of 64')
    if off['test_indices']!=indices or on['test_indices']!=indices or cache['indices']!=indices or bundle['indices']!=indices:
        raise RuntimeError('Saved source selection differs')
    if (on['result']['state']!='complete' or on['result']['samples']!=128 or
            on['result'].get('category_consistent_decoding_version')!='sampled-category-poi-v2'):
        raise RuntimeError('On output is not a completed sampled-category-v2 result')
    if not same_records(off['sequences'],bundle['original_off_records']) or not equivalent(off['batch_traces'],bundle['original_off_traces']):
        raise RuntimeError('Off output is not the original sealed subset')
    cache_matches_original(cache['batches'],bundle['original_time_items'])
    alignment,matched=audit_alignment(cache['batches'],off['sequences'],on['sequences'],bundle['poi_category'],bundle['poi_gps'])
    traces=audit_traces(cache['batches'],off['batch_traces'],on['batch_traces'])
    if on['result']['projection_settings']!=manifest['projection_settings']:raise RuntimeError('Projection settings differ')
    comparison,conditions=compare_metrics(bundle['references'],off['sequences'],on['sequences'],matched,bundle['poi_category'],indices)
    metrics_equal(comparison['metrics']['off'],off['metrics'])
    metrics_equal(comparison['metrics']['on'],on['result']['metrics'])
    return dict(revision=REVISION,alignment=alignment,traces=traces,comparison=comparison,per_condition=conditions,
        source_manifest_sha256=manifest_sha,source_state='failed',original_failure_preserved=True,
        on_spatial_seconds=on['result']['spatial_seconds'],source_provenance=provenance,
        scope='Original time-cache integrity and alignment; NOT the original exported-length invariant or an overall quality pass.')


def report(result):
    alignment=result['alignment'];comparison=result['comparison'];metrics=comparison['metrics']
    lines=['# 类别一致解码：原时间缓存基准的独立审计','',
        '独立审计通过仅指输入完整性、缓存对齐、原始随机流及指标复算；原实验仍为 failed，不改写原导出长度审计结论。',
        'NewYork，seed135398，测试索引0–127，两个batch64；没有新采样、训练、参数搜索或三种子显著性结论。','',
        '## 数据完整性','',f"- 关闭组 {alignment['off_events']} 个事件，开启组 {alignment['on_events']} 个事件。",
        f"- 开启组完整匹配原时间缓存；关闭组为有序子集；恢复 {alignment['restored_events']} 个缓存事件。",
        '- 时间、六项逐事件条件、六项轨迹级条件、合法POI/GPS及实际类别一致性均检查通过。',
        '- 两批空间/投影/距离随机流和原10×500投影预算一致；原来源哈希保留。',
        '- 恢复事件与旧导出器删除无效POI的行为一致；旧结果未保存被删除的原始token，不声称恢复其具体token值。','',
        '| 测试索引 | 缓存事件数 | 关闭 | 开启 | 恢复位置（从0开始） |','|---|---:|---:|---:|---|']
    for row in alignment['restored_rows']:
        lines.append(f"| {row['index']} | {row['cache_events']} | {row['off_events']} | {row['on_events']} | {row['restored_cache_positions']} |")
    lines+=['','## 原样指标与固定旧事件集合的敏感性复算','',
        '最后一列从开启组副本中只保留关闭组原先保留的事件位置；不修改开启组原文件，不替代主结果。',
        '| 指标（原量纲） | 关闭原样 | 开启原样 | 开启：固定旧事件集合 |','|---|---:|---:|---:|']
    keys=('category_poi_mismatch_rate','strict_ovr','pair_coverage','category_coverage','Distance','Radius','DailyLoc','G-RANK')
    for key in keys:
        values=[metrics[name][key] for name in ('off','on','on_matched_events')]
        lines.append('| '+key+' | '+' | '.join('未定义' if v is None else f'{v:.6f}' for v in values)+' |')
    values=[comparison['diagnostics'][name]['unfiltered_distance_jsd'] for name in ('off','on','on_matched_events')]
    lines.append('| Distance（无端点筛选） | '+' | '.join('未定义' if v is None else f'{v:.6f}' for v in values)+' |')
    fixed=comparison['common_endpoint_distance']
    lines+=['','## 端点筛选与解释','',
        f"原样Distance端点入选数：关闭 {len(comparison['diagnostics']['off']['endpoint_indices'])}，开启 {len(comparison['diagnostics']['on']['endpoint_indices'])}。",
        f"两组共同端点且双方/参考均多点的固定集合含 {fixed['count']} 条；Distance JSD：关闭 {fixed['jsd']['off']}，开启 {fixed['jsd']['on']}。",
        '筛选集合与导出事件数会影响统计值，因此不能仅凭原筛选Distance下降判定几何保真改善。',
        '完整开启组的Radius与无端点筛选Distance退化；类别一致性改善不等于所有分布指标改善。',
        f"开启组原采样耗时 {result['on_spatial_seconds']:.3f} 秒；关闭组复用，未构造配对端到端加速比。",'',
        '原failed状态、完整源输出与收据在source/run中逐字节保留；审计工具、输入来源和哈希见audit.json及source/provenance.json。']
    return '\n'.join(lines)+'\n'


def verify_audit(root):
    root=Path(root);audit=read(root/'audit.json');status=read(root/'status.json')
    if (audit.get('state')!='passed' or audit.get('revision')!=REVISION or not audit.get('no_sampling')
            or audit.get('source_failure_preserved') is not True or audit.get('scientific_quality_passed') is not None
            or audit.get('source_state')!='failed' or status.get('state')!='audit-complete'):
        raise RuntimeError('Not a complete independent cache audit')
    for name,expected in audit['files'].items():
        if sha256_file(checked_path(root,name))!=expected:raise RuntimeError('Audit file changed: '+name)
    verify_snapshot(root)
    if read(root/'source/run/status.json').get('error')!=EXPECTED_ERROR:raise RuntimeError('Original failure was altered')
    return dict(state='passed',scope='independent-time-cache-audit',source_state='failed',no_sampling=True,
                verified_files=len(audit['files'])+2,audit_sha256=sha256_file(root/'audit.json'),
                source_manifest_sha256=audit['source_manifest_sha256'])
