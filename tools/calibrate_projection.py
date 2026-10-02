"""Fixed train-only perfcal-v1 sweep. Does NOT launch a full OOD experiment."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import wandb
from omegaconf import OmegaConf
from experiment_io import atomic_json,sha256_file
from tools.perfcal_common import temperature_choice,batch_choice
from tools.run_newyork_ood import code_fingerprint


def report_text(manifest, results, recommendation, reasons, gpu_verified):
    lines=['# perfcal-v1 工程校准报告','',
           '**数据来自模型已经见过的训练集，不是独立验证集；未重跑 OOD 测试集。**','',
           '固定投影预算：10×50；其余窗口、步长和权重保持不变。','',
           '| 阶段 | GPU | T | batch | 状态 | 条/秒 | 显存峰值 GiB（reserved） | 严格 OVR | 类别对覆盖率 | 零梯度比例 |',
           '|---|---:|---:|---:|---|---:|---:|---:|---:|---:|']
    for r in results:
        if r['kind']=='regression':
            continue
        if r['state']=='complete':
            lines.append(f"| {r['kind']} | {r['physical_gpu']} | {r['temperature']} | {r['batch_size']} | 完成 | {r['samples_per_second']:.3f} | {r['peak_reserved_bytes']/2**30:.2f} | {r['metrics']['strict_ovr']:.2%} | {r['metrics']['pair_coverage']:.2%} | {r['zero_gradient_ratio']:.2%} |")
        else:
            lines.append(f"| {r['kind']} | {r['physical_gpu']} | {r.get('temperature')} | {r.get('batch_size')} | {r.get('error_type','失败')} | — | — | — | — | — |")
    lines+=['','## 数值回归和纯投影性能','']
    for r in results:
        if r['kind']=='regression':
            if r['state']=='complete':
                med=r['wall_seconds_median']
                lines.append(f"- 全部候选温度的实际输入最大数值差：{r['numerical_max_abs_diff']:.3g}；优化步数、提前停止和随机数消耗一致。")
                lines.append(f"- 同为10×50，三次中位数：参考实现 {med['reference']:.4f}s，优化实现 {med['optimized']:.4f}s，倍率 {med['reference']/med['optimized']:.3f}×。")
            else:
                lines.append(f"- 回归未通过：{r.get('error','unknown')}")
    lines+=['','## 完整采样与资源明细','',
            '| 候选 | 采样秒 | 投影秒（GPU事件） | 优化步数 | allocated / reserved GiB | GPU利用率均值 / 峰值 | 约束梯度范数均值 | 空输出率 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in results:
        if r['kind']!='regression' and r['state']=='complete':
            utilization = (f"{r['gpu_utilization_mean']:.1f}% / {r['gpu_utilization_peak']:.1f}%"
                           if r.get('gpu_utilization_mean') is not None else '不可用')
            lines.append(f"| {r['label']} | {r['sampling_seconds']:.2f} | {r['projection_gpu_seconds']:.2f} | {r['optimizer_steps']} | {r['peak_allocated_bytes']/2**30:.2f} / {r['peak_reserved_bytes']/2**30:.2f} | {utilization} | {r['gradient_norm_mean']:.3g} | {r['metrics']['empty_rate']:.2%} |")
    lines+=['','梯度探针仅统计每次投影首次更新前、有效且尚未满足约束的行，使用约束项梯度（不含 KL）。',
            '加载与预热不计入采样耗时，分别记录于各候选 result.json；三次中位数仅用于同为 10×50 的短投影比较。']
    lines+=['','## 推荐','']
    if recommendation and gpu_verified:
        lines.append(f"推荐采样 batch={recommendation['batch_size']}，投影温度={recommendation['temperature']}，outer=10，inner=50。")
        lines.append('已在第二张 GPU 复核数值稳定性、质量门槛和显存峰值。')
    else:
        lines.append('目前没有通过全部门槛和第二张 GPU 复核的可推荐配置；不自动启动完整实验。')
    lines+=['','## 筛选记录','']
    for label, failures in reasons.items():
        lines.append(f"- {label}: {'; '.join(failures) if failures else '通过主卡门槛'}")
    lines+=['','训练 batch、训练数据、检查点均未改动。采样 batch 会影响目标函数尺度和随机数分组，结果不能视为纯硬件差异。',
            '本报告不保证 OOD 严格违反率达到历史 55%；下一步完整测试需要用户另行确认。','']
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-experiment',required=True)
    parser.add_argument('--profile',default=str(ROOT/'config/calibration/perfcal-v1.yaml'))
    parser.add_argument('--output-dir',default=str(ROOT/'calibration_runs/perfcal-v1'))
    parser.add_argument('--indices',help='Optional JSON list of exactly 512 unique training indices')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    os.chdir(ROOT)
    output=Path(args.output_dir).resolve()
    output.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(output/'calibration.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    profile=OmegaConf.to_container(OmegaConf.load(args.profile),resolve=True)
    if (profile['reference_split']!='train' or profile['projection']['projection_outer_iters']!=10
            or profile['projection']['projection_inner_iters']!=50):
        raise ValueError('This calibration is restricted to train-only, 10x50')
    source=Path(args.source_experiment).resolve()
    source_manifest=json.loads((source/'manifest.json').read_text())
    checkpoint=source/'final.ckpt'
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if saved['epoch']!=999:
        raise RuntimeError('Expected completed 1000-epoch checkpoint')
    del saved
    train_path=ROOT/f"data/{profile['dataset']}/{profile['dataset']}_train.pkl"
    raw=torch.load(train_path,map_location='cpu',weights_only=False)
    count=len(raw['sequences'])
    del raw
    indices=(json.loads(Path(args.indices).read_text()) if args.indices else
             np.random.default_rng(profile['seed']).choice(count,profile['pool_size'],replace=False).tolist())
    if len(indices)!=profile['pool_size'] or len(set(indices))!=len(indices) or min(indices)<0 or max(indices)>=count:
        raise ValueError('Invalid calibration pool')
    hashes=code_fingerprint()
    for p in ('tools/calibrate_projection.py','tools/perfcal_common.py','tools/perfcal_worker.py',
              'tests/fixtures/projection_reference_f70beb4.py','config/calibration/perfcal-v1.yaml'):
        hashes[p]=sha256_file(ROOT/p)
    manifest=dict(revision='perfcal-v1',reference_split='train',in_sample=True,independent_validation=False,
                  source_run_id=source_manifest['run_id'],source_manifest_sha256=sha256_file(source/'manifest.json'),
                  checkpoint_sha256=sha256_file(checkpoint),train_sha256=sha256_file(train_path),
                  profile=profile,indices=indices,temperature_indices=indices[:profile['temperature_size']],
                  code_sha256=hashes,precision='float32; no autocast; TF32 disabled',
                  packages={name:importlib.metadata.version(name) for name in ('torch','pytorch-lightning','numpy','wandb')})
    if manifest['train_sha256']!=source_manifest['data_sha256']['train']:
        raise RuntimeError('Calibration training data differs from original data')
    manifest_path=output/'manifest.json'
    if manifest_path.exists():
        if not args.resume or json.loads(manifest_path.read_text())!=manifest:
            raise RuntimeError('Existing calibration requires identical inputs/code and explicit --resume')
    elif args.resume:
        raise RuntimeError('Cannot resume without original manifest')
    else:
        atomic_json(manifest_path,manifest)
        atomic_json(output/'indices.json',indices)
    run=wandb.init(project='Marionette',entity=source_manifest['entity'],
                   id=f"{source_manifest['run_id']}-perfcal-v1",name='perfcal-v1-train-engineering-calibration',
                   group=source_manifest['run_id'],job_type='engineering-calibration',mode='online',
                   resume='allow',dir=str(output),save_code=False,
                   config={'in_sample':True,'independent_validation':False,'profile':profile,
                           'source_checkpoint_sha256':manifest['checkpoint_sha256'],'train_indices':indices})
    results=[]
    phase='precheck'
    def status(state, **extra):
        atomic_json(output/'status.json',dict(state=state,phase=phase,updated_at=time.time(),pid=os.getpid(),**extra))
        print(json.dumps(dict(state=state,phase=phase,**extra)),flush=True)
    def execute(label,kind,temperature,batch_size,n,gpu):
        nonlocal phase
        phase=label
        status('running')
        folder=output/label
        folder.mkdir(exist_ok=True)
        job=dict(label=label,kind=kind,temperature=temperature,batch_size=batch_size,
                 indices=indices[:n],physical_gpu=gpu,source_run_id=source_manifest['run_id'],
                 checkpoint=str(checkpoint),profile=profile,output_dir=str(folder))
        path=folder/'job.json'
        if path.exists() and json.loads(path.read_text())!=job:
            raise RuntimeError('Candidate settings changed')
        atomic_json(path,job)
        result_path=folder/'result.json'
        if args.resume and result_path.exists():
            r=json.loads(result_path.read_text())
            if r['state'] not in ('complete','failed'):
                raise RuntimeError('Interrupted candidate needs a new isolated candidate directory')
        else:
            with (folder/'worker.log').open('w') as log:
                process=subprocess.run([sys.executable,'-u','-B','tools/perfcal_worker.py','--job',str(path)],
                    cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONUNBUFFERED='1',
                                     CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4',MPLBACKEND='Agg'),
                    stdout=log,stderr=subprocess.STDOUT)
            r=(json.loads(result_path.read_text()) if result_path.exists() else
               dict(label=label,kind=kind,temperature=temperature,batch_size=batch_size,physical_gpu=gpu,
                    state='failed',error_type='WorkerExit',error=f'exit {process.returncode}'))
            if process.returncode!=0 and r['state']=='complete':
                raise RuntimeError('Worker/result status contradiction')
        results.append(r)
        atomic_json(output/'results.json',results)
        scalars={f"calibration/{k}":v for k,v in r.items() if isinstance(v,(int,float,str,bool))}
        scalars.update({f"calibration/{k}":v for k,v in r.get('metrics',{}).items()})
        run.log(scalars)
        print(json.dumps({'candidate':label,'state':r['state'],'speed':r.get('samples_per_second'),
                          'metrics':r.get('metrics'),'error':r.get('error')}),flush=True)
        return r
    recommendation=None
    reasons={}
    gpu_verified=False
    try:
        # Preserve all prior safety regressions; full test set is never sampled here.
        with (output/'unit-tests.log').open('w') as log:
            subprocess.run([sys.executable,'-B','-m','unittest','discover','-s','tests','-v'],
                           stdout=log,stderr=subprocess.STDOUT,check=True)
        regression=execute('numerical-regression','regression',0.1,64,64,profile['primary_gpu'])
        if regression['state']!='complete':
            raise RuntimeError('Numerical equivalence gate failed; sweep not started')
        temperatures=[execute(f"temperature-{str(t).replace('.','p')}",'temperature',t,64,
                              profile['temperature_size'],profile['primary_gpu']) for t in profile['temperatures']]
        chosen_t=temperature_choice(temperatures)
        if chosen_t:
            batches=[execute(f'batch-{b}','batch',chosen_t['temperature'],b,profile['pool_size'],profile['primary_gpu'])
                     for b in profile['batch_sizes']]
            chosen_b,reasons=batch_choice(batches,profile)
            if chosen_b:
                verification=execute('second-gpu-verification','verification',chosen_t['temperature'],
                                     chosen_b['batch_size'],profile['pool_size'],profile['verification_gpu'])
                baseline=next(r for r in batches if r['batch_size']==64)
                gpu_verified=(verification['state']=='complete'
                    and verification['memory_fraction']<=profile['memory_fraction_limit']
                    and verification['metrics']['strict_ovr']<=baseline['metrics']['strict_ovr']+profile['strict_ovr_tolerance']
                    and verification['metrics']['pair_coverage']>=baseline['metrics']['pair_coverage']-profile['pair_coverage_tolerance'])
                recommendation=dict(batch_size=chosen_b['batch_size'],temperature=chosen_t['temperature'],
                                    projection_outer_iters=10,projection_inner_iters=50) if gpu_verified else None
        else:
            reasons['temperature_selection']=['No finite candidate with informative violating-constraint gradients']
        if sha256_file(checkpoint)!=manifest['checkpoint_sha256'] or sha256_file(train_path)!=manifest['train_sha256']:
            raise RuntimeError('Checkpoint or training data changed during calibration')
        atomic_json(output/'recommendation.json',dict(recommendation=recommendation,second_gpu_verified=gpu_verified,
                                                     rejection_reasons=reasons,in_sample=True))
        (output/'report.md').write_text(report_text(manifest,results,recommendation,reasons,gpu_verified),encoding='utf-8')
        run.summary['calibration_complete']=True
        run.summary['recommendation']=recommendation or 'No qualified configuration'
        run.summary['independent_validation']=False
        run.finish(exit_code=0)
        remote=wandb.Api(timeout=30).run(f"{source_manifest['entity']}/Marionette/{run.id}")
        if remote.summary.get('calibration_complete') is not True:
            raise RuntimeError('W&B calibration summary readback failed')
        phase='complete'
        status('complete',recommendation=recommendation,wandb_url=run.url,full_ood_sampling_started=False)
    except BaseException as exc:
        run.finish(exit_code=1)
        status('failed',error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        lock.close()


if __name__=='__main__':
    main()
