"""Spatial-only PO-CFG training; temporal weights are frozen and fingerprinted."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
import wandb
from tools.baseline_common import (precision,load_task,fresh_cfg,state_hash,atomic_checkpoint,
                                   references,rng_state,restore_rng,CFG_FORMAT)
from experiment_io import atomic_json,sha256_file


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--job',required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    os.chdir(ROOT)
    job=json.loads(Path(args.job).read_text())
    folder=Path(job['output_dir'])
    precision()
    task,dm,run_path=load_task(job['source'])
    temporal_hash=state_hash(task.tpp_model)
    dd=fresh_cfg(task,dm,run_path)
    _,_,_=references(dm,'train',list(range(len(dm.train_data.sequences))))
    if len(dm.train_data.sequences)!=3160 or dm.batch_size!=64:
        raise RuntimeError('CFG budget requires 3160 training sequences at batch 64')
    dd.train()
    optimizer=torch.optim.AdamW(dd.parameters(),lr=.001,weight_decay=0)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=.95,patience=1000)
    checkpoint_path=folder/'last.ckpt'
    epoch_start=0
    global_step=0
    pending_rng=None
    prior_seconds=0.0
    if checkpoint_path.exists():
        if not args.resume:
            raise RuntimeError('Existing CFG training requires explicit --resume')
        saved=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
        if saved['job']!=job or saved['temporal_sha256']!=temporal_hash:
            raise RuntimeError('Training resume inputs changed')
        dd.load_state_dict(saved['spatial_state_dict'],strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        dd.cfg_stats.update(saved['branch_counts'])
        epoch_start=saved['epoch']+1
        global_step=saved['global_step']
        pending_rng=saved['rng']
        prior_seconds=saved.get('training_seconds',0.0)
    run=wandb.init(project='Marionette',entity=job['entity'],id=job['wandb_id'],
        name=job['wandb_id'],group=job['group'],job_type='cfg-preflight' if job['epochs']==2 else 'cfg-training',
        mode='online',resume='allow',dir=str(folder),save_code=False,config=job)
    started=time.perf_counter()
    if pending_rng is not None:
        restore_rng(pending_rng)
    try:
        for epoch in range(epoch_start,job['epochs']):
            losses=[]
            for batch in dm.train_dataloader():
                batch=batch.to(task.device)
                optimizer.zero_grad(set_to_none=True)
                loss,_=dd.training_losses(batch)
                loss=loss.mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite CFG training loss')
                loss.backward()
                if not torch.stack([torch.isfinite(p.grad).all() for p in dd.parameters() if p.grad is not None]).all():
                    raise FloatingPointError('Non-finite CFG gradient')
                optimizer.step()
                if not torch.stack([torch.isfinite(p).all() for p in dd.parameters()]).all():
                    raise FloatingPointError('Non-finite CFG updated parameter')
                scheduler.step(loss.detach())
                losses.append(float(loss.detach()))
                global_step+=1
            if len(losses)!=50 or any(p.grad is not None for p in task.tpp_model.parameters()):
                raise RuntimeError('CFG training budget/frozen temporal invariant failed')
            if min(dd.cfg_stats['conditional_rows'],dd.cfg_stats['null_rows'])<=0:
                raise RuntimeError('Both CFG training branches must receive samples')
            saved=dict(format=CFG_FORMAT,job=job,epoch=epoch,global_step=global_step,
                trained=job['epochs']==1000 and epoch==999,spatial_state_dict=dd.state_dict(),
                optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),rng=rng_state(),
                temporal_sha256=temporal_hash,branch_counts=dict(dd.cfg_stats),
                training_seconds=prior_seconds+time.perf_counter()-started)
            atomic_checkpoint(checkpoint_path,saved)
            progress=dict(state='running',epoch=epoch+1,target_epochs=job['epochs'],global_step=global_step,
                loss=sum(losses)/len(losses),elapsed_seconds=prior_seconds+time.perf_counter()-started,
                conditional_rows=dd.cfg_stats['conditional_rows'],null_rows=dd.cfg_stats['null_rows'])
            atomic_json(folder/'progress.json',progress)
            run.log(progress)
            if epoch<2 or (epoch+1)%25==0:
                print(json.dumps(progress),flush=True)
        if state_hash(task.tpp_model)!=temporal_hash or global_step!=job['epochs']*50:
            raise RuntimeError('Final CFG frozen weights/update count mismatch')
        result=dict(state='complete',epochs=job['epochs'],global_step=global_step,
            checkpoint_sha256=sha256_file(checkpoint_path),temporal_sha256=temporal_hash,
            branch_counts=dict(dd.cfg_stats),training_seconds=prior_seconds+time.perf_counter()-started)
        run.summary.update(result)
        run.summary['complete']=True
        run.finish(exit_code=0)
        remote=wandb.Api(timeout=30).run(f"{job['entity']}/Marionette/{job['wandb_id']}")
        if remote.summary.get('complete') is not True:
            raise RuntimeError('CFG training W&B readback failed')
        result['wandb_url']=run.url
        atomic_json(folder/'result.json',result)
    except BaseException as exc:
        run.finish(exit_code=1)
        atomic_json(folder/'result.json',dict(state='failed',error_type=type(exc).__name__,error=str(exc)))
        raise


if __name__=='__main__':
    main()
