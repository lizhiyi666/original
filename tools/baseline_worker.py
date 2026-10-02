"""Isolated baseline3/4 sampling candidate, full shard, or fixed-empty replay."""
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
from baseline_models import configure_energy
from tools.baseline_common import precision,load_task,references,validate_alignment
from tools.perfcal_common import calibration_metrics,GpuMonitor
from experiment_io import atomic_json,publish_torch,sha256_file,seed_sampling,decode_preserving_empty


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--job',required=True)
    args=parser.parse_args()
    os.chdir(ROOT)
    job=json.loads(Path(args.job).read_text())
    folder=Path(job['output_dir'])
    result=dict(state='running',method=job['method'],scale=job['scale'],temperature=job.get('temperature',0),
                reference_split=job['split'],indices=job['indices'],physical_gpu=job['gpu'])
    atomic_json(folder/'result.json',result)
    monitor=None
    try:
        precision()
        task,dm,_=load_task(job['source'],job.get('cfg_checkpoint'))
        dd=task.discrete_diffusion
        dd.projection_call_count=0
        if job['method']=='baseline3':
            configure_energy(dd,job['temperature'],job['scale'])
        elif job['method']=='baseline4':
            if not getattr(dd,'cfg_trained',False):
                raise RuntimeError('Missing CFG training proof')
            dd.cfg_scale=job['scale']
            dd.reset_cfg_stats()
        else:
            raise ValueError('Unknown baseline worker method')
        raw,selected,refs=references(dm,job['split'],job['indices'])
        def temporal(items,seed):
            seed_sampling(seed)
            batch=Batch.from_sequence_list(items).to(task.device)
            with torch.no_grad():
                time_batch=task.tpp_model.sample(batch.batch_size,x_n=batch,tmax=batch.tmax)
            maximum=min(dd.condition_encoder.max_position_embeddings,
                        (dd.transformer.positional_encoding.num_embeddings-3)//2)
            if int(time_batch.unpadded_length.max())>maximum:
                raise RuntimeError('Temporal length exceeds unchanged spatial capacity')
            time_batch=time_batch.mask_check()
            time_batch.po_matrix=batch.po_matrix
            return time_batch
        if job.get('fixture'):
            saved=torch.load(job['fixture'],map_location='cpu',weights_only=False)
            frozen=saved['batch']
            if job['indices']!=list(range(64,128)) or int(frozen.unpadded_length[35])!=0:
                raise RuntimeError('Not the validated historical empty fixture')
            batches=[copy.deepcopy(frozen).to(task.device)]
        else:
            batches=None
        generated=[]
        temporal_empty=[]
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        monitor=GpuMonitor(job['gpu'])
        started=time.perf_counter()
        for offset in range(0,len(selected),64):
            batch_seed=135398+(job['indices'][offset] if job['split']=='test' else offset)
            if batches is None:
                batch=temporal(selected[offset:offset+64],batch_seed)
            else:
                seed_sampling(batch_seed)
                batch=batches[offset//64]
            temporal_empty.extend(job['indices'][offset+i] for i in torch.where(batch.unpadded_length==0)[0].tolist())
            output=decode_preserving_empty(task,batch,raw['poi_gps'],
                baseline_method='energy_guidance' if job['method']=='baseline3' else 'cfg')
            generated.extend(output)
            atomic_json(folder/'progress.json',dict(completed_samples=len(generated),total_samples=len(selected),
                                                     elapsed_seconds=time.perf_counter()-started))
        torch.cuda.synchronize()
        elapsed=time.perf_counter()-started
        result.update(monitor.finish())
        monitor=None
        validate_alignment(generated,refs,raw['poi_category'])
        if dd.projection_call_count:
            raise RuntimeError('Baseline unexpectedly used ALM')
        if batches is not None and len(generated[35]['checkins'])!=0:
            raise RuntimeError('Fixed empty index 99 was filled')
        reserved=torch.cuda.max_memory_reserved()
        allocated=torch.cuda.max_memory_allocated()
        if job['method']=='baseline3':
            stats=dict(dd.energy_guidance.stats)
            result.update(method_executed=stats['calls']>0,guidance_stats=stats,
                all_violation_gradients_zero=stats['probes']>0 and stats['probes']==stats['zero_probes'])
        else:
            stats=dict(dd.cfg_stats)
            result.update(method_executed=stats['calls']>0,cfg_stats=stats,
                          branch_difference=stats['max_branch_difference'])
        if not result['method_executed']:
            raise RuntimeError('Requested method did not execute')
        output_path=folder/'generated.pkl'
        publish_torch(output_path,dict(sequences=generated,indices=job['indices'],reference_split=job['split'],
            method=job['method'],job=job,temporal_empty_indices=temporal_empty,
            source_checkpoint_sha256=sha256_file(Path(job['source'])/'final.ckpt'),
            cfg_checkpoint_sha256=sha256_file(job['cfg_checkpoint']) if job.get('cfg_checkpoint') else None))
        result.update(state='complete',samples=len(generated),sampling_seconds=elapsed,
            samples_per_second=len(generated)/elapsed,peak_reserved_bytes=reserved,peak_allocated_bytes=allocated,
            memory_fraction=max(reserved,allocated)/torch.cuda.get_device_properties(0).total_memory,
            output_sha256=sha256_file(output_path),projection_calls=0,
            metrics=calibration_metrics(refs,generated,raw['poi_category']))
        atomic_json(folder/'result.json',result)
        print(json.dumps(result),flush=True)
    except BaseException as exc:
        if monitor is not None:
            result.update(monitor.finish())
        result.update(state='failed',error_type=type(exc).__name__,error=str(exc))
        atomic_json(folder/'result.json',result)
        raise


if __name__=='__main__':
    main()
