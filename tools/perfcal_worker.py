"""One isolated GPU calibration candidate. Never reads test conditions for sampling."""
import argparse
import contextlib
import gc
import io
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MPLBACKEND', 'Agg')
import torch
from datamodule import Batch
from evaluate_utils import get_task, get_run_data
from experiment_io import (atomic_json, publish_torch, sha256_file, seed_sampling,
                           strict_test_matrix, decode_preserving_empty, validate_sequences)
from constraint_projection import ConstraintProjection
from tools.perfcal_common import calibration_metrics, projector_kwargs, reference_projector_class, GpuMonitor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True)
    args = parser.parse_args()
    job = json.loads(Path(args.job).read_text())
    folder = Path(job['output_dir']).resolve()
    os.chdir(ROOT)
    result = dict(label=job['label'], kind=job['kind'], temperature=job.get('temperature'),
                  batch_size=job.get('batch_size'), physical_gpu=job['physical_gpu'],
                  engineering_calibration=True, independent_validation=False, state='running')
    monitor = None
    atomic_json(folder/'result.json', result)
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_num_threads(4)
        load_started = time.perf_counter()
        _, _, run_path = get_run_data(job['source_run_id'], ROOT/'wandb')
        task, dm = get_task(run_path, str(ROOT), job['checkpoint'])
        if str(task.device) == 'cpu':
            raise RuntimeError('CUDA is required')
        raw = torch.load(ROOT/f"data/{job['profile']['dataset']}/{job['profile']['dataset']}_train.pkl",
                         map_location='cpu', weights_only=False)
        indices = job['indices']
        if len(indices) != len(set(indices)) or min(indices) < 0 or max(indices) >= len(raw['sequences']):
            raise ValueError('Invalid training calibration indices')
        sequences = [dm.train_data.sequences[i] for i in indices]
        references = [raw['sequences'][i] for i in indices]
        for seq, ref in zip(sequences, references):
            seq.po_matrix = strict_test_matrix(ref, raw['poi_category'], raw['category_mapping'])
        dd = task.discrete_diffusion
        profile = job['profile']
        dd.projection_frequency = profile['projection']['projection_frequency']
        dd.projection_last_k_steps = profile['projection']['projection_last_k_steps']
        dd.projection_call_count = 0
        kwargs = projector_kwargs(dd, profile, job.get('temperature', 0.1))
        p = ConstraintProjection(**kwargs, device=str(task.device), collect_diagnostics=True, verbose=False)
        if p.projection_distance_kl_weight > 0:
            if job['kind'] == 'regression':
                raise ValueError('Frozen v1 equivalence checks require distance KL to be disabled')
            from distance_kl import attach_distance_reference
            attach_distance_reference(p, dm, profile['seed'])
        dd.constraint_projector = p
        result['load_seconds'] = time.perf_counter() - load_started
        result['checkpoint_sha256'] = sha256_file(job['checkpoint'])
        result['precision'] = dict(dtype='float32', autocast=False, matmul_tf32=False, cudnn_tf32=False)

        def temporal_batch(selected, seed):
            seed_sampling(seed)
            batch = Batch.from_sequence_list(selected).to(task.device)
            with torch.no_grad():
                temporal = task.tpp_model.sample(batch.batch_size, x_n=batch, tmax=batch.tmax)
            maximum = min(dd.condition_encoder.max_position_embeddings,
                          (dd.transformer.positional_encoding.num_embeddings-3)//2)
            if int(temporal.unpadded_length.max()) > maximum:
                raise RuntimeError('Generated length exceeds unchanged model capacity')
            temporal = temporal.mask_check()
            temporal.po_matrix = batch.po_matrix
            return temporal

        warm_started = time.perf_counter()
        reject_multiple = task.tpp_model.intensity_model.rejections_sample_multiple
        dd.use_constraint_projection = False
        warm = temporal_batch(sequences[:4], profile['seed'])
        decode_preserving_empty(task, warm, raw['poi_gps'])
        del warm
        task.tpp_model.intensity_model.rejections_sample_multiple = reject_multiple
        torch.cuda.synchronize()
        result['warmup_seconds'] = time.perf_counter()-warm_started
        gc.collect()
        dd.use_constraint_projection = True

        if job['kind'] == 'regression':
            snapshot = {}
            class Captured(Exception):
                pass
            def capture(logits, a, b, mask, constraint_mask=None):
                snapshot.update(logits=logits.detach().clone(), a=a.clone(), b=b.clone(),
                                mask=mask.clone(), constraint_mask=constraint_mask.clone() if constraint_mask is not None else None,
                                cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state())
                raise Captured()
            p.project_with_matrices = capture
            temporal = temporal_batch(sequences[:64], profile['seed'])
            try:
                dd.sample_fast(temporal)
            except Captured:
                pass
            if not snapshot:
                raise RuntimeError('No projection input captured')
            reference_cls = reference_projector_class(ROOT)

            def trial(cls, temperature=None):
                options = dict(kwargs)
                if cls is not ConstraintProjection:
                    for key in ('projection_distance_kl_weight', 'distance_paths', 'distance_topk',
                                'distance_bins', 'distance_temperature'):
                        options.pop(key, None)
                if temperature is not None:
                    options['gumbel_temperature'] = temperature
                instance = cls(**options, device=str(task.device))
                if isinstance(instance, ConstraintProjection):
                    instance.verbose = False
                torch.set_rng_state(snapshot['cpu_rng'])
                torch.cuda.set_rng_state(snapshot['cuda_rng'])
                torch.cuda.synchronize()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                wall = time.perf_counter()
                with contextlib.redirect_stdout(io.StringIO()):
                    output = instance.project_with_matrices(snapshot['logits'], snapshot['a'], snapshot['b'],
                        snapshot['mask'], snapshot['constraint_mask'])
                end.record()
                torch.cuda.synchronize()
                return (output, time.perf_counter()-wall, start.elapsed_time(end)/1000,
                        instance.last_projection_stats, torch.get_rng_state(), torch.cuda.get_rng_state())

            equivalence = []
            for temperature in profile['temperatures']:
                before, _, _, before_stats, before_cpu, before_cuda = trial(reference_cls, temperature)
                after, _, _, after_stats, after_cpu, after_cuda = trial(ConstraintProjection, temperature)
                torch.testing.assert_close(after, before, rtol=1e-5, atol=1e-6)
                if any(before_stats[k] != after_stats[k] for k in
                       ('optimizer_steps','outer_iterations','active_constraints','inactive_multiplier_max')):
                    raise RuntimeError('Optimization/early-stop behavior changed')
                if not torch.equal(before_cpu, after_cpu) or not torch.equal(before_cuda, after_cuda):
                    raise RuntimeError('Projection random-number consumption changed')
                equivalence.append(dict(temperature=temperature,
                    max_abs_diff=float((before-after).abs().max()),
                    reference_stats=before_stats, optimized_stats=after_stats, rng_equal=True))
                del before, after
            timings = {'reference': [], 'optimized': []}
            gpu_timings = {'reference': [], 'optimized': []}
            for _ in range(profile['projection_timing_repeats']):
                for name, cls in (('reference',reference_cls),('optimized',ConstraintProjection)):
                    output, wall, gpu, _, _, _ = trial(cls)
                    timings[name].append(wall)
                    gpu_timings[name].append(gpu)
                    del output
            result.update(state='complete', numerical_max_abs_diff=max(r['max_abs_diff'] for r in equivalence),
                          equivalence_by_temperature=equivalence,
                          wall_seconds_trials=timings, gpu_seconds_trials=gpu_timings,
                          wall_seconds_median={k:statistics.median(v) for k,v in timings.items()},
                          gpu_seconds_median={k:statistics.median(v) for k,v in gpu_timings.items()})
        else:
            original_project = p.project_with_matrices
            projection_times = []
            probe_stats = []
            def profiled(*values, **options):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                wall = time.perf_counter()
                output = original_project(*values, **options)
                projection_wall = time.perf_counter()-wall
                end.record()
                projection_times.append((start,end,projection_wall))
                probe_stats.append(dict(p.last_projection_stats))
                return output
            p.project_with_matrices = profiled
            dd.projection_call_count = 0
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            monitor = GpuMonitor(job['physical_gpu'])
            generated = []
            temporal_empty_indices = []
            started = time.perf_counter()
            for offset in range(0,len(indices),job['batch_size']):
                selected = sequences[offset:offset+job['batch_size']]
                temporal = temporal_batch(selected, profile['seed']+offset)
                temporal_empty_indices.extend(indices[offset+i] for i in torch.where(temporal.unpadded_length==0)[0].tolist())
                output = decode_preserving_empty(task,temporal,raw['poi_gps'])
                if len(output)!=len(selected):
                    raise ValueError('Lost samples during calibration')
                generated.extend(output)
                atomic_json(folder/'progress.json',dict(completed_samples=len(generated),
                    total_samples=len(indices), elapsed_seconds=time.perf_counter()-started,
                    projection_calls=len(projection_times)))
                del temporal, output
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-started
            result.update(monitor.finish())
            monitor = None
            validate_sequences(generated,raw['poi_category'])
            if len(generated)!=len(indices):
                raise ValueError('Calibration output count mismatch')
            metrics = calibration_metrics(references,generated,raw['poi_category'])
            probes = sum(s.get('unmet_probe_count',0) for s in probe_stats)
            zeros = sum(s.get('zero_gradient_probe_count',0) for s in probe_stats)
            grad_sum = sum(s.get('gradient_norm_sum',0) for s in probe_stats)
            allocated, reserved = torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved()
            total = torch.cuda.get_device_properties(0).total_memory
            output_path=folder/'generated.pkl'
            publish_torch(output_path,dict(sequences=generated,reference_indices=indices,reference_split='train',
                calibration=True,independent_validation=False,temporal_empty_indices=temporal_empty_indices,
                source_checkpoint_sha256=result['checkpoint_sha256'],seed=profile['seed']))
            result.update(state='complete',samples=len(generated),reference_indices=indices,
                          sampling_seconds=elapsed,samples_per_second=len(generated)/elapsed,
                          projection_wall_seconds=sum(t[2] for t in projection_times),
                          projection_gpu_seconds=sum(t[0].elapsed_time(t[1]) for t in projection_times)/1000,
                          projection_calls=len(projection_times),optimizer_steps=sum(s['optimizer_steps'] for s in probe_stats),
                          unmet_gradient_probes=probes,zero_gradient_probes=zeros,
                          zero_gradient_ratio=zeros/probes if probes else 0.0,
                          violating_probes_all_zero=probes>0 and zeros==probes,
                          gradient_norm_mean=grad_sum/probes if probes else 0.0,
                          gradient_norm_max=max((s.get('gradient_norm_max',0) for s in probe_stats),default=0.0),
                          peak_allocated_bytes=allocated,peak_reserved_bytes=reserved,total_device_bytes=total,
                          memory_fraction=max(allocated,reserved)/total,metrics=metrics,
                          output_sha256=sha256_file(output_path),temporal_empty_count=len(temporal_empty_indices))
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
