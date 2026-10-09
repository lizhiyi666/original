"""Isolated synthetic benchmark. Never loads checkpoints or writes experiment outputs.

python -B tools/benchmark_distance_kl.py --device cuda --backend legacy --output tmp/legacy.json
Two warmups and five synchronized measurements by default; no profiler in timed runs.
"""
import argparse
from contextlib import nullcontext
import gc
import json
from pathlib import Path
import statistics
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import distance_kl
from experiment_io import atomic_json, sha256_file
from tools.distance_benchmark_fixture import fixture, make_projector


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def measure(setup, device, warmups, repeats):
    samples, allocated, reserved = [], [], []
    for index in range(warmups + repeats):
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        run = setup()
        sync(device)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        begin = time.perf_counter()
        value = run()
        sync(device)
        elapsed = time.perf_counter() - begin
        if device.type == 'cuda':
            peak_a, peak_r = torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device)
            if peak_r > .8 * torch.cuda.get_device_properties(device).total_memory:
                raise RuntimeError('Benchmark exceeded 80% GPU reserved-memory ceiling; no smaller-batch fallback')
        else:
            peak_a = peak_r = None
        if index >= warmups:
            samples.append(elapsed)
            allocated.append(peak_a)
            reserved.append(peak_r)
        del value, run
    return dict(seconds=samples, median_seconds=statistics.median(samples),
                peak_allocated_bytes=max(allocated) if allocated[0] is not None else None,
                peak_reserved_bytes=max(reserved) if reserved[0] is not None else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--backend', choices=('legacy', 'batched'), default='legacy')
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--max-pois', type=int, default=16)
    parser.add_argument('--candidates', type=int, default=64)
    parser.add_argument('--seed', type=int, default=135398)
    parser.add_argument('--warmups', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite a previous benchmark')
    if args.warmups < 0 or args.repeats < 1:
        raise ValueError('Invalid measurement counts')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    cls_name = 'DistanceObjective' if args.backend == 'legacy' else 'BatchedDistanceObjective'
    cls = getattr(distance_kl, cls_name)
    # Also supports the untouched pre-optimization projector for baseline snapshots.
    import inspect
    backend = args.backend if 'distance_backend' in inspect.signature(make_projector.__globals__['ConstraintProjection']).parameters else None
    report = dict(state='running', fixture='synthetic-ragged-v1-not-production-replay',
        config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        torch_version=torch.__version__, threads=1, dtype='float32', tf32=False,
        gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        source_sha256={p: sha256_file(ROOT / p) for p in
            ('distance_kl.py', 'constraint_projection.py', 'tools/benchmark_distance_kl.py',
             'tools/distance_benchmark_fixture.py')}, timings={})
    try:
        d = fixture(device, args.batch, args.max_pois, args.candidates, args.seed)
        stochastic = not args.deterministic
        def objective():
            return cls(d['reference'], d['y'], d['poi'], d['active'])
        def loss_setup():
            obj = objective()
            noise = obj.noise(distance_kl.distance_generator(args.seed, device), stochastic)
            y = d['y'].clone().requires_grad_()
            def run():
                loss = obj.loss(y, noise)
                loss.backward()
                return loss
            return run
        def projection_setup(include_initialization):
            p, inputs = make_projector(d, backend, stochastic)
            obj = None if include_initialization else objective()
            def run():
                context = nullcontext() if include_initialization else patch('distance_kl.' + cls_name, return_value=obj)
                with context:
                    result = p.project_with_matrices(*inputs, poi_mask=d['poi'])
                if p.last_projection_stats['optimizer_steps'] != 500 or not torch.isfinite(result).all():
                    raise RuntimeError('Incomplete/nonfinite full projection')
                return result
            return run
        for label, setup in (
                ('distance_forward_backward', loss_setup),
                ('projection_prepared_geometry', lambda: projection_setup(False)),
                ('projection_including_geometry', lambda: projection_setup(True))):
            report['timings'][label] = measure(setup, device, args.warmups, args.repeats)
            atomic_json(args.output, report)
            print(json.dumps({label: report['timings'][label]}), flush=True)
        report['state'] = 'complete'
    except (RuntimeError, MemoryError) as exc:
        report.update(state='incomplete', error=str(exc), no_batch_reduction=True)
        raise
    finally:
        atomic_json(args.output, report)


if __name__ == '__main__':
    main()
