"""Metrics/selection for engineering calibration, not independent validation."""
import importlib.util
import math
from pathlib import Path
import statistics
import subprocess
import threading

import numpy as np
from evaluations.ovr import (dataset_ovr_with_coverage, _seq_cats_order,
                             _extract_reference_pairs_for_sequence,
                             _violation_rate_for_pair_in_generated)


def reference_projector_class(root):
    path = Path(root) / 'tests/fixtures/projection_reference_f70beb4.py'
    spec = importlib.util.spec_from_file_location('perfcal_frozen_reference', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ConstraintProjection


def projector_kwargs(dd, profile, temperature):
    p = profile['projection']
    return dict(num_classes=dd.num_classes, type_classes=dd.type_classes, num_spectial=dd.num_spectial,
                tau=p['projection_tau'], lambda_init=p['projection_lambda'], mu_init=p['projection_mu'],
                mu_alpha=p['projection_mu_alpha'], mu_max=p['projection_mu_max'],
                outer_iterations=p['projection_outer_iters'], inner_iterations=p['projection_inner_iters'],
                eta=p['projection_eta'], delta_tol=p['projection_delta_tol'],
                projection_existence_weight=p['projection_existence_weight'],
                use_gumbel_softmax=p['use_gumbel_softmax'], gumbel_temperature=temperature)


def calibration_metrics(reference, generated, poi_category):
    if len(reference) != len(generated) or not reference:
        raise ValueError('Calibration count mismatch')
    skip, strict, cat_coverage = dataset_ovr_with_coverage(reference, generated, poi_category)
    total = present = 0
    for target, sample in zip(reference, generated):
        pairs = _extract_reference_pairs_for_sequence(_seq_cats_order(target, poi_category))
        cats = _seq_cats_order(sample, poi_category)
        total += len(pairs)
        present += sum(_violation_rate_for_pair_in_generated(cats, a, b)[1] > 0 for a, b in pairs)
    metrics = dict(strict_ovr=float(strict), pair_coverage=present/total if total else float('nan'),
                   category_coverage=float(cat_coverage), ovr_skip=float(skip),
                   empty_count=sum(len(s['checkins']) == 0 for s in generated),
                   empty_rate=sum(len(s['checkins']) == 0 for s in generated)/len(generated))
    if not all(math.isfinite(x) for x in metrics.values()):
        raise FloatingPointError('Undefined/non-finite calibration metrics; not a valid candidate')
    return metrics


def temperature_choice(results):
    eligible = [r for r in results if r.get('state') == 'complete'
                and not r.get('violating_probes_all_zero', True)]
    if not eligible:
        return None
    return min(eligible, key=lambda r: (r['metrics']['strict_ovr'], -r['metrics']['pair_coverage'], r['sampling_seconds']))


def batch_choice(results, profile):
    reference = next((r for r in results if r.get('state') == 'complete' and r['batch_size'] == 64), None)
    if reference is None:
        return None, {r['label']: ['missing valid batch-64 reference'] for r in results}
    reasons, eligible = {}, []
    for r in results:
        failures = []
        if r.get('state') != 'complete':
            failures.append(r.get('error_type', 'failed'))
        else:
            if r['memory_fraction'] > profile['memory_fraction_limit']:
                failures.append('memory peak above 80% limit')
            if r['metrics']['strict_ovr'] > reference['metrics']['strict_ovr'] + profile['strict_ovr_tolerance']:
                failures.append('strict OVR worsened by more than 1 percentage point')
            if r['metrics']['pair_coverage'] < reference['metrics']['pair_coverage'] - profile['pair_coverage_tolerance']:
                failures.append('pair coverage dropped by more than 1 percentage point')
        reasons[r['label']] = failures
        if not failures:
            eligible.append(r)
    if not eligible:
        return None, reasons
    fastest = max(r['samples_per_second'] for r in eligible)
    near_best = [r for r in eligible if r['samples_per_second'] >= fastest * (1-profile['throughput_tie_fraction'])]
    return min(near_best, key=lambda r: r['batch_size']), reasons


class GpuMonitor:
    """One persistent nvidia-smi sampler, rather than spawning a process per reading."""
    def __init__(self, physical_gpu):
        self.rows = []
        self.error = None
        self.process = subprocess.Popen(['nvidia-smi', f'--id={physical_gpu}',
            '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits', '--loop-ms=500'],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        for line in self.process.stdout:
            try:
                utilization, memory = [float(value.strip()) for value in line.split(',')]
                self.rows.append((utilization, memory))
            except ValueError:
                self.error = 'unparseable nvidia-smi sample'

    def finish(self):
        self.process.terminate()
        try:
            self.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.thread.join(timeout=2)
        self.process.stdout.close()
        return dict(gpu_utilization_mean=statistics.mean(r[0] for r in self.rows) if self.rows else None,
                    gpu_utilization_peak=max((r[0] for r in self.rows), default=None),
                    nvml_peak_memory_mib=max((r[1] for r in self.rows), default=None),
                    gpu_monitor_samples=len(self.rows), gpu_monitor_error=self.error)
