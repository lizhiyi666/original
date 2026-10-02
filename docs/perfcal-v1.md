# perfcal-v1: train-only engineering calibration

This does not retrain the model or run the full OOD test set. The fixed pool contains 512
training conditions already seen by the checkpoint; its first 128 conditions screen temperature.
Indices are sampled without replacement with NumPy default_rng(135398), in draw order.

The profile is `config/calibration/perfcal-v1.yaml`: outer=10, inner=50, last 40 diffusion
steps, frequency 4, with all other ALM weights and learning rates unchanged. Temperatures are
0.1/0.5/1/2/3 at batch 64. Batch candidates are 64/128/256/512 at the selected temperature.
Loss normalization and global clipping still depend on batch size; cross-batch results are
new experimental settings, not bitwise-equivalent hardware-only changes.

The optimized projector caches only invocation-local invariants and removes logging-only
scalar transfers from inner iterations. It still checks loss before backward, gradient norm
before updating, and logits after updating. `tests/fixtures/projection_reference_f70beb4.py`
is frozen from commit f70beb4 for equivalence and three-repeat short projection timing.
Real-input numerical equivalence checks all five candidate temperatures, including RNG
consumption and early stopping. Gradient probes use only the constraint objective before
the first update of each projection call; numerical residue from KL is not counted.
Normal sampling candidates are each one full warmed pass over their fixed conditions;
the three-repeat medians refer to the isolated projection-kernel comparison, not full sampling.

```bash
conda activate pcdg-exp
cd /root/experiments/pcdg/perfcal-v1
python -u tools/calibrate_projection.py \
  --source-experiment /root/experiments/pcdg/original/experiment_runs/nyood1000s13539820261001
```

Each candidate runs in a fresh subprocess. GPU 0 is the primary calibration device; the
chosen configuration is checked on GPU 1. Peak allocated/reserved bytes and nvidia-smi
utilization samples are recorded. FP32 is used without autocast or TF32. OOM/NaN candidates
are recorded as failures; their batch sizes are never silently reduced.

Selection excludes temperatures whose violating-constraint gradient probes are all zero.
Temperature ties use strict OVR, then higher pair coverage, then shorter sampling time.
Batch eligibility requires peak reserved/allocated memory <=80%, strict OVR no more than
1 percentage point worse and pair coverage no more than 1 percentage point lower than
same-temperature batch 64. Within 5% of maximum throughput, choose the smaller batch.

Artifacts live in `calibration_runs/perfcal-v1/`: immutable manifest/indices, per-candidate
jobs/logs/results/generated outputs, recommendation.json and report.md. `--resume` reuses
only recorded terminal candidates with identical manifests. W&B uses the distinct
engineering-calibration job type and explicit in-sample labels. No full OOD sampling is
automatically started after selection.

## User-approved full OOD resampling

After explicit approval, use a new deployment `/root/experiments/pcdg/sampling-perfcal-v1`.
Keep the original training and calibration directories unchanged. Copy only the two input
PKLs into the new data directory so new generated files cannot overwrite earlier outputs.

```bash
python -u -B tools/run_calibrated_ood.py \
  --source-experiment /root/experiments/pcdg/original/experiment_runs/nyood1000s13539820261001 \
  --calibration /root/experiments/pcdg/perfcal-v1/calibration_runs/perfcal-v1
```

The runner verifies the completed calibration, code/data/checkpoint fingerprints and the
lowest measured strict-OVR configuration. It cannot train. It replays global indices
64..127 with the selected settings and requires empty index 99 to remain intact, then runs
both native and projected sampling for all 2108 OOD conditions on two GPUs. It freezes
temperature 3.0, batch 64 and projection 10x50 from the verified recommendation; no OOD
hyperparameter search is performed. `tools/sample_perfcal.py` preserves the FP32/no-TF32
policy. Outputs and W&B evaluation runs use the separate `perfcal-v1-ood` revision.
Existing manifests require identical inputs and an explicit `--resume`.
