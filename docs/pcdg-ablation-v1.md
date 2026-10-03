# PCDG ablation v1

The locked study is seven variants x seeds 135398/135399/135400 on all 2108
NewYork_PO1_OOD conditions, with batch 64 and two contiguous 1054-condition shards.
It reuses the original 1000-epoch base checkpoint, never CFG weights or retraining.
Full is regenerated; historical 38.05% is not a result under this new protocol.

All effective projection calls run exactly 10x50 updates, at steps 36,32,...,0.
Temperature=3, eta=1, clip norm=10, lambda=mu=1, mu_max=1000, mu_alpha=2.
KL remains model-to-projected with the existing batch denominator; penalties remain
separately computed for order/existence and summed, not normalized differently.
Variants disable projection, existence, order, KL, multiplier updates, or Gumbel
noise one at a time. No-Gumbel uses differentiable softmax at the same temperature.
Disabled heads are excluded from dual updates and stopping rules. Safety checks
and inactive rows are never ablated. Existing entry points retain legacy defaults.

Time caches are atomically published only after a complete rank, retaining the
temporal sampler's adaptive state across its batches. Temporal randomness follows
seed+global batch start. Spatial and projection generators use separate SHA-256
streams over compact JSON [version,seed,global_start,stream], independent of variant
and physical GPU. Global RNG leakage is checked. Every variant reads the same
sealed CPU time batches, and cannot call temporal sampling in its spatial stage.

```bash
python -u -B tools/run_pcdg_ablation.py \
  --source-experiment /root/experiments/pcdg/original/experiment_runs/nyood1000s13539820261001 \
  --preflight-indices /root/experiments/pcdg/perfcal-v1/calibration_runs/perfcal-v1/indices.json \
  --historical-fixture /root/experiments/pcdg/sampling-perfcal-v1-r2/experiment_runs/nyood1000s13539820261001_perfcal-v1-ood-r2/frozen-empty-regression/legacy-temporal-input.pkl
```

Run from an isolated `/root/experiments/pcdg/ablation-v1` deployment. `--checkpoint`
may identify an identical copy of the original checkpoint. `--cache-dir`, `--seeds`
and `--variants` are explicit inputs to the immutable manifest; Full is required.
Use `--resume` only with identical inputs/code. Incomplete cache ranks restart from
their initial temporal state; complete numerical outputs are reused. W&B sync has
three bounded readback attempts, and sync failure never requires resampling.

Preflight covers all seven variants on 64 train conditions and the historical
mixed empty fixture, plus the identical Full/cache on GPU1. Those results are
functional tests, never parameter selection. Formal variants execute serially,
each using both GPUs. Per-condition diagnostics retain original category tokens.
OVR decomposition uses the original macro weights, not 1-global-pair-coverage.
Undefined skip/distribution metrics stay null with validity counts, not zero.

Outputs under experiment_runs/pcdg-ablation-v1 include 21 generated.pkl files,
42 source shards, six rank caches, per-condition and RNG traces, W&B receipts,
summary CSV/JSON, within-seed paired differences, PNG/SVG figures and audit.json.
Sample standard deviation uses ddof=1. All uncertainty is conditional on one
training checkpoint; no training-seed significance claim is made.
