# Baseline suite v1

This suite preserves the completed `perfcal-v1-ood-r2` native/projection results.
Baseline2 derives from its exact native PKL, swapping only marks/POI/GPS while
retaining times and all context fields. Baseline3 uses deterministic soft energy
`sum(order + 5 * existence)` at the last 40 steps, frequency 4, with no ALM.
Baseline4 freezes the original temporal model and trains a fresh spatial PO-CFG
model for 1000 epochs / 50000 updates after a separate discarded 2-epoch preflight.
Only the PO condition drops (probability 0.1); the background remains present.

All methods use batch 64, seed 135398, FP32 without TF32. The nine energy candidates
are T=1/2/3 crossed with scale=1/10/100. CFG evaluates scales 0/1/1.5/2/3/5;
zero is a diagnostic, not a selectable candidate. All candidates use the same
previously saved 512 training indices. Selection is strict OVR, descending pair
coverage, smaller scale, then smaller energy temperature. This is in-sample
engineering calibration, never OOD tuning.

Run from a separate deployment `/root/experiments/pcdg/baseline-suite-v1`:

```bash
python -u -B tools/run_baseline_suite.py \
  --source-experiment /root/experiments/pcdg/original/experiment_runs/nyood1000s13539820261001 \
  --reference-run /root/experiments/pcdg/sampling-perfcal-v1-r2/experiment_runs/nyood1000s13539820261001_perfcal-v1-ood-r2 \
  --baseline1 /root/experiments/pcdg/sampling-perfcal-v1-r2/data/NewYork_PO1_OOD/NewYork_PO1_OOD_nyood1000s13539820261001_perfcal-v1-ood-r2_native_generated.pkl \
  --indices /root/experiments/pcdg/perfcal-v1/calibration_runs/perfcal-v1/indices.json
```

Use `--methods baseline2 baseline3 baseline4` to select an explicit subset before
starting a new suite. An existing suite requires identical code/inputs/methods and
`--resume`; CFG restores the optimizer, scheduler, counters and RNG from a completed
epoch, not weights alone. Source shards, candidate logs and failures are retained.
Each method has an independent W&B evaluation; CFG training/preflight and each
method's train calibration have separate records. `comparison.json`, `audit.json`
and `report.md` are published only after the selected methods complete.

The legacy `sample.py --baseline` aliases remain available. Energy/CFG cannot be
combined with ALM. CFG additionally requires `--cfg_checkpoint` pointing to a
completed suite spatial checkpoint. The old four-GPU shell wrappers are historical;
the new suite entry point is the supported two-GPU workflow.
