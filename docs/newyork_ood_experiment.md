# NewYork_PO1_OOD experiment

Run on the configured `pcdg` server in `/root/experiments/pcdg/original`,
inside `conda activate pcdg-exp`. Never commit credentials or dataset PKLs.

## Fixed formal profile

- 1000 epochs, batch 64, seed 135398, GPU 0; PO auxiliary loss disabled.
- Existing temporal/spatial architecture and optimizers; actual diffusion steps 100/256.
- Final checkpoint only; no test-set selection or hyperparameter search.
- Native then projected sampling, each on GPU 0/1 with 1054 ordered test conditions per worker.
- Batch 64; both methods reseed each global batch identically. Projection can consume additional
  random numbers within a batch, so this is not a claim of identical reverse-process noise.
- Projection: last 40 steps, every 4 steps, maximum 200 outer / 100 inner iterations;
  tau 0, lambda/eta/mu 1, mu max 1000, mu multiplier 2, tolerance 1e-6,
  Gumbel temperature 0.1, existence weight 5. Existing early termination is retained.

The stored matrices include edges from absent categories. `--constraint_source strict_test`
rebuilds constraints in memory using the same strict reference-pair function as OVR evaluation:
both categories occur and max-position(A) < min-position(B). PKLs/SVD/encodings are unchanged.

## Commands

```bash
python -m unittest discover -s tests -v
python tools/run_newyork_ood.py --run-id nyood-preflight-UNIQUE --stage preflight
python tools/run_newyork_ood.py --run-id nyood-formal-UNIQUE --preflight-id nyood-preflight-UNIQUE
```

Use a unique ID for each fresh experiment. Run the last command inside tmux. The runner checks
the preflight's code/data/environment fingerprints, then trains and runs both sampling/evaluation
stages automatically. The user-approved preflight uses 20 epochs, 50 batches per epoch at batch
size 64 (1000 training batches total), and 4 test conditions; its projection
budget is only 2x2 iterations for functional testing. It is never reused for formal training.
A separate full-budget one-sample benchmark runs after formal training, without changing settings.

```bash
# Resume weights, both optimizers/schedulers and epoch, not only the W&B run:
python tools/run_newyork_ood.py --run-id EXISTING_ID --resume
# Reverify/reuse matching completed shards and retry sampling/evaluation:
python tools/run_newyork_ood.py --run-id EXISTING_ID --stage sample --resume
```

An immutable `experiment_runs/<ID>/manifest.json` records settings and input/code fingerprints.
`status.json` records running/failed/complete and the current phase. A process lock prevents
concurrent execution of the same experiment. Failed worker processes block merging/evaluation;
source shards are never deleted. Dataset or code changes require a fresh run, not silent resume.

Outputs: final.ckpt, config_hydra.yaml, command/log files, benchmark.json, per-method metrics JSON,
comparison.json, and method-specific generated PKLs in the dataset directory. W&B sampling runs
are grouped with the training ID. Non-finite metrics are rejected, not reported as success.

At least 5 GiB free disk is required at each phase boundary. No automatic changes to batch size,
precision, epochs or projection parameters are allowed on errors.

## Approved singleton fix while training is running

`MixtureIntensity.sample` previously squeezed a one-sequence count tensor into a scalar.
The fix keeps at least one dimension; non-scalar count values and shapes are unchanged.
Do not replace files in the active training directory or rewrite its immutable manifest.

For the running `nyood1000s13539820261001` experiment, the fixed inference snapshot is
`/root/experiments/pcdg/sampling-singleton-fix`. It has independent copies of the two input
PKLs, and a `wandb` symlink for reading the existing training/preflight checkpoints.

```bash
# Run the regression test on the idle GPU, using the completed 20-epoch preflight:
CUDA_VISIBLE_DEVICES=1 python tools/validate_singleton_fix.py \
  --source-experiment /root/experiments/pcdg/original/experiment_runs/nyood1000s13539820261001 \
  --preflight-run-id nyoodpf1000b20261001d

# From the fixed snapshot, inside a separate tmux session:
python tools/continue_newyork_sampling.py --wait \
  --source-experiment /root/experiments/pcdg/original/experiment_runs/nyood1000s13539820261001 \
  --validation-receipt validation/singleton-validation.json
```

The continuation waits without interrupting training. It accepts only the original
controller's specifically verified singleton `IndexError` in the benchmark phase, after
the training lock is released and W&B/checkpoint verification confirms all 1000 epochs.
Any other failure stops the continuation. It does not silently recover training failures.
The old controller's expected benchmark failure is separate from successful model training.

The new sampling manifest records both code fingerprints, the untouched source manifest's
hash, the validation receipt, and the source experiment. Only the approved intensity file
may differ; architecture, parameters, package versions, data and checkpoints must match.
Sampling results and `status.json` are under the fixed snapshot, not the training directory.
Use `--resume` explicitly to retry that continuation with matching inputs and code.
