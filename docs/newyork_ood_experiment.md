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
stages automatically. Preflight uses 1 epoch, 3 batches of 8 and 4 test conditions; its projection
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
