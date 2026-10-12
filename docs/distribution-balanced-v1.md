# Distribution balance v1

This is a bounded, training-side engineering calibration on frozen checkpoints,
not an independent validation set: the base model saw these training records.
The original `same_category_v1`, evaluator v2, model weights, time caches and
historical results are unchanged. New sampling is enabled explicitly with
`distribution_balance=train_ref_v1` (standalone CLI or worker `balance_config`).

## Frozen method and selection

- ALM objective: existing token KL + rho * existing constraint loss + existing
  distance-KL term + lambda * mean(hour-category JSD, conditional-transition JSD).
  The new reference is the inherited training reference pool only. Self edges
  count; sequence boundaries do not. Unsupported reference rows contribute no
  regularizer. No new categorical RNG draws are consumed.
- Terminal geometry: original POI + model top11 + nearest-to-center12 + train
  top-frequency8, same actual category, deduplicated and filled by model score
  to at most32. Ties are stable by catalog ID. Geometry remains 200 steps,
  weights 1/2/0.1, 8 paths, lr0.05, noise refreshed every20 steps. Beta adds the
  mean of identity-aligned top1000 visitation JSD and soft expected DailyLoc JSD.
  Hard original/candidate batches are selected by the training-only objective.
- Grid: rho={0.25,0.5,1}, lambda={0,1}, beta={0,0.1,1}:18 configurations.
  Spatial results are cached by (city,pool,seed,rho,lambda), not beta; scoring
  is reused across the three refinements of each batch.
- Inherit screen512 / confirm512 / reference-rest and their temporal caches.
  Screen seed135398; retain at most2 jointly feasible city configurations.
  Confirm seeds135398/135399/135400; stop if no configuration passes. Rank by
  maximum then mean of the twelve JSD/JointGen ratios, then parameter tuple.
- Six JSDs must be <= JointGen; strict_ovr/ovr_skip/Unsat <= PCDG;
  pair/category coverage >= PCDG. Use each city's unrounded three-seed mean,
  only 1e-12 numerical tolerance, no old 1pp/5%/10% allowance. Undefined fails.

## Commands

Use `D:\Anaconda\envs\Marionette\python.exe -B -m unittest discover -s tests -q`
locally and `/root/anaconda3/envs/pcdg-exp/bin/python` on `ssh pcdg`.
Deploy the pushed commit into a NEW directory, never into a historical checkout.

```text
python -B tools/run_distribution_study.py --geometry-source <sealed-geo-run> --category-source <sealed-A-run> --preflight-only
python -u -B tools/run_distribution_study.py --geometry-source <sealed-geo-run> --category-source <sealed-A-run> --source-commit <verified-github-sha> --gpus 0,1
python -B tools/run_distribution_study.py --verify-only <sealed-output> --recompute
python -B tools/fetch_distribution_study.py
```

Run ID is `ny-ist-distribution-balanced-v1-20261012`. `--resume` is explicit,
identity-exact reuse, never an automatic restart. Half-published or conflicting
batches stop for investigation. Old baselines are checked/re-evaluated but never
sampled again; TPP calls are forbidden. GPU ceiling is80%, no batch/seed/precision
changes. On failure only this controller's own workers are terminated.

## Evidence and terminal states

Every batch and result has a manifest-bound receipt. New results retain original
indices/batch boundaries, source hashes, independent RNG diagnostics and model
state hashes. Refinement preserves sampled categories, conditions, times and
lengths. Final evaluation additionally reports unfiltered Distance and endpoint
selection counts; test metrics never select candidates or hyperparameters.

`stopped` with an audited0 formal results means the bounded calibration failed:
deliver the calibration analysis and do not run the test set. `complete` requires
6 formal results and21066 records; quality may still be false. The downloader
waits for controller exit, verifies the archive and every file, and CPU-recomputes
metrics from byte-identical copied evaluation inputs. Data, caches and logs stay
outside Git. The frozen configuration and all calibration evidence are retained.
