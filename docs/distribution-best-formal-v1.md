# Best confirmed candidate: explicit formal test authorization

On 2026-10-13 the user explicitly requested formal sampling with the selected
best candidate, followed by insertion into the existing method tables in
`C:\Users\19542\Downloads\g(⋅)设计方案.md`. Do not insert calibration rows or add
a new confirmation/v3 table. The original bounded study and its failed gates
remain sealed and unchanged.

## Frozen experiment

- New run: `ny-ist-distribution-best-formal-v1-20261013`.
- Candidate: `r0.5-c0-p1`, rho=0.5, lambda=0, beta=1. Selected by the existing
  max-then-mean of the twelve JSD/JointGen ratios, using the already confirmed
  three-seed candidates that preserve PCDG constraint gates. It does not meet
  all distribution gates; the explicit user override is recorded before test.
- Two cities; seeds 135398/135399/135400; New York 2108 and Istanbul 4914 test
  records per seed. Exactly six new results, 21066 records.
- Reuse the original checkpoints, training references, temporal caches,
  conditions and batch layout. Reuse original JointGen/PCDG baseline trajectories.
  Do not retrain, resample time/baselines, tune, change seeds/batch/precision,
  restart automatically or overwrite any historical artifacts.
- The new controller delegates only the formal phase to the existing audited
  distribution-study workers. No calibration phase is run again. Configuration
  and source fingerprints are frozen before workers start. Only its own workers
  are stopped on technical failure.

```text
python -B tools/run_best_distribution_formal.py --geometry-source <sealed-geometry> --category-source <sealed-category> --selection-source <sealed-v3-calibration> --preflight-only
python -u -B tools/run_best_distribution_formal.py --geometry-source <sealed-geometry> --category-source <sealed-category> --selection-source <sealed-v3-calibration> --source-commit <verified-GitHub-SHA> --gpus 0,1 --allow-confirmation-gate-failure
python -B tools/fetch_best_distribution_formal.py
```

## Evaluation and delivery

The existing document's method tables use evaluation v2, so the new method row
uses frozen server v2 values for a like-for-like table. Do not mix a v3 G-RANK
value with v2 baseline columns. Deterministic v3 evaluation is stored separately
for all six results and the two reused baselines per city/seed (18 metric sets),
and locally recomputed on identical evaluation inputs. Preserve the original
v2 values and require every non-G-RANK statistic to agree within 1e-12.

Both quality outcomes are reported; failed quality is not a reason to discard
the actual formal results. Acceptance requires controller/worker exit, sealed
6/6 and 21066 counts, completed transfer, archive and per-file SHA-256 checks,
and a passing `local-delivery-audit.json` including 18 CPU v3 metric sets.

Only after acceptance, insert `分布校正版（最佳候选）` as one new row into each of
the existing two-city method-distribution and method-constraint tables (four
rows total). Use three-seed means and sample SDs: four decimals for JSD, percent
units with two decimals for constraints. Keep other methods, historical tables,
and ablations. Add a short provenance/metric-version note outside the tables;
do not add a separate calibration table. Preserve intervening user document
edits; re-read and inspect its hash before patching. Synchronize the updated
manuscript into the repository and checkpoint/push before and after the edit.

At task recovery, inspect the existing remote run and exact PID commands first.
If healthy, monitor quietly; never launch another copy. If the local delivery
already exists, use the fetcher's `--verify-only`, never overwrite it. Technical
failure preserves the scene and requires a user decision.
