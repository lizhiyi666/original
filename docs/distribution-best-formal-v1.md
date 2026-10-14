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

## Delivery accepted on 2026-10-14

The existing run completed and was sealed; controller 26040 and workers 26051 /
26052 were confirmed exited. The server audit passed with 6/6 results and 21066
trajectory records. Original inputs, selection evidence, models and temporal
caches were preserved; there was no retraining, temporal/baseline resampling,
restart or additional candidate search during delivery.

- Runtime source commit: `a7e3c300d29f5a266320c90ae14d147782b9f599`.
- Remote result directory: `/root/experiments/pcdg/ny-ist-distribution-best-formal-v1-20261013/experiment_runs/ny-ist-distribution-best-formal-v1-20261013`.
- Local result directory: `experiment_runs/ny-ist-distribution-best-formal-v1-20261013`.
- Transfer process exited successfully; archive and per-file SHA-256 checks passed for 1383 delivered files (1381 audit entries plus `audit.json` and `status.json`).
- Local receipt: `experiment_runs/ny-ist-distribution-best-formal-v1-20261013/local-delivery-audit.json`, state `passed`, `transfer_exited=true`, `cpu_recomputed=true`.
- All 18 deterministic CPU v3 metric sets passed; maximum absolute difference was `1.1102230246251565e-16`, below `1e-12`. Frozen v2 values were retained for the manuscript.
- Archive SHA-256: `2332af438ecdc0e72924936853e52e0c85bb099790177a311a94f63f724a4880`.
- Sealed summary SHA-256: `8f9bc62fa0c98a73250461d60282ebf54290c1c71d6889a13660b6c8181f49de`.

Quality was not fully successful under either v2 or v3: New York
`CategoryTransition` and `G-RANK` remain above JointGen. Istanbul passes all
eleven gates, and both cities preserve all five PCDG constraint gates. These
are comparisons of three-seed means, not significance claims.

The downloaded manuscript `C:\Users\19542\Downloads\g(⋅)设计方案.md` and the
repository manuscript `g(⋅)设计方案.md` now contain exactly four new
`分布校正版（最佳候选）` method rows. All 22 metric cells (44 mean/SD values)
were checked against `formal.results.r0.5-c0-p1.aggregate`, including sample
SDs and percent conversion. Existing rows, historical text and ablations were
preserved; only obsolete best-value bold markers were moved. No separate
confirmation/v3 tables or duplicate baseline rows were added. Manuscript text
is identical after newline normalization; each file retains its existing line
endings. Istanbul Category uses the unrounded means to resolve the displayed
0.0423 tie: candidate 0.04229393378375002 versus Scheme A 0.04232732278162768.
