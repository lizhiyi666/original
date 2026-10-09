# Two-city method comparison and ablation study

This is a new study and reporting manifest; it never edits the existing NY ablation
manifest, cached temporal inputs, or generated results. The shared RNG version stays
`pcdg-ablation-v1`, so the 21 completed NY results remain valid anchors.

## Locked sources

- NewYork: 3160 train / 2108 test, original completed base and completed 1000-epoch
  PO-CFG checkpoint. Reuse all 21 ablation results and all six temporal rank caches.
- Istanbul: 7035 train / 4914 test from the user archive; checkpoint
  `run-20260414_010530-ublj731x/files/checkpoints/last.ckpt`, SHA-256
  `a20566fcbd7980fb22e265700c81e3dd891ae60573e4c1c0232e41b1ebe334ba`.
  `latest-run` is stale; April 18 has additional PO-encoder weights and is not a
  compatible unconditioned base. Never use strict=False or drop those weights.
- Istanbul has nine semantic categories and ten historical model slots. Pad only
  the in-memory constraint matrix; the final row/column remains zero. Preserve the
  original token IDs, vocabulary, checkpoint dimensions and masks.
- Historical Istanbul base batch=512, NY base batch=64. Historical Istanbul data
  identity is not retroactively proven by new hashes; disclose the provenance limit.

## Execution

From isolated `/root/experiments/pcdg/two-city-v1`:

```bash
python -u -B tools/run_two_city_study.py \
  --newyork-source /root/experiments/pcdg/original/experiment_runs/nyood1000s13539820261001 \
  --newyork-ablation /root/experiments/pcdg/ablation-v1/experiment_runs/pcdg-ablation-v1 \
  --newyork-cfg /root/experiments/pcdg/baseline-suite-v1/experiment_runs/nyood-baselines-v1-20261002/cfg-training/last.ckpt \
  --istanbul-inputs /root/experiments/pcdg/two-city-v1/inputs/istanbul \
  --delivery-root 'D:/桌面/轨迹/轨迹/轨迹生成/实验/original/experiment_runs/two-city-v1'
```

The input directory contains unchanged `base.ckpt`, `config_hydra.yaml`, and the two
Istanbul_PO1_OOD PKLs. `data/Istanbul_PO1_OOD` points to that imported directory;
`data/NewYork_PO1_OOD` references the already validated NY inputs. Expected hashes
and counts are checked before GPU work. Use --resume only with matching manifests.

Order: import and independently re-evaluate 21 NY results; verify the new loader
reproduces old NY Full/No Projection preflight outputs; generate nine NY method
results; validate Istanbul layout/all ablations/energy and an explicit test-only
empty fixture; run Istanbul ablations and non-CFG methods; discard a separate
two-epoch CFG preflight and train fresh spatial weights for 1000 epochs / 110000
updates; verify CFG and sample its three seeds. Temporal weights are frozen.

PCDG remains T=3, order=1, existence=5, KL=1, 10x50 without early stopping, at steps
36,32,...,0. Energy is T=3, scale=100, order+5*existence, one update at those steps.
CFG uses scale=1 with no additional extrapolation. No per-city parameter search.

The empty fixture is an explicitly labeled copy of the 64-condition preflight cache
with one row masked empty; it never enters training or formal OOD caches. Original
data is unchanged. Each production seed uses 2457 conditions per Istanbul GPU.

## Tables and audit

`tools/unified_tables.py` provides the only column/format/best-value definitions.
Both methods and ablations show Distance/Radius/Interval/DailyLoc/Category/G-RANK,
and separately OVR_strict/OVR_skip/pair coverage/category coverage/Unsat. JSD has four
decimals; constraint percentages two. Error is sample SD across three sampling seeds.
Best unrounded means are bolded per city, with exact ties; Full is not privileged.
Missing metrics use a dash, incomplete city panels are explicitly withheld.

JointGen is an alias to No Projection, PCDG to Full: identical source files, hashes,
and numbers. Outputs comprise 30 unique results per city, 60 total (39 newly derived
or sampled). No Projection/Full are not regenerated for the method table. Report
interim NY tables as partial until both cities and the final audit complete.

The registry separately records source-manifest SHA, result SHA, source shards,
reuse/new origin and delivery path. Final audit rechecks all 60 outputs, indices,
shared metrics, shard contents, and paired spatial RNG traces. W&B sync has bounded
retries and dedicated receipts; syncing failures do not require resampling/training.
Markdown and LaTeX booktabs fragments are derived outputs; original PDFs/manuscripts
remain untouched. `totalJSD`, empty counts, costs and diagnostics are supplemental.
