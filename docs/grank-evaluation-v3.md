# G-RANK evaluation v3

This opt-in, CPU-only re-evaluation fixes platform-dependent membership at a
top-1000 frequency tie. Legacy `evaluations/statistical_metrics.py`, its v2
default, sampler, regularizers and all archived artifacts remain unchanged.

## Definition

Count integer POI IDs, sort by **count descending, then POI ID ascending**, take
at most 1000 IDs separately for generated/reference data, and align their union
in ascending POI-ID order before applying the historical JSD normalization and
epsilon. Do not compare sorted popularity curves without identity alignment.

The historical GPS-length filter (`len(gps) > 1`) is unchanged. Distance's
endpoint filter does not affect G-RANK. All other statistical and constraint
metrics are obtained from v2 and must match their archived values within
`1e-12`. Only G-RANK, totalJSD and the explicit evaluation version may change.
No global function is patched and no global random state is consumed.

## Immutable re-evaluation

```text
python -B tools/reevaluate_distribution_v3.py run --source <sealed-v2-directory> --output <NEW-v3-directory> --evaluation-commit <verified-40-character-GitHub-SHA>
python -B tools/reevaluate_distribution_v3.py verify --output <v3-directory> --source <sealed-v2-directory> --recompute
python -B tools/reevaluate_distribution_v3.py compare --left <Windows-v3-directory> --right <downloaded-Linux-v3-directory>
```

The runner verifies every source audit hash before and after evaluation. Source
and output must be disjoint; an existing output, including a partial output, is
refused. PKLs are loaded only from the hash-checked, trusted experiment source.
Receipts, manifest identity, city, phase, seeds, exact indices, summary rows and
evaluation-input hashes are checked before use. Every audited baseline/result
must be represented in the immutable source summary.

Legacy code must match the original source hashes, allowing only explicit
LF/CRLF conversion for code/config text (never for datasets). The new manifest
records raw and LF-normalized code hashes and the evaluation commit. This makes
transport line endings visible without hiding substantive code changes.

`records.json` keeps the original v2 values beside v3 values, source hashes and
ranked POI/count fingerprints. `summary.json` recomputes both baselines and all
archived candidates, means, sample SDs, paired seed deltas, gates and rankings.
It keeps the original selection and terminal state, and flags changed screening
shortlists. This is retrospective analysis: new shortlist members are **not**
sampled or confirmed, and a newly passing archived candidate does not authorize
a formal run. There are no model/checkpoint loads, training or sampling calls.

Each platform first produces its own audit. The comparison requires matching
source fingerprints, evaluation commit and normalized code, Windows plus Linux
environment records, identical top-1000 membership/count fingerprints, matching
gate/ranking decisions and all metrics within `1e-12`. A platform's own audit
does not by itself assert cross-platform success.

The failed v2 `local-delivery-audit.json` is not fabricated or overwritten. A
successful v3 comparison is a separate delivery under a new metric version.
