# PCDG — WWW 2027 working manuscript

Updated 2026-10-18. ACM `sigconf,anonymous,review`; Research Track long paper.
Target area: Web Mining, Multimedia and Multilingual Content Analysis.
T1–T8 are complete; T9 has an outstanding official CCS export. This is a complete
advisor-review draft, **not an unconditional submission-ready sign-off**.

## Locked metadata and open decisions

- Title, abstract, five keywords and author order are unchanged from `c828598`.
  Do not add “distribution correction” back to the title.
- Zhiyi Li: `25s151146@stu.hit.edu.cn`; then corresponding author Kaiqi Zhao:
  `zhaokaiqi@hit.edu.cn`. Both HIT Shenzhen. Emails are in source; ORCIDs were not
  supplied and must not be guessed. Keep the review PDF anonymous.
- **Abstract conflict retained at the author's direction:** the phrase “eliminates
  residual category–POI inconsistency” conflicts with the sealed six-run mismatch
  range **0.0034%–0.0219%**. The body explains the unsupported-final-category catalog
  fallback. Await advisor decision; zero mismatch is unsupported. Pair coverage means
  both categories occur, not correct ordering, as the body now states explicitly.
- **CCS pending:** the official tool <https://dl.acm.org/ccs> remained behind security
  verification on 2026-10-18. The inherited block is marked unverified; replace it with
  an official export before full submission. No concept IDs were invented or changed.
- No OpenReview interaction or submission was performed. The author must register/upload.

## Deadlines and format

The [official WWW 2027 Research Track page](https://www2027.thewebconf.org/research-track-papers/),
checked 2026-10-18, gives abstract **18 October 2026 AoE** and full-paper
**25 October 2026 AoE**. End-of-day AoE converts to just before **20:00 Beijing on
19/26 October**, respectively. The old handoff reported OpenReview 13:00 UTC / 21:00
Beijing; that one-hour discrepancy remains unverified in the portal. Work to the earlier
official-site deadline and confirm the portal yourself. Long papers allow **8 main-text
pages / 12 total**, with Web/track relevance on page 1. Settle authors/order at registration.

## Build and verification

Run inside `paper-www2027/`:

```powershell
latexmk -pdf -interaction=nonstopmode main.tex
```

Current PDF: **11 pages**, conclusion **p7**, Appendix A starts **p8**. Build exits
**0** with no LaTeX errors, undefined references/citations, visible TODOs or overfull
boxes. Fonts are embedded; PDF text, metadata and links pass the anonymity checks.
Web relevance appears on page 1; final-size page renders have been inspected.

Do not treat exit code 12 as harmless. The old draft had a real `\Bbbk` redefinition
from redundant `amssymb`; that import was removed. Check file-line-style errors too,
not only lines beginning `!`. Preserve `latexmkrc` for Chinese-path BibTeX support.
Remaining nonfatal notices concern review-mode reference-block suppression, last-page
column balancing, underfull vertical boxes and incomplete legacy BibTeX fields.
The 52-entry database has not received a full metadata/claim-support audit; five directly
relevant entries were checked against primary sources.

From the repository root, regenerate figures/tables without executing experiments:

```powershell
& 'D:\Anaconda\envs\Marionette\python.exe' figure_build/generate_method_main_v2.py
& 'D:\Anaconda\envs\Marionette\python.exe' figure_build/generate_tradeoff.py
& 'D:\Anaconda\envs\Marionette\python.exe' figure_build/generate_paper_evidence.py
```

After building, run `figure_build/verify_paper.py` with a Python environment providing
`pypdf` (the app's bundled document runtime) and Poppler `pdffonts`. It checks locked
blocks, original numeric rows/precision/bolding, page budget, fonts, anonymity, TODOs,
log errors and protected-path diffs against `c828598`. Integrity passing does not resolve
CCS or the abstract conflict. Scratch renders stay in `ccfa-workfiles/writing/www2027/build/`
and are not committed. PDF/SVG exports live in `paper-www2027/figures/`.

## Completed work

| Task | Delivery |
| --- | --- |
| T1 | Combined fidelity/constraint table; full stage-wise table in Appendix C; compressed introduction/conclusion while retaining negative results and selection failures |
| T2 | Unsat–Distance two-city figure from existing v2 table means, equal linear axes, unshifted measured coordinates |
| T3 | Combined framework/reverse-step figure; distribution weights as objective inputs, not a fictitious serial fourth stage; fallback shown |
| T4 | Exact metric definitions, denominators, endpoint filter, catalog/token distinction and historical histogram edge cases |
| T5 | 18-row v2 grid, SHA-verified v3 selection source and unchanged screening/confirmation rankings and failures |
| T6 | Manifest counts and evidence-bounded OOD definition, retaining Istanbul split-provenance gap |
| T7 | Selected-geometry timing with projection-Full denominator; unavailable final-stack costs explicitly unmeasured |
| T8 | Related work aligned to projection, support-constrained decoding and spatial refinement; five central citations verified |
| T9 | Emails, no-op TODO, build/anonymity/fonts/page checks done; official CCS export pending |

## Scientific guardrails

- Marionette is the prior backbone; projection adapts CDD. The implemented token KL is
  **base-to-projected**; Appendix A specifies batch normalization.
- New York CategoryTransition 0.0183 vs JointGen 0.0070 and G-RANK 0.1914 vs 0.1612 remain
  worse. PO-CFG wins those two and Category there. Istanbul improves all six vs JointGen
  and five vs PO-CFG. Three-seed sample SDs do not establish significance.
- All main/ablation tables and the screening-grid table use **v2**. Final selection used
  **v3**. All 18 screening ranks and both confirmed-candidate ranks agree across versions.
  Neither confirmation passed all distribution gates; final sampling used an explicit
  override retaining constraint gates. Calibration was training-side, not independent
  validation. No test metric entered selection, pool construction or refinement objectives.
- Maximum v3 G-RANK shift 0.0032 is scoped to the **18 formal audit metric sets**.
  v2 already aligns retained POIs by identity; v3 changes tie handling at the cutoff.
- (d)→(e) jointly changes rho, beta and pool composition: original+top15+nearest16 becomes
  original+top11+nearest12+train-most-visited8. No single-factor attribution.
- Radius is `sqrt(mean(Haversine distance to arithmetic coordinate centroid))`, unit
  sqrt(km), not standard radius of gyration. Distance is conditioned on endpoint-category
  matching; eligible subsets vary by method. Unsat is relation-level, not sequence failure.
- New York train/test counts: 3160/2108; Istanbul: 7035/4914. New York reverses precedence
  preferences without guaranteeing disjoint complete constraint configurations.
- Measured 3.35%–3.98% cost increase is from the **old selected 200-step geometry qualification
  relative to projection Full**, not final PCDG relative to JointGen. It excludes temporal
  generation, loading and transfer. Final-stack overhead and PO-CFG GPU-hours are unmeasured.
- Hard category-conditioned POI-support masking **is implemented**, with fallback exceptions;
  the old contrary note was wrong. Complex DAG evaluation, adaptive scheduling, formal privacy
  and downstream utility remain unestablished. Do not edit sealed `experiment_runs/` or rerun.

## Camera-ready only

After acceptance: remove anonymous/review; confirm ORCIDs, copyright, conference/DOI/ISBN
metadata and the ACM reference block under official instructions. Do not make these changes
to the anonymous review PDF now. See `HANDOFF-codex.md` for evidence paths and next steps.
