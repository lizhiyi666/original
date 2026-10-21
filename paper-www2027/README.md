# PCDG — WWW 2027 working manuscript

Updated 2026-10-21. ACM `sigconf,anonymous,review`; Research Track long paper.
Target area: Web Mining, Multimedia and Multilingual Content Analysis.
T1–T9 and the authorized evidence-focused revision are complete, including the
official CCS export and the bibliography audit. Figure 1's four-panel redesign and
caption/accessibility integration are also complete. Scientific limitations below
remain; successful build and format checks are not a scientific acceptance judgment.
The submission-review revision is complete: reverse-step projection and decoder
notation, calibration gates, token-KL attribution, and registration notes are aligned.
See [REVIEW-AND-FIXES.md](REVIEW-AND-FIXES.md) for resolved issues, locked-abstract
concerns, evidence gaps, and author-only decisions. The current abstract is unchanged
from `9b05681`; no new abstract correction was made in this revision.

## Locked metadata and open decisions

- Title, five keywords and author order are unchanged from `c828598`.
  The abstract allows exactly one authorized change: `eliminates residual` to
  `reduces residual`. All other abstract text remains locked.
  Do not add “distribution correction” back to the title.
- Zhiyi Li: `25s151146@stu.hit.edu.cn`; then corresponding author Kaiqi Zhao:
  `zhaokaiqi@hit.edu.cn`. Both HIT Shenzhen. Emails are in source; ORCIDs were not
  supplied and must not be guessed. Keep the review PDF anonymous.
- **Abstract conflict corrected with authorization:** “reduces residual” agrees
  with the sealed six-run mismatch range **0.0034%–0.0219%**. The body retains the
  unsupported-final-category catalog fallback. Pair coverage means both categories
  occur, not correct ordering.
- **CCS complete:** five concepts were selected through the official tool
  <https://dl.acm.org/ccs> on 2026-10-18 and its Generate Code output replaced the
  inherited block. Location based services and Machine learning have weight 500,
  Data mining and Social networks 300, and Neural networks 100. All IDs, descriptions
  and weights agree with the supplied candidate code. The six newly suggested keywords
  were not substituted for the five locked keywords.
- No WWW submission-portal operations were performed. Public OpenReview literature
  records were read only for citation verification. Registration/upload remains with the author.

## Deadlines and format

The [official WWW 2027 Research Track page](https://www2027.thewebconf.org/research-track-papers/),
checked 2026-10-21, gives abstract **18 October 2026 AoE** and full-paper
**25 October 2026 AoE**. End-of-day AoE converts to just before **20:00 Beijing on
19/26 October**, respectively. The old handoff reported OpenReview 13:00 UTC / 21:00
Beijing; that one-hour discrepancy remains unverified in the portal. Work to the earlier
official-site deadline and confirm the portal yourself. Long papers allow **8 main-text
pages / 12 total**, with Web/track relevance on page 1. Settle authors/order at registration.

## Build and verification

Run inside `paper-www2027/`:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=../ccfa-workfiles/writing/www2027/build main.tex
```

Current PDF: **12 pages**, conclusion **p7**, Appendix A starts **p8**. Build exits
**0** with no LaTeX errors, undefined references/citations, visible TODOs or overfull
boxes. Fonts are embedded; PDF text, metadata and links pass the anonymity checks.
Web relevance appears on page 1; all 12 current page renders have been inspected.
The unchanged method table and trade-off figure are queued earlier in the source
to avoid an isolated float page. The inherited `balance=false` option is retained;
the current TeX Live `acmart` 2.16 reports it as unused, so it is not evidence that
column balancing was disabled. The actual current bibliography has no overfull box.
Fonts, margins and table styling were not reduced. Figure 1 was redesigned at the
existing 7 × 3.25 inch size; its live labels remain at least 8 pt.

Do not treat exit code 12 as harmless. The old draft had a real `\Bbbk` redefinition
from redundant `amssymb`; that import was removed. Check file-line-style errors too,
not only lines beginning `!`. Preserve `latexmkrc` for Chinese-path BibTeX support.
Remaining nonfatal notices concern review-mode reference-block suppression,
underfull vertical boxes and 34 BibTeX style warnings (missing optional addresses,
ICLR's non-paginated record, and AAAI volume/issue combinations). No missing fields
were invented to silence those warnings.
The database has **50 records, 45 actually cited**. All 50 received metadata checks;
six central references received targeted context checks. Background citations were
checked at identity/title/available-abstract level, not all read in full.
See [CITATION-AUDIT.md](CITATION-AUDIT.md) for sources, corrections and coverage limits.

From the repository root, regenerate figures/tables without executing experiments:

```powershell
& 'D:\Anaconda\envs\Marionette\python.exe' figure_build/generate_method_main_v2.py
& 'D:\Anaconda\envs\Marionette\python.exe' figure_build/generate_tradeoff.py
& 'D:\Anaconda\envs\Marionette\python.exe' figure_build/generate_paper_evidence.py
```

After building, run `figure_build/verify_paper.py` with a Python environment providing
`pypdf` (the app's bundled document runtime) and Poppler `pdffonts`. It checks locked
blocks, original numeric rows/precision/bolding, page budget, fonts, anonymity, TODOs,
log errors and protected-path diffs against `c828598`. It also validates exactly five
CCSXML entries and matching LaTeX descriptions against the verified official export.
Negative checks reject incorrect IDs/weights, duplicate descriptions and missing XML.
The strict abstract whitelist is tested by `figure_build/test_verify_paper.py`:
all eight tests pass, including rejection of other abstract edits, title/keyword
changes and author reordering. From the root, run `python -B -m unittest discover
-s figure_build -p test_verify_paper.py -v` with the same Python environment.
For the Figure 1 revision, the method body from §4.1 onward, all equations and
table files, other figures, scientific code and sealed results are unchanged from
`4389ee0`. Only the framework generator/tests, framework SVG/PDF, method opening,
caption/accessibility description, compiled paper and handoff documents changed.
The five figure-contract tests pass in the existing plotting environment; run
`python -B -m unittest discover -s figure_build -p test_generate_method_main_v2.py -v`
there. The document runtime provides `pypdf`; it need not provide Matplotlib.
Integrity passing does not close the scientific gaps below.
Scratch renders stay in `ccfa-workfiles/writing/www2027/build/`
and are not committed. PDF/SVG exports live in `paper-www2027/figures/`.
The current verified build has been copied to `paper-www2027/main.pdf`; its SHA256 is
`72db36579d257ccef77888fef305daf5ab6a55535a0fbd83fbdeb27a9621b487`.
Figure 1 is on p3. All twelve pages were rendered and inspected on 2026-10-21.
The recovered 13-page build placed two conclusion lines alone on p8. Concise wording
in the method placement, limitations and conclusion restored the 12-page layout
without changing fonts, margins, tables or figures. Source and regression checks
were not relaxed. The source baseline for this revision is `9b05681`, with recovered
edits checkpointed at `160b20a` before page fitting.

## Completed work

| Task | Delivery |
| --- | --- |
| T1 | Combined fidelity/constraint table; full stage-wise table in Appendix C; compressed introduction/conclusion while retaining negative results and selection failures |
| T2 | Unsat–Distance two-city figure from existing v2 table means, equal linear axes, unshifted measured coordinates |
| T3 | Four-panel framework: reverse-loop overview, logit projection, shared-Gumbel category-conditioned POI decoding, and same-frame refinement; objective inputs and fallback shown |
| T4 | Exact metric definitions, denominators, endpoint filter, catalog/token distinction and historical histogram edge cases |
| T5 | 18-row v2 grid, SHA-verified v3 selection source and unchanged screening/confirmation rankings and failures |
| T6 | Manifest counts and evidence-bounded OOD definition, retaining Istanbul split-provenance gap |
| T7 | Selected-geometry timing with projection-Full denominator; unavailable final-stack costs explicitly unmeasured |
| T8 | Related work aligned to projection, support-constrained decoding and spatial refinement; five central citations verified |
| T9 | Emails, no-op TODO, build/anonymity/fonts/page checks done; official five-concept CCS export integrated and XML/LaTeX consistency verified |

## Scientific guardrails

- Marionette is the prior backbone; projection adapts CDD. The implemented token KL is
  **base-to-projected**; Appendix A specifies batch normalization.
- Actual activities are catalog POI categories, not independently sampled category
  tokens. Decoding connects those representations; same-category refinement preserves
  constraint-relevant fields, not all distributional fidelity. EnergyGuide is the
  local order/existence-gradient baseline, not an exact reproduction of discrete CBG.
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
