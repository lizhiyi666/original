# PCDG — WWW 2027 submission (working draft)

ACM `acmart` (sigconf, **anonymous, review**) paper for The Web Conference 2027,
Research Track. Target track: **Web Mining, Multimedia and Multilingual Content
Analysis**.

## Hard deadlines (AoE, no extensions)
- **Abstract registration: 2026-10-18** (locks title + author list/order)
- **Full paper: 2026-10-25** — body **≤ 8 pages**, (refs + appendix) total ≤ 12
- Reviews 2026-12-15 · rebuttal 12-15–12-20 · notify 2027-01-04 · camera-ready 2027-01-31
- Short-paper fallback: abstract 2026-11-09, paper 2026-11-16

## Build
```bash
latexmk -pdf main.tex        # one command; latexmkrc handles the Chinese-path bibtex quirk
```
Produces `main.pdf` (currently **10 pages total**: body ends p8, appendix A–D p8–9, refs p9–10;
WWW limit is body ≤ 8 + total ≤ 12). Compiles clean: 0 undefined refs/cites, no overfull boxes.
`latexmk` may print exit code 12 from strict first-pass warnings — the PDF is still
correct and fully resolved. To clean aux files: `latexmk -c`.

> Note: classic BibTeX mishandles the non-ASCII (Chinese) directory path. `latexmkrc`
> works around it by passing BibTeX only the basename. If you ever build by hand, run
> `pdflatex main` → `bibtex main` → `pdflatex main` → `pdflatex main` **inside** this dir.

## Status by section (updated 2026-10-14)
| File | Status |
|---|---|
| `main.tex` | **Abstract rewritten** for the three-stage stack + the separable-trade-off claim. Title changed to "Training-Free Partial-Order Control and Distribution Correction for Discrete-Diffusion POI Check-in Generation" — **re-check before the 10-18 abstract registration, which locks it** |
| `sections/intro.tex` | **Updated**: 3-stage method paragraph, trade-off-decomposition paragraph, 4 contributions |
| `sections/related.tex` | Drafted (positions vs CDD / Marionette / ControlTraj; honest novelty delta) |
| `sections/problem.tex` | **Complete draft** (task, partial-order matrix, eval axes) |
| `sections/method.tex` | Stage (b) only — **still needs two subsections** for category-consistent POI decoding and distribution-aware same-category geometry refinement. Needs Fig.1 + Fig.2 |
| `sections/experiments.tex` | **Numeric findings filled from eval-v2.** New §5.4 decomposes the trade-off. Remaining: Fig.3, cost numbers |
| `sections/conclusion.tex` | Limitations expanded to (i)–(x); conclusion rewritten; new Appendix B (grid + selection) and metric-definition caveats |
| `tables/method-comparison.tex` | **Final eval-v2**, 7 rows (4 baselines + PCDG-P / PCDG-PD / PCDG) |
| `tables/ablation.tex` | **Final eval-v2**, 8 rows incl. `No Distance KL` |
| `tables/stagewise.tex` | **New** — stage-wise ablation (a)–(e), the core evidence for the new contribution |
| `references.bib` | 52 entries (copied from thesis); keys verified, bib resolves |

## What still must be done before submission
- `method.tex`: write the two missing subsections (stages (c) and (d)–(e)); `experiments.tex`
  carries a `\todo` pointing at this.
- Export **Fig.1** (framework) and **Fig.2** (one projected reverse step) from
  `../figure_build/` into `figures/`, following the "must-show / do-NOT-draw" list in
  `../method_innovation_map.md` §11–§15.
- Add **Fig.3**: Unsat (x) vs. Distance JSD (y), one marker per method plus a dashed path
  through stages (a)–(e) per city. This is now a core figure, not optional.
- Fill the remaining `\todo`s: metric definitions (`app:metrics`), cost numbers (`app:cost`),
  the 18-configuration grid table (`app:grid`), and the OOD train/test counts in §5.1.
- Page budget is tight (body already reaches p8). If Fig.1–3 overflow, move
  `tables/ablation.tex` into the appendix and keep the stage-wise table in the body.

## Honesty guardrails (do not overclaim)
- Base generator is **Marionette** (prior work), used frozen — credited, not claimed.
- Projection mechanism is **adapted from CDD** — cited; novelty = partial-order+existence
  energies, category-position restriction, distance-KL, the category-consistent decoder,
  the constraint-neutral geometry refinement, and the training-free empirical win.
- **New York `CategoryTransition` and `G-RANK` remain above the unconstrained backbone**, and
  PO-CFG still wins those two plus `Category` there. Never write "no fidelity loss".
- The ρ/λ/β configuration was chosen on a **training-side calibration split** (not an
  independent validation set) and **neither confirmed candidate cleared all pre-specified
  thresholds**; test metrics never entered selection. Never write "pre-registered and passed".
- Stage (d)→(e) changes **two** factors (ρ and β) — no single-factor attribution.
- `Radius` uses `sqrt(mean(distance_to_centroid))`, **not** standard radius of gyration; tables
  are internally comparable but not against differently defined radii elsewhere.
- All tables are evaluation **v2**; the deterministic G-RANK variant (v3) is a cross-check in
  `app:metrics` only. Never mix a v3 G-RANK into a v2 row.
- Error bars are three-seed sampling SDs at a fixed checkpoint — not significance, not CIs.
- `po_encoding` global condition, hard category→POI logit filtering, complex partial-order DAGs,
  adaptive scheduling, and a stable training-time partial-order loss are **NOT implemented**
  → future work only.

## Before camera-ready
Remove `review` from `\documentclass`, drop the `anonymous` option, fill real author/ORCID,
set the proper `\setcopyright`/`\acmConference`/DOI, make `\todo` a no-op, and re-enable the
ACM reference block (`\settopmatter{printacmref=true}`).
