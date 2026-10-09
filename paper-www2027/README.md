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
Produces `main.pdf` (currently **8 pages**, compiles clean: 0 undefined refs/cites).
`latexmk` may print exit code 12 from strict first-pass warnings — the PDF is still
correct and fully resolved. To clean aux files: `latexmk -c`.

> Note: classic BibTeX mishandles the non-ASCII (Chinese) directory path. `latexmkrc`
> works around it by passing BibTeX only the basename. If you ever build by hand, run
> `pdflatex main` → `bibtex main` → `pdflatex main` → `pdflatex main` **inside** this dir.

## Status by section
| File | Status |
|---|---|
| `sections/intro.tex` | Drafted (incl. required page-1 **Web-relevance** paragraph + 3 contributions) |
| `sections/related.tex` | Drafted (positions vs CDD / Marionette / ControlTraj; honest novelty delta) |
| `sections/problem.tex` | **Complete draft** (task, partial-order matrix, eval axes) |
| `sections/method.tex` | **Complete draft** — matches the code (order+existence energy, KL-ALM projection, distance-KL). Needs Fig.1 + Fig.2. |
| `sections/experiments.tex` | Scaffold: design is written; **numeric findings are TODO, pending distance-v2** |
| `sections/conclusion.tex` | Limitations + Conclusion + Appendix (algorithm box, metric/cost TODOs) |
| `tables/method-comparison.tex` | **Provisional** numbers from `integrated-20261005` (eval-v1 anchor) |
| `tables/ablation.tex` | **Provisional** (eval-v1 anchor; v2 adds a `No Distance KL` row) |
| `references.bib` | 52 entries (copied from thesis); keys verified, bib resolves |

## What is PROVISIONAL and must be refreshed from distance-v2 before submission
- All table numbers → swap to **evaluation v2** (CategoryTransition replaces Interval)
  on the fresh **66-result distance-v2** run; PCDG rows will reflect the distance-KL term.
- Add the `No Distance KL` ablation row.
- Fill every `\todo{...}` in `experiments.tex`, `method.tex`, `conclusion.tex`.
- Export **Fig.1** (framework) and **Fig.2** (one projected reverse step) from
  `../figure_build/` into `figures/`, following the "must-show / do-NOT-draw" list in
  `../method_innovation_map.md` §11–§15.
- Add **Fig.3**: Unsat (x) vs. Distance/Radius JSD (y) scatter, one point per method/city.

## Honesty guardrails (from the code cross-validation; do not overclaim)
- Base generator is **Marionette** (prior work), used frozen — credited, not claimed.
- Projection mechanism is **adapted from CDD** — cited; novelty = partial-order+existence
  energies, category-position restriction, distance-KL, and the training-free empirical win.
- Report the **fidelity cost** (radius/distance JSD) honestly; PO-CFG wins raw fidelity.
- `po_encoding` global condition, hard category→POI filtering, complex partial-order DAGs,
  adaptive scheduling, and a stable training-time partial-order loss are **NOT implemented**
  → future work only.

## Before camera-ready
Remove `review` from `\documentclass`, drop the `anonymous` option, fill real author/ORCID,
set the proper `\setcopyright`/`\acmConference`/DOI, make `\todo` a no-op, and re-enable the
ACM reference block (`\settopmatter{printacmref=true}`).
