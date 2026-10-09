# Method Main Final Source Synchronization Log

## Scope

Synchronized `method_main_final.drawio` with the already approved V2 static exports without changing the algorithm, layout, terminology, color system, panel hierarchy, or scientific meaning.

## 1. Added objects

The following previously flattened or missing static-export objects were restored as independent native Draw.io cells:

- Reverse diffusion loop title and axis text;
- Internal reverse-step cells: `x_t`, `Denoising`, `Category-position logits`, `① Category-order projection`, and `x_{t-1}`;
- Connectors for `x_t → Denoising → Category-position logits → projection → x_{t-1}`;
- `TRAINING` and `INFERENCE / GENERATION` lane labels and their auxiliary field labels;
- Training subtitles and shared-generator annotation;
- Direct reverse-step annotation for `Constraint specification / po_matrix`;
- Output metadata annotation and POI token cells;
- `M[A,B]=1`, `A → B`, order/existence penalty annotations;
- `category_mask` sequence-position annotation and editable position tokens `1, 0, 1, …, 1` with position labels;
- `position selector, not a POI candidate set` boundary annotation;
- ALM projection annotations;
- `POI logits bypass projection`;
- `No candidate filtering · No explicit POI transition restriction · Constraint acts during reverse diffusion`;
- Legend text cells and the existing `①` overview/detail correspondence.

All added objects are `mxCell` vertices or connectors. No image element was inserted.

## 2. Repaired objects

- Replaced the former text-only reverse-loop label with an editable outer loop cell plus editable internal reverse-step cells.
- Preserved the existing panel cells, module cells, colors, coordinates, and terminology.
- Preserved the `po_matrix` dashed constraint path and the Panel (a) → Panel (b) `①` callout.
- Preserved `category_mask` semantics as sequence-position selection, not POI candidate filtering.

## 3. Deleted objects

No scientific module or panel was deleted. The previous non-editable text embedded in the outer loop label was replaced by equivalent editable text/cells; this is a source-structure replacement, not a method change.

## 4. Export synchronization

Generated final deliverables:

- `method_main_final.drawio`
- `method_main_final.svg`
- `method_main_final.pdf`
- `method_main_final.png`

The static exports retain the approved V2 geometry and content. The Draw.io source now contains the same visible object inventory as the static figure at the level required for editing; all major boxes, labels, annotations, tokens, and connectors are native cells.

## 5. Verification

- Draw.io XML parse: passed.
- Native cells: 62 vertices and 18 connectors.
- Connector endpoint errors: 0.
- Off-canvas vertices: 0.
- Embedded image elements in Draw.io XML: 0.
- SVG image elements: 0.
- PDF raster-image objects: 0 (`pdfimages -list` returned no image rows).
- Required scientific labels found in the source: `x_t`, `x_{t-1}`, `category_mask`/sequence-position wording, `POI logits (unchanged)`, `No candidate filtering`, `No explicit POI transition restriction`, and `①` correspondence.

## 6. Equality statement

At the visual/content level, the final Draw.io source is synchronized with the final SVG/PDF/PNG. The four files were delivered from the same approved V2 geometry and terminology set; the Draw.io source additionally exposes every major static object as an editable cell. No algorithm code was modified.

## 7. Skill-based refinement pass (2026-09-07)

- Added explicit parent hierarchy for panels, reverse loop, output tokens, and category-mask tokens while preserving their rendered positions.
- Removed the duplicate `proj_a` vertex; the single `a_proj` cell is now the source and target of the reverse-step constraint callout.
- Converted reverse-axis notation to editable HTML subscript labels (`x_T`, `x_t`, `x_{t−1}`, `x_0`).
- Added editable divider and legend line primitives.
- Regenerated the skill preview at `method_main_final_preview.html`.

The structural validator remains `OK` (84 cells, no duplicate IDs, no images). The visual-quality heuristic still reports container/child overlap and decorative-token warnings because it does not resolve nested Draw.io coordinates or recognize semantic token strips; these are documented as conservative false positives. The duplicate projection and actual source-fidelity issue are resolved.
