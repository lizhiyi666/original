# Defect Log — V1

## Resolved during V1 rendering

- Reduced overlaps between overview generator title and its branch boxes.
- Reduced overlaps in the constraint-aware generation box.
- Separated forward-diffusion and reverse-denoising labels.
- Reduced Panel (b) annotation sizes and kept the split logits readable.

## Known limitations

1. The requested `academic-figures-drawer` skill is unavailable in the current environment; this V1 uses a local Matplotlib/vector fallback plus an editable Draw.io XML source.
2. The Draw.io XML has not been opened in diagrams.net in this environment because a Draw.io desktop executable was not found. XML parsing and structural inspection are still required before publication.
3. The repository does not contain `scripts/validate_visual_quality.py` or `scripts/validate_drawio.py`; the requested project-provided validators could not be executed.
4. Panel (a) uses a compact schematic token strip rather than a full data example; V2 may refine token alignment and the reverse-step callout.
5. The phrase “No candidate filtering / no explicit POI transition restriction” is intentionally included as a scientific guardrail; it may be moved to a caption or detail figure in a final thesis layout.

## Scientific audit status

- Input: present.
- Condition: present.
- Diffusion: present.
- Reverse generation: present.
- Category-order constraint: present and highlighted.
- Constraint mechanism: present in Panel (b).
- Output: present.

## V2 status (2026-09-06)

### Resolved

1. `po_matrix` is now shown as an independent constraint specification entering the reverse step, not as a condition-encoder output.
2. Training and inference/generation are separated into explicit lanes.
3. The category-order projection is embedded between category-position logits and the next reverse state inside the reverse diffusion loop.
4. Panel (a) and Panel (b) share callout `①`, making the detail relationship explicit.
5. The mask is labeled as selecting sequence positions; POI logits are explicitly marked unchanged.
6. The editable Draw.io XML root was repaired and structurally checked.

### Remaining limitations

1. Draw.io has not been opened in the diagrams.net desktop application in this environment; XML parsing and endpoint checks pass.
2. Repository visual validators are unavailable, so no project-validator result is claimed.
3. At thesis single-column scale, the reverse-loop inset and the two logits branches should receive one final print-size readability check.

### V2 scientific audit status

- Input and generation-time conditions: present and separated from training data.
- Condition representation: present only on the training/model-learning lane.
- Forward diffusion training and reverse diffusion generation: distinguished.
- Constraint specification source and reverse-step insertion: corrected.
- Category-position-only projection and unchanged POI logits: explicit.
- Final generated POI sequence: present.

## Skill-based source refinement (2026-09-07)

### Fixed

- Reparented panel, loop, output, and mask objects into explicit Draw.io containers with position-preserving relative geometry.
- Removed the duplicate `proj_a` projection node; `a_proj` is the sole reverse-step projection cell and carries the `①` callout.
- Added native editable reverse-step cells and line primitives for the divider and legend.
- Converted reverse diffusion axis notation to editable subscript labels.

### Validation notes

- `validate_drawio.py`: passed (84 cells, 64 vertices, 18 edges, no duplicate IDs, no embedded raster).
- `validate_visual_quality.py`: still emits conservative failures for filled containers and nested coordinates because its heuristic parser does not resolve Draw.io parent transforms; token-strip decoration warnings are intentional semantic mask/output tokens. No real P0/P1 overlap was observed in the generated figure.
- Preview generated with `make_drawio_preview.py` and served locally for inspection.
