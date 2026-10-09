# Layout Grid

## Canvas

- Logical canvas: 1600 × 1000 units.
- Orientation: landscape.
- Exported formats: PNG, SVG, PDF, editable Draw.io XML.

## Panel bounds

- Panel (a): x=35, y=480, w=1530, h=475.
- Panel (b): x=35, y=35, w=1530, h=405.

## Panel (a) module positions

- Input: x=75, y=610, w=260, h=245.
- Condition Representation: x=395, y=770, w=275, h=85.
- Conditional Joint Generator: x=395, y=545, w=560, h=185.
- Temporal Add-Thin: x=425, y=580, w=220, h=90.
- Category/POI Discrete Diffusion: x=700, y=580, w=220, h=90.
- Category-order Constraint-aware Generation: x=1005, y=555, w=280, h=170.
- Generated POI Sequence: x=1335, y=610, w=205, h=110.

## Panel (b) module positions

- Sample-level po_matrix: x=75, y=180, w=230, h=175.
- Constraint Parsing: x=360, y=205, w=215, h=125.
- Order + Existence Energy: x=630, y=177, w=275, h=180.
- KL-preserving ALM Projection: x=960, y=165, w=285, h=205.
- Category logits: x=1300, y=285, w=205, h=70.
- POI logits (unchanged): x=1300, y=175, w=205, h=70.
- Joint Gumbel-max Sampling: x=1285, y=62, w=235, h=70.

## Alignment rules

- Primary overview flow is left-to-right.
- Constraint detail chain is left-to-right.
- Category and POI logits are vertically stacked and converge at joint sampling.
- Accent control relation is the only long cross-module control arrow.
