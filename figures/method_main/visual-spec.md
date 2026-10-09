# Visual Specification

## Typography

- Landscape, paper-oriented sans-serif hierarchy.
- Panel titles: bold, largest visible text.
- Core modules: bold medium-size labels.
- Mechanism annotations: smaller muted text.
- No implementation-level class/function names in the main visual.

## Color semantics

- Blue/blue-gray: input, condition, output, and general model structure.
- Gray: temporal Add-Thin and unchanged/neutral distributions.
- Green: category/POI discrete diffusion branch.
- Terracotta accent: category-order contribution and all corresponding Panel (b) operations.
- White/neutral token cells: special or structural tokens.

The accent color is reserved for the contribution chain and is shared across both panels.

## Borders and boxes

- Large rounded light panels define Panel (a) and Panel (b).
- Core model boxes use moderate rounded corners and restrained fills.
- Contribution boxes use a thicker terracotta border and pale terracotta fill.
- No gradients, shadows, 3D effects, or decorative icons.

## Arrows

- Solid dark arrows: primary data flow.
- Dashed terracotta arrow: constraint/control relation from the sample-level matrix to reverse generation.
- Light dashed arrows: secondary condition/control links.
- Panel (b) uses a left-to-right constraint chain, then a split into category and POI logits before joint sampling.

## Spacing and density

- Two clearly bounded landscape panels.
- Panel (a) occupies the upper overview area; Panel (b) occupies the lower detail area.
- Main nodes are aligned on a common horizontal flow; the reverse-step detail is separated from the overview to avoid overloading the main path.
- The figure is intentionally compact but leaves whitespace around arrows and labels for later V2 refinement.

## Panel styles

- Panel (a): neutral overview with one accent control insertion.
- Panel (b): accent-dominant mechanism detail with gray POI-logit bypass.
