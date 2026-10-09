# Method Main Figure V1 Brief

## Scientific purpose

This figure presents a code-grounded overview of a conditional POI check-in sequence generator with category partial-order control. The central scientific message is that the partial-order mechanism is inserted into reverse discrete diffusion before joint category/POI sampling.

## Scope

- Panel (a) shows the end-to-end path from POI event representation and context to temporal/discrete generation and the generated POI sequence.
- Panel (b) expands one reverse step: sample-level `po_matrix` → constraint parsing → order/existence energy → KL-preserving ALM projection on category logits → joint sampling with unchanged POI logits.
- The diagram does not claim POI candidate filtering, explicit POI transition restriction, `po_encoding` conditioning, complete CFG, or complex-DAG adaptation.

## Core contribution highlighted

Category-order constraint-aware reverse generation is the only high-emphasis module. It is shown as an inference-time control layer, not as a new denoiser or a separate POI generator.

## Missing requested resource

The requested `academic-figures-drawer` skill was not installed or exposed in this environment. The figure was produced with a local editable Draw.io XML source and a static vector rendering fallback, following the supplied method-invention map and visual contract.
