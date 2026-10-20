"""Four-panel editable paper mechanism figure at 7 x 3.25 inches.

User-authorized code-first redesign (2026-10-20) after the built-in concept
renderer was unavailable. All points/logits are schematic, not measurements.
One shared catalog drives panels (c) and (d). No scientific modules are imported.
Source anchors: diffusion_transformer.py::_apply_category_consistent_poi_mask /
log_sample_categorical; constraint_projection.py; geometry_projection.py::
assert_invariants; paper-www2027/sections/method.tex.

Geometric-flow: 504 x 234 pt, Times New Roman, 9 pt labels, 10 pt headings,
8 pt secondary labels. Solid data flow; dashed conditions/objectives. Lower
panels approximately 28/42/30 percent. A=blue circle, B=orange square,
C=green triangle. SVG has live text and semantic artist IDs, no embedded image.
Default entry point and destinations remain; output overrides permit candidate
QA before replacing the publication exports.
"""
import argparse
from itertools import count
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper-www2027/figures"
BUILD = ROOT / "ccfa-workfiles/writing/www2027/build"
INK, MUTED, RULE = "#252b32", "#59616a", "#ccd2d6"
BLUE, ORANGE, GREEN = "#0072B2", "#D55E00", "#008560"
CATEGORY = {"A": (BLUE, "circle"), "B": (ORANGE, "square"),
            "C": (GREEN, "triangle")}
# Shared reference geometry in candidate fields and both refinement maps.
CATALOG = {
    "a1": ("A", .16, .23), "a2": ("A", .28, .68),
    "b1": ("B", .51, .84), "b2": ("B", .54, .36),
    "c1": ("C", .88, .19), "c2": ("C", .85, .68),
}
BEFORE = ("a1", "b1", "c1")
AFTER = ("a2", "b2", "c2")
_IDS = count()


def tag(artist, name):
    safe_name = re.sub(r"[^A-Za-z0-9_.:-]+", "-", name).strip("-")
    artist.set_gid(f"{safe_name}-{next(_IDS)}")
    return artist


def text(ax, x, y, label, size=9, color=INK, align="left", bold=False,
         name="label", **kwargs):
    return tag(ax.text(x, y, label, fontsize=size, color=color, ha=align,
                       va="center", weight="bold" if bold else "normal",
                       linespacing=1.12, **kwargs), name)


def line(ax, points, color=INK, width=.7, dashed=False, name="line", **kwargs):
    xs, ys = zip(*points)
    return tag(ax.plot(xs, ys, color=color, lw=width,
                       ls=(0, (2.5, 2)) if dashed else "-", **kwargs)[0], name)


def arrow(ax, points, color=INK, dashed=False, name="data-flow", width=.75):
    if len(points) > 2:
        line(ax, points[:-1], color, width, dashed, name)
    patch = FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
                            mutation_scale=6, lw=width, color=color,
                            linestyle=(0, (2.5, 2)) if dashed else "-",
                            shrinkA=0, shrinkB=0)
    ax.add_patch(tag(patch, name))


def rect(ax, x, y, width, height, edge=RULE, fill="white", name="region", lw=.6):
    patch = Rectangle((x, y), width, height, facecolor=fill,
                      edgecolor=edge, linewidth=lw)
    ax.add_patch(tag(patch, name))
    return patch


def node(ax, x, y, width, height, label, accent=INK):
    rect(ax, x, y, width, height, edge=accent,
         fill=to_rgba(accent, .055), name="operator-" + label.replace("\n", "-"))
    text(ax, x+width/2, y+height/2, label, align="center")


def symbol(ax, x, y, category, radius=3.2, label=None, inactive=False,
           selected=False, name="category"):
    color, shape = CATEGORY[category]
    edge = "#a4abb0" if inactive else color
    fill = "#e2e5e7" if inactive else to_rgba(color, .12)
    if shape == "circle":
        patch = Circle((x, y), radius, edgecolor=edge, facecolor=fill, lw=.85)
    elif shape == "square":
        patch = Rectangle((x-radius, y-radius), 2*radius, 2*radius,
                          edgecolor=edge, facecolor=fill, lw=.85)
    else:
        patch = Polygon([(x, y+radius*1.15), (x-radius*1.05, y-radius*.9),
                         (x+radius*1.05, y-radius*.9)], closed=True,
                        edgecolor=edge, facecolor=fill, lw=.85)
    ax.add_patch(tag(patch, f"{name}-{category}"))
    if label:
        text(ax, x, y, label, size=9, align="center", color=color, bold=True)
    if inactive:
        line(ax, [(x-radius-1, y-radius-1), (x+radius+1, y+radius+1)],
             color="#949ca3", width=.6, name="masked-out")
    if selected:
        ring = Circle((x, y), radius+2.0, facecolor="none", edgecolor=INK, lw=.65)
        ax.add_patch(tag(ring, "selected-poi"))


def location(bounds, key):
    x, y, width, height = bounds
    _, u, v = CATALOG[key]
    return x + width*u, y + height*v


def catalog(ax, bounds, category=None, chosen=(), route=False, labels=False):
    x, y, width, height = bounds
    rect(ax, x, y, width, height, fill="#fcfcfc", name="catalog-coordinate-frame")
    for fraction in (.25, .5, .75):
        line(ax, [(x+width*fraction, y), (x+width*fraction, y+height)],
             color="#e8ebed", width=.35, name="coordinate-grid")
        line(ax, [(x, y+height*fraction), (x+width, y+height*fraction)],
             color="#e8ebed", width=.35, name="coordinate-grid")
    if route:
        for a, b in zip(chosen, chosen[1:]):
            p, q = location(bounds, a), location(bounds, b)
            dx, dy = q[0]-p[0], q[1]-p[1]
            norm = (dx*dx+dy*dy)**.5
            p = (p[0]+dx/norm*4, p[1]+dy/norm*4)
            q = (q[0]-dx/norm*4, q[1]-dy/norm*4)
            arrow(ax, [p, q], width=.9, name=f"route-{a}-to-{b}")
    for key, (cat, _, _) in CATALOG.items():
        px, py = location(bounds, key)
        symbol(ax, px, py, cat, radius=2.6 if width < 60 else 3.0,
               inactive=category is not None and cat != category,
               selected=key in chosen and not route, name=f"catalog-{key}")
        if labels and key in chosen:
            ox, oy = {"a1": (-3, -9), "a2": (-1, 9), "b1": (0, 9),
                      "b2": (0, -10), "c1": (0, -9), "c2": (0, 9)}[key]
            text(ax, px+ox, py+oy, key, size=8, align="center")


def overview(ax):
    text(ax, 5, 225, "(a) Sampling-time control", size=10, bold=True)
    text(ax, 499, 225, "Time branch unchanged", size=8, color=MUTED, align="right")
    # Refinement is outside the loop, following its terminal exit.
    rect(ax, 47, 157, 333, 57, edge="#aeb7be", fill="#fafbfc", name="reverse-loop")
    text(ax, 54, 207, "Reverse diffusion", size=8, color=MUTED)
    text(ax, 191.5, 210, "Partial order M; ρ, λ", size=8, align="center", color=MUTED)
    arrow(ax, [(191.5, 206), (191.5, 202.5)], dashed=True, color=MUTED,
          name="partial-order-condition")
    node(ax, 55, 177, 73, 25, "Frozen\ndenoiser")
    node(ax, 153, 177, 77, 25, "(b) Logit\nprojection", ORANGE)
    node(ax, 255, 177, 116, 25, "(c) Category-consistent\ndecoding", BLUE)
    text(ax, 191.5, 168, "Controlled late steps", size=8, color=ORANGE, align="center")
    text(ax, 7, 198, "Time /", size=8, color=MUTED)
    text(ax, 7, 189, "context", size=8, color=MUTED)
    arrow(ax, [(36, 194), (53, 194)], dashed=True, color=MUTED, name="time-context")
    text(ax, 13, 176, "x", size=10, align="center")
    text(ax, 18, 172, "s", size=8)
    arrow(ax, [(28, 179), (53, 179)], name="noisy-state-input")
    arrow(ax, [(129, 189.5), (151, 189.5)])
    arrow(ax, [(231, 189.5), (253, 189.5)])
    arrow(ax, [(314, 176), (314, 161), (91, 161), (91, 176)], name="reverse-feedback")
    text(ax, 326, 167, "x", size=10, color=MUTED)
    text(ax, 331, 163, "s−1", size=8, color=MUTED)
    arrow(ax, [(372, 189.5), (415, 189.5)], name="terminal-exit")
    text(ax, 394, 199, "s = 0", size=8, align="center")
    text(ax, 458, 212, "Training targets; β", size=8, align="center", color=MUTED)
    arrow(ax, [(458, 207), (458, 203)], dashed=True, color=MUTED,
          name="training-refinement-target")
    node(ax, 417, 177, 82, 25, "(d) Same-category\nrefinement", GREEN)
    text(ax, 458, 164, "Catalog POI trajectory", size=8, align="center", color=MUTED)
    line(ax, [(5, 151), (499, 151)], color=RULE, name="overview-separator")


def logits(ax):
    text(ax, 5, 140, "(b) Logit projection", size=10, bold=True)
    text(ax, 5, 127, "Order + existence; λ·category JSD", size=8)
    text(ax, 32, 114, "Base", size=8, align="center", color=MUTED)
    text(ax, 110, 114, "Projected", size=8, align="center", color=ORANGE)
    for prefix, y in (("k", 93), ("l", 69)):
        for col, start in enumerate((8, 86)):
            for index in range(3):
                x = start+16*index
                rect(ax, x, y, 16, 13, edge=ORANGE if col else "#9aa4ad",
                     fill=to_rgba(ORANGE if col else BLUE, .07+.04*index),
                     name=f"{'projected' if col else 'base'}-{prefix}{index+1}")
                text(ax, x+8, y+6.5, f"{prefix}{index+1}", size=8, align="center")
        arrow(ax, [(60, y+6.5), (82, y+6.5)], color=ORANGE, name=f"project-{prefix}")
    arrow(ax, [(136, 127), (139, 127), (139, 99.5), (135, 99.5)],
          color=ORANGE, dashed=True, name="category-energies")
    text(ax, 8, 86.5, "Category positions", size=8, color=MUTED)
    text(ax, 8, 61.5, "POI positions", size=8, color=MUTED)
    text(ax, 104, 51, "Distance KL", size=8, color=ORANGE, align="center")
    arrow(ax, [(111, 56), (111, 68)], color=ORANGE, dashed=True, name="distance-objective")
    text(ax, 5, 37, "Training reference; weights", size=8, color=MUTED)
    arrow(ax, [(112, 37), (132, 37), (132, 51), (128, 51)],
          dashed=True, color=MUTED, name="training-distance-target")
    arrow(ax, [(132, 37), (142, 37), (142, 132), (106, 132)],
          dashed=True, color=MUTED, name="training-category-target")
    text(ax, 5, 23, "Token KL: base → projected", size=8, bold=True)


def decoding(ax):
    text(ax, 153, 140, "(c) Category-consistent decoding", size=10, bold=True)
    text(ax, 251, 123, "Sample category with draw g", size=9, align="center")
    centers = (183, 249, 315)
    for event, (x, cat, key) in enumerate(zip(centers, CATEGORY, BEFORE), 1):
        symbol(ax, x, 108, cat, radius=6.6, label=cat, name=f"event-{event}-category")
        arrow(ax, [(x, 99), (x, 91)], color=CATEGORY[cat][0], dashed=True,
              name=f"event-{event}-condition")
        catalog(ax, (x-25, 49, 50, 32), category=cat, chosen=(key,))
        arrow(ax, [(x, 47), (x, 40)], name=f"event-{event}-poi-reselection")
        symbol(ax, x-9, 32, cat, radius=3, name=f"event-{event}-decoded-poi")
        text(ax, x+1, 32, key, size=9)
    text(ax, 249, 86.5, "Filter catalog by sampled category", size=8, align="center")
    text(ax, 249, 21, "Reselect with same g; category tokens retained", size=8, align="center", bold=True)


def refinement(ax):
    text(ax, 361, 140, "(d) Same-category refinement", size=10, bold=True)
    text(ax, 393, 123, "Before", size=9, align="center")
    text(ax, 470, 123, "After", size=9, align="center")
    catalog(ax, (362, 64, 62, 47), chosen=BEFORE, route=True, labels=True)
    catalog(ax, (437, 64, 62, 47), chosen=AFTER, route=True, labels=True)
    arrow(ax, [(426, 88), (435, 88)], color=GREEN, name="same-category-replacement")
    text(ax, 430, 51, "Change: POI identity / position", size=8, align="center")
    text(ax, 430, 37, "Fixed: actual categories A → B → C", size=8, align="center", bold=True)
    text(ax, 430, 24, "Times, length, event order", size=9, align="center")


def make_figure():
    global _IDS
    _IDS = count()
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 9,
                         "mathtext.fontset": "stix", "pdf.fonttype": 42,
                         "svg.fonttype": "none", "svg.hashsalt": "pcdg-www2027"})
    fig = plt.figure(figsize=(7, 3.25))
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, 504), ylim=(0, 234))
    ax.axis("off")
    ax.set_gid("pcdg-framework")
    overview(ax)
    logits(ax)
    decoding(ax)
    refinement(ax)
    for x in (147, 355):
        line(ax, [(x, 18), (x, 146)], color=RULE, width=.55, name="detail-separator")
    text(ax, 5, 7, "Schematic. Solid: data flow; dashed: conditions / objectives.", size=8, color=MUTED)
    text(ax, 499, 7, "Final unsupported category → any catalog POI", size=8,
         color=MUTED, align="right")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for label in ax.texts:
        assert label.get_fontsize() >= 8
        assert label.get_fontfamily() == ["Times New Roman"]
        bbox = label.get_window_extent(renderer)
        assert fig.bbox.contains(bbox.x0, bbox.y0) and fig.bbox.contains(bbox.x1, bbox.y1), label.get_text()
    assert tuple(CATALOG[k][0] for k in BEFORE) == tuple(CATALOG[k][0] for k in AFTER)
    return fig


def main(output_dir=OUT, preview_dir=BUILD):
    output_dir, preview_dir = Path(output_dir), Path(preview_dir)
    fig = make_figure()
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "framework.pdf",
                metadata={"Creator": "Matplotlib", "CreationDate": None})
    fig.savefig(output_dir / "framework.svg", metadata={"Date": None})
    svg = output_dir / "framework.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    fig.savefig(preview_dir / "framework.png", dpi=220)
    plt.close(fig)
    print("Exported 4-panel framework.pdf/.svg and preview; 7 x 3.25 in; live text >= 8 pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--preview-dir", type=Path, default=BUILD)
    args = parser.parse_args()
    main(args.output_dir, args.preview_dir)
