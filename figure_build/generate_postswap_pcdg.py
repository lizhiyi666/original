"""Editable single-column schematic comparing PostSwap with PCDG.

The routes and POIs are deliberately schematic.  The figure encodes the
method distinction supported by the manuscript: post-generation reordering
keeps sampled POI identities, whereas PCDG controls category presence/order at
sampling time, decodes POIs with category support, and can refine within a
realized category.
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
BUILD = ROOT / "ccfa-workfiles/figures/intro-postswap-pcdg/build"

INK = "#252b32"
MUTED = "#59616a"
RULE = "#c9cfd4"
GRID = "#e7eaec"
BLUE, ORANGE, GREEN = "#0072B2", "#D55E00", "#009E73"
COLORS = {"S": BLUE, "W": ORANGE, "D": GREEN}
SHAPES = {"S": "circle", "W": "square", "D": "triangle"}
_IDS = count()


def tag(artist, name):
    safe = re.sub(r"[^A-Za-z0-9_.:-]+", "-", name).strip("-")
    artist.set_gid(f"{safe}-{next(_IDS)}")
    return artist


def txt(ax, x, y, label, size=7.4, color=INK, align="left", bold=False,
        name="label", **kwargs):
    return tag(ax.text(x, y, label, fontsize=size, color=color, ha=align,
                       va="center", weight="bold" if bold else "normal",
                       linespacing=1.08, **kwargs), name)


def line(ax, points, color=INK, width=.65, dashed=False, name="line"):
    xs, ys = zip(*points)
    return tag(ax.plot(xs, ys, color=color, lw=width,
                       ls=(0, (2.0, 1.7)) if dashed else "-")[0], name)


def arrow(ax, points, color=INK, width=.7, dashed=False, name="data-flow"):
    if len(points) > 2:
        line(ax, points[:-1], color=color, width=width, dashed=dashed,
             name=name + "-route")
    patch = FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
                            mutation_scale=5.2, lw=width, color=color,
                            linestyle=(0, (2.0, 1.7)) if dashed else "-",
                            shrinkA=0, shrinkB=0)
    ax.add_patch(tag(patch, name))


def box(ax, x, y, w, h, edge=RULE, fill="white", name="box", lw=.55,
        radius=0):
    # A square-cornered box keeps the source simple and remains fully editable.
    patch = Rectangle((x, y), w, h, edgecolor=edge, facecolor=fill,
                      linewidth=lw)
    ax.add_patch(tag(patch, name))
    return patch


def symbol(ax, x, y, cat, radius=4.2, label=None, muted=False, name="poi"):
    color = "#a6adb3" if muted else COLORS[cat]
    fill = "#e2e5e7" if muted else to_rgba(COLORS[cat], .14)
    if SHAPES[cat] == "circle":
        patch = Circle((x, y), radius, edgecolor=color, facecolor=fill, lw=.85)
    elif SHAPES[cat] == "square":
        patch = Rectangle((x-radius, y-radius), 2*radius, 2*radius,
                          edgecolor=color, facecolor=fill, lw=.85)
    else:
        patch = Polygon([(x, y+radius*1.15), (x-radius*1.05, y-radius*.9),
                         (x+radius*1.05, y-radius*.9)], closed=True,
                        edgecolor=color, facecolor=fill, lw=.85)
    ax.add_patch(tag(patch, name + "-" + cat))
    if label:
        txt(ax, x, y, label, size=7.1, align="center", color=color, bold=True,
            name=name + "-label")


def sequence(ax, x, y, cats, scale=1.0, name="sequence"):
    step = 23 * scale
    for i, cat in enumerate(cats):
        sx = x + i * step
        symbol(ax, sx, y, cat, radius=4.5 * scale, label=cat,
               name=f"{name}-{i}")
        if i + 1 < len(cats):
            arrow(ax, [(sx + 7*scale, y), (sx + step - 7*scale, y)],
                  color=MUTED, width=.55, name=f"{name}-edge-{i}")


def route_map(ax, bounds, points, route, name, callout=None):
    x, y, w, h = bounds
    box(ax, x, y, w, h, edge=RULE, fill="#fcfcfc", name=name + "-map")
    for fraction in (.33, .66):
        line(ax, [(x + fraction*w, y), (x + fraction*w, y+h)],
             color=GRID, width=.35, name=name + "-grid-v")
        line(ax, [(x, y + fraction*h), (x+w, y + fraction*h)],
             color=GRID, width=.35, name=name + "-grid-h")
    lookup = {}
    for key, (cat, u, v) in points.items():
        px, py = x + u*w, y + v*h
        lookup[key] = (px, py)
        symbol(ax, px, py, cat, radius=4.6, label=cat, name=name + "-poi-" + key)
    for i, (a, b) in enumerate(zip(route, route[1:])):
        p, q = lookup[a], lookup[b]
        dx, dy = q[0]-p[0], q[1]-p[1]
        norm = (dx*dx + dy*dy) ** .5
        p2 = (p[0] + 6*dx/norm, p[1] + 6*dy/norm)
        q2 = (q[0] - 6*dx/norm, q[1] - 6*dy/norm)
        arrow(ax, [p2, q2], color=INK, width=.8, name=name + "-route-" + str(i))
    if callout:
        txt(ax, x+w/2, y-7, callout, size=6.3, color=MUTED, align="center",
            name=name + "-callout")


def panel_frame(ax, x, y, w, h, name):
    box(ax, x, y, w, h, edge=RULE, fill="white", name=name + "-panel", lw=.6)


def draw_postswap(ax, x0, y0, w, h):
    panel_frame(ax, x0, y0, w, h, "postswap")
    txt(ax, x0+7, y0+h-12, "(a) Post-generation reorder", size=8.2, bold=True,
        name="postswap-title")
    txt(ax, x0+8, y0+h-31, "Sampled", size=6.9,
        color=MUTED, name="postswap-generated-label")
    sequence(ax, x0+72, y0+h-31, "WS", scale=.82, name="postswap-generated")
    txt(ax, x0+8, y0+h-52, "After reorder", size=6.9,
        color=MUTED, name="postswap-reordered-label")
    sequence(ax, x0+72, y0+h-52, "SW", scale=.82, name="postswap-reordered")
    points = {"w": ("W", .15, .72), "s": ("S", .86, .25)}
    route_map(ax, (x0+9, y0+22, w-18, 37), points, ("s", "w"),
              "postswap-route", None)
    txt(ax, x0+w/2, y0+12, "Same POIs; long inter-area move", size=6.4,
        color=MUTED, align="center", name="postswap-route-caption")
    txt(ax, x0+w/2, y0+3, "Missing category: unavailable.", size=6.4,
        color=MUTED, name="postswap-missing")


def draw_pcdg(ax, x0, y0, w, h):
    panel_frame(ax, x0, y0, w, h, "pcdg")
    txt(ax, x0+7, y0+h-12, "(b) PCDG during generation", size=8.2, bold=True,
        name="pcdg-title")
    # Three compact stages mirror the implemented stack without claiming a
    # deterministic nearest-POI rule.
    stages = [
        ("1", "Order + existence", ORANGE),
        ("2", "Category-aware decode", BLUE),
        ("3", "Same-category refine", GREEN),
    ]
    ys = [y0+h-37, y0+h-58, y0+h-79]
    for i, ((num, title, accent), yy) in enumerate(zip(stages, ys)):
        box(ax, x0+8, yy-7, w-16, 15, edge=to_rgba(accent, .85),
            fill=to_rgba(accent, .055), name=f"pcdg-stage-{num}", lw=.75)
        txt(ax, x0+13, yy, num, size=7.1, color=accent, bold=True,
            align="center", name=f"pcdg-stage-{num}-num")
        txt(ax, x0+25, yy, title, size=6.7, bold=True,
            name=f"pcdg-stage-{num}-title")
        if i + 1 < len(stages):
            arrow(ax, [(x0+w/2, yy-8), (x0+w/2, ys[i+1]+8)],
                  color=MUTED, width=.55, name=f"pcdg-stage-edge-{num}")
    points = {"s": ("S", .25, .25), "w": ("W", .68, .72)}
    route_map(ax, (x0+9, y0+22, w-18, 37), points, ("s", "w"),
              "pcdg-route", None)
    txt(ax, x0+w/2, y0+12, "One possible refined route", size=6.4,
        color=MUTED, align="center", name="pcdg-route-caption")
    txt(ax, x0+w/2, y0+3, "Categories preserved.", size=6.3,
        color=MUTED, align="center", name="pcdg-mapping")


def make_figure():
    global _IDS
    _IDS = count()
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 7.4,
                         "mathtext.fontset": "stix", "pdf.fonttype": 42,
                         "svg.fonttype": "none", "svg.hashsalt": "pcdg-intro"})
    fig = plt.figure(figsize=(3.33, 2.55))
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, 333), ylim=(0, 255))
    ax.axis("off")
    ax.set_gid("pcdg-postswap-intro")
    txt(ax, 166.5, 248, "Requested order: sightseeing before shopping", size=8.0,
        align="center", bold=True, name="requested-relation")
    txt(ax, 166.5, 235, "Both categories must occur; routes are schematic.", size=6.5,
        align="center", color=MUTED, name="schematic-note")
    draw_postswap(ax, 5, 10, 155, 221)
    draw_pcdg(ax, 173, 10, 155, 221)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for label in ax.texts:
        assert label.get_fontsize() >= 6.3
        assert label.get_fontfamily() == ["Times New Roman"]
        bbox = label.get_window_extent(renderer)
        assert fig.bbox.contains(bbox.x0, bbox.y0) and fig.bbox.contains(bbox.x1, bbox.y1), label.get_text()
    return fig


def main(output_dir=OUT, preview_dir=BUILD):
    output_dir, preview_dir = Path(output_dir), Path(preview_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    fig = make_figure()
    fig.savefig(output_dir / "postswap-pcdg.pdf",
                metadata={"Creator": "Matplotlib", "CreationDate": None})
    fig.savefig(output_dir / "postswap-pcdg.svg", metadata={"Date": None})
    svg = output_dir / "postswap-pcdg.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    fig.savefig(preview_dir / "postswap-pcdg.png", dpi=260)
    plt.close(fig)
    print("Exported postswap-pcdg.pdf/.svg and preview; 3.33 x 2.55 in; live text")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--preview-dir", type=Path, default=BUILD)
    args = parser.parse_args()
    main(args.output_dir, args.preview_dir)
