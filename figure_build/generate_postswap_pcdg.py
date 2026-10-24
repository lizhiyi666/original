"""Editable reference-guided Figure 1: identical maps, different POI choices.

All six POIs, coordinates and street geometry are shared by both panels.
Routes are illustrative, not measured samples or a shortest-path guarantee.
Icons are original geometric primitives; no external icon assets are required.
"""
import argparse
from itertools import count
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper-www2027/figures"
BUILD = ROOT / "ccfa-workfiles/figures/intro-postswap-pcdg/build"
INK, MUTED, RULE = "#243447", "#617080", "#cbd2d4"
COLORS = {"S": "#8261ad", "W": "#188da4", "D": "#d88422", "H": "#5073a8"}
ROUTE_COLORS = {"postswap": "#b95635", "pcdg": "#23866f"}
CATALOG = {
    "s-east": ("S", .67, .76), "w-west": ("W", .16, .76),
    "w-east": ("W", .76, .54), "d-west": ("D", .16, .30),
    "d-east": ("D", .625, .325), "h-east": ("H", .79, .13),
}
ROUTES = {
    "postswap": ("s-east", "w-west", "d-west", "h-east"),
    "pcdg": ("s-east", "w-east", "d-east", "h-east"),
}
MAP_BOUNDS = {"postswap": (16, 66, 319, 202), "pcdg": (365, 66, 319, 202)}
_IDS = count()


def tag(artist, name):
    safe = re.sub(r"[^A-Za-z0-9_.:-]+", "-", name).strip("-")
    artist.set_gid(f"{safe}-{next(_IDS)}")
    return artist


def txt(ax, x, y, label, size=8.5, color=INK, align="center", bold=False, name="label"):
    return tag(ax.text(x, y, label, fontsize=size, color=color, ha=align,
                       va="center", weight="bold" if bold else "normal"), name)


def line(ax, points, color, width=.85, name="line", zorder=3):
    xs, ys = zip(*points)
    return tag(ax.plot(xs, ys, color=color, lw=width, solid_capstyle="round",
                       solid_joinstyle="round", zorder=zorder)[0], name)


def rect(ax, x, y, w, h, color, name, radius=0, edge="none", lw=.6, zorder=1):
    if radius:
        patch = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                              facecolor=color, edgecolor=edge, linewidth=lw, zorder=zorder)
    else:
        patch = Rectangle((x, y), w, h, facecolor=color, edgecolor=edge,
                          linewidth=lw, zorder=zorder)
    ax.add_patch(tag(patch, name))
    return patch


def icon(ax, x, y, cat, radius=11.5, name="poi"):
    color = COLORS[cat]
    ax.add_patch(tag(Circle((x, y), radius, facecolor="white", edgecolor=color,
                            linewidth=.7, zorder=4), name + "-ring"))

    def stroke(points, suffix):
        return line(ax, [(x + a*radius, y + b*radius) for a, b in points], color,
                    width=.95, name=name + "-" + suffix, zorder=5)

    if cat == "S":
        stroke([(-.65, .28), (0, .67), (.65, .28), (-.65, .28)], "pediment")
        for i, u in enumerate((-.43, 0, .43)):
            stroke([(u, -.45), (u, .17)], f"column-{i}")
        stroke([(-.65, -.55), (.65, -.55)], "base")
    elif cat == "W":
        stroke([(-.64, .32), (-.46, .66), (.46, .66), (.64, .32), (-.64, .32)], "awning")
        stroke([(-.49, .22), (-.49, -.55), (.49, -.55), (.49, .22)], "shop")
        stroke([(-.15, -.55), (-.15, -.15), (.19, -.15), (.19, -.55)], "door")
    elif cat == "D":
        stroke([(-.35, -.65), (-.35, .58)], "fork-stem")
        stroke([(-.61, .58), (-.61, .09), (-.10, .09), (-.10, .58)], "fork")
        stroke([(.34, -.65), (.34, .59), (.64, .27), (.34, -.03)], "knife")
    elif cat == "H":
        stroke([(-.65, -.60), (-.65, .49)], "bed-head")
        stroke([(-.65, -.31), (.65, -.31), (.65, .05), (-.65, .05)], "bed")
        stroke([(.65, -.31), (.65, -.60)], "bed-foot")
        rect(ax, x-.49*radius, y+.10*radius, .32*radius, .22*radius,
             color, name + "-pillow", zorder=5)
    else:
        raise ValueError(cat)


def location(bounds, key):
    x, y, w, h = bounds
    _, u, v = CATALOG[key]
    return x + u*w, y + v*h


def draw_map(ax, bounds, name):
    x, y, w, h = bounds
    rect(ax, x, y, w, h, "#f2f3ef", name + "-map", radius=4, edge=RULE, lw=.85)
    for col in range(5):
        for row in range(4):
            rect(ax, x+(col*.2+.032)*w, y+(row*.25+.036)*h, .135*w, .171*h,
                 "#e1e5e4", f"{name}-block-{col}-{row}", radius=2.5)
    for index, (u, v, ph) in enumerate(((.425, .72, .235), (.382, .065, .30))):
        rect(ax, x+u*w, y+v*h, .146*w, ph*h, "#dfebd8",
             f"{name}-park-{index}", radius=4)
    for col in range(1, 5):
        rect(ax, x+col*.2*w-.008*w, y+2, .016*w, h-4, "white", f"{name}-street-v-{col}")
    for row, v in enumerate((.19, .46, .73)):
        rect(ax, x+2, y+v*h-.010*h, w-4, .020*h, "white", f"{name}-street-h-{row}")
    txt(ax, x+.82*w, y+.925*h, "East district", size=8, color=MUTED, name=name + "-east")
    txt(ax, x+.14*w, y+.058*h, "West district", size=8, color=MUTED, name=name + "-west")
    for key, (cat, _, _) in CATALOG.items():
        icon(ax, *location(bounds, key), cat, name=f"{name}-poi-{key}")


def draw_route(ax, bounds, name):
    route, color = ROUTES[name], ROUTE_COLORS[name]
    for index, (source, target) in enumerate(zip(route, route[1:])):
        a, b = location(bounds, source), location(bounds, target)
        dx, dy = b[0]-a[0], b[1]-a[1]
        length = (dx*dx+dy*dy)**.5
        start = (a[0]+12.8*dx/length, a[1]+12.8*dy/length)
        end = (b[0]-12.8*dx/length, b[1]-12.8*dy/length)
        patch = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=9,
                                lw=1.65, color=color, shrinkA=0, shrinkB=0, zorder=3)
        ax.add_patch(tag(patch, f"{name}-route-{index+1}-{source}-to-{target}"))
    for index, key in enumerate(route, 1):
        x, y = location(bounds, key)
        x, y = x+11.6, y+11.6
        ax.add_patch(tag(Circle((x, y), 6.8, facecolor=color, edgecolor="white",
                                linewidth=.45, zorder=6), f"{name}-visit-badge-{index}"))
        label = txt(ax, x, y, str(index), size=8, color="white", bold=True,
                    name=f"{name}-visit-number-{index}")
        label.set_zorder(7)


def make_figure():
    global _IDS
    _IDS = count()
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 8.5,
                         "mathtext.fontset": "stix", "pdf.fonttype": 42,
                         "svg.fonttype": "none", "svg.hashsalt": "pcdg-intro"})
    fig = plt.figure(figsize=(7, 3.25))
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, 700), ylim=(0, 325))
    ax.axis("off")
    ax.set_gid("pcdg-postswap-intro")
    txt(ax, 350, 315, "Activity-order constraint: Sightseeing before Shopping", size=10.5,
        bold=True, name="requested-relation")
    txt(ax, 350, 299, "Same map and POI catalog; different visit sequences and POI choices.",
        size=8.5, color=MUTED, name="shared-catalog-note")
    for name, title, first, second in (
        ("postswap", "(a) Post-generation reordering", "Order satisfied; distant POIs retained", "Long cross-district detour"),
        ("pcdg", "(b) PCDG", "Order satisfied; nearby POIs selected", "Spatially coherent visit sequence"),
    ):
        bounds = MAP_BOUNDS[name]
        cx = bounds[0] + bounds[2]/2
        txt(ax, cx, 282, title, size=10, bold=True, name=name + "-title")
        draw_map(ax, bounds, name)
        draw_route(ax, bounds, name)
        txt(ax, cx, 53, first, size=8.6, color=ROUTE_COLORS[name], bold=True,
            name=name + "-outcome")
        txt(ax, cx, 39, second, size=8.5, name=name + "-interpretation")
    for x, cat, label in ((115, "S", "Sightseeing"), (281, "W", "Shopping"),
                           (437, "D", "Dining"), (571, "H", "Hotel")):
        icon(ax, x, 15, cat, radius=8.5, name="legend-" + cat)
        txt(ax, x+14, 15, label, size=8.5, align="left", name="legend-label-" + cat)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for label in ax.texts:
        assert label.get_fontsize() >= 8
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
    print("Exported postswap-pcdg.pdf/.svg and preview; 7 x 3.25 in; live vector text")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--preview-dir", type=Path, default=BUILD)
    args = parser.parse_args()
    main(args.output_dir, args.preview_dir)
