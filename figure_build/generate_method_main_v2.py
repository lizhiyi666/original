"""Code-first WWW framework and reverse-step vector figure.
Extends the old two-panel diagram to the frozen full stack. Source topology:
diffusion_transformer.py (incl. unsupported-category fallback),
constraint_projection.py, geometry_projection.py, distribution_balance.py.
Importing creates no files. Coordinates are points at final 7-inch width.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper-www2027/figures"
BUILD = ROOT / "ccfa-workfiles/writing/www2027/build"
INK, BLUE, GREEN, ORANGE = "#222222", "#0072B2", "#009E73", "#D55E00"


def box(ax, xy, width, height, label, color=BLUE, fill="#eef5fa", fontsize=9):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x, y), width, height,
        boxstyle="round,pad=0,rounding_size=2", facecolor=fill,
        edgecolor=color, linewidth=.7))
    ax.text(x+width/2, y+height/2, label, ha="center", va="center",
            fontsize=fontsize, linespacing=1.08, color=INK)


def arrow(ax, points, control=False, color=INK):
    for a, b in zip(points[:-2], points[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=.75,
                ls="--" if control else "-")
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
        mutation_scale=7, lw=.75, linestyle="--" if control else "-",
        color=color, shrinkA=1, shrinkB=1))


def main():
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 9,
                         "mathtext.fontset": "stix", "pdf.fonttype": 42,
                         "svg.fonttype": "none"})
    fig = plt.figure(figsize=(7, 3.25))
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, 504), ylim=(0, 234))
    ax.axis("off")
    ax.text(3, 223, "(a) Sampling-time control on a frozen generator", fontsize=10, weight="bold")
    ax.add_patch(Rectangle((118, 142), 220, 55, facecolor="#f7f7f7", edgecolor="#aaaaaa", lw=.7))
    ax.text(228, 184, "Reverse diffusion", ha="center", fontsize=9, weight="bold")
    box(ax, (3, 145), 84, 28, "Time / context\nMask state", fill="white")
    box(ax, (127, 149), 91, 26, "Frozen denoiser")
    box(ax, (237, 149), 92, 26, "Projection +\ncategory-first decode", color=ORANGE, fill="#fff0e7", fontsize=8.7)
    arrow(ax, [(88, 162), (126, 162)])
    arrow(ax, [(219, 162), (236, 162)])
    arrow(ax, [(329, 151), (329, 138), (172, 138), (172, 148)])
    ax.text(247, 128, r"$x_s\ \rightarrow\ x_{s-1}$", ha="center", fontsize=8)
    box(ax, (362, 149), 89, 26, "Same-category\ngeometry refinement", color=GREEN, fill="#ecf7f2", fontsize=8.7)
    arrow(ax, [(339, 162), (361, 162)])
    ax.text(349, 177, r"$s=0$", ha="center", fontsize=8)
    arrow(ax, [(452, 162), (465, 162)])
    # Schematic output, not measured coordinates.
    for x, y, c in [(470, 151, BLUE), (480, 174, GREEN), (497, 157, BLUE)]:
        ax.plot(x, y, "o", color=c, ms=3.5)
    ax.plot([470, 480, 497], [151, 174, 157], color="#888888", lw=.7, zorder=0)
    ax.text(483, 139, "POIs", ha="center", fontsize=8.5)
    ax.text(408, 128, "Actual categories, times, order fixed", ha="center", fontsize=8.2)
    box(ax, (155, 202), 116, 15, r"Partial order $\mathbf{M}$; $\rho,\lambda$", color=ORANGE, fill="white", fontsize=8.5)
    arrow(ax, [(271, 209), (283, 209), (283, 176)], True, ORANGE)
    box(ax, (349, 202), 145, 15, "Training reference", color=GREEN, fill="white", fontsize=8.5)
    arrow(ax, [(406, 201), (406, 176)], True, GREEN)
    ax.text(415, 187, r"$\beta$", fontsize=9)
    arrow(ax, [(349, 209), (337, 209), (337, 179), (314, 179), (314, 176)], True, GREEN)
    ax.plot([3, 501], [118, 118], color="#cccccc", lw=.7)
    ax.text(3, 104, "(b) One controlled reverse step", fontsize=10, weight="bold")
    box(ax, (3, 61), 81, 26, "Category logits", fill="white")
    box(ax, (3, 21), 81, 26, "POI logits", fill="white")
    box(ax, (113, 21), 149, 66, "", color=ORANGE, fill="#fff0e7")
    ax.text(187.5, 77, "KL-anchored projection", ha="center", weight="bold", fontsize=9)
    ax.text(187.5, 61, r"$\rho\,\mathcal{L}_{\rm order+exist}+\lambda\,\mathcal{L}_{\rm cat}$", ha="center", fontsize=9)
    ax.text(187.5, 43, r"$w_{\rm dist}D_{\rm KL}(H_{\rm train}\Vert H_{\rm proj})$", ha="center", fontsize=9)
    ax.text(187.5, 27, "Soft gradients; hard dual updates", ha="center", fontsize=8.2)
    arrow(ax, [(85, 74), (112, 74)])
    arrow(ax, [(85, 34), (112, 34)])
    box(ax, (291, 63), 91, 24, "Sample category", color=ORANGE, fill="#fff0e7")
    box(ax, (291, 21), 91, 26, "Restrict POI support", color=ORANGE, fill="#fff0e7", fontsize=8.7)
    arrow(ax, [(263, 74), (290, 74)])
    arrow(ax, [(263, 34), (290, 34)])
    arrow(ax, [(336, 62), (336, 48)], True, ORANGE)
    box(ax, (411, 21), 90, 26, "Resample POI\nSame Gumbel draw", color=ORANGE, fill="#fff0e7", fontsize=8.5)
    arrow(ax, [(383, 34), (410, 34)])
    arrow(ax, [(383, 75), (456, 75), (456, 48)])
    ax.text(451, 84, "Category tokens retained", ha="center", fontsize=8.2)
    ax.text(3, 5, "Solid: samples / logits     Dashed: conditions / objectives", fontsize=8.2)
    ax.text(501, 5, "Unsupported final category: any catalog POI", ha="right", fontsize=8.2)
    OUT.mkdir(parents=True, exist_ok=True)
    BUILD.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "framework.pdf", metadata={"Creator": "Matplotlib", "CreationDate": None})
    fig.savefig(OUT / "framework.svg")
    fig.savefig(BUILD / "framework.png", dpi=220)
    plt.close(fig)
    print("Exported framework.pdf/.svg; live vector text, final width 7 inches")


if __name__ == "__main__":
    main()
