"""Plot archived evaluation-v2 table means; never load or rerun experiments.

Source: paper-www2027/tables/{method-comparison,stagewise}.tex. Shared
method/stage points are drawn once. Label offsets never move measured points.
CCFA small-multiple grammar adapted to two-metric scatter with linear shared
axes and shape/color encoding. SDs remain in the comparison table.
"""
from pathlib import Path
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper-www2027/figures"
BUILD = ROOT / "ccfa-workfiles/writing/www2027/build"
CITIES = ("Istanbul", "New York")


def read_rows(path):
    blocks, rows, city = [], {c: [] for c in CITIES}, None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(r"\begin{tabular}"):
            rows = {c: [] for c in CITIES}
        for name in CITIES:
            if r"\textbf{" + name + "}" in line:
                city = name
        if city and " & " in line and line.rstrip().endswith(r"\\"):
            cells = line.split(" & ")
            values = [re.search(r"\d+\.\d+", c) for c in cells[1:]]
            if any(values):
                rows[city].append((cells[0], [float(v.group()) for v in values if v]))
        if line.startswith(r"\end{tabular}"):
            blocks.append(rows)
    return blocks


def main():
    jsd, constraints = read_rows(ROOT / "paper-www2027/tables/method-comparison.tex")
    stages, = read_rows(ROOT / "paper-www2027/tables/stagewise.tex")
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 9,
                         "pdf.fonttype": 42, "svg.fonttype": "none", "svg.hashsalt": "pcdg-www2027",
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.65), sharex=True, sharey=True)
    fig.subplots_adjust(left=.078, right=.985, top=.89, bottom=.33, wspace=.13)
    names = ["(a) JointGen", "(b) PCDG-P", "(c) PCDG-PD", "(d) + geometry", "(e) PCDG"]
    colors = ["#555555", "#0072B2", "#009E73", "#CC79A7", "#D55E00"]
    shapes = ["o", "s", "^", "D", "*"]
    for ax, city in zip(axes, CITIES):
        assert len(jsd[city]) == len(constraints[city]) == 7
        assert len(stages[city]) == 5
        coords = [(v[-1], v[0]) for _, v in stages[city]]
        for i, mi in enumerate([0, 4, 5, None, 6]):
            if mi is not None:
                assert coords[i] == (constraints[city][mi][1][-1], jsd[city][mi][1][0])
        ax.plot(*zip(*coords), "--", color="#8a8a8a", lw=.9, zorder=1)
        offsets = [(0, 8), (0, 9), (-5, 10), (-11, 2), (12, 9)]
        for i, ((x, y), color, marker, off) in enumerate(zip(coords, colors, shapes, offsets)):
            ax.scatter(x, y, s=34 if i != 4 else 75, c=color, marker=marker,
                       edgecolors="white", linewidths=.45, zorder=3)
            ax.annotate(f"({chr(97+i)})", (x, y), xytext=off, textcoords="offset points",
                        ha="center", fontsize=9, color="#222222")
        for mi, name, marker, offset in [(1, "PostSwap", "v", (4, -15)),
                                          (2, "EnergyGuide", "X", (-5, 14)),
                                          (3, "PO-CFG", "P", (-5, -13))]:
            x, y = constraints[city][mi][1][-1], jsd[city][mi][1][0]
            ax.scatter(x, y, s=32, marker=marker, color="#777777", zorder=2)
            ax.annotate(name, (x, y), xytext=offset, textcoords="offset points",
                        ha="left" if mi == 1 else "right", fontsize=8.5)
        ax.set_title(city, fontsize=10, fontweight="bold", pad=7)
        ax.set_xlim(-4, 104)
        ax.set_ylim(0, .345)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_yticks([0, .1, .2, .3])
        ax.set_xlabel("Unsatisfied required relations (%)", labelpad=3)
        ax.grid(axis="y", color="#e7e7e7", lw=.6)
        ax.tick_params(length=3, pad=2)
    axes[0].set_ylabel("Distance JSD", labelpad=4)
    handles = [Line2D([], [], color=c, marker=m, linestyle="None", markersize=6,
                      label=n) for c, m, n in zip(colors, shapes, names)]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               columnspacing=1.1, handletextpad=.35, bbox_to_anchor=(.52, .035), fontsize=8.5)
    OUT.mkdir(parents=True, exist_ok=True)
    BUILD.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "tradeoff.pdf", metadata={"Creator": "Matplotlib", "CreationDate": None})
    fig.savefig(OUT / "tradeoff.svg", metadata={"Date": None})
    svg = OUT / "tradeoff.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    fig.savefig(BUILD / "tradeoff.png", dpi=220)
    plt.close(fig)
    print("Validated stage coordinates against table means; exported tradeoff.pdf/.svg")


if __name__ == "__main__":
    main()
