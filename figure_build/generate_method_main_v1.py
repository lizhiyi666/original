from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch


OUT = Path(__file__).resolve().parents[1] / "figures" / "method_main"
OUT.mkdir(parents=True, exist_ok=True)

W, H = 1600, 1000

COL = {
    "ink": "#17212B",
    "muted": "#5C6873",
    "line": "#8B98A5",
    "panel": "#F7F9FB",
    "blue": "#DCEAF5",
    "blue_edge": "#6F8FA8",
    "green": "#E4F0EA",
    "green_edge": "#7DA28C",
    "gray": "#EEF1F4",
    "gray_edge": "#A8B2BC",
    "accent": "#C45A3A",
    "accent_fill": "#FBE8E1",
    "white": "#FFFFFF",
}


def box(ax, x, y, w, h, text, fc, ec, lw=1.8, fs=18, weight="normal", radius=0.02,
        color=None, z=3, ha="center", va="center"):
    patch = FancyBboxPatch((x, y), w, h,
                           boxstyle=f"round,pad=0.012,rounding_size={radius * min(w, h)}",
                           linewidth=lw, edgecolor=ec, facecolor=fc, zorder=z)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha=ha, va=va, fontsize=fs,
            fontweight=weight, color=color or COL["ink"], zorder=z + 1,
            linespacing=1.18)
    return patch


def arrow(ax, x1, y1, x2, y2, color=None, lw=2.3, style="-|>", ls="-", z=4, mutation=18):
    arr = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                          mutation_scale=mutation, linewidth=lw,
                          linestyle=ls, color=color or COL["ink"],
                          shrinkA=5, shrinkB=5, zorder=z)
    ax.add_patch(arr)
    return arr


def token(ax, x, y, w, h, label, fc, ec=COL["line"], fs=12, hatch=None):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec,
                           linewidth=1.0, hatch=hatch, zorder=5))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fs, color=COL["ink"], zorder=6)


def draw_figure(path_png: Path, path_svg: Path, path_pdf: Path):
    fig, ax = plt.subplots(figsize=(16, 10), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    # Panels
    ax.add_patch(FancyBboxPatch((35, 480), 1530, 475, boxstyle="round,pad=0.012,rounding_size=18",
                                facecolor=COL["panel"], edgecolor="#C7D0D8", linewidth=2.0, zorder=0))
    ax.add_patch(FancyBboxPatch((35, 35), 1530, 405, boxstyle="round,pad=0.012,rounding_size=18",
                                facecolor="#FBFCFD", edgecolor="#C7D0D8", linewidth=2.0, zorder=0))
    ax.text(62, 918, "(a) Overall Framework", fontsize=24, fontweight="bold", color=COL["ink"])
    ax.text(62, 403, "(b) Category-order Constraint Mechanism", fontsize=24, fontweight="bold", color=COL["ink"])

    # Panel A: input
    box(ax, 75, 610, 260, 245, "", COL["white"], COL["blue_edge"], lw=1.8, fs=21, weight="bold")
    ax.text(205, 832, "Input", ha="center", fontsize=20, color=COL["ink"], fontweight="bold")
    ax.text(205, 802, "POI event sequence", ha="center", fontsize=14, color=COL["ink"], fontweight="bold")
    ax.text(205, 779, "time  ·  category  ·  POI", ha="center", fontsize=11, color=COL["muted"])
    # sequence strip
    sx, sy, tw, th = 103, 691, 40, 34
    labels = [("t", COL["gray"]), ("C₁", COL["green"]), ("P₁", COL["blue"]), ("C₂", COL["green"]), ("P₂", COL["blue"]), ("…", COL["gray"])]
    for i, (lab, fc) in enumerate(labels):
        token(ax, sx + i * (tw + 3), sy, tw, th, lab, fc, fs=11)
    ax.text(205, 660, "context conditions", ha="center", fontsize=12, color=COL["muted"])
    for i, lab in enumerate(["time", "ctx₁", "ctx₂", "…"]):
        token(ax, 111 + i * 43, 625, 37, 25, lab, COL["gray"], fs=9)
    arrow(ax, 345, 733, 390, 733, color=COL["ink"], lw=2.6)

    # Condition representation
    box(ax, 395, 770, 275, 85, "Condition\nRepresentation", COL["blue"], COL["blue_edge"], fs=15, weight="bold")
    arrow(ax, 532, 768, 532, 730, color=COL["line"], lw=1.8, ls="--")

    # Generator outer
    box(ax, 395, 545, 560, 185, "", "#F4F8FB", COL["blue_edge"], lw=2.0, fs=19, weight="bold", radius=0.025)
    ax.text(675, 708, "Conditional Joint Generator", ha="center", fontsize=18, fontweight="bold", color=COL["ink"])
    box(ax, 425, 580, 220, 90, "Temporal\nAdd-Thin", COL["gray"], COL["gray_edge"], fs=14, weight="bold")
    box(ax, 700, 580, 220, 90, "Category / POI\nDiscrete Diffusion", COL["green"], COL["green_edge"], fs=14, weight="bold")
    ax.text(535, 560, "continuous event times", ha="center", fontsize=10, color=COL["muted"])
    ax.text(810, 560, "joint discrete tokens", ha="center", fontsize=10, color=COL["muted"])
    arrow(ax, 670, 627, 700, 627, color=COL["line"], lw=1.7, ls="--")

    # Forward/reverse strip
    ax.text(600, 526, "Forward diffusion", ha="center", fontsize=11, color=COL["muted"])
    token(ax, 500, 493, 48, 25, "x₀", COL["white"], fs=11)
    token(ax, 570, 493, 48, 25, "xₜ", COL["gray"], fs=11)
    token(ax, 640, 493, 48, 25, "x_T", COL["gray"], fs=11)
    arrow(ax, 550, 505, 565, 505, color=COL["line"], lw=1.5, ls="--", mutation=12)
    arrow(ax, 620, 505, 635, 505, color=COL["line"], lw=1.5, ls="--", mutation=12)
    ax.text(850, 526, "Reverse denoising", ha="center", fontsize=11, color=COL["muted"])
    arrow(ax, 700, 505, 895, 505, color=COL["ink"], lw=2.2, mutation=16)
    ax.text(800, 493, "x_T → … → x₀", ha="center", fontsize=12, color=COL["ink"])

    # Constraint insertion and output
    arrow(ax, 955, 640, 1000, 640, color=COL["ink"], lw=2.6)
    box(ax, 1005, 555, 280, 170, "Category-order\nConstraint-aware\nGeneration", COL["accent_fill"], COL["accent"], lw=2.7, fs=15, weight="bold", color=COL["accent"], radius=0.03)
    ax.text(1145, 567, "inference / reverse diffusion", ha="center", fontsize=9.5, color=COL["accent"])
    ax.text(1145, 585, "category positions only", ha="center", fontsize=9.5, color=COL["accent"])
    arrow(ax, 1288, 640, 1330, 640, color=COL["ink"], lw=2.6)
    box(ax, 1335, 610, 205, 110, "Generated POI\nSequence", COL["white"], COL["blue_edge"], fs=14, weight="bold")
    ax.text(1438, 586, "POI₁ → POI₂ → … → POIₙ", ha="center", fontsize=10, color=COL["ink"])
    # condition side connection to innovation
    arrow(ax, 670, 812, 1085, 728, color=COL["accent"], lw=1.8, ls="--", mutation=15)
    ax.text(820, 786, "sample-level po_matrix", fontsize=11, color=COL["accent"], rotation=-12)

    # Panel B detail
    box(ax, 75, 180, 230, 175, "Sample-level\npo_matrix", COL["accent_fill"], COL["accent"], lw=2.2, fs=14, weight="bold", color=COL["accent"])
    ax.text(190, 207, "M[A,B] = 1", ha="center", fontsize=11, color=COL["accent"])
    ax.text(190, 225, "A ≺ B", ha="center", fontsize=12, color=COL["accent"], fontweight="bold")
    arrow(ax, 307, 267, 355, 267, color=COL["accent"], lw=2.2)
    box(ax, 360, 205, 215, 125, "Constraint\nParsing", COL["accent_fill"], COL["accent"], lw=2.2, fs=14, weight="bold", color=COL["accent"])
    ax.text(467, 222, "A → B", ha="center", fontsize=11, color=COL["accent"])
    arrow(ax, 578, 267, 625, 267, color=COL["accent"], lw=2.2)
    box(ax, 630, 177, 275, 180, "Order + Existence\nEnergy", COL["accent_fill"], COL["accent"], lw=2.2, fs=14, weight="bold", color=COL["accent"])
    ax.text(767, 224, "reverse-order penalty", ha="center", fontsize=10, color=COL["muted"])
    ax.text(767, 202, "existence penalty", ha="center", fontsize=10, color=COL["muted"])
    arrow(ax, 908, 267, 955, 267, color=COL["accent"], lw=2.2)
    box(ax, 960, 165, 285, 205, "KL-preserving\nALM Projection", COL["accent_fill"], COL["accent"], lw=2.4, fs=15, weight="bold", color=COL["accent"])
    ax.text(1102, 198, "Gumbel-softmax relaxation", ha="center", fontsize=10, color=COL["accent"])
    ax.text(1102, 218, "project category logits", ha="center", fontsize=10, color=COL["accent"], fontweight="bold")
    arrow(ax, 1248, 267, 1295, 267, color=COL["accent"], lw=2.2)
    # split logits
    box(ax, 1300, 285, 205, 70, "Category logits′", COL["accent_fill"], COL["accent"], lw=2.0, fs=13, weight="bold", color=COL["accent"])
    box(ax, 1300, 175, 205, 70, "POI logits\n(unchanged)", COL["gray"], COL["gray_edge"], lw=1.6, fs=12, weight="bold")
    arrow(ax, 1402, 285, 1402, 250, color=COL["accent"], lw=1.8)
    arrow(ax, 1402, 175, 1402, 145, color=COL["line"], lw=1.8)
    box(ax, 1285, 62, 235, 70, "Joint Gumbel-max\nSampling", COL["blue"], COL["blue_edge"], lw=2.0, fs=13, weight="bold")
    arrow(ax, 1402, 145, 1402, 136, color=COL["ink"], lw=2.0)
    ax.text(1400, 49, "next reverse state xₜ₋₁", ha="center", fontsize=10, color=COL["ink"])
    ax.text(88, 72, "No candidate filtering / no explicit POI transition restriction", fontsize=10, color=COL["muted"])

    # Legend
    ax.plot([86, 120], [458, 458], color=COL["ink"], lw=2.6)
    ax.text(128, 458, "data flow", va="center", fontsize=11, color=COL["muted"])
    ax.plot([260, 294], [458, 458], color=COL["accent"], lw=2.0, ls="--")
    ax.text(302, 458, "constraint / control", va="center", fontsize=11, color=COL["muted"])

    plt.savefig(path_png, dpi=220, bbox_inches="tight", facecolor="white")
    plt.savefig(path_svg, bbox_inches="tight", facecolor="white")
    plt.savefig(path_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def drawio_xml(path: Path):
    cells = []
    def add_node(cid, value, x, y, w, h, style):
        label = escape(value).replace('\\n', '&lt;br&gt;')
        cells.append(f'<mxCell id="{cid}" value="{label}" style="{style}" vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
    def add_edge(cid, source, target, style):
        cells.append(f'<mxCell id="{cid}" edge="1" parent="1" source="{source}" target="{target}" style="{style}"><mxGeometry relative="1" as="geometry"/></mxCell>')
    panel = 'rounded=1;whiteSpace=wrap;html=1;strokeColor=#C7D0D8;fillColor=#F7F9FB;strokeWidth=2;'
    panel2 = 'rounded=1;whiteSpace=wrap;html=1;strokeColor=#C7D0D8;fillColor=#FBFCFD;strokeWidth=2;'
    base = 'rounded=1;whiteSpace=wrap;html=1;strokeColor=#6F8FA8;fillColor=#FFFFFF;strokeWidth=2;fontSize=18;fontColor=#17212B;'
    blue = 'rounded=1;whiteSpace=wrap;html=1;strokeColor=#6F8FA8;fillColor=#DCEAF5;strokeWidth=2;fontSize=17;fontColor=#17212B;'
    green = 'rounded=1;whiteSpace=wrap;html=1;strokeColor=#7DA28C;fillColor=#E4F0EA;strokeWidth=2;fontSize=17;fontColor=#17212B;'
    gray = 'rounded=1;whiteSpace=wrap;html=1;strokeColor=#A8B2BC;fillColor=#EEF1F4;strokeWidth=1;fontSize=15;fontColor=#17212B;'
    accent = 'rounded=1;whiteSpace=wrap;html=1;strokeColor=#C45A3A;fillColor=#FBE8E1;strokeWidth=3;fontSize=18;fontColor=#C45A3A;fontStyle=1;'
    add_node('panel_a','(a) Overall Framework',35,20,1530,475,panel)
    add_node('panel_b','(b) Category-order Constraint Mechanism',35,560,1530,405,panel2)
    add_node('input','Input\\nPOI event sequence',75,100,260,245,base)
    add_node('cond','Condition\\nRepresentation',395,100,275,85,blue)
    add_node('gen','Conditional Joint Generator',395,225,560,185,panel)
    add_node('time','Temporal\\nAdd-Thin',425,275,220,95,gray)
    add_node('disc','Category / POI\\nDiscrete Diffusion',700,275,220,95,green)
    add_node('constraint','Category-order\\nConstraint-aware\\nGeneration',1005,230,280,170,accent)
    add_node('output','Generated POI\\nSequence',1335,235,205,110,base)
    add_edge('e1','input','gen','edgeStyle=orthogonalEdgeStyle;rounded=0;endArrow=block;strokeWidth=2;strokeColor=#17212B;')
    add_edge('e2','cond','gen','edgeStyle=orthogonalEdgeStyle;rounded=0;endArrow=block;dashed=1;strokeWidth=1;strokeColor=#8B98A5;')
    add_edge('e3','gen','constraint','edgeStyle=orthogonalEdgeStyle;rounded=0;endArrow=block;strokeWidth=2;strokeColor=#17212B;')
    add_edge('e4','constraint','output','edgeStyle=orthogonalEdgeStyle;rounded=0;endArrow=block;strokeWidth=2;strokeColor=#17212B;')
    add_node('matrix','Sample-level\\npo_matrix\\nM[A,B]=1; A≺B',75,675,230,175,accent)
    add_node('parse','Constraint\\nParsing',360,700,215,125,accent)
    add_node('energy','Order + Existence\\nEnergy',630,668,275,180,accent)
    add_node('proj','KL-preserving\\nALM Projection',960,660,285,205,accent)
    add_node('catlog','Category logits′',1300,675,205,70,accent)
    add_node('poilog','POI logits\\n(unchanged)',1300,785,205,70,gray)
    add_node('sample','Joint Gumbel-max\\nSampling',1285,898,235,70,blue)
    add_edge('b1','matrix','parse','edgeStyle=orthogonalEdgeStyle;rounded=0;endArrow=block;strokeWidth=2;strokeColor=#C45A3A;')
    add_edge('b2','parse','energy','edgeStyle=orthogonalEdgeStyle;rounded=0;endArrow=block;strokeWidth=2;strokeColor=#C45A3A;')
    add_edge('b3','energy','proj','edgeStyle=orthogonalEdgeStyle;rounded=0;endArrow=block;strokeWidth=2;strokeColor=#C45A3A;')
    add_edge('b4','proj','catlog','edgeStyle=orthogonalEdgeStyle;rounded=0;endArrow=block;strokeWidth=2;strokeColor=#C45A3A;')
    add_edge('b5','catlog','sample','edgeStyle=orthogonalEdgeStyle;rounded=0;endArrow=block;strokeWidth=2;strokeColor=#17212B;')
    add_edge('b6','poilog','sample','edgeStyle=orthogonalEdgeStyle;rounded=0;endArrow=block;strokeWidth=2;strokeColor=#8B98A5;')
    xml = '<mxfile host="app.diagrams.net" modified="2026-09-06T00:00:00.000Z" agent="Codex" version="24.7.17"><diagram id="method-main-v1" name="method_main_v1"><mxGraphModel dx="1600" dy="1000" grid="1" gridSize="10" page="1" pageScale="1" pageWidth="1600" pageHeight="1000" math="0" shadow="0"><root><mxCell id="0"/><mxCell id="1" parent="0"/>' + ''.join(cells) + '</root></mxGraphModel></diagram></mxfile>'
    path.write_text(xml, encoding='utf-8')


def main():
    draw_figure(OUT / 'method_main_v1.png', OUT / 'method_main_v1.svg', OUT / 'method_main_v1.pdf')
    drawio_xml(OUT / 'method_main_v1.drawio')


if __name__ == '__main__':
    main()
