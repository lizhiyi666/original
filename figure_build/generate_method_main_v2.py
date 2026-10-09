from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch


OUT = Path(__file__).resolve().parents[1] / "figures" / "method_main"
OUT.mkdir(parents=True, exist_ok=True)

W, H = 1600, 1050
COL = {
    "ink": "#17212B", "muted": "#5C6873", "line": "#8B98A5",
    "panel": "#F7F9FB", "blue": "#DCEAF5", "blue_edge": "#6F8FA8",
    "green": "#E4F0EA", "green_edge": "#7DA28C", "gray": "#EEF1F4",
    "gray_edge": "#A8B2BC", "accent": "#C45A3A", "accent_fill": "#FBE8E1",
    "white": "#FFFFFF",
}


def box(ax, x, y, w, h, text, fc, ec, lw=1.8, fs=16, weight="normal", color=None,
        radius=0.025, z=3):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.012,rounding_size={radius * min(w, h)}",
                       facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z)
    ax.add_patch(p)
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                fontweight=weight, color=color or COL["ink"], linespacing=1.15, zorder=z + 1)
    return p


def arrow(ax, x1, y1, x2, y2, color=None, lw=2.2, ls="-", mutation=16, z=4):
    p = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=mutation,
                        linewidth=lw, linestyle=ls, color=color or COL["ink"],
                        shrinkA=5, shrinkB=5, zorder=z)
    ax.add_patch(p)
    return p


def token(ax, x, y, w, h, label, fc, ec=COL["line"], fs=10, hatch=None):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=1.0,
                           hatch=hatch, zorder=5))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs,
            color=COL["ink"], zorder=6)


def draw_static(path_png: Path, path_svg: Path, path_pdf: Path):
    fig, ax = plt.subplots(figsize=(16, 10.5), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

    # Two panels; the internal horizontal split in (a) separates training from inference.
    ax.add_patch(FancyBboxPatch((35, 520), 1530, 495, boxstyle="round,pad=0.012,rounding_size=18",
                                facecolor=COL["panel"], edgecolor="#C7D0D8", linewidth=2.0, zorder=0))
    ax.add_patch(FancyBboxPatch((35, 35), 1530, 445, boxstyle="round,pad=0.012,rounding_size=18",
                                facecolor="#FBFCFD", edgecolor="#C7D0D8", linewidth=2.0, zorder=0))
    ax.text(62, 980, "(a) Overall Framework", fontsize=24, fontweight="bold", color=COL["ink"])
    ax.text(62, 445, "(b) Category-order Constraint Mechanism  ·  ① Detail", fontsize=22,
            fontweight="bold", color=COL["ink"])

    # Panel (a), training lane.
    ax.text(78, 920, "TRAINING", fontsize=13, fontweight="bold", color=COL["muted"])
    box(ax, 78, 790, 255, 100, "Observed POI\nevent sequences", COL["white"], COL["blue_edge"], fs=16, weight="bold")
    ax.text(205, 760, "time · category · POI · context", ha="center", fontsize=11, color=COL["muted"])
    box(ax, 405, 790, 255, 100, "Condition\nRepresentation", COL["blue"], COL["blue_edge"], fs=16, weight="bold")
    box(ax, 735, 775, 370, 130, "Conditional temporal +\ndiscrete diffusion training", "#F4F8FB", COL["blue_edge"], lw=2.0, fs=14, weight="bold")
    ax.text(920, 758, "Add-Thin + category/POI denoiser", ha="center", fontsize=10.5, color=COL["muted"])
    ax.text(920, 740, "forward noise / reverse denoising objective", ha="center", fontsize=9.5, color=COL["muted"])
    arrow(ax, 338, 840, 400, 840, lw=2.3)
    arrow(ax, 665, 840, 730, 840, lw=2.3)
    ax.text(1175, 840, "shared trained generator", ha="center", fontsize=11, color=COL["muted"])
    arrow(ax, 1110, 840, 1240, 840, color=COL["line"], lw=1.6, ls="--", mutation=14)

    # Divider and inference lane.
    ax.plot([70, 1530], [710, 710], color="#C7D0D8", linewidth=1.4, zorder=1)
    ax.text(78, 675, "INFERENCE / GENERATION", fontsize=13, fontweight="bold", color=COL["muted"])
    box(ax, 78, 555, 245, 100, "Generation-time\nconditions", COL["white"], COL["blue_edge"], fs=16, weight="bold")
    ax.text(200, 570, "time · context · length", ha="center", fontsize=10.5, color=COL["muted"])
    box(ax, 355, 555, 210, 100, "Initial mask /\nnoise state", COL["gray"], COL["gray_edge"], fs=16, weight="bold")
    arrow(ax, 328, 605, 350, 605, lw=2.3)

    # Reverse loop with an explicit insertion point.
    box(ax, 600, 535, 605, 145, "", "#F4F8FB", COL["blue_edge"], lw=2.2, radius=0.025)
    ax.text(902, 660, "Reverse Diffusion Loop", ha="center", fontsize=16, fontweight="bold", color=COL["ink"])
    ax.text(902, 640, "x_T → … → x_t → x_{t-1} → … → x_0", ha="center", fontsize=10.5, color=COL["muted"])
    # Internal step chain.
    box(ax, 620, 555, 75, 55, "x_t", COL["gray"], COL["gray_edge"], fs=13, weight="bold")
    box(ax, 715, 555, 125, 55, "Denoising", COL["green"], COL["green_edge"], fs=12, weight="bold")
    box(ax, 850, 555, 130, 55, "Category-position\nlogits", COL["white"], COL["blue_edge"], fs=10.5, weight="bold")
    box(ax, 1015, 550, 140, 65, "① Category-order\nprojection", COL["accent_fill"], COL["accent"], lw=2.6, fs=10.5, weight="bold", color=COL["accent"])
    box(ax, 1160, 555, 30, 55, "x\nₜ₋₁", COL["gray"], COL["gray_edge"], fs=10, weight="bold")
    arrow(ax, 700, 582, 710, 582, lw=1.8, mutation=12)
    arrow(ax, 840, 582, 845, 582, lw=1.8, mutation=12)
    arrow(ax, 985, 582, 1010, 582, color=COL["accent"], lw=2.0, mutation=12)
    arrow(ax, 1150, 582, 1155, 582, lw=1.8, mutation=12)
    ax.text(1080, 532, "before joint sampling", ha="center", fontsize=9.5, color=COL["accent"])
    arrow(ax, 568, 605, 595, 605, lw=2.3)

    # Constraint specification enters the reverse loop directly, not the condition encoder.
    box(ax, 1180, 680, 270, 58, "Constraint specification\npo_matrix", COL["accent_fill"], COL["accent"], lw=2.4, fs=11.5, weight="bold", color=COL["accent"])
    arrow(ax, 1180, 700, 1118, 620, color=COL["accent"], lw=2.0, ls="--", mutation=14)
    ax.text(1315, 748, "direct input to reverse step", ha="center", fontsize=9.5, color=COL["accent"])
    arrow(ax, 1208, 605, 1245, 605, lw=2.5)
    box(ax, 1250, 555, 255, 100, "Generated POI\nSequence", COL["white"], COL["blue_edge"], fs=16, weight="bold")
    ax.text(1378, 535, "time / category / POI / GPS", ha="center", fontsize=10.5, color=COL["muted"])
    for i, (lab, fc) in enumerate([("POI₁", COL["blue"]), ("POI₂", COL["green"]), ("…", COL["gray"]), ("POIₙ", COL["blue"])]):
        token(ax, 1275 + i * 53, 575, 46, 25, lab, fc, fs=9)

    # Panel (b), detailed constraint step.
    box(ax, 78, 225, 195, 130, "Sample-level\npo_matrix\n(A ≺ B)", COL["accent_fill"], COL["accent"], lw=2.3, fs=13, weight="bold", color=COL["accent"])
    ax.text(175, 238, "M[A,B]=1", ha="center", fontsize=10, color=COL["accent"])
    box(ax, 315, 240, 190, 100, "Constraint\nParsing", COL["accent_fill"], COL["accent"], lw=2.2, fs=14, weight="bold", color=COL["accent"])
    ax.text(410, 255, "A → B", ha="center", fontsize=11, color=COL["accent"])
    box(ax, 550, 220, 250, 140, "Order + Existence\nEnergy", COL["accent_fill"], COL["accent"], lw=2.2, fs=14, weight="bold", color=COL["accent"])
    ax.text(675, 254, "reverse-order penalty", ha="center", fontsize=10, color=COL["muted"])
    ax.text(675, 236, "existence penalty", ha="center", fontsize=10, color=COL["muted"])
    box(ax, 570, 125, 210, 70, "Category-position Mask", COL["blue"], COL["blue_edge"], lw=1.8, fs=11.2, weight="bold")
    ax.text(675, 135, "category_mask · sequence positions only", ha="center", fontsize=8.8, color=COL["muted"])
    # Position-mask schematic.
    for i, lab in enumerate([("p₁", "1"), ("p₂", "0"), ("p₃", "1"), ("…", "…"), ("pL", "1")]):
        token(ax, 585 + i * 37, 98, 31, 20, lab[1], COL["green"] if lab[1] == "1" else COL["gray"], fs=8)
        ax.text(600 + i * 37, 88, lab[0], ha="center", fontsize=8, color=COL["muted"])
    arrow(ax, 650, 198, 675, 215, color=COL["blue_edge"], lw=1.6, ls="--", mutation=12)
    ax.text(675, 68, "position selector, not a POI candidate set", ha="center", fontsize=9.5, color=COL["muted"])
    box(ax, 845, 205, 285, 170, "KL-preserving\nALM Projection", COL["accent_fill"], COL["accent"], lw=2.7, fs=15, weight="bold", color=COL["accent"])
    ax.text(987, 237, "Gumbel-softmax relaxation", ha="center", fontsize=10, color=COL["accent"])
    ax.text(987, 219, "modify category-position logits", ha="center", fontsize=10, color=COL["accent"], fontweight="bold")
    box(ax, 1180, 285, 170, 72, "Category-position\nlogits′", COL["accent_fill"], COL["accent"], lw=2.2, fs=12, weight="bold", color=COL["accent"])
    box(ax, 1180, 175, 170, 72, "POI logits\n(unchanged)", COL["gray"], COL["gray_edge"], lw=1.7, fs=12, weight="bold")
    box(ax, 1390, 215, 135, 100, "Joint\nSampling", COL["blue"], COL["blue_edge"], lw=2.0, fs=13, weight="bold")
    arrow(ax, 275, 290, 310, 290, color=COL["accent"], lw=2.1)
    arrow(ax, 510, 290, 545, 290, color=COL["accent"], lw=2.1)
    arrow(ax, 805, 290, 840, 290, color=COL["accent"], lw=2.1)
    arrow(ax, 1135, 290, 1175, 320, color=COL["accent"], lw=2.0)
    arrow(ax, 1355, 320, 1385, 280, color=COL["ink"], lw=2.0)
    arrow(ax, 1355, 210, 1385, 250, color=COL["line"], lw=2.0)
    ax.text(1185, 155, "POI logits bypass projection", fontsize=10, color=COL["muted"])
    ax.text(78, 48, "No candidate filtering · No explicit POI transition restriction · Constraint acts during reverse diffusion", fontsize=10.5, color=COL["muted"])

    # Legend.
    ax.plot([1110, 1140], [952, 952], color=COL["ink"], lw=2.4)
    ax.text(1148, 952, "data / model flow", va="center", fontsize=10, color=COL["muted"])
    ax.plot([1270, 1300], [952, 952], color=COL["accent"], lw=2.0, ls="--")
    ax.text(1308, 952, "constraint specification", va="center", fontsize=10, color=COL["muted"])

    plt.savefig(path_png, dpi=220, bbox_inches="tight", facecolor="white")
    plt.savefig(path_svg, bbox_inches="tight", facecolor="white")
    plt.savefig(path_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def drawio(path: Path):
    cells = []
    def node(cid, value, x, y, w, h, style):
        label = escape(value).replace('\\n', '&lt;br&gt;')
        cells.append(f'<mxCell id="{cid}" value="{label}" style="{style}" vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
    def edge(cid, s, t, style):
        cells.append(f'<mxCell id="{cid}" edge="1" parent="1" source="{s}" target="{t}" style="{style}"><mxGeometry relative="1" as="geometry"/></mxCell>')
    def textnode(cid, value, x, y, w, h, size=11, color='#5C6873', bold=False):
        style=f'whiteSpace=wrap;html=1;strokeColor=none;fillColor=none;fontSize={size};fontColor={color};'+('fontStyle=1;' if bold else '')
        node(cid, value, x, y, w, h, style)
    def tokennode(cid, value, x, y, w, h, fc, size=9):
        style=f'whiteSpace=wrap;html=1;strokeColor=#8B98A5;fillColor={fc};strokeWidth=1;fontSize={size};fontColor=#17212B;'
        node(cid, value, x, y, w, h, style)
    panel_a='rounded=1;whiteSpace=wrap;html=1;strokeColor=#C7D0D8;fillColor=#F7F9FB;strokeWidth=2;'
    panel_b='rounded=1;whiteSpace=wrap;html=1;strokeColor=#C7D0D8;fillColor=#FBFCFD;strokeWidth=2;'
    base='rounded=1;whiteSpace=wrap;html=1;strokeColor=#6F8FA8;fillColor=#FFFFFF;strokeWidth=2;fontSize=16;fontColor=#17212B;'
    blue='rounded=1;whiteSpace=wrap;html=1;strokeColor=#6F8FA8;fillColor=#DCEAF5;strokeWidth=2;fontSize=15;fontColor=#17212B;'
    green='rounded=1;whiteSpace=wrap;html=1;strokeColor=#7DA28C;fillColor=#E4F0EA;strokeWidth=2;fontSize=15;fontColor=#17212B;'
    gray='rounded=1;whiteSpace=wrap;html=1;strokeColor=#A8B2BC;fillColor=#EEF1F4;strokeWidth=1;fontSize=14;fontColor=#17212B;'
    accent='rounded=1;whiteSpace=wrap;html=1;strokeColor=#C45A3A;fillColor=#FBE8E1;strokeWidth=3;fontSize=15;fontColor=#C45A3A;fontStyle=1;'
    node('pa','(a) Overall Framework',35,20,1530,495,panel_a); node('pb','(b) Category-order Constraint Mechanism · ① Detail',35,570,1530,445,panel_b)
    node('train_seq','Observed POI\\nevent sequences',78,125,255,100,base); node('train_cond','Condition\\nRepresentation',405,125,255,100,blue); node('train_model','Conditional temporal +\\ndiscrete diffusion training',735,110,370,130,panel_a)
    node('gen_cond','Generation-time\\nconditions',78,360,245,100,base); node('noise','Initial mask /\\nnoise state',355,360,210,100,gray)
    node('loop','',600,335,605,145,panel_a)
    node('proj_a','① Category-order\\nprojection',1005,365,140,65,accent); node('po_a','Constraint specification\\npo_matrix',860,273,250,62,accent); node('out','Generated POI\\nSequence',1250,360,255,100,base)
    node('mat','Sample-level\\npo_matrix',78,660,195,130,accent); node('parse','Constraint\\nParsing',315,675,190,100,accent); node('energy','Order + Existence\\nEnergy',550,655,250,140,accent); node('cmask','Category-position Mask',570,860,210,70,blue); node('proj','KL-preserving\\nALM Projection',845,640,285,170,accent); node('cat','Category-position\\nlogits′',1180,660,170,72,accent); node('poi','POI logits\\n(unchanged)',1180,770,170,72,gray); node('joint','Joint\\nSampling',1390,705,135,100,blue)
    # Static-export details are represented as independent editable Draw.io cells.
    textnode('t_train','TRAINING',78,75,150,24,13,'#5C6873',True); textnode('t_train_fields','time · category · POI · context',80,230,255,24,11)
    textnode('t_train_sub1','Add-Thin + category/POI denoiser',735,245,370,24,10); textnode('t_train_sub2','forward noise / reverse denoising objective',735,263,370,22,9); textnode('t_shared','shared trained generator',1110,125,250,24,11)
    textnode('t_inf','INFERENCE / GENERATION',78,330,270,24,13,'#5C6873',True); textnode('t_inf_fields','time · context · length',80,445,245,24,10)
    textnode('loop_title','Reverse Diffusion Loop',700,345,405,28,16,'#17212B',True); textnode('loop_axis','x_T → … → x_t → x_{t-1} → … → x_0',700,370,405,22,10.5)
    node('a_xt','x_t',620,405,75,55,gray); node('a_denoise','Denoising',715,405,125,55,green); node('a_cat','Category-position\\nlogits',850,405,130,55,base); node('a_proj','① Category-order\\nprojection',1015,400,140,65,accent); node('a_prev','x\\nₜ₋₁',1160,405,30,55,gray)
    textnode('a_before','before joint sampling',1000,465,160,20,9.5,'#C45A3A'); textnode('po_direct','direct input to reverse step',1180,267,270,22,9.5,'#C45A3A'); textnode('out_fields','time / category / POI / GPS',1250,465,255,22,10.5)
    tokennode('out_tok1','POI₁',1275,425,46,25,'#DCEAF5'); tokennode('out_tok2','POI₂',1328,425,46,25,'#E4F0EA'); tokennode('out_tok3','…',1381,425,46,25,'#EEF1F4'); tokennode('out_tok4','POIₙ',1434,425,46,25,'#DCEAF5')
    textnode('b_mat_note','M[A,B]=1',110,790,130,22,10,'#C45A3A'); textnode('b_parse_note','A → B',350,760,120,22,11,'#C45A3A'); textnode('b_energy1','reverse-order penalty',550,735,250,22,10); textnode('b_energy2','existence penalty',550,753,250,22,10)
    textnode('b_mask_note','category_mask · sequence positions only',570,845,210,22,8.8); tokennode('mask1','1',585,875,31,20,'#E4F0EA',8); tokennode('mask0','0',622,875,31,20,'#EEF1F4',8); tokennode('mask2','1',659,875,31,20,'#E4F0EA',8); tokennode('maskdots','…',696,875,31,20,'#EEF1F4',8); tokennode('maskL','1',733,875,31,20,'#E4F0EA',8)
    textnode('maskp1','p₁',590,897,22,16,8); textnode('maskp2','p₂',627,897,22,16,8); textnode('maskp3','p₃',664,897,22,16,8); textnode('maskpd','…',701,897,22,16,8); textnode('maskpL','pL',738,897,22,16,8); textnode('mask_boundary','position selector, not a POI candidate set',510,928,330,22,9.5)
    textnode('proj_note1','Gumbel-softmax relaxation',845,773,285,22,10,'#C45A3A'); textnode('proj_note2','modify category-position logits',845,791,285,22,10,'#C45A3A',True); textnode('poi_bypass','POI logits bypass projection',1180,853,210,22,10); textnode('boundary','No candidate filtering · No explicit POI transition restriction · Constraint acts during reverse diffusion',78,962,1300,24,10.5)
    textnode('legend_flow','data / model flow',1148,45,120,22,10); textnode('legend_constraint','constraint specification',1308,45,180,22,10)
    es='edgeStyle=orthogonalEdgeStyle;rounded=0;endArrow=block;strokeWidth=2;strokeColor=#17212B;'; ecs='edgeStyle=orthogonalEdgeStyle;rounded=0;endArrow=block;strokeWidth=2;strokeColor=#C45A3A;'; ed='edgeStyle=orthogonalEdgeStyle;rounded=0;endArrow=block;dashed=1;strokeWidth=1.5;strokeColor=#C45A3A;'
    edge('e1','train_seq','train_cond',es); edge('e2','train_cond','train_model',es); edge('e3','gen_cond','loop',es); edge('e4','noise','loop',es); edge('e5','loop','out',es); edge('e6','po_a','proj_a',ed)
    edge('a1','a_xt','a_denoise',es); edge('a2','a_denoise','a_cat',es); edge('a3','a_cat','a_proj',ecs); edge('a4','a_proj','a_prev',es)
    edge('b1','mat','parse',ecs); edge('b2','parse','energy',ecs); edge('b3','energy','proj',ecs); edge('b4','proj','cat',ecs); edge('b5','cat','joint',es); edge('b6','poi','joint','edgeStyle=orthogonalEdgeStyle;rounded=0;endArrow=block;strokeWidth=2;strokeColor=#8B98A5;'); edge('b7','cmask','energy','edgeStyle=orthogonalEdgeStyle;rounded=0;endArrow=block;dashed=1;strokeWidth=1.5;strokeColor=#6F8FA8;'); edge('b8','proj_a','pb','edgeStyle=orthogonalEdgeStyle;rounded=0;endArrow=none;dashed=1;strokeWidth=1.2;strokeColor=#C45A3A;')
    xml='<mxfile host="app.diagrams.net" modified="2026-09-06T00:00:00.000Z" agent="Codex" version="24.7.17"><diagram id="method-main-v2" name="method_main_v2"><mxGraphModel dx="1600" dy="1050" grid="1" gridSize="10" page="1" pageScale="1" pageWidth="1600" pageHeight="1050" math="0" shadow="0"><root><mxCell id="0"/><mxCell id="1" parent="0"/>'+''.join(cells)+'</root></mxGraphModel></diagram></mxfile>'
    path.write_text(xml, encoding='utf-8')


def main():
    draw_static(OUT / 'method_main_v2.png', OUT / 'method_main_v2.svg', OUT / 'method_main_v2.pdf')
    drawio(OUT / 'method_main_v2.drawio')


if __name__ == '__main__':
    main()
