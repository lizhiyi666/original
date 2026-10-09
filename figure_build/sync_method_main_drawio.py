from pathlib import Path
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures" / "method_main"
SRC = FIG / "method_main_final.drawio"


def geom(cell):
    return cell.find("mxGeometry")


def set_parent_relative(cell, parent_id, parent_origin):
    g = geom(cell)
    if g is not None and "x" in g.attrib and "y" in g.attrib:
        g.set("x", str(float(g.attrib["x"]) - parent_origin[0]).rstrip("0").rstrip("."))
        g.set("y", str(float(g.attrib["y"]) - parent_origin[1]).rstrip("0").rstrip("."))
    cell.set("parent", parent_id)


def add_vertex(root, cid, value, x, y, w, h, style, parent="1"):
    c = ET.Element("mxCell", {"id": cid, "value": value, "style": style, "vertex": "1", "parent": parent})
    ET.SubElement(c, "mxGeometry", {"x": str(x), "y": str(y), "width": str(w), "height": str(h), "as": "geometry"})
    root.append(c)
    return c


def main():
    tree = ET.parse(SRC)
    root = tree.find("./diagram/mxGraphModel/root")
    cells = {c.get("id"): c for c in root.findall("mxCell")}

    # One projection node only: keep the in-loop node that is visible in the static export.
    duplicate = cells.pop("proj_a", None)
    if duplicate is not None:
        root.remove(duplicate)
    for e in root.findall("mxCell"):
        if e.get("id") == "e6":
            e.set("target", "a_proj")
        if e.get("id") == "b8":
            e.set("source", "a_proj")

    # Mark semantic containers and establish parent-child hierarchy while preserving absolute positions.
    for cid in ("pa", "pb", "loop", "out", "cmask"):
        if cid in cells:
            style = cells[cid].get("style", "")
            if "container=1;" not in style:
                cells[cid].set("style", style + "container=1;collapsible=0;")

    pa_origin = (35.0, 20.0)
    pb_origin = (35.0, 570.0)
    loop_origin = (600.0, 335.0)
    out_origin = (1250.0, 360.0)
    cmask_origin = (570.0, 860.0)

    panel_a = {
        "train_seq", "train_cond", "train_model", "gen_cond", "noise", "loop", "po_a", "out",
        "t_train", "t_train_fields", "t_train_sub1", "t_train_sub2", "t_shared", "t_inf", "t_inf_fields",
        "loop_title", "loop_axis", "po_direct", "out_fields", "a_before", "legend_flow", "legend_constraint",
    }
    panel_b = {
        "mat", "parse", "energy", "cmask", "proj", "cat", "poi", "joint", "b_mat_note", "b_parse_note",
        "b_energy1", "b_energy2", "b_mask_note", "mask_boundary", "proj_note1", "proj_note2", "poi_bypass", "boundary",
    }
    for cid in panel_a:
        if cid in cells:
            set_parent_relative(cells[cid], "pa", pa_origin)
    for cid in panel_b:
        if cid in cells:
            set_parent_relative(cells[cid], "pb", pb_origin)

    for cid in ("a_xt", "a_denoise", "a_cat", "a_proj", "a_prev"):
        if cid in cells:
            set_parent_relative(cells[cid], "loop", loop_origin)
    for cid in ("out_tok1", "out_tok2", "out_tok3", "out_tok4"):
        if cid in cells:
            set_parent_relative(cells[cid], "out", out_origin)
    for cid in ("mask1", "mask0", "mask2", "maskdots", "maskL", "maskp1", "maskp2", "maskp3", "maskpd", "maskpL"):
        if cid in cells:
            set_parent_relative(cells[cid], "cmask", cmask_origin)

    # Keep math notation editable and rendered by draw.io HTML labels.
    cells["loop_axis"].set("value", "x<sub>T</sub> → … → x<sub>t</sub> → x<sub>t−1</sub> → … → x<sub>0</sub>")
    cells["a_xt"].set("value", "x<sub>t</sub>")
    cells["a_prev"].set("value", "x<sub>t−1</sub>")

    # Add missing non-semantic layout primitives as editable line cells.
    line_style = "shape=line;html=1;strokeColor=#C7D0D8;strokeWidth=1;"
    if "training_inference_divider" not in cells:
        add_vertex(root, "training_inference_divider", "", 35, 690, 1460, 1, line_style, "pa")
    legend_line = "shape=line;html=1;strokeWidth=2;"
    if "legend_flow_line" not in cells:
        add_vertex(root, "legend_flow_line", "", 1075, 67, 30, 1, legend_line + "strokeColor=#17212B;", "pa")
    if "legend_constraint_line" not in cells:
        add_vertex(root, "legend_constraint_line", "", 1235, 67, 30, 1, legend_line + "strokeColor=#C45A3A;dashed=1;", "pa")

    # Use a stable XML declaration and explicit UTF-8 output.
    tree.write(SRC, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    main()
