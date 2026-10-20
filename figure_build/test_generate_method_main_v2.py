"""Local figure contract checks; no experiments or scientific-code imports."""
import io
import unittest
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt

import generate_method_main_v2 as figure


class FigureContractTests(unittest.TestCase):
    def setUp(self):
        self.fig = figure.make_figure()
        self.ax = self.fig.axes[0]

    def tearDown(self):
        plt.close(self.fig)

    def test_canvas_and_type_contract(self):
        self.assertEqual(tuple(self.fig.get_size_inches()), (7, 3.25))
        self.assertTrue(all(t.get_fontsize() >= 8 for t in self.ax.texts))
        self.assertTrue(all(t.get_fontfamily() == ["Times New Roman"] for t in self.ax.texts))

    def test_same_catalog_and_actual_categories(self):
        self.assertEqual(len(figure.BEFORE), 3)
        self.assertEqual(len(figure.BEFORE), len(figure.AFTER))
        for before, after in zip(figure.BEFORE, figure.AFTER):
            self.assertNotEqual(before, after)
            self.assertEqual(figure.CATALOG[before][0], figure.CATALOG[after][0])
        # Matched maps differ only by their horizontal offset, never by scale.
        for key in figure.CATALOG:
            bx, by = figure.location((362, 64, 62, 47), key)
            ax, ay = figure.location((437, 64, 62, 47), key)
            self.assertAlmostEqual(ax-bx, 75)
            self.assertEqual(ay, by)

    def test_required_mechanism_labels(self):
        labels = "\n".join(t.get_text() for t in self.ax.texts)
        for required in ("Controlled late steps", "Time branch unchanged",
                         "Partial order M; ρ, λ",
                         "Token KL: base → projected", "Distance KL",
                         "λ·category JSD", "Filter catalog by sampled category",
                         "Reselect with same g; category tokens retained",
                         "Fixed: actual categories A → B → C",
                         "Times, length, event order", "Final unsupported category"):
            self.assertIn(required, labels)
        self.assertNotIn("category KL", labels)

    def test_condition_arrows_are_dashed(self):
        arrows = [p for p in self.ax.patches
                  if (p.get_gid() or "").startswith("partial-order-condition-")
                  or ((p.get_gid() or "").startswith("event-")
                      and "-condition-" in p.get_gid())]
        self.assertEqual(len(arrows), 4)
        self.assertTrue(all(p.get_linestyle() == (0, (2.5, 2)) for p in arrows))

    def test_live_vector_svg_and_unique_semantic_ids(self):
        stream = io.StringIO()
        self.fig.savefig(stream, format="svg", metadata={"Date": None})
        root = ET.fromstring(stream.getvalue())
        ns = {"s": "http://www.w3.org/2000/svg"}
        self.assertGreater(len(root.findall(".//s:text", ns)), 50)
        self.assertEqual(root.findall(".//s:image", ns), [])
        ids = [e.attrib["id"] for e in root.iter() if "id" in e.attrib]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(not any(c.isspace() for c in i) for i in ids))
        self.assertTrue(any(i.startswith("reverse-feedback-") for i in ids))
        self.assertEqual(sum(i.startswith("masked-out-") for i in ids), 12)


if __name__ == "__main__":
    unittest.main()
