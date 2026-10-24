"""Reference-figure contracts; these tests never import experimental code."""
import io
import unittest
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import generate_postswap_pcdg as figure


class IntroFigureTests(unittest.TestCase):
    def setUp(self):
        self.fig = figure.make_figure()
        self.ax = self.fig.axes[0]

    def tearDown(self):
        plt.close(self.fig)

    def test_canvas_and_fonts(self):
        self.assertEqual(tuple(self.fig.get_size_inches()), (7, 3.25))
        self.assertTrue(all(t.get_fontsize() >= 8 for t in self.ax.texts))
        self.assertTrue(all(t.get_fontfamily() == ["Times New Roman"] for t in self.ax.texts))

    def test_shared_catalog_and_coordinates(self):
        self.assertEqual(len(figure.CATALOG), 6)
        self.assertEqual({p[0] for p in figure.CATALOG.values()}, {"S", "W", "D", "H"})
        left, right = figure.MAP_BOUNDS["postswap"], figure.MAP_BOUNDS["pcdg"]
        self.assertEqual(left[1:], right[1:])
        for key in figure.CATALOG:
            lx, ly = figure.location(left, key)
            rx, ry = figure.location(right, key)
            self.assertAlmostEqual(rx-lx, right[0]-left[0])
            self.assertEqual(ly, ry)

    def test_rendered_poi_centers_match_shared_catalog(self):
        rings = [p for p in self.ax.patches if isinstance(p, Circle)
                 and "-poi-" in (p.get_gid() or "") and "-ring-" in p.get_gid()]
        self.assertEqual(len(rings), 12)
        for name, bounds in figure.MAP_BOUNDS.items():
            for key in figure.CATALOG:
                ring = next(p for p in rings if p.get_gid().startswith(f"{name}-poi-{key}-ring-"))
                self.assertEqual(ring.center, figure.location(bounds, key))

    def test_route_order_and_alternative_choices(self):
        self.assertEqual(figure.ROUTES["postswap"], ("s-east", "w-west", "d-west", "h-east"))
        self.assertEqual(figure.ROUTES["pcdg"], ("s-east", "w-east", "d-east", "h-east"))
        for route in figure.ROUTES.values():
            self.assertEqual([figure.CATALOG[k][0] for k in route], list("SWDH"))
        arrows = [p for p in self.ax.patches if "-route-" in (p.get_gid() or "")]
        self.assertEqual(len(arrows), 6)
        for name in figure.ROUTES:
            numbers = [t.get_text() for t in self.ax.texts
                       if (t.get_gid() or "").startswith(name + "-visit-number-")]
            self.assertEqual(numbers, ["1", "2", "3", "4"])

    def test_text_bounds_and_no_text_overlap(self):
        renderer = self.fig.canvas.get_renderer()
        bounds = [t.get_window_extent(renderer) for t in self.ax.texts]
        for box in bounds:
            self.assertTrue(self.fig.bbox.contains(box.x0, box.y0))
            self.assertTrue(self.fig.bbox.contains(box.x1, box.y1))
        for i, box in enumerate(bounds):
            for j in range(i):
                self.assertFalse(box.overlaps(bounds[j]),
                                 (self.ax.texts[i].get_text(), self.ax.texts[j].get_text()))

    def test_live_vector_svg(self):
        stream = io.StringIO()
        self.fig.savefig(stream, format="svg", metadata={"Date": None})
        root = ET.fromstring(stream.getvalue())
        ns = {"s": "http://www.w3.org/2000/svg"}
        self.assertGreater(len(root.findall(".//s:text", ns)), 20)
        self.assertEqual(root.findall(".//s:image", ns), [])
        ids = [e.attrib["id"] for e in root.iter() if "id" in e.attrib]
        self.assertEqual(len(ids), len(set(ids)))


if __name__ == "__main__":
    unittest.main()
