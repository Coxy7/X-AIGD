from __future__ import annotations

import unittest

from PIL import Image

from xaigd_viewer.models import ArtifactLabel
from xaigd_viewer.render import draw_overlays, polygon_color


class RenderTests(unittest.TestCase):
    def test_polygon_color_is_deterministic(self) -> None:
        self.assertEqual(polygon_color("low-level-edge_shape"), polygon_color("low-level-edge_shape"))
        self.assertEqual(polygon_color("unknown"), (128, 128, 128, 92))

    def test_draw_overlays_skips_malformed_polygons(self) -> None:
        image = Image.new("RGB", (24, 24), "white")
        labels = [
            ArtifactLabel(
                category="low-level-edge_shape",
                points=((2.0, 2.0), (20.0, 2.0), (20.0, 20.0)),
            ),
            ArtifactLabel(
                category="low-level-color",
                points=((5.0, 5.0), (6.0, 6.0)),
            ),
        ]

        rendered, skipped = draw_overlays(image, labels)

        self.assertEqual(rendered.size, (24, 24))
        self.assertEqual(skipped, 1)
        self.assertNotEqual(rendered.getpixel((10, 2)), (255, 255, 255, 255))


if __name__ == "__main__":
    unittest.main()
