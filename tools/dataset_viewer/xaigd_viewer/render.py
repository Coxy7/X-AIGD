from __future__ import annotations

from typing import Iterable, Tuple

from PIL import Image, ImageDraw

from xaigd_viewer.constants import CATEGORY_COLORS
from xaigd_viewer.models import ArtifactLabel


def is_valid_polygon(label: ArtifactLabel) -> bool:
    return len(label.points) >= 3


def polygon_color(category: str) -> Tuple[int, int, int, int]:
    return CATEGORY_COLORS.get(category, {"rgba": (128, 128, 128, 92)})["rgba"]


def draw_overlays(image: Image.Image, labels: Iterable[ArtifactLabel]) -> Tuple[Image.Image, int]:
    canvas = image.convert("RGBA").copy()
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    line_width = 6

    skipped_count = 0
    for label in labels:
        if not is_valid_polygon(label):
            skipped_count += 1
            continue
        outline_rgba = polygon_color(label.category)[:3] + (255,)
        closed_points = [*label.points, label.points[0]]
        draw.line(closed_points, fill=outline_rgba, width=line_width, joint="curve")

    composed = Image.alpha_composite(canvas, overlay)
    return composed, skipped_count
