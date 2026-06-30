from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from metrics.constants import CATEGORY_SET, TRANSFORMS
from metrics.data import ImageRecord


@dataclass(frozen=True)
class MaskBuildStats:
    skipped_malformed_polygons: int = 0


def build_gt_mask(
    record: ImageRecord,
    categories: set[str],
) -> tuple[np.ndarray, MaskBuildStats]:
    unknown_categories = categories - CATEGORY_SET
    if unknown_categories:
        raise ValueError(f"Unknown requested categories: {sorted(unknown_categories)}")

    mask = np.zeros((record.height, record.width), dtype=np.uint8)
    skipped = 0
    for label in record.labels:
        if label.category not in categories:
            continue
        if len(label.points) < 3:
            skipped += 1
            continue
        points = np.array(label.points, dtype=np.int32)
        points[:, 0] = np.clip(points[:, 0], 0, record.width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, record.height - 1)
        cv2.fillPoly(mask, [points], 255)

    return mask, MaskBuildStats(skipped_malformed_polygons=skipped)


def transformed_shape(record: ImageRecord, transform: str) -> tuple[int, int]:
    if transform == "keep-original-size":
        return record.height, record.width
    if transform == "resize256-crop224":
        return 224, 224
    if transform == "resize518-crop518":
        return 518, 518
    raise ValueError(f"Unsupported transform {transform!r}; expected one of {TRANSFORMS}")


def transform_mask(mask: np.ndarray, transform: str) -> np.ndarray:
    if transform == "keep-original-size":
        return mask
    if transform == "resize256-crop224":
        resized = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        return resized[16:240, 16:240]
    if transform == "resize518-crop518":
        height, width = mask.shape[:2]
        if height < width:
            new_height, new_width = 518, int(width * 518 / height)
        else:
            new_height, new_width = int(height * 518 / width), 518
        resized = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        height, width = resized.shape[:2]
        start_height = (height - 518) // 2
        start_width = (width - 518) // 2
        return resized[start_height : start_height + 518, start_width : start_width + 518]
    raise ValueError(f"Unsupported transform {transform!r}; expected one of {TRANSFORMS}")
