from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from metrics.constants import CATEGORY_SET
from metrics.data import ImageRecord


@dataclass(frozen=True)
class MaskTransform:
    resize_size: tuple[int, int] | None = None
    resize_short_side: int | None = None
    center_crop: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if self.resize_size is not None and self.resize_short_side is not None:
            raise ValueError("resize_size and resize_short_side are mutually exclusive")
        if self.resize_size is not None:
            validate_size(self.resize_size, "resize_size")
        if self.resize_short_side is not None and self.resize_short_side <= 0:
            raise ValueError("resize_short_side must be positive")
        if self.center_crop is not None:
            validate_size(self.center_crop, "center_crop")


def validate_size(size: tuple[int, int], name: str) -> None:
    if len(size) != 2 or size[0] <= 0 or size[1] <= 0:
        raise ValueError(f"{name} must contain positive height and width values")


def build_gt_mask(
    record: ImageRecord,
    categories: set[str],
) -> np.ndarray:
    unknown_categories = categories - CATEGORY_SET
    if unknown_categories:
        raise ValueError(f"Unknown requested categories: {sorted(unknown_categories)}")

    mask = np.zeros((record.height, record.width), dtype=np.uint8)
    for label in record.labels:
        if label.category not in categories:
            continue
        if len(label.points) < 3:
            raise ValueError(
                f"Malformed polygon for {record.generator}/{record.uid} "
                f"in category {label.category!r}: expected at least 3 points, got {len(label.points)}"
            )
        points = np.array(label.points, dtype=np.int32)
        points[:, 0] = np.clip(points[:, 0], 0, record.width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, record.height - 1)
        cv2.fillPoly(mask, [points], 255)

    return mask


def transformed_shape(record: ImageRecord, transform: MaskTransform) -> tuple[int, int]:
    shape = apply_resize_shape((record.height, record.width), transform)
    return apply_center_crop_shape(shape, transform)


def transform_mask(mask: np.ndarray, transform: MaskTransform) -> np.ndarray:
    output_mask = apply_resize(mask, transform)
    return apply_center_crop(output_mask, transform)


def apply_resize(mask: np.ndarray, transform: MaskTransform) -> np.ndarray:
    if transform.resize_size is not None:
        height, width = transform.resize_size
        return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    if transform.resize_short_side is not None:
        height, width = mask.shape[:2]
        short_side = transform.resize_short_side
        if height < width:
            new_height, new_width = short_side, int(width * short_side / height)
        else:
            new_height, new_width = int(height * short_side / width), short_side
        return cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    return mask


def apply_resize_shape(shape: tuple[int, int], transform: MaskTransform) -> tuple[int, int]:
    if transform.resize_size is not None:
        return transform.resize_size
    if transform.resize_short_side is not None:
        height, width = shape
        short_side = transform.resize_short_side
        if height < width:
            return short_side, int(width * short_side / height)
        return int(height * short_side / width), short_side
    return shape


def apply_center_crop(mask: np.ndarray, transform: MaskTransform) -> np.ndarray:
    if transform.center_crop is None:
        return mask
    crop_height, crop_width = transform.center_crop
    height, width = mask.shape[:2]
    if crop_height > height or crop_width > width:
        raise ValueError(
            f"center_crop {(crop_height, crop_width)} exceeds transformed mask shape {(height, width)}"
        )
    start_height = (height - crop_height) // 2
    start_width = (width - crop_width) // 2
    return mask[start_height : start_height + crop_height, start_width : start_width + crop_width]


def apply_center_crop_shape(
    shape: tuple[int, int],
    transform: MaskTransform,
) -> tuple[int, int]:
    if transform.center_crop is None:
        return shape
    crop_height, crop_width = transform.center_crop
    height, width = shape
    if crop_height > height or crop_width > width:
        raise ValueError(
            f"center_crop {(crop_height, crop_width)} exceeds transformed mask shape {(height, width)}"
        )
    return crop_height, crop_width
