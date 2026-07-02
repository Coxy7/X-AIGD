from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_binary_mask(
    mask_path: Path,
    expected_shape: tuple[int, int],
    *,
    threshold: int = 127,
) -> np.ndarray:
    validate_prediction_threshold(threshold)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Failed to read prediction mask: {mask_path}")
    if mask.shape != expected_shape:
        raise ValueError(
            f"Prediction mask has shape {mask.shape}, expected {expected_shape}: {mask_path}"
        )
    return np.where(mask > threshold, 255, 0).astype(np.uint8)


def union_prediction_dir(
    prediction_dir: Path,
    expected_shape: tuple[int, int],
    *,
    threshold: int = 127,
) -> tuple[np.ndarray, bool]:
    validate_prediction_threshold(threshold)
    total_mask = np.zeros(expected_shape, dtype=np.uint8)
    if not prediction_dir.exists():
        return total_mask, True
    if not prediction_dir.is_dir():
        raise ValueError(f"Prediction path exists but is not a directory: {prediction_dir}")

    mask_files = sorted(prediction_dir.glob("*.png"))
    if not mask_files:
        return total_mask, True

    for mask_file in mask_files:
        total_mask = cv2.bitwise_or(
            total_mask,
            read_binary_mask(mask_file, expected_shape, threshold=threshold),
        )
    return total_mask, not bool(np.any(total_mask))


def validate_prediction_threshold(threshold: int) -> None:
    if not 0 <= threshold <= 255:
        raise ValueError(f"prediction threshold must be between 0 and 255, got {threshold}")
