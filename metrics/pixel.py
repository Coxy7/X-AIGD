from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from metrics.constants import CATEGORIES
from metrics.data import ImageRecord
from metrics.masks import build_gt_mask, transform_mask, transformed_shape
from metrics.predictions import union_prediction_dir


@dataclass
class PixelCounts:
    tp_0: int = 0
    fp_0: int = 0
    fn_0: int = 0
    tp_255: int = 0
    fp_255: int = 0
    fn_255: int = 0
    evaluated_images: int = 0
    zero_prediction_images: int = 0
    skipped_malformed_polygons: int = 0

    def add_masks(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> None:
        self.tp_0 += int(np.sum((pred_mask == 0) & (gt_mask == 0)))
        self.fp_0 += int(np.sum((pred_mask == 0) & (gt_mask == 255)))
        self.fn_0 += int(np.sum((pred_mask == 255) & (gt_mask == 0)))
        self.tp_255 += int(np.sum((pred_mask == 255) & (gt_mask == 255)))
        self.fp_255 += int(np.sum((pred_mask == 255) & (gt_mask == 0)))
        self.fn_255 += int(np.sum((pred_mask == 0) & (gt_mask == 255)))
        self.evaluated_images += 1

    def to_metrics_row(self, *, task: str, category: str | None = None) -> dict[str, float | int | str]:
        iou_0 = safe_div(self.tp_0, self.tp_0 + self.fp_0 + self.fn_0)
        iou_255 = safe_div(self.tp_255, self.tp_255 + self.fp_255 + self.fn_255)
        precision_255 = safe_div(self.tp_255, self.tp_255 + self.fp_255)
        recall_255 = safe_div(self.tp_255, self.tp_255 + self.fn_255)
        f1_255 = safe_div(2 * precision_255 * recall_255, precision_255 + recall_255)
        row: dict[str, float | int | str] = {
            "task": task,
            "category": category or "all",
            "TP_0": self.tp_0,
            "FP_0": self.fp_0,
            "FN_0": self.fn_0,
            "TP_255": self.tp_255,
            "FP_255": self.fp_255,
            "FN_255": self.fn_255,
            "iou_255": iou_255 * 100,
            "precision_255": precision_255 * 100,
            "recall_255": recall_255 * 100,
            "f1_255": f1_255 * 100,
            "mIoU": ((iou_0 + iou_255) / 2) * 100,
            "evaluated_images": self.evaluated_images,
            "zero_prediction_images": self.zero_prediction_images,
            "skipped_malformed_polygons": self.skipped_malformed_polygons,
        }
        return row


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate_category_agnostic(
    records: list[ImageRecord],
    prediction_root: Path,
    *,
    transform: str,
    prediction_threshold: int = 127,
) -> dict[str, float | int | str]:
    counts = PixelCounts()
    category_set = set(CATEGORIES)
    for record in records:
        expected_shape = transformed_shape(record, transform)
        pred_dir = prediction_root / record.generator / record.uid
        pred_mask, is_zero_prediction = union_prediction_dir(
            pred_dir,
            expected_shape,
            threshold=prediction_threshold,
        )
        gt_mask, stats = build_gt_mask(record, category_set)
        gt_mask = transform_mask(gt_mask, transform)
        counts.skipped_malformed_polygons += stats.skipped_malformed_polygons
        counts.zero_prediction_images += int(is_zero_prediction)
        counts.add_masks(pred_mask, gt_mask)
    return counts.to_metrics_row(task="category-agnostic")


def evaluate_fine_grained(
    records: list[ImageRecord],
    prediction_root: Path,
    *,
    transform: str,
    prediction_threshold: int = 127,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for category in CATEGORIES:
        counts = PixelCounts()
        for record in records:
            expected_shape = transformed_shape(record, transform)
            pred_dir = prediction_root / record.generator / category / record.uid
            pred_mask, is_zero_prediction = union_prediction_dir(
                pred_dir,
                expected_shape,
                threshold=prediction_threshold,
            )
            gt_mask, stats = build_gt_mask(record, {category})
            gt_mask = transform_mask(gt_mask, transform)
            counts.skipped_malformed_polygons += stats.skipped_malformed_polygons
            counts.zero_prediction_images += int(is_zero_prediction)
            counts.add_masks(pred_mask, gt_mask)
        rows.append(counts.to_metrics_row(task="fine-grained", category=category))
    return rows
