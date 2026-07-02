from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from metrics.constants import CATEGORIES
from metrics.data import ImageRecord
from metrics.masks import MaskTransform, build_gt_mask, transform_mask, transformed_shape
from metrics.predictions import union_prediction_dir


@dataclass
class PixelCounts:
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    evaluated_images: int = 0
    zero_prediction_images: int = 0

    def add_masks(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> None:
        self.true_positive += int(np.sum((pred_mask == 255) & (gt_mask == 255)))
        self.false_positive += int(np.sum((pred_mask == 255) & (gt_mask == 0)))
        self.false_negative += int(np.sum((pred_mask == 0) & (gt_mask == 255)))
        self.evaluated_images += 1

    def to_metrics_row(
        self,
        *,
        generator: str,
        task: str,
        category: str | None = None,
    ) -> dict[str, float | int | str]:
        iou = safe_div(
            self.true_positive,
            self.true_positive + self.false_positive + self.false_negative,
        )
        precision = safe_div(self.true_positive, self.true_positive + self.false_positive)
        recall = safe_div(self.true_positive, self.true_positive + self.false_negative)
        f1 = safe_div(2 * precision * recall, precision + recall)
        row: dict[str, float | int | str] = {
            "generator": generator,
            "task": task,
            "category": category or "all",
            "IoU": iou,
            "PixP": precision,
            "PixR": recall,
            "PixF1": f1,
            "evaluated_images": self.evaluated_images,
            "zero_prediction_images": self.zero_prediction_images,
        }
        return row


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate_category_agnostic(
    records: list[ImageRecord],
    prediction_root: Path,
    *,
    transform: MaskTransform,
    prediction_threshold: int = 127,
) -> dict[str, float | int | str]:
    counts = compute_category_agnostic_counts(
        records,
        prediction_root,
        transform=transform,
        prediction_threshold=prediction_threshold,
    )
    return counts.to_metrics_row(generator="all", task="category-agnostic")


def evaluate_category_agnostic_per_generator(
    records: list[ImageRecord],
    prediction_root: Path,
    *,
    transform: MaskTransform,
    prediction_threshold: int = 127,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for generator, generator_records in group_records_by_generator(records).items():
        counts = compute_category_agnostic_counts(
            generator_records,
            prediction_root,
            transform=transform,
            prediction_threshold=prediction_threshold,
        )
        rows.append(counts.to_metrics_row(generator=generator, task="category-agnostic"))
    return rows


def compute_category_agnostic_counts(
    records: list[ImageRecord],
    prediction_root: Path,
    *,
    transform: MaskTransform,
    prediction_threshold: int = 127,
) -> PixelCounts:
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
        gt_mask = build_gt_mask(record, category_set)
        gt_mask = transform_mask(gt_mask, transform)
        counts.zero_prediction_images += int(is_zero_prediction)
        counts.add_masks(pred_mask, gt_mask)
    return counts


def evaluate_fine_grained(
    records: list[ImageRecord],
    prediction_root: Path,
    *,
    transform: MaskTransform,
    prediction_threshold: int = 127,
) -> list[dict[str, float | int | str]]:
    category_counts = compute_fine_grained_counts(
        records,
        prediction_root,
        transform=transform,
        prediction_threshold=prediction_threshold,
    )
    return [
        category_counts[category].to_metrics_row(
            generator="all",
            task="fine-grained",
            category=category,
        )
        for category in CATEGORIES
    ]


def evaluate_fine_grained_per_generator(
    records: list[ImageRecord],
    prediction_root: Path,
    *,
    transform: MaskTransform,
    prediction_threshold: int = 127,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for generator, generator_records in group_records_by_generator(records).items():
        category_counts = compute_fine_grained_counts(
            generator_records,
            prediction_root,
            transform=transform,
            prediction_threshold=prediction_threshold,
        )
        for category in CATEGORIES:
            rows.append(
                category_counts[category].to_metrics_row(
                    generator=generator,
                    task="fine-grained",
                    category=category,
                )
            )
    return rows


def compute_fine_grained_counts(
    records: list[ImageRecord],
    prediction_root: Path,
    *,
    transform: MaskTransform,
    prediction_threshold: int = 127,
) -> dict[str, PixelCounts]:
    category_counts = {category: PixelCounts() for category in CATEGORIES}
    for record in records:
        expected_shape = transformed_shape(record, transform)
        for category in CATEGORIES:
            counts = category_counts[category]
            pred_dir = prediction_root / record.generator / category / record.uid
            pred_mask, is_zero_prediction = union_prediction_dir(
                pred_dir,
                expected_shape,
                threshold=prediction_threshold,
            )
            gt_mask = build_gt_mask(record, {category})
            gt_mask = transform_mask(gt_mask, transform)
            counts.zero_prediction_images += int(is_zero_prediction)
            counts.add_masks(pred_mask, gt_mask)
    return category_counts


def group_records_by_generator(records: list[ImageRecord]) -> dict[str, list[ImageRecord]]:
    records_by_generator: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        records_by_generator[record.generator].append(record)
    return {generator: records_by_generator[generator] for generator in sorted(records_by_generator)}
