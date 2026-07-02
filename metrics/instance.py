from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from metrics.constants import CATEGORIES, CATEGORY_SET
from metrics.data import ImageRecord
from metrics.masks import build_gt_mask


@dataclass(frozen=True)
class PredictedBox:
    generator: str
    uid: str
    category: str
    x_min: float
    y_min: float
    x_max: float
    y_max: float


@dataclass
class InstanceCounts:
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    evaluated_images: int = 0
    zero_prediction_images: int = 0
    invalid_prediction_boxes: int = 0

    def to_row(
        self,
        *,
        generator: str,
        category: str,
        overlap_threshold: float,
    ) -> dict[str, float | int | str]:
        precision = safe_div(self.true_positive, self.true_positive + self.false_positive)
        recall = safe_div(self.true_positive, self.true_positive + self.false_negative)
        f1 = safe_div(2 * precision * recall, precision + recall)
        threshold_label = format_threshold(overlap_threshold)
        return {
            "generator": generator,
            "category": category,
            f"P@{threshold_label}": precision,
            f"R@{threshold_label}": recall,
            f"F1@{threshold_label}": f1,
            "TP": self.true_positive,
            "FP": self.false_positive,
            "FN": self.false_negative,
            "evaluated_images": self.evaluated_images,
            "zero_prediction_images": self.zero_prediction_images,
            "invalid_prediction_boxes": self.invalid_prediction_boxes,
        }


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def format_threshold(threshold: float) -> str:
    return f"{threshold:g}"


def load_instance_predictions(csv_path: Path) -> dict[tuple[str, str, str], list[PredictedBox]]:
    required_columns = {"generator", "uid", "category", "x_min", "y_min", "x_max", "y_max"}
    predictions: dict[tuple[str, str, str], list[PredictedBox]] = defaultdict(list)
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"Prediction CSV is empty: {csv_path}")
        missing_columns = required_columns - set(reader.fieldnames)
        if missing_columns:
            raise ValueError(f"Prediction CSV is missing columns: {sorted(missing_columns)}")
        for row_number, row in enumerate(reader, start=2):
            category = row["category"]
            if category not in CATEGORY_SET:
                raise ValueError(f"Unknown prediction category at row {row_number}: {category!r}")
            try:
                box = PredictedBox(
                    generator=row["generator"],
                    uid=row["uid"],
                    category=category,
                    x_min=float(row["x_min"]),
                    y_min=float(row["y_min"]),
                    x_max=float(row["x_max"]),
                    y_max=float(row["y_max"]),
                )
            except ValueError as exc:
                raise ValueError(f"Invalid numeric bounding box at row {row_number}") from exc
            predictions[(box.generator, box.uid, box.category)].append(box)
    return dict(predictions)


def evaluate_instance_predictions(
    records: list[ImageRecord],
    predictions: dict[tuple[str, str, str], list[PredictedBox]],
    *,
    overlap_threshold: float = 0.5,
) -> tuple[list[dict[str, float | int | str]], list[dict[str, float | int | str]]]:
    if not 0 <= overlap_threshold <= 1:
        raise ValueError("overlap_threshold must be between 0 and 1")

    records_by_generator: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        records_by_generator[record.generator].append(record)

    per_generator_rows: list[dict[str, float | int | str]] = []
    overall_counts = {category: InstanceCounts() for category in CATEGORIES}
    for generator in sorted(records_by_generator):
        generator_records = records_by_generator[generator]
        for category in CATEGORIES:
            counts = InstanceCounts()
            for record in generator_records:
                image_counts = evaluate_instance_record(
                    record,
                    category,
                    predictions.get((record.generator, record.uid, category), []),
                    overlap_threshold=overlap_threshold,
                )
                merge_instance_counts(counts, image_counts)
                merge_instance_counts(overall_counts[category], image_counts)
            per_generator_rows.append(
                counts.to_row(
                    generator=generator,
                    category=category,
                    overlap_threshold=overlap_threshold,
                )
            )

    overall_rows = [
        overall_counts[category].to_row(
            generator="all",
            category=category,
            overlap_threshold=overlap_threshold,
        )
        for category in CATEGORIES
    ]
    return per_generator_rows, overall_rows


def evaluate_instance_record(
    record: ImageRecord,
    category: str,
    pred_boxes: list[PredictedBox],
    *,
    overlap_threshold: float,
) -> InstanceCounts:
    counts = InstanceCounts(evaluated_images=1, zero_prediction_images=int(not pred_boxes))
    composite_mask = build_gt_mask(record, {category})

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated_mask = cv2.dilate(composite_mask, kernel, iterations=1)
    num_labels, labels_im = cv2.connectedComponents(dilated_mask, connectivity=8)
    pred_matched = [0 for _ in pred_boxes]
    gt_matched = [0 for _ in range(1, num_labels)]

    for pred_idx, pred_box in enumerate(pred_boxes):
        for component_label in range(1, num_labels):
            component_idx = component_label - 1
            gt_instance_mask = (labels_im == component_label).astype(np.uint8) * 255
            gt_instance_mask = cv2.bitwise_and(gt_instance_mask, composite_mask)
            match_result = match_box_to_mask(
                gt_instance_mask,
                pred_box,
                record.width,
                record.height,
                overlap_threshold=overlap_threshold,
            )
            if match_result is True:
                pred_matched[pred_idx] = 1
                gt_matched[component_idx] = 1
            elif match_result is None:
                counts.invalid_prediction_boxes += 1
                break

    counts.true_positive = gt_matched.count(1)
    counts.false_positive = pred_matched.count(0)
    counts.false_negative = gt_matched.count(0)
    return counts


def match_box_to_mask(
    gt_instance_mask: np.ndarray,
    pred_box: PredictedBox,
    image_width: int,
    image_height: int,
    *,
    overlap_threshold: float,
) -> bool | None:
    if not (
        0 <= pred_box.x_min < pred_box.x_max <= image_width
        and 0 <= pred_box.y_min < pred_box.y_max <= image_height
    ):
        return None

    x_min = min(int(pred_box.x_min), image_width)
    x_max = min(int(pred_box.x_max), image_width)
    y_min = min(int(pred_box.y_min), image_height)
    y_max = min(int(pred_box.y_max), image_height)
    if not (0 <= x_min < x_max <= image_width and 0 <= y_min < y_max <= image_height):
        return None

    pred_mask = np.zeros((image_height, image_width), dtype=np.uint8)
    pred_mask[y_min:y_max, x_min:x_max] = 255
    pred_area = int(np.count_nonzero(pred_mask))
    if pred_area == 0:
        return None

    intersection = cv2.bitwise_and(gt_instance_mask, pred_mask)
    intersection_area = int(np.count_nonzero(intersection))
    return (intersection_area / pred_area) > overlap_threshold


def merge_instance_counts(target: InstanceCounts, source: InstanceCounts) -> None:
    target.true_positive += source.true_positive
    target.false_positive += source.false_positive
    target.false_negative += source.false_negative
    target.evaluated_images += source.evaluated_images
    target.zero_prediction_images += source.zero_prediction_images
    target.invalid_prediction_boxes += source.invalid_prediction_boxes
