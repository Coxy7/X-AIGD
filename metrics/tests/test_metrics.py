from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from metrics.data import ArtifactLabel, ImageRecord, normalize_record
from metrics.instance import (
    PredictedBox,
    evaluate_instance_predictions,
    load_instance_predictions,
    match_box_to_mask,
)
from metrics.masks import MaskTransform, build_gt_mask, transform_mask, transformed_shape
from metrics.pixel import evaluate_category_agnostic, evaluate_category_agnostic_per_generator
from metrics.predictions import union_prediction_dir


def make_record(
    *,
    labels: tuple[ArtifactLabel, ...],
    width: int = 8,
    height: int = 8,
    generator: str = "gen",
    uid: str = "uid",
) -> ImageRecord:
    return ImageRecord(generator=generator, uid=uid, width=width, height=height, labels=labels)


class MaskTests(unittest.TestCase):
    def test_gt_mask_uses_cv2_fillpoly_with_int32_truncation(self) -> None:
        record = make_record(
            labels=(
                ArtifactLabel(
                    category="low-level-edge_shape",
                    points=((1.9, 1.9), (5.9, 1.9), (1.9, 5.9)),
                ),
            )
        )

        mask = build_gt_mask(record, {"low-level-edge_shape"})
        expected = np.zeros((8, 8), dtype=np.uint8)
        cv2.fillPoly(expected, [np.array([(1, 1), (5, 1), (1, 5)], dtype=np.int32)], 255)

        np.testing.assert_array_equal(mask, expected)

    def test_gt_mask_rejects_malformed_polygon(self) -> None:
        record = make_record(
            labels=(
                ArtifactLabel(
                    category="low-level-edge_shape",
                    points=((1, 1), (2, 2)),
                ),
            )
        )

        with self.assertRaises(ValueError):
            build_gt_mask(record, {"low-level-edge_shape"})

    def test_nearest_transform_keeps_binary_values(self) -> None:
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:6, 2:6] = 255

        transformed = transform_mask(mask, MaskTransform(resize_size=(256, 256), center_crop=(224, 224)))

        self.assertEqual(transformed.shape, (224, 224))
        self.assertEqual(set(np.unique(transformed).tolist()), {0, 255})

    def test_transformed_shape_matches_transform(self) -> None:
        record = make_record(labels=(), width=10, height=6)

        self.assertEqual(transformed_shape(record, MaskTransform()), (6, 10))
        self.assertEqual(
            transformed_shape(record, MaskTransform(resize_size=(256, 256), center_crop=(224, 224))),
            (224, 224),
        )
        self.assertEqual(
            transformed_shape(record, MaskTransform(resize_short_side=518, center_crop=(518, 518))),
            (518, 518),
        )

    def test_transform_rejects_crop_larger_than_resized_shape(self) -> None:
        record = make_record(labels=(), width=10, height=6)

        with self.assertRaises(ValueError):
            transformed_shape(record, MaskTransform(resize_short_side=5, center_crop=(6, 6)))

    def test_unknown_dataset_category_raises(self) -> None:
        with self.assertRaises(ValueError):
            normalize_record(
                {
                    "generator": "gen",
                    "uid": "uid",
                    "width": 8,
                    "height": 8,
                    "labels": [
                        {"label": "unknown-category", "points": [[1, 1], [2, 1], [1, 2]]}
                    ],
                }
            )


class PredictionTests(unittest.TestCase):
    def test_missing_prediction_dir_is_zero_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            mask, is_zero = union_prediction_dir(Path(tmpdir) / "missing", (4, 4))

        self.assertTrue(is_zero)
        self.assertEqual(int(mask.sum()), 0)

    def test_prediction_threshold_rejects_out_of_range_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pred_dir = Path(tmpdir) / "missing"

            with self.assertRaises(ValueError):
                union_prediction_dir(pred_dir, (4, 4), threshold=-1)
            with self.assertRaises(ValueError):
                union_prediction_dir(pred_dir, (4, 4), threshold=256)

    def test_prediction_dimension_mismatch_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pred_dir = Path(tmpdir)
            cv2.imwrite(str(pred_dir / "mask.png"), np.zeros((5, 5), dtype=np.uint8))

            with self.assertRaises(ValueError):
                union_prediction_dir(pred_dir, (4, 4))

    def test_pixel_metric_outputs_paper_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pred_dir = root / "gen" / "uid"
            pred_dir.mkdir(parents=True)
            pred = np.zeros((8, 8), dtype=np.uint8)
            pred[1:6, 1:6] = 255
            cv2.imwrite(str(pred_dir / "mask.png"), pred)
            record = make_record(
                labels=(
                    ArtifactLabel(
                        category="low-level-edge_shape",
                        points=((1, 1), (3, 1), (3, 3), (1, 3)),
                    ),
                )
            )

            row = evaluate_category_agnostic(
                [record],
                root,
                transform=MaskTransform(),
            )

        self.assertEqual(row["evaluated_images"], 1)
        self.assertEqual(row["zero_prediction_images"], 0)
        self.assertEqual(row["generator"], "all")
        self.assertEqual(row["IoU"], 0.36)
        self.assertEqual(row["PixP"], 0.36)
        self.assertEqual(row["PixR"], 1.0)
        self.assertAlmostEqual(row["PixF1"], 0.5294117647058824)
        self.assertEqual(
            set(row),
            {
                "generator",
                "task",
                "category",
                "IoU",
                "PixP",
                "PixR",
                "PixF1",
                "evaluated_images",
                "zero_prediction_images",
            },
        )

    def test_pixel_metric_outputs_per_generator_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for generator in ("gen-a", "gen-b"):
                pred_dir = root / generator / "uid"
                pred_dir.mkdir(parents=True)
                pred = np.zeros((8, 8), dtype=np.uint8)
                pred[1:4, 1:4] = 255
                cv2.imwrite(str(pred_dir / "mask.png"), pred)

            records = [
                make_record(
                    generator="gen-a",
                    labels=(
                        ArtifactLabel(
                            category="low-level-edge_shape",
                            points=((1, 1), (3, 1), (3, 3), (1, 3)),
                        ),
                    ),
                ),
                make_record(
                    generator="gen-b",
                    labels=(
                        ArtifactLabel(
                            category="low-level-edge_shape",
                            points=((1, 1), (3, 1), (3, 3), (1, 3)),
                        ),
                    ),
                ),
            ]

            rows = evaluate_category_agnostic_per_generator(
                records,
                root,
                transform=MaskTransform(),
            )

        self.assertEqual([row["generator"] for row in rows], ["gen-a", "gen-b"])
        self.assertEqual([row["evaluated_images"] for row in rows], [1, 1])
        self.assertEqual([row["IoU"] for row in rows], [1.0, 1.0])


class InstanceTests(unittest.TestCase):
    def test_instance_csv_rejects_unknown_category(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "pred.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(
                    csv_file,
                    fieldnames=["generator", "uid", "category", "x_min", "y_min", "x_max", "y_max"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "generator": "gen",
                        "uid": "uid",
                        "category": "low-level-text",
                        "x_min": 0,
                        "y_min": 0,
                        "x_max": 1,
                        "y_max": 1,
                    }
                )

            with self.assertRaises(ValueError):
                load_instance_predictions(csv_path)

    def test_box_match_uses_intersection_over_prediction_area(self) -> None:
        gt_instance_mask = np.zeros((10, 10), dtype=np.uint8)
        gt_instance_mask[1:4, 1:4] = 255
        box = PredictedBox("gen", "uid", "low-level-edge_shape", 1, 1, 5, 5)

        self.assertTrue(
            match_box_to_mask(gt_instance_mask, box, 10, 10, overlap_threshold=0.5)
        )
        self.assertFalse(
            match_box_to_mask(gt_instance_mask, box, 10, 10, overlap_threshold=0.6)
        )

    def test_missing_instance_rows_are_zero_predictions(self) -> None:
        record = make_record(
            labels=(
                ArtifactLabel(
                    category="low-level-edge_shape",
                    points=((1, 1), (4, 1), (4, 4), (1, 4)),
                ),
            )
        )

        per_generator_rows, overall_rows = evaluate_instance_predictions([record], {})

        edge_specific = [
            row for row in per_generator_rows if row["category"] == "low-level-edge_shape"
        ][0]
        edge_overall = [
            row for row in overall_rows if row["category"] == "low-level-edge_shape"
        ][0]
        self.assertEqual(edge_specific["zero_prediction_images"], 1)
        self.assertEqual(edge_specific["FP"], 0)
        self.assertGreater(edge_specific["FN"], 0)
        self.assertEqual(edge_overall["FN"], edge_specific["FN"])

    def test_invalid_instance_box_is_counted_without_ground_truth_instances(self) -> None:
        record = make_record(labels=())
        invalid_box = PredictedBox("gen", "uid", "low-level-edge_shape", -1, 0, 2, 2)

        per_generator_rows, _ = evaluate_instance_predictions(
            [record],
            {("gen", "uid", "low-level-edge_shape"): [invalid_box]},
        )

        edge_specific = [
            row for row in per_generator_rows if row["category"] == "low-level-edge_shape"
        ][0]
        self.assertEqual(edge_specific["invalid_prediction_boxes"], 1)
        self.assertEqual(edge_specific["FP"], 1)
        self.assertEqual(edge_specific["FN"], 0)
        self.assertEqual(edge_specific["zero_prediction_images"], 0)

    def test_instance_metric_outputs_threshold_named_columns(self) -> None:
        record = make_record(
            labels=(
                ArtifactLabel(
                    category="low-level-edge_shape",
                    points=((1, 1), (4, 1), (4, 4), (1, 4)),
                ),
            )
        )
        box = PredictedBox("gen", "uid", "low-level-edge_shape", 1, 1, 4, 4)

        per_generator_rows, _ = evaluate_instance_predictions(
            [record],
            {("gen", "uid", "low-level-edge_shape"): [box]},
            overlap_threshold=0.5,
        )

        edge_specific = [
            row for row in per_generator_rows if row["category"] == "low-level-edge_shape"
        ][0]
        self.assertEqual(edge_specific["P@0.5"], 1.0)
        self.assertEqual(edge_specific["R@0.5"], 1.0)
        self.assertEqual(edge_specific["F1@0.5"], 1.0)
        self.assertEqual(
            set(edge_specific),
            {
                "generator",
                "category",
                "P@0.5",
                "R@0.5",
                "F1@0.5",
                "TP",
                "FP",
                "FN",
                "evaluated_images",
                "zero_prediction_images",
                "invalid_prediction_boxes",
            },
        )


if __name__ == "__main__":
    unittest.main()
