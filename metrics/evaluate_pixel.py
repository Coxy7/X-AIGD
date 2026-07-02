from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from metrics.data import load_hf_records
from metrics.masks import MaskTransform
from metrics.pixel import (
    evaluate_category_agnostic,
    evaluate_category_agnostic_per_generator,
    evaluate_fine_grained,
    evaluate_fine_grained_per_generator,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate X-AIGD pixel-level metrics.")
    parser.add_argument("--dataset-repo", default="Coxy7/X-AIGD")
    parser.add_argument("--split", default="labeled_test")
    parser.add_argument("--prediction-root", required=True, type=Path)
    parser.add_argument("--task", choices=["category-agnostic", "fine-grained"], required=True)
    parser.add_argument("--resize-size", nargs=2, type=int, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--resize-short-side", type=int, metavar="PIXELS")
    parser.add_argument("--center-crop", nargs=2, type=int, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--output-overall", required=True, type=Path)
    parser.add_argument("--output-per-generator", type=Path)
    parser.add_argument("--prediction-threshold", default=127, type=int)
    parser.add_argument("--revision")
    parser.add_argument("--cache-dir")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.resize_size and args.resize_short_side is not None:
        parser.error("--resize-size and --resize-short-side are mutually exclusive")
    validate_cli_size(parser, args.resize_size, "--resize-size")
    validate_cli_size(parser, args.center_crop, "--center-crop")
    if args.resize_short_side is not None and args.resize_short_side <= 0:
        parser.error("--resize-short-side must be positive")
    return args


def main() -> None:
    args = parse_args()
    per_generator_rows = None
    transform = build_transform(args)
    records = load_hf_records(
        args.dataset_repo,
        args.split,
        revision=args.revision,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    if args.task == "category-agnostic":
        overall_rows = [
            evaluate_category_agnostic(
                records,
                args.prediction_root,
                transform=transform,
                prediction_threshold=args.prediction_threshold,
            )
        ]
        if args.output_per_generator:
            per_generator_rows = evaluate_category_agnostic_per_generator(
                records,
                args.prediction_root,
                transform=transform,
                prediction_threshold=args.prediction_threshold,
            )
    else:
        overall_rows = evaluate_fine_grained(
            records,
            args.prediction_root,
            transform=transform,
            prediction_threshold=args.prediction_threshold,
        )
        if args.output_per_generator:
            per_generator_rows = evaluate_fine_grained_per_generator(
                records,
                args.prediction_root,
                transform=transform,
                prediction_threshold=args.prediction_threshold,
            )
    write_csv(args.output_overall, overall_rows)
    if args.output_per_generator and per_generator_rows is not None:
        write_csv(args.output_per_generator, per_generator_rows)


def build_transform(args: argparse.Namespace) -> MaskTransform:
    resize_size = cli_width_height_to_shape(args.resize_size)
    center_crop = cli_width_height_to_shape(args.center_crop)
    return MaskTransform(
        resize_size=resize_size,
        resize_short_side=args.resize_short_side,
        center_crop=center_crop,
    )


def cli_width_height_to_shape(size: list[int] | None) -> tuple[int, int] | None:
    if size is None:
        return None
    width, height = size
    return height, width


def validate_cli_size(
    parser: argparse.ArgumentParser,
    size: list[int] | None,
    option_name: str,
) -> None:
    if size is not None and any(value <= 0 for value in size):
        parser.error(f"{option_name} values must be positive")


def write_csv(output_path: Path, rows: list[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No metric rows to write")
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
