from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from metrics.constants import TRANSFORMS
from metrics.data import load_hf_records
from metrics.pixel import evaluate_category_agnostic, evaluate_fine_grained


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate X-AIGD pixel-level metrics.")
    parser.add_argument("--dataset-repo", default="Coxy7/X-AIGD")
    parser.add_argument("--split", default="labeled_test")
    parser.add_argument("--prediction-root", required=True, type=Path)
    parser.add_argument("--task", choices=["category-agnostic", "fine-grained"], required=True)
    parser.add_argument("--transform", choices=TRANSFORMS, required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prediction-threshold", default=127, type=int)
    parser.add_argument("--revision")
    parser.add_argument("--cache-dir")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_hf_records(
        args.dataset_repo,
        args.split,
        revision=args.revision,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    if args.task == "category-agnostic":
        rows = [
            evaluate_category_agnostic(
                records,
                args.prediction_root,
                transform=args.transform,
                prediction_threshold=args.prediction_threshold,
            )
        ]
    else:
        rows = evaluate_fine_grained(
            records,
            args.prediction_root,
            transform=args.transform,
            prediction_threshold=args.prediction_threshold,
        )
    write_csv(args.output, rows)


def write_csv(output_path: Path, rows: list[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No metric rows to write")
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
