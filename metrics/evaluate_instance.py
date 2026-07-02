from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from metrics.data import load_hf_records
from metrics.instance import (
    evaluate_instance_predictions,
    load_instance_predictions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate X-AIGD instance-level metrics.")
    parser.add_argument("--dataset-repo", default="Coxy7/X-AIGD")
    parser.add_argument("--split", default="labeled_test")
    parser.add_argument("--prediction-csv", required=True, type=Path)
    parser.add_argument("--output-per-generator", required=True, type=Path)
    parser.add_argument("--output-overall", required=True, type=Path)
    parser.add_argument("--overlap-threshold", default=0.5, type=float)
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
    predictions = load_instance_predictions(args.prediction_csv)
    per_generator_rows, overall_rows = evaluate_instance_predictions(
        records,
        predictions,
        overlap_threshold=args.overlap_threshold,
    )
    write_csv(args.output_per_generator, per_generator_rows)
    write_csv(args.output_overall, overall_rows)


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
