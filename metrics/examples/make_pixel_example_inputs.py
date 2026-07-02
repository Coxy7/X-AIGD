from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from metrics.constants import CATEGORIES
from metrics.data import load_hf_records
from metrics.masks import build_gt_mask


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "metrics" / "examples" / "generated" / "pixel_inputs"
FALSE_POSITIVE_BOX_SIZE = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate X-AIGD demo pixel prediction masks.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset-repo", default="Coxy7/X-AIGD-demo")
    parser.add_argument("--split", default="labeled_train")
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def write_mask(mask_path: Path, mask) -> None:
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(mask_path), mask):
        raise OSError(f"Failed to write mask: {mask_path}")


def add_false_positive_region(mask: np.ndarray) -> np.ndarray:
    output_mask = mask.copy()
    box_size = min(FALSE_POSITIVE_BOX_SIZE, output_mask.shape[0], output_mask.shape[1])
    for y_min in range(0, output_mask.shape[0] - box_size + 1, box_size):
        for x_min in range(0, output_mask.shape[1] - box_size + 1, box_size):
            if not np.any(output_mask[y_min : y_min + box_size, x_min : x_min + box_size]):
                output_mask[y_min : y_min + box_size, x_min : x_min + box_size] = 255
                return output_mask
    raise ValueError("Could not find a background region for the false-positive example mask.")


def main() -> None:
    args = parse_args()
    records = load_hf_records(
        args.dataset_repo,
        args.split,
        local_files_only=not args.allow_download,
    )
    category_agnostic_root = args.output_root / "category_agnostic_predictions"
    fine_grained_root = args.output_root / "fine_grained_predictions"
    for record in records:
        category_agnostic_mask = add_false_positive_region(build_gt_mask(record, set(CATEGORIES)))
        write_mask(
            category_agnostic_root / record.generator / record.uid / "mask.png",
            category_agnostic_mask,
        )
        for category in CATEGORIES:
            fine_grained_mask = add_false_positive_region(build_gt_mask(record, {category}))
            write_mask(
                fine_grained_root / record.generator / category / record.uid / "mask.png",
                fine_grained_mask,
            )
    print(f"Successfully generated pixel prediction masks for {len(records)} images.")
    print(f"Outputs written to: {args.output_root}")
    print("\nNext step: Run the evaluation commands described in metrics/examples/README.md.")


if __name__ == "__main__":
    main()
