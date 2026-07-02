# X-AIGD Evaluation Examples

This directory contains small, reproducible examples for the three evaluation tasks using the cached `Coxy7/X-AIGD-demo` dataset split `labeled_train`. Follow these steps to verify your setup and understand the formatting of evaluation inputs and outputs.

---

## Directory Overview

- **`instance_predictions.csv`**: A sample input CSV containing bounding box predictions for the demo dataset.
- **`make_pixel_example_inputs.py`**: A helper script that generates pixel-level prediction masks (PNGs) with true-positive regions and false-positive regions.
- **`expected_*.csv`**: Reference files containing the exact output you should get when running the evaluation commands.
- **`generated/`**: The output directory where your evaluation runs will write files.

---

## Prerequisites

Ensure that you are running all commands from the repository root directory. For the required Python packages, please refer to the Prerequisites section of [metrics/README.md](../README.md).

---

## Preparation for Pixel-Level Evaluations

Since PNG prediction masks are binary files, they are not committed to Git. Before running either of the pixel-level evaluation options, generate the sample prediction masks locally by running:

```bash
python metrics/examples/make_pixel_example_inputs.py
```

*Note: The script uses locally cached dataset files by default. If you need to download them, pass the `--allow-download` flag.*

---

## Evaluation Options

The three evaluation tasks are independent and suit different kinds of models. Choose the option that matches your model's outputs.

### Option A: Category-Agnostic Pixel Evaluation

Use this mode if your model detects artifact regions without predicting their specific fine-grained categories.

Run the category-agnostic evaluation script:
```bash
python metrics/evaluate_pixel.py \
  --dataset-repo Coxy7/X-AIGD-demo \
  --split labeled_train \
  --task category-agnostic \
  --prediction-root metrics/examples/generated/pixel_inputs/category_agnostic_predictions \
  --transform keep-original-size \
  --local-files-only \
  --output metrics/examples/generated/category_agnostic.csv
```
Verify that the output matches [expected_category_agnostic.csv](expected_category_agnostic.csv).

---

### Option B: Fine-Grained Pixel Evaluation

Use this mode if your model predicts pixel-level masks for each of the 7 fine-grained artifact categories.

Run the fine-grained evaluation script:
```bash
python metrics/evaluate_pixel.py \
  --dataset-repo Coxy7/X-AIGD-demo \
  --split labeled_train \
  --task fine-grained \
  --prediction-root metrics/examples/generated/pixel_inputs/fine_grained_predictions \
  --transform keep-original-size \
  --local-files-only \
  --output metrics/examples/generated/fine_grained.csv
```
Verify that the output matches [expected_fine_grained.csv](expected_fine_grained.csv).

---

### Option C: Instance-Level Evaluation

Use this mode if your model predicts individual bounding boxes with category labels. This task does not require generating pixel prediction masks first.

Run the instance-level evaluation script:
```bash
python metrics/evaluate_instance.py \
  --dataset-repo Coxy7/X-AIGD-demo \
  --split labeled_train \
  --prediction-csv metrics/examples/instance_predictions.csv \
  --output-per-generator metrics/examples/generated/instance_per_generator.csv \
  --output-overall metrics/examples/generated/instance_overall.csv \
  --local-files-only
```
Verify that the outputs match [expected_instance_overall.csv](expected_instance_overall.csv) and [expected_instance_per_generator.csv](expected_instance_per_generator.csv).
