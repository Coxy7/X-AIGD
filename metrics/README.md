# X-AIGD Metric Evaluation

This directory contains metric evaluation code for the X-AIGD dataset.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Data Source](#data-source)
  - [Ground-Truth Labels](#ground-truth-labels)
  - [Ground-Truth Masks](#ground-truth-masks)
- [Examples](#examples)
- [Pixel-Level Metrics](#pixel-level-metrics)
  - [Usage Examples](#usage-examples)
  - [Prediction Layout Requirements](#prediction-layout-requirements)
  - [Image Preprocessing and Mask Handling](#image-preprocessing-and-mask-handling)
  - [Evaluation Output Format](#evaluation-output-format)
  - [Metrics Explanation](#metrics-explanation)
- [Instance-Level Metrics](#instance-level-metrics)
  - [Usage Example](#usage-example)
  - [Prediction CSV Schema](#prediction-csv-schema)
  - [Ground-Truth Instance Generation](#ground-truth-instance-generation)
  - [Matching Strategy](#matching-strategy)
  - [Evaluation Output Format](#evaluation-output-format-1)
  - [Metrics Explanation](#metrics-explanation-1)

## Prerequisites

The evaluation scripts require the following Python libraries:

- `opencv-python`
- `numpy`
- `pyarrow`
- `huggingface-hub`

## Data Source

Ground-truth labels are read directly from the Hugging Face dataset rows. Note that the `image` itself is not decoded for metric computation.

### Ground-Truth Labels

The attributes `generator`, `uid`, `width`, `height`, and `labels` are read from the split Parquet files. The `labels` attribute must use one of the supported category names:

- `low-level-edge_shape`
- `low-level-texture`
- `low-level-color`
- `low-level-symbol`
- `high-level-semantics`
- `cognitive-level-commonsense`
- `cognitive-level-physics`

*Unknown categories are treated as errors.*

### Ground-Truth Masks

Ground-truth masks are generated in memory during evaluation:
1. Polygon points are discretized to valid pixel indices by converting them with `np.array(points, dtype=np.int32)`, which truncates fractional coordinates, and then clamping boundary coordinates such as `x == width` or `y == height` to the last valid pixel index.
2. The polygons are rasterized with `cv2.fillPoly(..., 255)`.
3. Polygons with fewer than three points are treated as annotation errors.

## Examples

See `metrics/examples/README.md` for runnable category-agnostic pixel, fine-grained pixel, and instance-level examples using the `Coxy7/X-AIGD-demo` dataset.

## Pixel-Level Metrics

### Usage Examples

**Category-Agnostic Pixel Evaluation:**

```bash
python metrics/evaluate_pixel.py \
  --dataset-repo Coxy7/X-AIGD \
  --split labeled_test \
  --task category-agnostic \
  --prediction-root /path/to/category-agnostic-predictions \
  --resize-size 256 256 \
  --center-crop 224 224 \
  --output-overall /tmp/xaigd-category-agnostic-overall.csv \
  --output-per-generator /tmp/xaigd-category-agnostic-per-generator.csv
```

**Fine-Grained Pixel Evaluation:**

```bash
python metrics/evaluate_pixel.py \
  --dataset-repo Coxy7/X-AIGD \
  --split labeled_test \
  --task fine-grained \
  --prediction-root /path/to/fine-grained-predictions \
  --output-overall /tmp/xaigd-fine-grained-overall.csv \
  --output-per-generator /tmp/xaigd-fine-grained-per-generator.csv
```

*Note: `--output-overall` is required, while `--output-per-generator` is optional.*

### Prediction Layout Requirements

Missing directories, empty directories, and directories whose PNG masks union to an all-zero mask are interpreted as zero predictions. Prediction masks with unexpected dimensions will result in errors.

**Category-Agnostic Layout:**

```text
prediction_root/
  {generator}/
    {uid}/
      *.png
```

**Fine-Grained Layout:**

```text
prediction_root/
  {generator}/
    {category}/
      {uid}/
        *.png
```

### Image Preprocessing and Mask Handling

All PNG masks in an image directory are binarized using `mask > 127` by default and merged with a pixel-wise OR operation. Change the binarization threshold with `--prediction-threshold`.

**Mask Resizing Options:**

- By default, masks are compared at their original image size.
- `--resize-size WIDTH HEIGHT`: Resizes the ground-truth mask to an explicit size before comparison.
- `--resize-short-side PIXELS`: Resizes the ground-truth mask so its shortest side has the requested length while preserving the aspect ratio.
- `--center-crop WIDTH HEIGHT`: Center-crops the ground-truth mask after any resizing step.

*Note: `--resize-size` and `--resize-short-side` are mutually exclusive.*

Masks are resized using nearest-neighbor interpolation. This approach keeps binary labels binary.

### Evaluation Output Format

Pixel-level evaluation reports artifact-region metrics as fractions in the range `[0, 1]`. 

The CSV output structure is as follows:

```text
generator, task, category, IoU, PixP, PixR, PixF1, evaluated_images, zero_prediction_images
```

- **Overall Rows:** The `generator` column is set to `all`.
- **Category-Agnostic:** The `category` column is set to `all`.
- **Fine-Grained:** Contains one row per category.

### Metrics Explanation

For pixel-level metrics, True Positives (`TP`), False Positives (`FP`), and False Negatives (`FN`) are counted over individual pixels:

- **`TP`**: Pixels predicted as artifact and labeled as artifact.
- **`FP`**: Pixels predicted as artifact but labeled as background.
- **`FN`**: Pixels predicted as background but labeled as artifact.

The metrics calculated are:

- **`IoU` (Intersection over Union)**: `TP / (TP + FP + FN)`
- **`PixP` (Pixel-level Precision)**: `TP / (TP + FP)`
- **`PixR` (Pixel-level Recall)**: `TP / (TP + FN)`
- **`PixF1` (Pixel-level F1-score)**: `2 * PixP * PixR / (PixP + PixR)`

---

## Instance-Level Metrics

### Usage Example

```bash
python metrics/evaluate_instance.py \
  --dataset-repo Coxy7/X-AIGD \
  --split labeled_test \
  --prediction-csv /path/to/predicted-boxes.csv \
  --output-overall /tmp/xaigd-instance-overall.csv \
  --output-per-generator /tmp/xaigd-instance-per-generator.csv
```

*Note: `--output-overall` is required, while `--output-per-generator` is optional.*

### Prediction CSV Schema

```text
generator,uid,category,x_min,y_min,x_max,y_max
```

- Each row represents one predicted bounding box.
- Missing rows for an image/category indicate zero predicted instances.
- Unknown categories will result in errors.
- Boxes falling outside the image bounds are counted as invalid, unmatched predictions.

### Ground-Truth Instance Generation

For each image and category, ground-truth instances are generated by:
1. Unioning the category masks.
2. Dilating the union once using a `5x5` rectangular kernel.
3. Converting the dilated mask to connected components.
4. Intersecting each component with the original, undilated category mask before measuring box overlap.

### Matching Strategy

A prediction is considered to match a ground-truth instance when:

```text
intersection_area / predicted_box_area > overlap_threshold
```

- **Threshold**: The default overlap threshold is `0.5`, which can be adjusted via `--overlap-threshold`. *Note that this is not IoU.*
- **Matching Process**: The matching is flag-based rather than a strict one-to-one assignment. The evaluator maintains a matched/unmatched flag for each predicted box and each ground-truth instance.
  - If a predicted box meets the overlap rule for a ground-truth instance, both flags are marked as matched.
  - Multiple predicted boxes can match the same ground-truth instance.
  - A single predicted box can match multiple ground-truth instances if it satisfies the overlap rule for each.

### Evaluation Output Format

Instance-level evaluation output includes overall rows and optionally per-generator rows.

Both output files contain metric columns named according to the selected overlap threshold (e.g., `@0.5` for the default `0.5` threshold). The CSV output structure is as follows:

```text
generator, category, P@0.5, R@0.5, F1@0.5, TP, FP, FN, evaluated_images, zero_prediction_images, invalid_prediction_boxes
```

- **Overall Rows:** The `generator` column is set to `all`.

### Metrics Explanation

For instance-level metrics, True Positives (`TP`), False Positives (`FP`), and False Negatives (`FN`) are counted based on matched and unmatched flags:

- **`TP`**: Number of matched ground-truth instances.
- **`FP`**: Number of unmatched predicted boxes. (Invalid boxes remain unmatched and contribute to `FP`).
- **`FN`**: Number of unmatched ground-truth instances.

Let `t` denote the overlap threshold (e.g., `0.5`). `P@t`, `R@t`, and `F1@t` are reported as fractions in the range `[0, 1]` and computed from the accumulated counts:

- **`P@t` (Precision)**: `TP / (TP + FP)`
- **`R@t` (Recall)**: `TP / (TP + FN)`
- **`F1@t` (F1-score)**: `2 * P@t * R@t / (P@t + R@t)`
