# X-AIGD Metric Evaluation

This directory contains metric code for the X-AIGD dataset.

## Data Source

Ground-truth labels are read directly from the Hugging Face dataset rows:

- `image` is not decoded for metric computation.
- `generator`, `uid`, `width`, `height`, and `labels` are read from the split
  Parquet files.
- `labels` must use the supported category names:
  - `low-level-edge_shape`
  - `low-level-texture`
  - `low-level-color`
  - `low-level-symbol`
  - `high-level-semantics`
  - `cognitive-level-commonsense`
  - `cognitive-level-physics`

Unknown categories are treated as errors.

Ground-truth masks are generated in memory. Polygon points are converted with `np.array(points, dtype=np.int32)`, then rasterized with `cv2.fillPoly(..., 255)`.  This means fractional coordinates are truncated before rasterization. Malformed polygons with fewer than three points are skipped and counted in the output diagnostics.

## Pixel-Level Metrics

Run category-agnostic pixel evaluation:

```bash
python metrics/evaluate_pixel.py \
  --dataset-repo Coxy7/X-AIGD \
  --split labeled_test \
  --task category-agnostic \
  --prediction-root /path/to/category-agnostic-predictions \
  --transform resize256-crop224 \
  --output /tmp/xaigd-category-agnostic.csv
```

Run fine-grained pixel evaluation:

```bash
python metrics/evaluate_pixel.py \
  --dataset-repo Coxy7/X-AIGD \
  --split labeled_test \
  --task fine-grained \
  --prediction-root /path/to/fine-grained-predictions \
  --transform keep-original-size \
  --output /tmp/xaigd-fine-grained.csv
```

Category-agnostic prediction layout:

```text
prediction_root/
  {generator}/
    {uid}/
      *.png
```

Fine-grained prediction layout:

```text
prediction_root/
  {generator}/
    {category}/
      {uid}/
        *.png
```

All PNG masks in an image directory are binarized with `mask > 127` and merged with a pixelwise OR. Missing directories or empty directories mean an all-zero prediction. Prediction masks with unexpected dimensions are errors.

Supported transforms:

- `keep-original-size`: compare at the original image size.
- `resize256-crop224`: resize the ground-truth mask to 256x256, then center
  crop 224x224.
- `resize518-crop518`: resize shortest side to 518, then center crop 518x518.

Masks are resized with nearest-neighbor interpolation. This keeps binary labels binary. Linear interpolation creates gray boundary pixels between 0 and 255, which changes foreground/background counts after thresholding and can silently drop pixels if exact 0/255 comparisons are used.

Pixel output columns include raw counts and percentages:

```text
task, category, TP_0, FP_0, FN_0, TP_255, FP_255, FN_255,
iou_255, precision_255, recall_255, f1_255, mIoU,
evaluated_images, zero_prediction_images, skipped_malformed_polygons
```

## Instance-Level Metrics

Run instance-level evaluation:

```bash
python metrics/evaluate_instance.py \
  --dataset-repo Coxy7/X-AIGD \
  --split labeled_test \
  --prediction-csv /path/to/predicted-boxes.csv \
  --output-per-generator /tmp/xaigd-instance-per-generator.csv \
  --output-overall /tmp/xaigd-instance-overall.csv
```

Prediction CSV schema:

```text
generator,uid,category,x_min,y_min,x_max,y_max
```

Each row is one predicted box. Missing rows for an image/category mean zero predicted instances. Unknown categories are errors. Boxes outside image bounds are counted as invalid unmatched predictions.

Ground-truth instances are built by unioning category masks, dilating once with a 5x5 rectangular kernel, and converting the result to connected components. A prediction matches a ground-truth instance when:

```text
intersection_area / predicted_box_area > overlap_threshold
```

The default overlap threshold is `0.5` and can be changed with
`--overlap-threshold`. This is not IoU.

Instance output includes per-generator rows and overall rows. Overall rows use
`all` in the `generator` column. Both files contain:

```text
generator, category, Precision, Recall, F1, TP, FP, FN, evaluated_images, zero_prediction_images, invalid_prediction_boxes, skipped_malformed_polygons
```
