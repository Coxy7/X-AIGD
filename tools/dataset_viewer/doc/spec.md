# X-AIGD Dataset Viewer Specification

This document describes the current behavior and development assumptions for the X-AIGD Dataset Viewer. It is a development reference, not user-facing documentation.

## Scope

- The viewer is a lightweight Streamlit app for exploring X-AIGD images, artifact annotations, and sample metadata.
- The implementation is currently tailored to X-AIGD and X-AIGD-demo.
- The code should remain small enough to adapt to related detection or segmentation-style datasets, but general-dataset support is not currently a product requirement.

## Supported Data

### Dataset Repositories

- `X-AIGD-demo` maps to `Coxy7/X-AIGD-demo`.
- `X-AIGD` maps to `Coxy7/X-AIGD`.

### Splits

- `labeled_train`
- `labeled_test`

Only labeled parquet split files are loaded. Unlabeled splits are not downloaded.

### Required Fields

- `image`: AI-generated image stored in the Hugging Face dataset.
- `generator`: Name of the text-to-image generator.
- `uid`: Unique image identifier. Different generators may share the same UID.
- `labels`: List of human-annotated artifacts.
- `width`, `height`: Image resolution. If either is missing, the viewer may infer dimensions from the image when possible.

Each label item should contain:

- `label`: Artifact category.
- `points`: Polygon coordinates in image space, formatted as `[[x1, y1], [x2, y2], ...]`.

## Artifact Categories

The supported X-AIGD artifact categories are:

| Category value | Display label |
| --- | --- |
| `low-level-edge_shape` | Edge & Shape |
| `low-level-texture` | Texture |
| `low-level-color` | Color |
| `low-level-symbol` | Symbol |
| `high-level-semantics` | Semantics |
| `cognitive-level-commonsense` | Commonsense |
| `cognitive-level-physics` | Physics |

The global category filter also includes `All`.

## Data Loading

- The selected split is downloaded through `huggingface_hub.snapshot_download`.
- The snapshot download is restricted to `data/{split}-*.parquet`.
- The parquet files are loaded with `datasets.load_dataset`.
- The image column is cast with `datasets.Image(decode=False)` so metadata can be loaded without decoding all images eagerly.
- Images are decoded lazily when they are displayed.
- A small LRU cache stores recently decoded images.
- Neighboring images may be prefetched to make next/previous navigation smoother.

## Viewer Behavior

### Main View

- Display one image at a time.
- Draw artifact polygons as colored outlines.
- Do not fill polygons.
- Show an overlay legend above the image.
- Show metadata below the image:
  - generator
  - UID
  - resolution
  - absolute image index
  - filtered-match index
  - visible-label count

### Dataset and Split Selection

- The sidebar provides dataset and split selectors.
- Changing dataset or split resets navigation to the first image and clears the image cache.

### Global Category Filtering

- `All` shows all records.
- Selecting a category limits next/previous navigation to records containing that category.
- Changing the category filter should not force the current image to jump away immediately.
- If the current image has no label for the selected category, the image remains visible and the app warns that the selected category is not present on the current image.
- If no image in the selected split contains the selected category, the app warns the user.

### Per-Image Overlay Selection

- The legend controls which categories are visible on the current image.
- Categories unavailable on the current image are disabled or visually dimmed.
- Per-image overlay selection is scoped to the current dataset, split, image, and global category filter.
- When the scope changes, visible overlay categories reset to categories available for the current image and filter.

### Navigation

- Previous/next buttons move through the active index list.
- The active index list is either all records or records matching the selected global category.
- Navigation wraps around the active index list.
- The info panel supports jumping to:
  - absolute image index
  - filtered matching-image index
- The sidebar supports jumping to a record by generator and UID.

### Keyboard Shortcuts

| Key | Action |
| --- | --- |
| `Left` / `A` | Previous image |
| `Right` / `D` | Next image |
| `F` | Toggle fullscreen image view |
| `S` | Show or hide all labels on the current image |
| `0` | Clear category filter |
| `1`-`7` | Select artifact categories in spec order |

## Adaptation Notes

Future dataset support should preferably be added by isolating dataset-specific loading and normalization logic. A compatible adapter should produce the same internal concepts:

- ordered image records
- generator and UID metadata when available
- image dimensions
- category labels
- polygon or overlay geometry
- per-category matching index lists

Potentially related annotation styles include polygon object annotations and segmentation masks that can be converted or rendered as overlays.
