# X-AIGD Dataset Viewer

A lightweight Streamlit app for exploring [X-AIGD](https://github.com/Coxy7/X-AIGD) images, artifact categories, polygon annotations, and sample metadata.

## Setup

Requirements:
- Python 3.10+
- A Python environment managed with your preferred tool, such as `venv` or `conda`
- Internet access for loading dataset splits from Hugging Face

Install the viewer package in editable mode:
```bash
python -m pip install -e .
```

## Usage

Start the app from the viewer directory:

```bash
streamlit run app.py
```

By default, the viewer loads the development subset `Coxy7/X-AIGD-demo` and opens the `labeled_train` split.

### Data Sources

The sidebar lets you switch between:

| Viewer option | Hugging Face dataset |
| --- | --- |
| `X-AIGD-demo` | [![X-AIGD-demo](https://img.shields.io/badge/Hugging%20Face-X--AIGD--demo-yellow?logo=huggingface)](https://huggingface.co/datasets/Coxy7/X-AIGD-demo) |
| `X-AIGD` | [![X-AIGD](https://img.shields.io/badge/Hugging%20Face-X--AIGD-yellow?logo=huggingface)](https://huggingface.co/datasets/Coxy7/X-AIGD) |

Available splits:

| Split | Purpose |
| --- | --- |
| `labeled_train` | Labeled training split |
| `labeled_test` | Labeled test split |

The app fetches only the selected labeled parquet split from the latest Hugging Face dataset snapshot. Unlabeled splits are not downloaded.

### Supported Interactions

- Browse X-AIGD samples image by image.
- Inspect artifact polygon overlays on top of each image.
- Switch between the demo dataset and the full dataset.
- Filter globally by artifact category.
- Toggle visible categories on the current image.
- Jump to an absolute image index or a filtered matching-image index.
- Jump directly to a sample by generator and UID.
- View generator, UID, resolution, image index, filtered-match index, and visible-label count.

### Keyboard Shortcuts

| Key | Action |
| --- | --- |
| `Left` / `A` | Previous image |
| `Right` / `D` | Next image |
| `F` | Toggle fullscreen image view |
| `S` | Show or hide all labels on the current image |
| `0` | Clear category filter |
| `1`-`7` | Select artifact categories in spec order |

These navigation and filtering controls are also available from the sidebar and buttons under the image.

### Troubleshooting

- If dataset loading fails, check your internet connection and confirm that `datasets` and `huggingface_hub` are installed in the active environment.
- If Streamlit cannot start, confirm that `streamlit` is installed in the active environment.
- For quick development and testing, use `Coxy7/X-AIGD-demo` before loading the full dataset.

## Development

<details>
<summary>Show development notes</summary>

### Specification

See `doc/spec.md` for the current development reference, including supported data fields, viewer behavior, filtering semantics, navigation rules, and adaptation notes.

### Unit Testing

Install the optional test dependency:

```bash
python -m pip install -e ".[dev]"
```

Run tests with:

```bash
python -m pytest
```

### Adapting to Other Datasets

This viewer supports X-AIGD and X-AIGD-demo out of the box. Similar datasets may be supported by adapting the dataset-loading layer to produce the same internal sample format used by the viewer.

Potentially compatible annotation styles include:

- polygon-style object annotations with category labels
- image-level metadata used for filtering or navigation

### Repository Layout

When included in the X-AIGD repository, this viewer is intended to live under:

```text
tools/dataset_viewer/
```

Run commands from the viewer directory unless otherwise noted.

The standalone development repository is [Coxy7/xaigd-dataset-viewer](https://github.com/Coxy7/xaigd-dataset-viewer).

</details>
