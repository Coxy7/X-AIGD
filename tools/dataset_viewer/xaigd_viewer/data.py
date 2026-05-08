from __future__ import annotations

from collections import OrderedDict
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PIL import Image

from xaigd_viewer.constants import ALL_CATEGORIES_OPTION, CATEGORY_ORDER
from xaigd_viewer.models import ArtifactLabel, ImageRecord, SplitData

ImageCacheKey = Tuple[str, str, int]


def normalize_label(raw_label: Dict[str, Any]) -> Optional[ArtifactLabel]:
    category = raw_label.get("label")
    raw_points = raw_label.get("points") or []
    if not isinstance(category, str):
        return None

    normalized_points = []
    for raw_point in raw_points:
        if not isinstance(raw_point, (list, tuple)) or len(raw_point) < 2:
            continue
        try:
            x_coord = float(raw_point[0])
            y_coord = float(raw_point[1])
        except (TypeError, ValueError):
            continue
        normalized_points.append((x_coord, y_coord))

    return ArtifactLabel(category=category, points=tuple(normalized_points))


def normalize_record(row: Dict[str, Any]) -> ImageRecord:
    image = row.get("image")
    width = row.get("width")
    height = row.get("height")
    if width is None or height is None:
        if hasattr(image, "size"):
            width, height = image.size
        else:
            raise ValueError("Image width/height missing from dataset row")

    labels = []
    for raw_label in row.get("labels") or []:
        if not isinstance(raw_label, dict):
            continue
        normalized_label = normalize_label(raw_label)
        if normalized_label is not None:
            labels.append(normalized_label)

    return ImageRecord(
        uid=str(row.get("uid", "")),
        generator=str(row.get("generator", "")),
        width=int(width),
        height=int(height),
        labels=tuple(labels),
    )


def build_matching_index_map(records: Iterable[ImageRecord]) -> Dict[str, Tuple[int, ...]]:
    record_list = list(records)
    matching = {ALL_CATEGORIES_OPTION: tuple(range(len(record_list)))}
    for category in CATEGORY_ORDER:
        matching[category] = tuple(matching_indices(record_list, category))
    return matching


def build_generator_uid_index(records: Iterable[ImageRecord]) -> Dict[Tuple[str, str], int]:
    index_by_generator_uid: Dict[Tuple[str, str], int] = {}
    for index, record in enumerate(records):
        index_by_generator_uid.setdefault((record.generator, record.uid), index)
    return index_by_generator_uid


def generator_options(records: Iterable[ImageRecord]) -> Tuple[str, ...]:
    seen_generators = set()
    ordered_generators = []
    for record in records:
        if record.generator in seen_generators:
            continue
        seen_generators.add(record.generator)
        ordered_generators.append(record.generator)
    return tuple(ordered_generators)


def load_split_data(dataset_repo: str, split: str) -> SplitData:
    from datasets import Image as HFImage
    from datasets import load_dataset
    from huggingface_hub import snapshot_download

    if split not in {"labeled_train", "labeled_test"}:
        raise ValueError(f"Unsupported split: {split}")

    snapshot_path = Path(
        snapshot_download(
            repo_id=dataset_repo,
            repo_type="dataset",
            allow_patterns=[f"data/{split}-*.parquet"],
        )
    )
    parquet_files = sorted(snapshot_path.glob(f"data/{split}-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files found for split {split!r} in dataset repo {dataset_repo!r}"
        )

    dataset = load_dataset(
        "parquet",
        data_files={split: [str(path) for path in parquet_files]},
        split=split,
    )
    dataset = dataset.cast_column("image", HFImage(decode=False))
    metadata_rows = dataset.remove_columns("image")
    records = tuple(normalize_record(row) for row in metadata_rows)
    return SplitData(
        dataset=dataset,
        records=records,
        matching_indices_by_category=build_matching_index_map(records),
        generator_options=generator_options(records),
        index_by_generator_uid=build_generator_uid_index(records),
    )


def load_image(split_data: SplitData, index: int) -> Image.Image:
    row = split_data.dataset[index]
    raw_image = row.get("image")

    if hasattr(raw_image, "load"):
        raw_image.load()
        return raw_image

    if not isinstance(raw_image, dict):
        raise TypeError(f"Unsupported image payload: {type(raw_image)!r}")

    image_bytes = raw_image.get("bytes")
    image_path = raw_image.get("path")
    image_source: BytesIO | str
    if image_bytes is not None:
        image_source = BytesIO(image_bytes)
    elif image_path:
        image_source = image_path
    else:
        raise ValueError("Image payload is missing both bytes and path")

    with Image.open(image_source) as decoded_image:
        decoded_image.load()
        return decoded_image.copy()


class ImageLRUCache:
    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("Image cache capacity must be positive")
        self.capacity = capacity
        self._cache: OrderedDict[ImageCacheKey, Image.Image] = OrderedDict()

    def __len__(self) -> int:
        return len(self._cache)

    def get(self, key: ImageCacheKey) -> Optional[Image.Image]:
        image = self._cache.get(key)
        if image is None:
            return None
        self._cache.move_to_end(key)
        return image

    def put(self, key: ImageCacheKey, image: Image.Image) -> None:
        self._cache[key] = image
        self._cache.move_to_end(key)
        if len(self._cache) > self.capacity:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        self._cache.clear()


def image_cache_key(dataset_repo: str, split: str, index: int) -> ImageCacheKey:
    return dataset_repo, split, index


def get_or_load_image(
    image_cache: ImageLRUCache,
    split_data: SplitData,
    dataset_repo: str,
    split: str,
    index: int,
) -> Image.Image:
    key = image_cache_key(dataset_repo, split, index)
    image = image_cache.get(key)
    if image is not None:
        return image

    image = load_image(split_data, index)
    image_cache.put(key, image)
    return image


def prefetch_neighbor_images(
    image_cache: ImageLRUCache,
    split_data: SplitData,
    dataset_repo: str,
    split: str,
    index: int,
    radius: int = 1,
) -> None:
    if radius < 1:
        return

    for offset in range(1, radius + 1):
        for neighbor_index in (index - offset, index + offset):
            if 0 <= neighbor_index < len(split_data.records):
                get_or_load_image(image_cache, split_data, dataset_repo, split, neighbor_index)


def filter_labels(record: ImageRecord, category: str) -> List[ArtifactLabel]:
    if category == ALL_CATEGORIES_OPTION:
        return list(record.labels)
    return [label for label in record.labels if label.category == category]


def matching_indices(records: Iterable[ImageRecord], category: str) -> List[int]:
    if category == ALL_CATEGORIES_OPTION:
        return [index for index, _ in enumerate(records)]

    matches = []
    for index, record in enumerate(records):
        if any(label.category == category for label in record.labels):
            matches.append(index)
    return matches
