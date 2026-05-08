from __future__ import annotations

from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from PIL import Image

from xaigd_viewer.constants import ALL_CATEGORIES_OPTION
from xaigd_viewer.data import (
    ImageLRUCache,
    build_generator_uid_index,
    filter_labels,
    generator_options,
    get_or_load_image,
    image_cache_key,
    load_image,
    load_split_data,
    matching_indices,
    normalize_record,
    prefetch_neighbor_images,
)
from xaigd_viewer.models import SplitData


def png_bytes(color: str) -> bytes:
    image = Image.new("RGB", (12, 10), color)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class NoSizeImage:
    @property
    def size(self) -> tuple[int, int]:
        raise AssertionError("metadata normalization should not inspect image size when width/height exist")


class FakeDataset:
    def __init__(self, rows):
        self.rows = rows
        self.cast_column_calls = []
        self.remove_columns_calls = []

    def cast_column(self, name, feature):
        self.cast_column_calls.append((name, getattr(feature, "decode", None)))
        return self

    def remove_columns(self, *names):
        if len(names) == 1 and isinstance(names[0], str):
            names = (names[0],)
        self.remove_columns_calls.append(tuple(names))
        return FakeDataset(
            [{key: value for key, value in row.items() if key not in names} for row in self.rows]
        )

    def __iter__(self):
        return iter(self.rows)

    def __getitem__(self, index):
        return self.rows[index]

    def __len__(self):
        return len(self.rows)


class DataNormalizationTests(unittest.TestCase):
    def test_normalize_record_keeps_expected_fields(self) -> None:
        row = {
            "image": Image.new("RGB", (128, 96), "white"),
            "generator": "demo-gen",
            "uid": "sample-1",
            "width": 128,
            "height": 96,
            "labels": [
                {
                    "label": "low-level-edge_shape",
                    "points": [[1, 2], [20.5, 5], [9, 18]],
                },
                {
                    "label": "low-level-texture",
                    "points": [[4, 4], ["bad", 6], [7, 9]],
                },
            ],
        }

        record = normalize_record(row)

        self.assertEqual(record.uid, "sample-1")
        self.assertEqual(record.generator, "demo-gen")
        self.assertEqual((record.width, record.height), (128, 96))
        self.assertFalse(hasattr(record, "image"))
        self.assertEqual(len(record.labels), 2)
        self.assertEqual(record.labels[0].points[1], (20.5, 5.0))
        self.assertEqual(record.labels[1].points, ((4.0, 4.0), (7.0, 9.0)))

    def test_normalize_record_allows_empty_labels(self) -> None:
        row = {
            "image": Image.new("RGB", (32, 32), "white"),
            "generator": "demo-gen",
            "uid": "sample-2",
            "labels": [],
        }

        record = normalize_record(row)

        self.assertEqual(record.labels, ())
        self.assertEqual((record.width, record.height), (32, 32))

    def test_normalize_record_uses_width_and_height_without_touching_image(self) -> None:
        row = {
            "image": NoSizeImage(),
            "generator": "demo-gen",
            "uid": "sample-4",
            "width": 48,
            "height": 24,
            "labels": [],
        }

        record = normalize_record(row)

        self.assertEqual((record.width, record.height), (48, 24))


class FilterTests(unittest.TestCase):
    def test_filter_labels_returns_all_when_unfiltered(self) -> None:
        record = normalize_record(
            {
                "image": Image.new("RGB", (32, 32), "white"),
                "generator": "demo-gen",
                "uid": "sample-3",
                "labels": [
                    {"label": "low-level-edge_shape", "points": [[1, 1], [2, 2], [3, 3]]},
                    {"label": "low-level-color", "points": [[4, 4], [5, 5], [6, 6]]},
                ],
            }
        )

        labels = filter_labels(record, ALL_CATEGORIES_OPTION)

        self.assertEqual(len(labels), 2)

    def test_matching_indices_only_returns_records_with_selected_category(self) -> None:
        records = [
            normalize_record(
                {
                    "image": Image.new("RGB", (16, 16), "white"),
                    "generator": "g1",
                    "uid": "1",
                    "labels": [{"label": "low-level-edge_shape", "points": [[1, 1], [2, 1], [2, 2]]}],
                }
            ),
            normalize_record(
                {
                    "image": Image.new("RGB", (16, 16), "white"),
                    "generator": "g2",
                    "uid": "2",
                    "labels": [{"label": "high-level-semantics", "points": [[1, 1], [2, 1], [2, 2]]}],
                }
            ),
            normalize_record(
                {
                    "image": Image.new("RGB", (16, 16), "white"),
                    "generator": "g3",
                    "uid": "3",
                    "labels": [],
                }
            ),
        ]

        labels = filter_labels(records[1], "high-level-semantics")
        matches = matching_indices(records, "high-level-semantics")

        self.assertEqual(len(labels), 1)
        self.assertEqual(matches, [1])

    def test_generator_uid_index_and_generator_options_are_ordered(self) -> None:
        records = [
            normalize_record(
                {
                    "image": Image.new("RGB", (16, 16), "white"),
                    "generator": "g1",
                    "uid": "u1",
                    "labels": [],
                }
            ),
            normalize_record(
                {
                    "image": Image.new("RGB", (16, 16), "white"),
                    "generator": "g2",
                    "uid": "u1",
                    "labels": [],
                }
            ),
            normalize_record(
                {
                    "image": Image.new("RGB", (16, 16), "white"),
                    "generator": "g1",
                    "uid": "u2",
                    "labels": [],
                }
            ),
        ]

        self.assertEqual(generator_options(records), ("g1", "g2"))
        self.assertEqual(
            build_generator_uid_index(records),
            {
                ("g1", "u1"): 0,
                ("g2", "u1"): 1,
                ("g1", "u2"): 2,
            },
        )


class SplitDataTests(unittest.TestCase):
    @mock.patch("huggingface_hub.snapshot_download")
    @mock.patch("datasets.load_dataset")
    def test_load_split_data_builds_metadata_only_records(
        self,
        load_dataset_mock,
        snapshot_download_mock,
    ) -> None:
        fake_dataset = FakeDataset(
            [
                {
                    "image": NoSizeImage(),
                    "generator": "g1",
                    "uid": "1",
                    "width": 32,
                    "height": 16,
                    "labels": [{"label": "low-level-edge_shape", "points": [[1, 1], [2, 1], [2, 2]]}],
                },
                {
                    "image": NoSizeImage(),
                    "generator": "g2",
                    "uid": "2",
                    "width": 64,
                    "height": 32,
                    "labels": [{"label": "high-level-semantics", "points": [[1, 1], [2, 1], [2, 2]]}],
                },
            ]
        )
        load_dataset_mock.return_value = fake_dataset
        with tempfile.TemporaryDirectory() as tmpdir:
            parquet_path = Path(tmpdir) / "data" / "labeled_test-00000-of-00001.parquet"
            parquet_path.parent.mkdir(parents=True, exist_ok=True)
            parquet_path.touch()
            snapshot_download_mock.return_value = tmpdir

            split_data = load_split_data("Coxy7/X-AIGD-demo", "labeled_test")

            self.assertIsInstance(split_data, SplitData)
            snapshot_download_mock.assert_called_once_with(
                repo_id="Coxy7/X-AIGD-demo",
                repo_type="dataset",
                allow_patterns=["data/labeled_test-*.parquet"],
            )
            load_dataset_mock.assert_called_once_with(
                "parquet",
                data_files={"labeled_test": [str(parquet_path)]},
                split="labeled_test",
            )
            self.assertEqual(fake_dataset.cast_column_calls, [("image", False)])
            self.assertEqual(fake_dataset.remove_columns_calls, [("image",)])
            self.assertEqual(len(split_data.records), 2)
            self.assertEqual(split_data.records[0].uid, "1")
            self.assertEqual(split_data.matching_indices_by_category[ALL_CATEGORIES_OPTION], (0, 1))
            self.assertEqual(split_data.matching_indices_by_category["high-level-semantics"], (1,))
            self.assertEqual(split_data.generator_options, ("g1", "g2"))
            self.assertEqual(split_data.index_by_generator_uid[("g2", "2")], 1)

    @mock.patch("huggingface_hub.snapshot_download")
    def test_load_split_data_raises_when_snapshot_has_no_matching_parquet(self, snapshot_download_mock) -> None:
        snapshot_download_mock.return_value = "/tmp/hf-cache/snapshots/revision-1"

        with self.assertRaisesRegex(FileNotFoundError, "labeled_test"):
            load_split_data("Coxy7/X-AIGD-demo", "labeled_test")

    def test_load_image_decodes_single_image_from_bytes(self) -> None:
        split_data = SplitData(
            dataset=FakeDataset(
                [
                    {"image": {"bytes": png_bytes("red"), "path": None}},
                ]
            ),
            records=(),
            matching_indices_by_category={},
            generator_options=(),
            index_by_generator_uid={},
        )

        image = load_image(split_data, 0)

        self.assertEqual(image.size, (12, 10))
        self.assertEqual(image.getpixel((0, 0)), (255, 0, 0))


class ImageCacheTests(unittest.TestCase):
    def test_lru_cache_returns_same_object_and_evicts_least_recently_used(self) -> None:
        cache = ImageLRUCache(capacity=2)
        red = Image.new("RGB", (4, 4), "red")
        blue = Image.new("RGB", (4, 4), "blue")
        green = Image.new("RGB", (4, 4), "green")

        first_key = image_cache_key("repo", "labeled_train", 0)
        second_key = image_cache_key("repo", "labeled_train", 1)
        third_key = image_cache_key("repo", "labeled_train", 2)

        cache.put(first_key, red)
        cache.put(second_key, blue)

        self.assertIs(cache.get(first_key), red)

        cache.put(third_key, green)

        self.assertIsNone(cache.get(second_key))
        self.assertIs(cache.get(first_key), red)
        self.assertIs(cache.get(third_key), green)

    def test_cache_keys_distinguish_splits(self) -> None:
        cache = ImageLRUCache(capacity=4)
        train_key = image_cache_key("repo", "labeled_train", 1)
        test_key = image_cache_key("repo", "labeled_test", 1)
        train_image = Image.new("RGB", (4, 4), "red")
        test_image = Image.new("RGB", (4, 4), "blue")

        cache.put(train_key, train_image)
        cache.put(test_key, test_image)

        self.assertIs(cache.get(train_key), train_image)
        self.assertIs(cache.get(test_key), test_image)

    @mock.patch("xaigd_viewer.data.load_image")
    def test_get_or_load_image_uses_cache(self, load_image_mock) -> None:
        cache = ImageLRUCache(capacity=2)
        image = Image.new("RGB", (4, 4), "red")
        load_image_mock.return_value = image
        split_data = SplitData(
            dataset=FakeDataset([]),
            records=(),
            matching_indices_by_category={},
            generator_options=(),
            index_by_generator_uid={},
        )

        loaded = get_or_load_image(cache, split_data, "repo", "labeled_test", 3)
        cached = get_or_load_image(cache, split_data, "repo", "labeled_test", 3)

        self.assertIs(loaded, image)
        self.assertIs(cached, image)
        load_image_mock.assert_called_once_with(split_data, 3)

    @mock.patch("xaigd_viewer.data.load_image")
    def test_prefetch_neighbor_images_warms_adjacent_indices(self, load_image_mock) -> None:
        load_image_mock.side_effect = [
            Image.new("RGB", (4, 4), "red"),
            Image.new("RGB", (4, 4), "blue"),
        ]
        cache = ImageLRUCache(capacity=4)
        split_data = SplitData(
            dataset=FakeDataset([]),
            records=(mock.sentinel.r0, mock.sentinel.r1, mock.sentinel.r2),
            matching_indices_by_category={},
            generator_options=(),
            index_by_generator_uid={},
        )

        prefetch_neighbor_images(cache, split_data, "repo", "labeled_test", 1)

        self.assertEqual(load_image_mock.call_args_list, [mock.call(split_data, 0), mock.call(split_data, 2)])
        self.assertEqual(len(cache), 2)

    @mock.patch("xaigd_viewer.data.load_image")
    def test_prefetch_neighbor_images_skips_out_of_bounds_indices(self, load_image_mock) -> None:
        load_image_mock.return_value = Image.new("RGB", (4, 4), "red")
        cache = ImageLRUCache(capacity=4)
        split_data = SplitData(
            dataset=FakeDataset([]),
            records=(mock.sentinel.r0, mock.sentinel.r1),
            matching_indices_by_category={},
            generator_options=(),
            index_by_generator_uid={},
        )

        prefetch_neighbor_images(cache, split_data, "repo", "labeled_test", 0)

        load_image_mock.assert_called_once_with(split_data, 1)


if __name__ == "__main__":
    unittest.main()
