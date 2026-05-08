from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

from xaigd_viewer.data import ImageLRUCache


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
APP_SPEC = importlib.util.spec_from_file_location("dataset_viewer_app", APP_PATH)
assert APP_SPEC is not None and APP_SPEC.loader is not None
app = importlib.util.module_from_spec(APP_SPEC)
APP_SPEC.loader.exec_module(app)


class AppStateTests(unittest.TestCase):
    def test_next_overlay_categories_hides_all_when_all_available_are_selected(self) -> None:
        next_categories = app.next_overlay_categories(
            ["low-level-edge_shape", "low-level-texture"],
            ["low-level-edge_shape", "low-level-texture"],
        )

        self.assertEqual(next_categories, [])

    def test_next_overlay_categories_shows_all_when_subset_is_selected(self) -> None:
        next_categories = app.next_overlay_categories(
            ["low-level-edge_shape"],
            ["low-level-edge_shape", "low-level-texture"],
        )

        self.assertEqual(
            next_categories,
            ["low-level-edge_shape", "low-level-texture"],
        )

    def test_apply_source_change_resets_navigation_and_image_cache(self) -> None:
        original_cache = ImageLRUCache(capacity=app.IMAGE_CACHE_CAPACITY)
        original_cache.put(("repo", "labeled_train", 0), object())
        state = {
            "source_key": "Coxy7/X-AIGD-demo:labeled_train",
            "current_index": 7,
            "image_cache": original_cache,
        }

        app.apply_source_change(state, "Coxy7/X-AIGD-demo:labeled_test")

        self.assertEqual(state["source_key"], "Coxy7/X-AIGD-demo:labeled_test")
        self.assertEqual(state["current_index"], 0)
        self.assertIsInstance(state["image_cache"], ImageLRUCache)
        self.assertIsNot(state["image_cache"], original_cache)
        self.assertEqual(len(state["image_cache"]), 0)

    def test_apply_source_change_keeps_existing_cache_when_source_unchanged(self) -> None:
        cache = ImageLRUCache(capacity=app.IMAGE_CACHE_CAPACITY)
        state = {
            "source_key": "Coxy7/X-AIGD-demo:labeled_train",
            "current_index": 4,
            "image_cache": cache,
        }

        app.apply_source_change(state, "Coxy7/X-AIGD-demo:labeled_train")

        self.assertEqual(state["current_index"], 4)
        self.assertIs(state["image_cache"], cache)

    def test_jump_to_record_returns_matching_index_for_generator_uid(self) -> None:
        split_data = app.SplitData(
            dataset=None,
            records=(),
            matching_indices_by_category={},
            generator_options=("g1", "g2"),
            index_by_generator_uid={("g1", "u1"): 3},
        )
        app.st.session_state.jump_record_generator = "g1"
        app.st.session_state.jump_record_uid = " u1 "
        app.st.session_state.source_key = "repo:labeled_train"

        error_message = app.jump_to_record(split_data)

        self.assertIsNone(error_message)
        self.assertEqual(app.st.session_state.current_index, 3)
        self.assertEqual(app.st.session_state.jump_record_uid, " u1 ")

    def test_jump_to_record_returns_error_when_generator_uid_is_missing(self) -> None:
        split_data = app.SplitData(
            dataset=None,
            records=(),
            matching_indices_by_category={},
            generator_options=("g1",),
            index_by_generator_uid={},
        )
        app.st.session_state.jump_record_generator = "g1"
        app.st.session_state.jump_record_uid = "missing"

        error_message = app.jump_to_record(split_data)

        self.assertEqual(error_message, "No image found for generator 'g1' with UID 'missing'.")


if __name__ == "__main__":
    unittest.main()
