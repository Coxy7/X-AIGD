from __future__ import annotations

import unittest

from xaigd_viewer.navigation import next_index


class NavigationTests(unittest.TestCase):
    def test_wraps_within_filtered_subset(self) -> None:
        self.assertEqual(next_index(3, 1, [1, 3, 8]), 8)
        self.assertEqual(next_index(8, 1, [1, 3, 8]), 1)
        self.assertEqual(next_index(1, -1, [1, 3, 8]), 8)

    def test_recovers_when_current_index_is_outside_filtered_subset(self) -> None:
        self.assertEqual(next_index(4, 1, [1, 3, 8]), 8)
        self.assertEqual(next_index(4, -1, [1, 3, 8]), 3)

    def test_wraps_when_no_later_or_earlier_match_exists(self) -> None:
        self.assertEqual(next_index(10, 1, [1, 3, 8]), 1)
        self.assertEqual(next_index(0, -1, [1, 3, 8]), 8)

    def test_no_matching_indices_keeps_current_image(self) -> None:
        self.assertEqual(next_index(5, 1, []), 5)
        self.assertEqual(next_index(5, -1, []), 5)


if __name__ == "__main__":
    unittest.main()
