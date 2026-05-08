from __future__ import annotations

from typing import List


def next_index(current_index: int, direction: int, filtered_indices: List[int]) -> int:
    if not filtered_indices:
        return current_index

    ordered_indices = sorted(filtered_indices)
    if current_index in ordered_indices:
        current_position = ordered_indices.index(current_index)
        next_position = (current_position + direction) % len(ordered_indices)
        return ordered_indices[next_position]

    if direction >= 0:
        for candidate in ordered_indices:
            if candidate > current_index:
                return candidate
        return ordered_indices[0]

    for candidate in reversed(ordered_indices):
        if candidate < current_index:
            return candidate
    return ordered_indices[-1]
