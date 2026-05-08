from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

Point = Tuple[float, float]


@dataclass(frozen=True)
class ArtifactLabel:
    category: str
    points: Tuple[Point, ...]


@dataclass(frozen=True)
class ImageRecord:
    uid: str
    generator: str
    width: int
    height: int
    labels: Tuple[ArtifactLabel, ...]


@dataclass(frozen=True)
class SplitData:
    dataset: Any
    records: Tuple[ImageRecord, ...]
    matching_indices_by_category: Dict[str, Tuple[int, ...]]
    generator_options: Tuple[str, ...]
    index_by_generator_uid: Dict[Tuple[str, str], int]
