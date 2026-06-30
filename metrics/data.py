from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pyarrow.parquet as pq

from metrics.constants import CATEGORY_SET


@dataclass(frozen=True)
class ArtifactLabel:
    category: str
    points: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class ImageRecord:
    generator: str
    uid: str
    width: int
    height: int
    labels: tuple[ArtifactLabel, ...]


def normalize_label(raw_label: dict) -> ArtifactLabel:
    category = raw_label.get("label")
    if category not in CATEGORY_SET:
        raise ValueError(f"Unknown artifact category in dataset labels: {category!r}")

    points = []
    for raw_point in raw_label.get("points") or ():
        if not isinstance(raw_point, (list, tuple)) or len(raw_point) < 2:
            continue
        points.append((float(raw_point[0]), float(raw_point[1])))

    return ArtifactLabel(category=category, points=tuple(points))


def normalize_record(row: dict) -> ImageRecord:
    return ImageRecord(
        generator=str(row["generator"]),
        uid=str(row["uid"]),
        width=int(row["width"]),
        height=int(row["height"]),
        labels=tuple(normalize_label(label) for label in (row.get("labels") or ())),
    )


def read_records_from_parquet_files(parquet_files: Iterable[Path]) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for parquet_file in sorted(parquet_files):
        table = pq.read_table(
            parquet_file,
            columns=["generator", "uid", "labels", "width", "height"],
        )
        records.extend(normalize_record(row) for row in table.to_pylist())
    return records


def load_hf_records(
    dataset_repo: str,
    split: str,
    *,
    revision: str | None = None,
    cache_dir: str | None = None,
    local_files_only: bool = False,
) -> list[ImageRecord]:
    from huggingface_hub import snapshot_download

    snapshot_path = Path(
        snapshot_download(
            repo_id=dataset_repo,
            repo_type="dataset",
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            allow_patterns=[f"data/{split}-*.parquet"],
        )
    )
    parquet_files = sorted(snapshot_path.glob(f"data/{split}-*.parquet"))
    if not parquet_files and local_files_only:
        parquet_files = find_split_parquets_in_cached_snapshots(snapshot_path, split)
    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files found for split {split!r} in dataset repo {dataset_repo!r}"
        )
    return read_records_from_parquet_files(parquet_files)


def find_split_parquets_in_cached_snapshots(snapshot_path: Path, split: str) -> list[Path]:
    snapshots_dir = snapshot_path.parent
    if snapshots_dir.name != "snapshots":
        return []
    candidates = []
    for candidate_snapshot in snapshots_dir.iterdir():
        if not candidate_snapshot.is_dir():
            continue
        parquet_files = sorted(candidate_snapshot.glob(f"data/{split}-*.parquet"))
        if parquet_files:
            candidates.append((candidate_snapshot.stat().st_mtime, parquet_files))
    if not candidates:
        return []
    return max(candidates, key=lambda item: item[0])[1]
