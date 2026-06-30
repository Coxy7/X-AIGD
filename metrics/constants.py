from __future__ import annotations

CATEGORIES = (
    "low-level-edge_shape",
    "low-level-texture",
    "low-level-color",
    "low-level-symbol",
    "high-level-semantics",
    "cognitive-level-commonsense",
    "cognitive-level-physics",
)

CATEGORY_SET = set(CATEGORIES)

TRANSFORMS = (
    "keep-original-size",
    "resize256-crop224",
    "resize518-crop518",
)
