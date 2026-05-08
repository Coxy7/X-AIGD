from __future__ import annotations

ALL_CATEGORIES_OPTION = "All"

CATEGORY_ORDER = [
    "low-level-edge_shape",
    "low-level-texture",
    "low-level-color",
    "low-level-symbol",
    "high-level-semantics",
    "cognitive-level-commonsense",
    "cognitive-level-physics",
]

CATEGORY_LABELS = {
    "low-level-edge_shape": "Edge & Shape",
    "low-level-texture": "Texture",
    "low-level-color": "Color",
    "low-level-symbol": "Symbol",
    "high-level-semantics": "Semantics",
    "cognitive-level-commonsense": "Commonsense",
    "cognitive-level-physics": "Physics",
}

CATEGORY_COLORS = {
    "low-level-edge_shape": {"hex": "#D1495B", "rgba": (209, 73, 91, 92)},
    "low-level-texture": {"hex": "#EDA72C", "rgba": (237, 167, 44, 92)},
    "low-level-color": {"hex": "#4C956C", "rgba": (76, 149, 108, 92)},
    "low-level-symbol": {"hex": "#2E86AB", "rgba": (46, 134, 171, 92)},
    "high-level-semantics": {"hex": "#5C4D7D", "rgba": (92, 77, 125, 92)},
    "cognitive-level-commonsense": {"hex": "#6C9A8B", "rgba": (108, 154, 139, 92)},
    "cognitive-level-physics": {"hex": "#A23E48", "rgba": (162, 62, 72, 92)},
}
