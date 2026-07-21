"""Classification helpers for visual-clearance subjects."""

from __future__ import annotations

import re
from typing import Any

_DISPLAY_WORDS = re.compile(
    r"\b(?:art|artwork|canvas|clock|frame|mirror|painting|photo|photograph|"
    r"picture|poster|print|tapestry)\b"
)
_EXCLUDED_WORDS = re.compile(
    r"\b(?:display|light|projection screen|screen|sconce|shelf|television|tv)\b"
)


def is_wall_mounted_visual_subject(obj: dict[str, Any]) -> bool:
    """Return whether an object is decorative wall content."""
    hints = obj.get("functional_hints") or {}
    scene_type = str(
        obj.get("object_type") or hints.get("scene_object_type") or ""
    ).strip().lower()
    if scene_type != "wall_mounted":
        return False
    text = " ".join(
        str(obj.get(key) or "").strip().lower().replace("_", " ").replace("-", " ")
        for key in ("id", "name", "description", "category", "category_norm")
    )
    identity = " ".join(
        str(obj.get(key) or "").strip().lower().replace("_", " ").replace("-", " ")
        for key in ("id", "name", "category", "category_norm")
    )
    return bool(_DISPLAY_WORDS.search(text)) and not _EXCLUDED_WORDS.search(identity)
