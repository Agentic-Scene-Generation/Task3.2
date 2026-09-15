"""Physical geometry helpers for structural opening proxies."""

from __future__ import annotations

from typing import Any

import numpy as np


def normalized_opening_type(opening: Any) -> str:
    value = getattr(opening, "opening_type", "")
    if hasattr(value, "value"):
        value = value.value
    normalized = str(value or "").strip().lower()
    return "opening" if normalized == "open" else normalized


def normalized_wall_direction(opening: Any) -> str:
    value = getattr(opening, "wall_direction", "")
    if hasattr(value, "value"):
        value = value.value
    normalized = str(value or "").strip().lower()
    return normalized.rsplit(".", 1)[-1]


def opening_physical_bounds(
    opening: Any,
    *,
    wall_thickness_m: float = 0.05,
    offset: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return the thin wall-plane AABB, never the interaction-clearance AABB."""
    try:
        center = np.asarray(
            getattr(opening, "center_world", [0.0, 0.0, 0.0]), dtype=float
        )
        width = max(0.0, float(getattr(opening, "width", 0.0) or 0.0))
        height = max(0.0, float(getattr(opening, "height", 0.0) or 0.0))
        sill = float(getattr(opening, "sill_height", 0.0) or 0.0)
    except (TypeError, ValueError):
        return None
    if center.size < 2 or width <= 0.0 or height <= 0.0:
        return None
    direction = normalized_wall_direction(opening)
    thickness = max(0.01, min(0.05, float(wall_thickness_m or 0.05)))
    if direction in {"north", "south"}:
        size_xy = np.array([width, thickness], dtype=float)
    elif direction in {"east", "west"}:
        size_xy = np.array([thickness, width], dtype=float)
    else:
        return None
    center_xy = center[:2] + (
        np.asarray(offset, dtype=float)[:2] if offset is not None else 0.0
    )
    lower = np.array(
        [center_xy[0] - size_xy[0] / 2.0, center_xy[1] - size_xy[1] / 2.0, sill]
    )
    upper = np.array(
        [
            center_xy[0] + size_xy[0] / 2.0,
            center_xy[1] + size_xy[1] / 2.0,
            sill + height,
        ]
    )
    return lower, upper
