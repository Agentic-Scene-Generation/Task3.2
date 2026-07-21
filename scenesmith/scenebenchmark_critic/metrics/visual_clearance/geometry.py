"""Small 2-D projection primitives for visual-clearance checks."""

from __future__ import annotations

from typing import Any


def wall_projection_rect(
    obj: dict[str, Any], *, direction: str
) -> tuple[float, float, float, float] | None:
    bbox = obj.get("bbox_world") or {}
    minimum = bbox.get("min") or []
    maximum = bbox.get("max") or []
    if len(minimum) < 3 or len(maximum) < 3:
        return None
    lateral_axis = 0 if direction in {"north", "south"} else 1
    return (
        float(minimum[lateral_axis]),
        float(minimum[2]),
        float(maximum[lateral_axis]),
        float(maximum[2]),
    )


def rect_area(rect: tuple[float, float, float, float] | None) -> float:
    if rect is None:
        return 0.0
    return max(rect[2] - rect[0], 0.0) * max(rect[3] - rect[1], 0.0)


def rect_intersection(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> tuple[float, float, float, float] | None:
    result = (
        max(first[0], second[0]),
        max(first[1], second[1]),
        min(first[2], second[2]),
        min(first[3], second[3]),
    )
    return result if rect_area(result) > 1e-9 else None


def wall_direction(obj: dict[str, Any]) -> str:
    placement = obj.get("placement_info") or {}
    surface_id = str(placement.get("parent_surface_id") or "").lower()
    for direction in ("north", "south", "east", "west"):
        if direction in surface_id:
            return direction
    explicit = str(
        obj.get("wall_direction")
        or placement.get("wall_direction")
        or placement.get("surface_direction")
        or ""
    ).strip().lower()
    if explicit in {"north", "south", "east", "west"}:
        return explicit
    # 2026-07-16 修改原因：导出场景的 parent_surface_id 有时只有数值
    # 后缀，不能因此跳过同墙壁挂重叠；由壁挂物薄轴推断墙法向。
    size = (obj.get("bbox_world") or {}).get("size") or []
    if len(size) >= 2:
        return "east" if float(size[0]) < float(size[1]) else "north"
    return ""
