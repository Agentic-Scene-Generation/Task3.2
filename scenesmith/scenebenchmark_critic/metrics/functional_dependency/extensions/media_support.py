"""Functional check for wall-mounted media alignment over a media console."""

from __future__ import annotations

import math

from typing import Any

MEDIA_CATEGORIES = {
    "display",
    "screen",
    "television",
    "tv",
    "wall_mounted_television",
    "wall_mounted_tv",
}
SUPPORT_CATEGORIES = {
    "entertainment_center",
    "media_console",
    "media_center",
    "tv_console",
    "tv_stand",
}


def evaluate_media_support_alignment(case_pack: dict[str, Any]) -> list[dict[str, Any]]:
    """Report wall-mounted media that is not centered over a matching console."""
    geometry = case_pack.get("scene_geometry") or {}
    objects = [obj for obj in geometry.get("objects") or [] if isinstance(obj, dict)]
    media = [obj for obj in objects if _is_wall_media(obj)]
    supports = [obj for obj in objects if _is_media_support(obj)]
    results: list[dict[str, Any]] = []

    for display in media:
        display_box = display.get("bbox_world") or {}
        display_center = _bbox_center(display_box)
        if display_center is None:
            continue
        candidates = [
            support
            for support in supports
            if str(support.get("room") or "") == str(display.get("room") or "")
        ] or supports
        if not candidates:
            continue
        support = min(
            candidates,
            key=lambda item: _xy_distance(display_box, item.get("bbox_world") or {}),
        )
        support_box = support.get("bbox_world") or {}
        support_center = _bbox_center(support_box)
        if support_center is None:
            continue

        display_surface_id = _surface_id(display)
        support_surface_id = _support_surface_id(
            support,
            objects,
            geometry.get("relations") or [],
        )
        target_wall_direction = _surface_direction(support_surface_id)
        target_window_ids = _window_ids_on_direction(
            geometry.get("scene_shell") or {},
            target_wall_direction,
            support_center=support_center,
            display_box=display_box,
        )
        axis = _wall_axis(display)
        lateral_offset = abs(display_center[axis] - support_center[axis])
        support_span = _bbox_span(support_box, axis)
        allowed_offset = max(0.15, min(0.4, 0.15 * support_span))
        display_bottom = _bbox_min(display_box, 2)
        support_top = _bbox_max(support_box, 2)
        vertical_gap = (
            display_bottom - support_top
            if display_bottom is not None and support_top is not None
            else None
        )
        # 2026-07-15 修改原因：仅比较当前 TV 墙面的局部轴会把“TV 在 east、
        # TV stand 在 south”误化成一个可沿 east 墙修复的偏移量。媒体显示器
        # 必须先回到支撑家具所在墙，再比较同墙中心线，适配任意房间方向。
        surface_mismatch = bool(
            display_surface_id
            and support_surface_id
            and (
                (
                    _surface_direction(display_surface_id)
                    and _surface_direction(support_surface_id)
                    and _surface_direction(display_surface_id)
                    != _surface_direction(support_surface_id)
                )
                or (
                    not _surface_direction(display_surface_id)
                    and not _surface_direction(support_surface_id)
                    and display_surface_id != support_surface_id
                )
            )
        )
        horizontal_ok = not surface_mismatch and lateral_offset <= allowed_offset
        vertical_ok = vertical_gap is None or vertical_gap >= -0.05
        display_id = str(display.get("id") or "")
        support_id = str(support.get("id") or "")
        check_id = f"media_support_alignment__{display_id}_{support_id}"
        if horizontal_ok and vertical_ok:
            label = "pass"
            reason = (
                f"Wall-mounted media `{display_id}` is centered over `{support_id}` "
                f"(lateral offset {lateral_offset:.2f}m, allowed {allowed_offset:.2f}m)."
            )
        else:
            label = "fail"
            failures = []
            if not horizontal_ok:
                if surface_mismatch:
                    failures.append(
                        f"display is on `{display_surface_id}` but support is on "
                        f"`{support_surface_id}`"
                    )
                else:
                    failures.append(
                        f"lateral offset {lateral_offset:.2f}m exceeds allowed "
                        f"{allowed_offset:.2f}m"
                    )
            if not vertical_ok:
                failures.append(
                    f"vertical gap {vertical_gap:.2f}m places media below the support"
                )
            if target_window_ids:
                failures.append(
                    f"target wall has opening(s) {', '.join(target_window_ids)}; "
                    "repair the opening before final media alignment"
                )
            reason = (
                f"Wall-mounted media `{display_id}` must be centered above `{support_id}`; "
                + "; ".join(failures)
                + "."
            )
        results.append(
            {
                "check_id": check_id,
                "metric": "functional_dependency",
                "label": label,
                "primary_object": display_id,
                "related_objects": [support_id],
                "confidence": 0.96,
                "reason": reason,
                "relation_type": "media_over_support_alignment",
                "diagnostics": {
                    "wall_axis": "x" if axis == 0 else "y",
                    "lateral_offset_m": round(lateral_offset, 4),
                    "allowed_lateral_offset_m": round(allowed_offset, 4),
                    "display_center_xy": [display_center[0], display_center[1]],
                    "support_center_xy": [support_center[0], support_center[1]],
                    "vertical_gap_m": (
                        round(vertical_gap, 4) if vertical_gap is not None else None
                    ),
                    "target_wall_surface_id": support_surface_id,
                    "display_wall_surface_id": display_surface_id,
                    "target_wall_direction": target_wall_direction,
                    "target_wall_window_ids": target_window_ids,
                    "surface_mismatch": surface_mismatch,
                },
                "evidence_refs": ["scene_geometry", "placement_info"],
                "scoring_tier": "core",
            }
        )
    return results


def _is_wall_media(obj: dict[str, Any]) -> bool:
    hints = obj.get("functional_hints") or {}
    types = {
        str(obj.get("object_type") or "").strip().lower().replace("-", "_"),
        str(hints.get("scene_object_type") or "").strip().lower().replace("-", "_"),
    }
    category = str(obj.get("category_norm") or obj.get("category") or "").lower()
    return bool(types & {"wall_mounted", "mounted"}) and (
        category in MEDIA_CATEGORIES or "television" in category or category == "tv"
    )


def _is_media_support(obj: dict[str, Any]) -> bool:
    category = str(obj.get("category_norm") or obj.get("category") or "").lower()
    text = " ".join(
        str(obj.get(key) or "").lower()
        for key in ("id", "name", "description", "category", "category_norm")
    )
    return category in SUPPORT_CATEGORIES or any(
        token in text
        for token in ("tv stand", "tv_stand", "media console", "entertainment center")
    )


def _wall_axis(obj: dict[str, Any]) -> int:
    surface_id = str(
        (obj.get("placement_info") or {}).get("parent_surface_id") or ""
    ).lower()
    return 0 if any(direction in surface_id for direction in ("north", "south")) else 1


def _surface_id(obj: dict[str, Any]) -> str:
    return str((obj.get("placement_info") or {}).get("parent_surface_id") or "")


def _surface_direction(surface_id: str) -> str:
    value = str(surface_id or "").lower()
    for direction in ("north", "south", "east", "west"):
        if direction in value:
            return direction
    return ""


def _support_surface_id(
    support: dict[str, Any],
    objects: list[dict[str, Any]],
    relations: list[dict[str, Any]],
) -> str:
    """Resolve the wall supporting a console, including snapped furniture."""
    explicit = _surface_id(support)
    if explicit:
        return explicit

    support_id = str(support.get("id") or "")
    for relation in relations:
        if not isinstance(relation, dict):
            continue
        if str(relation.get("subject_id") or relation.get("subject") or "") != support_id:
            continue
        target = str(relation.get("target_surface_id") or "")
        if target:
            return target

    support_box = support.get("bbox_world") or {}
    candidates = [
        obj
        for obj in objects
        if _is_wall(obj)
        and (
            not support.get("room")
            or str(obj.get("room") or "") == str(support.get("room") or "")
        )
    ]
    if not candidates:
        return ""
    nearest = min(
        candidates,
        key=lambda wall: _xy_box_gap(support_box, wall.get("bbox_world") or {}),
    )
    return str(nearest.get("id") or "")


def _is_wall(obj: dict[str, Any]) -> bool:
    category = str(obj.get("category_norm") or obj.get("category") or "").lower()
    object_type = str(obj.get("object_type") or "").lower()
    return category == "wall" or object_type == "wall"


def _xy_box_gap(first: dict[str, Any], second: dict[str, Any]) -> float:
    first_min, first_max = first.get("min"), first.get("max")
    second_min, second_max = second.get("min"), second.get("max")
    if not all(
        _valid_vec(value) for value in (first_min, first_max, second_min, second_max)
    ):
        return math.inf
    gap_x = max(
        float(second_min[0]) - float(first_max[0]),
        float(first_min[0]) - float(second_max[0]),
        0.0,
    )
    gap_y = max(
        float(second_min[1]) - float(first_max[1]),
        float(first_min[1]) - float(second_max[1]),
        0.0,
    )
    return math.hypot(gap_x, gap_y)


def _window_ids_on_direction(
    scene_shell: dict[str, Any],
    direction: str,
    *,
    support_center: list[float] | None = None,
    display_box: dict[str, Any] | None = None,
) -> list[str]:
    if not direction:
        return []
    if support_center is None or display_box is None:
        return []
    center = _bbox_center(display_box)
    size = display_box.get("size") or []
    if center is None or len(size) < 3:
        return []
    along_axis = 0 if direction in {"north", "south"} else 1
    # Use the larger horizontal extent because the TV may currently be mounted
    # on a perpendicular wall, where its world AABB's thin axis is the normal.
    half_display_span = max(float(size[0]), float(size[1])) / 2.0
    projected_min = float(support_center[along_axis]) - half_display_span
    projected_max = float(support_center[along_axis]) + half_display_span
    return sorted(
        str(window.get("id") or window.get("opening_id") or "")
        for window in scene_shell.get("windows") or []
        if isinstance(window, dict)
        and str(window.get("wall_direction") or "").lower() == direction
        and (window.get("id") or window.get("opening_id"))
        and _window_overlaps_projected_media(
            window, along_axis, projected_min, projected_max, center[2]
        )
    )


def _window_overlaps_projected_media(
    window: dict[str, Any],
    along_axis: int,
    projected_min: float,
    projected_max: float,
    center_z: float,
) -> bool:
    bbox = window.get("bbox") or {}
    minimum, maximum = bbox.get("min"), bbox.get("max")
    if _valid_vec(minimum) and _valid_vec(maximum):
        window_min = float(minimum[along_axis])
        window_max = float(maximum[along_axis])
        vertical_min = float(minimum[2])
        vertical_max = float(maximum[2])
        return (
            min(projected_max, window_max) > max(projected_min, window_min)
            and vertical_min <= center_z <= vertical_max
        )
    center = window.get("center") or window.get("position")
    width = float(window.get("width") or 0.0)
    if not _valid_vec(center) or width <= 0.0:
        return False
    return min(projected_max, float(center[along_axis]) + width / 2.0) > max(
        projected_min, float(center[along_axis]) - width / 2.0
    )


def _bbox_center(box: dict[str, Any]) -> list[float] | None:
    minimum, maximum = box.get("min"), box.get("max")
    if not _valid_vec(minimum) or not _valid_vec(maximum):
        return None
    return [
        (float(minimum[index]) + float(maximum[index])) / 2.0 for index in range(3)
    ]


def _bbox_span(box: dict[str, Any], axis: int) -> float:
    minimum, maximum = box.get("min"), box.get("max")
    if not _valid_vec(minimum) or not _valid_vec(maximum):
        return 1.0
    return abs(float(maximum[axis]) - float(minimum[axis]))


def _bbox_min(box: dict[str, Any], axis: int) -> float | None:
    value = box.get("min")
    return float(value[axis]) if _valid_vec(value) else None


def _bbox_max(box: dict[str, Any], axis: int) -> float | None:
    value = box.get("max")
    return float(value[axis]) if _valid_vec(value) else None


def _xy_distance(first: dict[str, Any], second: dict[str, Any]) -> float:
    a = _bbox_center(first)
    b = _bbox_center(second)
    if a is None or b is None:
        return math.inf
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _valid_vec(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and len(value) >= 3
