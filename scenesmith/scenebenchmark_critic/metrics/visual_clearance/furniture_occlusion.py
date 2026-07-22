"""Visual-clearance checks for decorative wall-mounted displays."""

from __future__ import annotations

import re

from typing import Any

RELATION_TYPE = "wall_mounted_visibility"
MAX_PASS_OCCLUSION = 0.05
FAIL_OCCLUSION = 0.20

_DISPLAY_WORDS = re.compile(
    r"\b(?:art|artwork|canvas|clock|frame|mirror|painting|photo|photograph|"
    r"picture|poster|print|shelf|shelving|tapestry)\b"
)
_EXCLUDED_WORDS = re.compile(
    r"\b(?:display|light|projection screen|screen|sconce|television|tv)\b"
)


def evaluate_wall_mounted_visibility(
    case_pack: dict[str, Any],
) -> list[dict[str, Any]]:
    """Check that wall decor and shelves remain visible past furniture."""
    geometry = case_pack.get("scene_geometry") or {}
    objects = [
        obj
        for obj in geometry.get("objects") or []
        if isinstance(obj, dict) and obj.get("id")
    ]
    displays = [obj for obj in objects if _is_wall_display(obj)]
    furniture = [obj for obj in objects if _scene_object_type(obj) == "furniture"]
    rooms = [room for room in geometry.get("rooms") or [] if isinstance(room, dict)]
    results: list[dict[str, Any]] = []
    for display in displays:
        result = _evaluate_display(display, furniture=furniture, rooms=rooms)
        if result is not None:
            results.append(result)
    return results


def _evaluate_display(
    display: dict[str, Any],
    *,
    furniture: list[dict[str, Any]],
    rooms: list[dict[str, Any]],
) -> dict[str, Any] | None:
    direction = _wall_direction(display)
    display_bounds = _bounds(display)
    room = _room_for_object(display, rooms)
    room_bounds = _room_bounds(room)
    if not direction or display_bounds is None or room_bounds is None:
        return None

    lateral_axis = 0 if direction in {"north", "south"} else 1
    normal_axis = 1 - lateral_axis
    display_rect = _projected_rect(display_bounds, lateral_axis)
    display_area = _rect_area(display_rect)
    if display_area <= 1e-8:
        return None

    normal_span = room_bounds[normal_axis + 3] - room_bounds[normal_axis]
    wall_clearance = max(0.35, min(0.75, 0.15 * normal_span))
    blocker_rows: list[dict[str, Any]] = []
    intersections: list[tuple[float, float, float, float]] = []
    for candidate in furniture:
        if str(candidate.get("room") or "") not in {
            "",
            str(display.get("room") or ""),
        }:
            continue
        candidate_bounds = _bounds(candidate)
        if candidate_bounds is None:
            continue
        wall_distance = _distance_from_wall(
            candidate_bounds,
            room_bounds=room_bounds,
            direction=direction,
        )
        if wall_distance is None or wall_distance > wall_clearance:
            continue
        candidate_rect = _projected_rect(candidate_bounds, lateral_axis)
        intersection = _rect_intersection(display_rect, candidate_rect)
        if intersection is None:
            continue
        fraction = _rect_area(intersection) / display_area
        if fraction <= 1e-6:
            continue
        intersections.append(intersection)
        blocker_rows.append(
            {
                "object_id": str(candidate["id"]),
                "category": str(
                    candidate.get("category_norm") or candidate.get("category") or ""
                ),
                "wall_distance_m": round(wall_distance, 4),
                "individual_occluded_fraction": round(fraction, 6),
                "projected_rect": [round(value, 4) for value in candidate_rect],
                "intersection_rect": [round(value, 4) for value in intersection],
            }
        )

    occluded_fraction = min(_rect_union_area(intersections) / display_area, 1.0)
    blocker_rows.sort(
        key=lambda row: (-row["individual_occluded_fraction"], row["object_id"])
    )
    blocking_ids = [row["object_id"] for row in blocker_rows]
    if occluded_fraction <= MAX_PASS_OCCLUSION:
        label = "pass"
    elif occluded_fraction < FAIL_OCCLUSION:
        label = "degraded"
    else:
        label = "fail"

    display_id = str(display["id"])
    percentage = 100.0 * occluded_fraction
    if label == "pass":
        reason = (
            f"Wall-mounted display `{display_id}` is visually clear on the "
            f"{direction} wall; nearby furniture covers {percentage:.1f}% of its "
            f"projected face (allowed {100 * MAX_PASS_OCCLUSION:.0f}%)."
        )
        repair_advice = ""
    else:
        blocker_text = ", ".join(f"`{item}`" for item in blocking_ids)
        reason = (
            f"Wall-mounted display `{display_id}` is visually occluded on the "
            f"{direction} wall: nearby furniture {blocker_text} covers "
            f"{percentage:.1f}% of its projected face; at most "
            f"{100 * MAX_PASS_OCCLUSION:.0f}% is allowed."
        )
        # 2026-07-15 修改原因：wall stage 发生在家具布局完成后；检测到高柜
        # 遮挡画、镜子或时钟时，应优先让 wall agent 在同墙移动装饰物，避免
        # 为修复次级装饰反向破坏已经验证过的家具拓扑与净空。
        repair_advice = (
            f"First, move `{display_id}` laterally or upward on the same {direction} wall "
            f"until its projected overlap with {blocker_text} is at most "
            f"{100 * MAX_PASS_OCCLUSION:.0f}%. Keep the complete display inside the "
            "wall bounds and clear of windows/doors; do not move the wardrobe or "
            "other accepted furniture solely to expose wall decor."
        )

    return {
        "check_id": f"wall_visibility__{display_id}",
        "metric": "visual_clearance",
        "label": label,
        "confidence": 0.97 if label != "pass" else 0.93,
        "primary_object": display_id,
        "related_objects": blocking_ids,
        "selected_related_objects": blocking_ids,
        "blocking_objects": blocking_ids,
        "relation_type": RELATION_TYPE,
        "reason": reason,
        "repair_advice": repair_advice,
        "diagnostics": {
            "wall_direction": direction,
            "projection_axes": {
                "lateral": "x" if lateral_axis == 0 else "y",
                "vertical": "z",
            },
            "display_projected_rect": [round(value, 4) for value in display_rect],
            "display_projected_area_m2": round(display_area, 6),
            "occluded_fraction": round(occluded_fraction, 6),
            "allowed_occluded_fraction": MAX_PASS_OCCLUSION,
            "fail_occluded_fraction": FAIL_OCCLUSION,
            "wall_furniture_clearance_m": round(wall_clearance, 4),
            "blockers": blocker_rows,
        },
        "evidence": {
            "constraint": "wall_display_projected_visibility",
            "coordinate_frame": f"{direction}_wall_lateral_z",
        },
        "evaluation_source": "scenesmith_wall_mounted_visibility",
        "scoring_tier": "core",
    }


def evaluate_furniture_occlusion(
    case_pack: dict[str, Any],
) -> list[dict[str, Any]]:
    """Name the legacy wall-visibility rule by its new metric ownership."""
    return evaluate_wall_mounted_visibility(case_pack)


def _is_wall_display(obj: dict[str, Any]) -> bool:
    if _scene_object_type(obj) != "wall_mounted":
        return False
    text = _object_text(obj)
    identity = " ".join(
        str(obj.get(key) or "").strip().lower().replace("_", "-").replace("-", " ")
        for key in ("id", "name", "category", "category_norm")
    )
    # 2026-07-15 修改原因：时钟描述可能包含 ``readable display``，不能因为
    # 描述里的普通 display 一词把 wall clock 排除；排除项只看资产身份字段。
    return bool(_DISPLAY_WORDS.search(text)) and not _EXCLUDED_WORDS.search(identity)


def _scene_object_type(obj: dict[str, Any]) -> str:
    hints = obj.get("functional_hints") or {}
    return (
        str(obj.get("object_type") or hints.get("scene_object_type") or "")
        .strip()
        .lower()
    )


def _object_text(obj: dict[str, Any]) -> str:
    return " ".join(
        str(obj.get(key) or "").strip().lower().replace("_", "-").replace("-", " ")
        for key in ("id", "name", "description", "category", "category_norm")
    )


def _wall_direction(obj: dict[str, Any]) -> str:
    surface_id = str(
        (obj.get("placement_info") or {}).get("parent_surface_id") or ""
    ).lower()
    for direction in ("north", "south", "east", "west"):
        if direction in surface_id:
            return direction
    return ""


def _room_for_object(
    obj: dict[str, Any], rooms: list[dict[str, Any]]
) -> dict[str, Any] | None:
    room_id = str(obj.get("room") or "")
    for room in rooms:
        if room_id and str(room.get("id") or "") == room_id:
            return room
    return rooms[0] if len(rooms) == 1 else None


def _bounds(
    obj: dict[str, Any],
) -> tuple[float, float, float, float, float, float] | None:
    bbox = obj.get("bbox_world") or {}
    minimum = bbox.get("min") or []
    maximum = bbox.get("max") or []
    if len(minimum) < 3 or len(maximum) < 3:
        return None
    return (
        float(minimum[0]),
        float(minimum[1]),
        float(minimum[2]),
        float(maximum[0]),
        float(maximum[1]),
        float(maximum[2]),
    )


def _room_bounds(
    room: dict[str, Any] | None,
) -> tuple[float, float, float, float, float, float] | None:
    bbox = (room or {}).get("bbox") or {}
    minimum = bbox.get("min") or []
    maximum = bbox.get("max") or []
    if len(minimum) < 3 or len(maximum) < 3:
        return None
    return (
        float(minimum[0]),
        float(minimum[1]),
        float(minimum[2]),
        float(maximum[0]),
        float(maximum[1]),
        float(maximum[2]),
    )


def _projected_rect(
    bounds: tuple[float, float, float, float, float, float], lateral_axis: int
) -> tuple[float, float, float, float]:
    return (
        bounds[lateral_axis],
        bounds[2],
        bounds[lateral_axis + 3],
        bounds[5],
    )


def _distance_from_wall(
    bounds: tuple[float, float, float, float, float, float],
    *,
    room_bounds: tuple[float, float, float, float, float, float],
    direction: str,
) -> float | None:
    if direction == "north":
        return max(room_bounds[4] - bounds[4], 0.0)
    if direction == "south":
        return max(bounds[1] - room_bounds[1], 0.0)
    if direction == "east":
        return max(room_bounds[3] - bounds[3], 0.0)
    if direction == "west":
        return max(bounds[0] - room_bounds[0], 0.0)
    return None


def _rect_area(rect: tuple[float, float, float, float]) -> float:
    return max(rect[2] - rect[0], 0.0) * max(rect[3] - rect[1], 0.0)


def _rect_intersection(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> tuple[float, float, float, float] | None:
    result = (
        max(first[0], second[0]),
        max(first[1], second[1]),
        min(first[2], second[2]),
        min(first[3], second[3]),
    )
    return result if _rect_area(result) > 1e-8 else None


def _rect_union_area(rectangles: list[tuple[float, float, float, float]]) -> float:
    """Return exact union area for a small set of axis-aligned rectangles."""
    if not rectangles:
        return 0.0
    xs = sorted({value for rect in rectangles for value in (rect[0], rect[2])})
    area = 0.0
    for left, right in zip(xs, xs[1:]):
        if right <= left:
            continue
        intervals = sorted(
            (rect[1], rect[3])
            for rect in rectangles
            if rect[0] < right and rect[2] > left
        )
        covered = 0.0
        current_start: float | None = None
        current_end: float | None = None
        for start, end in intervals:
            if current_start is None:
                current_start, current_end = start, end
            elif start <= current_end:
                current_end = max(current_end, end)
            else:
                covered += current_end - current_start
                current_start, current_end = start, end
        if current_start is not None and current_end is not None:
            covered += current_end - current_start
        area += (right - left) * covered
    return area
