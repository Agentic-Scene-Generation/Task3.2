"""Geometry targets for wall-backed storage furniture."""

from __future__ import annotations

import math

from typing import Any

from scenesmith.scenebenchmark_critic.core.geometry import (
    bbox_center_xy,
    object_category,
    object_footprint_polygon,
)

RELATION_TYPE = "wall_backed_storage_alignment"

_WALL_BACKED_STORAGE = {
    "bookcase",
    "bookshelf",
    "cabinet",
    "credenza",
    "dresser",
    "shelving",
    "shelving_unit",
    "sideboard",
    "storage_cabinet",
    "wardrobe",
}


def evaluate_wall_backed_storage_alignment(
    case_pack: dict[str, Any],
) -> list[dict[str, Any]]:
    geometry = case_pack.get("scene_geometry") or {}
    objects = [obj for obj in geometry.get("objects") or [] if isinstance(obj, dict)]
    walls = [obj for obj in objects if object_category(obj) == "wall"]
    room_center = _room_center(geometry)
    if not walls or room_center is None:
        return []

    results: list[dict[str, Any]] = []
    for obj in objects:
        if not _is_wall_backed_storage(obj):
            continue
        center = bbox_center_xy(obj)
        dining_context = _dining_sideboard_context(obj, objects)
        candidates = _candidate_poses(
            obj,
            walls,
            room_center,
            dining_context=dining_context,
        )
        if center is None or not candidates:
            continue
        preferred = [
            item
            for item in candidates
            if float(item.get("dining_table_wall_error_deg", 0.0)) <= 20.0
        ]
        nearest = min(preferred or candidates, key=lambda item: item["translation_m"])
        gap = float(nearest["wall_gap_m"])
        front_error = float(nearest["front_error_deg"])
        if gap <= 0.1 and front_error <= 20.0:
            label, confidence = "pass", 0.91
        elif gap <= 0.3 and front_error <= 45.0:
            label, confidence = "degraded", 0.87
        else:
            label, confidence = "fail", 0.94
        object_id = str(obj.get("id") or "")
        wall_id = str(nearest["wall_id"])
        results.append(
            {
                "check_id": f"wall_backed_storage__{object_id}",
                "metric": "functional_dependency",
                "label": label,
                "confidence": confidence,
                "primary_object": object_id,
                "related_objects": [wall_id],
                "selected_related_objects": [wall_id],
                "blocking_objects": [],
                "relation_type": RELATION_TYPE,
                "reason": (
                    f"Storage furniture {object_id!r} has a {gap:.2f} m footprint "
                    f"gap to its nearest usable wall {wall_id!r} and its front "
                    f"is {front_error:.0f} degrees from the inward wall normal."
                ),
                "repair_advice": (
                    "Use one of the critic-provided wall poses and accept it only "
                    "when whole-scene critic results improve."
                    if label != "pass"
                    else ""
                ),
                "diagnostics": {
                    "object_id": object_id,
                    "current_center_xy_m": [round(value, 6) for value in center],
                    "nearest_wall_gap_m": round(gap, 6),
                    "allowed_wall_gap_m": 0.1,
                    "front_error_deg": round(front_error, 6),
                    "allowed_front_error_deg": 20.0,
                    "candidate_poses": candidates,
                },
                "evidence": {
                    "constraint": "storage_backed_by_wall_with_front_facing_inward"
                },
                "evaluation_source": "scenesmith_wall_backed_storage_alignment",
                "scoring_tier": "core",
            }
        )
    return results


def _is_wall_backed_storage(obj: dict[str, Any]) -> bool:
    if str(obj.get("object_type") or "").lower() != "furniture":
        return False
    category = object_category(obj)
    group = str((obj.get("functional_hints") or {}).get("category_group") or "")
    return category in _WALL_BACKED_STORAGE or group == "storage"


def _room_center(geometry: dict[str, Any]) -> tuple[float, float] | None:
    rooms = geometry.get("rooms") or []
    if not rooms:
        return None
    bbox = (rooms[0].get("bbox") or {}) if isinstance(rooms[0], dict) else {}
    lower, upper = bbox.get("min") or [], bbox.get("max") or []
    if len(lower) < 2 or len(upper) < 2:
        return None
    return (float(lower[0]) + float(upper[0])) / 2.0, (
        float(lower[1]) + float(upper[1])
    ) / 2.0


def _candidate_poses(
    obj: dict[str, Any],
    walls: list[dict[str, Any]],
    room_center: tuple[float, float],
    *,
    dining_context: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> list[dict[str, Any]]:
    center = bbox_center_xy(obj)
    polygon = object_footprint_polygon(obj) or []
    if center is None or len(polygon) < 3:
        return []
    yaw_deg = float(obj.get("yaw_deg") or 0.0)
    yaw = math.radians(yaw_deg)
    local_x = (math.cos(yaw), math.sin(yaw))
    local_y = (-math.sin(yaw), math.cos(yaw))
    width = _extent(polygon, local_x)
    depth = _extent(polygon, local_y)
    candidates: list[dict[str, Any]] = []
    for wall in walls:
        wall_bbox = wall.get("bbox_world") or {}
        wall_center = wall_bbox.get("center") or []
        wall_size = wall_bbox.get("size") or []
        if len(wall_center) < 2 or len(wall_size) < 2:
            continue
        normal_index = 0 if float(wall_size[0]) <= float(wall_size[1]) else 1
        tangent_index = 1 - normal_index
        wall_coord = float(wall_center[normal_index])
        inward = 1.0 if room_center[normal_index] >= wall_coord else -1.0
        wall_half = float(wall_size[normal_index]) / 2.0
        wall_tangent_half = float(wall_size[tangent_index]) / 2.0
        current_normal_values = [point[normal_index] for point in polygon]
        wall_min = wall_coord - wall_half
        wall_max = wall_coord + wall_half
        current_gap = max(
            0.0,
            wall_min - max(current_normal_values),
            min(current_normal_values) - wall_max,
        )
        inward_normal = [0.0, 0.0]
        inward_normal[normal_index] = inward
        candidate_yaw = math.atan2(-inward_normal[0], inward_normal[1])
        candidate_yaw_deg = math.degrees(candidate_yaw) % 360.0
        axis_x = (math.cos(candidate_yaw), math.sin(candidate_yaw))
        axis_y = (-math.sin(candidate_yaw), math.cos(candidate_yaw))
        normal = (1.0, 0.0) if normal_index == 0 else (0.0, 1.0)
        tangent = (0.0, 1.0) if normal_index == 0 else (1.0, 0.0)
        dining_wall_error = (
            _axis_alignment_error_deg(tangent, dining_context[1])
            if dining_context is not None
            else 0.0
        )
        normal_half = (
            abs(_dot(axis_x, normal)) * width + abs(_dot(axis_y, normal)) * depth
        ) / 2.0
        tangent_half = (
            abs(_dot(axis_x, tangent)) * width + abs(_dot(axis_y, tangent)) * depth
        ) / 2.0
        target = [float(center[0]), float(center[1])]
        target[normal_index] = wall_coord + inward * (wall_half + normal_half + 0.02)
        tangent_limit = max(0.0, wall_tangent_half - tangent_half - 0.05)
        desired_tangent = (
            dining_context[0][tangent_index]
            if dining_context is not None and dining_wall_error <= 20.0
            else target[tangent_index]
        )
        target[tangent_index] = min(
            max(
                desired_tangent,
                float(wall_center[tangent_index]) - tangent_limit,
            ),
            float(wall_center[tangent_index]) + tangent_limit,
        )
        translation = math.hypot(target[0] - center[0], target[1] - center[1])
        candidates.append(
            {
                "wall_id": str(wall.get("id") or ""),
                "target_center_xy_m": [round(value, 6) for value in target],
                "target_yaw_deg": round(candidate_yaw_deg, 6),
                "front_error_deg": round(
                    abs((candidate_yaw_deg - yaw_deg + 180.0) % 360.0 - 180.0),
                    6,
                ),
                "wall_gap_m": round(current_gap, 6),
                "translation_m": round(translation, 6),
                "dining_table_wall_error_deg": round(dining_wall_error, 6),
            }
        )
    candidates.sort(
        key=lambda item: (
            float(item["dining_table_wall_error_deg"]) > 20.0,
            item["translation_m"],
            abs((float(item["target_yaw_deg"]) - yaw_deg + 180.0) % 360.0 - 180.0),
            item["wall_id"],
        )
    )
    return candidates[:8]


def _dining_sideboard_context(
    storage: dict[str, Any],
    objects: list[dict[str, Any]],
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Return the dining-table center and long axis for sideboard wall selection."""
    if object_category(storage) not in {"credenza", "sideboard"}:
        return None
    table = next(
        (obj for obj in objects if object_category(obj) == "dining_table"),
        None,
    )
    if table is None:
        return None
    center = bbox_center_xy(table)
    polygon = object_footprint_polygon(table) or []
    if center is None or len(polygon) < 3:
        return None
    yaw = math.radians(float(table.get("yaw_deg") or 0.0))
    local_x = (math.cos(yaw), math.sin(yaw))
    local_y = (-math.sin(yaw), math.cos(yaw))
    long_axis = (
        local_x if _extent(polygon, local_x) >= _extent(polygon, local_y) else local_y
    )
    return center, long_axis


def _axis_alignment_error_deg(
    first: tuple[float, float], second: tuple[float, float]
) -> float:
    dot = abs(_dot(first, second))
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


def _extent(points: list[tuple[float, float]], axis: tuple[float, float]) -> float:
    projections = [_dot(point, axis) for point in points]
    return max(projections) - min(projections)


def _dot(first: tuple[float, float], second: tuple[float, float]) -> float:
    return first[0] * second[0] + first[1] * second[1]
