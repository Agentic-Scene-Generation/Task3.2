"""Workstation focal alignment extension for functional dependency."""

from __future__ import annotations

import math

from typing import Any

from scenesmith.scenebenchmark_critic.core.geometry import (
    bbox_center_xy,
    front_vector,
    side_vector,
)

FOCUS_WORDS = ("monitor", "screen", "display", "laptop", "notebook_computer")
WORK_RELATION = "seating_to_work_surface"
RELATION_TYPE = "workstation_focal_alignment"


def evaluate_workstation_focal_alignment(
    case_pack: dict[str, Any],
) -> list[dict[str, Any]]:
    """Check chair alignment with the display group on its paired desk."""
    geometry = case_pack.get("scene_geometry") or {}
    objects = {
        str(obj["id"]): obj
        for obj in geometry.get("objects") or []
        if isinstance(obj, dict) and obj.get("id")
    }
    surface_owners = {
        str(region.get("region_id")): object_id
        for object_id, obj in objects.items()
        for region in obj.get("support_regions") or []
        if isinstance(region, dict) and region.get("region_id")
    }
    results: list[dict[str, Any]] = []
    for check in case_pack.get("checks") or []:
        if check.get("metric") != "functional_dependency":
            continue
        if check.get("relation_type") != WORK_RELATION:
            continue
        seat = objects.get(str(check.get("subject_id") or ""))
        if seat is None:
            continue
        for target_id in check.get("target_ids") or []:
            desk = objects.get(str(target_id))
            if desk is None or not _is_work_surface(desk):
                continue
            focus = [
                obj
                for obj in objects.values()
                if _is_focus(obj)
                and surface_owners.get(_parent_surface_id(obj)) == str(desk["id"])
            ]
            if not focus:
                continue
            result = _evaluate_pair(seat, desk, focus)
            if result is not None:
                results.append(result)
            # 2026-07-17 修改原因：原 workstation 检查只验证椅子是否朝向
            # 显示器中心，无法阻止 monitor 自身背向椅子；补充 display_faces_user
            # 结果，让 deterministic critic context 可以否决 LLM 的反向修复。
            results.extend(_evaluate_display_orientation(seat, desk, focus))
    return results


def _evaluate_pair(
    seat: dict[str, Any], desk: dict[str, Any], focus: list[dict[str, Any]]
) -> dict[str, Any] | None:
    seat_center = bbox_center_xy(seat)
    desk_center = bbox_center_xy(desk)
    focus_centers = [bbox_center_xy(obj) for obj in focus]
    if seat_center is None or desk_center is None or any(
        center is None for center in focus_centers
    ):
        return None
    focus_center = (
        sum(center[0] for center in focus_centers if center is not None) / len(focus),
        sum(center[1] for center in focus_centers if center is not None) / len(focus),
    )
    desk_side = side_vector(desk)
    desk_front = front_vector(desk)
    seat_lateral = _dot(_subtract(seat_center, desk_center), desk_side)
    focus_lateral = _dot(_subtract(focus_center, desk_center), desk_side)
    lateral_offset = abs(seat_lateral - focus_lateral)
    seat_half = _extent_along(seat, desk_side) / 2.0
    focus_half = max(_extent_along(obj, desk_side) for obj in focus) / 2.0
    desk_long_span = max(_extent_along(desk, desk_side), _extent_along(desk, desk_front))
    lateral_tolerance = max(seat_half, focus_half, 0.15 * desk_long_span)
    angle = _angle_to_target(seat, seat_center, focus_center)
    if angle is None:
        return None
    if lateral_offset > lateral_tolerance:
        label = "fail"
        priority = "lateral_alignment"
    elif angle <= 25.0:
        label = "pass"
        priority = "none"
    elif angle <= 45.0:
        label = "degraded"
        priority = "orientation"
    else:
        label = "fail"
        priority = "orientation"
    seat_id = str(seat["id"])
    desk_id = str(desk["id"])
    focus_ids = sorted(str(obj["id"]) for obj in focus)
    return {
        "check_id": f"workstation_focal_alignment__{seat_id}__{desk_id}",
        "metric": "functional_dependency",
        "label": label,
        "confidence": 0.97,
        "primary_object": seat_id,
        "related_objects": [desk_id, *focus_ids],
        "selected_related_objects": [desk_id, *focus_ids],
        "blocking_objects": [],
        "relation_type": RELATION_TYPE,
        "reason": (
            f"Work chair {seat_id!r} is {lateral_offset:.3f} m laterally from "
            f"the display-group center and {angle:.1f} degrees off focus."
        ),
        "repair_advice": (
            f"Move {seat_id!r} along the front edge of {desk_id!r} and rotate it "
            "toward the monitor/display group; do not treat a distant visitor chair "
            "as a workstation blocker."
        ),
        "diagnostics": {
            "seat_id": seat_id,
            "desk_id": desk_id,
            "focus_ids": focus_ids,
            "lateral_offset_m": round(lateral_offset, 6),
            "lateral_tolerance_m": round(lateral_tolerance, 6),
            "angle_to_focus_deg": round(angle, 6),
            "priority": priority,
        },
        "evidence": {
            "constraint": "workstation_seat_to_display_group_alignment",
            "desk_local_axis": "side/front",
        },
        "scoring_tier": "core",
    }


def _evaluate_display_orientation(
    seat: dict[str, Any], desk: dict[str, Any], focus: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Check that each display's semantic front points toward the work chair."""
    seat_center = bbox_center_xy(seat)
    if seat_center is None:
        return []

    results: list[dict[str, Any]] = []
    for display in focus:
        display_center = bbox_center_xy(display)
        if display_center is None:
            continue
        angle = _angle_to_target(display, display_center, seat_center)
        if angle is None:
            continue
        if angle <= 25.0:
            label = "pass"
            priority = "none"
        elif angle <= 45.0:
            label = "degraded"
            priority = "orientation"
        else:
            label = "fail"
            priority = "orientation"
        display_id = str(display["id"])
        seat_id = str(seat["id"])
        desk_id = str(desk["id"])
        results.append(
            {
                "check_id": f"display_faces_user__{display_id}__{seat_id}",
                "metric": "functional_dependency",
                "label": label,
                "confidence": 0.97,
                "primary_object": display_id,
                "related_objects": [seat_id, desk_id],
                "selected_related_objects": [seat_id, desk_id],
                "blocking_objects": [],
                "relation_type": "display_faces_user",
                "reason": (
                    f"Display {display_id!r} is {angle:.1f} degrees off the work "
                    f"chair {seat_id!r}."
                ),
                "repair_advice": (
                    f"Rotate or reposition {display_id!r} so its screen/front points "
                    f"toward {seat_id!r}; use the parent surface's local frame when "
                    "issuing a placement change."
                ),
                "diagnostics": {
                    "display_id": display_id,
                    "seat_id": seat_id,
                    "desk_id": desk_id,
                    "angle_to_user_deg": round(angle, 6),
                    "priority": priority,
                },
                "evidence": {
                    "constraint": "display_front_to_workstation_seat",
                    "desk_local_axis": "surface/front",
                },
                "scoring_tier": "core",
            }
        )
    return results


def _is_work_surface(obj: dict[str, Any]) -> bool:
    text = _identity(obj)
    return any(token in text for token in ("desk", "workstation", "work_table"))


def _is_focus(obj: dict[str, Any]) -> bool:
    return any(token in _identity(obj) for token in FOCUS_WORDS)


def _identity(obj: dict[str, Any]) -> str:
    return " ".join(
        str(obj.get(key) or "").strip().lower().replace("-", "_")
        for key in ("id", "name", "category", "category_norm", "asset_id")
    )


def _parent_surface_id(obj: dict[str, Any]) -> str:
    return str((obj.get("placement_info") or {}).get("parent_surface_id") or "")


def _subtract(
    first: tuple[float, float], second: tuple[float, float]
) -> tuple[float, float]:
    return first[0] - second[0], first[1] - second[1]


def _dot(vector: tuple[float, float], axis: tuple[float, float]) -> float:
    return vector[0] * axis[0] + vector[1] * axis[1]


def _extent_along(obj: dict[str, Any], axis: tuple[float, float]) -> float:
    polygon = obj.get("footprint_world") or []
    points = [
        (float(point[0]), float(point[1]))
        for point in polygon
        if isinstance(point, (list, tuple)) and len(point) >= 2
    ]
    if not points:
        bbox = obj.get("bbox_world") or {}
        size = bbox.get("size") or []
        if len(size) >= 2:
            local_side = side_vector(obj)
            local_front = front_vector(obj)
            return abs(_dot(local_side, axis)) * float(size[0]) + abs(
                _dot(local_front, axis)
            ) * float(size[1])
        return 0.0
    projections = [_dot(point, axis) for point in points]
    return max(projections) - min(projections)


def _angle_to_target(
    seat: dict[str, Any],
    seat_center: tuple[float, float],
    target: tuple[float, float],
) -> float | None:
    vector = _subtract(target, seat_center)
    norm = math.hypot(*vector)
    if norm <= 1e-6:
        return 0.0
    facing = front_vector(seat)
    dot = max(-1.0, min(1.0, _dot(facing, vector) / norm))
    return abs(math.degrees(math.acos(dot)))
