"""Coordinated wall layout checks for prompt-explicit study furniture groups."""

from __future__ import annotations

import math
import re

from typing import Any

from scenesmith.scenebenchmark_critic.core.geometry import (
    bbox_center_xy,
    front_vector,
    object_footprint_polygon,
)

RELATION_TYPE = "study_furniture_layout"
_STUDY_RE = re.compile(r"\b(?:study|home office)\b")
_BACK_WALL_DESK_RE = re.compile(
    r"\bdesk\b.{0,100}\b(?:centered|centred|against)\b.{0,60}\bback\s+wall\b",
    re.DOTALL,
)
_SIDE_WALL_SEATING_RE = re.compile(
    r"\b(?:guest|visitor)\s+(?:chairs?|seats?|armchairs?)\b.{0,100}" r"\bside\s+wall\b",
    re.DOTALL,
)


def evaluate_study_furniture_layout(case_pack: dict[str, Any]) -> list[dict[str, Any]]:
    """Evaluate the legacy coordinated study wall-layout heuristic.

    Contract mode evaluates each prompt-originated relation through the generic
    intent-contract extensions instead.  This legacy layout encodes implicit
    tangential wall slots for storage and guest seating; those positions are not
    authorized by phrases such as ``against the wall`` and must not become hard
    requirements or repair targets.
    """
    task = str(case_pack.get("task_instruction") or "")
    lowered = task.lower()
    contract_mode = str(case_pack.get("intent_contract_mode") or "legacy")
    if contract_mode == "contract":
        return []
    if not (
        _STUDY_RE.search(lowered)
        and _BACK_WALL_DESK_RE.search(lowered)
        and _SIDE_WALL_SEATING_RE.search(lowered)
    ):
        return []

    geometry = case_pack.get("scene_geometry") or {}
    objects = [
        obj
        for obj in geometry.get("objects") or []
        if isinstance(obj, dict) and obj.get("id")
    ]
    rooms = [
        room
        for room in geometry.get("rooms") or []
        if isinstance(room, dict) and room.get("id")
    ]
    if not objects or not rooms:
        return []

    desk = _first_matching(objects, _is_desk)
    office_chair = _first_matching(objects, _is_office_chair)
    bookshelf = _first_matching(objects, _is_bookshelf)
    guest_chairs = sorted(
        (obj for obj in objects if _is_guest_chair(obj)),
        key=lambda obj: str(obj["id"]),
    )
    if desk is None or office_chair is None or bookshelf is None or not guest_chairs:
        return []

    room_bbox = rooms[0].get("bbox") or {}
    room_min = room_bbox.get("min") or []
    room_max = room_bbox.get("max") or []
    if len(room_min) < 2 or len(room_max) < 2:
        return []
    min_x, min_y = float(room_min[0]), float(room_min[1])
    max_x, max_y = float(room_max[0]), float(room_max[1])
    if max_x <= min_x or max_y <= min_y:
        return []

    room_center_x = (min_x + max_x) / 2.0
    desk_width, desk_depth = _local_size(desk)
    office_width, office_depth = _local_size(office_chair)
    shelf_width, shelf_depth = _local_size(bookshelf)
    guest_sizes = [_local_size(chair) for chair in guest_chairs]

    wall_gap = max(0.05, 0.012 * min(max_x - min_x, max_y - min_y))
    desk_target = (
        room_center_x,
        max_y - desk_depth / 2.0 - wall_gap,
    )
    office_target = (
        room_center_x,
        desk_target[1] - desk_depth / 2.0 - office_depth / 2.0 - 0.08,
    )

    shelf_center = bbox_center_xy(bookshelf) or (room_center_x, 0.0)
    shelf_on_west = float(shelf_center[0]) <= room_center_x
    shelf_x = (
        min_x + shelf_depth / 2.0 + wall_gap
        if shelf_on_west
        else max_x - shelf_depth / 2.0 - wall_gap
    )
    shelf_y_min = min_y + shelf_width / 2.0 + 0.15
    shelf_y_max = desk_target[1] - desk_depth / 2.0 - shelf_width / 2.0 - 0.35
    shelf_y = (
        (shelf_y_min + shelf_y_max) / 2.0 if shelf_y_max >= shelf_y_min else shelf_y_min
    )

    guest_wall_x = max_x if shelf_on_west else min_x
    guest_front = (-1.0, 0.0) if shelf_on_west else (1.0, 0.0)
    guest_yaw = _yaw_for_front(guest_front)
    max_guest_width = max(width for width, _ in guest_sizes)
    guest_y_min = min_y + max_guest_width / 2.0 + 0.30
    guest_y_max = desk_target[1] - desk_depth / 2.0 - max_guest_width / 2.0 - 0.35
    if guest_y_max <= guest_y_min:
        return []
    guest_y_slots = _segment_centers(guest_y_min, guest_y_max, len(guest_chairs))

    poses: list[dict[str, Any]] = [
        _pose_diagnostics(desk, desk_target, 180.0, role="desk"),
        _pose_diagnostics(office_chair, office_target, 0.0, role="office_chair"),
        _pose_diagnostics(
            bookshelf,
            (shelf_x, shelf_y),
            -90.0 if shelf_on_west else 90.0,
            role="bookshelf",
        ),
    ]
    for chair, (_, chair_depth), y_slot in zip(
        guest_chairs, guest_sizes, guest_y_slots, strict=True
    ):
        chair_x = (
            guest_wall_x - chair_depth / 2.0 - wall_gap
            if shelf_on_west
            else guest_wall_x + chair_depth / 2.0 + wall_gap
        )
        poses.append(
            _pose_diagnostics(
                chair,
                (chair_x, y_slot),
                guest_yaw,
                role="guest_chair",
            )
        )

    tolerance = max(0.18, 0.05 * min(max_x - min_x, max_y - min_y))
    aligned = all(
        float(pose["deviation_m"]) <= tolerance
        and float(pose["facing_error_deg"]) <= 10.0
        for pose in poses
    )
    related = [str(pose["object_id"]) for pose in poses]
    failures = [
        f"`{pose['object_id']}` is outside its {pose['role']} wall-layout pose"
        for pose in poses
        if float(pose["deviation_m"]) > tolerance
        or float(pose["facing_error_deg"]) > 10.0
    ]
    # ``shadow`` is a migration/observation mode: keep the historical
    # coordinated-wall diagnosis visible in reports, but do not let its
    # unprompted tangential slots reject an otherwise valid study.  Only the
    # explicit contract may become authoritative in that mode.  ``legacy``
    # remains available for reproducibility of the former behaviour.
    scoring_tier = "auxiliary" if contract_mode == "shadow" else "core"
    return [
        {
            "check_id": f"fd_study_{RELATION_TYPE}",
            "metric": "functional_dependency",
            "label": "pass" if aligned else "fail",
            "confidence": 0.94,
            "primary_object": str(desk["id"]),
            "related_objects": related,
            "selected_related_objects": related,
            "blocking_objects": [],
            "relation_type": RELATION_TYPE,
            "reason": (
                "The study desk is centered on the back wall, its office chair is "
                "aligned in front, and storage and guest seating occupy opposite "
                "side walls with every usable front perpendicular to its wall."
                if aligned
                else "A prompt-explicit study wall group requires a centered back-wall "
                "desk, its office chair, and non-overlapping storage and guest zones "
                "on opposite side walls. " + "; ".join(failures)
            ),
            "repair_advice": (
                "Move the desk, office chair, bookshelf, and all guest chairs as one "
                "coordinated wall layout; do not rotate a guest chair in place when "
                "storage occupies the same wall segment."
                if not aligned
                else ""
            ),
            "diagnostics": {
                "room_bounds_xy": [min_x, min_y, max_x, max_y],
                "storage_wall": "west" if shelf_on_west else "east",
                "guest_wall": "east" if shelf_on_west else "west",
                "member_poses": poses,
                "allowed_deviation_m": tolerance,
                "allowed_facing_error_deg": 10.0,
            },
            "evidence": {
                "layout": "room_relative_study_wall_group",
                "zone_separation": "storage_opposite_guest_seating",
            },
            "evaluation_source": "scenesmith_study_furniture_layout",
            "scoring_tier": scoring_tier,
        }
    ]


def _identity_text(obj: dict[str, Any]) -> str:
    return " ".join(
        str(obj.get(key) or "").lower().replace("_", " ")
        for key in ("id", "category", "category_norm", "name", "description")
    )


def _is_desk(obj: dict[str, Any]) -> bool:
    text = _identity_text(obj)
    return "desk" in text and "chair" not in text


def _is_office_chair(obj: dict[str, Any]) -> bool:
    text = _identity_text(obj)
    return "office" in text and any(token in text for token in ("chair", "seat"))


def _is_guest_chair(obj: dict[str, Any]) -> bool:
    text = _identity_text(obj)
    return any(role in text for role in ("guest", "visitor")) and any(
        token in text for token in ("chair", "seat", "armchair")
    )


def _is_bookshelf(obj: dict[str, Any]) -> bool:
    text = _identity_text(obj)
    return any(token in text for token in ("bookshelf", "bookcase", "shelving unit"))


def _first_matching(
    objects: list[dict[str, Any]], predicate: Any
) -> dict[str, Any] | None:
    matches = sorted(
        (obj for obj in objects if predicate(obj)), key=lambda obj: str(obj["id"])
    )
    return matches[0] if matches else None


def _local_size(obj: dict[str, Any]) -> tuple[float, float]:
    front = front_vector(obj)
    tangent = (-front[1], front[0])
    return _extent_along(obj, tangent), _extent_along(obj, front)


def _extent_along(obj: dict[str, Any], axis: tuple[float, float]) -> float:
    polygon = object_footprint_polygon(obj) or []
    if polygon:
        projections = [x * axis[0] + y * axis[1] for x, y in polygon]
        return max(projections) - min(projections)
    size = (obj.get("bbox_world") or {}).get("size") or []
    if len(size) < 2:
        return 0.5
    return abs(axis[0]) * float(size[0]) + abs(axis[1]) * float(size[1])


def _pose_diagnostics(
    obj: dict[str, Any],
    target_xy: tuple[float, float],
    target_yaw_deg: float,
    *,
    role: str,
) -> dict[str, Any]:
    center = bbox_center_xy(obj) or target_xy
    deviation = math.hypot(
        float(center[0]) - target_xy[0], float(center[1]) - target_xy[1]
    )
    target_front = _front_for_yaw(target_yaw_deg)
    current_front = front_vector(obj)
    dot = max(
        -1.0,
        min(
            1.0,
            current_front[0] * target_front[0] + current_front[1] * target_front[1],
        ),
    )
    return {
        "object_id": str(obj["id"]),
        "role": role,
        "target_center_xy_m": [float(target_xy[0]), float(target_xy[1])],
        "target_yaw_deg": float(target_yaw_deg),
        "deviation_m": round(deviation, 4),
        "facing_error_deg": round(math.degrees(math.acos(dot)), 2),
    }


def _segment_centers(lower: float, upper: float, count: int) -> list[float]:
    step = (upper - lower) / count
    return [lower + (index + 0.5) * step for index in range(count)]


def _front_for_yaw(yaw_deg: float) -> tuple[float, float]:
    yaw = math.radians(yaw_deg)
    return -math.sin(yaw), math.cos(yaw)


def _yaw_for_front(front: tuple[float, float]) -> float:
    return math.degrees(math.atan2(-front[0], front[1]))
