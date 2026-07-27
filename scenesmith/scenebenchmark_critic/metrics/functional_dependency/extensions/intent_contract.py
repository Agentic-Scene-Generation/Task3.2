"""Geometry evaluators for prompt-originated intent-contract constraints."""

from __future__ import annotations

import math

from typing import Any

from scenesmith.scenebenchmark_critic.core.geometry import (
    bbox_center_xy,
    object_category,
)
from scenesmith.scenebenchmark_critic.intent_contract import (
    bound_ids,
    contract_constraints,
    is_hard_constraint,
    selected_ids,
)


def evaluate_intent_contract_extensions(
    case_pack: dict[str, Any],
) -> list[dict[str, Any]]:
    """Evaluate relations that do not map to a pairwise FD rule.

    Pairwise constraints (chair->desk, object->support, furniture->wall) are
    materialized as normal functional-dependency checks.  This extension owns
    group/room-relative relations so their geometry stays reusable rather than
    being encoded in a prompt-specific evaluator.
    """
    mode = str(case_pack.get("intent_contract_mode") or "legacy")
    if mode == "legacy":
        return []
    geometry = case_pack.get("scene_geometry") or {}
    objects = [
        item
        for item in geometry.get("objects") or []
        if isinstance(item, dict) and item.get("id")
    ]
    if not objects:
        return []
    results: list[dict[str, Any]] = []
    for constraint in contract_constraints(
        case_pack,
        relations=(
            "required_count",
            "centered_in_room",
            "centered_on_wall",
            "flanking",
        ),
    ):
        relation = str(constraint.get("relation") or "")
        tier = _tier(constraint, mode)
        if relation == "required_count":
            result = _evaluate_required_count(constraint, objects, tier)
        elif relation == "centered_in_room":
            result = _evaluate_centered_in_room(constraint, geometry, objects, tier)
        elif relation == "centered_on_wall":
            result = _evaluate_centered_on_wall(constraint, geometry, objects, tier)
        else:
            result = _evaluate_flanking(constraint, objects, tier)
        if result is not None:
            results.extend(result if isinstance(result, list) else [result])
    return results


def _evaluate_required_count(
    constraint: dict[str, Any], objects: list[dict[str, Any]], tier: str
) -> dict[str, Any] | None:
    selector = constraint.get("subjects") or {}
    required = int(selector.get("count") or 1)
    matched = selected_ids(selector, objects)
    category = str(selector.get("category") or "required_object")
    return _result(
        constraint,
        suffix=category,
        label="pass" if len(matched) >= required else "fail",
        primary=matched[0] if matched else category,
        related=matched,
        relation_type="required_count",
        tier=tier,
        reason=(
            f"Prompt/room contract requires at least {required} `{category}` object(s); "
            f"found {len(matched)}."
        ),
        diagnostics={"required_count": required, "observed_ids": matched},
    )


def _evaluate_centered_in_room(
    constraint: dict[str, Any],
    geometry: dict[str, Any],
    objects: list[dict[str, Any]],
    tier: str,
) -> list[dict[str, Any]]:
    bounds = _room_bounds(geometry)
    if bounds is None:
        return []
    min_x, min_y, max_x, max_y = bounds
    center = ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)
    scale = min(max_x - min_x, max_y - min_y)
    allowed = max(0.15, 0.08 * scale)
    by_id = {str(obj["id"]): obj for obj in objects}
    results: list[dict[str, Any]] = []
    for object_id in bound_ids(constraint.get("subjects"), objects):
        obj = by_id.get(object_id)
        current = bbox_center_xy(obj) if obj is not None else None
        if current is None:
            continue
        offset = math.hypot(current[0] - center[0], current[1] - center[1])
        label = "pass" if offset <= allowed else "fail"
        results.append(
            _result(
                constraint,
                suffix=object_id,
                label=label,
                primary=object_id,
                related=[],
                # Reuse the generic group-safe repair path already used for
                # room-centering; the target remains room-relative.
                relation_type="room_center_alignment",
                tier=tier,
                reason=(
                    f"`{object_id}` is {offset:.2f}m from the prompt-requested room "
                    f"center (allowed {allowed:.2f}m)."
                ),
                diagnostics={
                    "room_center_xy": [round(center[0], 6), round(center[1], 6)],
                    "offset_m": round(offset, 6),
                    "allowed_offset_m": round(allowed, 6),
                    "room_bounds_xy": list(bounds),
                },
            )
        )
    return results


def _evaluate_centered_on_wall(
    constraint: dict[str, Any],
    geometry: dict[str, Any],
    objects: list[dict[str, Any]],
    tier: str,
) -> list[dict[str, Any]]:
    bounds = _room_bounds(geometry)
    walls = [obj for obj in objects if object_category(obj) == "wall"]
    if bounds is None or not walls:
        return []
    by_id = {str(obj["id"]): obj for obj in objects}
    results: list[dict[str, Any]] = []
    for object_id in bound_ids(constraint.get("subjects"), objects):
        obj = by_id.get(object_id)
        current = bbox_center_xy(obj) if obj is not None else None
        if current is None:
            continue
        pose = _centered_wall_pose(obj, walls, bounds)
        if pose is None:
            continue
        target, yaw, tangent_error, normal_error, allowed_tangent = pose
        yaw_error = abs(
            (float(obj.get("yaw_deg") or 0.0) - yaw + 180.0) % 360.0 - 180.0
        )
        label = (
            "pass"
            if tangent_error <= allowed_tangent
            and normal_error <= 0.10
            and yaw_error <= 20.0
            else "fail"
        )
        results.append(
            _result(
                constraint,
                suffix=object_id,
                label=label,
                primary=object_id,
                related=[],
                relation_type="centered_on_wall",
                tier=tier,
                reason=(
                    f"`{object_id}` must be centered and inward-facing on its selected wall; "
                    f"tangent error {tangent_error:.2f}m (allowed {allowed_tangent:.2f}m), "
                    f"wall offset {normal_error:.2f}m, yaw error {yaw_error:.0f}deg."
                ),
                diagnostics={
                    "target_center_xy_m": [round(target[0], 6), round(target[1], 6)],
                    "target_yaw_deg": round(yaw, 6),
                    "tangent_error_m": round(tangent_error, 6),
                    "normal_error_m": round(normal_error, 6),
                    "allowed_tangent_error_m": round(allowed_tangent, 6),
                    "yaw_error_deg": round(yaw_error, 6),
                },
            )
        )
    return results


def _evaluate_flanking(
    constraint: dict[str, Any], objects: list[dict[str, Any]], tier: str
) -> dict[str, Any] | None:
    subjects = bound_ids(constraint.get("subjects"), objects)
    targets = bound_ids(constraint.get("targets"), objects)
    if len(subjects) < 2 or not targets:
        return None
    by_id = {str(obj["id"]): obj for obj in objects}
    target = by_id.get(targets[0])
    target_center = bbox_center_xy(target) if target is not None else None
    if target_center is None:
        return None
    # Use the target's local side axis, so a rotated coffee table remains a
    # valid flanking reference.  At least one subject must occur on each side.
    yaw = math.radians(float(target.get("yaw_deg") or 0.0))
    side = (-math.sin(yaw), math.cos(yaw))
    signs: list[int] = []
    valid_subjects: list[str] = []
    for subject_id in subjects:
        center = bbox_center_xy(by_id.get(subject_id))
        if center is None:
            continue
        lateral = (center[0] - target_center[0]) * side[0] + (
            center[1] - target_center[1]
        ) * side[1]
        signs.append(1 if lateral > 0.05 else -1 if lateral < -0.05 else 0)
        valid_subjects.append(subject_id)
    label = "pass" if 1 in signs and -1 in signs else "fail"
    return _result(
        constraint,
        suffix=targets[0],
        label=label,
        primary=targets[0],
        related=valid_subjects,
        relation_type="flanking",
        tier=tier,
        reason=(
            f"Prompt requests seating to flank `{targets[0]}`; observed local-side signs "
            f"are {signs}."
        ),
        diagnostics={"subject_ids": valid_subjects, "side_signs": signs},
    )


def _room_bounds(geometry: dict[str, Any]) -> tuple[float, float, float, float] | None:
    rooms = geometry.get("rooms") or []
    if not rooms or not isinstance(rooms[0], dict):
        return None
    bbox = rooms[0].get("bbox") or {}
    lower, upper = bbox.get("min") or [], bbox.get("max") or []
    if len(lower) < 2 or len(upper) < 2:
        return None
    bounds = tuple(float(value) for value in (*lower[:2], *upper[:2]))
    if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        return None
    return bounds


def _centered_wall_pose(
    obj: dict[str, Any],
    walls: list[dict[str, Any]],
    bounds: tuple[float, float, float, float],
) -> tuple[tuple[float, float], float, float, float, float] | None:
    center = bbox_center_xy(obj)
    if center is None:
        return None
    room_center = ((bounds[0] + bounds[2]) / 2.0, (bounds[1] + bounds[3]) / 2.0)
    wall = min(walls, key=lambda item: _wall_distance_sq(center, item))
    wall_bbox = wall.get("bbox_world") or {}
    wall_center = wall_bbox.get("center") or []
    wall_size = wall_bbox.get("size") or []
    obj_size = (obj.get("bbox_world") or {}).get("size") or []
    if len(wall_center) < 2 or len(wall_size) < 2 or len(obj_size) < 2:
        return None
    normal_axis = 0 if float(wall_size[0]) <= float(wall_size[1]) else 1
    tangent_axis = 1 - normal_axis
    wall_coord = float(wall_center[normal_axis])
    inward = 1.0 if room_center[normal_axis] >= wall_coord else -1.0
    wall_half = float(wall_size[normal_axis]) / 2.0
    object_half = float(obj_size[normal_axis]) / 2.0
    target = [float(wall_center[0]), float(wall_center[1])]
    target[normal_axis] = wall_coord + inward * (wall_half + object_half + 0.03)
    yaw = math.degrees(
        math.atan2(
            -(inward if normal_axis == 0 else 0.0), inward if normal_axis == 1 else 0.0
        )
    )
    tangent_error = abs(float(center[tangent_axis]) - float(target[tangent_axis]))
    normal_error = abs(float(center[normal_axis]) - float(target[normal_axis]))
    allowed_tangent = max(
        0.15, 0.08 * min(bounds[2] - bounds[0], bounds[3] - bounds[1])
    )
    return (
        (float(target[0]), float(target[1])),
        yaw,
        tangent_error,
        normal_error,
        allowed_tangent,
    )


def _wall_distance_sq(center: tuple[float, float], wall: dict[str, Any]) -> float:
    wall_center = bbox_center_xy(wall)
    if wall_center is None:
        return math.inf
    return (center[0] - wall_center[0]) ** 2 + (center[1] - wall_center[1]) ** 2


def _tier(constraint: dict[str, Any], mode: str) -> str:
    if mode == "shadow":
        return "ignored"
    return "core" if is_hard_constraint(constraint) else "auxiliary"


def _result(
    constraint: dict[str, Any],
    *,
    suffix: str,
    label: str,
    primary: str,
    related: list[str],
    relation_type: str,
    tier: str,
    reason: str,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "check_id": f"intent_extension__{constraint['constraint_id']}__{suffix}",
        "metric": "functional_dependency",
        "label": label,
        "confidence": 0.96,
        "primary_object": primary,
        "related_objects": related,
        "selected_related_objects": related,
        "blocking_objects": [],
        "relation_type": relation_type,
        "reason": reason,
        "diagnostics": diagnostics,
        "evidence": {"intent_constraint": constraint},
        "evaluation_source": "scenesmith_intent_contract",
        "scoring_tier": tier,
    }
