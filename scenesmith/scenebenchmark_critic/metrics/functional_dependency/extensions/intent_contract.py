"""Geometry evaluators for prompt-originated intent-contract constraints."""

from __future__ import annotations

import math

from typing import Any

from scenesmith.scenebenchmark_critic.core.geometry import (
    bbox_center_xy,
    front_vector,
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
            "centered_between",
            "between",
            "in_front_of",
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
        elif relation in {"centered_between", "between"}:
            result = _evaluate_between(constraint, objects, tier)
        elif relation == "in_front_of":
            result = _evaluate_in_front_of(
                constraint,
                case_pack=case_pack,
                objects=objects,
                tier=tier,
            )
        else:
            result = _evaluate_flanking(constraint, objects, tier)
        if result is not None:
            results.extend(result if isinstance(result, list) else [result])
    return results


def _evaluate_between(
    constraint: dict[str, Any], objects: list[dict[str, Any]], tier: str
) -> list[dict[str, Any]]:
    """Evaluate one object against the segment defined by two named anchors."""
    subject_ids = bound_ids(constraint.get("subjects"), objects)
    targets = constraint.get("targets") or {}
    first_ids = bound_ids(targets, objects)
    secondary_category = str(targets.get("secondary_category") or "")
    if not subject_ids or not secondary_category:
        return []
    secondary_selector: dict[str, Any] = {
        "category": secondary_category,
        "quantifier": "all",
    }
    if targets.get("secondary_role"):
        secondary_selector["role"] = targets["secondary_role"]
    if targets.get("secondary_count"):
        secondary_selector["count"] = targets["secondary_count"]
    second_ids = bound_ids(secondary_selector, objects)
    if len(first_ids) != 1 or len(second_ids) != 1 or first_ids == second_ids:
        return []

    by_id = {str(obj["id"]): obj for obj in objects}
    first_center = bbox_center_xy(by_id.get(first_ids[0]))
    second_center = bbox_center_xy(by_id.get(second_ids[0]))
    if first_center is None or second_center is None:
        return []
    segment = (
        second_center[0] - first_center[0],
        second_center[1] - first_center[1],
    )
    segment_length = math.hypot(*segment)
    if segment_length <= 1e-6:
        return []
    axis = (segment[0] / segment_length, segment[1] / segment_length)
    side = (-axis[1], axis[0])
    midpoint = (
        (first_center[0] + second_center[0]) / 2.0,
        (first_center[1] + second_center[1]) / 2.0,
    )
    relation = str(constraint.get("relation") or "between")
    centered = relation == "centered_between"
    lateral_tolerance = max(0.18, min(0.45, 0.12 * segment_length))
    midpoint_tolerance = max(0.15, min(0.35, 0.08 * segment_length))

    results: list[dict[str, Any]] = []
    for subject_id in subject_ids:
        subject_center = bbox_center_xy(by_id.get(subject_id))
        if subject_center is None:
            continue
        from_first = (
            subject_center[0] - first_center[0],
            subject_center[1] - first_center[1],
        )
        longitudinal = from_first[0] * axis[0] + from_first[1] * axis[1]
        fraction = longitudinal / segment_length
        lateral = abs(from_first[0] * side[0] + from_first[1] * side[1])
        midpoint_error = math.hypot(
            subject_center[0] - midpoint[0], subject_center[1] - midpoint[1]
        )
        if centered:
            label = "pass" if midpoint_error <= midpoint_tolerance else "fail"
        elif 0.10 <= fraction <= 0.90 and lateral <= lateral_tolerance:
            label = "pass"
        elif 0.0 <= fraction <= 1.0 and lateral <= 2.0 * lateral_tolerance:
            label = "degraded"
        else:
            label = "fail"
        relation_type = (
            "centered_between_alignment" if centered else "between_alignment"
        )
        results.append(
            _result(
                constraint,
                suffix=f"{subject_id}__{first_ids[0]}__{second_ids[0]}",
                label=label,
                primary=subject_id,
                related=[first_ids[0], second_ids[0]],
                relation_type=relation_type,
                tier=tier,
                reason=(
                    f"`{subject_id}` must be {'centered ' if centered else ''}between "
                    f"`{first_ids[0]}` and `{second_ids[0]}`; segment fraction is "
                    f"{fraction:.2f}, lateral offset is {lateral:.2f}m, and midpoint "
                    f"error is {midpoint_error:.2f}m."
                ),
                diagnostics={
                    "anchor_ids": [first_ids[0], second_ids[0]],
                    "segment_fraction": round(fraction, 6),
                    "lateral_offset_m": round(lateral, 6),
                    "lateral_tolerance_m": round(lateral_tolerance, 6),
                    "midpoint_error_m": round(midpoint_error, 6),
                    "midpoint_tolerance_m": round(midpoint_tolerance, 6),
                    "target_center_xy_m": [
                        round(midpoint[0], 6),
                        round(midpoint[1], 6),
                    ],
                },
            )
        )
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


def _evaluate_in_front_of(
    constraint: dict[str, Any],
    *,
    case_pack: dict[str, Any],
    objects: list[dict[str, Any]],
    tier: str,
) -> list[dict[str, Any]]:
    """Evaluate prompt-explicit front placement and lateral centerline alignment.

    ``in front of`` is directional: the subject must lie along the target's
    usable front axis.  For a single subject/target pair it also denotes the
    same centerline, which prevents a centered rug, chair, or table from being
    disconnected laterally from the object it is explicitly in front of.
    """
    subject_ids = bound_ids(constraint.get("subjects"), objects)
    target_ids = bound_ids(constraint.get("targets"), objects)
    # Multiple target instances are ambiguous without an explicit pairing.
    # Declining that case is safer than binding by scene order or proximity.
    if not subject_ids or len(target_ids) != 1:
        return []
    by_id = {str(obj["id"]): obj for obj in objects}
    target_id = target_ids[0]
    target = by_id.get(target_id)
    target_center = bbox_center_xy(target)
    if target is None or target_center is None:
        return []

    front = front_vector(target)
    front_norm = math.hypot(*front)
    if front_norm <= 1e-6:
        return []
    front = (front[0] / front_norm, front[1] / front_norm)
    side = (-front[1], front[0])
    min_forward = max(0.15, _projected_half_extent(target, front) * 0.70)
    centered_anchor_ids = _centered_anchor_ids(case_pack, objects)

    results: list[dict[str, Any]] = []
    for subject_id in subject_ids:
        subject = by_id.get(subject_id)
        subject_center = bbox_center_xy(subject)
        if subject is None or subject_center is None:
            continue
        delta = (
            subject_center[0] - target_center[0],
            subject_center[1] - target_center[1],
        )
        forward_distance = delta[0] * front[0] + delta[1] * front[1]
        lateral_signed = delta[0] * side[0] + delta[1] * side[1]
        lateral_error = abs(lateral_signed)
        lateral_tolerance = max(
            0.18,
            min(
                0.40,
                0.15
                * (
                    _projected_half_extent(subject, side)
                    + _projected_half_extent(target, side)
                ),
            ),
        )
        forward_error = max(0.0, min_forward - forward_distance)
        if forward_error <= 1e-6 and lateral_error <= lateral_tolerance:
            label = "pass"
        elif forward_distance >= 0.0 and lateral_error <= lateral_tolerance * 2.0:
            label = "degraded"
        else:
            label = "fail"

        repair_object_id, repair_center = _front_alignment_repair_pose(
            subject_id=subject_id,
            subject_center=subject_center,
            target_id=target_id,
            target_center=target_center,
            front=front,
            side=side,
            forward_distance=forward_distance,
            lateral_signed=lateral_signed,
            minimum_forward_distance=min_forward,
            centered_anchor_ids=centered_anchor_ids,
        )
        diagnostics: dict[str, Any] = {
            "subject_id": subject_id,
            "target_id": target_id,
            "target_front_xy": [round(front[0], 6), round(front[1], 6)],
            "target_side_xy": [round(side[0], 6), round(side[1], 6)],
            "forward_distance_m": round(forward_distance, 6),
            "minimum_forward_distance_m": round(min_forward, 6),
            "lateral_offset_m": round(lateral_error, 6),
            "lateral_tolerance_m": round(lateral_tolerance, 6),
        }
        if repair_object_id is not None and repair_center is not None:
            diagnostics["repair_object_id"] = repair_object_id
            diagnostics["repair_target_center_xy_m"] = [
                round(repair_center[0], 6),
                round(repair_center[1], 6),
            ]
        results.append(
            _result(
                constraint,
                suffix=f"{subject_id}__{target_id}",
                label=label,
                primary=subject_id,
                related=[target_id],
                relation_type="front_axis_alignment",
                tier=tier,
                reason=(
                    f"`{subject_id}` must be in front of and laterally aligned with "
                    f"`{target_id}`; forward distance is {forward_distance:.2f}m "
                    f"(minimum {min_forward:.2f}m), lateral offset is "
                    f"{lateral_error:.2f}m (allowed {lateral_tolerance:.2f}m)."
                ),
                diagnostics=diagnostics,
            )
        )
    return results


def _centered_anchor_ids(
    case_pack: dict[str, Any], objects: list[dict[str, Any]]
) -> set[str]:
    """Return objects that another relation may not displace from their center."""
    anchored: set[str] = set()
    for constraint in contract_constraints(
        case_pack,
        relations=("centered_in_room", "centered_on_wall"),
        include_auxiliary=False,
    ):
        anchored.update(bound_ids(constraint.get("subjects"), objects))
    return anchored


def _front_alignment_repair_pose(
    *,
    subject_id: str,
    subject_center: tuple[float, float],
    target_id: str,
    target_center: tuple[float, float],
    front: tuple[float, float],
    side: tuple[float, float],
    forward_distance: float,
    lateral_signed: float,
    minimum_forward_distance: float,
    centered_anchor_ids: set[str],
) -> tuple[str | None, tuple[float, float] | None]:
    """Choose one non-anchor object to move while retaining explicit centers."""
    subject_anchored = subject_id in centered_anchor_ids
    target_anchored = target_id in centered_anchor_ids
    if subject_anchored and target_anchored:
        return None, None

    # Moving a wall-backed target along its tangent preserves its wall gap and
    # yaw.  Use that option only when the centered subject is already in front
    # of it, so the repair cannot solve a lateral error by breaking the prompt's
    # front/back ordering.
    if (
        subject_anchored
        and not target_anchored
        and forward_distance >= minimum_forward_distance
    ):
        return (
            target_id,
            (
                target_center[0] + side[0] * lateral_signed,
                target_center[1] + side[1] * lateral_signed,
            ),
        )

    if target_anchored and subject_anchored:
        return None, None
    desired_forward = max(minimum_forward_distance, forward_distance)
    return (
        subject_id,
        (
            target_center[0] + front[0] * desired_forward,
            target_center[1] + front[1] * desired_forward,
        ),
    )


def _projected_half_extent(
    obj: dict[str, Any], direction: tuple[float, float]
) -> float:
    """Return a conservative horizontal half extent along ``direction``."""
    bbox = obj.get("bbox_world") or {}
    size = bbox.get("size") or []
    if len(size) < 2:
        return 0.0
    try:
        return 0.5 * (
            abs(float(direction[0])) * float(size[0])
            + abs(float(direction[1])) * float(size[1])
        )
    except (TypeError, ValueError):
        return 0.0


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
    # Local +Y is the canonical front axis, so local +X is the side axis.  The
    # old implementation accidentally used +Y and accepted chairs placed at
    # the front/back of an anchor as a valid left/right flank.
    yaw = math.radians(float(target.get("yaw_deg") or 0.0))
    side = (math.cos(yaw), math.sin(yaw))
    signs: list[int] = []
    valid_subjects: list[str] = []
    subject_rows: list[tuple[str, dict[str, Any], float]] = []
    for subject_id in subjects:
        subject = by_id.get(subject_id)
        center = bbox_center_xy(subject)
        if center is None:
            continue
        lateral = (center[0] - target_center[0]) * side[0] + (
            center[1] - target_center[1]
        ) * side[1]
        signs.append(1 if lateral > 0.05 else -1 if lateral < -0.05 else 0)
        valid_subjects.append(subject_id)
        subject_rows.append((subject_id, subject, lateral))
    label = "pass" if 1 in signs and -1 in signs else "fail"
    target_half = _projected_half_extent(target, side)
    gap = max(0.18, 0.12 * target_half)
    target_slots: list[dict[str, Any]] = []
    ordered = sorted(subject_rows, key=lambda row: (row[2], row[0]))
    for index, (subject_id, subject, _) in enumerate(ordered):
        sign = -1.0 if index == 0 else 1.0
        distance = target_half + _projected_half_extent(subject, side) + gap
        slot = (
            target_center[0] + sign * side[0] * distance,
            target_center[1] + sign * side[1] * distance,
        )
        to_target = (target_center[0] - slot[0], target_center[1] - slot[1])
        target_slots.append(
            {
                "object_id": subject_id,
                "target_center_xy_m": [round(slot[0], 6), round(slot[1], 6)],
                "target_yaw_deg": round(
                    math.degrees(math.atan2(-to_target[0], to_target[1])), 6
                ),
            }
        )
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
        diagnostics={
            "subject_ids": valid_subjects,
            "side_signs": signs,
            "target_side_xy": [round(side[0], 6), round(side[1], 6)],
            "target_slots": target_slots,
        },
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
