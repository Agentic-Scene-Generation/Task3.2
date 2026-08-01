"""Geometry evaluators for prompt-originated intent-contract constraints."""

from __future__ import annotations

import math

from typing import Any

from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import nearest_points, unary_union

from scenesmith.scenebenchmark_critic.core.geometry import (
    bbox_center_xy,
    front_vector,
    object_category,
    object_footprint_polygon,
)
from scenesmith.scenebenchmark_critic.intent_contract import (
    bound_ids,
    contract_constraints,
    selected_ids,
    selector_match_count,
)
from scenesmith.scenebenchmark_critic.relation_registry import (
    CEILING_MOUNTED_CATEGORIES,
    MANIPULAND_CATEGORIES,
    ROOM_RELATIVE_WALL_CATEGORIES,
    STAGE_ORDER,
    WALL_MOUNTED_CATEGORIES,
    relation_spec,
)
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.semantics import (
    _is_work_surface_target,
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
    geometry = case_pack.get("scene_geometry") or {}
    objects = [
        item
        for item in geometry.get("objects") or []
        if isinstance(item, dict) and item.get("id")
    ]
    results: list[dict[str, Any]] = []
    for constraint in contract_constraints(case_pack):
        relation = str(constraint.get("relation") or "")
        binding_result = _binding_state_result(case_pack, constraint, objects)
        if binding_result is not None:
            results.append(binding_result)
            continue
        evaluator = _EXTENSION_EVALUATORS.get(relation_spec(relation).evaluator)
        if evaluator is None:
            continue
        result = evaluator(constraint, geometry, objects, case_pack, "core")
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
    same_category_pair = (
        str(targets.get("category") or "") == secondary_category
        and str(targets.get("role") or "")
        == str(targets.get("secondary_role") or "")
    )
    if same_category_pair and first_ids == second_ids and len(first_ids) == 2:
        first_ids, second_ids = [first_ids[0]], [first_ids[1]]
    elif len(first_ids) != 1 or len(second_ids) != 1 or first_ids == second_ids:
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
    match_count = selector_match_count(selector, objects)
    category = str(selector.get("category") or "required_object")
    return _result(
        constraint,
        suffix=category,
        label="pass" if match_count >= required else "fail",
        primary=matched[0] if matched else category,
        related=matched,
        relation_type="required_count",
        tier=tier,
        reason=(
            f"Prompt/room contract requires at least {required} `{category}` object(s); "
            f"found {match_count}."
        ),
        diagnostics={
            "required_count": required,
            "observed_count": match_count,
            "observed_ids": matched,
        },
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


def _evaluate_operation_zone_at_wall(
    constraint: dict[str, Any],
    geometry: dict[str, Any],
    objects: list[dict[str, Any]],
    tier: str,
) -> list[dict[str, Any]]:
    """Center a work surface while reserving its wall-side operator zone."""
    bounds = _room_bounds(geometry)
    subject_ids = bound_ids(constraint.get("subjects"), objects)
    walls = [obj for obj in objects if object_category(obj) == "wall"]
    if bounds is None or len(subject_ids) != 1 or not walls:
        return []
    by_id = {str(obj["id"]): obj for obj in objects}
    subject_id = subject_ids[0]
    subject = by_id.get(subject_id)
    center = bbox_center_xy(subject) if subject is not None else None
    if subject is None or center is None:
        return []

    wall = min(
        walls,
        key=lambda item: (_wall_normal_distance(center, item), str(item.get("id"))),
    )
    frame = _wall_frame(wall, bounds)
    if frame is None:
        return []
    normal_axis, tangent_axis, inward, inner_face, wall_center = frame
    wallward = [0.0, 0.0]
    wallward[normal_axis] = -inward
    wallward_xy = (float(wallward[0]), float(wallward[1]))
    subject_half_depth = _projected_half_extent(subject, front_vector(subject))
    room_depth = bounds[2 + normal_axis] - bounds[normal_axis]
    required_clearance = min(1.0, max(0.65, 0.10 * room_depth))
    target = [float(center[0]), float(center[1])]
    target[tangent_axis] = float(wall_center[tangent_axis])
    target[normal_axis] = inner_face + inward * (
        subject_half_depth + required_clearance
    )
    target_xy = (float(target[0]), float(target[1]))
    target_yaw = math.degrees(math.atan2(-wallward_xy[0], wallward_xy[1]))

    tangent_error = abs(center[tangent_axis] - target_xy[tangent_axis])
    normal_error = abs(center[normal_axis] - target_xy[normal_axis])
    observed_clearance = max(
        0.0,
        inward * (center[normal_axis] - inner_face) - subject_half_depth,
    )
    facing_error = _front_error_deg(subject, wallward_xy)
    allowed_tangent = max(
        0.18, 0.05 * min(bounds[2] - bounds[0], bounds[3] - bounds[1])
    )
    allowed_normal = 0.15
    aligned = (
        tangent_error <= allowed_tangent
        and normal_error <= allowed_normal
        and facing_error <= 15.0
    )
    return [
        _result(
            constraint,
            suffix=f"{subject_id}__{wall['id']}",
            label="pass" if aligned else "fail",
            primary=subject_id,
            related=[str(wall["id"])],
            relation_type="operation_zone_at_wall",
            tier=tier,
            reason=(
                f"`{subject_id}` must be centered on its functional wall with its "
                f"usable front facing the wall and a {required_clearance:.2f}m "
                f"operator zone between them; tangent error is {tangent_error:.2f}m, "
                f"normal error is {normal_error:.2f}m, facing error is "
                f"{facing_error:.0f}deg, and current clearance is "
                f"{observed_clearance:.2f}m."
            ),
            diagnostics={
                "wall_id": str(wall["id"]),
                "target_center_xy_m": [round(value, 6) for value in target_xy],
                "target_yaw_deg": round(target_yaw, 6),
                "tangent_error_m": round(tangent_error, 6),
                "allowed_tangent_error_m": round(allowed_tangent, 6),
                "normal_error_m": round(normal_error, 6),
                "allowed_normal_error_m": allowed_normal,
                "facing_error_deg": round(facing_error, 6),
                "allowed_facing_error_deg": 15.0,
                "operation_clearance_m": round(observed_clearance, 6),
                "required_operation_clearance_m": round(required_clearance, 6),
            },
        )
    ]


def _evaluate_instructional_surface_alignment(
    constraint: dict[str, Any],
    geometry: dict[str, Any],
    objects: list[dict[str, Any]],
    tier: str,
) -> list[dict[str, Any]]:
    """Keep a later-created focal surface on its presenter wall and centerline."""
    bounds = _room_bounds(geometry)
    subject_ids = bound_ids(constraint.get("subjects"), objects)
    target_ids = bound_ids(constraint.get("targets"), objects)
    walls = [obj for obj in objects if object_category(obj) == "wall"]
    if bounds is None or len(subject_ids) != 1 or len(target_ids) != 1 or not walls:
        return []
    by_id = {str(obj["id"]): obj for obj in objects}
    subject_id, target_id = subject_ids[0], target_ids[0]
    subject, target = by_id.get(subject_id), by_id.get(target_id)
    subject_center = bbox_center_xy(subject) if subject is not None else None
    target_center = bbox_center_xy(target) if target is not None else None
    if (
        subject is None
        or target is None
        or subject_center is None
        or target_center is None
    ):
        return []

    subject_wall = min(
        walls,
        key=lambda item: (
            _wall_normal_distance(subject_center, item),
            str(item.get("id")),
        ),
    )
    target_wall = min(
        walls,
        key=lambda item: (
            _wall_normal_distance(target_center, item),
            str(item.get("id")),
        ),
    )
    frame = _wall_frame(target_wall, bounds)
    if frame is None:
        return []
    normal_axis, tangent_axis, inward, inner_face, _ = frame
    target_xy = [float(subject_center[0]), float(subject_center[1])]
    target_xy[tangent_axis] = float(target_center[tangent_axis])
    if str(subject_wall["id"]) == str(target_wall["id"]):
        target_xy[normal_axis] = float(subject_center[normal_axis])
    else:
        target_xy[normal_axis] = inner_face
    inward_vector = [0.0, 0.0]
    inward_vector[normal_axis] = inward
    target_yaw = math.degrees(math.atan2(-inward_vector[0], inward_vector[1]))
    tangent_error = abs(subject_center[tangent_axis] - target_center[tangent_axis])
    same_wall = str(subject_wall["id"]) == str(target_wall["id"])
    facing_error = _front_error_deg(subject, tuple(inward_vector))
    allowed_tangent = max(
        0.15, 0.04 * min(bounds[2] - bounds[0], bounds[3] - bounds[1])
    )
    aligned = same_wall and tangent_error <= allowed_tangent and facing_error <= 20.0
    return [
        _result(
            constraint,
            suffix=f"{subject_id}__{target_id}",
            label="pass" if aligned else "fail",
            primary=subject_id,
            related=[target_id, str(target_wall["id"])],
            relation_type="instructional_surface_alignment",
            tier=tier,
            reason=(
                f"`{subject_id}` must share the presenter wall and centerline of "
                f"`{target_id}`; same-wall={same_wall}, tangent error is "
                f"{tangent_error:.2f}m, and inward-facing error is "
                f"{facing_error:.0f}deg."
            ),
            diagnostics={
                "subject_wall_id": str(subject_wall["id"]),
                "presenter_wall_id": str(target_wall["id"]),
                "target_center_xy_m": [round(value, 6) for value in target_xy],
                "target_yaw_deg": round(target_yaw, 6),
                "same_wall": same_wall,
                "tangent_error_m": round(tangent_error, 6),
                "allowed_tangent_error_m": round(allowed_tangent, 6),
                "facing_error_deg": round(facing_error, 6),
                "allowed_facing_error_deg": 20.0,
            },
        )
    ]


def _evaluate_in_front_of(
    constraint: dict[str, Any],
    *,
    case_pack: dict[str, Any],
    objects: list[dict[str, Any]],
    tier: str,
) -> list[dict[str, Any]]:
    return _evaluate_axial_relation(
        constraint,
        case_pack=case_pack,
        objects=objects,
        tier=tier,
        behind=False,
    )


def _evaluate_behind(
    constraint: dict[str, Any],
    *,
    case_pack: dict[str, Any],
    objects: list[dict[str, Any]],
    tier: str,
) -> list[dict[str, Any]]:
    return _evaluate_axial_relation(
        constraint,
        case_pack=case_pack,
        objects=objects,
        tier=tier,
        behind=True,
    )


def _evaluate_axial_relation(
    constraint: dict[str, Any],
    *,
    case_pack: dict[str, Any],
    objects: list[dict[str, Any]],
    tier: str,
    behind: bool,
) -> list[dict[str, Any]]:
    """Evaluate front/rear placement and lateral centerline alignment.

    The subject must lie along the requested target axis. For a single pair the
    relation also denotes the same centerline, preventing a disconnected
    lateral placement.
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
    if behind:
        front = (-front[0], -front[1])
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
            "target_relation_axis_xy": [round(front[0], 6), round(front[1], 6)],
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
                relation_type=(
                    "rear_axis_alignment" if behind else "front_axis_alignment"
                ),
                tier=tier,
                reason=(
                    f"`{subject_id}` must be "
                    f"{'behind' if behind else 'in front of'} and laterally aligned with "
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


def _binding_state_result(
    case_pack: dict[str, Any],
    constraint: dict[str, Any],
    objects: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Represent missing/ambiguous endpoints instead of silently dropping them."""
    relation = str(constraint.get("relation") or "")
    if relation == "required_count":
        return None
    stage = _normalized_stage(str(case_pack.get("stage") or "adhoc"))
    expected_stage = _constraint_stage(constraint)
    before_expected = STAGE_ORDER.index(stage) < STAGE_ORDER.index(expected_stage)
    subject_selector = constraint.get("subjects") or {}
    entrance_route = relation == "clear_access" and str(
        subject_selector.get("category") or ""
    ) in {"door", "entrance", "entry"}
    if entrance_route:
        subject_matches = [
            str(item.get("id") or item.get("opening_id") or "")
            for item in (
                ((case_pack.get("scene_geometry") or {}).get("scene_shell") or {}).get(
                    "doors"
                )
                or []
            )
            if str(item.get("id") or item.get("opening_id") or "")
        ]
        subject_ids = subject_matches
    else:
        subject_matches = selected_ids(subject_selector, objects)
        subject_ids = bound_ids(subject_selector, objects)
    virtual_targets = {"room", "ceiling"}
    target_selector = constraint.get("targets") or {}
    target_category = str(target_selector.get("category") or "")
    positional_wall_target = (
        relation in {"against_wall", "centered_on_wall"}
        and target_category in ROOM_RELATIVE_WALL_CATEGORIES
    )
    if target_category in virtual_targets:
        target_matches = [target_category]
        target_ids = target_matches
    elif positional_wall_target:
        # The relation evaluator selects the nearest physical wall for each
        # subject. Room-relative labels are semantic selectors, never asset
        # categories such as ``back_wall_0``.
        wall_selector = {"category": "wall", "quantifier": "all"}
        target_matches = selected_ids(wall_selector, objects)
        target_ids = bound_ids(wall_selector, objects)
    else:
        target_matches = selected_ids(target_selector, objects)
        target_ids = bound_ids(target_selector, objects)
    secondary_category = str(target_selector.get("secondary_category") or "")
    secondary_matches: list[str] = []
    secondary_ids: list[str] = []
    if secondary_category:
        secondary_selector = {
            "category": secondary_category,
            "quantifier": "all",
            **(
                {"role": target_selector["secondary_role"]}
                if target_selector.get("secondary_role")
                else {}
            ),
            **(
                {"count": target_selector["secondary_count"]}
                if target_selector.get("secondary_count")
                else {}
            ),
        }
        secondary_matches = selected_ids(secondary_selector, objects)
        secondary_ids = bound_ids(secondary_selector, objects)

    missing = (
        not subject_matches
        or (relation_spec(relation).target_arity > 0 and not target_matches)
        or (relation_spec(relation).target_arity == 2 and not secondary_matches)
    )
    ambiguous = (
        bool(subject_matches and not subject_ids)
        or bool(target_matches and not target_ids)
        or bool(secondary_matches and not secondary_ids)
    )
    pairwise = relation not in {
        "against_wall",
        "centered_on_wall",
        "on_wall",
        "operation_zone_at_wall",
        "flanking",
        "surround",
        "distributed_evenly",
        "one_per_side",
    }
    target_is_existential = str(target_selector.get("quantifier") or "") in {
        "at_least",
        "minimum",
    }
    try:
        subject_count = int(subject_selector.get("count"))
        target_count = int(target_selector.get("count"))
    except (TypeError, ValueError):
        subject_count = target_count = 0
    equal_group = (
        len(subject_ids) > 1
        and len(subject_ids) == len(target_ids)
        and (
            (subject_count > 1 and subject_count == target_count)
            or (
                str(subject_selector.get("quantifier") or "") == "all"
                and str(target_selector.get("quantifier") or "") == "all"
            )
        )
    )
    same_category_anchor_pair = (
        relation in {"between", "centered_between"}
        and str(target_selector.get("category") or "") == secondary_category
        and set(target_ids) == set(secondary_ids)
        and len(target_ids) == 2
    )
    if (
        pairwise
        and relation_spec(relation).target_arity > 0
        and not target_is_existential
        and not equal_group
    ):
        ambiguous = ambiguous or (
            not same_category_anchor_pair
            and (len(target_ids) > 1 or len(secondary_ids) > 1)
        )
    if not missing and not ambiguous:
        return None

    state = "pending" if before_expected and missing else "failed"
    label = "unknown" if state == "pending" else "fail"
    reason_kind = "missing" if missing else "ambiguous"
    return _result(
        constraint,
        suffix="binding",
        label=label,
        primary=(
            subject_ids[0]
            if subject_ids
            else str(subject_selector.get("category") or "unbound")
        ),
        related=target_ids + secondary_ids,
        relation_type=relation,
        tier="auxiliary" if state == "pending" else "core",
        reason=(
            f"Constraint endpoint binding is {reason_kind} at stage `{stage}`; "
            f"earliest required stage is `{expected_stage}`."
        ),
        diagnostics={
            "contract_state": state,
            "current_stage": stage,
            "earliest_stage": expected_stage,
            "subject_matches": subject_matches,
            "target_matches": target_matches,
            "secondary_target_matches": secondary_matches,
            "binding_issue": reason_kind,
        },
        contract_state=state,
    )


def _normalized_stage(stage: str) -> str:
    normalized = str(stage or "").strip().lower()
    aliases = {
        "scene_after_furniture": "furniture",
        "furniture_relation_repair": "furniture",
        "scene_after_wall_objects": "wall_mounted",
        "wall_visual_clearance_repair": "wall_mounted",
        "scene_after_ceiling_objects": "ceiling_mounted",
        "scene_after_manipulands": "manipuland",
        "final_scene": "final",
        "final": "final",
    }
    if normalized in aliases:
        return aliases[normalized]
    if normalized in STAGE_ORDER:
        return normalized
    for token, resolved in (
        ("manipuland", "manipuland"),
        ("ceiling", "ceiling_mounted"),
        ("wall", "wall_mounted"),
        ("furniture", "furniture"),
    ):
        if token in normalized:
            return resolved
    return "final"


def _constraint_stage(constraint: dict[str, Any]) -> str:
    relation = str(constraint.get("relation") or "")
    stage = str(constraint.get("stage") or relation_spec(relation).earliest_stage)
    targets = constraint.get("targets") or {}
    categories = {
        str((constraint.get("subjects") or {}).get("category") or ""),
        str(targets.get("category") or ""),
        str(targets.get("secondary_category") or ""),
    }
    if relation == "on_wall" or categories & WALL_MOUNTED_CATEGORIES:
        stage = "wall_mounted"
    elif relation == "hang_from_ceiling" or categories & CEILING_MOUNTED_CATEGORIES:
        stage = "ceiling_mounted"
    elif (
        relation == "on_top_of"
        and str((constraint.get("subjects") or {}).get("category") or "")
        in MANIPULAND_CATEGORIES
    ):
        stage = "manipuland"
    return stage


def _evaluate_group_distribution(
    constraint: dict[str, Any], objects: list[dict[str, Any]], tier: str
) -> dict[str, Any] | None:
    relation = str(constraint.get("relation") or "")
    subject_ids = bound_ids(constraint.get("subjects"), objects)
    target_ids = bound_ids(constraint.get("targets"), objects)
    by_id = {str(obj["id"]): obj for obj in objects}
    centers = [
        (object_id, bbox_center_xy(by_id.get(object_id))) for object_id in subject_ids
    ]
    centers = [
        (object_id, center) for object_id, center in centers if center is not None
    ]
    if len(centers) < 2 or not target_ids:
        return None
    target_id = target_ids[0]
    target = by_id.get(target_id)
    target_center = bbox_center_xy(target) if target is not None else None
    if relation != "distributed_evenly" and target_center is None:
        return None

    diagnostics: dict[str, Any] = {"subject_ids": [row[0] for row in centers]}
    if relation == "distributed_evenly":
        nearest = []
        for index, (_, center) in enumerate(centers):
            nearest.append(
                min(
                    math.hypot(center[0] - other[0], center[1] - other[1])
                    for other_index, (_, other) in enumerate(centers)
                    if other_index != index
                )
            )
        mean = sum(nearest) / len(nearest)
        coefficient = (
            math.sqrt(sum((value - mean) ** 2 for value in nearest) / len(nearest))
            / mean
            if mean > 1e-6
            else math.inf
        )
        passed = coefficient <= 0.35
        diagnostics.update(
            {"nearest_neighbor_distances_m": nearest, "spacing_cv": coefficient}
        )
    else:
        angles = sorted(
            math.atan2(center[1] - target_center[1], center[0] - target_center[0])
            for _, center in centers
        )
        gaps = [
            (angles[(index + 1) % len(angles)] - angle) % (2.0 * math.pi)
            for index, angle in enumerate(angles)
        ]
        if relation == "surround":
            passed = len(angles) >= 3 and max(gaps) <= math.pi
        else:
            yaw = math.radians(float(target.get("yaw_deg") or 0.0))
            axes = ((math.cos(yaw), math.sin(yaw)), (-math.sin(yaw), math.cos(yaw)))
            sides: set[tuple[int, int]] = set()
            for _, center in centers:
                delta = (center[0] - target_center[0], center[1] - target_center[1])
                projections = (
                    delta[0] * axes[0][0] + delta[1] * axes[0][1],
                    delta[0] * axes[1][0] + delta[1] * axes[1][1],
                )
                axis = 0 if abs(projections[0]) >= abs(projections[1]) else 1
                sides.add((axis, 1 if projections[axis] >= 0 else -1))
            if relation == "one_per_side" and len(centers) == 2:
                # A pair has no way to occupy all four cardinal sides.  It is
                # correctly distributed when both objects occupy opposing
                # sides of the same target-local axis (for example, the two
                # bedside tables flanking a bed).
                occupied_axes = {axis for axis, _sign in sides}
                occupied_signs = {sign for _axis, sign in sides}
                passed = (
                    len(sides) == 2
                    and len(occupied_axes) == 1
                    and occupied_signs == {-1, 1}
                )
            else:
                passed = len(centers) == 4 and len(sides) == 4
            diagnostics["occupied_sides"] = sorted([list(side) for side in sides])
        diagnostics["angular_gaps_deg"] = [math.degrees(gap) for gap in gaps]

    return _result(
        constraint,
        suffix=target_id,
        label="pass" if passed else "fail",
        primary=target_id,
        related=[row[0] for row in centers],
        relation_type=relation,
        tier=tier,
        reason=f"Compiled `{relation}` group relation {'passes' if passed else 'fails'}.",
        diagnostics=diagnostics,
    )


def _evaluate_corner_of_room(
    constraint: dict[str, Any],
    geometry: dict[str, Any],
    objects: list[dict[str, Any]],
    tier: str,
) -> list[dict[str, Any]]:
    bounds = _room_bounds(geometry)
    if bounds is None:
        return []
    corners = (
        (bounds[0], bounds[1]),
        (bounds[0], bounds[3]),
        (bounds[2], bounds[1]),
        (bounds[2], bounds[3]),
    )
    allowed = max(0.5, 0.12 * math.hypot(bounds[2] - bounds[0], bounds[3] - bounds[1]))
    by_id = {str(obj["id"]): obj for obj in objects}
    results = []
    for object_id in bound_ids(constraint.get("subjects"), objects):
        center = bbox_center_xy(by_id.get(object_id))
        if center is None:
            continue
        distance = min(math.hypot(center[0] - x, center[1] - y) for x, y in corners)
        results.append(
            _result(
                constraint,
                suffix=object_id,
                label="pass" if distance <= allowed else "fail",
                primary=object_id,
                related=[],
                relation_type="corner_of_room",
                tier=tier,
                reason=f"`{object_id}` is {distance:.2f}m from its nearest room corner (allowed {allowed:.2f}m).",
                diagnostics={
                    "nearest_corner_distance_m": distance,
                    "allowed_distance_m": allowed,
                },
            )
        )
    return results


def _evaluate_across_from(
    constraint: dict[str, Any], objects: list[dict[str, Any]], tier: str
) -> list[dict[str, Any]]:
    subjects = bound_ids(constraint.get("subjects"), objects)
    targets = bound_ids(constraint.get("targets"), objects)
    if not subjects or not targets:
        return []
    by_id = {str(obj["id"]): obj for obj in objects}
    if set(subjects) == set(targets) and len(subjects) == 2:
        first_id, second_id = subjects
        first = by_id.get(first_id)
        second = by_id.get(second_id)
        first_center = bbox_center_xy(first)
        second_center = bbox_center_xy(second)
        if (
            first is None
            or second is None
            or first_center is None
            or second_center is None
        ):
            return []
        direction = (
            second_center[0] - first_center[0],
            second_center[1] - first_center[1],
        )
        reverse = (-direction[0], -direction[1])
        distance = math.hypot(*direction)
        errors = (_front_error_deg(first, direction), _front_error_deg(second, reverse))
        work_surface_pair = _is_work_surface_target(first) and _is_work_surface_target(
            second
        )
        axis_errors = tuple(min(error, abs(180.0 - error)) for error in errors)
        front_angle = _vector_angle_deg(front_vector(first), front_vector(second))
        opposition_error = abs(180.0 - front_angle)
        passed = distance >= 0.5 and (
            (
                work_surface_pair
                and max(axis_errors) <= 45.0
                and opposition_error <= 45.0
            )
            or (not work_surface_pair and max(errors) <= 45.0)
        )
        orientation_reason = (
            f"work-surface axis errors are {axis_errors[0]:.0f}deg and "
            f"{axis_errors[1]:.0f}deg; front-axis opposition error is "
            f"{opposition_error:.0f}deg"
            if work_surface_pair
            else (
                f"mutual facing errors are {errors[0]:.0f}deg and "
                f"{errors[1]:.0f}deg"
            )
        )
        return [
            _result(
                constraint,
                suffix=f"{first_id}__{second_id}",
                label="pass" if passed else "fail",
                primary=first_id,
                related=[second_id],
                relation_type="across_from",
                tier=tier,
                reason=(
                    f"`{first_id}` and `{second_id}` are {distance:.2f}m apart; "
                    f"{orientation_reason}."
                ),
                diagnostics={
                    "distance_m": distance,
                    "mutual_facing_error_deg": list(errors),
                    "axis_alignment_error_deg": list(axis_errors),
                    "front_axis_opposition_error_deg": opposition_error,
                    "work_surface_pair": work_surface_pair,
                },
            )
        ]
    if len(targets) != 1:
        return []
    target_id = targets[0]
    target_center = bbox_center_xy(by_id.get(target_id))
    if target_center is None:
        return []
    results = []
    for subject_id in subjects:
        subject = by_id.get(subject_id)
        center = bbox_center_xy(subject)
        if subject is None or center is None:
            continue
        direction = (target_center[0] - center[0], target_center[1] - center[1])
        distance = math.hypot(*direction)
        facing_error = _front_error_deg(subject, direction)
        passed = distance >= 0.5 and facing_error <= 45.0
        results.append(
            _result(
                constraint,
                suffix=f"{subject_id}__{target_id}",
                label="pass" if passed else "fail",
                primary=subject_id,
                related=[target_id],
                relation_type="across_from",
                tier=tier,
                reason=f"`{subject_id}` across-from distance is {distance:.2f}m and facing error is {facing_error:.0f}deg.",
                diagnostics={"distance_m": distance, "facing_error_deg": facing_error},
            )
        )
    return results


def _evaluate_clear_access(
    constraint: dict[str, Any],
    objects: list[dict[str, Any]],
    tier: str,
    geometry: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if str((constraint.get("subjects") or {}).get("category") or "") in {
        "door",
        "entrance",
        "entry",
    }:
        return _evaluate_entrance_routes(
            constraint,
            objects,
            tier,
            geometry=geometry or {},
        )
    by_id = {str(obj["id"]): obj for obj in objects}
    subject_ids = bound_ids(constraint.get("subjects"), objects)
    target_ids = bound_ids(constraint.get("targets"), objects)
    if (
        set(subject_ids) == set(target_ids)
        and len(subject_ids) == 2
        and all(
            _is_work_surface_target(by_id.get(object_id)) for object_id in subject_ids
        )
    ):
        return _evaluate_workstation_aisle(
            constraint,
            subject_ids,
            by_id,
            tier,
            geometry=geometry or {},
        )

    required_depth = float(
        relation_spec("clear_access").thresholds.get("min_clearance_m", 0.8)
    )
    results = []
    for subject_id in subject_ids:
        subject = by_id.get(subject_id)
        center = bbox_center_xy(subject)
        if subject is None or center is None:
            continue
        front = front_vector(subject)
        norm = math.hypot(*front)
        if norm <= 1e-6:
            continue
        front = (front[0] / norm, front[1] / norm)
        side = (-front[1], front[0])
        size = (subject.get("bbox_world") or {}).get("size") or [0.6, 0.6]
        width = (
            0.5 * (abs(side[0]) * float(size[0]) + abs(side[1]) * float(size[1])) + 0.3
        )
        blockers = []
        for other_id, other in by_id.items():
            if other_id == subject_id or object_category(other) in {
                "wall",
                "floor",
                "ceiling",
            }:
                continue
            other_center = bbox_center_xy(other)
            if other_center is None:
                continue
            delta = (other_center[0] - center[0], other_center[1] - center[1])
            forward = delta[0] * front[0] + delta[1] * front[1]
            lateral = abs(delta[0] * side[0] + delta[1] * side[1])
            if 0.0 < forward < required_depth and lateral < width:
                blockers.append(other_id)
        results.append(
            _result(
                constraint,
                suffix=subject_id,
                label="pass" if not blockers else "fail",
                primary=subject_id,
                related=blockers,
                relation_type="clear_access",
                tier=tier,
                reason=f"`{subject_id}` access zone has {len(blockers)} blocker(s).",
                diagnostics={
                    "blocking_ids": blockers,
                    "required_depth_m": required_depth,
                    "half_width_m": width,
                },
            )
        )
    return results


def _evaluate_entrance_routes(
    constraint: dict[str, Any],
    objects: list[dict[str, Any]],
    tier: str,
    *,
    geometry: dict[str, Any],
) -> list[dict[str, Any]]:
    """Check connected walkable space from any entrance to each destination."""
    bounds = _room_bounds(geometry)
    doors = ((geometry.get("scene_shell") or {}).get("doors") or [])
    target_ids = bound_ids(constraint.get("targets"), objects)
    if bounds is None or not doors or not target_ids:
        return []

    by_id = {str(obj["id"]): obj for obj in objects}
    clearance = float(
        relation_spec("clear_access").thresholds.get("min_clearance_m", 0.8)
    )
    radius = clearance / 2.0
    floor = box(*bounds).buffer(-radius)
    if floor.is_empty:
        return []

    results: list[dict[str, Any]] = []
    for target_id in target_ids:
        target = by_id.get(target_id)
        target_center = bbox_center_xy(target)
        if target is None or target_center is None:
            continue
        obstacles: list[tuple[str, Polygon]] = []
        for object_id, obj in by_id.items():
            if object_id == target_id or object_category(obj) in {
                "ceiling",
                "floor",
                "wall",
            }:
                continue
            lower = ((obj.get("bbox_world") or {}).get("min") or [])
            if len(lower) >= 3 and float(lower[2]) > 0.25:
                continue
            footprint = object_footprint_polygon(obj)
            if len(footprint) < 3:
                continue
            polygon = Polygon(footprint)
            if polygon.is_empty or polygon.area <= 1e-9:
                continue
            obstacles.append(
                (object_id, polygon.buffer(radius, join_style="mitre"))
            )
        blocked = unary_union([polygon for _, polygon in obstacles]) if obstacles else None
        walkable = floor.difference(blocked) if blocked is not None else floor
        components = (
            list(walkable.geoms)
            if getattr(walkable, "geom_type", "") == "MultiPolygon"
            else [walkable]
        )

        reachable = False
        selected_door = ""
        selected_start: Point | None = None
        selected_end: Point | None = None
        for door in doors:
            center = door.get("center") or door.get("position") or []
            if len(center) < 2 or walkable.is_empty:
                continue
            start, _ = nearest_points(walkable, Point(float(center[0]), float(center[1])))
            end, _ = nearest_points(walkable, Point(*target_center))
            selected_door = str(door.get("id") or door.get("opening_id") or "entrance")
            selected_start, selected_end = start, end
            if any(
                component.distance(start) <= 1e-7
                and component.distance(end) <= 1e-7
                for component in components
                if not component.is_empty
            ):
                reachable = True
                break

        blocking_ids: list[str] = []
        if not reachable and selected_start is not None and selected_end is not None:
            direct = LineString([selected_start, selected_end]).buffer(radius)
            blocking_ids = [
                object_id
                for object_id, polygon in obstacles
                if polygon.intersects(direct)
            ][:8]
        results.append(
            _result(
                constraint,
                suffix=f"{selected_door or 'entrance'}__{target_id}",
                label="pass" if reachable else "fail",
                primary=selected_door or "entrance",
                related=[target_id],
                relation_type="clear_access",
                tier=tier,
                reason=(
                    f"Entrance route to `{target_id}` is "
                    f"{'connected' if reachable else 'blocked'} for a "
                    f"{clearance:.2f}m-wide passage."
                ),
                diagnostics={
                    "evaluation_mode": "entrance_route",
                    "entrance_id": selected_door,
                    "destination_id": target_id,
                    "required_clearance_m": clearance,
                    "blocking_ids": blocking_ids,
                },
            )
        )
    return results


def _evaluate_workstation_aisle(
    constraint: dict[str, Any],
    desk_ids: list[str],
    by_id: dict[str, dict[str, Any]],
    tier: str,
    *,
    geometry: dict[str, Any],
) -> list[dict[str, Any]]:
    first_id, second_id = sorted(desk_ids)
    first, second = by_id[first_id], by_id[second_id]
    first_center = bbox_center_xy(first)
    second_center = bbox_center_xy(second)
    if first_center is None or second_center is None:
        return []
    direction = (
        second_center[0] - first_center[0],
        second_center[1] - first_center[1],
    )
    center_distance = math.hypot(*direction)
    if center_distance <= 1e-6:
        return []
    axis = (direction[0] / center_distance, direction[1] / center_distance)
    side = (-axis[1], axis[0])
    first_depth = _projected_half_extent(first, axis)
    second_depth = _projected_half_extent(second, axis)
    first_width = _projected_half_extent(first, side)
    second_width = _projected_half_extent(second, side)
    free_depth = center_distance - first_depth - second_depth
    half_depth = max(0.0, free_depth / 2.0)
    half_width = min(first_width, second_width)
    midpoint = (
        (first_center[0] + second_center[0]) / 2.0,
        (first_center[1] + second_center[1]) / 2.0,
    )
    aisle = [
        (
            midpoint[0] + axis[0] * longitudinal + side[0] * lateral,
            midpoint[1] + axis[1] * longitudinal + side[1] * lateral,
        )
        for longitudinal, lateral in (
            (-half_depth, -half_width),
            (half_depth, -half_width),
            (half_depth, half_width),
            (-half_depth, half_width),
        )
    ]
    endpoint_ids = {first_id, second_id}
    surface_owners = _support_surface_owners(geometry, by_id, endpoint_ids)
    blockers = []
    for object_id, obj in by_id.items():
        if (
            object_id in endpoint_ids
            or object_category(obj) in {"wall", "floor", "ceiling"}
            or _is_elevated_or_endpoint_supported(obj, surface_owners)
        ):
            continue
        footprint = object_footprint_polygon(obj)
        if footprint and _convex_polygons_overlap(aisle, footprint):
            blockers.append(object_id)

    required_clearance = float(
        relation_spec("clear_access").thresholds.get("min_clearance_m", 0.8)
    )
    sufficient_size = (
        free_depth >= required_clearance and 2.0 * half_width >= required_clearance
    )
    return [
        _result(
            constraint,
            suffix=f"{first_id}__{second_id}",
            label="pass" if sufficient_size and not blockers else "fail",
            primary=first_id,
            related=blockers or [second_id],
            relation_type="clear_access",
            tier=tier,
            reason=(
                f"Workstation aisle between `{first_id}` and `{second_id}` has "
                f"{len(blockers)} blocker(s), {free_depth:.2f}m depth, and "
                f"{2.0 * half_width:.2f}m width."
            ),
            diagnostics={
                "evaluation_mode": "between_workstations",
                "aisle_endpoint_ids": [first_id, second_id],
                "blocking_ids": blockers,
                "free_depth_m": round(free_depth, 6),
                "aisle_width_m": round(2.0 * half_width, 6),
                "required_clearance_m": required_clearance,
            },
        )
    ]


def _projected_half_extent(obj: dict[str, Any], axis: tuple[float, float]) -> float:
    center = bbox_center_xy(obj)
    footprint = object_footprint_polygon(obj)
    if center is None or not footprint:
        return 0.0
    return max(
        abs((point[0] - center[0]) * axis[0] + (point[1] - center[1]) * axis[1])
        for point in footprint
    )


def _support_surface_owners(
    geometry: dict[str, Any],
    by_id: dict[str, dict[str, Any]],
    endpoint_ids: set[str],
) -> dict[str, str]:
    owners: dict[str, str] = {}
    for object_id in endpoint_ids:
        for region in by_id[object_id].get("support_regions") or []:
            region_id = str(region.get("region_id") or "")
            if region_id:
                owners[region_id] = object_id
    for relation in geometry.get("relations") or []:
        if str(relation.get("relation_type") or "") != "placed_on_surface":
            continue
        surface_id = str(relation.get("target_surface_id") or "")
        owner_id = str(relation.get("object") or "")
        if surface_id and owner_id in endpoint_ids:
            owners[surface_id] = owner_id
    return owners


def _is_elevated_or_endpoint_supported(
    obj: dict[str, Any], surface_owners: dict[str, str]
) -> bool:
    placement = obj.get("placement_info") or {}
    if str(placement.get("parent_surface_id") or "") in surface_owners:
        return True
    bbox = obj.get("bbox_world") or {}
    lower = bbox.get("min") or []
    return len(lower) >= 3 and float(lower[2]) > 0.25


def _convex_polygons_overlap(
    first: list[tuple[float, float]], second: list[tuple[float, float]]
) -> bool:
    for polygon in (first, second):
        for index, point in enumerate(polygon):
            next_point = polygon[(index + 1) % len(polygon)]
            axis = (-(next_point[1] - point[1]), next_point[0] - point[0])
            if math.hypot(*axis) <= 1e-9:
                continue
            first_values = [
                vertex[0] * axis[0] + vertex[1] * axis[1] for vertex in first
            ]
            second_values = [
                vertex[0] * axis[0] + vertex[1] * axis[1] for vertex in second
            ]
            if max(first_values) < min(second_values) or max(second_values) < min(
                first_values
            ):
                return False
    return True


def _evaluate_hang_from_ceiling(
    constraint: dict[str, Any],
    geometry: dict[str, Any],
    objects: list[dict[str, Any]],
    tier: str,
) -> list[dict[str, Any]]:
    rooms = geometry.get("rooms") or []
    if not rooms:
        return []
    room_bbox = rooms[0].get("bbox") or {}
    room_max = room_bbox.get("max") or []
    if len(room_max) < 3:
        return []
    ceiling_z = float(room_max[2])
    by_id = {str(obj["id"]): obj for obj in objects}
    results = []
    for subject_id in bound_ids(constraint.get("subjects"), objects):
        bbox = (by_id.get(subject_id) or {}).get("bbox_world") or {}
        upper = bbox.get("max") or []
        if len(upper) < 3:
            continue
        gap = abs(ceiling_z - float(upper[2]))
        results.append(
            _result(
                constraint,
                suffix=subject_id,
                label="pass" if gap <= 0.08 else "fail",
                primary=subject_id,
                related=[],
                relation_type="mounted_to_ceiling",
                tier=tier,
                reason=(
                    f"`{subject_id}` ceiling attachment gap is {gap:.2f}m "
                    "(allowed 0.08m)."
                ),
                diagnostics={"ceiling_gap_m": gap, "allowed_gap_m": 0.08},
            )
        )
    return results


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


def _wall_normal_distance(center: tuple[float, float], wall: dict[str, Any]) -> float:
    wall_bbox = wall.get("bbox_world") or {}
    wall_center = wall_bbox.get("center") or []
    wall_size = wall_bbox.get("size") or []
    if len(wall_center) < 2 or len(wall_size) < 2:
        return math.inf
    normal_axis = 0 if float(wall_size[0]) <= float(wall_size[1]) else 1
    return abs(float(center[normal_axis]) - float(wall_center[normal_axis]))


def _wall_frame(
    wall: dict[str, Any],
    bounds: tuple[float, float, float, float],
) -> tuple[int, int, float, float, tuple[float, float]] | None:
    wall_bbox = wall.get("bbox_world") or {}
    wall_center = wall_bbox.get("center") or []
    wall_size = wall_bbox.get("size") or []
    if len(wall_center) < 2 or len(wall_size) < 2:
        return None
    normal_axis = 0 if float(wall_size[0]) <= float(wall_size[1]) else 1
    tangent_axis = 1 - normal_axis
    room_center = (
        (bounds[0] + bounds[2]) / 2.0,
        (bounds[1] + bounds[3]) / 2.0,
    )
    wall_coord = float(wall_center[normal_axis])
    inward = 1.0 if room_center[normal_axis] >= wall_coord else -1.0
    inner_face = wall_coord + inward * float(wall_size[normal_axis]) / 2.0
    return (
        normal_axis,
        tangent_axis,
        inward,
        inner_face,
        (float(wall_center[0]), float(wall_center[1])),
    )


def _front_error_deg(obj: dict[str, Any], target_front: tuple[float, float]) -> float:
    return _vector_angle_deg(front_vector(obj), target_front)


def _vector_angle_deg(
    current: tuple[float, float], target_front: tuple[float, float]
) -> float:
    current_norm = math.hypot(*current)
    target_norm = math.hypot(*target_front)
    if current_norm <= 1e-6 or target_norm <= 1e-6:
        return 180.0
    dot = (current[0] * target_front[0] + current[1] * target_front[1]) / (
        current_norm * target_norm
    )
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


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
    contract_state: str = "evaluated",
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
        "contract_state": contract_state,
    }


# Relation names are resolved to these evaluator keys by relation_registry.
# The adapters normalize the small signature differences of the geometry helpers.
_EXTENSION_EVALUATORS = {
    "required_count": lambda constraint, _geometry, objects, _case_pack, tier: (
        _evaluate_required_count(constraint, objects, tier)
    ),
    "centered_in_room": lambda constraint, geometry, objects, _case_pack, tier: (
        _evaluate_centered_in_room(constraint, geometry, objects, tier)
    ),
    "centered_on_wall": lambda constraint, geometry, objects, _case_pack, tier: (
        _evaluate_centered_on_wall(constraint, geometry, objects, tier)
    ),
    "centered_between": lambda constraint, _geometry, objects, _case_pack, tier: (
        _evaluate_between(constraint, objects, tier)
    ),
    "between": lambda constraint, _geometry, objects, _case_pack, tier: (
        _evaluate_between(constraint, objects, tier)
    ),
    "in_front_of": lambda constraint, _geometry, objects, case_pack, tier: (
        _evaluate_in_front_of(
            constraint,
            case_pack=case_pack,
            objects=objects,
            tier=tier,
        )
    ),
    "behind": lambda constraint, _geometry, objects, case_pack, tier: (
        _evaluate_behind(
            constraint,
            case_pack=case_pack,
            objects=objects,
            tier=tier,
        )
    ),
    "flanking": lambda constraint, _geometry, objects, _case_pack, tier: (
        _evaluate_flanking(constraint, objects, tier)
    ),
    "distributed_evenly": lambda constraint, _geometry, objects, _case_pack, tier: (
        _evaluate_group_distribution(constraint, objects, tier)
    ),
    "one_per_side": lambda constraint, _geometry, objects, _case_pack, tier: (
        _evaluate_group_distribution(constraint, objects, tier)
    ),
    "surround": lambda constraint, _geometry, objects, _case_pack, tier: (
        _evaluate_group_distribution(constraint, objects, tier)
    ),
    "corner_of_room": lambda constraint, geometry, objects, _case_pack, tier: (
        _evaluate_corner_of_room(constraint, geometry, objects, tier)
    ),
    "across_from": lambda constraint, _geometry, objects, _case_pack, tier: (
        _evaluate_across_from(constraint, objects, tier)
    ),
    "clear_access": lambda constraint, geometry, objects, _case_pack, tier: (
        _evaluate_clear_access(constraint, objects, tier, geometry)
    ),
    "mounted_to_ceiling": lambda constraint, geometry, objects, _case_pack, tier: (
        _evaluate_hang_from_ceiling(constraint, geometry, objects, tier)
    ),
    "operation_zone_at_wall": lambda constraint, geometry, objects, _case_pack, tier: (
        _evaluate_operation_zone_at_wall(constraint, geometry, objects, tier)
    ),
    "instructional_surface_alignment": lambda constraint, geometry, objects, _case_pack, tier: (
        _evaluate_instructional_surface_alignment(constraint, geometry, objects, tier)
    ),
}
