"""Geometry evaluators for prompt-originated intent-contract constraints."""

from __future__ import annotations

import math
import re
from itertools import permutations

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
    _normalize_selector_category,
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
    _is_media_target,
    _is_seating_subject,
    _is_work_surface_target,
)
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.relations import (
    _eval_facing_relation,
    _media_lateral_axis_diagnostics,
)
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.seat_surface_assignment import (
    assign_work_seats_to_surfaces,
    room_bounds_from_case_pack,
)


_ENTRANCE_CATEGORIES = frozenset({"door", "entrance", "entry"})
_VIRTUAL_ROUTE_CATEGORIES = frozenset(
    {
        "access_route",
        "circulation",
        "circulation_path",
        "entry_route",
        "entrance_route",
        "path",
        "route",
        "walking_path",
        "walking_route",
        "walkway",
    }
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
        if relation == "edge_distribution":
            # The dedicated edge evaluator owns complete subject binding and
            # slot assignment.  It must run before generic relation binding.
            continue
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
    same_category_pair = str(
        targets.get("category") or ""
    ) == secondary_category and str(targets.get("role") or "") == str(
        targets.get("secondary_role") or ""
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
    exact = str(selector.get("quantifier") or "").lower() == "exactly"
    category = str(selector.get("category") or "required_object")
    if _normalize_selector_category(category) == "table_setting":
        component_selectors = {
            "plate": {"category": "plate", "quantifier": "at_least"},
            "cutlery": {"category": "cutlery", "quantifier": "at_least"},
            "glass": {"category": "glass", "quantifier": "at_least"},
        }
        component_counts = {
            name: selector_match_count(component, objects)
            for name, component in component_selectors.items()
        }
        match_count = min(component_counts.values(), default=0)
        matched = sorted(
            {
                object_id
                for component in component_selectors.values()
                for object_id in selected_ids(component, objects)
            }
        )
        return _result(
            constraint,
            suffix=category,
            label="pass" if match_count >= required else "fail",
            primary=matched[0] if matched else category,
            related=matched,
            relation_type="required_count",
            tier=tier,
            reason=(
                f"Prompt/room contract requires at least {required} usable table "
                f"setting(s); component counts are {component_counts} and support "
                f"{match_count} complete setting(s)."
            ),
            diagnostics={
                "required_count": required,
                "observed_count": match_count,
                "component_counts": component_counts,
                "observed_ids": matched,
            },
        )

    matched = selected_ids(selector, objects)
    match_count = selector_match_count(selector, objects)
    count_ok = match_count == required if exact else match_count >= required
    requirement = "exactly" if exact else "at least"
    return _result(
        constraint,
        suffix=category,
        label="pass" if count_ok else "fail",
        primary=matched[0] if matched else category,
        related=matched,
        relation_type="required_count",
        tier=tier,
        reason=(
            f"Prompt/room contract requires {requirement} {required} `{category}` object(s); "
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


def _evaluate_faces_room(
    constraint: dict[str, Any],
    geometry: dict[str, Any],
    objects: list[dict[str, Any]],
    tier: str,
) -> list[dict[str, Any]]:
    """Evaluate a prompt's ``faces into the room`` clause against room center."""
    target_category = _normalize_selector_category(
        (constraint.get("targets") or {}).get("category")
    )
    if target_category != "room":
        return []
    bounds = _room_bounds(geometry)
    if bounds is None:
        return []
    room_center = ((bounds[0] + bounds[2]) / 2.0, (bounds[1] + bounds[3]) / 2.0)
    allowed_error = float(relation_spec("faces").thresholds.get("max_angle_deg", 60.0))
    by_id = {str(obj["id"]): obj for obj in objects}
    results: list[dict[str, Any]] = []
    for object_id in bound_ids(constraint.get("subjects"), objects):
        obj = by_id.get(object_id)
        center = bbox_center_xy(obj) if obj is not None else None
        if center is None:
            continue
        desired = (room_center[0] - center[0], room_center[1] - center[1])
        error = (
            0.0
            if math.hypot(*desired) <= 1e-6
            else _vector_angle_deg(front_vector(obj), desired)
        )
        results.append(
            _result(
                constraint,
                suffix=object_id,
                label="pass" if error <= allowed_error else "fail",
                primary=object_id,
                related=[],
                relation_type="faces",
                tier=tier,
                reason=(
                    f"`{object_id}` faces into the room."
                    if error <= allowed_error
                    else f"`{object_id}` faces {error:.0f} degrees away from the room interior."
                ),
                diagnostics={
                    "target_center_xy_m": [round(center[0], 6), round(center[1], 6)],
                    "facing_target_xy_m": [
                        round(room_center[0], 6),
                        round(room_center[1], 6),
                    ],
                    "current_front_xy": list(front_vector(obj)),
                    "facing_error_deg": round(error, 6),
                    "allowed_facing_error_deg": allowed_error,
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


def _evaluate_centered_above(
    constraint: dict[str, Any],
    geometry: dict[str, Any],
    objects: list[dict[str, Any]],
    tier: str,
) -> list[dict[str, Any]]:
    """Evaluate a horizontally centered object above another object."""
    subject_ids = bound_ids(constraint.get("subjects"), objects)
    target_ids = bound_ids(constraint.get("targets"), objects)
    if not subject_ids or not target_ids:
        return []
    by_id = {str(obj["id"]): obj for obj in objects}
    bounds = _room_bounds(geometry)
    walls = [obj for obj in objects if object_category(obj) == "wall"]
    results: list[dict[str, Any]] = []
    for subject_id in subject_ids:
        subject = by_id.get(subject_id)
        subject_center = bbox_center_xy(subject) if subject is not None else None
        subject_box = (subject or {}).get("bbox_world") or {}
        subject_min = subject_box.get("min") or []
        if subject_center is None or len(subject_min) < 3:
            continue
        best_target: tuple[dict[str, Any], float, str, str] | None = None
        for target_id in target_ids:
            target = by_id.get(target_id)
            target_center = bbox_center_xy(target) if target is not None else None
            target_box = (target or {}).get("bbox_world") or {}
            target_max = target_box.get("max") or []
            if target_center is None or len(target_max) < 3:
                continue
            alignment_axis = "xy"
            alignment_error = math.hypot(
                subject_center[0] - target_center[0],
                subject_center[1] - target_center[1],
            )
            if bounds is not None and walls:
                subject_wall = min(
                    walls, key=lambda wall: _wall_normal_distance(subject_center, wall)
                )
                target_wall = min(
                    walls, key=lambda wall: _wall_normal_distance(target_center, wall)
                )
                frame = _wall_frame(subject_wall, bounds)
                if frame is not None and str(subject_wall["id"]) == str(
                    target_wall["id"]
                ):
                    _, tangent_axis, _, _, _ = frame
                    alignment_error = abs(
                        subject_center[tangent_axis] - target_center[tangent_axis]
                    )
                    alignment_axis = "wall_tangent"
            candidate = (target, alignment_error, alignment_axis, target_id)
            if best_target is None or candidate[1] < best_target[1]:
                best_target = candidate
        if best_target is None:
            continue
        target, alignment_error, alignment_axis, target_id = best_target
        target_box = target.get("bbox_world") or {}
        target_max = target_box.get("max") or []
        target_size = target_box.get("size") or []
        subject_size = subject_box.get("size") or []
        size_scale = max(
            [
                *(
                    float(value)
                    for value in target_size[:2]
                    if isinstance(value, (int, float))
                ),
                *(
                    float(value)
                    for value in subject_size[:2]
                    if isinstance(value, (int, float))
                ),
                0.0,
            ]
        )
        allowed_alignment = max(0.15, min(0.35, 0.20 * size_scale))
        vertical_gap = float(subject_min[2]) - float(target_max[2])
        label = (
            "pass"
            if alignment_error <= allowed_alignment and vertical_gap >= -0.03
            else "fail"
        )
        results.append(
            _result(
                constraint,
                suffix=f"{subject_id}__{target_id}",
                label=label,
                primary=subject_id,
                related=[target_id],
                relation_type="centered_above",
                tier=tier,
                reason=(
                    f"`{subject_id}` must be centered above `{target_id}`; "
                    f"{alignment_axis} alignment error is {alignment_error:.2f}m "
                    f"(allowed {allowed_alignment:.2f}m) and vertical gap is "
                    f"{vertical_gap:.2f}m."
                ),
                diagnostics={
                    "alignment_axis": alignment_axis,
                    "alignment_error_m": round(alignment_error, 6),
                    "allowed_alignment_error_m": round(allowed_alignment, 6),
                    "vertical_gap_m": round(vertical_gap, 6),
                    "subject_bottom_z_m": round(float(subject_min[2]), 6),
                    "target_top_z_m": round(float(target_max[2]), 6),
                },
            )
        )
    return results


def _evaluate_mounted_to_wall(
    constraint: dict[str, Any],
    geometry: dict[str, Any],
    objects: list[dict[str, Any]],
    tier: str,
) -> list[dict[str, Any]]:
    """Verify a mounted object is adjacent to and faces into a physical wall."""
    bounds = _room_bounds(geometry)
    walls = [obj for obj in objects if object_category(obj) == "wall"]
    if bounds is None or not walls:
        return []
    by_id = {str(obj["id"]): obj for obj in objects}
    results: list[dict[str, Any]] = []
    for subject_id in bound_ids(constraint.get("subjects"), objects):
        subject = by_id.get(subject_id)
        center = bbox_center_xy(subject) if subject is not None else None
        if center is None:
            continue
        wall = min(walls, key=lambda item: _wall_normal_distance(center, item))
        frame = _wall_frame(wall, bounds)
        if frame is None:
            continue
        normal_axis, _, inward, inner_face, _ = frame
        subject_box = (subject or {}).get("bbox_world") or {}
        subject_size = subject_box.get("size") or []
        normal_size = (
            float(subject_size[normal_axis]) if len(subject_size) > normal_axis else 0.0
        )
        wall_gap = abs(float(center[normal_axis]) - inner_face)
        allowed_gap = max(0.10, normal_size / 2.0 + 0.08)
        inward_vector = (inward, 0.0) if normal_axis == 0 else (0.0, inward)
        facing_error = _front_error_deg(subject, inward_vector)
        label = "pass" if wall_gap <= allowed_gap and facing_error <= 25.0 else "fail"
        results.append(
            _result(
                constraint,
                suffix=f"{subject_id}__{wall['id']}",
                label=label,
                primary=subject_id,
                related=[str(wall["id"])],
                relation_type="mounted_to_wall",
                tier=tier,
                reason=(
                    f"`{subject_id}` must be mounted to `{wall['id']}` and face "
                    f"into the room; wall gap is {wall_gap:.2f}m (allowed "
                    f"{allowed_gap:.2f}m) and facing error is {facing_error:.0f}deg."
                ),
                diagnostics={
                    "wall_id": str(wall["id"]),
                    "wall_gap_m": round(wall_gap, 6),
                    "allowed_wall_gap_m": round(allowed_gap, 6),
                    "facing_error_deg": round(facing_error, 6),
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
    """Keep a wall-backed work surface and its inward operator zone usable."""
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
    inward_vector = [0.0, 0.0]
    inward_vector[normal_axis] = inward
    inward_xy = (float(inward_vector[0]), float(inward_vector[1]))
    subject_half_depth = _projected_half_extent(subject, inward_xy)
    room_depth = bounds[2 + normal_axis] - bounds[normal_axis]
    required_clearance = min(1.0, max(0.65, 0.10 * room_depth))
    # Asset extents and placement tools each round independently. Treat a
    # centimeter-scale miss at an otherwise unobstructed ergonomic boundary as
    # numerical noise, while preserving meaningful blocking failures.
    clearance_tolerance = 0.02
    target_wall_gap = 0.08
    allowed_wall_gap = min(0.30, max(0.15, 0.04 * room_depth))
    target = [float(center[0]), float(center[1])]
    target[tangent_axis] = float(wall_center[tangent_axis])
    target[normal_axis] = inner_face + inward * (subject_half_depth + target_wall_gap)
    target_xy = (float(target[0]), float(target[1]))
    target_yaw = math.degrees(math.atan2(-inward_xy[0], inward_xy[1]))

    tangent_error = abs(center[tangent_axis] - target_xy[tangent_axis])
    wall_gap = max(
        0.0,
        inward * (center[normal_axis] - inner_face) - subject_half_depth,
    )
    observed_clearance, blocking_object_id = _front_operation_clearance(
        subject=subject,
        objects=objects,
        bounds=bounds,
        inward_xy=inward_xy,
    )
    facing_error = _front_error_deg(subject, inward_xy)
    allowed_tangent = max(
        0.18, 0.05 * min(bounds[2] - bounds[0], bounds[3] - bounds[1])
    )
    aligned = (
        tangent_error <= allowed_tangent
        and wall_gap <= allowed_wall_gap
        and facing_error <= 15.0
        and observed_clearance + clearance_tolerance >= required_clearance
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
                f"rear edge near the wall, usable front facing into the room, "
                f"and a {required_clearance:.2f}m clear operator zone in front; "
                f"tangent error is {tangent_error:.2f}m, wall gap is "
                f"{wall_gap:.2f}m, facing error is {facing_error:.0f}deg, and "
                f"front clearance is "
                f"{observed_clearance:.2f}m."
            ),
            diagnostics={
                "wall_id": str(wall["id"]),
                "target_center_xy_m": [round(value, 6) for value in target_xy],
                "target_yaw_deg": round(target_yaw, 6),
                "tangent_error_m": round(tangent_error, 6),
                "allowed_tangent_error_m": round(allowed_tangent, 6),
                "wall_gap_m": round(wall_gap, 6),
                "allowed_wall_gap_m": round(allowed_wall_gap, 6),
                "facing_error_deg": round(facing_error, 6),
                "allowed_facing_error_deg": 15.0,
                "operation_clearance_m": round(observed_clearance, 6),
                "required_operation_clearance_m": round(required_clearance, 6),
                "operation_clearance_tolerance_m": clearance_tolerance,
                "blocking_object_id": blocking_object_id,
            },
        )
    ]


def _front_operation_clearance(
    *,
    subject: dict[str, Any],
    objects: list[dict[str, Any]],
    bounds: tuple[float, float, float, float],
    inward_xy: tuple[float, float],
) -> tuple[float, str | None]:
    """Measure clear depth before a workstation within its usable front band."""
    subject_id = str(subject.get("id") or "")
    center = bbox_center_xy(subject)
    if center is None:
        return 0.0, None
    normal_axis = 0 if abs(inward_xy[0]) > abs(inward_xy[1]) else 1
    tangent_axis = 1 - normal_axis
    tangent_xy = [0.0, 0.0]
    tangent_xy[tangent_axis] = 1.0
    tangent_vector = (float(tangent_xy[0]), float(tangent_xy[1]))
    subject_normal_half = _projected_half_extent(subject, inward_xy)
    subject_tangent_half = _projected_half_extent(subject, tangent_vector)
    room_limit = (
        bounds[2 + normal_axis] if inward_xy[normal_axis] > 0 else bounds[normal_axis]
    )
    front_edge = center[normal_axis] + inward_xy[normal_axis] * subject_normal_half
    nearest_clearance = max(0.0, inward_xy[normal_axis] * (room_limit - front_edge))
    blocking_object_id: str | None = None

    for other in objects:
        if str(other.get("id") or "") == subject_id:
            continue
        if str(other.get("object_type") or "").lower() != "furniture":
            continue
        other_center = bbox_center_xy(other)
        if other_center is None:
            continue
        other_normal_half = _projected_half_extent(other, inward_xy)
        other_tangent_half = _projected_half_extent(other, tangent_vector)
        forward_distance = (other_center[0] - center[0]) * inward_xy[0] + (
            other_center[1] - center[1]
        ) * inward_xy[1]
        if forward_distance + other_normal_half <= subject_normal_half:
            continue
        tangent_distance = abs(
            (other_center[0] - center[0]) * tangent_vector[0]
            + (other_center[1] - center[1]) * tangent_vector[1]
        )
        if tangent_distance > subject_tangent_half + other_tangent_half:
            continue
        clearance = max(0.0, forward_distance - subject_normal_half - other_normal_half)
        if clearance < nearest_clearance:
            nearest_clearance = clearance
            blocking_object_id = str(other.get("id") or "") or None
    return nearest_clearance, blocking_object_id


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
    target_is_existential = str(
        (constraint.get("targets") or {}).get("quantifier") or ""
    ) in {"at_least", "minimum"}
    paired_targets: dict[str, str] = {}
    if len(target_ids) > 1 and not target_is_existential:
        paired_targets = _hard_paired_axial_targets(
            case_pack,
            constraint,
            objects,
            subject_ids,
            target_ids,
        )
    # Multiple universal targets remain ambiguous without explicit pairing.
    # For an existential target, geometry can select the best satisfying
    # candidate without inventing a semantic identity. Equal work-seat groups
    # reuse the globally resolved prompt-authorized one-to-one assignment.
    if (
        not subject_ids
        or not target_ids
        or (
            len(target_ids) != 1
            and not target_is_existential
            and len(paired_targets) != len(subject_ids)
        )
    ):
        return []
    by_id = {str(obj["id"]): obj for obj in objects}
    centered_anchor_ids = _centered_anchor_ids(case_pack, objects)

    results: list[dict[str, Any]] = []
    for subject_id in subject_ids:
        subject = by_id.get(subject_id)
        subject_center = bbox_center_xy(subject)
        if subject is None or subject_center is None:
            continue
        candidate_target_ids = (
            [paired_targets[subject_id]] if subject_id in paired_targets else target_ids
        )
        candidates = [
            candidate
            for target_id in candidate_target_ids
            if (
                candidate := _axial_candidate(
                    subject,
                    subject_center,
                    by_id.get(target_id),
                    target_id,
                    behind=behind,
                )
            )
            is not None
        ]
        if not candidates:
            continue
        candidate = min(
            candidates,
            key=lambda item: (
                {"pass": 0, "degraded": 1, "fail": 2}[item["label"]],
                item["forward_error"],
                item["lateral_error"],
                item["target_id"],
            ),
        )
        target_id = candidate["target_id"]
        target_center = candidate["target_center"]
        front = candidate["front"]
        side = candidate["side"]
        min_forward = candidate["min_forward"]
        forward_distance = candidate["forward_distance"]
        lateral_signed = candidate["lateral_signed"]
        lateral_error = candidate["lateral_error"]
        lateral_tolerance = candidate["lateral_tolerance"]
        label = candidate["label"]

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
        if paired_targets:
            diagnostics["group_pairing"] = "global_minimum_cost_one_to_one"
            diagnostics["paired_target_id"] = target_id
        elif len(target_ids) > 1:
            diagnostics["candidate_target_ids"] = list(target_ids)
            diagnostics["existential_target_selection"] = True
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


def _hard_paired_axial_targets(
    case_pack: dict[str, Any],
    constraint: dict[str, Any],
    objects: list[dict[str, Any]],
    subject_ids: list[str],
    target_ids: list[str],
) -> dict[str, str]:
    """Resolve an equal work-seat group only when the prompt authorizes pairing."""
    if len(subject_ids) <= 1 or len(subject_ids) != len(target_ids):
        return {}
    by_id = {str(obj.get("id") or ""): obj for obj in objects}
    if not all(_is_seating_subject(by_id.get(item) or {}) for item in subject_ids):
        return {}
    if not all(_is_work_surface_target(by_id.get(item)) for item in target_ids):
        return {}

    subject_set = set(subject_ids)
    target_set = set(target_ids)
    pairing_authorized = False
    for pairing in contract_constraints(
        case_pack,
        relations=("paired_with",),
        include_auxiliary=False,
    ):
        if str(pairing.get("strength") or "hard").lower() != "hard":
            continue
        pairing_subjects = set(bound_ids(pairing.get("subjects"), objects))
        pairing_targets = set(bound_ids(pairing.get("targets"), objects))
        if (pairing_subjects == subject_set and pairing_targets == target_set) or (
            pairing_subjects == target_set and pairing_targets == subject_set
        ):
            pairing_authorized = True
            break
    if not pairing_authorized:
        return {}

    assignments = assign_work_seats_to_surfaces(
        objects,
        task_instruction=str(
            case_pack.get("task_instruction")
            or case_pack.get("original_task_instruction")
            or ""
        ),
        room_type=str(case_pack.get("room_type") or ""),
        room_bounds=room_bounds_from_case_pack(case_pack),
    )
    resolved = {
        assignment.seat_id: assignment.surface_id
        for assignment in assignments
        if assignment.seat_id in subject_set and assignment.surface_id in target_set
    }
    if set(resolved) != subject_set or set(resolved.values()) != target_set:
        return {}
    return resolved


def _axial_candidate(
    subject: dict[str, Any],
    subject_center: tuple[float, float],
    target: dict[str, Any] | None,
    target_id: str,
    *,
    behind: bool,
) -> dict[str, Any] | None:
    target_center = bbox_center_xy(target)
    if target is None or target_center is None:
        return None
    front = front_vector(target)
    front_norm = math.hypot(*front)
    if front_norm <= 1e-6:
        return None
    front = (front[0] / front_norm, front[1] / front_norm)
    if behind:
        front = (-front[0], -front[1])
    side = (-front[1], front[0])
    # A center-to-center threshold based only on the target's extent can mark
    # an overlapping subject as "in front".  Include both projected half
    # extents so a deterministic repair has room to place a real object in the
    # target's usable front zone without creating a collision.
    min_forward = max(
        0.15,
        _projected_half_extent(target, front)
        + _projected_half_extent(subject, front)
        + 0.03,
    )
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
    return {
        "target_id": target_id,
        "target_center": target_center,
        "front": front,
        "side": side,
        "min_forward": min_forward,
        "forward_distance": forward_distance,
        "lateral_signed": lateral_signed,
        "lateral_error": lateral_error,
        "lateral_tolerance": lateral_tolerance,
        "forward_error": forward_error,
        "label": label,
    }


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
    entrance_route = relation == "clear_access" and (
        _is_entrance_selector(subject_selector)
        or _is_virtual_route_selector(subject_selector)
    )
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
        if not subject_matches and subject_ids:
            subject_matches = subject_ids
    virtual_targets = {"room", "ceiling"}
    target_selector = constraint.get("targets") or {}
    target_category = str(target_selector.get("category") or "")
    positional_wall_target = relation in {
        "against_wall",
        "centered_on_wall",
        "on_wall",
    } and target_category in {"wall", *ROOM_RELATIVE_WALL_CATEGORIES}
    if (
        _is_virtual_route_selector(subject_selector)
        and _normalize_selector_category(target_category) in _ENTRANCE_CATEGORIES
    ):
        # Compiler-shaped ``route -> entrance`` describes connectivity from
        # the entrance into the room. Neither endpoint is a generated object.
        target_matches = ["room"]
        target_ids = target_matches
    elif target_category in virtual_targets:
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
        if not target_matches and target_ids:
            target_matches = target_ids
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
        if not secondary_matches and secondary_ids:
            secondary_matches = secondary_ids

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
        "edge_distribution",
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
        _normalize_selector_category(
            str((constraint.get("subjects") or {}).get("category") or "")
        ),
        _normalize_selector_category(str(targets.get("category") or "")),
        _normalize_selector_category(str(targets.get("secondary_category") or "")),
    }
    if relation == "on_wall" or categories & WALL_MOUNTED_CATEGORIES:
        stage = "wall_mounted"
    elif relation == "hang_from_ceiling" or categories & CEILING_MOUNTED_CATEGORIES:
        stage = "ceiling_mounted"
    elif categories & MANIPULAND_CATEGORIES or "table_setting" in categories:
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
            if relation == "edge_distribution" and len(centers) == 2:
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


def _evaluate_corner_distribution(
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
    subject_ids = bound_ids(constraint.get("subjects"), objects)
    if not subject_ids:
        return []
    by_id = {str(obj["id"]): obj for obj in objects}
    centers = [bbox_center_xy(by_id.get(object_id)) for object_id in subject_ids]
    if any(center is None for center in centers) or len(subject_ids) > len(corners):
        return []
    typed_centers = [center for center in centers if center is not None]
    best: tuple[float, tuple[int, ...]] | None = None
    for assignment in permutations(range(len(corners)), len(subject_ids)):
        cost = sum(
            math.hypot(
                typed_centers[index][0] - corners[corner_index][0],
                typed_centers[index][1] - corners[corner_index][1],
            )
            for index, corner_index in enumerate(assignment)
        )
        if best is None or cost < best[0]:
            best = (cost, assignment)
    if best is None:
        return []
    allowed = max(
        0.5,
        0.12 * math.hypot(bounds[2] - bounds[0], bounds[3] - bounds[1]),
    )
    results: list[dict[str, Any]] = []
    for index, corner_index in enumerate(best[1]):
        center = typed_centers[index]
        corner = corners[corner_index]
        distance = math.hypot(center[0] - corner[0], center[1] - corner[1])
        assignment = {
            "object_id": subject_ids[index],
            "corner_index": corner_index,
            "target_corner_xy_m": [round(corner[0], 6), round(corner[1], 6)],
            "distance_m": round(distance, 6),
        }
        results.append(
            _result(
                constraint,
                suffix=subject_ids[index],
                label="pass" if distance <= allowed else "fail",
                primary=subject_ids[index],
                related=[
                    object_id
                    for object_id in subject_ids
                    if object_id != subject_ids[index]
                ],
                relation_type="corner_distribution",
                tier=tier,
                reason=(
                    f"`{subject_ids[index]}` is {distance:.2f}m from its assigned "
                    f"distinct room corner (allowed {allowed:.2f}m)."
                ),
                diagnostics={
                    "assignment": assignment,
                    "distinct_corner_count": len(set(best[1])),
                    "allowed_distance_m": allowed,
                },
            )
        )
    return results


def _evaluate_one_per_support(
    constraint: dict[str, Any],
    geometry: dict[str, Any],
    objects: list[dict[str, Any]],
    tier: str,
) -> dict[str, Any] | None:
    subject_ids = bound_ids(constraint.get("subjects"), objects)
    target_ids = bound_ids(constraint.get("targets"), objects)
    if not subject_ids or not target_ids:
        return None
    by_id = {str(obj["id"]): obj for obj in objects}
    target_set = set(target_ids)
    surface_owners = _support_surface_owners(geometry, by_id, target_set)
    assignments: dict[str, str] = {}
    counts = {target_id: 0 for target_id in target_ids}
    for subject_id in subject_ids:
        placement = by_id.get(subject_id, {}).get("placement_info") or {}
        surface_id = str(placement.get("parent_surface_id") or "")
        owner_id = surface_owners.get(surface_id, "")
        if owner_id in target_set:
            assignments[subject_id] = owner_id
            counts[owner_id] += 1
    passed = len(assignments) == len(subject_ids) and all(
        count == 1 for count in counts.values()
    )
    unassigned = [
        subject_id for subject_id in subject_ids if subject_id not in assignments
    ]
    duplicates = [target_id for target_id, count in counts.items() if count > 1]
    missing_targets = [target_id for target_id, count in counts.items() if count == 0]
    duplicate_subjects = [
        subject_id
        for target_id in duplicates
        for subject_id, owner_id in assignments.items()
        if owner_id == target_id
    ]
    repair_subject_id = (
        unassigned[0]
        if unassigned
        else (duplicate_subjects[-1] if duplicate_subjects else subject_ids[0])
    )
    return _result(
        constraint,
        suffix="one_per_support",
        label="pass" if passed else "fail",
        primary=repair_subject_id,
        related=target_ids,
        relation_type="one_per_support",
        tier=tier,
        reason=(
            f"Matched {len(assignments)}/{len(subject_ids)} subject(s) to distinct "
            f"support owners; missing targets={missing_targets}, duplicates={duplicates}."
        ),
        diagnostics={
            "support_assignments": assignments,
            "target_subject_counts": counts,
            "unassigned_subject_ids": unassigned,
            "missing_target_ids": missing_targets,
            "duplicate_target_ids": duplicates,
            "repair_subject_id": repair_subject_id,
        },
    )


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
        target = by_id.get(target_id)
        center = bbox_center_xy(subject)
        if subject is None or target is None or center is None:
            continue
        if _is_seating_subject(subject) and _is_media_target(target):
            label, _confidence, reason = _eval_facing_relation(
                subject, target, "seating_to_media"
            )
            axis = _media_lateral_axis_diagnostics(subject, target)
            diagnostics = axis[3] if axis is not None else {}
            diagnostics["distance_m"] = math.hypot(
                target_center[0] - center[0], target_center[1] - center[1]
            )
            results.append(
                _result(
                    constraint,
                    suffix=f"{subject_id}__{target_id}",
                    label=label,
                    primary=subject_id,
                    related=[target_id],
                    relation_type="seating_to_media",
                    tier=tier,
                    reason=reason,
                    diagnostics=diagnostics,
                )
            )
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
    *,
    case_pack: dict[str, Any],
) -> list[dict[str, Any]]:
    subject_selector = constraint.get("subjects") or {}
    if _is_entrance_selector(subject_selector):
        return _evaluate_entrance_routes(
            constraint,
            objects,
            tier,
            geometry=geometry or {},
        )
    if _is_virtual_route_selector(subject_selector):
        normalized = dict(constraint)
        normalized["subjects"] = {
            "category": "entrance",
            "count": 1,
            "quantifier": "all",
        }
        target_selector = constraint.get("targets") or {}
        if (
            _normalize_selector_category(target_selector.get("category"))
            in _ENTRANCE_CATEGORIES
        ):
            normalized["targets"] = {
                "category": "room",
                "count": 1,
                "quantifier": "all",
            }
        return _evaluate_entrance_routes(
            normalized,
            objects,
            tier,
            geometry=geometry or {},
        )
    if _requests_global_circulation(constraint):
        normalized = dict(constraint)
        normalized["subjects"] = {
            "category": "entrance",
            "count": 1,
            "quantifier": "all",
        }
        normalized["targets"] = dict(subject_selector)
        return _evaluate_entrance_routes(
            normalized,
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
        authorized_occupant_ids = _authorized_clear_access_occupants(
            case_pack,
            objects,
            subject_id,
        )
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
            if (
                other_id == subject_id
                or other_id in authorized_occupant_ids
                or object_category(other)
                in {
                    "wall",
                    "floor",
                    "ceiling",
                }
            ):
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
                    "authorized_occupant_ids": sorted(authorized_occupant_ids),
                    "blocking_ids": blockers,
                    "required_depth_m": required_depth,
                    "half_width_m": width,
                },
            )
        )
    return results


def _is_entrance_selector(selector: dict[str, Any]) -> bool:
    return (
        _normalize_selector_category(selector.get("category")) in _ENTRANCE_CATEGORIES
    )


def _is_virtual_route_selector(selector: dict[str, Any]) -> bool:
    category = _normalize_selector_category(selector.get("category"))
    role = str(selector.get("role") or "").strip().lower().replace("-", "_")
    return category in _VIRTUAL_ROUTE_CATEGORIES or role in {
        "access_route",
        "circulation_path",
        "walking_route",
    }


def _requests_global_circulation(constraint: dict[str, Any]) -> bool:
    evidence = str(constraint.get("evidence_span") or "").lower()
    return bool(
        re.search(
            r"\b(?:without\s+(?:blocking|obstructing)|while\s+(?:preserving|maintaining))\s+"
            r"(?:the\s+)?(?:circulation|traffic(?:\s+flow)?|walkway|walking\s+path)\b",
            evidence,
        )
    )


def _authorized_clear_access_occupants(
    case_pack: dict[str, Any],
    objects: list[dict[str, Any]],
    access_subject_id: str,
) -> set[str]:
    """Return named functional occupants that may use an access zone.

    A clear-access zone is normally circulation-only, but prompts can reserve
    part of it for a user-facing companion object, such as a stool at a
    dressing table or a chair at a workstation.  Local companions require
    explicit ``in_front_of`` and ``faces`` rows.  Group workstations instead
    use the existing functional-dependency chair-to-desk assignment, but only
    when the original hard contract explicitly says the group is both paired
    and facing.  This deliberately does not authorize a category in general
    or infer a nearest target from the current pose.
    """
    pairs_by_relation: dict[str, set[tuple[str, str]]] = {
        "in_front_of": set(),
        "faces": set(),
    }
    for relation, pairs in pairs_by_relation.items():
        for relation_constraint in contract_constraints(
            case_pack,
            relations=(relation,),
            include_auxiliary=False,
        ):
            if str(relation_constraint.get("strength") or "hard").lower() != "hard":
                continue
            subject_ids = bound_ids(relation_constraint.get("subjects"), objects)
            target_ids = bound_ids(relation_constraint.get("targets"), objects)
            # Multiple target IDs do not encode an object pairing.  Declining
            # that ambiguous contract is safer than exempting an unrelated
            # obstacle from every matching access zone.
            if len(target_ids) != 1:
                continue
            target_id = target_ids[0]
            pairs.update((subject_id, target_id) for subject_id in subject_ids)

    authorized = {
        subject_id
        for subject_id, target_id in (
            pairs_by_relation["in_front_of"] & pairs_by_relation["faces"]
        )
        if target_id == access_subject_id
    }
    authorized.update(
        _hard_paired_workstation_occupants(case_pack, objects, access_subject_id)
    )
    return authorized


def _hard_paired_workstation_occupants(
    case_pack: dict[str, Any],
    objects: list[dict[str, Any]],
    access_subject_id: str,
) -> set[str]:
    """Return concrete work seats explicitly paired and facing their desk.

    ``paired_with`` may name a group of desks and chairs, so it cannot bind an
    individual chair by itself.  Reuse the functional-dependency check's
    already-resolved seat-to-surface pairing, then require hard group-level
    ``paired_with`` and ``faces`` evidence before treating that chair as an
    intentional access-zone occupant.
    """
    by_id = {str(obj.get("id") or ""): obj for obj in objects}
    pairing_constraints = contract_constraints(
        case_pack, relations=("paired_with",), include_auxiliary=False
    )
    facing_constraints = contract_constraints(
        case_pack, relations=("faces",), include_auxiliary=False
    )
    if not pairing_constraints or not facing_constraints:
        return set()

    authorized: set[str] = set()
    for check in case_pack.get("checks") or []:
        if (
            not isinstance(check, dict)
            or str(check.get("metric") or "") != "functional_dependency"
            or str(check.get("relation_type") or "") != "seating_to_work_surface"
            or [str(item) for item in (check.get("target_ids") or [])]
            != [access_subject_id]
        ):
            continue
        seat_id = str(check.get("subject_id") or "")
        if not seat_id or not _is_seating_subject(by_id.get(seat_id) or {}):
            continue
        if _contract_pairs_and_faces_ids(
            pairing_constraints,
            facing_constraints,
            objects,
            seat_id,
            access_subject_id,
        ):
            authorized.add(seat_id)
    return authorized


def _contract_pairs_and_faces_ids(
    pairing_constraints: list[dict[str, Any]],
    facing_constraints: list[dict[str, Any]],
    objects: list[dict[str, Any]],
    seat_id: str,
    surface_id: str,
) -> bool:
    """Check prompt evidence for one resolved work-seat assignment."""
    paired = False
    for constraint in pairing_constraints:
        subject_ids = set(bound_ids(constraint.get("subjects"), objects))
        target_ids = set(bound_ids(constraint.get("targets"), objects))
        if (seat_id in subject_ids and surface_id in target_ids) or (
            seat_id in target_ids and surface_id in subject_ids
        ):
            paired = True
            break
    if not paired:
        return False
    return any(
        seat_id in set(bound_ids(constraint.get("subjects"), objects))
        and surface_id in set(bound_ids(constraint.get("targets"), objects))
        for constraint in facing_constraints
    )


def _evaluate_entrance_routes(
    constraint: dict[str, Any],
    objects: list[dict[str, Any]],
    tier: str,
    *,
    geometry: dict[str, Any],
) -> list[dict[str, Any]]:
    """Check connected walkable space from any entrance to each destination."""
    bounds = _room_bounds(geometry)
    doors = (geometry.get("scene_shell") or {}).get("doors") or []
    if bounds is None or not doors:
        return []

    target_selector = constraint.get("targets") or {}
    target_category = _normalize_selector_category(target_selector.get("category"))
    if target_category == "room":
        # ``room`` is a virtual endpoint, rather than a generated scene
        # object.  Route toward its interior center and later project that
        # point onto the walkable region, so a center occupied by furniture
        # still expresses access to the room rather than to that furniture.
        target_destinations = [
            (
                "room",
                ((bounds[0] + bounds[2]) / 2.0, (bounds[1] + bounds[3]) / 2.0),
            )
        ]
    else:
        by_id = {str(obj["id"]): obj for obj in objects}
        target_destinations = []
        for target_id in bound_ids(target_selector, objects):
            target_center = bbox_center_xy(by_id.get(target_id))
            if target_center is not None:
                target_destinations.append((target_id, target_center))
    if not target_destinations:
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
    for target_id, target_center in target_destinations:
        obstacles: list[tuple[str, Polygon]] = []
        for object_id, obj in by_id.items():
            if object_id == target_id or object_category(obj) in {
                "ceiling",
                "floor",
                "wall",
            }:
                continue
            lower = (obj.get("bbox_world") or {}).get("min") or []
            if len(lower) >= 3 and float(lower[2]) > 0.25:
                continue
            footprint = object_footprint_polygon(obj)
            if len(footprint) < 3:
                continue
            polygon = Polygon(footprint)
            if polygon.is_empty or polygon.area <= 1e-9:
                continue
            obstacles.append((object_id, polygon.buffer(radius, join_style="mitre")))
        blocked = (
            unary_union([polygon for _, polygon in obstacles]) if obstacles else None
        )
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
            start, _ = nearest_points(
                walkable, Point(float(center[0]), float(center[1]))
            )
            end, _ = nearest_points(walkable, Point(*target_center))
            selected_door = str(door.get("id") or door.get("opening_id") or "entrance")
            selected_start, selected_end = start, end
            if any(
                component.distance(start) <= 1e-7 and component.distance(end) <= 1e-7
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
                    "destination_target_xy_m": [
                        round(target_center[0], 6),
                        round(target_center[1], 6),
                    ],
                    "destination_walkable_xy_m": (
                        [round(selected_end.x, 6), round(selected_end.y, 6)]
                        if selected_end is not None
                        else None
                    ),
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
    "centered_above": lambda constraint, geometry, objects, _case_pack, tier: (
        _evaluate_centered_above(constraint, geometry, objects, tier)
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
    "faces": lambda constraint, geometry, objects, _case_pack, tier: (
        _evaluate_faces_room(constraint, geometry, objects, tier)
    ),
    "flanking": lambda constraint, _geometry, objects, _case_pack, tier: (
        _evaluate_flanking(constraint, objects, tier)
    ),
    "distributed_evenly": lambda constraint, _geometry, objects, _case_pack, tier: (
        _evaluate_group_distribution(constraint, objects, tier)
    ),
    "edge_distribution": lambda constraint, _geometry, objects, _case_pack, tier: [],
    "surround": lambda constraint, _geometry, objects, _case_pack, tier: (
        _evaluate_group_distribution(constraint, objects, tier)
    ),
    "corner_of_room": lambda constraint, geometry, objects, _case_pack, tier: (
        _evaluate_corner_of_room(constraint, geometry, objects, tier)
    ),
    "corner_distribution": lambda constraint, geometry, objects, _case_pack, tier: (
        _evaluate_corner_distribution(constraint, geometry, objects, tier)
    ),
    "one_per_support": lambda constraint, geometry, objects, _case_pack, tier: (
        _evaluate_one_per_support(constraint, geometry, objects, tier)
    ),
    "across_from": lambda constraint, _geometry, objects, _case_pack, tier: (
        _evaluate_across_from(constraint, objects, tier)
    ),
    "clear_access": lambda constraint, geometry, objects, case_pack, tier: (
        _evaluate_clear_access(
            constraint,
            objects,
            tier,
            geometry,
            case_pack=case_pack,
        )
    ),
    "mounted_to_ceiling": lambda constraint, geometry, objects, _case_pack, tier: (
        _evaluate_hang_from_ceiling(constraint, geometry, objects, tier)
    ),
    "mounted_to_wall": lambda constraint, geometry, objects, _case_pack, tier: (
        _evaluate_mounted_to_wall(constraint, geometry, objects, tier)
    ),
    "operation_zone_at_wall": lambda constraint, geometry, objects, _case_pack, tier: (
        _evaluate_operation_zone_at_wall(constraint, geometry, objects, tier)
    ),
    "instructional_surface_alignment": lambda constraint, geometry, objects, _case_pack, tier: (
        _evaluate_instructional_surface_alignment(constraint, geometry, objects, tier)
    ),
}
