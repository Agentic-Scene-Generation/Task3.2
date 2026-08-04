"""Generic rectangular-target edge distribution evaluator."""

from __future__ import annotations

import itertools
import math
from typing import Any

from scipy.optimize import linear_sum_assignment

from scenesmith.scenebenchmark_critic.core.geometry import (
    bbox_center_xy,
    front_vector,
    object_footprint_polygon,
)
from scenesmith.scenebenchmark_critic.intent_contract import (
    contract_constraints,
    selected_ids,
    selector_match_count,
)

RELATION_TYPE = "edge_distribution"
_TARGET_GAP_M = 0.05
_FACING_TOLERANCE_DEG = 10.0


def evaluate_edge_distribution(case_pack: dict[str, Any]) -> list[dict[str, Any]]:
    """Evaluate every hard edge-distribution relation in a case pack.

    Binding is intentionally strict: the relation must account for every
    matching subject and exactly one target.  Ambiguity becomes an unresolved
    hard result instead of a pose-dependent subset choice.
    """
    geometry = case_pack.get("scene_geometry") or {}
    objects = [
        obj
        for obj in geometry.get("objects") or []
        if isinstance(obj, dict) and obj.get("id")
    ]
    by_id = {str(obj["id"]): obj for obj in objects}
    results: list[dict[str, Any]] = []
    for constraint in contract_constraints(
        case_pack, relations=(RELATION_TYPE,), include_auxiliary=False
    ):
        subject_selector = constraint.get("subjects") or {}
        target_selector = constraint.get("targets") or {}
        subject_ids = selected_ids(subject_selector, objects)
        target_ids = selected_ids(target_selector, objects)
        expected_subject_count = int(subject_selector.get("count") or 0)
        expected_target_count = int(target_selector.get("count") or 0)
        target_id = target_ids[0] if len(target_ids) == 1 else ""

        if (
            not subject_ids
            or expected_subject_count <= 0
            or selector_match_count(subject_selector, objects) != expected_subject_count
        ):
            results.append(
                _unresolved(
                    constraint,
                    target_id=target_id,
                    related=subject_ids,
                    reason=(
                        "edge_distribution subjects selector must match every "
                        "object of the declared category exactly once"
                    ),
                )
            )
            continue
        if expected_target_count != 1 or len(target_ids) != 1:
            results.append(
                _unresolved(
                    constraint,
                    target_id=target_id,
                    related=subject_ids,
                    reason="edge_distribution target selector must bind exactly one object",
                )
            )
            continue

        target = by_id.get(target_id)
        subjects = [by_id[object_id] for object_id in subject_ids if object_id in by_id]
        if target is None or len(subjects) != len(subject_ids):
            results.append(
                _unresolved(
                    constraint,
                    target_id=target_id,
                    related=subject_ids,
                    reason="edge_distribution endpoint object is missing from geometry",
                )
            )
            continue
        shape = _target_rectangle(target)
        if shape is None:
            results.append(
                _unresolved(
                    constraint,
                    target_id=target_id,
                    related=subject_ids,
                    reason=(
                        "edge_distribution requires a non-square rectangular target "
                        "with a stable long/short edge frame"
                    ),
                )
            )
            continue

        result = _evaluate_bound_distribution(
            constraint,
            target,
            subjects,
            shape=shape,
        )
        results.append(result)
    return results


def _unresolved(
    constraint: dict[str, Any],
    *,
    target_id: str,
    related: list[str],
    reason: str,
) -> dict[str, Any]:
    return {
        "check_id": f"fd_{target_id or 'unresolved'}_{RELATION_TYPE}",
        "metric": "functional_dependency",
        "label": "unresolved",
        "confidence": 1.0,
        "primary_object": target_id,
        "related_objects": sorted(related),
        "selected_related_objects": sorted(related),
        "blocking_objects": [],
        "relation_type": RELATION_TYPE,
        "reason": reason,
        "diagnostics": {"unresolved": True, "unresolved_reason": reason},
        "evidence": {"intent_constraint": constraint},
        "evaluation_source": "scenesmith_edge_distribution",
        "scoring_tier": "core",
    }


def _target_rectangle(
    target: dict[str, Any],
) -> (
    tuple[tuple[float, float], float, float, tuple[float, float], tuple[float, float]]
    | None
):
    identity = " ".join(
        str(value).lower()
        for value in (
            target.get("id"),
            target.get("name"),
            target.get("category"),
            target.get("category_norm"),
        )
        if value
    )
    if any(token in identity for token in ("round", "circular", "oval", "ellipse")):
        return None
    shape = str(
        (target.get("functional_hints") or {}).get("footprint_shape")
        or target.get("footprint_shape")
        or "rectangular"
    ).lower()
    if shape not in {"rectangular", "rectangle", "box", ""}:
        return None
    center = bbox_center_xy(target)
    if center is None:
        return None
    yaw = math.radians(float(target.get("yaw_deg") or 0.0))
    tangent_x = (math.cos(yaw), math.sin(yaw))
    tangent_y = (-math.sin(yaw), math.cos(yaw))
    polygon = object_footprint_polygon(target) or []
    if len(polygon) != 4 or not _is_rectangle(polygon):
        return None
    width = _extent(polygon, tangent_x)
    depth = _extent(polygon, tangent_y)
    if min(width, depth) <= 1e-6 or max(width, depth) / min(width, depth) < 1.1:
        return None
    return center, width, depth, tangent_x, tangent_y


def _is_rectangle(polygon: list[tuple[float, float]]) -> bool:
    if len(polygon) != 4:
        return False
    points = [(float(x), float(y)) for x, y in polygon]
    edges = [
        (
            points[(index + 1) % 4][0] - points[index][0],
            points[(index + 1) % 4][1] - points[index][1],
        )
        for index in range(4)
    ]
    lengths = [math.hypot(x, y) for x, y in edges]
    if min(lengths) <= 1e-6:
        return False
    for index in range(4):
        current = edges[index]
        following = edges[(index + 1) % 4]
        if abs(current[0] * following[0] + current[1] * following[1]) > (
            1e-5 * lengths[index] * lengths[(index + 1) % 4]
        ):
            return False
    return True


def _extent(polygon: list[tuple[float, float]], axis: tuple[float, float]) -> float:
    values = [point[0] * axis[0] + point[1] * axis[1] for point in polygon]
    return max(values) - min(values)


def _evaluate_bound_distribution(
    constraint: dict[str, Any],
    target: dict[str, Any],
    subjects: list[dict[str, Any]],
    *,
    shape: tuple[
        tuple[float, float], float, float, tuple[float, float], tuple[float, float]
    ],
) -> dict[str, Any]:
    center, width, depth, tangent_x, tangent_y = shape
    long_edges = ("front", "back") if width >= depth else ("left", "right")
    short_edges = ("left", "right") if width >= depth else ("front", "back")
    orientation = str(constraint.get("orientation") or "unconstrained")
    edge_options: list[list[tuple[str, int]]] = []
    for group in constraint.get("groups") or []:
        edge_class = str(group.get("edge_class") or "")
        counts = [int(value) for value in group.get("counts_per_edge") or []]
        if edge_class not in {"long", "short"} or len(counts) != 2:
            return _unresolved(
                constraint,
                target_id=str(target["id"]),
                related=[str(obj["id"]) for obj in subjects],
                reason="edge_distribution contains an invalid edge group",
            )
        physical_edges = long_edges if edge_class == "long" else short_edges
        pairs = [(counts[0], counts[1])]
        if counts[0] != counts[1]:
            pairs.append((counts[1], counts[0]))
        edge_options.append(
            [
                [
                    (physical_edges[0], pair[0]),
                    (physical_edges[1], pair[1]),
                ]
                for pair in pairs
            ]
        )

    best: tuple[float, list[dict[str, Any]], list[dict[str, Any]]] | None = None
    for choices in itertools.product(*edge_options):
        edge_counts = dict(item for pair in choices for item in pair)
        slots = [
            (edge, tangent)
            for edge, count in edge_counts.items()
            for tangent in _segment_centers(_edge_length(edge, width, depth), count)
        ]
        if len(slots) != len(subjects):
            continue
        costs = []
        candidate_diagnostics: list[list[dict[str, Any]]] = []
        for subject in subjects:
            subject_center = bbox_center_xy(subject)
            if subject_center is None:
                costs.append([1e12] * len(slots))
                candidate_diagnostics.append([{} for _ in slots])
                continue
            row: list[float] = []
            row_diagnostics: list[dict[str, Any]] = []
            for edge, tangent in slots:
                target_xy = _slot_center(
                    subject,
                    edge,
                    tangent,
                    width=width,
                    depth=depth,
                    center=center,
                    tangent_x=tangent_x,
                    tangent_y=tangent_y,
                    orientation=orientation,
                )
                distance = (subject_center[0] - target_xy[0]) ** 2 + (
                    subject_center[1] - target_xy[1]
                ) ** 2
                row.append(distance)
                row_diagnostics.append(
                    _slot_diagnostics(
                        subject,
                        target,
                        edge=edge,
                        tangent=tangent,
                        target_xy=target_xy,
                        center=center,
                        width=width,
                        depth=depth,
                        tangent_x=tangent_x,
                        tangent_y=tangent_y,
                        spacing=str(
                            next(
                                (
                                    group.get("spacing")
                                    for group in constraint.get("groups") or []
                                    if group.get("edge_class")
                                    == ("long" if edge in long_edges else "short")
                                ),
                                "equal_segments",
                            )
                        ),
                    )
                )
            costs.append(row)
            candidate_diagnostics.append(row_diagnostics)
        row_indices, col_indices = linear_sum_assignment(costs)
        assignment_cost = sum(
            costs[row][col] for row, col in zip(row_indices, col_indices)
        )
        assigned = [
            candidate_diagnostics[row][col]
            for row, col in zip(row_indices, col_indices)
        ]
        if best is None or assignment_cost < best[0]:
            best = (assignment_cost, assigned, slots)

    if best is None:
        return _unresolved(
            constraint,
            target_id=str(target["id"]),
            related=[str(obj["id"]) for obj in subjects],
            reason="edge_distribution has no complete physical edge assignment",
        )

    _annotate_pass_fail(best[1], orientation=orientation)
    if orientation not in {"toward_target", "away_from_target"}:
        for item in best[1]:
            item["facing_target_xy_m"] = None
    failures = [item["failure"] for item in best[1] if item.get("failure")]
    diagnostics = best[1]
    return {
        "check_id": f"fd_{target['id']}_{RELATION_TYPE}",
        "metric": "functional_dependency",
        "label": "fail" if failures else "pass",
        "confidence": 0.95 if failures else 0.92,
        "primary_object": str(target["id"]),
        "related_objects": sorted(str(obj["id"]) for obj in subjects),
        "selected_related_objects": sorted(str(obj["id"]) for obj in subjects),
        "blocking_objects": [],
        "relation_type": RELATION_TYPE,
        "reason": (
            "edge distribution passes"
            if not failures
            else "edge distribution failed: " + "; ".join(failures)
        ),
        "diagnostics": {
            "edge_slots": diagnostics,
            "seat_slots": diagnostics,
            "topology_repair_slots": [
                {
                    "seat_id": item["object_id"],
                    "edge": item["edge"],
                    "target_center_xy_m": item["target_center_xy_m"],
                    "current_front_xy": item["current_front_xy"],
                    "facing_target_xy_m": item["facing_target_xy_m"],
                }
                for item in diagnostics
            ],
            "orientation": orientation,
        },
        "evidence": {
            "distribution": "target_local_rectangle_equal_edge_segments",
            "intent_constraint": constraint,
        },
        "evaluation_source": "scenesmith_edge_distribution",
        "scoring_tier": "core",
    }


def _segment_centers(edge_length: float, count: int) -> list[float]:
    if count <= 0:
        return []
    segment = edge_length / count
    return [-edge_length / 2.0 + (index + 0.5) * segment for index in range(count)]


def _edge_length(edge: str, width: float, depth: float) -> float:
    return depth if edge in {"left", "right"} else width


def _slot_center(
    subject: dict[str, Any],
    edge: str,
    tangent: float,
    *,
    width: float,
    depth: float,
    center: tuple[float, float],
    tangent_x: tuple[float, float],
    tangent_y: tuple[float, float],
    orientation: str,
) -> tuple[float, float]:
    normal_axis = tangent_x if edge in {"left", "right"} else tangent_y
    desired_front: tuple[float, float] | None = None
    if orientation == "toward_target":
        desired_front = _edge_inward_vector(
            edge, tangent_x=tangent_x, tangent_y=tangent_y
        )
    elif orientation == "away_from_target":
        inward = _edge_inward_vector(edge, tangent_x=tangent_x, tangent_y=tangent_y)
        desired_front = (-inward[0], -inward[1])
    span = _footprint_extent_at_facing(subject, normal_axis, desired_front)
    outward = (span or 0.5) / 2.0 + _TARGET_GAP_M
    if edge == "left":
        local = (-width / 2.0 - outward, tangent)
    elif edge == "right":
        local = (width / 2.0 + outward, tangent)
    elif edge == "front":
        local = (tangent, -depth / 2.0 - outward)
    else:
        local = (tangent, depth / 2.0 + outward)
    return (
        center[0] + local[0] * tangent_x[0] + local[1] * tangent_y[0],
        center[1] + local[0] * tangent_x[1] + local[1] * tangent_y[1],
    )


def _footprint_extent(obj: dict[str, Any], axis: tuple[float, float]) -> float | None:
    polygon = object_footprint_polygon(obj) or []
    if polygon:
        return _extent(polygon, axis)
    size = (obj.get("bbox_world") or {}).get("size") or []
    if len(size) >= 2:
        return abs(axis[0]) * float(size[0]) + abs(axis[1]) * float(size[1])
    return None


def _footprint_extent_at_facing(
    obj: dict[str, Any],
    axis: tuple[float, float],
    desired_front: tuple[float, float] | None,
) -> float | None:
    """Project the footprint after the relation repair applies its facing yaw.

    A chair's world footprint can be narrower before the repair than after a
    quarter turn toward the target.  Sizing its edge slot from the old pose
    therefore permits a table collision even though the relation itself passes.
    Rotate the current footprint about its center into the requested facing
    direction before computing the normal span.
    """
    polygon = object_footprint_polygon(obj) or []
    center = bbox_center_xy(obj)
    current_front = front_vector(obj)
    if (
        desired_front is None
        or len(polygon) < 3
        or center is None
        or math.hypot(*current_front) <= 1e-6
        or math.hypot(*desired_front) <= 1e-6
    ):
        extent = _footprint_extent(obj, axis)
        if extent is not None:
            return extent
        size = (obj.get("bbox_world") or {}).get("size") or []
        return max((abs(float(value)) for value in size[:2]), default=0.5)

    dot = current_front[0] * desired_front[0] + current_front[1] * desired_front[1]
    cross = current_front[0] * desired_front[1] - current_front[1] * desired_front[0]
    angle = math.atan2(cross, dot)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    rotated = [
        (
            center[0] + cosine * (point[0] - center[0]) - sine * (point[1] - center[1]),
            center[1] + sine * (point[0] - center[0]) + cosine * (point[1] - center[1]),
        )
        for point in polygon
    ]
    return _extent(rotated, axis)


def _slot_diagnostics(
    subject: dict[str, Any],
    target: dict[str, Any],
    *,
    edge: str,
    tangent: float,
    target_xy: tuple[float, float],
    center: tuple[float, float],
    width: float,
    depth: float,
    tangent_x: tuple[float, float],
    tangent_y: tuple[float, float],
    spacing: str,
) -> dict[str, Any]:
    subject_center = bbox_center_xy(subject) or target_xy
    dx = subject_center[0] - center[0]
    dy = subject_center[1] - center[1]
    local_x = dx * tangent_x[0] + dy * tangent_x[1]
    local_y = dx * tangent_y[0] + dy * tangent_y[1]
    current_tangent = local_y if edge in {"left", "right"} else local_x
    edge_length = _edge_length(edge, width, depth)
    tangent_deviation = abs(current_tangent - tangent)
    tangent_span = (
        _footprint_extent(
            subject, tangent_y if edge in {"left", "right"} else tangent_x
        )
        or 0.5
    )
    allowed_tangent = max(0.08, min(0.35 * tangent_span, 0.08 * edge_length))
    normal_span = (
        _footprint_extent(
            subject, tangent_x if edge in {"left", "right"} else tangent_y
        )
        or 0.5
    )
    normal_value = local_x if edge in {"left", "right"} else local_y
    sign = -1.0 if edge in {"left", "front"} else 1.0
    boundary = width / 2.0 if edge in {"left", "right"} else depth / 2.0
    outside_gap = sign * normal_value - boundary - normal_span / 2.0
    nearest_edge = _nearest_edge(local_x, local_y, width=width, depth=depth)
    front = front_vector(subject)
    inward = _edge_inward_vector(edge, tangent_x=tangent_x, tangent_y=tangent_y)
    facing_error = _angle_error(front, inward)
    tangent_vector = tangent_y if edge in {"left", "right"} else tangent_x
    parallel_error = min(
        _angle_error(front, tangent_vector),
        _angle_error(front, (-tangent_vector[0], -tangent_vector[1])),
    )
    facing_target = (
        target_xy[0] + inward[0],
        target_xy[1] + inward[1],
    )
    failures: list[str] = []
    if outside_gap < -0.02:
        failures.append("subject overlaps target footprint")
    if (
        nearest_edge != edge
        and _edge_distance(local_x, local_y, edge, width=width, depth=depth) > 0.15
    ):
        failures.append(f"subject is not associated with the finite {edge} edge")
    if spacing == "equal_segments" and tangent_deviation > allowed_tangent:
        failures.append(
            f"subject is {tangent_deviation:.2f}m from its equal-segment slot"
        )
    return {
        "object_id": str(subject["id"]),
        "edge": edge,
        "nearest_edge": nearest_edge,
        "tangent_position_m": round(current_tangent, 6),
        "target_position_m": round(tangent, 6),
        "deviation_m": round(tangent_deviation, 6),
        "allowed_deviation_m": round(allowed_tangent, 6),
        "outside_gap_m": round(outside_gap, 6),
        "target_center_xy_m": [round(value, 6) for value in target_xy],
        "current_front_xy": [round(value, 6) for value in front],
        "facing_target_xy_m": [round(value, 6) for value in facing_target],
        "facing_error_deg": round(facing_error, 3),
        "parallel_error_deg": round(parallel_error, 3),
        "failure": "; ".join(failures),
    }


def _annotate_pass_fail(diagnostics: list[dict[str, Any]], *, orientation: str) -> None:
    for item in diagnostics:
        if (
            orientation == "toward_target"
            and item["facing_error_deg"] > _FACING_TOLERANCE_DEG
        ):
            item["failure"] = "; ".join(
                value
                for value in (
                    item.get("failure"),
                    "subject does not face target inward",
                )
                if value
            )
        elif (
            orientation == "away_from_target"
            and abs(item["facing_error_deg"] - 180.0) > _FACING_TOLERANCE_DEG
        ):
            item["failure"] = "; ".join(
                value
                for value in (
                    item.get("failure"),
                    "subject does not face away from target",
                )
                if value
            )
        elif (
            orientation == "parallel_to_edge"
            and item["parallel_error_deg"] > _FACING_TOLERANCE_DEG
        ):
            item["failure"] = "; ".join(
                value
                for value in (
                    item.get("failure"),
                    "subject is not parallel to its edge",
                )
                if value
            )


def _angle_error(first: tuple[float, float], second: tuple[float, float]) -> float:
    first_norm = math.hypot(*first)
    second_norm = math.hypot(*second)
    if first_norm <= 1e-6 or second_norm <= 1e-6:
        return 180.0
    cosine = (first[0] * second[0] + first[1] * second[1]) / (first_norm * second_norm)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _nearest_edge(local_x: float, local_y: float, *, width: float, depth: float) -> str:
    candidates = {
        "left": _edge_distance(local_x, local_y, "left", width=width, depth=depth),
        "right": _edge_distance(local_x, local_y, "right", width=width, depth=depth),
        "front": _edge_distance(local_x, local_y, "front", width=width, depth=depth),
        "back": _edge_distance(local_x, local_y, "back", width=width, depth=depth),
    }
    return min(candidates, key=lambda edge: (candidates[edge], edge))


def _edge_inward_vector(
    edge: str,
    *,
    tangent_x: tuple[float, float],
    tangent_y: tuple[float, float],
) -> tuple[float, float]:
    """Return the target-local unit normal pointing into the rectangle."""
    if edge == "left":
        return tangent_x
    if edge == "right":
        return -tangent_x[0], -tangent_x[1]
    if edge == "front":
        return tangent_y
    return -tangent_y[0], -tangent_y[1]


def _edge_distance(
    local_x: float,
    local_y: float,
    edge: str,
    *,
    width: float,
    depth: float,
) -> float:
    half_width = width / 2.0
    half_depth = depth / 2.0
    if edge == "left":
        return math.hypot(
            local_x + half_width, local_y - min(max(local_y, -half_depth), half_depth)
        )
    if edge == "right":
        return math.hypot(
            local_x - half_width, local_y - min(max(local_y, -half_depth), half_depth)
        )
    if edge == "front":
        return math.hypot(
            local_x - min(max(local_x, -half_width), half_width), local_y + half_depth
        )
    return math.hypot(
        local_x - min(max(local_x, -half_width), half_width), local_y - half_depth
    )
