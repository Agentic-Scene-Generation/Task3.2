"""Coordinated classroom desk-and-chair distribution checks."""

from __future__ import annotations

import math

from typing import Any

from scenesmith.scenebenchmark_critic.core.geometry import (
    bbox_center_xy,
    front_vector,
    object_footprint_polygon,
)
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.seat_surface_assignment import (
    _solve_rectangular_costs,
    assign_work_seats_to_surfaces,
    classroom_surface_cohort,
    is_assignable_work_surface,
    room_bounds_from_case_pack,
    work_seat_candidates,
)
from scenesmith.scenebenchmark_critic.intent_contract import (
    bound_ids,
    contract_constraints,
    contract_relation_requested,
)

RELATION_TYPE = "classroom_workstation_distribution"
_CLASSROOM_HINTS = ("classroom", "school", "student desk", "student chair")


def evaluate_classroom_workstation_distribution(
    case_pack: dict[str, Any],
) -> list[dict[str, Any]]:
    """Require an N-desk student grid plus one distinct front-zone teacher desk."""
    task = str(case_pack.get("task_instruction") or "")
    room_type = str(case_pack.get("room_type") or "")
    contract_mode = str(case_pack.get("intent_contract_mode") or "legacy")
    if contract_mode == "contract" and not (
        contract_relation_requested(case_pack, "distributed_evenly")
        or _instructional_layout_requested(task, room_type)
    ):
        return []
    if not any(hint in f"{room_type} {task}".lower() for hint in _CLASSROOM_HINTS):
        return []
    geometry = case_pack.get("scene_geometry") or {}
    objects = [
        obj
        for obj in geometry.get("objects") or []
        if isinstance(obj, dict) and obj.get("id")
    ]
    room_bounds = room_bounds_from_case_pack(case_pack)
    if room_bounds is None:
        return []
    seats = work_seat_candidates(
        objects,
        task_instruction=task,
        room_type=room_type,
    )
    surfaces = [obj for obj in objects if is_assignable_work_surface(obj)]
    cohort = classroom_surface_cohort(
        surfaces,
        seat_count=len(seats),
        room_bounds=room_bounds,
    )
    if cohort is None:
        return []
    assignments = assign_work_seats_to_surfaces(
        objects,
        task_instruction=task,
        room_type=room_type,
        room_bounds=room_bounds,
    )
    student_surface_ids = {str(surface["id"]) for surface in cohort.student_surfaces}
    student_assignments = [
        assignment
        for assignment in assignments
        if assignment.surface_id in student_surface_ids
    ]
    if len(student_assignments) != len(cohort.student_surfaces):
        return []
    objects_by_id = {str(obj["id"]): obj for obj in objects}
    assignment_by_surface = {
        assignment.surface_id: assignment for assignment in student_assignments
    }
    if set(assignment_by_surface) != student_surface_ids:
        return []

    instructional_surface, instructional_front = _instructional_anchor(
        objects, room_bounds
    )
    layout = _layout_targets(
        cohort.student_surfaces,
        cohort.teacher_surface,
        assignment_by_surface,
        objects_by_id,
        objects=objects,
        room_bounds=room_bounds,
        front=instructional_front or cohort.front_vector_xy,
        instructional_surface=instructional_surface,
    )
    if layout is None:
        return []
    workstation_slots, teacher_slot = layout
    teacher_id = str(cohort.teacher_surface["id"])
    teacher_managed_by_contract = teacher_id in _operation_zone_subject_ids(
        case_pack, objects
    )
    if teacher_managed_by_contract:
        teacher_slot = {
            **teacher_slot,
            "aligned": True,
            "managed_by_contract": "operation_zone_at_wall",
        }
    all_aligned = (
        all(slot["aligned"] for slot in workstation_slots) and teacher_slot["aligned"]
    )
    related = sorted(
        {
            teacher_id,
            *(str(slot["surface_id"]) for slot in workstation_slots),
            *(str(slot["seat_id"]) for slot in workstation_slots),
        }
    )
    failures = [
        f"`{slot['surface_id']}`/`{slot['seat_id']}` is outside its coordinated grid slot"
        for slot in workstation_slots
        if not slot["aligned"]
    ]
    if not teacher_managed_by_contract and not teacher_slot["aligned"]:
        failures.append(f"`{teacher_id}` is outside the separate front teacher zone")
    reason = (
        "Student desks form an evenly distributed classroom grid and each chair "
        "occupies the matching desk-local seating slot; the remaining desk is "
        "separated into the front teacher zone."
        if all_aligned
        else "A classroom with N student chairs and N+1 desks requires one front-zone "
        "teacher desk and an evenly distributed student workstation grid. "
        + "; ".join(failures)
    )
    return [
        {
            "check_id": f"fd_classroom_{RELATION_TYPE}",
            "metric": "functional_dependency",
            "label": "pass" if all_aligned else "fail",
            "confidence": 0.94,
            "primary_object": teacher_id,
            "related_objects": related,
            "selected_related_objects": related,
            "blocking_objects": [],
            "relation_type": RELATION_TYPE,
            "reason": reason,
            "repair_advice": (
                "Move the teacher desk and all student desk-chair pairs as one "
                "coordinated layout; do not repair chairs independently."
                if not all_aligned
                else ""
            ),
            "diagnostics": {
                "front_vector_xy": list(instructional_front or cohort.front_vector_xy),
                "room_bounds_xy": list(room_bounds),
                "workstation_slots": workstation_slots,
                "teacher_slot": teacher_slot,
            },
            "evidence": {
                "distribution": "room_relative_student_grid",
                "pairing": "one_chair_per_student_desk",
                "teacher_desk_rule": "one_extra_front_zone_surface",
            },
            "evaluation_source": "scenesmith_classroom_workstation_distribution",
            "scoring_tier": "core",
        }
    ]


def _operation_zone_subject_ids(
    case_pack: dict[str, Any], objects: list[dict[str, Any]]
) -> set[str]:
    subject_ids: set[str] = set()
    for constraint in contract_constraints(
        case_pack,
        relations=("operation_zone_at_wall",),
        include_auxiliary=False,
    ):
        subject_ids.update(bound_ids(constraint.get("subjects"), objects))
    return subject_ids


def _layout_targets(
    student_surfaces: tuple[dict[str, Any], ...],
    teacher_surface: dict[str, Any],
    assignment_by_surface: dict[str, Any],
    objects_by_id: dict[str, dict[str, Any]],
    *,
    objects: list[dict[str, Any]],
    room_bounds: tuple[float, float, float, float],
    front: tuple[float, float],
    instructional_surface: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
    count = len(student_surfaces)
    if count < 2:
        return None
    front_axis = 0 if abs(front[0]) > abs(front[1]) else 1
    lateral_axis = 1 - front_axis
    front_sign = front[front_axis]
    lateral = (0.0, 1.0) if lateral_axis == 1 else (1.0, 0.0)
    lower = room_bounds[:2]
    upper = room_bounds[2:]
    p_values = (front_sign * lower[front_axis], front_sign * upper[front_axis])
    p_min, p_max = min(p_values), max(p_values)
    q_min, q_max = lower[lateral_axis], upper[lateral_axis]
    room_depth = p_max - p_min
    room_width = q_max - q_min

    student_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for surface in student_surfaces:
        surface_id = str(surface["id"])
        assignment = assignment_by_surface.get(surface_id)
        seat = objects_by_id.get(str(getattr(assignment, "seat_id", "")))
        if assignment is None or seat is None:
            return None
        student_pairs.append((surface, seat))

    desk_half_depth = max(_local_depth(surface) / 2.0 for surface, _ in student_pairs)
    chair_half_depth = max(_local_depth(seat) / 2.0 for _, seat in student_pairs)
    chair_offset = max(desk_half_depth + chair_half_depth + 0.08, 0.45)
    teacher_half_depth = _local_depth(teacher_surface) / 2.0
    grid_front_p = p_max - teacher_half_depth - max(0.45, 0.07 * room_depth)
    teacher_p = grid_front_p

    # A classroom's instructional board is a usable focal surface, not a wall
    # decoration.  The teacher desk belongs in the front teaching zone, but it
    # must stay on the student side of the board's standing/reading clearance.
    # Without this bound, a room-relative "front-zone" target can incorrectly
    # place the desk behind (or directly against) a blackboard.
    if instructional_surface is not None:
        board_center = bbox_center_xy(instructional_surface)
        if board_center is not None:
            board_p = front[0] * board_center[0] + front[1] * board_center[1]
            board_half_depth = _extent_along(instructional_surface, front) / 2.0
            access_clearance = min(1.0, max(0.65, 0.10 * room_depth))
            teacher_p = min(
                teacher_p,
                board_p - board_half_depth - teacher_half_depth - access_clearance,
            )
    student_p_min = (
        p_min + max(0.55, 0.06 * room_depth) + chair_offset + chair_half_depth
    )
    student_p_max = (
        grid_front_p
        - teacher_half_depth
        - desk_half_depth
        - max(0.65, 0.08 * room_depth)
    )
    if student_p_max <= student_p_min:
        return None

    max_lateral_half = max(
        _local_width(obj) / 2.0 for pair in student_pairs for obj in pair
    )
    lateral_margin = max(0.7, max_lateral_half + 0.45)
    usable_q_min, usable_q_max = q_min + lateral_margin, q_max - lateral_margin
    if usable_q_max <= usable_q_min:
        return None

    columns = max(
        1,
        min(
            count,
            int(round(math.sqrt(count * room_width / max(room_depth, 1e-6)))),
        ),
    )
    rows = math.ceil(count / columns)
    p_slots = _segment_centers(student_p_min, student_p_max, rows)
    slots: list[tuple[float, float]] = []
    remaining = count
    for row_index, p_slot in enumerate(reversed(p_slots)):
        row_count = min(columns, remaining)
        remaining -= row_count
        q_slots = _segment_centers(usable_q_min, usable_q_max, row_count)
        slots.extend(_world_xy(p_slot, q, front_axis, front_sign) for q in q_slots)
    if len(slots) != count:
        return None

    costs = tuple(
        tuple(_distance(surface, slot) for slot in slots)
        for surface, _ in student_pairs
    )
    slot_indices = _solve_rectangular_costs(costs)
    # A student's viewing direction is toward the teaching focal surface, while
    # the usable front of their desk is the edge facing the seated student.
    # These directions are intentionally opposite: keeping them equal makes a
    # visually plausible classroom grid violate the paired desk--chair use
    # contract and prevents the atomic repair from being accepted.
    desk_front = (-front[0], -front[1])
    desk_yaw = _yaw_for_front(desk_front)
    chair_yaw = _yaw_for_front(front)
    tolerance = min(0.55, max(0.28, 0.06 * min(room_width, room_depth)))
    diagnostics: list[dict[str, Any]] = []
    for pair_index, (surface, seat) in enumerate(student_pairs):
        desk_target = slots[slot_indices[pair_index]]
        chair_target = (
            desk_target[0] - front[0] * chair_offset,
            desk_target[1] - front[1] * chair_offset,
        )
        desk_distance = _distance(surface, desk_target)
        chair_distance = _distance(seat, chair_target)
        desk_yaw_error = _front_error_deg(surface, desk_front)
        chair_yaw_error = _front_error_deg(seat, front)
        aligned = (
            desk_distance <= tolerance
            and chair_distance <= tolerance
            and desk_yaw_error <= 15.0
            and chair_yaw_error <= 15.0
        )
        diagnostics.append(
            {
                "surface_id": str(surface["id"]),
                "seat_id": str(seat["id"]),
                "target_surface_center_xy_m": list(desk_target),
                "target_surface_yaw_deg": desk_yaw,
                "target_seat_center_xy_m": list(chair_target),
                "target_seat_yaw_deg": chair_yaw,
                "surface_deviation_m": round(desk_distance, 4),
                "seat_deviation_m": round(chair_distance, 4),
                "surface_facing_error_deg": round(desk_yaw_error, 2),
                "seat_facing_error_deg": round(chair_yaw_error, 2),
                "allowed_deviation_m": tolerance,
                "allowed_facing_error_deg": 15.0,
                "aligned": aligned,
            }
        )

    teacher_q = (q_min + q_max) / 2.0
    if instructional_surface is not None:
        board_center = bbox_center_xy(instructional_surface)
        if board_center is not None:
            teacher_q = board_center[lateral_axis]
    teacher_target = _world_xy(teacher_p, teacher_q, front_axis, front_sign)
    teacher_front = (-front[0], -front[1])
    teacher_distance = _distance(teacher_surface, teacher_target)
    teacher_yaw_error = _front_error_deg(teacher_surface, teacher_front)
    teacher_slot = {
        "surface_id": str(teacher_surface["id"]),
        "target_surface_center_xy_m": list(teacher_target),
        "target_surface_yaw_deg": _yaw_for_front(teacher_front),
        "surface_deviation_m": round(teacher_distance, 4),
        "surface_facing_error_deg": round(teacher_yaw_error, 2),
        "allowed_deviation_m": tolerance,
        "allowed_facing_error_deg": 15.0,
        "aligned": teacher_distance <= tolerance and teacher_yaw_error <= 15.0,
    }
    if instructional_surface is not None:
        teacher_slot["instructional_surface_id"] = str(instructional_surface["id"])
        teacher_slot["instructional_access_clearance_m"] = round(
            max(0.0, grid_front_p - teacher_p), 4
        )
    return diagnostics, teacher_slot


def _instructional_layout_requested(task: str, room_type: str) -> bool:
    """Recognize the reusable teacher/student instructional-room topology."""
    context = f"{room_type} {task}".lower()
    return (
        any(hint in context for hint in _CLASSROOM_HINTS)
        and any(token in context for token in ("teacher", "instructor", "lecturer"))
        and "student" in context
        and any(token in context for token in ("desk", "table", "workstation"))
    )


def _instructional_anchor(
    objects: list[dict[str, Any]],
    room_bounds: tuple[float, float, float, float],
) -> tuple[dict[str, Any] | None, tuple[float, float] | None]:
    """Find a teaching focal surface and the room-relative direction it anchors."""
    min_x, min_y, max_x, max_y = room_bounds
    spans = (max_x - min_x, max_y - min_y)
    candidates: list[tuple[float, str, dict[str, Any], tuple[float, float]]] = []
    for obj in objects:
        text = " ".join(
            str(obj.get(key) or "")
            for key in ("id", "name", "description", "category", "type")
        ).lower()
        if not any(
            token in text
            for token in (
                "chalkboard",
                "blackboard",
                "whiteboard",
                "projection screen",
                "projector screen",
                "teaching screen",
            )
        ):
            continue
        center = bbox_center_xy(obj)
        if center is None:
            continue
        boundary_options = (
            (abs(center[0] - min_x) / spans[0], (-1.0, 0.0)),
            (abs(max_x - center[0]) / spans[0], (1.0, 0.0)),
            (abs(center[1] - min_y) / spans[1], (0.0, -1.0)),
            (abs(max_y - center[1]) / spans[1], (0.0, 1.0)),
        )
        distance, front = min(boundary_options, key=lambda item: item[0])
        candidates.append((distance, str(obj["id"]), obj, front))
    if not candidates:
        return None, None
    _, _, surface, front = min(candidates, key=lambda item: item[:2])
    return surface, front


def _segment_centers(lower: float, upper: float, count: int) -> list[float]:
    step = (upper - lower) / count
    return [lower + (index + 0.5) * step for index in range(count)]


def _world_xy(
    front_coordinate: float,
    lateral_coordinate: float,
    front_axis: int,
    front_sign: float,
) -> tuple[float, float]:
    values = [0.0, 0.0]
    values[front_axis] = front_sign * front_coordinate
    values[1 - front_axis] = lateral_coordinate
    return float(values[0]), float(values[1])


def _extent_along(obj: dict[str, Any], axis: tuple[float, float]) -> float:
    polygon = object_footprint_polygon(obj) or []
    if polygon:
        projected = [x * axis[0] + y * axis[1] for x, y in polygon]
        return max(projected) - min(projected)
    size = (obj.get("bbox_world") or {}).get("size") or []
    if len(size) < 2:
        return 0.5
    return abs(axis[0]) * float(size[0]) + abs(axis[1]) * float(size[1])


def _local_depth(obj: dict[str, Any]) -> float:
    return _extent_along(obj, front_vector(obj))


def _local_width(obj: dict[str, Any]) -> float:
    front = front_vector(obj)
    return _extent_along(obj, (-front[1], front[0]))


def _distance(obj: dict[str, Any], target: tuple[float, float]) -> float:
    center = bbox_center_xy(obj)
    if center is None:
        return float("inf")
    return math.hypot(center[0] - target[0], center[1] - target[1])


def _front_error_deg(obj: dict[str, Any], target: tuple[float, float]) -> float:
    current = front_vector(obj)
    dot = max(-1.0, min(1.0, current[0] * target[0] + current[1] * target[1]))
    return math.degrees(math.acos(dot))


def _yaw_for_front(front: tuple[float, float]) -> float:
    return math.degrees(math.atan2(-front[0], front[1]))
