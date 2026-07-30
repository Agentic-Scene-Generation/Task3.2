"""Stable one-to-one assignment of work seats to usable work surfaces."""

from __future__ import annotations

import functools
import math
import re

from dataclasses import dataclass
from typing import Any

from scenesmith.scenebenchmark_critic.core.geometry import (
    bbox_center_xy,
    front_vector,
    object_category,
    object_footprint_polygon,
)
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.profiles import (
    object_function_profile,
)
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.semantics import (
    _is_seating_subject,
    _is_work_surface_target,
)

ASSIGNMENT_SOURCE = "scenesmith_seat_surface_global_assignment"
_NON_WORK_SEATING = {"armchair", "bench", "dining_chair", "loveseat", "sofa", "stool"}
_NON_WORK_SURFACES = {
    "bar_table",
    "coffee_table",
    "dining_table",
    "end_table",
    "nightstand",
    "side_table",
}
_WORK_RELATION_TOKENS = {
    "computer_desk",
    "conference_table",
    "desk",
    "office_desk",
    "table",
    "work_surface",
    "work_table",
    "workstation",
    "writing_desk",
}
_WORK_CONTEXT_HINTS = (
    "classroom",
    "computer desk",
    "desk and chair",
    "desk with a chair",
    "office",
    "school",
    "student chair",
    "student desk",
    "study",
    "workstation",
)
_CLASSROOM_CONTEXT_HINTS = ("classroom", "school", "student chair", "student desk")
_INSTRUCTOR_ROLE_TOKENS = ("teacher", "instructor", "lecturer", "professor", "faculty")
_LEARNER_ROLE_TOKENS = ("student", "learner", "pupil")
_DINING_CONTEXT_HINTS = (
    "dining",
    "restaurant",
    "breakfast",
    "lunch",
    "dinner",
    "meal",
)
_WORK_CONTEXT_OVERRIDES = (
    "classroom",
    "computer",
    "desk",
    "office",
    "school",
    "study",
    "workstation",
)
_TRAILING_INSTANCE_RE = re.compile(r"(?:[_\s-]+[0-9a-z]+)$", re.IGNORECASE)


@dataclass(frozen=True)
class SeatSurfaceAssignment:
    seat_id: str
    surface_id: str
    target_center_xy: tuple[float, float]
    target_yaw_deg: float
    side: str
    cost: float
    evidence_sources: tuple[str, ...]

    def evidence(self) -> dict[str, Any]:
        return {
            "pairing": "global_minimum_cost_one_to_one",
            "topology_required": True,
            "assignment_source": ASSIGNMENT_SOURCE,
            "target_slot": {
                "center_xy": [round(value, 4) for value in self.target_center_xy],
                "yaw_deg": round(self.target_yaw_deg, 2),
                "surface_side": self.side,
            },
            "annotation_sources": list(self.evidence_sources),
            "assignment_cost": round(self.cost, 4),
        }


@dataclass(frozen=True)
class ClassroomSurfaceCohort:
    """Student surfaces plus the distinct focal desk and inferred room front."""

    student_surfaces: tuple[dict[str, Any], ...]
    teacher_surface: dict[str, Any]
    front_vector_xy: tuple[float, float]


def assign_work_seats_to_surfaces(
    objects: list[dict[str, Any]] | dict[str, dict[str, Any]],
    *,
    task_instruction: str = "",
    room_type: str = "",
    fixed_pairs: dict[str, str] | None = None,
    room_bounds: tuple[float, float, float, float] | None = None,
) -> list[SeatSurfaceAssignment]:
    """Assign work-oriented seats to surface-local chair slots one-to-one.

    Functional annotations establish candidacy. Current geometry selects among
    valid surfaces, but never decides whether a work-seat dependency exists.
    """
    values = list(objects.values()) if isinstance(objects, dict) else list(objects)
    resolved_room_bounds = room_bounds or _room_bounds_from_objects(values)
    all_surfaces = [obj for obj in values if _is_candidate_surface(obj)]
    all_seats = [obj for obj in values if _is_candidate_seat(obj)]
    if not all_seats or not all_surfaces:
        return []
    seats = work_seat_candidates(
        values,
        task_instruction=task_instruction,
        room_type=room_type,
    )
    if not seats:
        return []

    context = f"{room_type} {task_instruction}".lower()
    classroom_context = any(hint in context for hint in _CLASSROOM_CONTEXT_HINTS)
    if classroom_context:
        classroom = classroom_surface_cohort(
            all_surfaces,
            seat_count=len(seats),
            room_bounds=resolved_room_bounds,
        )
        if classroom is not None:
            student_seats, instructor_seats = _classroom_seat_cohorts(seats)
            student_assignments = _assign_seat_surface_cohort(
                student_seats,
                list(classroom.student_surfaces),
                fixed_pairs=fixed_pairs,
                room_bounds=resolved_room_bounds,
            )
            instructor_assignments = _assign_seat_surface_cohort(
                instructor_seats,
                [classroom.teacher_surface],
                fixed_pairs=fixed_pairs,
                room_bounds=resolved_room_bounds,
            )
            return sorted(
                [*student_assignments, *instructor_assignments],
                key=lambda item: item.seat_id,
            )

    surfaces = _surface_cohort(
        all_surfaces,
        seat_count=len(seats),
        classroom_context=classroom_context,
        room_bounds=resolved_room_bounds,
    )
    if not surfaces:
        return []
    return _assign_seat_surface_cohort(
        seats,
        surfaces,
        fixed_pairs=fixed_pairs,
        room_bounds=resolved_room_bounds,
    )


def _assign_seat_surface_cohort(
    seats: list[dict[str, Any]],
    surfaces: list[dict[str, Any]],
    *,
    fixed_pairs: dict[str, str] | None,
    room_bounds: tuple[float, float, float, float] | None,
) -> list[SeatSurfaceAssignment]:
    """Assign one role-compatible seat/surface cohort without cross-role fallback."""
    seats = sorted(seats, key=_object_id)
    surfaces = sorted(surfaces, key=_object_id)

    seats_by_id = {_object_id(obj): obj for obj in seats}
    surfaces_by_id = {_object_id(obj): obj for obj in surfaces}
    requested_fixed = dict(fixed_pairs or {})
    for seat in seats:
        seat_role = _indexed_role(seat)
        if seat_role is None or seat_role[0] != "chair":
            continue
        matching_surface = next(
            (
                surface
                for surface in surfaces
                if _indexed_role(surface) == ("desk", seat_role[1])
            ),
            None,
        )
        if matching_surface is not None:
            requested_fixed.setdefault(_object_id(seat), _object_id(matching_surface))
    fixed: list[SeatSurfaceAssignment] = []
    used_seats: set[str] = set()
    used_surfaces: set[str] = set()
    for seat_id, surface_id in sorted(requested_fixed.items()):
        seat = seats_by_id.get(str(seat_id))
        surface = surfaces_by_id.get(str(surface_id))
        if seat is None or surface is None or surface_id in used_surfaces:
            continue
        fixed.append(_pair_assignment(seat, surface, room_bounds))
        used_seats.add(str(seat_id))
        used_surfaces.add(str(surface_id))

    free_seats = [obj for obj in seats if _object_id(obj) not in used_seats]
    free_surfaces = [obj for obj in surfaces if _object_id(obj) not in used_surfaces]
    assigned = _minimum_cost_pairs(free_seats, free_surfaces, room_bounds=room_bounds)
    return sorted([*fixed, *assigned], key=lambda item: item.seat_id)


def is_assignable_work_seat(obj: dict[str, Any]) -> bool:
    """Return whether an object can participate in a work-seat assignment."""
    return _is_candidate_seat(obj)


def is_assignable_work_surface(obj: dict[str, Any]) -> bool:
    """Return whether an object can participate in a work-seat assignment."""
    return _is_candidate_surface(obj)


def work_seat_candidates(
    objects: list[dict[str, Any]] | dict[str, dict[str, Any]],
    *,
    task_instruction: str = "",
    room_type: str = "",
) -> list[dict[str, Any]]:
    """Return seats whose functional role requires a distinct work surface."""
    values = list(objects.values()) if isinstance(objects, dict) else list(objects)
    if is_dining_context(task_instruction=task_instruction, room_type=room_type):
        return []
    all_seats = [obj for obj in values if _is_candidate_seat(obj)]
    all_surfaces = [obj for obj in values if _is_candidate_surface(obj)]
    context = f"{room_type} {task_instruction}".lower()
    work_context = any(hint in context for hint in _WORK_CONTEXT_HINTS)
    classroom_context = any(hint in context for hint in _CLASSROOM_CONTEXT_HINTS)
    return [
        seat
        for seat in all_seats
        if _seat_has_work_intent(
            seat,
            work_context=work_context,
            classroom_context=classroom_context,
            seat_count=len(all_seats),
            surface_count=len(all_surfaces),
        )
    ]


def is_dining_context(*, task_instruction: str = "", room_type: str = "") -> bool:
    """Return true only for an explicit dining context without work semantics."""
    context = f"{room_type} {task_instruction}".lower()
    return bool(
        any(hint in context for hint in _DINING_CONTEXT_HINTS)
        and not any(hint in context for hint in _WORK_CONTEXT_OVERRIDES)
    )


def _is_candidate_seat(obj: dict[str, Any]) -> bool:
    return (
        bool(_object_id(obj))
        and _is_seating_subject(obj)
        and object_function_profile(obj).is_seating
        and object_category(obj) not in _NON_WORK_SEATING
        and bbox_center_xy(obj) is not None
    )


def _is_candidate_surface(obj: dict[str, Any]) -> bool:
    category = object_category(obj)
    if (
        not _object_id(obj)
        or category in _NON_WORK_SURFACES
        or not _is_work_surface_target(obj)
        or not object_function_profile(obj).is_work_surface
        or bbox_center_xy(obj) is None
    ):
        return False
    identity = _identity(obj)
    return category in {
        "computer_desk",
        "desk",
        "office_desk",
        "table",
        "work_table",
        "writing_desk",
    } or any(token in identity for token in ("desk", "work_table", "workstation"))


def _seat_has_work_intent(
    seat: dict[str, Any],
    *,
    work_context: bool,
    classroom_context: bool,
    seat_count: int,
    surface_count: int,
) -> bool:
    category = object_category(seat)
    if category == "office_chair":
        return True
    identity = _identity(seat)
    if any(
        token in identity for token in ("office", "student", "task_chair", "work_chair")
    ):
        return True
    hints = seat.get("functional_hints") or {}
    relations: list[str] = []
    for key in ("explicit_target_relation", "target_relation"):
        raw = hints.get(key)
        values = raw if isinstance(raw, list) else [raw]
        relations.extend(_normalize_token(value) for value in values if value)
    if set(relations) & _WORK_RELATION_TOKENS:
        return True
    if classroom_context and seat_count >= 2 and surface_count >= 2:
        return True
    return work_context and seat_count == 1 and surface_count == 1


def _surface_cohort(
    surfaces: list[dict[str, Any]],
    *,
    seat_count: int,
    classroom_context: bool = False,
    room_bounds: tuple[float, float, float, float] | None = None,
) -> list[dict[str, Any]]:
    """Keep repeated workstation families ahead of singleton focal desks."""
    if classroom_context:
        classroom = classroom_surface_cohort(
            surfaces,
            seat_count=seat_count,
            room_bounds=room_bounds,
        )
        if classroom is not None:
            return list(classroom.student_surfaces)
    if seat_count < 2 or len(surfaces) < 2:
        return surfaces
    family_counts: dict[str, int] = {}
    for surface in surfaces:
        family = _instance_family(surface)
        family_counts[family] = family_counts.get(family, 0) + 1
    repeated = [
        surface
        for surface in surfaces
        if family_counts.get(_instance_family(surface), 0) >= 2
    ]
    return repeated or surfaces


def classroom_surface_cohort(
    surfaces: list[dict[str, Any]],
    *,
    seat_count: int,
    room_bounds: tuple[float, float, float, float] | None,
) -> ClassroomSurfaceCohort | None:
    """Separate one front-zone teacher desk from repeated student desks.

    Generic agents often instantiate every requested desk from one asset family, so
    names and categories cannot distinguish the teacher desk. The teacher desk is
    instead the desk closest to a room boundary and closest to that wall's lateral
    centerline. This remains stable after the student grid is spread through the room.
    """
    if room_bounds is None:
        return None
    min_x, min_y, max_x, max_y = room_bounds
    span_x, span_y = max_x - min_x, max_y - min_y
    if min(span_x, span_y) <= 1e-6:
        return None
    room_center = ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)
    boundary_specs = (
        (0, -1.0, min_x, span_y),
        (0, 1.0, max_x, span_y),
        (1, -1.0, min_y, span_x),
        (1, 1.0, max_y, span_x),
    )
    explicit = [
        surface
        for surface in surfaces
        if _has_role_token(surface, _INSTRUCTOR_ROLE_TOKENS)
    ]
    if len(explicit) > 1:
        return None
    if not explicit and (seat_count < 2 or len(surfaces) != seat_count + 1):
        return None
    ranked: list[tuple[float, float, str, dict[str, Any], tuple[float, float]]] = []
    candidates = explicit or surfaces
    for surface in candidates:
        center = bbox_center_xy(surface)
        if center is None:
            continue
        for axis, sign, boundary, lateral_span in boundary_specs:
            normal_span = span_x if axis == 0 else span_y
            lateral_axis = 1 - axis
            boundary_gap = abs(float(center[axis]) - boundary)
            lateral_offset = abs(
                float(center[lateral_axis]) - room_center[lateral_axis]
            )
            score = boundary_gap / normal_span + 0.35 * lateral_offset / lateral_span
            front = (sign, 0.0) if axis == 0 else (0.0, sign)
            ranked.append((score, boundary_gap, _object_id(surface), surface, front))
    if not ranked:
        return None
    _, _, teacher_id, teacher, front = min(ranked, key=lambda item: item[:3])
    students = tuple(
        sorted(
            (surface for surface in surfaces if _object_id(surface) != teacher_id),
            key=_object_id,
        )
    )
    if not explicit and len(students) != seat_count:
        return None
    return ClassroomSurfaceCohort(students, teacher, front)


def _classroom_seat_cohorts(
    seats: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Partition explicit instructional roles before geometry-based assignment."""
    instructors = [
        seat for seat in seats if _has_role_token(seat, _INSTRUCTOR_ROLE_TOKENS)
    ]
    learners = [seat for seat in seats if _has_role_token(seat, _LEARNER_ROLE_TOKENS)]
    if learners:
        return learners, instructors
    return [seat for seat in seats if seat not in instructors], instructors


def _has_role_token(obj: dict[str, Any], tokens: tuple[str, ...]) -> bool:
    identity = _identity(obj)
    return any(
        re.search(rf"(?:^|[^a-z0-9]){re.escape(token)}(?:[^a-z0-9]|$)", identity)
        for token in tokens
    )


def _minimum_cost_pairs(
    seats: list[dict[str, Any]],
    surfaces: list[dict[str, Any]],
    *,
    room_bounds: tuple[float, float, float, float] | None = None,
) -> list[SeatSurfaceAssignment]:
    if not seats or not surfaces:
        return []
    matrix = [
        [_pair_assignment(seat, surface, room_bounds) for surface in surfaces]
        for seat in seats
    ]
    if max(len(seats), len(surfaces)) > 12:
        candidates = sorted(
            (assignment for row in matrix for assignment in row),
            key=lambda item: (item.cost, item.seat_id, item.surface_id),
        )
        selected: list[SeatSurfaceAssignment] = []
        used_seats: set[str] = set()
        used_surfaces: set[str] = set()
        for assignment in candidates:
            if (
                assignment.seat_id in used_seats
                or assignment.surface_id in used_surfaces
            ):
                continue
            selected.append(assignment)
            used_seats.add(assignment.seat_id)
            used_surfaces.add(assignment.surface_id)
            if len(selected) == min(len(seats), len(surfaces)):
                break
        return selected

    if len(seats) <= len(surfaces):
        indices = _solve_rectangular_costs(
            tuple(tuple(item.cost for item in row) for row in matrix)
        )
        return [
            matrix[seat_index][surface_index]
            for seat_index, surface_index in enumerate(indices)
        ]

    transposed = tuple(
        tuple(
            matrix[seat_index][surface_index].cost for seat_index in range(len(seats))
        )
        for surface_index in range(len(surfaces))
    )
    seat_indices = _solve_rectangular_costs(transposed)
    return [
        matrix[seat_index][surface_index]
        for surface_index, seat_index in enumerate(seat_indices)
    ]


def _solve_rectangular_costs(costs: tuple[tuple[float, ...], ...]) -> tuple[int, ...]:
    row_count = len(costs)
    column_count = len(costs[0])

    @functools.lru_cache(maxsize=None)
    def solve(row: int, used_mask: int) -> tuple[float, tuple[int, ...]]:
        if row == row_count:
            return 0.0, ()
        best = (math.inf, ())
        for column in range(column_count):
            if used_mask & (1 << column):
                continue
            remaining_cost, remaining = solve(row + 1, used_mask | (1 << column))
            candidate = (costs[row][column] + remaining_cost, (column, *remaining))
            if candidate < best:
                best = candidate
        return best

    return solve(0, 0)[1]


def _pair_assignment(
    seat: dict[str, Any],
    surface: dict[str, Any],
    room_bounds: tuple[float, float, float, float] | None = None,
) -> SeatSurfaceAssignment:
    seat_center = bbox_center_xy(seat)
    surface_center = bbox_center_xy(surface)
    assert seat_center is not None and surface_center is not None
    surface_front = front_vector(surface)
    surface_half_depth = _extent_along(surface, surface_front) / 2.0
    seat_half_depth = _extent_along(seat, surface_front) / 2.0
    offset = max(surface_half_depth + seat_half_depth + 0.12, 0.45)

    options: list[tuple[float, tuple[float, float], float, str]] = []
    # A work surface can be used from either side when room geometry can reject
    # an impossible slot. Without that geometry, retain the established desk
    # semantic fallback; the current chair pose is never used to choose it.
    sides = (
        ((1.0, "front"), (-1.0, "back"))
        if room_bounds is not None or "desk" not in _identity(surface)
        else ((-1.0, "back"),)
    )
    for sign, side in sides:
        slot = (
            surface_center[0] + sign * surface_front[0] * offset,
            surface_center[1] + sign * surface_front[1] * offset,
        )
        desired_front = (-sign * surface_front[0], -sign * surface_front[1])
        yaw = _yaw_for_front(desired_front)
        distance = math.hypot(slot[0] - seat_center[0], slot[1] - seat_center[1])
        yaw_error = _yaw_distance_deg(float(seat.get("yaw_deg") or 0.0), yaw)
        cost = distance + 0.0025 * yaw_error
        options.append((cost, slot, yaw, side))
    valid_options = [
        option
        for option in options
        if room_bounds is None
        or _slot_fits_room(seat, option[1], option[2], room_bounds)
    ]
    if valid_options:
        options = valid_options
    elif room_bounds is not None:
        # Keep an assignment available for malformed/tiny rooms, but never
        # expose a target center outside the usable room rectangle.
        options = [
            (
                cost,
                _clamp_slot_to_room(seat, slot, yaw, room_bounds),
                yaw,
                side,
            )
            for cost, slot, yaw, side in options
        ]
    cost, slot, yaw, side = min(options, key=lambda item: (item[0], item[3]))

    subject_role = _indexed_role(seat)
    target_role = _indexed_role(surface)
    if subject_role and target_role:
        if subject_role[0] == "chair" and target_role == ("desk", subject_role[1]):
            cost -= 0.2
        elif subject_role[0] == "chair" and target_role[0] == "desk":
            cost += 0.2
    return SeatSurfaceAssignment(
        seat_id=_object_id(seat),
        surface_id=_object_id(surface),
        target_center_xy=slot,
        target_yaw_deg=yaw,
        side=side,
        cost=cost,
        evidence_sources=tuple(
            sorted(_annotation_sources(seat) | _annotation_sources(surface))
        ),
    )


def room_bounds_from_case_pack(
    case_pack: dict[str, Any],
) -> tuple[float, float, float, float] | None:
    """Read the usable XY room rectangle when it is present in a case pack."""
    geometry = case_pack.get("scene_geometry") or {}
    rooms = geometry.get("rooms") or []
    for room in rooms:
        if not isinstance(room, dict):
            continue
        bbox = room.get("bbox") or {}
        lower = bbox.get("min") or []
        upper = bbox.get("max") or []
        if len(lower) >= 2 and len(upper) >= 2:
            bounds = tuple(float(value) for value in (*lower[:2], *upper[:2]))
            if bounds[0] < bounds[2] and bounds[1] < bounds[3]:
                return (
                    bounds[0],
                    bounds[1],
                    bounds[2],
                    bounds[3],
                )
    return None


def _room_bounds_from_objects(
    objects: list[dict[str, Any]],
) -> tuple[float, float, float, float] | None:
    """Infer inner room bounds from axis-aligned wall geometry when available."""
    walls = [obj for obj in objects if object_category(obj) == "wall"]
    vertical: list[tuple[float, float]] = []
    horizontal: list[tuple[float, float]] = []
    for wall in walls:
        bbox = wall.get("bbox_world") or {}
        center = bbox.get("center") or []
        size = bbox.get("size") or []
        if len(center) < 2 or len(size) < 2:
            continue
        cx, cy = float(center[0]), float(center[1])
        sx, sy = abs(float(size[0])), abs(float(size[1]))
        if sx <= 1e-6 or sy <= 1e-6:
            continue
        if sx <= sy:
            vertical.append((cx, sx / 2.0))
        else:
            horizontal.append((cy, sy / 2.0))

    x_bounds = _inner_axis_bounds(vertical)
    y_bounds = _inner_axis_bounds(horizontal)
    if x_bounds is None or y_bounds is None:
        return None
    return x_bounds[0], y_bounds[0], x_bounds[1], y_bounds[1]


def _inner_axis_bounds(
    walls: list[tuple[float, float]],
) -> tuple[float, float] | None:
    if len(walls) < 2:
        return None
    lower_center, lower_half = min(walls, key=lambda item: item[0])
    upper_center, upper_half = max(walls, key=lambda item: item[0])
    lower = lower_center + lower_half
    upper = upper_center - upper_half
    return (lower, upper) if lower < upper else None


def _slot_fits_room(
    seat: dict[str, Any],
    center: tuple[float, float],
    yaw_deg: float,
    room_bounds: tuple[float, float, float, float],
) -> bool:
    half_x, half_y = _rotated_footprint_half_extents(seat, yaw_deg)
    min_x, min_y, max_x, max_y = room_bounds
    return (
        min_x + half_x <= center[0] <= max_x - half_x
        and min_y + half_y <= center[1] <= max_y - half_y
    )


def _clamp_slot_to_room(
    seat: dict[str, Any],
    center: tuple[float, float],
    yaw_deg: float,
    room_bounds: tuple[float, float, float, float],
) -> tuple[float, float]:
    half_x, half_y = _rotated_footprint_half_extents(seat, yaw_deg)
    min_x, min_y, max_x, max_y = room_bounds
    low_x, high_x = min_x + half_x, max_x - half_x
    low_y, high_y = min_y + half_y, max_y - half_y
    return (
        min(max(center[0], low_x), high_x),
        min(max(center[1], low_y), high_y),
    )


def _rotated_footprint_half_extents(
    obj: dict[str, Any], yaw_deg: float
) -> tuple[float, float]:
    polygon = object_footprint_polygon(obj) or []
    center = bbox_center_xy(obj)
    if len(polygon) < 3 or center is None:
        size = (obj.get("bbox_world") or {}).get("size") or []
        return (
            abs(float(size[0])) / 2.0 if len(size) >= 1 else 0.25,
            abs(float(size[1])) / 2.0 if len(size) >= 2 else 0.25,
        )

    current_yaw = math.radians(float(obj.get("yaw_deg") or 0.0))
    current_u = (math.cos(current_yaw), math.sin(current_yaw))
    current_v = (-current_u[1], current_u[0])
    local_u = [
        (point[0] - center[0]) * current_u[0] + (point[1] - center[1]) * current_u[1]
        for point in polygon
    ]
    local_v = [
        (point[0] - center[0]) * current_v[0] + (point[1] - center[1]) * current_v[1]
        for point in polygon
    ]
    half_u = max(abs(value) for value in local_u)
    half_v = max(abs(value) for value in local_v)
    target_yaw = math.radians(yaw_deg)
    return (
        abs(math.cos(target_yaw)) * half_u + abs(math.sin(target_yaw)) * half_v,
        abs(math.sin(target_yaw)) * half_u + abs(math.cos(target_yaw)) * half_v,
    )


def _extent_along(obj: dict[str, Any], axis: tuple[float, float]) -> float:
    polygon = object_footprint_polygon(obj) or []
    if polygon:
        projections = [point[0] * axis[0] + point[1] * axis[1] for point in polygon]
        return max(projections) - min(projections)
    size = (obj.get("bbox_world") or {}).get("size") or []
    return max(float(size[0]), float(size[1])) if len(size) >= 2 else 0.5


def _yaw_for_front(vector: tuple[float, float]) -> float:
    return math.degrees(math.atan2(-vector[0], vector[1])) % 360.0


def _yaw_distance_deg(first: float, second: float) -> float:
    return abs((first - second + 180.0) % 360.0 - 180.0)


def _annotation_sources(obj: dict[str, Any]) -> set[str]:
    hints = obj.get("functional_hints") or {}
    sources = {
        str(value)
        for value in (
            hints.get("asset_annotation_source"),
            hints.get("classification_source"),
            object_function_profile(obj).source,
        )
        if value
    }
    return sources or {"inferred_geometry"}


def _instance_family(obj: dict[str, Any]) -> str:
    identity = str(obj.get("id") or obj.get("name") or object_category(obj)).lower()
    return _TRAILING_INSTANCE_RE.sub("", identity).strip("_- ")


def _indexed_role(obj: dict[str, Any]) -> tuple[str, int] | None:
    identity = _identity(obj)
    match = re.search(r"\b(?:student[_\s-]*)?(chair|desk)[_\s-]*(\d+)\b", identity)
    return (match.group(1), int(match.group(2))) if match else None


def _identity(obj: dict[str, Any]) -> str:
    return " ".join(
        str(obj.get(key) or "").strip().lower().replace("-", "_")
        for key in (
            "id",
            "name",
            "category",
            "category_norm",
            "description",
            "asset_id",
        )
    )


def _normalize_token(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _object_id(obj: dict[str, Any]) -> str:
    return str(obj.get("id") or "")
