"""Deterministic future-capacity validation for generated floor plans."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from scenesmith.agent_utils.house import HouseLayout, OpeningType, Wall
from scenesmith.scene_expert.explicit_geometry import (
    normalize_polygon_vertices,
    polygon_area_m2,
)
from scenesmith.scene_expert.schemas import (
    FloorPlanReservation,
    FloorPlanReservationManifest,
)


_OPENING_MARGIN_M = 0.15


def _normalized_room_type(value: str) -> str:
    return "_".join(str(value or "").strip().lower().replace("-", " ").split())


def _rooms_for_type(layout: HouseLayout, room_type: str) -> list[Any]:
    rooms = list(layout.placed_rooms)
    normalized = _normalized_room_type(room_type)
    if not normalized:
        return rooms
    matching_ids = {
        spec.room_id
        for spec in layout.room_specs
        if _normalized_room_type(spec.room_type) == normalized
    }
    return [room for room in rooms if room.room_id in matching_ids]


@dataclass
class FloorPlanReservationValidation:
    future_capacity: str = "ok"
    opening_budget: str = "ok"
    issues: list[dict[str, Any]] = field(default_factory=list)
    advisories: list[dict[str, Any]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.future_capacity == "ok" and self.opening_budget == "ok"


def adaptive_implicit_window_budget(area_m2: float) -> int:
    if area_m2 <= 25.0:
        return 1
    if area_m2 <= 50.0:
        return 2
    return 3


def opening_free_spans(
    wall: Wall, margin_m: float = _OPENING_MARGIN_M
) -> list[tuple[float, float]]:
    blocked: list[tuple[float, float]] = []
    for opening in wall.openings:
        if opening.opening_type == OpeningType.OPEN:
            return []
        # HouseLayout stores the opening's left edge, not its center.
        opening_start = float(opening.position_along_wall)
        opening_end = opening_start + max(0.0, float(opening.width))
        blocked.append(
            (
                max(0.0, opening_start - margin_m),
                min(
                    float(wall.length),
                    opening_end + margin_m,
                ),
            )
        )
    merged: list[list[float]] = []
    for start, end in sorted(blocked):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    spans: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in merged:
        if start > cursor:
            spans.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < wall.length:
        spans.append((cursor, float(wall.length)))
    return spans


def _candidate_walls(
    layout: HouseLayout, reservation: FloorPlanReservation
) -> list[Wall]:
    rooms = _rooms_for_type(layout, reservation.room_type)
    walls = [wall for room in rooms for wall in room.walls]
    role = reservation.wall_role.strip().lower()
    # Match the room-local wall names used by generated scene geometry.  In that
    # coordinate frame front/back are north/south and left/right are west/east;
    # their meaning must not change when the designer moves the entrance.
    direction_for_role = {
        "north": "north",
        "front": "north",
        "south": "south",
        "back": "south",
        "east": "east",
        "right": "east",
        "west": "west",
        "left": "west",
    }
    direction = direction_for_role.get(role)
    if direction:
        walls = [
            wall
            for wall in walls
            if getattr(wall.direction, "value", None) == direction
        ]
    return walls


def _wall_free_capacity_spans(layout: HouseLayout) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    seen_walls: set[str] = set()
    for room in layout.placed_rooms:
        for wall in room.walls:
            if wall.wall_id in seen_walls:
                continue
            seen_walls.add(wall.wall_id)
            spans.extend(
                {
                    "wall": wall,
                    "start": start,
                    "end": end,
                }
                for start, end in opening_free_spans(wall)
            )
    return spans


def _wall_capacity_issues(
    layout: HouseLayout,
    reservations: list[FloorPlanReservation],
    *,
    spans: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    spans = spans if spans is not None else _wall_free_capacity_spans(layout)

    requests: list[tuple[float, FloorPlanReservation]] = []
    for reservation in reservations:
        requests.extend(
            (reservation.min_wall_width_m, reservation)
            for _ in range(reservation.count)
        )
    requests.sort(key=lambda item: item[0], reverse=True)
    issues: list[dict[str, Any]] = []
    for width, reservation in requests:
        candidates = [
            span
            for span in spans
            if span["end"] - span["start"] >= width
            and span["wall"] in _candidate_walls(layout, reservation)
        ]
        if not candidates:
            issues.append(
                {
                    "issue_type": "insufficient_opening_free_wall_capacity",
                    "reservation_id": reservation.reservation_id,
                    "required_width_m": width,
                    "count": reservation.count,
                }
            )
            continue
        selected = min(
            candidates,
            key=lambda span: span["end"] - span["start"],
        )
        selected["start"] += width
    return issues


def _opening_adjacency_capacity_issues(
    layout: HouseLayout,
    reservations: list[FloorPlanReservation],
    *,
    spans: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Allocate non-overlapping wall capacity immediately beside windows."""
    spans = spans if spans is not None else _wall_free_capacity_spans(layout)
    capacities: list[dict[str, Any]] = []
    for room in layout.placed_rooms:
        for wall in room.walls:
            for opening in wall.openings:
                if opening.opening_type != OpeningType.WINDOW:
                    continue
                opening_start = float(opening.position_along_wall)
                opening_end = opening_start + max(0.0, float(opening.width))
                left_edge = max(0.0, opening_start - _OPENING_MARGIN_M)
                right_edge = min(float(wall.length), opening_end + _OPENING_MARGIN_M)
                for span in spans:
                    if span["wall"].wall_id != wall.wall_id:
                        continue
                    start, end = span["start"], span["end"]
                    side = ""
                    if abs(end - left_edge) <= 1e-6:
                        side = "left"
                    elif abs(start - right_edge) <= 1e-6:
                        side = "right"
                    if side:
                        capacities.append(
                            {
                                "wall": wall,
                                "window_id": opening.opening_id,
                                "side": side,
                                "span": span,
                            }
                        )

    issues: list[dict[str, Any]] = []
    requests: list[tuple[float, FloorPlanReservation]] = []
    for reservation in reservations:
        requests.extend(
            (reservation.min_wall_width_m, reservation)
            for _ in range(reservation.count)
        )
    requests.sort(key=lambda item: item[0], reverse=True)
    for width, reservation in requests:
        candidate_wall_ids = {
            wall.wall_id for wall in _candidate_walls(layout, reservation)
        }
        matching_window_exists = any(
            wall.wall_id in candidate_wall_ids
            and any(
                opening.opening_type == OpeningType.WINDOW for opening in wall.openings
            )
            for room in layout.placed_rooms
            for wall in room.walls
        )
        matching = [
            capacity
            for capacity in capacities
            if capacity["wall"].wall_id in candidate_wall_ids
        ]
        if not matching:
            issues.append(
                {
                    "issue_type": (
                        "insufficient_window_side_capacity"
                        if matching_window_exists
                        else "missing_matching_window_for_opening_adjacency"
                    ),
                    "reservation_id": reservation.reservation_id,
                    "required_width_m": width,
                    "count": reservation.count,
                }
            )
            continue
        feasible = [
            capacity
            for capacity in matching
            if capacity["span"]["end"] - capacity["span"]["start"] >= width
        ]
        if not feasible:
            issues.append(
                {
                    "issue_type": "insufficient_window_side_capacity",
                    "reservation_id": reservation.reservation_id,
                    "required_width_m": width,
                    "count": reservation.count,
                }
            )
            continue
        selected = min(
            feasible,
            key=lambda capacity: (
                capacity["span"]["end"] - capacity["span"]["start"],
                str(capacity["window_id"]),
                str(capacity["side"]),
            ),
        )
        if selected["side"] == "left":
            selected["span"]["end"] -= width
        else:
            selected["span"]["start"] += width
    return issues


def _axis_feasible(first: Wall, second: Wall, width: float) -> bool:
    if first.direction is None or second.direction is None:
        return False
    if first.direction.value == second.direction.value:
        return False
    opposite = {
        "north": "south",
        "south": "north",
        "east": "west",
        "west": "east",
    }
    if opposite[first.direction.value] != second.direction.value:
        return False
    half = width / 2.0
    first_centers = [
        (start + half, end - half)
        for start, end in opening_free_spans(first)
        if end - start >= width
    ]
    second_centers = [
        (start + half, end - half)
        for start, end in opening_free_spans(second)
        if end - start >= width
    ]
    return any(
        max(a0, b0) <= min(a1, b1)
        for a0, a1 in first_centers
        for b0, b1 in second_centers
    )


def _window_counts_by_wall(layout: HouseLayout) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for window in layout.windows:
        direction = window.wall_direction.value if window.wall_direction else ""
        key = (window.room_id, direction or window.boundary_label)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _room_floor_area(room: Any) -> float:
    """Use the exact polygon area when the room has a non-rectangular footprint."""
    vertices = getattr(room, "footprint_vertices", None)
    if vertices:
        try:
            return polygon_area_m2(vertices)
        except (TypeError, ValueError):
            pass
    return float(room.width) * float(room.depth)


def _area_tolerance(expected_area: float, geometry: Any) -> float:
    return max(
        float(getattr(geometry, "absolute_area_tolerance_m2", 0.10)),
        abs(expected_area) * float(getattr(geometry, "relative_area_tolerance", 0.01)),
    )


def _explicit_geometry_issues(
    layout: HouseLayout, manifest: FloorPlanReservationManifest
) -> list[dict[str, Any]]:
    """Verify that an advisory is granted only to the requested final geometry."""
    geometry = manifest.explicit_geometry
    if not geometry.detected or not geometry.verify_geometry_match:
        return []
    if len(layout.placed_rooms) != 1:
        return [
            {
                "issue_type": "explicit_geometry_mismatch",
                "reason": "expected_single_room",
                "expected_mode": geometry.mode,
                "actual_room_count": len(layout.placed_rooms),
            }
        ]

    room = layout.placed_rooms[0]
    actual_area = _room_floor_area(room)
    issue_base = {
        "issue_type": "explicit_geometry_mismatch",
        "expected_mode": geometry.mode,
        "expected_area_m2": geometry.expected_area_m2,
        "actual_area_m2": actual_area,
        "evidence": list(geometry.evidence),
    }
    issues: list[dict[str, Any]] = []

    if geometry.mode == "room":
        expected_dimensions = (
            geometry.expected_width_m,
            geometry.expected_length_m,
        )
        if None in expected_dimensions:
            return [{**issue_base, "reason": "missing_expected_dimensions"}]
        actual_dimensions = (float(room.width), float(room.depth))
        tolerance = float(geometry.vertex_tolerance_m)
        same_order = all(
            abs(actual - expected) <= tolerance
            for actual, expected in zip(actual_dimensions, expected_dimensions)
        )
        swapped_order = all(
            abs(actual - expected) <= tolerance
            for actual, expected in zip(
                actual_dimensions, reversed(expected_dimensions)
            )
        )
        if not same_order and not swapped_order:
            issues.append(
                {
                    **issue_base,
                    "reason": "dimensions",
                    "expected_dimensions_m": list(expected_dimensions),
                    "actual_dimensions_m": list(actual_dimensions),
                }
            )
    elif geometry.mode == "polygon":
        actual_vertices = getattr(room, "footprint_vertices", None)
        try:
            normalized_actual = normalize_polygon_vertices(actual_vertices or [])
        except (TypeError, ValueError):
            normalized_actual = []
        expected_vertices = list(geometry.expected_vertices)
        tolerance = float(geometry.vertex_tolerance_m)
        vertices_match = len(normalized_actual) == len(expected_vertices) and all(
            abs(actual[0] - expected[0]) <= tolerance
            and abs(actual[1] - expected[1]) <= tolerance
            for actual, expected in zip(normalized_actual, expected_vertices)
        )
        if not vertices_match:
            issues.append(
                {
                    **issue_base,
                    "reason": "vertices",
                    "expected_vertex_count": len(expected_vertices),
                    "actual_vertex_count": len(normalized_actual),
                }
            )
    else:
        issues.append({**issue_base, "reason": "unknown_geometry_mode"})

    if abs(actual_area - geometry.expected_area_m2) > _area_tolerance(
        geometry.expected_area_m2, geometry
    ):
        issues.append({**issue_base, "reason": "area"})
    return issues


def validate_floor_plan_reservations(
    layout: HouseLayout,
    manifest: FloorPlanReservationManifest | dict[str, Any] | None,
) -> FloorPlanReservationValidation:
    if manifest is None:
        return FloorPlanReservationValidation()
    if isinstance(manifest, dict):
        manifest = FloorPlanReservationManifest.model_validate(manifest)
    if not manifest.enabled:
        return FloorPlanReservationValidation()

    result = FloorPlanReservationValidation()
    result.issues.extend(_explicit_geometry_issues(layout, manifest))
    hard_reservations = [item for item in manifest.reservations if item.hard]
    wall_capacity_spans = _wall_free_capacity_spans(layout)
    wall_reservations = [
        item for item in hard_reservations if item.kind == "wall_anchor"
    ]
    opening_adjacency_reservations = [
        item for item in hard_reservations if item.kind == "opening_adjacency"
    ]
    result.issues.extend(
        _opening_adjacency_capacity_issues(
            layout,
            opening_adjacency_reservations,
            spans=wall_capacity_spans,
        )
    )
    result.issues.extend(
        _wall_capacity_issues(
            layout,
            wall_reservations,
            spans=wall_capacity_spans,
        )
    )
    zone_area_by_room_type: dict[str, float] = {}
    for reservation in hard_reservations:
        if reservation.kind != "functional_zone":
            continue
        room_type = _normalized_room_type(reservation.room_type)
        zone_area_by_room_type[room_type] = (
            zone_area_by_room_type.get(room_type, 0.0) + reservation.min_zone_area_m2
        )
    for room_type, required_zone_area in zone_area_by_room_type.items():
        available_zone_area = sum(
            _room_floor_area(room) for room in _rooms_for_type(layout, room_type)
        )
        if available_zone_area < required_zone_area:
            issue = {
                "room_type": room_type,
                "required_m2": required_zone_area,
                "available_m2": available_zone_area,
            }
            geometry = manifest.explicit_geometry
            if geometry.detected and geometry.functional_zone_area_policy == "advisory":
                result.advisories.append(
                    {
                        **issue,
                        "issue_type": "functional_zone_area_below_advisory_minimum",
                        "policy": "explicit_geometry_advisory",
                        "blocking": False,
                    }
                )
            else:
                result.issues.append(
                    {**issue, "issue_type": "insufficient_functional_zone_area"}
                )

    for reservation in hard_reservations:
        if reservation.kind in {
            "functional_zone",
            "wall_anchor",
            "opening_adjacency",
        }:
            continue
        walls = _candidate_walls(layout, reservation)
        if reservation.kind == "opposed_anchor_pair" and not any(
            _axis_feasible(first, second, reservation.min_wall_width_m)
            for first in walls
            for second in walls
            if first.room_id == second.room_id
        ):
            result.issues.append(
                {
                    "issue_type": "missing_opposed_opening_free_media_axis",
                    "reservation_id": reservation.reservation_id,
                    "required_width_m": reservation.min_wall_width_m,
                }
            )

    if manifest.preserve_entrance_route and not any(
        door.door_type == "exterior" for door in layout.doors
    ):
        result.issues.append({"issue_type": "missing_exterior_entrance"})

    total_windows = len(layout.windows)
    if (
        manifest.explicit_window_required
        and total_windows < manifest.explicit_window_count
    ):
        result.issues.append(
            {
                "issue_type": "missing_explicit_windows",
                "required": manifest.explicit_window_count,
                "actual": total_windows,
            }
        )
    if manifest.adaptive_window_budget:
        implicit_windows = max(0, total_windows - manifest.explicit_window_count)
        budget = sum(
            adaptive_implicit_window_budget(room.width * room.depth)
            for room in layout.placed_rooms
        )
        if implicit_windows > budget:
            result.issues.append(
                {
                    "issue_type": "implicit_window_budget_exceeded",
                    "implicit_windows": implicit_windows,
                    "budget": budget,
                    "explicit_windows": manifest.explicit_window_count,
                }
            )
        wall_counts = _window_counts_by_wall(layout)
        wall_overflow = sum(
            max(0, count - manifest.max_implicit_windows_per_wall)
            for count in wall_counts.values()
        )
        if wall_overflow > manifest.explicit_window_count:
            result.issues.append(
                {
                    "issue_type": "implicit_windows_per_wall_exceeded",
                    "max_per_wall": manifest.max_implicit_windows_per_wall,
                }
            )

    capacity_issues = [
        issue
        for issue in result.issues
        if issue["issue_type"]
        not in {"implicit_window_budget_exceeded", "implicit_windows_per_wall_exceeded"}
    ]
    budget_issues = [
        issue
        for issue in result.issues
        if issue["issue_type"]
        in {"implicit_window_budget_exceeded", "implicit_windows_per_wall_exceeded"}
    ]
    if capacity_issues:
        result.future_capacity = "error: " + "; ".join(
            issue["issue_type"] for issue in capacity_issues
        )
    if budget_issues:
        result.opening_budget = "error: " + "; ".join(
            issue["issue_type"] for issue in budget_issues
        )
    return result
