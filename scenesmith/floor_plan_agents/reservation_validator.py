"""Deterministic future-capacity validation for generated floor plans."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from scenesmith.agent_utils.house import HouseLayout, OpeningType, Wall
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
        walls = [wall for wall in walls if wall.direction.value == direction]
    return walls


def _wall_capacity_issues(
    layout: HouseLayout, reservations: list[FloorPlanReservation]
) -> list[dict[str, Any]]:
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
                    "remaining": end - start,
                }
                for start, end in opening_free_spans(wall)
            )

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
            if span["remaining"] >= width
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
        selected = min(candidates, key=lambda span: span["remaining"])
        selected["remaining"] -= width
    return issues


def _axis_feasible(first: Wall, second: Wall, width: float) -> bool:
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
    hard_reservations = [item for item in manifest.reservations if item.hard]
    wall_reservations = [
        item for item in hard_reservations if item.kind == "wall_anchor"
    ]
    result.issues.extend(_wall_capacity_issues(layout, wall_reservations))
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
            room.width * room.depth for room in _rooms_for_type(layout, room_type)
        )
        if available_zone_area < required_zone_area:
            result.issues.append(
                {
                    "issue_type": "insufficient_functional_zone_area",
                    "room_type": room_type,
                    "required_m2": required_zone_area,
                    "available_m2": available_zone_area,
                }
            )

    for reservation in hard_reservations:
        if reservation.kind in {"functional_zone", "wall_anchor"}:
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
