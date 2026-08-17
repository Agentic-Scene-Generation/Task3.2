"""Authoritative live-layout context for the floor-plan critic."""

from __future__ import annotations

from collections.abc import Iterable

from scenesmith.agent_utils.house import HouseLayout, Wall


def _format_intervals(intervals: Iterable[tuple[float, float]]) -> str:
    return ", ".join(f"{start:.2f}-{end:.2f}m" for start, end in intervals)


def _opening_free_spans(wall: Wall) -> list[tuple[float, float]]:
    """Return maximal opening-free wall intervals in wall-local coordinates."""
    occupied = sorted(
        (
            max(0.0, opening.position_along_wall),
            min(wall.length, opening.position_along_wall + opening.width),
        )
        for opening in wall.openings
    )
    merged: list[tuple[float, float]] = []
    for start, end in occupied:
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    spans: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in merged:
        if start > cursor:
            spans.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < wall.length:
        spans.append((cursor, wall.length))
    return spans


def _wall_label(layout: HouseLayout, wall: Wall) -> str:
    """Find the ASCII boundary label for a wall when one is available."""
    for label, boundary in layout.boundary_labels.items():
        room_a, room_b, direction = boundary
        direction_value = getattr(direction, "value", direction)
        if (
            wall.room_id == room_a
            and room_b is None
            and direction_value == wall.direction.value
        ):
            return label
    return "unlabelled"


def format_floor_plan_critic_context(layout: HouseLayout) -> str:
    """Describe exact live geometry for use alongside the visual floor render.

    The floor-plan renderer is intentionally useful for material and presentation
    review, but its perspective image is not a reliable source for dimensions or
    opening identity.  This compact context is generated directly from the same
    ``HouseLayout`` consumed by validation and geometry export.
    """
    lines = [
        "# AUTHORITATIVE STRUCTURED FLOOR-PLAN STATE",
        "This snapshot is generated directly from the live HouseLayout. For room "
        "dimensions, wall directions, exterior status, opening counts, opening "
        "positions, and opening-free spans, it overrides visual inference. Use "
        "the render only for material and visible presentation assessment. Do not "
        "request an architectural change because an image appears to contradict "
        "these exact fields.",
        "",
        "Validation state: "
        f"placement_valid={layout.placement_valid}; "
        f"connectivity_valid={layout.connectivity_valid}.",
    ]

    if not layout.placed_rooms:
        lines.append("No placed room geometry is available yet.")
        return "\n".join(lines)

    lines.append(f"Placed rooms ({len(layout.placed_rooms)}):")
    for room in layout.placed_rooms:
        area = room.width * room.depth
        lines.append(
            f"- {room.room_id}: {room.width:.2f}m x {room.depth:.2f}m "
            f"({area:.2f}m2), position=({room.position[0]:.2f}, "
            f"{room.position[1]:.2f})."
        )
        for wall in sorted(room.walls, key=lambda item: item.direction.value):
            label = _wall_label(layout, wall)
            opening_text = (
                "; ".join(
                    f"{opening.opening_id}={opening.opening_type.value} "
                    f"[{opening.position_along_wall:.2f}-"
                    f"{opening.position_along_wall + opening.width:.2f}m]"
                    for opening in sorted(
                        wall.openings,
                        key=lambda item: (item.position_along_wall, item.opening_id),
                    )
                )
                or "none"
            )
            spans = _opening_free_spans(wall)
            longest = max((end - start for start, end in spans), default=0.0)
            lines.append(
                f"  - wall {label} ({wall.direction.value}, "
                f"{'exterior' if wall.is_exterior else 'interior'}): "
                f"length={wall.length:.2f}m; openings={opening_text}; "
                f"opening-free spans={_format_intervals(spans) or 'none'} "
                f"(longest={longest:.2f}m)."
            )
    return "\n".join(lines)
