"""Deterministic room-containment checks for floor-standing objects."""

from __future__ import annotations

from typing import Any

from shapely.geometry import Polygon

from scenesmith.scenebenchmark_critic.core.geometry import (
    floor_polygon_for_object,
    load_geometry,
    object_footprint_polygon,
)


_CONTAINMENT_TOLERANCE_M = 0.01


def evaluate_room_containment(case_pack: dict[str, Any]) -> list[dict[str, Any]]:
    """Require each floor-standing furniture footprint to remain in its room."""
    store = load_geometry(case_pack)
    if store is None:
        return []

    results: list[dict[str, Any]] = []
    for obj in store.objects.values():
        if str(obj.get("object_type") or "").lower() != "furniture":
            continue
        object_id = str(obj.get("id") or "")
        footprint = object_footprint_polygon(obj)
        room_points = floor_polygon_for_object(store, obj)
        if not object_id or not footprint or not room_points:
            continue
        try:
            object_polygon = Polygon(footprint)
            room_polygon = Polygon(room_points)
        except (TypeError, ValueError):
            continue
        if (
            object_polygon.is_empty
            or room_polygon.is_empty
            or not object_polygon.is_valid
            or not room_polygon.is_valid
        ):
            continue

        tolerated_room = room_polygon.buffer(_CONTAINMENT_TOLERANCE_M, join_style=2)
        contained = bool(tolerated_room.covers(object_polygon))
        outside_area = float(object_polygon.difference(room_polygon).area)
        result = {
            "check_id": f"room_containment__{object_id}",
            "metric": "functional_dependency",
            "label": "pass" if contained else "fail",
            "confidence": 1.0,
            "primary_object": object_id,
            "related_objects": [],
            "selected_related_objects": [],
            "blocking_objects": [],
            "relation_type": "room_containment",
            "reason": (
                f"`{object_id}` footprint is inside the room boundary."
                if contained
                else (
                    f"`{object_id}` footprint extends outside the room by "
                    f"{outside_area:.4f} square metres."
                )
            ),
            "diagnostics": {
                "object_id": object_id,
                "footprint_world": [[float(x), float(y)] for x, y in footprint],
                "room_floor_polygon": [[float(x), float(y)] for x, y in room_points],
                "outside_area_m2": outside_area,
                "tolerance_m": _CONTAINMENT_TOLERANCE_M,
            },
            "evidence": {"constraint": "floor_object_inside_room"},
            "evaluation_source": "scenesmith_room_containment",
            "scoring_tier": "core",
        }
        results.append(result)
    return results
