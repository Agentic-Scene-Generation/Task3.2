"""Bounded, read-only scene observations for cross-task memory adaptation.

No SceneSmith/Drake imports, no geometry inference, and no mutation of the task.
Unavailable measurements are omitted instead of fabricated.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math

from typing import Any

console_logger = logging.getLogger(__name__)


def state_hash(value: Any) -> str:
    """Fingerprint a JSON observation, independent of timestamps and Git."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()
    ).hexdigest()


def observed_roles(state: dict | None) -> list[str]:
    """Observed roles are retrieval evidence, never additional requirements."""
    return list(
        dict.fromkeys(
            str(row.get("category") or row.get("name") or "")
            for row in (state or {}).get("objects", [])
            if row.get("category") or row.get("name")
        )
    )


def _vector(value: Any) -> list[float]:
    result = [round(float(v), 4) for v in value]
    return result if all(math.isfinite(v) for v in result) else []


def build_memory_scene_state(scene: Any | None, *, max_objects: int = 96) -> dict:
    """Observe without letting a diagnostic failure abort native generation."""
    try:
        return _observe_memory_scene_state(scene, max_objects=max_objects)
    except Exception as exc:
        console_logger.warning("Memory scene observation unavailable: %s", exc)
        payload = {
            "schema_version": "memory-scene-state.v1",
            "objects": [],
            "room": {},
            "observation_error": type(exc).__name__,
        }
        return {**payload, "fingerprint": state_hash(payload)}


def _observe_memory_scene_state(scene: Any | None, *, max_objects: int) -> dict:
    """Observe actual objects, poses, bounds and explicit supports without tools."""
    objects = getattr(scene, "objects", {}) or {}
    rows = []
    for object_id, obj in sorted(objects.items(), key=lambda item: str(item[0]))[
        : max(0, max_objects)
    ]:
        row: dict[str, Any] = {
            "object_id": str(object_id),
            "name": str(getattr(obj, "name", "") or "")[:240],
        }
        category = getattr(obj, "category", "")
        row["category"] = str(getattr(category, "value", category) or "")
        kind = getattr(obj, "object_type", "")
        row["object_type"] = str(getattr(kind, "value", kind) or "")
        row["immutable"] = bool(getattr(obj, "immutable", False))
        try:
            row["translation"] = _vector(obj.transform.translation())
            matrix = obj.transform.rotation().matrix()
            row["yaw_deg"] = round(
                math.degrees(math.atan2(float(matrix[1][0]), float(matrix[0][0]))), 3
            )
        except Exception:
            pass
        try:
            bounds = obj.compute_world_bounds()
            if bounds is not None:
                row["bbox_min"], row["bbox_max"] = _vector(bounds[0]), _vector(
                    bounds[1]
                )
                row["size"] = _vector([bounds[1][i] - bounds[0][i] for i in range(3)])
        except Exception as exc:
            row["bounds_unavailable"] = type(exc).__name__
        surfaces = getattr(obj, "support_surfaces", []) or []
        row["support_surface_count"] = len(surfaces)
        row["support_surfaces"] = []
        for surface in surfaces[:8]:
            surface_row = {
                "surface_id": str(getattr(surface, "surface_id", "")),
                "bounds_frame": "surface_local",
            }
            try:
                surface_row.update(
                    {
                        "bbox_min": _vector(surface.bounding_box_min),
                        "bbox_max": _vector(surface.bounding_box_max),
                        "world_translation": _vector(surface.transform.translation()),
                    }
                )
            except Exception:
                pass
            row["support_surfaces"].append(surface_row)
        support = getattr(obj, "support_object_id", None)
        if support is not None:
            row["support_object_id"] = str(support)
        rows.append(row)
    geometry = getattr(scene, "room_geometry", None)
    room = {}
    for name in ("width", "length", "height"):
        value = getattr(geometry, name, None)
        if isinstance(value, (float, int)) and math.isfinite(value):
            room[name + "_m"] = float(value)
    payload = {
        "schema_version": "memory-scene-state.v1",
        "coordinate_frame": "current_room_world; never copy source-world coordinates",
        "room": room,
        "objects": rows,
        "object_count": len(objects),
        "omitted_objects": max(0, len(objects) - len(rows)),
    }
    return {**payload, "fingerprint": state_hash(payload)}


def format_memory_scene_state(state: dict) -> str:
    """Render structured observations as bounded JSON for the existing Planner."""
    return json.dumps(state, ensure_ascii=False, sort_keys=True)
