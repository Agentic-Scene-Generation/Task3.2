"""Translate persisted physics observations into canonical critic results."""

from __future__ import annotations

from typing import Any

from scenesmith.scenebenchmark_critic.object_taxonomy import (
    constraint_evaluation_stage,
    execution_owner,
    is_structural_anchor,
)


_EXEMPT_CLASSES = frozenset({"expected_support", "soft_furnishing"})
_STRUCTURAL_OBJECT_TYPES = frozenset({"floor", "room", "structural", "wall"})
_STRUCTURAL_CATEGORIES = frozenset({"ceiling", "floor", "room", "wall"})


def evaluate_physics_collision_evidence(
    case_pack: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return canonical results from structured final-physics evidence.

    This intentionally consumes an additive artifact written by the physics
    validator. It never infers collisions from log text or treats absent physics
    evidence as a clean scene.
    """
    evidence = case_pack.get("physics_evidence")
    if not isinstance(evidence, dict):
        return [
            _unavailable_result("No persisted final physics evidence is available.")
        ]
    if not evidence.get("available", False):
        return [
            _unavailable_result(
                str(
                    evidence.get("error") or "Final physics validation was unavailable."
                ),
                evidence=evidence,
            )
        ]

    tolerance = _positive_float(evidence.get("penetration_tolerance_m"), 0.001)
    rows = evidence.get("collisions")
    if not isinstance(rows, list):
        return [
            _unavailable_result(
                "Final physics evidence has no collision list.", evidence=evidence
            )
        ]

    results: list[dict[str, Any]] = []
    objects_by_id = _scene_objects_by_id(case_pack)
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        first = str(row.get("object_a_id") or row.get("subject_id") or "")
        second = str(row.get("object_b_id") or row.get("target_id") or "")
        if not first or not second:
            continue
        collision_class = str(row.get("classification") or "hard").strip().lower()
        depth = _positive_float(row.get("penetration_depth_m"), 0.0)
        wall_side = str(row.get("wall_side") or "").strip().lower()
        signature = (*sorted((first, second)), wall_side)
        if signature in seen:
            continue
        seen.add(signature)
        if collision_class in _EXEMPT_CLASSES or depth <= tolerance:
            continue
        endpoint_owners = {
            object_id: _endpoint_execution_owner(objects_by_id.get(object_id))
            for object_id in (first, second)
        }
        movable_owners = [owner for owner in endpoint_owners.values() if owner]
        repair_owner = (
            constraint_evaluation_stage(*movable_owners) if movable_owners else ""
        )
        diagnostics = {
            "penetration_depth_m": depth,
            "classification": collision_class,
            "wall_side": wall_side,
            "source_phase": evidence.get("source_phase") or "final_physics",
            "endpoint_generation_owners": endpoint_owners,
        }
        if repair_owner:
            diagnostics["earliest_stage"] = repair_owner
        results.append(
            {
                "check_id": "physics_collision__" + "__".join(signature),
                "metric": "physics_collision",
                "relation_type": "physics_penetration",
                "primary_object": first,
                "related_objects": [second],
                "label": "fail",
                "confidence": 1.0,
                "scoring_tier": "core",
                "reason": (
                    f"Final physics reports {depth * 100:.2f}cm penetration between "
                    f"{first} and {second}."
                ),
                "diagnostics": diagnostics,
                "evidence": {"physics_evidence": dict(row)},
            }
        )
    if results:
        return results
    return [
        {
            "check_id": "physics_collision__scene",
            "metric": "physics_collision",
            "relation_type": "physics_penetration",
            "primary_object": "scene",
            "related_objects": [],
            "label": "pass",
            "confidence": 1.0,
            "scoring_tier": "core",
            "reason": "Final physics found no non-exempt collision above tolerance.",
            "diagnostics": {
                "collision_count": len(rows),
                "penetration_tolerance_m": tolerance,
                "source_phase": evidence.get("source_phase") or "final_physics",
            },
            "evidence": {"physics_evidence": evidence},
        }
    ]


def _unavailable_result(
    reason: str, *, evidence: dict[str, Any] | None = None
) -> dict[str, Any]:
    return {
        "check_id": "physics_collision__unavailable",
        "metric": "physics_collision",
        "relation_type": "physics_penetration",
        "primary_object": "scene",
        "related_objects": [],
        "label": "unknown",
        "confidence": 0.0,
        "scoring_tier": "core",
        "reason": reason,
        "diagnostics": {"physics_available": False},
        "evidence": {"physics_evidence": evidence or {}},
    }


def _positive_float(value: Any, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if numeric >= 0.0 else default


def _scene_objects_by_id(case_pack: dict[str, Any]) -> dict[str, dict[str, Any]]:
    geometry = case_pack.get("scene_geometry")
    objects = geometry.get("objects") if isinstance(geometry, dict) else []
    if not isinstance(objects, list):
        return {}
    return {
        str(obj.get("id")): obj
        for obj in objects
        if isinstance(obj, dict) and str(obj.get("id") or "")
    }


def _endpoint_execution_owner(obj: dict[str, Any] | None) -> str:
    """Return a movable endpoint's generation owner, excluding room geometry."""
    if not isinstance(obj, dict):
        return ""
    hints = obj.get("metadata_hints")
    hints = hints if isinstance(hints, dict) else {}
    category = str(obj.get("category_norm") or obj.get("category") or "")
    object_type = (
        str(obj.get("object_type") or hints.get("scene_object_type") or "")
        .strip()
        .lower()
    )
    normalized_category = category.strip().lower()
    if (
        object_type in _STRUCTURAL_OBJECT_TYPES
        or normalized_category in _STRUCTURAL_CATEGORIES
        or is_structural_anchor(category)
    ):
        return ""
    return execution_owner(category, existing_owner=object_type)
