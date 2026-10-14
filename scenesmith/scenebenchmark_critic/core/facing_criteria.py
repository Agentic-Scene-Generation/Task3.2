"""Effective facing criteria shared by evaluation and semantic identity."""

from __future__ import annotations

import math
from typing import Any


def face_dependency_requires_distance(dependency: dict[str, Any]) -> bool:
    """Preserve the evaluator's legacy HSSD orientation-only behavior."""
    if dependency.get("distance_required") is False:
        return False
    if not str(dependency.get("source") or "").startswith("hssd_annotations:"):
        return True
    try:
        distance = float(dependency.get("max_distance_m"))
    except (TypeError, ValueError):
        return False
    return math.isfinite(distance) and distance >= 0.0


def facing_dependency_criteria(dependency: dict[str, Any]) -> dict[str, Any]:
    """Read the existing facing predicate without inferring scene semantics."""

    def number(key: str, default: float) -> float:
        try:
            return float(dependency.get(key))
        except (TypeError, ValueError):
            return default

    return {
        "distance_required": face_dependency_requires_distance(dependency),
        "max_distance_m": number("max_distance_m", 1.8),
        "max_angle_deg": number("max_angle_deg", 60.0),
        "subject_face": str(dependency.get("subject_face") or "front").strip().lower(),
        "target_face": str(dependency.get("target_face") or "any").strip().lower(),
    }
