from __future__ import annotations

import math

from typing import Any


def _label_rank(label: str) -> int:
    return {"fail": 0, "unknown": 0, "degraded": 1, "pass": 2}.get(label, 0)


def _profile_diagnostics(item: dict[str, Any]) -> dict[str, Any]:
    distance = float(item["min_reach_distance_m"])
    return {
        "label": item["label"],
        "access_ratio": item["access_ratio"],
        "access_side": item["access_side"],
        "zone_scores": item["zone_scores"],
        "reachable_stance_count": item["reachable_stance_count"],
        "min_reach_distance_m": None if math.isinf(distance) else distance,
        "reach_posture": item.get("reach_posture"),
        "reach_origin_height_m": item.get("reach_origin_height_m"),
        "blocking_objects": item["blocking_objects"],
    }


def _unknown(check: dict[str, Any], reason: str) -> dict[str, Any]:
    return _result(
        check,
        label="unknown",
        reason=reason,
        confidence=0.0,
        blocking_objects=[],
        diagnostics={},
    )


def _result(
    check: dict[str, Any],
    *,
    label: str,
    reason: str,
    confidence: float,
    blocking_objects: list[str],
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "check_id": check.get("check_id"),
        "metric": "spatial_accessibility",
        "label": label,
        "reason": reason,
        "blocking_objects": blocking_objects,
        "confidence": confidence,
        "evaluation_source": "rule_spatial_accessibility",
        "diagnostics": diagnostics,
    }
