"""Explicit relative preferences between imperfect, independently executed scenes."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

RELATIVE_POLICY = "verified_relative_v1"


def report_profile(report: dict[str, Any]) -> dict[str, Any]:
    """Summarize label comparability and physical regressions, retaining report identity."""
    case = report.get("case_pack") or {}
    physics = case.get("physics_evidence") or {}
    summary = (report.get("summary") or {}).get("scene_summary") or {}
    constraints = (case.get("intent_contract") or {}).get("constraints") or []
    depths = [
        float(row["penetration_depth_m"]) for row in physics.get("collisions") or []
    ]
    if (
        physics.get("available") is not True
        or not constraints
        or any(not math.isfinite(value) or value < 0 for value in depths)
    ):
        return {}
    return {
        "schema_version": RELATIVE_POLICY,
        "report_sha256": hashlib.sha256(
            json.dumps(report, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest(),
        "constraints": sorted(str(row["constraint_id"]) for row in constraints),
        "unknown_checks": int(summary.get("unknown", 0)),
        "hard_failures": int(summary.get("fail", 0)),
        "quality_score": summary.get("score"),
        "collision_count": len(depths),
        "max_penetration_m": max(depths, default=0.0),
        "total_penetration_m": sum(depths),
    }


def relative_order(chosen: Any, rejected: Any, minimum_margin: float) -> bool:
    """Require measured improvement with no physical or hard-failure regression."""
    if not math.isfinite(minimum_margin) or minimum_margin <= 0:
        return False
    profiles = []
    for row in (chosen, rejected):
        if not (
            row.outcome.execution_complete is True
            and row.outcome.tool_call_valid is True
            and row.evidence.authoritative
            and row.evidence.kind in {"deterministic", "critic_and_deterministic"}
            and row.evidence.verdict in {"accepted", "rejected"}
            and row.outcome.hard_violation_count is not None
        ):
            return False
        profile = row.evidence.details.get("relative_profile") or {}
        if (
            profile.get("schema_version") != RELATIVE_POLICY
            or not profile.get("report_sha256")
            or not profile.get("constraints")
            or profile.get("unknown_checks") != 0
            or profile.get("hard_failures") != row.outcome.hard_violation_count
            or profile.get("quality_score") != row.evidence.quality_score
        ):
            return False
        profiles.append(profile)
    better, worse = profiles
    if better["constraints"] != worse["constraints"]:
        return False
    for key in ("collision_count", "max_penetration_m", "total_penetration_m"):
        values = (better.get(key), worse.get(key))
        if any(
            not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
            for value in values
        ):
            return False
        if values[0] > values[1] + 1e-9:
            return False
    left, right = chosen.evidence.quality_score, rejected.evidence.quality_score
    if (
        left is None
        or right is None
        or not math.isfinite(left)
        or not math.isfinite(right)
        or not 0 <= right <= 1
        or not 0.5 <= left <= 1
        or left - right < minimum_margin
    ):
        return False
    failures = chosen.outcome.hard_violation_count
    other_failures = rejected.outcome.hard_violation_count
    return failures <= other_failures and (failures == 0 or failures < other_failures)


def select_relative(
    records: list[Any], minimum_margin: float
) -> tuple[Any, Any] | None:
    """Choose one widest supported contrast, never order by model identity or latency."""
    options = [
        (left, right)
        for left in records
        for right in records
        if left.trajectory_id != right.trajectory_id
        and relative_order(left, right, minimum_margin)
    ]
    return max(
        options,
        key=lambda pair: (
            pair[0].evidence.quality_score - pair[1].evidence.quality_score,
            pair[0].trajectory_id,
            pair[1].trajectory_id,
        ),
        default=None,
    )
