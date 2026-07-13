"""Shared failure taxonomy and repair plan data structures.

SceneExpert should not grow one rule per observed failure.  This module keeps
hard failures in a small taxonomy so deterministic operators and LLM repair
planning can talk about the same problem classes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class FailureCategory(str, Enum):
    MISSING_REQUIRED_OBJECT = "missing_required_object"
    COLLISION_OR_OVERLAP = "collision_or_overlap"
    DOOR_OR_OPENING_CLEARANCE = "door_or_opening_clearance"
    WINDOW_OR_WALL_ACCESS = "window_or_wall_access"
    SUPPORT_INVALID = "support_invalid"
    OUT_OF_BOUNDS = "out_of_bounds"
    ASSET_INVALID = "asset_invalid"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ClassifiedFailure:
    category: FailureCategory
    reason: str
    object_id: str = ""
    severity: str = "hard"
    deterministic: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RepairPlan:
    plan_id: str
    stage: str
    failures: list[ClassifiedFailure]
    operators: list[str] = field(default_factory=list)
    max_attempts: int = 1
    created_at: str = field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )

    @property
    def categories(self) -> set[FailureCategory]:
        return {failure.category for failure in self.failures}

    def to_log_text(self) -> str:
        failure_text = "; ".join(
            f"{failure.category.value}:{failure.reason}" for failure in self.failures
        )
        return (
            f"RepairPlan[{self.stage}] operators={self.operators} "
            f"failures={failure_text}"
        )


def classify_hard_reasons(
    reasons: list[str] | tuple[str, ...] | None,
) -> list[ClassifiedFailure]:
    """Classify deterministic hard-check strings into stable failure classes."""
    failures: list[ClassifiedFailure] = []
    for reason in reasons or []:
        text = str(reason or "").lower()
        category = FailureCategory.UNKNOWN
        if "missing required" in text or "missing_object" in text:
            category = FailureCategory.MISSING_REQUIRED_OBJECT
        elif "door" in text or "open connection" in text or "opening" in text:
            category = FailureCategory.DOOR_OR_OPENING_CLEARANCE
        elif "window" in text or "wall access" in text:
            category = FailureCategory.WINDOW_OR_WALL_ACCESS
        elif "collision" in text or "overlap" in text or "penetration" in text:
            category = FailureCategory.COLLISION_OR_OVERLAP
        elif (
            "support" in text
            or "surface" in text
            or "fell off" in text
            or "fallen" in text
        ):
            category = FailureCategory.SUPPORT_INVALID
        elif "out of bounds" in text or "outside room" in text or "boundary" in text:
            category = FailureCategory.OUT_OF_BOUNDS
        elif (
            "geometry construction failed" in text
            or "missing mesh" in text
            or "invalid mesh" in text
            or "hssd" in text
            or "sdf" in text
            or "qhull" in text
        ):
            category = FailureCategory.ASSET_INVALID
        failures.append(
            ClassifiedFailure(
                category=category,
                reason=str(reason),
                deterministic=category is not FailureCategory.UNKNOWN,
            )
        )
    if not failures:
        failures.append(
            ClassifiedFailure(
                category=FailureCategory.UNKNOWN,
                reason="unknown hard-check failure",
                deterministic=False,
            )
        )
    return failures


def build_repair_plan(
    *,
    stage: str,
    hard_reasons: list[str] | tuple[str, ...] | None,
    max_attempts: int = 1,
) -> RepairPlan:
    failures = classify_hard_reasons(hard_reasons)
    operators: list[str] = []
    categories = {failure.category for failure in failures}
    if FailureCategory.ASSET_INVALID in categories:
        operators.append("replace_invalid_asset")
    if FailureCategory.MISSING_REQUIRED_OBJECT in categories:
        operators.append("ensure_required_object")
    if FailureCategory.COLLISION_OR_OVERLAP in categories:
        operators.append("separate_overlapping_bboxes")
    if FailureCategory.DOOR_OR_OPENING_CLEARANCE in categories:
        operators.append("clear_forbidden_opening_zones")
    if FailureCategory.WINDOW_OR_WALL_ACCESS in categories:
        operators.append("avoid_soft_window_wall_zones")
    if FailureCategory.SUPPORT_INVALID in categories:
        operators.append("resnap_to_valid_support_surface")
    if FailureCategory.OUT_OF_BOUNDS in categories:
        operators.append("project_inside_room_bounds")
    if not operators:
        operators.append("fallback_local_repair_instruction")
    return RepairPlan(
        plan_id=f"{stage}_{abs(hash(tuple(str(f.reason) for f in failures))) & 0xffffffff:08x}",
        stage=stage,
        failures=failures,
        operators=operators,
        max_attempts=max_attempts,
    )
