"""Frozen, opt-in four-candidate collection; no quality-conditioned execution."""
from __future__ import annotations

from typing import Any

POOL_PROTOCOL = "sceneexpert.fixed_candidate_pool.v1"


def pool_policy(count: int) -> dict[str, Any]:
    """Declare the only approved extension, without changing a legacy A/B run."""
    if type(count) is not int or count != 4:
        raise ValueError("the fixed candidate pool requires exactly four candidates")
    return {
        "protocol": POOL_PROTOCOL,
        "candidates": ["A", "B", "C", "D"],
        "candidate_count": 4,
        "execution_order": ["B", "C", "D"],
        "canonical": "A",
        "quality_conditioning": False,
        "early_stop_on_preference": False,
        "independence_unit": "original_task_id",
    }


def candidate_names(snapshot: dict[str, Any]) -> tuple[str, ...]:
    """Fail closed on a changed pool declaration; absent means legacy A/B."""
    policy = snapshot.get("candidate_pool")
    if policy is None:
        return ("A", "B")
    if policy != pool_policy(4):
        raise ValueError("candidate pool differs from the frozen four-candidate protocol")
    if snapshot.get("decision_boundary") != "first_mutation":
        raise ValueError("candidate pools require an exact first-mutation request")
    return tuple(policy["candidates"])


def validate_pool_execution(snapshot: dict[str, Any], execution: dict[str, Any]) -> None:
    """Require the entire preregistered pool, including true worker exit codes."""
    names = candidate_names(snapshot)
    if len(names) == 2:
        return
    if execution.get("policy") != snapshot["candidate_pool"]:
        raise ValueError("candidate pool execution policy mismatch")
    if execution.get("scheduled") != list(names[1:]):
        raise ValueError("candidate pool schedule changed")
    exits = execution.get("worker_exits")
    if not isinstance(exits, dict) or list(exits) != list(names[1:]):
        raise ValueError("candidate pool omitted or reordered a scheduled candidate")
    if any(type(code) is not int or code != 0 for code in exits.values()):
        raise ValueError("candidate pool contains incomplete or failed execution")
