"""Frozen, opt-in four-candidate collection; no quality-conditioned execution."""

from __future__ import annotations

from typing import Any

POOL_PROTOCOL = "sceneexpert.fixed_candidate_pool.v1"
PAIR_LOCAL_PROTOCOL = "sceneexpert.fixed_candidate_pool.pair_local_v2"


def shared_failure(message: str) -> bool:
    """Recognize native source/request/restoration guards, never a poor design."""
    return message.startswith(
        (
            "paired runtime source changed",
            "snapshot files changed",
            "restored scene differs",
            "reconstructed tool schemas differ",
            "reconstructed model settings differ",
            "restored shared workflow state differs",
            "restored safety-controller state differs",
            "shadow sessions must be empty",
            "shadow decision session must be empty",
            "shadow decision request differs",
            "paired collection observed a different request model",
            "scope sampling ",
            "scope sample has no actual wire request evidence",
            "cannot restart a partial shadow sampling invocation",
            "executed first response differs from selected scope sample",
            "candidate evaluator changed the state being scored",
        )
    )


def pool_policy(count: int, completion_policy: str = "all_four_v1") -> dict[str, Any]:
    """Declare the only approved extension, without changing a legacy A/B run."""
    if type(count) is not int or count != 4:
        raise ValueError("the fixed candidate pool requires exactly four candidates")
    if completion_policy not in {"all_four_v1", "pair_local_v2"}:
        raise ValueError("unsupported candidate pool completion policy")
    return {
        "protocol": PAIR_LOCAL_PROTOCOL
        if completion_policy == "pair_local_v2"
        else POOL_PROTOCOL,
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
    if policy not in (pool_policy(4), pool_policy(4, "pair_local_v2")):
        raise ValueError(
            "candidate pool differs from the frozen four-candidate protocol"
        )
    if snapshot.get("decision_boundary") != "first_mutation":
        raise ValueError("candidate pools require an exact first-mutation request")
    return tuple(policy["candidates"])


def is_pair_local(snapshot: dict[str, Any]) -> bool:
    """Recognize only an exact, explicitly frozen v2 policy."""
    candidate_names(snapshot)
    return snapshot.get("candidate_pool", {}).get("protocol") == PAIR_LOCAL_PROTOCOL


def validate_pool_execution(
    snapshot: dict[str, Any], execution: dict[str, Any]
) -> None:
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
    if is_pair_local(snapshot):
        slots = execution.get("slots")
        if not isinstance(slots, dict) or list(slots) != list(names):
            raise ValueError("pair-local execution must account for all four slots")
        if execution.get("terminal") is not True:
            raise ValueError("pair-local execution has no terminal record")
        for name, slot in slots.items():
            state, code = slot.get("state"), slot.get("exit_code")
            if state not in {"completed", "failed", "unstarted"}:
                raise ValueError("pair-local slot is not terminal")
            if name == "A":
                if code is not None:
                    raise ValueError("canonical A has no independent worker exit")
            elif code != exits[name] or (code is not None and type(code) is not int):
                raise ValueError("pair-local slot and actual worker exit disagree")
            if name != "A" and state == "completed" and code != 0:
                raise ValueError("completed worker requires an actual zero exit")
            if state == "unstarted" and code is not None:
                raise ValueError("unstarted slot cannot have a numeric exit")
            if state != "completed" and not slot.get("reason"):
                raise ValueError(
                    "incomplete slot requires a failure or unstarted reason"
                )
        return
    if any(type(code) is not int or code != 0 for code in exits.values()):
        raise ValueError("candidate pool contains incomplete or failed execution")
