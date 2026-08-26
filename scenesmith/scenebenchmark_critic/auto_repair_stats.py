"""Append-only, JSON-safe audit events for critic-driven deterministic repairs."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any
from uuid import uuid4


SCHEMA_VERSION = "scenesmith.auto_repair_stats.v1"
_MAX_CALLS = 100


def record_auto_repair_call(
    scene: Any,
    *,
    module: str,
    stage: str,
    status: str,
    skip_reason: str | None = None,
    candidate_budget: int | None = None,
    candidates_evaluated: int = 0,
    accepted_rounds: int = 0,
    fix_records: int = 0,
    object_ids: Iterable[str] = (),
    relation_types: Iterable[str] = (),
) -> dict[str, Any]:
    """Record one finalized call without retaining live transaction objects.

    Callers must invoke this only after their owning transaction commits.  The
    helper intentionally has no references to evaluator or scene classes so it
    remains safe during checkpoint JSON serialization.
    """
    metadata = getattr(scene, "metadata", None)
    if not isinstance(metadata, dict):
        return {}
    stats = metadata.get("auto_repair_stats")
    if not isinstance(stats, dict) or stats.get("schema_version") != SCHEMA_VERSION:
        stats = {"schema_version": SCHEMA_VERSION, "calls": [], "totals": {}}
        metadata["auto_repair_stats"] = stats
    calls = stats.setdefault("calls", [])
    totals = stats.setdefault("totals", {})
    event = {
        "call_id": str(uuid4()),
        "parent_call_id": None,
        "module": str(module),
        "stage": str(stage),
        "status": str(status),
        "skip_reason": skip_reason,
        "candidate_budget": candidate_budget,
        "candidates_evaluated": int(candidates_evaluated),
        "accepted_rounds": int(accepted_rounds),
        "fix_records": int(fix_records),
        "object_ids": sorted({str(value) for value in object_ids}),
        "relation_types": sorted({str(value) for value in relation_types}),
    }
    calls.append(event)
    if len(calls) > _MAX_CALLS:
        dropped = len(calls) - _MAX_CALLS
        del calls[:dropped]
        stats["dropped_call_count"] = int(stats.get("dropped_call_count", 0)) + dropped
    bucket = totals.setdefault(
        str(module),
        {"calls": 0, "candidates_evaluated": 0, "accepted_rounds": 0, "fix_records": 0,
         "skipped": 0, "rolled_back": 0},
    )
    if status == "committed":
        for key in ("calls", "candidates_evaluated", "accepted_rounds", "fix_records"):
            bucket[key] += int(event[key]) if key != "calls" else 1
    elif status == "skipped":
        bucket["skipped"] += 1
    elif status == "rolled_back":
        bucket["rolled_back"] += 1
    return event
