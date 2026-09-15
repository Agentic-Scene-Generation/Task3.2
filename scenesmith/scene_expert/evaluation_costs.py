"""All-attempt cost accounting with explicit incomplete/overlapping evidence."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path

from scenesmith.scene_expert.evaluation_io import finite_number, read_object, read_rows


def timestamp(value: str) -> float | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.timestamp() if parsed.tzinfo is not None else None
    except (ValueError, TypeError):
        return None


def attempt_directories(scene_dir: Path, scene_index: int) -> list[Path]:
    """Deduplicate symlink aliases; include native clean-process retry archives."""
    archives = sorted(
        (scene_dir.parent / "failed_attempts").glob(
            f"scene_{scene_index:03d}_attempt_*"
        )
    )
    return list(
        dict.fromkeys(
            path.resolve() for path in [*archives, scene_dir] if path.is_dir()
        )
    )


def _trace(scene_dir: Path, scene_index: int, warnings: list[str]) -> dict:
    # Prefer attempt-local evidence; the parent final trace may belong to a later retry.
    for suffix in ("", "_partial"):
        path = scene_dir / f"scene_expert/trace/trace_{scene_index:06d}{suffix}.json"
        if path.is_file():
            return read_object(path, warnings)
    if scene_dir.name == f"scene_{scene_index:03d}":
        path = scene_dir.parent / f"traces/trace_{scene_index:06d}.json"
        if path.is_file():
            return read_object(path, warnings)
    return {}


def collect_attempt_costs(scene_dir: Path, scene_index: int) -> dict:
    """Malformed evidence disables cost claims instead of breaking the collector."""
    try:
        return _collect_attempt_costs(scene_dir, scene_index)
    except (OSError, TypeError, ValueError, KeyError, AttributeError) as exc:
        return {
            "schema_version": "memory-cost.v1",
            "attempts": [],
            "attempt_count": 0,
            "expected_attempt_count": None,
            "all_attempt_cost_complete": False,
            "all_attempt_time_sec": None,
            "observed_attempt_time_lower_bound_sec": None,
            "all_attempt_trace_time_sec": None,
            "final_attempt_trace_time_sec": None,
            "warnings": [f"cost_evidence_invalid:{type(exc).__name__}"],
        }


def _collect_attempt_costs(scene_dir: Path, scene_index: int) -> dict:
    """Never add nested Planner, Designer, tool and render intervals as wall time."""
    warnings: list[str] = []
    attempts = []
    for path in attempt_directories(scene_dir, scene_index):
        status = (
            read_object(path / "scene_status.json", warnings)
            if (path / "scene_status.json").is_file()
            else {}
        )
        trace = _trace(path, scene_index, warnings)
        trace_start = timestamp(
            (trace.get("runtime_identity") or {}).get("scene_started_at", "")
        )
        status_start = timestamp(status.get("started_at", ""))
        start = status_start if status_start is not None else trace_start
        if status.get("started_at") and status_start is None:
            warnings.append(f"attempt_start_evidence_invalid:{path.name}")
            start = None
        if (
            status_start is not None
            and trace_start is not None
            and abs(status_start - trace_start) > 0.001
        ):
            warnings.append(f"attempt_start_evidence_conflict:{path.name}")
            start = None
        end = timestamp(status.get("updated_at", ""))
        terminal = status.get("status") in {
            "completed",
            "completed_with_quality_issues",
            "failed",
        }
        wall = (
            end - start
            if terminal and start is not None and end is not None and end >= start
            else None
        )
        role_costs = defaultdict(
            lambda: {
                "calls": 0,
                "failed_calls": 0,
                "elapsed_observed_sec": 0.0,
                "elapsed_missing_calls": 0,
                "tokens_observed": 0,
                "token_usage_missing_calls": 0,
            }
        )
        debug = read_rows(path / "scene_expert/timing/llm_calls.jsonl", warnings)
        # Files contain completed/failed calls, not additional provider attempts
        # reconstructed from traces. Never add mirrored public-bank events.
        for row in debug:
            if row.get("event_kind", "llm") != "llm":
                continue
            role = str(row.get("agent_role") or "unknown")
            bucket = role_costs[role]
            bucket["calls"] += 1
            bucket["failed_calls"] += bool(row.get("error"))
            elapsed = finite_number(row.get("elapsed_sec"))
            bucket["elapsed_missing_calls"] += elapsed is None
            bucket["elapsed_observed_sec"] += elapsed or 0.0
            tokens = finite_number((row.get("token_usage") or {}).get("total_tokens"))
            bucket["token_usage_missing_calls"] += tokens is None
            bucket["tokens_observed"] += tokens or 0
        retrieval = read_rows(
            path / "scene_expert/timing/memory_retrieval.jsonl", warnings
        )
        hybrid = [row for row in retrieval if row.get("retriever_type") == "hybrid"]
        retrieval = hybrid or retrieval  # Wrapper timings mirror hybrid timings.
        retrieval_values = [
            finite_number(row.get("total_sec", row.get("elapsed_sec")))
            for row in retrieval
        ]
        measured_modules = defaultdict(
            lambda: {
                "events": 0,
                "elapsed_observed_sec": 0.0,
                "elapsed_missing_events": 0,
            }
        )
        for event in read_rows(
            path / "scene_expert/timing/stage_working_timing.jsonl", warnings
        ):
            bucket = measured_modules[str(event.get("module") or "unknown")]
            value = finite_number(event.get("elapsed_sec"))
            bucket["events"] += 1
            bucket["elapsed_missing_events"] += value is None
            bucket["elapsed_observed_sec"] += value or 0.0
        attempts.append(
            {
                "scene_dir": str(path),
                "attempt": status.get("attempt"),
                "status": status.get("status", "missing"),
                "error": status.get("error", ""),
                "failure": status.get("failure") or {},
                "started_at": start,
                "start_time_source": (
                    "unavailable"
                    if start is None
                    else "scene_status" if status_start is not None else "legacy_trace"
                ),
                "finished_at": end,
                "end_to_end_sec": wall,
                "trace_generation_sec": finite_number(trace.get("total_time_sec")),
                "runtime_identity": trace.get("runtime_identity") or {},
                "role_costs": dict(role_costs),
                "retrieval_observed_sec": sum(
                    value for value in retrieval_values if value is not None
                ),
                "retrieval_missing_events": sum(
                    value is None for value in retrieval_values
                ),
                "calls_observed": len(
                    [row for row in debug if row.get("event_kind", "llm") == "llm"]
                ),
                "tool_render_wait_decomposition": "unknown_not_inferred_from_nested_intervals",
            }
        )
        attempts[-1]["measured_module_intervals"] = dict(measured_modules)
    attempt_numbers = [row["attempt"] for row in attempts]
    positive = [value for value in attempt_numbers if type(value) is int and value >= 1]
    expected = max(positive, default=1)
    identity_complete = len(positive) == len(attempts) and sorted(positive) == list(
        range(1, expected + 1)
    )
    complete = (
        bool(attempts)
        and identity_complete
        and all(row["end_to_end_sec"] is not None for row in attempts)
    )
    ordered = sorted(attempts, key=lambda row: row.get("started_at") or 0)
    if complete and any(
        a["finished_at"] > b["started_at"] for a, b in zip(ordered, ordered[1:])
    ):
        complete = False
        warnings.append("overlapping_attempt_intervals")
    if not identity_complete:
        warnings.append("missing_or_duplicate_attempt_identity")
    if not complete:
        warnings.append("all_attempt_wall_cost_incomplete")
    observed = sum(row["end_to_end_sec"] or 0.0 for row in attempts)
    traces = [row["trace_generation_sec"] for row in attempts]
    return {
        "schema_version": "memory-cost.v1",
        "attempts": attempts,
        "attempt_count": len(attempts),
        "expected_attempt_count": expected,
        "all_attempt_cost_complete": complete,
        "all_attempt_time_sec": (
            ordered[-1]["finished_at"] - ordered[0]["started_at"] if complete else None
        ),
        "attempt_service_time_sec": observed if complete else None,
        "observed_attempt_time_lower_bound_sec": observed,
        "all_attempt_trace_time_sec": (
            sum(value for value in traces if value is not None)
            if traces
            and all(value is not None for value in traces)
            and identity_complete
            else None
        ),
        "final_attempt_trace_time_sec": (
            attempts[-1]["trace_generation_sec"] if attempts else None
        ),
        "cost_interpretation": "Per-case elapsed span includes all attempts and retry gaps after first worker start. Also report summed attempt service time. It is not whole-job makespan. Nested role/module intervals are not additive wall time; missing usage is not zero.",
        "warnings": warnings,
    }
