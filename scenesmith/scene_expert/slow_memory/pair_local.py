"""Fail-closed admission for a preregistered, fully accounted four-slot pool."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from scenesmith.scene_expert.slow_memory.candidate_pool import (
    candidate_names,
    is_pair_local,
    validate_pool_execution,
)
from scenesmith.scene_expert.slow_memory.paired import (
    EXPECTED_MODEL,
    digest,
    load_trajectories,
    read_json,
    tree_hashes,
    validate_candidate_artifacts,
)


def validate_frozen_inputs(group: Path, snapshot: dict[str, Any]) -> None:
    """Require the original wire request, SDK boundary, budget and private base."""
    if not is_pair_local(snapshot):
        raise ValueError("not an explicitly frozen pair-local protocol")
    if tree_hashes(group / "input_scene") != snapshot["files"]:
        raise ValueError("snapshot_files_changed")
    wire = read_json(group / "A/first_request.json")
    boundary = read_json(group / "decision_boundary.json")
    if wire.get("model") != EXPECTED_MODEL or snapshot.get("model") != EXPECTED_MODEL:
        raise ValueError("unexpected_model")
    if boundary.get("wire_request_sha256") != digest(wire):
        raise ValueError("frozen_wire_request_changed")
    if boundary.get("sdk_input_sha256") != digest(snapshot["input"]):
        raise ValueError("frozen_SDK_input_changed")
    if boundary.get("forced_tool_choice") is not False:
        raise ValueError("forced_tool_choice_is_forbidden")
    prefix, remaining = snapshot["prefix_model_turns"], snapshot["remaining_max_turns"]
    maximum = snapshot["cfg"]["agents"]["designer_agent"]["max_turns"]
    if (
        any(type(value) is not int for value in (prefix, remaining, maximum))
        or prefix < 0
        or remaining < 1
        or prefix + remaining != maximum
        or boundary.get("prefix_model_turns") != prefix
    ):
        raise ValueError("native_budget_mismatch")
    if boundary.get("selection") != snapshot.get("boundary_selection"):
        raise ValueError("decision_boundary_selection_changed")
    provenance = snapshot.get("code_provenance", {})
    if provenance.get("identity_kind") != "source_bundle_sha256" or not provenance.get(
        "source_hashes"
    ):
        raise ValueError("frozen_source_provenance_missing")
    expected_hash = hashlib.sha256(
        json.dumps(
            provenance["source_hashes"], sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    if provenance.get("source_bundle_hash") != expected_hash:
        raise ValueError("frozen_source_provenance_hash_mismatch")


def validate_retained_sampling(
    directory: Path, snapshot: dict[str, Any], request: dict[str, Any]
) -> str | None:
    """Check retained draws even when a worker never produced a candidate result."""
    policy = snapshot.get("shadow_sampling")
    if not policy:
        return None
    from scenesmith.scene_expert.slow_memory.scope_sampling import (
        has_design,
        sampling_policy,
        validate_sampling_proof,
    )

    if policy != sampling_policy(policy["max_samples"]):
        raise ValueError("scope sampling policy differs from frozen snapshot")
    paths = sorted((directory / "scope_samples").glob("*.json"))
    if len(paths) > policy["max_samples"]:
        raise ValueError("partial scope sampling exceeded its budget")
    previous_direct = False
    for ordinal, path in enumerate(paths):
        row = read_json(path)
        direct = has_design(row["response_items"])
        if (
            path.name != f"sample_{ordinal:03d}.json"
            or row.get("ordinal") != ordinal
            or previous_direct
            or not row.get("requests")
            or any(value != request for value in row["requests"])
            or row.get("has_direct_design") != direct
            or row.get("tools_executed_before_selection") is not False
        ):
            raise ValueError("partial scope sampling draw or budget mismatch")
        previous_direct = direct
    if (directory / "scope_sampling.json").exists():
        return validate_sampling_proof(directory, policy, request)
    return None


def validate_local_group(
    group: Path,
    snapshot: dict[str, Any],
    *,
    require_fresh_physics: bool = True,
) -> tuple[list[Path], dict[str, Any], list[str]]:
    """Exclude local incomplete/nondirect slots; shared drift invalidates the group."""
    reasons: list[str] = []
    candidates: dict[str, dict[str, Any]] = {}
    sources: list[Path] = []
    exclusions: dict[str, str] = {}
    execution: dict[str, Any] = {}
    names = candidate_names(snapshot)
    try:
        validate_frozen_inputs(group, snapshot)
        execution = read_json(group / "candidate_pool_execution.json")
        validate_pool_execution(snapshot, execution)
        status = read_json(group / "status.json")
        if execution.get("global_error") or status.get("global_error"):
            raise ValueError("recorded_global_execution_failure")
        complete = all(
            slot["state"] == "completed" for slot in execution["slots"].values()
        )
        if (
            execution.get("pool_complete") is not complete
            or status.get("pool_complete") is not complete
        ):
            raise ValueError("pool_completion_record_mismatch")
        if status.get("status") != ("completed" if complete else "pool_failed"):
            raise ValueError("pool_status_mismatch")
        request = read_json(group / "A/first_request.json")
        sampling_hashes: dict[str, str] = {}
        for name in names:
            directory = group / name
            slot = execution["slots"][name]
            failure_path = directory / "failure.json"
            if (
                failure_path.exists()
                and read_json(failure_path).get("failure_scope") == "global"
            ):
                raise ValueError(f"{name}_recorded_shared_guard_failure")
            if name != "A":
                exit_path = directory / "worker_exit.json"
                if slot["state"] != "unstarted":
                    actual = read_json(exit_path)
                    if actual.get("exit_code") != slot["exit_code"]:
                        raise ValueError(f"{name}_actual_worker_exit_mismatch")
                    if slot["state"] == "completed" and (
                        actual.get("process_started") is not True or actual.get("error")
                    ):
                        raise ValueError(f"{name}_completion_without_successful_worker")
                    proof = read_json(group / f"continuation_proof_{name}.json")
                    if any(
                        proof[f"{key}_before"] != proof[f"{key}_after"]
                        for key in ("canonical", "assets", "memory")
                    ):
                        raise ValueError(f"{name}_changed_canonical_state_or_memory")
                elif exit_path.exists() or any(
                    (directory / path).exists()
                    for path in ("first_request.json", "raw_state.json", "result.json")
                ):
                    raise ValueError(f"{name}_unstarted_slot_has_execution_artifacts")
            wire_path = directory / "first_request.json"
            if wire_path.exists() and read_json(wire_path) != request:
                raise ValueError(f"{name}_effective_request_mismatch")
            retained_sampling = (
                validate_retained_sampling(directory, snapshot, request)
                if name != "A"
                else None
            )
            result_path = directory / "result.json"
            if result_path.exists():
                candidate = read_json(result_path)
                if candidate.get("snapshot_hash") != digest(snapshot):
                    raise ValueError(f"{name}_snapshot_mismatch")
            if slot["state"] != "completed":
                exclusions[name] = slot["reason"]
                continue
            candidate = read_json(result_path)
            if name != "A" and snapshot.get("shadow_sampling"):
                if retained_sampling is None:
                    raise ValueError(f"{name}_scope_sampling_proof_missing")
                sampling_hashes[name] = retained_sampling
                if sampling_hashes[name] != candidate.get("shadow_sampling_sha256"):
                    raise ValueError(f"{name}_scope_sampling_proof_mismatch")
            candidate_errors = validate_candidate_artifacts(
                group,
                snapshot,
                name,
                candidate,
                require_fresh_physics=require_fresh_physics,
                names=names,
                sampling_hashes=sampling_hashes,
            )
            if candidate_errors:
                reasons.extend(candidate_errors)
                continue
            rows, diagnostics = load_trajectories(
                [directory / "slow_memory/trajectories.jsonl"]
            )
            if diagnostics or len(rows) != 1:
                raise ValueError(f"{name}_invalid_candidate_capture")
            if rows[0].provenance.get("code_provenance") != snapshot["code_provenance"]:
                raise ValueError(f"{name}_trajectory_source_mismatch")
            from scenesmith.scene_expert.slow_memory.decision_scope import (
                first_turn_has_design,
            )

            if not first_turn_has_design(rows[0].completion_messages):
                exclusions[name] = (
                    "complete natural continuation is not a direct first-action target"
                )
                continue
            candidates[name] = candidate
        if len(candidates) < 2:
            reasons.append("fewer_than_two_complete_direct_candidates")
        if not reasons:
            sources = [
                group / name / "slow_memory/trajectories.jsonl" for name in candidates
            ]
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        reasons.append(f"incomplete_or_changed_shared_evidence: {exc}")
    details = {
        "group": group.name,
        "valid": not reasons,
        "errors": reasons,
        "protocol": snapshot["candidate_pool"]["protocol"],
        "pool_complete": execution.get("pool_complete", False),
        "pair_candidate_pool_eligible": not reasons,
        "complete_direct_candidates": list(candidates),
        "excluded_slots": exclusions,
        "slots": execution.get("slots", {}),
        "native_failure_preserved": not execution.get("pool_complete", False),
    }
    return sources, details, reasons
