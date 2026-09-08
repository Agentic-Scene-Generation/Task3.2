"""Reconcile prepared memory with observed Designer payloads and native evidence.

An action here is a successful, related tool mutation, NOT proof that a whole
procedure was followed or that memory caused it. Missing evidence stays unknown.
"""

from __future__ import annotations

import json
import re

from pathlib import Path
from typing import Any

from scenesmith.scene_expert.evaluation_costs import timestamp
from scenesmith.scene_expert.evaluation_io import (
    local_reference,
    read_object,
    read_rows,
)
from scenesmith.scene_expert.memory.evidence import (
    evidence_hash,
    resolve_constraint_evidence,
)
from scenesmith.scene_expert.memory.skill_policy import (
    _category_compatible,
    _endpoints_match,
    _relation,
)
from scenesmith.scene_expert.memory.state import state_hash


def _text(value: Any) -> str:
    return (
        value
        if isinstance(value, str)
        else json.dumps(value, ensure_ascii=False, sort_keys=True)
    )


def _leaves(value: Any) -> set[str]:
    if isinstance(value, dict):
        return {leaf for item in value.values() for leaf in _leaves(item)}
    if isinstance(value, list):
        return {leaf for item in value for leaf in _leaves(item)}
    return {str(value)}


def _json_value(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def related_mutations(
    payload: dict, item: dict, stage_entry: dict | None = None
) -> list[dict]:
    """Require successful result + same call ID + exact bound object, not prose."""
    ids = {
        str(oid)
        for binding in (item.get("adaptation") or {}).get("bindings", [])
        for oid in binding.get("object_ids", [])
    }
    entry = stage_entry or {}
    previous_ids = {
        str(row.get("object_id"))
        for row in (
            (entry.get("injection") or {}).get("current_scene_state") or {}
        ).get("objects", [])
    }
    for binding in (item.get("adaptation") or {}).get("bindings", []):
        if binding.get("object_ids"):
            continue
        # Empty bindings mean a role to be created. Confirm the new result ID
        # against the actual post-stage object inventory, never a model summary.
        for row in (entry.get("post_scene_state") or {}).get("objects", []):
            object_id = str(row.get("object_id") or "")
            if (
                object_id
                and object_id not in previous_ids
                and any(
                    _category_compatible(
                        binding.get("current_role", ""), row.get(key, "")
                    )
                    for key in ("name", "category")
                )
            ):
                ids.add(object_id)
    if not ids:
        return []  # A not-yet-created role is not an observed object identity.
    trace = payload.get("agent_trace") or {}
    outputs = {}
    ambiguous = set()
    for result in trace.get("tool_results", []):
        call_id = str(result.get("tool_call_id") or "")
        if call_id in outputs:
            ambiguous.add(call_id)
        outputs[call_id] = _json_value(result.get("output"))
    found = []
    for call in trace.get("tool_calls", []):
        function = call.get("function") or {}
        name = str(function.get("name") or "")
        if not re.match(
            r"^(?:place|move|rotate|update|remove|add|create|arrange|set_object)(?:_|$)",
            name,
        ):
            continue
        call_id = str(call.get("id") or "")
        result = outputs.get(call_id)
        if not call_id or call_id in ambiguous or not isinstance(result, dict):
            continue
        success = result.get("success") is True or result.get("status") in {
            "success",
            "completed",
        }
        if not success or result.get("error") or result.get("success") is False:
            continue
        matched = ids & (
            _leaves(_json_value(function.get("arguments"))) | _leaves(result)
        )
        if matched:
            found.append(
                {
                    "tool_call_id": call_id,
                    "tool_name": name,
                    "object_ids": sorted(matched),
                    "arguments_hash": evidence_hash(function.get("arguments")),
                    "result_hash": evidence_hash(result),
                }
            )
    return found


def target_observations(item: dict, stage: str, stage_entry: dict) -> list[dict]:
    """Join current hard rows by semantics, then reuse per-constraint verification."""
    # Older audit payloads may omit unrelated adaptation fields. Preserve that
    # read compatibility, but never credit outcomes for unselected source rows.
    relations = (item.get("source") or {}).get("spatial_relations", [])
    indices = (item.get("adaptation") or {}).get("source_relation_indices")
    if indices is not None:
        if (
            not isinstance(indices, list)
            or (relations and not indices)
            or any(type(i) is not int or not 0 <= i < len(relations) for i in indices)
            or len(set(indices)) != len(indices)
        ):
            raise ValueError("Invalid accepted relation scope")
        relations = [relations[i] for i in indices]
    context = stage_entry.get("relation_context") or {}
    results = []
    seen = set()
    for constraint in context.get("hard_constraints", []):
        constraint_id = str(constraint.get("constraint_id") or "")
        if not constraint_id or constraint_id in seen:
            continue
        if not any(
            _relation(row.get("relation_type"))
            == _relation(constraint.get("relation") or constraint.get("relation_type"))
            and _endpoints_match(
                str(row.get("subject_role") or ""),
                str(row.get("target_role") or ""),
                constraint,
            )
            for row in relations
        ):
            continue
        seen.add(constraint_id)
        status, evidence = resolve_constraint_evidence(
            constraint, {**stage_entry, "stage": stage}
        )
        results.append(
            {
                "constraint_id": constraint_id,
                "status": status,
                "evidence": [row.model_dump(mode="json") for row in evidence],
            }
        )
    return results


def collect_memory_usage(
    scene_dir: Path, activity: dict, *, stage_filter: str = ""
) -> dict:
    """Malformed optional evidence must not abort generation or claim application."""
    try:
        return _collect_memory_usage(
            scene_dir.resolve(), activity, stage_filter=stage_filter
        )
    except (OSError, TypeError, ValueError, KeyError, AttributeError) as exc:
        return {
            "schema_version": "memory-usage.v1",
            "items": [],
            "initial_checkpoints": {},
            "warnings": [f"memory_usage_evidence_invalid:{type(exc).__name__}"],
            "evidence_complete": False,
        }


def _collect_memory_usage(
    scene_dir: Path, activity: dict, *, stage_filter: str = ""
) -> dict:
    """Read actual request audit files; a prepared context alone is not delivery."""
    warnings: list[str] = []
    payloads: dict[str, list[tuple[str, dict, dict]]] = {}
    seen_paths = set()
    for call in read_rows(scene_dir / "scene_expert/timing/llm_calls.jsonl", warnings):
        if call.get("agent_role") != "designer" or call.get("event") not in {
            "request_initial_design",
            "request_design_change",
        }:
            continue
        stage = str(call.get("stage") or "")
        if stage_filter and stage != stage_filter:
            continue
        path = local_reference(scene_dir, str(call.get("payload_ref") or ""))
        if path is None or not path.is_file():
            warnings.append(f"missing_designer_payload:{stage}")
            continue
        if path in seen_paths:
            continue
        seen_paths.add(path)
        payload = read_object(path, warnings)
        payload_hash = evidence_hash(payload)
        if (
            payload.get("stage") != stage
            or payload.get("agent_role") != "designer"
            or payload.get("event") != call.get("event")
        ):
            warnings.append(f"payload_identity_mismatch:{path}")
            continue
        payloads.setdefault(stage, []).append(
            (
                str(path.relative_to(scene_dir.resolve())),
                {**call, "payload_hash": payload_hash},
                payload,
            )
        )
    rows = []
    entries = [
        (row["stage"], row["entry"])
        for row in activity.get("stage_attempt_history", [])
    ]
    entries += list((activity.get("stages") or {}).items())
    for observation_index, (stage, entry) in enumerate(entries, 1):
        if stage_filter and stage != stage_filter:
            continue
        injection = entry.get("injection") or {}
        decisions: dict[tuple[str, str], list[dict]] = {}
        for decision in injection.get("adaptation_decisions", []):
            decision_key = (
                str(decision.get("memory_type") or ""),
                str(decision.get("memory_id") or ""),
            )
            decisions.setdefault(decision_key, []).append(decision)
        accepted = {
            (
                str((item.get("source") or {}).get("memory_type")),
                str((item.get("source") or {}).get("memory_id")),
            ): item
            for item in injection.get("accepted_items", [])
        }
        for source in (entry.get("retrieval") or {}).get("selections", []):
            key = (
                str(source.get("memory_type") or ""),
                str(source.get("memory_id") or ""),
            )
            item = accepted.get(key)
            evidence = decisions.get(key, [])
            # Older exports may omit these decisions. Unknown is not a model
            # rejection, and retrieval pre-filter reasons are a different step.
            rejection_reasons = sorted(
                {
                    str(reason)
                    for decision in evidence
                    for reason in decision.get("reasons", [])
                    if reason
                }
            )
            requests = []
            targets = target_observations(item, stage, entry) if item else []
            if item:
                for reference, call, payload in payloads.get(stage, []):
                    called_at = timestamp(call.get("created_at", ""))
                    if "prepared_at_epoch" in entry and (
                        called_at is None
                        or called_at < entry["prepared_at_epoch"]
                        or called_at > entry.get("finished_at_epoch", float("inf"))
                    ):
                        continue
                    # The exact full accepted item must occur in the actual request,
                    # not just in a metadata/context snapshot or model's output.
                    text = str(item.get("text") or "")
                    matched = bool(text) and " ".join(text.split()) in " ".join(
                        _text(payload.get("prompt", "")).split()
                    )
                    if not matched:
                        continue
                    if (item.get("source") or {}).get("content_hash") != source.get(
                        "content_hash"
                    ):
                        warnings.append(f"accepted_source_hash_mismatch:{stage}:{key}")
                        continue
                    requests.append(
                        {
                            "payload_ref": reference,
                            "payload_hash": call["payload_hash"],
                            "event": call["event"],
                            "created_at": call.get("created_at"),
                            "stage_execution_attempt": call.get(
                                "stage_execution_attempt"
                            ),
                            "payload_observed": True,
                            "provider_completed": not bool(
                                call.get("error") or payload.get("error")
                            ),
                            "related_mutations": related_mutations(
                                payload, item, entry
                            ),
                        }
                    )
            delivered = any(request["provider_completed"] for request in requests)
            actions = [
                action
                for request in requests
                if request["provider_completed"]
                for action in request["related_mutations"]
            ]
            target_verified = (
                True
                if targets and all(row["status"] == "verified_pass" for row in targets)
                else (
                    False
                    if any(row["status"] == "verified_fail" for row in targets)
                    else None
                )
            )
            rows.append(
                {
                    "stage": stage,
                    "stage_observation_index": observation_index,
                    "memory_type": key[0],
                    "memory_id": key[1],
                    "content_hash": source.get("content_hash", ""),
                    "source_task_ids": source.get("source_task_ids", []),
                    "retrieved": True,
                    "accepted": item is not None,
                    "adaptation_decision_observed": bool(evidence),
                    "adaptation_decisions": evidence,
                    "adaptation_rejection_reasons": rejection_reasons,
                    "payload_observed": bool(requests),
                    "delivered": (
                        delivered
                        if delivered or payloads.get(stage)
                        else None if item else False
                    ),
                    "action_observed": True if delivered and actions else None,
                    "target_verified": target_verified,
                    "first_critic_pass_round": None,
                    "requests": requests,
                    "target_observations": targets,
                    "causal_benefit": None,
                }
            )
    checkpoints = {}
    for stage, candidates in payloads.items():
        initial = candidates  # Restored furniture candidates may start with repair.
        if initial:
            first = min(
                initial,
                key=lambda candidate: str(
                    candidate[1].get("created_at") or candidate[0]
                ),
            )
            state = (first[2].get("context_snapshot") or {}).get("decision_state") or {}
            if state.get("fingerprint") == state_hash(
                {key: value for key, value in state.items() if key != "fingerprint"}
            ) and not state.get("observation_error"):
                checkpoints[stage] = {
                    "fingerprint": state["fingerprint"],
                    "payload_ref": first[0],
                }
    return {
        "schema_version": "memory-usage.v1",
        "items": rows,
        "initial_checkpoints": checkpoints,
        "warnings": sorted(set(warnings)),
        "evidence_complete": not warnings,
        "interpretation": "Delivery, related mutation, and target attainment are distinct; none alone establishes causal gain. A null first_critic_pass_round means the per-round check sequence was not captured; do not infer it from final stage success.",
    }
