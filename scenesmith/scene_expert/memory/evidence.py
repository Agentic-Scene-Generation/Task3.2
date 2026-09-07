"""Per-constraint evidence adapters; never infer geometry from stage success."""

from __future__ import annotations

import hashlib
import json

from copy import deepcopy
from typing import Any

from pydantic import ValidationError

from scenesmith.scene_expert.memory.schemas import MemoryCheckEvidence


def evidence_hash(value: Any) -> str:
    """Hash exact JSON evidence deterministically, without changing the source."""
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def constraint_observation(result: dict, constraint: dict, stage: str) -> dict:
    """Preserve a native result as an auditable observation, not a new score."""
    diagnostics = result.get("diagnostics")
    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
    evidence = result.get("evidence")
    evidence = evidence if isinstance(evidence, dict) else {}
    return MemoryCheckEvidence(
        constraint_id=str(constraint.get("constraint_id") or ""),
        check_id=str(result.get("check_id") or ""),
        stage=stage,
        label=str(result.get("label") or "unknown").lower(),
        evaluation_state=str(
            result.get("contract_state") or diagnostics.get("evaluation_state") or ""
        ),
        scoring_tier=str(result.get("scoring_tier") or "core"),
        result_hash=evidence_hash(result),
        constraint_hash=evidence_hash(constraint),
        metric=str(result.get("metric") or ""),
        observations={
            "evidence": evidence,
            "diagnostics": diagnostics,
            "primary_object": result.get("primary_object"),
            "related_objects": result.get("related_objects", []),
            "evaluation_source": result.get("evaluation_source", ""),
        },
    ).model_dump(mode="json")


def resolve_constraint_evidence(
    constraint: dict,
    stage_evidence: dict,
) -> tuple[str, list[MemoryCheckEvidence]]:
    """Join exact stage/constraint content; uncertainty blocks pass promotion.

    Multiple object checks for one relation must all pass. A fail is retained
    even if another object passed. Unknown/degraded checks are not discarded.
    Older reports containing only aggregate counts remain requirement-only.
    """
    report = stage_evidence.get("verify_report") or {}
    if not isinstance(report, dict):
        return "requirement_only", []
    hard_report = report.get("hard_check_report") or {}
    if not isinstance(hard_report, dict):
        return "requirement_only", []
    constraint_id = str(constraint.get("constraint_id") or "")
    if not constraint_id:
        return "requirement_only", []
    stage = str(stage_evidence.get("stage") or report.get("stage") or "")
    matches: list[MemoryCheckEvidence] = []
    invalid_match = False
    for row in hard_report.get("constraint_evidence", []) or []:
        if not isinstance(row, dict) or row.get("constraint_id") != constraint_id:
            continue
        try:
            item = MemoryCheckEvidence.model_validate(row)
        except (ValidationError, TypeError):
            invalid_match = True
            continue
        if (
            item.stage != stage
            or item.constraint_hash != evidence_hash(constraint)
            or not item.check_id
            or not item.result_hash
        ):
            invalid_match = True
            continue
        matches.append(
            item.model_copy(
                update={
                    "scene_state_path": str(
                        stage_evidence.get("scene_state_path") or ""
                    )
                }
            )
        )
    if any(
        item.label == "fail"
        and item.scoring_tier == "core"
        and item.evaluation_state not in {"unknown", "deferred", "not_applicable"}
        for item in matches
    ):
        return "verified_fail", matches
    if invalid_match:
        return "inconclusive", matches
    if not matches:
        return "requirement_only", []
    if all(
        item.label == "pass"
        and item.scoring_tier == "core"
        and item.evaluation_state not in {"unknown", "deferred", "not_applicable"}
        for item in matches
    ):
        return "verified_pass", matches
    return "inconclusive", matches


def writer_prompt_evidence(payload: dict) -> dict:
    """Keep every check outcome in the LLM input without repeating raw geometry.

    Full observations and hashes stay in the trace and deterministic writer
    input. The model sees outcomes/bindings; it does not assign verification.
    """
    projected = deepcopy(payload)
    for stage in projected.get("stages", []) or []:
        if not isinstance(stage, dict):
            continue
        report = stage.get("verify_report") or {}
        if not isinstance(report, dict):
            continue
        hard_report = report.get("hard_check_report") or {}
        if (
            not isinstance(hard_report, dict)
            or "constraint_evidence" not in hard_report
        ):
            continue
        rows = []
        for item in hard_report["constraint_evidence"] or []:
            if not isinstance(item, dict):
                continue
            row = {
                key: item.get(key)
                for key in (
                    "constraint_id",
                    "check_id",
                    "label",
                    "evaluation_state",
                    "scoring_tier",
                    "metric",
                )
            }
            observations = item.get("observations") or {}
            observations = observations if isinstance(observations, dict) else {}
            row["primary_object"] = observations.get("primary_object")
            row["related_objects"] = observations.get("related_objects", [])
            rows.append(row)
        hard_report["constraint_evidence"] = rows
    return projected
