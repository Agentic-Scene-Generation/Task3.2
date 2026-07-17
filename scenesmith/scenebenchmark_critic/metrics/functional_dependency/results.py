from __future__ import annotations

from typing import Any

from scenesmith.scenebenchmark_critic.core.models import (
    FunctionalDependencyProposal,
)


def _target_eval_payload(
    target: dict[str, Any],
    label: str,
    confidence: float,
    reason: str,
    relation_type: str,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    base_score = {"pass": 1.0, "degraded": 0.55, "fail": 0.0, "unknown": 0.0}.get(
        label, 0.0
    )
    payload = {
        "target_id": str(target.get("id") or ""),
        "label": label,
        "confidence": confidence,
        "reason": reason,
        "relation_type": relation_type,
        "semantic_score": 1.0 if label != "fail" else 0.0,
        "distance_score": base_score,
        "orientation_score": base_score,
        "support_score": base_score,
    }
    if evidence:
        payload["evidence"] = evidence
        if "support_evidence_score" in evidence:
            payload["support_evidence_score"] = evidence["support_evidence_score"]
    return payload


def _fd_label_rank(label: str) -> int:
    return {"fail": 0, "unknown": 0, "degraded": 1, "pass": 2}.get(label, 0)


def _relation_label_rank(relation_type: str, label: str) -> int:
    if relation_type == "object_on_support":
        return {"fail": 0, "degraded": 1, "unknown": 2, "pass": 3}.get(label, 0)
    return _fd_label_rank(label)


def _empty_fd_diagnostics() -> dict[str, Any]:
    return {
        "semantic_score": 0.0,
        "distance_score": 0.0,
        "orientation_score": 0.0,
        "support_score": 0.0,
        "cardinality_score": 0.0,
        "selected_target_ids": [],
        "target_evaluations": [],
    }


def _fd_diagnostics_from_targets(
    scored: list[dict[str, Any]], *, selected: list[str]
) -> dict[str, Any]:
    if not scored:
        return _empty_fd_diagnostics()
    selected_set = set(selected)
    chosen = [item for item in scored if item["target_id"] in selected_set] or [
        max(scored, key=lambda item: _fd_label_rank(item["label"]))
    ]

    def avg(key: str) -> float:
        return float(
            sum(float(item.get(key) or 0.0) for item in chosen) / max(len(chosen), 1)
        )

    diagnostics = {
        "semantic_score": avg("semantic_score"),
        "distance_score": avg("distance_score"),
        "orientation_score": avg("orientation_score"),
        "support_score": avg("support_score"),
        "cardinality_score": min(len(chosen), 2) / 2.0,
        "selected_target_ids": [item["target_id"] for item in chosen],
        "target_evaluations": scored,
    }
    for item in chosen:
        evidence = item.get("evidence")
        if isinstance(evidence, dict) and evidence:
            diagnostics.update(evidence)
            break
    return diagnostics


def _proposal_to_check(
    case_pack: dict[str, Any], proposal: FunctionalDependencyProposal
) -> dict[str, Any]:
    target_suffix = "__to__" + "__".join(proposal.target_ids)
    check_id = f"functional_dependency__{proposal.subject_id}{target_suffix}"
    return {
        "check_id": check_id,
        "metric": "functional_dependency",
        "subject_id": proposal.subject_id,
        "target_ids": list(proposal.target_ids),
        "affordance": "contextual",
        "question": (
            f"Task: {case_pack.get('task_instruction') or 'No task instruction provided.'}\n"
            f"Evaluate only the `functional_dependency` metric for `{proposal.subject_id}`. "
            f"Expected relation `{proposal.relation_type}`: {proposal.expected_use}."
        ),
        "prompt_text": "",
        "evidence": {},
        "response_schema": {},
        "relation_type": proposal.relation_type,
        "expected_use": proposal.expected_use,
        "proposal_reason": proposal.reason,
        "check_source": "fd_relation_proposer",
        **_check_scoring_tier_payload(proposal.scoring_tier),
    }


def _check_key(check: dict[str, Any]) -> tuple[str, tuple[str, ...], str]:
    return (
        str(check.get("subject_id") or ""),
        tuple(str(item) for item in (check.get("target_ids") or []) if str(item)),
        str(check.get("relation_type") or check.get("metric") or ""),
    )


def _unknown(check: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "check_id": check.get("check_id"),
        "metric": "functional_dependency",
        "label": "unknown",
        "reason": reason,
        "blocking_objects": [],
        "confidence": 0.0,
        "evaluation_source": "rule_functional_dependency",
        "primary_object": str(check.get("subject_id") or ""),
        "related_objects": [
            str(item) for item in (check.get("target_ids") or []) if str(item)
        ],
        "selected_related_objects": [],
        "relation_type": check.get("relation_type"),
        "diagnostics": _empty_fd_diagnostics(),
        **_result_scoring_tier_payload(check.get("scoring_tier")),
    }


def _normalize_scoring_tier(value: Any) -> str:
    tier = str(value or "core").strip().lower()
    if tier in {"core", "auxiliary", "ignored"}:
        return tier
    return "core"


def _check_scoring_tier_payload(value: Any) -> dict[str, str]:
    tier = _normalize_scoring_tier(value)
    return {"scoring_tier": "ignored"} if tier == "ignored" else {}


def _result_scoring_tier_payload(value: Any) -> dict[str, str]:
    tier = _normalize_scoring_tier(value)
    return {"scoring_tier": "ignored"} if tier == "ignored" else {}


def _scoring_tier_rank(scoring_tier: str) -> int:
    return {"core": 0, "auxiliary": 1, "ignored": 2}.get(
        _normalize_scoring_tier(scoring_tier), 3
    )
