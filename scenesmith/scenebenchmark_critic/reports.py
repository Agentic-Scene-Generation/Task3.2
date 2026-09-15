"""Report serialization for embedded critic results."""

from __future__ import annotations

import json

from pathlib import Path
from typing import Any

from scenesmith.scenebenchmark_critic.config import CriticConfig
from scenesmith.scenebenchmark_critic.aggregation import aggregate_results


def build_evaluation_payload(
    *,
    case_pack: dict[str, Any],
    results: list[dict[str, Any]],
    stage: str,
    scope: str,
    config: CriticConfig,
) -> dict[str, Any]:
    summary = aggregate_results(results, case_pack=case_pack)
    gate = _gate_status(summary, config)
    return {
        "schema_version": "scenesmith.scenebenchmark_critic.report.v2",
        "scope": scope,
        "stage": stage,
        "case_pack": case_pack,
        "results": results,
        "summary": summary,
        "gate": gate,
    }


def write_report(output_dir: Path, payload: dict[str, Any]) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "scenebenchmark_critic.json"
    md_path = output_dir / "scenebenchmark_critic.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(format_markdown_report(payload), encoding="utf-8")
    return json_path, md_path


def format_markdown_report(payload: dict[str, Any]) -> str:
    summary = payload.get("summary") or {}
    scene_summary = summary.get("scene_summary") or {}
    lines = [
        "# SceneBenchmark Critic",
        "",
        f"- Scope: `{payload.get('scope')}`",
        f"- Stage: `{payload.get('stage')}`",
        f"- Gate: `{(payload.get('gate') or {}).get('label', 'n/a')}`",
        f"- Checks: {scene_summary.get('total_checks', 0)}",
        f"- All / excluded auxiliary / ignored: "
        f"{scene_summary.get('all_checks', scene_summary.get('total_checks', 0))}/"
        f"{scene_summary.get('excluded_auxiliary', 0)}/"
        f"{scene_summary.get('excluded_ignored', 0)}",
        f"- Pass/degraded/fail/unknown: {scene_summary.get('pass', 0)}/"
        f"{scene_summary.get('degraded', 0)}/{scene_summary.get('fail', 0)}/"
        f"{scene_summary.get('unknown', 0)}",
        f"- Score: {_fmt(scene_summary.get('score'))}",
        "",
        "## Intent Contract",
        "",
    ]
    execution = ((payload.get("case_pack") or {}).get("intent_contract") or {}).get(
        "execution"
    ) or []
    execution_by_constraint = {
        str(row.get("constraint_id") or ""): row for row in execution
    }
    if not execution:
        lines.append("No compiled intent constraints.")
    for row in execution:
        dependencies = _format_dependency_states(row, execution_by_constraint)
        subjects = ", ".join(row.get("subject_ids") or []) or "unbound"
        targets = ", ".join(row.get("target_ids") or []) or "none"
        provenance = str(
            row.get("inference_reason") or row.get("evidence_span") or ""
        ).strip()
        provenance_text = f"; rationale={provenance}" if provenance else ""
        lines.append(
            f"- `{row.get('constraint_id')}` [{row.get('source')}] "
            f"`{row.get('relation')}`: **{row.get('state')}**; "
            f"subjects={subjects}; targets={targets}; dependencies={dependencies}; "
            f"repair={row.get('repair_strategy') or 'none'}{provenance_text}"
        )
    coverage_requirements = (
        (payload.get("case_pack") or {}).get("intent_contract") or {}
    ).get("coverage_requirements") or []
    coverage_results = {
        str((result.get("diagnostics") or {}).get("requirement_id")): result
        for result in payload.get("results") or []
        if (result.get("diagnostics") or {}).get("requirement_id")
    }
    lines.extend(["", "## Coverage Requirements", ""])
    if not coverage_requirements:
        lines.append("No explicit coverage requirements.")
    for requirement in coverage_requirements:
        if not isinstance(requirement, dict):
            continue
        requirement_id = str(requirement.get("requirement_id") or "")
        result = coverage_results.get(requirement_id) or {}
        diagnostics = result.get("diagnostics") or {}
        status = str(diagnostics.get("coverage_status") or "unreported")
        label = str(result.get("label") or "unknown")
        evidence_span = str(
            requirement.get("evidence_span") or diagnostics.get("evidence_span") or ""
        ).strip()
        evidence_text = f"; evidence={evidence_span}" if evidence_span else ""
        lines.append(
            f"- `{requirement_id}` [{requirement.get('kind')}] "
            f"`{requirement.get('normalized')}`: status=`{status}`, "
            f"disposition=`{requirement.get('disposition', 'compiled')}`, "
            f"label=`{label}`, stage=`{requirement.get('earliest_stage')}`"
            f"{evidence_text}"
        )
    lines.extend(["", "## Issues", ""])
    issue_rows = [
        result for result in payload.get("results") or [] if _is_prompt_issue(result)
    ]
    if not issue_rows:
        lines.append("No degraded or failed checks.")
    for result in issue_rows:
        lines.extend(
            [
                f"### {result.get('check_id')}",
                "",
                f"- Metric: `{result.get('metric')}`",
                f"- Label: `{result.get('label')}`",
                f"- Subject: `{result.get('primary_object')}`",
                f"- Related: `{', '.join(result.get('related_objects') or []) or 'none'}`",
                f"- Reason: {result.get('reason')}",
                "",
            ]
        )
    return "\n".join(lines)


def format_prompt_context(payload: dict[str, Any], *, max_issues: int = 8) -> str:
    results = payload.get("results") or []
    execution_by_constraint = {
        str(row.get("constraint_id") or ""): row
        for row in (
            ((payload.get("case_pack") or {}).get("intent_contract") or {}).get(
                "execution"
            )
            or []
        )
        if isinstance(row, dict) and row.get("constraint_id")
    }
    counted_results = [
        result for result in results if not _is_non_authoritative_scoring_tier(result)
    ]
    issues = [result for result in results if _is_prompt_issue(result)][:max_issues]
    if not issues:
        return (
            "SceneBenchmark geometry critic: no degraded or failed checks in "
            f"{len(counted_results)} counted rule checks."
        )
    lines = [
        "SceneBenchmark geometry critic found rule-level issues. Use this as "
        "geometric evidence alongside visual critique:"
    ]
    for result in issues:
        related = ", ".join(result.get("related_objects") or [])
        suffix = f" related={related}" if related else ""
        lines.append(
            f"- {result.get('label')}: {result.get('metric')} "
            f"subject={result.get('primary_object')}{suffix}. "
            f"{result.get('reason')}"
        )
        repair_advice = str(result.get("repair_advice") or "").strip()
        if repair_advice:
            lines.append(f"  Repair priority: {repair_advice}")
        if str(result.get("check_id") or "").startswith("window_clearance__"):
            lines.append(
                "  Repair priority: shrink the window first, then move it. Remove "
                "only an implicit window when the floor-plan contract permits; "
                "do not move primary furniture merely to preserve the opening."
            )
        constraint = (result.get("evidence") or {}).get("intent_constraint") or {}
        if constraint:
            execution = execution_by_constraint.get(
                str(constraint.get("constraint_id") or ""), {}
            )
            dependency_ids = _format_dependency_states(
                execution, execution_by_constraint
            )
            binding = execution.get("subject_ids") or [result.get("primary_object")]
            binding_text = ", ".join(str(item) for item in binding if item) or "unbound"
            lines.append(
                "  Contract: "
                f"{constraint.get('constraint_id')} source={constraint.get('source')} "
                f"relation={constraint.get('relation')} "
                f"state={execution.get('state') or result.get('contract_state') or 'failed'} "
                f"bound_subjects={binding_text} dependencies={dependency_ids}"
            )
            rationale = str(
                constraint.get("inference_reason")
                or constraint.get("evidence_span")
                or ""
            ).strip()
            if rationale:
                lines.append(f"  Contract rationale: {rationale}")
    return "\n".join(lines)


def _format_dependency_states(
    row: dict[str, Any], execution_by_constraint: dict[str, dict[str, Any]]
) -> str:
    dependency_ids = [
        str(value) for value in row.get("dependency_constraint_ids") or [] if value
    ]
    if not dependency_ids:
        return "none"
    return ", ".join(
        f"{constraint_id}:{execution_by_constraint.get(constraint_id, {}).get('state') or 'unknown'}"
        for constraint_id in dependency_ids
    )


def _is_prompt_issue(result: dict[str, Any]) -> bool:
    diagnostics = result.get("diagnostics") or {}
    if diagnostics.get("requirement_id"):
        return result.get("label") in {"fail", "degraded", "unknown"}
    if result.get("prompt_actionable_auxiliary"):
        return result.get("label") in {"fail", "degraded", "unknown"}
    constraint = (result.get("evidence") or {}).get("intent_constraint") or {}
    return (
        str(result.get("contract_state") or "") == "failed"
        and result.get("label")
        in {
            "fail",
            "degraded",
            "unknown",
        }
        and str(constraint.get("source") or "")
        in {
            "explicit_prompt",
            "task_compiler_inventory",
            "model_inferred",
            "room_ontology",
            "deterministic_fallback",
        }
        and not _is_non_authoritative_scoring_tier(result)
    )


def _is_non_authoritative_scoring_tier(result: dict[str, Any]) -> bool:
    return str(result.get("scoring_tier") or "").strip().lower() in {
        "ignored",
        "auxiliary",
    }


def _gate_status(summary: dict[str, Any], config: CriticConfig) -> dict[str, Any]:
    scene_summary = summary.get("scene_summary") or {}
    fail_count = int(scene_summary.get("fail") or 0)
    degraded_count = int(scene_summary.get("degraded") or 0)
    blocked = config.hard_gate and (
        fail_count >= config.fail_gate_threshold
        or degraded_count >= config.degraded_gate_threshold
    )
    return {
        "enabled": config.hard_gate,
        "blocked": blocked,
        "label": "fail" if blocked else "report_only",
        "fail_count": fail_count,
        "degraded_count": degraded_count,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"
