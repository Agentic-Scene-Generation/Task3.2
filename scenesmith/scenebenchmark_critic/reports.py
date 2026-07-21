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
        "schema_version": "scenesmith.scenebenchmark_critic.report.v1",
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
        f"- All checks / ignored: {scene_summary.get('all_checks', scene_summary.get('total_checks', 0))}/"
        f"{scene_summary.get('excluded_ignored', 0)}",
        f"- Pass/degraded/fail/unknown: {scene_summary.get('pass', 0)}/"
        f"{scene_summary.get('degraded', 0)}/{scene_summary.get('fail', 0)}/"
        f"{scene_summary.get('unknown', 0)}",
        f"- Score: {_fmt(scene_summary.get('score'))}",
        "",
        "## Issues",
        "",
    ]
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
    counted_results = [
        result for result in results if not _is_ignored_scoring_tier(result)
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
                "  Repair priority: shrink the window first, then move it, then "
                "remove it; only move otherwise appropriate furniture afterward."
            )
    return "\n".join(lines)


def _is_prompt_issue(result: dict[str, Any]) -> bool:
    return result.get("label") in {
        "fail",
        "degraded",
        "unknown",
    } and not _is_ignored_scoring_tier(result)


def _is_ignored_scoring_tier(result: dict[str, Any]) -> bool:
    return str(result.get("scoring_tier") or "").strip().lower() == "ignored"


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
