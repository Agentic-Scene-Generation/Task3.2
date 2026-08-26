"""Paired SceneEval promotion gate for a trained Slow Memory adapter."""

from __future__ import annotations

import json
import statistics
import time

from pathlib import Path
from typing import Any


_COMPLETE = {"completed", "completed_with_quality_issues"}


def _load_run_metrics(path_or_root: Path) -> dict[str, Any]:
    path = Path(path_or_root)
    if path.is_dir():
        candidate = path / "metrics" / "run_metrics.json"
        path = candidate if candidate.is_file() else path / "run_metrics.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"run metrics must be a JSON object: {path}")
    return payload


def _rate(values: list[bool]) -> float | None:
    return sum(values) / len(values) if values else None


def _delta(after: float | None, before: float | None) -> float | None:
    return after - before if after is not None and before is not None else None


def _metric_number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _identity_values(metrics: dict[str, Any], key: str) -> list[str]:
    return sorted(
        str(value)
        for value in (metrics.get("experiment_identity") or {}).get(key, [])
        if str(value)
    )


def evaluate_scene_level_promotion(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Require controlled paired gains before an adapter becomes deployable."""

    thresholds = thresholds or {}
    failures: list[str] = []
    warnings: list[str] = []
    baseline_rows = {
        str(row.get("case_id") or ""): row
        for row in baseline.get("scenes") or []
        if isinstance(row, dict) and row.get("case_id")
    }
    candidate_rows = {
        str(row.get("case_id") or ""): row
        for row in candidate.get("scenes") or []
        if isinstance(row, dict) and row.get("case_id")
    }
    if set(baseline_rows) != set(candidate_rows):
        failures.append("case sets differ between base and adapter runs")
    baseline_models = _identity_values(baseline, "models")
    candidate_models = _identity_values(candidate, "models")
    if not baseline_models or not candidate_models:
        failures.append("model identity is missing")
    elif baseline_models == candidate_models:
        failures.append("base and adapter runs unexpectedly use the same model")
    for key in ("code_revisions",):
        if _identity_values(baseline, key) != _identity_values(candidate, key):
            failures.append(f"controlled identity mismatch: {key}")
    for key in ("git_status_hash", "source_hashes"):
        left = (baseline.get("code_provenance") or {}).get(key)
        right = (candidate.get("code_provenance") or {}).get(key)
        if not left or left != right:
            failures.append(f"controlled code provenance mismatch: {key}")
    left_memory = sorted(
        str(value)
        for value in (baseline.get("memory_identity") or {}).get("memory_bank_ids", [])
    )
    right_memory = sorted(
        str(value)
        for value in (candidate.get("memory_identity") or {}).get("memory_bank_ids", [])
    )
    if left_memory != right_memory:
        failures.append("memory bank snapshot differs between base and adapter runs")
    if thresholds.get("require_memory_snapshot", True) and not left_memory:
        failures.append(
            "controlled evaluation is missing a frozen memory bank snapshot"
        )
    if not baseline.get("quality_comparison_ready"):
        failures.append("base run is not quality-comparison ready")
    if not candidate.get("quality_comparison_ready"):
        failures.append("adapter run is not quality-comparison ready")

    pairs: list[dict[str, Any]] = []
    for case_id in sorted(set(baseline_rows) & set(candidate_rows)):
        before = baseline_rows[case_id]
        after = candidate_rows[case_id]
        prompt_match = str(before.get("prompt") or "") == str(after.get("prompt") or "")
        if not prompt_match:
            failures.append(f"prompt mismatch: {case_id}")
        before_score = _metric_number(before.get("critic_score"))
        after_score = _metric_number(after.get("critic_score"))
        score_delta = _delta(after_score, before_score)
        before_complete = str(before.get("status") or "") in _COMPLETE
        after_complete = str(after.get("status") or "") in _COMPLETE
        before_hard = bool(
            before.get("hard_constraint_pass", before.get("critic_zero_fail", False))
        )
        after_hard = bool(
            after.get("hard_constraint_pass", after.get("critic_zero_fail", False))
        )
        before_scene_pass = bool(before.get("sceneexpert_pass_scene", False))
        after_scene_pass = bool(after.get("sceneexpert_pass_scene", False))
        before_relation = _metric_number(before.get("relation_satisfaction"))
        after_relation = _metric_number(after.get("relation_satisfaction"))
        relation_delta = _delta(after_relation, before_relation)
        win = bool(
            (not before_complete and after_complete)
            or (
                before_complete
                and after_complete
                and score_delta is not None
                and score_delta > 0
            )
            or (not before_hard and after_hard)
            or (not before_scene_pass and after_scene_pass)
        )
        loss = bool(
            (before_complete and not after_complete)
            or (
                before_complete
                and after_complete
                and score_delta is not None
                and score_delta < 0
            )
            or (before_hard and not after_hard)
            or (before_scene_pass and not after_scene_pass)
        )
        pairs.append(
            {
                "case_id": case_id,
                "prompt_match": prompt_match,
                "baseline_complete": before_complete,
                "candidate_complete": after_complete,
                "baseline_hard_pass": before_hard,
                "candidate_hard_pass": after_hard,
                "baseline_scene_pass": before_scene_pass,
                "candidate_scene_pass": after_scene_pass,
                "baseline_critic_score": before_score,
                "candidate_critic_score": after_score,
                "critic_score_delta": score_delta,
                "baseline_relation_satisfaction": before_relation,
                "candidate_relation_satisfaction": after_relation,
                "relation_satisfaction_delta": relation_delta,
                "win": win and not loss,
                "loss": loss,
            }
        )

    minimum_pairs = int(thresholds.get("minimum_paired_cases", 20))
    if len(pairs) < minimum_pairs:
        failures.append(f"paired cases {len(pairs)} < required {minimum_pairs}")
    score_deltas = [
        float(row["critic_score_delta"])
        for row in pairs
        if row["critic_score_delta"] is not None
    ]
    relation_deltas = [
        float(row["relation_satisfaction_delta"])
        for row in pairs
        if row["relation_satisfaction_delta"] is not None
    ]
    summary = {
        "paired_cases": len(pairs),
        "baseline_completion_rate": _rate([row["baseline_complete"] for row in pairs]),
        "candidate_completion_rate": _rate(
            [row["candidate_complete"] for row in pairs]
        ),
        "baseline_hard_pass_rate": _rate([row["baseline_hard_pass"] for row in pairs]),
        "candidate_hard_pass_rate": _rate(
            [row["candidate_hard_pass"] for row in pairs]
        ),
        "baseline_scene_pass_rate": _rate(
            [row["baseline_scene_pass"] for row in pairs]
        ),
        "candidate_scene_pass_rate": _rate(
            [row["candidate_scene_pass"] for row in pairs]
        ),
        "mean_critic_score_delta": (
            statistics.fmean(score_deltas) if score_deltas else None
        ),
        "mean_relation_satisfaction_delta": (
            statistics.fmean(relation_deltas) if relation_deltas else None
        ),
        "win_rate": _rate([row["win"] for row in pairs]),
        "loss_rate": _rate([row["loss"] for row in pairs]),
    }
    summary["completion_rate_delta"] = _delta(
        summary["candidate_completion_rate"], summary["baseline_completion_rate"]
    )
    summary["hard_pass_rate_delta"] = _delta(
        summary["candidate_hard_pass_rate"], summary["baseline_hard_pass_rate"]
    )
    summary["scene_pass_rate_delta"] = _delta(
        summary["candidate_scene_pass_rate"], summary["baseline_scene_pass_rate"]
    )
    gates = (
        (
            "completion_rate_delta",
            float(thresholds.get("minimum_completion_rate_delta", 0.0)),
        ),
        (
            "hard_pass_rate_delta",
            float(thresholds.get("minimum_hard_pass_rate_delta", 0.0)),
        ),
        (
            "scene_pass_rate_delta",
            float(thresholds.get("minimum_scene_pass_rate_delta", 0.0)),
        ),
        (
            "mean_critic_score_delta",
            float(thresholds.get("minimum_mean_critic_score_delta", 0.0)),
        ),
        (
            "mean_relation_satisfaction_delta",
            float(thresholds.get("minimum_relation_satisfaction_delta", 0.0)),
        ),
    )
    for name, minimum in gates:
        value = summary.get(name)
        if value is None or float(value) < minimum:
            failures.append(f"{name}={value} is below {minimum}")
    win_minus_loss = float(summary["win_rate"] or 0.0) - float(
        summary["loss_rate"] or 0.0
    )
    summary["win_minus_loss_rate"] = win_minus_loss
    minimum_net_win = float(thresholds.get("minimum_win_minus_loss_rate", 0.0))
    if win_minus_loss < minimum_net_win:
        failures.append(
            f"win_minus_loss_rate={win_minus_loss:.6f} is below {minimum_net_win}"
        )
    if not score_deltas:
        warnings.append("no paired critic scores were available")
    if not relation_deltas:
        failures.append("no paired spatial-relation metric was available")
    promotable = not failures
    return {
        "schema_version": "sceneexpert.dpo_scene_promotion.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "promotable": promotable,
        "status": "scene_gate_passed" if promotable else "scene_gate_failed",
        "baseline_run_id": str(baseline.get("run_id") or ""),
        "candidate_run_id": str(candidate.get("run_id") or ""),
        "baseline_models": baseline_models,
        "candidate_models": candidate_models,
        "thresholds": thresholds,
        "summary": summary,
        "failures": sorted(set(failures)),
        "warnings": sorted(set(warnings)),
        "pairs": pairs,
    }


def evaluate_scene_level_paths(
    *,
    baseline: Path,
    candidate: Path,
    thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return evaluate_scene_level_promotion(
        _load_run_metrics(baseline),
        _load_run_metrics(candidate),
        thresholds=thresholds,
    )
