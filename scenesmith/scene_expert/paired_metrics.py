"""Build a guarded cold/warm comparison from two run-metrics bundles."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from scenesmith.scene_expert.experiment_identity import stable_source_bundle_hash

SCHEMA_VERSION = "sceneexpert.paired_metrics.v4"
PAIR_COLUMNS = (
    "case_id",
    "prompt_match",
    "shared_base_match",
    "baseline_status",
    "treatment_status",
    "baseline_generation_status",
    "treatment_generation_status",
    "baseline_requirement_status",
    "treatment_requirement_status",
    "baseline_quality_status",
    "treatment_quality_status",
    "baseline_required_coverage",
    "treatment_required_coverage",
    "required_coverage_delta",
    "outcome_transition",
    "memory_benefit_signal",
    "baseline_time_sec",
    "treatment_time_sec",
    "time_delta_sec",
    "speedup_ratio",
    "baseline_critic_score",
    "treatment_critic_score",
    "critic_score_delta",
    "baseline_sceneexpert_score",
    "treatment_sceneexpert_score",
    "sceneexpert_score_delta",
    "baseline_hard_constraint_pass",
    "treatment_hard_constraint_pass",
    "baseline_relation_satisfaction",
    "treatment_relation_satisfaction",
    "relation_satisfaction_delta",
    "treatment_memory_retrieved",
    "treatment_memory_injected",
    "treatment_cross_task_verified",
)


def _read_metrics(path_or_root: str | Path) -> tuple[dict[str, Any], Path]:
    path = Path(path_or_root).resolve()
    if path.is_dir():
        candidate = path / "metrics" / "run_metrics.json"
        path = candidate if candidate.is_file() else path / "run_metrics.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Run metrics must be a JSON object: {path}")
    return payload, path


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: Iterable[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return round(statistics.fmean(clean), 6) if clean else None


def _median(values: Iterable[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return round(float(statistics.median(clean)), 6) if clean else None


def _delta(after: float | None, before: float | None) -> float | None:
    if after is None or before is None:
        return None
    return round(after - before, 6)


def _paired_sign_statistics(
    values: Iterable[float | None],
    *,
    lower_is_better: bool = False,
) -> dict[str, Any]:
    """Return an exact two-sided sign test without optional dependencies."""
    clean = [float(value) for value in values if value not in {None, 0.0}]
    if lower_is_better:
        clean = [-value for value in clean]
    wins = sum(value > 0 for value in clean)
    losses = sum(value < 0 for value in clean)
    observed = wins + losses
    if not observed:
        return {
            "non_tied_pairs": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": None,
            "two_sided_sign_test_p": None,
        }
    tail = sum(math.comb(observed, index) for index in range(min(wins, losses) + 1))
    p_value = min(1.0, 2.0 * tail / (2**observed))
    return {
        "non_tied_pairs": observed,
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / observed, 6),
        "two_sided_sign_test_p": round(p_value, 8),
    }


def _scene_index(
    metrics: dict[str, Any], label: str, warnings: list[str]
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in metrics.get("scenes") or []:
        if not isinstance(row, dict):
            continue
        case_id = str(row.get("case_id") or "")
        if not case_id:
            warnings.append(f"{label}_scene_missing_case_id")
            continue
        if case_id in indexed:
            warnings.append(f"{label}_duplicate_case_id:{case_id}")
            continue
        indexed[case_id] = row
    return indexed


def _identity_values(metrics: dict[str, Any], key: str) -> list[str]:
    values = (metrics.get("experiment_identity") or {}).get(key) or []
    return sorted(str(value) for value in values if str(value))


_COMPLETED_STATUSES = {"completed", "completed_with_quality_issues"}


def _is_completed(status: str) -> bool:
    return status in _COMPLETED_STATUSES


def _benefit_signal(
    *,
    before_status: str,
    after_status: str,
    time_delta: float | None,
    quality_delta: float | None,
) -> str:
    before_complete = _is_completed(before_status)
    after_complete = _is_completed(after_status)
    if not before_complete and after_complete:
        return "rescued"
    if before_complete and not after_complete:
        return "regressed"
    if not before_complete and not after_complete:
        return "both_incomplete"
    faster = time_delta is not None and time_delta < 0
    slower = time_delta is not None and time_delta > 0
    better = quality_delta is not None and quality_delta > 0
    worse = quality_delta is not None and quality_delta < 0
    if faster and better:
        return "faster_and_better"
    if faster and worse:
        return "faster_quality_tradeoff"
    if slower and better:
        return "better_speed_tradeoff"
    if slower and worse:
        return "slower_and_worse"
    return "mixed_or_tied"


def compare_run_metrics(
    baseline: dict[str, Any], treatment: dict[str, Any]
) -> dict[str, Any]:
    """Return paired deltas and refuse causal claims when controls are invalid."""
    warnings: list[str] = []
    baseline_rows = _scene_index(baseline, "baseline", warnings)
    treatment_rows = _scene_index(treatment, "treatment", warnings)
    baseline_cases = set(baseline_rows)
    treatment_cases = set(treatment_rows)
    if baseline_cases != treatment_cases:
        warnings.append(
            "case_set_mismatch:"
            f"baseline_only={sorted(baseline_cases - treatment_cases)}:"
            f"treatment_only={sorted(treatment_cases - baseline_cases)}"
        )

    identity_checks: dict[str, bool] = {}
    semantic_identity_keys = {
        "experiment_names",
        "control_signatures",
        "models",
    }
    for key in (
        "experiment_names",
        "experiment_signatures",
        "control_signatures",
        "config_hashes",
        "models",
        "code_revisions",
    ):
        baseline_values = _identity_values(baseline, key)
        treatment_values = _identity_values(treatment, key)
        matches = bool(baseline_values) and baseline_values == treatment_values
        identity_checks[key] = matches
        if key in semantic_identity_keys and not matches:
            warnings.append(
                f"identity_mismatch:{key}:"
                f"baseline={baseline_values}:treatment={treatment_values}"
            )
    baseline_provenance = baseline.get("code_provenance") or {}
    treatment_provenance = treatment.get("code_provenance") or {}
    baseline_bundle = str(baseline_provenance.get("source_bundle_hash") or "")
    treatment_bundle = str(treatment_provenance.get("source_bundle_hash") or "")
    if not baseline_bundle and isinstance(
        baseline_provenance.get("source_hashes"), dict
    ):
        baseline_bundle = stable_source_bundle_hash(
            baseline_provenance["source_hashes"]
        )
    if not treatment_bundle and isinstance(
        treatment_provenance.get("source_hashes"), dict
    ):
        treatment_bundle = stable_source_bundle_hash(
            treatment_provenance["source_hashes"]
        )
    bundle_matches = bool(baseline_bundle) and baseline_bundle == treatment_bundle
    identity_checks["code_provenance.source_bundle_hash"] = bundle_matches
    if not bundle_matches:
        warnings.append("identity_mismatch:code_provenance.source_bundle_hash")
    for key in ("git_status_hash", "source_hashes"):
        before_value = baseline_provenance.get(key)
        after_value = treatment_provenance.get(key)
        matches = bool(before_value) and before_value == after_value
        identity_checks[f"code_provenance.{key}"] = matches
    for key in (
        "memory_bank_ids",
        "memory_dirs",
        "snapshot_fingerprints",
        "snapshot_revisions",
    ):
        before_values = sorted(
            str(value)
            for value in (baseline.get("memory_identity") or {}).get(key, [])
            if str(value)
        )
        after_values = sorted(
            str(value)
            for value in (treatment.get("memory_identity") or {}).get(key, [])
            if str(value)
        )
        matches = bool(before_values) and before_values == after_values
        identity_checks[f"memory_identity.{key}"] = matches
        if not matches:
            warnings.append(f"identity_mismatch:memory_identity.{key}")

    baseline_memory_identity = baseline.get("memory_identity") or {}
    treatment_memory_identity = treatment.get("memory_identity") or {}
    snapshots_frozen = bool(
        baseline_memory_identity.get("frozen_all_unchanged") is True
        and treatment_memory_identity.get("frozen_all_unchanged") is True
        and baseline_memory_identity.get("snapshot_identity_stable") is True
        and treatment_memory_identity.get("snapshot_identity_stable") is True
    )
    identity_checks["memory_identity.frozen_all_unchanged"] = snapshots_frozen
    if not snapshots_frozen:
        warnings.append("memory_snapshot_not_frozen")

    baseline_contract = baseline.get("evaluation_contract") or {}
    treatment_contract = treatment.get("evaluation_contract") or {}
    pair_ids_match = bool(baseline_contract.get("pair_ids")) and (
        baseline_contract.get("pair_ids") == treatment_contract.get("pair_ids")
    )
    dimension_matches = baseline_contract.get("controlled_dimensions") == [
        "fast_memory_retrieval"
    ] and treatment_contract.get("controlled_dimensions") == ["fast_memory_retrieval"]
    arms_valid = baseline_contract.get("arms") == [
        "memory_off"
    ] and treatment_contract.get("arms") == ["memory_on"]
    contracts_ready = bool(
        baseline_contract.get("contract_ready") is True
        and treatment_contract.get("contract_ready") is True
    )
    identity_checks["evaluation_contract.pair_id"] = pair_ids_match
    identity_checks["evaluation_contract.controlled_dimension"] = dimension_matches
    identity_checks["evaluation_contract.arms"] = arms_valid
    identity_checks["evaluation_contract.ready"] = contracts_ready
    for key, passed in (
        ("pair_id", pair_ids_match),
        ("controlled_dimension", dimension_matches),
        ("arms", arms_valid),
        ("ready", contracts_ready),
    ):
        if not passed:
            warnings.append(f"evaluation_contract_mismatch:{key}")

    pairs: list[dict[str, Any]] = []
    prompt_mismatch = False
    shared_base_mismatch = False
    for case_id in sorted(baseline_cases & treatment_cases):
        before = baseline_rows[case_id]
        after = treatment_rows[case_id]
        before_time = _number(before.get("trace_time_sec"))
        after_time = _number(after.get("trace_time_sec"))
        before_critic = _number(before.get("critic_score"))
        after_critic = _number(after.get("critic_score"))
        before_wrapper = _number(before.get("sceneexpert_overall_score"))
        after_wrapper = _number(after.get("sceneexpert_overall_score"))
        before_relation = _number(before.get("relation_satisfaction"))
        after_relation = _number(after.get("relation_satisfaction"))
        time_delta = _delta(after_time, before_time)
        critic_delta = _delta(after_critic, before_critic)
        wrapper_delta = _delta(after_wrapper, before_wrapper)
        before_status = str(before.get("status") or "")
        after_status = str(after.get("status") or "")
        before_required_coverage = _number(before.get("required_coverage"))
        after_required_coverage = _number(after.get("required_coverage"))
        if not _is_completed(before_status) and _is_completed(after_status):
            outcome_transition = "rescued"
        elif _is_completed(before_status) and not _is_completed(after_status):
            outcome_transition = "regressed"
        elif _is_completed(before_status) and _is_completed(after_status):
            outcome_transition = "both_completed"
        else:
            outcome_transition = "both_incomplete"
        prompt_match = str(before.get("prompt") or "") == str(after.get("prompt") or "")
        prompt_mismatch = prompt_mismatch or not prompt_match
        before_shared_base = str(before.get("shared_base_fingerprint") or "")
        after_shared_base = str(after.get("shared_base_fingerprint") or "")
        shared_base_match = bool(before_shared_base) and (
            before_shared_base == after_shared_base
        )
        shared_base_mismatch = shared_base_mismatch or not shared_base_match
        pairs.append(
            {
                "case_id": case_id,
                "prompt_match": prompt_match,
                "shared_base_match": shared_base_match,
                "baseline_status": before_status,
                "treatment_status": after_status,
                "baseline_generation_status": str(
                    before.get("generation_status") or "unknown"
                ),
                "treatment_generation_status": str(
                    after.get("generation_status") or "unknown"
                ),
                "baseline_requirement_status": str(
                    before.get("requirement_status") or "unknown"
                ),
                "treatment_requirement_status": str(
                    after.get("requirement_status") or "unknown"
                ),
                "baseline_quality_status": str(
                    before.get("quality_status") or "unknown"
                ),
                "treatment_quality_status": str(
                    after.get("quality_status") or "unknown"
                ),
                "baseline_required_coverage": before_required_coverage,
                "treatment_required_coverage": after_required_coverage,
                "required_coverage_delta": _delta(
                    after_required_coverage,
                    before_required_coverage,
                ),
                "outcome_transition": outcome_transition,
                "memory_benefit_signal": _benefit_signal(
                    before_status=before_status,
                    after_status=after_status,
                    time_delta=time_delta,
                    quality_delta=critic_delta,
                ),
                "baseline_time_sec": before_time,
                "treatment_time_sec": after_time,
                "time_delta_sec": time_delta,
                "speedup_ratio": (
                    round(before_time / after_time, 6)
                    if before_time is not None and after_time not in {None, 0.0}
                    else None
                ),
                "baseline_critic_score": before_critic,
                "treatment_critic_score": after_critic,
                "critic_score_delta": critic_delta,
                "baseline_sceneexpert_score": before_wrapper,
                "treatment_sceneexpert_score": after_wrapper,
                "sceneexpert_score_delta": wrapper_delta,
                "baseline_hard_constraint_pass": before.get("hard_constraint_pass"),
                "treatment_hard_constraint_pass": after.get("hard_constraint_pass"),
                "baseline_relation_satisfaction": before_relation,
                "treatment_relation_satisfaction": after_relation,
                "relation_satisfaction_delta": _delta(
                    after_relation,
                    before_relation,
                ),
                "treatment_memory_retrieved": bool(
                    after.get("memory_retrieved_stages")
                ),
                "treatment_memory_injected": bool(
                    after.get("memory_injection_verified_stages")
                ),
                "treatment_cross_task_verified": bool(
                    after.get("memory_cross_task_verified_stages")
                ),
            }
        )
    if prompt_mismatch:
        warnings.append("prompt_mismatch")
    identity_checks["shared_base.per_case_fingerprint"] = bool(pairs) and not (
        shared_base_mismatch
    )
    if shared_base_mismatch:
        warnings.append("shared_base_fingerprint_mismatch")

    baseline_ready = bool(baseline.get("quality_comparison_ready"))
    treatment_ready = bool(treatment.get("quality_comparison_ready"))
    if not baseline_ready:
        warnings.append("baseline_not_quality_ready")
    if not treatment_ready:
        warnings.append("treatment_not_quality_ready")

    baseline_flags_valid = bool(baseline_rows) and all(
        row.get("component_flags", {}).get("fast_memory_retrieval") is False
        and row.get("component_flags", {}).get("memory_writer") is False
        and row.get("component_flags", {}).get("slow_memory_capture") is True
        for row in baseline_rows.values()
    )
    treatment_flags_valid = bool(treatment_rows) and all(
        row.get("component_flags", {}).get("fast_memory_retrieval") is True
        and row.get("component_flags", {}).get("memory_writer") is False
        and row.get("component_flags", {}).get("slow_memory_capture") is True
        for row in treatment_rows.values()
    )
    identity_checks["component_flags.baseline_memory_off_full"] = baseline_flags_valid
    identity_checks["component_flags.treatment_memory_on_full"] = treatment_flags_valid
    if not baseline_flags_valid:
        warnings.append("baseline_component_contract_invalid")
    if not treatment_flags_valid:
        warnings.append("treatment_component_contract_invalid")
    baseline_memory_isolated = all(
        not row.get("memory_retrieved_stages")
        and not row.get("memory_injection_verified_stages")
        for row in baseline_rows.values()
    )
    treatment_delivery_observed = any(
        row.get("memory_injection_verified_stages")
        and row.get("memory_cross_task_verified_stages")
        for row in treatment_rows.values()
    )
    identity_checks["memory_delivery.baseline_isolated"] = baseline_memory_isolated
    identity_checks["memory_delivery.treatment_cross_task_injected"] = (
        treatment_delivery_observed
    )
    if not baseline_memory_isolated:
        warnings.append("baseline_memory_delivery_not_isolated")
    if not treatment_delivery_observed:
        warnings.append("treatment_memory_delivery_not_observed")
    required_identity_keys = {
        "control_signatures",
        "models",
        "code_provenance.source_bundle_hash",
        "memory_identity.memory_bank_ids",
        "memory_identity.snapshot_fingerprints",
        "memory_identity.snapshot_revisions",
        "memory_identity.frozen_all_unchanged",
        "evaluation_contract.pair_id",
        "evaluation_contract.controlled_dimension",
        "evaluation_contract.arms",
        "evaluation_contract.ready",
        "shared_base.per_case_fingerprint",
        "component_flags.baseline_memory_off_full",
        "component_flags.treatment_memory_on_full",
        "memory_delivery.baseline_isolated",
        "memory_delivery.treatment_cross_task_injected",
    }
    outcome_comparison_ready = bool(
        pairs
        and baseline_cases == treatment_cases
        and not prompt_mismatch
        and all(identity_checks.get(key, False) for key in required_identity_keys)
    )
    completed_pairs = [
        row
        for row in pairs
        if _is_completed(row["baseline_status"])
        and _is_completed(row["treatment_status"])
    ]
    quality_delta_ready = bool(
        outcome_comparison_ready
        and completed_pairs
        and baseline_ready
        and treatment_ready
    )
    claim_status = "not_ready"
    if quality_delta_ready:
        claim_status = "paired_quality_and_outcomes_ready"
    elif outcome_comparison_ready:
        claim_status = "paired_outcomes_ready_with_partial_quality"
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline_run_id": str(baseline.get("run_id") or ""),
        "treatment_run_id": str(treatment.get("run_id") or ""),
        "comparison_ready": outcome_comparison_ready,
        "outcome_comparison_ready": outcome_comparison_ready,
        "quality_delta_ready": quality_delta_ready,
        "claim_status": claim_status,
        "identity_checks": identity_checks,
        "data_quality_warnings": sorted(set(warnings)),
        "summary": {
            "paired_cases": len(pairs),
            "completed_pairs": len(completed_pairs),
            "rescued_cases": sum(
                row["outcome_transition"] == "rescued" for row in pairs
            ),
            "regressed_cases": sum(
                row["outcome_transition"] == "regressed" for row in pairs
            ),
            "faster_completed_pairs": sum(
                row["time_delta_sec"] is not None and row["time_delta_sec"] < 0
                for row in completed_pairs
            ),
            "slower_completed_pairs": sum(
                row["time_delta_sec"] is not None and row["time_delta_sec"] > 0
                for row in completed_pairs
            ),
            "critic_quality_wins": sum(
                row["critic_score_delta"] is not None and row["critic_score_delta"] > 0
                for row in completed_pairs
            ),
            "critic_quality_losses": sum(
                row["critic_score_delta"] is not None and row["critic_score_delta"] < 0
                for row in completed_pairs
            ),
            "baseline_completion_rate": (baseline.get("summary") or {}).get(
                "completion_rate"
            ),
            "treatment_completion_rate": (treatment.get("summary") or {}).get(
                "completion_rate"
            ),
            "mean_time_delta_sec": _mean(
                row["time_delta_sec"] for row in completed_pairs
            ),
            "median_speedup_ratio": _median(
                row["speedup_ratio"] for row in completed_pairs
            ),
            "mean_critic_score_delta": _mean(
                row["critic_score_delta"] for row in completed_pairs
            ),
            "mean_sceneexpert_score_delta": _mean(
                row["sceneexpert_score_delta"] for row in completed_pairs
            ),
            "mean_relation_satisfaction_delta": _mean(
                row["relation_satisfaction_delta"] for row in completed_pairs
            ),
            "hard_constraint_pass_wins": sum(
                row["baseline_hard_constraint_pass"] is False
                and row["treatment_hard_constraint_pass"] is True
                for row in completed_pairs
            ),
            "hard_constraint_pass_losses": sum(
                row["baseline_hard_constraint_pass"] is True
                and row["treatment_hard_constraint_pass"] is False
                for row in completed_pairs
            ),
            "mean_required_coverage_delta": _mean(
                row["required_coverage_delta"] for row in pairs
            ),
            "requirement_coverage_wins": sum(
                row["required_coverage_delta"] is not None
                and row["required_coverage_delta"] > 0
                for row in pairs
            ),
            "requirement_coverage_losses": sum(
                row["required_coverage_delta"] is not None
                and row["required_coverage_delta"] < 0
                for row in pairs
            ),
            "cross_task_memory_verified_pairs": sum(
                row["treatment_cross_task_verified"] for row in completed_pairs
            ),
            "speed_sign_test": _paired_sign_statistics(
                (row["time_delta_sec"] for row in completed_pairs),
                lower_is_better=True,
            ),
            "critic_score_sign_test": _paired_sign_statistics(
                row["critic_score_delta"] for row in completed_pairs
            ),
            "relation_satisfaction_sign_test": _paired_sign_statistics(
                row["relation_satisfaction_delta"] for row in completed_pairs
            ),
        },
        "pairs": pairs,
    }


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    temporary.replace(path)


def write_paired_metrics(
    metrics: dict[str, Any], output_dir: str | Path
) -> dict[str, str]:
    """Write lossless JSON, flat CSV, and a guarded Markdown summary."""
    root = Path(output_dir).resolve()
    json_path = root / "paired_metrics.json"
    csv_path = root / "paired_scene_metrics.csv"
    markdown_path = root / "paired_metrics.md"
    _atomic_write(
        json_path,
        json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    )
    csv_tmp = csv_path.with_suffix(csv_path.suffix + ".tmp")
    csv_tmp.parent.mkdir(parents=True, exist_ok=True)
    with csv_tmp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=PAIR_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(metrics.get("pairs") or [])
    csv_tmp.replace(csv_path)
    lines = [
        "# Harness + memory paired metrics",
        "",
        f"- Baseline: `{metrics['baseline_run_id']}`",
        f"- Treatment: `{metrics['treatment_run_id']}`",
        f"- Comparison ready: `{str(metrics['comparison_ready']).lower()}`",
        f"- Claim status: `{metrics['claim_status']}`",
        "",
        "## Paired KPIs",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in metrics["summary"].items():
        lines.append(f"| `{key}` | {value} |")
    lines.extend(["", "## Data quality", ""])
    warnings = metrics.get("data_quality_warnings") or []
    lines.extend(f"- `{warning}`" for warning in warnings)
    if not warnings:
        lines.append("- No structural comparison warnings detected.")
    _atomic_write(markdown_path, "\n".join(lines) + "\n")
    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "markdown": str(markdown_path),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--treatment", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    baseline, _ = _read_metrics(args.baseline)
    treatment, _ = _read_metrics(args.treatment)
    metrics = compare_run_metrics(baseline, treatment)
    write_paired_metrics(metrics, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
