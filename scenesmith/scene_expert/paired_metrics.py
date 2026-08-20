"""Build a guarded cold/warm comparison from two run-metrics bundles."""

from __future__ import annotations

import argparse
import csv
import json
import statistics

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = "sceneexpert.paired_metrics.v1"
PAIR_COLUMNS = (
    "case_id",
    "prompt_match",
    "baseline_status",
    "treatment_status",
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
    for key in ("experiment_names", "config_hashes", "models", "code_revisions"):
        baseline_values = _identity_values(baseline, key)
        treatment_values = _identity_values(treatment, key)
        matches = bool(baseline_values) and baseline_values == treatment_values
        identity_checks[key] = matches
        if not matches:
            warnings.append(
                f"identity_mismatch:{key}:"
                f"baseline={baseline_values}:treatment={treatment_values}"
            )
    baseline_provenance = baseline.get("code_provenance") or {}
    treatment_provenance = treatment.get("code_provenance") or {}
    for key in ("git_status_hash", "source_hashes"):
        before_value = baseline_provenance.get(key)
        after_value = treatment_provenance.get(key)
        matches = bool(before_value) and before_value == after_value
        identity_checks[f"code_provenance.{key}"] = matches
        if not matches:
            warnings.append(f"identity_mismatch:code_provenance.{key}")
    for key in ("memory_bank_ids", "memory_dirs"):
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

    pairs: list[dict[str, Any]] = []
    prompt_mismatch = False
    for case_id in sorted(baseline_cases & treatment_cases):
        before = baseline_rows[case_id]
        after = treatment_rows[case_id]
        before_time = _number(before.get("trace_time_sec"))
        after_time = _number(after.get("trace_time_sec"))
        before_critic = _number(before.get("critic_score"))
        after_critic = _number(after.get("critic_score"))
        before_wrapper = _number(before.get("sceneexpert_overall_score"))
        after_wrapper = _number(after.get("sceneexpert_overall_score"))
        prompt_match = str(before.get("prompt") or "") == str(after.get("prompt") or "")
        prompt_mismatch = prompt_mismatch or not prompt_match
        pairs.append(
            {
                "case_id": case_id,
                "prompt_match": prompt_match,
                "baseline_status": str(before.get("status") or ""),
                "treatment_status": str(after.get("status") or ""),
                "baseline_time_sec": before_time,
                "treatment_time_sec": after_time,
                "time_delta_sec": _delta(after_time, before_time),
                "speedup_ratio": (
                    round(before_time / after_time, 6)
                    if before_time is not None and after_time not in {None, 0.0}
                    else None
                ),
                "baseline_critic_score": before_critic,
                "treatment_critic_score": after_critic,
                "critic_score_delta": _delta(after_critic, before_critic),
                "baseline_sceneexpert_score": before_wrapper,
                "treatment_sceneexpert_score": after_wrapper,
                "sceneexpert_score_delta": _delta(after_wrapper, before_wrapper),
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

    baseline_ready = bool(baseline.get("quality_comparison_ready"))
    treatment_ready = bool(treatment.get("quality_comparison_ready"))
    if not baseline_ready:
        warnings.append("baseline_not_quality_ready")
    if not treatment_ready:
        warnings.append("treatment_not_quality_ready")
    comparison_ready = bool(
        pairs
        and baseline_cases == treatment_cases
        and not prompt_mismatch
        and baseline_ready
        and treatment_ready
        and all(identity_checks.values())
    )
    completed_pairs = [
        row
        for row in pairs
        if row["baseline_status"] in {"completed", "completed_with_quality_issues"}
        and row["treatment_status"] in {"completed", "completed_with_quality_issues"}
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline_run_id": str(baseline.get("run_id") or ""),
        "treatment_run_id": str(treatment.get("run_id") or ""),
        "comparison_ready": comparison_ready,
        "claim_status": "paired_deltas_ready" if comparison_ready else "not_ready",
        "identity_checks": identity_checks,
        "data_quality_warnings": sorted(set(warnings)),
        "summary": {
            "paired_cases": len(pairs),
            "completed_pairs": len(completed_pairs),
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
            "cross_task_memory_verified_pairs": sum(
                row["treatment_cross_task_verified"] for row in completed_pairs
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
