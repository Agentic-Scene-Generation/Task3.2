"""Audit collection completeness separately from preference-training readiness."""

from __future__ import annotations

import hashlib
import json

from collections import Counter
from pathlib import Path
from typing import Any

from scenesmith.scene_expert.slow_memory.dpo import (
    DEFAULT_TRAINING_TASK_TYPES,
    export_dpo_dataset,
    load_trajectories,
)
from scenesmith.scene_expert.slow_memory.schemas import TrajectoryRecord


def audit_collection(
    run_root: Path,
    output_dir: Path,
    *,
    expected_model: str,
    min_pairs: int = 0,
) -> dict[str, Any]:
    """Export a diagnostic DPO probe and report whether further collection is safe.

    ``min_pairs=0`` checks an observer-collection pilot. It does not make an empty
    export trainable. Positive values additionally require a valid DPO package.
    Scene-quality failures remain observations; they are never rewritten as passes.
    """
    if min_pairs < 0:
        raise ValueError("min_pairs must be nonnegative")
    run_root, output_dir = Path(run_root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = export_dpo_dataset(
        trajectory_sources=[run_root], output_dir=output_dir / "dpo_probe"
    )
    records, load_errors = load_trajectories([run_root])
    designers = [r for r in records if r.task_type in DEFAULT_TRAINING_TASK_TYPES]
    errors = sorted({row["reason"] for row in load_errors})
    if not designers:
        errors.append("no_designer_records")
    if any(r.model_id != expected_model for r in records):
        errors.append("unexpected_model")
    if any(not r.prompt_complete or not r.response_complete for r in designers):
        errors.append("incomplete_designer_payload")

    # Verify the returned media too: a successful runtime manifest cannot prove
    # that the archive retained the images. Includes tool-observation images.
    media_errors: list[dict[str, str]] = []
    checked_media: set[tuple[str, str]] = set()
    for path in sorted(run_root.rglob("trajectories*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = TrajectoryRecord.model_validate_json(line)
            except ValueError:
                continue  # load_trajectories already reports malformed rows.
            references = [
                *row.image_refs,
                *(row.provenance.get("tool_media_refs") or []),
            ]
            for ref in references:
                if not isinstance(ref, dict):
                    continue
                value, expected = str(ref.get("path") or ""), str(
                    ref.get("sha256") or ""
                )
                media_path = Path(value)
                if not media_path.is_absolute():
                    media_path = path.parent / media_path
                key = (str(media_path.resolve()), expected)
                if key in checked_media:
                    continue
                checked_media.add(key)
                try:
                    digest = hashlib.sha256(media_path.read_bytes()).hexdigest()
                    reason = (
                        "missing_media_hash"
                        if not expected
                        else ("media_hash_mismatch" if digest != expected else "")
                    )
                except OSError:
                    reason = "missing_media"
                if reason:
                    media_errors.append({"reason": reason, "path": str(media_path)})
    if media_errors:
        errors.append("media_integrity_failed")

    metrics_path = run_root / "metrics" / "run_metrics.json"
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if not isinstance(metrics, dict) or not isinstance(
            metrics.get("summary"), dict
        ):
            raise ValueError("run metrics must contain a summary object")
    except (OSError, ValueError):
        metrics = {}
        errors.append("missing_or_invalid_run_metrics")
    summary = metrics.get("summary") or {}
    expected_scenes = summary.get("expected_scenes", 0)
    generation_complete = bool(
        isinstance(expected_scenes, int)
        and expected_scenes > 0
        and summary.get("completed_scenes") == expected_scenes
        and summary.get("missing_or_nonterminal_scenes") == 0
        and summary.get("failed_scenes") == 0
    )
    if not generation_complete:
        errors.append("generation_incomplete")
    captured_scenes = {(r.run_id, r.scene_id) for r in designers}
    if len(captured_scenes) != expected_scenes:
        errors.append("designer_scene_coverage_mismatch")

    traces: dict[tuple[str, ...], dict[str, Any]] = {}
    trace_payloads: dict[tuple[str, ...], str] = {}
    for path in sorted(run_root.rglob("trace_*.json")):
        parts = path.relative_to(run_root).parts
        if "shared_base" in parts or path.stem.endswith("_partial"):
            continue
        try:
            trace = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(trace, dict) or not isinstance(
                trace.get("final_report"), dict
            ):
                raise ValueError("final trace must contain a report object")
        except (OSError, ValueError):
            errors.append("invalid_final_trace")
            continue
        # Both the per-scene debug tree and experiment-level traces/ directory
        # contain this trace, and ACP may copy both through latest-run too.
        # Preserve the branch/batch scope so independent runs are not collapsed.
        boundary = next(
            (i for i, part in enumerate(parts) if part in {"hydra", "latest-run"}),
            0,
        )
        key = (*parts[:boundary], str(trace.get("scene_id")))
        payload = json.dumps(trace, sort_keys=True)
        if key in trace_payloads:
            if trace_payloads[key] != payload:
                errors.append("conflicting_final_trace")
            continue
        trace_payloads[key] = payload
        report = trace.get("final_report") or {}
        traces[key] = {
            "scene_id": trace.get("scene_id"),
            "status": trace.get("status"),
            "completed_stages": report.get("completed_stages", []),
            "missing_stages": report.get("missing_stages", []),
            "generation_status": report.get("generation_status"),
            "pass_scene": report.get("pass_scene"),
            "deterministic_pass": report.get("deterministic_pass"),
            "issues": [
                {"stage": stage.get("stage"), **issue}
                for stage in trace.get("stages") or []
                for issue in (stage.get("verify_report") or {}).get("issues") or []
            ],
        }
        if (
            report.get("generation_status") != "complete"
            or report.get("missing_stages")
            or set(report.get("completed_stages") or [])
            != {
                "floor_plan",
                "furniture",
                "wall_mounted",
                "ceiling_mounted",
                "manipuland",
            }
        ):
            errors.append("incomplete_final_trace")
    if len(traces) != expected_scenes:
        errors.append("final_trace_coverage_mismatch")
    if {t["scene_id"] for t in traces.values()} != {r.scene_id for r in designers}:
        errors.append("trace_designer_scene_mismatch")

    pair_count = manifest["stats"]["eligible_pair_count"]
    export_ready = bool(manifest["validation"]["valid"] and pair_count > 0)
    collection_ready = not errors
    if pair_count < min_pairs or (min_pairs > 0 and not export_ready):
        errors.append("dpo_pair_gate_failed")
    context_counts = Counter(
        (r.task_id, r.stage, r.task_type, r.context_hash) for r in designers
    )
    result = {
        "schema_version": "sceneexpert.collection_audit.v1",
        "run_id": metrics.get("run_id"),
        "source_code_revision": (metrics.get("code_provenance") or {}).get(
            "git_revision"
        ),
        "expected_model": expected_model,
        "model_counts": dict(Counter(r.model_id for r in records)),
        "generation_complete": generation_complete,
        "observer_collection_ready": collection_ready,
        "dpo_export_ready": export_ready,
        "training_preflight_status": "not_run",
        "min_pairs": min_pairs,
        "gate_passed": not errors,
        "errors": sorted(set(errors)),
        "scene_summary": {
            k: summary.get(k)
            for k in (
                "expected_scenes",
                "completed_scenes",
                "degraded_scenes",
                "failed_scenes",
                "missing_or_nonterminal_scenes",
                "mean_required_coverage",
                "hard_constraint_pass_rate",
                "memory_writer_persisted_records",
            )
        },
        "final_traces": list(traces.values()),
        "trajectory_count": len(records),
        "designer_count": len(designers),
        "designer_verdict_counts": dict(Counter(r.evidence.verdict for r in designers)),
        "designer_context_count": len(context_counts),
        "contexts_with_multiple_candidates": sum(
            n > 1 for n in context_counts.values()
        ),
        "checked_media_count": len(checked_media),
        "media_errors": media_errors,
        "dpo_stats": manifest["stats"],
        "dpo_validation": manifest["validation"],
        "next_step": (
            "review_pairs_and_run_training_preflight"
            if export_ready
            else "collect_independently_executed_candidates_from_identical_decision_contexts"
        ),
    }
    (output_dir / "collection_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result
