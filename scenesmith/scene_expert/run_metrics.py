"""Build independent, auditable metrics for one critic-probe output root.

The collector is intentionally read-only with respect to generation artifacts.
It accepts partial and failed runs, records data-quality gaps explicitly, and
writes only below ``<output_root>/metrics``.  This makes speed, quality, memory
delivery, writer health, and repair ownership comparable across repeated runs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import statistics

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from scenesmith.scene_expert.experiment_identity import stable_source_bundle_hash

console_logger = logging.getLogger(__name__)

SCHEMA_VERSION = "sceneexpert.run_metrics.v7"
SCENE_COLUMNS = (
    "run_id",
    "batch_id",
    "scene_index",
    "scene_id",
    "case_id",
    "prompt",
    "status",
    "error",
    "attempt",
    "failure_class",
    "failure_stage",
    "failure_error_type",
    "failure_recordable",
    "failure_reason",
    "failure_retryable",
    "failure_root_error_type",
    "failure_stage_execution_attempt",
    "trace_status",
    "trace_time_sec",
    "trace_degraded",
    "critic_report_count",
    "critic_effective_checks",
    "critic_pass",
    "critic_degraded",
    "critic_fail",
    "critic_unknown",
    "critic_score",
    "critic_zero_fail",
    "sceneexpert_overall_score",
    "sceneexpert_pass_scene",
    "generation_status",
    "requirement_status",
    "quality_status",
    "required_coverage",
    "hard_constraint_pass",
    "hard_violation_count",
    "relation_satisfaction",
    "experiment_name",
    "config_hash",
    "experiment_signature",
    "control_signature",
    "evaluation_pair_id",
    "evaluation_dimension",
    "evaluation_arm",
    "evaluation_require_frozen_memory",
    "shared_base_fingerprint",
    "source_bundle_hash",
    "model",
    "code_revision",
    "code_dirty",
    "task_compiler_source",
    "task_compiler_degraded",
    "global_planner_llm_stages",
    "global_planner_fallback_stages",
    "global_planner_noop_stages",
    "brief_injection_verified_stages",
    "stage_policies",
    "stage_agent_invoked_stages",
    "missing_stage_agent_invocation_stages",
    "optional_asset_recommendation_count",
    "required_first_instruction_stages",
    "required_first_missing_instruction_stages",
    "optional_autonomy_preserved_stages",
    "required_objects_by_stage",
    "required_satisfied_objects_by_stage",
    "required_missing_objects_by_stage",
    "requirement_status_by_stage",
    "observed_stages",
    "memory_retrieved_stages",
    "memory_retrieved_ids",
    "memory_injected_stages",
    "memory_injection_verified_stages",
    "memory_cross_task_verified_stages",
    "memory_retrieved_source_task_ids",
    "memory_retrieved_source_run_ids",
    "memory_bank_ids",
    "memory_bank_revisions",
    "memory_dirs",
    "memory_snapshot_fingerprint",
    "memory_snapshot_revision",
    "memory_snapshot_unchanged",
    "memory_read_only",
    "memory_zero_result_reasons",
    "memory_zero_result_events",
    "memory_retrieval_time_sec",
    "memory_writer_status",
    "memory_writer_candidate_count",
    "memory_writer_noop_reason",
    "memory_writer_persisted",
    "memory_writer_promoted",
    "memory_writer_added",
    "memory_writer_merged",
    "memory_writer_fallback_written",
    "memory_writer_degraded",
    "llm_skill_candidate_count",
    "bootstrap_skill_eligible_stage_count",
    "bootstrap_skill_candidate_count",
    "bootstrap_skill_persisted_candidate_count",
    "bootstrap_skill_rejected_count",
    "skill_persisted_candidate_count",
    "skill_promoted_active_count",
    "skill_rejected_count",
    "skill_rejection_reasons",
    "skill_store_candidate_added",
    "skill_store_candidate_merged",
    "scenesmith_repair_events",
    "scenesmith_repairs_accepted",
    "scenesmith_repairs_rejected",
    "scenesmith_repairs_resolved",
    "scenesmith_repair_strategies",
    "scenesmith_repair_affected_objects",
    "scenesmith_repair_timing_events",
    "scenesmith_repair_overhead_sec",
    "scene_rescued_by_scenesmith_repair",
    "sceneexpert_repair_plans",
    "sceneexpert_repairs_executed",
    "component_flags",
)


def _read_json(path: Path, warnings: list[str]) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        warnings.append(f"unreadable_json:{path}:{type(exc).__name__}")
        return {}
    if not isinstance(payload, dict):
        warnings.append(f"non_object_json:{path}")
        return {}
    return payload


def _read_jsonl(path: Path, warnings: list[str]) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        warnings.append(f"unreadable_jsonl:{path}:{type(exc).__name__}")
        return records
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            warnings.append(f"malformed_jsonl:{path}:{line_number}")
            continue
        if isinstance(value, dict):
            records.append(value)
        else:
            warnings.append(f"non_object_jsonl:{path}:{line_number}")
    return records


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _as_float(value: Any) -> float | None:
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


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _unique(values: Iterable[str]) -> list[str]:
    return sorted({str(value) for value in values if str(value)})


def _manifest_rows(output_root: Path, warnings: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    critic_root = output_root / "critic_on"
    for batch_dir in sorted(critic_root.glob("batch_*")):
        if not batch_dir.is_dir():
            continue
        manifest = batch_dir / "batch_cases.csv"
        if not manifest.is_file():
            warnings.append(f"missing_batch_manifest:{manifest}")
            continue
        try:
            with manifest.open("r", encoding="utf-8", newline="") as stream:
                for raw in csv.DictReader(stream):
                    scene_index = _as_int(raw.get("scene_index"))
                    rows.append(
                        {
                            "batch_dir": batch_dir,
                            "batch_id": batch_dir.name,
                            "scene_index": scene_index,
                            "scene_id": f"scene_{scene_index:03d}",
                            "case_id": str(raw.get("case_id") or ""),
                            "prompt": str(raw.get("prompt") or ""),
                        }
                    )
        except (OSError, UnicodeError, csv.Error) as exc:
            warnings.append(
                f"unreadable_batch_manifest:{manifest}:{type(exc).__name__}"
            )
    return rows


def _discover_rows(output_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for batch_dir in sorted((output_root / "critic_on").glob("batch_*")):
        if not batch_dir.is_dir():
            continue
        hydra_dir = batch_dir / "hydra"
        scene_root = hydra_dir if hydra_dir.is_dir() else batch_dir
        for scene_dir in sorted(scene_root.glob("scene_*")):
            try:
                scene_index = int(scene_dir.name.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
            rows.append(
                {
                    "batch_dir": batch_dir,
                    "batch_id": batch_dir.name,
                    "scene_index": scene_index,
                    "scene_id": scene_dir.name,
                    "case_id": "",
                    "prompt": "",
                }
            )
    return rows


def _scene_dir(manifest_row: dict[str, Any]) -> Path:
    batch_dir = Path(manifest_row["batch_dir"])
    hydra_dir = batch_dir / "hydra"
    root = hydra_dir if hydra_dir.is_dir() else batch_dir
    return root / str(manifest_row["scene_id"])


def _load_trace(
    scene_dir: Path,
    scene_index: int,
    warnings: list[str],
) -> tuple[dict[str, Any], str]:
    hydra_dir = scene_dir.parent
    final_candidates = (
        hydra_dir / "traces" / f"trace_{scene_index:06d}.json",
        scene_dir / "scene_expert" / "trace" / f"trace_{scene_index:06d}.json",
    )
    for path in final_candidates:
        if path.is_file():
            return _read_json(path, warnings), str(path)
    partial = (
        scene_dir / "scene_expert" / "trace" / f"trace_{scene_index:06d}_partial.json"
    )
    if partial.is_file():
        return _read_json(partial, warnings), str(partial)
    return {}, ""


def _stage_payloads(
    scene_dir: Path,
    trace: dict[str, Any],
    warnings: list[str],
) -> list[dict[str, Any]]:
    stages = [item for item in trace.get("stages") or [] if isinstance(item, dict)]
    known = {str(item.get("stage") or "") for item in stages}
    stage_dir = scene_dir / "scene_expert" / "stages"
    for path in sorted(stage_dir.glob("*_pre.json")):
        payload = _read_json(path, warnings)
        stage = str(payload.get("stage") or "")
        if stage and stage not in known:
            stages.append(payload)
            known.add(stage)
    return stages


def _memory_metrics(
    scene_dir: Path,
    stages: list[dict[str, Any]],
    prompt: str,
    warnings: list[str],
) -> dict[str, Any]:
    retrieved_stages: list[str] = []
    retrieved_ids: list[str] = []
    injected_stages: list[str] = []
    verified_stages: list[str] = []
    cross_task_stages: list[str] = []
    source_task_ids: dict[str, list[str]] = {}
    source_run_ids: dict[str, list[str]] = {}
    bank_ids: list[str] = []
    bank_revisions: list[int] = []
    current_task_id = "task_" + hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
    for stage_payload in stages:
        stage = str(stage_payload.get("stage") or "")
        pack = stage_payload.get("memory_pack") or {}
        ids = [
            *list(pack.get("success_case_ids") or []),
            *list(pack.get("failure_case_ids") or []),
            *list(pack.get("skill_names") or []),
        ]
        evidence = stage_payload.get("execution_evidence") or {}
        task_provenance = pack.get("retrieved_source_task_ids") or {}
        run_provenance = pack.get("retrieved_source_run_ids") or {}
        if isinstance(task_provenance, dict):
            for memory_id, values in task_provenance.items():
                key = str(memory_id)
                source_task_ids[key] = _unique(
                    [*source_task_ids.get(key, []), *(values or [])]
                )
        if isinstance(run_provenance, dict):
            for memory_id, values in run_provenance.items():
                key = str(memory_id)
                source_run_ids[key] = _unique(
                    [*source_run_ids.get(key, []), *(values or [])]
                )
        if any(
            source_id != current_task_id
            for memory_id in ids
            for source_id in source_task_ids.get(str(memory_id), [])
        ):
            cross_task_stages.append(stage)
        bank_id = str(pack.get("memory_bank_id") or "")
        if bank_id:
            bank_ids.append(bank_id)
        revision = pack.get("memory_bank_revision")
        if isinstance(revision, int):
            bank_revisions.append(revision)
        if ids:
            retrieved_stages.append(stage)
            retrieved_ids.extend(str(value) for value in ids)
        memory_injected = bool(
            evidence.get("injected_memory_hash")
            or evidence.get("placement_reference_injected")
            or (
                ids
                and evidence.get("final_injection_hash")
                and evidence.get("designer_prompt_contains_memory")
            )
        )
        if memory_injected:
            injected_stages.append(stage)
        if bool(evidence.get("designer_prompt_contains_memory")) or bool(
            evidence.get("placement_reference_injected")
        ):
            verified_stages.append(stage)

    timing_path = scene_dir / "scene_expert" / "timing" / "memory_retrieval.jsonl"
    timing_rows = _read_jsonl(timing_path, warnings)
    detailed_rows = [
        row for row in timing_rows if str(row.get("retriever_type") or "") == "hybrid"
    ]
    if not detailed_rows:
        detailed_rows = timing_rows
    memory_dirs = _unique(str(row.get("memory_dir") or "") for row in detailed_rows)
    bank_ids.extend(str(row.get("memory_bank_id") or "") for row in detailed_rows)
    bank_revisions.extend(
        int(row["memory_bank_revision"])
        for row in detailed_rows
        if isinstance(row.get("memory_bank_revision"), int)
    )
    zero_reasons = _unique(
        str(row.get("zero_result_reason") or "") for row in detailed_rows
    )
    retrieval_time = sum(
        _as_float(row.get("total_sec")) or _as_float(row.get("elapsed_sec")) or 0.0
        for row in detailed_rows
    )
    zero_events = [
        {
            "stage": str(row.get("stage") or ""),
            "reason": str(row.get("zero_result_reason") or ""),
            "active_stage_records": dict(row.get("active_stage_records") or {}),
            "banks": list(row.get("banks") or []),
        }
        for row in detailed_rows
        if row.get("zero_result_reason")
    ]
    return {
        "memory_retrieved_stages": _unique(retrieved_stages),
        "memory_retrieved_ids": _unique(retrieved_ids),
        "memory_injected_stages": _unique(injected_stages),
        "memory_injection_verified_stages": _unique(verified_stages),
        "memory_cross_task_verified_stages": _unique(cross_task_stages),
        "memory_retrieved_source_task_ids": source_task_ids,
        "memory_retrieved_source_run_ids": source_run_ids,
        "memory_bank_ids": _unique(bank_ids),
        "memory_bank_revisions": sorted(set(bank_revisions)),
        "memory_dirs": memory_dirs,
        "memory_zero_result_reasons": zero_reasons,
        "memory_zero_result_events": zero_events,
        "memory_retrieval_time_sec": round(retrieval_time, 6),
    }


def _component_execution_metrics(
    trace: dict[str, Any], stages: list[dict[str, Any]]
) -> dict[str, Any]:
    """Separate configured flags from observed model/fallback execution."""
    component_status = trace.get("component_status") or {}
    task_status = component_status.get("task_compiler") or {}
    planner_by_status: dict[str, list[str]] = {"ok": [], "fallback": [], "no_op": []}
    brief_verified: list[str] = []
    stage_agent_invoked: list[str] = []
    missing_stage_agent_invocation: list[str] = []
    stage_policies: dict[str, str] = {}
    optional_recommendation_count = 0
    required_first_instruction_stages: list[str] = []
    required_first_missing_instruction_stages: list[str] = []
    optional_autonomy_preserved_stages: list[str] = []
    required_objects_by_stage: dict[str, list[str]] = {}
    required_satisfied_by_stage: dict[str, list[str]] = {}
    required_missing_by_stage: dict[str, list[str]] = {}
    requirement_status_by_stage: dict[str, str] = {}
    required_coverage_by_stage: dict[str, float] = {}
    for stage_payload in stages:
        stage = str(stage_payload.get("stage") or "")
        planner_status = str(
            (stage_payload.get("planner_trace") or {}).get("status") or ""
        )
        if planner_status in planner_by_status:
            planner_by_status[planner_status].append(stage)
        evidence = stage_payload.get("execution_evidence") or {}
        stage_policy = str(evidence.get("stage_policy") or "unknown")
        if stage_policy != "unknown" or stage not in stage_policies:
            stage_policies[stage] = stage_policy
        optional_recommendation_count += len(
            evidence.get("optional_asset_recommendations") or []
        )
        required_objects = [
            str(item) for item in evidence.get("required_objects") or [] if str(item)
        ]
        required_satisfied = [
            str(item)
            for item in evidence.get("required_satisfied_objects") or []
            if str(item)
        ]
        required_missing = [
            str(item)
            for item in evidence.get("required_missing_objects") or []
            if str(item)
        ]
        if required_objects or stage not in required_objects_by_stage:
            required_objects_by_stage[stage] = required_objects
        if required_satisfied or stage not in required_satisfied_by_stage:
            required_satisfied_by_stage[stage] = required_satisfied
        if required_missing or stage not in required_missing_by_stage:
            required_missing_by_stage[stage] = required_missing
        requirement_status = str(evidence.get("requirement_status") or "unknown")
        if requirement_status != "unknown" or stage not in requirement_status_by_stage:
            requirement_status_by_stage[stage] = requirement_status
        required_coverage = _as_float(evidence.get("required_coverage"))
        if required_coverage is not None and required_objects:
            required_coverage_by_stage[stage] = required_coverage
        if bool(evidence.get("required_first_instruction_delivered")):
            required_first_instruction_stages.append(stage)
        elif bool(evidence.get("required_first_instruction_applicable")):
            required_first_missing_instruction_stages.append(stage)
        if bool(evidence.get("optional_autonomy_preserved")):
            optional_autonomy_preserved_stages.append(stage)
        if bool(evidence.get("stage_agent_invoked")):
            stage_agent_invoked.append(stage)
        else:
            missing_stage_agent_invocation.append(stage)
        if bool(evidence.get("designer_prompt_contains_brief")):
            brief_verified.append(stage)
    return {
        "task_compiler_source": str(task_status.get("source") or "not_observed"),
        "task_compiler_degraded": bool(task_status.get("degraded", False)),
        "global_planner_llm_stages": _unique(planner_by_status["ok"]),
        "global_planner_fallback_stages": _unique(planner_by_status["fallback"]),
        "global_planner_noop_stages": _unique(planner_by_status["no_op"]),
        "brief_injection_verified_stages": _unique(brief_verified),
        "stage_policies": stage_policies,
        "stage_agent_invoked_stages": _unique(stage_agent_invoked),
        "missing_stage_agent_invocation_stages": sorted(
            set(missing_stage_agent_invocation) - set(stage_agent_invoked)
        ),
        "optional_asset_recommendation_count": optional_recommendation_count,
        "required_first_instruction_stages": _unique(required_first_instruction_stages),
        "required_first_missing_instruction_stages": sorted(
            set(required_first_missing_instruction_stages)
            - set(required_first_instruction_stages)
        ),
        "optional_autonomy_preserved_stages": _unique(
            optional_autonomy_preserved_stages
        ),
        "required_objects_by_stage": required_objects_by_stage,
        "required_satisfied_objects_by_stage": required_satisfied_by_stage,
        "required_missing_objects_by_stage": required_missing_by_stage,
        "requirement_status_by_stage": requirement_status_by_stage,
        "required_coverage": (
            round(
                sum(
                    len(required_objects_by_stage.get(stage) or []) * value
                    for stage, value in required_coverage_by_stage.items()
                )
                / sum(
                    len(required_objects_by_stage.get(stage) or [])
                    for stage in required_coverage_by_stage
                ),
                6,
            )
            if required_coverage_by_stage
            else None
        ),
    }


def _critic_metrics(scene_dir: Path, warnings: list[str]) -> dict[str, Any]:
    report_paths = sorted(
        scene_dir.glob(
            "room_*/scenebenchmark_critic/final_scene/scenebenchmark_critic.json"
        )
    )
    if not report_paths:
        report_paths = sorted(
            scene_dir.glob(
                "**/scenebenchmark_critic/final_scene/scenebenchmark_critic.json"
            )
        )
    reports = [_read_json(path, warnings) for path in report_paths]
    summaries = [
        (report.get("summary") or {}).get("scene_summary") or {} for report in reports
    ]
    totals = {
        key: sum(_as_int(summary.get(key)) for summary in summaries)
        for key in ("effective_checks", "pass", "degraded", "fail", "unknown")
    }
    score_sum = 0.0
    score_weight = 0
    for summary in summaries:
        score = _as_float(summary.get("score"))
        weight = _as_int(summary.get("effective_checks"))
        if score is not None and weight > 0:
            score_sum += score * weight
            score_weight += weight
    score = round(score_sum / score_weight, 6) if score_weight else None
    return {
        "critic_report_count": len(reports),
        "critic_effective_checks": totals["effective_checks"],
        "critic_pass": totals["pass"],
        "critic_degraded": totals["degraded"],
        "critic_fail": totals["fail"],
        "critic_unknown": totals["unknown"],
        "critic_score": score,
        "critic_zero_fail": (
            totals["effective_checks"] > 0
            and totals["fail"] == 0
            and totals["unknown"] == 0
        ),
    }


def _writer_metrics(
    scene_dir: Path,
    trace: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    status = dict((trace.get("component_status") or {}).get("memory_writer") or {})
    debug_path = scene_dir / "scene_expert" / "memory" / "memory_writer_debug.json"
    if debug_path.is_file():
        debug = _read_json(debug_path, warnings)
        result_status = debug.get("result_status") or {}
        status = {**debug, **result_status, **status}
    store_apply = status.get("store_apply") or {}
    return {
        "memory_writer_status": str(
            status.get("write_status") or status.get("status") or "not_observed"
        ),
        "memory_writer_candidate_count": _as_int(status.get("candidate_count")),
        "memory_writer_noop_reason": str(status.get("noop_reason") or ""),
        "memory_writer_persisted": _as_int(
            status.get("persisted_count", status.get("promoted_count"))
        ),
        "memory_writer_promoted": _as_int(status.get("promoted_count")),
        "memory_writer_added": _as_int(store_apply.get("added")),
        "memory_writer_merged": _as_int(store_apply.get("merged")),
        "memory_writer_fallback_written": bool(status.get("fallback_written", False)),
        "memory_writer_degraded": bool(status.get("degraded", False)),
        "llm_skill_candidate_count": _as_int(status.get("llm_skill_candidate_count")),
        "bootstrap_skill_eligible_stage_count": _as_int(
            status.get("bootstrap_skill_eligible_stage_count")
        ),
        "bootstrap_skill_candidate_count": _as_int(
            status.get("bootstrap_skill_candidate_count")
        ),
        "bootstrap_skill_persisted_candidate_count": _as_int(
            status.get("bootstrap_skill_persisted_candidate_count")
        ),
        "bootstrap_skill_rejected_count": _as_int(
            status.get("bootstrap_skill_rejected_count")
        ),
        "skill_persisted_candidate_count": _as_int(
            status.get("skill_persisted_candidate_count")
        ),
        "skill_promoted_active_count": _as_int(
            store_apply.get(
                "skill_promoted_active", status.get("skill_promoted_active_count")
            )
        ),
        "skill_rejected_count": _as_int(status.get("skill_rejected_count")),
        "skill_rejection_reasons": dict(status.get("skill_rejection_reasons") or {}),
        "skill_store_candidate_added": _as_int(
            store_apply.get("skill_candidate_added")
        ),
        "skill_store_candidate_merged": _as_int(
            store_apply.get("skill_candidate_merged")
        ),
    }


def _repair_metrics(
    scene_dir: Path,
    stages: list[dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    raw_sceneexpert_actions = [
        action
        for stage in stages
        for action in stage.get("repair_actions") or []
        if isinstance(action, dict)
    ]
    actions_by_key: dict[tuple[str, ...], dict[str, Any]] = {}
    for action in raw_sceneexpert_actions:
        key = tuple(
            str(action.get(field) or "")
            for field in (
                "repair_owner",
                "repair_type",
                "failure_type",
                "repair_action",
            )
        )
        current = actions_by_key.get(key)
        if current is None or str(action.get("execution_status")) == "executed":
            actions_by_key[key] = action
    sceneexpert_actions = list(actions_by_key.values())
    repair_events = _read_jsonl(
        scene_dir / "scene_expert" / "timing" / "repair_events.jsonl",
        warnings,
    )
    core_events = [
        event
        for event in repair_events
        if str(event.get("repair_owner") or "scenesmith_core") == "scenesmith_core"
    ]
    executed = [
        action
        for action in sceneexpert_actions
        if str(action.get("execution_status") or "planned") == "executed"
    ]
    accepted = [event for event in core_events if event.get("status") == "accepted"]
    rejected = [event for event in core_events if event.get("status") == "rejected"]
    resolved = [
        event
        for event in core_events
        if bool((event.get("detail") or {}).get("resolved", False))
    ]
    timing_events = _read_jsonl(
        scene_dir / "scene_expert" / "timing" / "stage_working_timing.jsonl",
        warnings,
    )
    repair_timing_events = [
        event
        for event in timing_events
        if str(event.get("module") or "") == "deterministic_repair"
    ]
    timing_overhead = sum(
        _as_float(event.get("elapsed_sec")) or 0.0 for event in repair_timing_events
    )
    return {
        "scenesmith_repair_events": len(core_events),
        "scenesmith_repairs_accepted": len(accepted),
        "scenesmith_repairs_rejected": len(rejected),
        "scenesmith_repairs_resolved": len(resolved),
        "scenesmith_repair_strategies": _unique(
            str(event.get("strategy") or "") for event in core_events
        ),
        "scenesmith_repair_affected_objects": sum(
            len(event.get("affected_objects") or []) for event in core_events
        ),
        "scenesmith_repair_timing_events": len(repair_timing_events),
        "scenesmith_repair_overhead_sec": round(
            timing_overhead
            or sum(
                _as_float((event.get("detail") or {}).get("elapsed_sec")) or 0.0
                for event in core_events
            ),
            6,
        ),
        "scene_rescued_by_scenesmith_repair": bool(resolved),
        "sceneexpert_repair_plans": len(sceneexpert_actions),
        "sceneexpert_repairs_executed": len(executed),
    }


def _verification_metrics(stages: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize final per-stage hard and relation evidence without re-scoring."""

    final_reports: dict[str, dict[str, Any]] = {}
    for stage in stages:
        report = stage.get("verify_report")
        if isinstance(report, dict):
            final_reports[str(stage.get("stage") or "")] = report
    hard_passes: list[bool] = []
    violation_count = 0
    relation_scores: list[float] = []
    for report in final_reports.values():
        issues = [
            issue for issue in report.get("issues") or [] if isinstance(issue, dict)
        ]
        violation_count += len(issues)
        hard_report = (
            report.get("hard_check_report")
            if isinstance(report.get("hard_check_report"), dict)
            else {}
        )
        hard_valid = hard_report.get("hard_valid", hard_report.get("hard_passed"))
        if not isinstance(hard_valid, bool):
            hard_valid = bool(report.get("pass_stage")) and not issues
        hard_passes.append(hard_valid)
        relation_value = hard_report.get(
            "relation_satisfaction",
            hard_report.get("constraint_satisfaction_rate"),
        )
        if isinstance(relation_value, (int, float)):
            relation_scores.append(max(0.0, min(1.0, float(relation_value))))
    return {
        "hard_constraint_pass": all(hard_passes) if hard_passes else None,
        "hard_violation_count": violation_count if final_reports else None,
        "relation_satisfaction": (
            round(statistics.fmean(relation_scores), 6) if relation_scores else None
        ),
    }


def _scene_metrics(
    output_root: Path,
    run_id: str,
    manifest_row: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    scene_dir = _scene_dir(manifest_row)
    status_path = scene_dir / "scene_status.json"
    status_payload = _read_json(status_path, warnings) if status_path.is_file() else {}
    trace, _trace_path = _load_trace(
        scene_dir, int(manifest_row["scene_index"]), warnings
    )
    stages = _stage_payloads(scene_dir, trace, warnings)
    final_report = trace.get("final_report") or {}
    memory_identity = dict(trace.get("memory_identity") or {})
    evaluation_contract = dict(trace.get("evaluation_contract") or {})
    shared_base_identity = dict(evaluation_contract.get("shared_base_identity") or {})
    memory_snapshot = dict(
        (trace.get("component_status") or {}).get("memory_snapshot") or {}
    )
    code_provenance = dict(trace.get("code_provenance") or {})
    source_bundle_hash = str(code_provenance.get("source_bundle_hash") or "")
    if not source_bundle_hash:
        source_hashes = code_provenance.get("source_hashes") or {}
        if isinstance(source_hashes, dict):
            source_bundle_hash = stable_source_bundle_hash(source_hashes)
    if source_bundle_hash:
        code_provenance["source_bundle_hash"] = source_bundle_hash
    status = str(status_payload.get("status") or "missing")
    failure = status_payload.get("failure")
    failure = failure if isinstance(failure, dict) else {}
    failure_class = str(failure.get("failure_class") or "")
    if status == "failed" and not failure_class:
        failure_class = "unclassified_legacy"
    generation_status = str(final_report.get("generation_status") or "")
    if not generation_status:
        if status in {"completed", "completed_with_quality_issues"}:
            generation_status = "complete"
        elif status == "failed":
            generation_status = "failed"
        else:
            generation_status = "unknown"
    row: dict[str, Any] = {
        "run_id": run_id,
        "batch_id": manifest_row["batch_id"],
        "scene_index": int(manifest_row["scene_index"]),
        "scene_id": manifest_row["scene_id"],
        "case_id": manifest_row.get("case_id", ""),
        "prompt": str(status_payload.get("prompt") or manifest_row.get("prompt") or ""),
        "status": status,
        "error": str(status_payload.get("error") or trace.get("error") or ""),
        "attempt": _as_int(status_payload.get("attempt")),
        "failure_class": failure_class,
        "failure_stage": str(failure.get("stage") or ""),
        "failure_error_type": str(failure.get("error_type") or ""),
        "failure_recordable": failure.get("recordable") is True,
        "failure_reason": str(failure.get("reason") or ""),
        "failure_retryable": failure.get("retryable") is True,
        "failure_root_error_type": str(failure.get("root_error_type") or ""),
        "failure_stage_execution_attempt": _as_int(
            failure.get("stage_execution_attempt")
        ),
        "trace_status": str(trace.get("status") or "missing"),
        "trace_time_sec": _as_float(trace.get("total_time_sec")),
        "trace_degraded": bool(trace.get("degraded", False)),
        "sceneexpert_overall_score": _as_float(final_report.get("overall_score")),
        "sceneexpert_pass_scene": (
            bool(final_report.get("pass_scene")) if final_report else None
        ),
        "generation_status": generation_status,
        "requirement_status": str(final_report.get("requirement_status") or "unknown"),
        "quality_status": str(final_report.get("quality_status") or "unknown"),
        "experiment_name": str(trace.get("experiment_name") or ""),
        "config_hash": str(trace.get("config_hash") or ""),
        "experiment_signature": str(trace.get("experiment_signature") or ""),
        "control_signature": str(trace.get("control_signature") or ""),
        "evaluation_pair_id": str(evaluation_contract.get("pair_id") or ""),
        "evaluation_dimension": str(
            evaluation_contract.get("controlled_dimension") or ""
        ),
        "evaluation_arm": str(evaluation_contract.get("arm") or ""),
        "evaluation_require_frozen_memory": evaluation_contract.get(
            "require_frozen_memory"
        ),
        "shared_base_fingerprint": str(shared_base_identity.get("fingerprint") or ""),
        "source_bundle_hash": source_bundle_hash,
        "model": str(trace.get("model") or ""),
        "code_revision": str(
            (trace.get("code_provenance") or {}).get("git_revision") or ""
        ),
        "code_dirty": (trace.get("code_provenance") or {}).get("dirty"),
        "code_provenance": code_provenance,
        "observed_stages": _unique(str(stage.get("stage") or "") for stage in stages),
        "component_flags": dict(trace.get("component_flags") or {}),
        "memory_snapshot_fingerprint": str(
            memory_identity.get("content_fingerprint") or ""
        ),
        "memory_snapshot_revision": _as_int(memory_identity.get("revision")),
        "memory_snapshot_unchanged": memory_snapshot.get("unchanged"),
        "memory_read_only": memory_identity.get("read_only"),
    }
    row.update(_component_execution_metrics(trace, stages))
    row.update(_critic_metrics(scene_dir, warnings))
    row.update(_memory_metrics(scene_dir, stages, row["prompt"], warnings))
    if memory_identity.get("bank_id"):
        row["memory_bank_ids"] = _unique(
            [*row["memory_bank_ids"], str(memory_identity["bank_id"])]
        )
    if memory_identity.get("revision") is not None:
        row["memory_bank_revisions"] = sorted(
            {
                *row["memory_bank_revisions"],
                _as_int(memory_identity.get("revision")),
            }
        )
    if memory_identity.get("memory_dir"):
        row["memory_dirs"] = _unique(
            [*row["memory_dirs"], str(memory_identity["memory_dir"])]
        )
    row.update(_writer_metrics(scene_dir, trace, warnings))
    row.update(_repair_metrics(scene_dir, stages, warnings))
    row.update(_verification_metrics(stages))
    return row


def collect_run_metrics(
    output_root: str | Path,
    *,
    run_id: str = "",
    process_exit_code: int | None = None,
) -> dict[str, Any]:
    """Collect one run without mutating any generation artifact."""
    root = Path(output_root).resolve()
    resolved_run_id = run_id or root.name
    warnings: list[str] = []
    manifest_rows = _manifest_rows(root, warnings)
    if not manifest_rows:
        warnings.append("no_critic_on_manifests:falling_back_to_scene_discovery")
        manifest_rows = _discover_rows(root)
    scene_rows = [
        _scene_metrics(root, resolved_run_id, row, warnings) for row in manifest_rows
    ]

    completed_statuses = {"completed", "completed_with_quality_issues"}
    completed = [row for row in scene_rows if row["status"] in completed_statuses]
    failed = [row for row in scene_rows if row["status"] == "failed"]
    missing = [
        row
        for row in scene_rows
        if row["status"] not in completed_statuses | {"failed"}
    ]
    degraded = [
        row
        for row in completed
        if row["status"] == "completed_with_quality_issues" or row["trace_degraded"]
    ]
    critic_observed = [row for row in completed if row["critic_effective_checks"] > 0]
    trace_observed = [row for row in scene_rows if row["trace_status"] != "missing"]
    retrieved = [row for row in scene_rows if row["memory_retrieved_stages"]]
    injected = [row for row in scene_rows if row["memory_injection_verified_stages"]]
    cross_task_verified = [
        row for row in scene_rows if row["memory_cross_task_verified_stages"]
    ]
    retrieved_stage_count = sum(
        len(row["memory_retrieved_stages"]) for row in scene_rows
    )
    injected_stage_count = sum(
        len(row["memory_injection_verified_stages"]) for row in scene_rows
    )
    writer_observed = [
        row for row in scene_rows if row["memory_writer_status"] != "not_observed"
    ]
    writer_failures = [
        row
        for row in writer_observed
        if row["memory_writer_status"]
        in {"model_failure_no_write", "exception_no_write"}
        or row["memory_writer_degraded"]
    ]
    writer_noops = [
        row
        for row in writer_observed
        if row["memory_writer_status"] == "no_valid_candidates"
    ]
    fallback_writes = [
        row for row in writer_observed if row["memory_writer_fallback_written"]
    ]

    expected = len(scene_rows)
    if not expected:
        warnings.append("no_scenes_discovered")
    if process_exit_code not in {None, 0}:
        warnings.append(f"run_process_failed:{process_exit_code}")
    if missing:
        warnings.append(f"nonterminal_or_missing_scenes:{len(missing)}")
    if completed and len(critic_observed) < len(completed):
        warnings.append(
            "completed_scenes_missing_final_critic:"
            f"{len(completed) - len(critic_observed)}"
        )
    if scene_rows and len(trace_observed) < len(scene_rows):
        warnings.append(f"scenes_missing_trace:{len(scene_rows) - len(trace_observed)}")
    if fallback_writes:
        warnings.append(f"fallback_memory_writes_detected:{len(fallback_writes)}")
    if retrieved and not cross_task_verified:
        warnings.append("retrieved_memory_missing_cross_task_provenance")
    missing_stage_invocations = sum(
        len(row["missing_stage_agent_invocation_stages"]) for row in scene_rows
    )
    if missing_stage_invocations:
        warnings.append(
            f"missing_native_stage_agent_invocations:{missing_stage_invocations}"
        )

    revisions = _unique(row["code_revision"] for row in scene_rows)
    config_hashes = _unique(row["config_hash"] for row in scene_rows)
    experiment_signatures = _unique(row["experiment_signature"] for row in scene_rows)
    control_signatures = _unique(row["control_signature"] for row in scene_rows)
    evaluation_pair_ids = _unique(row["evaluation_pair_id"] for row in scene_rows)
    evaluation_dimensions = _unique(row["evaluation_dimension"] for row in scene_rows)
    evaluation_arms = _unique(row["evaluation_arm"] for row in scene_rows)
    shared_base_fingerprints = _unique(
        row["shared_base_fingerprint"] for row in scene_rows
    )
    source_bundle_hashes = _unique(row["source_bundle_hash"] for row in scene_rows)
    experiment_names = _unique(row["experiment_name"] for row in scene_rows)
    models = _unique(row["model"] for row in scene_rows)
    provenance_by_signature = {
        json.dumps(row["code_provenance"], sort_keys=True, default=str): row[
            "code_provenance"
        ]
        for row in scene_rows
        if row["code_provenance"]
    }
    if len(revisions) > 1:
        warnings.append(f"mixed_code_revisions:{len(revisions)}")
    if len(source_bundle_hashes) > 1:
        warnings.append(f"mixed_source_bundle_hashes:{len(source_bundle_hashes)}")
    elif len(provenance_by_signature) > 1:
        # Git/path metadata may legitimately vary across workers. Source bytes,
        # not worktree layout, own controlled code identity.
        console_logger.info(
            "Run contains %d provenance envelopes with one source bundle",
            len(provenance_by_signature),
        )
    if len(experiment_signatures) > 1:
        warnings.append(f"mixed_experiment_signatures:{len(experiment_signatures)}")
    if len(control_signatures) > 1:
        warnings.append(f"mixed_control_signatures:{len(control_signatures)}")
    if not config_hashes:
        warnings.append("missing_config_hash")
    if not experiment_signatures:
        warnings.append("missing_experiment_signature")
    if not control_signatures:
        warnings.append("missing_control_signature")
    if not source_bundle_hashes:
        warnings.append("missing_source_bundle_hash")
    if not experiment_names:
        warnings.append("missing_experiment_name")

    memory_bank_ids = _unique(
        bank_id for row in scene_rows for bank_id in row["memory_bank_ids"]
    )
    memory_dirs = _unique(
        memory_dir for row in scene_rows for memory_dir in row["memory_dirs"]
    )
    memory_bank_revisions = sorted(
        {
            int(revision)
            for row in scene_rows
            for revision in row["memory_bank_revisions"]
        }
    )
    memory_snapshot_fingerprints = _unique(
        row["memory_snapshot_fingerprint"] for row in scene_rows
    )
    memory_snapshot_revisions = sorted(
        {
            int(row["memory_snapshot_revision"])
            for row in scene_rows
            if row["memory_snapshot_fingerprint"]
        }
    )
    evaluation_active = bool(
        evaluation_pair_ids or evaluation_dimensions or evaluation_arms
    )
    memory_snapshot_all_unchanged = bool(
        scene_rows
        and all(row["memory_snapshot_unchanged"] is True for row in scene_rows)
    )
    memory_snapshot_identity_stable = bool(
        len(memory_snapshot_fingerprints) == 1 and len(memory_snapshot_revisions) == 1
    )
    shared_base_all_present = bool(
        scene_rows and all(row["shared_base_fingerprint"] for row in scene_rows)
    )
    memory_read_only_all = bool(
        scene_rows and all(row["memory_read_only"] is True for row in scene_rows)
    )
    frozen_required_all = bool(
        scene_rows
        and all(row["evaluation_require_frozen_memory"] is True for row in scene_rows)
    )
    if evaluation_active:
        if not memory_snapshot_fingerprints:
            warnings.append("evaluation_missing_memory_snapshot_fingerprint")
        if not memory_snapshot_all_unchanged:
            warnings.append("evaluation_memory_snapshot_changed_or_unverified")
        if not memory_snapshot_identity_stable:
            warnings.append("evaluation_memory_snapshot_mixed_across_scenes")
        if len(evaluation_pair_ids) != 1:
            warnings.append("evaluation_pair_id_missing_or_mixed")
        if len(evaluation_dimensions) != 1:
            warnings.append("evaluation_dimension_missing_or_mixed")
        if len(evaluation_arms) != 1:
            warnings.append("evaluation_arm_missing_or_mixed")
        if not shared_base_all_present:
            warnings.append("evaluation_shared_base_fingerprint_missing")
        if not memory_read_only_all:
            warnings.append("evaluation_memory_bank_not_read_only")
        if not frozen_required_all:
            warnings.append("evaluation_frozen_contract_not_required")
    requirement_observed = [
        row
        for row in scene_rows
        if row["requirement_status"] not in {"", "unknown", "not_applicable"}
    ]
    quality_observed = [
        row for row in scene_rows if row["quality_status"] not in {"", "unknown"}
    ]
    required_first_applicable_count = sum(
        len(row["required_first_instruction_stages"])
        + len(row["required_first_missing_instruction_stages"])
        for row in scene_rows
    )
    required_first_delivered_count = sum(
        len(row["required_first_instruction_stages"]) for row in scene_rows
    )
    auto_stage_count = sum(
        sum(policy == "auto" for policy in row["stage_policies"].values())
        for row in scene_rows
    )
    optional_autonomy_preserved_count = sum(
        len(row["optional_autonomy_preserved_stages"]) for row in scene_rows
    )
    skill_rejection_reasons: dict[str, int] = {}
    for row in scene_rows:
        for reason, count in row["skill_rejection_reasons"].items():
            reason_text = str(reason)
            skill_rejection_reasons[reason_text] = int(
                skill_rejection_reasons.get(reason_text, 0)
            ) + _as_int(count)

    summary = {
        "expected_scenes": expected,
        "completed_scenes": len(completed),
        "degraded_scenes": len(degraded),
        "failed_scenes": len(failed),
        "recorded_failed_scenes": sum(row["failure_recordable"] for row in failed),
        "failure_class_counts": dict(
            sorted(Counter(row["failure_class"] for row in failed).items())
        ),
        "missing_or_nonterminal_scenes": len(missing),
        "completion_rate": _rate(len(completed), expected),
        "generation_complete_rate": _rate(
            sum(row["generation_status"] == "complete" for row in scene_rows),
            expected,
        ),
        "required_satisfaction_rate": _rate(
            sum(row["requirement_status"] == "satisfied" for row in scene_rows),
            len(requirement_observed),
        ),
        "mean_required_coverage": _mean(row["required_coverage"] for row in scene_rows),
        "quality_pass_rate": _rate(
            sum(row["quality_status"] == "passed" for row in scene_rows),
            len(quality_observed),
        ),
        "required_first_instruction_delivery_rate": _rate(
            required_first_delivered_count,
            required_first_applicable_count,
        ),
        "optional_autonomy_preservation_rate": _rate(
            optional_autonomy_preserved_count,
            auto_stage_count,
        ),
        "trace_coverage": _rate(len(trace_observed), expected),
        "mean_scene_time_sec": _mean(row["trace_time_sec"] for row in scene_rows),
        "median_scene_time_sec": _median(row["trace_time_sec"] for row in scene_rows),
        "critic_coverage": _rate(len(critic_observed), len(completed)),
        "critic_mean_score": _mean(row["critic_score"] for row in critic_observed),
        "critic_median_score": _median(row["critic_score"] for row in critic_observed),
        "critic_zero_fail_rate": _rate(
            sum(bool(row["critic_zero_fail"]) for row in critic_observed),
            len(critic_observed),
        ),
        "hard_constraint_pass_rate": _rate(
            sum(row["hard_constraint_pass"] is True for row in completed),
            sum(row["hard_constraint_pass"] is not None for row in completed),
        ),
        "mean_relation_satisfaction": _mean(
            row["relation_satisfaction"] for row in completed
        ),
        "memory_retrieval_scene_coverage": _rate(len(retrieved), expected),
        "memory_injection_scene_coverage": _rate(len(injected), expected),
        "memory_cross_task_verified_scene_coverage": _rate(
            len(cross_task_verified), expected
        ),
        "memory_injection_delivery_rate": _rate(
            injected_stage_count, retrieved_stage_count
        ),
        "unique_retrieved_memory_ids": len(
            {
                memory_id
                for row in scene_rows
                for memory_id in row["memory_retrieved_ids"]
            }
        ),
        "memory_zero_result_reasons": sorted(
            {
                reason
                for row in scene_rows
                for reason in row["memory_zero_result_reasons"]
            }
        ),
        "memory_writer_observed_scenes": len(writer_observed),
        "memory_writer_failure_scenes": len(writer_failures),
        "memory_writer_noop_scenes": len(writer_noops),
        "memory_writer_candidate_records": sum(
            row["memory_writer_candidate_count"] for row in scene_rows
        ),
        "memory_writer_persisted_records": sum(
            row["memory_writer_persisted"] for row in scene_rows
        ),
        "memory_writer_promoted_records": sum(
            row["memory_writer_promoted"] for row in scene_rows
        ),
        "memory_writer_added_records": sum(
            row["memory_writer_added"] for row in scene_rows
        ),
        "memory_writer_merged_records": sum(
            row["memory_writer_merged"] for row in scene_rows
        ),
        "memory_writer_fallback_writes": len(fallback_writes),
        "llm_skill_candidate_count": sum(
            row["llm_skill_candidate_count"] for row in scene_rows
        ),
        "bootstrap_skill_eligible_stage_count": sum(
            row["bootstrap_skill_eligible_stage_count"] for row in scene_rows
        ),
        "bootstrap_skill_candidate_count": sum(
            row["bootstrap_skill_candidate_count"] for row in scene_rows
        ),
        "bootstrap_skill_persisted_candidate_count": sum(
            row["bootstrap_skill_persisted_candidate_count"] for row in scene_rows
        ),
        "bootstrap_skill_rejected_count": sum(
            row["bootstrap_skill_rejected_count"] for row in scene_rows
        ),
        "skill_persisted_candidate_count": sum(
            row["skill_persisted_candidate_count"] for row in scene_rows
        ),
        "skill_promoted_active_count": sum(
            row["skill_promoted_active_count"] for row in scene_rows
        ),
        "skill_rejected_count": sum(row["skill_rejected_count"] for row in scene_rows),
        "skill_rejection_reasons": dict(sorted(skill_rejection_reasons.items())),
        "skill_store_candidate_added": sum(
            row["skill_store_candidate_added"] for row in scene_rows
        ),
        "skill_store_candidate_merged": sum(
            row["skill_store_candidate_merged"] for row in scene_rows
        ),
        "scenesmith_repair_events": sum(
            row["scenesmith_repair_events"] for row in scene_rows
        ),
        "scenesmith_repairs_accepted": sum(
            row["scenesmith_repairs_accepted"] for row in scene_rows
        ),
        "scenesmith_repairs_rejected": sum(
            row["scenesmith_repairs_rejected"] for row in scene_rows
        ),
        "scenesmith_repairs_resolved": sum(
            row["scenesmith_repairs_resolved"] for row in scene_rows
        ),
        "scenesmith_repair_affected_objects": sum(
            row["scenesmith_repair_affected_objects"] for row in scene_rows
        ),
        "scenesmith_repair_timing_events": sum(
            row["scenesmith_repair_timing_events"] for row in scene_rows
        ),
        "scenesmith_repair_overhead_sec": round(
            sum(row["scenesmith_repair_overhead_sec"] for row in scene_rows), 6
        ),
        "scenes_rescued_by_scenesmith_repair": sum(
            bool(row["scene_rescued_by_scenesmith_repair"])
            and row["status"] in {"completed", "completed_with_quality_issues"}
            for row in scene_rows
        ),
        "sceneexpert_repair_plans": sum(
            row["sceneexpert_repair_plans"] for row in scene_rows
        ),
        "sceneexpert_repairs_executed": sum(
            row["sceneexpert_repairs_executed"] for row in scene_rows
        ),
        "task_compiler_llm_scenes": sum(
            row["task_compiler_source"] == "llm" for row in scene_rows
        ),
        "task_compiler_degraded_scenes": sum(
            row["task_compiler_degraded"] for row in scene_rows
        ),
        "global_planner_llm_stage_count": sum(
            len(row["global_planner_llm_stages"]) for row in scene_rows
        ),
        "global_planner_fallback_stage_count": sum(
            len(row["global_planner_fallback_stages"]) for row in scene_rows
        ),
        "brief_injection_verified_stage_count": sum(
            len(row["brief_injection_verified_stages"]) for row in scene_rows
        ),
        "native_stage_agent_invocation_count": sum(
            len(row["stage_agent_invoked_stages"]) for row in scene_rows
        ),
        "missing_native_stage_agent_invocation_count": missing_stage_invocations,
        "native_stage_agent_invocation_rate": _rate(
            sum(len(row["stage_agent_invoked_stages"]) for row in scene_rows),
            sum(len(row["observed_stages"]) for row in scene_rows),
        ),
        "optional_asset_recommendation_count": sum(
            row["optional_asset_recommendation_count"] for row in scene_rows
        ),
    }
    quality_comparison_ready = bool(
        expected
        and process_exit_code in {None, 0}
        and not failed
        and not missing
        and len(critic_observed) == expected
        and len(trace_observed) == expected
        and not missing_stage_invocations
    )
    memory_closed_loop_observed = bool(
        summary["memory_writer_promoted_records"] > 0
        and summary["unique_retrieved_memory_ids"] > 0
        and cross_task_verified
        and injected
    )
    provenance_variants = list(provenance_by_signature.values())
    run_code_provenance = dict(provenance_variants[0]) if provenance_variants else {}
    if len(provenance_variants) > 1:
        run_code_provenance["variants"] = provenance_variants
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": resolved_run_id,
        "output_root": str(root),
        "process_exit_code": process_exit_code,
        "quality_comparison_ready": quality_comparison_ready,
        "memory_closed_loop_observed": memory_closed_loop_observed,
        "experiment_identity": {
            "experiment_names": experiment_names,
            "config_hashes": config_hashes,
            "exact_config_hashes": config_hashes,
            "experiment_signatures": experiment_signatures,
            "control_signatures": control_signatures,
            "source_bundle_hashes": source_bundle_hashes,
            "models": models,
            "code_revisions": revisions,
            "code_dirty_values": sorted(
                {
                    row["code_dirty"]
                    for row in scene_rows
                    if isinstance(row["code_dirty"], bool)
                }
            ),
        },
        "code_provenance": run_code_provenance,
        "memory_identity": {
            "scope": "persistent_configured_bank",
            "memory_dirs": memory_dirs,
            "memory_bank_ids": memory_bank_ids,
            "memory_bank_revision_min": (
                min(memory_bank_revisions) if memory_bank_revisions else None
            ),
            "memory_bank_revision_max": (
                max(memory_bank_revisions) if memory_bank_revisions else None
            ),
            "snapshot_fingerprints": memory_snapshot_fingerprints,
            "snapshot_revisions": memory_snapshot_revisions,
            "frozen_all_unchanged": memory_snapshot_all_unchanged,
            "snapshot_identity_stable": memory_snapshot_identity_stable,
        },
        "evaluation_contract": {
            "active": evaluation_active,
            "pair_ids": evaluation_pair_ids,
            "controlled_dimensions": evaluation_dimensions,
            "arms": evaluation_arms,
            "shared_base_fingerprints": shared_base_fingerprints,
            "shared_base_all_present": shared_base_all_present,
            "memory_read_only_all": memory_read_only_all,
            "frozen_required_all": frozen_required_all,
            "contract_ready": bool(
                evaluation_active
                and len(evaluation_pair_ids) == 1
                and len(evaluation_dimensions) == 1
                and len(evaluation_arms) == 1
                and shared_base_all_present
                and memory_snapshot_all_unchanged
                and memory_snapshot_identity_stable
                and memory_read_only_all
                and frozen_required_all
            ),
        },
        "summary": summary,
        "data_quality_warnings": _unique(warnings),
        "scenes": scene_rows,
    }


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    return value


def _markdown(metrics: dict[str, Any]) -> str:
    summary = metrics["summary"]
    lines = [
        "# SceneExpert Run Metrics",
        "",
        f"- Run: `{metrics['run_id']}`",
        f"- Comparison ready: `{str(metrics['quality_comparison_ready']).lower()}`",
        f"- Cross-task memory loop observed: "
        f"`{str(metrics.get('memory_closed_loop_observed', False)).lower()}`",
        f"- Process exit code: `{metrics.get('process_exit_code')}`",
        "",
        "## Core KPIs",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key in (
        "expected_scenes",
        "completed_scenes",
        "degraded_scenes",
        "failed_scenes",
        "recorded_failed_scenes",
        "missing_or_nonterminal_scenes",
        "completion_rate",
        "generation_complete_rate",
        "required_satisfaction_rate",
        "mean_required_coverage",
        "quality_pass_rate",
        "mean_scene_time_sec",
        "critic_coverage",
        "critic_mean_score",
        "critic_zero_fail_rate",
        "memory_retrieval_scene_coverage",
        "memory_injection_scene_coverage",
        "memory_injection_delivery_rate",
        "memory_cross_task_verified_scene_coverage",
        "memory_writer_promoted_records",
        "memory_writer_persisted_records",
        "llm_skill_candidate_count",
        "bootstrap_skill_eligible_stage_count",
        "bootstrap_skill_candidate_count",
        "bootstrap_skill_persisted_candidate_count",
        "bootstrap_skill_rejected_count",
        "skill_persisted_candidate_count",
        "skill_promoted_active_count",
        "skill_rejected_count",
        "skill_store_candidate_added",
        "skill_store_candidate_merged",
        "memory_writer_noop_scenes",
        "memory_writer_failure_scenes",
        "memory_writer_fallback_writes",
        "scenesmith_repair_events",
        "scenesmith_repairs_accepted",
        "scenesmith_repairs_rejected",
        "scenesmith_repairs_resolved",
        "scenes_rescued_by_scenesmith_repair",
        "scenesmith_repair_timing_events",
        "scenesmith_repair_overhead_sec",
        "sceneexpert_repair_plans",
        "sceneexpert_repairs_executed",
        "task_compiler_llm_scenes",
        "task_compiler_degraded_scenes",
        "global_planner_llm_stage_count",
        "global_planner_fallback_stage_count",
        "brief_injection_verified_stage_count",
        "native_stage_agent_invocation_count",
        "missing_native_stage_agent_invocation_count",
        "native_stage_agent_invocation_rate",
        "optional_asset_recommendation_count",
        "required_first_instruction_delivery_rate",
        "optional_autonomy_preservation_rate",
    ):
        lines.append(f"| `{key}` | {summary.get(key)} |")
    lines.extend(
        [
            "",
            "## Scene failures",
            "",
            f"- Failure class counts: `{summary.get('failure_class_counts', {})}`",
        ]
    )
    experiment_identity = metrics.get("experiment_identity") or {}
    lines.extend(
        [
            "",
            "## Experiment identity",
            "",
            "- Exact configuration hashes: "
            f"`{experiment_identity.get('exact_config_hashes', [])}`",
            "- Semantic experiment signatures: "
            f"`{experiment_identity.get('experiment_signatures', [])}`",
            "- Source-bundle hashes: "
            f"`{experiment_identity.get('source_bundle_hashes', [])}`",
            "- Git revisions (diagnostic only): "
            f"`{experiment_identity.get('code_revisions', [])}`",
        ]
    )
    identity = metrics.get("memory_identity") or {}
    lines.extend(
        [
            "",
            "## Memory identity",
            "",
            f"- Scope: `{identity.get('scope', '')}`",
            f"- Directories: `{identity.get('memory_dirs', [])}`",
            f"- Bank IDs: `{identity.get('memory_bank_ids', [])}`",
            "- Revision range: "
            f"`{identity.get('memory_bank_revision_min')}` → "
            f"`{identity.get('memory_bank_revision_max')}`",
        ]
    )
    lines.extend(["", "## Data quality", ""])
    warnings = metrics.get("data_quality_warnings") or []
    if warnings:
        lines.extend(f"- `{warning}`" for warning in warnings)
    else:
        lines.append("- No structural data-quality warnings detected.")
    return "\n".join(lines) + "\n"


def write_run_metrics(metrics: dict[str, Any]) -> dict[str, str]:
    """Write the machine-readable and human-readable metrics bundle."""
    output_root = Path(str(metrics["output_root"]))
    metrics_dir = output_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    run_json = metrics_dir / "run_metrics.json"
    run_md = metrics_dir / "run_metrics.md"
    scene_jsonl = metrics_dir / "scene_metrics.jsonl"
    scene_csv = metrics_dir / "scene_metrics.csv"
    _atomic_write_text(
        run_json,
        json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    )
    _atomic_write_text(run_md, _markdown(metrics))
    _atomic_write_text(
        scene_jsonl,
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in metrics.get("scenes") or []
        ),
    )
    csv_tmp = scene_csv.with_suffix(scene_csv.suffix + ".tmp")
    with csv_tmp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SCENE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in metrics.get("scenes") or []:
            writer.writerow({key: _csv_value(row.get(key)) for key in SCENE_COLUMNS})
    csv_tmp.replace(scene_csv)
    return {
        "run_json": str(run_json),
        "run_markdown": str(run_md),
        "scene_jsonl": str(scene_jsonl),
        "scene_csv": str(scene_csv),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--process-exit-code", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parser().parse_args(argv)
    metrics = collect_run_metrics(
        args.output_root,
        run_id=args.run_id,
        process_exit_code=args.process_exit_code,
    )
    paths = write_run_metrics(metrics)
    summary = metrics["summary"]
    console_logger.info(
        "SceneExpert run metrics: %s (completed=%d degraded=%d failed=%d "
        "missing=%d failure_classes=%s)",
        paths["run_json"],
        summary["completed_scenes"],
        summary["degraded_scenes"],
        summary["failed_scenes"],
        summary["missing_or_nonterminal_scenes"],
        summary["failure_class_counts"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
