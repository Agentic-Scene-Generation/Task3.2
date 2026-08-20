"""Per-scene, human-readable audit trail for fast-memory activity."""

from __future__ import annotations

import json
import os
import time

from pathlib import Path
from typing import Any

from scenesmith.scene_expert.memory.schemas import MemoryUtilityObservation
from scenesmith.scene_expert.schemas import (
    MemoryInjectionBundle,
    MemoryPack,
    SceneTaskSpec,
    StageExecutionEvidence,
    StageRelationContext,
    StageVerifyReport,
)


class MemoryActivityLogger:
    """Persist retrieval → planning → injection → outcome evidence per scene."""

    SCHEMA_VERSION = "sceneexpert.memory_activity.v1"

    def __init__(
        self,
        output_dir: str | Path,
        *,
        scene_id: str,
        task_spec: SceneTaskSpec,
        experiment_signature: str = "",
        task_id: str = "",
        run_id: str = "",
    ) -> None:
        self._output_dir = Path(output_dir)
        self._json_path = self._output_dir / "memory_activity.json"
        self._markdown_path = self._output_dir / "memory_activity.md"
        self._payload: dict[str, Any] = {
            "schema_version": self.SCHEMA_VERSION,
            "scene_id": scene_id,
            "task_id": task_id,
            "run_id": run_id,
            "experiment_signature": experiment_signature,
            "task_spec": task_spec.model_dump(mode="json"),
            "stages": {},
            "writer": {"status": "not_run"},
            "main_deterministic_repair_events": [],
            "main_deterministic_repair_timing": [],
        }

    def record_pre_stage(
        self,
        *,
        stage: str,
        memory_pack: MemoryPack,
        relation_context: StageRelationContext | None,
        planner_trace: dict[str, Any],
        injection_bundle: MemoryInjectionBundle,
        execution_evidence: StageExecutionEvidence,
    ) -> None:
        """Record every retrieval choice and the exact downstream injection."""
        stage_entry = self._payload["stages"].setdefault(stage, {})
        stage_entry.update(
            {
                "retrieval": memory_pack.model_dump(mode="json"),
                "relation_context": (
                    relation_context.model_dump(mode="json")
                    if relation_context is not None
                    else None
                ),
                "planner_trace": planner_trace,
                "injection": injection_bundle.model_dump(mode="json"),
                "execution_evidence": execution_evidence.model_dump(mode="json"),
            }
        )
        self._save()

    def record_post_stage(
        self,
        *,
        stage: str,
        verify_report: StageVerifyReport | None,
        repair_actions: list[Any],
        scene_state_path: str,
    ) -> None:
        """Attach the authoritative critic result to the selected memory."""
        stage_entry = self._payload["stages"].setdefault(stage, {})
        retrieval = stage_entry.get("retrieval") or {}
        execution = stage_entry.get("execution_evidence") or {}
        observations = []
        for selection in retrieval.get("selections", []) or []:
            if not isinstance(selection, dict):
                continue
            observations.append(
                MemoryUtilityObservation(
                    memory_id=str(selection.get("memory_id") or ""),
                    memory_type=str(selection.get("memory_type") or "success"),
                    task_id=self._payload["task_id"],
                    run_id=self._payload["run_id"],
                    stage=stage,
                    selected_rank=int(selection.get("rank") or 0),
                    retrieval_score=selection.get("score"),
                    injected=bool(execution.get("designer_prompt_contains_memory")),
                    stage_passed=(
                        verify_report.pass_stage if verify_report is not None else None
                    ),
                    # A single run establishes delivery and downstream outcome,
                    # not causal utility. Paired evaluation may classify this
                    # observation later without rewriting the source artifact.
                    outcome="unknown",
                    evidence_ref=scene_state_path,
                ).model_dump(mode="json")
            )
        stage_entry.update(
            {
                "scene_state_path": scene_state_path,
                "verify_report": (
                    verify_report.model_dump(mode="json")
                    if verify_report is not None
                    else None
                ),
                "scene_expert_repair_actions": [
                    (
                        action.model_dump(mode="json")
                        if hasattr(action, "model_dump")
                        else action
                    )
                    for action in repair_actions
                ],
                "utility_observations": observations,
            }
        )
        self._save()

    def record_writer(
        self,
        *,
        proposed_ops: list[Any],
        writer_trace: dict[str, Any],
        apply_summary: dict[str, Any] | None,
        error: str = "",
    ) -> None:
        """Record candidate generation and the final promotion decision."""
        self._payload["writer"] = {
            "status": "failed" if error else "completed",
            "proposed_ops": [
                op.model_dump(mode="json") if hasattr(op, "model_dump") else op
                for op in proposed_ops
            ],
            "writer_trace": writer_trace,
            "store_apply": apply_summary or {},
            "error": error,
        }
        self._save()

    def capture_main_repair_events(
        self,
        path: str | Path,
        timing_path: str | Path | None = None,
    ) -> None:
        """Copy main's deterministic repair events without changing main logic."""
        event_path = Path(path)
        events: list[dict[str, Any]] = []
        if event_path.exists():
            for line in event_path.read_text(encoding="utf-8").splitlines():
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict):
                    events.append(item)
        self._payload["main_deterministic_repair_events"] = events
        timing_events: list[dict[str, Any]] = []
        resolved_timing_path = Path(timing_path) if timing_path else None
        if resolved_timing_path is not None and resolved_timing_path.exists():
            for line in resolved_timing_path.read_text(encoding="utf-8").splitlines():
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (
                    isinstance(item, dict)
                    and str(item.get("module") or "") == "deterministic_repair"
                ):
                    timing_events.append(item)
        self._payload["main_deterministic_repair_timing"] = timing_events
        self._save()

    def _save(self) -> None:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._payload["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        json_text = json.dumps(
            self._payload, ensure_ascii=False, indent=2, sort_keys=True, default=str
        )
        self._atomic_write(self._json_path, json_text + "\n")
        self._atomic_write(self._markdown_path, self._to_markdown())

    @staticmethod
    def _atomic_write(path: Path, text: str) -> None:
        temp_path = path.with_suffix(path.suffix + ".tmp")
        temp_path.write_text(text, encoding="utf-8")
        os.replace(temp_path, path)

    def _to_markdown(self) -> str:
        lines = [
            "# SceneExpert memory activity",
            "",
            f"- Scene: `{self._payload['scene_id']}`",
            f"- Experiment signature: `{self._payload['experiment_signature']}`",
            "",
        ]
        for stage, entry in self._payload["stages"].items():
            retrieval = entry.get("retrieval") or {}
            injection = entry.get("injection") or {}
            report = entry.get("verify_report") or {}
            lines.extend(
                [
                    f"## {stage}",
                    "",
                    "- Retrieved: "
                    f"{len(retrieval.get('success_case_ids', []))} success, "
                    f"{len(retrieval.get('failure_case_ids', []))} failure, "
                    f"{len(retrieval.get('skill_names', []))} skill records.",
                    "- Selected IDs: `"
                    + ", ".join(injection.get("selected_memory_ids", []))
                    + "`",
                    "- Injection hash: `"
                    + str(
                        (entry.get("execution_evidence") or {}).get(
                            "final_injection_hash", ""
                        )
                    )
                    + "`",
                    f"- Stage passed: `{report.get('pass_stage', 'not verified')}`",
                    f"- Scene state: `{entry.get('scene_state_path', '')}`",
                    "",
                ]
            )
        writer = self._payload.get("writer") or {}
        lines.extend(
            [
                "## Memory writer",
                "",
                f"- Status: `{writer.get('status', 'not_run')}`",
                f"- Proposed operations: `{len(writer.get('proposed_ops', []))}`",
                f"- Store result: `{json.dumps(writer.get('store_apply', {}), ensure_ascii=False)}`",
                "",
                "## Main deterministic repair",
                "",
                "- Events captured: `"
                + str(len(self._payload.get("main_deterministic_repair_events", [])))
                + "`",
                "- Timed attempts: `"
                + str(len(self._payload.get("main_deterministic_repair_timing", [])))
                + "`",
                "",
            ]
        )
        return "\n".join(lines)
