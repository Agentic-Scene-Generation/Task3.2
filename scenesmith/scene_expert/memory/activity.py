"""Per-scene, human-readable audit trail for fast-memory activity."""

from __future__ import annotations

import json
import os
import re
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
    ) -> list[MemoryUtilityObservation]:
        """Attach the authoritative critic result to the selected memory."""
        stage_entry = self._payload["stages"].setdefault(stage, {})
        retrieval = stage_entry.get("retrieval") or {}
        injection = stage_entry.get("injection") or {}
        execution = stage_entry.get("execution_evidence") or {}
        planner_selected = set(injection.get("planner_selected_skill_names") or [])
        prompt_delivered = set(injection.get("prompt_delivered_skill_names") or [])
        skill_decisions = {
            str(item.get("skill_name") or ""): item
            for item in retrieval.get("skill_filter_decisions", []) or []
            if isinstance(item, dict) and str(item.get("skill_name") or "")
        }
        observations: list[MemoryUtilityObservation] = []
        for selection in retrieval.get("selections", []) or []:
            if not isinstance(selection, dict):
                continue
            memory_id = str(selection.get("memory_id") or "")
            memory_type = str(selection.get("memory_type") or "success")
            delivered = bool(
                memory_type == "skill"
                and memory_id in prompt_delivered
                and execution.get("designer_prompt_contains_memory")
            )
            outcome, outcome_basis = self._classify_outcome(
                memory_type=memory_type,
                delivered=delivered,
                skill_decision=skill_decisions.get(memory_id),
                verify_report=verify_report,
            )
            observations.append(
                MemoryUtilityObservation(
                    memory_id=memory_id,
                    memory_type=memory_type,
                    task_id=self._payload["task_id"],
                    run_id=self._payload["run_id"],
                    stage=stage,
                    selected_rank=int(selection.get("rank") or 0),
                    retrieval_score=selection.get("score"),
                    injected=(
                        delivered
                        if memory_type == "skill"
                        else bool(execution.get("designer_prompt_contains_memory"))
                    ),
                    retrieved=True,
                    planner_selected=(
                        memory_type == "skill" and memory_id in planner_selected
                    ),
                    prompt_delivered=delivered,
                    stage_passed=(
                        verify_report.pass_stage if verify_report is not None else None
                    ),
                    outcome=outcome,
                    outcome_basis=outcome_basis,
                    evidence_ref=scene_state_path,
                )
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
                "utility_observations": [
                    observation.model_dump(mode="json") for observation in observations
                ],
            }
        )
        self._save()
        return observations

    @staticmethod
    def _classify_outcome(
        *,
        memory_type: str,
        delivered: bool,
        skill_decision: dict[str, Any] | None,
        verify_report: StageVerifyReport | None,
    ) -> tuple[str, str]:
        """Label verified skill co-occurrence without claiming causal impact."""
        if memory_type != "skill" or not delivered or verify_report is None:
            return "unknown", "not_a_delivered_skill"
        hard_report = verify_report.hard_check_report or {}
        if verify_report.pass_stage and hard_report.get("hard_valid") is not False:
            return "positive", "verified_stage_pass_after_skill_delivery"
        if hard_report.get("hard_valid") is not False:
            return "unknown", "failure_not_owned_by_deterministic_hard_contract"

        evidence_text = " ".join(
            [
                *[str(value) for value in hard_report.get("failed_checks", [])],
                *[
                    " ".join([issue.issue_type, issue.object_name, issue.description])
                    for issue in verify_report.issues
                ],
            ]
        )
        decision = dict(skill_decision or {})
        relevance_text = " ".join(
            [
                *[str(value) for value in decision.get("matched_constraint_ids", [])],
                *[str(value) for value in decision.get("required_relation_types", [])],
                *[str(value) for value in decision.get("required_object_roles", [])],
            ]
        )
        evidence_tokens = MemoryActivityLogger._meaningful_tokens(evidence_text)
        relevance_tokens = MemoryActivityLogger._meaningful_tokens(relevance_text)
        if relevance_tokens and evidence_tokens & relevance_tokens:
            return "negative", "relevant_hard_failure_after_skill_delivery"
        if not relevance_tokens:
            return "unknown", "skill_lacks_structured_hard_contract_scope"
        return "unknown", "hard_failure_not_relevant_to_delivered_skill"

    @staticmethod
    def _meaningful_tokens(text: str) -> set[str]:
        ignored = {
            "and",
            "avoid",
            "for",
            "from",
            "must",
            "object",
            "place",
            "room",
            "skill",
            "stage",
            "the",
            "this",
            "with",
        }
        return {
            token
            for token in re.split(r"[^a-z0-9_]+", str(text or "").casefold())
            if len(token) >= 3 and token not in ignored
        }

    def record_skill_learning(self, *, summary: dict[str, Any]) -> None:
        """Attach the single end-of-scene durable-bank update to the audit."""
        self._payload["skill_learning"] = dict(summary)
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
                    "- Skill funnel: retrieved=`"
                    + ", ".join(injection.get("retrieved_skill_names", []))
                    + "`; planner-selected=`"
                    + ", ".join(injection.get("planner_selected_skill_names", []))
                    + "`; prompt-delivered=`"
                    + ", ".join(injection.get("prompt_delivered_skill_names", []))
                    + "`.",
                    f"- Scene state: `{entry.get('scene_state_path', '')}`",
                    "",
                ]
            )
        writer = self._payload.get("writer") or {}
        lines.extend(
            [
                "## Skill learning",
                "",
                "- Store result: `"
                + json.dumps(
                    self._payload.get("skill_learning", {}), ensure_ascii=False
                )
                + "`",
                "",
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
