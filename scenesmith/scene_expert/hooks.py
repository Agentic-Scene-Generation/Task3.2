"""SceneExpertHookRunner: pre/post-stage hooks injected into SceneSmith's _generate_room.

This is the main integration point between SceneExpert and SceneSmith.
It is created once per scene (in _generate_single_scene) and passed down to
_generate_room, where it is called before and after each stage agent runs.

Pre-stage hook:  Memory retrieval → StageBrief → injects into scene.text_description
Post-stage hook: Stage verification → Repair decision → Trace logging

Ablation mode controls which components are active:
  "disabled"         → hooks are never created; SceneSmith runs as-is
  "harness_only"     → Harness FSM + GlobalPlanner, NO memory retrieval
  "harness_memory"   → Harness FSM + GlobalPlanner + FastMemory (MVP default)
  "full"             → harness_memory + future LoRA (placeholder)
"""

from __future__ import annotations

import logging
import os
import time
import hashlib
import json
from pathlib import Path
from typing import Any

from scenesmith.agent_utils.room import RoomScene
from scenesmith.scene_expert.context_bundle import build_stage_context_bundle
from scenesmith.scene_expert.global_planner import GlobalPlanner
from scenesmith.scene_expert.harness import STAGE_ORDER, Harness
from scenesmith.scene_expert.memory.retriever import MemoryRetriever
from scenesmith.scene_expert.memory.store import FastMemoryStore
from scenesmith.scene_expert.memory.writer import MemoryWriter
from scenesmith.scene_expert.memory.schemas import FailureCase, SuccessCase
from scenesmith.scene_expert.memory.text_builder import build_embedding_text
from scenesmith.scene_expert.repair_controller import RepairController
from scenesmith.scene_expert.repair_taxonomy import classify_hard_reasons
from scenesmith.scene_expert.schemas import (
    FullVerifyReport,
    MemoryPack,
    RepairResult,
    SceneTaskSpec,
    StageBrief,
    StageVerifyReport,
)
from scenesmith.scene_expert.task_compiler import TaskCompiler
from scenesmith.scene_expert.trace_logger import TraceLogger
from scenesmith.scene_expert.verifier import FullVerifier, StageVerifier

console_logger = logging.getLogger(__name__)

# Valid ablation modes
ABLATION_MODES = frozenset(["disabled", "harness_only", "harness_memory", "full"])


def _empty_memory_pack() -> MemoryPack:
    return MemoryPack(success_hints=[], failure_hints=[], skill_texts=[])


def _stable_config_hash(cfg_dict: dict) -> str:
    """Return a short stable hash for trace reproducibility metadata."""
    try:
        payload = json.dumps(cfg_dict, sort_keys=True, default=str)
    except TypeError:
        payload = repr(cfg_dict)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _deep_merge_dicts(base: dict, override: dict) -> dict:
    """Merge nested dicts without mutating either input."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _cfg_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(value)


def _cfg_int(value: Any, default: int) -> int:
    if value is None or value == "":
        return default
    return int(value)


def _cfg_float(value: Any, default: float) -> float:
    if value is None or value == "":
        return default
    return float(value)


def _compact_memory_text(text: str, max_chars: int = 300) -> str:
    """Compress multiline memory hints for prompt/directive injection."""
    compact = " ".join(text.strip().split())
    return compact if len(compact) <= max_chars else compact[: max_chars - 3] + "..."


def _extend_unique(target: list[str], items: list[str]) -> list[str]:
    """Append non-empty unique strings while preserving order."""
    seen = {item.strip() for item in target if item.strip()}
    for item in items:
        text = item.strip()
        if text and text not in seen:
            target.append(text)
            seen.add(text)
    return target


def _skill_name_from_text(skill_text: str) -> str:
    first_line = skill_text.strip().splitlines()[0] if skill_text.strip() else ""
    if first_line.startswith("[Skill:") and first_line.endswith("]"):
        return first_line[len("[Skill:") : -1].strip()
    return _compact_memory_text(first_line, max_chars=80)


def _apply_memory_to_stage_brief(
    stage_brief: StageBrief,
    memory_pack: MemoryPack,
) -> StageBrief:
    """Make retrieved memory survive even if GlobalPlanner underuses it."""
    success_rules = [
        "Retrieved success memory: " + _compact_memory_text(hint)
        for hint in memory_pack.success_hints[:3]
    ]
    failure_rules = [
        _compact_memory_text(hint)
        for hint in memory_pack.failure_hints[:3]
    ]
    critic_checks = [
        "Verify retrieved failure is avoided: " + _compact_memory_text(hint)
        for hint in memory_pack.failure_hints[:3]
    ]
    skill_names = [
        name
        for text in memory_pack.skill_texts[:3]
        if (name := _skill_name_from_text(text))
    ]

    return stage_brief.model_copy(
        update={
            "constraints_for_designer": _extend_unique(
                list(stage_brief.constraints_for_designer),
                success_rules,
            ),
            "failure_patterns_to_avoid": _extend_unique(
                list(stage_brief.failure_patterns_to_avoid),
                failure_rules,
            ),
            "checks_for_critic": _extend_unique(
                list(stage_brief.checks_for_critic),
                critic_checks,
            ),
            "recommended_skills": _extend_unique(
                list(stage_brief.recommended_skills),
                skill_names,
            ),
        }
    )


def _format_memory_directives(memory_pack: MemoryPack) -> str:
    """Format retrieved memory as a direct hook-level prompt block."""
    parts: list[str] = []
    if memory_pack.success_hints:
        parts.append("Positive guidance from retrieved memory:")
        parts.extend(
            f"  - {_compact_memory_text(hint)}"
            for hint in memory_pack.success_hints[:3]
        )
    if memory_pack.failure_hints:
        parts.append("Negative constraints from retrieved memory:")
        parts.extend(
            f"  - {_compact_memory_text(hint)}"
            for hint in memory_pack.failure_hints[:3]
        )
    if memory_pack.skill_texts:
        parts.append("Reusable skills from retrieved memory:")
        parts.extend(
            f"  - {_compact_memory_text(skill)}"
            for skill in memory_pack.skill_texts[:2]
        )
    if not parts:
        return ""
    return (
        "=== SceneExpert Retrieved Memory Directives ===\n"
        + "\n".join(parts)
        + "\n=== End Retrieved Memory Directives ==="
    )


def _build_hybrid_retriever(
    memory_store: FastMemoryStore,
    memory_dir: str,
    memory_cfg: dict,
    ret_cfg: dict,
    timing_path: Path | None = None,
):
    """Construct the optional hybrid retriever from memory config."""
    if not (
        memory_store.success_cases
        or memory_store.failure_cases
        or memory_store.skills
    ):
        console_logger.info(
            "Hybrid memory requested but memory store is empty; using "
            "lightweight lexical retriever and skipping BGE-M3 initialization."
        )
        return MemoryRetriever(
            store=memory_store,
            max_success=_cfg_int(ret_cfg.get("max_success_cases"), 3),
            max_failure=_cfg_int(ret_cfg.get("max_failure_cases"), 3),
            max_skills=_cfg_int(ret_cfg.get("max_skills"), 2),
        )

    from scenesmith.scene_expert.memory.embedding import SceneMemoryEmbedder
    from scenesmith.scene_expert.memory.hybrid_retriever import HybridMemoryRetriever
    from scenesmith.scene_expert.memory.scoring import HybridScoreWeights

    emb_cfg = memory_cfg.get("embedding", {})
    idx_cfg = memory_cfg.get("index", {})
    backend = idx_cfg.get("backend", "numpy")
    if backend != "numpy":
        raise NotImplementedError(
            f"SceneExpert hybrid memory currently supports numpy index only, got {backend!r}."
        )

    weights_cfg = memory_cfg.get("hybrid_weights", {})
    weights = HybridScoreWeights(
        embedding_similarity=_cfg_float(weights_cfg.get("embedding_similarity"), 0.45),
        object_overlap=_cfg_float(weights_cfg.get("object_overlap"), 0.20),
        room_stage_match=_cfg_float(weights_cfg.get("room_stage_match"), 0.15),
        memory_quality_score=_cfg_float(weights_cfg.get("memory_quality_score"), 0.10),
        recency_or_verified=_cfg_float(weights_cfg.get("recency_or_verified"), 0.10),
    )

    embedder = SceneMemoryEmbedder(
        model_dir=emb_cfg.get("model_dir"),
        model_id=emb_cfg.get("model_id", "BAAI/bge-m3"),
        device=emb_cfg.get("device", "cpu"),
        batch_size=_cfg_int(emb_cfg.get("batch_size"), 8),
        max_length=_cfg_int(emb_cfg.get("max_length"), 512),
        normalize=_cfg_bool(emb_cfg.get("normalize"), True),
    )
    return HybridMemoryRetriever(
        store=memory_store,
        memory_dir=memory_dir,
        embedder=embedder,
        index_dir=idx_cfg.get("dir"),
        max_success=_cfg_int(ret_cfg.get("max_success_cases"), 3),
        max_failure=_cfg_int(ret_cfg.get("max_failure_cases"), 3),
        max_skills=_cfg_int(ret_cfg.get("max_skills"), 2),
        recall_top_k=_cfg_int(ret_cfg.get("recall_top_k"), 30),
        sim_threshold=_cfg_float(ret_cfg.get("sim_threshold"), 0.0),
        object_overlap_threshold=_cfg_float(
            ret_cfg.get("object_overlap_threshold"),
            0.15,
        ),
        weights=weights,
        require_indexes=_cfg_bool(idx_cfg.get("require_ready"), True),
        auto_build_indexes=_cfg_bool(idx_cfg.get("auto_build_missing"), False),
        timing_path=timing_path,
    )


class SceneExpertHookRunner:
    """Per-scene hook runner that wraps SceneSmith stage execution.

    One instance is created per scene (prompt). It holds the task spec,
    all SceneExpert module references, and accumulated per-stage trace data.

    Thread safety: NOT thread-safe. Use one instance per scene.
    """

    def __init__(
        self,
        prompt: str,
        scene_id: int,
        output_dir: Path,
        mode: str,
        task_spec: SceneTaskSpec,
        harness: Harness,
        global_planner: GlobalPlanner,
        retriever: Any | None,
        stage_verifier: StageVerifier,
        full_verifier: FullVerifier,
        repair_controller: RepairController,
        memory_writer: MemoryWriter | None,
        memory_store: FastMemoryStore | None,
        qwen_model: str,
        experiment_name: str = "",
        config_hash: str = "",
        start_stage: str = "floor_plan",
    ) -> None:
        self._prompt = prompt
        self._scene_id = scene_id
        self._output_dir = output_dir
        self._mode = mode
        self._scene_debug_dir = output_dir / f"scene_{scene_id:03d}" / "scene_expert"
        self._retrieval_timing_path = (
            self._scene_debug_dir / "timing" / "memory_retrieval.jsonl"
        )
        self._context_debug_dir = self._scene_debug_dir / "context_bundles"

        self._task_spec = task_spec
        self._harness = harness
        self._global_planner = global_planner
        self._retriever = retriever
        self._stage_verifier = stage_verifier
        self._full_verifier = full_verifier
        self._repair_controller = repair_controller
        self._memory_writer = memory_writer
        self._memory_store = memory_store
        self._qwen_model = qwen_model
        self._experiment_name = experiment_name
        self._config_hash = config_hash
        self._start_stage = start_stage
        self._stage_order_baseline = self._initial_completed_stages(start_stage)
        self._room_start_stage = "furniture" if start_stage == "floor_plan" else start_stage
        self._room_stage_order_baseline = self._initial_completed_stages(
            self._room_start_stage
        )

        self._trace_logger = TraceLogger(
            output_dir=str(output_dir),
            scene_index=scene_id,
            prompt=prompt,
            experiment_name=experiment_name,
            config_hash=config_hash,
        )
        self._stage_reports: list[StageVerifyReport] = []
        self._completed_stages: list[str] = list(self._stage_order_baseline)
        self._qwen_calls = 0

        # Current stage state (populated in pre_stage, consumed in post_stage)
        self._current_stage: str = ""
        self._current_memory_pack: MemoryPack = _empty_memory_pack()
        self._current_stage_brief: StageBrief | None = None
        self._stage_start_time: float = 0.0
        # Original text_description per stage (so we can restore if needed)
        self._original_text_descriptions: dict[str, str] = {}
        self._last_injected_floor_plan_prompt: str = prompt

    def _save_context_bundle(
        self,
        *,
        stage: str,
        agent_role: str,
        event: str,
        scene: RoomScene | None = None,
        prompt: Any = "",
        last_hard_issues: list[str] | None = None,
    ) -> None:
        """Save a structured pre-LLM context snapshot for audit/debug."""
        try:
            bundle = build_stage_context_bundle(
                stage=stage,
                agent_role=agent_role,
                event=event,
                task_spec=self._task_spec,
                stage_brief=self._current_stage_brief,
                scene=scene,
                memory_pack=self._current_memory_pack,
                history_summary=self._build_scene_state_summary()
                if scene is not None
                else "",
                last_hard_issues=last_hard_issues or [],
                prompt=prompt,
                trace_id=f"trace_{self._scene_id:06d}",
                scene_id=f"scene_{self._scene_id:03d}",
                metadata={
                    "mode": self._mode,
                    "experiment_name": self._experiment_name,
                    "config_hash": self._config_hash,
                },
            )
            safe_stage = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in stage)
            safe_event = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in event)
            path = (
                self._context_debug_dir
                / safe_stage
                / f"{int(time.time() * 1000)}_{agent_role}_{safe_event}.json"
            )
            bundle.save(path)
        except Exception as e:
            console_logger.warning(
                "[SceneExpert] Failed to save StageContextBundle for %s/%s: %s",
                stage,
                event,
                e,
            )

    def _record_memory_retrieval_timing(
        self,
        *,
        stage: str,
        elapsed_sec: float,
        pack: MemoryPack | None = None,
        error: str = "",
    ) -> None:
        """Record pre-stage memory retrieval timing even for empty/fallback stores."""
        try:
            record = {
                "schema_version": "1.0",
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "stage": stage,
                "module": "scene_expert_memory_retrieval",
                "retriever": type(self._retriever).__name__
                if self._retriever is not None
                else "none",
                "elapsed_sec": round(float(elapsed_sec), 6),
                "success_hints": len(pack.success_hints) if pack else 0,
                "failure_hints": len(pack.failure_hints) if pack else 0,
                "skills": len(pack.skill_texts) if pack else 0,
                "has_placement_reference": bool(pack and pack.placement_reference),
                "error": error,
            }
            self._retrieval_timing_path.parent.mkdir(parents=True, exist_ok=True)
            with self._retrieval_timing_path.open(
                "a",
                encoding="utf-8",
                newline="\n",
            ) as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except Exception as timing_error:
            console_logger.warning(
                "Failed to record SceneExpert memory retrieval timing: %s",
                timing_error,
            )

    def _stage_score_quality(self, report: StageVerifyReport) -> float:
        if not report.scores:
            return 0.0
        return max(0.0, min(1.0, sum(report.scores.values()) / len(report.scores)))

    def _commit_stage_memory(
        self,
        *,
        stage: str,
        verify_report: StageVerifyReport | None,
        scene_state_path: str,
        repair_actions: list[RepairResult],
    ) -> None:
        """Continuously commit post-stage verifier results to the shared bank."""
        if (
            self._memory_store is None
            or verify_report is None
            or self._mode not in ("harness_memory", "full")
        ):
            return
        try:
            quality = self._stage_score_quality(verify_report)
            event = {
                "schema_version": "1.0",
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "event_type": "stage_verify",
                "trace_id": f"trace_{self._scene_id:06d}",
                "scene_id": f"scene_{self._scene_id:03d}",
                "stage": stage,
                "scene_state_path": scene_state_path,
                "pass_stage": verify_report.pass_stage,
                "quality_score": quality,
                "scores": verify_report.scores,
                "issues": [issue.model_dump() for issue in verify_report.issues],
                "repair_actions": [
                    action.model_dump()
                    if hasattr(action, "model_dump")
                    else getattr(action, "__dict__", str(action))
                    for action in repair_actions
                ],
                "critique_summary": verify_report.critique_summary[:2000],
            }
            self._memory_store.append_event(event)

            digest = hashlib.sha1(
                json.dumps(event, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()[:12]
            if verify_report.pass_stage and quality >= 0.75:
                case = SuccessCase(
                    case_id=f"success_{self._task_spec.room_type}_{stage}_{digest}",
                    room_type=self._task_spec.room_type,
                    style=self._task_spec.style,
                    stage=stage,
                    task_signature=self._stage_required_objects(stage),
                    required_objects=self._stage_required_objects(stage),
                    functional_zones=self._task_spec.functional_zones,
                    scene_summary=(
                        f"{stage} passed SceneExpert stage verifier in "
                        f"trace_{self._scene_id:06d}."
                    ),
                    successful_pattern=[
                        verify_report.critique_summary[:900]
                        or f"{stage} passed with quality_score={quality:.2f}."
                    ],
                    positive_guidance=[
                        "Use as a weak positive prior; adapt geometry to the "
                        "current room and re-check hard constraints."
                    ],
                    scores=verify_report.scores,
                    trace_ref=f"trace_{self._scene_id:06d}",
                    quality_score=quality,
                    confidence=0.4,
                    created_at=event["created_at"],
                )
                if not case.embedding_text:
                    case = case.model_copy(
                        update={"embedding_text": build_embedding_text(case)}
                    )
                self._memory_store.add_success_case(case)
            elif not verify_report.pass_stage and verify_report.issues:
                reasons = [
                    issue.description or issue.issue_type for issue in verify_report.issues
                ]
                classified = classify_hard_reasons(reasons)
                case = FailureCase(
                    failure_id=f"failure_{self._task_spec.room_type}_{stage}_{digest}",
                    room_type=self._task_spec.room_type,
                    stage=stage,
                    object=verify_report.issues[0].object_name,
                    failure_type=classified[0].category.value,
                    bad_pattern=verify_report.issues[0].description,
                    failure_reason="; ".join(reasons)[:900],
                    repair_action=(
                        repair_actions[0].repair_action
                        if repair_actions
                        else "Run stage repair loop, re-render, and re-score before accepting."
                    ),
                    repair_verified=False,
                    required_objects=self._stage_required_objects(stage),
                    functional_zones=self._task_spec.functional_zones,
                    scene_summary=(
                        f"{stage} failed SceneExpert stage verifier in "
                        f"trace_{self._scene_id:06d}."
                    ),
                    quality_score=quality,
                    confidence=0.55,
                    created_at=event["created_at"],
                    scope="stage",
                    is_deterministic=all(item.deterministic for item in classified),
                    negative_constraint="; ".join(reasons)[:700],
                    critic_check="Verify this failure class before stage acceptance.",
                    trace_ref=f"trace_{self._scene_id:06d}",
                )
                if not case.embedding_text:
                    case = case.model_copy(
                        update={"embedding_text": build_embedding_text(case)}
                    )
                self._memory_store.add_failure_case(case)
        except Exception as e:
            console_logger.warning(
                "[SceneExpert] Stage-level public memory commit failed for %s: %s",
                stage,
                e,
            )

    def _stage_required_objects(self, stage: str) -> list[str]:
        if stage in ("floor_plan", "furniture"):
            return list(self._task_spec.required_large_objects)
        if stage == "wall_mounted":
            return list(self._task_spec.required_wall_objects)
        if stage == "ceiling_mounted":
            return list(self._task_spec.required_ceiling_objects)
        if stage == "manipuland":
            return list(self._task_spec.required_small_objects)
        return []

    # ------------------------------------------------------------------
    # Pre-stage hook: called BEFORE the SceneSmith stage agent runs
    # ------------------------------------------------------------------

    def pre_floor_plan(self) -> str:
        """Prepare SceneExpert context for the house-level floor_plan stage.

        Floor plan generation runs in an isolated subprocess and receives only a
        prompt string, so this returns an enhanced prompt instead of mutating a
        RoomScene.
        """
        stage = "floor_plan"
        console_logger.info(f"[SceneExpert/{self._mode}] pre_stage: {stage}")
        self._validate_stage_transition(stage)
        self._current_stage = stage
        self._stage_start_time = time.time()
        self._qwen_calls = 0

        if self._retriever is not None and self._mode in ("harness_memory", "full"):
            try:
                retrieval_start = time.time()
                self._current_memory_pack = self._retriever.retrieve(
                    self._task_spec, stage
                )
                retrieval_elapsed = time.time() - retrieval_start
                n_hints = (
                    len(self._current_memory_pack.success_hints)
                    + len(self._current_memory_pack.failure_hints)
                )
                self._record_memory_retrieval_timing(
                    stage=stage,
                    elapsed_sec=retrieval_elapsed,
                    pack=self._current_memory_pack,
                )
                console_logger.info(
                    f"[SceneExpert] Memory retrieved for {stage}: "
                    f"{n_hints} hints, {len(self._current_memory_pack.skill_texts)} skills "
                    f"in {retrieval_elapsed:.2f}s"
                )
            except Exception as e:
                self._record_memory_retrieval_timing(
                    stage=stage,
                    elapsed_sec=time.time() - retrieval_start
                    if "retrieval_start" in locals()
                    else 0.0,
                    pack=None,
                    error=str(e),
                )
                console_logger.warning(f"Memory retrieval failed for {stage}: {e}")
                self._current_memory_pack = _empty_memory_pack()
        else:
            self._current_memory_pack = _empty_memory_pack()

        self._current_stage_brief = None
        if self._mode in ("harness_only", "harness_memory", "full"):
            try:
                planner_start = time.time()
                context = self._harness.build_context(
                    stage=stage,
                    task_spec=self._task_spec,
                    memory_pack=self._current_memory_pack,
                )
                self._current_stage_brief = self._global_planner.generate_stage_brief(
                    context=context,
                    scene_state_summary="No floor plan has been generated yet.",
                )
                self._current_stage_brief = _apply_memory_to_stage_brief(
                    self._current_stage_brief,
                    self._current_memory_pack,
                )
                self._qwen_calls += 1
                console_logger.info(
                    f"[SceneExpert] StageBrief generated for {stage}: "
                    f"{len(self._current_stage_brief.constraints_for_designer)} constraints "
                    f"in {time.time() - planner_start:.2f}s"
                )
            except Exception as e:
                console_logger.warning(
                    f"GlobalPlanner failed for {stage}, running without StageBrief: {e}"
                )

        enhanced = self._prompt
        if self._current_stage_brief is not None:
            enhanced += "\n\n" + self._current_stage_brief.to_injection_text()
        memory_directives = _format_memory_directives(self._current_memory_pack)
        if memory_directives:
            enhanced += "\n\n" + memory_directives
        if self._current_memory_pack.placement_reference:
            enhanced += "\n\n" + self._current_memory_pack.placement_reference
        self._last_injected_floor_plan_prompt = enhanced
        self._save_context_bundle(
            stage=stage,
            agent_role="global_planner",
            event="pre_floor_plan",
            prompt=enhanced,
        )
        self._trace_logger.save_stage_context(
            stage=stage,
            memory_pack=self._current_memory_pack,
            stage_brief=self._current_stage_brief,
            phase="pre",
        )
        return enhanced

    def post_floor_plan(self, scene_dir: Path) -> None:
        """Verify and log the house-level floor_plan stage."""
        stage = "floor_plan"
        console_logger.info(f"[SceneExpert/{self._mode}] post_stage: {stage}")

        scene_state_info = self._extract_floor_plan_state_info(scene_dir)
        verify_report: StageVerifyReport | None = None
        repair_actions: list[RepairResult] = []
        try:
            verify_start = time.time()
            verify_report = self._stage_verifier.verify(
                stage=stage,
                stage_output_dir=str(scene_dir),
                task_spec=self._task_spec,
                stage_brief=self._current_stage_brief,
                scene_state_info=scene_state_info,
            )
            console_logger.info(
                "[SceneExpertTiming] stage=%s module=stage_verifier elapsed=%.2fs",
                stage,
                time.time() - verify_start,
            )
            self._stage_reports.append(verify_report)

            if not verify_report.pass_stage:
                console_logger.warning(
                    f"[SceneExpert] Stage {stage} FAILED verification: "
                    f"issues={[i.issue_type for i in verify_report.issues]}"
                )
                decision = self._harness.decide_repair(stage, verify_report)
                if decision.should_repair:
                    repair_result = self._repair_controller.repair(
                        repair_type=decision.strategy,
                        stage=stage,
                        verify_report=verify_report,
                        scene_path=str(scene_dir),
                        stage_brief=self._current_stage_brief,
                        task_spec=self._task_spec,
                    )
                    repair_actions.append(repair_result)
                    self._repair_controller.record_failure_to_memory(
                        stage=stage,
                        room_type=self._task_spec.room_type,
                        repair_result=repair_result,
                        verify_report=verify_report,
                        repair_verified=False,
                    )
            else:
                console_logger.info(f"[SceneExpert] Stage {stage} PASSED verification")
        except Exception as e:
            console_logger.warning(f"[SceneExpert] Verification failed for {stage}: {e}")

        elapsed = time.time() - self._stage_start_time
        self._commit_stage_memory(
            stage=stage,
            verify_report=verify_report,
            scene_state_path=str(scene_dir),
            repair_actions=repair_actions,
        )
        self._trace_logger.log_stage(
            stage=stage,
            memory_pack=self._current_memory_pack,
            stage_brief=self._current_stage_brief,
            scene_state_path=str(scene_dir),
            verify_report=verify_report,
            repair_actions=repair_actions,
            qwen_calls=self._qwen_calls,
            stage_time_sec=round(elapsed, 1),
        )
        self._trace_logger.save_stage_context(
            stage=stage,
            memory_pack=self._current_memory_pack,
            stage_brief=self._current_stage_brief,
            phase="post",
        )
        self._trace_logger.save_stage_visual_manifest(stage, str(scene_dir))
        self._completed_stages.append(stage)
        console_logger.info(
            "[SceneExpertTiming] stage=%s module=stage_total elapsed=%.2fs",
            stage,
            elapsed,
        )

    def pre_stage(self, stage: str, scene: RoomScene) -> None:
        """Retrieve memory, generate StageBrief, inject into scene.text_description.

        Called from _generate_room immediately before each stage's agent is built.

        Args:
            stage: Current stage name (e.g., "furniture").
            scene: The RoomScene that will be passed to the stage agent.
        """
        console_logger.info(f"[SceneExpert/{self._mode}] pre_stage: {stage}")
        self._validate_stage_transition(stage)
        self._current_stage = stage
        self._stage_start_time = time.time()
        self._qwen_calls = 0

        # Save original text_description for restoration after stage
        self._original_text_descriptions[stage] = scene.text_description

        # --- Step 1: Memory retrieval (skip in harness_only mode) ---
        if self._retriever is not None and self._mode in ("harness_memory", "full"):
            try:
                retrieval_start = time.time()
                self._current_memory_pack = self._retriever.retrieve(
                    self._task_spec, stage
                )
                retrieval_elapsed = time.time() - retrieval_start
                n_hints = (
                    len(self._current_memory_pack.success_hints)
                    + len(self._current_memory_pack.failure_hints)
                )
                self._record_memory_retrieval_timing(
                    stage=stage,
                    elapsed_sec=retrieval_elapsed,
                    pack=self._current_memory_pack,
                )
                console_logger.info(
                    f"[SceneExpert] Memory retrieved for {stage}: "
                    f"{n_hints} hints, {len(self._current_memory_pack.skill_texts)} skills "
                    f"in {retrieval_elapsed:.2f}s"
                )
            except Exception as e:
                self._record_memory_retrieval_timing(
                    stage=stage,
                    elapsed_sec=time.time() - retrieval_start
                    if "retrieval_start" in locals()
                    else 0.0,
                    pack=None,
                    error=str(e),
                )
                console_logger.warning(f"Memory retrieval failed for {stage}: {e}")
                self._current_memory_pack = _empty_memory_pack()
        else:
            self._current_memory_pack = _empty_memory_pack()

        # --- Step 2: Global Planner -> StageBrief ---
        self._current_stage_brief = None
        if self._mode in ("harness_only", "harness_memory", "full"):
            try:
                planner_start = time.time()
                scene_state_summary = self._build_scene_state_summary()
                context = self._harness.build_context(
                    stage=stage,
                    task_spec=self._task_spec,
                    memory_pack=self._current_memory_pack,
                )
                self._current_stage_brief = self._global_planner.generate_stage_brief(
                    context=context,
                    scene_state_summary=scene_state_summary,
                )
                self._current_stage_brief = _apply_memory_to_stage_brief(
                    self._current_stage_brief,
                    self._current_memory_pack,
                )
                self._qwen_calls += 1
                console_logger.info(
                    f"[SceneExpert] StageBrief generated for {stage}: "
                    f"{len(self._current_stage_brief.constraints_for_designer)} constraints "
                    f"in {time.time() - planner_start:.2f}s"
                )
            except Exception as e:
                console_logger.warning(
                    f"GlobalPlanner failed for {stage}, running without StageBrief: {e}"
                )

        # --- Step 3: Inject StageBrief into scene.text_description ---
        memory_directives = _format_memory_directives(self._current_memory_pack)
        if self._current_stage_brief is not None:
            brief_text = self._current_stage_brief.to_injection_text()
            injection_text = brief_text
            if memory_directives:
                injection_text += "\n\n" + memory_directives
            enhanced = (
                scene.text_description
                + "\n\n"
                + injection_text
            )
            scene.text_description = enhanced
            setattr(scene, "scene_expert_brief", injection_text)
            if memory_directives:
                setattr(scene, "scene_expert_memory_directives", memory_directives)
            briefs = getattr(scene, "scene_expert_briefs", {})
            if not isinstance(briefs, dict):
                briefs = {}
            briefs[stage] = injection_text
            setattr(scene, "scene_expert_briefs", briefs)
            console_logger.debug(
                f"[SceneExpert] Injected StageBrief into scene.text_description for {stage}"
            )
        elif memory_directives:
            scene.text_description = scene.text_description + "\n\n" + memory_directives
            setattr(scene, "scene_expert_memory_directives", memory_directives)

        # --- Step 4: Inject placement reference directly (bypasses GlobalPlanner) ---
        # This gives the designer exact coordinates/surfaces from the best historical
        # run, so it doesn't have to guess layout from scratch.
        placement_ref = self._current_memory_pack.placement_reference
        if placement_ref:
            scene.text_description = scene.text_description + "\n\n" + placement_ref
            console_logger.info(
                f"[SceneExpert] Injected placement reference for {stage} "
                f"({placement_ref.count(chr(10))+1} lines)"
            )
        setattr(scene, "scene_expert_task_spec", self._task_spec.model_dump())
        setattr(scene, "scene_expert_stage", stage)
        self._save_context_bundle(
            stage=stage,
            agent_role="designer",
            event="pre_stage",
            scene=scene,
            prompt=scene.text_description,
        )
        self._trace_logger.save_stage_context(
            stage=stage,
            memory_pack=self._current_memory_pack,
            stage_brief=self._current_stage_brief,
            phase="pre",
        )

    # ------------------------------------------------------------------
    # Post-stage hook: called AFTER the SceneSmith stage agent completes
    # ------------------------------------------------------------------

    def post_stage(self, stage: str, scene: RoomScene, room_dir: Path) -> None:
        """Verify stage output, log trace entry, optionally record to memory.

        Called from _generate_room immediately after the stage's checkpoint is saved.
        Repair is NOT executed here (would require re-running the agent, which is
        complex within _generate_room). Instead, repair instructions are logged for
        the MemoryWriter to learn from.

        Args:
            stage: Completed stage name.
            scene: The RoomScene after stage completion.
            room_dir: Room output directory (for finding scores.yaml).
        """
        console_logger.info(f"[SceneExpert/{self._mode}] post_stage: {stage}")

        # Restore original text_description (keep scene clean for next stage)
        if stage in self._original_text_descriptions:
            scene.text_description = self._original_text_descriptions[stage]

        # Extract lightweight scene state info for rule checks
        scene_state_info = self._extract_scene_state_info_from_scene(scene)

        # Verify stage
        verify_report: StageVerifyReport | None = None
        repair_actions: list[RepairResult] = []
        try:
            verify_start = time.time()
            verify_report = self._stage_verifier.verify(
                stage=stage,
                stage_output_dir=str(room_dir),
                task_spec=self._task_spec,
                stage_brief=self._current_stage_brief,
                scene_state_info=scene_state_info,
            )
            console_logger.info(
                "[SceneExpertTiming] stage=%s module=stage_verifier elapsed=%.2fs",
                stage,
                time.time() - verify_start,
            )
            self._stage_reports.append(verify_report)

            if not verify_report.pass_stage:
                console_logger.warning(
                    f"[SceneExpert] Stage {stage} FAILED verification: "
                    f"issues={[i.issue_type for i in verify_report.issues]}"
                )
                # Log repair decision for trace (actual re-execution not done here)
                decision = self._harness.decide_repair(stage, verify_report)
                if decision.should_repair:
                    repair_result = self._repair_controller.repair(
                        repair_type=decision.strategy,
                        stage=stage,
                        verify_report=verify_report,
                        scene_path=str(room_dir),
                        stage_brief=self._current_stage_brief,
                        task_spec=self._task_spec,
                    )
                    repair_actions.append(repair_result)
                    # Record failure to memory for future runs
                    self._repair_controller.record_failure_to_memory(
                        stage=stage,
                        room_type=self._task_spec.room_type,
                        repair_result=repair_result,
                        verify_report=verify_report,
                        repair_verified=False,  # can't verify without re-running
                    )
            else:
                console_logger.info(f"[SceneExpert] Stage {stage} PASSED verification")

        except Exception as e:
            console_logger.warning(f"[SceneExpert] Verification failed for {stage}: {e}")

        # Log stage trace entry
        elapsed = time.time() - self._stage_start_time
        self._commit_stage_memory(
            stage=stage,
            verify_report=verify_report,
            scene_state_path=str(room_dir),
            repair_actions=repair_actions,
        )
        self._trace_logger.log_stage(
            stage=stage,
            memory_pack=self._current_memory_pack,
            stage_brief=self._current_stage_brief,
            scene_state_path=str(room_dir),
            verify_report=verify_report,
            repair_actions=repair_actions,
            qwen_calls=self._qwen_calls,
            stage_time_sec=round(elapsed, 1),
        )
        self._trace_logger.save_stage_context(
            stage=stage,
            memory_pack=self._current_memory_pack,
            stage_brief=self._current_stage_brief,
            phase="post",
        )
        self._trace_logger.save_stage_visual_manifest(stage, str(room_dir))
        self._completed_stages.append(stage)
        console_logger.info(
            "[SceneExpertTiming] stage=%s module=stage_total elapsed=%.2fs",
            stage,
            elapsed,
        )

    # ------------------------------------------------------------------
    # Finalize: called after all stages complete
    # ------------------------------------------------------------------

    def finalize(self, final_scene_path: str) -> None:
        """Run full verifier, save trace, update memory.

        Called from _generate_single_scene after _run_sequential_room_generation
        returns and before the function exits.

        Args:
            final_scene_path: Path to the final scene output directory.
        """
        console_logger.info(f"[SceneExpert/{self._mode}] finalizing scene {self._scene_id:03d}")
        finalize_start = time.time()

        # Full verifier
        full_report = FullVerifyReport()
        try:
            full_verify_start = time.time()
            full_report = self._full_verifier.verify(
                stage_reports=self._stage_reports,
                final_scene_path=final_scene_path,
            )
            console_logger.info(
                "[SceneExpertTiming] stage=full_scene module=full_verifier elapsed=%.2fs",
                time.time() - full_verify_start,
            )
        except Exception as e:
            console_logger.warning(f"FullVerifier failed: {e}")

        # Save trace
        final_path = Path(final_scene_path)
        combined_path = (
            final_path
            if final_path.name == "combined_house"
            else final_path / "combined_house"
        )
        exports = {
            "scene_dir": final_scene_path,
            "drake": str(combined_path / "house.dmd.yaml"),
            "blend": str(combined_path / "house.blend"),
        }
        trace_dict = self._trace_logger.finalize(
            full_report=full_report,
            exports=exports,
            model=self._qwen_model,
        )
        trace_path = self._trace_logger.save(trace_dict)
        console_logger.info(f"[SceneExpert] Trace saved to {trace_path}")

        # Memory update (skip in harness_only mode)
        if (
            self._memory_writer is not None
            and self._memory_store is not None
            and self._mode in ("harness_memory", "full")
        ):
            try:
                memory_start = time.time()
                trace_summary = self._trace_logger.build_trace_summary()
                related_old_memory = self._format_related_memory_for_writer()
                ops = self._memory_writer.write(
                    trace_summary=trace_summary,
                    full_report=full_report,
                    related_old_memory=related_old_memory,
                )
                self._trace_logger.save_memory_update_ops(ops, full_report)
                self._memory_store.apply_updates(ops)
                console_logger.info(
                    f"[SceneExpert] Memory updated: {len(ops)} ops applied "
                    f"in {time.time() - memory_start:.2f}s"
                )
            except Exception as e:
                console_logger.warning(f"Memory update failed (non-fatal): {e}")
                self._trace_logger.save_memory_update_ops([], full_report)

        console_logger.info(
            f"[SceneExpert] Scene {self._scene_id:03d} complete: "
            f"overall={full_report.overall_score:.2f} "
            f"pass={'YES' if full_report.pass_scene else 'NO'} "
            f"mode={self._mode}"
        )
        console_logger.info(
            "[SceneExpertTiming] stage=full_scene module=finalize_total elapsed=%.2fs",
            time.time() - finalize_start,
        )

    def save_partial_trace(self, error: str = "") -> None:
        """Persist a partial trace from an exception path."""
        try:
            path = self._trace_logger.save_partial(status="failed", error=error)
            console_logger.info(f"[SceneExpert] Partial trace saved to {path}")
        except Exception as save_error:
            console_logger.warning(
                f"[SceneExpert] Failed to save partial trace: {save_error}"
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _initial_completed_stages(self, start_stage: str) -> list[str]:
        """Return the stage-order prefix already satisfied by a resumed run."""
        if start_stage not in STAGE_ORDER:
            return []
        return STAGE_ORDER[: STAGE_ORDER.index(start_stage)]

    def _validate_stage_transition(self, stage: str) -> None:
        """Enforce Harness FSM order while tolerating sequential multi-room runs."""
        try:
            self._harness.validate_stage_order(self._completed_stages, stage)
            return
        except ValueError:
            # _generate_room runs a full room pipeline per room. When a new room
            # starts, the same per-scene hook sees the start stage again. Reset the
            # FSM baseline for that room instead of treating it as an LLM skip.
            if stage == self._room_start_stage and self._completed_stages:
                console_logger.info(
                    "[SceneExpert] Resetting Harness stage-order baseline for "
                    f"new room at stage '{stage}'"
                )
                self._completed_stages = list(self._room_stage_order_baseline)
                self._harness.validate_stage_order(self._completed_stages, stage)
                return
            raise

    def _build_scene_state_summary(self) -> str:
        """Build a text summary of completed stages for the GlobalPlanner."""
        if not self._completed_stages:
            return "Empty scene — no objects placed yet."
        return "Completed stages: " + ", ".join(self._completed_stages)

    def _extract_floor_plan_state_info(self, scene_dir: Path) -> dict:
        """Extract lightweight floor-plan facts for rule-based verification."""
        layout_path = scene_dir / "house_layout.json"
        if not layout_path.exists():
            return {"layout_exists": False, "room_count": 0, "rooms": []}
        try:
            with layout_path.open() as f:
                data = json.load(f)
        except Exception as e:
            return {
                "layout_exists": False,
                "room_count": 0,
                "rooms": [],
                "layout_error": str(e),
            }

        rooms = data.get("room_specs") or data.get("rooms") or []
        if isinstance(rooms, dict):
            rooms = list(rooms.values())
        if not isinstance(rooms, list):
            rooms = []
        return {
            "layout_exists": True,
            "room_count": len(rooms),
            "rooms": rooms,
        }

    def _format_related_memory_for_writer(self) -> str:
        """Build compact related-memory context for MemoryWriter deduplication."""
        if self._retriever is None:
            return ""

        lines: list[str] = []
        seen: set[str] = set()
        for stage in STAGE_ORDER:
            try:
                pack = self._retriever.retrieve(self._task_spec, stage)
            except Exception:
                continue
            for item in (
                pack.success_hints
                + pack.failure_hints
                + pack.skill_texts
                + ([pack.placement_reference] if pack.placement_reference else [])
            ):
                text = item.strip()
                if not text or text in seen:
                    continue
                seen.add(text)
                lines.append(f"- [{stage}] {text}")
        return "\n".join(lines[:24])

    def _extract_scene_state_info_from_scene(self, scene: RoomScene) -> dict:
        """Extract object names from the live RoomScene for rule-based checks."""
        try:
            names = [
                obj.name
                for obj in scene.objects.values()
                if hasattr(obj, "name") and obj.name
            ]
            return {"object_names": names}
        except Exception:
            return {"object_names": []}


# ------------------------------------------------------------------
# Factory function
# ------------------------------------------------------------------

def build_hook_runner(
    prompt: str,
    scene_id: int,
    output_dir: Path,
    cfg_dict: dict,
) -> SceneExpertHookRunner | None:
    """Build a SceneExpertHookRunner from config.

    Returns None if scene_expert is disabled (ablation mode "disabled" or
    scene_expert config block missing).

    Args:
        prompt: Raw scene prompt.
        scene_id: Scene index.
        output_dir: Base experiment output directory.
        cfg_dict: Full Hydra config as plain dict.

    Returns:
        Configured SceneExpertHookRunner, or None if disabled.
    """
    # Ablation configs set experiment.scene_expert. The root scene_expert block is
    # a disabled default and also carries memory sub-config defaults.
    root_se_cfg = cfg_dict.get("scene_expert", {})
    exp_se_cfg = cfg_dict.get("experiment", {}).get("scene_expert")
    se_cfg = exp_se_cfg or root_se_cfg
    if not se_cfg:
        return None
    memory_cfg = _deep_merge_dicts(
        root_se_cfg.get("memory", {}),
        se_cfg.get("memory", {}),
    )

    mode = se_cfg.get("mode", "disabled")
    if mode == "disabled" or not se_cfg.get("enabled", False):
        return None

    if mode not in ABLATION_MODES:
        console_logger.warning(
            f"Unknown scene_expert.mode={mode!r}. "
            f"Valid: {sorted(ABLATION_MODES)}. Disabling SceneExpert."
        )
        return None

    console_logger.info(f"[SceneExpert] Building hook runner (mode={mode})")

    # Model / API settings (shared with SceneSmith agents)
    model = cfg_dict.get("furniture_agent", {}).get("openai", {}).get(
        "model",
        cfg_dict.get("llm", {}).get("model_id", "Qwen/Qwen3.5-35B-A3B"),
    )
    api_base = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
    api_key = os.environ.get("OPENAI_API_KEY", "dummy")

    # Memory system (skip if harness_only)
    memory_dir = memory_cfg.get(
        "dir",
        cfg_dict.get("paths", {}).get("memory_dir", "outputs/scene_expert_memory"),
    )
    use_memory = mode in ("harness_memory", "full")
    scene_debug_dir = output_dir / f"scene_{scene_id:03d}" / "scene_expert"
    os.environ["SCENEEXPERT_LLM_DEBUG_PATH"] = str(
        scene_debug_dir / "timing" / "scene_expert_llm_calls.jsonl"
    )
    if not use_memory:
        os.environ.pop("SCENEEXPERT_ACTIVE_MEMORY_BANK_DIR", None)

    memory_store: FastMemoryStore | None = None
    retriever: Any | None = None
    memory_writer: MemoryWriter | None = None

    if use_memory:
        ret_cfg = memory_cfg.get("retrieval", {})
        memory_store = FastMemoryStore(memory_dir)
        os.environ["SCENEEXPERT_ACTIVE_MEMORY_BANK_DIR"] = str(memory_dir)
        retriever_type = memory_cfg.get("retriever_type", "lexical")
        if retriever_type == "hybrid":
            retriever = _build_hybrid_retriever(
                memory_store=memory_store,
                memory_dir=memory_dir,
                memory_cfg=memory_cfg,
                ret_cfg=ret_cfg,
                timing_path=scene_debug_dir / "timing" / "memory_retrieval.jsonl",
            )
        elif retriever_type == "lexical":
            retriever = MemoryRetriever(
                store=memory_store,
                max_success=_cfg_int(ret_cfg.get("max_success_cases"), 3),
                max_failure=_cfg_int(ret_cfg.get("max_failure_cases"), 3),
                max_skills=_cfg_int(ret_cfg.get("max_skills"), 2),
            )
        else:
            raise ValueError(
                f"Unsupported SceneExpert memory retriever_type={retriever_type!r}. "
                "Use 'lexical' or 'hybrid'."
            )
        memory_writer = MemoryWriter(
            model=model,
            api_base_url=api_base,
            api_key=api_key,
            debug_dir=scene_debug_dir / "memory",
        )

    # Verifier thresholds
    ver_cfg = se_cfg.get("verifier", {})
    stage_verifier = StageVerifier(
        pass_threshold=ver_cfg.get("stage_pass_threshold", 0.6)
    )
    full_verifier = FullVerifier(
        pass_threshold=ver_cfg.get("full_pass_threshold", 0.7)
    )

    # Build TaskCompiler and compile the task spec
    from omegaconf import OmegaConf
    task_compiler = TaskCompiler(model=model, api_base_url=api_base, api_key=api_key)
    try:
        task_spec = task_compiler.compile(prompt)
    except Exception as e:
        console_logger.warning(
            f"TaskCompiler failed, using fallback task spec from prompt text: {e}"
        )
        from scenesmith.scene_expert.task_compiler import _fallback_spec_from_prompt
        task_spec = _fallback_spec_from_prompt(prompt)

    # Harness (always active when mode != "disabled")
    from omegaconf import OmegaConf
    se_omega = OmegaConf.create(se_cfg)
    harness = Harness(se_omega)
    harness.reset()

    global_planner = GlobalPlanner(
        model=model, api_base_url=api_base, api_key=api_key
    )
    repair_controller = RepairController(memory_store=memory_store)
    start_stage = (
        cfg_dict.get("experiment", {})
        .get("pipeline", {})
        .get("start_stage", "floor_plan")
    )

    return SceneExpertHookRunner(
        prompt=prompt,
        scene_id=scene_id,
        output_dir=output_dir,
        mode=mode,
        task_spec=task_spec,
        harness=harness,
        global_planner=global_planner,
        retriever=retriever,
        stage_verifier=stage_verifier,
        full_verifier=full_verifier,
        repair_controller=repair_controller,
        memory_writer=memory_writer,
        memory_store=memory_store,
        qwen_model=model,
        experiment_name=cfg_dict.get("name", ""),
        config_hash=_stable_config_hash(cfg_dict),
        start_stage=start_stage,
    )
