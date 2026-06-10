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
from pathlib import Path

from omegaconf import DictConfig

from scenesmith.agent_utils.room import RoomScene
from scenesmith.scene_expert.global_planner import GlobalPlanner
from scenesmith.scene_expert.harness import STAGE_ORDER, Harness
from scenesmith.scene_expert.memory.retriever import MemoryRetriever
from scenesmith.scene_expert.memory.store import FastMemoryStore
from scenesmith.scene_expert.memory.writer import MemoryWriter
from scenesmith.scene_expert.repair_controller import RepairController
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
        retriever: MemoryRetriever | None,
        stage_verifier: StageVerifier,
        full_verifier: FullVerifier,
        repair_controller: RepairController,
        memory_writer: MemoryWriter | None,
        memory_store: FastMemoryStore | None,
        qwen_model: str,
    ) -> None:
        self._prompt = prompt
        self._scene_id = scene_id
        self._output_dir = output_dir
        self._mode = mode

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

        self._trace_logger = TraceLogger(
            output_dir=str(output_dir), scene_index=scene_id, prompt=prompt
        )
        self._stage_reports: list[StageVerifyReport] = []
        self._completed_stages: list[str] = []
        self._qwen_calls = 0

        # Current stage state (populated in pre_stage, consumed in post_stage)
        self._current_stage: str = ""
        self._current_memory_pack: MemoryPack = _empty_memory_pack()
        self._current_stage_brief: StageBrief | None = None
        self._stage_start_time: float = 0.0
        # Original text_description per stage (so we can restore if needed)
        self._original_text_descriptions: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Pre-stage hook: called BEFORE the SceneSmith stage agent runs
    # ------------------------------------------------------------------

    def pre_stage(self, stage: str, scene: RoomScene) -> None:
        """Retrieve memory, generate StageBrief, inject into scene.text_description.

        Called from _generate_room immediately before each stage's agent is built.

        Args:
            stage: Current stage name (e.g., "furniture").
            scene: The RoomScene that will be passed to the stage agent.
        """
        console_logger.info(f"[SceneExpert/{self._mode}] pre_stage: {stage}")
        self._current_stage = stage
        self._stage_start_time = time.time()
        self._qwen_calls = 0

        # Save original text_description for restoration after stage
        self._original_text_descriptions[stage] = scene.text_description

        # --- Step 1: Memory retrieval (skip in harness_only mode) ---
        if self._retriever is not None and self._mode in ("harness_memory", "full"):
            try:
                self._current_memory_pack = self._retriever.retrieve(
                    self._task_spec, stage
                )
                n_hints = (
                    len(self._current_memory_pack.success_hints)
                    + len(self._current_memory_pack.failure_hints)
                )
                console_logger.info(
                    f"[SceneExpert] Memory retrieved for {stage}: "
                    f"{n_hints} hints, {len(self._current_memory_pack.skill_texts)} skills"
                )
            except Exception as e:
                console_logger.warning(f"Memory retrieval failed for {stage}: {e}")
                self._current_memory_pack = _empty_memory_pack()
        else:
            self._current_memory_pack = _empty_memory_pack()

        # --- Step 2: Global Planner → StageBrief (skip for floor_plan in MVP) ---
        self._current_stage_brief = None
        if stage != "floor_plan" and self._mode in ("harness_only", "harness_memory", "full"):
            try:
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
                self._qwen_calls += 1
                console_logger.info(
                    f"[SceneExpert] StageBrief generated for {stage}: "
                    f"{len(self._current_stage_brief.constraints_for_designer)} constraints"
                )
            except Exception as e:
                console_logger.warning(
                    f"GlobalPlanner failed for {stage}, running without StageBrief: {e}"
                )

        # --- Step 3: Inject StageBrief into scene.text_description ---
        if self._current_stage_brief is not None:
            enhanced = (
                scene.text_description
                + "\n\n"
                + self._current_stage_brief.to_injection_text()
            )
            scene.text_description = enhanced
            console_logger.debug(
                f"[SceneExpert] Injected StageBrief into scene.text_description for {stage}"
            )

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
            verify_report = self._stage_verifier.verify(
                stage=stage,
                stage_output_dir=str(room_dir),
                task_spec=self._task_spec,
                stage_brief=self._current_stage_brief,
                scene_state_info=scene_state_info,
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
        self._trace_logger.log_stage(
            stage=stage,
            memory_pack=self._current_memory_pack,
            stage_brief=self._current_stage_brief,
            scene_state_path=str(room_dir),
            verify_report=verify_report,
            repair_actions=repair_actions,
            qwen_calls=self._qwen_calls,
        )
        self._completed_stages.append(stage)
        console_logger.debug(
            f"[SceneExpert] Logged trace for {stage} in {elapsed:.1f}s"
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

        # Full verifier
        full_report = FullVerifyReport()
        try:
            full_report = self._full_verifier.verify(
                stage_reports=self._stage_reports,
                final_scene_path=final_scene_path,
            )
        except Exception as e:
            console_logger.warning(f"FullVerifier failed: {e}")

        # Save trace
        exports = {
            "scene_dir": final_scene_path,
            "drake": str(Path(final_scene_path) / "combined_house" / "house.dmd.yaml"),
            "blend": str(Path(final_scene_path) / "combined_house" / "house.blend"),
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
                trace_summary = self._trace_logger.build_trace_summary()
                ops = self._memory_writer.write(
                    trace_summary=trace_summary,
                    full_report=full_report,
                )
                self._memory_store.apply_updates(ops)
                console_logger.info(
                    f"[SceneExpert] Memory updated: {len(ops)} ops applied"
                )
            except Exception as e:
                console_logger.warning(f"Memory update failed (non-fatal): {e}")

        console_logger.info(
            f"[SceneExpert] Scene {self._scene_id:03d} complete: "
            f"overall={full_report.overall_score:.2f} "
            f"pass={'YES' if full_report.pass_scene else 'NO'} "
            f"mode={self._mode}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_scene_state_summary(self) -> str:
        """Build a text summary of completed stages for the GlobalPlanner."""
        if not self._completed_stages:
            return "Empty scene — no objects placed yet."
        return "Completed stages: " + ", ".join(self._completed_stages)

    def _extract_scene_state_info_from_scene(self, scene: RoomScene) -> dict:
        """Extract object names from the live RoomScene for rule-based checks."""
        try:
            names = [
                obj.name
                for obj in scene.objects
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
    # scene_expert config lives under cfg.experiment (set by ablation yamls), not at root.
    se_cfg = cfg_dict.get("experiment", {}).get("scene_expert", {})
    if not se_cfg:
        return None

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
        "model", "Qwen/Qwen3.5-35B-A3B"
    )
    api_base = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
    api_key = os.environ.get("OPENAI_API_KEY", "dummy")

    # Memory system (skip if harness_only)
    memory_dir = se_cfg.get("memory", {}).get("dir", "outputs/scene_expert_memory")
    use_memory = mode in ("harness_memory", "full")

    memory_store: FastMemoryStore | None = None
    retriever: MemoryRetriever | None = None
    memory_writer: MemoryWriter | None = None

    if use_memory:
        ret_cfg = se_cfg.get("memory", {}).get("retrieval", {})
        memory_store = FastMemoryStore(memory_dir)
        retriever = MemoryRetriever(
            store=memory_store,
            max_success=ret_cfg.get("max_success_cases", 3),
            max_failure=ret_cfg.get("max_failure_cases", 3),
            max_skills=ret_cfg.get("max_skills", 2),
        )
        memory_writer = MemoryWriter(
            model=model, api_base_url=api_base, api_key=api_key
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
    )
