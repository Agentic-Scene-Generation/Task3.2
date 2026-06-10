"""SceneExpertPipeline: orchestrates the full SceneExpert MVP online loop.

Architecture: runs SceneSmith stage-by-stage (using start_stage/stop_stage),
inserting SceneExpert's pre/post hooks between each stage:

  For each stage:
    Pre:  Memory retrieval → StageBrief generation → prompt injection
    In:   SceneSmith stage execution (unchanged)
    Post: Stage verification → Repair (if needed) → Trace logging

After all stages:
  Full verification → Memory update → Save trace

StageBrief injection: patches the prompt string in cfg_dict before passing
it to SceneSmith's _generate_single_scene. The StageBrief is formatted as
additional context appended to the original room prompt.
"""

from __future__ import annotations

import copy
import logging
import os
import time
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

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
    StageBrief,
    StageVerifyReport,
)
from scenesmith.scene_expert.task_compiler import TaskCompiler
from scenesmith.scene_expert.trace_logger import TraceLogger
from scenesmith.scene_expert.verifier import FullVerifier, StageVerifier

console_logger = logging.getLogger(__name__)


def _get_stage_output_dir(scene_dir: Path, stage: str) -> str:
    """Return the expected SceneSmith output directory for a stage."""
    mapping = {
        "floor_plan": str(scene_dir),
        "furniture": str(scene_dir / "room_main"),
        "wall_mounted": str(scene_dir / "room_main"),
        "ceiling_mounted": str(scene_dir / "room_main"),
        "manipuland": str(scene_dir / "room_main"),
    }
    return mapping.get(stage, str(scene_dir))


def _extract_scene_state_info(scene_dir: Path, stage: str) -> dict:
    """Extract lightweight scene state info from SceneSmith output directory.

    Reads object names from saved scene JSON files for rule-based verification.

    Returns:
        Dict with "object_names" key (list of object name strings).
    """
    import json as _json

    # Look for scene state JSON files
    state_files = {
        "furniture": "scene_after_furniture",
        "wall_mounted": "scene_after_wall_objects",
        "ceiling_mounted": "scene_after_ceiling_objects",
        "manipuland": "final_scene",
    }

    state_name = state_files.get(stage, "")
    if not state_name:
        return {"object_names": []}

    # Search common save paths
    candidates = [
        scene_dir / "room_main" / f"{state_name}.json",
        scene_dir / f"{state_name}.json",
        scene_dir / "room_main" / state_name / "scene.json",
    ]

    for candidate in candidates:
        if candidate.exists():
            try:
                with candidate.open() as f:
                    data = _json.load(f)
                # Extract object names from common schema patterns
                objects = data.get("objects", data.get("scene_objects", []))
                if isinstance(objects, list):
                    names = []
                    for obj in objects:
                        if isinstance(obj, dict):
                            name = obj.get("name", obj.get("object_name", obj.get("id", "")))
                            if name:
                                names.append(str(name))
                    return {"object_names": names}
            except Exception as e:
                console_logger.debug(f"Could not parse scene state from {candidate}: {e}")

    return {"object_names": []}


def _build_enhanced_prompt(original_prompt: str, stage_brief: StageBrief) -> str:
    """Append StageBrief as structured context to the original room prompt.

    The enhanced prompt is what SceneSmith's stage agents will see.
    """
    brief_text = stage_brief.to_injection_text()
    return f"{original_prompt}\n\n{brief_text}"


class SceneExpertPipeline:
    """SceneExpert MVP online closed-loop pipeline.

    Wraps SceneSmith's stage-by-stage execution with expert pre/post hooks.
    Can be used as a drop-in replacement for direct IndoorSceneGeneration use.

    Usage:
        pipeline = SceneExpertPipeline(cfg)
        pipeline.run_scene(
            prompt="A bedroom with ...",
            scene_id=0,
            output_dir=Path("outputs/2026-05-18/..."),
            cfg_dict=OmegaConf.to_container(cfg, resolve=True),
        )
    """

    def __init__(self, cfg: DictConfig) -> None:
        self._cfg = cfg
        se_cfg = getattr(cfg, "scene_expert", None)

        # Model settings
        model = cfg.furniture_agent.openai.model
        api_base = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
        api_key = os.environ.get("OPENAI_API_KEY", "dummy")

        # Memory system
        memory_dir = (
            se_cfg.memory.dir if se_cfg and hasattr(se_cfg, "memory") else "outputs/scene_expert_memory"
        )
        self._memory_store = FastMemoryStore(memory_dir)
        max_success = getattr(getattr(se_cfg, "memory", None), "retrieval", None)
        self._retriever = MemoryRetriever(
            store=self._memory_store,
            max_success=3,
            max_failure=3,
            max_skills=2,
        )

        # Qwen3 modules
        self._task_compiler = TaskCompiler(
            model=model, api_base_url=api_base, api_key=api_key
        )
        self._global_planner = GlobalPlanner(
            model=model, api_base_url=api_base, api_key=api_key
        )
        self._memory_writer = MemoryWriter(
            model=model, api_base_url=api_base, api_key=api_key
        )

        # Deterministic modules
        self._harness = Harness(se_cfg if se_cfg else DictConfig({}))
        self._stage_verifier = StageVerifier(
            pass_threshold=(
                se_cfg.verifier.stage_pass_threshold
                if se_cfg and hasattr(se_cfg, "verifier")
                else 0.6
            )
        )
        self._full_verifier = FullVerifier(
            pass_threshold=(
                se_cfg.verifier.full_pass_threshold
                if se_cfg and hasattr(se_cfg, "verifier")
                else 0.7
            )
        )
        self._repair_controller = RepairController(memory_store=self._memory_store)

        console_logger.info("SceneExpertPipeline initialized")

    def run_scene(
        self,
        prompt: str,
        scene_id: int,
        output_dir: Path,
        cfg_dict: dict,
    ) -> tuple[str, dict, FullVerifyReport]:
        """Run the full SceneExpert pipeline for one scene.

        Args:
            prompt: Natural-language scene description.
            scene_id: Integer scene index (used for directory naming).
            output_dir: Base experiment output directory.
            cfg_dict: SceneSmith config as plain dict (from OmegaConf.to_container).

        Returns:
            (scene_path, trace_dict, full_report) tuple.
        """
        from scenesmith.experiments.indoor_scene_generation import (
            _generate_single_scene,
            PIPELINE_STAGES,
        )

        scene_dir = output_dir / f"scene_{scene_id:03d}"
        scene_dir.mkdir(parents=True, exist_ok=True)

        trace_logger = TraceLogger(
            output_dir=str(output_dir), scene_index=scene_id, prompt=prompt
        )
        self._harness.reset()

        qwen_call_count = 0

        # --- Step 1: TaskCompiler ---
        console_logger.info(f"[SceneExpert] scene_{scene_id:03d}: compiling task spec")
        try:
            task_spec = self._task_compiler.compile(prompt)
            qwen_call_count += 1
        except Exception as e:
            console_logger.warning(f"TaskCompiler failed, continuing without task spec: {e}")
            from scenesmith.scene_expert.schemas import SceneTaskSpec
            task_spec = SceneTaskSpec(room_type="room", style="standard")

        # Determine stage range from config
        pipeline_cfg = cfg_dict.get("experiment", {}).get("pipeline", {})
        start_stage = pipeline_cfg.get("start_stage", "floor_plan")
        stop_stage = pipeline_cfg.get("stop_stage", "manipuland")

        start_idx = PIPELINE_STAGES.index(start_stage)
        stop_idx = PIPELINE_STAGES.index(stop_stage)
        stages_to_run = PIPELINE_STAGES[start_idx: stop_idx + 1]

        # --- Step 2: Stage-by-stage loop ---
        stage_reports: list[StageVerifyReport] = []
        completed_stages: list[str] = []

        for stage in stages_to_run:
            stage_start = time.time()
            console_logger.info(f"[SceneExpert] === Stage: {stage} ===")

            # 2a. Memory retrieval
            memory_pack: MemoryPack = self._retriever.retrieve(task_spec, stage)

            # 2b. Build harness context
            context = self._harness.build_context(
                stage=stage,
                task_spec=task_spec,
                memory_pack=memory_pack,
            )

            # 2c. Generate StageBrief (skip for floor_plan in MVP — geometry is less memory-sensitive)
            stage_brief: StageBrief | None = None
            scene_state_summary = self._build_scene_state_summary(scene_dir, completed_stages)

            if stage != "floor_plan":
                try:
                    context = self._harness.build_context(
                        stage=stage, task_spec=task_spec, memory_pack=memory_pack
                    )
                    stage_brief = self._global_planner.generate_stage_brief(
                        context=context, scene_state_summary=scene_state_summary
                    )
                    qwen_call_count += 1
                    context = self._harness.build_context(
                        stage=stage,
                        task_spec=task_spec,
                        memory_pack=memory_pack,
                        stage_brief=stage_brief,
                    )
                except Exception as e:
                    console_logger.warning(f"GlobalPlanner failed for {stage}: {e}")

            # 2d. Inject StageBrief into prompt
            enhanced_prompt = prompt
            if stage_brief:
                enhanced_prompt = _build_enhanced_prompt(prompt, stage_brief)
                console_logger.info(
                    f"[SceneExpert] Injected StageBrief for {stage}: "
                    f"{len(stage_brief.constraints_for_designer)} constraints"
                )

            # 2e. Execute SceneSmith for this stage only
            stage_cfg_dict = self._build_stage_cfg_dict(
                cfg_dict=cfg_dict,
                stage=stage,
                enhanced_prompt=enhanced_prompt,
            )

            try:
                _generate_single_scene(
                    prompt=enhanced_prompt,
                    scene_id=scene_id,
                    output_dir=output_dir,
                    cfg_dict=stage_cfg_dict,
                )
            except Exception as e:
                console_logger.error(f"[SceneExpert] SceneSmith stage {stage} failed: {e}")
                # Log partial trace and continue to next stage
                trace_logger.log_stage(
                    stage=stage,
                    memory_pack=memory_pack,
                    stage_brief=stage_brief,
                    scene_state_path=_get_stage_output_dir(scene_dir, stage),
                    verify_report=None,
                    repair_actions=[],
                    qwen_calls=qwen_call_count,
                )
                completed_stages.append(stage)
                continue

            # 2f. Verify stage
            stage_output_dir = _get_stage_output_dir(scene_dir, stage)
            scene_state_info = _extract_scene_state_info(scene_dir, stage)
            verify_report = self._stage_verifier.verify(
                stage=stage,
                stage_output_dir=stage_output_dir,
                task_spec=task_spec,
                stage_brief=stage_brief,
                scene_state_info=scene_state_info,
            )
            stage_reports.append(verify_report)

            # 2g. Repair loop
            repair_actions: list[RepairResult] = []
            repair_attempt = 0

            while not verify_report.pass_stage:
                decision = self._harness.decide_repair(stage, verify_report)
                if not decision.should_repair:
                    console_logger.info(
                        f"[SceneExpert] Skipping repair for {stage}: {decision.reason}"
                    )
                    break

                console_logger.info(
                    f"[SceneExpert] Repair attempt {repair_attempt+1} for {stage}: "
                    f"{decision.strategy}"
                )

                repair_result = self._repair_controller.repair(
                    repair_type=decision.strategy,
                    stage=stage,
                    verify_report=verify_report,
                    scene_path=stage_output_dir,
                    stage_brief=stage_brief,
                    task_spec=task_spec,
                )
                repair_actions.append(repair_result)

                if decision.strategy == "stage_regeneration":
                    # Re-run this stage with updated prompt incorporating repair action
                    repair_prompt = (
                        enhanced_prompt + f"\n\n[REPAIR INSTRUCTION]\n{repair_result.repair_action}"
                    )
                    repair_cfg_dict = self._build_stage_cfg_dict(
                        cfg_dict=cfg_dict,
                        stage=stage,
                        enhanced_prompt=repair_prompt,
                    )
                    try:
                        _generate_single_scene(
                            prompt=repair_prompt,
                            scene_id=scene_id,
                            output_dir=output_dir,
                            cfg_dict=repair_cfg_dict,
                        )
                        # Re-verify after regeneration
                        verify_report = self._stage_verifier.verify(
                            stage=stage,
                            stage_output_dir=stage_output_dir,
                            task_spec=task_spec,
                            stage_brief=stage_brief,
                            scene_state_info=_extract_scene_state_info(scene_dir, stage),
                        )
                        repair_result.repair_verified = verify_report.pass_stage
                        stage_reports[-1] = verify_report  # update latest
                    except Exception as e:
                        console_logger.error(f"Stage regeneration failed for {stage}: {e}")

                elif decision.strategy == "local_repair":
                    # Local repair: the instruction will be passed in on next designer call
                    # For MVP, we re-run stage with the repair instruction appended
                    repair_prompt = (
                        enhanced_prompt + f"\n\n[REPAIR INSTRUCTION]\n{repair_result.repair_action}"
                    )
                    repair_cfg_dict = self._build_stage_cfg_dict(
                        cfg_dict=cfg_dict,
                        stage=stage,
                        enhanced_prompt=repair_prompt,
                    )
                    try:
                        _generate_single_scene(
                            prompt=repair_prompt,
                            scene_id=scene_id,
                            output_dir=output_dir,
                            cfg_dict=repair_cfg_dict,
                        )
                        verify_report = self._stage_verifier.verify(
                            stage=stage,
                            stage_output_dir=stage_output_dir,
                            task_spec=task_spec,
                            stage_brief=stage_brief,
                            scene_state_info=_extract_scene_state_info(scene_dir, stage),
                        )
                        repair_result.repair_verified = verify_report.pass_stage
                        stage_reports[-1] = verify_report
                    except Exception as e:
                        console_logger.error(f"Local repair execution failed for {stage}: {e}")

                # Record failure to memory
                self._repair_controller.record_failure_to_memory(
                    stage=stage,
                    room_type=task_spec.room_type,
                    repair_result=repair_result,
                    verify_report=verify_report,
                    repair_verified=repair_result.repair_verified,
                )

                repair_attempt += 1
                qwen_call_count += 1

            # 2h. Log stage trace
            stage_time = time.time() - stage_start
            trace_logger.log_stage(
                stage=stage,
                memory_pack=memory_pack,
                stage_brief=stage_brief,
                scene_state_path=stage_output_dir,
                verify_report=verify_report if stage_reports else None,
                repair_actions=repair_actions,
                qwen_calls=qwen_call_count,
            )
            completed_stages.append(stage)

        # --- Step 3: Full verifier ---
        console_logger.info("[SceneExpert] Running full scene verification")
        full_report = self._full_verifier.verify(stage_reports=stage_reports)

        # --- Step 4: Export paths ---
        final_scene_path = str(scene_dir)
        exports = {
            "scene_dir": final_scene_path,
            "drake": str(scene_dir / "combined_house" / "house.dmd.yaml"),
            "blend": str(scene_dir / "combined_house" / "house.blend"),
        }

        # --- Step 5: Finalize and save trace ---
        trace_summary = trace_logger.build_trace_summary()
        trace_dict = trace_logger.finalize(
            full_report=full_report,
            exports=exports,
            model=self._cfg.furniture_agent.openai.model,
        )
        trace_path = trace_logger.save(trace_dict)
        console_logger.info(f"[SceneExpert] Trace saved to {trace_path}")

        # --- Step 6: Memory update ---
        console_logger.info("[SceneExpert] Updating fast memory")
        try:
            memory_ops = self._memory_writer.write(
                trace_summary=trace_summary,
                full_report=full_report,
            )
            qwen_call_count += 1
            self._memory_store.apply_updates(memory_ops)
        except Exception as e:
            console_logger.warning(f"Memory update failed (non-fatal): {e}")

        console_logger.info(
            f"[SceneExpert] scene_{scene_id:03d} complete: "
            f"overall={full_report.overall_score:.2f} "
            f"pass={'YES' if full_report.pass_scene else 'NO'} "
            f"qwen_calls={qwen_call_count}"
        )

        return final_scene_path, trace_dict, full_report

    def _build_stage_cfg_dict(
        self, cfg_dict: dict, stage: str, enhanced_prompt: str
    ) -> dict:
        """Build a cfg_dict for running a single stage.

        Sets start_stage=stop_stage=stage so SceneSmith runs only that stage.
        Does NOT set resume_from_path (pipeline manages checkpoints by running
        stages in sequence — each stage's output is consumed by the next).
        """
        stage_cfg = copy.deepcopy(cfg_dict)
        stage_cfg["experiment"]["pipeline"]["start_stage"] = stage
        stage_cfg["experiment"]["pipeline"]["stop_stage"] = stage
        # Keep resume_from_path as-is (None for normal sequential execution)
        return stage_cfg

    def _build_scene_state_summary(self, scene_dir: Path, completed_stages: list[str]) -> str:
        """Build a text summary of the current scene state from completed stages."""
        if not completed_stages:
            return "Empty scene — no objects placed yet."

        parts = []
        for stage in completed_stages:
            info = _extract_scene_state_info(scene_dir, stage)
            objects = info.get("object_names", [])
            if objects:
                parts.append(f"{stage}: {', '.join(objects[:10])}")
            else:
                parts.append(f"{stage}: completed (object names unavailable)")

        return "Current scene state:\n" + "\n".join(f"  - {p}" for p in parts)
