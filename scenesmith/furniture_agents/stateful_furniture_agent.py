"""
Stateful furniture agent with natural conversation between persistent agents.

This module implements a furniture placement workflow using persistent
SQLiteSession agents that maintain conversation memory across interactions.
"""

import json
import logging
import math
import re
import shutil
import time

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from agents import Agent, FunctionTool, Runner, RunResult
from agents.exceptions import MaxTurnsExceeded
from omegaconf import DictConfig
from pydrake.all import RigidTransform, RollPitchYaw

from scenesmith.agent_utils.asset_manager import AssetGenerationRequest
from scenesmith.agent_utils.bailian_image_editor import BailianContextImageEditor
from scenesmith.agent_utils.base_stateful_agent import (
    BaseStatefulAgent,
    HardStateEvaluation,
    log_agent_usage,
)
from scenesmith.agent_utils.context_image_generation import (
    OpenAICompatibleContextImageEditor,
)
from scenesmith.agent_utils.context_image_quality import (
    ContextImageQualityEvaluator,
    ContextImageQualityGateConfig,
    evaluate_context_image_deterministic,
    file_sha256,
    write_context_image_quality_report,
)
from scenesmith.agent_utils.furniture_image_layout import (
    LAYOUT_ARTIFACT_NAME,
    build_grounded_furniture_layout_reference,
)
from scenesmith.agent_utils.furniture_layout_constraint_contract import (
    build_furniture_layout_constraint_contract,
    format_furniture_layout_constraint_contract,
)
from scenesmith.agent_utils.furniture_layout_planning import (
    build_bedroom_anchor_plan,
    format_bedroom_anchor_guidance,
    is_bedroom_scene,
)
from scenesmith.agent_utils.furniture_placement_order import (
    build_furniture_placement_order_reference,
)
from scenesmith.agent_utils.furniture_safety import (
    furniture_category_satisfies,
    furniture_object_category_matches,
    infer_furniture_category,
    infer_furniture_object_category,
)
from scenesmith.agent_utils.mesh_physics_analyzer import MeshPhysicsAnalysis
from scenesmith.agent_utils.okcodex_image_editor import OKCodexContextImageEditor
from scenesmith.agent_utils.openrouter_image_editor import (
    OpenRouterContextImageEditor,
)
from scenesmith.agent_utils.placement_noise import PlacementNoiseMode
from scenesmith.agent_utils.reachability import (
    compute_reachability,
    format_reachability_for_critic,
)
from scenesmith.agent_utils.room import (
    AgentType,
    ObjectType,
    RoomScene,
    SceneObject,
    copy_scene_object_with_new_pose,
)
from scenesmith.agent_utils.scoring import (
    FurnitureCritiqueWithScores,
    log_agent_response,
)
from scenesmith.agent_utils.sdf_generator import generate_drake_sdf
from scenesmith.agent_utils.seating_orientation_guard import (
    align_seating_to_nearest_surface,
)
from scenesmith.agent_utils.stage_placement_order_config import (
    append_placement_order_reference,
)
from scenesmith.agent_utils.thin_covering_generator import generate_thin_covering_sdf
from scenesmith.agent_utils.workflow_tools import WorkflowTools
from scenesmith.floor_plan_agents.tools.polygon_geometry import (
    room_geometry_covers_object,
)
from scenesmith.furniture_agents.base_furniture_agent import BaseFurnitureAgent
from scenesmith.furniture_agents.tools.furniture_tools import FurnitureTools
from scenesmith.furniture_agents.tools.scene_tools import SceneTools
from scenesmith.furniture_agents.tools.vision_tools import VisionTools
from scenesmith.prompts.registry import FurnitureAgentPrompts
from scenesmith.scene_expert.repair_taxonomy import FailureCategory, build_repair_plan
from scenesmith.scenebenchmark_critic.api import seating_orientation_targets
from scenesmith.scenebenchmark_critic.config import critic_config_from_any
from scenesmith.scenebenchmark_critic.furniture_relation_repair import (
    improve_furniture_relations,
)
from scenesmith.scenebenchmark_critic.intent_contract import constraint_mode
from scenesmith.utils.logging import BaseLogger

console_logger = logging.getLogger(__name__)


REPAIR_ASSET_SPECS: dict[str, tuple[str, list[float]]] = {
    "bed": (
        "Compact standard double bed with headboard, mattress, pillows, and bedding",
        [1.60, 2.05, 0.80],
    ),
    "twin_bed": (
        "Compact single twin bed with mattress and headboard",
        [1.0, 2.0, 0.75],
    ),
    "nightstand": ("Compact bedside nightstand with drawer", [0.45, 0.42, 0.55]),
    "wardrobe": ("Compact wardrobe closet with simple doors", [0.90, 0.55, 2.00]),
    "dresser": ("Low dresser chest with storage drawers", [1.10, 0.48, 0.85]),
    "desk": ("Practical rectangular work desk", [1.10, 0.60, 0.75]),
    "student_desk": (
        "Compact student classroom desk with a writing surface and storage shelf",
        [1.05, 0.50, 0.75],
    ),
    "teacher_desk": (
        "Larger teacher classroom desk with a broad work surface and modesty panel",
        [1.40, 0.65, 0.76],
    ),
    "office_chair": (
        "Ergonomic office task chair with an adjustable back",
        [0.60, 0.60, 1.05],
    ),
    "guest_chair": (
        "Compact upholstered guest chair with a fixed wooden frame",
        [0.60, 0.65, 0.90],
    ),
    "dining_chair": ("Simple upright dining chair", [0.50, 0.55, 0.90]),
    "chair": ("Simple upright task chair", [0.50, 0.50, 0.90]),
    "student_chair": ("Simple upright student classroom chair", [0.50, 0.50, 0.90]),
    "sofa": ("Compact upholstered two-seat sofa", [1.70, 0.85, 0.90]),
    "table": ("Practical rectangular table", [1.20, 0.80, 0.75]),
    "cabinet": ("Compact freestanding storage cabinet", [0.90, 0.45, 1.10]),
    "bookshelf": ("Compact freestanding bookshelf", [0.90, 0.35, 1.80]),
    "plant": ("Large indoor potted floor plant", [0.60, 0.60, 1.20]),
    "rug": ("Square low-pile area rug", [1.80, 1.80, 0.03]),
    "armchair": ("Compact upholstered armchair", [0.75, 0.75, 0.95]),
    "floor_lamp": ("Slim standing floor lamp", [0.40, 0.40, 1.60]),
    "tv_stand": ("Low media console TV stand", [1.60, 0.45, 0.65]),
    "sideboard": ("Compact dining room sideboard", [1.40, 0.45, 0.80]),
}

_WALL_BACKED_STORAGE_CATEGORIES = {
    "bookshelf",
    "cabinet",
    "dresser",
    "sideboard",
    "tv_stand",
    "wardrobe",
}

_SHALLOW_FURNITURE_COLLISION_RE = re.compile(
    r"^\s*-\s*(?P<first>[^\s]+)\s+collides with\s+"
    r"(?P<second>[^\s]+)\s+\((?P<depth>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>mm|cm|m)\s+penetration\)",
    re.IGNORECASE | re.MULTILINE,
)


class StatefulFurnitureAgent(BaseStatefulAgent, BaseFurnitureAgent):
    """Natural conversation between persistent agents with proper image injection."""

    @property
    def agent_type(self) -> AgentType:
        """Return agent type for collision filtering."""
        return AgentType.FURNITURE

    def __init__(
        self,
        cfg: DictConfig,
        logger: BaseLogger,
        geometry_server_host: str = "127.0.0.1",
        geometry_server_port: int = 7000,
        hssd_server_host: str = "127.0.0.1",
        hssd_server_port: int = 7001,
        articulated_server_host: str = "127.0.0.1",
        articulated_server_port: int = 7002,
        materials_server_host: str = "127.0.0.1",
        materials_server_port: int = 7008,
        num_workers: int = 1,
        render_gpu_id: int | None = None,
    ):
        # Initialize base agent (sessions, checkpoint state, prompt registry).
        BaseStatefulAgent.__init__(
            self,
            cfg=cfg,
            logger=logger,
            geometry_server_host=geometry_server_host,
            geometry_server_port=geometry_server_port,
            hssd_server_host=hssd_server_host,
            hssd_server_port=hssd_server_port,
        )
        # Initialize furniture-specific base class.
        BaseFurnitureAgent.__init__(
            self,
            cfg=cfg,
            logger=logger,
            geometry_server_host=geometry_server_host,
            geometry_server_port=geometry_server_port,
            hssd_server_host=hssd_server_host,
            hssd_server_port=hssd_server_port,
            articulated_server_host=articulated_server_host,
            articulated_server_port=articulated_server_port,
            materials_server_host=materials_server_host,
            materials_server_port=materials_server_port,
            num_workers=num_workers,
            render_gpu_id=render_gpu_id,
        )

        # Create persistent agent sessions using base class method.
        self.designer_session, self.critic_session = self._create_sessions()

        # Context image for designer initialization (furniture-specific).
        self.context_image_path: Path | None = None
        # Grounding is intentionally independent from visual publication.  A
        # rejected concept may still yield a safely degraded language contract.
        self._grounding_candidate_path: Path | None = None
        self._grounding_quality_mode: str = "none"
        # Lazily created only when context generation and qwen_local are enabled.
        self._qwen_context_image_editor: OpenAICompatibleContextImageEditor | None = (
            None
        )
        # Lazily created only when context generation and okcodex are enabled.
        self._okcodex_context_image_editor: OKCodexContextImageEditor | None = None
        # Lazily created only when context generation and bailian are enabled.
        self._bailian_context_image_editor: BailianContextImageEditor | None = None
        # Lazily created only when context generation and openrouter are enabled.
        self._openrouter_context_image_editor: OpenRouterContextImageEditor | None = (
            None
        )
        # Populated per scene only when the optional feature is enabled.
        self._placement_order_reference: str = ""
        # Independent image-derived contract; never aliases the legacy reference.
        self._context_image_layout_reference: str = ""
        self._layout_constraint_contract: dict[str, Any] = {}
        self._layout_constraint_contract_text: str = ""

    def _create_designer_agent(self, tools: list[FunctionTool]) -> Agent:
        """Create designer agent with tools.

        Args:
            tools: Tools to provide to the designer

        Returns:
            Configured designer agent
        """
        designer_config = self.cfg.agents.designer_agent
        designer_prompt_enum = FurnitureAgentPrompts[designer_config.prompt]
        return super()._create_designer_agent(
            tools=tools,
            prompt_enum=designer_prompt_enum,
            has_reference_image=self.context_image_path is not None,
        )

    def _create_critic_tools(self) -> list[FunctionTool]:
        """Create critic tools with read-only scene access.

        Returns:
            List of tools for the critic (read-only scene validation tools)
        """
        vision_tools = VisionTools(
            scene=self.scene,
            rendering_manager=self.rendering_manager,
            cfg=self.cfg,
            blender_server=self.blender_server,
        )
        scene_tools = SceneTools(scene=self.scene, cfg=self.cfg)
        self._critic_vision_tools = vision_tools
        self._critic_scene_tools = scene_tools

        # Return vision tools + read-only scene tools.
        # Note: check_physics is NOT included since physics_context is already
        # injected via the critique runner instruction template.
        return [
            vision_tools.tools["observe_scene"],
            scene_tools.tools["get_current_scene_state"],
            scene_tools.tools["check_facing_tool"],
        ]

    def _create_critic_agent(
        self, scene: RoomScene, tools: list[FunctionTool]
    ) -> Agent:
        """Create critic agent with scene context.

        Args:
            scene: RoomScene to provide context for the critic
            tools: Tools to provide to the critic

        Returns:
            Configured critic agent with structured output
        """
        critic_config = self.cfg.agents.critic_agent
        critic_prompt_enum = FurnitureAgentPrompts[critic_config.prompt]
        # The planner's StageBrief and memory are useful designer guidance but
        # are mutable, model-produced text.  They must not become the critic's
        # statement of prompt truth or authorize a visual failure.
        original_task = getattr(scene, "scene_expert_original_description", "")
        return super()._create_critic_agent(
            tools=tools,
            prompt_enum=critic_prompt_enum,
            output_type=FurnitureCritiqueWithScores,
            scene_description=original_task or scene.text_description,
        )

    def _create_planner_agent(
        self, scene: RoomScene, tools: list[FunctionTool]
    ) -> Agent:
        """Create planner agent with scene-specific context.

        Args:
            scene: RoomScene to provide context for the planner
            tools: Tools to provide to the planner

        Returns:
            Configured planner agent
        """
        planner_config = self.cfg.agents.planner_agent
        planner_prompt_enum = FurnitureAgentPrompts[planner_config.prompt]
        return super()._create_planner_agent(
            tools=tools,
            prompt_enum=planner_prompt_enum,
            scene_prompt=scene.text_description,
            max_critique_rounds=self.cfg.max_critique_rounds,
            reset_single_category_threshold=self.cfg.reset_single_category_threshold,
            reset_total_sum_threshold=self.cfg.reset_total_sum_threshold,
            early_finish_min_score=self.cfg.early_finish_min_score,
        )

    def _create_designer_tools(self) -> list[FunctionTool]:
        """Create designer tools with captured dependencies.

        Returns:
            List of tools for the designer agent.
        """
        vision_tools = VisionTools(
            scene=self.scene,
            rendering_manager=self.rendering_manager,
            cfg=self.cfg,
            blender_server=self.blender_server,
            safety_controller=getattr(self, "furniture_safety_controller", None),
        )
        self.furniture_tools = FurnitureTools(
            scene=self.scene,
            asset_manager=self.asset_manager,
            cfg=self.cfg,
            safety_controller=getattr(self, "furniture_safety_controller", None),
        )
        scene_tools = SceneTools(scene=self.scene, cfg=self.cfg)
        workflow_tools = WorkflowTools()

        return [
            *vision_tools.tools.values(),
            *self.furniture_tools.tools.values(),
            *scene_tools.tools.values(),
            *workflow_tools.tools.values(),
        ]

    def _render_empty_room(self) -> Path:
        """Render top-down view of empty room showing doors/windows.

        Uses furniture_selection mode which disables coordinate grid/frame.
        Pass annotate_object_types=[] to disable all labels and bounding boxes.
        Result: clean room geometry with doors/windows visible but unlabeled.

        Returns:
            Path to directory containing rendered image.
        """
        return self.rendering_manager.render_scene(
            scene=self.scene,
            blender_server=self.blender_server,
            include_objects=[],  # Empty room only
            render_name="empty_room_context",
            rendering_mode="furniture_selection",  # Disables grid/frame
            annotate_object_types=[],  # Disables all labels/bboxes
            show_opening_labels=False,  # Qwen input must not contain text overlays
        )

    def _get_context_image_editor(self) -> Any:
        """Resolve the editor after the top-level context switch is enabled."""
        context_cfg = self.cfg.context_image_generation
        backend = str(context_cfg.get("backend", "inherit")).strip().lower()
        if backend == "inherit":
            return self.asset_manager.image_generator
        if backend == "qwen_local":
            if self._qwen_context_image_editor is None:
                qwen_cfg = context_cfg.get("qwen_local")
                if qwen_cfg is None:
                    raise ValueError(
                        "context_image_generation.qwen_local config is required "
                        "when backend=qwen_local"
                    )
                self._qwen_context_image_editor = OpenAICompatibleContextImageEditor(
                    qwen_cfg
                )
            return self._qwen_context_image_editor
        if backend == "okcodex":
            if self._okcodex_context_image_editor is None:
                okcodex_cfg = context_cfg.get("okcodex")
                if okcodex_cfg is None:
                    raise ValueError(
                        "context_image_generation.okcodex config is required "
                        "when backend=okcodex"
                    )
                self._okcodex_context_image_editor = OKCodexContextImageEditor(
                    okcodex_cfg
                )
            return self._okcodex_context_image_editor
        if backend == "bailian":
            if self._bailian_context_image_editor is None:
                bailian_cfg = context_cfg.get("bailian")
                if bailian_cfg is None:
                    raise ValueError(
                        "context_image_generation.bailian config is required "
                        "when backend=bailian"
                    )
                self._bailian_context_image_editor = BailianContextImageEditor(
                    bailian_cfg
                )
            return self._bailian_context_image_editor
        if backend == "openrouter":
            if self._openrouter_context_image_editor is None:
                openrouter_cfg = context_cfg.get("openrouter")
                if openrouter_cfg is None:
                    raise ValueError(
                        "context_image_generation.openrouter config is required "
                        "when backend=openrouter"
                    )
                self._openrouter_context_image_editor = OpenRouterContextImageEditor(
                    openrouter_cfg
                )
            return self._openrouter_context_image_editor
        raise ValueError(
            "Unknown context_image_generation.backend="
            f"{backend!r}; expected 'inherit', 'qwen_local', 'okcodex', "
            "'bailian', or 'openrouter'"
        )

    def _generate_and_save_context_image(
        self,
        scene: RoomScene,
        image_editor: Any,
        quality_gate: ContextImageQualityGateConfig,
    ) -> Path | None:
        """Generate and save context image for design guidance.

        Renders an empty room showing doors/windows, then uses image editing
        to add suggested furniture placement.

        Args:
            scene: RoomScene to generate context image for.

        Returns:
            Path to saved context image.
        """
        console_logger.info("Generating context image for scene...")

        # Render empty room showing doors/windows.
        room_render_dir = self._render_empty_room()
        # Get the top-down image from the render directory.
        room_render = room_render_dir / "0_top.png"

        # Generate context image using the render as reference.
        # Save alongside the input render for easy association.
        output_path = room_render_dir / "context_edited.png"
        if quality_gate.enabled:
            return self._generate_quality_gated_context_image(
                scene=scene,
                image_editor=image_editor,
                room_render=room_render,
                output_path=output_path,
                quality_gate=quality_gate,
            )
        image_path = image_editor.generate_furniture_context_image(
            reference_image_path=room_render,
            scene_description=self._context_scene_description(scene),
            width_m=scene.room_geometry.width,
            length_m=scene.room_geometry.length,
            output_path=output_path,
        )
        self._grounding_candidate_path = Path(image_path)
        self._grounding_quality_mode = "full_reference"
        console_logger.info(f"Context image saved to: {image_path}")
        return image_path

    def _generate_quality_gated_context_image(
        self,
        *,
        scene: RoomScene,
        image_editor: Any,
        room_render: Path,
        output_path: Path,
        quality_gate: ContextImageQualityGateConfig,
    ) -> Path | None:
        """Generate independent candidates and publish the best usable image."""
        room_render_dir = room_render.parent
        output_path.unlink(missing_ok=True)
        output_path.with_suffix(".metadata.json").unlink(missing_ok=True)
        report_path = room_render_dir / "context_image_quality.json"
        self._grounding_candidate_path = None
        self._grounding_quality_mode = "none"
        report: dict[str, Any] = {
            "schema_version": 3,
            "quality_gate": {
                "enabled": quality_gate.enabled,
                "max_regenerations": quality_gate.max_regenerations,
                "max_attempts": quality_gate.max_attempts,
                "min_score": quality_gate.min_score,
            },
            "original_image": {
                "path": str(room_render),
                "sha256": file_sha256(room_render),
            },
            "attempts": [],
            "final_status": "running",
            "accepted_attempt": None,
            "best_attempt": None,
            "best_score": None,
            "selection_mode": None,
            "selection_reason": None,
            "designer_reference_path": None,
            "designer_visual_reference_path": None,
            "grounding_candidate_path": None,
            "grounding_quality_mode": "none",
        }
        evaluator = ContextImageQualityEvaluator(vlm_service=self.vlm_service)
        editor_config = getattr(image_editor, "config", None)
        base_seed = getattr(editor_config, "seed", None)
        best_candidate_path: Path | None = None
        best_attempt_number: int | None = None
        best_rank: tuple[float, int, int] | None = None
        best_visual_passed = False
        grounding_candidate_path: Path | None = None
        grounding_attempt_number: int | None = None
        grounding_quality_mode = "none"
        grounding_rank: tuple[int, float, float, int] | None = None
        grounding_mode_rank = {
            "none": 0,
            "inventory_only": 1,
            "relations_only": 2,
            "contract_only": 3,
            "full_reference": 4,
        }

        reasoning_config = getattr(self.cfg.openai, "reasoning_effort", None)
        verbosity_config = getattr(self.cfg.openai, "verbosity", None)
        reasoning_effort = str(getattr(reasoning_config, "asset_validation", "none"))
        if reasoning_effort.strip().lower() == "none":
            reasoning_effort = "low"
        verbosity = str(getattr(verbosity_config, "asset_validation", "low"))

        def publish_candidate(candidate_path: Path) -> None:
            shutil.copy2(candidate_path, output_path)
            candidate_metadata = candidate_path.with_suffix(".metadata.json")
            if candidate_metadata.exists():
                shutil.copy2(
                    candidate_metadata,
                    output_path.with_suffix(".metadata.json"),
                )

        for attempt_number in range(1, quality_gate.max_attempts + 1):
            candidate_path = room_render_dir / (
                f"context_edited_attempt_{attempt_number:02d}.png"
            )
            effective_seed = (
                (int(base_seed) + attempt_number - 1) % (2**32)
                if base_seed is not None
                else None
            )
            attempt: dict[str, Any] = {
                "attempt": attempt_number,
                "candidate_path": str(candidate_path),
                "seed": effective_seed,
                "generation_status": "running",
                "quality_status": "not_run",
                "fallback_eligible": False,
            }
            report["attempts"].append(attempt)
            console_logger.info(
                "Generating independent context image candidate %d/%d (seed=%s)",
                attempt_number,
                quality_gate.max_attempts,
                effective_seed,
            )
            generation_start = time.monotonic()
            try:
                image_editor.generate_furniture_context_image(
                    reference_image_path=room_render,
                    scene_description=self._context_scene_description(scene),
                    width_m=scene.room_geometry.width,
                    length_m=scene.room_geometry.length,
                    output_path=candidate_path,
                    seed_override=effective_seed,
                )
                attempt["generation_status"] = "success"
                attempt["generation_seconds"] = time.monotonic() - generation_start
                attempt["candidate_sha256"] = file_sha256(candidate_path)
            except Exception as exc:
                attempt["generation_status"] = "error"
                attempt["generation_seconds"] = time.monotonic() - generation_start
                attempt["error_type"] = type(exc).__name__
                attempt["error"] = str(exc)[:1000]
                report["final_status"] = (
                    "retrying_after_edit_error"
                    if attempt_number < quality_gate.max_attempts
                    else "edit_error_exhausted"
                )
                write_context_image_quality_report(report_path, report)
                console_logger.warning(
                    "Context image candidate %d/%d failed: %s",
                    attempt_number,
                    quality_gate.max_attempts,
                    exc,
                )
                continue

            # Every successfully generated image remains eligible for a degraded
            # grounding attempt even when its judge call later fails.
            unjudged_rank = (1, 0.0, 0.0, -attempt_number)
            if grounding_rank is None or unjudged_rank > grounding_rank:
                grounding_rank = unjudged_rank
                grounding_candidate_path = candidate_path
                grounding_attempt_number = attempt_number
                grounding_quality_mode = "inventory_only"

            try:
                deterministic_result = evaluate_context_image_deterministic(
                    room_render, candidate_path
                )
            except Exception as exc:
                deterministic_result = {
                    "passed": False,
                    "reasons": [
                        "deterministic gate error: " f"{type(exc).__name__}: {exc}"
                    ],
                }
            attempt["deterministic_result"] = deterministic_result

            console_logger.info(
                "Evaluating context image candidate %d/%d with VLM",
                attempt_number,
                quality_gate.max_attempts,
            )
            quality_start = time.monotonic()
            try:
                quality_result = evaluator.evaluate(
                    original_image_path=room_render,
                    candidate_image_path=candidate_path,
                    scene_description=self._context_scene_description(scene),
                    model=str(self.cfg.openai.model),
                    min_score=quality_gate.min_score,
                    reasoning_effort=reasoning_effort,
                    verbosity=verbosity,
                )
            except Exception as exc:
                attempt["quality_status"] = "error"
                attempt["quality_seconds"] = time.monotonic() - quality_start
                attempt["quality_error_type"] = type(exc).__name__
                attempt["quality_error"] = str(exc)[:1000]
                report["final_status"] = (
                    "retrying_after_judge_error"
                    if attempt_number < quality_gate.max_attempts
                    else "judge_error_exhausted"
                )
                write_context_image_quality_report(report_path, report)
                console_logger.warning(
                    "Context image quality judge failed for candidate %d/%d; "
                    "continuing with independent candidates: %s",
                    attempt_number,
                    quality_gate.max_attempts,
                    exc,
                )
                continue

            attempt["quality_status"] = (
                "passed" if quality_result.passed else "rejected"
            )
            attempt["quality_seconds"] = time.monotonic() - quality_start
            attempt["quality_result"] = quality_result.to_dict()
            attempt["fallback_eligible"] = quality_result.fallback_eligible
            attempt["grounding_quality_mode"] = quality_result.grounding_quality_mode

            candidate_grounding_rank = (
                grounding_mode_rank[quality_result.grounding_quality_mode],
                quality_result.grounding_utility_score,
                quality_result.quality_score,
                -attempt_number,
            )
            candidate_quality_mode = quality_result.grounding_quality_mode
            if not deterministic_result.get("passed", False):
                candidate_quality_mode = (
                    "inventory_only"
                    if not deterministic_result.get("content_nonempty", True)
                    else "relations_only"
                )
                attempt["grounding_quality_mode"] = candidate_quality_mode
                candidate_grounding_rank = (
                    grounding_mode_rank[candidate_quality_mode],
                    quality_result.grounding_utility_score,
                    quality_result.quality_score,
                    -attempt_number,
                )
            if grounding_rank is None or candidate_grounding_rank > grounding_rank:
                grounding_rank = candidate_grounding_rank
                grounding_candidate_path = candidate_path
                grounding_attempt_number = attempt_number
                grounding_quality_mode = candidate_quality_mode

            # A fallback may relax only the normal score threshold. Never hand
            # Designer an image that violates a structural invariant (including
            # render-style fidelity), because it can corrupt the layout contract.
            if quality_result.fallback_eligible and deterministic_result.get(
                "passed", False
            ):
                candidate_rank = (
                    quality_result.quality_score,
                    int(quality_result.openings_clear),
                    -attempt_number,
                )
                if best_rank is None or candidate_rank > best_rank:
                    best_rank = candidate_rank
                    best_candidate_path = candidate_path
                    best_attempt_number = attempt_number
                    report["best_attempt"] = attempt_number
                    report["best_score"] = quality_result.quality_score
                    best_visual_passed = quality_result.passed

            reasons = "; ".join(quality_result.reasons) or "no material issue"
            report["final_status"] = (
                "retrying_after_rejection"
                if attempt_number < quality_gate.max_attempts
                else "rejected_exhausted"
            )
            write_context_image_quality_report(report_path, report)
            log = (
                console_logger.info if quality_result.passed else console_logger.warning
            )
            log(
                "Context image candidate %d/%d %s (score %.1f, fallback_eligible=%s): %s",
                attempt_number,
                quality_gate.max_attempts,
                "passed" if quality_result.passed else "rejected",
                quality_result.quality_score,
                quality_result.fallback_eligible,
                reasons,
            )

        if best_candidate_path is not None and best_attempt_number is not None:
            publish_candidate(best_candidate_path)
            # When Designer receives a visual, its language contract must describe
            # that exact image rather than a separately ranked candidate.
            selected_attempt = report["attempts"][best_attempt_number - 1]
            grounding_candidate_path = best_candidate_path
            grounding_attempt_number = best_attempt_number
            grounding_quality_mode = selected_attempt.get(
                "grounding_quality_mode", "full_reference"
            )
            report["final_status"] = (
                "accepted" if best_visual_passed else "best_candidate_fallback"
            )
            report["accepted_attempt"] = best_attempt_number
            report["selection_mode"] = (
                "best_scored_pass" if best_visual_passed else "best_effort_fallback"
            )
            report["selection_reason"] = (
                "scored all generated candidates and selected the highest-ranked "
                "structurally eligible candidate"
            )
            report["designer_reference_path"] = str(output_path)
            report["designer_visual_reference_path"] = str(output_path)
            report["grounding_candidate_path"] = (
                str(grounding_candidate_path) if grounding_candidate_path else None
            )
            report["grounding_attempt"] = grounding_attempt_number
            report["grounding_quality_mode"] = grounding_quality_mode
            self._grounding_candidate_path = grounding_candidate_path
            self._grounding_quality_mode = grounding_quality_mode
            write_context_image_quality_report(report_path, report)
            console_logger.warning(
                "Context image gate exhausted %d attempt(s); using best candidate "
                "%d with score %.1f: %s",
                quality_gate.max_attempts,
                best_attempt_number,
                float(report["best_score"]),
                output_path,
            )
            return output_path

        self._grounding_candidate_path = grounding_candidate_path
        self._grounding_quality_mode = grounding_quality_mode
        report["grounding_candidate_path"] = (
            str(grounding_candidate_path) if grounding_candidate_path else None
        )
        report["grounding_attempt"] = grounding_attempt_number
        report["grounding_quality_mode"] = grounding_quality_mode
        generated_count = sum(
            attempt.get("generation_status") == "success"
            for attempt in report["attempts"]
        )
        scored_count = sum(
            attempt.get("quality_status") in {"passed", "rejected"}
            for attempt in report["attempts"]
        )
        report["final_status"] = (
            "generation_error"
            if generated_count == 0
            else "judge_error" if scored_count == 0 else "no_eligible_candidate"
        )
        report["selection_mode"] = "none"
        report["selection_reason"] = (
            "no candidate was safe enough to publish as a Designer image; "
            "the best generated candidate remains available for degraded grounding"
        )
        report["designer_reference_path"] = None
        report["designer_visual_reference_path"] = None
        write_context_image_quality_report(report_path, report)
        console_logger.warning(
            "Context image gate exhausted %d attempt(s) with no publishable visual; "
            "continuing with grounding mode %s",
            quality_gate.max_attempts,
            grounding_quality_mode,
        )
        return None

    async def add_furniture(self, scene: RoomScene) -> None:
        """Add furniture to a scene.

        Args:
            scene: RoomScene to add furniture to (mutated in place)
        """
        # Store everything as instance variables for closure access.
        self.scene = scene
        self.context_image_path = None
        self._grounding_candidate_path = None
        self._grounding_quality_mode = "none"
        self._context_image_layout_reference = ""
        safety_description = getattr(
            scene,
            "scene_expert_original_description",
            scene.text_description,
        )
        self._configure_furniture_safety_for_scene(safety_description)
        self._layout_constraint_contract = build_furniture_layout_constraint_contract(
            scene, getattr(self, "furniture_safety_controller", None)
        )
        self._layout_constraint_contract_text = (
            format_furniture_layout_constraint_contract(
                self._layout_constraint_contract
            )
        )
        try:
            write_context_image_quality_report(
                Path(scene.scene_dir) / "furniture_layout_constraint_contract.json",
                self._layout_constraint_contract,
            )
        except (OSError, TypeError):
            console_logger.debug("Could not persist layout contract", exc_info=True)
        self._placement_order_reference = build_furniture_placement_order_reference(
            cfg=self.cfg,
            scene_prompt=safety_description,
            scene_dir=scene.scene_dir,
            vlm_service=self.vlm_service,
            model=self.cfg.openai.model,
            room_dimensions={
                "length_m": scene.room_geometry.length,
                "width_m": scene.room_geometry.width,
            },
        )

        # Generate context image if configured. If generation fails, continue without it.
        if self.cfg.context_image_generation.enabled:
            # Resolve static backend configuration outside the runtime fail-open
            # block. This branch is unreachable when the feature is disabled.
            image_editor = self._get_context_image_editor()
            quality_gate = ContextImageQualityGateConfig.from_config(
                self.cfg.context_image_generation.get("quality_gate")
            )
            try:
                self.context_image_path = self._generate_and_save_context_image(
                    scene,
                    image_editor,
                    quality_gate,
                )
            except Exception as e:
                console_logger.warning(
                    f"Context image generation failed, continuing without it: {e}"
                )
                self.context_image_path = None

        # Grounding is independent from visual publication. Rejected images stay
        # hidden from Designer but may still produce a degraded language contract.
        grounded_cfg = self.cfg.context_image_generation.get("grounded_layout")
        if grounded_cfg is not None and self._grounding_candidate_path is not None:
            self._context_image_layout_reference = (
                build_grounded_furniture_layout_reference(
                    image_path=self._grounding_candidate_path,
                    scene_prompt=safety_description,
                    cfg=grounded_cfg,
                    vlm_service=self.vlm_service,
                    model=str(self.cfg.openai.model),
                    artifact_dir=self._grounding_candidate_path.parent,
                    quality_mode=self._grounding_quality_mode,
                )
            )

        # Create designer, critic, and planner with tools once for this scene.
        designer_tools = self._create_designer_tools()
        self.designer = self._create_designer_agent(tools=designer_tools)
        critic_tools = self._create_critic_tools()
        self.critic = self._create_critic_agent(scene=scene, tools=critic_tools)
        planner_tools = self._create_planner_tools()
        self.planner = self._create_planner_agent(scene=scene, tools=planner_tools)

        # Get runner instruction from prompt registry.
        runner_instruction = self.prompt_registry.get_prompt(
            prompt_enum=FurnitureAgentPrompts.STATEFUL_PLANNER_RUNNER_INSTRUCTION,
        )

        # Run the furniture placement workflow.
        result: RunResult | None = None
        try:
            result = await Runner.run(
                starting_agent=self.planner,
                input=runner_instruction,
                max_turns=self.cfg.agents.planner_agent.max_turns,
                run_config=self._create_run_config(),
            )
        except MaxTurnsExceeded as error:
            self._recover_from_planner_turn_limit(error)

        if result is not None:
            log_agent_usage(result=result, agent_name="PLANNER (FURNITURE)")

        if result is not None and result.final_output:
            log_agent_response(
                response=result.final_output, agent_name="PLANNER (FURNITURE)"
            )

        pre_final_hard_state = self._evaluate_current_hard_state()
        _, _, pre_final_actions = self._try_deterministic_repair_for_hard_state(
            pre_final_hard_state,
            source="post_planner_pre_final_critique",
        )
        if pre_final_actions:
            console_logger.info(
                "Deterministic furniture repair before final critique: %s",
                "; ".join(pre_final_actions),
            )
        self._converge_prompt_required_inventory(source="before final critique")

        seating_fixes = align_seating_to_nearest_surface(
            scene,
            allowed_targets_by_seat=seating_orientation_targets(scene, config=self.cfg),
        )
        if seating_fixes:
            console_logger.info(
                "Deterministic seating orientation guard before final critique: %s",
                "; ".join(
                    f"{fix.subject_id}->{fix.target_id}" for fix in seating_fixes
                ),
            )

        # Compute final critique and scores for completed scene.
        # Check if scene changed since last checkpoint to avoid redundant critique.
        current_scene_hash = self.scene.content_hash()

        if self._critic_failed:
            console_logger.warning(
                "Skipping final furniture critique because critic scoring already "
                "failed in this stage"
            )
        elif self._can_skip_final_critique(current_scene_hash):
            console_logger.info(
                "Scene unchanged since last critique, skipping final critique"
            )
        else:
            console_logger.info(
                "Scene changed since last critique, computing final critique"
            )
            # Pass update_checkpoint=False to preserve N-1 checkpoint for reset check.
            try:
                await self._request_critique_impl(update_checkpoint=False)
            except Exception:
                self._critic_failed = True
                console_logger.exception(
                    "Final furniture critique failed; preserving the best available "
                    "hard-valid checkpoint instead of restarting the planner"
                )

        # Validate final scene and save scores.
        await self._finalize_scene_and_scores()
        # Finalization may restore an earlier best-scoring checkpoint. Enforce
        # requested inventory once more so a checkpoint with an unpenalized
        # duplicate cannot become the persisted furniture-stage scene.
        self._converge_prompt_required_inventory(source="after finalization")

    def _recover_from_planner_turn_limit(self, error: MaxTurnsExceeded) -> list[str]:
        """Continue only when bounded deterministic repair restores hard validity.

        The planner may exhaust its conversational turn budget after it has
        already placed a nearly valid scene.  Preserve the useful partial state
        only when the existing geometry-only repair closes the remaining hard
        issue; otherwise keep the original failure behavior.
        """
        hard_state = self._evaluate_current_hard_state()
        repaired_state, _, actions = self._try_deterministic_repair_for_hard_state(
            hard_state,
            source="planner_turn_limit",
        )
        if repaired_state is None or not repaired_state.hard_valid:
            raise error
        console_logger.warning(
            "Furniture planner exhausted its turn budget; deterministic repair "
            "restored hard validity: %s",
            "; ".join(actions) if actions else "no repair was required",
        )
        return actions

    def _get_final_scores_directory(self) -> Path:
        """Get the directory path for saving final furniture placement state.

        Returns:
            Path to scene_states/furniture directory.
        """
        return self.logger.output_dir / "scene_states" / "furniture"

    def _get_critique_prompt_enum(self) -> Any:
        """Get the prompt enum for critic runner instruction.

        Returns:
            Furniture-specific critic instruction prompt.
        """
        return FurnitureAgentPrompts.STATEFUL_CRITIC_RUNNER_INSTRUCTION

    def _get_initial_design_prompt_enum(self) -> Any:
        """Get the prompt enum for initial design instruction.

        Returns:
            Furniture-specific initial design instruction prompt.
        """
        return FurnitureAgentPrompts.DESIGNER_INITIAL_INSTRUCTION

    def _get_initial_design_prompt_kwargs(self) -> dict:
        """Get prompt kwargs for initial design instruction.

        Returns:
            Dict with scene description, room boundary, and reference image flag.
        """
        room_geometry = self.scene.room_geometry
        return {
            "scene_description": self.scene.text_description,
            "has_reference_image": self.context_image_path is not None,
            "room_length": room_geometry.length,
            "room_width": room_geometry.width,
            "room_local_footprint_vertices": (
                room_geometry.room_local_footprint_vertices
            ),
        }

    def _build_initial_design_input(self, instruction: str) -> str | list[dict]:
        """Add deterministic room-aware bedroom guidance to the initial design."""
        instruction = append_placement_order_reference(
            instruction,
            self._placement_order_reference,
        )
        instruction = append_placement_order_reference(
            instruction,
            self._context_image_layout_reference,
        )
        instruction = append_placement_order_reference(
            instruction,
            self._layout_constraint_contract_text,
        )
        safety_cfg = getattr(self.cfg, "furniture_safety_controller", None)
        bedroom_cfg = getattr(safety_cfg, "bedroom_layout", None)
        guidance = ""
        if not self._is_polygon_room():
            guidance = format_bedroom_anchor_guidance(
                scene=self.scene,
                cfg=bedroom_cfg,
            )
        else:
            console_logger.info(
                "Skipping rectangle/cardinal bedroom guidance for polygon room %s",
                getattr(self.scene, "room_id", "unknown"),
            )
        if guidance:
            instruction = (
                f"{instruction}\n\n"
                "# Deterministic Room-Aware Layout Guidance\n"
                f"{guidance}"
            )
        return super()._build_initial_design_input(instruction)

    def _context_scene_description(self, scene: RoomScene) -> str:
        original_description = (
            getattr(scene, "scene_expert_original_description", None)
            or scene.text_description
        )
        contract = getattr(self, "_layout_constraint_contract_text", "")
        if not contract:
            return original_description
        return f"{original_description}\n\n{contract}"

    async def _request_initial_design_impl(self) -> str:
        """Run the initial designer, then repair only prompt-authorized relations.

        The planner auto-scores the result immediately after this method returns.
        Performing an eligible deterministic repair here therefore prevents the
        first critic render from observing a known-bad, but otherwise complete,
        LLM layout.  This remains deliberately inactive for legacy/shadow
        rollout: only hard constraints compiled from the immutable prompt may
        move furniture before that first critique.
        """
        result = await super()._request_initial_design_impl()
        self._repair_initial_contract_layout()
        return result

    def _repair_initial_contract_layout(self) -> list[str]:
        """Repair contract-authorized furniture relations before first critique.

        Both repair mechanisms are geometry-only and retain their own
        whole-scene acceptance/rollback checks.  In particular, this does not
        ask an LLM or VLM to infer a pose, and it never activates from a
        StageBrief, current layout, or legacy prompt heuristic.
        """
        critic_config = critic_config_from_any(self.cfg)
        if (
            not critic_config.enabled
            or not critic_config.metric_enabled("functional_dependency")
            or constraint_mode(critic_config) != "contract"
        ):
            return []

        relation_fixes = improve_furniture_relations(
            self.scene,
            config=critic_config,
        )
        seating_fixes = align_seating_to_nearest_surface(
            self.scene,
            allowed_targets_by_seat=seating_orientation_targets(
                self.scene,
                config=critic_config,
            ),
        )
        if not relation_fixes and not seating_fixes:
            return []

        # The next automatic critic request must render and evaluate the pose
        # just accepted above rather than use the designer's pre-repair cache.
        self.rendering_manager.clear_cache()
        self._reset_critic_candidate_cache()
        actions = [f"{fix.object_id}:{fix.relation_type}" for fix in relation_fixes] + [
            f"{fix.subject_id}->{fix.target_id}:seating_orientation"
            for fix in seating_fixes
        ]
        console_logger.info(
            "Initial prompt-contract furniture repair before first critique: %s",
            "; ".join(actions),
        )
        return actions

    def _get_context_image_path(self) -> Path | None:
        """Get the AI-generated context image for initial design.

        Returns:
            Path to context image if available, None otherwise.
        """
        return self.context_image_path

    def _get_design_change_prompt_enum(self) -> Any:
        """Get the prompt enum for design change instruction.

        Returns:
            Furniture-specific design change instruction prompt.
        """
        return FurnitureAgentPrompts.DESIGNER_CRITIQUE_INSTRUCTION_STATEFUL

    def _set_placement_noise_profile(self, mode: PlacementNoiseMode) -> None:
        """Set placement noise profile for furniture tools.

        Args:
            mode: Placement noise mode (NATURAL or PERFECT).
        """
        self.furniture_tools.set_noise_profile(mode)

    def _attempt_deterministic_repair(
        self, hard_state: HardStateEvaluation
    ) -> tuple[bool, list[str]]:
        if not self.scene:
            return False, []

        actions: list[str] = []
        polygon_original_transforms: dict[str, tuple[Any, RigidTransform]] = {}
        if self._is_polygon_room():
            for object_id, obj in getattr(self.scene, "objects", {}).items():
                if (
                    getattr(obj, "object_type", None) == ObjectType.FURNITURE
                    and not getattr(obj, "immutable", False)
                    and room_geometry_covers_object(self.scene.room_geometry, obj)
                ):
                    polygon_original_transforms[str(object_id)] = (
                        obj.object_id,
                        obj.transform,
                    )

        def rollback_new_polygon_violations() -> None:
            rolled_back: list[str] = []
            for object_id, (
                scene_object_id,
                original_transform,
            ) in polygon_original_transforms.items():
                obj = self.scene.get_object(scene_object_id)
                if obj is None or room_geometry_covers_object(
                    self.scene.room_geometry, obj
                ):
                    continue
                obj.transform = original_transform
                rolled_back.append(object_id)
            if rolled_back:
                actions.append(
                    "rolled back polygon-unsafe repair moves: " + ", ".join(rolled_back)
                )
                self.rendering_manager.clear_cache()

        reasons = " ".join(hard_state.hard_reasons or []).lower()
        repair_plan = build_repair_plan(
            stage=self.agent_type.value,
            hard_reasons=hard_state.hard_reasons,
            max_attempts=1,
        )
        console_logger.info("Deterministic furniture %s", repair_plan.to_log_text())

        required_counts = self._repair_required_counts()
        for category in required_counts:
            if not self._category_matches_missing_reason(category, reasons):
                continue
            added = self._ensure_required_furniture_asset(category)
            if added:
                actions.append(
                    f"added {added} missing {category} asset(s) from local/HSSD bank"
                )

        if "geometry construction failed" in reasons:
            replaced = self._replace_geometry_failed_furniture_assets(reasons)
            if replaced:
                actions.append(
                    f"replaced {replaced} geometry-failed furniture asset(s)"
                )
        if (
            FailureCategory.DOOR_OR_OPENING_CLEARANCE in repair_plan.categories
            and self._repair_forbidden_zone_conflicts(include_windows=False)
        ):
            actions.append("cleared deterministic door/opening forbidden zones")

        if not is_bedroom_scene(self.scene):
            if "collisions" in reasons:
                if self._repair_generic_wall_collisions():
                    actions.append(
                        "moved generic furniture away from room walls to the deterministic margin"
                    )
                actions.extend(self._repair_shallow_furniture_collisions())
            removed_excess = self._remove_excess_required_furniture(required_counts)
            if removed_excess:
                actions.append(
                    f"removed {removed_excess} duplicate prompt-required furniture asset(s)"
                )
            rollback_new_polygon_violations()
            return bool(actions), actions

        if self._is_polygon_room():
            console_logger.info(
                "Skipping rectangle-only bed, nightstand, and wardrobe anchor "
                "repairs for polygon room %s",
                getattr(self.scene, "room_id", "unknown"),
            )
            rollback_new_polygon_violations()
            return bool(actions), actions

        if self._anchor_existing_bed():
            actions.append("anchored bed to deterministic bedroom head wall")
        if self._repair_bedside_nightstands():
            actions.append("repositioned nightstands to deterministic bedside anchors")
        if "dresser" in reasons and self._repair_dresser_opposite_bed_wall_anchor():
            actions.append("anchored dresser to the wall opposite the bed")
        repaired_storage_pair = False
        if self._prompt_requires_wardrobe_next_to_dresser():
            repaired_storage_pair = self._repair_wardrobe_next_to_dresser()
            if repaired_storage_pair:
                actions.append("placed wardrobe against the wall next to dresser")
        if (
            not repaired_storage_pair
            and (
                "window access warning" in reasons
                or "wardrobe" in reasons
                or "closet" in reasons
                or "collisions" in reasons
                or FailureCategory.WINDOW_OR_WALL_ACCESS in repair_plan.categories
            )
            and self._repair_wardrobe_wall_anchor()
        ):
            actions.append("moved wardrobe to a deterministic wall/corner anchor")

        removed_excess = self._remove_excess_required_furniture(required_counts)
        if removed_excess:
            actions.append(
                f"removed {removed_excess} duplicate prompt-required furniture asset(s)"
            )

        rollback_new_polygon_violations()
        return bool(actions), actions

    def _remove_excess_required_furniture(self, required_counts: dict[str, int]) -> int:
        """Converge prompt-counted inventory after repair/fallback asset creation."""
        if self.scene is None or not hasattr(self.scene, "objects"):
            return 0
        removed = 0
        for category, required in required_counts.items():
            objects = self._furniture_by_category(category)
            excess = len(objects) - int(required or 0)
            if excess <= 0:
                continue
            objects.sort(key=lambda obj: self._duplicate_keep_key(category, obj))
            for obj in objects[-excess:]:
                self.scene.remove_object(obj.object_id)
                removed += 1
                console_logger.info(
                    "Deterministic inventory repair removed excess %s asset %s",
                    category,
                    obj.object_id,
                )
        return removed

    def _converge_prompt_required_inventory(self, *, source: str) -> int:
        """Remove prompt-counted duplicates independently of hard-check status."""
        required_counts = self._repair_required_counts()
        removed = self._remove_excess_required_furniture(required_counts)
        if removed:
            console_logger.info(
                "Deterministic inventory convergence %s removed %d duplicate "
                "prompt-required furniture asset(s)",
                source,
                removed,
            )
            self.rendering_manager.clear_cache()
            self._reset_critic_candidate_cache()
        return removed

    def _duplicate_keep_key(self, category: str, obj: SceneObject) -> tuple[Any, ...]:
        """Prefer normal, wall-backed storage when a counted category is duplicated."""
        metadata = getattr(obj, "metadata", {}) or {}
        placeholder = bool(metadata.get("repair_placeholder"))
        dining_wall_penalty = (
            self._dining_sideboard_wall_penalty(obj)
            if category in {"credenza", "sideboard"}
            else 0.0
        )
        wall_distance = (
            self._nearest_room_boundary_distance(obj)
            if category in _WALL_BACKED_STORAGE_CATEGORIES
            else 0.0
        )
        return (placeholder, dining_wall_penalty, wall_distance, str(obj.object_id))

    def _dining_sideboard_wall_penalty(self, obj: SceneObject) -> float:
        """Prefer the wall parallel to the dining table's long axis."""
        if self.scene is None or getattr(self.scene, "room_geometry", None) is None:
            return 0.0
        table = next(
            (
                candidate
                for candidate in self.scene.objects.values()
                if "dining"
                in (
                    f"{candidate.object_id} {candidate.name} "
                    f"{candidate.description}"
                ).lower()
                and "table"
                in (
                    f"{candidate.object_id} {candidate.name} "
                    f"{candidate.description}"
                ).lower()
            ),
            None,
        )
        bounds = self._room_bounds_xy()
        object_bounds = obj.compute_world_bounds()
        if (
            table is None
            or bounds is None
            or object_bounds is None
            or table.bbox_min is None
            or table.bbox_max is None
        ):
            return 0.0

        min_x, min_y, max_x, max_y = bounds
        lower, upper = object_bounds
        boundary_gaps = {
            "west": abs(float(lower[0]) - min_x),
            "east": abs(max_x - float(upper[0])),
            "south": abs(float(lower[1]) - min_y),
            "north": abs(max_y - float(upper[1])),
        }
        nearest_wall = min(boundary_gaps, key=boundary_gaps.get)
        wall_tangent = (
            np.array([0.0, 1.0])
            if nearest_wall in {"west", "east"}
            else np.array([1.0, 0.0])
        )

        local_size = np.asarray(table.bbox_max) - np.asarray(table.bbox_min)
        local_long_axis = (
            np.array([1.0, 0.0, 0.0])
            if float(local_size[0]) >= float(local_size[1])
            else np.array([0.0, 1.0, 0.0])
        )
        world_long_axis = table.transform.rotation().matrix() @ local_long_axis
        alignment = abs(float(np.dot(world_long_axis[:2], wall_tangent)))
        return round(1.0 - min(1.0, alignment), 6)

    def _nearest_room_boundary_distance(self, obj: SceneObject) -> float:
        bounds = self._room_bounds_xy()
        object_bounds = obj.compute_world_bounds()
        if bounds is None or object_bounds is None:
            return float("inf")
        min_x, min_y, max_x, max_y = bounds
        lower, upper = object_bounds
        return max(
            0.0,
            min(
                float(lower[0]) - min_x,
                max_x - float(upper[0]),
                float(lower[1]) - min_y,
                max_y - float(upper[1]),
            ),
        )

    def _repair_generic_wall_collisions(self) -> bool:
        """Move non-bedroom furniture back inside a conservative wall margin.

        The furniture designer can place a thin rug or a duplicate/repair asset
        exactly on the room-boundary AABB.  Drake then reports a small collision
        with the wall thickness even though the object appears visually inside the
        room.  Bedroom repairs already use the deterministic wall margin, but
        generic rooms previously left this case to the planner/LLM.  Refit every
        mutable furniture object once; the operation is idempotent and preserves
        the requested object set.
        """
        if self.scene is None or self._room_bounds_xy() is None:
            return False

        changed = False
        for obj in self.scene.objects.values():
            if getattr(obj, "immutable", False):
                continue
            if getattr(obj, "object_type", None) != ObjectType.FURNITURE:
                continue
            transform = self._fit_transform_inside_room(obj, obj.transform)
            if self._transform_close(obj.transform, transform):
                continue
            if not self._move_object_if_repair_safe(obj, transform):
                continue
            changed = True
            console_logger.info(
                "Deterministic generic wall repair moved %s (%s) inside room margin",
                obj.object_id,
                obj.name,
            )
        return changed

    def _repair_shallow_furniture_collisions(self) -> list[str]:
        """Separate one reported shallow furniture collision without a layout guess.

        This is deliberately a narrow geometry fallback for small mesh
        penetrations introduced by asset placement or snapping.  It never uses
        object categories, room names, or a VLM judgement.  Deep collisions are
        left for the planner because automatically moving them risks changing a
        meaningful prompt relationship.
        """
        if self.scene is None:
            return []

        max_penetration = max(
            0.0,
            float(
                self._repair_cfg_value("collision_separation_max_penetration_m", 0.08)
            ),
        )
        clearance = max(
            0.005,
            float(self._repair_cfg_value("collision_separation_margin_m", 0.025)),
        )
        reported = self._reported_shallow_furniture_collisions(
            self._get_cached_physics_context(),
            max_penetration_m=max_penetration,
        )
        if not reported:
            return []

        objects_by_id = {
            str(object_id): obj for object_id, obj in self.scene.objects.items()
        }
        before_pairs = self._furniture_aabb_overlap_pairs()
        for first_id, second_id, penetration in reported:
            first = objects_by_id.get(first_id)
            second = objects_by_id.get(second_id)
            if first is None or second is None:
                continue
            candidates = [
                obj
                for obj in (first, second)
                if not getattr(obj, "immutable", False)
                and getattr(obj, "object_type", None) == ObjectType.FURNITURE
            ]
            if not candidates:
                continue
            candidates.sort(key=self._collision_repair_candidate_key)
            for moving in candidates:
                other = second if moving is first else first
                transform = self._safe_shallow_collision_transform(
                    moving,
                    other,
                    penetration=penetration,
                    clearance=clearance,
                    before_pairs=before_pairs,
                )
                if transform is None:
                    continue
                if not self._move_object_if_repair_safe(moving, transform):
                    continue
                return [
                    "separated shallow collision "
                    f"{first_id}<->{second_id} by moving {moving.object_id}"
                ]
        return []

    def _reported_shallow_furniture_collisions(
        self,
        physics_context: str,
        *,
        max_penetration_m: float,
    ) -> list[tuple[str, str, float]]:
        """Parse only bounded, object-addressable collisions from physics output."""
        scale = {"mm": 0.001, "cm": 0.01, "m": 1.0}
        result: list[tuple[str, str, float]] = []
        for match in _SHALLOW_FURNITURE_COLLISION_RE.finditer(
            str(physics_context or "")
        ):
            penetration = (
                float(match.group("depth")) * scale[match.group("unit").lower()]
            )
            if 0.0 < penetration <= max_penetration_m:
                result.append(
                    (
                        match.group("first"),
                        match.group("second"),
                        penetration,
                    )
                )
        return result

    def _collision_repair_candidate_key(self, obj: SceneObject) -> tuple[float, str]:
        """Prefer moving the smaller object when both are equally modifiable."""
        bounds = obj.compute_world_bounds()
        area = float("inf")
        if bounds is not None:
            lower, upper = bounds
            area = max(0.0, float(upper[0] - lower[0])) * max(
                0.0, float(upper[1] - lower[1])
            )
        return area, str(obj.object_id)

    def _safe_shallow_collision_transform(
        self,
        moving: SceneObject,
        other: SceneObject,
        *,
        penetration: float,
        clearance: float,
        before_pairs: set[frozenset[str]],
    ) -> RigidTransform | None:
        moving_bounds = moving.compute_world_bounds()
        other_bounds = other.compute_world_bounds()
        if moving_bounds is None or other_bounds is None:
            return None

        allowed_axes = self._collision_separation_axes(moving)
        if not allowed_axes:
            return None
        overlap_x, overlap_y = self._xy_overlap_depths(moving_bounds, other_bounds)
        axes = sorted(allowed_axes, key=lambda axis: (overlap_x, overlap_y)[axis])
        old_translation = np.asarray(moving.transform.translation(), dtype=float)
        other_translation = np.asarray(other.transform.translation(), dtype=float)
        # Drake reports mesh penetration, while the conservative AABB used to
        # reject new conflicts can overlap farther along the chosen axis.  The
        # larger value guarantees that this candidate clears both signals.

        for axis in axes:
            separation = max(penetration, (overlap_x, overlap_y)[axis]) + clearance
            delta = old_translation[axis] - other_translation[axis]
            signs = (-1.0, 1.0) if abs(delta) < 1e-4 else (1.0 if delta > 0 else -1.0,)
            for sign in signs:
                translation = old_translation.copy()
                translation[axis] += sign * separation
                candidate = RigidTransform(R=moving.transform.rotation(), p=translation)
                candidate = self._fit_transform_inside_room(moving, candidate)
                if not self._transform_inside_exact_room(moving, candidate):
                    continue
                actual_shift = float(
                    candidate.translation()[axis] - old_translation[axis]
                )
                if abs(actual_shift) < separation * 0.5:
                    continue
                after_pairs = self._furniture_aabb_overlap_pairs(
                    overrides={str(moving.object_id): candidate}
                )
                if after_pairs - before_pairs:
                    continue
                return candidate
        return None

    def _collision_separation_axes(self, obj: SceneObject) -> tuple[int, ...]:
        """Keep wall-backed objects on their wall by moving along its tangent."""
        bounds = obj.compute_world_bounds()
        room_bounds = self._room_bounds_xy()
        if bounds is None or room_bounds is None:
            return (0, 1)
        lower, upper = bounds
        min_x, min_y, max_x, max_y = room_bounds
        anchor_distance = max(
            0.0,
            float(self._repair_cfg_value("wall_anchor_preservation_distance_m", 0.16)),
        )
        x_wall = min(abs(float(lower[0]) - min_x), abs(max_x - float(upper[0])))
        y_wall = min(abs(float(lower[1]) - min_y), abs(max_y - float(upper[1])))
        if x_wall <= anchor_distance and y_wall > anchor_distance:
            return (1,)
        if y_wall <= anchor_distance and x_wall > anchor_distance:
            return (0,)
        if x_wall <= anchor_distance and y_wall <= anchor_distance:
            return ()
        return (0, 1)

    def _furniture_aabb_overlap_pairs(
        self,
        *,
        overrides: dict[str, RigidTransform] | None = None,
    ) -> set[frozenset[str]]:
        """Return AABB-overlap pairs for rejecting candidates with new conflicts."""
        if self.scene is None:
            return set()
        overrides = overrides or {}
        furniture: list[tuple[str, SceneObject, tuple[np.ndarray, np.ndarray]]] = []
        for object_id, obj in self.scene.objects.items():
            if (
                getattr(obj, "immutable", False)
                or getattr(obj, "object_type", None) != ObjectType.FURNITURE
            ):
                continue
            transform = overrides.get(str(object_id))
            bounds = (
                self._bounds_for_transform(obj, transform)
                if transform is not None
                else obj.compute_world_bounds()
            )
            if bounds is not None:
                furniture.append((str(object_id), obj, bounds))
        pairs: set[frozenset[str]] = set()
        for index, (first_id, _first, first_bounds) in enumerate(furniture):
            for second_id, _second, second_bounds in furniture[index + 1 :]:
                overlap_x, overlap_y = self._xy_overlap_depths(
                    first_bounds, second_bounds
                )
                first_lower, first_upper = first_bounds
                second_lower, second_upper = second_bounds
                overlap_z = max(
                    0.0,
                    float(
                        min(first_upper[2], second_upper[2])
                        - max(first_lower[2], second_lower[2])
                    ),
                )
                if overlap_x > 1e-4 and overlap_y > 1e-4 and overlap_z > 1e-4:
                    pairs.add(frozenset((first_id, second_id)))
        return pairs

    def _replace_geometry_failed_furniture_assets(self, reasons: str) -> int:
        """Replace required furniture whose SDF/mesh cannot be loaded by Drake."""
        if self.scene is None:
            return 0

        configured_categories = list(self._repair_required_counts())
        categories: list[str] = []
        for category in configured_categories:
            if category in reasons:
                categories.append(category)
        if "closet" in reasons or "armoire" in reasons:
            categories.append("wardrobe")
        categories = list(dict.fromkeys(categories))
        if not categories:
            return 0

        replaced = 0
        for category in categories:
            current_objects = list(self._furniture_by_category(category))
            if not current_objects:
                continue
            self._remember_geometry_failed_assets(current_objects)
            failed_signatures = self._geometry_failed_asset_signatures()
            replacement_signatures: set[str] = set()
            for old_obj in current_objects:
                replacement = self._get_or_generate_repair_asset(
                    category,
                    exclude_asset_signatures=failed_signatures | replacement_signatures,
                )
                if replacement is None:
                    console_logger.warning(
                        "Deterministic repair could not replace geometry-failed %s %s",
                        category,
                        old_obj.object_id,
                    )
                    continue
                old_id = old_obj.object_id
                self.scene.remove_object(old_id)
                if self._place_repair_asset(category, replacement):
                    replacement_signatures.update(
                        self._asset_signature_values(replacement)
                    )
                    console_logger.info(
                        "Deterministic repair replaced geometry-failed %s %s",
                        category,
                        old_id,
                    )
                    replaced += 1
                else:
                    # If placement failed, restore the original object so repair does
                    # not make the candidate worse.
                    self.scene.add_object(old_obj)
        return replaced

    def _geometry_failed_asset_signatures(self) -> set[str]:
        signatures = getattr(self, "_geometry_failed_repair_asset_signatures", None)
        if signatures is None:
            signatures = set()
            self._geometry_failed_repair_asset_signatures = signatures
        return signatures

    def _remember_geometry_failed_assets(self, objects: list[SceneObject]) -> None:
        signatures = self._geometry_failed_asset_signatures()
        for obj in objects:
            signatures.update(self._asset_signature_values(obj))

    def _asset_signature_values(self, asset: SceneObject) -> set[str]:
        signatures: set[str] = set()
        for attr in ("sdf_path", "geometry_path"):
            value = getattr(asset, attr, None)
            if value:
                signatures.add(f"{attr}:{Path(value)}")
        metadata = getattr(asset, "metadata", {}) or {}
        hssd_mesh_id = metadata.get("hssd_mesh_id")
        if hssd_mesh_id:
            signatures.add(f"hssd_mesh_id:{hssd_mesh_id}")
        asset_source = metadata.get("asset_source")
        if asset_source and hssd_mesh_id:
            signatures.add(f"source_mesh:{asset_source}:{hssd_mesh_id}")
        return signatures

    def _asset_matches_excluded_signature(
        self,
        asset: SceneObject,
        excluded: set[str],
    ) -> bool:
        if not excluded:
            return False
        return bool(self._asset_signature_values(asset) & excluded)

    def _repair_cfg_value(self, key: str, default: Any) -> Any:
        safety_cfg = getattr(self.cfg, "furniture_safety_controller", None)
        repair_cfg = getattr(safety_cfg, "deterministic_repair", None)
        if repair_cfg is None:
            return default
        try:
            return repair_cfg.get(key, default)
        except Exception:
            return getattr(repair_cfg, key, default)

    def _category_for_object(self, object_id: Any, obj: SceneObject) -> str | None:
        metadata = getattr(obj, "metadata", {}) or {}
        semantic_name = str(metadata.get("semantic_name") or "").strip().lower()
        if semantic_name in REPAIR_ASSET_SPECS:
            return semantic_name
        name = getattr(obj, "name", "")
        description = getattr(obj, "description", "")
        text = f"{object_id} {name} {description}".lower()
        controller = getattr(self, "furniture_safety_controller", None)
        infer_category = getattr(controller, "infer_object_category", None)
        if callable(infer_category):
            category = infer_category(f"{object_id} {name}")
            if category is None:
                category = infer_category(str(description or ""))
            if category:
                return category
        else:
            category = infer_furniture_object_category(object_id, name, description)
            if category:
                return category
        if "nightstand" in text or "bedside" in text:
            return "nightstand"
        if any(term in text for term in ("wardrobe", "closet", "armoire")):
            return "wardrobe"
        if "bed" in text:
            return "bed"
        return None

    def _furniture_by_category(self, category: str) -> list[SceneObject]:
        if self.scene is None:
            return []
        result: list[SceneObject] = []
        for object_id, obj in self.scene.objects.items():
            if getattr(obj, "immutable", False):
                continue
            object_type = getattr(obj, "object_type", None)
            value = getattr(object_type, "value", object_type)
            if str(value).lower() != "furniture":
                continue
            object_category = self._category_for_object(object_id, obj)
            if object_category == category or furniture_object_category_matches(
                object_id,
                getattr(obj, "name", ""),
                getattr(obj, "description", ""),
                category,
            ):
                result.append(obj)
        return result

    def _required_count(self, category: str) -> int:
        return int(self._repair_required_counts().get(category, 0) or 0)

    def _repair_required_counts(self) -> dict[str, int]:
        """Use TaskCompiler's role-specific inventory when it is available."""
        controller = getattr(self, "furniture_safety_controller", None)
        counts = dict(getattr(controller, "required_counts", {}) or {})
        task_spec = getattr(self.scene, "scene_expert_task_spec", None)
        if not task_spec:
            metadata = getattr(self.scene, "metadata", {}) or {}
            if isinstance(metadata, dict):
                task_spec = metadata.get("scene_expert_task_spec")
        required = (
            task_spec.get("required_large_objects", [])
            if isinstance(task_spec, dict)
            else getattr(task_spec, "required_large_objects", [])
        )
        semantic_counts: dict[str, int] = {}
        for item in required or []:
            category = self._repair_category_for_task_label(item)
            if category in REPAIR_ASSET_SPECS:
                semantic_counts[category] = semantic_counts.get(category, 0) + 1
        intent_constraints = (
            task_spec.get("intent_constraints", [])
            if isinstance(task_spec, dict)
            else getattr(task_spec, "intent_constraints", [])
        )
        for constraint in intent_constraints or []:
            if not isinstance(constraint, dict):
                continue
            selector = constraint.get("subjects") or constraint.get("subject")
            if not isinstance(selector, dict):
                continue
            category = self._repair_category_for_task_label(
                selector.get("category") or ""
            )
            if category not in REPAIR_ASSET_SPECS:
                continue
            try:
                count = int(selector.get("count") or 0)
            except (TypeError, ValueError):
                count = 0
            if count > 0:
                semantic_counts[category] = max(semantic_counts.get(category, 0), count)
        if semantic_counts:
            for generic in ("desk", "chair"):
                specialized = sum(
                    count
                    for category, count in semantic_counts.items()
                    if category.endswith(f"_{generic}")
                )
                if specialized:
                    counts.pop(generic, None)
            counts.update(semantic_counts)
        return counts

    @staticmethod
    def _repair_category_for_task_label(value: Any) -> str:
        """Map TaskCompiler text to a repair category without losing role semantics."""
        text = str(value or "")
        inferred = infer_furniture_category(text)
        if inferred in REPAIR_ASSET_SPECS:
            return inferred
        normalized = re.sub(r"(?<=[a-z])['\u2019]s\b", "", text.lower())
        return re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")

    @staticmethod
    def _category_matches_missing_reason(category: str, reasons: str) -> bool:
        """Match generic hard failures to role-specific inventory categories."""
        if f"missing required {category}" in reasons:
            return True
        return any(
            f"missing required {parent}" in reasons
            for parent in REPAIR_ASSET_SPECS
            if parent != category and furniture_category_satisfies(category, parent)
        )

    def _ensure_required_furniture_asset(self, category: str) -> int:
        required = self._required_count(category)
        if required <= 0:
            return 0
        current = len(self._furniture_by_category(category))
        missing = max(0, required - current)
        if missing <= 0:
            return 0

        added = 0
        for _ in range(missing):
            asset = self._get_or_generate_repair_asset(category)
            if asset is None:
                console_logger.warning(
                    "Deterministic repair could not find or generate %s asset",
                    category,
                )
                break
            if self._place_repair_asset(category, asset):
                added += 1
        return added

    def _get_or_generate_repair_asset(
        self,
        category: str,
        exclude_sdf_paths: set[str] | None = None,
        exclude_asset_signatures: set[str] | None = None,
    ) -> SceneObject | None:
        exclude_sdf_paths = exclude_sdf_paths or set()
        exclude_asset_signatures = set(exclude_asset_signatures or set())
        exclude_asset_signatures.update(
            f"sdf_path:{Path(path)}" for path in exclude_sdf_paths
        )
        for asset in self.asset_manager.list_available_assets():
            if self._asset_matches_excluded_signature(asset, exclude_asset_signatures):
                continue
            if (
                self._category_for_object(getattr(asset, "object_id", ""), asset)
                == category
            ):
                return asset

        spec = REPAIR_ASSET_SPECS.get(category)
        if spec is None:
            return None
        description, dimensions = spec

        request = AssetGenerationRequest(
            object_descriptions=[description],
            short_names=[category],
            object_type=ObjectType.FURNITURE,
            desired_dimensions=[dimensions],
            style_context="deterministic repair asset",
            scene_id=(
                self.scene.scene_dir.name if self.scene else "deterministic_repair"
            ),
        )
        result = self.asset_manager.generate_assets(request)
        for asset in result.successful_assets:
            if self._asset_matches_excluded_signature(asset, exclude_asset_signatures):
                console_logger.warning(
                    "Deterministic repair rejected generated %s asset %s because "
                    "it matches a known geometry-failed signature",
                    category,
                    asset.object_id,
                )
                continue
            return asset
        return self._create_placeholder_repair_asset(category, dimensions)

    def _create_placeholder_repair_asset(
        self,
        category: str,
        dimensions: list[float],
    ) -> SceneObject | None:
        if self.scene is None:
            return None
        try:
            repair_root = (
                self.scene.scene_dir
                / "generated_assets"
                / "furniture"
                / "repair_placeholders"
                / f"{category}_{int(time.time() * 1000)}"
            )
            repair_root.mkdir(parents=True, exist_ok=True)
            width, depth, height = [float(v) for v in dimensions]
            # generate_drake_sdf expects the visual and collision meshes in
            # glTF's Y-up frame. Encode the SceneSmith depth/height axes as
            # glTF Z/Y so the SDF exporter converts them back to X/Y/Z.
            mesh = trimesh.creation.box(extents=[width, height, depth])
            mesh.apply_translation([0.0, height / 2.0, 0.0])
            gltf_path = repair_root / f"{category}_placeholder.gltf"
            sdf_path = repair_root / f"{category}_placeholder.sdf"
            mesh.export(gltf_path)
            if category == "rug":
                # Rugs are floor coverings, not rigid furniture.  Keep the
                # visual mesh in the scene but omit collision geometry so a
                # fallback rug cannot collide with walls or furniture merely
                # because all retrieved rug meshes were invalid.
                generate_thin_covering_sdf(
                    visual_mesh_path=gltf_path,
                    output_path=sdf_path,
                    model_name=f"{category}_placeholder",
                )
            else:
                physics = MeshPhysicsAnalysis(
                    up_axis="+Z",
                    front_axis="+Y",
                    material="wood",
                    mass_kg=max(1.0, width * depth * height * 35.0),
                    mass_range_kg=(1.0, max(1.0, width * depth * height * 50.0)),
                )
                generate_drake_sdf(
                    visual_mesh_path=gltf_path,
                    collision_pieces=[mesh.copy()],
                    physics_analysis=physics,
                    output_path=sdf_path,
                    asset_name=f"{category}_placeholder",
                )
            object_id = self.asset_manager.registry.generate_unique_id(
                f"{category}_repair_placeholder"
            )
            placeholder = SceneObject(
                object_id=object_id,
                object_type=ObjectType.FURNITURE,
                name=category,
                description=f"deterministic placeholder {category}",
                transform=RigidTransform(),
                geometry_path=gltf_path,
                sdf_path=sdf_path,
                bbox_min=np.asarray([-width / 2.0, -depth / 2.0, 0.0], dtype=float),
                bbox_max=np.asarray([width / 2.0, depth / 2.0, height], dtype=float),
                metadata={
                    "asset_source": (
                        "thin_covering"
                        if category == "rug"
                        else "deterministic_placeholder"
                    ),
                    "repair_placeholder": True,
                    "generation_timestamp": time.time(),
                    **(
                        {
                            "width_m": width,
                            "depth_m": depth,
                            "shape": "rectangular",
                            "is_wall_covering": False,
                        }
                        if category == "rug"
                        else {}
                    ),
                },
            )
            self.asset_manager.registry.register(placeholder)
            console_logger.warning(
                "Deterministic repair created placeholder %s asset %s after "
                "available assets were missing or geometry-failed",
                category,
                placeholder.object_id,
            )
            return placeholder
        except Exception:
            console_logger.exception(
                "Deterministic repair failed creating placeholder %s asset",
                category,
            )
            return None

    def _place_repair_asset(self, category: str, asset: SceneObject) -> bool:
        if self.scene is None:
            return False
        x, y, yaw = self._default_repair_pose(category)
        try:
            scene_object = copy_scene_object_with_new_pose(
                scene=self.scene,
                original=asset,
                x=x,
                y=y,
                z=0.0,
                roll=0.0,
                pitch=0.0,
                yaw=math.radians(yaw),
            )
            transform = self._grounded_transform(scene_object, x=x, y=y, yaw_deg=yaw)
            transform = self._fit_transform_inside_room(scene_object, transform)
            if self._is_polygon_room() or category not in (
                "bed",
                "nightstand",
                "wardrobe",
                "twin_bed",
            ):
                transform = self._best_generic_repair_transform(
                    scene_object,
                    fallback=transform,
                )
            if not room_geometry_covers_object(
                self.scene.room_geometry,
                scene_object,
                transform=transform,
            ):
                console_logger.warning(
                    "Deterministic repair could not place %s inside the exact "
                    "polygon footprint",
                    category,
                )
                return False
            scene_object.transform = transform
            self.scene.add_object(scene_object)
            console_logger.info(
                "Deterministic repair placed %s asset %s as %s",
                category,
                asset.object_id,
                scene_object.object_id,
            )
            return True
        except Exception:
            console_logger.exception("Deterministic repair failed placing %s", category)
            return False

    def _best_generic_repair_transform(
        self,
        obj: SceneObject,
        *,
        fallback: RigidTransform,
    ) -> RigidTransform:
        """Choose a low-overlap in-bounds pose for non-bedroom repair assets."""
        room_bounds = self._room_bounds_xy()
        if room_bounds is None or self.scene is None:
            return fallback
        min_x, min_y, max_x, max_y = room_bounds
        fractions = (0.12, 0.30, 0.50, 0.70, 0.88)
        zones = self._opening_forbidden_zones(include_windows=False)
        best = fallback
        best_penalty = float("inf")
        for fx in fractions:
            for fy in fractions:
                x = min_x + (max_x - min_x) * fx
                y = min_y + (max_y - min_y) * fy
                for yaw in (0.0, 90.0):
                    candidate = self._grounded_transform(obj, x=x, y=y, yaw_deg=yaw)
                    candidate = self._fit_transform_inside_room(obj, candidate)
                    if not room_geometry_covers_object(
                        self.scene.room_geometry,
                        obj,
                        transform=candidate,
                    ):
                        continue
                    bounds = self._bounds_for_transform(obj, candidate)
                    if bounds is None:
                        continue
                    penalty = self._zone_overlap_penalty(bounds, zones)
                    for existing in self.scene.objects.values():
                        if getattr(existing, "immutable", False):
                            continue
                        existing_type = getattr(existing, "object_type", None)
                        existing_value = getattr(existing_type, "value", existing_type)
                        if str(existing_value).lower() != "furniture":
                            continue
                        try:
                            existing_bounds = existing.compute_world_bounds()
                        except Exception as exc:
                            console_logger.warning(
                                "Skipping invalid obstacle %s while placing %s: %s",
                                getattr(existing, "object_id", "unknown"),
                                getattr(obj, "object_id", "repair_asset"),
                                exc,
                            )
                            continue
                        if existing_bounds is None:
                            continue
                        overlap_x, overlap_y = self._xy_overlap_depths(
                            bounds,
                            existing_bounds,
                        )
                        penalty += overlap_x * overlap_y * 1000.0
                    if penalty < best_penalty:
                        best = candidate
                        best_penalty = penalty
                    if penalty <= 1e-6:
                        return candidate
        return best

    def _default_repair_pose(self, category: str) -> tuple[float, float, float]:
        room_bounds = self._room_bounds_xy()
        if room_bounds is None:
            return 0.0, 0.0, 0.0
        min_x, min_y, max_x, max_y = room_bounds
        if category == "wardrobe":
            return max_x - 0.5, max_y - 0.6, 180.0
        if category == "nightstand":
            return min_x + 0.8, min_y + 0.8, 0.0
        plan = build_bedroom_anchor_plan(self.scene, self._bedroom_layout_cfg())
        wall = plan.bed_head_wall if plan else "north"
        return 0.0, 0.0, self._yaw_for_head_wall(wall)

    def _anchor_existing_bed(self) -> bool:
        beds = self._furniture_by_category("bed")
        if not beds or self.scene is None:
            return False
        bed = beds[0]
        plan = build_bedroom_anchor_plan(self.scene, self._bedroom_layout_cfg())
        wall = plan.bed_head_wall if plan and plan.bed_head_wall else "north"
        yaw = self._yaw_for_head_wall(wall)
        current = np.asarray(bed.transform.translation(), dtype=float)
        transform = self._grounded_transform(
            bed, x=float(current[0]), y=float(current[1]), yaw_deg=yaw
        )
        transform = self._snap_transform_to_wall(bed, transform, wall)
        transform = self._fit_transform_inside_room(bed, transform)
        if self._transform_close(bed.transform, transform):
            return False
        return self._move_object_if_repair_safe(bed, transform)

    def _repair_bedside_nightstands(self) -> bool:
        beds = self._furniture_by_category("bed")
        if not beds:
            return False
        needed = self._required_count("nightstand")
        if needed > len(self._furniture_by_category("nightstand")):
            self._ensure_required_furniture_asset("nightstand")
        nightstands = self._furniture_by_category("nightstand")[:2]
        if len(nightstands) < 2:
            return False

        bed = beds[0]
        bed_dims = self._local_size(bed, [1.60, 2.05, 0.80])
        bed_center = np.asarray(bed.transform.translation(), dtype=float)
        rotation = np.asarray(bed.transform.rotation().matrix(), dtype=float)
        lateral = rotation @ np.array([1.0, 0.0, 0.0])
        # Bed assets point +Y toward the foot; bedside furniture belongs at
        # the opposite (headboard) end.
        head = -(rotation @ np.array([0.0, 1.0, 0.0]))
        yaw = math.degrees(RollPitchYaw(bed.transform.rotation()).yaw_angle())
        gap = float(self._repair_cfg_value("nightstand_gap_m", 0.08))

        changed = False
        for side, nightstand in zip((-1.0, 1.0), nightstands):
            ns_dims = self._local_size(nightstand, [0.45, 0.42, 0.55])
            target = (
                bed_center
                + side * lateral * (bed_dims[0] / 2 + ns_dims[0] / 2 + gap)
                + head * max(0.0, bed_dims[1] / 2 - ns_dims[1] / 2 - 0.10)
            )
            transform = self._grounded_transform(
                nightstand,
                x=float(target[0]),
                y=float(target[1]),
                yaw_deg=yaw,
            )
            transform = self._fit_transform_inside_room(nightstand, transform)
            if not self._transform_close(nightstand.transform, transform):
                changed = (
                    self._move_object_if_repair_safe(nightstand, transform) or changed
                )
        return changed

    def _repair_wardrobe_wall_anchor(self) -> bool:
        wardrobes = self._furniture_by_category("wardrobe")
        if not wardrobes or self.scene is None:
            return False
        wardrobe = wardrobes[0]
        room_bounds = self._room_bounds_xy()
        if room_bounds is None:
            return False
        candidates = self._wardrobe_candidate_transforms(wardrobe)
        forbidden_zones = self._opening_forbidden_zones(include_windows=False)
        obstacles = self._furniture_by_category("bed") + self._furniture_by_category(
            "nightstand"
        )
        best_transform = None
        best_score = -1e9
        for transform, wall_opening_penalty in candidates:
            bounds = self._bounds_for_transform(wardrobe, transform)
            if bounds is None:
                continue
            # A coarse wall-level opening penalty is insufficient near corners:
            # furniture anchored to an adjacent wall can still intersect the
            # actual door/open-connection clearance volume.  Treat that geometry
            # as a hard candidate filter so this repair cannot undo the generic
            # forbidden-zone repair that runs immediately before it.
            if self._zone_overlap_penalty(bounds, forbidden_zones) > 1e-6:
                continue
            overlap_penalty = 0.0
            for obstacle in obstacles:
                obstacle_bounds = obstacle.compute_world_bounds()
                if obstacle_bounds is None:
                    continue
                overlap_x, overlap_y = self._xy_overlap_depths(bounds, obstacle_bounds)
                overlap_penalty += overlap_x * overlap_y * 100.0
            center = np.asarray(transform.translation(), dtype=float)
            bed_center = (
                np.asarray(obstacles[0].transform.translation(), dtype=float)
                if obstacles
                else np.zeros(3)
            )
            distance_score = float(np.linalg.norm(center[:2] - bed_center[:2]))
            score = distance_score - overlap_penalty - wall_opening_penalty
            # Hard collisions must never win merely because the candidate is
            # farther from the bed. If no fully valid wall candidate exists,
            # retain the current pose for a later generic repair instead of
            # introducing a known hard failure.
            if overlap_penalty > 1e-5:
                continue
            if score > best_score:
                best_score = score
                best_transform = transform

        if best_transform is None or self._transform_close(
            wardrobe.transform, best_transform
        ):
            return False
        return self._move_object_if_repair_safe(wardrobe, best_transform)

    def _repair_dresser_opposite_bed_wall_anchor(self) -> bool:
        """Back the dresser against the wall faced by the foot of the bed."""
        dressers = self._furniture_by_category("dresser")
        beds = self._furniture_by_category("bed")
        if not dressers or not beds or self.scene is None:
            return False

        dresser = dressers[0]
        bed = beds[0]
        plan = build_bedroom_anchor_plan(self.scene, self._bedroom_layout_cfg())
        head_wall = plan.bed_head_wall if plan and plan.bed_head_wall else "north"
        opposite_wall = {
            "north": "south",
            "south": "north",
            "east": "west",
            "west": "east",
        }.get(head_wall, "south")
        bed_center = np.asarray(bed.transform.translation(), dtype=float)
        x = float(bed_center[0])
        y = float(bed_center[1])
        transform = self._grounded_transform(
            dresser,
            x=x,
            y=y,
            yaw_deg=self._yaw_for_inward_wall(opposite_wall),
        )
        transform = self._snap_transform_to_wall(dresser, transform, opposite_wall)
        transform = self._fit_transform_inside_room(dresser, transform)
        if self._transform_close(dresser.transform, transform):
            return False
        return self._move_object_if_repair_safe(dresser, transform)

    def _prompt_requires_wardrobe_next_to_dresser(self) -> bool:
        if self.scene is None:
            return False
        text = str(
            getattr(self.scene, "scene_expert_original_description", "")
            or getattr(self.scene, "text_description", "")
            or ""
        ).lower()
        wardrobe = r"(?:wardrobe|closet|armoire)"
        dresser = r"(?:dresser|chest\s+of\s+drawers)"
        return bool(
            re.search(
                rf"{wardrobe}.{{0,50}}(?:next|adjacent)\s+to.{{0,30}}{dresser}", text
            )
            or re.search(
                rf"{dresser}.{{0,50}}(?:next|adjacent)\s+to.{{0,30}}{wardrobe}", text
            )
        )

    def _repair_wardrobe_next_to_dresser(self) -> bool:
        wardrobes = self._furniture_by_category("wardrobe")
        dressers = self._furniture_by_category("dresser")
        beds = self._furniture_by_category("bed")
        if not wardrobes or not dressers or not beds or self.scene is None:
            return False
        wardrobe = wardrobes[0]
        dresser = dressers[0]
        dresser_bounds = dresser.compute_world_bounds()
        if dresser_bounds is None:
            return False

        plan = build_bedroom_anchor_plan(self.scene, self._bedroom_layout_cfg())
        head_wall = plan.bed_head_wall if plan and plan.bed_head_wall else "north"
        wall = {
            "north": "south",
            "south": "north",
            "east": "west",
            "west": "east",
        }.get(head_wall, "south")
        dresser_min, dresser_max = dresser_bounds
        wardrobe_size = self._local_size(wardrobe, [0.90, 0.55, 2.00])
        gap = float(self._repair_cfg_value("storage_pair_gap_m", 0.08))
        dresser_center = np.asarray(dresser.transform.translation(), dtype=float)
        candidates: list[RigidTransform] = []
        if wall in ("north", "south"):
            for x in (
                float(dresser_min[0]) - wardrobe_size[0] / 2.0 - gap,
                float(dresser_max[0]) + wardrobe_size[0] / 2.0 + gap,
            ):
                candidates.append(
                    self._grounded_transform(
                        wardrobe,
                        x=x,
                        y=float(dresser_center[1]),
                        yaw_deg=self._yaw_for_inward_wall(wall),
                    )
                )
        else:
            for y in (
                float(dresser_min[1]) - wardrobe_size[1] / 2.0 - gap,
                float(dresser_max[1]) + wardrobe_size[1] / 2.0 + gap,
            ):
                candidates.append(
                    self._grounded_transform(
                        wardrobe,
                        x=float(dresser_center[0]),
                        y=y,
                        yaw_deg=self._yaw_for_inward_wall(wall),
                    )
                )

        obstacles = self._furniture_by_category("bed") + self._furniture_by_category(
            "nightstand"
        )
        original = np.asarray(wardrobe.transform.translation(), dtype=float)
        best_transform = None
        best_score = -1e18
        for candidate in candidates:
            candidate = self._snap_transform_to_wall(wardrobe, candidate, wall)
            candidate = self._fit_transform_inside_room(wardrobe, candidate)
            bounds = self._bounds_for_transform(wardrobe, candidate)
            if bounds is None:
                continue
            overlap_penalty = 0.0
            for obstacle in obstacles:
                obstacle_bounds = obstacle.compute_world_bounds()
                if obstacle_bounds is None:
                    continue
                overlap_x, overlap_y = self._xy_overlap_depths(bounds, obstacle_bounds)
                overlap_penalty += overlap_x * overlap_y * 500.0
            move_penalty = float(
                np.linalg.norm(np.asarray(candidate.translation())[:2] - original[:2])
            )
            score = -overlap_penalty - move_penalty
            if score > best_score:
                best_score = score
                best_transform = candidate
        if best_transform is None or self._transform_close(
            wardrobe.transform, best_transform
        ):
            return False
        return self._move_object_if_repair_safe(wardrobe, best_transform)

    def _repair_forbidden_zone_conflicts(self, include_windows: bool = False) -> bool:
        """Move objects out of door/opening clearance zones using generic anchors."""
        if self.scene is None:
            return False
        zones = self._opening_forbidden_zones(include_windows=include_windows)
        if not zones:
            return False
        blockers = self._objects_overlapping_zones(zones)
        if not blockers:
            return False

        changed = False
        # Move less-central storage first. Beds/nightstands get their bedroom
        # relation repair before this method runs, so they are only moved if they
        # still block a hard opening zone.
        category_priority = {"wardrobe": 0, "nightstand": 1, "bed": 2}
        blockers.sort(
            key=lambda item: (
                category_priority.get(
                    self._category_for_object(item[0], item[1]) or "", 9
                ),
                -item[2],
            )
        )
        for object_id, obj, original_penalty in blockers:
            transform = self._best_forbidden_zone_repair_transform(obj, zones)
            if transform is None:
                continue
            new_penalty = self._zone_overlap_penalty_for_transform(
                obj, transform, zones
            )
            if new_penalty + 1e-5 >= original_penalty:
                continue
            if not self._move_object_if_repair_safe(obj, transform):
                continue
            console_logger.info(
                "Deterministic forbidden-zone repair moved %s from penalty %.4f to %.4f",
                object_id,
                original_penalty,
                new_penalty,
            )
            changed = True
        return changed

    def _opening_forbidden_zones(
        self, include_windows: bool = False
    ) -> list[tuple[str, str, np.ndarray, np.ndarray]]:
        if self.scene is None or self.scene.room_geometry is None:
            return []
        zones: list[tuple[str, str, np.ndarray, np.ndarray]] = []
        for opening in list(getattr(self.scene.room_geometry, "openings", []) or []):
            opening_type_raw = getattr(opening, "opening_type", "")
            opening_type = str(
                getattr(opening_type_raw, "value", opening_type_raw)
            ).lower()
            if opening_type not in ("door", "open") and not (
                include_windows and opening_type == "window"
            ):
                continue
            bounds = self._opening_clearance_bounds(opening)
            if bounds is None:
                continue
            zone_min, zone_max = bounds
            zones.append(
                (
                    str(getattr(opening, "opening_id", f"{opening_type}_{len(zones)}")),
                    opening_type,
                    zone_min,
                    zone_max,
                )
            )
        return zones

    def _opening_clearance_bounds(
        self, opening: Any
    ) -> tuple[np.ndarray, np.ndarray] | None:
        zone_min = getattr(opening, "clearance_bbox_min", None)
        zone_max = getattr(opening, "clearance_bbox_max", None)
        if zone_min is not None and zone_max is not None:
            return np.asarray(zone_min, dtype=float), np.asarray(zone_max, dtype=float)

            opening_type_raw = getattr(opening, "opening_type", "")
            opening_type = str(
                getattr(opening_type_raw, "value", opening_type_raw)
            ).lower()
        if opening_type != "open":
            return None
        try:
            wall_direction_raw = getattr(opening, "wall_direction", "")
            wall_direction = str(
                getattr(wall_direction_raw, "value", wall_direction_raw)
            ).lower()
            center = np.asarray(getattr(opening, "center_world"), dtype=float)
            width = float(getattr(opening, "width"))
            clearance_cfg = getattr(self.cfg, "clearance_zones", None)
            passage = float(getattr(clearance_cfg, "passage_size", 0.8))
            depth = float(getattr(clearance_cfg, "open_connection_clearance", 1.0))
            half_width = max(width, passage) / 2.0
            min_x = max_x = float(center[0])
            min_y = max_y = float(center[1])
            if wall_direction in ("north", "south"):
                min_x = float(center[0]) - half_width
                max_x = float(center[0]) + half_width
                if wall_direction == "north":
                    min_y = float(center[1]) - depth
                    max_y = float(center[1])
                else:
                    min_y = float(center[1])
                    max_y = float(center[1]) + depth
            else:
                min_y = float(center[1]) - half_width
                max_y = float(center[1]) + half_width
                if wall_direction == "east":
                    min_x = float(center[0]) - depth
                    max_x = float(center[0])
                else:
                    min_x = float(center[0])
                    max_x = float(center[0]) + depth
            return (
                np.asarray([min_x, min_y, 0.0], dtype=float),
                np.asarray([max_x, max_y, 2.5], dtype=float),
            )
        except Exception:
            return None

    def _objects_overlapping_zones(
        self, zones: list[tuple[str, str, np.ndarray, np.ndarray]]
    ) -> list[tuple[str, SceneObject, float]]:
        if self.scene is None:
            return []
        blockers: list[tuple[str, SceneObject, float]] = []
        for object_id, obj in self.scene.objects.items():
            if getattr(obj, "immutable", False):
                continue
            if getattr(obj, "object_type", None) in (ObjectType.WALL, ObjectType.FLOOR):
                continue
            if (getattr(obj, "metadata", {}) or {}).get(
                "asset_source"
            ) == "thin_covering":
                continue
            bounds = obj.compute_world_bounds()
            if bounds is None:
                continue
            penalty = self._zone_overlap_penalty(bounds, zones)
            if penalty > 1e-6:
                blockers.append((str(object_id), obj, penalty))
        return blockers

    def _zone_overlap_penalty(
        self,
        bounds: tuple[np.ndarray, np.ndarray],
        zones: list[tuple[str, str, np.ndarray, np.ndarray]],
    ) -> float:
        penalty = 0.0
        obj_min, obj_max = bounds
        for _, zone_type, zone_min, zone_max in zones:
            overlap_x = min(float(obj_max[0]), float(zone_max[0])) - max(
                float(obj_min[0]), float(zone_min[0])
            )
            overlap_y = min(float(obj_max[1]), float(zone_max[1])) - max(
                float(obj_min[1]), float(zone_min[1])
            )
            if overlap_x > 0.0 and overlap_y > 0.0:
                weight = 1000.0 if zone_type in ("door", "open") else 150.0
                penalty += overlap_x * overlap_y * weight
        return penalty

    def _zone_overlap_penalty_for_transform(
        self,
        obj: SceneObject,
        transform: RigidTransform,
        zones: list[tuple[str, str, np.ndarray, np.ndarray]],
    ) -> float:
        bounds = self._bounds_for_transform(obj, transform)
        if bounds is None:
            return 1e9
        return self._zone_overlap_penalty(bounds, zones)

    def _best_forbidden_zone_repair_transform(
        self,
        obj: SceneObject,
        zones: list[tuple[str, str, np.ndarray, np.ndarray]],
    ) -> RigidTransform | None:
        candidates = self._generic_wall_candidate_transforms(obj)
        if not candidates:
            return None
        obstacles = [
            other
            for other in self._furniture_by_category("bed")
            + self._furniture_by_category("nightstand")
            + self._furniture_by_category("wardrobe")
            if other.object_id != obj.object_id
        ]
        best_transform = None
        best_score = -1e18
        original_center = np.asarray(obj.transform.translation(), dtype=float)
        for transform in candidates:
            bounds = self._bounds_for_transform(obj, transform)
            if bounds is None:
                continue
            zone_penalty = self._zone_overlap_penalty(bounds, zones)
            overlap_penalty = 0.0
            for obstacle in obstacles:
                obstacle_bounds = obstacle.compute_world_bounds()
                if obstacle_bounds is None:
                    continue
                overlap_x, overlap_y = self._xy_overlap_depths(bounds, obstacle_bounds)
                overlap_penalty += overlap_x * overlap_y * 400.0
            center = np.asarray(transform.translation(), dtype=float)
            move_penalty = (
                float(np.linalg.norm(center[:2] - original_center[:2])) * 0.15
            )
            wall_bonus = 0.25
            score = wall_bonus - zone_penalty - overlap_penalty - move_penalty
            if score > best_score:
                best_score = score
                best_transform = transform
        return best_transform

    def _generic_wall_candidate_transforms(
        self, obj: SceneObject
    ) -> list[RigidTransform]:
        room_bounds = self._room_bounds_xy()
        if room_bounds is None:
            return []
        min_x, min_y, max_x, max_y = room_bounds
        margin = float(self._repair_cfg_value("wall_margin_m", 0.08))
        candidates: list[tuple[str, float, float, float]] = []
        for wall in ("north", "south"):
            y = max_y - margin if wall == "north" else min_y + margin
            for x in (min_x + 0.65, 0.0, max_x - 0.65):
                candidates.append((wall, x, y, self._yaw_for_inward_wall(wall)))
        for wall in ("east", "west"):
            x = max_x - margin if wall == "east" else min_x + margin
            for y in (min_y + 0.65, 0.0, max_y - 0.65):
                candidates.append((wall, x, y, self._yaw_for_inward_wall(wall)))

        transforms: list[RigidTransform] = []
        for wall, x, y, yaw in candidates:
            transform = self._grounded_transform(obj, x=x, y=y, yaw_deg=yaw)
            transform = self._snap_transform_to_wall(obj, transform, wall)
            transform = self._fit_transform_inside_room(obj, transform)
            transforms.append(transform)
        return transforms

    def _wardrobe_candidate_transforms(
        self, wardrobe: SceneObject
    ) -> list[tuple[RigidTransform, float]]:
        room_bounds = self._room_bounds_xy()
        if room_bounds is None:
            return []
        min_x, min_y, max_x, max_y = room_bounds
        plan = build_bedroom_anchor_plan(self.scene, self._bedroom_layout_cfg())
        wall_openings = plan.wall_openings if plan else {}
        margin = 0.08
        candidates: list[tuple[str, float, float, float]] = []
        for wall in ("north", "south"):
            y = max_y - margin if wall == "north" else min_y + margin
            for x in (min_x + 0.7, 0.0, max_x - 0.7):
                candidates.append((wall, x, y, self._yaw_for_inward_wall(wall)))
        for wall in ("east", "west"):
            x = max_x - margin if wall == "east" else min_x + margin
            for y in (min_y + 0.7, 0.0, max_y - 0.7):
                candidates.append((wall, x, y, self._yaw_for_inward_wall(wall)))

        transforms: list[tuple[RigidTransform, float]] = []
        for wall, x, y, yaw in candidates:
            transform = self._grounded_transform(wardrobe, x=x, y=y, yaw_deg=yaw)
            transform = self._snap_transform_to_wall(wardrobe, transform, wall)
            transform = self._fit_transform_inside_room(wardrobe, transform)
            opening_penalty = 5.0 if wall_openings.get(wall) else 0.0
            transforms.append((transform, opening_penalty))
        return transforms

    def _bedroom_layout_cfg(self) -> Any:
        safety_cfg = getattr(self.cfg, "furniture_safety_controller", None)
        return getattr(safety_cfg, "bedroom_layout", None)

    def _is_polygon_room(self) -> bool:
        """Return whether the current room uses an exact polygon footprint."""
        scene = getattr(self, "scene", None)
        if scene is None:
            return False
        room_geometry = getattr(scene, "room_geometry", None)
        return isinstance(
            getattr(room_geometry, "footprint_vertices", None),
            (list, tuple),
        )

    def _room_bounds_xy(self) -> tuple[float, float, float, float] | None:
        if self.scene is None or self.scene.room_geometry is None:
            return None
        length = float(getattr(self.scene.room_geometry, "length", 0.0) or 0.0)
        width = float(getattr(self.scene.room_geometry, "width", 0.0) or 0.0)
        if length <= 0 or width <= 0:
            return None
        return (-length / 2, -width / 2, length / 2, width / 2)

    def _local_size(self, obj: SceneObject, default: list[float]) -> np.ndarray:
        if obj.bbox_min is None or obj.bbox_max is None:
            return np.asarray(default, dtype=float)
        return np.abs(
            np.asarray(obj.bbox_max, dtype=float)
            - np.asarray(obj.bbox_min, dtype=float)
        )

    def _grounded_transform(
        self, obj: SceneObject, *, x: float, y: float, yaw_deg: float
    ) -> RigidTransform:
        transform = RigidTransform(
            rpy=RollPitchYaw(0.0, 0.0, math.radians(yaw_deg)),
            p=[x, y, 0.0],
        )
        furniture_tools = getattr(self, "furniture_tools", None)
        if furniture_tools is not None:
            transform, _ = furniture_tools._ground_transform_to_floor_if_needed(
                scene_obj=obj,
                transform=transform,
            )
        return transform

    def _bounds_for_transform(
        self, obj: SceneObject, transform: RigidTransform
    ) -> tuple[np.ndarray, np.ndarray] | None:
        furniture_tools = getattr(self, "furniture_tools", None)
        if furniture_tools is not None:
            return furniture_tools._world_bounds_for_transform(obj, transform)
        old_transform = obj.transform
        obj.transform = transform
        try:
            return obj.compute_world_bounds()
        finally:
            obj.transform = old_transform

    def _snap_transform_to_wall(
        self, obj: SceneObject, transform: RigidTransform, wall: str
    ) -> RigidTransform:
        room_bounds = self._room_bounds_xy()
        bounds = self._bounds_for_transform(obj, transform)
        if room_bounds is None or bounds is None:
            return transform
        min_x, min_y, max_x, max_y = room_bounds
        world_min, world_max = bounds
        margin = float(self._repair_cfg_value("wall_margin_m", 0.08))
        translation = np.asarray(transform.translation(), dtype=float).copy()
        if wall == "north":
            translation[1] += max_y - margin - float(world_max[1])
        elif wall == "south":
            translation[1] += min_y + margin - float(world_min[1])
        elif wall == "east":
            translation[0] += max_x - margin - float(world_max[0])
        elif wall == "west":
            translation[0] += min_x + margin - float(world_min[0])
        return RigidTransform(R=transform.rotation(), p=translation)

    def _fit_transform_inside_room(
        self, obj: SceneObject, transform: RigidTransform
    ) -> RigidTransform:
        # An AABB clamp cannot project onto a concave polygon.  For polygon rooms
        # the repair is transactional: accept a fully covered candidate or retain
        # the original pose unchanged.
        if self._is_polygon_room():
            return (
                transform
                if self._transform_inside_exact_room(obj, transform)
                else obj.transform
            )
        room_bounds = self._room_bounds_xy()
        bounds = self._bounds_for_transform(obj, transform)
        if room_bounds is None or bounds is None:
            return transform
        min_x, min_y, max_x, max_y = room_bounds
        world_min, world_max = bounds
        margin = max(
            0.03,
            float(self._repair_cfg_value("wall_margin_m", 0.08)),
        )
        translation = np.asarray(transform.translation(), dtype=float).copy()
        if world_min[0] < min_x + margin:
            translation[0] += min_x + margin - float(world_min[0])
        if world_max[0] > max_x - margin:
            translation[0] -= float(world_max[0]) - (max_x - margin)
        if world_min[1] < min_y + margin:
            translation[1] += min_y + margin - float(world_min[1])
        if world_max[1] > max_y - margin:
            translation[1] -= float(world_max[1]) - (max_y - margin)
        return RigidTransform(R=transform.rotation(), p=translation)

    def _transform_inside_exact_room(
        self, obj: SceneObject, transform: RigidTransform
    ) -> bool:
        if self.scene is None:
            return False
        return room_geometry_covers_object(
            self.scene.room_geometry,
            obj,
            transform=transform,
        )

    def _move_object_if_repair_safe(
        self, obj: SceneObject, transform: RigidTransform
    ) -> bool:
        """Commit a repair move only if it adds no polygon/opening/collision failure."""
        if self.scene is None:
            return False
        if not self._transform_inside_exact_room(obj, transform):
            console_logger.info(
                "Rejected deterministic repair for %s: exact polygon containment",
                obj.object_id,
            )
            return False
        zones = self._opening_forbidden_zones(include_windows=False)
        if (
            zones
            and self._zone_overlap_penalty_for_transform(obj, transform, zones) > 1e-6
        ):
            console_logger.info(
                "Rejected deterministic repair for %s: door/opening clearance",
                obj.object_id,
            )
            return False
        before_pairs = self._furniture_aabb_overlap_pairs()
        after_pairs = self._furniture_aabb_overlap_pairs(
            overrides={str(obj.object_id): transform}
        )
        if after_pairs - before_pairs:
            console_logger.info(
                "Rejected deterministic repair for %s: new furniture collision",
                obj.object_id,
            )
            return False
        self.scene.move_object(obj.object_id, transform)
        return True

    def _yaw_for_head_wall(self, wall: str) -> float:
        # The bed tool/render arrow is the foot direction, so it must point
        # inward while the headboard faces the selected wall.
        return {
            "north": 180.0,
            "south": 0.0,
            "east": 90.0,
            "west": -90.0,
        }.get(wall, 0.0)

    def _yaw_for_inward_wall(self, wall: str) -> float:
        return {
            "north": 180.0,
            "south": 0.0,
            "east": 90.0,
            "west": -90.0,
        }.get(wall, 0.0)

    def _xy_overlap_depths(
        self,
        bounds_a: tuple[np.ndarray, np.ndarray],
        bounds_b: tuple[np.ndarray, np.ndarray],
    ) -> tuple[float, float]:
        min_a, max_a = bounds_a
        min_b, max_b = bounds_b
        return (
            max(0.0, float(min(max_a[0], max_b[0]) - max(min_a[0], min_b[0]))),
            max(0.0, float(min(max_a[1], max_b[1]) - max(min_a[1], min_b[1]))),
        )

    def _transform_close(self, a: RigidTransform, b: RigidTransform) -> bool:
        a_t = np.asarray(a.translation(), dtype=float)
        b_t = np.asarray(b.translation(), dtype=float)
        a_yaw = RollPitchYaw(a.rotation()).yaw_angle()
        b_yaw = RollPitchYaw(b.rotation()).yaw_angle()
        return bool(
            np.allclose(a_t, b_t, atol=1e-3)
            and abs(math.atan2(math.sin(a_yaw - b_yaw), math.cos(a_yaw - b_yaw))) < 1e-3
        )

    def _get_extra_critique_kwargs(self) -> dict[str, Any]:
        """Get extra kwargs for critic prompt (reachability context).

        Computes room reachability and formats it for critic context injection.
        This allows the critic to score reachability based on computed metrics.

        Returns:
            Dict with reachability_context and robot_width for prompt template.
        """
        robot_width = self.cfg.reachability.robot_width
        result = compute_reachability(scene=self.scene, robot_width=robot_width)
        reachability_context = format_reachability_for_critic(result)

        return {
            "reachability_context": reachability_context,
            "robot_width": robot_width,
            "reference_adherence_context": self._reference_adherence_context(),
        }

    def _reference_adherence_context(self) -> str:
        """Build a structured, image-free reference report for later critiques."""
        grounding_path = getattr(self, "_grounding_candidate_path", None)
        if grounding_path is None or self.scene is None:
            return ""
        artifact_path = Path(grounding_path).parent / LAYOUT_ARTIFACT_NAME
        try:
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return ""
        normalized = (payload.get("vlm") or {}).get("normalized") or {}
        items = normalized.get("items") or []
        if not items:
            return ""

        expected = Counter()
        for item in items:
            name = str(item.get("furniture_name", ""))
            expected[infer_furniture_category(name) or name.strip().lower()] += 1
        observed = Counter()
        for obj in self.scene.objects.values():
            if getattr(obj, "object_type", None) != ObjectType.FURNITURE:
                continue
            category = infer_furniture_object_category(
                obj.object_id, obj.name, obj.description
            )
            if category:
                observed[category] += 1

        matched = sum(
            min(count, observed.get(category, 0))
            for category, count in expected.items()
        )
        missing = {
            category: count - observed.get(category, 0)
            for category, count in expected.items()
            if observed.get(category, 0) < count
        }
        coverage = matched / max(1, sum(expected.values()))
        audit = {
            "schema_version": 1,
            "source_quality_mode": payload.get("source_quality_mode", "unknown"),
            "inventory_coverage": round(coverage, 4),
            "expected_counts": dict(expected),
            "observed_counts": dict(observed),
            "missing_reference_items": missing,
            "layout_contract": payload.get("layout_contract", ""),
        }
        try:
            write_context_image_quality_report(
                artifact_path.parent / "reference_adherence.json", audit
            )
        except OSError:
            console_logger.debug("Could not persist reference adherence", exc_info=True)
        missing_text = (
            ", ".join(
                f"{category} x{count}" for category, count in sorted(missing.items())
            )
            or "none"
        )
        return (
            f"Reference quality mode: {audit['source_quality_mode']}. "
            f"Detected-reference inventory coverage: {coverage:.0%}; missing: "
            f"{missing_text}.\n{audit['layout_contract']}"
        )
