"""
Stateful furniture agent with natural conversation between persistent agents.

This module implements a furniture placement workflow using persistent
SQLiteSession agents that maintain conversation memory across interactions.
"""

import copy
import json
import logging
import math
import time

from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from agents import Agent, FunctionTool
from omegaconf import DictConfig
from pydrake.all import RigidTransform, RollPitchYaw

from scenesmith.agent_utils.asset_manager import AssetGenerationRequest
from scenesmith.agent_utils.base_stateful_agent import (
    BaseStatefulAgent,
    HardStateEvaluation,
    log_agent_usage,
)
from scenesmith.agent_utils.furniture_functional_layout import (
    choose_functional_anchor_wall,
    format_functional_layout_guidance,
    functional_layout_family,
    furnishable_room_bounds_xy,
)
from scenesmith.agent_utils.furniture_layout_planning import (
    build_bedroom_anchor_plan,
    evaluate_bedroom_layout_plausibility,
    format_bedroom_anchor_guidance,
    is_bedroom_scene,
)
from scenesmith.agent_utils.mesh_physics_analyzer import MeshPhysicsAnalysis
from scenesmith.agent_utils.placement_noise import PlacementNoiseMode
from scenesmith.agent_utils.reachability import (
    compute_reachability,
    format_reachability_for_critic,
)
from scenesmith.scene_expert.repair_taxonomy import (
    FailureCategory,
    build_repair_plan,
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
from scenesmith.agent_utils.workflow_tools import WorkflowTools
from scenesmith.furniture_agents.base_furniture_agent import BaseFurnitureAgent
from scenesmith.furniture_agents.tools.furniture_tools import FurnitureTools
from scenesmith.furniture_agents.tools.scene_tools import SceneTools
from scenesmith.furniture_agents.tools.vision_tools import VisionTools
from scenesmith.prompts.registry import FurnitureAgentPrompts
from scenesmith.utils.logging import BaseLogger

console_logger = logging.getLogger(__name__)


REPAIR_ASSET_SPECS: dict[str, tuple[str, list[float]]] = {
    "bed": (
        "Compact standard double bed with headboard, mattress, pillows, and bedding",
        [1.60, 2.05, 0.80],
    ),
    "twin_bed": ("Compact single twin bed with mattress and headboard", [1.0, 2.0, 0.75]),
    "nightstand": ("Compact bedside nightstand with drawer", [0.45, 0.42, 0.55]),
    "wardrobe": ("Compact wardrobe closet with simple doors", [0.90, 0.55, 2.00]),
    "dresser": ("Low dresser chest with storage drawers", [1.10, 0.48, 0.85]),
    "desk": ("Practical rectangular work desk", [1.10, 0.60, 0.75]),
    "student_desk": (
        "Standard individual classroom student desk with writing surface and storage",
        [0.70, 0.55, 0.75],
    ),
    "teacher_desk": (
        "Full-size classroom teacher desk with enclosed front and drawers",
        [1.40, 0.70, 0.75],
    ),
    "chair": (
        "Standard upright classroom student chair with seat and backrest",
        [0.48, 0.50, 0.85],
    ),
    "sofa": ("Compact upholstered two-seat sofa", [1.70, 0.85, 0.90]),
    "table": ("Practical rectangular table", [1.20, 0.80, 0.75]),
    "cabinet": ("Compact freestanding storage cabinet", [0.90, 0.45, 1.10]),
    "bookshelf": ("Compact freestanding bookshelf", [0.90, 0.35, 1.80]),
    "plant": ("Large indoor potted floor plant", [0.60, 0.60, 1.20]),
    "rug": ("Square low-pile area rug", [1.80, 1.80, 0.03]),
}


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
        self._allow_functional_layout_repair = False

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
        return super()._create_critic_agent(
            tools=tools,
            prompt_enum=critic_prompt_enum,
            output_type=FurnitureCritiqueWithScores,
            scene_description=scene.text_description,
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
        )

    def _generate_and_save_context_image(self, scene: RoomScene) -> Path:
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
        image_path = (
            self.asset_manager.image_generator.generate_furniture_context_image(
                reference_image_path=room_render,
                scene_description=scene.text_description,
                width_m=scene.room_geometry.width,
                length_m=scene.room_geometry.length,
                output_path=output_path,
            )
        )

        console_logger.info(f"Context image saved to: {image_path}")
        return image_path

    async def add_furniture(self, scene: RoomScene) -> None:
        """Add furniture to a scene.

        Args:
            scene: RoomScene to add furniture to (mutated in place)
        """
        # Store everything as instance variables for closure access.
        self.scene = scene
        self._configure_stage_runtime(scene)
        safety_description = getattr(
            scene,
            "scene_expert_original_description",
            scene.text_description,
        )
        self._configure_furniture_safety_for_scene(safety_description)

        # Generate context image if configured. If generation fails, continue without it.
        if self.cfg.context_image_generation.enabled:
            try:
                self.context_image_path = self._generate_and_save_context_image(scene)
            except Exception as e:
                console_logger.warning(
                    f"Context image generation failed, continuing without it: {e}"
                )
                self.context_image_path = None

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
        result = await self._run_agent_with_stage_sla(
            starting_agent=self.planner,
            input=runner_instruction,
            role="planner",
            event="planner_workflow",
            configured_max_turns=self.cfg.agents.planner_agent.max_turns,
            run_config=self._create_run_config(),
        )
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
            Dict with scene description and reference image flag.
        """
        return {
            "scene_description": self.scene.text_description,
            "has_reference_image": self.context_image_path is not None,
        }

    def _build_initial_design_input(self, instruction: str) -> str | list[dict]:
        """Add deterministic room-aware functional guidance to initial design."""
        safety_cfg = getattr(self.cfg, "furniture_safety_controller", None)
        bedroom_cfg = getattr(safety_cfg, "bedroom_layout", None)
        guidance_blocks = [format_bedroom_anchor_guidance(
            scene=self.scene,
            cfg=bedroom_cfg,
        )]
        guidance_blocks.append(
            format_functional_layout_guidance(
                scene=self.scene,
                cfg=getattr(safety_cfg, "functional_layout", None),
            )
        )
        guidance = "\n\n".join(block for block in guidance_blocks if block)
        if guidance:
            instruction = (
                f"{instruction}\n\n"
                "# Deterministic Room-Aware Layout Guidance\n"
                f"{guidance}"
            )
        return super()._build_initial_design_input(instruction)

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
        reasons = " ".join(hard_state.hard_reasons or []).lower()
        repair_plan = build_repair_plan(
            stage=self.agent_type.value,
            hard_reasons=hard_state.hard_reasons,
            max_attempts=1,
        )
        console_logger.info("Deterministic furniture %s", repair_plan.to_log_text())

        controller = getattr(self, "furniture_safety_controller", None)
        required_counts = getattr(controller, "required_counts", {}) or {}
        for category in required_counts:
            if f"missing required {category}" not in reasons:
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
        replaced_invalid = self._replace_invalid_furniture_assets(hard_state)
        if replaced_invalid:
            actions.append(
                f"replaced {replaced_invalid} invalid furniture asset(s)"
            )
        relation_changed = False
        if (
            FailureCategory.DOOR_OR_OPENING_CLEARANCE in repair_plan.categories
            and self._repair_forbidden_zone_conflicts(include_windows=False)
        ):
            actions.append("cleared deterministic door/opening forbidden zones")
            relation_changed = True

        window_conflict = (
            "window access warning" in reasons
            or FailureCategory.WINDOW_OR_WALL_ACCESS in repair_plan.categories
        )
        if window_conflict and self._repair_forbidden_zone_conflicts(
            include_windows=True
        ):
            actions.append("cleared deterministic window forbidden zones")
            relation_changed = True

        if is_bedroom_scene(self.scene):
            bed_repair_needed = (
                "missing required bed" in reasons
                or "bedroom plausibility: bed" in reasons
                or window_conflict
            )
            nightstand_repair_needed = (
                bed_repair_needed
                or "missing required nightstand" in reasons
                or "bedroom relation:" in reasons
            )
            if bed_repair_needed and self._anchor_existing_bed():
                actions.append("anchored bed to deterministic bedroom head wall")
                relation_changed = True
            if nightstand_repair_needed and self._repair_bedside_nightstands():
                actions.append(
                    "repositioned nightstands to deterministic bedside anchors"
                )
                relation_changed = True
            if (
                window_conflict
                or "wardrobe" in reasons
                or "closet" in reasons
            ) and self._repair_wardrobe_wall_anchor():
                actions.append("moved wardrobe to a deterministic wall/corner anchor")
                relation_changed = True

        if getattr(self, "_allow_functional_layout_repair", False):
            functional_action = self._repair_functional_layout()
            if functional_action:
                actions.append(functional_action)
                relation_changed = True

        # Structured collision pairs survive the verifier boundary. If a
        # bedroom relation operator already moved objects, defer collision
        # handling until the next re-evaluation so stale pairs are not applied.
        if not relation_changed:
            repaired_collisions = self._repair_structured_collisions(hard_state)
            if repaired_collisions:
                actions.append(
                    f"separated {repaired_collisions} structured collision pair(s)"
                )

        return bool(actions), actions

    def capture_agent_candidate(self) -> dict[str, Any] | None:
        """Export the best hard-valid candidate from the current agent attempt."""
        controller = getattr(self, "furniture_safety_controller", None)
        if controller is None or controller.best_scene_state is None:
            hard_state = self._evaluate_current_hard_state()
            if hard_state is None or not hard_state.hard_valid or self.scene is None:
                return None
            return {
                "scene_state": copy.deepcopy(self.scene.to_state_dict()),
                "scores": None,
                "render_dir": None,
                "weighted_score": None,
                "score_source": "unscored_hard_valid",
            }
        return {
            "scene_state": copy.deepcopy(controller.best_scene_state),
            "scores": (
                copy.deepcopy(controller.best_scores)
                if getattr(controller, "best_score_source", "unavailable")
                == "vlm_critic"
                else None
            ),
            "render_dir": controller.best_render_dir,
            "weighted_score": (
                float(controller.best_weighted_score)
                if (
                    controller.best_scores is not None
                    and getattr(controller, "best_score_source", "unavailable")
                    == "vlm_critic"
                )
                else None
            ),
            "score_source": getattr(
                controller,
                "best_score_source",
                "unscored_hard_valid",
            ),
        }

    @staticmethod
    def prefer_agent_candidate(
        current: dict[str, Any] | None,
        candidate: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Prefer trusted critic score, else retain the first hard-valid state."""
        if candidate is None:
            return current
        if current is None:
            return candidate
        current_score = current.get("weighted_score")
        candidate_score = candidate.get("weighted_score")
        if candidate_score is None:
            return current
        if current_score is None or float(candidate_score) > float(current_score):
            return candidate
        return current

    def restore_agent_candidate(self, candidate: dict[str, Any]) -> None:
        """Restore a cross-regeneration best candidate into agent/controller state."""
        self._restore_furniture_scene_state(candidate["scene_state"])
        controller = self.furniture_safety_controller
        controller.best_scene_state = copy.deepcopy(candidate["scene_state"])
        controller.best_scores = copy.deepcopy(candidate.get("scores"))
        controller.best_score_source = str(
            candidate.get("score_source", "unscored_hard_valid")
        )
        controller.best_render_dir = candidate.get("render_dir")
        controller.best_weighted_score = float(
            candidate.get("weighted_score")
            if candidate.get("weighted_score") is not None
            else 0.0
        )
        controller.best_reasons = ["best pure-agent candidate across regenerations"]
        self.previous_scores = copy.deepcopy(candidate.get("scores"))
        self.final_render_dir = candidate.get("render_dir")
        self.rendering_manager.clear_cache()

    def should_regenerate_for_quality(
        self, candidate: dict[str, Any] | None
    ) -> tuple[bool, str]:
        """Request another agent design only for a real critic score below target."""
        if not getattr(self.scene, "scene_expert_stage_budget", None):
            return False, "SceneExpert quality fallback is inactive for this run"
        if candidate is None or candidate.get("score_source") != "vlm_critic":
            return False, "visual critic unavailable; not interpreted as a failure"
        weighted_score = candidate.get("weighted_score")
        if weighted_score is None:
            return False, "visual critic score unavailable"
        threshold = float(self.furniture_safety_controller.accept_score_threshold)
        if float(weighted_score) >= threshold:
            return (
                False,
                f"trusted critic score {weighted_score:.3f} meets {threshold:.3f}",
            )
        scores = candidate.get("scores")
        critique = str(getattr(scores, "critique", "") or "").strip()
        critique_hint = f"; critic: {critique[:1200]}" if critique else ""
        return (
            True,
            f"trusted critic score {float(weighted_score):.3f} below "
            f"{threshold:.3f}{critique_hint}",
        )

    @staticmethod
    def _candidate_score_summary(candidate: dict[str, Any] | None) -> dict[str, Any]:
        if candidate is None:
            return {"score_source": "unavailable", "weighted_score": None}
        scores = candidate.get("scores")
        return {
            "score_source": candidate.get("score_source", "unavailable"),
            "weighted_score": candidate.get("weighted_score"),
            "scores": (
                {score.name: score.grade for score in scores.get_scores()}
                if scores is not None
                else {}
            ),
        }

    async def compare_deterministic_fallback(
        self,
        *,
        agent_candidate: dict[str, Any],
        trigger: str,
        regeneration_attempts: int,
    ) -> dict[str, Any]:
        """Render and score one deterministic fallback without making it authoritative.

        The controller keeps the pure-agent candidate as incumbent. The fallback
        wins only when deterministic hard checks pass and a real VLM critic score
        improves it by the configured minimum delta.
        """
        self.restore_agent_candidate(agent_candidate)
        agent_hash = self.scene.content_hash()
        agent_hard_state = self._evaluate_current_hard_state()
        self.logger.log_scene(
            scene=self.scene,
            name="furniture_agent_best_pre_deterministic",
        )
        self.rendering_manager.clear_cache()
        agent_render_dir = self.rendering_manager.render_scene(
            scene=self.scene,
            blender_server=self.blender_server,
            rendering_mode="furniture",
            render_name="agent_best_pre_deterministic",
        )
        if agent_candidate.get("scores") is not None:
            self._write_score_artifacts(
                response=agent_candidate["scores"],
                images_dir=agent_render_dir,
                physics_context=self._get_cached_physics_context(),
                event="fallback_agent_incumbent",
            )

        comparison: dict[str, Any] = {
            "schema_version": "1.0",
            "trigger": trigger,
            "agent_regeneration_attempts": regeneration_attempts,
            "deterministic_layout_family": (
                "bedroom"
                if is_bedroom_scene(self.scene)
                else functional_layout_family(self.scene) or "unsupported"
            ),
            "agent_candidate": self._candidate_score_summary(agent_candidate),
            "agent_hard_valid": bool(
                agent_hard_state is None or agent_hard_state.hard_valid
            ),
            "agent_hard_reasons": (
                [] if agent_hard_state is None else agent_hard_state.hard_reasons
            ),
            "deterministic_actions": [],
            "deterministic_candidate": {
                "score_source": "not_generated",
                "weighted_score": None,
            },
            "selection": "agent_best_pre_deterministic",
            "selection_reason": "deterministic candidate was not generated",
            "agent_render_dir": str(agent_render_dir),
        }
        # Persist the trigger and pure-agent baseline before attempting any
        # optional fallback work. A renderer/critic interruption must still
        # leave a complete explanation of why no deterministic image appeared.
        self._write_fallback_comparison(comparison)

        controller_cfg = getattr(self.cfg, "furniture_safety_controller", None)
        bedroom_cfg = getattr(controller_cfg, "bedroom_layout", None)
        functional_cfg = getattr(controller_cfg, "functional_layout", None)
        functional_fallback_enabled = bool(
            getattr(functional_cfg, "deterministic_fallback_enabled", False)
        )
        fallback_enabled = (
            bool(
                getattr(
                    bedroom_cfg,
                    "deterministic_fallback_enabled",
                    functional_fallback_enabled,
                )
            )
            if is_bedroom_scene(self.scene)
            else functional_fallback_enabled
        )
        if not fallback_enabled:
            comparison["selection_reason"] = (
                "deterministic layout fallback is disabled by configuration"
            )
            self._write_fallback_comparison(comparison)
            return comparison

        self._allow_functional_layout_repair = True
        try:
            action = self._repair_functional_layout()
        finally:
            self._allow_functional_layout_repair = False
        if not action:
            comparison["selection_reason"] = (
                "no applicable deterministic room-layout operator changed the "
                "agent layout"
            )
            self._write_fallback_comparison(comparison)
            return comparison

        comparison["deterministic_actions"] = [action]
        deterministic_hard_state = self._evaluate_current_hard_state()
        comparison["deterministic_hard_valid"] = bool(
            deterministic_hard_state is None or deterministic_hard_state.hard_valid
        )
        comparison["deterministic_hard_reasons"] = (
            []
            if deterministic_hard_state is None
            else deterministic_hard_state.hard_reasons
        )
        self.logger.log_scene(
            scene=self.scene,
            name="furniture_deterministic_candidate",
        )
        self.rendering_manager.clear_cache()
        deterministic_render_dir = self.rendering_manager.render_scene(
            scene=self.scene,
            blender_server=self.blender_server,
            rendering_mode="furniture",
            render_name="deterministic_candidate",
        )

        deterministic_hash = self.scene.content_hash()
        comparison["deterministic_render_dir"] = str(deterministic_render_dir)
        comparison["deterministic_scene_changed"] = deterministic_hash != agent_hash
        # The deterministic render is a required diagnostic artifact, even if
        # its later VLM comparison times out or the process is interrupted.
        self._write_fallback_comparison(comparison)
        self._stage_runtime_phase = "fallback"
        try:
            if (
                deterministic_hard_state is not None
                and not deterministic_hard_state.hard_valid
            ):
                self.restore_agent_candidate(agent_candidate)
                comparison["deterministic_candidate"] = {
                    "score_source": "hard_check_only",
                    "weighted_score": None,
                }
                comparison["selection_reason"] = (
                    "deterministic candidate failed physical hard checks"
                )
            else:
                # A failure on the agent candidate does not prove that the
                # independent deterministic candidate is unscorable. Retry its
                # visual decision with the fresh per-evaluation critic context.
                self._critic_failed = False
                self._last_trusted_critic_candidate = None
                await self._request_critique_impl(update_checkpoint=False)
                deterministic_candidate = self._last_trusted_critic_candidate
                comparison["deterministic_candidate"] = self._candidate_score_summary(
                    deterministic_candidate
                )
                if deterministic_candidate is None:
                    self.restore_agent_candidate(agent_candidate)
                    comparison["deterministic_candidate"] = {
                        "score_source": self._last_score_provenance.get(
                            "score_source", "critic_unavailable"
                        ),
                        "weighted_score": None,
                    }
                    comparison["selection_reason"] = (
                        "deterministic candidate rendered, but its visual critic "
                        "did not complete; pure-agent candidate preserved"
                    )
                elif self.scene.content_hash() == deterministic_hash:
                    comparison["selection"] = "deterministic_candidate"
                    comparison["selection_reason"] = (
                        "hard-valid deterministic candidate meaningfully improved "
                        "the trusted critic score"
                    )
                else:
                    comparison["selection_reason"] = (
                        "deterministic candidate did not produce a trusted, "
                        "meaningful critic improvement"
                    )
        except Exception as exc:
            console_logger.exception(
                "Deterministic fallback comparison failed; restoring pure-agent "
                "candidate"
            )
            self.restore_agent_candidate(agent_candidate)
            comparison["deterministic_candidate"] = {
                "score_source": "critic_error",
                "weighted_score": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
            comparison["selection_reason"] = (
                "fallback critic failed; pure-agent candidate preserved"
            )
        finally:
            self._stage_runtime_phase = "agent"

        # Preserve the diagnostic render path even when the controller restores
        # the incumbent after scoring.
        self._write_fallback_comparison(comparison)
        await self._finalize_scene_and_scores()
        return comparison

    def _write_fallback_comparison(self, comparison: dict[str, Any]) -> Path:
        output_path = (
            self.logger.output_dir
            / "scene_states"
            / "furniture"
            / "fallback_comparison.json"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(comparison, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        console_logger.info("Furniture fallback comparison saved to %s", output_path)
        return output_path

    def _replace_invalid_furniture_assets(self, hard_state: HardStateEvaluation) -> int:
        """Replace placeholder or dimension-invalid required furniture assets."""
        if self.scene is None:
            return 0
        invalid_ids = {
            str(issue.object_a_id)
            for issue in getattr(hard_state, "issues", [])
            if getattr(issue, "issue_type", "") == "asset_invalid"
            and getattr(issue, "object_a_id", "")
        }
        if not invalid_ids:
            return 0
        invalid_objects = [
            obj
            for object_id, obj in self.scene.objects.items()
            if str(object_id) in invalid_ids
        ]
        if not invalid_objects:
            return 0

        excluded: set[str] = set()
        for obj in invalid_objects:
            excluded.update(self._asset_signature_values(obj))

        replaced = 0
        for old_obj in invalid_objects:
            category = self._category_for_object(old_obj.object_id, old_obj)
            if not category or self._required_count(category) <= 0:
                continue
            replacement = self._get_or_generate_repair_asset(
                category,
                exclude_asset_signatures=excluded,
            )
            if replacement is None:
                continue
            old_id = old_obj.object_id
            self.scene.remove_object(old_id)
            if self._place_repair_asset(category, replacement):
                replaced += 1
            else:
                self.scene.add_object(old_obj)
        return replaced

    def _repair_functional_layout(self) -> str:
        if self.scene is None or not hasattr(self.scene, "objects"):
            return ""
        if is_bedroom_scene(self.scene):
            bedroom_actions = self._repair_bedroom_layout()
            if bedroom_actions:
                return "normalized bedroom fallback: " + "; ".join(
                    bedroom_actions
                )
        family = functional_layout_family(self.scene)
        if family == "living_room" and self._repair_living_room_layout():
            return "normalized sofa, rug, and plants into one conversation zone"
        if family == "classroom" and self._repair_classroom_layout():
            return "normalized classroom desk-chair pairs and front teaching zone"
        return ""

    def _repair_bedroom_layout(self) -> list[str]:
        """Build one coherent bedroom fallback from existing safe operators.

        This method is invoked only by the separately rendered fallback path.
        It reacts to deterministic plausibility evidence instead of normalizing
        every low-scoring bedroom indiscriminately. Moving the bed also moves its
        dependent nightstands and rechecks wardrobe anchoring as one candidate.
        """
        if self.scene is None:
            return []
        report = evaluate_bedroom_layout_plausibility(
            scene=self.scene,
            cfg=self._bedroom_layout_cfg(),
        )
        issue_text = " ".join(report.issues).lower()
        if not issue_text:
            return []

        bed_relation_issue = any(
            term in issue_text
            for term in (
                "bed headboard faces",
                "bed headboard is not anchored",
                "bed headboard overlaps",
            )
        )
        nightstand_relation_issue = "nightstands are not on opposite" in issue_text
        wardrobe_relation_issue = "wardrobe is floating away" in issue_text

        actions: list[str] = []
        bed_changed = bed_relation_issue and self._anchor_existing_bed()
        if bed_changed:
            actions.append("anchored bed headboard to the preferred solid wall")

        nightstands_changed = (
            bed_changed or nightstand_relation_issue
        ) and self._repair_bedside_nightstands()
        if nightstands_changed:
            actions.append("placed paired nightstands beside the bed headboard")

        wardrobe_changed = (
            bed_changed or nightstands_changed or wardrobe_relation_issue
        ) and self._repair_wardrobe_wall_anchor()
        if wardrobe_changed:
            actions.append("anchored wardrobe to a non-opening wall or corner")
        return actions

    def _repair_living_room_layout(self) -> bool:
        if self.scene is None:
            return False
        sofas = self._furniture_by_category("sofa")
        if not sofas:
            return False
        wall = choose_functional_anchor_wall(self.scene, "living_room")
        if wall is None:
            return False
        sofa = sofas[0]
        yaw = self._yaw_for_inward_wall(wall)
        transform = self._grounded_transform(sofa, x=0.0, y=0.0, yaw_deg=yaw)
        transform = self._snap_transform_to_wall(sofa, transform, wall)
        transform = self._fit_transform_inside_room(sofa, transform)
        changed = False
        if not self._transform_close(sofa.transform, transform):
            self.scene.move_object(sofa.object_id, transform)
            changed = True

        sofa_center = np.asarray(transform.translation(), dtype=float)
        rotation = np.asarray(transform.rotation().matrix(), dtype=float)
        lateral = rotation @ np.asarray([1.0, 0.0, 0.0])
        forward = rotation @ np.asarray([0.0, 1.0, 0.0])
        sofa_dims = self._local_size(sofa, [1.70, 0.85, 0.90])

        rugs = self._furniture_by_category("rug")
        if rugs:
            rug = rugs[0]
            rug_dims = self._local_size(rug, [1.80, 1.80, 0.03])
            distance = max(
                0.75,
                sofa_dims[1] / 2.0 + rug_dims[1] / 2.0 - 0.20,
            )
            target = sofa_center + forward * distance
            rug_transform = self._grounded_transform(
                rug,
                x=float(target[0]),
                y=float(target[1]),
                yaw_deg=yaw,
            )
            rug_transform = self._fit_transform_inside_room(rug, rug_transform)
            if not self._transform_close(rug.transform, rug_transform):
                self.scene.move_object(rug.object_id, rug_transform)
                changed = True

        plants = self._furniture_by_category("plant")[:2]
        if len(plants) == 2:
            for side, plant in zip((-1.0, 1.0), plants):
                plant_dims = self._local_size(plant, [0.60, 0.60, 1.20])
                target = (
                    sofa_center
                    + lateral
                    * side
                    * (sofa_dims[0] / 2.0 + plant_dims[0] / 2.0 + 0.15)
                    + forward * 0.05
                )
                plant_transform = self._grounded_transform(
                    plant,
                    x=float(target[0]),
                    y=float(target[1]),
                    yaw_deg=yaw,
                )
                plant_transform = self._fit_transform_inside_room(
                    plant, plant_transform
                )
                if not self._transform_close(plant.transform, plant_transform):
                    self.scene.move_object(plant.object_id, plant_transform)
                    changed = True
        return changed

    def _repair_classroom_layout(self) -> bool:
        if self.scene is None:
            return False
        desks = sorted(
            self._furniture_by_category("student_desk"),
            key=lambda obj: str(obj.object_id),
        )
        chairs = sorted(
            self._furniture_by_category("chair"),
            key=lambda obj: str(obj.object_id),
        )
        teacher_desks = self._furniture_by_category("teacher_desk")
        if not desks:
            return False
        wall = choose_functional_anchor_wall(self.scene, "classroom")
        room_bounds = self._room_bounds_xy()
        if wall is None or room_bounds is None:
            return False

        inward_xy = {
            "north": np.asarray([0.0, -1.0]),
            "south": np.asarray([0.0, 1.0]),
            "east": np.asarray([-1.0, 0.0]),
            "west": np.asarray([1.0, 0.0]),
        }[wall]
        lateral_xy = np.asarray([inward_xy[1], -inward_xy[0]])
        min_x, min_y, max_x, max_y = room_bounds
        wall_center = {
            "north": np.asarray([0.0, max_y]),
            "south": np.asarray([0.0, min_y]),
            "east": np.asarray([max_x, 0.0]),
            "west": np.asarray([min_x, 0.0]),
        }[wall]
        student_yaw = self._yaw_for_head_wall(wall)
        teacher_yaw = self._yaw_for_inward_wall(wall)
        changed = False

        teacher_depth = 0.70
        if teacher_desks:
            teacher = teacher_desks[0]
            teacher_depth = float(
                self._local_size(teacher, [1.40, 0.70, 0.75])[1]
            )
            teacher_transform = self._grounded_transform(
                teacher,
                x=float(wall_center[0]),
                y=float(wall_center[1]),
                yaw_deg=teacher_yaw,
            )
            teacher_transform = self._snap_transform_to_wall(
                teacher, teacher_transform, wall
            )
            teacher_transform = self._fit_transform_inside_room(
                teacher, teacher_transform
            )
            if not self._transform_close(teacher.transform, teacher_transform):
                self.scene.move_object(teacher.object_id, teacher_transform)
                changed = True
            metadata = dict(getattr(teacher, "metadata", {}) or {})
            metadata.update(
                {"functional_zone": "classroom_front", "front_wall": wall}
            )
            teacher.metadata = metadata

        safety_cfg = getattr(self.cfg, "furniture_safety_controller", None)
        functional_cfg = getattr(safety_cfg, "functional_layout", None)
        classroom_cfg = getattr(functional_cfg, "classroom", None)
        columns = max(
            1,
            min(
                len(desks),
                int(getattr(classroom_cfg, "preferred_columns", 3) or 3),
            ),
        )
        sample_dims = self._local_size(desks[0], [0.70, 0.55, 0.75])
        lateral_room_span = (max_x - min_x) if wall in ("north", "south") else (
            max_y - min_y
        )
        column_spacing = min(
            max(float(sample_dims[0]) + 0.45, 1.15),
            max(0.85, (lateral_room_span - 0.8) / max(1, columns)),
        )
        row_spacing = max(float(sample_dims[1]) + 0.95, 1.45)
        first_row_distance = teacher_depth / 2.0 + 1.35

        for index, desk in enumerate(desks):
            row = index // columns
            column = index % columns
            lateral_offset = (column - (columns - 1) / 2.0) * column_spacing
            desk_xy = (
                wall_center
                + inward_xy * (first_row_distance + row * row_spacing)
                + lateral_xy * lateral_offset
            )
            desk_transform = self._grounded_transform(
                desk,
                x=float(desk_xy[0]),
                y=float(desk_xy[1]),
                yaw_deg=student_yaw,
            )
            desk_transform = self._fit_transform_inside_room(desk, desk_transform)
            if not self._transform_close(desk.transform, desk_transform):
                self.scene.move_object(desk.object_id, desk_transform)
                changed = True

            if index >= len(chairs):
                continue
            chair = chairs[index]
            desk_depth = float(self._local_size(desk, [0.70, 0.55, 0.75])[1])
            chair_depth = float(self._local_size(chair, [0.48, 0.50, 0.85])[1])
            chair_distance = desk_depth / 2.0 + chair_depth / 2.0 + 0.12
            chair_xy = desk_xy + inward_xy * chair_distance
            chair_transform = self._grounded_transform(
                chair,
                x=float(chair_xy[0]),
                y=float(chair_xy[1]),
                yaw_deg=student_yaw,
            )
            chair_transform = self._fit_transform_inside_room(
                chair, chair_transform
            )
            if not self._transform_close(chair.transform, chair_transform):
                self.scene.move_object(chair.object_id, chair_transform)
                changed = True
        return changed

    def _repair_structured_collisions(self, hard_state: HardStateEvaluation) -> int:
        if self.scene is None:
            return 0
        collision_issues = [
            issue
            for issue in getattr(hard_state, "issues", [])
            if getattr(issue, "issue_type", "") == "collision_or_overlap"
        ]
        repaired = 0
        moved_ids: set[str] = set()
        for issue in collision_issues:
            object_a = self._scene_object_by_string_id(issue.object_a_id)
            object_b = self._scene_object_by_string_id(issue.object_b_id)
            movable = [
                obj
                for obj in (object_a, object_b)
                if obj is not None
                and not getattr(obj, "immutable", False)
                and str(getattr(obj.object_type, "value", obj.object_type)).lower()
                == "furniture"
            ]
            if not movable:
                continue

            def move_priority(obj: SceneObject) -> tuple[int, float]:
                category = self._category_for_object(obj.object_id, obj)
                required = bool(category and self._required_count(category) > 0)
                bounds = obj.compute_world_bounds()
                if bounds is None:
                    footprint = float("inf")
                else:
                    bounds_min = np.asarray(bounds[0], dtype=float)
                    bounds_max = np.asarray(bounds[1], dtype=float)
                    footprint = float(
                        np.prod(np.maximum(0.0, bounds_max[:2] - bounds_min[:2]))
                    )
                return (1 if required else 0, footprint)

            movable.sort(key=move_priority)
            obj = movable[0]
            if str(obj.object_id) in moved_ids:
                continue
            other = object_b if obj is object_a else object_a
            room_boundary_id = next(
                (
                    str(candidate_id)
                    for candidate_id in (issue.object_a_id, issue.object_b_id)
                    if str(candidate_id).startswith("room_geometry::")
                ),
                "",
            )
            if room_boundary_id:
                transform = self._move_away_from_room_boundary_transform(
                    obj,
                    room_boundary_id=room_boundary_id,
                    penetration_depth_m=float(
                        getattr(issue, "penetration_depth_m", 0.0) or 0.0
                    ),
                )
            else:
                transform = self._best_collision_separation_transform(obj, other)
            if transform is None or self._transform_close(obj.transform, transform):
                continue
            old_penalty = self._furniture_placement_penalty(
                obj, obj.transform, exclude_object_id=str(obj.object_id)
            )
            new_penalty = self._furniture_placement_penalty(
                obj, transform, exclude_object_id=str(obj.object_id)
            )
            # Boundary motion is driven by Drake's measured wall penetration;
            # the furniture-only AABB penalty does not include room walls and can
            # therefore remain numerically unchanged after a valid inward snap.
            if not room_boundary_id and new_penalty + 1e-5 >= old_penalty:
                continue
            self.scene.move_object(obj.object_id, transform)
            moved_ids.add(str(obj.object_id))
            repaired += 1
            console_logger.info(
                "Deterministic collision repair moved %s away from %s "
                "(penalty %.4f -> %.4f)",
                obj.object_id,
                getattr(other, "object_id", "unknown"),
                old_penalty,
                new_penalty,
            )
        return repaired

    def _move_away_from_room_boundary_transform(
        self,
        obj: SceneObject,
        *,
        room_boundary_id: str,
        penetration_depth_m: float,
    ) -> RigidTransform | None:
        """Translate furniture inward from the specific wall it penetrates."""
        boundary = room_boundary_id.lower()
        inward_xy: tuple[float, float] | None = None
        if "north" in boundary:
            inward_xy = (0.0, -1.0)
        elif "south" in boundary:
            inward_xy = (0.0, 1.0)
        elif "east" in boundary:
            inward_xy = (-1.0, 0.0)
        elif "west" in boundary:
            inward_xy = (1.0, 0.0)
        if inward_xy is None:
            return self._best_generic_repair_transform(
                obj,
                fallback=obj.transform,
                exclude_object_id=str(obj.object_id),
            )

        gap = float(self._repair_cfg_value("wall_clearance_gap_m", 0.03))
        distance = max(0.0, float(penetration_depth_m)) + max(0.0, gap)
        translation = np.array(obj.transform.translation(), dtype=float, copy=True)
        translation[0] += inward_xy[0] * distance
        translation[1] += inward_xy[1] * distance
        candidate = RigidTransform(R=obj.transform.rotation(), p=translation)
        return self._fit_transform_inside_room(obj, candidate)

    def _scene_object_by_string_id(self, object_id: str) -> SceneObject | None:
        if self.scene is None:
            return None
        for candidate_id, obj in self.scene.objects.items():
            if str(candidate_id) == str(object_id):
                return obj
        return None

    def _best_collision_separation_transform(
        self,
        obj: SceneObject,
        other: SceneObject | None,
    ) -> RigidTransform | None:
        if other is None:
            return self._best_generic_repair_transform(
                obj,
                fallback=obj.transform,
                exclude_object_id=str(obj.object_id),
            )
        obj_bounds = obj.compute_world_bounds()
        other_bounds = other.compute_world_bounds()
        if obj_bounds is None or other_bounds is None:
            return None
        obj_min = np.asarray(obj_bounds[0], dtype=float)
        obj_max = np.asarray(obj_bounds[1], dtype=float)
        other_min = np.asarray(other_bounds[0], dtype=float)
        other_max = np.asarray(other_bounds[1], dtype=float)
        obj_center = (obj_min + obj_max) / 2.0
        current_translation = np.asarray(obj.transform.translation(), dtype=float)
        origin_offset = current_translation[:2] - obj_center[:2]
        half_size = (obj_max[:2] - obj_min[:2]) / 2.0
        gap = float(self._repair_cfg_value("collision_separation_gap_m", 0.08))
        candidate_centers = (
            np.asarray([other_min[0] - half_size[0] - gap, obj_center[1]]),
            np.asarray([other_max[0] + half_size[0] + gap, obj_center[1]]),
            np.asarray([obj_center[0], other_min[1] - half_size[1] - gap]),
            np.asarray([obj_center[0], other_max[1] + half_size[1] + gap]),
        )
        yaw = math.degrees(RollPitchYaw(obj.transform.rotation()).yaw_angle())
        best: RigidTransform | None = None
        best_penalty = float("inf")
        for center in candidate_centers:
            xy = center + origin_offset
            candidate = self._grounded_transform(
                obj, x=float(xy[0]), y=float(xy[1]), yaw_deg=yaw
            )
            candidate = self._fit_transform_inside_room(obj, candidate)
            penalty = self._furniture_placement_penalty(
                obj, candidate, exclude_object_id=str(obj.object_id)
            )
            if penalty < best_penalty:
                best = candidate
                best_penalty = penalty
        return best

    def _replace_geometry_failed_furniture_assets(self, reasons: str) -> int:
        """Replace required furniture whose SDF/mesh cannot be loaded by Drake."""
        if self.scene is None:
            return 0

        controller = getattr(self, "furniture_safety_controller", None)
        configured_categories = list(
            (getattr(controller, "required_counts", {}) or {}).keys()
        )
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
        text = (
            f"{object_id} {getattr(obj, 'name', '')} "
            f"{getattr(obj, 'description', '')}"
        ).lower()
        controller = getattr(self, "furniture_safety_controller", None)
        if controller is not None:
            category = controller.infer_object_category(text)
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
            if self._category_for_object(object_id, obj) == category:
                result.append(obj)
        return result

    def _required_count(self, category: str) -> int:
        controller = getattr(self, "furniture_safety_controller", None)
        if not controller:
            return 0
        return int(getattr(controller, "required_counts", {}).get(category, 0) or 0)

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
            mesh = trimesh.creation.box(extents=[width, depth, height])
            mesh.apply_translation([0.0, 0.0, height / 2.0])
            gltf_path = repair_root / f"{category}_placeholder.gltf"
            sdf_path = repair_root / f"{category}_placeholder.sdf"
            mesh.export(gltf_path)
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
                    "asset_source": "deterministic_placeholder",
                    "repair_placeholder": True,
                    "generation_timestamp": time.time(),
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
            if category not in ("bed", "nightstand", "wardrobe", "twin_bed"):
                transform = self._best_generic_repair_transform(
                    scene_object,
                    fallback=transform,
                )
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
        exclude_object_id: str = "",
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
                    bounds = self._bounds_for_transform(obj, candidate)
                    if bounds is None:
                        continue
                    penalty = self._furniture_placement_penalty(
                        obj,
                        candidate,
                        exclude_object_id=exclude_object_id,
                    )
                    if penalty < best_penalty:
                        best = candidate
                        best_penalty = penalty
                    if penalty <= 1e-6:
                        return candidate
        return best

    def _furniture_placement_penalty(
        self,
        obj: SceneObject,
        transform: RigidTransform,
        *,
        exclude_object_id: str = "",
    ) -> float:
        if self.scene is None:
            return float("inf")
        bounds = self._bounds_for_transform(obj, transform)
        if bounds is None:
            return float("inf")
        penalty = self._zone_overlap_penalty(
            bounds,
            self._opening_forbidden_zones(include_windows=False),
        )
        for existing_id, existing in self.scene.objects.items():
            if str(existing_id) == exclude_object_id:
                continue
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
            overlap_x, overlap_y = self._xy_overlap_depths(bounds, existing_bounds)
            penalty += overlap_x * overlap_y * 1000.0
        return penalty

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
        self.scene.move_object(bed.object_id, transform)
        return True

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
        head = rotation @ np.array([0.0, 1.0, 0.0])
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
                self.scene.move_object(nightstand.object_id, transform)
                changed = True
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
        obstacles = self._furniture_by_category("bed") + self._furniture_by_category(
            "nightstand"
        )
        opening_zones = self._opening_forbidden_zones(include_windows=True)
        best_transform = None
        best_score = -1e9
        for transform, wall_opening_penalty in candidates:
            bounds = self._bounds_for_transform(wardrobe, transform)
            if bounds is None:
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
            exact_opening_penalty = self._zone_overlap_penalty(
                bounds,
                opening_zones,
            )
            score = (
                distance_score
                - overlap_penalty
                - wall_opening_penalty
                - exact_opening_penalty
            )
            if score > best_score:
                best_score = score
                best_transform = transform

        if best_transform is None or self._transform_close(
            wardrobe.transform, best_transform
        ):
            return False
        self.scene.move_object(wardrobe.object_id, best_transform)
        return True

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
            self.scene.move_object(obj.object_id, transform)
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

    def _room_bounds_xy(self) -> tuple[float, float, float, float] | None:
        if self.scene is None:
            return None
        return furnishable_room_bounds_xy(self.scene)

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
        room_bounds = self._room_bounds_xy()
        bounds = self._bounds_for_transform(obj, transform)
        if room_bounds is None or bounds is None:
            return transform
        min_x, min_y, max_x, max_y = room_bounds
        world_min, world_max = bounds
        margin = 0.03
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

    def _yaw_for_head_wall(self, wall: str) -> float:
        return {
            "north": 0.0,
            "south": 180.0,
            "east": -90.0,
            "west": 90.0,
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
        }
