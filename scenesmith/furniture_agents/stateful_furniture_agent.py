"""
Stateful furniture agent with natural conversation between persistent agents.

This module implements a furniture placement workflow using persistent
SQLiteSession agents that maintain conversation memory across interactions.
"""

import copy
import logging
import math
import re
import time

from pathlib import Path
from typing import Any, Callable

import numpy as np
import trimesh

from agents import Agent, FunctionTool, Runner, RunResult
from agents.exceptions import MaxTurnsExceeded
from omegaconf import DictConfig
from pydrake.all import RigidTransform, RollPitchYaw

from scenesmith.agent_utils.asset_manager import AssetGenerationRequest
from scenesmith.agent_utils.base_stateful_agent import (
    BaseStatefulAgent,
    HardStateEvaluation,
    log_agent_usage,
)
from scenesmith.agent_utils.clearance_zones import (
    AABB_INTERSECTION_EPSILON_M,
    WALL_HEIGHT_TOLERANCE_M,
    aabb_overlap_depths,
    compute_door_clearance_violations,
    compute_window_clearance_violations,
    door_swing_clearance_bounds,
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
    FURNITURE_CATEGORY_COMPONENT_SHADOWS,
    furniture_category_satisfies,
    furniture_object_category_matches,
    infer_furniture_category,
    infer_furniture_object_category,
)
from scenesmith.agent_utils.house import HouseLayout
from scenesmith.agent_utils.mesh_physics_analyzer import MeshPhysicsAnalysis
from scenesmith.agent_utils.placement_noise import PlacementNoiseMode
from scenesmith.agent_utils.physics_validation import compute_scene_collisions
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
from scenesmith.agent_utils.stage_placement_order_config import (
    append_placement_order_reference,
)
from scenesmith.agent_utils.seating_orientation_guard import (
    align_seating_to_nearest_surface,
)
from scenesmith.agent_utils.thin_covering_generator import generate_thin_covering_sdf
from scenesmith.agent_utils.workflow_tools import WorkflowTools
from scenesmith.furniture_agents.base_furniture_agent import BaseFurnitureAgent
from scenesmith.furniture_agents.tools.furniture_tools import FurnitureTools
from scenesmith.furniture_agents.tools.scene_tools import SceneTools
from scenesmith.furniture_agents.tools.vision_tools import VisionTools
from scenesmith.prompts.registry import FurnitureAgentPrompts
from scenesmith.scene_expert.repair_taxonomy import FailureCategory, build_repair_plan
from scenesmith.scenebenchmark_critic.api import (
    evaluate_room_scene,
    seating_orientation_targets,
)
from scenesmith.scenebenchmark_critic.config import critic_config_from_any
from scenesmith.scenebenchmark_critic.furniture_relation_repair import (
    improve_furniture_relations,
    unresolved_furniture_relation_failures,
)
from scenesmith.scenebenchmark_critic.intent_contract import (
    intent_contract_required_counts,
)
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
    "sofa_chair": (
        "Single-seat upholstered sofa chair with a supportive back",
        [0.85, 0.85, 0.90],
    ),
    "guest_chair": (
        "Compact upholstered guest chair with a fixed wooden frame",
        [0.60, 0.65, 0.90],
    ),
    "dining_chair": ("Simple upright dining chair", [0.50, 0.55, 0.90]),
    "chair": ("Simple upright task chair", [0.50, 0.50, 0.90]),
    "student_chair": ("Simple upright student classroom chair", [0.50, 0.50, 0.90]),
    "stool": ("Compact low upholstered stool", [0.40, 0.40, 0.45]),
    "sofa": ("Compact upholstered two-seat sofa", [1.70, 0.85, 0.90]),
    "table": ("Practical rectangular table", [1.20, 0.80, 0.75]),
    "coffee_table": ("Low rectangular coffee table", [1.10, 0.60, 0.45]),
    "dining_table": (
        "Rectangular dining table for five place settings",
        [1.80, 0.85, 0.75],
    ),
    "conference_table": ("Rectangular conference table", [2.40, 1.10, 0.75]),
    "dressing_table": (
        "Low freestanding dressing table with drawers and no integrated mirror",
        [1.20, 0.50, 0.75],
    ),
    "cabinet": ("Compact freestanding storage cabinet", [0.90, 0.45, 1.10]),
    "storage_cabinet": ("Compact freestanding storage cabinet", [0.90, 0.45, 1.10]),
    "bookshelf": ("Compact freestanding bookshelf", [0.90, 0.35, 1.80]),
    "plant": ("Large indoor potted floor plant", [0.60, 0.60, 1.20]),
    "rug": ("Square low-pile area rug", [1.80, 1.80, 0.03]),
    "armchair": ("Compact upholstered armchair", [0.75, 0.75, 0.95]),
    "floor_lamp": ("Slim standing floor lamp", [0.40, 0.40, 1.60]),
    "speaker": (
        "Tall floor-standing speaker tower with a compact footprint",
        [0.40, 0.35, 1.20],
    ),
    "tv_stand": ("Low media console TV stand", [1.60, 0.45, 0.65]),
    "television": ("Slim flat-screen television display", [1.10, 0.18, 0.65]),
    "sideboard": ("Compact dining room sideboard", [1.40, 0.45, 0.80]),
    "water_dispenser": ("Freestanding bottled water dispenser", [0.35, 0.40, 1.10]),
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
_OPENING_SAFE_WALL_MARGIN_M = 0.03


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
        house_layout: HouseLayout | None = None,
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
        self.house_layout = house_layout
        self._window_repair_service: Any | None = None
        # Populated per scene only when the optional feature is enabled.
        self._placement_order_reference: str = ""

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

    async def add_furniture(
        self,
        scene: RoomScene,
        *,
        resume_from_initial_render: bool = False,
    ) -> None:
        """Add furniture to a scene.

        Args:
            scene: RoomScene to add furniture to (mutated in place)
        """
        # Store everything as instance variables for closure access.
        self.scene = scene
        safety_description = getattr(
            scene,
            "scene_expert_original_description",
            scene.text_description,
        )
        self._configure_furniture_safety_for_scene(safety_description)
        self._synchronize_task_required_counts()
        pre_disk_restore_hash = scene.content_hash()
        loaded_disk_checkpoint = self._load_furniture_hard_valid_checkpoint(
            restore_when_current_invalid=True,
        )
        if loaded_disk_checkpoint and scene.content_hash() != pre_disk_restore_hash:
            resume_from_initial_render = True
        self._placement_order_reference = ""
        if not resume_from_initial_render:
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
        if self.cfg.context_image_generation.enabled and not resume_from_initial_render:
            try:
                self.context_image_path = self._generate_and_save_context_image(scene)
            except Exception as e:
                console_logger.warning(
                    f"Context image generation failed, continuing without it: {e}"
                )
                self.context_image_path = None

        if resume_from_initial_render:
            # A restored render can already contain a complete but physically
            # invalid support pair plus unrelated clearance violations. First use
            # the normal deterministic hard-state repair so a relation candidate
            # is not rejected solely because of an independent old violation.
            restored_hard_state = self._evaluate_current_hard_state()
            _, _, restored_repair_actions = (
                self._try_deterministic_repair_for_hard_state(
                    restored_hard_state,
                    source="restored_furniture_render_pre_first_critique",
                )
            )
            if restored_repair_actions:
                console_logger.info(
                    "Deterministic repair for restored furniture render before "
                    "first critique: %s",
                    "; ".join(restored_repair_actions),
                )
            # Resolve remaining prompt-authorized geometry before the planner has
            # a chance to spend its bounded tool budget on it.
            self._repair_contract_layout_before_first_critique()

        # Create designer, critic, and planner with tools once for this scene.
        designer_tools = self._create_designer_tools()
        self.designer = self._create_designer_agent(tools=designer_tools)
        critic_tools = self._create_critic_tools()
        self.critic = self._create_critic_agent(scene=scene, tools=critic_tools)
        self._planner_skip_initial_design = resume_from_initial_render
        planner_tools = self._create_planner_tools()
        self._planner_skip_initial_design = False
        self.planner = self._create_planner_agent(scene=scene, tools=planner_tools)

        # Get runner instruction from prompt registry.
        runner_instruction = self.prompt_registry.get_prompt(
            prompt_enum=FurnitureAgentPrompts.STATEFUL_PLANNER_RUNNER_INSTRUCTION,
        )
        if resume_from_initial_render:
            runner_instruction += (
                "\n\nRESUME MODE: this scene already contains a restored furniture "
                "render. Do not create another "
                "initial design. Start by calling request_critique(), then use "
                "request_design_change() only to address concrete feedback."
            )

        # Run the furniture placement workflow.
        result: RunResult | None = None
        try:
            result = await self._run_planner_workflow(
                runner_input=runner_instruction,
                max_turns=self.cfg.agents.planner_agent.max_turns,
                require_initial_design=not resume_from_initial_render,
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

        window_actions = self._repair_substantial_window_clearance()
        if window_actions:
            console_logger.info(
                "Deterministic window-clearance repair before final critique: %s",
                "; ".join(window_actions),
            )

        seating_fixes = self._align_seating_with_hard_state_guard(
            seating_orientation_targets(scene, config=self.cfg)
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
        self._ensure_furniture_checkpoint_integrity(
            source="inventory convergence",
        )

    async def resume_from_initial_render(self, scene: RoomScene) -> None:
        """Compatibility entry point for a saved initial furniture render."""
        await self.resume_from_furniture_render(scene)

    async def resume_from_furniture_render(self, scene: RoomScene) -> None:
        """Continue furniture critique/repair from any saved furniture render."""
        await self.add_furniture(scene, resume_from_initial_render=True)

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
            delegation_failure = self._planner_terminal_failure_text()
            if delegation_failure:
                remaining = "; ".join(
                    getattr(repaired_state or hard_state, "hard_reasons", None) or []
                )
                raise RuntimeError(
                    f"Furniture planner stopped after {delegation_failure}; "
                    f"remaining hard constraints: {remaining or 'unknown'}"
                ) from error
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
            Dict with scene description and reference image flag.
        """
        return {
            "scene_description": self.scene.text_description,
            "has_reference_image": self.context_image_path is not None,
        }

    def _build_initial_design_input(self, instruction: str) -> str | list[dict]:
        """Add deterministic room-aware bedroom guidance to the initial design."""
        instruction = append_placement_order_reference(
            instruction,
            self._placement_order_reference,
        )
        safety_cfg = getattr(self.cfg, "furniture_safety_controller", None)
        bedroom_cfg = getattr(safety_cfg, "bedroom_layout", None)
        guidance = format_bedroom_anchor_guidance(
            scene=self.scene,
            cfg=bedroom_cfg,
        )
        if guidance:
            instruction = (
                f"{instruction}\n\n"
                "# Deterministic Room-Aware Layout Guidance\n"
                f"{guidance}"
            )
        return super()._build_initial_design_input(instruction)

    async def _request_initial_design_impl(self) -> str:
        """Run the initial designer, then repair only prompt-authorized relations.

        The planner auto-scores the result immediately after this method returns.
        Performing an eligible deterministic repair here therefore prevents the
        first critic render from observing a known-bad, but otherwise complete,
        LLM layout. Only hard constraints compiled from the immutable prompt
        may move furniture before that first critique.
        """
        result = await super()._request_initial_design_impl()
        self._repair_initial_contract_layout()
        return result

    def _is_furniture_relation_candidate_hard_valid(self, scene: RoomScene) -> bool:
        """Keep critic relation repairs from introducing physical hard failures."""
        if scene is not self.scene:
            return True
        hard_state = self._evaluate_current_furniture_hard_state()
        return hard_state is None or hard_state.hard_valid

    def _relation_candidate_preserves_hard_baseline(
        self,
        baseline: set[str],
        *,
        allow_deferred_window_repair: bool = False,
    ) -> Callable[[RoomScene], bool]:
        """Build a no-new-hard-failure gate for one relation-repair transaction.

        A furniture stage can begin relation repair with unrelated collisions or
        other known hard failures. Requiring every intermediate candidate to be
        globally valid would prevent that repair from making progress. Instead,
        compare each candidate with the transaction's initial hard-state
        fingerprint. This rejects only the target that creates a new violation
        (for example, a storage-wall alignment that enters a door clearance),
        while retaining independent accepted targets such as a TV support pose.
        """

        baseline_severity = self._hard_violation_severity_profile()

        def validator(scene: RoomScene) -> bool:
            if scene is not self.scene:
                return True
            introduced = self._hard_violation_fingerprints() - baseline
            if allow_deferred_window_repair:
                introduced = {
                    fingerprint
                    for fingerprint in introduced
                    if not self._is_window_hard_fingerprint(fingerprint)
                }
            severity_regressions = self._hard_severity_regressions(
                baseline_severity,
                self._hard_violation_severity_profile(),
            )
            if introduced or severity_regressions:
                console_logger.debug(
                    "Rejecting relation candidate with hard-state regression: %s",
                    "; ".join(sorted(introduced) + severity_regressions),
                )
                return False
            return True

        return validator

    @staticmethod
    def _is_window_hard_fingerprint(fingerprint: str) -> bool:
        return fingerprint.startswith("window:") or fingerprint.startswith(
            "hard:substantial_window_occlusion"
        )

    def _hard_violation_fingerprints(self) -> set[str]:
        """Return stable IDs for hard failures relevant to repair acceptance."""
        scene = getattr(self, "scene", None)
        if scene is None:
            return set()

        fingerprints: set[str] = set()
        hard_state = self._evaluate_current_furniture_hard_state()
        checkpoint_state = self._checkpoint_eligible_furniture_hard_state(hard_state)
        for reason in getattr(checkpoint_state or hard_state, "hard_reasons", []) or []:
            fingerprints.add(self._stable_hard_reason_id(str(reason)))
        try:
            for violation in compute_door_clearance_violations(scene):
                fingerprints.add(
                    "door:" f"{violation.door_label}:{violation.furniture_id}"
                )
        except Exception:
            console_logger.debug(
                "Could not collect structured door violations for repair transaction",
                exc_info=True,
            )
        try:
            for violation in compute_window_clearance_violations(scene):
                fingerprints.add(
                    "window:" f"{violation.window_label}:{violation.furniture_id}"
                )
        except Exception:
            console_logger.debug(
                "Could not collect structured window violations for repair transaction",
                exc_info=True,
            )
        try:
            for first_id, second_id, _depth in self._reported_furniture_collisions(
                self._get_cached_physics_context()
            ):
                pair = ":".join(sorted((first_id, second_id)))
                fingerprints.add(f"collision:{pair}")
        except Exception:
            console_logger.debug(
                "Could not collect structured collision pairs for repair transaction",
                exc_info=True,
            )
        return fingerprints

    @staticmethod
    def _stable_hard_reason_id(reason: str) -> str:
        text = " ".join(str(reason or "").lower().split())
        missing = re.search(r"missing required\s+([^:]+)", text)
        if missing:
            category = re.sub(r"[^a-z0-9_]+", "_", missing.group(1)).strip("_")
            return f"required_count:{category or 'unknown'}"
        relation = re.search(
            r"unresolved prompt-core furniture relation:\s*([^;]+)", text
        )
        if relation:
            identity = re.sub(r"\s+", "", relation.group(1))
            return f"relation:{identity}"
        for section in (
            "collisions",
            "door clearance violations",
            "open connection blocked",
            "wall height exceeded",
            "geometry construction failed",
            "fallen or below-floor",
        ):
            if section in text:
                return f"physics:{section.replace(' ', '_')}"
        normalized = re.sub(r"[^a-z0-9_]+", "_", text).strip("_")
        return f"hard:{normalized or 'unknown'}"

    def _hard_violation_severity_profile(self) -> dict[str, float]:
        """Return structured magnitudes for stable hard-state identities."""
        scene = getattr(self, "scene", None)
        if scene is None:
            return {}
        profile: dict[str, float] = {}
        try:
            for violation in compute_door_clearance_violations(scene):
                key = f"door:{violation.door_label}:{violation.furniture_id}"
                profile[key] = max(
                    profile.get(key, 0.0),
                    float(getattr(violation, "penetration_depth", 0.0) or 0.0),
                )
        except Exception:
            pass
        try:
            for violation in compute_window_clearance_violations(scene):
                key = f"window:{violation.window_label}:{violation.furniture_id}"
                top = float(getattr(violation, "furniture_top_height", 0.0) or 0.0)
                sill = float(getattr(violation, "sill_height", 0.0) or 0.0)
                profile[key] = max(profile.get(key, 0.0), max(0.0, top - sill))
        except Exception:
            pass
        try:
            for first_id, second_id, depth in self._reported_furniture_collisions(
                self._get_cached_physics_context()
            ):
                pair = ":".join(sorted((first_id, second_id)))
                key = f"collision:{pair}"
                profile[key] = max(profile.get(key, 0.0), float(depth))
        except Exception:
            pass
        return profile

    @staticmethod
    def _hard_severity_regressions(
        before: dict[str, float],
        after: dict[str, float],
        *,
        tolerance: float = 1e-6,
    ) -> list[str]:
        return [
            f"{key} worsened {before[key]:.6f}->{after[key]:.6f}"
            for key in sorted(before.keys() & after.keys())
            if after[key] > before[key] + tolerance
        ]

    def _begin_hard_state_transaction(
        self,
    ) -> tuple[dict[str, Any], set[str], dict[str, float]] | None:
        """Snapshot the scene before a deterministic geometry repair."""
        scene = getattr(self, "scene", None)
        serialize = getattr(scene, "to_state_dict", None)
        restore = getattr(scene, "restore_from_state_dict", None)
        if not callable(serialize) or not callable(restore):
            return None
        try:
            return (
                copy.deepcopy(serialize()),
                self._hard_violation_fingerprints(),
                self._hard_violation_severity_profile(),
            )
        except Exception:
            console_logger.debug(
                "Could not create hard-state repair transaction snapshot", exc_info=True
            )
            return None

    def _commit_hard_state_transaction(
        self,
        transaction: tuple[Any, ...] | None,
        *,
        source: str,
    ) -> bool:
        """Reject a candidate that adds any hard failure to its baseline."""
        if transaction is None:
            return True
        snapshot, before, *severity_state = transaction
        before_severity = severity_state[0] if severity_state else {}
        after = self._hard_violation_fingerprints()
        introduced = sorted(after - before)
        severity_regressions = self._hard_severity_regressions(
            before_severity,
            self._hard_violation_severity_profile(),
        )
        if not introduced and not severity_regressions:
            return True
        self.scene.restore_from_state_dict(snapshot)
        if getattr(self, "rendering_manager", None) is not None:
            self.rendering_manager.clear_cache()
        self._reset_critic_candidate_cache()
        console_logger.info(
            "Rejected deterministic %s repair because it introduced hard failures: %s",
            source,
            "; ".join(introduced + severity_regressions),
        )
        return False

    def _restore_hard_state_transaction(
        self,
        transaction: tuple[Any, ...] | None,
        *,
        source: str,
        reasons: list[str],
    ) -> None:
        """Restore an outer deterministic-repair snapshot after critic rejection."""
        if transaction is None:
            return
        snapshot, *_before = transaction
        self.scene.restore_from_state_dict(snapshot)
        if getattr(self, "rendering_manager", None) is not None:
            self.rendering_manager.clear_cache()
        self._reset_critic_candidate_cache()
        console_logger.info(
            "Rejected deterministic %s repair: %s",
            source,
            "; ".join(reasons),
        )

    @staticmethod
    def _critic_core_window_failures(payload: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            result
            for result in payload.get("results") or []
            if str(result.get("check_id") or "").startswith("window_clearance__")
            and str(result.get("scoring_tier") or "").lower() == "core"
            and str(result.get("label") or "").lower() == "fail"
        ]

    def _checkpoint_eligible_furniture_hard_state(
        self, hard_state: HardStateEvaluation | None
    ) -> HardStateEvaluation | None:
        """Keep substantially window-blocked scenes out of rollback checkpoints."""
        checkpoint_state = super()._checkpoint_eligible_furniture_hard_state(hard_state)
        if checkpoint_state is None or not checkpoint_state.hard_valid:
            return checkpoint_state
        critic_config = critic_config_from_any(getattr(self, "cfg", {}))
        if not critic_config.enabled or not critic_config.metric_enabled(
            "interaction_clearance"
        ):
            return checkpoint_state
        try:
            payload = evaluate_room_scene(
                self.scene,
                config=critic_config,
                stage="furniture_checkpoint_window_gate",
                annotate_assets=False,
            )
        except Exception:
            console_logger.warning(
                "Could not evaluate window clearance for checkpoint safety",
                exc_info=True,
            )
            return checkpoint_state
        window_failures = self._critic_core_window_failures(payload)
        if not window_failures:
            return checkpoint_state
        gated_state = copy.deepcopy(checkpoint_state)
        gated_state.hard_valid = False
        gated_state.hard_reasons.extend(
            f"substantial window occlusion: {result.get('check_id')}"
            for result in window_failures
            if f"substantial window occlusion: {result.get('check_id')}"
            not in gated_state.hard_reasons
        )
        return gated_state

    @staticmethod
    def _critic_core_failure_ids(payload: dict[str, Any]) -> set[str]:
        return {
            str(result.get("check_id") or f"result_{index}")
            for index, result in enumerate(payload.get("results") or [])
            if str(result.get("scoring_tier") or "core").lower() == "core"
            and str(result.get("label") or "").lower() == "fail"
        }

    @staticmethod
    def _critic_passing_explicit_relation_ids(payload: dict[str, Any]) -> set[str]:
        passing: set[str] = set()
        for index, result in enumerate(payload.get("results") or []):
            if str(result.get("label") or "").lower() != "pass":
                continue
            constraint = (result.get("evidence") or {}).get("intent_constraint") or {}
            if (
                not constraint
                or str(constraint.get("relation") or "") == "required_count"
            ):
                continue
            if str(constraint.get("strength") or "hard").lower() != "hard":
                continue
            passing.add(str(result.get("check_id") or f"result_{index}"))
        return passing

    @staticmethod
    def _critic_explicit_relation_ids_for_objects(
        payload: dict[str, Any],
        object_ids: set[str],
    ) -> set[str]:
        """Return hard prompt relation checks that bind any selected object."""
        relation_ids: set[str] = set()
        for index, result in enumerate(payload.get("results") or []):
            constraint = (result.get("evidence") or {}).get("intent_constraint") or {}
            if (
                not constraint
                or str(constraint.get("relation") or "") == "required_count"
                or str(constraint.get("strength") or "hard").lower() != "hard"
            ):
                continue
            bound_ids = {
                str(result.get("primary_object") or ""),
                *(str(value) for value in result.get("related_objects") or [] if value),
                *(
                    str(value)
                    for value in result.get("selected_related_objects") or []
                    if value
                ),
            }
            if bound_ids & object_ids:
                relation_ids.add(str(result.get("check_id") or f"result_{index}"))
        return relation_ids

    def _get_window_repair_service(self) -> Any | None:
        existing = getattr(self, "_window_repair_service", None)
        if existing is not None:
            return existing
        if getattr(self, "house_layout", None) is None:
            return None
        floor_plan_cfg = getattr(self.cfg, "floor_plan_geometry_config", None)
        if floor_plan_cfg is None:
            return None
        from scenesmith.wall_agents.tools.window_tools import WindowRepairTools

        self._window_repair_service = WindowRepairTools(
            scene=self.scene,
            house_layout=self.house_layout,
            floor_plan_cfg=floor_plan_cfg,
            room_output_dir=self.logger.output_dir,
            refresh_wall_surfaces=lambda: None,
            rendering_manager=self.rendering_manager,
            logger=self.logger,
        )
        return self._window_repair_service

    def _repair_substantial_window_clearance(self) -> list[str]:
        """Migrate blocked windows first, then fall back to moving furniture."""
        if self.scene is None:
            return []
        critic_config = critic_config_from_any(self.cfg)
        if not critic_config.enabled or not critic_config.metric_enabled(
            "interaction_clearance"
        ):
            return []
        baseline = evaluate_room_scene(
            self.scene,
            config=critic_config,
            stage="furniture_window_clearance_repair_baseline",
            annotate_assets=False,
        )
        target_results = self._critic_core_window_failures(baseline)
        if not target_results:
            return []
        if not any(result.get("primary_object") for result in target_results):
            return []

        actions: list[str] = []
        service = self._get_window_repair_service()
        current_baseline = baseline
        remaining_results = list(target_results)
        if service is not None:
            for target in sorted(
                target_results,
                key=lambda item: str(item.get("check_id") or ""),
            ):
                window_id = str(target.get("primary_object") or "")
                check_id = str(target.get("check_id") or "")
                if not window_id or not check_id:
                    continue
                baseline_core = self._critic_core_failure_ids(current_baseline)
                protected_relations = self._critic_passing_explicit_relation_ids(
                    current_baseline
                )
                window_relations = self._critic_explicit_relation_ids_for_objects(
                    current_baseline,
                    {window_id},
                )
                accepted_payload: dict[str, Any] | None = None

                def accept_candidate(
                    _candidate: dict[str, Any],
                ) -> tuple[bool, str]:
                    nonlocal accepted_payload
                    self._reset_critic_candidate_cache()
                    fresh_payload = evaluate_room_scene(
                        self.scene,
                        config=critic_config,
                        stage="furniture_window_migration_candidate",
                        annotate_assets=False,
                    )
                    results_by_id = {
                        str(result.get("check_id") or ""): result
                        for result in fresh_payload.get("results") or []
                    }
                    reasons: list[str] = []
                    target_result = results_by_id.get(check_id)
                    if target_result is None:
                        reasons.append("target window check disappeared")
                    elif str(target_result.get("label") or "").lower() != "pass":
                        reasons.append(
                            "target window clearance is not pass "
                            f"({target_result.get('label') or 'unknown'})"
                        )
                    fresh_core = self._critic_core_failure_ids(fresh_payload)
                    introduced = sorted(fresh_core - baseline_core)
                    if introduced:
                        reasons.append("new core failures: " + ", ".join(introduced))
                    fresh_relations = self._critic_passing_explicit_relation_ids(
                        fresh_payload
                    )
                    lost = sorted(protected_relations - fresh_relations)
                    if lost:
                        reasons.append("lost explicit relations: " + ", ".join(lost))
                    unresolved_window_relations = sorted(
                        window_relations - fresh_relations
                    )
                    if unresolved_window_relations:
                        reasons.append(
                            "window relation is not satisfied: "
                            + ", ".join(unresolved_window_relations)
                        )
                    if reasons:
                        return False, "; ".join(reasons)
                    accepted_payload = fresh_payload
                    return True, "accepted"

                migration = service.migrate_window_atomically(
                    window_id=window_id,
                    accept_candidate=accept_candidate,
                )
                if not migration.success:
                    console_logger.info(
                        "No accepted migration for %s; furniture fallback remains "
                        "eligible: %s",
                        window_id,
                        migration.reason,
                    )
                    continue
                actions.append(
                    f"migrated {window_id} from {migration.old_wall_direction}"
                    f"@{migration.old_position_along_wall:.3f} to "
                    f"{migration.new_wall_direction}"
                    f"@{migration.new_position_along_wall:.3f}"
                )
                if accepted_payload is not None:
                    current_baseline = accepted_payload
                remaining_results = [
                    result
                    for result in remaining_results
                    if str(result.get("primary_object") or "") != window_id
                ]

        if not remaining_results:
            self._reset_critic_candidate_cache()
            return actions

        target_check_ids = {str(result.get("check_id")) for result in remaining_results}
        target_window_ids = {
            str(result.get("primary_object") or "")
            for result in remaining_results
            if result.get("primary_object")
        }
        target_blocker_ids = {
            str(object_id)
            for result in remaining_results
            for object_id in (
                result.get("blocking_objects")
                or (result.get("diagnostics") or {}).get("core_blocking_objects")
                or []
            )
        }
        if not target_window_ids or not target_blocker_ids:
            return actions

        transaction = self._begin_hard_state_transaction()
        baseline_core_failures = self._critic_core_failure_ids(current_baseline)
        protected_relations = self._critic_passing_explicit_relation_ids(
            current_baseline
        )
        changed = self._repair_forbidden_zone_conflicts(
            include_windows=True,
            opening_ids=target_window_ids,
            blocker_ids=target_blocker_ids,
        )
        if not changed:
            return actions
        self.rendering_manager.clear_cache()
        self._reset_critic_candidate_cache()
        fresh = evaluate_room_scene(
            self.scene,
            config=critic_config,
            stage="furniture_window_clearance_repair_fresh",
            annotate_assets=False,
        )
        fresh_result_ids = {
            str(result.get("check_id") or "") for result in fresh.get("results") or []
        }
        fresh_core_failures = self._critic_core_failure_ids(fresh)
        fresh_relations = self._critic_passing_explicit_relation_ids(fresh)
        rejection_reasons: list[str] = []
        if not target_check_ids.issubset(fresh_result_ids):
            rejection_reasons.append("target window check disappeared")
        unresolved = sorted(target_check_ids & fresh_core_failures)
        if unresolved:
            rejection_reasons.append("target blockers remain: " + ", ".join(unresolved))
        introduced = sorted(fresh_core_failures - baseline_core_failures)
        if introduced:
            rejection_reasons.append("new core failures: " + ", ".join(introduced))
        lost_relations = sorted(protected_relations - fresh_relations)
        if lost_relations:
            rejection_reasons.append(
                "lost explicit relations: " + ", ".join(lost_relations)
            )
        if rejection_reasons:
            self._restore_hard_state_transaction(
                transaction,
                source="window-clearance",
                reasons=rejection_reasons,
            )
            return actions
        actions.append(
            "cleared substantial window occlusion "
            f"for {', '.join(sorted(target_window_ids))} by moving "
            f"{', '.join(sorted(target_blocker_ids))}"
        )
        return actions

    def _align_seating_with_hard_state_guard(
        self, allowed_targets_by_seat: dict[str, set[str]]
    ) -> list[Any]:
        """Apply seating orientation only when it preserves all hard invariants."""
        transaction = self._begin_hard_state_transaction()
        fixes = align_seating_to_nearest_surface(
            self.scene,
            allowed_targets_by_seat=allowed_targets_by_seat,
        )
        if fixes and not self._commit_hard_state_transaction(
            transaction, source="seating orientation"
        ):
            return []
        return fixes

    def _repair_contract_layout_before_first_critique(self) -> list[str]:
        """Repair contract-authorized furniture relations before first critique.

        Both repair mechanisms are geometry-only and retain their own
        whole-scene acceptance/rollback checks.  In particular, this does not
        ask an LLM or VLM to infer a pose, and it never activates from a
        StageBrief or the current layout.
        """
        critic_config = critic_config_from_any(self.cfg)
        if not critic_config.enabled or not critic_config.metric_enabled(
            "functional_dependency"
        ):
            return []

        transaction = self._begin_hard_state_transaction()
        baseline_hard_failures = (
            transaction[1]
            if transaction is not None
            else self._hard_violation_fingerprints()
        )
        relation_fixes = improve_furniture_relations(
            self.scene,
            config=critic_config,
            candidate_validator=self._relation_candidate_preserves_hard_baseline(
                baseline_hard_failures,
                allow_deferred_window_repair=True,
            ),
        )
        window_actions = (
            self._repair_substantial_window_clearance() if relation_fixes else []
        )
        seating_fixes = self._align_seating_with_hard_state_guard(
            seating_orientation_targets(self.scene, config=critic_config)
        )
        if not relation_fixes and not seating_fixes and not window_actions:
            return []
        if not self._commit_hard_state_transaction(
            transaction, source="prompt-contract relation"
        ):
            return []

        # The next automatic critic request must render and evaluate the pose
        # just accepted above rather than use the designer's pre-repair cache.
        self.rendering_manager.clear_cache()
        self._reset_critic_candidate_cache()
        actions = [f"{fix.object_id}:{fix.relation_type}" for fix in relation_fixes] + [
            f"{fix.subject_id}->{fix.target_id}:seating_orientation"
            for fix in seating_fixes
        ]
        actions.extend(window_actions)
        affected_objects = [
            {
                "object_id": fix.object_id,
                "relation_type": fix.relation_type,
                "check_id": getattr(fix, "check_id", None),
                "before": {
                    "xy": list(getattr(fix, "old_xy", ()) or ()),
                    "yaw_deg": getattr(fix, "old_yaw_deg", None),
                },
                "after": {
                    "xy": list(getattr(fix, "new_xy", ()) or ()),
                    "yaw_deg": getattr(fix, "new_yaw_deg", None),
                },
            }
            for fix in relation_fixes
        ] + [
            {
                "object_id": fix.subject_id,
                "target_id": fix.target_id,
                "relation_type": "seating_orientation",
                "before": {"yaw_deg": getattr(fix, "old_yaw_deg", None)},
                "after": {"yaw_deg": getattr(fix, "new_yaw_deg", None)},
                "angle_to_target_deg": getattr(fix, "angle_to_target_deg", None),
            }
            for fix in seating_fixes
        ]
        working_memory = getattr(self, "stage_working_memory", None)
        if working_memory is not None:
            working_memory.record_repair_event(
                source="pre_first_critique",
                strategy="prompt_contract_furniture_relations",
                status="accepted",
                trigger_reasons=[
                    str(item.get("check_id") or item["relation_type"])
                    for item in affected_objects
                ],
                actions=actions,
                affected_objects=affected_objects,
                detail={
                    "relation_fix_count": len(relation_fixes),
                    "seating_fix_count": len(seating_fixes),
                },
            )
        console_logger.info(
            "Prompt-contract furniture repair before first critique: %s",
            "; ".join(actions),
        )
        return actions

    def _repair_initial_contract_layout(self) -> list[str]:
        """Preserve the initial-design repair hook used by planner integrations."""
        return self._repair_contract_layout_before_first_critique()

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

        required_counts = self._repair_required_counts()
        inventory_changed = False
        for category in required_counts:
            if not self._category_matches_missing_reason(category, reasons):
                continue
            added = self._ensure_required_furniture_asset(category)
            if added:
                inventory_changed = True
                actions.append(
                    f"added {added} missing {category} asset(s) from local/HSSD bank"
                )

        if "geometry construction failed" in reasons:
            replaced = self._replace_geometry_failed_furniture_assets(reasons)
            if replaced:
                inventory_changed = True
                actions.append(
                    f"replaced {replaced} geometry-failed furniture asset(s)"
                )
        if "wall height exceeded" in reasons:
            grounded = self._ground_elevated_floor_furniture()
            if grounded:
                actions.append(
                    f"grounded {grounded} elevated floor furniture object(s)"
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
                # Clear independent mesh contacts before the support repair.
                # Its whole-scene hard-state gate would otherwise reject a
                # correct Z-axis support move because of an unrelated pair.
                # The shallow repair itself preserves hard support pairs, so it
                # cannot horizontally split an object from its required surface.
                actions.extend(self._repair_shallow_furniture_collisions())
                actions.extend(self._repair_bounded_furniture_collisions())
                actions.extend(
                    self._repair_prompt_contract_relations(
                        "after physical collision repair"
                    )
                )
                # A relation target may occupy the same region as an old
                # collision. Check once more so an existing hard fingerprint
                # cannot hide a pair reintroduced by relation repair.
                actions.extend(self._repair_bounded_furniture_collisions())
            removed_excess = self._remove_excess_required_furniture(required_counts)
            if removed_excess:
                actions.append(
                    f"removed {removed_excess} duplicate prompt-required furniture asset(s)"
                )
            if inventory_changed:
                if self._repair_forbidden_zone_conflicts(include_windows=False):
                    actions.append("cleared deterministic door/opening forbidden zones")
                # Newly restored inventory is placed one object at a time. Even
                # the low-overlap placement heuristic can leave a tiny mesh
                # collision (for example a chair and a floor plant). Clear that
                # bounded geometry residue before relation repair: its strict
                # whole-scene validator otherwise rejects an otherwise valid
                # support or seating candidate because of an unrelated pair.
                actions.extend(self._repair_shallow_furniture_collisions())
                actions.extend(self._repair_relations_after_inventory_change())
            elif "unresolved prompt-core furniture relation" in reasons:
                # A later design change can break a hard prompt relation even
                # when the inventory is already complete.  Re-run the same
                # geometry-only repair path so the safety controller has a
                # deterministic recovery before rejecting the stage.
                actions.extend(self._repair_unresolved_prompt_contract_relations())
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
        if inventory_changed:
            if self._repair_forbidden_zone_conflicts(include_windows=False):
                actions.append("cleared deterministic door/opening forbidden zones")
            actions.extend(self._repair_relations_after_inventory_change())
            # A free-wall relation is an explicit request to reserve a separate
            # wall for its subject.  Once relation repair has anchored that
            # subject, re-evaluate storage anchors so a wardrobe does not keep
            # the same wall merely because it was placed before the relation.
            if (
                self._free_wall_anchor_objects(excluding_object_id="wardrobe_0")
                and self._repair_wardrobe_wall_anchor()
            ):
                actions.append(
                    "moved wardrobe away from a prompt-reserved free-wall anchor"
                )
                actions.extend(self._repair_relations_after_inventory_change())
        elif "unresolved prompt-core furniture relation" in reasons:
            actions.extend(self._repair_unresolved_prompt_contract_relations())

        # Bedroom anchors and relation repairs can reintroduce a wall or opening
        # conflict even when the current hard failure came from another source.
        # Always run both geometry-only safety passes last so their final pose is
        # authoritative for the next hard-state evaluation.
        if self._repair_forbidden_zone_conflicts(include_windows=False):
            actions.append("revalidated deterministic door/opening forbidden zones")
        if "collisions" in reasons:
            if self._repair_generic_wall_collisions():
                actions.append("moved bedroom furniture away from room-wall collisions")
            # Moving a bed out of a door or window zone can leave a shallow
            # collision with an optional foot bench or other small furniture.
            # Use the same bounded geometry repair as other rooms after the
            # bed/nightstand poses have settled.
            actions.extend(self._repair_shallow_furniture_collisions())

        return bool(actions), actions

    def _ground_elevated_floor_furniture(
        self, tolerance_m: float = WALL_HEIGHT_TOLERANCE_M
    ) -> int:
        """Lower accidentally elevated furniture whose top exceeds wall height."""
        if self.scene is None or self.scene.room_geometry is None:
            return 0
        wall_height = float(
            getattr(self.scene.room_geometry, "wall_height", 0.0) or 0.0
        )
        if wall_height <= 0.0:
            return 0

        grounded = 0
        for obj in self.scene.objects.values():
            if obj.object_type != ObjectType.FURNITURE:
                continue
            try:
                bounds = obj.compute_world_bounds()
            except Exception:
                continue
            if bounds is None:
                continue
            bottom_z = float(bounds[0][2])
            top_z = float(bounds[1][2])
            if (
                not np.isfinite(bottom_z)
                or not np.isfinite(top_z)
                or bottom_z <= tolerance_m
                or top_z <= wall_height + tolerance_m
            ):
                continue
            translation = np.asarray(obj.transform.translation(), dtype=float).copy()
            translation[2] -= bottom_z
            transform = RigidTransform(R=obj.transform.rotation(), p=translation)
            self.scene.move_object(obj.object_id, transform)
            grounded += 1
        return grounded

    def _repair_relations_after_inventory_change(self) -> list[str]:
        """Bind newly added furniture into the prompt's hard relations."""
        return self._repair_prompt_contract_relations("after inventory repair")

    def _repair_unresolved_prompt_contract_relations(self) -> list[str]:
        """Repair hard prompt relations after a design change without inventory churn."""
        return self._repair_prompt_contract_relations("after hard constraint failure")

    def _repair_prompt_contract_relations(self, action_context: str) -> list[str]:
        """Apply prompt-authorized geometry repairs and report their trigger context."""
        critic_config = critic_config_from_any(self.cfg)
        if not critic_config.enabled or not critic_config.metric_enabled(
            "functional_dependency"
        ):
            return []

        transaction = self._begin_hard_state_transaction()
        baseline_hard_failures = (
            transaction[1]
            if transaction is not None
            else self._hard_violation_fingerprints()
        )
        relation_fixes = improve_furniture_relations(
            self.scene,
            config=critic_config,
            candidate_validator=self._relation_candidate_preserves_hard_baseline(
                baseline_hard_failures,
                allow_deferred_window_repair=True,
            ),
        )
        window_actions = (
            self._repair_substantial_window_clearance() if relation_fixes else []
        )
        seating_fixes = self._align_seating_with_hard_state_guard(
            seating_orientation_targets(self.scene, config=critic_config)
        )
        if (
            relation_fixes or seating_fixes
        ) and not self._commit_hard_state_transaction(
            transaction, source="prompt-contract relation"
        ):
            return []
        actions = [
            f"bound {fix.object_id} via {fix.relation_type} {action_context}"
            for fix in relation_fixes
        ]
        actions.extend(window_actions)
        actions.extend(
            f"aligned {fix.subject_id} toward {fix.target_id} {action_context}"
            for fix in seating_fixes
        )
        return actions

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

        bed_head_wall: str | None = None
        if is_bedroom_scene(self.scene):
            plan = build_bedroom_anchor_plan(
                self.scene,
                self._bedroom_layout_cfg(),
            )
            bed_head_wall = plan.bed_head_wall if plan else "north"

        changed = False
        for obj in self.scene.objects.values():
            if getattr(obj, "immutable", False):
                continue
            if getattr(obj, "object_type", None) != ObjectType.FURNITURE:
                continue
            transform = self._fit_transform_inside_room(obj, obj.transform)
            if (
                bed_head_wall is not None
                and self._category_for_object(obj.object_id, obj) == "bed"
            ):
                # The conservative generic margin can undo the narrow lateral
                # slot selected by the bedroom anchor and re-block its window.
                transform = self._opening_safe_bed_transform(
                    bed=obj,
                    transform=transform,
                    wall=bed_head_wall,
                    fallback=transform,
                )
            if self._transform_close(obj.transform, transform):
                continue
            self.scene.move_object(obj.object_id, transform)
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
            if self._is_hard_prompt_support_pair(first_id, second_id):
                # The support repair owns this pair and moves the subject in Z.
                # Separating it in XY would invalidate an explicit on_top_of
                # contract before that repair gets a chance to run.
                continue
            # Earlier bedroom repairs can already have resolved a collision
            # described by the cached physics context. Never move furniture
            # for an obsolete report.
            if frozenset((first_id, second_id)) not in before_pairs:
                continue
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
            moved_ids = self._apply_best_collision_repair_candidate(
                first=first,
                second=second,
                penetration=penetration,
                clearance=clearance,
                before_pairs=before_pairs,
            )
            if moved_ids:
                return [
                    "separated shallow collision "
                    f"{first_id}<->{second_id} by moving {','.join(moved_ids)}"
                ]
        return []

    def _is_hard_prompt_support_pair(self, first_id: str, second_id: str) -> bool:
        """Whether a collision is the two endpoints of a hard ``on_top_of``.

        Surface support normally implies XY overlap. Treating that overlap as a
        generic furniture collision can push a TV off its console or a monitor
        off its desk, even though the deterministic relation repair has the
        correct Z-axis placement. The check intentionally reads only immutable
        hard prompt contracts, not inferred functional-dependency proposals.
        """
        if self.scene is None:
            return False
        first = self.scene.objects.get(first_id)
        second = self.scene.objects.get(second_id)
        if first is None or second is None:
            return False
        first_category = self._category_for_object(first_id, first)
        second_category = self._category_for_object(second_id, second)
        if not first_category or not second_category:
            return False

        contract = getattr(self.scene, "scenebenchmark_intent_contract", None)
        constraints = contract.get("constraints") if isinstance(contract, dict) else []
        for constraint in constraints or []:
            if not isinstance(constraint, dict):
                continue
            if (
                str(constraint.get("relation") or "") != "on_top_of"
                or str(constraint.get("strength") or "hard").lower() != "hard"
            ):
                continue
            relation_stage = str(constraint.get("stage") or "").strip().lower()
            if relation_stage not in {"", "furniture"}:
                # A future-stage support contract cannot protect a current
                # collision from the generic separator before its subject exists.
                continue
            subjects = constraint.get("subjects") or {}
            targets = constraint.get("targets") or {}
            subject_category = self._repair_category_for_task_label(
                str(subjects.get("category") or "")
            )
            target_category = self._repair_category_for_task_label(
                str(targets.get("category") or "")
            )
            if not subject_category or not target_category:
                continue
            if (
                furniture_category_satisfies(first_category, subject_category)
                and furniture_category_satisfies(second_category, target_category)
            ) or (
                furniture_category_satisfies(second_category, subject_category)
                and furniture_category_satisfies(first_category, target_category)
            ):
                subject_obj, support_obj = (
                    (first, second)
                    if furniture_category_satisfies(first_category, subject_category)
                    else (second, first)
                )
                if (
                    subject_obj.object_type != ObjectType.FURNITURE
                    or support_obj.object_type != ObjectType.FURNITURE
                ):
                    continue
                subject_bounds = subject_obj.compute_world_bounds()
                support_bounds = support_obj.compute_world_bounds()
                if subject_bounds is None or support_bounds is None:
                    continue
                subject_height = float(subject_bounds[1][2] - subject_bounds[0][2])
                support_width = float(support_bounds[1][0] - support_bounds[0][0])
                support_depth = float(support_bounds[1][1] - support_bounds[0][1])
                if subject_height > 0.0 and support_width > 0.0 and support_depth > 0.0:
                    return True
        return False

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

    def _reported_furniture_collisions(
        self, physics_context: str
    ) -> list[tuple[str, str, float]]:
        """Parse all object-addressable furniture collisions for bounded search."""
        scale = {"mm": 0.001, "cm": 0.01, "m": 1.0}
        result: list[tuple[str, str, float]] = []
        for match in _SHALLOW_FURNITURE_COLLISION_RE.finditer(
            str(physics_context or "")
        ):
            penetration = (
                float(match.group("depth")) * scale[match.group("unit").lower()]
            )
            if penetration > 0.0:
                result.append(
                    (match.group("first"), match.group("second"), penetration)
                )
        return result

    def _repair_bounded_furniture_collisions(self) -> list[str]:
        """Separate one deep collision using bounded geometry candidates."""
        if self.scene is None:
            return []
        room_bounds = self._room_bounds_xy()
        if room_bounds is None:
            return []
        min_x, min_y, max_x, max_y = room_bounds
        max_translation = min(1.5, math.hypot(max_x - min_x, max_y - min_y))
        clearance = max(
            0.005,
            float(self._repair_cfg_value("collision_separation_margin_m", 0.025)),
        )
        reported = self._reported_furniture_collisions(
            self._get_cached_physics_context()
        )
        if not reported:
            return []

        objects_by_id = {
            str(object_id): obj for object_id, obj in self.scene.objects.items()
        }
        before_pairs = self._furniture_aabb_overlap_pairs()
        for first_id, second_id, penetration in reported:
            pair = frozenset((first_id, second_id))
            if pair not in before_pairs or self._is_hard_prompt_support_pair(
                first_id, second_id
            ):
                continue
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
            moved_ids = self._apply_best_collision_repair_candidate(
                first=first,
                second=second,
                penetration=penetration,
                clearance=clearance,
                before_pairs=before_pairs,
                max_translation=max_translation,
            )
            if moved_ids:
                return [
                    "separated bounded collision "
                    f"{first_id}<->{second_id} by moving {','.join(moved_ids)}"
                ]
        return []

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

    def _apply_best_collision_repair_candidate(
        self,
        *,
        first: SceneObject,
        second: SceneObject,
        penetration: float,
        clearance: float,
        before_pairs: set[frozenset[str]],
        max_translation: float | None = None,
    ) -> tuple[str, ...]:
        """Rank single-object and relation-group collision ejection candidates."""
        target_pair = frozenset((str(first.object_id), str(second.object_id)))
        baseline_relations = self._hard_relation_failure_ids()
        baseline_openings = self._opening_violation_count()
        ranked: list[
            tuple[
                tuple[int, float, int, int, float, float, tuple[str, ...]],
                dict[str, RigidTransform],
            ]
        ] = []
        movable = [
            obj
            for obj in (first, second)
            if not getattr(obj, "immutable", False)
            and getattr(obj, "object_type", None) == ObjectType.FURNITURE
        ]
        movable.sort(key=self._collision_repair_candidate_key)
        for moving in movable:
            other = second if moving is first else first
            transforms = self._safe_shallow_collision_transforms(
                moving,
                other,
                penetration=penetration,
                clearance=clearance,
                before_pairs=before_pairs,
                max_translation=max_translation,
            )
            relation_groups = self._hard_relation_groups_for_object(
                str(moving.object_id)
            )
            for transform in transforms:
                override_candidates = [
                    {str(moving.object_id): transform},
                    *self._rigid_group_collision_overrides(
                        moving,
                        transform,
                        relation_groups,
                    ),
                ]
                for overrides in override_candidates:
                    after_pairs = self._furniture_aabb_overlap_pairs(
                        overrides=overrides
                    )
                    if target_pair in after_pairs or after_pairs - before_pairs:
                        continue
                    metrics = self._collision_candidate_metrics(overrides)
                    if metrics is None:
                        continue
                    pair_count, max_depth, relation_ids, opening_count, containment = (
                        metrics
                    )
                    if target_pair in relation_ids.get("collision_pairs", set()):
                        continue
                    relation_failures = set(
                        relation_ids.get("relation_failures", set())
                    )
                    if relation_failures - baseline_relations or containment:
                        continue
                    movement = sum(
                        float(
                            np.linalg.norm(
                                np.asarray(candidate.translation(), dtype=float)[:2]
                                - np.asarray(
                                    self.scene.objects[
                                        object_id
                                    ].transform.translation(),
                                    dtype=float,
                                )[:2]
                            )
                        )
                        for object_id, candidate in overrides.items()
                    )
                    moved_area = sum(
                        self._collision_repair_candidate_key(
                            self.scene.objects[object_id]
                        )[0]
                        for object_id in overrides
                    )
                    moved_ids = tuple(sorted(overrides))
                    score = (
                        pair_count,
                        max_depth,
                        len(relation_failures),
                        max(0, opening_count - baseline_openings),
                        movement,
                        moved_area,
                        moved_ids,
                    )
                    ranked.append((score, overrides))
        if not ranked:
            return ()
        ranked.sort(key=lambda item: item[0])
        best = ranked[0][1]
        for object_id, transform in best.items():
            self.scene.move_object(self.scene.objects[object_id].object_id, transform)
        return tuple(sorted(best))

    def _rigid_group_collision_overrides(
        self,
        moving: SceneObject,
        moving_transform: RigidTransform,
        relation_groups: list[tuple[str, ...]],
    ) -> list[dict[str, RigidTransform]]:
        old_translation = np.asarray(moving.transform.translation(), dtype=float)
        delta = (
            np.asarray(moving_transform.translation(), dtype=float) - old_translation
        )
        candidates: list[dict[str, RigidTransform]] = []
        for group in relation_groups:
            overrides: dict[str, RigidTransform] = {}
            valid = True
            for object_id in group:
                obj = self.scene.objects.get(object_id)
                if (
                    obj is None
                    or obj.object_type != ObjectType.FURNITURE
                    or getattr(obj, "immutable", False)
                ):
                    valid = False
                    break
                candidate = RigidTransform(
                    R=obj.transform.rotation(),
                    p=np.asarray(obj.transform.translation(), dtype=float) + delta,
                )
                fitted = self._fit_transform_inside_room(obj, candidate)
                if (
                    float(
                        np.linalg.norm(
                            np.asarray(fitted.translation(), dtype=float)
                            - np.asarray(candidate.translation(), dtype=float)
                        )
                    )
                    > 1e-5
                ):
                    valid = False
                    break
                overrides[object_id] = fitted
            if valid and len(overrides) > 1:
                candidates.append(overrides)
        return candidates

    def _hard_relation_failure_ids(self) -> set[str]:
        try:
            return {
                str(result.get("check_id") or "")
                for result in unresolved_furniture_relation_failures(
                    self.scene,
                    config=critic_config_from_any(self.cfg),
                )
                if result.get("check_id")
            }
        except Exception:
            return set()

    def _hard_relation_groups_for_object(
        self,
        object_id: str,
    ) -> list[tuple[str, ...]]:
        try:
            payload = evaluate_room_scene(
                self.scene,
                config=critic_config_from_any(self.cfg),
                stage="furniture_collision_relation_groups",
                annotate_assets=False,
            )
        except Exception:
            return []
        groups: set[tuple[str, ...]] = set()
        for result in payload.get("results") or []:
            constraint = (result.get("evidence") or {}).get("intent_constraint") or {}
            if str(constraint.get("strength") or "").lower() != "hard":
                continue
            ids = {
                str(result.get("primary_object") or ""),
                *(
                    str(value)
                    for value in (
                        result.get("selected_related_objects")
                        or result.get("related_objects")
                        or []
                    )
                    if value
                ),
            }
            ids.discard("")
            furniture_ids = tuple(
                sorted(
                    candidate_id
                    for candidate_id in ids
                    if candidate_id in self.scene.objects
                    and self.scene.objects[candidate_id].object_type
                    == ObjectType.FURNITURE
                )
            )
            if object_id in furniture_ids and len(furniture_ids) > 1:
                groups.add(furniture_ids)
        return sorted(groups)

    def _opening_violation_count(self) -> int:
        try:
            return len(compute_door_clearance_violations(self.scene)) + len(
                compute_window_clearance_violations(self.scene)
            )
        except Exception:
            return 0

    def _collision_candidate_metrics(
        self,
        overrides: dict[str, RigidTransform],
    ) -> tuple[int, float, dict[str, Any], int, int] | None:
        originals = {
            object_id: self.scene.objects[object_id].transform
            for object_id in overrides
            if object_id in self.scene.objects
        }
        if len(originals) != len(overrides):
            return None
        try:
            for object_id, transform in overrides.items():
                self.scene.objects[object_id].transform = transform
            collision_pairs: set[frozenset[str]] = set()
            max_depth = 0.0
            try:
                for collision in compute_scene_collisions(self.scene):
                    first_id = str(collision.object_a_id)
                    second_id = str(collision.object_b_id)
                    first_obj = self.scene.objects.get(first_id)
                    second_obj = self.scene.objects.get(second_id)
                    if (
                        first_obj is None
                        or second_obj is None
                        or first_obj.object_type != ObjectType.FURNITURE
                        or second_obj.object_type != ObjectType.FURNITURE
                    ):
                        continue
                    collision_pairs.add(frozenset((first_id, second_id)))
                    max_depth = max(
                        max_depth,
                        float(getattr(collision, "penetration_depth", 0.0) or 0.0),
                    )
            except Exception:
                collision_pairs = self._furniture_aabb_overlap_pairs()
                max_depth = self._maximum_furniture_aabb_penetration()
            relation_failures = self._hard_relation_failure_ids()
            opening_count = self._opening_violation_count()
            containment = self._furniture_containment_violation_count()
            return (
                len(collision_pairs),
                max_depth,
                {
                    "collision_pairs": collision_pairs,
                    "relation_failures": relation_failures,
                },
                opening_count,
                containment,
            )
        finally:
            for object_id, transform in originals.items():
                self.scene.objects[object_id].transform = transform

    def _maximum_furniture_aabb_penetration(self) -> float:
        furniture = [
            obj
            for obj in self.scene.objects.values()
            if obj.object_type == ObjectType.FURNITURE
            and obj.compute_world_bounds() is not None
        ]
        maximum = 0.0
        for index, first in enumerate(furniture):
            first_bounds = first.compute_world_bounds()
            if first_bounds is None:
                continue
            for second in furniture[index + 1 :]:
                second_bounds = second.compute_world_bounds()
                if second_bounds is None:
                    continue
                overlap_x, overlap_y = self._xy_overlap_depths(
                    first_bounds,
                    second_bounds,
                )
                overlap_z = max(
                    0.0,
                    float(
                        min(first_bounds[1][2], second_bounds[1][2])
                        - max(first_bounds[0][2], second_bounds[0][2])
                    ),
                )
                if overlap_x > 0.0 and overlap_y > 0.0 and overlap_z > 0.0:
                    maximum = max(maximum, min(overlap_x, overlap_y, overlap_z))
        return maximum

    def _furniture_containment_violation_count(self) -> int:
        room_bounds = self._room_bounds_xy()
        if room_bounds is None:
            return 0
        min_x, min_y, max_x, max_y = room_bounds
        count = 0
        for obj in self.scene.objects.values():
            if obj.object_type != ObjectType.FURNITURE:
                continue
            bounds = obj.compute_world_bounds()
            if bounds is None:
                continue
            if (
                float(bounds[0][0]) < min_x - 1e-6
                or float(bounds[1][0]) > max_x + 1e-6
                or float(bounds[0][1]) < min_y - 1e-6
                or float(bounds[1][1]) > max_y + 1e-6
            ):
                count += 1
        return count

    def _safe_shallow_collision_transform(
        self,
        moving: SceneObject,
        other: SceneObject,
        *,
        penetration: float,
        clearance: float,
        before_pairs: set[frozenset[str]],
        max_translation: float | None = None,
    ) -> RigidTransform | None:
        candidates = self._safe_shallow_collision_transforms(
            moving,
            other,
            penetration=penetration,
            clearance=clearance,
            before_pairs=before_pairs,
            max_translation=max_translation,
        )
        return candidates[0] if candidates else None

    def _safe_shallow_collision_transforms(
        self,
        moving: SceneObject,
        other: SceneObject,
        *,
        penetration: float,
        clearance: float,
        before_pairs: set[frozenset[str]],
        max_translation: float | None = None,
    ) -> list[RigidTransform]:
        moving_bounds = moving.compute_world_bounds()
        other_bounds = other.compute_world_bounds()
        if moving_bounds is None or other_bounds is None:
            return []

        allowed_axes = self._collision_separation_axes(moving)
        if not allowed_axes:
            return []
        overlap_x, overlap_y = self._xy_overlap_depths(moving_bounds, other_bounds)
        axes = sorted(allowed_axes, key=lambda axis: (overlap_x, overlap_y)[axis])
        old_translation = np.asarray(moving.transform.translation(), dtype=float)
        other_translation = np.asarray(other.transform.translation(), dtype=float)
        # Drake reports mesh penetration, while the conservative AABB used to
        # reject new conflicts can overlap farther along the chosen axis.  The
        # larger value guarantees that this candidate clears both signals.

        candidates: list[RigidTransform] = []
        for axis in axes:
            separation = max(penetration, (overlap_x, overlap_y)[axis]) + clearance
            delta = old_translation[axis] - other_translation[axis]
            signs = (-1.0, 1.0) if abs(delta) < 1e-4 else (1.0 if delta > 0 else -1.0,)
            for sign in signs:
                translation = old_translation.copy()
                translation[axis] += sign * separation
                candidate = RigidTransform(R=moving.transform.rotation(), p=translation)
                candidate = self._fit_transform_inside_room(moving, candidate)
                actual_shift = float(
                    candidate.translation()[axis] - old_translation[axis]
                )
                if abs(actual_shift) < separation * 0.5:
                    continue
                if (
                    max_translation is not None
                    and float(
                        np.linalg.norm(
                            np.asarray(candidate.translation(), dtype=float)[:2]
                            - old_translation[:2]
                        )
                    )
                    > max_translation
                ):
                    continue
                after_pairs = self._furniture_aabb_overlap_pairs(
                    overrides={str(moving.object_id): candidate}
                )
                target_pair = frozenset((str(moving.object_id), str(other.object_id)))
                if target_pair in after_pairs or after_pairs - before_pairs:
                    continue
                candidates.append(candidate)
        candidates.sort(
            key=lambda transform: float(
                np.linalg.norm(
                    np.asarray(transform.translation(), dtype=float)[:2]
                    - old_translation[:2]
                )
            )
        )
        return candidates

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
        """Merge inventory counts with authoritative prompt-contract counts."""
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
            if category:
                semantic_counts[category] = semantic_counts.get(category, 0) + 1
        contract_counts: dict[str, int] = {}
        for contract_category, count in intent_contract_required_counts(
            self.scene, stage="furniture"
        ).items():
            category = self._repair_category_for_task_label(contract_category)
            if category and count > 0:
                contract_counts[category] = count
        if semantic_counts:
            counts.update(semantic_counts)
        if contract_counts:
            # A contract subtype replaces an overlapping generic inventory
            # category (for example ``office_chair`` replaces ``chair``).
            # Remove only non-contract entries: two explicit contract roles
            # in the same family remain independently authoritative.
            for contract_category in contract_counts:
                for category in list(counts):
                    if category in contract_counts or category == contract_category:
                        continue
                    if furniture_category_satisfies(
                        contract_category, category
                    ) or furniture_category_satisfies(category, contract_category):
                        counts.pop(category, None)
            counts.update(contract_counts)
        authoritative_categories = set(semantic_counts) | set(contract_counts)
        for category in authoritative_categories:
            for shadowed in FURNITURE_CATEGORY_COMPONENT_SHADOWS.get(category, ()):
                if shadowed not in authoritative_categories:
                    counts.pop(shadowed, None)
        for generic in ("desk", "chair"):
            if generic in contract_counts:
                continue
            specialized = sum(
                count
                for category, count in counts.items()
                if category != generic and category.endswith(f"_{generic}")
            )
            if specialized and counts.get(generic, 0) <= specialized:
                counts.pop(generic, None)
        return counts

    def _synchronize_task_required_counts(self) -> None:
        """Make tool guards and deterministic repair share TaskCompiler counts."""
        controller = getattr(self, "furniture_safety_controller", None)
        if controller is None or not getattr(controller, "enabled", False):
            return
        counts = self._repair_required_counts()
        if not counts:
            return
        controller.required_counts = counts
        # ``counts`` already merges the immutable contract with TaskCompiler
        # inventory. Replacing terms prevents an obsolete generic label (such
        # as ``table`` after a ``dressing_table`` contract) from remaining a
        # required-object guard beside its specialized replacement.
        controller.required_terms = set(counts)
        try:
            self.stage_working_memory.set_required_counts(counts)
        except Exception as exc:
            console_logger.warning(
                "Failed to synchronize TaskCompiler furniture requirements: %s",
                exc,
            )

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

    def _repair_request_semantic_name(self, category: str) -> str:
        """Use an unambiguous hard-contract label for repair retrieval."""
        compatible_contract_categories: list[str] = []
        for contract_category in intent_contract_required_counts(
            self.scene, stage="furniture"
        ):
            canonical = self._repair_category_for_task_label(contract_category)
            if not canonical or canonical in compatible_contract_categories:
                continue
            if (
                canonical == category
                or furniture_category_satisfies(category, canonical)
                or furniture_category_satisfies(canonical, category)
            ):
                compatible_contract_categories.append(canonical)
        if category in compatible_contract_categories:
            return category
        if len(compatible_contract_categories) == 1:
            return compatible_contract_categories[0]
        return category

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
        semantic_name = self._repair_request_semantic_name(category)

        request = AssetGenerationRequest(
            object_descriptions=[description],
            short_names=[semantic_name],
            object_type=ObjectType.FURNITURE,
            desired_dimensions=[dimensions],
            style_context="deterministic repair asset",
            scene_id=(
                self.scene.scene_dir.name if self.scene else "deterministic_repair"
            ),
            semantic_name_candidates=[[semantic_name]],
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
                    "semantic_name": category,
                    "semantic_name_source": "deterministic_repair",
                    "category_norm": category,
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
        # Preserve an in-room fallback. A wall anchor is preferred only when it
        # clears every opening; otherwise a soft window warning must not turn
        # an already feasible interior layout into a wall collision.
        fallback = self._grounded_transform(
            bed, x=float(current[0]), y=float(current[1]), yaw_deg=yaw
        )
        fallback = self._fit_transform_inside_room(bed, fallback)
        anchored = self._grounded_transform(
            bed, x=float(current[0]), y=float(current[1]), yaw_deg=yaw
        )
        anchored = self._snap_transform_to_wall(bed, anchored, wall)
        anchored = self._fit_transform_inside_room(bed, anchored)
        transform = self._opening_safe_bed_transform(
            bed=bed,
            transform=anchored,
            wall=wall,
            fallback=fallback,
        )
        if self._transform_close(bed.transform, transform):
            return False
        self.scene.move_object(bed.object_id, transform)
        return True

    def _opening_safe_bed_transform(
        self,
        *,
        bed: SceneObject,
        transform: RigidTransform,
        wall: str,
        fallback: RigidTransform | None = None,
    ) -> RigidTransform:
        """Preserve a wall anchor when possible, otherwise use a safe interior pose.

        A bed can clear a window on its head wall by shifting sideways, yet still
        block a door or window on the perpendicular wall in a compact room.  In
        that case there is no usable wall anchor, so prefer the nearest interior
        placement that leaves every opening clearance zone unobstructed.
        """
        fallback = fallback if fallback is not None else transform
        wall_candidate = self._best_window_safe_bed_anchor_transform(
            bed=bed,
            transform=transform,
            wall=wall,
        )
        if self._bed_transform_clears_openings(bed, wall_candidate):
            return wall_candidate
        interior_candidate = self._best_opening_safe_interior_bed_transform(
            bed=bed,
            transform=wall_candidate,
        )
        # No complete escape exists when other required furniture occupies the
        # only interior slot. Retain the in-room fallback rather than swapping
        # one window warning for a perpendicular-wall collision.
        return interior_candidate if interior_candidate is not None else fallback

    def _bed_transform_clears_openings(
        self, bed: SceneObject, transform: RigidTransform
    ) -> bool:
        zones = self._opening_forbidden_zones(include_windows=True)
        if not zones:
            return True
        return self._zone_overlap_penalty_for_transform(bed, transform, zones) <= 1e-6

    def _best_opening_safe_interior_bed_transform(
        self,
        *,
        bed: SceneObject,
        transform: RigidTransform,
    ) -> RigidTransform | None:
        """Find the closest interior bed pose when no wall anchor clears openings."""
        room_bounds = self._room_bounds_xy()
        bounds = self._bounds_for_transform(bed, transform)
        if room_bounds is None or bounds is None:
            return None

        zones = self._opening_forbidden_zones(include_windows=True)
        if not zones:
            return None

        lower, upper = bounds
        half_span = (upper - lower) / 2.0
        min_x, min_y, max_x, max_y = room_bounds
        margin = max(0.03, float(self._repair_cfg_value("wall_margin_m", 0.08)))
        x_min = min_x + float(half_span[0]) + margin
        x_max = max_x - float(half_span[0]) - margin
        y_min = min_y + float(half_span[1]) + margin
        y_max = max_y - float(half_span[1]) - margin
        if x_min > x_max or y_min > y_max:
            return None

        base_translation = np.asarray(transform.translation(), dtype=float)
        x_values = {
            min(max(float(base_translation[0]), x_min), x_max),
            0.0,
            x_min,
            x_max,
        }
        y_values = {
            min(max(float(base_translation[1]), y_min), y_max),
            0.0,
            y_min,
            y_max,
        }
        for _, _, zone_min, zone_max in zones:
            x_values.update(
                (
                    float(zone_min[0]) - float(half_span[0]) - margin,
                    float(zone_max[0]) + float(half_span[0]) + margin,
                )
            )
            y_values.update(
                (
                    float(zone_min[1]) - float(half_span[1]) - margin,
                    float(zone_max[1]) + float(half_span[1]) + margin,
                )
            )

        best: tuple[float, RigidTransform] | None = None
        for x in x_values:
            if x < x_min - 1e-6 or x > x_max + 1e-6:
                continue
            for y in y_values:
                if y < y_min - 1e-6 or y > y_max + 1e-6:
                    continue
                translation = base_translation.copy()
                translation[0] = float(x)
                translation[1] = float(y)
                candidate = RigidTransform(R=transform.rotation(), p=translation)
                candidate = self._fit_transform_inside_room(
                    bed, candidate, margin_m=margin
                )
                if not self._bed_transform_clears_openings(bed, candidate):
                    continue
                if self._bed_candidate_hits_unrelated_furniture(bed, candidate):
                    continue
                displacement = float(
                    np.linalg.norm(
                        np.asarray(candidate.translation()[:2]) - base_translation[:2]
                    )
                )
                if best is None or displacement < best[0]:
                    best = (displacement, candidate)
        return best[1] if best is not None else None

    def _bed_candidate_hits_unrelated_furniture(
        self, bed: SceneObject, transform: RigidTransform
    ) -> bool:
        if self.scene is None:
            return False
        candidate_bounds = self._bounds_for_transform(bed, transform)
        if candidate_bounds is None:
            return True
        for object_id, obj in self.scene.objects.items():
            if object_id == bed.object_id or getattr(obj, "immutable", False):
                continue
            if getattr(obj, "object_type", None) != ObjectType.FURNITURE:
                continue
            # Bedside anchors are recomputed immediately after the bed moves.
            # Other furniture must not be displaced implicitly by this fallback.
            if self._category_for_object(object_id, obj) in {"bed", "nightstand"}:
                continue
            obstacle_bounds = obj.compute_world_bounds()
            if obstacle_bounds is None:
                continue
            overlap_x, overlap_y = self._xy_overlap_depths(
                candidate_bounds, obstacle_bounds
            )
            if overlap_x > 1e-4 and overlap_y > 1e-4:
                return True
        return False

    def _best_window_safe_bed_anchor_transform(
        self,
        *,
        bed: SceneObject,
        transform: RigidTransform,
        wall: str,
    ) -> RigidTransform:
        """Shift a wall-anchored bed sideways when its headboard blocks an opening."""
        bounds = self._bounds_for_transform(bed, transform)
        room_bounds = self._room_bounds_xy()
        if bounds is None or room_bounds is None:
            return transform
        if not self._bed_anchor_overlaps_opening(bounds, wall):
            return transform

        tangent_axis = 0 if wall in {"north", "south"} else 1
        lower, upper = bounds
        current_center = float((lower[tangent_axis] + upper[tangent_axis]) / 2.0)
        half_span = float((upper[tangent_axis] - lower[tangent_axis]) / 2.0)
        room_min = float(room_bounds[tangent_axis]) + 0.03 + half_span
        room_max = float(room_bounds[tangent_axis + 2]) - 0.03 - half_span
        if room_min > room_max:
            return transform

        candidates = [min(max(current_center, room_min), room_max), 0.0]
        for opening in list(getattr(self.scene.room_geometry, "openings", []) or []):
            opening_wall = str(
                getattr(
                    getattr(opening, "wall_direction", ""),
                    "value",
                    getattr(opening, "wall_direction", ""),
                )
            ).lower()
            opening_type = str(
                getattr(
                    getattr(opening, "opening_type", ""),
                    "value",
                    getattr(opening, "opening_type", ""),
                )
            ).lower()
            if opening_wall != wall or opening_type not in {"window", "door", "open"}:
                continue
            center = getattr(opening, "center_world", None)
            width = getattr(opening, "width", None)
            try:
                opening_center = float(center[tangent_axis])
                opening_half_span = float(width) / 2.0
            except (IndexError, TypeError, ValueError):
                continue
            clearance = half_span + opening_half_span + 0.04
            candidates.extend((opening_center - clearance, opening_center + clearance))

        best: tuple[float, RigidTransform] | None = None
        for tangent_center in candidates:
            tangent_center = min(max(float(tangent_center), room_min), room_max)
            translation = np.asarray(transform.translation(), dtype=float).copy()
            translation[tangent_axis] += tangent_center - current_center
            candidate = RigidTransform(R=transform.rotation(), p=translation)
            # Keep the normal wall anchor conservative, but a lateral opening
            # avoidance move may need the same 3 cm boundary margin used by
            # deterministic relation repairs to fit in a narrow valid slot.
            candidate = self._fit_transform_inside_room(
                bed,
                candidate,
                margin_m=_OPENING_SAFE_WALL_MARGIN_M,
            )
            candidate_bounds = self._bounds_for_transform(bed, candidate)
            if candidate_bounds is None or self._bed_anchor_overlaps_opening(
                candidate_bounds, wall
            ):
                continue
            displacement = float(
                np.linalg.norm(
                    np.asarray(candidate.translation()[:2])
                    - np.asarray(transform.translation()[:2])
                )
            )
            if best is None or displacement < best[0]:
                best = (displacement, candidate)
        return best[1] if best is not None else transform

    def _bed_anchor_overlaps_opening(
        self,
        bounds: tuple[np.ndarray, np.ndarray],
        wall: str,
    ) -> bool:
        """Return whether a bed headboard span intersects an opening on its wall."""
        if self.scene is None or self.scene.room_geometry is None:
            return False
        tangent_axis = 0 if wall in {"north", "south"} else 1
        bed_min, bed_max = bounds
        for opening in list(getattr(self.scene.room_geometry, "openings", []) or []):
            opening_wall = str(
                getattr(
                    getattr(opening, "wall_direction", ""),
                    "value",
                    getattr(opening, "wall_direction", ""),
                )
            ).lower()
            opening_type = str(
                getattr(
                    getattr(opening, "opening_type", ""),
                    "value",
                    getattr(opening, "opening_type", ""),
                )
            ).lower()
            if opening_wall != wall or opening_type not in {"window", "door", "open"}:
                continue
            center = getattr(opening, "center_world", None)
            width = getattr(opening, "width", None)
            try:
                opening_min = float(center[tangent_axis]) - float(width) / 2.0
                opening_max = float(center[tangent_axis]) + float(width) / 2.0
            except (IndexError, TypeError, ValueError):
                continue
            if max(float(bed_min[tangent_axis]), opening_min) <= min(
                float(bed_max[tangent_axis]), opening_max
            ):
                return True
        return False

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
        forbidden_zones = self._opening_forbidden_zones(include_windows=False)
        bed_bounds = self._bounds_for_transform(bed, bed.transform)
        overlap_tolerance = float(
            self._repair_cfg_value("nightstand_bed_overlap_tolerance_m", 0.03)
        )
        for side, nightstand in zip((-1.0, 1.0), nightstands):
            ns_dims = self._local_size(nightstand, [0.45, 0.42, 0.55])
            target = (
                bed_center
                + side * lateral * (bed_dims[0] / 2 + ns_dims[0] / 2 + gap)
                + head * max(0.0, bed_dims[1] / 2 - ns_dims[1] / 2 - 0.10)
            )
            candidates: list[tuple[float, float, float, RigidTransform]] = []
            # A door can occupy the nominal head-side slot. Search both axes in
            # the bed-local frame so the nightstand can move inward or toward
            # the head wall while remaining a reachable bedside surface.
            for head_retreat_step in range(9):
                head_retreat = head_retreat_step * 0.08
                for inward_step in range(6):
                    inward = inward_step * 0.06
                    candidate_target = (
                        target + head * head_retreat - side * lateral * inward
                    )
                    raw_candidate = self._grounded_transform(
                        nightstand,
                        x=float(candidate_target[0]),
                        y=float(candidate_target[1]),
                        yaw_deg=yaw,
                    )
                    candidate = raw_candidate
                    candidate = self._fit_transform_inside_room(nightstand, candidate)
                    bounds = self._bounds_for_transform(nightstand, candidate)
                    if bounds is None:
                        continue
                    zone_penalty = self._zone_overlap_penalty(bounds, forbidden_zones)
                    if zone_penalty > 1e-6:
                        # A clearance repair may need to use the last few
                        # centimetres beside a wall; keep a positive margin.
                        candidate = self._fit_transform_inside_room(
                            nightstand, raw_candidate, margin_m=0.03
                        )
                    bounds = self._bounds_for_transform(nightstand, candidate)
                    if bounds is None:
                        continue
                    zone_penalty = self._zone_overlap_penalty(bounds, forbidden_zones)
                    if bed_bounds is not None:
                        overlap_x, overlap_y = self._xy_overlap_depths(
                            bed_bounds, bounds
                        )
                        if (
                            overlap_x > overlap_tolerance
                            and overlap_y > overlap_tolerance
                        ):
                            continue
                    displacement = float(
                        np.linalg.norm(
                            np.asarray(candidate.translation()[:2]) - target[:2]
                        )
                    )
                    candidates.append(
                        (
                            zone_penalty,
                            displacement,
                            head_retreat + inward,
                            candidate,
                        )
                    )
            if not candidates:
                continue
            _, _, _, transform = min(
                candidates, key=lambda item: (item[0], item[1], item[2])
            )
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
        reserved_free_wall_objects = self._free_wall_anchor_objects(
            excluding_object_id=str(wardrobe.object_id)
        )
        reserved_free_walls = {
            wall
            for obj in reserved_free_wall_objects
            if (wall := self._nearest_room_wall(obj)) is not None
        }
        forbidden_zones = self._opening_forbidden_zones(include_windows=False)
        # A wardrobe anchor must not trade a wall/window violation for an
        # overlap with a dresser, desk, or any other existing furniture.
        obstacles = [
            obj
            for object_id, obj in self.scene.objects.items()
            if str(object_id) != str(wardrobe.object_id)
            and getattr(obj, "object_type", None) == ObjectType.FURNITURE
        ]
        beds = self._furniture_by_category("bed")
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
                lower, upper = bounds
                obstacle_lower, obstacle_upper = obstacle_bounds
                overlap_z = max(
                    0.0,
                    float(
                        min(upper[2], obstacle_upper[2])
                        - max(lower[2], obstacle_lower[2])
                    ),
                )
                if overlap_x > 1e-5 and overlap_y > 1e-5 and overlap_z > 1e-5:
                    overlap_penalty += overlap_x * overlap_y * 100.0
            center = np.asarray(transform.translation(), dtype=float)
            bed_center = (
                np.asarray(beds[0].transform.translation(), dtype=float)
                if beds
                else np.zeros(3)
            )
            distance_score = float(np.linalg.norm(center[:2] - bed_center[:2]))
            candidate_wall = self._nearest_room_wall(wardrobe, transform=transform)
            # "Free wall" is a role-qualified, prompt-authored constraint. A
            # large storage anchor should not occupy that same physical wall
            # when another object was explicitly assigned to it. This remains
            # a soft candidate preference: it never forces an invalid wall when
            # every distinct-wall option is blocked by an opening or collision.
            reserved_wall_penalty = (
                6.0
                if candidate_wall is not None and candidate_wall in reserved_free_walls
                else 0.0
            )
            score = (
                distance_score
                - overlap_penalty
                - wall_opening_penalty
                - reserved_wall_penalty
            )
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
        self.scene.move_object(wardrobe.object_id, best_transform)
        return True

    def _free_wall_anchor_objects(
        self, *, excluding_object_id: str
    ) -> list[SceneObject]:
        """Return furniture assigned to an explicitly free wall by the contract."""
        if self.scene is None:
            return []
        contract = getattr(self.scene, "scenebenchmark_intent_contract", {}) or {}
        categories: set[str] = set()
        for constraint in contract.get("constraints") or []:
            if str(constraint.get("relation") or "") != "against_wall":
                continue
            if str(constraint.get("strength") or "hard").lower() != "hard":
                continue
            target = constraint.get("targets") or {}
            if str(target.get("role") or "").strip().lower() != "free":
                continue
            category = str((constraint.get("subjects") or {}).get("category") or "")
            if category:
                categories.add(category)
        anchors: list[SceneObject] = []
        for category in sorted(categories):
            anchors.extend(
                obj
                for obj in self._furniture_by_category(category)
                if str(obj.object_id) != excluding_object_id
            )
        return anchors

    def _nearest_room_wall(
        self,
        obj: SceneObject,
        *,
        transform: RigidTransform | None = None,
    ) -> str | None:
        """Identify the closest physical room wall for an object's AABB."""
        room_bounds = self._room_bounds_xy()
        bounds = (
            self._bounds_for_transform(obj, transform)
            if transform is not None
            else obj.compute_world_bounds()
        )
        if room_bounds is None or bounds is None:
            return None
        min_x, min_y, max_x, max_y = room_bounds
        lower, upper = bounds
        return min(
            (
                ("west", abs(float(lower[0]) - min_x)),
                ("east", abs(max_x - float(upper[0]))),
                ("south", abs(float(lower[1]) - min_y)),
                ("north", abs(max_y - float(upper[1]))),
            ),
            key=lambda item: (item[1], item[0]),
        )[0]

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
        self.scene.move_object(dresser.object_id, transform)
        return True

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
        self.scene.move_object(wardrobe.object_id, best_transform)
        return True

    def _repair_forbidden_zone_conflicts(
        self,
        include_windows: bool = False,
        *,
        opening_ids: set[str] | None = None,
        blocker_ids: set[str] | None = None,
    ) -> bool:
        """Move objects out of door/opening clearance zones using generic anchors."""
        if self.scene is None:
            return False
        transaction = self._begin_hard_state_transaction()
        zones = self._opening_forbidden_zones(include_windows=include_windows)
        if opening_ids is not None:
            zones = [zone for zone in zones if zone[0] in opening_ids]
        if not zones:
            return False
        blockers = self._objects_overlapping_zones(zones)
        if blocker_ids is not None:
            blockers = [item for item in blockers if item[0] in blocker_ids]
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
            minimum_improvement = 1000.0 * AABB_INTERSECTION_EPSILON_M**2
            if new_penalty >= original_penalty - minimum_improvement:
                continue
            self.scene.move_object(obj.object_id, transform)
            console_logger.info(
                "Deterministic forbidden-zone repair moved %s from penalty %.4f to %.4f",
                object_id,
                original_penalty,
                new_penalty,
            )
            changed = True
        if changed and not self._commit_hard_state_transaction(
            transaction, source="forbidden-zone"
        ):
            return False
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
        opening_type_raw = getattr(opening, "opening_type", "")
        opening_type = str(getattr(opening_type_raw, "value", opening_type_raw)).lower()
        if zone_min is not None and zone_max is not None:
            if opening_type == "door":
                swing_bounds = door_swing_clearance_bounds(opening)
                if swing_bounds is not None:
                    zone_min, zone_max = swing_bounds
            return np.asarray(zone_min, dtype=float), np.asarray(zone_max, dtype=float)

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
            if penalty > 0.0:
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
            overlap_x, overlap_y, _ = aabb_overlap_depths(
                list(obj_min), list(obj_max), list(zone_min), list(zone_max)
            )
            if (
                overlap_x > AABB_INTERSECTION_EPSILON_M
                and overlap_y > AABB_INTERSECTION_EPSILON_M
            ):
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

    def _zone_ejection_candidate_transforms(
        self,
        obj: SceneObject,
        zones: list[tuple[str, str, np.ndarray, np.ndarray]],
    ) -> list[RigidTransform]:
        """Return minimum axis translations that eject an object from a zone."""
        bounds = obj.compute_world_bounds()
        if bounds is None:
            return []
        obj_min, obj_max = bounds
        original = np.asarray(obj.transform.translation(), dtype=float)
        margin = max(
            0.005,
            float(self._repair_cfg_value("collision_separation_margin_m", 0.025)),
        )
        candidates: list[RigidTransform] = []
        for _, _, zone_min, zone_max in zones:
            overlap_x, overlap_y, _ = aabb_overlap_depths(
                list(obj_min), list(obj_max), list(zone_min), list(zone_max)
            )
            if (
                overlap_x <= AABB_INTERSECTION_EPSILON_M
                or overlap_y <= AABB_INTERSECTION_EPSILON_M
            ):
                continue
            deltas = (
                (0, float(zone_min[0] - obj_max[0] - margin)),
                (0, float(zone_max[0] - obj_min[0] + margin)),
                (1, float(zone_min[1] - obj_max[1] - margin)),
                (1, float(zone_max[1] - obj_min[1] + margin)),
            )
            for axis, delta in deltas:
                translation = original.copy()
                translation[axis] += delta
                candidates.append(
                    self._fit_transform_inside_room(
                        obj,
                        RigidTransform(R=obj.transform.rotation(), p=translation),
                    )
                )
        return candidates

    def _furniture_overlap_penalty_for_transform(
        self, obj: SceneObject, transform: RigidTransform
    ) -> float:
        """Return total 3D AABB overlap volume with other furniture."""
        bounds = self._bounds_for_transform(obj, transform)
        if bounds is None:
            return float("inf")
        penalty = 0.0
        for other_id, other in self.scene.objects.items():
            if (
                str(other_id) == str(obj.object_id)
                or getattr(other, "immutable", False)
                or getattr(other, "object_type", None) != ObjectType.FURNITURE
            ):
                continue
            other_bounds = other.compute_world_bounds()
            if other_bounds is None:
                continue
            overlap_x, overlap_y = self._xy_overlap_depths(bounds, other_bounds)
            overlap_z = max(
                0.0,
                float(
                    min(bounds[1][2], other_bounds[1][2])
                    - max(bounds[0][2], other_bounds[0][2])
                ),
            )
            penalty += overlap_x * overlap_y * overlap_z
        return penalty

    def _best_forbidden_zone_repair_transform(
        self,
        obj: SceneObject,
        zones: list[tuple[str, str, np.ndarray, np.ndarray]],
    ) -> RigidTransform | None:
        candidates = self._zone_ejection_candidate_transforms(obj, zones)
        candidates.extend(self._generic_wall_candidate_transforms(obj))
        room_bounds = self._room_bounds_xy()
        if room_bounds is not None:
            min_x, min_y, max_x, max_y = room_bounds
            current_yaw = RollPitchYaw(obj.transform.rotation()).yaw_angle()
            grid_step = max(
                0.4,
                float(self._repair_cfg_value("forbidden_zone_grid_step_m", 0.6)),
            )
            x_values = np.arange(min_x + 0.35, max_x - 0.34, grid_step)
            y_values = np.arange(min_y + 0.35, max_y - 0.34, grid_step)
            for x in x_values:
                for y in y_values:
                    candidates.append(
                        self._fit_transform_inside_room(
                            obj,
                            self._grounded_transform(
                                obj,
                                x=float(x),
                                y=float(y),
                                yaw_deg=math.degrees(current_yaw),
                            ),
                        )
                    )
        if not candidates:
            return None
        before_pairs = self._furniture_aabb_overlap_pairs()
        best_transform: RigidTransform | None = None
        best_score: tuple[float, int, float, float] | None = None
        original_center = np.asarray(obj.transform.translation(), dtype=float)
        for transform in candidates:
            bounds = self._bounds_for_transform(obj, transform)
            if bounds is None:
                continue
            after_pairs = self._furniture_aabb_overlap_pairs(
                overrides={str(obj.object_id): transform}
            )
            if after_pairs - before_pairs:
                continue
            zone_penalty = self._zone_overlap_penalty(bounds, zones)
            center = np.asarray(transform.translation(), dtype=float)
            score = (
                zone_penalty,
                len(after_pairs),
                self._furniture_overlap_penalty_for_transform(obj, transform),
                float(np.linalg.norm(center[:2] - original_center[:2])),
            )
            if best_score is None or score < best_score:
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
        geometry = getattr(self.scene, "room_geometry", None)
        if geometry is None:
            return None
        length = float(getattr(geometry, "length", 0.0) or 0.0)
        width = float(getattr(geometry, "width", 0.0) or 0.0)
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
        self,
        obj: SceneObject,
        transform: RigidTransform,
        *,
        margin_m: float | None = None,
    ) -> RigidTransform:
        room_bounds = self._room_bounds_xy()
        bounds = self._bounds_for_transform(obj, transform)
        if room_bounds is None or bounds is None:
            return transform
        min_x, min_y, max_x, max_y = room_bounds
        world_min, world_max = bounds
        margin = (
            max(0.0, float(margin_m))
            if margin_m is not None
            else max(0.03, float(self._repair_cfg_value("wall_margin_m", 0.08)))
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
        }
