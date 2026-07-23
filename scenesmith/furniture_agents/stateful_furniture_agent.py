"""
Stateful furniture agent with natural conversation between persistent agents.

This module implements a furniture placement workflow using persistent
SQLiteSession agents that maintain conversation memory across interactions.
"""

import logging
import math
import time

from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from agents import Agent, FunctionTool, Runner, RunResult
from omegaconf import DictConfig
from pydrake.all import RigidTransform, RollPitchYaw

from scenesmith.agent_utils.asset_manager import AssetGenerationRequest
from scenesmith.agent_utils.base_stateful_agent import (
    BaseStatefulAgent,
    HardStateEvaluation,
    log_agent_usage,
)
from scenesmith.agent_utils.furniture_layout_planning import (
    build_bedroom_anchor_plan,
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
    "twin_bed": (
        "Compact single twin bed with mattress and headboard",
        [1.0, 2.0, 0.75],
    ),
    "nightstand": ("Compact bedside nightstand with drawer", [0.45, 0.42, 0.55]),
    "wardrobe": ("Compact wardrobe closet with simple doors", [0.90, 0.55, 2.00]),
    "dresser": ("Low dresser chest with storage drawers", [1.10, 0.48, 0.85]),
    "desk": ("Practical rectangular work desk", [1.10, 0.60, 0.75]),
    "chair": ("Simple upright task chair", [0.50, 0.50, 0.90]),
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
        result: RunResult = await Runner.run(
            starting_agent=self.planner,
            input=runner_instruction,
            max_turns=self.cfg.agents.planner_agent.max_turns,
            run_config=self._create_run_config(),
        )
        log_agent_usage(result=result, agent_name="PLANNER (FURNITURE)")

        if result.final_output:
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
        """Add deterministic room-aware bedroom guidance to the initial design."""
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
        if (
            FailureCategory.DOOR_OR_OPENING_CLEARANCE in repair_plan.categories
            and self._repair_forbidden_zone_conflicts(include_windows=False)
        ):
            actions.append("cleared deterministic door/opening forbidden zones")

        if not is_bedroom_scene(self.scene):
            return bool(actions), actions

        if self._anchor_existing_bed():
            actions.append("anchored bed to deterministic bedroom head wall")
        if self._repair_bedside_nightstands():
            actions.append("repositioned nightstands to deterministic bedside anchors")
        if "dresser" in reasons and self._repair_dresser_opposite_bed_wall_anchor():
            actions.append("anchored dresser to the wall opposite the bed")
        if (
            "window access warning" in reasons
            or "wardrobe" in reasons
            or "closet" in reasons
            or "collisions" in reasons
            or FailureCategory.WINDOW_OR_WALL_ACCESS in repair_plan.categories
        ) and self._repair_wardrobe_wall_anchor():
            actions.append("moved wardrobe to a deterministic wall/corner anchor")

        return bool(actions), actions

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
            # generate_drake_sdf expects the visual and collision meshes in
            # glTF's Y-up frame. Encode the SceneSmith depth/height axes as
            # glTF Z/Y so the SDF exporter converts them back to X/Y/Z.
            mesh = trimesh.creation.box(extents=[width, height, depth])
            mesh.apply_translation([0.0, height / 2.0, 0.0])
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
        best_transform = None
        best_score = -1e9
        fallback_transform = None
        fallback_score = -1e9
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
            score = distance_score - overlap_penalty - wall_opening_penalty
            if score > fallback_score:
                fallback_score = score
                fallback_transform = transform
            # A hard collision must never win merely because it is farther from
            # the bed. Keep a fallback only for pathological rooms where every
            # candidate overlaps an existing object.
            if overlap_penalty > 1e-5:
                continue
            if score > best_score:
                best_score = score
                best_transform = transform

        if best_transform is None:
            best_transform = fallback_transform

        if best_transform is None or self._transform_close(
            wardrobe.transform, best_transform
        ):
            return False
        self.scene.move_object(wardrobe.object_id, best_transform)
        return True

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
        collision_clearance = float(
            self._repair_cfg_value("wardrobe_wall_clearance_m", 0.35)
        )
        for wall, x, y, yaw in candidates:
            transform = self._grounded_transform(wardrobe, x=x, y=y, yaw_deg=yaw)
            transform = self._snap_transform_to_wall(wardrobe, transform, wall)
            translation = np.asarray(transform.translation(), dtype=float).copy()
            if wall == "north":
                translation[1] -= collision_clearance
            elif wall == "south":
                translation[1] += collision_clearance
            elif wall == "east":
                translation[0] -= collision_clearance
            else:
                translation[0] += collision_clearance
            transform = RigidTransform(R=transform.rotation(), p=translation)
            transform = self._fit_transform_inside_room(wardrobe, transform)
            opening_penalty = 5.0 if wall_openings.get(wall) else 0.0
            transforms.append((transform, opening_penalty))
        return transforms

    def _bedroom_layout_cfg(self) -> Any:
        safety_cfg = getattr(self.cfg, "furniture_safety_controller", None)
        return getattr(safety_cfg, "bedroom_layout", None)

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
