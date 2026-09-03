"""
Stateful manipuland agent with planner/designer/critic workflow.

This module implements manipuland placement using persistent agents that work
per-furniture, with fresh contexts for each furniture surface to bound token usage.
"""

import copy
import json
import logging
import math
import re

from pathlib import Path
from typing import Any

from agents import Agent, FunctionTool, Runner, RunResult, custom_span
from agents.exceptions import MaxTurnsExceeded
import numpy as np
from omegaconf import DictConfig, OmegaConf
from pydrake.all import RigidTransform, RollPitchYaw

from scenesmith.agent_utils.asset_scaling_policy import agent_rescale_tools_enabled
from scenesmith.agent_utils.base_stateful_agent import (
    BaseStatefulAgent,
    log_agent_usage,
)
from scenesmith.agent_utils.hssd_retrieval.support_surface_loader import (
    hssd_support_surface_path,
)
from scenesmith.agent_utils.manipuland_placement_order import (
    build_manipuland_placement_order_reference,
)
from scenesmith.agent_utils.physical_feasibility import (
    apply_per_furniture_postprocessing,
)
from scenesmith.agent_utils.physics_validation import compute_scene_collisions
from scenesmith.agent_utils.placement_noise import PlacementNoiseMode
from scenesmith.agent_utils.rendering_manager import RenderingManager
from scenesmith.agent_utils.room import (
    AgentType,
    ObjectType,
    RoomScene,
    SupportSurface,
    UniqueID,
    extract_and_propagate_support_surfaces,
)
from scenesmith.agent_utils.scene_analyzer import FurnitureSelection, SceneAnalyzer
from scenesmith.agent_utils.scoring import (
    ManipulandCritiqueWithScores,
    log_agent_response,
)
from scenesmith.agent_utils.stage_placement_order_config import (
    append_placement_order_reference,
)
from scenesmith.agent_utils.support_surface_extraction import (
    SupportSurfaceExtractionConfig,
)
from scenesmith.agent_utils.workflow_tools import WorkflowTools
from scenesmith.manipuland_agents.base_manipuland_agent import BaseManipulandAgent
from scenesmith.manipuland_agents.cross_stage_inventory import (
    contract_manipuland_support_cohorts,
    existing_floor_covering_ids,
    is_floor_target,
    is_single_explicit_required_category_request,
    is_single_floor_covering_request,
    satisfied_furniture_owned_floor_requirements,
)
from scenesmith.manipuland_agents.tools.manipuland_tools import ManipulandTools
from scenesmith.manipuland_agents.tools.vision_tools import ManipulandVisionTools
from scenesmith.prompts.registry import ManipulandAgentPrompts
from scenesmith.scenebenchmark_critic import room_scene_to_case_pack
from scenesmith.scenebenchmark_critic.manipuland_targets import (
    classify_manipuland_furniture,
    infer_prompt_manipuland_obligations,
)
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.extensions.dining_place_setting import (
    _usable_seat_front,
    evaluate_dining_place_setting_alignment,
)
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.extensions.manipuland_completeness import (
    evaluate_manipuland_completeness,
)
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.constants import (
    SEATING,
)
from scenesmith.scenebenchmark_critic.object_taxonomy import canonical_object_category
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.seat_surface_assignment import (
    assign_work_seats_to_surfaces,
    room_bounds_from_case_pack,
)
from scenesmith.scenebenchmark_critic.intent_contract import (
    is_hard_constraint,
    intent_contract_constraints_for_scene,
    selected_ids,
    selector_match_count,
)
from scenesmith.utils.logging import BaseLogger

console_logger = logging.getLogger(__name__)


_EXPLICIT_FLOOR_PLACEMENT_PATTERN = re.compile(
    r"\b(?:on|onto|at|placed\s+on|resting\s+on)\s+(?:the\s+)?floor\b|"
    r"\bfloor[-\s]+(?:standing|placed|item)\b",
    re.IGNORECASE,
)

# A bounding-box top is only a last resort for a deterministic, prompt-required
# support target. These categories describe furniture whose primary function is
# to provide a horizontal surface; accepting arbitrary furniture here would hide
# bad asset annotations and allow unsafe placements on sofas, chairs, or plants.
_REQUIRED_BBOX_SUPPORT_CATEGORIES = frozenset(
    {
        "dining_table",
        "coffee_table",
        "table",
        "desk",
        "nightstand",
        "sideboard",
        "bookshelf",
        "dresser",
        "tv_stand",
    }
)

_UPHOLSTERED_SEAT_MANIPULAND_TOKENS = frozenset(
    {"cushion", "pillow", "bolster", "blanket", "throw"}
)


def _selector_category(selector: Any) -> str:
    if isinstance(selector, dict):
        value = selector.get("category")
    else:
        value = getattr(selector, "category", "")
    return "_".join(str(value or "").strip().lower().split())


def _normalized_asset_path(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    return str(Path(text).resolve())


def _serialized_translation(value: Any) -> np.ndarray | None:
    if not isinstance(value, dict):
        return None
    translation = value.get("translation")
    if not isinstance(translation, (list, tuple)) or len(translation) < 3:
        return None
    try:
        return np.asarray(translation[:3], dtype=float)
    except (TypeError, ValueError):
        return None


def _is_monitor_category(category: str) -> bool:
    return _selector_category({"category": category}) in {
        "computer_display",
        "computer_monitor",
        "display",
        "monitor",
        "screen",
    }


def _geometry_center_xy(record: dict[str, Any] | None) -> tuple[float, float] | None:
    center = ((record or {}).get("bbox_world") or {}).get("center") or []
    if len(center) < 2:
        return None
    return float(center[0]), float(center[1])


def _yaw_distance_deg(first: float, second: float) -> float:
    return abs((first - second + 180.0) % 360.0 - 180.0)


class StatefulManipulandAgent(BaseStatefulAgent, BaseManipulandAgent):
    """Manipuland placement with planner/designer/critic agents per furniture.

    Workflow:
    1. Initial analysis: Identify which furniture to populate
    2. Per-furniture loop: Create fresh agents for each furniture surface
    3. Per-furniture workflow: Planner coordinates designer/critic
    4. Agent-driven termination: Planner decides when surface is complete
    """

    @property
    def agent_type(self) -> AgentType:
        """Return agent type for collision filtering."""
        return AgentType.MANIPULAND

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
        # Initialize manipuland-specific base class.
        BaseManipulandAgent.__init__(
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

        # Initialize pending images for image injection during critique.
        self.pending_images: list[dict[str, Any]] = []

        # Current furniture selection context (set per-furniture in workflow).
        self.current_furniture_selection: FurnitureSelection | None = None

        # Context image for manipuland designer initialization (per-furniture).
        self.manipuland_context_image_path: Path | None = None
        # Cleared and rebuilt for each furniture item when enabled.
        self._placement_order_reference: str = ""

    def _render_furniture_for_context(self) -> Path:
        """Render furniture with clean angled front view for context image input.

        Uses furniture_selection mode with empty annotate_object_types to get
        a clean render without any labels, bounding boxes, or coordinate overlays.
        For articulated furniture, opens joints to show interior surfaces.
        Includes context furniture (e.g., chairs around a table) for spatial reference.

        Uses adaptive camera elevation based on furniture type:
        - Tables (1 surface): High elevation (60°) - looking down at surface
        - Shelves (multiple surfaces): Low elevation (30°) - see all levels from front

        Camera is positioned to view the furniture's front face (+Y in local frame),
        accounting for the furniture's world rotation.

        Special case for floor: Renders top-down view of entire room with all
        furniture visible, similar to observe_scene. This provides spatial context
        for floor item placement (rugs, floor lamps, etc.).

        Returns:
            Path to directory containing rendered images.
        """
        furniture = self.scene.get_object(self.current_furniture_id)

        # Special case: Floor needs top-down view of entire room with all furniture.
        # This provides spatial context for floor item placement.
        if furniture.object_type == ObjectType.FLOOR:
            # Include all furniture objects for room context.
            all_furniture_ids = [
                obj.object_id
                for obj in self.scene.objects.values()
                if obj.object_type == ObjectType.FURNITURE
            ]
            return self.rendering_manager.render_scene(
                scene=self.scene,
                blender_server=self.blender_server,
                include_objects=[self.current_furniture_id] + all_furniture_ids,
                exclude_room_geometry=False,  # Include floor/walls for context
                rendering_mode="furniture_selection",  # Disables grid/frame
                annotate_object_types=[],  # Disables all labels/bboxes
                render_name=f"context_input_{self.current_furniture_id}",
                # Top-down view for floor context.
                include_vertical_views=True,  # Include top view
                override_side_view_count=0,  # No side views, just top
            )

        # Get context furniture IDs from current selection.
        context_ids = (
            self.current_furniture_selection.context_furniture_ids
            if self.current_furniture_selection
            else []
        )

        # Include current furniture + validated context furniture (same pattern as
        # observe_scene).
        valid_context_ids = [
            ctx_id for ctx_id in context_ids if ctx_id in self.scene.objects
        ]
        include_objects = [self.current_furniture_id] + valid_context_ids

        # Check if furniture is articulated (has doors/drawers).
        is_articulated = furniture.metadata.get("is_articulated", False)

        # Determine elevation based on furniture type (number of support surfaces).
        # Tables with 1 surface benefit from high angle looking down at surface.
        # Shelves with multiple surfaces need low angle to see all levels.
        num_surfaces = (
            len(furniture.support_surfaces) if furniture.support_surfaces else 1
        )
        if num_surfaces == 1:
            elevation = 60.0  # High angle - looking down at table surface
        else:
            elevation = 30.0  # Low angle - see all shelf levels from front

        # Calculate camera azimuth to view the furniture's front face.
        # Furniture "front" is +Y in local frame. We need to find where that
        # points in world frame and position the camera there.
        # For a Z-rotation (yaw) of θ, the camera should be at azimuth = 90° + θ.
        rotation_matrix = furniture.transform.rotation().matrix()
        # Extract yaw (Z rotation) from rotation matrix: atan2(R[1,0], R[0,0]).
        yaw_rad = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        # Camera azimuth: 90° (front at +Y) + furniture yaw rotation.
        front_azimuth = 90.0 + math.degrees(yaw_rad)

        return self.rendering_manager.render_scene(
            scene=self.scene,
            blender_server=self.blender_server,
            include_objects=include_objects,
            exclude_room_geometry=True,  # Furniture only, no floor/walls
            rendering_mode="furniture_selection",  # Disables grid/frame
            annotate_object_types=[],  # Disables all labels/bboxes
            articulated_open=is_articulated,  # Open joints to show interior surfaces
            context_furniture_ids=valid_context_ids,  # For proper visibility in render
            render_name=f"context_input_{self.current_furniture_id}",
            # Render single angled view from furniture's front face.
            include_vertical_views=False,  # No pure top/bottom views
            override_side_view_count=1,  # Single angled view
            side_view_start_azimuth_degrees=front_azimuth,  # Front of furniture
            side_view_elevation_degrees=elevation,  # Adaptive elevation
        )

    def _get_furniture_dimensions(self, furniture) -> str:
        """Compute human-readable furniture dimensions from bbox.

        Args:
            furniture: SceneObject with bbox_min and bbox_max.

        Returns:
            Human-readable dimensions string.
        """
        if furniture.bbox_min is None or furniture.bbox_max is None:
            return "dimensions unknown"

        dims = furniture.bbox_max - furniture.bbox_min
        width, depth, height = dims[0], dims[1], dims[2]
        return f"{width:.2f}m wide × {depth:.2f}m deep × {height:.2f}m tall"

    def _generate_manipuland_context_image(self) -> Path | None:
        """Generate context image for manipuland placement.

        Renders the furniture and uses image editing API to add suggested objects.
        This provides visual guidance for the manipuland designer agent.

        Returns:
            Path to generated context image, or None if generation fails or disabled.
        """
        if not self.cfg.context_image_generation.enabled:
            return None

        render_dir = self._render_furniture_for_context()

        selection = self.current_furniture_selection
        furniture = self.scene.get_object(selection.furniture_id)

        # Select correct image based on furniture type.
        # Floor uses top-down view; other furniture uses angled front view.
        if furniture.object_type == ObjectType.FLOOR:
            render = render_dir / "0_top.png"
        else:
            render = render_dir / "0_side.png"

        try:
            return self.asset_manager.image_generator.generate_manipuland_context_image(
                reference_image_path=render,
                furniture_description=furniture.description,
                furniture_dimensions=self._get_furniture_dimensions(furniture),
                suggested_items=selection.suggested_items,
                prompt_constraints=selection.prompt_constraints,
                style_notes=selection.style_notes,
                output_path=render_dir / "context_edited.png",
            )
        except Exception as e:
            console_logger.warning(f"Context image generation failed: {e}")
            return None

    def _get_context_image_path(self) -> Path | None:
        """Get the AI-generated context image for initial design.

        Returns:
            Path to manipuland context image if available, None otherwise.
        """
        return self.manipuland_context_image_path

    def _create_designer_tools(
        self,
        current_furniture_id: UniqueID,
        support_surfaces: dict[str, SupportSurface],
    ) -> list[FunctionTool]:
        """Create designer tools with captured dependencies.

        Args:
            current_furniture_id: ID of furniture being populated.
            support_surfaces: Dictionary mapping surface_id to SupportSurface.

        Returns:
            List of tools for the designer agent.
        """
        # Get context furniture from current selection.
        context_ids = []
        if self.current_furniture_selection:
            context_ids = self.current_furniture_selection.context_furniture_ids

        vision_tools = ManipulandVisionTools(
            scene=self.scene,
            rendering_manager=self.rendering_manager,
            cfg=self.cfg,
            current_furniture_id=current_furniture_id,
            blender_server=self.blender_server,
            context_furniture_ids=context_ids,
        )
        self.manipuland_tools = ManipulandTools(
            scene=self.scene,
            asset_manager=self.asset_manager,
            cfg=self.cfg,
            current_furniture_id=current_furniture_id,
            support_surfaces=support_surfaces,
            fulfilled_floor_requirements=(
                satisfied_furniture_owned_floor_requirements(
                    self.scene,
                    current_furniture_id,
                    self.current_furniture_selection.suggested_items,
                )
                if self.current_furniture_selection is not None
                else None
            ),
        )
        workflow_tools = WorkflowTools()

        return [
            *vision_tools.tools.values(),
            *self.manipuland_tools.tools.values(),
            *workflow_tools.tools.values(),
        ]

    def _create_designer_agent(
        self, tools: list[FunctionTool], furniture_description: str
    ) -> Agent:
        """Create designer agent with furniture-specific context.

        Args:
            tools: Tools to provide to the designer.
            furniture_description: Description of furniture being populated.

        Returns:
            Configured designer agent.
        """
        designer_config = self.cfg.agents.designer_agent
        designer_prompt_enum = ManipulandAgentPrompts[designer_config.prompt]

        # Get structured assignment context from current furniture selection.
        selection = self.current_furniture_selection
        if not selection:
            raise ValueError("No current furniture selection set")

        return super()._create_designer_agent(
            tools=tools,
            prompt_enum=designer_prompt_enum,
            furniture_description=furniture_description,
            suggested_items=selection.suggested_items,
            prompt_constraints=selection.prompt_constraints,
            style_notes=selection.style_notes,
            has_reference_image=self.manipuland_context_image_path is not None,
            asset_rescaling_enabled=agent_rescale_tools_enabled(self.cfg),
        )

    def _create_critic_agent(
        self, tools: list[FunctionTool], furniture_description: str
    ) -> Agent:
        """Create critic agent with furniture-specific context.

        Args:
            tools: Tools to provide to the critic.
            furniture_description: Description of furniture being populated.

        Returns:
            Configured critic agent with structured output.
        """
        critic_config = self.cfg.agents.critic_agent
        critic_prompt_enum = ManipulandAgentPrompts[critic_config.prompt]

        # Get structured assignment context from current furniture selection.
        selection = self.current_furniture_selection
        if not selection:
            raise ValueError("No current furniture selection set")

        return super()._create_critic_agent(
            tools=tools,
            prompt_enum=critic_prompt_enum,
            output_type=ManipulandCritiqueWithScores,
            furniture_description=furniture_description,
            suggested_items=selection.suggested_items,
            prompt_constraints=selection.prompt_constraints,
            style_notes=selection.style_notes,
            asset_rescaling_enabled=agent_rescale_tools_enabled(self.cfg),
        )

    def _create_planner_agent(
        self, tools: list[FunctionTool], furniture_description: str
    ) -> Agent:
        """Create planner agent with furniture-specific context.

        Args:
            tools: Tools to provide to the planner.
            furniture_description: Description of furniture being populated.

        Returns:
            Configured planner agent.
        """
        planner_config = self.cfg.agents.planner_agent
        planner_prompt_enum = ManipulandAgentPrompts[planner_config.prompt]
        single_threshold = self.cfg.reset_single_category_threshold
        total_threshold = self.cfg.reset_total_sum_threshold

        # Get structured assignment context from current furniture selection.
        selection = self.current_furniture_selection
        if not selection:
            raise ValueError("No current furniture selection set")

        planner = super()._create_planner_agent(
            tools=tools,
            prompt_enum=planner_prompt_enum,
            furniture_description=furniture_description,
            suggested_items=selection.suggested_items,
            prompt_constraints=selection.prompt_constraints,
            style_notes=selection.style_notes,
            max_critique_rounds=self.cfg.max_critique_rounds,
            reset_single_category_threshold=single_threshold,
            reset_total_sum_threshold=total_threshold,
            early_finish_min_score=self.cfg.early_finish_min_score,
        )
        if self._placement_order_reference and isinstance(planner.instructions, str):
            planner.instructions = append_placement_order_reference(
                planner.instructions,
                self._placement_order_reference,
            )
        return planner

    def _create_tools_for_furniture(
        self, furniture_id: UniqueID
    ) -> tuple[list[FunctionTool], list[FunctionTool], list[FunctionTool]]:
        """Create tools for planner, designer, and critic.

        Args:
            furniture_id: ID of current furniture.

        Returns:
            Tuple of (planner_tools, designer_tools, critic_tools).
        """
        # Get all support surfaces for this furniture.
        furniture = self.scene.get_object(furniture_id)
        if not furniture or not furniture.support_surfaces:
            raise ValueError(f"Furniture {furniture_id} has no support surfaces")

        # Build dict mapping surface_id strings to SupportSurface objects.
        support_surfaces = {
            str(surface.surface_id): surface for surface in furniture.support_surfaces
        }

        # Create designer tools using base class helper method.
        # This ensures consistency with furniture agent architecture and includes
        # WorkflowTools for task management.
        designer_tools = self._create_designer_tools(
            current_furniture_id=furniture_id, support_surfaces=support_surfaces
        )

        # Planner gets all designer tools (same access).
        planner_tools = designer_tools

        # Create critic tools using helper method.
        critic_tools = self._create_critic_tools(furniture_id=furniture_id)

        return planner_tools, designer_tools, critic_tools

    def _get_initial_design_prompt_enum(self) -> Any:
        """Get the prompt enum for initial design instruction.

        Returns:
            Manipuland-specific initial design instruction prompt.
        """
        return ManipulandAgentPrompts.DESIGNER_INITIAL_INSTRUCTION

    def _get_initial_design_prompt_kwargs(self) -> dict:
        """Get prompt kwargs for initial design instruction.

        Returns:
            Dict with has_reference_image flag.
        """
        return {
            "has_reference_image": self.manipuland_context_image_path is not None,
        }

    def _build_initial_design_input(self, instruction: str) -> str | list[dict]:
        """Append the current soft reference only when one was generated."""
        instruction = append_placement_order_reference(
            instruction,
            self._placement_order_reference,
        )
        return super()._build_initial_design_input(instruction)

    def _get_design_change_prompt_enum(self) -> Any:
        """Get the prompt enum for design change instruction.

        Returns:
            Manipuland-specific design change instruction prompt.
        """
        return ManipulandAgentPrompts.DESIGNER_CRITIQUE_INSTRUCTION

    def _get_critique_prompt_enum(self) -> Any:
        """Get the prompt enum for critic runner instruction.

        Returns:
            Manipuland-specific critic instruction prompt.
        """
        return ManipulandAgentPrompts.MANIPULAND_CRITIC_RUNNER_INSTRUCTION

    def _set_placement_noise_profile(self, mode: PlacementNoiseMode) -> None:
        """Set placement noise profile for manipuland tools.

        Args:
            mode: Placement noise mode (NATURAL or PERFECT).
        """
        self.manipuland_tools.set_noise_profile(mode)

    def _create_critic_tools(self, furniture_id: UniqueID) -> list[FunctionTool]:
        """Create critic tools with read-only scene access.

        Args:
            furniture_id: ID of furniture being critiqued (for context rendering).

        Returns:
            List of tools for the critic (read-only scene validation tools).
        """
        # Get context furniture from current selection.
        context_ids = []
        if self.current_furniture_selection:
            context_ids = self.current_furniture_selection.context_furniture_ids

        # Create vision tools for critic (read-only operations).
        vision_tools = ManipulandVisionTools(
            scene=self.scene,
            rendering_manager=self.rendering_manager,
            cfg=self.cfg,
            current_furniture_id=furniture_id,
            blender_server=self.blender_server,
            context_furniture_ids=context_ids,
        )
        self._critic_vision_tools = vision_tools
        self._critic_scene_tools = self.manipuland_tools

        # Critic gets read-only tools (observe only).
        # Note: check_physics is NOT included since physics_context is already
        # injected via the critique runner instruction template.
        return [
            vision_tools.tools["observe_scene"],
            self.manipuland_tools.tools["get_current_scene_state"],
        ]

    def _setup_furniture_context(self, furniture_selection: FurnitureSelection) -> None:
        """Set up per-furniture rendering and analysis context.

        Args:
            furniture_selection: Selection data for this furniture including
                suggested items, prompt constraints, and style notes.
        """
        # Clear pending images from previous furniture iteration.
        # This prevents image leakage if session callback somehow doesn't trigger.
        self.pending_images = []
        self._placement_order_reference = ""

        furniture_id = furniture_selection.furniture_id

        # Create per-furniture rendering manager with subdirectory.
        self.rendering_manager = RenderingManager(
            cfg=self.cfg.rendering,
            logger=self.logger,
            subdirectory=f"manipulands_{furniture_id}",
        )

        # Update scene_analyzer to use per-furniture rendering manager.
        self.scene_analyzer = SceneAnalyzer(
            vlm_service=self.vlm_service,
            rendering_manager=self.rendering_manager,
            cfg=self.cfg,
            blender_server=self.blender_server,
        )

        # Store current furniture selection for agent creation.
        self.current_furniture_id = furniture_id
        self.current_furniture_selection = furniture_selection

    def _initialize_checkpoint_state(self) -> None:
        """Reset checkpoint state for new furniture iteration.

        Called at the start of each furniture iteration to clear checkpoint
        state from the previous furniture piece. The attributes themselves
        were initialized in __init__().
        """
        # Reset checkpoint state to None for new furniture iteration.
        self.previous_scene_checkpoint = None
        self.scene_checkpoint = None
        self.previous_checkpoint_scores = None
        self.checkpoint_scores = None
        self.previous_scores = None
        self.previous_checkpoint_render_dir = None
        self.checkpoint_render_dir = None
        self.final_render_dir = None
        self.checkpoint_scene_hash = None
        self._last_scored_scene_hash = None
        # Keep placement_style as-is (it persists across furniture iterations).

    def _final_hard_validation_enabled(self) -> bool:
        """Require manipuland targets to finish without hard physics violations."""
        try:
            configured = OmegaConf.select(
                self.cfg,
                "fail_stage_on_unresolved_hard_constraints",
                default=False,
            )
        except Exception:
            configured = getattr(
                self.cfg,
                "fail_stage_on_unresolved_hard_constraints",
                False,
            )
        return bool(configured)

    def _evaluate_current_hard_state(self, physics_context: str | None = None) -> Any:
        """Add prompt inventory and dining contracts at a target commit boundary."""
        hard_state = super()._evaluate_current_hard_state(physics_context)
        reasons = self._current_target_cardinality_failures()
        reasons.extend(self._current_target_dining_contract_failures())
        if not reasons:
            return hard_state
        hard_state.hard_valid = False
        hard_state.hard_reasons.extend(
            reason for reason in reasons if reason not in hard_state.hard_reasons
        )
        return hard_state

    def _current_target_cardinality_failures(self) -> list[str]:
        """Return hard count/support failures owned by this furniture target."""
        furniture_id = str(getattr(self, "current_furniture_id", "") or "")
        if not furniture_id or getattr(self, "scene", None) is None:
            return []
        case_pack = room_scene_to_case_pack(
            self.scene, stage="manipuland_target_cardinality"
        )
        objects = [
            item
            for item in (case_pack.get("scene_geometry") or {}).get("objects") or []
            if isinstance(item, dict) and item.get("id")
        ]
        failures: list[str] = []
        scene_objects = getattr(self.scene, "objects", {}) or {}
        object_by_id = (
            {str(object_id): obj for object_id, obj in scene_objects.items()}
            if isinstance(scene_objects, dict)
            else {}
        )
        target = object_by_id.get(furniture_id)
        support_surface_ids = {
            str(surface.surface_id)
            for surface in getattr(target, "support_surfaces", []) or []
            if getattr(surface, "surface_id", None)
        }
        support_cohorts = contract_manipuland_support_cohorts(self.scene)
        for cohort in support_cohorts:
            if cohort.target_id != furniture_id:
                continue
            subject_ids = selected_ids(cohort.subject_selector, objects)
            observed_ids = []
            for subject_id in subject_ids:
                subject = object_by_id.get(str(subject_id))
                placement = getattr(subject, "placement_info", None)
                if (
                    placement is not None
                    and str(placement.parent_surface_id) in support_surface_ids
                ):
                    observed_ids.append(str(subject_id))
            if len(observed_ids) < cohort.required_count:
                failures.append(
                    "support_capacity_or_wrong_support: prompt-required "
                    f"{cohort.category} cohort {cohort.constraint_id} on "
                    f"{furniture_id} requires {cohort.required_count} distinct "
                    f"supported instance(s), found {len(observed_ids)}"
                )

        for constraint in intent_contract_constraints_for_scene(self.scene):
            if not isinstance(constraint, dict):
                continue
            subjects = constraint.get("subjects") or {}
            targets = constraint.get("targets") or {}
            relation = str(constraint.get("relation") or "")
            if (
                str(constraint.get("stage") or "") != "manipuland"
                or not is_hard_constraint(constraint)
                or (relation in {"on_top_of", "one_per_support"} and support_cohorts)
                or str(subjects.get("quantifier") or "") != "exactly"
                or not targets
            ):
                continue
            target_ids = selected_ids(targets, objects)
            # Multi-support contracts are committed only after all targets exist;
            # enforcing them during the first target would reject valid partial work.
            if target_ids != [furniture_id]:
                continue
            try:
                expected = int(subjects.get("count") or 0)
            except (TypeError, ValueError):
                continue
            if expected <= 0:
                continue
            observed = selector_match_count(subjects, objects)
            if observed != expected:
                category = _selector_category(subjects) or "object"
                failures.append(
                    "prompt-required exact count for "
                    f"{category} on {furniture_id}: expected {expected}, "
                    f"found {observed}"
                )
        return failures

    def _current_target_dining_contract_failures(self) -> list[str]:
        """Return failed joint dining predicates owned by the current table only."""
        furniture_id = getattr(self, "current_furniture_id", None)
        if not furniture_id or getattr(self, "scene", None) is None:
            return []
        status = self._dining_joint_contract_status(furniture_id)
        if status is None or status["valid"]:
            return []
        return [
            f"dining joint contract for {furniture_id}: {reason}"
            for reason in status["failures"]
        ]

    def _apply_per_furniture_postprocessing(self, furniture_id: UniqueID) -> bool:
        """Settle one target's manipulands before its final scored critique."""
        postprocessing_cfg = self.cfg.per_furniture_postprocessing
        if not postprocessing_cfg.enabled:
            return True

        simulation_cfg = postprocessing_cfg.simulation
        simulation_html_path = None
        if simulation_cfg.save_html:
            simulation_html_path = (
                self.scene.scene_dir
                / "simulation"
                / "per_furniture"
                / f"{furniture_id}_simulation.html"
            )
        self.scene, success = apply_per_furniture_postprocessing(
            full_scene=self.scene,
            furniture_id=furniture_id,
            config=postprocessing_cfg,
            simulation_html_path=simulation_html_path,
            return_success=True,
        )
        self.rendering_manager.clear_cache()
        self._reset_critic_candidate_cache()
        return success

    def _dining_contract_results(
        self, furniture_id: UniqueID
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Return alignment and completeness results for one dining surface."""
        table_id = str(furniture_id)
        case_pack = room_scene_to_case_pack(
            self.scene, stage="dining_place_setting_joint_validation"
        )
        alignment = next(
            (
                result
                for result in evaluate_dining_place_setting_alignment(case_pack)
                if str(result.get("primary_object") or "") == table_id
            ),
            None,
        )
        completeness = next(
            (
                result
                for result in evaluate_manipuland_completeness(case_pack)
                if str(result.get("primary_object") or "") == table_id
            ),
            None,
        )
        return alignment, completeness

    def _dining_support_bindings_valid(
        self,
        furniture_id: UniqueID,
        alignment: dict[str, Any],
    ) -> bool:
        """Check that every aligned setting object remains bound to this table."""
        table = self.scene.get_object(furniture_id)
        if table is None:
            return False
        surface_ids = {surface.surface_id for surface in table.support_surfaces}
        if not surface_ids:
            return False
        related_ids = {
            str(object_id) for object_id in alignment.get("related_objects") or []
        }
        for object_id in related_ids:
            obj = self.scene.get_object(UniqueID(object_id))
            if obj is None or obj.object_type != ObjectType.MANIPULAND:
                continue
            placement = obj.placement_info
            if placement is None or placement.parent_surface_id not in surface_ids:
                return False
        return True

    def _dining_physics_valid(self, furniture_id: UniqueID) -> bool:
        """Run strict collision validation for this furniture's manipulands."""
        return not self._dining_collisions(furniture_id)

    @staticmethod
    def _dining_result_failure_detail(
        predicate: str,
        result: dict[str, Any] | None,
    ) -> str:
        """Format a bounded deterministic failure reason for logs and exceptions."""
        if result is None:
            return f"{predicate}: no result"
        detail = " ".join(str(result.get("reason") or "failed").split())
        if len(detail) > 320:
            detail = f"{detail[:317]}..."
        return f"{predicate}: {detail}"

    def _dining_joint_contract_status(
        self,
        furniture_id: UniqueID,
        *,
        check_physics: bool = True,
    ) -> dict[str, Any] | None:
        """Evaluate one table's semantic, support, and physics commit contract.

        ``None`` means this target has no dining place-setting contract.  A
        predicate is ``None`` only when an earlier predicate makes it unsafe or
        pointless to evaluate; it is never treated as a pass.
        """
        alignment, completeness = self._dining_contract_results(furniture_id)
        if alignment is None and completeness is None:
            return None

        alignment_valid = bool(alignment and alignment.get("label") == "pass")
        completeness_valid = bool(completeness and completeness.get("label") == "pass")
        failures: list[str] = []
        if not alignment_valid:
            failures.append(self._dining_result_failure_detail("alignment", alignment))
        if not completeness_valid:
            failures.append(
                self._dining_result_failure_detail("completeness", completeness)
            )

        support_valid: bool | None = None
        if alignment_valid:
            support_valid = self._dining_support_bindings_valid(furniture_id, alignment)
            if not support_valid:
                failures.append(
                    "support: aligned place-setting objects are not all bound "
                    f"to {furniture_id}"
                )

        physics_valid: bool | None = None
        if check_physics and alignment_valid and completeness_valid and support_valid:
            physics_valid = self._dining_physics_valid(furniture_id)
            if not physics_valid:
                failures.append(
                    "physics: dining-target collision validation did not pass"
                )

        valid = bool(
            alignment_valid
            and completeness_valid
            and support_valid
            and (physics_valid if check_physics else True)
        )
        return {
            "furniture_id": str(furniture_id),
            "alignment": alignment_valid,
            "completeness": completeness_valid,
            "support": support_valid,
            "physics": physics_valid,
            "valid": valid,
            "failures": failures,
            "alignment_result": alignment,
            "completeness_result": completeness,
        }

    def _remove_duplicate_composite_members(self, furniture_id: UniqueID) -> list[str]:
        """Remove separately placed copies already represented by a composite."""
        furniture = self.scene.get_object(furniture_id)
        if furniture is None:
            return []
        surface_ids = {
            str(surface.surface_id) for surface in furniture.support_surfaces
        }
        if not surface_ids:
            return []

        composites = [
            obj
            for obj in self.scene.objects.values()
            if obj.object_type == ObjectType.MANIPULAND
            and getattr(obj, "placement_info", None) is not None
            and str(obj.placement_info.parent_surface_id) in surface_ids
            and str((obj.metadata or {}).get("composite_type") or "")
            in {"filled_container", "stack", "pile"}
        ]
        removed: list[str] = []
        claimed_ids: set[UniqueID] = set()
        for composite in composites:
            metadata = composite.metadata or {}
            members: list[dict[str, Any]] = []
            if metadata.get("composite_type") == "filled_container":
                container = metadata.get("container_asset")
                if isinstance(container, dict):
                    members.append(container)
                members.extend(
                    member
                    for member in metadata.get("fill_assets") or []
                    if isinstance(member, dict)
                )
            else:
                members.extend(
                    member
                    for member in metadata.get("member_assets") or []
                    if isinstance(member, dict)
                )

            for member in members:
                member_sdf = _normalized_asset_path(member.get("sdf_path"))
                member_translation = _serialized_translation(member.get("transform"))
                if member_sdf is None or member_translation is None:
                    continue
                candidates: list[tuple[float, Any]] = []
                for obj in self.scene.objects.values():
                    if (
                        obj.object_id == composite.object_id
                        or obj.object_id in claimed_ids
                        or obj.object_type != ObjectType.MANIPULAND
                        or (obj.metadata or {}).get("composite_type")
                        or getattr(obj, "immutable", False)
                        or _normalized_asset_path(obj.sdf_path) != member_sdf
                    ):
                        continue
                    placement = getattr(obj, "placement_info", None)
                    if (
                        placement is None
                        or str(placement.parent_surface_id) not in surface_ids
                    ):
                        continue
                    distance = float(
                        np.linalg.norm(
                            np.asarray(obj.transform.translation(), dtype=float)
                            - member_translation
                        )
                    )
                    if distance <= 0.03:
                        candidates.append((distance, obj))
                if not candidates:
                    continue
                _, duplicate = min(candidates, key=lambda item: item[0])
                claimed_ids.add(duplicate.object_id)
                if self.scene.remove_object(duplicate.object_id):
                    removed.append(str(duplicate.object_id))

        if removed:
            console_logger.info(
                "Removed %d duplicate composite member instance(s) from %s: %s",
                len(removed),
                furniture_id,
                ", ".join(sorted(removed)),
            )
            self.rendering_manager.clear_cache()
            self._reset_critic_candidate_cache()
        return removed

    def _dining_collisions(self, furniture_id: UniqueID) -> list[Any]:
        """Return hard collisions scoped to one furniture target."""
        physics_cfg = self.cfg.physics_validation
        return compute_scene_collisions(
            scene=self.scene,
            penetration_threshold=float(physics_cfg.object_penetration_threshold_m),
            floor_penetration_tolerance=float(
                physics_cfg.floor_penetration_tolerance_m
            ),
            current_furniture_id=furniture_id,
            manipuland_furniture_tolerance_m=float(
                physics_cfg.manipuland_furniture_tolerance_m
            ),
        )

    @staticmethod
    def _dining_collision_score(collisions: list[Any]) -> tuple[int, float]:
        """Prefer fewer contacts, then less total penetration."""
        return len(collisions), sum(
            max(0.0, float(collision.penetration_depth)) for collision in collisions
        )

    def _resolve_dining_companion_collisions(
        self,
        furniture_id: UniqueID,
        alignment: dict[str, Any],
    ) -> bool:
        """Move companions within their seat lanes using Drake as the oracle."""
        assignments = (alignment.get("diagnostics") or {}).get("assignments") or []
        companion_rows = {
            str(companion_id): row
            for row in assignments
            for companion_id in row.get("companion_ids") or []
        }
        anchor_rows = {
            str(row.get("anchor_id") or ""): row
            for row in assignments
            if row.get("anchor_id")
        }
        if not companion_rows:
            return self._dining_physics_valid(furniture_id)

        case_pack = room_scene_to_case_pack(
            self.scene, stage="dining_collision_resolution"
        )
        geometry_objects = [
            item
            for item in (case_pack.get("scene_geometry") or {}).get("objects") or []
            if isinstance(item, dict) and item.get("id")
        ]
        geometry_by_id = {str(item["id"]): item for item in geometry_objects}
        table_geometry = geometry_by_id.get(str(furniture_id))
        table_center = _geometry_center_xy(table_geometry)
        table = self.scene.get_object(furniture_id)
        if table is None or table_center is None:
            return False
        surface_map = {
            str(surface.surface_id): surface for surface in table.support_surfaces
        }
        seat_laterals: dict[str, tuple[float, float]] = {}
        for row in assignments:
            seat_id = str(row.get("seat_id") or "")
            seat = geometry_by_id.get(seat_id)
            seat_center = _geometry_center_xy(seat)
            if seat is None or seat_center is None:
                continue
            facing = _usable_seat_front(seat, seat_center, table_center)
            seat_laterals[seat_id] = (-facing[1], facing[0])

        collisions = self._dining_collisions(furniture_id)
        initial_score = self._dining_collision_score(collisions)
        if not collisions:
            return True
        transaction_snapshot = copy.deepcopy(self.scene.to_state_dict())
        max_iterations = max(1, min(12, 2 * len(companion_rows)))
        moved_ids: set[str] = set()

        for _ in range(max_iterations):
            current_score = self._dining_collision_score(collisions)
            involved_ids = {
                object_id
                for collision in collisions
                for object_id in (collision.object_a_id, collision.object_b_id)
            }
            accepted = False

            # Move a complete place setting outward before disturbing its
            # internal plate/cutlery/glass alignment. This handles centerpiece
            # collisions that cannot be solved by moving companions alone.
            for anchor_id, row in sorted(anchor_rows.items()):
                group_ids = {
                    anchor_id,
                    *(str(item) for item in row.get("companion_ids") or []),
                }
                if not (group_ids & involved_ids):
                    continue
                anchor_geometry = geometry_by_id.get(anchor_id)
                anchor_center = _geometry_center_xy(anchor_geometry)
                if anchor_center is None:
                    anchor_object = self.scene.get_object(UniqueID(anchor_id))
                    anchor_center = self.manipuland_tools._object_world_xy(
                        anchor_object
                    )
                if anchor_center is None:
                    continue
                radial = np.asarray(anchor_center, dtype=float) - np.asarray(
                    table_center, dtype=float
                )
                norm = float(np.linalg.norm(radial))
                if norm <= 1e-6:
                    continue
                radial /= norm
                for magnitude in (0.015, 0.03, 0.06, 0.10, 0.15):
                    candidate_snapshot = copy.deepcopy(self.scene.to_state_dict())
                    moved = True
                    for group_id in sorted(group_ids):
                        scene_object = self.scene.get_object(UniqueID(group_id))
                        object_xy = self.manipuland_tools._object_world_xy(scene_object)
                        placement = getattr(scene_object, "placement_info", None)
                        if (
                            scene_object is None
                            or object_xy is None
                            or placement is None
                        ):
                            moved = False
                            break
                        selected = (
                            self.manipuland_tools._select_dining_surface_position(
                                surface_map=surface_map,
                                scene_object=scene_object,
                                target_xy=(
                                    object_xy[0] + magnitude * float(radial[0]),
                                    object_xy[1] + magnitude * float(radial[1]),
                                ),
                                preferred_surface_id=str(placement.parent_surface_id),
                            )
                        )
                        if selected is None:
                            moved = False
                            break
                        surface, position = selected
                        if not self.manipuland_tools._move_dining_object(
                            scene_object, surface, position
                        ).get("success"):
                            moved = False
                            break
                    candidate_alignment, candidate_completeness = (
                        self._dining_contract_results(furniture_id)
                    )
                    candidate_collisions = (
                        self._dining_collisions(furniture_id) if moved else []
                    )
                    valid = bool(
                        moved
                        and candidate_alignment is not None
                        and candidate_alignment.get("label") == "pass"
                        and candidate_completeness is not None
                        and candidate_completeness.get("label") == "pass"
                        and self._dining_support_bindings_valid(
                            furniture_id, candidate_alignment
                        )
                        and self._dining_collision_score(candidate_collisions)
                        < current_score
                    )
                    if valid:
                        collisions = candidate_collisions
                        accepted = True
                        console_logger.info(
                            "Dining collision repair translated setting %s "
                            "outward %.1fcm (%s -> %s)",
                            anchor_id,
                            magnitude * 100.0,
                            current_score,
                            self._dining_collision_score(collisions),
                        )
                        break
                    self.scene.restore_from_state_dict(candidate_snapshot)
                if accepted:
                    break
            if accepted:
                if not collisions:
                    break
                continue

            candidate_ids = sorted(
                involved_ids & companion_rows.keys(),
                key=lambda object_id: (object_id in moved_ids, object_id),
            )
            if not candidate_ids:
                break

            penetration_by_id = {
                object_id: max(
                    (
                        float(collision.penetration_depth)
                        for collision in collisions
                        if object_id in (collision.object_a_id, collision.object_b_id)
                    ),
                    default=0.0,
                )
                for object_id in candidate_ids
            }
            for object_id in candidate_ids:
                row = companion_rows[object_id]
                lateral = seat_laterals.get(str(row.get("seat_id") or ""))
                scene_object = self.scene.get_object(UniqueID(object_id))
                anchor = self.scene.get_object(
                    UniqueID(str(row.get("anchor_id") or ""))
                )
                object_xy = self.manipuland_tools._object_world_xy(scene_object)
                anchor_xy = self.manipuland_tools._object_world_xy(anchor)
                if (
                    lateral is None
                    or scene_object is None
                    or object_xy is None
                    or anchor_xy is None
                    or scene_object.placement_info is None
                ):
                    continue
                dimensions = self.manipuland_tools._dining_footprint_dimensions(
                    scene_object
                )
                step = max(
                    0.012,
                    min(
                        0.04,
                        penetration_by_id[object_id] + 0.008,
                        0.5 * float(min(dimensions)),
                    ),
                )
                max_distance = max(0.06, min(0.18, 2.0 * float(max(dimensions))))
                side = (
                    1.0
                    if (object_xy[0] - anchor_xy[0]) * lateral[0]
                    + (object_xy[1] - anchor_xy[1]) * lateral[1]
                    >= 0.0
                    else -1.0
                )
                candidate_snapshot = copy.deepcopy(self.scene.to_state_dict())
                distances: list[float] = []
                distance = step
                while distance < max_distance:
                    distances.append(distance)
                    distance *= 2.0
                distances.append(max_distance)
                for direction in (side, -side):
                    for magnitude in distances:
                        distance = magnitude * direction
                        target_xy = (
                            object_xy[0] + distance * lateral[0],
                            object_xy[1] + distance * lateral[1],
                        )
                        selected = (
                            self.manipuland_tools._select_dining_surface_position(
                                surface_map=surface_map,
                                scene_object=scene_object,
                                target_xy=target_xy,
                                preferred_surface_id=str(
                                    scene_object.placement_info.parent_surface_id
                                ),
                            )
                        )
                        if selected is None:
                            continue
                        surface, position = selected
                        move = self.manipuland_tools._move_dining_object(
                            scene_object, surface, position
                        )
                        if not move.get("success"):
                            self.scene.restore_from_state_dict(candidate_snapshot)
                            scene_object = self.scene.get_object(UniqueID(object_id))
                            continue
                        candidate_alignment, candidate_completeness = (
                            self._dining_contract_results(furniture_id)
                        )
                        if (
                            candidate_alignment is None
                            or candidate_alignment.get("label") != "pass"
                            or candidate_completeness is None
                            or candidate_completeness.get("label") != "pass"
                            or not self._dining_support_bindings_valid(
                                furniture_id, candidate_alignment
                            )
                        ):
                            self.scene.restore_from_state_dict(candidate_snapshot)
                            scene_object = self.scene.get_object(UniqueID(object_id))
                            continue
                        candidate_collisions = self._dining_collisions(furniture_id)
                        if (
                            self._dining_collision_score(candidate_collisions)
                            < current_score
                        ):
                            collisions = candidate_collisions
                            moved_ids.add(object_id)
                            accepted = True
                            console_logger.info(
                                "Dining collision repair moved %s %.1fcm within "
                                "seat %s lateral lane (%s -> %s)",
                                object_id,
                                abs(distance) * 100.0,
                                row.get("seat_id"),
                                current_score,
                                self._dining_collision_score(collisions),
                            )
                            break
                        self.scene.restore_from_state_dict(candidate_snapshot)
                        scene_object = self.scene.get_object(UniqueID(object_id))
                    if accepted:
                        break
                if accepted:
                    break
            if not accepted or not collisions:
                break

        if collisions:
            self.scene.restore_from_state_dict(transaction_snapshot)
            console_logger.warning(
                "Dining companion collision repair could not find a lane-valid "
                "solution for %s (%s -> %s); restored candidate",
                furniture_id,
                initial_score,
                self._dining_collision_score(collisions),
            )
            return False
        return True

    def _repair_dining_alignment_after_physics(self, furniture_id: UniqueID) -> bool:
        """Jointly commit post-physics dining alignment and physical validity."""
        if not self._code_level_auto_repair_enabled():
            return False
        initial_status = self._dining_joint_contract_status(furniture_id)
        if initial_status is None or initial_status["valid"]:
            return False

        settled_snapshot = copy.deepcopy(self.scene.to_state_dict())
        console_logger.info(
            "Physical settling broke dining alignment for %s; attempting one "
            "joint semantic/physics repair",
            furniture_id,
        )
        completion_result = json.loads(
            self.manipuland_tools._complete_dining_place_settings_impl(
                table_id=str(furniture_id)
            )
        )
        if not completion_result.get("success"):
            return False
        raw_result = self.manipuland_tools._align_dining_place_settings_impl(
            table_id=str(furniture_id)
        )
        try:
            repair_result = json.loads(raw_result)
        except (TypeError, json.JSONDecodeError):
            repair_result = {}
        if repair_result.get("restored"):
            self.scene.restore_from_state_dict(settled_snapshot)
            self.rendering_manager.clear_cache()
            self._reset_critic_candidate_cache()
            return False

        candidate_status = self._dining_joint_contract_status(
            furniture_id,
            check_physics=False,
        )
        if (
            candidate_status is None
            or not candidate_status["valid"]
            or not self._resolve_dining_companion_collisions(
                furniture_id, candidate_status["alignment_result"]
            )
        ):
            self.scene.restore_from_state_dict(settled_snapshot)
            self.rendering_manager.clear_cache()
            self._reset_critic_candidate_cache()
            return False

        postprocessing_valid = self._apply_per_furniture_postprocessing(furniture_id)
        repaired_status = (
            self._dining_joint_contract_status(furniture_id)
            if postprocessing_valid
            else None
        )
        if (
            postprocessing_valid
            and repaired_status is not None
            and repaired_status["valid"]
        ):
            console_logger.info(
                "Joint dining alignment/physics repair passed for %s",
                furniture_id,
            )
            return True

        self.scene.restore_from_state_dict(settled_snapshot)
        self.rendering_manager.clear_cache()
        self._reset_critic_candidate_cache()
        console_logger.warning(
            "Joint dining repair rejected for %s; restored settled scene "
            "(alignment=%s, completeness=%s, support=%s, physics=%s)",
            furniture_id,
            repaired_status and repaired_status["alignment"],
            repaired_status and repaired_status["completeness"],
            repaired_status and repaired_status["support"],
            repaired_status and repaired_status["physics"],
        )
        return False

    def _setup_furniture_agents(
        self, furniture_id: UniqueID, furniture_description: str
    ) -> None:
        """Create agents and sessions for this furniture piece.

        Args:
            furniture_id: ID of furniture being populated.
            furniture_description: Human-readable furniture description.
        """
        # Create fresh tools and agents for this furniture.
        # First create designer/critic tools.
        (
            _,  # planner_tools created later after agents exist
            designer_tools,
            critic_tools,
        ) = self._create_tools_for_furniture(furniture_id)

        # Create sessions using base class helper.
        # Sessions are stored as instance variables for planner tool closures.
        self.designer_session, self.critic_session = self._create_sessions(
            session_prefix=f"{furniture_id}_"
        )

        # Create agents using base class helpers with override methods.
        self.designer = self._create_designer_agent(
            tools=designer_tools, furniture_description=furniture_description
        )

        self.critic = self._create_critic_agent(
            tools=critic_tools, furniture_description=furniture_description
        )

        # Now create planner tools (can reference self.designer/critic/sessions).
        planner_tools = self._create_planner_tools()

        # Create planner agent using base class helper with override method.
        self.planner = self._create_planner_agent(
            tools=planner_tools, furniture_description=furniture_description
        )

    async def _run_furniture_workflow(self, furniture_id: UniqueID) -> None:
        """Execute the multi-agent workflow for a furniture piece.

        Args:
            furniture_id: ID of furniture being populated.
        """
        # Get runner instruction for planner to start workflow.
        planner_runner_prompt = (
            ManipulandAgentPrompts.MANIPULAND_PLANNER_RUNNER_INSTRUCTION
        )
        runner_instruction = self.prompt_registry.get_prompt(
            prompt_enum=planner_runner_prompt,
        )

        result: RunResult | None = None
        try:
            result = await self._run_planner_workflow(
                runner_input=runner_instruction,
                max_turns=self.cfg.agents.planner_agent.max_turns,
            )
        except MaxTurnsExceeded:
            # Tool side effects are committed before the runner reports its turn
            # limit. Continue through deterministic repair, physics validation,
            # and final scoring; any unresolved hard failure still propagates and
            # triggers the enclosing per-target transaction rollback.
            console_logger.warning(
                "Manipuland planner exhausted its turn budget for %s; "
                "finalizing the committed candidate deterministically",
                furniture_id,
            )

        if result is not None:
            log_agent_usage(result=result, agent_name="PLANNER (MANIPULAND)")

        if result is not None and result.final_output:
            log_agent_response(
                response=result.final_output, agent_name="PLANNER (MANIPULAND)"
            )

        # The final critic can identify a bad one-to-one dining assignment even
        # when the planner does not execute its repair recommendation. Enforce the
        # same deterministic contract before the final scored critique.
        self._enforce_monitor_work_seat_orientation(furniture_id)
        self._enforce_dining_place_setting_alignment(furniture_id)

        # Projection and simulation must settle the candidate before the final
        # critic observes it. Otherwise pre-repair collisions can be scored and
        # persisted as a completed furniture target.
        self._apply_per_furniture_postprocessing(furniture_id)
        self._repair_dining_alignment_after_physics(furniture_id)

        # Compute final critique and scores for completed furniture.
        # Check if scene changed since last checkpoint to avoid redundant critique.
        current_scene_hash = self.scene.content_hash()

        if self._can_skip_final_critique(current_scene_hash):
            console_logger.info(
                "Scene unchanged since last critique, skipping final critique"
            )
        else:
            console_logger.info(
                "Scene changed since last critique, computing final critique"
            )
            # Pass update_checkpoint=False to preserve N-1 checkpoint for reset check.
            await self._request_critique_impl(update_checkpoint=False)

        # Validate final scene and save scores.
        await self._finalize_scene_and_scores()

        # Final score rollback can restore the N-1 checkpoint after the joint
        # dining transaction above. Re-assert the hard dining contract at the
        # actual commit boundary so a softer VLM score cannot replace a
        # chair-aligned, physically valid candidate with a stale layout.
        self._finalize_dining_joint_contract(furniture_id)

        console_logger.info(
            f"Completed manipuland placement for furniture {furniture_id}"
        )

    def _finalize_dining_joint_contract(self, furniture_id: UniqueID) -> None:
        """Revalidate dining semantics and physics after score-based rollback."""
        initial_status = self._dining_joint_contract_status(furniture_id)
        if initial_status is None:
            return
        if initial_status["valid"]:
            return

        if not self._code_level_auto_repair_enabled():
            console_logger.warning(
                "Dining contract is invalid for %s, but code-level automatic "
                "repair is disabled",
                furniture_id,
            )
            return

        console_logger.info(
            "Final score rollback invalidated the dining joint contract for %s; "
            "reapplying deterministic semantic/physics repair",
            furniture_id,
        )
        self._repair_dining_alignment_after_physics(furniture_id)

        final_status = self._dining_joint_contract_status(furniture_id)
        if final_status is None or not final_status["valid"]:
            failures = (
                "; ".join(final_status["failures"])
                if final_status is not None
                else "contract evaluation disappeared"
            )
            raise RuntimeError(
                "Manipuland finalization left an unresolved dining semantic/physics "
                f"contract for {furniture_id}: {failures}"
            )

        console_logger.info(
            "Final dining joint contract passed after score rollback for %s",
            furniture_id,
        )

    def _enforce_dining_place_setting_alignment(self, furniture_id: UniqueID) -> bool:
        """Repair a failed dining place-setting contract before final scoring."""
        if not self._code_level_auto_repair_enabled():
            return False
        table_id = str(furniture_id)
        removed_duplicates = self._remove_duplicate_composite_members(furniture_id)

        def current_result() -> dict[str, Any] | None:
            case_pack = room_scene_to_case_pack(
                self.scene, stage="dining_place_setting_final_repair"
            )
            return next(
                (
                    result
                    for result in evaluate_dining_place_setting_alignment(case_pack)
                    if str(result.get("primary_object") or "") == table_id
                ),
                None,
            )

        before = current_result()
        if before is None or before.get("label") != "fail":
            return bool(removed_duplicates)

        console_logger.info(
            "Applying deterministic dining place-setting alignment for %s",
            table_id,
        )
        self.manipuland_tools._align_dining_place_settings_impl(table_id=table_id)

        after = current_result()
        if after is not None and after.get("label") == "pass":
            console_logger.info(
                "Deterministic dining place-setting alignment passed for %s",
                table_id,
            )
            return True

        unresolved = (
            "metric produced no result" if after is None else after.get("reason")
        )
        console_logger.warning(
            "Dining place-setting alignment remains unresolved for %s: %s",
            table_id,
            unresolved,
        )
        return False

    def _enforce_monitor_work_seat_orientation(self, furniture_id: UniqueID) -> bool:
        """Point monitors on this work surface towards its assigned work seat."""
        if not self._code_level_auto_repair_enabled():
            return False
        if not self._task_requires_monitor_work_seat_facing():
            return False
        tools = getattr(self, "manipuland_tools", None)
        support_surfaces = getattr(tools, "support_surfaces", {})
        if not tools or not support_surfaces:
            return False

        case_pack = room_scene_to_case_pack(
            self.scene, stage="monitor_work_seat_orientation_repair"
        )
        geometry = case_pack.get("scene_geometry") or {}
        objects = [
            item
            for item in geometry.get("objects") or []
            if isinstance(item, dict) and item.get("id")
        ]
        by_id = {str(item["id"]): item for item in objects}
        original_task = str(
            getattr(self.scene, "scene_expert_original_description", "")
            or case_pack.get("original_task_instruction")
            or case_pack.get("task_instruction")
            or ""
        )
        assignments = assign_work_seats_to_surfaces(
            objects,
            task_instruction=original_task,
            room_type=str(case_pack.get("room_type") or ""),
            room_bounds=room_bounds_from_case_pack(case_pack),
        )
        assigned_seat_id = next(
            (
                assignment.seat_id
                for assignment in assignments
                if assignment.surface_id == str(furniture_id)
            ),
            "",
        )
        chair = by_id.get(assigned_seat_id)
        chair_center = _geometry_center_xy(chair)
        if chair_center is None:
            return False

        moved = False
        for record in objects:
            if not _is_monitor_category(str(record.get("category") or "")):
                continue
            placement = record.get("placement_info") or {}
            surface_id = str(placement.get("parent_surface_id") or "")
            surface = support_surfaces.get(surface_id)
            monitor_id = str(record["id"])
            monitor = self.scene.get_object(UniqueID(monitor_id))
            monitor_center = _geometry_center_xy(record)
            if surface is None or monitor is None or monitor_center is None:
                continue
            direction = (
                chair_center[0] - monitor_center[0],
                chair_center[1] - monitor_center[1],
            )
            if math.hypot(*direction) <= 1e-6:
                continue
            desired_world_yaw = math.degrees(math.atan2(-direction[0], direction[1]))
            current_world_yaw = float(record.get("yaw_deg") or 0.0)
            if _yaw_distance_deg(current_world_yaw, desired_world_yaw) <= 1.0:
                continue
            position = getattr(
                getattr(monitor, "placement_info", None), "position_2d", None
            )
            if position is None or len(position) < 2:
                continue
            surface_yaw = math.degrees(
                RollPitchYaw(surface.transform.rotation()).yaw_angle()
            )
            local_yaw = (desired_world_yaw - surface_yaw) % 360.0
            tools._move_manipuland_impl(
                object_id=monitor_id,
                surface_id=surface_id,
                position_x=float(position[0]),
                position_z=float(position[1]),
                rotation_degrees=local_yaw,
            )
            moved = True
            console_logger.info(
                "Aligned monitor %s toward assigned work seat %s on %s",
                monitor_id,
                assigned_seat_id,
                furniture_id,
            )
        return moved

    def _code_level_auto_repair_enabled(self) -> bool:
        """Return whether non-physical deterministic scene edits are allowed."""
        config = getattr(self.cfg, "code_level_auto_repair", None)
        if config is None:
            return True
        return bool(getattr(config, "enabled", True))

    def _task_requires_monitor_work_seat_facing(self) -> bool:
        for constraint in intent_contract_constraints_for_scene(self.scene):
            relation = (
                str(constraint.get("relation") or "")
                if isinstance(constraint, dict)
                else str(getattr(constraint, "relation", "") or "")
            )
            subjects = (
                constraint.get("subjects") or {}
                if isinstance(constraint, dict)
                else getattr(constraint, "subjects", None)
            )
            targets = (
                constraint.get("targets") or {}
                if isinstance(constraint, dict)
                else getattr(constraint, "targets", None)
            )
            subject_category = _selector_category(subjects)
            target_category = _selector_category(targets)
            if (
                relation == "faces"
                and _is_monitor_category(subject_category)
                and target_category in {"chair", "office_chair"}
            ):
                return True
        return False

    def _get_final_scores_directory(self) -> Path:
        """Get the directory path for saving per-furniture manipuland placement state.

        Returns:
            Path to scene_states/manipuland_furniture_{id} directory.
        """
        return (
            self.logger.output_dir
            / "scene_states"
            / f"manipuland_furniture_{self.current_furniture_id}"
        )

    @staticmethod
    def _required_target_bbox_top_surfaces(
        *,
        scene: RoomScene,
        furniture: Any,
        furniture_id: UniqueID,
        selection: FurnitureSelection,
        config: SupportSurfaceExtractionConfig,
    ) -> list[SupportSurface]:
        """Build one conservative top surface for a hard support obligation.

        HSSD can contain an explicit but empty support-surface annotation. The
        normal mesh fallback is appropriate for missing metadata, but can be
        prohibitively expensive for a high-poly asset whose semantic role is
        unambiguous. This fallback is deliberately narrower: it requires an
        explicit prompt obligation, a known support-furniture category, and a
        finite usable inset inside the object bounds. Placement, physics, and
        final contract validation remain responsible for rejecting an unsuitable
        asset or arrangement.
        """
        if not selection.is_prompt_required:
            return []
        category = classify_manipuland_furniture(furniture, furniture_id)
        if category not in _REQUIRED_BBOX_SUPPORT_CATEGORIES:
            return []

        bbox_min = getattr(furniture, "bbox_min", None)
        bbox_max = getattr(furniture, "bbox_max", None)
        transform = getattr(furniture, "transform", None)
        if bbox_min is None or bbox_max is None or transform is None:
            return []
        lower = np.asarray(bbox_min, dtype=float)
        upper = np.asarray(bbox_max, dtype=float)
        if (
            lower.shape != (3,)
            or upper.shape != (3,)
            or not np.all(np.isfinite(np.concatenate((lower, upper))))
        ):
            return []

        dimensions = upper - lower
        if np.any(dimensions <= 0.0):
            return []
        # Keep placements away from edge geometry while retaining most of a
        # valid tabletop. The cap avoids eliminating compact nightstands.
        inset_m = min(0.08, 0.05 * float(np.min(dimensions[:2])))
        half_extents = dimensions[:2] / 2.0 - inset_m
        if (
            np.any(half_extents <= 0.0)
            or 4.0 * float(np.prod(half_extents)) < config.min_surface_area_m2
            or float(np.min(half_extents)) < config.min_inscribed_radius_m
        ):
            return []

        offset_m = config.surface_offset_m
        local_center = np.array(
            [
                (lower[0] + upper[0]) / 2.0,
                (lower[1] + upper[1]) / 2.0,
                upper[2] + offset_m,
            ]
        )
        clearance_m = max(config.min_clearance_m, config.top_surface_clearance_m)
        return [
            SupportSurface(
                surface_id=scene.generate_surface_id(),
                bounding_box_min=np.array(
                    [-half_extents[0], -half_extents[1], offset_m]
                ),
                bounding_box_max=np.array(
                    [half_extents[0], half_extents[1], offset_m + clearance_m]
                ),
                transform=transform @ RigidTransform(p=local_center),
                mesh=None,
            )
        ]

    @staticmethod
    def _required_target_support_surface_policy(
        *,
        scene: RoomScene,
        furniture: Any,
        furniture_id: UniqueID,
        selection: FurnitureSelection,
        config: SupportSurfaceExtractionConfig,
    ) -> str:
        """Select the relaxed HSSD seat policy only for a hard soft-furnishing cohort."""
        if not selection.is_prompt_required:
            return "general"
        if config.recompute_hssd_surfaces:
            return "general"
        metadata = getattr(furniture, "metadata", {}) or {}
        mesh_id = str(metadata.get("hssd_mesh_id") or "")
        if metadata.get("asset_source") != "hssd" or not mesh_id:
            return "general"
        if not hssd_support_surface_path(
            mesh_id=mesh_id, data_dir=config.hssd_data_dir
        ).exists():
            return "general"

        category_candidates = (
            metadata.get("semantic_name"),
            metadata.get("category"),
            getattr(furniture, "name", ""),
            getattr(furniture, "description", ""),
            str(furniture_id),
        )
        target_category = next(
            (
                category
                for value in category_candidates
                if (category := canonical_object_category(value)) in SEATING
            ),
            "",
        )
        if not target_category:
            return "general"

        for cohort in contract_manipuland_support_cohorts(scene):
            subject_tokens = set(canonical_object_category(cohort.category).split("_"))
            if (
                cohort.target_id == str(furniture_id)
                and cohort.relation == "on_top_of"
                and subject_tokens & _UPHOLSTERED_SEAT_MANIPULAND_TOKENS
            ):
                return "upholstered_seat"
        return "general"

    async def add_manipulands(self, scene: RoomScene) -> None:
        """Add manipulands to furniture surfaces in the scene.

        This method implements a two-phase workflow:
        1. VLM-based furniture analysis to identify which pieces need manipulands
        2. Per-furniture multi-agent workflow (planner/designer/critic) to
           populate selected furniture with appropriate small objects

        The scene is mutated in place to add manipuland objects. Fresh agent
        contexts are created for each furniture piece to bound token usage.

        Side effects:
        - Scene objects are added (manipulands placed on furniture)
        - Support surfaces are extracted and assigned to furniture
        - Render cache is cleared before processing
        - Per-furniture subdirectories created under logger output directory
        - Checkpoint state saved after each critique iteration
        - Final scores copied to furniture_<id>/final_scene/ directories

        Requirements:
        - Furniture must have geometry_path (non-None)
        - Furniture must have valid bounding boxes (bbox_min, bbox_max)
        - Scene must have text_description for agent context

        Args:
            scene: RoomScene with furniture already placed. Furniture objects must
                have geometry and bounding boxes to be considered for manipuland
                placement.

        Raises:
            Exception: If support surface extraction fails (indicates invalid
                furniture geometry). Agent execution errors are logged but do
                not halt processing of remaining furniture.
        """
        console_logger.info("Starting manipuland placement")
        self.scene = scene

        # Clear render cache to ensure fresh renders for manipulands.
        # This prevents cache key collisions when object IDs are reused.
        self.rendering_manager.clear_cache()

        # Phase 1: Initial analysis - identify which furniture to populate.
        furniture_data = await self._analyze_furniture_for_placement(scene)
        furniture_data = self._recover_contract_required_manipuland_targets(
            scene=scene,
            furniture_data=furniture_data,
        )
        furniture_data = self._recover_prompt_required_manipuland_targets(
            scene=scene,
            furniture_data=furniture_data,
        )
        furniture_data = self._route_explicit_floor_selections(
            scene=scene,
            furniture_data=furniture_data,
        )
        furniture_data = self._skip_realized_floor_covering_targets(
            scene=scene,
            furniture_data=furniture_data,
        )
        furniture_data = self._skip_satisfied_furniture_owned_floor_targets(
            scene=scene,
            furniture_data=furniture_data,
        )

        if not furniture_data:
            console_logger.info("No furniture identified for manipuland placement")
            return

        console_logger.info(
            f"Identified {len(furniture_data)} furniture pieces to populate"
        )
        target_furniture_ids = self._get_target_furniture_ids()
        if target_furniture_ids:
            original_count = len(furniture_data)
            furniture_data = [
                selection
                for selection in furniture_data
                if str(selection.furniture_id) in target_furniture_ids
            ]
            console_logger.info(
                "Filtered manipuland targets from %d to %d using explicit IDs: %s",
                original_count,
                len(furniture_data),
                sorted(target_furniture_ids),
            )
            if not furniture_data:
                console_logger.warning(
                    "None of the requested manipuland target IDs were selected: %s",
                    sorted(target_furniture_ids),
                )
                return
        max_target_furniture = self._get_max_target_furniture()
        if max_target_furniture > 0 and len(furniture_data) > max_target_furniture:
            original_count = len(furniture_data)
            furniture_data = self._select_manipuland_targets(
                scene=scene,
                furniture_data=furniture_data,
                max_target_furniture=max_target_furniture,
            )
            console_logger.info(
                "Limited manipuland targets from %d to %d: %s",
                original_count,
                len(furniture_data),
                [str(selection.furniture_id) for selection in furniture_data],
            )

        # Phase 1b: Select context furniture for each selection.
        if self.cfg.context_furniture.enabled:
            # Get path to furniture_selection images (already rendered).
            furniture_selection_dir = (
                self.rendering_manager._base_output_dir
                / "scene_renders"
                / "furniture_selection"
            )
            images_dir = (
                furniture_selection_dir if furniture_selection_dir.exists() else None
            )

            context_map = self.scene_analyzer.select_context_furniture(
                scene=scene,
                furniture_selections=furniture_data,
                furniture_selection_images_dir=images_dir,
            )

            # Attach context to each selection.
            for selection in furniture_data:
                selection.context_furniture_ids = context_map.get(
                    selection.furniture_id, []
                )

        # Phase 2: Per-furniture loop.
        for furniture_selection in furniture_data:
            furniture_id = furniture_selection.furniture_id
            # Create custom span for this furniture's manipuland placement.
            with custom_span(
                name=f"manipulands_{furniture_id}",
                data={"furniture_id": str(furniture_id)},
            ):
                console_logger.info(f"Populating furniture: {furniture_id}")
                if furniture_selection.suggested_items:
                    console_logger.info(
                        f"Suggested items: {furniture_selection.suggested_items}"
                    )
                    console_logger.info(
                        f"Prompt constraints: {furniture_selection.prompt_constraints}"
                    )
                    console_logger.info(
                        f"Style notes: {furniture_selection.style_notes}"
                    )

                # Extract support surface for this furniture.
                furniture = scene.get_object(furniture_id)
                if not furniture:
                    console_logger.warning(
                        f"Furniture {furniture_id} not found, skipping"
                    )
                    continue

                # Extract all support surfaces using HSM algorithm.
                hsm_config = SupportSurfaceExtractionConfig.from_config(
                    cfg=self.cfg.support_surface_extraction
                )
                surface_policy = self._required_target_support_surface_policy(
                    scene=self.scene,
                    furniture=furniture,
                    furniture_id=furniture_id,
                    selection=furniture_selection,
                    config=hsm_config,
                )
                if surface_policy == "upholstered_seat":
                    console_logger.info(
                        "Using upholstered-seat HSSD support policy for hard "
                        "prompt-owned target %s",
                        furniture_id,
                    )
                try:
                    surfaces = extract_and_propagate_support_surfaces(
                        scene=self.scene,
                        furniture_object=furniture,
                        config=hsm_config,
                        surface_policy=surface_policy,
                    )
                except (FileNotFoundError, ValueError) as error:
                    if not furniture_selection.is_prompt_required:
                        raise
                    console_logger.warning(
                        "Support-surface extraction failed for prompt-required "
                        "%s: %s",
                        furniture_id,
                        error,
                    )
                    surfaces = []

                if not surfaces:
                    surfaces = self._required_target_bbox_top_surfaces(
                        scene=self.scene,
                        furniture=furniture,
                        furniture_id=furniture_id,
                        selection=furniture_selection,
                        config=hsm_config,
                    )
                    if surfaces:
                        furniture.support_surfaces = surfaces
                        furniture.metadata["support_surface_source"] = (
                            "required_bbox_top_fallback"
                        )
                        console_logger.warning(
                            "Using conservative bbox-top support fallback for "
                            "prompt-required furniture %s",
                            furniture_id,
                        )

                console_logger.info(
                    f"Extracted {len(surfaces)} support surface(s) for {furniture_id}"
                )

                # Skip furniture with no support surfaces (e.g., plants, unsuitable geometry).
                if not surfaces:
                    if furniture_selection.is_prompt_required:
                        raise RuntimeError(
                            "Prompt-required manipuland target "
                            f"{furniture_id} has no verified support surface"
                        )
                    console_logger.warning(
                        f"No support surfaces found for {furniture_id}, skipping manipuland placement"
                    )
                    continue

                target_scene_snapshot = copy.deepcopy(self.scene.to_state_dict())
                retry_budget = (
                    self._get_required_target_retry_attempts()
                    if furniture_selection.is_prompt_required
                    else 0
                )
                last_error: Exception | None = None
                retry_guidance = ""
                for target_attempt in range(1, retry_budget + 2):
                    if target_attempt > 1:
                        self.scene.restore_from_state_dict(target_scene_snapshot)
                        console_logger.warning(
                            "Retrying prompt-required manipuland target %s "
                            "(%d/%d) from pre-target snapshot",
                            furniture_id,
                            target_attempt,
                            retry_budget + 1,
                        )
                    attempt_selection = copy.copy(furniture_selection)
                    if retry_guidance:
                        attempt_selection.prompt_constraints = (
                            f"{furniture_selection.prompt_constraints}\n"
                            "RETRY DIAGNOSTIC: The previous candidate was rolled "
                            f"back. {retry_guidance} Choose smaller-footprint assets "
                            "or a different collision-free arrangement while "
                            "preserving every required count and relation."
                        )
                    try:
                        self._setup_furniture_context(attempt_selection)
                        self.manipuland_context_image_path = (
                            self._generate_manipuland_context_image()
                        )
                        self._initialize_checkpoint_state()

                        furniture_obj = scene.get_object(furniture_id)
                        furniture_description = (
                            furniture_obj.description if furniture_obj else "furniture"
                        )
                        scene_prompt = getattr(
                            self.scene,
                            "scene_expert_original_description",
                            self.scene.text_description,
                        )
                        self._placement_order_reference = (
                            build_manipuland_placement_order_reference(
                                cfg=self.cfg,
                                scene_prompt=scene_prompt,
                                scene_dir=self.scene.scene_dir,
                                vlm_service=self.vlm_service,
                                model=self.cfg.openai.model,
                                furniture_id=furniture_id,
                                furniture_description=furniture_description,
                                suggested_items=attempt_selection.suggested_items,
                                prompt_constraints=attempt_selection.prompt_constraints,
                                style_notes=attempt_selection.style_notes,
                                support_surfaces={
                                    str(surface.surface_id): surface
                                    for surface in surfaces
                                },
                            )
                        )
                        self._setup_furniture_agents(
                            furniture_id=furniture_id,
                            furniture_description=furniture_description,
                        )
                        await self._run_furniture_workflow(furniture_id)
                        last_error = None
                        break
                    except Exception as error:
                        last_error = error
                        retry_guidance = self._target_failure_diagnostic(
                            furniture_id, error
                        )
                        self.scene.restore_from_state_dict(target_scene_snapshot)
                        self.rendering_manager.clear_cache()
                        self._reset_critic_candidate_cache()
                        console_logger.error(
                            "Manipuland target attempt failed for %s "
                            "(attempt %d/%d); restored pre-target scene: %s",
                            furniture_id,
                            target_attempt,
                            retry_budget + 1,
                            retry_guidance,
                            exc_info=True,
                        )

                if last_error is not None:
                    if (
                        furniture_selection.is_prompt_required
                        and self._final_hard_validation_enabled()
                    ):
                        raise RuntimeError(
                            "Prompt-required manipuland target "
                            f"{furniture_id} failed after {retry_budget + 1} "
                            f"attempt(s): {retry_guidance}"
                        ) from last_error
                    continue

        console_logger.info("Manipuland placement complete")

    def _get_max_target_furniture(self) -> int:
        try:
            value = OmegaConf.select(self.cfg, "max_target_furniture")
        except Exception:
            value = getattr(self.cfg, "max_target_furniture", 0)
        try:
            return max(0, int(value or 0))
        except Exception:
            return 0

    def _get_required_target_retry_attempts(self) -> int:
        try:
            value = OmegaConf.select(
                self.cfg, "required_target_retry_attempts", default=1
            )
        except Exception:
            value = getattr(self.cfg, "required_target_retry_attempts", 1)
        try:
            return max(0, int(value or 0))
        except Exception:
            return 1

    def _target_failure_diagnostic(
        self, furniture_id: UniqueID, error: Exception
    ) -> str:
        """Return compact deterministic feedback for a clean target retry."""
        details = [str(error).strip() or type(error).__name__]
        try:
            collisions = self._dining_collisions(furniture_id)
        except Exception:
            collisions = []
        if collisions:
            pairs = [
                f"{item.object_a_id}<->{item.object_b_id} "
                f"({float(item.penetration_depth):.3f}m)"
                for item in collisions[:8]
            ]
            details.append("remaining collisions: " + ", ".join(pairs))
        return "; ".join(details)

    def _get_target_furniture_ids(self) -> set[str]:
        """Return an optional exact-ID allowlist for checkpoint replays."""
        try:
            value = OmegaConf.select(self.cfg, "target_furniture_ids", default=[])
        except Exception:
            value = getattr(self.cfg, "target_furniture_ids", [])
        if not value:
            return set()
        if isinstance(value, str):
            value = value.split(",")
        return {str(object_id).strip() for object_id in value if str(object_id).strip()}

    def _skip_realized_floor_covering_targets(
        self,
        *,
        scene: RoomScene,
        furniture_data: list[FurnitureSelection],
    ) -> list[FurnitureSelection]:
        """Skip floor assignments already realized by an earlier scene stage."""
        existing_ids = existing_floor_covering_ids(scene)
        if not existing_ids:
            return furniture_data

        retained: list[FurnitureSelection] = []
        for selection in furniture_data:
            if is_floor_target(
                scene, selection.furniture_id
            ) and is_single_floor_covering_request(selection.suggested_items):
                console_logger.info(
                    "Skipping redundant floor-covering target %s; already realized by %s",
                    selection.furniture_id,
                    existing_ids,
                )
                continue
            retained.append(selection)
        return retained

    def _skip_satisfied_furniture_owned_floor_targets(
        self,
        *,
        scene: RoomScene,
        furniture_data: list[FurnitureSelection],
    ) -> list[FurnitureSelection]:
        """Skip only single floor requirements already fulfilled by furniture."""
        retained: list[FurnitureSelection] = []
        for selection in furniture_data:
            fulfilled = satisfied_furniture_owned_floor_requirements(
                scene,
                selection.furniture_id,
                selection.suggested_items,
            )
            if not fulfilled:
                retained.append(selection)
                continue
            if is_single_explicit_required_category_request(
                selection.suggested_items,
                tuple(fulfilled),
            ):
                console_logger.info(
                    "Skipping fulfilled furniture-owned floor target %s; inventory=%s",
                    selection.furniture_id,
                    fulfilled,
                )
                continue

            inventory_note = "; ".join(
                f"{category} already realized by furniture as {', '.join(object_ids)}"
                for category, object_ids in sorted(fulfilled.items())
            )
            selection.prompt_constraints = "\n".join(
                value
                for value in (
                    selection.prompt_constraints,
                    "Cross-stage inventory (do not regenerate): " + inventory_note,
                )
                if value
            )
            retained.append(selection)
        return retained

    def _route_explicit_floor_selections(
        self,
        *,
        scene: RoomScene,
        furniture_data: list[FurnitureSelection],
    ) -> list[FurnitureSelection]:
        """Move explicit floor-item requests off an accidentally selected surface.

        The VLM selection can correctly describe an item as floor-standing while
        assigning it to a nearby dresser or desk.  A surface-local workflow has
        no way to place that item on the room floor, so retain the original
        placement instructions but target the floor support surface instead.
        """
        floor_id = next(
            (
                object_id
                for object_id, obj in getattr(scene, "objects", {}).items()
                if is_floor_target(scene, object_id)
            ),
            None,
        )
        if floor_id is None:
            return furniture_data

        routed: list[FurnitureSelection] = []
        floor_selection: FurnitureSelection | None = None
        for selection in furniture_data:
            placement_text = " ".join(
                (
                    selection.suggested_items or "",
                    selection.prompt_constraints or "",
                    selection.style_notes or "",
                )
            )
            if is_floor_target(scene, selection.furniture_id) or not (
                _EXPLICIT_FLOOR_PLACEMENT_PATTERN.search(placement_text)
            ):
                routed.append(selection)
                if selection.furniture_id == floor_id:
                    floor_selection = selection
                continue

            if floor_selection is None:
                floor_selection = FurnitureSelection(
                    furniture_id=floor_id,
                    suggested_items=selection.suggested_items,
                    prompt_constraints=selection.prompt_constraints,
                    style_notes=selection.style_notes,
                    context_furniture_ids=[selection.furniture_id],
                )
                routed.append(floor_selection)
            else:
                floor_selection.suggested_items = "\n".join(
                    value
                    for value in (
                        floor_selection.suggested_items,
                        selection.suggested_items,
                    )
                    if value
                )
                floor_selection.prompt_constraints = "\n".join(
                    value
                    for value in (
                        floor_selection.prompt_constraints,
                        selection.prompt_constraints,
                    )
                    if value
                )
                floor_selection.style_notes = "\n".join(
                    value
                    for value in (floor_selection.style_notes, selection.style_notes)
                    if value
                )
                if selection.furniture_id not in floor_selection.context_furniture_ids:
                    floor_selection.context_furniture_ids.append(selection.furniture_id)
            console_logger.info(
                "Rerouted explicit floor-item request from %s to %s",
                selection.furniture_id,
                floor_id,
            )
        return routed

    def _recover_prompt_required_manipuland_targets(
        self,
        *,
        scene: RoomScene,
        furniture_data: list[FurnitureSelection],
    ) -> list[FurnitureSelection]:
        """Add fallback targets for explicit, non-optional prompt obligations."""
        description = str(
            getattr(scene, "scene_expert_original_description", "")
            or getattr(scene, "text_description", "")
            or ""
        )
        obligations = infer_prompt_manipuland_obligations(description)
        if not obligations:
            return furniture_data

        recovered = list(furniture_data)
        for obligation in obligations:
            selected_ids: set[UniqueID] = set()
            for selection in recovered:
                if (
                    classify_manipuland_furniture(
                        scene.get_object(selection.furniture_id),
                        selection.furniture_id,
                    )
                    == obligation.category
                ):
                    selection.is_prompt_required = True
                    selected_ids.add(selection.furniture_id)
            missing = max(0, obligation.target_count - len(selected_ids))
            if missing <= 0:
                continue
            candidates = [
                obj
                for object_id, obj in scene.objects.items()
                if object_id not in selected_ids
                and not getattr(obj, "immutable", False)
                and classify_manipuland_furniture(obj, object_id) == obligation.category
            ]
            candidates.sort(key=lambda obj: str(obj.object_id))
            for obj in candidates[:missing]:
                recovered.append(
                    FurnitureSelection(
                        furniture_id=obj.object_id,
                        suggested_items=f"REQUIRED: {obligation.required_items}",
                        prompt_constraints=(
                            "Deterministic critic recovery: this furniture has "
                            "explicit small-object obligations in the scene prompt."
                        ),
                        style_notes="Follow the requested quantity and distribution exactly.",
                        is_prompt_required=True,
                    )
                )
                console_logger.warning(
                    "Recovered prompt-required manipuland target omitted by VLM: %s (%s)",
                    obj.object_id,
                    obligation.category,
                )
        return recovered

    def _recover_contract_required_manipuland_targets(
        self,
        *,
        scene: RoomScene,
        furniture_data: list[FurnitureSelection],
    ) -> list[FurnitureSelection]:
        """Recover support targets directly from the compiled hard contract."""
        recovered = list(furniture_data)
        by_target = {str(selection.furniture_id): selection for selection in recovered}
        for cohort in contract_manipuland_support_cohorts(scene):
            selection = by_target.get(cohort.target_id)
            target_object = next(
                (
                    obj
                    for object_id, obj in scene.objects.items()
                    if str(object_id) == cohort.target_id
                ),
                None,
            )
            if target_object is None:
                continue
            count_word = (
                "exactly" if cohort.quantifier in {"exactly", "exact"} else "at least"
            )
            category_label = cohort.category.replace("_", " ")
            inventory_line = (
                f"REQUIRED CONTRACT COHORT {cohort.constraint_id}: place {count_word} "
                f"{cohort.required_count} distinct {category_label} instance(s) on "
                f"{cohort.target_id}. This target owns its share of "
                f"{cohort.cohort_total} across {', '.join(cohort.target_ids)}."
            )
            constraint_line = (
                f"Contract relation {cohort.relation}: every counted {category_label} "
                "must be placed on a verified support surface of this exact target. "
                "For repeated identical items, generate or retrieve one reusable asset "
                "template and call place_manipuland_on_surface separately for each "
                "instance. If the verified surfaces cannot fit the required distinct "
                "instances without overlap, stop with a support_capacity failure."
            )
            if selection is None:
                selection = FurnitureSelection(
                    furniture_id=next(
                        object_id
                        for object_id in scene.objects
                        if str(object_id) == cohort.target_id
                    ),
                    suggested_items=inventory_line,
                    prompt_constraints=constraint_line,
                    style_notes="Preserve hard count and support provenance.",
                    context_furniture_ids=[
                        object_id
                        for object_id in scene.objects
                        if str(object_id) in set(cohort.target_ids)
                        and str(object_id) != cohort.target_id
                    ],
                    is_prompt_required=True,
                )
                recovered.append(selection)
                by_target[cohort.target_id] = selection
                console_logger.warning(
                    "Recovered contract-required manipuland target omitted by VLM: %s",
                    cohort.target_id,
                )
                continue
            selection.is_prompt_required = True
            if inventory_line not in (selection.suggested_items or ""):
                selection.suggested_items = "\n".join(
                    value
                    for value in (selection.suggested_items, inventory_line)
                    if value
                )
            if constraint_line not in (selection.prompt_constraints or ""):
                selection.prompt_constraints = "\n".join(
                    value
                    for value in (selection.prompt_constraints, constraint_line)
                    if value
                )
        return recovered

    def _select_manipuland_targets(
        self,
        *,
        scene: RoomScene,
        furniture_data: list[FurnitureSelection],
        max_target_furniture: int,
    ) -> list[FurnitureSelection]:
        def priority(selection: FurnitureSelection) -> tuple[int, int, str]:
            obj = scene.get_object(selection.furniture_id)
            text = " ".join(
                [
                    str(selection.furniture_id),
                    getattr(obj, "name", "") if obj is not None else "",
                    getattr(obj, "description", "") if obj is not None else "",
                    selection.suggested_items or "",
                    selection.prompt_constraints or "",
                ]
            ).lower()
            required_boost = 0 if "required" in text else 1
            if "nightstand" in text or "bedside" in text:
                category_rank = 0
            elif any(term in text for term in ("desk", "table", "dresser", "cabinet")):
                category_rank = 1
            elif any(term in text for term in ("shelf", "bookcase", "bookshelf")):
                category_rank = 2
            elif "bed" in text:
                category_rank = 3
            elif any(term in text for term in ("wardrobe", "closet", "armoire")):
                category_rank = 4
            else:
                category_rank = 5
            return (required_boost, category_rank, str(selection.furniture_id))

        ranked = sorted(furniture_data, key=priority)
        required_count = sum(priority(selection)[0] == 0 for selection in ranked)
        effective_limit = max(max_target_furniture, required_count)
        return ranked[:effective_limit]

    async def _analyze_furniture_for_placement(
        self, scene: RoomScene
    ) -> list[FurnitureSelection]:
        """Analyze which furniture should have manipulands.

        Delegates to SceneAnalyzer for VLM-based furniture selection.

        Args:
            scene: RoomScene with furniture.

        Returns:
            List of FurnitureSelection objects with assignment context.
        """
        return self.scene_analyzer.analyze_furniture_for_manipulands(
            scene=scene,
            prompt_enum=ManipulandAgentPrompts.ANALYZE_FURNITURE_FOR_PLACEMENT,
        )
