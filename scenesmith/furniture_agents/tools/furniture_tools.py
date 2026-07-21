import logging
import math
import time

from typing import Any

import numpy as np

from agents import function_tool
from omegaconf import DictConfig
from pydrake.all import RigidTransform, RollPitchYaw

from scenesmith.agent_utils.action_logger import log_scene_action
from scenesmith.agent_utils.asset_manager import (
    AssetGenerationRequest,
    AssetGenerationResult as DomainAssetGenerationResult,
    AssetManager,
)
from scenesmith.agent_utils.furniture_functional_layout import (
    furnishable_room_bounds_xy,
)
from scenesmith.agent_utils.furniture_layout_planning import (
    apply_bedroom_asset_size_policy,
)
from scenesmith.agent_utils.loop_detector import LoopDetector
from scenesmith.agent_utils.placement_noise import (
    PlacementNoiseMode,
    apply_placement_noise,
)
from scenesmith.agent_utils.rescale_result import RescaleErrorType, RescaleResult
from scenesmith.agent_utils.rescale_helpers import rescale_object_common
from scenesmith.agent_utils.response_datatypes import (
    AssetGenerationResult,
    GeneratedAsset,
)
from scenesmith.agent_utils.room import (
    ObjectType,
    RoomScene,
    SceneObject,
    UniqueID,
    copy_scene_object_with_new_pose,
)
from scenesmith.utils.geometry_utils import compute_aabb_corners
from scenesmith.furniture_agents.tools.response_dataclasses import (
    AssetInfo,
    AvailableAssetsResult,
    FurnitureErrorType,
    FurnitureOperationResult,
    FurniturePlacementResult,
    Position3D,
    Rotation3D,
)

console_logger = logging.getLogger(__name__)


class FurnitureTools:
    """
    Agent-callable tools for furniture asset generation and placement in 3D scenes.

    Provides a two-phase workflow for the designer agent:
    1. Asset Generation: Creates 3D furniture from text descriptions via the text-to-3D
       pipeline (GPT images → Hunyuan3D geometry → Drake SDF)
    2. Scene Operations: Places, moves, and removes furniture using generated assets

    Tools exposed:
    - generate_assets: Batch generate 3D furniture from descriptions
    - add_furniture_to_scene_tool: Place furniture at specific coordinates
    - move_furniture_tool: Reposition existing furniture
    - remove_furniture_tool: Delete furniture from scene
    """

    def __init__(
        self,
        scene: RoomScene,
        asset_manager: AssetManager,
        cfg: DictConfig,
        safety_controller: Any | None = None,
    ):
        """Initialize furniture tools.

        Args:
            scene: RoomScene instance to manipulate.
            asset_manager: Asset manager for generating 3D assets.
            cfg: Configuration object containing loop detection settings.
        """
        self.scene = scene
        self.asset_manager = asset_manager
        self.cfg = cfg
        self.safety_controller = safety_controller

        # Initialize placement noise configuration.
        # Start with natural profile as default until planner sets it.
        self.placement_noise_config = cfg.placement_noise
        self.active_noise_profile = self.placement_noise_config.natural_profile

        # Initialize loop detector from config.
        loop_config = cfg.loop_detection
        loop_detector = LoopDetector(
            max_attempts=loop_config.max_repeated_attempts,
            window_size=loop_config.tracking_window,
            enabled=loop_config.enabled,
            default_error_factory=self._create_loop_error_response,
        )

        # Apply loop detection to implementation methods.
        self._add_furniture_to_scene_impl = loop_detector(
            self._add_furniture_to_scene_impl
        )
        self._move_furniture_impl = loop_detector(self._move_furniture_impl)
        self._remove_furniture_impl = loop_detector(self._remove_furniture_impl)

        # Create tool closures that use the protected methods.
        self.tools = self._create_tool_closures()

    def set_noise_profile(self, mode: PlacementNoiseMode) -> None:
        """Update the active noise profile based on placement style.

        Args:
            mode: Placement noise mode (NATURAL or PERFECT).
        """
        if mode == PlacementNoiseMode.NATURAL:
            self.active_noise_profile = self.placement_noise_config.natural_profile
            console_logger.info("Placement noise set to NATURAL profile")
        elif mode == PlacementNoiseMode.PERFECT:
            self.active_noise_profile = self.placement_noise_config.perfect_profile
            console_logger.info("Placement noise set to PERFECT profile")
        else:
            console_logger.warning(
                f"Unsupported noise mode {mode}, keeping current profile"
            )

    def _safety_denial_generate_assets(self) -> str | None:
        controller = self.safety_controller
        if controller is None or not getattr(controller, "enabled", False):
            return None
        allowed, message = controller.record_generate_assets()
        if allowed:
            return None
        console_logger.info(message)
        return message

    def _safety_denial_move(self, object_id: str) -> str | None:
        controller = self.safety_controller
        if controller is None or not getattr(controller, "enabled", False):
            return None
        allowed, message = controller.record_move(object_id=object_id)
        if allowed:
            return None
        console_logger.info(message)
        return message

    def _safety_denial_add(self, asset_text: str) -> str | None:
        controller = self.safety_controller
        if controller is None or not getattr(controller, "enabled", False):
            return None
        allowed, message = controller.record_add(
            scene=self.scene,
            asset_text=asset_text,
        )
        if allowed:
            return None
        console_logger.info(message)
        return message

    def _safety_denial_remove(
        self, object_id: str, scene_obj: SceneObject
    ) -> str | None:
        controller = self.safety_controller
        if controller is None or not getattr(controller, "enabled", False):
            return None
        object_text = f"{scene_obj.name} {scene_obj.description}"
        allowed, message = controller.record_remove(
            object_id=object_id,
            object_text=object_text,
            scene=self.scene,
        )
        if allowed:
            return None
        console_logger.info(message)
        return message

    def _safety_denial_rescale(
        self,
        object_id: str,
        scale_factor: float,
    ) -> str | None:
        controller = self.safety_controller
        if controller is None or not getattr(controller, "enabled", False):
            return None

        scene_obj = self.scene.get_object(UniqueID(object_id))
        object_text = ""
        current_dimensions = None
        if scene_obj is not None:
            object_text = f"{scene_obj.name} {scene_obj.description}"
            if scene_obj.bbox_min is not None and scene_obj.bbox_max is not None:
                size = scene_obj.bbox_max - scene_obj.bbox_min
                current_dimensions = (
                    float(size[0]),
                    float(size[1]),
                    float(size[2]),
                )

        allowed, message = controller.record_rescale(
            object_id=object_id,
            scale_factor=scale_factor,
            object_text=object_text,
            current_dimensions=current_dimensions,
        )
        if allowed:
            return None
        console_logger.info(message)
        return message

    def _check_floor_bounds(self, x: float, y: float) -> tuple[bool, str]:
        """Check if position (center point) is within floor plan bounds.

        Args:
            x: X coordinate in meters.
            y: Y coordinate in meters.

        Returns:
            (is_valid, error_message) - error_message is empty string if valid.
        """
        room_geometry = self.scene.room_geometry

        # Floor bounds: [-length/2, length/2] × [-width/2, width/2].
        min_x = -room_geometry.length / 2
        max_x = room_geometry.length / 2
        min_y = -room_geometry.width / 2
        max_y = room_geometry.width / 2

        if not (min_x <= x <= max_x and min_y <= y <= max_y):
            error_msg = (
                f"Position ({x:.3f}, {y:.3f}) is out of floor plan bounds. "
                f"Valid bounds: X=[{min_x:.3f}, {max_x:.3f}], "
                f"Y=[{min_y:.3f}, {max_y:.3f}]"
            )
            return False, error_msg

        return True, ""

    def _get_room_bounds_xy(self) -> tuple[float, float, float, float] | None:
        return furnishable_room_bounds_xy(self.scene)

    def _room_bounds_tolerance(self) -> float:
        safety_cfg = getattr(self.cfg, "furniture_safety_controller", None)
        if safety_cfg is not None:
            try:
                return float(safety_cfg.get("room_bounds_tolerance_m", 0.02))
            except Exception:
                return float(getattr(safety_cfg, "room_bounds_tolerance_m", 0.02))
        return 0.02

    def _floor_penetration_tolerance(self) -> float:
        physics_cfg = getattr(self.cfg, "physics_validation", None)
        if physics_cfg is not None:
            try:
                return float(physics_cfg.get("floor_penetration_tolerance_m", 0.03))
            except Exception:
                return float(
                    getattr(physics_cfg, "floor_penetration_tolerance_m", 0.03)
                )
        return 0.03

    def _world_bounds_for_transform(
        self, scene_obj: SceneObject, transform: RigidTransform
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if scene_obj.bbox_min is None or scene_obj.bbox_max is None:
            return None
        corners = compute_aabb_corners(scene_obj.bbox_min, scene_obj.bbox_max)
        world_corners = np.array([transform @ corner for corner in corners])
        return np.min(world_corners, axis=0), np.max(world_corners, axis=0)

    def _ground_transform_to_floor_if_needed(
        self,
        scene_obj: SceneObject,
        transform: RigidTransform,
    ) -> tuple[RigidTransform, float]:
        """Lift floor-standing furniture so its full bbox rests on the floor.

        Retrieved assets do not always use the floor contact point as their local
        origin. A bed whose local bbox has min_z=-0.97 should still be placeable
        at a floor position; otherwise every valid x/y pose is rejected before
        the planner has a chance to repair the layout.
        """
        world_bounds = self._world_bounds_for_transform(scene_obj, transform)
        if world_bounds is None:
            return transform, 0.0

        world_min, _ = world_bounds
        bottom_z = float(world_min[2])
        floor_tolerance = self._floor_penetration_tolerance()
        if bottom_z >= -floor_tolerance:
            return transform, 0.0

        translation = np.array(transform.translation(), dtype=float)
        lift = -bottom_z
        translation[2] += lift
        grounded_transform = RigidTransform(R=transform.rotation(), p=translation)
        return grounded_transform, lift

    def _check_object_bounds_for_transform(
        self, scene_obj: SceneObject, transform: RigidTransform
    ) -> tuple[bool, str]:
        """Validate the full object AABB, not just its center point."""
        room_bounds = self._get_room_bounds_xy()
        world_bounds = self._world_bounds_for_transform(scene_obj, transform)
        if room_bounds is None or world_bounds is None:
            return True, ""

        min_x, min_y, max_x, max_y = room_bounds
        world_min, world_max = world_bounds
        tolerance = self._room_bounds_tolerance()
        floor_tolerance = self._floor_penetration_tolerance()

        violations = []
        if world_min[0] < min_x - tolerance or world_max[0] > max_x + tolerance:
            violations.append(
                f"x=[{world_min[0]:.3f}, {world_max[0]:.3f}] outside "
                f"[{min_x:.3f}, {max_x:.3f}]"
            )
        if world_min[1] < min_y - tolerance or world_max[1] > max_y + tolerance:
            violations.append(
                f"y=[{world_min[1]:.3f}, {world_max[1]:.3f}] outside "
                f"[{min_y:.3f}, {max_y:.3f}]"
            )
        if world_min[2] < -floor_tolerance:
            violations.append(
                f"bottom z={world_min[2]:.3f} below floor tolerance "
                f"{floor_tolerance:.3f}"
            )

        if violations:
            return (
                False,
                f"Full bounding box for {scene_obj.name} would leave the room: "
                + "; ".join(violations),
            )
        return True, ""

    def _create_loop_error_response(
        self, method_name: str, attempt_count: int, args: tuple, kwargs: dict
    ) -> str:
        """Create furniture-specific error response for loop detection."""
        # Extract object_id from kwargs or args if available.
        object_id = kwargs.get("object_id", "")
        if not object_id and args and len(args) > 1:
            object_id = str(args[1])  # First arg after self

        # Create context-specific diagnostic message.
        if method_name == "_remove_furniture_impl":
            base_name = object_id.rsplit("_", 1)[0] if "_" in object_id else object_id
            diagnostic_message = (
                f"Loop detected: You've tried to remove '{object_id}' {attempt_count} "
                f"times.\n\n"
                f"This means one of:\n"
                f"1. Wrong object name - missing ID postfix (e.g., using '{base_name}' "
                f"instead of '{base_name}_0', '{base_name}_1', etc.)\n"
                f"2. Object was already removed\n"
                f"3. Object doesn't exist with that ID\n\n"
                f"CRITICAL: ALL objects have sequential postfixes (_0, _1, _2, ...).\n"
                f"Base names without postfixes NEVER exist.\n\n"
                f"Recovery procedure (execute in order):\n"
                f"1. Call get_current_scene_state() to see current object IDs with "
                f"postfixes\n"
                f"2. Find objects whose names start with '{base_name}' "
                f"(e.g., '{base_name}_0', '{base_name}_1')\n"
                f"3. Use exact object_id from get_current_scene_state() including postfix\n"
                f"4. If object not in scene, report it was already removed\n\n"
                f"First call get_current_scene_state(), then retry with correct ID."
            )
            suggested_action = (
                "Call get_current_scene_state() to discover object IDs with postfixes"
            )
        elif method_name == "_move_furniture_impl":
            diagnostic_message = (
                f"Loop detected: You've tried to move '{object_id}' {attempt_count} "
                f"times.\n\n"
                f"Causes:\n"
                f"1. Wrong object ID - IDs have sequential postfixes (_0, _1, _2, ...)\n"
                f"2. Object doesn't exist\n"
                f"3. Position/rotation causing collision or validation failure\n\n"
                f"CRITICAL: ALL objects have sequential postfixes. Base names without "
                f"postfixes NEVER exist.\n\n"
                f"Recovery procedure:\n"
                f"1. Call get_current_scene_state() to verify object exists with correct "
                f"ID\n"
                f"2. If collision issue, try different coordinates\n"
                f"3. Check for obstacles blocking the target position"
            )
            suggested_action = "Call get_current_scene_state() to verify object ID"
        else:
            diagnostic_message = (
                f"Loop detected: {attempt_count} identical calls to {method_name}"
            )
            suggested_action = (
                "Call get_current_scene_state() to refresh state, then try different "
                "approach"
            )

        return FurnitureOperationResult(
            success=False,
            message=diagnostic_message,
            object_id=object_id,
            error_type=FurnitureErrorType.LOOP_DETECTED,
            suggested_action=suggested_action,
        ).to_json()

    def _create_tool_closures(self) -> dict[str, Any]:
        """Create closure-based tools that capture self."""

        @function_tool
        def generate_assets(
            object_descriptions: list[str],
            short_names: list[str],
            desired_dimensions: list[list[float]],
            style_context: str | None = None,
        ) -> str:
            """Create 3D furniture models from descriptions with specified dimensions.

            Generate floor-standing furniture items only. This tool is restricted
            to furniture that sits flat on the floor.

            DO NOT generate:
            - Manipulands (small objects meant for surfaces like books, vases, cups)
            - Carpets or rugs
            - Wall decorations

            ONLY generate furniture items that rest directly on the floor.

            You MUST specify dimensions for each object considering the
            relative sizes of other objects in the scene. Use realistic furniture
            proportions.

            Args:
                object_descriptions: List of furniture descriptions to generate
                    (e.g., "Modern oak dining table", "Leather office chair").
                short_names: List of short filesystem-safe names corresponding to
                    each description (e.g., "dining_table", "office_chair").
                desired_dimensions: List of [width, depth, height] in meters for each
                    object. Width (X-axis), depth (Y-axis), and height (Z-axis) specify
                    the object's dimensions in the room coordinate system. Width is
                    left-right, depth is front-back, height is up-down. Predict
                    dimensions considering other objects in the scene.
                    Example: [[1.8, 0.9, 0.75], [0.5, 0.5, 0.9]] for table and chair.
                style_context: Optional style context for visual consistency
                    (e.g., "modern minimalist living room").

            Returns:
                IDs and details of the created furniture models.
            """
            console_logger.info("Tool called: generate_assets")
            console_logger.info(
                f"Generating batch of {len(object_descriptions)} assets: "
                f"{object_descriptions}"
            )
            safety_denial = self._safety_denial_generate_assets()
            if safety_denial:
                return safety_denial

            safety_cfg = getattr(self.cfg, "furniture_safety_controller", None)
            bedroom_cfg = getattr(safety_cfg, "bedroom_layout", None)
            size_policy_result = apply_bedroom_asset_size_policy(
                scene=self.scene,
                object_descriptions=object_descriptions,
                short_names=short_names,
                desired_dimensions=desired_dimensions,
                cfg=bedroom_cfg,
            )
            if size_policy_result.notes:
                console_logger.info(
                    "Bedroom asset size policy applied: %s",
                    "; ".join(size_policy_result.notes),
                )

            request = AssetGenerationRequest(
                object_descriptions=size_policy_result.object_descriptions,
                short_names=size_policy_result.short_names,
                object_type=ObjectType.FURNITURE,
                desired_dimensions=size_policy_result.desired_dimensions,
                style_context=style_context,
                scene_id=self.scene.scene_dir.name,
            )
            return self._generate_assets_impl(request)

        @function_tool
        def add_furniture_to_scene_tool(
            asset_id: str,
            x: float,
            y: float,
            yaw: float = 0.0,
        ) -> str:
            """Place furniture in the room at a specific floor position.

            Furniture sits flat on the floor at z=0 with upright orientation.
            You can only control the x, y position and yaw rotation (rotation
            around the vertical axis).

            Each placement gets a unique ID so you can move or remove it later.
            The same furniture model can be placed multiple times.

            Use 'list_available_assets' to see what furniture you can place.

            Args:
                asset_id: ID of the furniture to place.
                x: X position in the room (meters).
                y: Y position in the room (meters).
                yaw: Yaw rotation in degrees around vertical axis (default: 0.0).
                    Positive values rotate counterclockwise in top-down view.

            Returns:
                The unique ID for this placement and confirmation of success.
            """
            return self._add_furniture_to_scene_impl(
                asset_id=asset_id,
                x=x,
                y=y,
                z=0.0,
                roll=0.0,
                pitch=0.0,
                yaw=yaw,
            )

        @function_tool
        def move_furniture_tool(
            object_id: str,
            x: float,
            y: float,
            yaw: float = 0.0,
        ) -> str:
            """Move existing furniture to a new floor position.

            Furniture sits flat on the floor at z=0 with upright orientation.
            You can only control the x, y position and yaw rotation (rotation
            around the vertical axis).

            Use this to relocate furniture that's already in the room. You need
            the object ID from when you placed it or from 'get_current_scene_state'.

            Args:
                object_id: ID of the furniture item to move.
                x: New X position in the room (meters).
                y: New Y position in the room (meters).
                yaw: New yaw rotation in degrees around vertical axis (default: 0.0).
                    Positive values rotate counterclockwise in top-down view.

            Returns:
                Confirmation that the furniture was moved successfully.
            """
            return self._move_furniture_impl(
                object_id=object_id,
                x=x,
                y=y,
                z=0.0,
                roll=0.0,
                pitch=0.0,
                yaw=yaw,
            )

        @function_tool
        def remove_furniture_tool(object_id: str) -> str:
            """Remove furniture from the room.

            Use this to delete furniture you no longer want. You need the object ID
            from when you placed it or from 'get_current_scene_state'.

            Args:
                object_id: ID of the furniture item to remove.

            Returns:
                Confirmation that the furniture was removed successfully.
            """
            return self._remove_furniture_impl(object_id)

        @function_tool
        def list_available_assets() -> str:
            """See all furniture models you can place with their dimensions.

            This shows you all the furniture that's available for placing in the
            room, including precise dimensions (width, depth, height) to help with
            spatial planning. Use the IDs from this list with 'add_furniture_to_scene_tool'
            to actually place items. You can place the same model multiple times.

            Returns:
                List of furniture with their IDs, names, descriptions, and dimensions.
            """
            return self._list_available_assets_impl()

        @function_tool
        def rescale_furniture_tool(object_id: str, scale_factor: float) -> str:
            """Resize furniture by a uniform scale factor.

            IMPORTANT: This rescales the underlying ASSET. All instances of the same
            asset (e.g., all 4 dining chairs) will be affected. This is usually what
            you want - if one chair is too small, they all are.

            Use this when proportions are correct but size is wrong.
            For shape/proportion issues, regenerate the asset instead.

            Args:
                object_id: ID of the furniture item to rescale.
                scale_factor: Scale multiplier (e.g., 1.5 = 50% larger, 0.8 = 20% smaller).

            Returns:
                Result with new dimensions and list of affected objects.
            """
            return self._rescale_furniture_impl(object_id, scale_factor)

        return {
            "generate_assets": generate_assets,
            "add_furniture_to_scene_tool": add_furniture_to_scene_tool,
            "move_furniture_tool": move_furniture_tool,
            "remove_furniture_tool": remove_furniture_tool,
            "rescale_furniture_tool": rescale_furniture_tool,
            "list_available_assets": list_available_assets,
        }

    @log_scene_action
    def _add_furniture_to_scene_impl(
        self,
        asset_id: str,
        x: float,
        y: float,
        z: float,
        roll: float = 0.0,
        pitch: float = 0.0,
        yaw: float = 0.0,
    ) -> str:
        """Implementation for placing an asset from the registry into the scene.

        Creates a new scene object instance with a unique object_id from the asset
        template.

        Rotations are in degrees.
        """
        console_logger.info("Tool called: add_furniture_to_scene_tool")
        try:
            console_logger.debug(f"Attempting to place asset: {asset_id}")

            # Convert string ID to UniqueID.
            try:
                unique_id = UniqueID(asset_id)
            except Exception:
                return self._create_failure_result(
                    asset_id=asset_id,
                    message=f"Invalid asset ID format: {asset_id}",
                    error_type=FurnitureErrorType.ASSET_NOT_FOUND,
                )

            # Get the asset from registry.
            original_asset = self.asset_manager.get_asset_by_id(unique_id)
            if not original_asset:
                available_assets = self.asset_manager.list_available_assets()
                available_ids = [str(a.object_id) for a in available_assets]
                return self._create_failure_result(
                    asset_id=asset_id,
                    message=f"Asset {asset_id} not found in registry. "
                    f"Available: {available_ids}",
                    error_type=FurnitureErrorType.ASSET_NOT_FOUND,
                )

            safety_denial = self._safety_denial_add(
                asset_text=(
                    f"{original_asset.name} "
                    f"{getattr(original_asset, 'description', '')}"
                )
            )
            if safety_denial:
                return self._create_failure_result(
                    asset_id=asset_id,
                    message=safety_denial,
                    error_type=FurnitureErrorType.INVALID_POSITION,
                )

            console_logger.debug(
                f"Placing asset {asset_id} ({original_asset.name}) at position "
                f"({x}, {y}, {z}), rotation "
                f"({roll:.1f}°, {pitch:.1f}°, {yaw:.1f}°)"
            )

            # Validate position is within floor plan bounds.
            is_valid, error_msg = self._check_floor_bounds(x=x, y=y)
            if not is_valid:
                return self._create_failure_result(
                    asset_id=asset_id,
                    message=error_msg,
                    error_type=FurnitureErrorType.POSITION_OUT_OF_BOUNDS,
                )

            # Create new scene object with unique ID and specified pose.
            # Convert degrees to radians for Drake's RigidTransform.
            scene_object = copy_scene_object_with_new_pose(
                scene=self.scene,
                original=original_asset,
                x=x,
                y=y,
                z=z,
                roll=math.radians(roll),
                pitch=math.radians(pitch),
                yaw=math.radians(yaw),
            )

            base_transform, base_lift = self._ground_transform_to_floor_if_needed(
                scene_obj=scene_object,
                transform=scene_object.transform,
            )
            if base_lift > 0:
                console_logger.info(
                    "Auto-grounded furniture asset '%s' by lifting %.3fm before "
                    "room-bound validation",
                    original_asset.name,
                    base_lift,
                )
            scene_object.transform = base_transform

            # Apply placement noise for realistic variation.
            noisy_transform = apply_placement_noise(
                transform=scene_object.transform,
                position_xy_std_meters=self.active_noise_profile.position_xy_std_meters,
                rotation_yaw_std_degrees=self.active_noise_profile.rotation_yaw_std_degrees,
            )
            noisy_transform, noisy_lift = self._ground_transform_to_floor_if_needed(
                scene_obj=scene_object,
                transform=noisy_transform,
            )
            if noisy_lift > 0:
                console_logger.info(
                    "Auto-grounded noisy furniture pose for '%s' by lifting %.3fm",
                    original_asset.name,
                    noisy_lift,
                )
            valid_noisy, noisy_error = self._check_object_bounds_for_transform(
                scene_obj=scene_object,
                transform=noisy_transform,
            )
            if valid_noisy:
                scene_object.transform = noisy_transform
            else:
                valid_base, base_error = self._check_object_bounds_for_transform(
                    scene_obj=scene_object,
                    transform=base_transform,
                )
                if not valid_base:
                    return self._create_failure_result(
                        asset_id=asset_id,
                        message=base_error,
                        error_type=FurnitureErrorType.POSITION_OUT_OF_BOUNDS,
                    )
                console_logger.info(
                    "Placement noise would violate room bounds for %s; using "
                    "un-noised transform. Noise error: %s",
                    scene_object.object_id,
                    noisy_error,
                )
                scene_object.transform = base_transform

            # Add to scene.
            self.scene.add_object(scene_object)

            # Log what changed.
            new_position = scene_object.transform.translation()
            new_rpy = RollPitchYaw(scene_object.transform.rotation())
            new_roll, new_pitch, new_yaw = (
                math.degrees(new_rpy.roll_angle()),
                math.degrees(new_rpy.pitch_angle()),
                math.degrees(new_rpy.yaw_angle()),
            )
            console_logger.info(
                f"Successfully placed asset '{original_asset.name}' as object "
                f"'{scene_object.object_id}' at position ({new_position[0]:.3f}, "
                f"{new_position[1]:.3f}, {new_position[2]:.3f}) and "
                f"rotation ({new_roll:.1f}°, {new_pitch:.1f}°, {new_yaw:.1f}°)"
            )

            return self._create_success_result(
                asset_id=asset_id, furniture_obj=scene_object
            )

        except Exception as e:
            console_logger.error(f"Error placing asset '{asset_id}': {e}")
            return self._create_failure_result(
                asset_id=asset_id,
                message=f"Failed to place asset: {str(e)}",
            )

    def _create_success_result(self, asset_id: str, furniture_obj: SceneObject) -> str:
        """Create success result for furniture placement."""
        position = furniture_obj.transform.translation()
        rpy = RollPitchYaw(furniture_obj.transform.rotation())

        return FurniturePlacementResult(
            success=True,
            message=(
                f"Successfully placed asset '{furniture_obj.name}' as object "
                f"'{furniture_obj.object_id}'. "
                f"Use object_id '{furniture_obj.object_id}' for remove/move operations."
            ),
            asset_id=asset_id,
            object_id=str(furniture_obj.object_id),
            position=Position3D(x=position[0], y=position[1], z=position[2]),
            rotation=Rotation3D(
                roll=math.degrees(rpy.roll_angle()),  # Convert radians to degrees
                pitch=math.degrees(rpy.pitch_angle()),
                yaw=math.degrees(rpy.yaw_angle()),
            ),
            has_geometry=bool(furniture_obj.geometry_path),
        ).to_json()

    def _create_failure_result(
        self, asset_id: str, message: str, error_type: FurnitureErrorType | None = None
    ) -> str:
        """Create failure result for furniture placement."""
        return FurniturePlacementResult(
            success=False,
            message=message,
            asset_id=asset_id,
            object_id="",
            position=Position3D(x=0.0, y=0.0, z=0.0),
            rotation=Rotation3D(roll=0.0, pitch=0.0, yaw=0.0),
            has_geometry=False,
            error_type=error_type,
        ).to_json()

    @log_scene_action
    def _move_furniture_impl(
        self,
        object_id: str,
        x: float,
        y: float,
        z: float,
        roll: float,
        pitch: float,
        yaw: float,
    ) -> str:
        """
        Implementation for moving furniture to absolute pose. Rotations are in degrees.
        """
        console_logger.info("Tool called: move_furniture_tool")
        try:
            # Convert string ID to UniqueID.
            unique_id = UniqueID(object_id)

            # Check if object exists.
            scene_obj = self.scene.get_object(unique_id)
            if scene_obj is None:
                return FurnitureOperationResult(
                    success=False,
                    message=f"Object with ID '{object_id}' not found in scene",
                    object_id=object_id,
                    error_type=FurnitureErrorType.OBJECT_NOT_FOUND,
                ).to_json()

            # Check if object is immutable.
            if scene_obj.immutable:
                return FurnitureOperationResult(
                    success=False,
                    message=(
                        f"Cannot move {scene_obj.name}: architectural element is "
                        "immutable"
                    ),
                    object_id=object_id,
                    error_type=FurnitureErrorType.IMMUTABLE_OBJECT,
                    suggested_action=(
                        "Walls and architectural elements cannot be repositioned"
                    ),
                ).to_json()

            safety_denial = self._safety_denial_move(object_id=object_id)
            if safety_denial:
                return FurnitureOperationResult(
                    success=False,
                    message=safety_denial,
                    object_id=object_id,
                    error_type=FurnitureErrorType.INVALID_POSITION,
                    suggested_action="Request a critique or finish with the best checkpoint.",
                ).to_json()

            # Validate position is within floor plan bounds.
            is_valid, error_msg = self._check_floor_bounds(x=x, y=y)
            if not is_valid:
                return FurnitureOperationResult(
                    success=False,
                    message=error_msg,
                    object_id=object_id,
                    error_type=FurnitureErrorType.POSITION_OUT_OF_BOUNDS,
                ).to_json()

            # Get current position and rotation.
            current_transform = scene_obj.transform
            current_position = current_transform.translation()
            current_rpy = RollPitchYaw(current_transform.rotation())

            new_position = np.array([x, y, z])
            new_rotation = np.array([roll, pitch, yaw])
            current_rotation = np.array(
                [
                    math.degrees(current_rpy.roll_angle()),
                    math.degrees(current_rpy.pitch_angle()),
                    math.degrees(current_rpy.yaw_angle()),
                ]
            )  # Current rotation in degrees for comparison

            # Check if both position and rotation are unchanged.
            position_unchanged = np.allclose(current_position, new_position, atol=1e-6)
            rotation_unchanged = np.allclose(current_rotation, new_rotation, atol=1e-6)

            if position_unchanged and rotation_unchanged:
                console_logger.info(
                    f"Furniture '{scene_obj.name}'/'{object_id}' is already at position "
                    f"({x}, {y}, {z}) and rotation ({roll}, {pitch}, {yaw}) - no "
                    "movement needed"
                )
                return FurnitureOperationResult(
                    success=False,
                    message=f"{scene_obj.name} is already at the target position and "
                    "rotation - no movement needed",
                    object_id=object_id,
                    error_type=FurnitureErrorType.NO_MOVEMENT,
                    current_position=Position3D(
                        x=current_position[0],
                        y=current_position[1],
                        z=current_position[2],
                    ),
                    attempted_position=Position3D(x=x, y=y, z=z),
                    current_rotation=Rotation3D(
                        roll=current_rotation[0],
                        pitch=current_rotation[1],
                        yaw=current_rotation[2],
                    ),
                    attempted_rotation=Rotation3D(roll=roll, pitch=pitch, yaw=yaw),
                    suggested_action="Try moving to a different position or rotation",
                ).to_json()

            # Create new transform with absolute position and rotation.
            # Convert degrees to radians for Drake's RigidTransform.
            new_rpy = RollPitchYaw(
                math.radians(roll), math.radians(pitch), math.radians(yaw)
            )
            new_transform = RigidTransform(rpy=new_rpy, p=[x, y, z])
            new_transform, base_lift = self._ground_transform_to_floor_if_needed(
                scene_obj=scene_obj,
                transform=new_transform,
            )
            if base_lift > 0:
                console_logger.info(
                    "Auto-grounded furniture '%s'/'%s' by lifting %.3fm before "
                    "move validation",
                    scene_obj.name,
                    object_id,
                    base_lift,
                )

            noisy_transform = apply_placement_noise(
                transform=new_transform,
                position_xy_std_meters=self.active_noise_profile.position_xy_std_meters,
                rotation_yaw_std_degrees=self.active_noise_profile.rotation_yaw_std_degrees,
            )
            noisy_transform, noisy_lift = self._ground_transform_to_floor_if_needed(
                scene_obj=scene_obj,
                transform=noisy_transform,
            )
            if noisy_lift > 0:
                console_logger.info(
                    "Auto-grounded noisy furniture move for '%s' by lifting %.3fm",
                    object_id,
                    noisy_lift,
                )
            valid_noisy, noisy_error = self._check_object_bounds_for_transform(
                scene_obj=scene_obj,
                transform=noisy_transform,
            )
            if valid_noisy:
                new_transform = noisy_transform
            else:
                valid_base, base_error = self._check_object_bounds_for_transform(
                    scene_obj=scene_obj,
                    transform=new_transform,
                )
                if not valid_base:
                    return FurnitureOperationResult(
                        success=False,
                        message=base_error,
                        object_id=object_id,
                        error_type=FurnitureErrorType.POSITION_OUT_OF_BOUNDS,
                    ).to_json()
                console_logger.info(
                    "Placement noise would violate room bounds for %s; using "
                    "un-noised transform. Noise error: %s",
                    object_id,
                    noisy_error,
                )

            # Update object to new absolute pose.
            self.scene.move_object(object_id=unique_id, new_transform=new_transform)

            # Log what changed.
            changes = []
            if not position_unchanged:
                new_pos = new_transform.translation()
                changes.append(
                    f"position from ({current_position[0]:.3f}, "
                    f"{current_position[1]:.3f}, {current_position[2]:.3f}) to "
                    f"({new_pos[0]:.3f}, {new_pos[1]:.3f}, {new_pos[2]:.3f})"
                )
            if not rotation_unchanged:
                new_rpy = RollPitchYaw(new_transform.rotation())
                new_roll, new_pitch, new_yaw = (
                    math.degrees(new_rpy.roll_angle()),
                    math.degrees(new_rpy.pitch_angle()),
                    math.degrees(new_rpy.yaw_angle()),
                )
                changes.append(
                    f"rotation from ({current_rotation[0]:.3f}°, "
                    f"{current_rotation[1]:.3f}°, {current_rotation[2]:.3f}°) to "
                    f"({new_roll:.3f}°, {new_pitch:.3f}°, {new_yaw:.3f}°)"
                )

            console_logger.info(
                f"Moved furniture '{scene_obj.name}'/'{object_id}': {' and '.join(changes)}"
            )

            return FurnitureOperationResult(
                success=True,
                message=f"Successfully moved {scene_obj.name} to new position and "
                "rotation",
                object_id=object_id,
            ).to_json()

        except Exception as e:
            console_logger.error(f"Error moving furniture '{object_id}': {e}")
            return FurnitureOperationResult(
                success=False,
                message=f"Failed to move furniture: {str(e)}",
                object_id=object_id,
            ).to_json()

    @log_scene_action
    def _remove_furniture_impl(self, object_id: str) -> str:
        """Implementation for removing furniture."""
        console_logger.info("Tool called: remove_furniture_tool")
        try:
            # Convert string ID to UniqueID.
            unique_id = UniqueID(object_id)

            # Check if object exists.
            scene_obj = self.scene.get_object(unique_id)
            if scene_obj is None:
                base_name = (
                    object_id.rsplit("_", 1)[0] if "_" in object_id else object_id
                )
                return FurnitureOperationResult(
                    success=False,
                    message=(
                        f"Object with ID '{object_id}' not found in scene.\n\n"
                        f"Causes:\n"
                        f"1. Missing ID postfix - IDs have random postfixes like "
                        f"'{base_name}_a1b2c3'\n"
                        f"2. Object already removed\n"
                        f"3. Typo in object_id\n\n"
                        f"Call get_current_scene_state() to see current object IDs with "
                        f"postfixes. Find objects whose names start with '{base_name}'."
                    ),
                    object_id=object_id,
                    error_type=FurnitureErrorType.OBJECT_NOT_FOUND,
                    suggested_action="Call get_current_scene_state() to verify object IDs",
                ).to_json()

            # Check if object is immutable.
            if scene_obj.immutable:
                return FurnitureOperationResult(
                    success=False,
                    message=(
                        f"Cannot remove {scene_obj.name}: architectural element is "
                        "immutable"
                    ),
                    object_id=object_id,
                    error_type=FurnitureErrorType.IMMUTABLE_OBJECT,
                    suggested_action=(
                        "Walls and architectural elements cannot be removed"
                    ),
                ).to_json()

            safety_denial = self._safety_denial_remove(
                object_id=object_id,
                scene_obj=scene_obj,
            )
            if safety_denial:
                return FurnitureOperationResult(
                    success=False,
                    message=safety_denial,
                    object_id=object_id,
                    error_type=FurnitureErrorType.IMMUTABLE_OBJECT,
                    suggested_action="Move required furniture locally instead of deleting it.",
                ).to_json()

            # Remove from scene.
            removed = self.scene.remove_object(unique_id)

            if not removed:
                # Log detailed information for debugging.
                scene_ids = list(self.scene.objects.keys())
                console_logger.info(
                    f"Failed to remove object '{object_id}' from scene. "
                    f"Object exists in scene (get_object succeeded) but remove_object "
                    f"returned False."
                )
                console_logger.info(f"Attempted to remove ID: {object_id}")
                console_logger.info(f"Attempted to remove ID repr: {repr(unique_id)}")
                console_logger.info(
                    f"Current scene object IDs ({len(scene_ids)}): {scene_ids}"
                )
                return FurnitureOperationResult(
                    success=False,
                    message=(
                        f"Object {object_id} exists but could not be removed from "
                        f"scene"
                    ),
                    object_id=object_id,
                    error_type=FurnitureErrorType.OBJECT_NOT_FOUND,
                ).to_json()

            console_logger.info(f"Removed furniture '{scene_obj.name}' from scene")
            return FurnitureOperationResult(
                success=True,
                message=f"Successfully removed {scene_obj.name} from scene",
                object_id=object_id,
            ).to_json()

        except Exception as e:
            console_logger.error(f"Error removing furniture '{object_id}': {e}")
            return FurnitureOperationResult(
                success=False,
                message=f"Failed to remove furniture: {str(e)}",
                object_id=object_id,
            ).to_json()

    @log_scene_action
    def _rescale_furniture_impl(self, object_id: str, scale_factor: float) -> str:
        """Implementation for rescaling furniture."""
        console_logger.info(
            f"Tool called: rescale_furniture (id={object_id}, scale={scale_factor})"
        )
        safety_denial = self._safety_denial_rescale(
            object_id=object_id,
            scale_factor=scale_factor,
        )
        if safety_denial:
            return RescaleResult(
                success=False,
                message=safety_denial,
                object_id=object_id,
                error_type=RescaleErrorType.INVALID_SCALE_FACTOR,
            ).to_json()

        result = rescale_object_common(
            scene=self.scene,
            object_id=object_id,
            scale_factor=scale_factor,
            object_type_name="furniture",
            asset_registry=self.asset_manager.registry,
        )
        return result.to_json()

    def _add_duplicate_warning(self, message_parts: list[str]) -> None:
        """Add duplicate warning to message if duplicates were detected."""
        duplicate_info = self.asset_manager.last_duplicate_info
        if duplicate_info:
            total_duplicates = sum(len(indices) for indices in duplicate_info.values())
            message_parts.append("")
            message_parts.append("⚠️  DUPLICATES REMOVED:")
            message_parts.append(
                f"You requested {total_duplicates} duplicate item(s). "
                "I generated each unique item only once:"
            )
            for desc, indices in duplicate_info.items():
                count = len(indices) + 1  # +1 for the original
                message_parts.append(f"  - '{desc}' (requested {count} times)")
            message_parts.append("")
            message_parts.append(
                "REMINDER: To place multiple identical items, use "
                "add_furniture_to_scene_tool with the SAME asset_id at different "
                "positions. Do NOT generate the same asset multiple times."
            )

    def _build_partial_success_message(
        self,
        result: DomainAssetGenerationResult,
        generated_assets: list[GeneratedAsset],
    ) -> tuple[str, str]:
        """Build message for partial success case."""
        message_parts = [
            f"Generated {len(generated_assets)} asset(s) successfully, but "
            f"{len(result.failed_assets)} failed:"
        ]

        # List successful assets.
        if generated_assets:
            message_parts.append("\n✓ SUCCESSFUL:")
            for asset in generated_assets:
                message_parts.append(f"  - {asset.name} (ID: {asset.object_id})")

        # List failed assets with error details.
        message_parts.append("\n✗ FAILED:")
        failure_details = []
        for failed in result.failed_assets:
            message_parts.append(f"  - {failed.description}: {failed.error_message}")
            failure_details.append(f"- {failed.description}: {failed.error_message}")

        has_fatal_setup_error = any(
            "Fatal asset retrieval setup error" in failed.error_message
            for failed in result.failed_assets
        )
        if has_fatal_setup_error:
            message_parts.append(
                "\nRECOMMENDATION: Stop calling generate_assets. This is an "
                "environment setup error, not a prompt problem. Fix the missing "
                "retrieval dependency first, then rerun the scene."
            )
        else:
            message_parts.append(
                "\nRECOMMENDATION: Regenerate only the failed assets with adjusted "
                "prompts if needed."
            )

        # Add duplicate warning if applicable.
        self._add_duplicate_warning(message_parts)

        return "\n".join(message_parts), "\n".join(failure_details)

    def _build_full_success_message(
        self, generated_assets: list[GeneratedAsset], object_type: ObjectType
    ) -> str:
        """Build message for full success case."""
        message_parts = [
            f"Successfully generated {len(generated_assets)} unique "
            f"{object_type.value} asset(s):"
        ]

        # List generated assets with IDs.
        for asset in generated_assets:
            message_parts.append(f"  - {asset.name} (ID: {asset.object_id})")

        # Add duplicate warning if applicable.
        self._add_duplicate_warning(message_parts)

        return "\n".join(message_parts)

    def _generate_assets_impl(self, request: AssetGenerationRequest) -> str:
        """Implementation for generating assets with partial success handling."""
        console_logger.info(
            f"Generating batch of {len(request.object_descriptions)} assets"
        )
        start_time = time.time()

        # Generate assets using the asset manager.
        result = self.asset_manager.generate_assets(request)

        # Convert successful assets to DTOs.
        generated_assets = [
            GeneratedAsset(
                name=obj.name,
                object_id=str(obj.object_id),
                description=obj.description,
                width=(
                    float(obj.bbox_max[0] - obj.bbox_min[0])
                    if obj.bbox_min is not None and obj.bbox_max is not None
                    else None
                ),
                depth=(
                    float(obj.bbox_max[1] - obj.bbox_min[1])
                    if obj.bbox_min is not None and obj.bbox_max is not None
                    else None
                ),
                height=(
                    float(obj.bbox_max[2] - obj.bbox_min[2])
                    if obj.bbox_min is not None and obj.bbox_max is not None
                    else None
                ),
            )
            for obj in result.successful_assets
        ]

        elapsed_time = time.time() - start_time

        # Handle partial success.
        if result.has_failures:
            console_logger.warning(
                f"Asset generation completed with {len(result.failed_assets)} "
                f"failure(s) and {len(result.successful_assets)} success(es) in "
                f"{elapsed_time:.2f} seconds"
            )

            message, failure_details = self._build_partial_success_message(
                result=result, generated_assets=generated_assets
            )

            return AssetGenerationResult(
                success=False,
                assets=generated_assets,
                message=message,
                successful_count=len(generated_assets),
                failed_count=len(result.failed_assets),
                failures=failure_details,
            ).to_json()

        # All succeeded.
        console_logger.info(
            f"Successfully generated {len(generated_assets)} assets in batch in "
            f"{elapsed_time:.2f} seconds"
        )

        message = self._build_full_success_message(
            generated_assets=generated_assets, object_type=request.object_type
        )

        return AssetGenerationResult(
            success=True,
            assets=generated_assets,
            message=message,
        ).to_json()

    def _list_available_assets_impl(self) -> str:
        """List all assets available for reuse.

        Returns:
            JSON response with list of available assets.
        """
        console_logger.info("Tool called: list_available_assets")
        try:
            available_assets = self.asset_manager.list_available_assets()

            asset_infos = [
                AssetInfo.from_scene_object(asset) for asset in available_assets
            ]

            result = AvailableAssetsResult(
                success=True,
                assets=asset_infos,
                count=len(asset_infos),
                message=f"Found {len(asset_infos)} available assets for reuse",
            )

            console_logger.info(f"Listed {len(asset_infos)} available assets")
            return result.to_json()

        except Exception as e:
            result = AvailableAssetsResult(
                success=False,
                assets=[],
                count=0,
                message=f"Failed to list available assets: {e}",
            )
            return result.to_json()
