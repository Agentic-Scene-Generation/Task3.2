"""Physical feasibility post-processing for scene collision resolution.

This module provides two-stage post-processing adapted from scene_gen repository:
1. Projection - IK-based collision resolution with configurable DOF constraints
2. Simulation - Physics settling to static equilibrium (always full 6DOF)

See: https://github.com/nepfaff/steerable-scene-generation/blob/main/steerable_scene_generation/algorithms/scene_diffusion/postprocessing.py
"""

import logging
import signal
import tempfile
import time

from collections.abc import Collection
from pathlib import Path

import numpy as np
import trimesh

from omegaconf import DictConfig
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    BodyIndex,
    Context,
    DiagramBuilder,
    DiscreteContactApproximation,
    EventStatus,
    InverseKinematics,
    IpoptSolver,
    LoadModelDirectives,
    MeshcatVisualizer,
    ModelInstanceIndex,
    MultibodyPlant,
    ProcessModelDirectives,
    Quaternion,
    RigidTransform,
    RotationMatrix,
    SceneGraph,
    Simulator,
    SnoptSolver,
    SolverOptions,
    StartMeshcat,
)
from pydrake.geometry import CollisionFilterDeclaration, GeometrySet, QueryObject, Role
from pydrake.geometry.optimization import HPolyhedron, VPolytope

from scenesmith.agent_utils.drake_utils import (
    create_drake_plant_and_scene_graph_from_scene,
)
from scenesmith.agent_utils.physics_validation import compute_scene_collisions
from scenesmith.agent_utils.room import (
    ObjectType,
    RoomScene,
    SceneObject,
    UniqueID,
    deserialize_rigid_transform,
    serialize_rigid_transform,
)
from scenesmith.utils.geometry_utils import safe_convex_hull_2d

console_logger = logging.getLogger(__name__)


def _find_surface_owner(
    scene: RoomScene, surface_id: UniqueID
) -> tuple[UniqueID | None, bool]:
    """Find the furniture or floor that owns a support surface.

    Args:
        scene: RoomScene containing objects.
        surface_id: ID of the support surface to find owner for.

    Returns:
        Tuple of (owner_id, is_floor) where:
        - owner_id: UniqueID of the owning furniture/floor, or None if not found
        - is_floor: True if owner is a floor object, False otherwise
    """
    for obj in scene.objects.values():
        if obj.object_type in (ObjectType.FURNITURE, ObjectType.FLOOR):
            for surface in obj.support_surfaces:
                if surface.surface_id == surface_id:
                    is_floor = obj.object_type == ObjectType.FLOOR
                    return obj.object_id, is_floor
    return None, False


def compute_tilt_angle_degrees(transform: RigidTransform) -> float:
    """Compute tilt angle (deviation from upright) in degrees.

    Measures how much the object's local up-vector (Z-axis) deviates from the
    world up-vector. This captures combined roll+pitch rotation while ignoring
    yaw (turning in place).

    Args:
        transform: Object's world-frame pose.

    Returns:
        Tilt angle in degrees (0 = perfectly upright, 90 = horizontal, 180 = inverted).
    """
    # Get object's local Z-axis (up) in world frame.
    object_up = transform.rotation().matrix() @ np.array([0.0, 0.0, 1.0])
    world_up = np.array([0.0, 0.0, 1.0])

    # Compute angle between vectors.
    cos_tilt = np.clip(np.dot(object_up, world_up), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_tilt)))


def _furniture_simulation_instability_reason(
    obj: SceneObject,
    pre_simulation_transform: RigidTransform,
    tilt_threshold_degrees: float,
) -> str | None:
    """Describe a furniture pose that cannot be a plausible settled result."""
    translation = np.asarray(obj.transform.translation(), dtype=float)
    rotation = np.asarray(obj.transform.rotation().matrix(), dtype=float)
    if not np.all(np.isfinite(translation)) or not np.all(np.isfinite(rotation)):
        return "non-finite simulated pose"

    tilt_angle = compute_tilt_angle_degrees(obj.transform)
    if tilt_angle > tilt_threshold_degrees:
        return (
            f"tilt={tilt_angle:.1f}deg > " f"threshold={tilt_threshold_degrees:.1f}deg"
        )

    displacement = float(
        np.linalg.norm(translation - pre_simulation_transform.translation())
    )
    try:
        extent = float(
            np.linalg.norm(
                np.asarray(obj.bbox_max, dtype=float)
                - np.asarray(obj.bbox_min, dtype=float)
            )
        )
    except (TypeError, ValueError):
        extent = 0.0
    if not np.isfinite(extent) or extent < 0.0:
        extent = 0.0
    # Dynamic settling should be local. A multi-object-length jump indicates a
    # broken collision proxy or numerical explosion, even when the final local
    # up-axis happens to remain below the tilt threshold.
    displacement_limit = max(1.0, 2.0 * extent)
    if displacement > displacement_limit:
        return (
            f"displacement={displacement:.2f}m > "
            f"scale-aware limit={displacement_limit:.2f}m"
        )
    return None


def _world_bbox_bottom_z(obj: SceneObject, transform: RigidTransform) -> float | None:
    """Return the bottom of an object's rendered AABB at a given transform."""
    if obj.bbox_min is None or obj.bbox_max is None:
        return None
    try:
        bbox_min = np.asarray(obj.bbox_min, dtype=float)
        bbox_max = np.asarray(obj.bbox_max, dtype=float)
    except (TypeError, ValueError):
        return None
    if (
        bbox_min.shape != (3,)
        or bbox_max.shape != (3,)
        or not np.all(np.isfinite(bbox_min))
        or not np.all(np.isfinite(bbox_max))
    ):
        return None

    corners = np.asarray(
        [
            transform @ np.asarray([x, y, z], dtype=float)
            for x in (bbox_min[0], bbox_max[0])
            for y in (bbox_min[1], bbox_max[1])
            for z in (bbox_min[2], bbox_max[2])
        ]
    )
    bottom_z = float(np.min(corners[:, 2]))
    return bottom_z if np.isfinite(bottom_z) else None


def _restore_furniture_that_fell_through_floor(
    scene: RoomScene,
    pre_simulation_transforms: dict[UniqueID, RigidTransform],
    floor_z: float = 0.0,
    tolerance_m: float = 0.05,
) -> list[UniqueID]:
    """Restore furniture when simulation moves a valid rendered bbox below floor.

    This targets invalid collision-proxy outcomes rather than ordinary settling:
    an object is restored only when its rendered bounds started above the floor
    tolerance and ended below it. Objects that already penetrated the floor are
    left unchanged so this safeguard cannot hide an invalid input layout.
    """
    restored: list[UniqueID] = []
    floor_limit = floor_z - tolerance_m
    for obj_id, pre_transform in pre_simulation_transforms.items():
        obj = scene.get_object(obj_id)
        if obj is None or obj.object_type != ObjectType.FURNITURE:
            continue
        pre_bottom_z = _world_bbox_bottom_z(obj, pre_transform)
        post_bottom_z = _world_bbox_bottom_z(obj, obj.transform)
        if (
            pre_bottom_z is None
            or post_bottom_z is None
            or pre_bottom_z < floor_limit
            or post_bottom_z >= floor_limit
        ):
            continue
        obj.transform = pre_transform
        restored.append(obj_id)
        console_logger.warning(
            "Restored furniture %s after simulation moved its rendered bounds "
            "through the floor (bottom_z %.3fm -> %.3fm, limit %.3fm)",
            obj_id,
            pre_bottom_z,
            post_bottom_z,
            floor_limit,
        )
    return restored


def _restore_anomalous_furniture_projection(
    scene: RoomScene,
    pre_projection_transforms: dict[UniqueID, RigidTransform],
) -> list[UniqueID]:
    """Rollback malformed projection poses and scale-inconsistent room exits."""
    restored: list[UniqueID] = []
    for obj_id, pre_transform in pre_projection_transforms.items():
        obj = scene.get_object(obj_id)
        if obj is None or obj.object_type != ObjectType.FURNITURE:
            continue
        reason = _furniture_simulation_instability_reason(
            obj,
            pre_transform,
            tilt_threshold_degrees=45.0,
        )
        if reason is None:
            continue
        if reason.startswith("displacement=") and not _projection_exits_room(
            scene, obj
        ):
            # A large solve can be legitimate for a severely overlapped input.
            # Treat displacement as anomalous only when it also leaves the
            # room's plausible horizontal/vertical extent.
            continue
        obj.transform = pre_transform
        restored.append(obj_id)
        console_logger.warning(
            "Restored furniture %s after anomalous non-penetration projection: %s",
            obj_id,
            reason,
        )
    return restored


def _projection_exits_room(scene: RoomScene, obj: SceneObject) -> bool:
    """Return whether projected rendered bounds lie outside the room envelope."""
    room_geometry = scene.room_geometry
    if room_geometry is None:
        return False
    try:
        bounds = obj.compute_world_bounds()
    except (TypeError, ValueError):
        bounds = None
    if bounds is None:
        point = np.asarray(obj.transform.translation(), dtype=float)
        world_min = world_max = point
    else:
        world_min = np.asarray(bounds[0], dtype=float)
        world_max = np.asarray(bounds[1], dtype=float)
    if not np.all(np.isfinite(world_min)) or not np.all(np.isfinite(world_max)):
        return True

    half_length = max(0.0, float(room_geometry.length)) / 2.0
    half_width = max(0.0, float(room_geometry.width)) / 2.0
    wall_height = max(0.0, float(room_geometry.wall_height))
    margin_m = max(0.05, float(room_geometry.wall_thickness))
    return bool(
        (half_length > 0.0 and world_min[0] < -half_length - margin_m)
        or (half_length > 0.0 and world_max[0] > half_length + margin_m)
        or (half_width > 0.0 and world_min[1] < -half_width - margin_m)
        or (half_width > 0.0 and world_max[1] > half_width + margin_m)
        or world_min[2] < -margin_m
        or (wall_height > 0.0 and world_max[2] > wall_height + margin_m)
    )


def _restore_collectively_unstable_instances(
    scene: RoomScene,
    pre_simulation_transforms: dict[UniqueID, RigidTransform],
    fallen_ids: list[UniqueID],
    tilt_threshold_degrees: float,
    restored_through_floor_ids: Collection[UniqueID] = (),
) -> list[UniqueID]:
    """Keep an intact set when one shared collision proxy topples every copy.

    A bad SDF/collision proxy can make every instance of one asset fall during a
    single simulation pass.  The final timestep can leave one copy just below
    the removal threshold while the others have crossed it, so treat a group as
    collectively unstable when every copy has toppled, is nearly toppled, or
    was rolled back after falling through the floor. Restoring that set is
    safer than deleting an otherwise valid repeated furniture group. Genuine
    isolated falls retain the existing removal behaviour.
    """
    fallen_set = set(fallen_ids)
    restored_through_floor_set = set(restored_through_floor_ids)
    groups: dict[str, list[UniqueID]] = {}
    for obj in scene.objects.values():
        if obj.object_type != ObjectType.FURNITURE:
            continue
        asset_group_key = _furniture_asset_group_key(obj)
        if asset_group_key is None or obj.object_id not in pre_simulation_transforms:
            continue
        groups.setdefault(asset_group_key, []).append(obj.object_id)

    restored: list[UniqueID] = []
    near_topple_threshold_degrees = 0.9 * tilt_threshold_degrees
    for asset_group_key, object_ids in groups.items():
        if len(object_ids) < 2 or not any(
            obj_id in fallen_set for obj_id in object_ids
        ):
            continue
        if any(
            compute_tilt_angle_degrees(pre_simulation_transforms[obj_id])
            > tilt_threshold_degrees
            for obj_id in object_ids
        ):
            continue
        if not all(
            obj_id in fallen_set
            or obj_id in restored_through_floor_set
            or (
                (obj := scene.get_object(obj_id)) is not None
                and compute_tilt_angle_degrees(obj.transform)
                >= near_topple_threshold_degrees
            )
            for obj_id in object_ids
        ):
            continue
        for obj_id in object_ids:
            obj = scene.get_object(obj_id)
            if obj is not None:
                obj.transform = pre_simulation_transforms[obj_id]
                restored.append(obj_id)
        console_logger.warning(
            "Restored %d collectively unstable furniture instance(s) for shared "
            "collision proxy %s (near-topple threshold=%.1fdeg)",
            len(object_ids),
            asset_group_key,
            near_topple_threshold_degrees,
        )
    return restored


def _furniture_asset_group_key(obj: SceneObject) -> str | None:
    """Return a stable provenance key for grouping repeated furniture assets."""
    metadata = obj.metadata or {}
    asset_source = str(metadata.get("asset_source") or "unknown")
    for field_name in ("hssd_mesh_id", "objaverse_mesh_id", "articulated_id"):
        asset_id = metadata.get(field_name)
        if asset_id not in (None, ""):
            # 2026-07-31: each placement has a fresh SDF directory, so source IDs
            # are required to detect a shared bad collision proxy across copies.
            return f"{asset_source}:{field_name}:{asset_id}"

    asset_path = obj.sdf_path or obj.geometry_path
    return str(asset_path) if asset_path is not None else None


def _create_drake_plant_for_ik(
    scene: RoomScene,
    builder: DiagramBuilder,
    weld_furniture: bool = False,
    time_step: float = 0.0,
    free_objects: list[UniqueID] | None = None,
    include_objects: list[UniqueID] | None = None,
) -> tuple[
    MultibodyPlant,
    SceneGraph,
    dict[UniqueID, tuple[ModelInstanceIndex, BodyIndex]],
    dict[UniqueID, dict],
]:
    """Create Drake plant configured for IK optimization.

    Args:
        scene: RoomScene to load into the plant.
        builder: DiagramBuilder to use.
        weld_furniture: If True, weld furniture (only manipulands are free).
                        If False, all objects are free bodies.
        time_step: Physics time step (0.0 for kinematics-only).
        free_objects: If provided, only these specific objects will be free
            bodies. Overrides weld_furniture for these objects.
        include_objects: If provided, only include these objects in the Drake
            plant. Objects not in this list are excluded entirely (not just
            welded). Useful for performance optimization when only a subset of
            objects are relevant for collision checking.

    Returns:
        Tuple of (plant, scene_graph, object_indices, composite_info) where:
        - object_indices maps object ID to (model_instance_index, body_index)
        - composite_info maps composite ID to original transforms for delta computation
    """
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=time_step)
    if time_step > 0.0:
        plant.set_discrete_contact_approximation(DiscreteContactApproximation.kLagged)

    # Generate Drake directive with composite members welded for rigid unit behavior.
    directive_yaml = scene.to_drake_directive(
        include_objects=include_objects,
        weld_furniture=weld_furniture,
        free_objects=free_objects,
        exclude_room_geometry=False,
        weld_stack_members=True,
    )

    # Write directive to temporary file and load it.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(directive_yaml)
        temp_directive_path = f.name

    try:
        directives = LoadModelDirectives(temp_directive_path)
        ProcessModelDirectives(directives, plant, parser=None)
    finally:
        Path(temp_directive_path).unlink(missing_ok=True)

    plant.Finalize()

    # Build mapping from object ID to Drake indices.
    # Model names follow pattern: {name}_{id_suffix} from to_drake_directive().
    object_indices: dict[UniqueID, tuple[ModelInstanceIndex, BodyIndex]] = {}
    composite_info: dict[UniqueID, dict] = {}

    for obj in scene.objects.values():
        # Handle composite objects (stacks) by tracking bottom member.
        if obj.metadata.get("composite_type") == "stack":
            member_assets = obj.metadata.get("member_assets", [])
            if not member_assets:
                continue

            # Reconstruct bottom member model name (same pattern as to_drake_directive).
            bottom_member = member_assets[0]
            member_name = bottom_member.get("name", "stack_member")
            member_id = bottom_member.get("asset_id", "unknown")
            id_suffix = member_id.split("_")[-1][:8]
            stack_suffix = str(obj.object_id).split("_")[-1][:4]
            model_name = (
                f"{member_name.lower().replace(' ', '_')}_{id_suffix}_s{stack_suffix}_0"
            )

            try:
                model_idx = plant.GetModelInstanceByName(model_name)
                body_indices = plant.GetBodyIndices(model_idx)
                if len(body_indices) > 0:
                    body_idx = body_indices[0]
                    object_indices[obj.object_id] = (model_idx, body_idx)

                    # Store original bottom transform for delta computation.
                    composite_info[obj.object_id] = {
                        "original_bottom_transform": deserialize_rigid_transform(
                            bottom_member.get("transform", {})
                        ),
                        "member_assets": member_assets,
                    }
            except RuntimeError:
                console_logger.warning(f"Model {model_name} not found in plant")
            continue

        # Handle filled containers by tracking container as reference member.
        if obj.metadata.get("composite_type") == "filled_container":
            container_asset = obj.metadata.get("container_asset")
            fill_assets = obj.metadata.get("fill_assets", [])
            if not container_asset:
                continue

            # Reconstruct container model name (same pattern as to_drake_directive).
            container_name = container_asset.get("name", "container")
            container_id = container_asset.get("asset_id", "unknown")
            id_suffix = container_id.split("_")[-1][:8]
            fill_suffix = str(obj.object_id).split("_")[-1][:4]
            model_name = f"{container_name.lower().replace(' ', '_')}_{id_suffix}_f{fill_suffix}_c"

            try:
                model_idx = plant.GetModelInstanceByName(model_name)
                body_indices = plant.GetBodyIndices(model_idx)
                if len(body_indices) > 0:
                    body_idx = body_indices[0]
                    object_indices[obj.object_id] = (model_idx, body_idx)

                    # Store original container transform and assets for delta computation.
                    composite_info[obj.object_id] = {
                        "original_bottom_transform": deserialize_rigid_transform(
                            container_asset.get("transform", {})
                        ),
                        "container_asset": container_asset,
                        "fill_assets": fill_assets,
                        "composite_type": "filled_container",
                    }
            except RuntimeError:
                console_logger.warning(f"Model {model_name} not found in plant")
            continue

        # Handle piles by tracking first member (same structure as stack).
        if obj.metadata.get("composite_type") == "pile":
            member_assets = obj.metadata.get("member_assets", [])
            if not member_assets:
                continue

            # Reconstruct first member model name (same pattern as to_drake_directive).
            first_member = member_assets[0]
            member_name = first_member.get("name", "pile_member")
            member_id = first_member.get("asset_id", "unknown")
            id_suffix = member_id.split("_")[-1][:8]
            pile_suffix = str(obj.object_id).split("_")[-1][:4]
            model_name = (
                f"{member_name.lower().replace(' ', '_')}_{id_suffix}_p{pile_suffix}_0"
            )

            try:
                model_idx = plant.GetModelInstanceByName(model_name)
                body_indices = plant.GetBodyIndices(model_idx)
                if len(body_indices) > 0:
                    body_idx = body_indices[0]
                    object_indices[obj.object_id] = (model_idx, body_idx)

                    # Store original first member transform for delta computation.
                    composite_info[obj.object_id] = {
                        "original_bottom_transform": deserialize_rigid_transform(
                            first_member.get("transform", {})
                        ),
                        "member_assets": member_assets,
                        "composite_type": "pile",
                    }
            except RuntimeError:
                console_logger.warning(f"Model {model_name} not found in plant")
            continue

        if obj.sdf_path is None:
            continue

        # Reconstruct model name as done in to_drake_directive().
        id_suffix = str(obj.object_id).split("_")[-1][:8]
        model_name = f"{obj.name.lower().replace(' ', '_')}_{id_suffix}"

        try:
            model_idx = plant.GetModelInstanceByName(model_name)
            body_indices = plant.GetBodyIndices(model_idx)
            if len(body_indices) > 0:
                body_idx = body_indices[0]
                object_indices[obj.object_id] = (model_idx, body_idx)
        except RuntimeError:
            console_logger.warning(f"Model {model_name} not found in plant")

    return plant, scene_graph, object_indices, composite_info


def _update_scene_from_plant(
    scene: RoomScene,
    plant: MultibodyPlant,
    plant_context: Context,
    object_indices: dict[UniqueID, tuple[ModelInstanceIndex, BodyIndex]],
    composite_info: dict[UniqueID, dict] | None = None,
    operation_name: str = "Projection",
) -> None:
    """Update Scene object poses from Drake plant context.

    Args:
        scene: RoomScene to update (modified in place).
        plant: Drake MultibodyPlant.
        plant_context: Plant context with current poses.
        object_indices: Mapping from object ID to Drake indices.
        composite_info: Optional mapping from composite ID to original transforms.
            Used to update all composite member transforms based on reference member delta.
            Handles stacks, filled_containers, and piles.
        operation_name: Name of the operation for logging (e.g., "Projection" or
            "Simulation").
    """
    for obj_id, (model_idx, body_idx) in object_indices.items():
        obj = scene.get_object(obj_id)
        if obj is None:
            continue

        body = plant.get_body(body_idx)
        if not body.is_floating():
            # Welded body - pose is fixed.
            continue

        # Get updated pose from plant.
        positions = plant.GetPositions(plant_context, model_idx)
        if len(positions) < 7:
            # Not a floating body (quaternion + translation = 7 DOF).
            continue

        # Drake uses [qw, qx, qy, qz, x, y, z] for floating bodies.
        quaternion = positions[:4]
        translation = positions[4:7]

        # Normalize quaternion to handle numerical drift.
        quaternion = quaternion / np.linalg.norm(quaternion)

        # Create new RigidTransform.
        rotation = RotationMatrix(Quaternion(wxyz=quaternion))
        new_transform = RigidTransform(rotation, translation)

        # Log pose change if significant.
        old_translation = obj.transform.translation()
        delta_translation = new_transform.translation() - old_translation
        translation_change = np.linalg.norm(delta_translation)

        # Compute rotation change as angle between old and new orientations.
        old_rotation = obj.transform.rotation()
        delta_rotation = new_transform.rotation().multiply(old_rotation.inverse())
        rotation_angle_deg = np.degrees(delta_rotation.ToAngleAxis().angle())

        if (
            translation_change > 0.001 or rotation_angle_deg > 0.1
        ):  # 1mm or 0.1° threshold.
            console_logger.info(
                f"{operation_name} moved {obj.object_id}: "
                f"delta=({delta_translation[0]:.4f}, {delta_translation[1]:.4f}, "
                f"{delta_translation[2]:.4f}), rot={rotation_angle_deg:.2f}°"
            )

        # Update object transform.
        obj.transform = new_transform

    # Update composite member transforms based on reference member delta.
    if composite_info:
        for composite_id, info in composite_info.items():
            if composite_id not in object_indices:
                continue

            model_idx, body_idx = object_indices[composite_id]
            body = plant.get_body(body_idx)

            if not body.is_floating():
                continue

            # Get new bottom pose from plant.
            positions = plant.GetPositions(plant_context, model_idx)
            if len(positions) < 7:
                continue

            quaternion = positions[:4]
            translation = positions[4:7]
            quaternion = quaternion / np.linalg.norm(quaternion)
            rotation = RotationMatrix(Quaternion(wxyz=quaternion))
            new_bottom_transform = RigidTransform(rotation, translation)

            # Compute delta transform from original bottom position.
            orig_bottom = info["original_bottom_transform"]
            t_delta = new_bottom_transform @ orig_bottom.inverse()

            # Log pose change if significant.
            delta_translation = (
                new_bottom_transform.translation() - orig_bottom.translation()
            )
            translation_change = np.linalg.norm(delta_translation)

            # Compute rotation change for composite.
            delta_rotation = new_bottom_transform.rotation().multiply(
                orig_bottom.rotation().inverse()
            )
            rotation_angle_deg = np.degrees(delta_rotation.ToAngleAxis().angle())

            if (
                translation_change > 0.001 or rotation_angle_deg > 0.1
            ):  # 1mm or 0.1° threshold.
                console_logger.info(
                    f"{operation_name} moved composite {composite_id} reference member: "
                    f"delta=({delta_translation[0]:.4f}, {delta_translation[1]:.4f}, "
                    f"{delta_translation[2]:.4f}), rot={rotation_angle_deg:.2f}°"
                )

            # Apply delta to all member transforms.
            composite_obj = scene.get_object(composite_id)
            if composite_obj is None:
                continue

            # Check composite type to determine how to update transforms.
            composite_type = info.get("composite_type")

            if composite_type == "filled_container":
                # Filled container: update container_asset + fill_assets.
                container_asset = info.get("container_asset")
                if container_asset:
                    old_transform = deserialize_rigid_transform(
                        container_asset["transform"]
                    )
                    new_container_transform = t_delta @ old_transform
                    container_asset["transform"] = serialize_rigid_transform(
                        new_container_transform
                    )

                fill_assets = info.get("fill_assets", [])
                for fill_asset in fill_assets:
                    old_transform = deserialize_rigid_transform(fill_asset["transform"])
                    new_fill_transform = t_delta @ old_transform
                    fill_asset["transform"] = serialize_rigid_transform(
                        new_fill_transform
                    )

                # Update metadata with new transforms.
                composite_obj.metadata["container_asset"] = container_asset
                composite_obj.metadata["fill_assets"] = fill_assets
            else:
                # Stack or pile: update member_assets (same structure).
                updated_members = []
                for member in info["member_assets"]:
                    old_transform = deserialize_rigid_transform(member["transform"])
                    new_member_transform = t_delta @ old_transform
                    member["transform"] = serialize_rigid_transform(
                        new_member_transform
                    )
                    updated_members.append(member)

                # Update metadata with new transforms.
                composite_obj.metadata["member_assets"] = updated_members

            # Also update composite transform to match new reference member position.
            # This keeps bbox aligned with visual mesh positions.
            composite_obj.transform = new_bottom_transform


def _apply_self_collision_filtering(
    plant: MultibodyPlant, scene_graph: SceneGraph
) -> int:
    """Apply collision filtering to exclude self-collisions within each model.

    Articulated models (e.g., cabinets with doors/drawers) have internal collisions
    between their parts that are impossible to resolve without significant joint
    movement. These self-collisions cause MinimumDistanceConstraint to fail.

    This function filters out collisions between geometries within the same model,
    so the solver only needs to resolve inter-object and floor/wall penetrations.

    Args:
        plant: Finalized MultibodyPlant with models loaded.
        scene_graph: SceneGraph connected to the plant.

    Returns:
        Number of models that had self-collision filtering applied.
    """
    filter_manager = scene_graph.collision_filter_manager()
    inspector = scene_graph.model_inspector()
    models_filtered = 0

    # Iterate over all model instances (skip world model at index 0).
    world_model = plant.world_body().model_instance()
    for i in range(plant.num_model_instances()):
        model_idx = ModelInstanceIndex(i)
        if model_idx == world_model:
            continue

        # Collect all proximity geometries for this model.
        model_geometry_ids = []
        for body_idx in plant.GetBodyIndices(model_idx):
            frame_id = plant.GetBodyFrameIdOrThrow(body_idx)
            geom_ids = inspector.GetGeometries(frame_id, Role.kProximity)
            model_geometry_ids.extend(geom_ids)

        # Only apply filtering if model has multiple geometries.
        if len(model_geometry_ids) > 1:
            geometry_set = GeometrySet(model_geometry_ids)
            filter_manager.Apply(
                CollisionFilterDeclaration().ExcludeWithin(geometry_set)
            )
            models_filtered += 1

    if models_filtered > 0:
        console_logger.debug(
            f"Applied self-collision filtering to {models_filtered} models"
        )

    return models_filtered


def solve_non_penetration_ik(
    builder: DiagramBuilder,
    plant: MultibodyPlant,
    scene_graph: SceneGraph,
    influence_distance: float = 0.02,
    fix_rotation: bool = True,
    fix_z: bool = False,
    solver_name: str = "snopt",
    iteration_limit: int = 5000,
    time_limit_s: float = 360.0,
    xy_regions: dict[BodyIndex, HPolyhedron] | None = None,
    fixed_z_bodies: Collection[BodyIndex] | None = None,
) -> tuple[Context | None, bool]:
    """Solve IK for non-penetration projection of free bodies.

    Shared utility for projecting free-floating bodies to resolve penetrations.
    Builds diagram, sets up IK with proper quaternion/position costs, adds
    non-penetration constraint, and solves.

    Automatically applies self-collision filtering to exclude internal collisions
    within each model (e.g., cabinet doors vs body). This prevents the solver from
    trying to resolve impossible self-penetrations in articulated models.

    Args:
        builder: DiagramBuilder with plant and scene_graph added (not yet built).
        plant: Finalized MultibodyPlant with free bodies to project.
        scene_graph: SceneGraph connected to the plant (for collision filtering).
        influence_distance: Collision influence distance in meters.
        fix_rotation: If True, hard-constrain rotations to initial values.
        fix_z: If True, hard-constrain Z positions (XY-only projection).
        solver_name: NLP solver ("snopt" or "ipopt").
        iteration_limit: Max solver iterations.
        time_limit_s: Max solver time in seconds.
        xy_regions: Optional per-body 2D convex region constraints. Maps each
            BodyIndex to an HPolyhedron (Ax <= b) defining the allowed XY
            positions for that body's origin. Each body can have a different
            region based on its footprint.

            Computed via Pontryagin difference: for each object, the feasible
            region is surface_hpoly.PontryaginDifference(object_footprint_hpoly).
            This accounts for object shape - long objects (knives) can be
            placed closer to edges when oriented parallel.
        fixed_z_bodies: Optional bodies whose current world Z translation must
            remain fixed even when ``fix_z`` is False. This preserves explicit
            support assignments while allowing unsupported bodies to use full
            3D projection.

    Returns:
        Tuple of (plant_context, success).
        On success, plant_context has the projected positions applied.
        Caller can extract body poses via plant.EvalBodyPoseInWorld().
        On failure, returns (None, False).
    """
    # Apply self-collision filtering before building diagram.
    # This excludes internal collisions within articulated models (e.g., cabinet
    # doors vs body) that are impossible to resolve without significant joint movement.
    _apply_self_collision_filtering(plant=plant, scene_graph=scene_graph)

    # Build diagram to connect plant and scene_graph.
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)

    # Set up IK.
    ik = InverseKinematics(plant, plant_context)
    q_vars = ik.q()
    prog = ik.prog()

    # Get initial positions.
    q0 = plant.GetPositions(plant_context)
    if len(q0) == 0:
        console_logger.warning("No DOFs found for plant. Skipping projection.")
        return None, False

    # Add costs and constraints for each free body.
    for body_idx in plant.GetFloatingBaseBodies():
        body = plant.get_body(body_idx)
        q_start_idx = body.floating_positions_start()
        model_idx = cyclopean_get_model_instance_for_body(plant, body_idx)

        # Quaternion variables [qw, qx, qy, qz].
        model_quat_vars = q_vars[q_start_idx : q_start_idx + 4]
        quat0 = q0[q_start_idx : q_start_idx + 4]

        # Add quadratic cost to stay close to initial orientation.
        # For quaternion q and q0, the cost approximates 1-cos(θ) = 2 - 2*(qᵀq₀)².
        prog.AddQuadraticCost(
            -4 * np.outer(quat0, quat0),
            np.zeros((4,)),
            2,
            model_quat_vars,
            is_convex=False,
        )

        # Position variables [x, y, z].
        model_pos_vars = q_vars[q_start_idx + 4 : q_start_idx + 7]
        pos0 = q0[q_start_idx + 4 : q_start_idx + 7]

        # Add quadratic cost to stay close to initial position.
        prog.AddQuadraticErrorCost(np.eye(3), pos0, model_pos_vars)

        # Fix rotation if requested.
        if fix_rotation:
            model_q = plant.GetPositions(plant_context, model_idx)
            model_quat = model_q[:4]
            prog.AddBoundingBoxConstraint(model_quat, model_quat, model_quat_vars)

        # Fix Z if requested.
        if fix_z or (fixed_z_bodies and body_idx in fixed_z_bodies):
            model_q = plant.GetPositions(plant_context, model_idx)
            model_z = model_q[6]  # Z is at index 6 (after quat + x + y).
            z_var = q_vars[q_start_idx + 6]
            prog.AddBoundingBoxConstraint(model_z, model_z, [z_var])

        # Add XY convex region constraints if provided.
        # Each region is an HPolyhedron from Pontryagin difference:
        # feasible_region = surface - object_footprint.
        if xy_regions and body_idx in xy_regions:
            region = xy_regions[body_idx]  # HPolyhedron
            x_var = q_vars[q_start_idx + 4]
            y_var = q_vars[q_start_idx + 5]
            xy_vars = np.array([x_var, y_var])
            # AddPointInSetConstraints adds linear constraints Ax <= b.
            region.AddPointInSetConstraints(prog, xy_vars)

    # Add minimum distance constraint (non-penetration).
    # LowerBound ensures objects are at least 1e-5m distance apart.
    ik.AddMinimumDistanceLowerBoundConstraint(1e-5, influence_distance)

    # Set initial guess.
    prog.SetInitialGuess(q_vars, q0)

    # Configure solver.
    options = SolverOptions()
    if solver_name == "snopt":
        solver = SnoptSolver()
        if not solver.available():
            raise ValueError("SNOPT solver not available")
        options.SetOption(solver.id(), "Major feasibility tolerance", 1e-3)
        options.SetOption(solver.id(), "Major optimality tolerance", 1e-3)
        options.SetOption(solver.id(), "Major iterations limit", iteration_limit)
        options.SetOption(solver.id(), "Time limit", time_limit_s)
        options.SetOption(solver.id(), "Timing level", 3)
    elif solver_name == "ipopt":
        solver = IpoptSolver()
        if not solver.available():
            raise ValueError("IPOPT solver not available")
        options.SetOption(solver.id(), "max_iter", iteration_limit)
    else:
        raise ValueError(f"Invalid solver: {solver_name}")

    # Solve with hard timeout enforcement.
    # SNOPT's internal "Time limit" option is unreliable in edge cases (can hang
    # indefinitely). Use SIGALRM as external enforcement with grace period.
    hard_timeout_s = int(time_limit_s) + 60

    def hard_timeout_handler(signum, frame):
        raise TimeoutError(
            f"Projection hard timeout: solver did not respect {time_limit_s}s limit"
        )

    old_handler = signal.signal(signal.SIGALRM, hard_timeout_handler)
    signal.alarm(hard_timeout_s)
    try:
        result = solver.Solve(prog, None, options)
        success = result.is_success()
    except TimeoutError as e:
        console_logger.error(f"Hard timeout triggered: {e}")
        return None, False
    except (SystemExit, RuntimeError) as e:
        console_logger.warning(f"Solver failed with error: {e}")
        return None, False
    finally:
        signal.alarm(0)  # Cancel alarm.
        signal.signal(signal.SIGALRM, old_handler)  # Restore handler.

    if not success:
        solution_result = result.get_solution_result()
        console_logger.warning(f"Projection failed: {solution_result.name}")
        infeasible = result.GetInfeasibleConstraintNames(prog)
        if infeasible:
            console_logger.warning(f"Infeasible constraints: {infeasible}")
        return None, False

    # Apply solution.
    solution = result.GetSolution(q_vars)
    if not np.all(np.isfinite(solution)):
        console_logger.warning("Solver returned non-finite values")
        return None, False

    plant.SetPositions(plant_context, solution)
    return plant_context, True


def cyclopean_get_model_instance_for_body(
    plant: MultibodyPlant, body_idx: BodyIndex
) -> ModelInstanceIndex:
    """Get model instance for a body (workaround for Drake API gap)."""
    body = plant.get_body(body_idx)
    return body.model_instance()


def _get_colliding_object_depths(
    scene: RoomScene, penetration_threshold: float = 1e-5
) -> dict[UniqueID, float]:
    """Return the maximum current penetration depth for each colliding object.

    Uses broadphase collision detection to efficiently find penetrating pairs.
    This is used to reduce DOFs in large scene projection by only making
    colliding objects free bodies.

    Note: Only returns scene object IDs, not room geometry IDs (floor, walls).
    Objects colliding with room geometry are included, but "floor"/"wall" IDs
    are filtered out since room geometry is always welded.

    Args:
        scene: RoomScene to check for collisions.
        penetration_threshold: Minimum penetration depth to consider (meters).

    Returns:
        Mapping from scene object ID to its maximum penetration depth.
    """

    def is_room_geometry_id(object_id: str) -> bool:
        """Check if an object ID refers to room geometry (floor, wall, etc.)."""
        # Room geometry IDs can be:
        # - Simple: "floor", "wall"
        # - Prefixed: "room_geometry::floor", "room_geometry::wall_collision"
        room_geometry_patterns = ("floor", "wall", "room_geometry")
        object_id_lower = object_id.lower()
        return any(pattern in object_id_lower for pattern in room_geometry_patterns)

    collisions = compute_scene_collisions(
        scene=scene,
        penetration_threshold=penetration_threshold,
        floor_penetration_tolerance=0.01,  # 1cm tolerance for floor/surface resting.
        current_furniture_id=None,
    )

    collision_depths: dict[UniqueID, float] = {}
    for collision in collisions:
        # Only add scene object IDs, skip room geometry.
        if not is_room_geometry_id(collision.object_a_id):
            object_id = UniqueID(collision.object_a_id)
            collision_depths[object_id] = max(
                collision_depths.get(object_id, 0.0), collision.penetration_depth
            )
        if not is_room_geometry_id(collision.object_b_id):
            object_id = UniqueID(collision.object_b_id)
            collision_depths[object_id] = max(
                collision_depths.get(object_id, 0.0), collision.penetration_depth
            )

    return collision_depths


def _get_colliding_object_ids(
    scene: RoomScene, penetration_threshold: float = 1e-5
) -> set[UniqueID]:
    """Identify scene objects whose penetration exceeds the threshold."""
    return set(
        _get_colliding_object_depths(scene, penetration_threshold=penetration_threshold)
    )


def _supported_projection_displacement_limit(
    obj: SceneObject,
    *,
    penetration_depth: float,
    influence_distance: float,
) -> float:
    """Bound collision cleanup motion using collision severity and object scale."""
    footprint_short_side = 0.0
    try:
        bounds = obj.compute_world_bounds()
    except (TypeError, ValueError):
        bounds = None
    if bounds is not None:
        dimensions = np.asarray(bounds[1], dtype=float) - np.asarray(
            bounds[0], dtype=float
        )
        finite_xy = dimensions[:2][np.isfinite(dimensions[:2])]
        if finite_xy.size:
            footprint_short_side = max(0.0, float(np.min(np.abs(finite_xy))))

    return max(
        0.05,
        2.0 * max(0.0, float(influence_distance)),
        3.0 * max(0.0, float(penetration_depth)),
        0.5 * footprint_short_side,
    )


def apply_non_penetration_projection(
    scene: RoomScene,
    influence_distance: float = 0.02,
    solver_name: str = "snopt",
    iteration_limit: int = 5000,
    time_limit_s: float = 360.0,
    weld_furniture: bool = False,
    xy_only: bool = True,
    fix_rotation: bool = True,
    large_scene_optimization_threshold: int = 100,
    collision_penetration_threshold_m: float = 0.001,
) -> tuple[RoomScene, bool]:
    """Apply IK-based non-penetration projection to resolve collisions.

    This is Stage 1 of physical feasibility post-processing. Uses Drake's
    InverseKinematics with minimum distance constraints to resolve penetrations.

    For large scenes (above threshold), uses collision pre-check optimization
    to reduce DOFs by only making colliding objects free bodies.

    Args:
        scene: RoomScene to project.
        influence_distance: Collision influence distance in meters.
        solver_name: NLP solver ("snopt" or "ipopt").
        iteration_limit: Max solver iterations.
        time_limit_s: Max solver time in seconds.
        weld_furniture: If True, weld furniture (project manipulands only).
                        If False, all objects are free for optimization.
        xy_only: If True, only allow XY translation (fix Z).
                 If False, allow XYZ translation.
        fix_rotation: If True, fix rotations during projection (default).
                      If False, allow rotation optimization.
        large_scene_optimization_threshold: When scene has more objects than
            this threshold, only colliding objects are made free in IK.
            Set to 0 to always use optimization, very high to disable.
        collision_penetration_threshold_m: Minimum penetration depth (meters)
            to consider an object as colliding. Objects with penetration below
            this are treated as surface contacts and excluded from optimization.

    Returns:
        Tuple of (projected_scene, success_flag).
        On failure: returns (original_scene, False).
    """
    start_time = time.time()
    total_objects = len(scene.objects)
    console_logger.info(
        f"Starting non-penetration projection (weld_furniture={weld_furniture}, "
        f"xy_only={xy_only}, fix_rotation={fix_rotation}, objects={total_objects})"
    )

    # Final manipuland cleanup must never move unrelated, stable objects. Besides
    # reducing DOFs, this preserves the support assignment already established by
    # each per-furniture workflow. Furniture projection retains its historical
    # threshold because all furniture may need to cooperate to resolve a layout.
    free_object_ids: list[UniqueID] | None = None
    collision_depths: dict[UniqueID, float] = {}
    if weld_furniture or total_objects > large_scene_optimization_threshold:
        collision_depths = _get_colliding_object_depths(
            scene, penetration_threshold=collision_penetration_threshold_m
        )
        colliding_ids = set(collision_depths)

        if not colliding_ids:
            elapsed = time.time() - start_time
            console_logger.info(
                f"No collisions detected, skipping projection ({elapsed:.2f}s)"
            )
            return scene, True

        if weld_furniture:
            free_object_ids = [
                object_id
                for object_id in colliding_ids
                if (
                    (obj := scene.get_object(object_id)) is not None
                    and obj.object_type == ObjectType.MANIPULAND
                )
            ]
            if not free_object_ids:
                console_logger.warning(
                    "Welded projection found collisions but no movable manipulands; "
                    "leaving the scene unchanged"
                )
                return scene, False
        else:
            free_object_ids = list(colliding_ids)

        console_logger.info(
            f"Collision-scoped projection: {len(colliding_ids)}/{total_objects} "
            f"objects colliding (DOF: {total_objects * 7} -> "
            f"{len(free_object_ids) * 7})"
        )

    pre_projection_transforms = {
        object_id: RigidTransform(
            R=obj.transform.rotation(), p=obj.transform.translation()
        )
        for object_id, obj in scene.objects.items()
    }
    supported_displacement_limits = (
        {
            object_id: _supported_projection_displacement_limit(
                obj,
                penetration_depth=collision_depths.get(object_id, 0.0),
                influence_distance=influence_distance,
            )
            for object_id, obj in scene.objects.items()
            if weld_furniture and object_id in collision_depths
        }
        if weld_furniture
        else {}
    )

    try:
        # Create Drake plant for IK.
        builder = DiagramBuilder()
        plant, scene_graph, object_indices, composite_info = _create_drake_plant_for_ik(
            scene=scene,
            builder=builder,
            weld_furniture=weld_furniture,
            time_step=0.0,
            free_objects=free_object_ids,
        )

        if not object_indices:
            console_logger.warning("No free bodies for projection. Skipping.")
            return scene, True

        xy_regions: dict[BodyIndex, HPolyhedron] = {}
        fixed_z_bodies: set[BodyIndex] = set()
        if weld_furniture:
            for obj_id, (_, body_idx) in object_indices.items():
                obj = scene.get_object(obj_id)
                if obj is None or obj.object_type != ObjectType.MANIPULAND:
                    continue
                region = _object_support_feasible_region(scene, obj)
                if region is not None:
                    xy_regions[body_idx] = region
                    # A valid support assignment is a hard vertical invariant:
                    # resolving a lateral object-object collision by moving the
                    # object through its support surface creates a guaranteed fall
                    # during simulation. Keep unsupported bodies free in Z.
                    fixed_z_bodies.add(body_idx)

        # Solve using shared utility.
        plant_context, success = solve_non_penetration_ik(
            builder=builder,
            plant=plant,
            scene_graph=scene_graph,
            influence_distance=influence_distance,
            fix_rotation=fix_rotation,
            fix_z=xy_only,
            solver_name=solver_name,
            iteration_limit=iteration_limit,
            time_limit_s=time_limit_s,
            xy_regions=xy_regions or None,
            fixed_z_bodies=fixed_z_bodies or None,
        )

        if success and plant_context is not None:
            # Update scene poses from plant.
            _update_scene_from_plant(
                scene=scene,
                plant=plant,
                plant_context=plant_context,
                object_indices=object_indices,
                composite_info=composite_info,
            )
            anomalous_moves: list[tuple[UniqueID, float, float, float]] = []
            for object_id, displacement_limit in supported_displacement_limits.items():
                obj = scene.get_object(object_id)
                initial_transform = pre_projection_transforms.get(object_id)
                if obj is None or initial_transform is None:
                    continue
                displacement = float(
                    np.linalg.norm(
                        obj.transform.translation() - initial_transform.translation()
                    )
                )
                if not np.isfinite(displacement) or displacement > displacement_limit:
                    anomalous_moves.append(
                        (
                            object_id,
                            collision_depths.get(object_id, 0.0),
                            displacement,
                            displacement_limit,
                        )
                    )
            if anomalous_moves:
                for object_id, penetration, displacement, limit in anomalous_moves:
                    console_logger.warning(
                        "Rejecting supported-object projection for %s: initial "
                        "penetration=%.4fm, displacement=%.4fm, limit=%.4fm",
                        object_id,
                        penetration,
                        displacement,
                        limit,
                    )
                for object_id, transform in pre_projection_transforms.items():
                    obj = scene.get_object(object_id)
                    if obj is not None:
                        obj.transform = transform
                console_logger.warning(
                    "Rolled back supported-object projection atomically after "
                    "%d anomalous move(s)",
                    len(anomalous_moves),
                )
                return scene, False
            elapsed = time.time() - start_time
            console_logger.info(f"Projection succeeded in {elapsed:.2f}s")
        else:
            elapsed = time.time() - start_time
            console_logger.warning(f"Projection failed after {elapsed:.2f}s")

        return scene, success

    except Exception as e:
        for object_id, transform in pre_projection_transforms.items():
            obj = scene.get_object(object_id)
            if obj is not None:
                obj.transform = transform
        console_logger.error(f"Projection failed with exception: {e}")
        return scene, False


def _is_plausible_floor_fallback_penetration(
    obj: SceneObject, penetration_m: float
) -> bool:
    """Return whether a reported floor penetration fits the object's scale.

    Drake collision proxies occasionally report a floor penetration larger than
    the rendered furniture itself. Lifting by that value puts an otherwise
    correct object in mid-air, where the simulation then topples and removes
    it. Keep the fallback conservative when a rendered bounding box is
    available; objects without bounds retain the historical fallback behavior.
    """
    try:
        world_bounds = obj.compute_world_bounds()
    except (TypeError, ValueError):
        world_bounds = None
    if world_bounds is None:
        return True

    world_min, world_max = world_bounds
    vertical_extent_m = float(world_max[2] - world_min[2])
    if not np.isfinite(vertical_extent_m) or vertical_extent_m <= 0.0:
        return True
    return penetration_m <= vertical_extent_m


def _apply_floor_penetration_fallback(
    scene: RoomScene, margin_m: float = 0.001
) -> tuple[RoomScene, int]:
    """Lift furniture above floor when NLP projection fails.

    Uses Drake's point-pair penetration query to find floor penetrations and
    lifts each penetrating furniture piece by the exact penetration
    depth plus a small margin.

    Args:
        scene: RoomScene with potentially penetrating furniture.
        margin_m: Safety margin above floor (default 1mm).

    Returns:
        Tuple of (updated scene, number of objects lifted).
    """
    try:
        builder = DiagramBuilder()
        _, scene_graph = create_drake_plant_and_scene_graph_from_scene(
            scene=scene, builder=builder, weld_furniture=False
        )
        diagram = builder.Build()
        context = diagram.CreateDefaultContext()
        scene_graph_context = scene_graph.GetMyContextFromRoot(context)
        query_object: QueryObject = scene_graph.get_query_output_port().Eval(
            scene_graph_context
        )
        all_pairs = query_object.ComputePointPairPenetration()
    except Exception as exc:
        console_logger.warning(
            "floor_penetration_query_failed scene=%s query=point_pair_penetration "
            "error_type=%s error=%s; preserving the original scene",
            getattr(scene, "room_id", "unknown"),
            type(exc).__name__,
            exc,
        )
        return scene, 0

    # Track max penetration per furniture piece.
    furniture_penetrations: dict[UniqueID, float] = {}
    inspector = query_object.inspector()

    for pair in all_pairs:
        # Get object names from geometry IDs.
        name_a = inspector.GetName(pair.id_A)
        name_b = inspector.GetName(pair.id_B)

        # Check if this is a floor collision (floor may be named "floor" or "ground").
        name_a_lower = name_a.lower()
        name_b_lower = name_b.lower()
        is_floor_a = "floor" in name_a_lower or "ground" in name_a_lower
        is_floor_b = "floor" in name_b_lower or "ground" in name_b_lower

        if not (is_floor_a or is_floor_b):
            continue  # Not a floor collision.

        # Get the non-floor object name.
        other_name = name_b if is_floor_a else name_a
        penetration_depth = max(0.0, float(pair.depth))

        # Find furniture ID from geometry name.
        for obj in scene.objects.values():
            if obj.object_type != ObjectType.FURNITURE:
                continue
            # Match by checking if object ID is in the geometry name.
            obj_id_str = str(obj.object_id)
            if (
                obj_id_str in other_name
                or obj.name.lower().replace(" ", "_") in other_name.lower()
            ):
                furn_id = obj.object_id
                # Track max penetration for this furniture.
                current_max = furniture_penetrations.get(furn_id, 0.0)
                furniture_penetrations[furn_id] = max(current_max, penetration_depth)
                break

    # Lift each penetrating furniture piece.
    lifted_count = 0
    for furn_id, penetration in furniture_penetrations.items():
        if penetration > 0:
            lift_amount = penetration + margin_m
            obj = scene.get_object(furn_id)
            if obj is None:
                continue
            if not _is_plausible_floor_fallback_penetration(obj, penetration):
                world_bounds = obj.compute_world_bounds()
                vertical_extent_m = float(world_bounds[1][2] - world_bounds[0][2])
                console_logger.warning(
                    "Skipping floor fallback for %s (%s): reported floor "
                    "penetration %.3fm exceeds object vertical extent %.3fm; "
                    "collision proxy is not credible",
                    obj.name,
                    obj.object_id,
                    penetration,
                    vertical_extent_m,
                )
                continue

            # Create new transform with lifted Z.
            old_transform = obj.transform
            new_translation = old_transform.translation().copy()
            new_translation[2] += lift_amount

            new_transform = RigidTransform(old_transform.rotation(), new_translation)
            obj.transform = new_transform

            console_logger.info(
                f"Floor fallback: lifted {obj.name} ({obj.object_id}) by "
                f"{lift_amount*1000:.1f}mm (penetration was {penetration*1000:.1f}mm)"
            )
            lifted_count += 1

    return scene, lifted_count


def get_object_xy_footprint(
    mesh: "trimesh.Trimesh", rotation: RotationMatrix
) -> VPolytope:
    """Get object's 2D XY footprint after applying rotation.

    Computes the convex hull of the object's mesh vertices projected onto
    the XY plane, after applying the given rotation. Used with Pontryagin
    difference to compute feasible placement regions.

    Args:
        mesh: Object's trimesh mesh.
        rotation: Fixed rotation to apply.

    Returns:
        VPolytope representing the 2D footprint of the rotated object.

    Raises:
        ValueError: If mesh has no vertices.
    """
    if mesh.vertices.shape[0] == 0:
        raise ValueError("Cannot compute footprint for mesh with no vertices")

    # Apply rotation to mesh vertices.
    rotated_vertices: np.ndarray = (rotation.matrix() @ mesh.vertices.T).T

    # Project to XY and compute convex hull.
    xy_vertices = rotated_vertices[:, :2]

    hull, processed_vertices = safe_convex_hull_2d(xy_vertices)
    if hull is None:
        # Degenerate hull (collinear vertices, e.g., very thin knife).
        # Fall back to AABB of XY vertices.
        lb = xy_vertices.min(axis=0)
        ub = xy_vertices.max(axis=0)
        # Ensure non-zero dimensions (add small epsilon if needed).
        epsilon = 1e-4
        if ub[0] - lb[0] < epsilon:
            lb[0] -= epsilon / 2
            ub[0] += epsilon / 2
        if ub[1] - lb[1] < epsilon:
            lb[1] -= epsilon / 2
            ub[1] += epsilon / 2
        return VPolytope.MakeBox(lb=lb, ub=ub)

    hull_vertices = processed_vertices[hull.vertices]
    # VPolytope expects 2xN array (dim x num_vertices).
    return VPolytope(vertices=hull_vertices.T)


def _find_support_surface(scene: RoomScene, surface_id: UniqueID):
    """Return the support surface with ``surface_id``, if it still exists."""
    for owner in scene.objects.values():
        for surface in owner.support_surfaces:
            if str(surface.surface_id) == str(surface_id):
                return surface
    return None


def _object_support_feasible_region(
    scene: RoomScene, obj: SceneObject
) -> HPolyhedron | None:
    """Build the world-XY region that keeps an object on its assigned surface."""
    placement = obj.placement_info
    if placement is None:
        return None
    surface = _find_support_surface(scene, placement.parent_surface_id)
    if surface is None:
        console_logger.warning(
            "Cannot constrain %s to missing support surface %s",
            obj.object_id,
            placement.parent_surface_id,
        )
        return None

    surface_hpoly = HPolyhedron(vpoly=surface.get_xy_convex_hull())
    obj_mesh = None
    geometry_path = obj.geometry_path
    if geometry_path is None and obj.metadata.get("composite_type") == "stack":
        members = obj.metadata.get("member_assets", [])
        if members:
            geometry_path = members[0].get("geometry_path")
    elif (
        geometry_path is None
        and obj.metadata.get("composite_type") == "filled_container"
    ):
        container = obj.metadata.get("container_asset") or {}
        geometry_path = container.get("geometry_path")

    if geometry_path is not None:
        try:
            obj_mesh = trimesh.load(geometry_path, force="mesh")
        except Exception as exc:
            console_logger.warning(
                "Failed to load support footprint for %s: %s",
                obj.object_id,
                exc,
            )

    if obj_mesh is None:
        # Constraining the body origin is less exact than the footprint, but still
        # prevents an unsupported lateral jump during global collision cleanup.
        return surface_hpoly

    if obj.scale_factor != 1.0:
        obj_mesh.vertices *= obj.scale_factor
    footprint = HPolyhedron(
        vpoly=get_object_xy_footprint(obj_mesh, obj.transform.rotation())
    )
    try:
        region = surface_hpoly.PontryaginDifference(footprint)
        if not region.IsEmpty():
            return region
    except Exception as exc:
        console_logger.warning(
            "Failed to compute support region for %s: %s", obj.object_id, exc
        )
    return surface_hpoly


def apply_surface_projection(
    scene: RoomScene,
    surface: "SupportSurface",
    object_ids: list[UniqueID],
    influence_distance: float = 0.02,
    solver_name: str = "snopt",
    iteration_limit: int = 5000,
    time_limit_s: float = 120.0,
) -> tuple[RoomScene, bool, list[UniqueID], float]:
    """Project specified objects on a surface to resolve penetrations.

    Solves for XY translations only - all rotations (roll, pitch, yaw) are
    fixed. Objects are constrained to stay within the surface boundary using
    Pontryagin difference (surface hull - object footprint).

    Uses adaptive scope for performance optimization:
    - For furniture surfaces: includes only parent furniture + its manipulands
    - For floor surfaces: includes all furniture + floor manipulands
    This prevents loading the entire scene into Drake when only a subset of
    objects are relevant for collision checking.

    Args:
        scene: RoomScene to project.
        surface: SupportSurface defining boundary for projection.
        object_ids: Objects to project. Can be 1+ objects. These are the
            movable objects (free bodies in IK).
        influence_distance: Collision influence distance.
        solver_name: NLP solver ("snopt" or "ipopt").
        iteration_limit: Max solver iterations.
        time_limit_s: Max solver time.

    Returns:
        Tuple of (projected_scene, success, moved_object_ids, max_displacement).
    """
    start_time = time.time()
    console_logger.debug(
        f"Starting surface projection for {len(object_ids)} objects on "
        f"surface {surface.surface_id}"
    )

    if not object_ids:
        console_logger.debug("No objects to project.")
        return scene, True, [], 0.0

    # Fail early if any object is a pile - piles have complex footprints that
    # cannot be accurately represented by a single member's mesh.
    for obj_id in object_ids:
        obj = scene.get_object(obj_id)
        if obj is not None and obj.metadata.get("composite_type") == "pile":
            raise ValueError(
                f"Cannot resolve penetrations for pile object {obj_id}. "
                "Piles have scattered members whose combined footprint cannot be "
                "accurately computed. Instead, move other objects or recreate "
                "the pile at a different location."
            )

    # Record original positions for displacement tracking.
    original_positions: dict[UniqueID, np.ndarray] = {}
    for obj_id in object_ids:
        obj = scene.get_object(obj_id)
        if obj is not None:
            original_positions[obj_id] = obj.transform.translation().copy()

    # Get surface boundary as HPolyhedron.
    surface_vpoly = surface.get_xy_convex_hull()
    surface_hpoly = HPolyhedron(vpoly=surface_vpoly)

    # Build per-object feasible regions using Pontryagin difference.
    # This accounts for object footprint - long objects (knives) can be
    # placed closer to edges when oriented parallel.
    object_feasible_regions: dict[UniqueID, HPolyhedron] = {}
    for obj_id in object_ids:
        obj = scene.get_object(obj_id)
        if obj is None:
            continue

        # Get mesh for the object. For composites, use the reference member's mesh.
        obj_mesh = None
        if obj.geometry_path is not None:
            try:
                obj_mesh = trimesh.load(obj.geometry_path, force="mesh")
            except Exception as e:
                console_logger.warning(f"Failed to load mesh for {obj_id}: {e}")

        composite_type = obj.metadata.get("composite_type")

        if obj_mesh is None and composite_type == "stack":
            # Stack: use bottom member's mesh.
            member_assets = obj.metadata.get("member_assets", [])
            if member_assets:
                bottom_geometry = member_assets[0].get("geometry_path")
                if bottom_geometry:
                    try:
                        obj_mesh = trimesh.load(bottom_geometry, force="mesh")
                    except Exception as e:
                        console_logger.warning(f"Failed to load stack bottom mesh: {e}")

        if obj_mesh is None and composite_type == "filled_container":
            # Filled container: use container's mesh.
            container_asset = obj.metadata.get("container_asset")
            if container_asset:
                container_geometry = container_asset.get("geometry_path")
                if container_geometry:
                    try:
                        obj_mesh = trimesh.load(container_geometry, force="mesh")
                    except Exception as e:
                        console_logger.warning(f"Failed to load container mesh: {e}")

        if obj_mesh is None:
            # No mesh available - use surface bounds as fallback.
            console_logger.warning(
                f"No mesh for {obj_id}, using surface bounds as feasible region"
            )
            object_feasible_regions[obj_id] = surface_hpoly
            continue

        # Apply scale_factor to mesh for correct footprint dimensions.
        if obj.scale_factor != 1.0:
            obj_mesh.vertices *= obj.scale_factor

        # Get object's current rotation (fixed during projection).
        obj_rotation = obj.transform.rotation()

        # Compute object's XY footprint after rotation.
        object_vpoly = get_object_xy_footprint(obj_mesh, obj_rotation)
        object_hpoly = HPolyhedron(vpoly=object_vpoly)

        # Pontryagin difference: set of all center positions where entire
        # object footprint stays within surface boundary.
        try:
            feasible_region = surface_hpoly.PontryaginDifference(object_hpoly)
            if feasible_region.IsEmpty():
                console_logger.warning(
                    f"Object {obj_id} is too large for surface {surface.surface_id}"
                )
                # Use surface bounds as fallback.
                feasible_region = surface_hpoly
            object_feasible_regions[obj_id] = feasible_region
        except Exception as e:
            console_logger.warning(
                f"Failed to compute feasible region for {obj_id}: {e}. "
                "Using surface bounds."
            )
            object_feasible_regions[obj_id] = surface_hpoly

    # Compute adaptive scope based on surface type.
    # For furniture surfaces: include parent furniture + its manipulands.
    # For floor surfaces: include all furniture + floor manipulands.
    owner_id, is_floor = _find_surface_owner(scene=scene, surface_id=surface.surface_id)

    include_objects: list[UniqueID] = []
    if is_floor:
        # Floor surface: include all furniture as obstacles.
        for obj in scene.objects.values():
            if obj.object_type == ObjectType.FURNITURE:
                include_objects.append(obj.object_id)
        # Include all manipulands on the floor.
        if owner_id is not None:
            floor_obj = scene.get_object(owner_id)
            if floor_obj is not None:
                for surf in floor_obj.support_surfaces:
                    for manip in scene.get_objects_on_surface(surf.surface_id):
                        if manip.object_id not in include_objects:
                            include_objects.append(manip.object_id)
        console_logger.debug(
            f"Floor surface mode: including {len(include_objects)} objects "
            f"(all furniture + floor manipulands)"
        )
    elif owner_id is not None:
        # Furniture surface: include parent furniture + its manipulands.
        include_objects.append(owner_id)
        furniture = scene.get_object(owner_id)
        if furniture is not None:
            for surf in furniture.support_surfaces:
                for manip in scene.get_objects_on_surface(surf.surface_id):
                    if manip.object_id not in include_objects:
                        include_objects.append(manip.object_id)
        console_logger.debug(
            f"Furniture surface mode: including {len(include_objects)} objects "
            f"(furniture {owner_id} + its manipulands)"
        )
    else:
        # Unknown surface owner - fall back to all objects.
        console_logger.warning(
            f"Could not find owner for surface {surface.surface_id}, "
            "using all objects for collision checking"
        )
        include_objects = None  # type: ignore[assignment]

    try:
        # Create Drake plant using shared helper (handles composite objects properly).
        builder = DiagramBuilder()
        plant, scene_graph, object_indices, composite_info = _create_drake_plant_for_ik(
            scene=scene,
            builder=builder,
            weld_furniture=True,
            time_step=0.0,
            free_objects=list(object_ids),
            include_objects=include_objects,
        )

        if not object_indices:
            console_logger.warning("No free bodies found for projection.")
            return scene, True, [], 0.0

        # Filter object_indices to only include requested object_ids.
        filtered_indices = {k: v for k, v in object_indices.items() if k in object_ids}

        if not filtered_indices:
            console_logger.warning("No matching free bodies found for projection.")
            return scene, True, [], 0.0

        # Build xy_regions mapping (BodyIndex -> HPolyhedron).
        xy_regions: dict[BodyIndex, HPolyhedron] = {}
        for obj_id, (_, body_idx) in filtered_indices.items():
            if obj_id in object_feasible_regions:
                xy_regions[body_idx] = object_feasible_regions[obj_id]

        # Solve using shared utility.
        plant_context, success = solve_non_penetration_ik(
            builder=builder,
            plant=plant,
            scene_graph=scene_graph,
            influence_distance=influence_distance,
            fix_rotation=True,
            fix_z=True,
            solver_name=solver_name,
            iteration_limit=iteration_limit,
            time_limit_s=time_limit_s,
            xy_regions=xy_regions,
        )

        if not success or plant_context is None:
            elapsed = time.time() - start_time
            console_logger.warning(f"Surface projection failed after {elapsed:.2f}s")
            return scene, False, [], 0.0

        # Update scene using shared helper (handles composite member transforms).
        # Filter to exclude obstacle composites - only update movable objects.
        # (composite_info contains all composites in plant, including welded obstacles)
        filtered_composite_info = {
            k: v for k, v in composite_info.items() if k in object_ids
        }
        _update_scene_from_plant(
            scene=scene,
            plant=plant,
            plant_context=plant_context,
            object_indices=filtered_indices,
            composite_info=filtered_composite_info,
            operation_name="Surface projection",
        )

        # Compute displacements for moved objects.
        moved_ids: list[UniqueID] = []
        max_displacement = 0.0
        for obj_id in object_ids:
            obj = scene.get_object(obj_id)
            if obj is None:
                continue

            if obj_id in original_positions:
                old_pos = original_positions[obj_id]
                new_pos = obj.transform.translation()
                displacement = np.linalg.norm(new_pos - old_pos)

                if displacement > 1e-6:  # Non-trivial movement.
                    moved_ids.append(obj_id)
                    max_displacement = max(max_displacement, displacement)
                    console_logger.info(f"  {obj_id}: moved {displacement:.4f}m")

        # Update placement_info for moved objects (world → surface conversion).
        # This keeps surface-relative coordinates in sync after physics resolution.
        for obj_id in moved_ids:
            obj = scene.get_object(obj_id)
            if obj is None or obj.placement_info is None:
                continue

            # Convert new world position back to surface-relative coordinates.
            new_pos_2d, new_rot_2d = surface.from_world_pose(obj.transform)
            obj.placement_info.position_2d = new_pos_2d.copy()
            # Note: rotation_2d should be unchanged since fix_rotation=True,
            # but update it anyway for correctness and future-proofing.
            obj.placement_info.rotation_2d = new_rot_2d

        elapsed = time.time() - start_time
        console_logger.info(
            f"Surface projection succeeded in {elapsed:.2f}s. "
            f"Moved {len(moved_ids)} objects, max displacement: {max_displacement:.4f}m"
        )

        return scene, True, moved_ids, max_displacement

    except Exception as e:
        console_logger.error(f"Surface projection failed with exception: {e}")
        return scene, False, [], 0.0


def apply_forward_simulation(
    scene: RoomScene,
    simulation_time_s: float = 5.0,
    time_step_s: float = 1e-3,
    timeout_s: float = 300.0,
    weld_furniture: bool = True,
    output_html_path: Path | None = None,
    remove_fallen_furniture: bool = False,
    fallen_tilt_threshold_degrees: float = 45.0,
    remove_fallen_manipulands: bool = False,
    fallen_manipuland_floor_z: float = -0.5,
    fallen_manipuland_near_floor_z: float = 0.02,
    fallen_manipuland_z_displacement: float = 0.3,
) -> tuple[RoomScene, list[UniqueID]]:
    """Apply forward simulation to settle scene to static equilibrium.

    This is Stage 2 of physical feasibility post-processing. Objects settle
    under gravity after projection resolves penetrations.

    Always allows full 6DOF motion (xyz + roll/pitch/yaw) for free bodies.

    Args:
        scene: RoomScene to simulate.
        simulation_time_s: Simulation duration in seconds.
        time_step_s: Physics time step in seconds.
        timeout_s: Maximum wall-clock time for simulation.
        weld_furniture: If True, weld furniture (simulate manipulands only).
                        If False, all objects are simulated.
        output_html_path: If provided, save meshcat visualization to this HTML file.
        remove_fallen_furniture: If True, remove furniture objects that fall over
            (tilt beyond threshold) during simulation.
        fallen_tilt_threshold_degrees: Tilt angle threshold in degrees to consider
            furniture "fallen" (only used if remove_fallen_furniture is True).
        remove_fallen_manipulands: If True, remove manipuland objects that fall
            off furniture surfaces during simulation.
        fallen_manipuland_floor_z: Absolute Z threshold for floor penetration.
            Manipulands below this Z are removed (physics bug).
        fallen_manipuland_near_floor_z: Object bottom below this Z is considered
            on floor (used with z_displacement check).
        fallen_manipuland_z_displacement: Z drop threshold. Manipulands that drop
            more than this AND end up on floor are removed.

    Returns:
        Tuple of (scene, removed_ids) where:
        - scene: Scene with updated poses after simulation
        - removed_ids: List of UniqueIDs of fallen objects (furniture and manipulands)
    """
    start_time = time.time()
    console_logger.info(
        f"Starting forward simulation (weld_furniture={weld_furniture}, "
        f"time={simulation_time_s}s)"
    )

    meshcat = None
    visualizer = None

    try:
        # Create Drake plant for simulation.
        builder = DiagramBuilder()
        plant, scene_graph, object_indices, composite_info = _create_drake_plant_for_ik(
            scene=scene,
            builder=builder,
            weld_furniture=weld_furniture,
            time_step=time_step_s,
        )

        if not object_indices:
            console_logger.warning("No free bodies for simulation. Skipping.")
            return scene, []

        # Store pre-simulation Z positions for fallen manipuland detection.
        pre_sim_z: dict[UniqueID, float] = {}
        if remove_fallen_manipulands:
            for obj in scene.objects.values():
                if obj.object_type == ObjectType.MANIPULAND:
                    pre_sim_z[obj.object_id] = obj.transform.translation()[2]

        # Set up visualization if HTML output is requested.
        if output_html_path is not None:
            meshcat = StartMeshcat()
            visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

        # Build diagram.
        diagram = builder.Build()
        context = diagram.CreateDefaultContext()
        plant_context = plant.GetMyContextFromRoot(context)

        # Set up timeout monitor.
        sim_start = time.time()

        def timeout_monitor(_: Context) -> EventStatus:
            if time.time() - sim_start > timeout_s:
                return EventStatus.ReachedTermination(None, "timeout")
            return EventStatus.DidNothing()

        # Track upright source poses so a shared bad collision proxy cannot
        # erase every repeated instance of an otherwise valid asset.
        pre_simulation_transforms = {
            obj.object_id: RigidTransform(
                R=obj.transform.rotation(), p=obj.transform.translation()
            )
            for obj in scene.objects.values()
            if obj.object_type == ObjectType.FURNITURE
        }

        # Run simulation.
        simulator = Simulator(diagram, context)
        simulator.set_monitor(timeout_monitor)

        # Start recording if visualizing.
        if visualizer is not None:
            visualizer.StartRecording()

        simulator.AdvanceTo(simulation_time_s)

        # Stop recording and export HTML if visualizing.
        if visualizer is not None and meshcat is not None:
            visualizer.StopRecording()
            visualizer.PublishRecording()

            html = meshcat.StaticHtml()
            output_html_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_html_path, "w") as f:
                f.write(html)
            console_logger.info(f"Saved simulation HTML to {output_html_path}")

        # Update scene poses from plant.
        _update_scene_from_plant(
            scene=scene,
            plant=plant,
            plant_context=plant_context,
            object_indices=object_indices,
            composite_info=composite_info,
            operation_name="Simulation",
        )

        restored_through_floor_ids = _restore_furniture_that_fell_through_floor(
            scene,
            pre_simulation_transforms,
        )

        # Detect and remove physically unstable furniture if enabled.
        removed_ids: list[UniqueID] = []
        if remove_fallen_furniture:
            # Only check furniture objects (not manipulands, walls, etc.).
            furniture_ids = [
                obj.object_id
                for obj in scene.objects.values()
                if obj.object_type == ObjectType.FURNITURE
            ]
            for obj_id in furniture_ids:
                obj = scene.get_object(obj_id)
                if obj is None:
                    continue
                pre_simulation_transform = pre_simulation_transforms.get(obj_id)
                if pre_simulation_transform is None:
                    continue
                instability_reason = _furniture_simulation_instability_reason(
                    obj,
                    pre_simulation_transform,
                    fallen_tilt_threshold_degrees,
                )
                if instability_reason is not None:
                    console_logger.warning(
                        f"Removing physically unstable furniture {obj_id}: "
                        f"{instability_reason}"
                    )
                    removed_ids.append(obj_id)

            restored_ids = _restore_collectively_unstable_instances(
                scene,
                pre_simulation_transforms,
                removed_ids,
                fallen_tilt_threshold_degrees,
                restored_through_floor_ids,
            )
            if restored_ids:
                restored_set = set(restored_ids)
                removed_ids = [
                    obj_id for obj_id in removed_ids if obj_id not in restored_set
                ]

            for obj_id in removed_ids:
                scene.remove_object(obj_id)

            if removed_ids:
                console_logger.info(
                    f"Removed {len(removed_ids)} fallen furniture item(s): {removed_ids}"
                )

        # Detect and remove fallen manipulands if enabled.
        if remove_fallen_manipulands:
            furniture_removed_count = len(removed_ids)
            manipuland_ids = [
                obj.object_id
                for obj in scene.objects.values()
                if obj.object_type == ObjectType.MANIPULAND
            ]
            for obj_id in manipuland_ids:
                obj = scene.get_object(obj_id)
                if obj is None:
                    continue

                current_z = obj.transform.translation()[2]

                # Check 1: Floor penetration (physics bug).
                if current_z < fallen_manipuland_floor_z:
                    console_logger.warning(
                        f"Removing fallen manipuland {obj_id}: "
                        f"z={current_z:.4f}m < floor_z={fallen_manipuland_floor_z}m"
                    )
                    scene.remove_object(obj_id)
                    removed_ids.append(obj_id)
                    continue

                # Check 2: On floor + significant Z drop (fell off furniture).
                # Use world-frame bounds (handles rotation).
                world_bounds = obj.compute_world_bounds()
                if world_bounds is None:
                    continue
                world_bbox_min, _ = world_bounds
                bottom_z = world_bbox_min[2]
                is_on_floor = bottom_z < fallen_manipuland_near_floor_z

                if obj_id in pre_sim_z and is_on_floor:
                    z_delta = current_z - pre_sim_z[obj_id]
                    if z_delta < -fallen_manipuland_z_displacement:
                        console_logger.warning(
                            f"Removing fallen manipuland {obj_id}: "
                            f"bottom_z={bottom_z:.4f}m, z_delta={z_delta:.4f}m"
                        )
                        scene.remove_object(obj_id)
                        removed_ids.append(obj_id)

            fallen_manipuland_count = len(removed_ids) - furniture_removed_count
            if fallen_manipuland_count > 0:
                console_logger.info(
                    f"Removed {fallen_manipuland_count} fallen manipuland(s)"
                )

        elapsed = time.time() - start_time
        console_logger.info(f"Simulation completed in {elapsed:.2f}s")

        return scene, removed_ids

    except Exception as e:
        console_logger.error(f"Simulation failed with exception: {e}")
        return scene, []

    finally:
        # Explicitly delete Meshcat on the main thread to avoid threading issues.
        # Drake's Meshcat destructor asserts it must be called from the thread
        # that created it. Without explicit deletion, Python's GC might destroy
        # the Meshcat from a ThreadPoolExecutor worker thread, causing a crash.
        if meshcat is not None:
            del meshcat


def apply_physical_feasibility_postprocessing(
    scene: RoomScene,
    weld_furniture: bool,
    projection_enabled: bool = True,
    projection_influence_distance: float = 0.02,
    projection_solver_name: str = "snopt",
    projection_iteration_limit: int = 5000,
    projection_time_limit_s: float = 360.0,
    projection_xy_only: bool = True,
    projection_fix_rotation: bool = True,
    large_scene_optimization_threshold: int = 100,
    collision_penetration_threshold_m: float = 0.001,
    simulation_enabled: bool = True,
    simulation_time_s: float = 5.0,
    simulation_time_step_s: float = 1e-3,
    simulation_timeout_s: float = 300.0,
    simulation_html_path: Path | None = None,
    remove_fallen_furniture: bool = False,
    fallen_tilt_threshold_degrees: float = 45.0,
    remove_fallen_manipulands: bool = False,
    fallen_manipuland_floor_z: float = -0.5,
    fallen_manipuland_near_floor_z: float = 0.02,
    fallen_manipuland_z_displacement: float = 0.3,
) -> tuple[RoomScene, bool, list[UniqueID]]:
    """Apply complete physical feasibility post-processing pipeline.

    Combines projection (Stage 1) and simulation (Stage 2) with graceful
    error handling. On any failure, returns original scene unchanged.

    Args:
        scene: RoomScene to process.
        weld_furniture: If True, weld furniture (process manipulands only).
        projection_enabled: Whether to run projection stage.
        projection_influence_distance: Collision influence distance.
        projection_solver_name: NLP solver name.
        projection_iteration_limit: Max solver iterations.
        projection_time_limit_s: Max solver time in seconds.
        projection_xy_only: If True, only optimize XY translation.
        projection_fix_rotation: If True, fix rotations during projection.
        large_scene_optimization_threshold: When scene has more objects than
            this threshold, only colliding objects are made free in IK.
        collision_penetration_threshold_m: Minimum penetration depth (meters)
            to consider an object as colliding for large scene optimization.
        simulation_enabled: Whether to run simulation stage.
        simulation_time_s: Simulation duration.
        simulation_time_step_s: Physics time step.
        simulation_timeout_s: Max wall-clock time for simulation.
        simulation_html_path: If provided, save meshcat visualization to this HTML file.
        remove_fallen_furniture: If True, remove furniture that falls over during
            simulation.
        fallen_tilt_threshold_degrees: Tilt angle threshold for fallen detection.
        remove_fallen_manipulands: If True, remove manipuland objects that fall
            off furniture surfaces during simulation.
        fallen_manipuland_floor_z: Absolute Z threshold for floor penetration.
        fallen_manipuland_near_floor_z: Object bottom below this Z is on floor.
        fallen_manipuland_z_displacement: Z drop threshold for detecting falling.

    Returns:
        Tuple of (processed_scene, projection_success, removed_ids).
    """
    pre_projection_transforms = {
        obj.object_id: RigidTransform(
            R=obj.transform.rotation(), p=obj.transform.translation()
        )
        for obj in scene.objects.values()
        if obj.object_type == ObjectType.FURNITURE
    }

    # Stage 1: Projection.
    if projection_enabled:
        scene, projection_success = apply_non_penetration_projection(
            scene=scene,
            influence_distance=projection_influence_distance,
            solver_name=projection_solver_name,
            iteration_limit=projection_iteration_limit,
            time_limit_s=projection_time_limit_s,
            weld_furniture=weld_furniture,
            xy_only=projection_xy_only,
            fix_rotation=projection_fix_rotation,
            large_scene_optimization_threshold=large_scene_optimization_threshold,
            collision_penetration_threshold_m=collision_penetration_threshold_m,
        )

        if not projection_success and not weld_furniture:
            # Only apply floor fallback when furniture is free to move.
            # When weld_furniture=True, furniture is fixed and can't penetrate floor.
            console_logger.warning(
                "Projection failed, applying floor penetration fallback"
            )
            # Lift furniture above floor to prevent tipping during simulation.
            scene, lifted_count = _apply_floor_penetration_fallback(
                scene=scene, margin_m=0.001
            )
            if lifted_count > 0:
                console_logger.info(
                    f"Floor fallback: lifted {lifted_count} furniture piece(s)"
                )
            else:
                console_logger.warning(
                    "Floor fallback: no floor penetrations found, "
                    "projection may have failed for other reasons"
                )
        elif not projection_success:
            console_logger.warning(
                "Projection failed (furniture welded, skipping floor fallback)"
            )
        elif not weld_furniture:
            _restore_anomalous_furniture_projection(
                scene,
                pre_projection_transforms,
            )

    # Stage 2: Simulation (runs regardless of projection result).
    removed_ids: list[UniqueID] = []
    if simulation_enabled:
        scene, removed_ids = apply_forward_simulation(
            scene=scene,
            simulation_time_s=simulation_time_s,
            time_step_s=simulation_time_step_s,
            timeout_s=simulation_timeout_s,
            weld_furniture=weld_furniture,
            output_html_path=simulation_html_path,
            remove_fallen_furniture=remove_fallen_furniture,
            fallen_tilt_threshold_degrees=fallen_tilt_threshold_degrees,
            remove_fallen_manipulands=remove_fallen_manipulands,
            fallen_manipuland_floor_z=fallen_manipuland_floor_z,
            fallen_manipuland_near_floor_z=fallen_manipuland_near_floor_z,
            fallen_manipuland_z_displacement=fallen_manipuland_z_displacement,
        )

    return scene, projection_success if projection_enabled else True, removed_ids


def apply_per_furniture_postprocessing(
    full_scene: RoomScene,
    furniture_id: UniqueID,
    config: DictConfig,
    simulation_html_path: Path | None = None,
    *,
    return_success: bool = False,
) -> RoomScene | tuple[RoomScene, bool]:
    """Run post-processing for a single furniture piece and its manipulands.

    Creates a subset scene containing only the target furniture, its manipulands,
    and room structure (walls/floor), then runs the full post-processing pipeline.
    The manipuland poses are then merged back into the full scene.

    This enables solving smaller, more tractable subproblems before the final
    combined post-processing pass.

    Args:
        full_scene: The complete scene with all objects.
        furniture_id: ID of the furniture piece to process.
        config: Post-processing configuration with projection and simulation settings.
        simulation_html_path: If provided, save meshcat visualization to this HTML file.

    Returns:
        The full scene with updated manipuland poses.
    """
    from scenesmith.agent_utils.room import SceneObject  # Avoid circular import.

    # Build subset scene: walls + floor + this furniture + its manipulands.
    subset_objects: dict[UniqueID, SceneObject] = {}

    # Add walls and floor.
    for obj in full_scene.objects.values():
        if obj.object_type in [ObjectType.WALL, ObjectType.FLOOR]:
            subset_objects[obj.object_id] = obj

    # Add target furniture.
    furniture = full_scene.get_object(furniture_id)
    if furniture is None:
        console_logger.warning(
            f"Furniture {furniture_id} not found, skipping per-furniture post-processing"
        )
        return (full_scene, True) if return_success else full_scene

    subset_objects[furniture.object_id] = furniture

    # Add manipulands on this furniture's surfaces.
    manipuland_ids: list[UniqueID] = []
    for surface in furniture.support_surfaces:
        for manip in full_scene.get_objects_on_surface(surface.surface_id):
            subset_objects[manip.object_id] = manip
            manipuland_ids.append(manip.object_id)

    # Skip if no manipulands to process.
    if not manipuland_ids:
        console_logger.info(
            f"No manipulands on furniture {furniture_id}, skipping post-processing"
        )
        return (full_scene, True) if return_success else full_scene

    console_logger.info(
        f"Running per-furniture post-processing for {furniture_id} "
        f"with {len(manipuland_ids)} manipuland(s)"
    )
    pose_snapshot = {
        object_id: RigidTransform(
            R=obj.transform.rotation(), p=obj.transform.translation()
        )
        for object_id in [furniture_id, *manipuland_ids]
        if (obj := full_scene.get_object(object_id)) is not None
    }

    # Create subset scene.
    subset_scene = RoomScene(
        room_geometry=full_scene.room_geometry,
        scene_dir=full_scene.scene_dir,
        objects=subset_objects,
        text_description=full_scene.text_description,
    )

    # Run post-processing on subset (furniture welded).
    # Fallen furniture removal not needed: furniture is welded here.
    projection_cfg = config.projection
    simulation_cfg = config.simulation
    processed_subset, success, _ = apply_physical_feasibility_postprocessing(
        scene=subset_scene,
        weld_furniture=True,
        projection_enabled=projection_cfg.enabled,
        projection_influence_distance=projection_cfg.influence_distance,
        projection_solver_name=projection_cfg.solver_name,
        projection_iteration_limit=projection_cfg.iteration_limit,
        projection_time_limit_s=projection_cfg.time_limit_s,
        projection_xy_only=projection_cfg.xy_only,
        projection_fix_rotation=projection_cfg.fix_rotation,
        simulation_enabled=simulation_cfg.enabled,
        simulation_time_s=simulation_cfg.simulation_time_s,
        simulation_time_step_s=simulation_cfg.time_step_s,
        simulation_timeout_s=simulation_cfg.timeout_s,
        simulation_html_path=simulation_html_path,
        remove_fallen_furniture=False,
        remove_fallen_manipulands=simulation_cfg.remove_fallen_manipulands,
        fallen_manipuland_floor_z=simulation_cfg.fallen_manipuland_floor_z,
        fallen_manipuland_near_floor_z=simulation_cfg.fallen_manipuland_near_floor_z,
        fallen_manipuland_z_displacement=simulation_cfg.fallen_manipuland_z_displacement,
    )

    if not success:
        console_logger.warning(
            f"Per-furniture post-processing for {furniture_id} had issues"
        )

    processed_furniture = processed_subset.get_object(furniture_id)
    original_furniture_pose = pose_snapshot[furniture_id]
    furniture_moved = processed_furniture is None or not (
        np.allclose(
            processed_furniture.transform.GetAsMatrix4(),
            original_furniture_pose.GetAsMatrix4(),
            atol=1e-9,
            rtol=0.0,
        )
    )
    if furniture_moved:
        console_logger.error(
            "Welded per-furniture post-processing moved support %s; rolling back "
            "the entire transaction",
            furniture_id,
        )
        success = False

    if not success:
        for object_id, transform in pose_snapshot.items():
            obj = full_scene.get_object(object_id)
            if obj is not None:
                obj.transform = transform
        if return_success:
            return full_scene, False
        return full_scene

    # Even a numerically insignificant solver update must not leak into the
    # support pose shared by the full scene.
    furniture.transform = original_furniture_pose

    # Merge manipuland poses back to full scene.
    for manip_id in manipuland_ids:
        processed_manip = processed_subset.get_object(manip_id)
        if processed_manip:
            full_scene.move_object(
                object_id=manip_id, new_transform=processed_manip.transform
            )
        else:
            # Object was removed during post-processing (e.g., fell off furniture).
            full_scene.remove_object(manip_id)

    if return_success:
        return full_scene, success
    return full_scene
