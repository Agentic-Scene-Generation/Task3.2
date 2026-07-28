"""Shared physics utilities for manipuland simulation."""

import logging

import trimesh

from scenesmith.agent_utils.room import SceneObject
from scenesmith.utils.mesh_loading import load_object_collision_geometry

console_logger = logging.getLogger(__name__)


def compute_collision_bounds(
    collision_meshes: list[trimesh.Trimesh],
) -> tuple[float, float]:
    """Get z_min and z_max from collision geometry.

    Args:
        collision_meshes: List of collision mesh pieces (convex hulls).

    Returns:
        Tuple of (z_min, z_max) representing collision bounds along Z-axis.

    Raises:
        ValueError: If collision_meshes is empty.
    """
    if not collision_meshes:
        raise ValueError("No collision meshes provided")

    z_min = min(m.vertices[:, 2].min() for m in collision_meshes)
    z_max = max(m.vertices[:, 2].max() for m in collision_meshes)
    return float(z_min), float(z_max)


def load_collision_bounds_for_scene_object(
    obj: SceneObject,
) -> tuple[float, float]:
    """Load collision geometry and compute z-bounds for a SceneObject.

    Uses the SDF as the authoritative collision scale. Runtime rescale tools
    update that SDF transactionally.

    Args:
        obj: SceneObject with sdf_path.

    Returns:
        Tuple of (z_min, z_max) from collision geometry.

    Raises:
        ValueError: If object has no SDF path or no collision geometry.
    """
    if not obj.sdf_path:
        raise ValueError(f"Object {obj.name} has no SDF path")

    collision_meshes = load_object_collision_geometry(obj)
    if not collision_meshes:
        raise ValueError(f"No collision geometry found for {obj.name}")

    return compute_collision_bounds(collision_meshes)
