from pathlib import Path

import numpy as np
import trimesh

from omegaconf import OmegaConf
from pydrake.math import RigidTransform

from scenesmith.agent_utils.room import ObjectType, SceneObject, UniqueID
from scenesmith.furniture_agents.tools import snapping_helpers


class _NoCollisionManager:
    """Exercise the footprint guard without an earlier mesh collision."""

    def __init__(self) -> None:
        self._objs = {}

    def add_object(self, name, mesh, transform=None) -> None:
        self._objs[name] = (mesh, transform)

    def min_distance_other(self, other) -> float:
        return float("inf")


class _PenetratingCollisionManager(_NoCollisionManager):
    """Force the push-out path without relying on a local FCL installation."""

    def min_distance_other(self, other) -> float:
        return -0.1


def _north_wall() -> SceneObject:
    return SceneObject(
        object_id=UniqueID("north_wall"),
        object_type=ObjectType.WALL,
        name="north_wall",
        description="north wall",
        transform=RigidTransform(p=[0.0, 2.2, 1.35]),
        bbox_min=np.array([-2.5, -0.05, -1.35]),
        bbox_max=np.array([2.5, 0.05, 1.35]),
        immutable=True,
    )


def _object(object_id: str, position: tuple[float, float, float]) -> SceneObject:
    return SceneObject(
        object_id=UniqueID(object_id),
        object_type=ObjectType.FURNITURE,
        name=object_id,
        description=object_id,
        transform=RigidTransform(p=np.asarray(position, dtype=float)),
        geometry_path=Path(f"/{object_id}.gltf"),
        sdf_path=Path(f"/{object_id}.sdf"),
    )


def test_axis_snap_stops_chair_front_at_table_footprint(monkeypatch) -> None:
    chair = _object("dining_chair", (0.0, -1.0, 0.25))
    table = _object("dining_table", (0.0, 0.0, 0.75))
    chair_mesh = trimesh.creation.box(extents=(0.5, 0.5, 0.5))
    tabletop_mesh = trimesh.creation.box(extents=(2.0, 1.0, 0.1))

    def collision_geometry(obj):
        return [chair_mesh if obj is chair else tabletop_mesh]

    monkeypatch.setattr(
        snapping_helpers, "load_object_collision_geometry", collision_geometry
    )
    monkeypatch.setattr(
        snapping_helpers.trimesh.collision,
        "CollisionManager",
        _NoCollisionManager,
    )
    cfg = OmegaConf.create(
        {
            "snap_to_object": {
                "iterative_snap_step_m": 0.01,
                "max_snap_distance_m": 2.0,
            }
        }
    )

    movement, distance = snapping_helpers.snap_with_iterative_collision_check(
        obj=chair,
        target=table,
        direction=np.array([0.0, 1.0, 0.0]),
        cfg=cfg,
    )

    np.testing.assert_allclose(movement, [0.0, 0.24, 0.0], atol=1e-8)
    assert abs(distance - 0.24) < 1e-8
    chair_front_y = chair.transform.translation()[1] + movement[1] + 0.25
    assert chair_front_y < -0.5


def test_floor_furniture_wall_snap_never_changes_elevation(monkeypatch) -> None:
    """Wall AABB mid-height must not lift a floor-standing furniture object."""
    wardrobe = _object("wardrobe", (0.0, 1.5, 0.0))
    wall = _north_wall()
    wardrobe_mesh = trimesh.creation.box(extents=(1.0, 0.5, 2.0))
    monkeypatch.setattr(
        snapping_helpers, "load_object_collision_geometry", lambda _: [wardrobe_mesh]
    )
    monkeypatch.setattr(
        snapping_helpers.trimesh.collision,
        "CollisionManager",
        _PenetratingCollisionManager,
    )
    cfg = OmegaConf.create(
        {
            "snap_to_object": {
                "iterative_snap_step_m": 0.01,
                "snap_margin_m": 0.01,
            }
        }
    )

    movement, _ = snapping_helpers.snap_mesh_to_aabb(wardrobe, wall, cfg)

    assert movement[2] == 0.0
    assert abs(movement[1]) > 1e-6


def test_floor_furniture_wall_closest_point_snap_never_changes_elevation(
    monkeypatch,
) -> None:
    """The separated closest-point path must also remain in the floor plane."""
    wardrobe = _object("wardrobe", (0.0, 1.5, 0.0))
    wall = _north_wall()
    wardrobe_mesh = trimesh.creation.box(extents=(1.0, 0.5, 2.0))
    monkeypatch.setattr(
        snapping_helpers, "load_object_collision_geometry", lambda _: [wardrobe_mesh]
    )
    monkeypatch.setattr(
        snapping_helpers.trimesh.collision,
        "CollisionManager",
        _NoCollisionManager,
    )
    cfg = OmegaConf.create(
        {
            "snap_to_object": {
                "iterative_snap_step_m": 0.01,
                "snap_margin_m": 0.01,
            }
        }
    )

    movement, _ = snapping_helpers.snap_mesh_to_aabb(wardrobe, wall, cfg)

    assert movement[2] == 0.0
    assert abs(movement[1]) > 1e-6
