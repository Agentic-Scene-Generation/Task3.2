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
