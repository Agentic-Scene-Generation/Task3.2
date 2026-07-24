"""Tests for deterministic seating orientation repair."""

from pathlib import Path

import numpy as np

from pydrake.math import RigidTransform, RollPitchYaw

from scenesmith.agent_utils.room import ObjectType, RoomScene, SceneObject, UniqueID
from scenesmith.agent_utils.seating_orientation_guard import (
    _front_angle_to_target_deg,
    align_seating_to_nearest_surface,
)


def _object(
    object_id: str,
    object_type: ObjectType,
    position: tuple[float, float, float],
    size: tuple[float, float, float],
    *,
    yaw_deg: float = 0.0,
) -> SceneObject:
    half_size = np.asarray(size, dtype=float) / 2.0
    return SceneObject(
        object_id=UniqueID(object_id),
        object_type=object_type,
        name=object_id,
        description=object_id,
        transform=RigidTransform(
            rpy=RollPitchYaw(0.0, 0.0, np.deg2rad(yaw_deg)),
            p=np.asarray(position, dtype=float),
        ),
        bbox_min=-half_size,
        bbox_max=half_size,
    )


def _scene(*objects: SceneObject) -> RoomScene:
    return RoomScene(
        room_geometry=None,
        scene_dir=Path("."),
        objects={obj.object_id: obj for obj in objects},
    )


def test_repairs_seat_seventy_three_degrees_from_coffee_table() -> None:
    chair = _object(
        "armchair_1", ObjectType.FURNITURE, (1.5, -0.5, 0.5), (0.8, 0.8, 1.0)
    )
    table = _object(
        "coffee_table_0",
        ObjectType.FURNITURE,
        (0.0, -0.05, 0.25),
        (1.0, 0.6, 0.5),
    )

    assert 72.0 < _front_angle_to_target_deg(chair, table) < 74.0
    fixes = align_seating_to_nearest_surface(_scene(chair, table))

    assert [(fix.subject_id, fix.target_id) for fix in fixes] == [
        ("armchair_1", "coffee_table_0")
    ]
    assert _front_angle_to_target_deg(chair, table) < 1e-6


def test_standalone_wall_chair_keeps_wall_normal_priority() -> None:
    chair = _object(
        "guest_chair_0", ObjectType.FURNITURE, (2.0, 0.0, 0.5), (0.8, 0.8, 1.0)
    )
    table = _object(
        "desk_0", ObjectType.FURNITURE, (0.8, 0.0, 0.4), (0.8, 1.0, 0.8)
    )
    east_wall = _object(
        "east_wall", ObjectType.WALL, (2.5, 0.0, 1.35), (0.1, 5.0, 2.7)
    )

    fixes = align_seating_to_nearest_surface(_scene(chair, table, east_wall))

    assert [(fix.subject_id, fix.target_id) for fix in fixes] == [
        ("guest_chair_0", "east_wall")
    ]
    front = chair.transform.rotation().matrix() @ np.array([0.0, 1.0, 0.0])
    np.testing.assert_allclose(front[:2], [-1.0, 0.0], atol=1e-7)


def test_seat_within_forty_five_degrees_is_unchanged() -> None:
    chair = _object(
        "armchair_0",
        ObjectType.FURNITURE,
        (0.0, -1.0, 0.5),
        (0.8, 0.8, 1.0),
        yaw_deg=-30.0,
    )
    table = _object(
        "coffee_table_0", ObjectType.FURNITURE, (0.0, 0.0, 0.25), (1.0, 0.6, 0.5)
    )
    old_transform = chair.transform.GetAsMatrix4().copy()

    fixes = align_seating_to_nearest_surface(_scene(chair, table))

    assert fixes == []
    np.testing.assert_allclose(chair.transform.GetAsMatrix4(), old_transform)
