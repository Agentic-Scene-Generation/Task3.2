"""Tests for deterministic seating orientation repair."""

import hashlib

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from pydrake.math import RigidTransform, RollPitchYaw

from scenesmith.agent_utils.room import ObjectType, RoomScene, SceneObject, UniqueID
from scenesmith.agent_utils.seating_orientation_guard import (
    _front_angle_to_target_deg,
    _is_functional_surface,
    align_seating_to_nearest_surface,
)
from scenesmith.scenebenchmark_critic.api import seating_orientation_targets
from scenesmith.scenebenchmark_critic.intent_contract import SCHEMA_VERSION


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
    table = _object("desk_0", ObjectType.FURNITURE, (0.8, 0.0, 0.4), (0.8, 1.0, 0.8))
    east_wall = _object("east_wall", ObjectType.WALL, (2.5, 0.0, 1.35), (0.1, 5.0, 2.7))

    fixes = align_seating_to_nearest_surface(_scene(chair, table, east_wall))

    assert [(fix.subject_id, fix.target_id) for fix in fixes] == [
        ("guest_chair_0", "east_wall")
    ]
    front = chair.transform.rotation().matrix() @ np.array([0.0, 1.0, 0.0])
    np.testing.assert_allclose(front[:2], [-1.0, 0.0], atol=1e-7)
    chair_bounds = chair.compute_world_bounds()
    wall_bounds = east_wall.compute_world_bounds()
    assert chair_bounds is not None and wall_bounds is not None
    assert abs(wall_bounds[0][0] - chair_bounds[1][0] - 0.03) < 1e-7


def test_wall_guest_chairs_already_facing_desk_use_wall_normal_priority() -> None:
    chairs = [
        _object(
            "guest_chair_0",
            ObjectType.FURNITURE,
            (1.85, 0.7, 0.5),
            (0.55, 0.66, 1.0),
            yaw_deg=45.0,
        ),
        _object(
            "guest_chair_1",
            ObjectType.FURNITURE,
            (1.85, -0.6, 0.5),
            (0.55, 0.66, 1.0),
            yaw_deg=30.0,
        ),
    ]
    desk = _object(
        "study_desk_0",
        ObjectType.FURNITURE,
        (0.0, 1.86, 0.4),
        (1.6, 0.66, 0.8),
    )
    east_wall = _object(
        "east_wall", ObjectType.WALL, (2.475, 0.0, 1.4), (0.05, 4.5, 2.8)
    )
    fixes = align_seating_to_nearest_surface(_scene(*chairs, desk, east_wall))

    assert {fix.subject_id for fix in fixes} == {
        "guest_chair_0",
        "guest_chair_1",
    }
    assert {fix.target_id for fix in fixes} == {"east_wall"}
    wall_bounds = east_wall.compute_world_bounds()
    assert wall_bounds is not None
    for chair in chairs:
        front = chair.transform.rotation().matrix() @ np.array([0.0, 1.0, 0.0])
        np.testing.assert_allclose(front[:2], [-1.0, 0.0], atol=1e-7)
        chair_bounds = chair.compute_world_bounds()
        assert chair_bounds is not None
        assert abs(wall_bounds[0][0] - chair_bounds[1][0] - 0.03) < 1e-7


def test_explicit_prompt_facing_contract_keeps_wall_guest_chairs_desk_relative() -> (
    None
):
    chair = _object(
        "guest_chair_0",
        ObjectType.FURNITURE,
        (1.85, 0.7, 0.5),
        (0.55, 0.66, 1.0),
        yaw_deg=45.0,
    )
    desk = _object(
        "study_desk_0",
        ObjectType.FURNITURE,
        (0.0, 1.86, 0.4),
        (1.6, 0.66, 0.8),
    )
    east_wall = _object(
        "east_wall", ObjectType.WALL, (2.475, 0.0, 1.4), (0.05, 4.5, 2.8)
    )
    scene = _scene(chair, desk, east_wall)
    setattr(
        scene,
        "_scenebenchmark_orientation_contracts",
        {
            "guest_chair_0": {
                "relation_type": "furniture_faces_furniture",
                "target_ids": ["study_desk_0"],
                "prompt_explicit_facing": True,
            }
        },
    )
    old_transform = chair.transform.GetAsMatrix4().copy()

    fixes = align_seating_to_nearest_surface(scene)

    assert fixes == []
    np.testing.assert_allclose(chair.transform.GetAsMatrix4(), old_transform)
    assert _front_angle_to_target_deg(chair, desk) <= 45.0


def test_wall_chairs_just_outside_old_ratio_are_backed_and_face_inward() -> None:
    chairs = [
        _object(
            f"office_chair_{index}",
            ObjectType.FURNITURE,
            (-1.9, -1.5 + index * 0.8, 0.45),
            (0.6, 0.61, 0.9),
        )
        for index in range(2)
    ]
    for chair in chairs:
        chair.description = "ergonomic office chair with adjustable height"
    west_wall = _object(
        "west_wall", ObjectType.WALL, (-2.5, 0.0, 1.35), (0.05, 4.5, 2.7)
    )

    fixes = align_seating_to_nearest_surface(_scene(*chairs, west_wall))

    assert not any(_is_functional_surface(chair) for chair in chairs)
    assert {fix.subject_id for fix in fixes} == {"office_chair_0", "office_chair_1"}
    wall_bounds = west_wall.compute_world_bounds()
    assert wall_bounds is not None
    for chair in chairs:
        front = chair.transform.rotation().matrix() @ np.array([0.0, 1.0, 0.0])
        np.testing.assert_allclose(front[:2], [1.0, 0.0], atol=1e-7)
        chair_bounds = chair.compute_world_bounds()
        assert chair_bounds is not None
        assert abs(chair_bounds[0][0] - wall_bounds[1][0] - 0.03) < 1e-7


def test_stool_at_wall_is_backed_and_faces_inward() -> None:
    stool = _object(
        "stool_0", ObjectType.FURNITURE, (-2.0, 0.0, 0.45), (0.55, 0.55, 0.9)
    )
    west_wall = _object(
        "west_wall", ObjectType.WALL, (-2.5, 0.0, 1.35), (0.05, 4.5, 2.7)
    )

    fixes = align_seating_to_nearest_surface(_scene(stool, west_wall))

    assert [(fix.subject_id, fix.target_id) for fix in fixes] == [
        ("stool_0", "west_wall")
    ]
    front = stool.transform.rotation().matrix() @ np.array([0.0, 1.0, 0.0])
    np.testing.assert_allclose(front[:2], [1.0, 0.0], atol=1e-7)


def test_edge_stool_with_one_footprint_gap_is_snapped_to_wall() -> None:
    stool = _object(
        "stool_0", ObjectType.FURNITURE, (-1.9, 0.0, 0.45), (0.55, 0.55, 0.9)
    )
    west_wall = _object(
        "west_wall", ObjectType.WALL, (-2.5, 0.0, 1.35), (0.05, 4.5, 2.7)
    )

    fixes = align_seating_to_nearest_surface(_scene(stool, west_wall))

    assert [(fix.subject_id, fix.target_id) for fix in fixes] == [
        ("stool_0", "west_wall")
    ]
    stool_bounds = stool.compute_world_bounds()
    wall_bounds = west_wall.compute_world_bounds()
    assert stool_bounds is not None and wall_bounds is not None
    assert abs(stool_bounds[0][0] - wall_bounds[1][0] - 0.03) < 1e-7
    front = stool.transform.rotation().matrix() @ np.array([0.0, 1.0, 0.0])
    np.testing.assert_allclose(front[:2], [1.0, 0.0], atol=1e-7)


def test_dining_chair_near_wall_remains_table_relative() -> None:
    chair = _object(
        "dining_chair_0", ObjectType.FURNITURE, (-1.9, 0.0, 0.45), (0.6, 0.6, 0.9)
    )
    table = _object(
        "dining_table_0", ObjectType.FURNITURE, (-0.9, 0.0, 0.4), (1.6, 0.9, 0.8)
    )
    west_wall = _object(
        "west_wall", ObjectType.WALL, (-2.5, 0.0, 1.35), (0.05, 4.5, 2.7)
    )

    fixes = align_seating_to_nearest_surface(_scene(chair, table, west_wall))

    assert [(fix.subject_id, fix.target_id) for fix in fixes] == [
        ("dining_chair_0", "dining_table_0")
    ]
    assert _front_angle_to_target_deg(chair, table) < 1e-6


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


def test_unconstrained_sofa_defaults_to_nearest_wall_parallel_axis() -> None:
    sofa = _object(
        "sectional_sofa_0",
        ObjectType.FURNITURE,
        (1.35, 0.0, 0.45),
        (1.8, 0.8, 0.9),
        yaw_deg=27.0,
    )
    side_table = _object(
        "side_table_0",
        ObjectType.FURNITURE,
        (0.5, 0.0, 0.3),
        (0.4, 0.4, 0.6),
    )
    east_wall = _object("east_wall", ObjectType.WALL, (2.5, 0.0, 1.35), (0.1, 5.0, 2.7))
    scene = _scene(sofa, side_table, east_wall)
    scene.room_geometry = SimpleNamespace(length=5.0, width=5.0)

    fixes = align_seating_to_nearest_surface(
        scene,
        allowed_targets_by_seat={"sectional_sofa_0": set()},
    )

    assert [(fix.subject_id, fix.target_id) for fix in fixes] == [
        ("sectional_sofa_0", "east_wall")
    ]
    front = sofa.transform.rotation().matrix() @ np.array([0.0, 1.0, 0.0])
    np.testing.assert_allclose(front[:2], [-1.0, 0.0], atol=1e-7)


def test_contract_target_can_be_wall_mounted_media() -> None:
    sofa = _object(
        "sofa_0",
        ObjectType.FURNITURE,
        (0.0, -0.5, 0.45),
        (1.8, 0.8, 0.9),
        yaw_deg=180.0,
    )
    television = _object(
        "television_0",
        ObjectType.WALL_MOUNTED,
        (0.0, 1.2, 1.3),
        (1.2, 0.1, 0.7),
    )

    fixes = align_seating_to_nearest_surface(
        _scene(sofa, television),
        allowed_targets_by_seat={"sofa_0": {"television_0"}},
    )

    assert [(fix.subject_id, fix.target_id) for fix in fixes] == [
        ("sofa_0", "television_0")
    ]
    assert _front_angle_to_target_deg(sofa, television) < 1e-6


def test_across_from_contract_authorizes_media_but_not_next_to_target() -> None:
    sofa = _object(
        "sofa_0",
        ObjectType.FURNITURE,
        (0.0, -0.5, 0.45),
        (1.8, 0.8, 0.9),
        yaw_deg=180.0,
    )
    television = _object(
        "television_0",
        ObjectType.WALL_MOUNTED,
        (0.0, 1.2, 1.3),
        (1.2, 0.1, 0.7),
    )
    side_table = _object(
        "side_table_0",
        ObjectType.FURNITURE,
        (1.0, -0.5, 0.3),
        (0.4, 0.4, 0.6),
    )
    scene = _scene(sofa, television, side_table)
    scene.text_description = "A sofa sits across from a wall-mounted television with a side table next to it."
    scene.scenebenchmark_intent_contract = {
        "schema_version": SCHEMA_VERSION,
        "prompt_sha256": hashlib.sha256(
            " ".join(scene.text_description.split()).encode("utf-8")
        ).hexdigest(),
        "constraints": [
            {
                "relation": "across_from",
                "stage": "wall_mounted",
                "strength": "hard",
                "subjects": {"category": "sofa", "count": 1},
                "targets": {"category": "television", "count": 1},
                "source": "explicit_prompt",
                "evidence_span": "sofa sits across from a wall-mounted television",
            },
            {
                "relation": "next_to",
                "stage": "furniture",
                "strength": "hard",
                "subjects": {"category": "side_table", "count": 1},
                "targets": {"category": "sofa", "count": 1},
                "source": "explicit_prompt",
                "evidence_span": "side table next to it",
            },
        ],
    }

    targets = seating_orientation_targets(scene)

    assert targets == {"sofa_0": {"television_0"}}


def test_wall_backed_sofa_snap_rolls_back_new_window_obstruction() -> None:
    sofa = _object(
        "sofa_0",
        ObjectType.FURNITURE,
        (1.5, 0.0, 0.5),
        (1.8, 0.8, 1.0),
        yaw_deg=-90.0,
    )
    east_wall = _object(
        "east_wall",
        ObjectType.WALL,
        (2.5, 0.0, 1.35),
        (0.1, 5.0, 2.7),
    )
    scene = _scene(sofa, east_wall)
    scene.room_geometry = SimpleNamespace(
        length=5.0,
        width=5.0,
        openings=[
            SimpleNamespace(
                opening_id="window_0",
                opening_type="window",
                clearance_bbox_min=[2.0, -0.75, 0.0],
                clearance_bbox_max=[2.5, 0.75, 2.0],
                sill_height=0.5,
                height=1.5,
            )
        ],
    )
    old_transform = sofa.transform.GetAsMatrix4().copy()

    fixes = align_seating_to_nearest_surface(
        scene,
        allowed_targets_by_seat={"sofa_0": {"east_wall"}},
    )

    assert fixes == []
    np.testing.assert_allclose(sofa.transform.GetAsMatrix4(), old_transform)


def test_l_sofa_explicit_rotation_rolls_back_when_footprint_leaves_room() -> None:
    sofa = _object(
        "l_sofa_0",
        ObjectType.FURNITURE,
        (1.3, 0.0, 0.45),
        (2.8, 1.0, 0.9),
        yaw_deg=90.0,
    )
    television = _object(
        "television_0",
        ObjectType.WALL_MOUNTED,
        (1.3, 1.8, 1.3),
        (1.2, 0.1, 0.7),
    )
    scene = _scene(sofa, television)
    scene.room_geometry = SimpleNamespace(length=4.0, width=4.0)
    old_transform = sofa.transform.GetAsMatrix4().copy()

    fixes = align_seating_to_nearest_surface(
        scene,
        allowed_targets_by_seat={"l_sofa_0": {"television_0"}},
    )

    assert fixes == []
    np.testing.assert_allclose(sofa.transform.GetAsMatrix4(), old_transform)
