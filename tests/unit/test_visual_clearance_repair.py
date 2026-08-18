import xml.etree.ElementTree as ET

from pathlib import Path

import numpy as np

from pydrake.math import RigidTransform, RotationMatrix

from scenesmith.agent_utils.house import RoomGeometry, WallDirection
from scenesmith.agent_utils.room import (
    ObjectType,
    PlacementInfo,
    RoomScene,
    SceneObject,
    UniqueID,
)
from scenesmith.scenebenchmark_critic.config import CriticConfig
from scenesmith.scenebenchmark_critic import visual_clearance_repair
from scenesmith.scenebenchmark_critic.metrics.visual_clearance.classification import (
    is_wall_mounted_visual_subject,
)
from scenesmith.scenebenchmark_critic.visual_clearance_repair import (
    improve_wall_visual_clearance,
)
from scenesmith.wall_agents.tools.wall_surface import WallSurface
from scenesmith.wall_agents.tools.wall_surface import (
    extract_wall_surfaces_from_room_geometry,
)


def _object(
    object_id: str,
    object_type: ObjectType,
    *,
    position: tuple[float, float, float],
    size: tuple[float, float, float],
    placement: PlacementInfo | None = None,
) -> SceneObject:
    half_x, half_y = size[0] / 2.0, size[1] / 2.0
    return SceneObject(
        object_id=UniqueID(object_id),
        object_type=object_type,
        name=object_id.rsplit("_", 1)[0],
        description=object_id.replace("_", " "),
        transform=RigidTransform(p=position),
        placement_info=placement,
        bbox_min=np.array([-half_x, -half_y, -size[2] / 2.0]),
        bbox_max=np.array([half_x, half_y, size[2] / 2.0]),
    )


def _scene(tmp_path: Path) -> tuple[RoomScene, WallSurface]:
    room_geometry = RoomGeometry(
        sdf_tree=ET.ElementTree(ET.Element("sdf")),
        sdf_path=tmp_path / "room.sdf",
        width=4.5,
        length=5.0,
        wall_height=2.7,
        wall_thickness=0.05,
    )
    scene = RoomScene(
        room_geometry=room_geometry,
        scene_dir=tmp_path,
        room_id="bedroom",
        room_type="bedroom",
    )
    surface = WallSurface(
        surface_id=UniqueID("bedroom_north"),
        wall_id="north_wall",
        wall_direction=WallDirection.NORTH,
        bounding_box_min=[0.0, 0.0, 0.0],
        bounding_box_max=[5.0, 0.0, 2.7],
        transform=RigidTransform(
            R=RotationMatrix.Identity(), p=np.array([-2.5, 2.25, 0.0])
        ),
        excluded_regions=[],
    )
    scene.add_object(
        _object(
            "wardrobe_0",
            ObjectType.FURNITURE,
            position=(-1.6, 1.9, 1.0),
            size=(0.8, 0.7, 2.0),
        )
    )
    return scene, surface


def test_wall_shelf_is_visual_clearance_subject() -> None:
    assert is_wall_mounted_visual_subject(
        {
            "id": "shelf_oak_0",
            "object_type": "wall_mounted",
            "name": "shelf_oak",
            "description": "floating wooden shelf",
        }
    )


def test_guard_moves_occluded_clock_on_same_wall(tmp_path: Path) -> None:
    scene, surface = _scene(tmp_path)
    placement = PlacementInfo(
        parent_surface_id=surface.surface_id,
        position_2d=np.array([0.9, 1.8]),
        rotation_2d=0.0,
        placement_method="wall_placement",
    )
    clock = _object(
        "clock_minimal_0",
        ObjectType.WALL_MOUNTED,
        position=(-1.6, 2.25, 1.8),
        size=(0.3, 0.05, 0.3),
        placement=placement,
    )
    scene.add_object(clock)

    fixes = improve_wall_visual_clearance(
        scene,
        wall_surfaces=[surface],
        config=CriticConfig(enabled=True, metrics=("visual_clearance",)),
    )

    assert len(fixes) == 1
    assert fixes[0].object_id == "clock_minimal_0"
    assert fixes[0].wall_surface_id == "bedroom_north"
    assert fixes[0].new_issue_count == 0
    assert clock.placement_info is not None
    assert clock.placement_info.parent_surface_id == surface.surface_id
    assert not np.allclose(clock.placement_info.position_2d, [0.9, 1.8])


def test_guard_moves_occluded_clock_to_another_wall_when_current_wall_is_full(
    tmp_path: Path,
) -> None:
    scene, _unused_surface = _scene(tmp_path)
    surfaces = extract_wall_surfaces_from_room_geometry(
        scene.room_geometry, room_id=scene.room_id
    )
    north = next(
        surface for surface in surfaces if surface.wall_direction == WallDirection.NORTH
    )
    placement = PlacementInfo(
        parent_surface_id=north.surface_id,
        position_2d=np.array([0.9, 1.8]),
        rotation_2d=0.0,
        placement_method="wall_placement",
    )
    clock = _object(
        "clock_minimal_0",
        ObjectType.WALL_MOUNTED,
        position=(-1.6, 2.2, 1.8),
        size=(0.3, 0.05, 0.3),
        placement=placement,
    )
    clock.transform = north.to_world_pose(0.9, 1.8)
    scene.add_object(clock)
    north.excluded_regions = [(0.0, 0.0, north.length, north.height)]

    report = improve_wall_visual_clearance(
        scene,
        wall_surfaces=surfaces,
        config=CriticConfig(enabled=True, metrics=("visual_clearance",)),
    )

    assert len(report) == 1
    assert report[0].old_wall_surface_id == str(north.surface_id)
    assert report[0].new_wall_surface_id != str(north.surface_id)
    assert report[0].old_occlusion_fraction > 0.2
    assert report[0].new_occlusion_fraction <= 0.05
    assert report.rejections == ()
    assert clock.placement_info is not None
    assert clock.placement_info.parent_surface_id != north.surface_id


def test_guard_reports_no_legal_candidate_and_rolls_back(tmp_path: Path) -> None:
    scene, surface = _scene(tmp_path)
    placement = PlacementInfo(
        parent_surface_id=surface.surface_id,
        position_2d=np.array([0.9, 1.8]),
        rotation_2d=0.0,
        placement_method="wall_placement",
    )
    clock = _object(
        "clock_minimal_0",
        ObjectType.WALL_MOUNTED,
        position=(-1.6, 2.25, 1.8),
        size=(0.3, 0.05, 0.3),
        placement=placement,
    )
    scene.add_object(clock)
    surface.excluded_regions = [(0.0, 0.0, surface.length, surface.height)]
    old_transform = clock.transform

    report = improve_wall_visual_clearance(
        scene,
        wall_surfaces=[surface],
        config=CriticConfig(enabled=True, metrics=("visual_clearance",)),
    )

    assert len(report) == 0
    assert report.rejections[0].object_id == "clock_minimal_0"
    assert report.rejections[0].reason == "no_legal_wall_candidate"
    np.testing.assert_allclose(
        clock.transform.GetAsMatrix4(), old_transform.GetAsMatrix4()
    )
    assert clock.placement_info is placement


def test_cross_wall_candidate_cannot_degrade_passing_explicit_relation(
    tmp_path: Path, monkeypatch
) -> None:
    scene, _unused_surface = _scene(tmp_path)
    surfaces = extract_wall_surfaces_from_room_geometry(
        scene.room_geometry, room_id=scene.room_id
    )
    north = next(
        surface for surface in surfaces if surface.wall_direction == WallDirection.NORTH
    )
    placement = PlacementInfo(
        parent_surface_id=north.surface_id,
        position_2d=np.array([0.9, 1.8]),
        rotation_2d=0.0,
        placement_method="wall_placement",
    )
    clock = _object(
        "clock_minimal_0",
        ObjectType.WALL_MOUNTED,
        position=(-1.6, 2.2, 1.8),
        size=(0.3, 0.05, 0.3),
        placement=placement,
    )
    clock.transform = north.to_world_pose(0.9, 1.8)
    scene.add_object(clock)
    north.excluded_regions = [(0.0, 0.0, north.length, north.height)]

    def fake_evaluate(_scene, _config):
        on_original_wall = (
            clock.placement_info is not None
            and clock.placement_info.parent_surface_id == north.surface_id
        )
        return {
            "results": [
                {
                    "check_id": "wall_visibility__clock_minimal_0",
                    "metric": "visual_clearance",
                    "relation_type": "wall_mounted_visibility",
                    "scoring_tier": "core",
                    "label": "fail" if on_original_wall else "pass",
                    "primary_object": "clock_minimal_0",
                    "diagnostics": {
                        "occluded_fraction": 0.8 if on_original_wall else 0.0
                    },
                },
                {
                    "check_id": "intent_wall_relation",
                    "metric": "functional_dependency",
                    "relation_type": "on_wall",
                    "scoring_tier": "core",
                    "label": "pass" if on_original_wall else "degraded",
                    "primary_object": "clock_minimal_0",
                    "evidence": {
                        "intent_constraint": {
                            "relation": "on_wall",
                            "strength": "hard",
                        }
                    },
                },
            ]
        }

    monkeypatch.setattr(visual_clearance_repair, "_evaluate", fake_evaluate)
    old_transform = clock.transform

    report = improve_wall_visual_clearance(
        scene,
        wall_surfaces=surfaces,
        config=CriticConfig(enabled=True, metrics=("visual_clearance",)),
    )

    assert len(report) == 0
    assert report.rejections[0].reason == "no_candidate_resolved_visual_issue"
    np.testing.assert_allclose(
        clock.transform.GetAsMatrix4(), old_transform.GetAsMatrix4()
    )
    assert clock.placement_info is placement


def test_guard_repairs_two_occluded_wall_objects_and_invalidates_each_move(
    tmp_path: Path,
) -> None:
    scene, surface = _scene(tmp_path)
    second_wardrobe = _object(
        "wardrobe_1",
        ObjectType.FURNITURE,
        position=(1.6, 1.9, 1.0),
        size=(0.8, 0.7, 2.0),
    )
    scene.add_object(second_wardrobe)
    for index, local_x in enumerate((0.9, 4.1)):
        placement = PlacementInfo(
            parent_surface_id=surface.surface_id,
            position_2d=np.array([local_x, 1.8]),
            rotation_2d=0.0,
            placement_method="wall_placement",
        )
        clock = _object(
            f"clock_minimal_{index}",
            ObjectType.WALL_MOUNTED,
            position=(-1.6 if index == 0 else 1.6, 2.25, 1.8),
            size=(0.3, 0.05, 0.3),
            placement=placement,
        )
        scene.add_object(clock)
    invalidations: list[None] = []

    report = improve_wall_visual_clearance(
        scene,
        wall_surfaces=[surface],
        config=CriticConfig(enabled=True, metrics=("visual_clearance",)),
        on_accept=lambda: invalidations.append(None),
    )

    assert {fix.object_id for fix in report} == {
        "clock_minimal_0",
        "clock_minimal_1",
    }
    assert len(invalidations) == 2
    assert report[-1].new_issue_count == 0
