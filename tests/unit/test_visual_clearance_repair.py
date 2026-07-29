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
from scenesmith.scenebenchmark_critic.metrics.visual_clearance.classification import (
    is_wall_mounted_visual_subject,
)
from scenesmith.scenebenchmark_critic.visual_clearance_repair import (
    improve_wall_visual_clearance,
)
from scenesmith.wall_agents.tools.wall_surface import WallSurface


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
