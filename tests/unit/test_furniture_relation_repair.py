import math
import xml.etree.ElementTree as ET

from pathlib import Path

import numpy as np

from pydrake.math import RigidTransform, RollPitchYaw

from scenesmith.agent_utils.house import RoomGeometry
from scenesmith.agent_utils.room import ObjectType, RoomScene, SceneObject, UniqueID
from scenesmith.scenebenchmark_critic.api import evaluate_room_scene
from scenesmith.scenebenchmark_critic.config import CriticConfig
from scenesmith.scenebenchmark_critic.furniture_relation_repair import (
    _candidate_improves,
    _score_payload,
    improve_furniture_relations,
)
from scenesmith.utils.geometry_utils import compute_optimal_facing_yaw


def _object(
    object_id: str,
    name: str,
    position: tuple[float, float, float],
    size: tuple[float, float, float],
    *,
    yaw_deg: float = 0.0,
    object_type: ObjectType = ObjectType.FURNITURE,
) -> SceneObject:
    half = np.asarray(size, dtype=float) / 2.0
    return SceneObject(
        object_id=UniqueID(object_id),
        object_type=object_type,
        name=name,
        description=name.replace("_", " "),
        transform=RigidTransform(
            rpy=RollPitchYaw(0.0, 0.0, math.radians(yaw_deg)),
            p=np.asarray(position, dtype=float),
        ),
        bbox_min=-half,
        bbox_max=half,
    )


def _scene(tmp_path: Path, *objects: SceneObject, text: str = "") -> RoomScene:
    walls = [
        _object(
            "west_wall",
            "wall",
            (-2.5, 0.0, 1.35),
            (0.1, 4.0, 2.7),
            object_type=ObjectType.WALL,
        ),
        _object(
            "east_wall",
            "wall",
            (2.5, 0.0, 1.35),
            (0.1, 4.0, 2.7),
            object_type=ObjectType.WALL,
        ),
        _object(
            "south_wall",
            "wall",
            (0.0, -2.0, 1.35),
            (5.0, 0.1, 2.7),
            object_type=ObjectType.WALL,
        ),
        _object(
            "north_wall",
            "wall",
            (0.0, 2.0, 1.35),
            (5.0, 0.1, 2.7),
            object_type=ObjectType.WALL,
        ),
    ]
    geometry = RoomGeometry(
        sdf_tree=ET.ElementTree(ET.Element("sdf")),
        sdf_path=tmp_path / "room.sdf",
        walls=walls,
        width=4.0,
        length=5.0,
        wall_height=2.7,
        wall_thickness=0.1,
    )
    return RoomScene(
        room_geometry=geometry,
        scene_dir=tmp_path,
        room_id="test_room",
        room_type="study" if "study" in text else "dining_room",
        text_description=text,
        objects={obj.object_id: obj for obj in objects},
    )


def _facing_yaw(origin: tuple[float, float], target=(0.0, 0.0)) -> float:
    return compute_optimal_facing_yaw(
        origin_a=np.array([origin[0], origin[1], 0.0]),
        target_point=np.array([target[0], target[1], 0.0]),
    )


def test_repairs_rotated_dining_slots_without_moving_unrelated_furniture(
    tmp_path: Path,
) -> None:
    yaw = math.radians(30.0)
    tx = np.array([math.cos(yaw), math.sin(yaw)])
    ty = np.array([-math.sin(yaw), math.cos(yaw)])
    table = _object(
        "surface_alpha", "dining_table", (0.0, 0.0, 0.4), (2.0, 0.8, 0.8), yaw_deg=30.0
    )
    positions = [
        0.5 * tx - 0.9 * ty,
        -0.5 * tx + 0.9 * ty,
        1.25 * tx,
        -1.25 * tx,
    ]
    seats = [
        _object(
            f"seat_variant_{index}",
            "dining_chair",
            (float(position[0]), float(position[1]), 0.45),
            (0.5, 0.5, 0.9),
            yaw_deg=_facing_yaw((float(position[0]), float(position[1]))),
        )
        for index, position in enumerate(positions)
    ]
    unrelated = _object(
        "decorative_console", "console_table", (1.8, 1.2, 0.4), (0.8, 0.4, 0.8)
    )
    old_unrelated = unrelated.transform.GetAsMatrix4().copy()
    scene = _scene(tmp_path, table, *seats, unrelated)
    config = CriticConfig(enabled=True, metrics=("functional_dependency",))

    fixes = improve_furniture_relations(scene, config=config)

    assert {fix.object_id for fix in fixes} == {"seat_variant_0", "seat_variant_1"}
    np.testing.assert_allclose(unrelated.transform.GetAsMatrix4(), old_unrelated)
    payload = evaluate_room_scene(scene, config=config, stage="test")
    result = next(
        item
        for item in payload["results"]
        if item.get("relation_type") == "dining_seat_distribution"
    )
    assert result["label"] == "pass"


def test_repairs_generic_storage_to_wall(tmp_path: Path) -> None:
    storage = _object(
        "storage_piece_alpha",
        "shelving_unit",
        (-1.0, 0.0, 0.9),
        (0.8, 0.3, 1.8),
    )
    storage.metadata["category"] = "shelving_unit"
    scene = _scene(tmp_path, storage)

    fixes = improve_furniture_relations(
        scene,
        config=CriticConfig(enabled=True, metrics=("functional_dependency",)),
    )

    assert [(fix.object_id, fix.relation_type) for fix in fixes] == [
        ("storage_piece_alpha", "wall_backed_storage_alignment")
    ]
    assert storage.transform.translation()[0] < -1.8


def test_rejects_assigned_work_seat_slot_outside_room(tmp_path: Path) -> None:
    desk = _object("work_surface", "office_desk", (0.0, -1.55, 0.4), (1.4, 0.8, 0.8))
    chair = _object("task_seat", "office_chair", (0.0, -0.3, 0.45), (0.6, 0.6, 0.9))
    old_transform = chair.transform.GetAsMatrix4().copy()
    scene = _scene(tmp_path, desk, chair, text="A study with a desk and chair")

    fixes = improve_furniture_relations(
        scene,
        config=CriticConfig(enabled=True, metrics=("functional_dependency",)),
        max_translation_m=3.0,
    )

    assert all(fix.object_id != "task_seat" for fix in fixes)
    np.testing.assert_allclose(chair.transform.GetAsMatrix4(), old_transform)


def test_candidate_score_rejects_new_issue() -> None:
    baseline = {
        "results": [
            {
                "check_id": "target",
                "label": "fail",
                "relation_type": "seating_to_work_surface",
            }
        ]
    }
    candidate = {
        "results": [
            {
                "check_id": "target",
                "label": "pass",
                "relation_type": "seating_to_work_surface",
            },
            {
                "check_id": "new_collision",
                "label": "fail",
                "relation_type": "collision",
            },
        ]
    }

    assert not _candidate_improves(
        baseline,
        candidate,
        baseline_score=_score_payload(baseline),
        check_id="target",
    )
