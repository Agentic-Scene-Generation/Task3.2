import hashlib
import json
import math
import xml.etree.ElementTree as ET

from pathlib import Path

import numpy as np
import pytest

from pydrake.math import RigidTransform, RollPitchYaw

from scenesmith.agent_utils.house import ClearanceOpeningData, RoomGeometry
from scenesmith.agent_utils.room import (
    AgentType,
    ObjectType,
    PlacementInfo,
    RoomScene,
    SceneObject,
    SupportSurface,
    UniqueID,
)
from scenesmith.scenebenchmark_critic import furniture_relation_repair
from scenesmith.scenebenchmark_critic.api import evaluate_room_scene
from scenesmith.scenebenchmark_critic.config import CriticConfig
from scenesmith.scenebenchmark_critic.furniture_relation_repair import (
    _RepairTarget,
    _candidate_improves,
    _prioritize_coordinated_seating_targets,
    _score_payload,
    improve_furniture_relations,
    unresolved_furniture_relation_failures,
)
from scenesmith.scenebenchmark_critic.intent_contract import (
    SCHEMA_VERSION,
    build_intent_contract,
)
from scenesmith.utils.geometry_utils import compute_optimal_facing_yaw
from scenesmith.wall_agents.tools.wall_surface import (
    extract_wall_surfaces_from_room_geometry,
)


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
        room_type=(
            "classroom"
            if "classroom" in text
            else "study" if "study" in text else "dining_room"
        ),
        text_description=text,
        objects={obj.object_id: obj for obj in objects},
    )


def _attach_intent_contract(scene: RoomScene, constraints: list[dict]) -> None:
    scene.scenebenchmark_intent_contract = {
        "schema_version": SCHEMA_VERSION,
        "prompt_sha256": hashlib.sha256(
            " ".join(scene.text_description.split()).encode("utf-8")
        ).hexdigest(),
        "constraints": constraints,
    }


def _attach_fixture_contract(scene: RoomScene) -> None:
    """Construct a v4 fixture contract without restoring runtime fallback behavior."""
    contract = build_intent_contract(
        scene.text_description,
        room_type=scene.room_type,
        task_spec=getattr(scene, "scene_expert_task_spec", None),
    )
    for constraint in contract["constraints"]:
        if constraint.get("source") == "model_inferred" and not constraint.get(
            "inference_reason"
        ):
            constraint["inference_reason"] = "fixture-only inferred relation"
    scene.scenebenchmark_intent_contract = contract


def _complete_fixture_contract_evidence(scene: RoomScene) -> None:
    """Make hand-authored fixture constraints valid independent-contract rows."""
    contract = scene.scenebenchmark_intent_contract
    for constraint in contract["constraints"]:
        constraint.setdefault("evidence_span", scene.text_description)


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
    scene = _scene(
        tmp_path,
        table,
        *seats,
        unrelated,
        text=(
            "A dining room with a dining table and four dining chairs arranged "
            "around it with one on each side."
        ),
    )
    _attach_fixture_contract(scene)
    config = CriticConfig(enabled=True, metrics=("functional_dependency",))

    unresolved_before = unresolved_furniture_relation_failures(scene, config=config)
    assert {result.get("relation_type") for result in unresolved_before} == {
        "edge_distribution"
    }

    fixes = improve_furniture_relations(scene, config=config)

    assert {fix.object_id for fix in fixes} == {
        f"seat_variant_{index}" for index in range(4)
    }
    np.testing.assert_allclose(unrelated.transform.GetAsMatrix4(), old_unrelated)
    payload = evaluate_room_scene(scene, config=config, stage="test")
    result = next(
        item
        for item in payload["results"]
        if item.get("relation_type") == "edge_distribution"
    )
    assert result["label"] == "pass"
    assert unresolved_furniture_relation_failures(scene, config=config) == []


def test_one_per_edge_dining_repair_is_atomic_and_table_local(tmp_path: Path) -> None:
    table = _object(
        "dining_table_0",
        "dining_table",
        (0.0, 0.0, 0.4),
        (2.0, 0.8, 0.8),
    )
    seats = [
        _object(
            f"dining_chair_{index}",
            "dining_chair",
            (*xy, 0.45),
            (0.5, 0.5, 0.9),
            yaw_deg=yaw,
        )
        for index, (xy, yaw) in enumerate(
            (
                ((0.0, 1.88), 180.0),
                ((0.0, -0.83), 0.0),
                ((2.13, 0.0), 90.0),
                ((-2.13, -1.6), -90.0),
            )
        )
    ]
    scene = _scene(
        tmp_path,
        table,
        *seats,
        text=(
            "A dining room with a dining table in the center and four dining "
            "chairs arranged around it with one on each of the four sides."
        ),
    )
    _attach_intent_contract(
        scene,
        [
            {
                "relation": "edge_distribution",
                "subjects": {
                    "category": "dining_chair",
                    "count": 4,
                    "quantifier": "all",
                },
                "targets": {
                    "category": "dining_table",
                    "count": 1,
                    "quantifier": "all",
                },
                "edge_frame": "target_local_rectangle",
                "groups": [
                    {
                        "edge_class": "long",
                        "counts_per_edge": [1, 1],
                        "spacing": "equal_segments",
                    },
                    {
                        "edge_class": "short",
                        "counts_per_edge": [1, 1],
                        "spacing": "equal_segments",
                    },
                ],
                "orientation": "toward_target",
                "source": "explicit_prompt",
                "evidence_span": "four chairs with one on each of the four sides",
            }
        ],
    )
    old_table = table.transform.GetAsMatrix4().copy()
    config = CriticConfig(enabled=True, metrics=("functional_dependency",))

    fixes = improve_furniture_relations(scene, config=config)

    assert {fix.object_id for fix in fixes} == {
        f"dining_chair_{index}" for index in range(4)
    }
    np.testing.assert_allclose(table.transform.GetAsMatrix4(), old_table)
    payload = evaluate_room_scene(scene, config=config, stage="one_per_edge_after")
    result = next(
        item
        for item in payload["results"]
        if item.get("relation_type") == "edge_distribution"
    )
    assert result["label"] == "pass"
    for slot in result["diagnostics"]["seat_slots"]:
        assert not slot["failure"]
        assert slot["facing_error_deg"] <= 10.0


def test_centered_table_and_edge_group_repair_as_one_candidate(tmp_path: Path) -> None:
    table = _object("dining_table_0", "dining_table", (0.0, -1.2, 0.4), (2.0, 0.8, 0.8))
    seats = [
        _object(
            f"dining_chair_{index}",
            "dining_chair",
            (*xy, 0.45),
            (0.5, 0.5, 0.9),
            yaw_deg=yaw,
        )
        for index, (xy, yaw) in enumerate(
            (
                ((-1.25, -1.2), 0.0),
                ((1.25, -1.2), 180.0),
                ((0.0, -2.0), 90.0),
                ((0.0, -0.4), -90.0),
            )
        )
    ]
    scene = _scene(
        tmp_path,
        table,
        *seats,
        text="A dining room with a dining table in the center and four dining chairs.",
    )
    _attach_intent_contract(
        scene,
        [
            {
                "relation": "centered_in_room",
                "subjects": {"category": "dining_table", "count": 1},
                "targets": {"category": "room", "count": 1},
                "source": "explicit_prompt",
                "evidence_span": "table in the center",
            },
            {
                "relation": "edge_distribution",
                "subjects": {
                    "category": "dining_chair",
                    "count": 4,
                    "quantifier": "all",
                },
                "targets": {"category": "dining_table", "count": 1},
                "edge_frame": "target_local_rectangle",
                "groups": [
                    {
                        "edge_class": "long",
                        "counts_per_edge": [1, 1],
                        "spacing": "equal_segments",
                    },
                    {
                        "edge_class": "short",
                        "counts_per_edge": [1, 1],
                        "spacing": "equal_segments",
                    },
                ],
                "orientation": "toward_target",
                "source": "explicit_prompt",
                "evidence_span": "one chair on each side",
            },
        ],
    )
    config = CriticConfig(enabled=True, metrics=("functional_dependency",))

    fixes = improve_furniture_relations(scene, config=config)

    assert {fix.object_id for fix in fixes} == {
        "dining_table_0",
        *{f"dining_chair_{index}" for index in range(4)},
    }
    results = evaluate_room_scene(scene, config=config, stage="centered_edge_after")[
        "results"
    ]
    assert {
        result["relation_type"]: result["label"]
        for result in results
        if result.get("relation_type") in {"room_center_alignment", "edge_distribution"}
    } == {"room_center_alignment": "pass", "edge_distribution": "pass"}


def test_repairs_room_facing_contract_with_yaw_only_move(tmp_path: Path) -> None:
    chair = _object(
        "guest_chair_0",
        "guest_chair",
        (0.0, -1.4, 0.45),
        (0.5, 0.5, 0.9),
        yaw_deg=180.0,
    )
    scene = _scene(
        tmp_path,
        chair,
        text="A guest chair against the wall facing into the room.",
    )
    _attach_intent_contract(
        scene,
        [
            {
                "relation": "faces",
                "subjects": {"category": "guest_chair", "count": 1},
                "targets": {"category": "room", "count": 1},
                "source": "explicit_prompt",
                "evidence_span": "guest chair facing into the room",
            }
        ],
    )
    config = CriticConfig(enabled=True, metrics=("functional_dependency",))

    fixes = improve_furniture_relations(scene, config=config)

    assert [(fix.object_id, fix.relation_type) for fix in fixes] == [
        ("guest_chair_0", "faces")
    ]
    result = next(
        result
        for result in evaluate_room_scene(
            scene, config=config, stage="room_facing_after"
        )["results"]
        if result.get("relation_type") == "faces"
    )
    assert result["label"] == "pass"


def test_repairs_reversed_mixed_table_edge_topology_atomically(tmp_path: Path) -> None:
    table = _object(
        "conference_table_0", "conference_table", (0.0, 0.0, 0.4), (3.3, 0.8, 0.8)
    )
    positions = [
        *[(-1.95, y) for y in (-0.95, 0.0, 0.95)],
        *[(1.95, y) for y in (-0.95, 0.0, 0.95)],
        (0.0, 0.95),
    ]
    seats = [
        _object(
            f"office_chair_{index}",
            "office_chair",
            (*position, 0.45),
            (0.5, 0.5, 0.9),
            yaw_deg=_facing_yaw(position),
        )
        for index, position in enumerate(positions)
    ]
    scene = _scene(
        tmp_path,
        table,
        *seats,
        text=(
            "A meeting room with one rectangular conference table and seven office "
            "chairs. Arrange six office chairs in two equal groups of three, evenly "
            "spaced along the table's two long sides. Place one remaining office chair "
            "centered along one short side, facing the table. Keep the opposite short "
            "side free of chairs."
        ),
    )
    _attach_fixture_contract(scene)
    config = CriticConfig(enabled=True, metrics=("functional_dependency",))

    fixes = improve_furniture_relations(scene, config=config)

    assert {fix.object_id for fix in fixes} <= {seat.object_id for seat in seats}
    assert len(fixes) >= 6
    result = next(
        item
        for item in evaluate_room_scene(scene, config=config, stage="test")["results"]
        if item.get("relation_type") == "edge_distribution"
    )
    assert result["label"] == "pass"
    edges = [slot["edge"] for slot in result["diagnostics"]["seat_slots"]]
    assert edges.count("front") == 3
    assert edges.count("back") == 3
    assert edges.count("left") + edges.count("right") == 1


def test_repairs_reversed_mixed_topology_for_generic_fallback_table(
    tmp_path: Path,
) -> None:
    """A fallback table must not prevent an independent seating repair."""
    table = _object("table_0", "table", (0.0, 0.0, 0.34), (1.31, 0.8, 0.67))
    positions = [
        *[(-1.3, y) for y in (-1.0, 0.0, 1.0)],
        *[(1.3, y) for y in (-1.0, 0.0, 1.0)],
        (0.0, 2.2),
    ]
    seats = [
        _object(
            f"office_chair_{index}",
            "office_chair",
            (*position, 0.44),
            (0.6, 0.61, 0.87),
            yaw_deg=_facing_yaw(position),
        )
        for index, position in enumerate(positions)
    ]
    scene = _scene(
        tmp_path,
        table,
        *seats,
        text=(
            "A meeting room with one rectangular conference table and seven office "
            "chairs. Arrange six office chairs in two equal groups of three, evenly "
            "spaced along the table's two long sides. Place one remaining office chair "
            "centered along one short side, facing the table. Keep the opposite short "
            "side free of chairs."
        ),
    )
    scene.room_type = "meeting_room"
    _attach_fixture_contract(scene)
    config = CriticConfig(enabled=True, metrics=("functional_dependency",))

    fixes = improve_furniture_relations(scene, config=config)

    assert len(fixes) >= 6
    result = next(
        item
        for item in evaluate_room_scene(scene, config=config, stage="test")["results"]
        if item.get("relation_type") == "edge_distribution"
    )
    assert result["label"] == "pass"


def test_rotated_asymmetric_edge_seats_do_not_overlap_table(tmp_path: Path) -> None:
    """Seat slots must use the footprint after their inward-facing rotation."""
    table = _object(
        "conference_table_0", "conference_table", (0.0, 0.0, 0.4), (3.0, 0.8, 0.8)
    )
    positions = [
        *[(x, y) for y in (-1.4, 1.4) for x in (-1.0, 0.0, 1.0)],
        (2.2, 0.0),
    ]
    seats = [
        _object(
            f"office_chair_{index}",
            "office_chair",
            (*position, 0.45),
            (0.4, 0.9, 0.9),
            yaw_deg=0.0,
        )
        for index, position in enumerate(positions)
    ]
    scene = _scene(
        tmp_path,
        table,
        *seats,
        text=(
            "A meeting room with one rectangular conference table and seven office "
            "chairs. Arrange six office chairs in two equal groups of three, evenly "
            "spaced along the table's two long sides. Place one remaining office chair "
            "centered along one short side, facing the table. Keep the opposite short "
            "side free of chairs."
        ),
    )
    _attach_fixture_contract(scene)
    config = CriticConfig(enabled=True, metrics=("functional_dependency",))

    improve_furniture_relations(scene, config=config)

    result = next(
        item
        for item in evaluate_room_scene(scene, config=config, stage="asymmetric_after")[
            "results"
        ]
        if item.get("relation_type") == "edge_distribution"
    )
    assert result["label"] == "pass"
    table_lower, table_upper = table.compute_world_bounds()
    for seat in seats:
        lower, upper = seat.compute_world_bounds()
        overlaps_x = (
            float(lower[0]) < float(table_upper[0]) - 0.01
            and float(upper[0]) > float(table_lower[0]) + 0.01
        )
        overlaps_y = (
            float(lower[1]) < float(table_upper[1]) - 0.01
            and float(upper[1]) > float(table_lower[1]) + 0.01
        )
        assert not (overlaps_x and overlaps_y), seat.object_id


def test_sideboard_wall_phrase_does_not_bind_dining_chairs_to_wall(
    tmp_path: Path,
) -> None:
    table = _object("dining_table_0", "dining_table", (0.0, 0.0, 0.4), (2.0, 0.8, 0.8))
    seats = [
        _object(
            f"dining_chair_{index}",
            "dining_chair",
            (*xy, 0.45),
            (0.5, 0.5, 0.9),
            yaw_deg=_facing_yaw(xy),
        )
        for index, xy in enumerate(((0.0, -0.9), (0.0, 0.9), (-1.3, 0.0), (1.3, 0.0)))
    ]
    sideboard = _object("sideboard_0", "sideboard", (0.0, -1.7, 0.4), (1.2, 0.4, 0.8))
    scene = _scene(
        tmp_path,
        table,
        *seats,
        sideboard,
        text=(
            "A dining room with a dining table in the center, four dining chairs "
            "arranged around it with one on each side, a sideboard against the "
            "wall behind the chairs on one side, and table settings for four."
        ),
    )
    _attach_fixture_contract(scene)
    config = CriticConfig(enabled=True, metrics=("functional_dependency",))

    payload = evaluate_room_scene(scene, config=config, stage="dining_sideboard_prompt")

    constraints = payload["case_pack"]["intent_contract"]["constraints"]
    assert any(
        row["relation"] == "against_wall" and row["subjects"]["category"] == "sideboard"
        for row in constraints
    )
    assert not any(
        row["relation"] == "against_wall"
        and row["subjects"]["category"] == "dining_chair"
        for row in constraints
    )


def test_instructional_contract_repairs_furniture_then_wall_surface(
    tmp_path: Path,
) -> None:
    teacher = _object(
        "instructor_desk_0",
        "instructor_desk",
        (1.75, 1.55, 0.4),
        (1.4, 0.7, 0.8),
        yaw_deg=180.0,
    )
    text = (
        "A classroom with student desks and chairs. An instructor desk sits near "
        "a chalkboard for the teaching area."
    )
    scene = _scene(tmp_path, teacher, text=text)
    scene.scene_expert_task_spec = {
        "room_type": "classroom",
        "required_large_objects": ["student desk", "chair", "instructor desk"],
        "required_wall_objects": ["chalkboard"],
        "functional_zones": ["teaching_zone", "student_seating_zone"],
    }
    _attach_fixture_contract(scene)
    config = CriticConfig(
        enabled=True,
        metrics=("functional_dependency",),
    )

    before = evaluate_room_scene(scene, config=config, stage="furniture_before")
    operation = next(
        item
        for item in before["results"]
        if item.get("relation_type") == "operation_zone_at_wall"
    )
    assert operation["label"] == "fail"
    assert operation["diagnostics"]["wall_id"] == "north_wall"

    furniture_fixes = improve_furniture_relations(scene, config=config)

    assert ("instructor_desk_0", "operation_zone_at_wall") in {
        (fix.object_id, fix.relation_type) for fix in furniture_fixes
    }
    teacher_center = teacher.transform.translation()
    teacher_yaw = math.degrees(RollPitchYaw(teacher.transform.rotation()).yaw_angle())
    assert abs(teacher_center[0]) < 1e-6
    assert abs(teacher_yaw) < 1e-6
    furniture_after = evaluate_room_scene(scene, config=config, stage="furniture_after")
    operation = next(
        item
        for item in furniture_after["results"]
        if item.get("relation_type") == "operation_zone_at_wall"
    )
    assert operation["label"] == "pass"
    assert operation["diagnostics"]["operation_clearance_m"] >= 0.65
    pending_alignment = next(
        item
        for item in furniture_after["results"]
        if item.get("relation_type") == "instructional_surface_alignment"
    )
    assert pending_alignment["contract_state"] == "pending"

    wall_surfaces = {
        surface.wall_direction.value: surface
        for surface in extract_wall_surfaces_from_room_geometry(
            scene.room_geometry,
            room_id=scene.room_id,
        )
    }
    east_surface = wall_surfaces["east"]
    chalkboard = _object(
        "chalkboard_0",
        "chalkboard",
        (0.0, 0.0, 0.0),
        (1.8, 0.05, 1.1),
        object_type=ObjectType.WALL_MOUNTED,
    )
    chalkboard.transform = east_surface.to_world_pose(
        position_x=1.3,
        position_z=1.5,
    )
    chalkboard.placement_info = PlacementInfo(
        parent_surface_id=east_surface.surface_id,
        position_2d=np.array([1.3, 1.5]),
        rotation_2d=0.0,
        placement_method="wall_placement",
    )
    scene.add_object(chalkboard)
    wall_before = evaluate_room_scene(scene, config=config, stage="wall_before")
    alignment = next(
        item
        for item in wall_before["results"]
        if item.get("relation_type") == "instructional_surface_alignment"
    )
    assert alignment["label"] == "fail"
    assert not alignment["diagnostics"]["same_wall"]

    wall_fixes = improve_furniture_relations(
        scene,
        config=config,
        allowed_relation_types={"instructional_surface_alignment"},
    )

    assert ("chalkboard_0", "instructional_surface_alignment") in {
        (fix.object_id, fix.relation_type) for fix in wall_fixes
    }
    assert abs(chalkboard.transform.translation()[0]) < 1e-6
    assert chalkboard.placement_info is not None
    north_surface = wall_surfaces["north"]
    assert chalkboard.placement_info.parent_surface_id == north_surface.surface_id
    expected_transform = north_surface.to_world_pose(
        position_x=float(chalkboard.placement_info.position_2d[0]),
        position_z=float(chalkboard.placement_info.position_2d[1]),
        rotation_deg=math.degrees(chalkboard.placement_info.rotation_2d),
    )
    np.testing.assert_allclose(
        chalkboard.transform.GetAsMatrix4(),
        expected_transform.GetAsMatrix4(),
        atol=1e-7,
    )

    serialized = json.loads(json.dumps(chalkboard.to_dict(scene_dir=tmp_path)))
    restored_chalkboard = SceneObject.from_dict(serialized, scene_dir=tmp_path)
    scene.objects[restored_chalkboard.object_id] = restored_chalkboard
    wall_after = evaluate_room_scene(scene, config=config, stage="wall_after")
    alignment = next(
        item
        for item in wall_after["results"]
        if item.get("relation_type") == "instructional_surface_alignment"
    )
    assert alignment["label"] == "pass"
    assert restored_chalkboard.placement_info is not None
    assert (
        restored_chalkboard.placement_info.parent_surface_id == north_surface.surface_id
    )


def test_relation_allowlist_leaves_unrequested_repairs_untouched(
    tmp_path: Path,
) -> None:
    storage = _object(
        "storage_piece_alpha",
        "shelving_unit",
        (-1.0, 0.0, 0.9),
        (0.8, 0.3, 1.8),
    )
    storage.metadata["category"] = "shelving_unit"
    scene = _scene(tmp_path, storage)
    original_transform = storage.transform.GetAsMatrix4().copy()

    fixes = improve_furniture_relations(
        scene,
        config=CriticConfig(enabled=True, metrics=("functional_dependency",)),
        allowed_relation_types={"instructional_surface_alignment"},
    )

    assert fixes == []
    np.testing.assert_allclose(storage.transform.GetAsMatrix4(), original_transform)


def test_paired_work_surfaces_rotate_toward_their_assigned_seats(
    tmp_path: Path,
) -> None:
    desk_positions = (
        (-1.5, 0.8),
        (0.0, 0.8),
        (1.5, 0.8),
        (-1.5, -0.5),
        (0.0, -0.5),
        (1.5, -0.5),
    )
    desks = [
        _object(
            f"student_desk_{index}",
            "student_desk",
            (*position, 0.375),
            (1.05, 0.5, 0.75),
            yaw_deg=0.0,
        )
        for index, position in enumerate(desk_positions)
    ]
    chairs = [
        _object(
            f"student_chair_{index}",
            "student_chair",
            (position[0], position[1] - 0.585, 0.45),
            (0.5, 0.51, 0.9),
            yaw_deg=0.0,
        )
        for index, position in enumerate(desk_positions)
    ]
    # Rendered asset choice keeps these role-level names for exact selector
    # binding. Functional dependency evaluation must still recognize their
    # desk/chair families rather than treating the names as opaque categories.
    for obj in (*desks, *chairs):
        obj.metadata["semantic_name"] = obj.name
    scene = _scene(
        tmp_path,
        *desks,
        *chairs,
        text=(
            "A classroom with six student desks, each with a chair. A teacher's "
            "desk sits at the front near the chalkboard."
        ),
    )
    _attach_fixture_contract(scene)
    config = CriticConfig(
        enabled=True,
        metrics=("functional_dependency",),
    )

    before = evaluate_room_scene(scene, config=config, stage="paired_surface_before")
    paired_surface_checks = {
        check["check_id"]
        for check in before["case_pack"]["checks"]
        if check.get("relation_type") == "furniture_faces_furniture"
        and (check.get("evidence") or {}).get("paired_surface_facing")
    }
    before_results = [
        result
        for result in before["results"]
        if result.get("check_id") in paired_surface_checks
    ]
    assert len(paired_surface_checks) == 6
    assert {result["label"] for result in before_results} == {"fail"}
    assert not any(
        result.get("relation_type") == "classroom_workstation_distribution"
        for result in before["results"]
    )

    fixes = improve_furniture_relations(scene, config=config)

    assert {fix.object_id for fix in fixes} == {
        f"student_desk_{index}" for index in range(6)
    }
    assert {fix.relation_type for fix in fixes} == {"furniture_faces_furniture"}
    for desk in desks:
        yaw = math.degrees(RollPitchYaw(desk.transform.rotation()).yaw_angle())
        assert abs(abs(yaw) - 180.0) < 1e-6

    after = evaluate_room_scene(scene, config=config, stage="paired_surface_after")
    pair_results = [
        result
        for result in after["results"]
        if result.get("check_id") in paired_surface_checks
        or (
            result.get("relation_type") == "seating_to_work_surface"
            and str(result.get("check_id") or "").startswith("intent_contract__")
        )
    ]
    assert len(pair_results) == 12
    assert {result["label"] for result in pair_results} == {"pass"}
    assert not any(
        result.get("relation_type")
        in {"seating_to_work_surface", "furniture_faces_furniture"}
        for result in unresolved_furniture_relation_failures(scene, config=config)
    )


def test_repairs_generic_near_contract_for_unrelated_categories(
    tmp_path: Path,
) -> None:
    armchair = _object("armchair_0", "armchair", (-1.6, 0.0, 0.45), (0.7, 0.7, 0.9))
    plant = _object("plant_0", "plant", (1.4, 0.0, 0.45), (0.6, 0.6, 0.9))
    scene = _scene(tmp_path, armchair, plant)
    _attach_intent_contract(
        scene,
        [
            {
                "relation": "near",
                "subjects": {"category": "armchair", "count": 1},
                "targets": {"category": "plant", "count": 1},
                "source": "explicit_prompt",
                "evidence_span": "an armchair near a plant",
            }
        ],
    )
    config = CriticConfig(enabled=True, metrics=("functional_dependency",))

    before = evaluate_room_scene(scene, config=config, stage="generic_near_before")
    near_before = next(
        result
        for result in before["results"]
        if result.get("relation_type") == "generic_near_relation"
    )
    assert near_before["label"] == "fail"

    fixes = improve_furniture_relations(scene, config=config)

    assert [(fix.object_id, fix.relation_type) for fix in fixes] == [
        ("armchair_0", "generic_near_relation")
    ]
    after = evaluate_room_scene(scene, config=config, stage="generic_near_after")
    near_after = next(
        result
        for result in after["results"]
        if result.get("relation_type") == "generic_near_relation"
    )
    assert near_after["label"] == "pass"
    assert near_after["contract_state"] == "passed"


def test_repairs_corner_of_room_contract(tmp_path: Path) -> None:
    wardrobe = _object("wardrobe_0", "wardrobe", (-1.6, 1.1, 0.9), (0.8, 0.6, 1.8))
    scene = _scene(tmp_path, wardrobe)
    _attach_intent_contract(
        scene,
        [
            {
                "relation": "corner_of_room",
                "subjects": {"category": "wardrobe", "count": 1},
                "targets": {"category": "room"},
                "source": "explicit_prompt",
                "evidence_span": "wardrobe in the corner of the room",
            }
        ],
    )
    config = CriticConfig(enabled=True, metrics=("functional_dependency",))

    before = evaluate_room_scene(scene, config=config, stage="corner_before")
    corner_before = next(
        result
        for result in before["results"]
        if result.get("relation_type") == "corner_of_room"
    )
    assert corner_before["label"] == "fail"

    fixes = improve_furniture_relations(scene, config=config)

    assert [(fix.object_id, fix.relation_type) for fix in fixes] == [
        ("wardrobe_0", "corner_of_room")
    ]
    after = evaluate_room_scene(scene, config=config, stage="corner_after")
    corner_after = next(
        result
        for result in after["results"]
        if result.get("relation_type") == "corner_of_room"
    )
    assert corner_after["label"] == "pass"
    assert corner_after["contract_state"] == "passed"


def test_does_not_infer_generic_storage_to_wall(tmp_path: Path) -> None:
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

    assert fixes == []
    np.testing.assert_allclose(storage.transform.translation()[:2], [-1.0, 0.0])


def test_repairs_window_blocking_wall_backed_bookshelf(tmp_path: Path) -> None:
    bookshelf = _object(
        "bookshelf_0",
        "bookshelf",
        (0.0, -1.78, 0.92),
        (1.0, 0.36, 1.84),
    )
    bookshelf.metadata["category"] = "bookshelf"
    scene = _scene(tmp_path, bookshelf, text="A study with a bookshelf.")
    scene.room_geometry.openings = [
        ClearanceOpeningData(
            opening_id="window_0",
            opening_type="window",
            wall_direction="south",
            center_world=[0.0, -2.0, 1.5],
            width=1.5,
            sill_height=0.9,
            height=1.2,
            clearance_bbox_min=[-0.75, -2.1, 0.0],
            clearance_bbox_max=[0.75, -1.65, 2.1],
            wall_start=[-2.5, -2.0],
            wall_end=[2.5, -2.0],
            position_along_wall=2.5,
        )
    ]
    config = CriticConfig(enabled=True)

    before = evaluate_room_scene(scene, config=config, stage="window_before")
    assert (
        next(
            result
            for result in before["results"]
            if result["check_id"] == "window_clearance__window_0"
        )["label"]
        == "fail"
    )

    fixes = improve_furniture_relations(scene, config=config)

    assert [(fix.object_id, fix.relation_type) for fix in fixes] == [
        ("bookshelf_0", "window_clearance")
    ]
    assert abs(float(bookshelf.transform.translation()[0])) >= 1.27
    after = evaluate_room_scene(scene, config=config, stage="window_after")
    result_by_id = {result["check_id"]: result for result in after["results"]}
    assert result_by_id["window_clearance__window_0"]["label"] == "pass"
    assert "wall_backed_storage__bookshelf_0" not in result_by_id


def test_repairs_multiple_wall_backed_stools_with_wall_normal_orientation(
    tmp_path: Path,
) -> None:
    stools = [
        _object(
            f"stool_{index}",
            "stool",
            (-2.0, -0.7 + index, 0.45),
            (0.55, 0.65, 0.9),
        )
        for index in range(2)
    ]
    west_wall = _object(
        "west_wall",
        "wall",
        (-2.5, 0.0, 1.35),
        (0.1, 4.0, 2.7),
        object_type=ObjectType.WALL,
    )
    scene = _scene(
        tmp_path,
        *stools,
        west_wall,
        text="A study with two stools placed against the side wall.",
    )
    _attach_intent_contract(
        scene,
        [
            {
                "relation": "against_wall",
                "subjects": {"category": "stool", "count": 2},
                "targets": {"category": "wall"},
                "source": "explicit_prompt",
                "evidence_span": "two stools placed against the side wall",
            }
        ],
    )

    fixes = improve_furniture_relations(
        scene,
        config=CriticConfig(enabled=True, metrics=("functional_dependency",)),
    )

    assert {fix.object_id for fix in fixes} == {"stool_0", "stool_1"}
    assert {fix.relation_type for fix in fixes} == {"back_against_wall"}
    wall_bounds = west_wall.compute_world_bounds()
    assert wall_bounds is not None
    for stool in stools:
        front = stool.transform.rotation().matrix() @ np.array([0.0, 1.0, 0.0])
        np.testing.assert_allclose(front[:2], [1.0, 0.0], atol=1e-7)
        stool_bounds = stool.compute_world_bounds()
        assert stool_bounds is not None
        assert abs(stool_bounds[0][0] - wall_bounds[1][0] - 0.03) < 1e-7


def test_wall_backed_targets_offer_small_move_from_just_outside_tolerance(
    tmp_path: Path,
) -> None:
    chair = _object(
        "guest_chair_0",
        "guest_chair",
        (-1.85, 0.0, 0.45),
        (0.60, 0.65, 0.90),
        yaw_deg=-90.0,
    )
    scene = _scene(
        tmp_path,
        chair,
        text="A study with a guest chair against the side wall.",
    )

    targets = furniture_relation_repair._wall_backed_targets(
        scene,
        "guest_chair_0",
        "west_wall",
    )

    assert targets
    target_center, target_yaw = targets[0]
    assert target_yaw == pytest.approx(-90.0)
    # The current AABB gap is 0.275m, only 0.025m over the 0.25m contract
    # threshold. Prefer the 0.23m in-tolerance pose over a 0.03m flush pose.
    assert target_center[0] == pytest.approx(-1.895)


def test_wall_backed_repair_uses_door_safe_lateral_pose(tmp_path: Path) -> None:
    sofa = _object("sofa_0", "sofa", (-1.6, -1.1, 0.4), (1.6, 0.9, 0.8), yaw_deg=0.0)
    scene = _scene(
        tmp_path,
        sofa,
        text="A living room with a sofa against the side wall.",
    )
    scene.room_type = "living_room"
    _attach_fixture_contract(scene)
    door = ClearanceOpeningData(
        opening_id="door_0",
        opening_type="door",
        wall_direction="west",
        center_world=[-2.5, -0.2, 1.05],
        width=0.9,
        sill_height=0.0,
        height=2.1,
        clearance_bbox_min=[-2.5, -0.65, 0.0],
        clearance_bbox_max=[-1.7, 0.25, 2.1],
        wall_start=[-2.5, -2.0],
        wall_end=[-2.5, 2.0],
        position_along_wall=1.35,
    )
    scene.room_geometry.openings = [door]
    config = CriticConfig(
        enabled=True,
        metrics=("functional_dependency",),
    )

    fixes = improve_furniture_relations(scene, config=config)

    assert [(fix.object_id, fix.relation_type) for fix in fixes] == [
        ("sofa_0", "back_against_wall")
    ]
    bounds = sofa.compute_world_bounds()
    assert bounds is not None
    # The south side of this doorway is too short for the rotated sofa.  The
    # repair must choose the remaining north-side wall segment, not put the
    # object in the door zone and rely on a later physics repair to move it.
    assert float(bounds[0][1]) >= float(door.clearance_bbox_max[1])
    result = next(
        item
        for item in evaluate_room_scene(scene, config=config, stage="door_safe_after")[
            "results"
        ]
        if item.get("relation_type") == "back_against_wall"
    )
    assert result["label"] == "pass"


def test_wall_backed_repair_keeps_nearby_required_relation(tmp_path: Path) -> None:
    sofa = _object(
        "sofa_0", "sofa", (-1.0, -1.07, 0.38), (1.7, 0.95, 0.76), yaw_deg=90.0
    )
    tv_stand = _object(
        "tv_stand_0", "tv_stand", (0.0, 0.8, 0.255), (1.6, 0.55, 0.51), yaw_deg=90.0
    )
    armchair_0 = _object(
        "armchair_0", "armchair", (-1.9, -1.52, 0.48), (0.7, 0.75, 0.96)
    )
    armchair_1 = _object("armchair_1", "armchair", (-1.9, 0.8, 0.48), (0.7, 0.75, 0.96))
    table = _object(
        "table_0", "table", (-1.0, 0.8, 0.34), (1.31, 0.8, 0.67), yaw_deg=90.0
    )
    rug = _object("rug_0", "rug", (1.0, -1.02, 0.012), (1.8, 1.8, 0.024))
    floor_lamp = _object(
        "floor_lamp_0", "floor_lamp", (-1.9, 1.52, 0.63), (0.4, 0.4, 1.27)
    )
    scene = _scene(
        tmp_path,
        armchair_0,
        armchair_1,
        sofa,
        table,
        rug,
        floor_lamp,
        tv_stand,
        text=(
            "A living room with a sofa against the back wall facing a TV stand "
            "and television on the opposite wall, a coffee table centered "
            "between the sofa and TV stand, two armchairs flanking the coffee "
            "table near each end of the sofa, and a floor lamp beside one "
            "armchair. A small rug lies between the coffee table and TV stand."
        ),
    )
    scene.room_type = "living_room"
    _attach_intent_contract(
        scene,
        [
            {
                "relation": "against_wall",
                "subjects": {"category": "sofa", "count": 1},
                "targets": {"category": "wall"},
                "source": "explicit_prompt",
                "evidence_span": "sofa against the back wall",
            },
            {
                "relation": "against_wall",
                "subjects": {"category": "tv_stand", "count": 1},
                "targets": {"category": "wall"},
                "source": "explicit_prompt",
                "evidence_span": "TV stand on the opposite wall",
            },
            {
                "relation": "near",
                "subjects": {"category": "armchair", "count": 2, "quantifier": "all"},
                "targets": {"category": "sofa", "count": 1},
                "source": "explicit_prompt",
                "evidence_span": "armchairs near each end of the sofa",
            },
        ],
    )
    config = CriticConfig(
        enabled=True,
        metrics=("functional_dependency",),
    )

    fixes = improve_furniture_relations(scene, config=config)

    assert {fix.object_id for fix in fixes} == {"sofa_0", "tv_stand_0"}
    # The sofa was already close to the south wall, but rotating it flush would
    # turn the required armchair-near-sofa relation into a new core failure.
    # A legal in-tolerance wall gap keeps both prompt relations valid.
    assert float(sofa.transform.translation()[1]) > -1.3
    result_by_object = {
        str(item.get("primary_object")): item
        for item in evaluate_room_scene(scene, config=config, stage="nearby_after")[
            "results"
        ]
        if item.get("relation_type") == "back_against_wall"
    }
    assert result_by_object["sofa_0"]["label"] == "pass"
    assert result_by_object["tv_stand_0"]["label"] == "pass"


def test_freestanding_television_repair_restores_media_support(tmp_path: Path) -> None:
    tv_stand = _object("tv_stand_0", "tv_stand", (0.0, 1.1, 0.3), (1.4, 0.5, 0.6))
    television = _object(
        "television_0", "television", (1.6, -0.2, 0.31), (0.9, 0.18, 0.6)
    )
    scene = _scene(
        tmp_path,
        tv_stand,
        television,
        text="A living room with a television placed on a TV stand.",
    )
    scene.room_type = "living_room"
    _attach_fixture_contract(scene)
    config = CriticConfig(
        enabled=True,
        metrics=("functional_dependency",),
    )

    before = next(
        item
        for item in evaluate_room_scene(scene, config=config, stage="media_before")[
            "results"
        ]
        if item.get("relation_type") == "object_on_support"
        and item.get("primary_object") == "television_0"
    )
    assert before["label"] == "fail"
    assert [
        (item.get("primary_object"), item.get("relation_type"))
        for item in unresolved_furniture_relation_failures(scene, config=config)
    ] == [("television_0", "object_on_support")]

    fixes = improve_furniture_relations(scene, config=config)

    assert [(fix.object_id, fix.relation_type) for fix in fixes] == [
        ("television_0", "object_on_support")
    ]
    tv_bounds = television.compute_world_bounds()
    stand_bounds = tv_stand.compute_world_bounds()
    assert tv_bounds is not None and stand_bounds is not None
    np.testing.assert_allclose(
        (tv_bounds[0][:2] + tv_bounds[1][:2]) / 2.0,
        (stand_bounds[0][:2] + stand_bounds[1][:2]) / 2.0,
    )
    assert math.isclose(
        float(tv_bounds[0][2]), float(stand_bounds[1][2]) + 0.01, abs_tol=1e-6
    )
    after = next(
        item
        for item in evaluate_room_scene(scene, config=config, stage="media_after")[
            "results"
        ]
        if item.get("relation_type") == "object_on_support"
        and item.get("primary_object") == "television_0"
    )
    assert after["label"] == "pass"


def test_media_support_repair_moves_pair_clear_of_window(tmp_path: Path) -> None:
    tv_stand = _object("tv_stand_0", "tv_stand", (0.0, 1.7, 0.3), (1.4, 0.5, 0.6))
    television = _object(
        "television_0", "television", (-2.0, -1.2, 0.31), (0.9, 0.18, 0.6)
    )
    scene = _scene(
        tmp_path,
        tv_stand,
        television,
        text="A living room with a television placed on a TV stand.",
    )
    scene.room_type = "living_room"
    scene.room_geometry.openings = [
        ClearanceOpeningData(
            opening_id="window_0",
            opening_type="window",
            wall_direction="north",
            center_world=[0.0, 2.0, 1.5],
            width=1.5,
            sill_height=0.9,
            height=1.2,
            clearance_bbox_min=[-0.75, 1.5, 0.0],
            clearance_bbox_max=[0.75, 2.1, 2.1],
            wall_start=[-2.5, 2.0],
            wall_end=[2.5, 2.0],
            position_along_wall=2.5,
        )
    ]
    _attach_intent_contract(
        scene,
        [
            {
                "relation": "on_top_of",
                "stage": "furniture",
                "strength": "hard",
                "subjects": {"category": "television", "count": 1},
                "targets": {"category": "tv_stand", "count": 1},
                "source": "explicit_prompt",
                "evidence_span": "television placed on a TV stand",
            }
        ],
    )
    config = CriticConfig(
        enabled=True,
        metrics=("functional_dependency", "interaction_clearance"),
    )

    fixes = improve_furniture_relations(scene, config=config)

    assert {fix.object_id for fix in fixes} == {"television_0", "tv_stand_0"}
    payload = evaluate_room_scene(scene, config=config, stage="media_window_after")
    result_by_id = {result["check_id"]: result for result in payload["results"]}
    support_result = next(
        result
        for result in payload["results"]
        if result.get("relation_type") == "object_on_support"
        and result.get("primary_object") == "television_0"
    )
    assert support_result["label"] == "pass"
    assert result_by_id["window_clearance__window_0"]["label"] == "pass"


def test_media_support_pair_preserves_hard_between_dependent(tmp_path: Path) -> None:
    tv_stand = _object("tv_stand_0", "tv_stand", (0.0, 1.7, 0.3), (1.4, 0.5, 0.6))
    television = _object(
        "television_0", "television", (-2.0, -1.2, 0.31), (0.9, 0.18, 0.6)
    )
    sofa = _object("sofa_0", "sofa", (0.0, -1.2, 0.4), (2.0, 0.8, 0.8))
    coffee_table = _object(
        "coffee_table_0", "coffee_table", (0.0, 0.25, 0.2), (1.2, 0.7, 0.4)
    )
    rug = _object("rug_0", "rug", (0.0, 0.75, 0.01), (1.4, 1.4, 0.02))
    scene = _scene(
        tmp_path,
        tv_stand,
        television,
        sofa,
        coffee_table,
        rug,
        text="A living room with a television placed on a TV stand.",
    )
    scene.room_type = "living_room"
    scene.room_geometry.openings = [
        ClearanceOpeningData(
            opening_id="window_0",
            opening_type="window",
            wall_direction="north",
            center_world=[0.0, 2.0, 1.5],
            width=1.5,
            sill_height=0.9,
            height=1.2,
            clearance_bbox_min=[-0.75, 1.5, 0.0],
            clearance_bbox_max=[0.75, 2.1, 2.1],
            wall_start=[-2.5, 2.0],
            wall_end=[2.5, 2.0],
            position_along_wall=2.5,
        )
    ]
    _attach_intent_contract(
        scene,
        [
            {
                "relation": "on_top_of",
                "stage": "furniture",
                "strength": "hard",
                "subjects": {"category": "television", "count": 1},
                "targets": {"category": "tv_stand", "count": 1},
                "source": "explicit_prompt",
                "evidence_span": "television placed on a TV stand",
            },
            {
                "relation": "between",
                "stage": "furniture",
                "strength": "hard",
                "subjects": {"category": "rug", "count": 1},
                "targets": {
                    "category": "coffee_table",
                    "count": 1,
                    "secondary_category": "tv_stand",
                    "secondary_count": 1,
                },
                "source": "explicit_prompt",
                "evidence_span": "rug lies between the coffee table and TV stand",
            },
            {
                "relation": "centered_between",
                "stage": "furniture",
                "strength": "hard",
                "subjects": {"category": "coffee_table", "count": 1},
                "targets": {
                    "category": "sofa",
                    "count": 1,
                    "secondary_category": "tv_stand",
                    "secondary_count": 1,
                },
                "source": "explicit_prompt",
                "evidence_span": "coffee table centered between the sofa and TV stand",
            },
        ],
    )
    config = CriticConfig(
        enabled=True,
        metrics=("functional_dependency", "interaction_clearance"),
    )
    old_rug_center = (
        rug.compute_world_bounds()[0][:2] + rug.compute_world_bounds()[1][:2]
    ) / 2.0
    old_table_center = (
        coffee_table.compute_world_bounds()[0][:2]
        + coffee_table.compute_world_bounds()[1][:2]
    ) / 2.0

    fixes = improve_furniture_relations(scene, config=config)

    assert {fix.object_id for fix in fixes} == {
        "television_0",
        "tv_stand_0",
        "rug_0",
        "coffee_table_0",
    }
    new_rug_center = (
        rug.compute_world_bounds()[0][:2] + rug.compute_world_bounds()[1][:2]
    ) / 2.0
    new_table_center = (
        coffee_table.compute_world_bounds()[0][:2]
        + coffee_table.compute_world_bounds()[1][:2]
    ) / 2.0
    assert not np.allclose(new_rug_center, old_rug_center)
    assert not np.allclose(new_table_center, old_table_center)
    payload = evaluate_room_scene(
        scene, config=config, stage="media_window_between_after"
    )
    result_by_id = {result["check_id"]: result for result in payload["results"]}
    support_result = next(
        result
        for result in payload["results"]
        if result.get("relation_type") == "object_on_support"
        and result.get("primary_object") == "television_0"
    )
    assert support_result["label"] == "pass"
    assert result_by_id["window_clearance__window_0"]["label"] == "pass"
    assert (
        next(
            result
            for result in payload["results"]
            if result.get("relation_type") == "between_alignment"
            and result.get("primary_object") == "rug_0"
        )["label"]
        == "pass"
    )
    assert (
        next(
            result
            for result in payload["results"]
            if result.get("relation_type") == "centered_between_alignment"
            and result.get("primary_object") == "coffee_table_0"
        )["label"]
        == "pass"
    )


def test_floor_plant_support_failure_is_not_repaired_or_hard_gated(
    tmp_path: Path, monkeypatch
) -> None:
    plant = _object("large_plant_0", "large_plant", (-1.5, 0.8, 0.4), (0.6, 0.6, 0.8))
    sofa = _object("sofa_0", "sofa", (0.0, -1.2, 0.4), (1.8, 0.8, 0.8))
    scene = _scene(tmp_path, plant, sofa, text="A living room with a floor plant.")
    scene.room_type = "living_room"
    config = CriticConfig(enabled=True, metrics=("functional_dependency",))
    payload = {
        "results": [
            {
                "check_id": "support__large_plant_0_sofa_0",
                "label": "fail",
                "scoring_tier": "core",
                "relation_type": "object_on_support",
                "primary_object": "large_plant_0",
                "selected_related_objects": ["sofa_0"],
                "diagnostics": {},
            }
        ]
    }
    monkeypatch.setattr(furniture_relation_repair, "_evaluate", lambda *_: payload)

    assert unresolved_furniture_relation_failures(scene, config=config) == []
    assert improve_furniture_relations(scene, config=config) == []


def test_implicit_media_support_failure_is_not_repaired_or_hard_gated(
    tmp_path: Path, monkeypatch
) -> None:
    tv_stand = _object("tv_stand_0", "tv_stand", (0.0, 1.1, 0.3), (1.4, 0.5, 0.6))
    television = _object(
        "television_0", "television", (1.6, -0.2, 0.31), (0.9, 0.18, 0.6)
    )
    scene = _scene(
        tmp_path,
        tv_stand,
        television,
        text="A living room with a television and a TV stand.",
    )
    scene.room_type = "living_room"
    config = CriticConfig(
        enabled=True,
        metrics=("functional_dependency",),
    )
    payload = {
        "case_pack": {"checks": []},
        "results": [
            {
                "check_id": "fd_television_0_tv_stand_0_object_on_support",
                "label": "fail",
                "scoring_tier": "core",
                "relation_type": "object_on_support",
                "primary_object": "television_0",
                "selected_related_objects": ["tv_stand_0"],
                "diagnostics": {},
            }
        ],
    }
    monkeypatch.setattr(furniture_relation_repair, "_evaluate", lambda *_: payload)

    assert unresolved_furniture_relation_failures(scene, config=config) == []
    assert improve_furniture_relations(scene, config=config) == []


def test_hard_furniture_inventory_contract_is_stage_gated(
    tmp_path: Path, monkeypatch
) -> None:
    scene = _scene(tmp_path, text="An office with two office chairs.")
    config = CriticConfig(enabled=True, metrics=("functional_dependency",))
    failure = {
        "check_id": "intent_required_office_chair",
        "label": "fail",
        "contract_state": "failed",
        "scoring_tier": "core",
        "relation_type": "required_count",
        "primary_object": "office_chair",
        "evidence": {
            "intent_constraint": {
                "relation": "required_count",
                "stage": "furniture",
                "strength": "hard",
            }
        },
    }
    monkeypatch.setattr(
        furniture_relation_repair,
        "_evaluate",
        lambda *_args, **_kwargs: {"case_pack": {}, "results": [failure]},
    )

    assert unresolved_furniture_relation_failures(scene, config=config) == [failure]


def test_manipuland_support_contract_is_deferred_from_furniture_gate(
    tmp_path: Path, monkeypatch
) -> None:
    scene = _scene(tmp_path, text="A study with a monitor on a desk.")
    config = CriticConfig(enabled=True, metrics=("functional_dependency",))
    failure = {
        "check_id": "intent_monitor_on_desk",
        "label": "fail",
        "contract_state": "failed",
        "scoring_tier": "core",
        "relation_type": "object_on_support",
        "primary_object": "computer_monitor",
        "selected_related_objects": ["study_desk_0"],
        "evidence": {
            "intent_constraint": {
                "relation": "on_top_of",
                "stage": "manipuland",
                "strength": "hard",
            }
        },
    }
    monkeypatch.setattr(
        furniture_relation_repair,
        "_evaluate",
        lambda *_args, **_kwargs: {"case_pack": {}, "results": [failure]},
    )

    assert unresolved_furniture_relation_failures(scene, config=config) == []


def test_repairs_work_seat_to_in_room_side_when_other_side_is_outside(
    tmp_path: Path,
) -> None:
    desk = _object("work_surface", "office_desk", (0.0, -1.55, 0.4), (1.4, 0.8, 0.8))
    chair = _object("task_seat", "office_chair", (0.0, -0.3, 0.45), (0.6, 0.6, 0.9))
    old_transform = chair.transform.GetAsMatrix4().copy()
    scene = _scene(tmp_path, desk, chair, text="A study with a desk and chair")

    fixes = improve_furniture_relations(
        scene,
        config=CriticConfig(enabled=True, metrics=("functional_dependency",)),
        max_translation_m=3.0,
    )

    assert [fix.object_id for fix in fixes] == ["task_seat"]
    assert not np.allclose(chair.transform.GetAsMatrix4(), old_transform)
    assert -2.0 < chair.transform.translation()[1] < 0.0


def test_relation_repair_rejects_candidate_failing_hard_validator(
    tmp_path: Path,
) -> None:
    desk = _object("work_surface", "office_desk", (0.0, -1.55, 0.4), (1.4, 0.8, 0.8))
    chair = _object("task_seat", "office_chair", (0.0, -0.3, 0.45), (0.6, 0.6, 0.9))
    scene = _scene(tmp_path, desk, chair, text="A study with a desk and chair")
    old_transform = chair.transform.GetAsMatrix4().copy()

    fixes = improve_furniture_relations(
        scene,
        config=CriticConfig(enabled=True, metrics=("functional_dependency",)),
        candidate_validator=lambda _: False,
    )

    assert fixes == []
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


def test_candidate_score_allows_soft_degradation_when_hard_failures_drop() -> None:
    baseline = {
        "results": [
            {
                "check_id": "dining_slots",
                "label": "fail",
                "relation_type": "edge_distribution",
            },
            {
                "check_id": "chair_facing",
                "label": "fail",
                "relation_type": "furniture_faces_furniture",
            },
            {
                "check_id": "table_access",
                "label": "pass",
                "relation_type": "spatial_accessibility",
            },
        ]
    }
    candidate = {
        "results": [
            {
                "check_id": "dining_slots",
                "label": "pass",
                "relation_type": "edge_distribution",
            },
            {
                "check_id": "chair_facing",
                "label": "pass",
                "relation_type": "furniture_faces_furniture",
            },
            {
                "check_id": "table_access",
                "label": "degraded",
                "relation_type": "spatial_accessibility",
            },
        ]
    }

    assert _candidate_improves(
        baseline,
        candidate,
        baseline_score=_score_payload(baseline),
        check_id="dining_slots",
    )


def test_candidate_score_rejects_soft_regression_without_hard_fail_reduction() -> None:
    baseline = {
        "results": [
            {
                "check_id": "dining_slots",
                "label": "fail",
                "relation_type": "edge_distribution",
            },
            {
                "check_id": "table_access",
                "label": "pass",
                "relation_type": "spatial_accessibility",
            },
        ]
    }
    candidate = {
        "results": [
            {
                "check_id": "dining_slots",
                "label": "fail",
                "relation_type": "edge_distribution",
            },
            {
                "check_id": "table_access",
                "label": "degraded",
                "relation_type": "spatial_accessibility",
            },
        ]
    }

    assert not _candidate_improves(
        baseline,
        candidate,
        baseline_score=_score_payload(baseline),
        check_id="dining_slots",
    )


def test_multiple_failed_work_seats_get_atomic_candidate_before_individuals() -> None:
    targets = [
        _RepairTarget(
            f"student_chair_{index}",
            "seating_to_work_surface",
            f"seat_{index}",
            (float(index), 1.0),
            0.0,
        )
        for index in range(3)
    ]

    candidates = _prioritize_coordinated_seating_targets(targets)

    assert len(candidates) == 4
    assert candidates[0].object_id == "student_chair_0"
    assert {pose.object_id for pose in candidates[0].member_poses} == {
        "student_chair_0",
        "student_chair_1",
        "student_chair_2",
    }
    assert candidates[1:] == targets


class _NonCopyableMesh:
    def __deepcopy__(self, _memo):
        raise AssertionError("candidate evaluation must not deepcopy support meshes")


def _relation_payload(*, label: str, new_issue: bool = False) -> dict:
    results = [
        {
            "check_id": "seat_to_surface",
            "label": label,
            "relation_type": "seating_to_work_surface",
            "primary_object": "task_seat",
            "diagnostics": {
                "seat_surface_assignment": {
                    "target_slot": {
                        "center_xy": [1.0, 0.0],
                        "yaw_deg": 0.0,
                    }
                }
            },
        }
    ]
    if new_issue:
        results.append(
            {
                "check_id": "new_issue",
                "label": "fail",
                "relation_type": "collision",
            }
        )
    return {"results": results}


def _scene_with_supported_child(
    tmp_path: Path,
) -> tuple[RoomScene, SceneObject, SceneObject, SupportSurface]:
    seat = _object("task_seat", "office_chair", (0.0, 0.0, 0.45), (0.6, 0.6, 0.9))
    surface = SupportSurface(
        surface_id=UniqueID("seat_surface"),
        bounding_box_min=np.array([-0.2, -0.2, 0.0]),
        bounding_box_max=np.array([0.2, 0.2, 0.0]),
        transform=RigidTransform(p=[0.0, 0.0, 0.9]),
        mesh=_NonCopyableMesh(),
    )
    seat.support_surfaces = [surface]
    child = _object(
        "supported_item",
        "notebook",
        (0.0, 0.0, 0.95),
        (0.2, 0.2, 0.1),
        object_type=ObjectType.MANIPULAND,
    )
    child.placement_info = PlacementInfo(
        parent_surface_id=surface.surface_id,
        position_2d=np.zeros(2),
        rotation_2d=0.0,
    )
    return (
        _scene(tmp_path, seat, child, text="A study with a desk and chair"),
        seat,
        child,
        surface,
    )


def test_rejected_candidate_restores_pose_without_copying_support_mesh(
    tmp_path: Path, monkeypatch
) -> None:
    scene, seat, child, surface = _scene_with_supported_child(tmp_path)
    original_surface_list = seat.support_surfaces
    old_seat = seat.transform.GetAsMatrix4().copy()
    old_child = child.transform.GetAsMatrix4().copy()
    old_surface = surface.transform.GetAsMatrix4().copy()

    evaluations = iter(
        [
            _relation_payload(label="fail"),
            _relation_payload(label="pass", new_issue=True),
        ]
    )

    def fake_evaluate(current_scene, _config):
        payload = next(evaluations)
        if payload["results"][0]["label"] == "pass":
            current_scene.objects[seat.object_id].support_surfaces.clear()
        return payload

    monkeypatch.setattr(
        furniture_relation_repair,
        "_evaluate",
        fake_evaluate,
    )

    fixes = improve_furniture_relations(
        scene,
        config=CriticConfig(enabled=True, metrics=("functional_dependency",)),
        max_repairs=1,
    )

    assert fixes == []
    np.testing.assert_allclose(seat.transform.GetAsMatrix4(), old_seat)
    np.testing.assert_allclose(child.transform.GetAsMatrix4(), old_child)
    np.testing.assert_allclose(surface.transform.GetAsMatrix4(), old_surface)
    assert seat.support_surfaces is original_surface_list
    assert seat.support_surfaces == [surface]
    assert surface.mesh is seat.support_surfaces[0].mesh


def test_accepted_candidate_moves_support_surface_and_child_together(
    tmp_path: Path, monkeypatch
) -> None:
    scene, seat, child, surface = _scene_with_supported_child(tmp_path)
    old_child_x = float(child.transform.translation()[0])
    old_surface_x = float(surface.transform.translation()[0])

    evaluations = iter(
        [
            _relation_payload(label="fail"),
            _relation_payload(label="pass"),
        ]
    )
    monkeypatch.setattr(
        furniture_relation_repair,
        "_evaluate",
        lambda _scene, _config: next(evaluations),
    )
    fixes = improve_furniture_relations(
        scene,
        config=CriticConfig(enabled=True, metrics=("functional_dependency",)),
        max_repairs=1,
    )

    assert [fix.object_id for fix in fixes] == ["task_seat"]
    assert abs(float(seat.transform.translation()[0]) - 1.0) < 1e-8
    assert abs(float(child.transform.translation()[0]) - old_child_x - 1.0) < 1e-8
    assert abs(float(surface.transform.translation()[0]) - old_surface_x - 1.0) < 1e-8
    assert surface.mesh is seat.support_surfaces[0].mesh


def test_dining_chair_repair_does_not_move_table_support_children(
    tmp_path: Path, monkeypatch
) -> None:
    table = _object(
        "dining_table",
        "dining_table",
        (0.0, 0.0, 0.4),
        (2.0, 0.8, 0.8),
    )
    table_surface = SupportSurface(
        surface_id=UniqueID("table_surface"),
        bounding_box_min=np.array([-0.8, -0.3, 0.0]),
        bounding_box_max=np.array([0.8, 0.3, 0.0]),
        transform=RigidTransform(p=[0.0, 0.0, 0.8]),
        mesh=_NonCopyableMesh(),
    )
    table.support_surfaces = [table_surface]
    tableware = _object(
        "place_setting",
        "plate",
        (0.0, 0.0, 0.85),
        (0.2, 0.2, 0.1),
        object_type=ObjectType.MANIPULAND,
    )
    tableware.placement_info = PlacementInfo(
        parent_surface_id=table_surface.surface_id,
        position_2d=np.zeros(2),
        rotation_2d=0.0,
    )
    chair = _object(
        "dining_chair",
        "dining_chair",
        (0.0, -1.2, 0.45),
        (0.5, 0.5, 0.9),
    )
    scene = _scene(tmp_path, table, chair, tableware)
    old_table = table.transform.GetAsMatrix4().copy()
    old_surface = table_surface.transform.GetAsMatrix4().copy()
    old_tableware = tableware.transform.GetAsMatrix4().copy()

    evaluations = iter(
        [
            {
                "results": [
                    {
                        "check_id": "dining_slots",
                        "label": "fail",
                        "relation_type": "edge_distribution",
                        "diagnostics": {
                            "seat_slots": [
                                {
                                    "seat_id": "dining_chair",
                                    "aligned": False,
                                    "target_center_xy_m": [1.0, 0.0],
                                }
                            ]
                        },
                    }
                ]
            },
            {
                "results": [
                    {
                        "check_id": "dining_slots",
                        "label": "pass",
                        "relation_type": "edge_distribution",
                        "diagnostics": {"seat_slots": []},
                    }
                ]
            },
        ]
    )
    monkeypatch.setattr(
        furniture_relation_repair,
        "_evaluate",
        lambda _scene, _config: next(evaluations),
    )

    fixes = improve_furniture_relations(
        scene,
        config=CriticConfig(enabled=True, metrics=("functional_dependency",)),
        max_repairs=1,
    )

    assert [fix.object_id for fix in fixes] == ["dining_chair"]
    np.testing.assert_allclose(table.transform.GetAsMatrix4(), old_table)
    np.testing.assert_allclose(table_surface.transform.GetAsMatrix4(), old_surface)
    np.testing.assert_allclose(tableware.transform.GetAsMatrix4(), old_tableware)


def test_dining_chair_repair_uses_outward_slot_when_exact_pose_blocks_access(
    tmp_path: Path, monkeypatch
) -> None:
    table = _object(
        "dining_table",
        "dining_table",
        (0.0, 0.0, 0.4),
        (2.0, 0.8, 0.8),
    )
    chair = _object(
        "dining_chair",
        "dining_chair",
        (-1.8, 1.8, 0.45),
        (0.5, 0.5, 0.9),
    )
    scene = _scene(tmp_path, table, chair)
    failed_distribution = {
        "check_id": "dining_slots",
        "label": "fail",
        "relation_type": "edge_distribution",
        "diagnostics": {
            "seat_slots": [
                {
                    "seat_id": "dining_chair",
                    "aligned": False,
                    "facing_aligned": False,
                    "target_center_xy_m": [0.0, 1.0],
                    "facing_target_xy_m": [0.0, 0.0],
                    "current_front_xy": [0.0, -1.0],
                    "allowed_normal_deviation_m": 0.12,
                }
            ]
        },
    }
    passed_distribution = {
        "check_id": "dining_slots",
        "label": "pass",
        "relation_type": "edge_distribution",
        "diagnostics": {"seat_slots": []},
    }
    evaluations = iter(
        [
            {"results": [failed_distribution]},
            {
                "results": [
                    passed_distribution,
                    {
                        "check_id": "spatial_accessibility__dining_chair",
                        "label": "degraded",
                        "relation_type": "spatial_accessibility",
                    },
                ]
            },
            {"results": [passed_distribution]},
        ]
    )
    monkeypatch.setattr(
        furniture_relation_repair,
        "_evaluate",
        lambda _scene, _config: next(evaluations),
    )

    fixes = improve_furniture_relations(
        scene,
        config=CriticConfig(enabled=True, metrics=("functional_dependency",)),
        max_repairs=1,
    )

    assert [fix.object_id for fix in fixes] == ["dining_chair"]
    np.testing.assert_allclose(chair.transform.translation()[:2], [0.0, 1.06])


def test_room_center_repair_coordinates_wall_storage_access(
    tmp_path: Path, monkeypatch
) -> None:
    table = _object(
        "dining_table",
        "dining_table",
        (-1.0, 0.0, 0.4),
        (1.8, 1.0, 0.8),
    )
    table_surface = SupportSurface(
        surface_id=UniqueID("table_surface"),
        bounding_box_min=np.array([-0.8, -0.4, 0.0]),
        bounding_box_max=np.array([0.8, 0.4, 0.0]),
        transform=RigidTransform(p=[-1.0, 0.0, 0.8]),
        mesh=_NonCopyableMesh(),
    )
    table.support_surfaces = [table_surface]
    tableware = _object(
        "place_setting",
        "plate",
        (-1.0, 0.0, 0.85),
        (0.2, 0.2, 0.1),
        object_type=ObjectType.MANIPULAND,
    )
    tableware.placement_info = PlacementInfo(
        parent_surface_id=table_surface.surface_id,
        position_2d=np.zeros(2),
        rotation_2d=0.0,
    )
    chair = _object(
        "dining_chair",
        "dining_chair",
        (-1.0, -1.0, 0.45),
        (0.5, 0.5, 0.9),
    )
    sideboard = _object(
        "sideboard",
        "sideboard",
        (0.0, -1.7, 0.4),
        (1.2, 0.4, 0.8),
    )
    scene = _scene(
        tmp_path,
        table,
        chair,
        sideboard,
        tableware,
        text="A dining room with a dining table in the center and one dining chair.",
    )
    evaluations = iter(
        [
            {
                "results": [
                    {
                        "check_id": "table_center",
                        "label": "fail",
                        "relation_type": "room_center_alignment",
                        "primary_object": "dining_table",
                        "related_objects": ["dining_chair"],
                        "diagnostics": {"room_center_xy": [0.0, 0.0]},
                    }
                ]
            },
            {
                "results": [
                    {
                        "check_id": "table_center",
                        "label": "pass",
                        "relation_type": "room_center_alignment",
                        "primary_object": "dining_table",
                        "related_objects": ["dining_chair"],
                        "diagnostics": {"room_center_xy": [0.0, 0.0]},
                    }
                ]
            },
        ]
    )
    monkeypatch.setattr(
        furniture_relation_repair,
        "_evaluate",
        lambda _scene, _config: next(evaluations),
    )
    coordinated_calls: list[bool] = []

    def fake_access_repair(current_scene, **kwargs):
        coordinated_calls.append(bool(kwargs.get("repair_degraded")))
        moved = (
            current_scene.objects[sideboard.object_id].transform.translation().copy()
        )
        moved[0] = 0.8
        current_scene.move_object(
            object_id=sideboard.object_id,
            new_transform=RigidTransform(R=sideboard.transform.rotation(), p=moved),
        )
        return []

    monkeypatch.setattr(
        furniture_relation_repair,
        "improve_storage_front_access",
        fake_access_repair,
    )

    fixes = improve_furniture_relations(
        scene,
        config=CriticConfig(enabled=True, metrics=("functional_dependency",)),
        max_repairs=1,
    )

    assert [fix.object_id for fix in fixes] == ["dining_table"]
    assert coordinated_calls == [True]
    np.testing.assert_allclose(table.transform.translation()[:2], [0.0, 0.0])
    np.testing.assert_allclose(table_surface.transform.translation()[:2], [0.0, 0.0])
    np.testing.assert_allclose(tableware.transform.translation()[:2], [0.0, 0.0])
    np.testing.assert_allclose(chair.transform.translation()[:2], [0.0, -1.0])
    np.testing.assert_allclose(sideboard.transform.translation()[:2], [0.8, -1.7])


def test_room_center_repair_does_not_move_nearby_sofa(
    tmp_path: Path, monkeypatch
) -> None:
    rug = _object("rug_0", "rug", (0.0, 0.4, 0.015), (2.5, 2.5, 0.03))
    sofa = _object(
        "two_seater_sofa_0",
        "two_seater_sofa",
        (0.0, -1.48, 0.44),
        (1.54, 0.9, 0.87),
    )
    scene = _scene(
        tmp_path,
        rug,
        sofa,
        text="A living room with a rug in the center of the room.",
    )
    evaluations = iter(
        [
            {
                "results": [
                    {
                        "check_id": "rug_center",
                        "label": "fail",
                        "relation_type": "room_center_alignment",
                        "primary_object": "rug_0",
                        "related_objects": [],
                        "diagnostics": {"room_center_xy": [0.0, 0.0]},
                    }
                ]
            },
            {
                "results": [
                    {
                        "check_id": "rug_center",
                        "label": "pass",
                        "relation_type": "room_center_alignment",
                        "primary_object": "rug_0",
                        "related_objects": [],
                        "diagnostics": {"room_center_xy": [0.0, 0.0]},
                    }
                ]
            },
        ]
    )
    monkeypatch.setattr(
        furniture_relation_repair,
        "_evaluate",
        lambda _scene, _config: next(evaluations),
    )

    old_sofa_transform = sofa.transform.GetAsMatrix4().copy()
    fixes = improve_furniture_relations(
        scene,
        config=CriticConfig(enabled=True, metrics=("functional_dependency",)),
        max_repairs=1,
    )

    assert [fix.object_id for fix in fixes] == ["rug_0"]
    np.testing.assert_allclose(rug.transform.translation()[:2], [0.0, 0.0])
    np.testing.assert_allclose(sofa.transform.GetAsMatrix4(), old_sofa_transform)


def test_room_center_repair_prefers_nearest_compliant_pose(
    tmp_path: Path, monkeypatch
) -> None:
    rug = _object("rug_0", "rug", (0.05, 0.325, 0.015), (2.0, 2.0, 0.03))
    scene = _scene(tmp_path, rug, text="A living room with a rug in the middle.")
    failed_center = {
        "check_id": "rug_center",
        "label": "fail",
        "relation_type": "room_center_alignment",
        "primary_object": "rug_0",
        "related_objects": [],
        "diagnostics": {
            "room_center_xy": [0.0, 0.0],
            "offset_m": math.hypot(0.05, 0.325),
            "allowed_offset_m": 0.32,
        },
    }
    passed_center = {**failed_center, "label": "pass"}
    evaluations = iter(
        [
            {"results": [failed_center]},
            {"results": [passed_center]},
            {"results": [passed_center]},
        ]
    )
    monkeypatch.setattr(
        furniture_relation_repair,
        "_evaluate",
        lambda _scene, _config: next(evaluations),
    )
    monkeypatch.setattr(
        furniture_relation_repair,
        "improve_storage_front_access",
        lambda *_args, **_kwargs: [],
    )

    fixes = improve_furniture_relations(
        scene,
        config=CriticConfig(enabled=True, metrics=("functional_dependency",)),
        max_repairs=1,
    )

    assert [(fix.object_id, fix.relation_type) for fix in fixes] == [
        ("rug_0", "room_center_alignment")
    ]
    # The repair enters the centered-in-room tolerance, instead of moving a
    # broad object all the way to the room origin and risking a clearance loss.
    np.testing.assert_allclose(
        math.hypot(*rug.transform.translation()[:2]), 0.31, atol=1e-7
    )
    np.testing.assert_allclose(
        rug.transform.translation()[:2] / math.hypot(*rug.transform.translation()[:2]),
        np.array([0.05, 0.325]) / math.hypot(0.05, 0.325),
    )


def test_repairs_explicit_front_alignment_without_moving_centered_rug(
    tmp_path: Path,
) -> None:
    sofa = _object(
        "sofa_0",
        "sofa",
        (-1.57, -1.445, 0.38),
        (1.7, 0.95, 0.76),
        yaw_deg=0.0,
    )
    rug = _object("rug_0", "rug", (0.0, 0.0, 0.015), (1.8, 1.8, 0.03))
    scene = _scene(
        tmp_path,
        sofa,
        rug,
        text=(
            "A living room with a two-seater sofa against the wall, a square rug "
            "in the middle in front of the sofa."
        ),
    )
    scene.room_type = "living_room"
    scene.scenebenchmark_intent_contract = {
        "schema_version": SCHEMA_VERSION,
        "prompt_sha256": hashlib.sha256(
            " ".join(scene.text_description.split()).encode("utf-8")
        ).hexdigest(),
        "constraints": [
            {
                "constraint_id": "sofa_wall",
                "relation": "against_wall",
                "subjects": {"category": "sofa", "count": 1},
                "targets": {"category": "wall"},
                "source": "explicit_prompt",
                "strength": "hard",
            },
            {
                "constraint_id": "rug_center",
                "relation": "centered_in_room",
                "subjects": {"category": "rug", "count": 1},
                "targets": {"category": "room"},
                "source": "explicit_prompt",
                "strength": "hard",
            },
            {
                "constraint_id": "rug_front",
                "relation": "in_front_of",
                "subjects": {"category": "rug", "count": 1},
                "targets": {"category": "sofa", "count": 1},
                "source": "explicit_prompt",
                "strength": "hard",
            },
        ],
    }
    _complete_fixture_contract_evidence(scene)
    config = CriticConfig(
        enabled=True,
        metrics=("functional_dependency",),
    )

    before = evaluate_room_scene(scene, config=config, stage="front_alignment_before")
    before_result = next(
        item
        for item in before["results"]
        if item.get("relation_type") == "front_axis_alignment"
    )
    assert before_result["label"] == "fail"

    fixes = improve_furniture_relations(scene, config=config)

    assert [(fix.object_id, fix.relation_type) for fix in fixes] == [
        ("sofa_0", "front_axis_alignment")
    ]
    np.testing.assert_allclose(sofa.transform.translation()[:2], [0.0, -1.445])
    np.testing.assert_allclose(rug.transform.translation()[:2], [0.0, 0.0])

    after = evaluate_room_scene(scene, config=config, stage="front_alignment_after")
    results = {
        str(item.get("relation_type")): item
        for item in after["results"]
        if item.get("relation_type") in {"front_axis_alignment", "back_against_wall"}
    }
    assert results["front_axis_alignment"]["label"] == "pass"
    assert results["back_against_wall"]["label"] == "pass"


def test_front_alignment_moves_explicit_near_dependents_with_sofa(
    tmp_path: Path,
) -> None:
    sofa = _object(
        "sofa_0",
        "sofa",
        (-1.57, -1.445, 0.38),
        (1.7, 0.95, 0.76),
        yaw_deg=0.0,
    )
    rug = _object("rug_0", "rug", (0.0, 0.0, 0.015), (1.8, 1.8, 0.03))
    plants = [
        _object("plant_0", "plant", (-1.9, 0.8, 0.45), (0.6, 0.6, 0.9)),
        _object("plant_1", "plant", (-2.14, 1.52, 0.45), (0.6, 0.6, 0.9)),
    ]
    scene = _scene(
        tmp_path,
        sofa,
        rug,
        *plants,
        text=(
            "A living room with a two-seater sofa against the wall, a square rug "
            "in the middle in front of the sofa, and two large plants near the sofa."
        ),
    )
    scene.room_type = "living_room"
    scene.scenebenchmark_intent_contract = {
        "schema_version": SCHEMA_VERSION,
        "prompt_sha256": hashlib.sha256(
            " ".join(scene.text_description.split()).encode("utf-8")
        ).hexdigest(),
        "constraints": [
            {
                "constraint_id": "sofa_wall",
                "relation": "against_wall",
                "subjects": {"category": "sofa", "count": 1},
                "targets": {"category": "wall"},
                "source": "explicit_prompt",
                "strength": "hard",
            },
            {
                "constraint_id": "rug_center",
                "relation": "centered_in_room",
                "subjects": {"category": "rug", "count": 1},
                "targets": {"category": "room"},
                "source": "explicit_prompt",
                "strength": "hard",
            },
            {
                "constraint_id": "rug_front",
                "relation": "in_front_of",
                "subjects": {"category": "rug", "count": 1},
                "targets": {"category": "sofa", "count": 1},
                "source": "explicit_prompt",
                "strength": "hard",
            },
            {
                "constraint_id": "plants_near_sofa",
                "relation": "near",
                "subjects": {"category": "plant", "count": 2},
                "targets": {"category": "sofa", "count": 1},
                "source": "explicit_prompt",
                "strength": "hard",
            },
        ],
    }
    _complete_fixture_contract_evidence(scene)
    config = CriticConfig(enabled=True, metrics=("functional_dependency",))

    before = evaluate_room_scene(scene, config=config, stage="near_group_before")
    assert any(
        item.get("relation_type") == "front_axis_alignment"
        and item.get("label") == "fail"
        for item in before["results"]
    )

    fixes = improve_furniture_relations(scene, config=config)

    assert {fix.object_id for fix in fixes} == {"sofa_0", "plant_0", "plant_1"}
    assert {fix.relation_type for fix in fixes} == {"front_axis_alignment"}
    np.testing.assert_allclose(sofa.transform.translation()[:2], [0.0, -1.445])
    np.testing.assert_allclose(rug.transform.translation()[:2], [0.0, 0.0])

    after = evaluate_room_scene(scene, config=config, stage="near_group_after")
    hard_results = [
        item
        for item in after["results"]
        if item.get("relation_type")
        in {"back_against_wall", "front_axis_alignment", "generic_near_relation"}
        or (
            (item.get("evidence") or {}).get("intent_constraint", {}).get("relation")
            == "centered_in_room"
        )
    ]
    assert hard_results
    assert {item["label"] for item in hard_results} == {"pass"}


def test_front_alignment_does_not_reintroduce_door_sweep_blockage(
    tmp_path: Path,
) -> None:
    sofa = _object(
        "sofa_0",
        "sofa",
        (-1.945, -1.3, 0.38),
        (1.2, 0.95, 0.76),
        yaw_deg=-90.0,
    )
    rug = _object("rug_0", "rug", (0.0, -0.2, 0.015), (1.8, 1.8, 0.03))
    scene = _scene(
        tmp_path,
        sofa,
        rug,
        text=(
            "A living room with a two-seater sofa against the wall, a square rug "
            "in the middle in front of the sofa."
        ),
    )
    scene.room_type = "living_room"
    scene.room_geometry.openings = [
        ClearanceOpeningData(
            opening_id="door_0",
            opening_type="door",
            wall_direction="west",
            center_world=[-2.5, -0.2, 1.05],
            width=0.9,
            sill_height=0.0,
            height=2.1,
            clearance_bbox_min=[-2.5, -0.65, 0.0],
            clearance_bbox_max=[-1.7, 0.25, 2.1],
            wall_start=[-2.5, -2.0],
            wall_end=[-2.5, 2.0],
            position_along_wall=1.35,
        )
    ]
    scene.scenebenchmark_intent_contract = {
        "schema_version": SCHEMA_VERSION,
        "prompt_sha256": hashlib.sha256(
            " ".join(scene.text_description.split()).encode("utf-8")
        ).hexdigest(),
        "constraints": [
            {
                "constraint_id": "sofa_wall",
                "relation": "against_wall",
                "subjects": {"category": "sofa", "count": 1},
                "targets": {"category": "wall"},
                "source": "explicit_prompt",
                "strength": "hard",
            },
            {
                "constraint_id": "rug_center",
                "relation": "centered_in_room",
                "subjects": {"category": "rug", "count": 1},
                "targets": {"category": "room"},
                "source": "explicit_prompt",
                "strength": "hard",
            },
            {
                "constraint_id": "rug_front",
                "relation": "in_front_of",
                "subjects": {"category": "rug", "count": 1},
                "targets": {"category": "sofa", "count": 1},
                "source": "explicit_prompt",
                "strength": "hard",
            },
        ],
    }
    _complete_fixture_contract_evidence(scene)
    config = CriticConfig(
        enabled=True,
        metrics=("functional_dependency", "interaction_clearance"),
    )

    before = evaluate_room_scene(scene, config=config, stage="door_safe_before")
    before_by_id = {result["check_id"]: result for result in before["results"]}
    assert before_by_id["door_clearance__door_0"]["label"] == "pass"
    assert (
        next(
            result
            for result in before["results"]
            if result.get("relation_type") == "front_axis_alignment"
        )["label"]
        == "fail"
    )
    original_transform = sofa.transform.GetAsMatrix4().copy()

    fixes = improve_furniture_relations(scene, config=config)

    assert fixes == []
    np.testing.assert_allclose(sofa.transform.GetAsMatrix4(), original_transform)
    after = evaluate_room_scene(scene, config=config, stage="door_safe_after")
    after_by_id = {result["check_id"]: result for result in after["results"]}
    assert after_by_id["door_clearance__door_0"]["label"] == "pass"


def test_repairs_two_anchor_living_room_group(tmp_path: Path) -> None:
    sofa = _object("sofa_0", "sofa", (0.0, -1.5, 0.4), (2.2, 0.8, 0.8))
    tv_stand = _object("tv_stand_0", "tv_stand", (0.0, 1.7, 0.3), (1.4, 0.5, 0.6))
    coffee_table = _object(
        "coffee_table_0", "coffee_table", (-1.0, 0.0, 0.25), (1.0, 0.6, 0.5)
    )
    rug = _object("rug_0", "rug", (1.0, 0.85, 0.02), (1.2, 0.7, 0.03))
    chairs = [
        _object(
            f"armchair_{index}",
            "armchair",
            (-1.7, y, 0.4),
            (0.65, 0.7, 0.8),
        )
        for index, y in enumerate((-0.5, 0.7))
    ]
    prompt = (
        "A living room with a sofa and TV stand, a coffee table centered between "
        "the sofa and TV stand, and two armchairs flanking the coffee table. "
        "A small rug lies between the coffee table and TV stand."
    )
    scene = _scene(tmp_path, sofa, tv_stand, coffee_table, rug, *chairs, text=prompt)
    scene.room_type = "living_room"
    _attach_fixture_contract(scene)
    config = CriticConfig(
        enabled=True,
        metrics=("functional_dependency",),
    )

    before = evaluate_room_scene(scene, config=config, stage="living_group_before")
    before_labels = {
        str(item.get("relation_type")): item.get("label")
        for item in before["results"]
        if item.get("relation_type")
        in {"centered_between_alignment", "between_alignment", "flanking"}
    }
    assert before_labels == {
        "centered_between_alignment": "fail",
        "between_alignment": "fail",
        "flanking": "fail",
    }

    fixes = improve_furniture_relations(scene, config=config)

    assert {fix.relation_type for fix in fixes} == {
        "centered_between_alignment",
        "between_alignment",
        "flanking",
    }
    np.testing.assert_allclose(coffee_table.transform.translation()[:2], [0.0, 0.1])
    np.testing.assert_allclose(rug.transform.translation()[:2], [0.0, 0.9])
    chair_x = sorted(float(chair.transform.translation()[0]) for chair in chairs)
    assert chair_x[0] < -0.9
    assert chair_x[1] > 0.9

    after = evaluate_room_scene(scene, config=config, stage="living_group_after")
    after_labels = {
        str(item.get("relation_type")): item.get("label")
        for item in after["results"]
        if item.get("relation_type")
        in {"centered_between_alignment", "between_alignment", "flanking"}
    }
    assert after_labels == {
        "centered_between_alignment": "pass",
        "between_alignment": "pass",
        "flanking": "pass",
    }


def test_centered_anchor_repair_preserves_passing_flanking_clearance(
    tmp_path: Path,
) -> None:
    sofa = _object("sofa_0", "sofa", (0.0, -1.46, 0.425), (2.46, 0.9, 0.85))
    tv_stand = _object("tv_stand_0", "tv_stand", (0.0, 1.695, 0.6), (1.73, 0.45, 1.2))
    coffee_table = _object(
        "coffee_table_0", "coffee_table", (0.075, 0.87, 0.225), (1.0, 0.6, 0.45)
    )
    chairs = [
        _object(
            f"armchair_{index}",
            "armchair",
            (x, 0.0, 0.39),
            (0.75, 0.8, 0.78),
            yaw_deg=yaw,
        )
        for index, (x, yaw) in enumerate(((-1.0, -53.0), (1.0, 55.0)))
    ]
    scene = _scene(
        tmp_path,
        sofa,
        tv_stand,
        coffee_table,
        *chairs,
        text=(
            "A living room with a sofa and TV stand, a coffee table centered "
            "between the sofa and TV stand, and two armchairs flanking the "
            "coffee table."
        ),
    )
    scene.room_type = "living_room"
    _attach_fixture_contract(scene)
    config = CriticConfig(
        enabled=True,
        metrics=("functional_dependency",),
    )

    before = evaluate_room_scene(scene, config=config, stage="group_clearance_before")
    labels = {
        str(item.get("relation_type")): item.get("label")
        for item in before["results"]
        if item.get("relation_type") in {"centered_between_alignment", "flanking"}
    }
    assert labels == {"centered_between_alignment": "fail", "flanking": "pass"}

    fixes = improve_furniture_relations(scene, config=config)

    assert {fix.object_id for fix in fixes} == {
        "coffee_table_0",
        "armchair_0",
        "armchair_1",
    }
    assert {fix.relation_type for fix in fixes} == {"centered_between_alignment"}
    assert float(chairs[0].transform.translation()[0]) < -1.15
    assert float(chairs[1].transform.translation()[0]) > 1.15
    table_bounds = coffee_table.compute_world_bounds()
    assert table_bounds is not None
    for chair in chairs:
        chair_bounds = chair.compute_world_bounds()
        assert chair_bounds is not None
        assert float(chair_bounds[1][0]) <= float(table_bounds[0][0]) or float(
            chair_bounds[0][0]
        ) >= float(table_bounds[1][0])

    after = evaluate_room_scene(scene, config=config, stage="group_clearance_after")
    assert {
        str(item.get("relation_type")): item.get("label")
        for item in after["results"]
        if item.get("relation_type") in {"centered_between_alignment", "flanking"}
    } == {"centered_between_alignment": "pass", "flanking": "pass"}


def test_repairs_paired_workstation_aisle_without_moving_other_furniture(
    tmp_path: Path,
) -> None:
    first = _object("desk_0", "desk", (0.0, -0.65, 0.4), (1.2, 0.6, 0.8))
    second = _object("desk_1", "desk", (0.0, 0.65, 0.4), (1.2, 0.6, 0.8))
    scene = _scene(
        tmp_path,
        first,
        second,
        text="An office has two desks with a clear walking path between the workstations.",
    )
    scene.room_type = "office"
    scene.scene_expert_original_description = scene.text_description
    scene.scene_expert_task_spec = {
        "compiler_status": "ok",
        "required_large_objects": ["desk", "desk"],
    }
    _attach_intent_contract(
        scene,
        [
            {
                "relation": "clear_access",
                "subjects": {"category": "desk", "count": 2},
                "targets": {"category": "desk", "count": 2},
                "source": "explicit_prompt",
                "evidence_span": "clear walking path between the workstations",
            }
        ],
    )
    config = CriticConfig(enabled=True, metrics=("functional_dependency",))

    before = evaluate_room_scene(
        scene, config=config, stage="furniture_relation_repair"
    )
    before_clear_access = next(
        item
        for item in before["results"]
        if item.get("relation_type") == "clear_access"
        and item.get("diagnostics", {}).get("evaluation_mode") == "between_workstations"
    )
    assert before_clear_access["label"] == "fail"
    assert before_clear_access["diagnostics"]["free_depth_m"] == pytest.approx(0.7)

    fixes = improve_furniture_relations(scene, config=config)

    assert {fix.object_id for fix in fixes} == {"desk_0", "desk_1"}
    assert {fix.relation_type for fix in fixes} == {"clear_access"}
    after = evaluate_room_scene(scene, config=config, stage="furniture_relation_repair")
    after_clear_access = next(
        item
        for item in after["results"]
        if item.get("relation_type") == "clear_access"
        and item.get("diagnostics", {}).get("evaluation_mode") == "between_workstations"
    )
    assert after_clear_access["label"] == "pass"
    assert after_clear_access["diagnostics"]["free_depth_m"] >= 0.8


def test_repairs_work_seats_blocking_a_paired_workstation_aisle(
    tmp_path: Path,
) -> None:
    first = _object("desk_0", "desk", (0.0, -1.0, 0.4), (1.2, 0.6, 0.8))
    second = _object("desk_1", "desk", (0.0, 1.0, 0.4), (1.2, 0.6, 0.8))
    first_chair = _object(
        "office_chair_0", "office_chair", (0.0, -0.45, 0.45), (0.6, 0.6, 0.9)
    )
    second_chair = _object(
        "office_chair_1", "office_chair", (0.0, 0.45, 0.45), (0.6, 0.6, 0.9)
    )
    scene = _scene(
        tmp_path,
        first,
        second,
        first_chair,
        second_chair,
        text="An office has two desks with a clear walking path between the workstations.",
    )
    scene.room_type = "office"
    scene.scene_expert_original_description = scene.text_description
    scene.scene_expert_task_spec = {
        "compiler_status": "ok",
        "required_large_objects": ["desk", "desk"],
    }
    _attach_intent_contract(
        scene,
        [
            {
                "relation": "clear_access",
                "subjects": {"category": "desk", "count": 2},
                "targets": {"category": "desk", "count": 2},
                "source": "explicit_prompt",
                "evidence_span": "clear walking path between the workstations",
            }
        ],
    )
    config = CriticConfig(enabled=True, metrics=("functional_dependency",))

    before = evaluate_room_scene(
        scene, config=config, stage="furniture_relation_repair"
    )
    before_clear_access = next(
        item
        for item in before["results"]
        if item.get("relation_type") == "clear_access"
        and item.get("diagnostics", {}).get("evaluation_mode") == "between_workstations"
    )
    assert before_clear_access["diagnostics"]["blocking_ids"] == [
        "office_chair_0",
        "office_chair_1",
    ]

    fixes = improve_furniture_relations(scene, config=config)

    assert {fix.object_id for fix in fixes} == {
        "desk_0",
        "desk_1",
        "office_chair_0",
        "office_chair_1",
    }
    assert float(first.transform.translation()[1]) > -1.0
    assert float(second.transform.translation()[1]) < 1.0
    assert float(first_chair.transform.translation()[1]) < -1.0
    assert float(second_chair.transform.translation()[1]) > 1.0
    after = evaluate_room_scene(scene, config=config, stage="furniture_relation_repair")
    after_clear_access = next(
        item
        for item in after["results"]
        if item.get("relation_type") == "clear_access"
        and item.get("diagnostics", {}).get("evaluation_mode") == "between_workstations"
    )
    assert after_clear_access["label"] == "pass"
    assert after_clear_access["diagnostics"]["blocking_ids"] == []
