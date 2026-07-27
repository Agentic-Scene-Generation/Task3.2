import math
import xml.etree.ElementTree as ET

from pathlib import Path

import numpy as np

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
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.orientation_contracts import (
    CONTRACT_ATTR,
)
from scenesmith.scenebenchmark_critic.prompt_context import format_agent_prompt_context
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
        room_type=(
            "classroom"
            if "classroom" in text
            else "study" if "study" in text else "dining_room"
        ),
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

    unresolved_before = unresolved_furniture_relation_failures(scene, config=config)
    assert {result.get("relation_type") for result in unresolved_before} == {
        "dining_seat_distribution"
    }

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
        if item.get("relation_type") == "dining_seat_distribution"
    )
    assert result["label"] == "pass"
    for slot in result["diagnostics"]["seat_slots"]:
        assert slot["aligned"]
        assert slot["facing_aligned"]


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
    config = CriticConfig(enabled=True, metrics=("functional_dependency",))

    evaluate_room_scene(scene, config=config, stage="dining_sideboard_prompt")

    contracts = getattr(scene, CONTRACT_ATTR)
    assert {contracts[seat.object_id]["relation_type"] for seat in seats} == {
        "seating_to_work_surface"
    }
    assert {contracts[seat.object_id]["target_ids"][0] for seat in seats} == {
        "dining_table_0"
    }


def test_repairs_generic_classroom_as_atomic_student_grid(tmp_path: Path) -> None:
    desk_positions = (
        (-3.04, -3.8),
        (-3.04, -2.0),
        (-3.04, 0.0),
        (-3.04, 2.0),
        (-1.6, -3.8),
        (-1.6, -2.0),
        (-1.6, 0.0),
    )
    chair_positions = (
        (-1.6, 2.0),
        (-1.6, 3.8),
        (0.0, -3.8),
        (0.0, -2.0),
        (0.0, 0.0),
        (0.0, 2.0),
    )
    desks = [
        _object(f"desk_{index}", "desk", (*xy, 0.3), (1.13, 0.6, 0.6))
        for index, xy in enumerate(desk_positions)
    ]
    chairs = [
        _object(f"chair_{index}", "chair", (*xy, 0.4), (0.51, 0.5, 0.8))
        for index, xy in enumerate(chair_positions)
    ]
    text = (
        "A classroom with six student desks, each with a chair. A teacher's "
        "desk sits at the front near the chalkboard. Arrange student desks in "
        "rows across the classroom with all workstations facing the front."
    )
    scene = _scene(tmp_path, *desks, *chairs, text=text)
    scene.room_geometry.length = 8.0
    scene.room_geometry.width = 10.0
    scene.room_geometry.walls = [
        _object(
            "west_wall",
            "wall",
            (-4.0, 0.0, 1.35),
            (0.1, 10.0, 2.7),
            object_type=ObjectType.WALL,
        ),
        _object(
            "east_wall",
            "wall",
            (4.0, 0.0, 1.35),
            (0.1, 10.0, 2.7),
            object_type=ObjectType.WALL,
        ),
        _object(
            "south_wall",
            "wall",
            (0.0, -5.0, 1.35),
            (8.0, 0.1, 2.7),
            object_type=ObjectType.WALL,
        ),
        _object(
            "north_wall",
            "wall",
            (0.0, 5.0, 1.35),
            (8.0, 0.1, 2.7),
            object_type=ObjectType.WALL,
        ),
    ]
    config = CriticConfig(enabled=True, metrics=("functional_dependency",))

    before = evaluate_room_scene(scene, config=config, stage="classroom_before")
    result = next(
        item
        for item in before["results"]
        if item.get("relation_type") == "classroom_workstation_distribution"
    )
    assert result["label"] == "fail"

    fixes = improve_furniture_relations(scene, config=config)

    assert {fix.object_id for fix in fixes} == {
        *(f"desk_{index}" for index in range(7)),
        *(f"chair_{index}" for index in range(6)),
    }
    after = evaluate_room_scene(scene, config=config, stage="classroom_after")
    result = next(
        item
        for item in after["results"]
        if item.get("relation_type") == "classroom_workstation_distribution"
    )
    assert result["label"] == "pass"
    slots = result["diagnostics"]["workstation_slots"]
    lateral_centers = {
        round(slot["target_surface_center_xy_m"][1], 3) for slot in slots
    }
    assert len(lateral_centers) == 3


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
    config = CriticConfig(
        enabled=True,
        metrics=("functional_dependency",),
        constraint_mode="contract",
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
    assert unresolved_furniture_relation_failures(scene, config=config) == []


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
    front = storage.transform.rotation().matrix() @ np.array([0.0, 1.0, 0.0])
    np.testing.assert_allclose(front[:2], [1.0, 0.0], atol=1e-7)


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
    assert result_by_id["wall_backed_storage__bookshelf_0"]["label"] == "pass"


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


def test_prompt_facing_guest_chairs_override_cached_orientation_contracts(
    tmp_path: Path,
) -> None:
    desk = _object(
        "study_desk_0",
        "study_desk",
        (0.0, 1.79, 0.4),
        (1.46, 0.8, 0.8),
        yaw_deg=180.0,
    )
    chair_positions = ((2.05, 0.5), (2.05, -0.5))
    chairs = [
        _object(
            f"guest_chair_{index}",
            "guest_chair",
            (position[0], position[1], 0.45),
            (0.55, 0.66, 0.9),
            yaw_deg=_facing_yaw(position, (0.0, 1.79)),
        )
        for index, position in enumerate(chair_positions)
    ]
    scene = _scene(
        tmp_path,
        desk,
        *chairs,
        text="A study with a desk and two guest chairs.",
    )
    config = CriticConfig(enabled=True, metrics=("functional_dependency",))

    initial_payload = evaluate_room_scene(
        scene, config=config, stage="before_prompt_face"
    )
    initial_contracts = {
        check.get("subject_id"): check
        for check in initial_payload["case_pack"]["checks"]
        if check.get("check_source") == "scenesmith_orientation_contract"
    }
    assert (
        initial_contracts["guest_chair_0"]["relation_type"] == "seating_to_work_surface"
    )

    scene.text_description = (
        "A study with a desk centered against the back wall and two guest chairs "
        "against the side wall facing the desk."
    )
    payload = evaluate_room_scene(scene, config=config, stage="after_prompt_face")
    contracts = {
        check.get("subject_id"): check
        for check in payload["case_pack"]["checks"]
        if check.get("check_source") == "scenesmith_orientation_contract"
    }
    expected_ids = {str(chair.object_id) for chair in chairs}
    assert expected_ids <= contracts.keys()
    for chair_id in expected_ids:
        assert contracts[chair_id]["relation_type"] == "furniture_faces_furniture"
        assert contracts[chair_id]["target_ids"] == ["study_desk_0"]

    result_by_check = {result["check_id"]: result for result in payload["results"]}
    contract_results = [
        result_by_check[contracts[chair_id]["check_id"]] for chair_id in expected_ids
    ]
    assert {result["label"] for result in contract_results} == {"pass"}
    assert all("angle 0deg" in result["reason"] for result in contract_results)

    context = format_agent_prompt_context(payload, agent_type=AgentType.FURNITURE)
    assert context.count("result=pass") >= 2
    assert "A deterministic `result=pass` is authoritative" in context
    assert "`guest_chair_0`: `furniture_faces_furniture` -> `study_desk_0`" in context


def test_prompt_wall_backed_guest_chairs_stay_free_of_study_desk(
    tmp_path: Path,
) -> None:
    desk = _object(
        "study_desk_0",
        "study_desk",
        (0.0, 1.79, 0.4),
        (1.46, 0.8, 0.8),
        yaw_deg=180.0,
    )
    office_chair = _object(
        "office_chair_0",
        "office_chair",
        (0.0, 0.915, 0.45),
        (0.6, 0.6, 0.9),
        yaw_deg=0.0,
    )
    guest_positions = ((1.35, 0.55), (1.35, -0.55))
    guests = [
        _object(
            f"guest_chair_{index}",
            "guest_chair",
            (position[0], position[1], 0.45),
            (0.55, 0.66, 0.9),
            yaw_deg=_facing_yaw(position, (0.0, 1.79)),
        )
        for index, position in enumerate(guest_positions)
    ]
    scene = _scene(
        tmp_path,
        desk,
        office_chair,
        *guests,
        text=(
            "A study with a desk centered against the back wall, an office chair "
            "tucked under the desk, and two guest chairs against the side wall "
            "with their usable fronts perpendicular to the wall and facing into "
            "the room."
        ),
    )
    config = CriticConfig(enabled=True, metrics=("functional_dependency",))

    payload = evaluate_room_scene(scene, config=config, stage="prompt_wall_backed")
    contracts = {
        check.get("subject_id"): check
        for check in payload["case_pack"]["checks"]
        if check.get("check_source") == "scenesmith_orientation_contract"
    }
    guest_ids = {str(guest.object_id) for guest in guests}

    assert contracts["office_chair_0"]["relation_type"] == "seating_to_work_surface"
    assert contracts["office_chair_0"]["target_ids"] == ["study_desk_0"]
    for guest_id in guest_ids:
        contract = contracts[guest_id]
        assert contract["relation_type"] == "back_against_wall"
        assert contract["target_ids"] == ["east_wall"]
        assert contract["evidence"]["prompt_explicit_wall"] is True
    assert not any(
        check.get("subject_id") in guest_ids
        and check.get("relation_type")
        in {"seating_to_work_surface", "furniture_faces_furniture"}
        and "study_desk_0" in {str(item) for item in check.get("target_ids") or []}
        for check in payload["case_pack"]["checks"]
    )

    fixes = improve_furniture_relations(scene, config=config)
    assert guest_ids <= {fix.object_id for fix in fixes}
    assert {fix.relation_type for fix in fixes if fix.object_id in guest_ids} == {
        "back_against_wall"
    }

    east_wall = next(
        wall for wall in scene.room_geometry.walls if str(wall.object_id) == "east_wall"
    )
    east_bounds = east_wall.compute_world_bounds()
    assert east_bounds is not None
    for guest in guests:
        front = guest.transform.rotation().matrix() @ np.array([0.0, 1.0, 0.0])
        np.testing.assert_allclose(front[:2], [-1.0, 0.0], atol=1e-7)
        guest_bounds = guest.compute_world_bounds()
        assert guest_bounds is not None
        assert abs(east_bounds[0][0] - guest_bounds[1][0] - 0.03) < 1e-7

    repaired_payload = evaluate_room_scene(
        scene, config=config, stage="prompt_wall_backed_repaired"
    )
    result_by_check = {
        result["check_id"]: result for result in repaired_payload["results"]
    }
    repaired_contracts = {
        check.get("subject_id"): check
        for check in repaired_payload["case_pack"]["checks"]
        if check.get("check_source") == "scenesmith_orientation_contract"
    }
    assert {
        result_by_check[repaired_contracts[guest_id]["check_id"]]["label"]
        for guest_id in guest_ids
    } == {"pass"}
    assert repaired_contracts["office_chair_0"]["target_ids"] == ["study_desk_0"]


def test_back_wall_sofa_uses_adjacent_wall_not_coffee_table_wall(
    tmp_path: Path,
) -> None:
    sofa = _object("sofa_0", "sofa", (0.0, -1.5, 0.4), (2.2, 0.9, 0.8), yaw_deg=0.0)
    tv_stand = _object(
        "tv_stand_0", "tv_stand", (0.0, 1.65, 0.5), (1.7, 0.45, 1.0), yaw_deg=180.0
    )
    coffee_table = _object(
        "coffee_table_0", "coffee_table", (0.0, 0.1, 0.25), (1.2, 0.7, 0.5)
    )
    scene = _scene(
        tmp_path,
        sofa,
        tv_stand,
        coffee_table,
        text=(
            "A living room with a sofa against the back wall facing a TV stand "
            "on the opposite wall, and a coffee table between them."
        ),
    )
    config = CriticConfig(enabled=True, metrics=("functional_dependency",))

    payload = evaluate_room_scene(scene, config=config, stage="back_wall_sofa")
    contract = next(
        check
        for check in payload["case_pack"]["checks"]
        if check.get("check_source") == "scenesmith_orientation_contract"
        and check.get("subject_id") == "sofa_0"
    )
    results_by_id = {result["check_id"]: result for result in payload["results"]}

    assert contract["relation_type"] == "back_against_wall"
    assert contract["target_ids"] == ["south_wall"]
    assert results_by_id[contract["check_id"]]["label"] == "pass"


def test_later_stage_brief_overrides_earlier_guest_desk_facing(tmp_path: Path) -> None:
    desk = _object(
        "study_desk_0",
        "study_desk",
        (0.0, 1.79, 0.4),
        (1.46, 0.8, 0.8),
        yaw_deg=180.0,
    )
    guests = [
        _object(
            f"guest_chair_{index}",
            "guest_chair",
            (1.9, y, 0.45),
            (0.55, 0.66, 0.9),
            yaw_deg=_facing_yaw((1.9, y), (0.0, 1.79)),
        )
        for index, y in enumerate((0.5, -0.5))
    ]
    scene = _scene(
        tmp_path,
        desk,
        *guests,
        text=(
            "A study with two guest chairs against the side wall facing the desk.\n\n"
            "=== SceneExpert Stage Brief: furniture ===\n"
            "Place the two guest chairs against a side wall, oriented with their "
            "fronts facing into the room. Verify both guest chairs are against a "
            "side wall and facing inward."
        ),
    )
    config = CriticConfig(enabled=True, metrics=("functional_dependency",))

    payload = evaluate_room_scene(scene, config=config, stage="later_inward_brief")
    contracts = {
        check.get("subject_id"): check
        for check in payload["case_pack"]["checks"]
        if check.get("check_source") == "scenesmith_orientation_contract"
    }
    for guest in guests:
        contract = contracts[str(guest.object_id)]
        assert contract["relation_type"] == "back_against_wall"
        assert contract["target_ids"] == ["east_wall"]

    fixes = improve_furniture_relations(scene, config=config)

    assert {fix.object_id for fix in fixes} == {
        "guest_chair_0",
        "guest_chair_1",
    }
    for guest in guests:
        front = guest.transform.rotation().matrix() @ np.array([0.0, 1.0, 0.0])
        np.testing.assert_allclose(front[:2], [-1.0, 0.0], atol=1e-7)


def test_repairs_study_as_coordinated_opposite_wall_layout(tmp_path: Path) -> None:
    desk = _object("desk_0", "desk", (-1.0, -1.45, 0.4), (1.4, 0.8, 0.8), yaw_deg=90.0)
    office_chair = _object(
        "office_chair_0",
        "office_chair",
        (-1.9, -1.7, 0.45),
        (0.6, 0.62, 0.9),
        yaw_deg=-78.0,
    )
    guests = [
        _object(
            f"guest_chair_{index}",
            "guest_chair",
            (-2.0, -0.9 + 0.9 * index, 0.4),
            (0.55, 0.72, 0.8),
            yaw_deg=yaw,
        )
        for index, yaw in enumerate((-90.0, -145.0))
    ]
    bookshelf = _object(
        "bookshelf_0",
        "bookshelf",
        (-2.15, 0.0, 1.0),
        (1.0, 0.5, 2.0),
        yaw_deg=-90.0,
    )
    scene = _scene(
        tmp_path,
        desk,
        office_chair,
        *guests,
        bookshelf,
        text=(
            "A study with a desk centered against the back wall, an office chair "
            "tucked under the desk, two guest chairs against the side wall with "
            "their usable fronts perpendicular to the wall and facing into the "
            "room, and a bookshelf on the adjacent wall."
        ),
    )
    config = CriticConfig(
        enabled=True,
        metrics=("functional_dependency",),
        # This test preserves the historical coordinated-wall repair. Contract
        # mode intentionally suppresses this scene-specific legacy extension.
        constraint_mode="legacy",
    )

    before = evaluate_room_scene(scene, config=config, stage="study_before")
    before_result = next(
        result
        for result in before["results"]
        if result.get("relation_type") == "study_furniture_layout"
    )
    assert before_result["label"] == "fail"

    fixes = improve_furniture_relations(scene, config=config)

    group_ids = {
        "desk_0",
        "office_chair_0",
        "guest_chair_0",
        "guest_chair_1",
        "bookshelf_0",
    }
    assert group_ids <= {fix.object_id for fix in fixes}
    assert desk.transform.translation()[1] > 1.0
    assert abs(desk.transform.translation()[0]) < 1e-7
    assert bookshelf.transform.translation()[0] < -1.5
    for guest in guests:
        assert guest.transform.translation()[0] > 1.5
        front = guest.transform.rotation().matrix() @ np.array([0.0, 1.0, 0.0])
        np.testing.assert_allclose(front[:2], [-1.0, 0.0], atol=1e-7)

    after = evaluate_room_scene(scene, config=config, stage="study_after")
    after_result = next(
        result
        for result in after["results"]
        if result.get("relation_type") == "study_furniture_layout"
    )
    assert after_result["label"] == "pass"


def test_prompt_guest_facing_does_not_apply_to_earlier_office_chair(
    tmp_path: Path,
) -> None:
    desk = _object("study_desk_0", "study_desk", (0.0, 1.79, 0.4), (1.46, 0.8, 0.8))
    office_chair = _object(
        "office_chair_0",
        "office_chair",
        (0.0, 1.15, 0.45),
        (0.6, 0.6, 0.9),
        yaw_deg=_facing_yaw((0.0, 1.15), (0.0, 1.79)),
    )
    guests = [
        _object(
            f"guest_chair_{index}",
            "guest_chair",
            (1.95, y, 0.45),
            (0.55, 0.66, 0.9),
            yaw_deg=_facing_yaw((1.95, y), (0.0, 1.79)),
        )
        for index, y in enumerate((0.5, -0.5))
    ]
    scene = _scene(
        tmp_path,
        desk,
        office_chair,
        *guests,
        text=(
            "A study with a desk centered against the back wall, an office chair "
            "tucked under the desk, a computer monitor on the desk, two guest "
            "chairs against the side wall facing the desk."
        ),
    )

    payload = evaluate_room_scene(
        scene,
        config=CriticConfig(enabled=True, metrics=("functional_dependency",)),
        stage="role_scoped_prompt_face",
    )
    contracts = {
        check.get("subject_id"): check
        for check in payload["case_pack"]["checks"]
        if check.get("check_source") == "scenesmith_orientation_contract"
    }

    assert contracts["office_chair_0"]["relation_type"] == "seating_to_work_surface"
    assert {contracts[str(guest.object_id)]["relation_type"] for guest in guests} == {
        "furniture_faces_furniture"
    }


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
                "relation_type": "dining_seat_distribution",
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
                "relation_type": "dining_seat_distribution",
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
                "relation_type": "dining_seat_distribution",
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
                "relation_type": "dining_seat_distribution",
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
                        "relation_type": "dining_seat_distribution",
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
                        "relation_type": "dining_seat_distribution",
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
        "relation_type": "dining_seat_distribution",
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
        "relation_type": "dining_seat_distribution",
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
