import math

from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from pydrake.all import RigidTransform, RollPitchYaw

from scenesmith.agent_utils.room import ObjectType, UniqueID
from scenesmith.agent_utils.scene_analyzer import FurnitureSelection
from scenesmith.manipuland_agents.cross_stage_inventory import (
    existing_floor_covering_ids,
    redundant_floor_covering_request_indices,
)
from scenesmith.manipuland_agents.stateful_manipuland_agent import (
    StatefulManipulandAgent,
)
from scenesmith.scenebenchmark_critic.manipuland_targets import (
    infer_prompt_manipuland_obligations,
)


def _object(object_id: str, name: str):
    return SimpleNamespace(
        object_id=UniqueID(object_id),
        name=name,
        description=name,
        immutable=False,
    )


def test_dining_prompt_requires_table_and_sideboard_targets() -> None:
    prompt = (
        "A dining room with a dining table and table settings for four including "
        "plates, cutlery, and glasses. A centerpiece vase sits in the middle of "
        "the table, and a set of coasters sits on the sideboard."
    )

    obligations = infer_prompt_manipuland_obligations(prompt)

    assert [(item.category, item.target_count) for item in obligations] == [
        ("dining_table", 1),
        ("sideboard", 1),
    ]


def test_recovery_adds_missing_dining_table_without_duplicate_sideboard() -> None:
    table = _object("dining_table_0", "dining table")
    sideboard = _object("sideboard_0", "sideboard")
    scene = SimpleNamespace(
        scene_expert_original_description=(
            "A dining room with a dining table and table settings for four including "
            "plates, cutlery, and glasses. A centerpiece vase sits in the middle of "
            "the table, and a set of coasters sits on the sideboard."
        ),
        text_description="",
        objects={table.object_id: table, sideboard.object_id: sideboard},
        get_object=lambda object_id: {
            table.object_id: table,
            sideboard.object_id: sideboard,
        }.get(object_id),
    )
    agent = object.__new__(StatefulManipulandAgent)
    selections = [
        FurnitureSelection(
            furniture_id=sideboard.object_id,
            suggested_items="REQUIRED: coasters",
            prompt_constraints="prompt",
            style_notes="",
        )
    ]

    recovered = agent._recover_prompt_required_manipuland_targets(
        scene=scene, furniture_data=selections
    )

    assert [selection.furniture_id for selection in recovered] == [
        sideboard.object_id,
        table.object_id,
    ]


def test_final_dining_alignment_runs_when_planner_ignored_failed_contract() -> None:
    agent = object.__new__(StatefulManipulandAgent)
    agent.scene = Mock()
    agent.manipuland_tools = Mock()
    failed = {
        "primary_object": "dining_table_0",
        "label": "fail",
        "reason": "plates are not assigned to seat-facing lanes",
    }
    passed = {
        "primary_object": "dining_table_0",
        "label": "pass",
        "reason": "all settings are aligned",
    }

    with (
        patch(
            "scenesmith.manipuland_agents.stateful_manipuland_agent.room_scene_to_case_pack",
            return_value={"scene_geometry": {}},
        ),
        patch(
            "scenesmith.manipuland_agents.stateful_manipuland_agent.evaluate_dining_place_setting_alignment",
            side_effect=[[failed], [passed]],
        ),
    ):
        repaired = agent._enforce_dining_place_setting_alignment(
            UniqueID("dining_table_0")
        )

    assert repaired
    agent.manipuland_tools._align_dining_place_settings_impl.assert_called_once_with(
        table_id="dining_table_0"
    )


def test_final_dining_alignment_leaves_passing_contract_unchanged() -> None:
    agent = object.__new__(StatefulManipulandAgent)
    agent.scene = Mock()
    agent.manipuland_tools = Mock()
    passed = {
        "primary_object": "dining_table_0",
        "label": "pass",
        "reason": "all settings are aligned",
    }

    with (
        patch(
            "scenesmith.manipuland_agents.stateful_manipuland_agent.room_scene_to_case_pack",
            return_value={"scene_geometry": {}},
        ),
        patch(
            "scenesmith.manipuland_agents.stateful_manipuland_agent.evaluate_dining_place_setting_alignment",
            return_value=[passed],
        ),
    ):
        repaired = agent._enforce_dining_place_setting_alignment(
            UniqueID("dining_table_0")
        )

    assert not repaired
    agent.manipuland_tools._align_dining_place_settings_impl.assert_not_called()


def test_bilateral_bedside_prompt_recovers_both_nightstands() -> None:
    prompt = (
        "A nightstand with a table lamp on each side of the bed. An alarm clock "
        "sits on one nightstand and a book on the other."
    )

    obligations = infer_prompt_manipuland_obligations(prompt)

    assert [(item.category, item.target_count) for item in obligations] == [
        ("nightstand", 2)
    ]


def test_tv_stand_prompt_recovers_television_target_but_not_wall_mounts() -> None:
    obligations = infer_prompt_manipuland_obligations(
        "A living room with a TV stand and television on the opposite wall."
    )

    assert [(item.category, item.target_count) for item in obligations] == [
        ("tv_stand", 1)
    ]

    wall_mounted = infer_prompt_manipuland_obligations(
        "A living room with a TV stand below a wall-mounted television."
    )

    assert wall_mounted == []


def test_recovery_adds_tv_stand_omitted_by_vlm() -> None:
    tv_stand = _object("tv_stand_0", "TV stand")
    scene = SimpleNamespace(
        scene_expert_original_description=(
            "A living room with a TV stand and television on the opposite wall."
        ),
        text_description="",
        objects={tv_stand.object_id: tv_stand},
        get_object=lambda object_id: {tv_stand.object_id: tv_stand}.get(object_id),
    )
    agent = object.__new__(StatefulManipulandAgent)

    recovered = agent._recover_prompt_required_manipuland_targets(
        scene=scene, furniture_data=[]
    )

    assert [selection.furniture_id for selection in recovered] == [tv_stand.object_id]
    assert "television" in recovered[0].suggested_items


def test_each_desk_prompt_recovers_explicit_desk_count() -> None:
    prompt = (
        "A collaborative office with two desks facing each other, with an office "
        "chair and computer monitor at each desk."
    )

    obligations = infer_prompt_manipuland_obligations(prompt)

    assert [(item.category, item.target_count) for item in obligations] == [("desk", 2)]


def test_required_manipuland_targets_can_exceed_optional_target_cap() -> None:
    desk_0 = _object("office_desk_0", "office desk")
    desk_1 = _object("office_desk_1", "office desk")
    cabinet = _object("filing_cabinet_0", "filing cabinet")
    objects = {
        desk_0.object_id: desk_0,
        desk_1.object_id: desk_1,
        cabinet.object_id: cabinet,
    }
    scene = SimpleNamespace(get_object=lambda object_id: objects.get(object_id))
    selections = [
        FurnitureSelection(
            furniture_id=obj.object_id,
            suggested_items=f"REQUIRED: {item}",
            prompt_constraints="prompt",
            style_notes="",
        )
        for obj, item in (
            (desk_0, "monitor"),
            (desk_1, "monitor"),
            (cabinet, "printer"),
        )
    ]
    agent = object.__new__(StatefulManipulandAgent)

    selected = agent._select_manipuland_targets(
        scene=scene,
        furniture_data=selections,
        max_target_furniture=2,
    )

    assert {selection.furniture_id for selection in selected} == set(objects)


def test_monitor_work_seat_repair_keeps_surface_position_and_fixes_local_yaw() -> None:
    monitor = SimpleNamespace(
        object_id=UniqueID("computer_monitor_0"),
        placement_info=SimpleNamespace(position_2d=np.array([0.1, -0.2])),
    )
    objects = {monitor.object_id: monitor}
    scene = SimpleNamespace(
        room_type="office",
        scene_expert_original_description="An office with a monitor facing a chair.",
        scene_expert_task_spec={
            "intent_constraints": [
                {
                    "relation": "faces",
                    "subjects": {"category": "computer_monitor"},
                    "targets": {"category": "office_chair"},
                    "source": "model_inferred",
                }
            ]
        },
        get_object=lambda object_id: objects.get(object_id),
    )
    agent = object.__new__(StatefulManipulandAgent)
    agent.scene = scene
    tools = Mock()
    tools.support_surfaces = {
        "desk_top": SimpleNamespace(
            transform=RigidTransform(RollPitchYaw([0.0, 0.0, math.pi]), np.zeros(3))
        )
    }
    agent.manipuland_tools = tools
    case_pack = {
        "room_type": "office",
        "scene_geometry": {
            "rooms": [{"bbox": {"min": [-3, -3, 0], "max": [3, 3, 2.7]}}],
            "objects": [
                {
                    "id": "office_desk_0",
                    "category": "office_desk",
                    "bbox_world": {"center": [0.0, 0.0, 0.4], "size": [1.2, 0.6, 0.8]},
                },
                {
                    "id": "office_chair_0",
                    "category": "office_chair",
                    "bbox_world": {"center": [0.0, 1.0, 0.4], "size": [0.5, 0.5, 0.8]},
                },
                {
                    "id": "computer_monitor_0",
                    "category": "computer_monitor",
                    "yaw_deg": 180.0,
                    "placement_info": {"parent_surface_id": "desk_top"},
                    "bbox_world": {"center": [0.0, 0.0, 1.0], "size": [0.5, 0.2, 0.6]},
                },
            ],
        },
    }

    with (
        patch(
            "scenesmith.manipuland_agents.stateful_manipuland_agent.room_scene_to_case_pack",
            return_value=case_pack,
        ),
        patch(
            "scenesmith.manipuland_agents.stateful_manipuland_agent.assign_work_seats_to_surfaces",
            return_value=[
                SimpleNamespace(surface_id="office_desk_0", seat_id="office_chair_0")
            ],
        ),
    ):
        repaired = agent._enforce_monitor_work_seat_orientation(
            UniqueID("office_desk_0")
        )

    assert repaired
    tools._move_manipuland_impl.assert_called_once_with(
        object_id="computer_monitor_0",
        surface_id="desk_top",
        position_x=0.1,
        position_z=-0.2,
        rotation_degrees=180.0,
    )


def test_existing_furniture_rug_skips_redundant_floor_target() -> None:
    floor_id = UniqueID("floor_living_room")
    rug_id = UniqueID("rug_0")
    floor = SimpleNamespace(object_id=floor_id, object_type=ObjectType.FLOOR)
    rug = SimpleNamespace(
        object_id=rug_id,
        object_type=ObjectType.FURNITURE,
        name="rug",
        description="area rug",
        metadata={"semantic_name": "rug"},
    )
    objects = {floor_id: floor, rug_id: rug}
    scene = SimpleNamespace(
        objects=objects,
        get_object=lambda object_id: objects.get(object_id),
    )
    selection = FurnitureSelection(
        furniture_id=floor_id,
        suggested_items="REQUIRED: small rug",
        prompt_constraints="small rug between the table and television stand",
        style_notes="",
    )
    agent = object.__new__(StatefulManipulandAgent)

    filtered = agent._skip_realized_floor_covering_targets(
        scene=scene, furniture_data=[selection]
    )

    assert filtered == []
    assert existing_floor_covering_ids(scene) == ["rug_0"]


def test_explicit_floor_item_is_rerouted_from_nearby_furniture() -> None:
    floor_id = UniqueID("floor_bedroom")
    dresser_id = UniqueID("dresser_0")
    floor = SimpleNamespace(object_id=floor_id, object_type=ObjectType.FLOOR)
    dresser = SimpleNamespace(object_id=dresser_id, object_type=ObjectType.FURNITURE)
    objects = {floor_id: floor, dresser_id: dresser}
    scene = SimpleNamespace(
        objects=objects,
        get_object=lambda object_id: objects.get(object_id),
    )
    selection = FurnitureSelection(
        furniture_id=dresser_id,
        suggested_items="REQUIRED: wastebasket",
        prompt_constraints="small wastebasket near the dresser",
        style_notes="Floor item placed adjacent to dresser base.",
    )
    agent = object.__new__(StatefulManipulandAgent)

    routed = agent._route_explicit_floor_selections(
        scene=scene, furniture_data=[selection]
    )

    assert [item.furniture_id for item in routed] == [floor_id]
    assert routed[0].suggested_items == "REQUIRED: wastebasket"
    assert routed[0].prompt_constraints == "small wastebasket near the dresser"
    assert routed[0].context_furniture_ids == [dresser_id]


def test_mixed_floor_target_is_retained_but_duplicate_covering_is_filtered() -> None:
    floor_id = UniqueID("floor_living_room")
    rug_id = UniqueID("area_rug_0")
    floor = SimpleNamespace(object_id=floor_id, object_type=ObjectType.FLOOR)
    rug = SimpleNamespace(
        object_id=rug_id,
        object_type=ObjectType.FURNITURE,
        name="area_rug",
        description="",
        metadata={},
    )
    objects = {floor_id: floor, rug_id: rug}
    scene = SimpleNamespace(
        objects=objects,
        get_object=lambda object_id: objects.get(object_id),
    )
    selection = FurnitureSelection(
        furniture_id=floor_id,
        suggested_items="REQUIRED: rug and planter",
        prompt_constraints="",
        style_notes="",
    )
    agent = object.__new__(StatefulManipulandAgent)

    retained = agent._skip_realized_floor_covering_targets(
        scene=scene, furniture_data=[selection]
    )
    skipped = redundant_floor_covering_request_indices(
        scene,
        floor_id,
        ["Small rectangular carpet", "Ceramic floor planter"],
        ["small_carpet", "planter"],
    )

    assert retained == [selection]
    assert skipped == [0]


def test_floor_covering_request_is_allowed_when_scene_has_none() -> None:
    floor_id = UniqueID("floor_living_room")
    floor = SimpleNamespace(object_id=floor_id, object_type=ObjectType.FLOOR)
    scene = SimpleNamespace(
        objects={floor_id: floor},
        get_object=lambda object_id: floor if object_id == floor_id else None,
    )

    skipped = redundant_floor_covering_request_indices(
        scene,
        floor_id,
        ["Small rug"],
        ["small_rug"],
    )

    assert skipped == []
