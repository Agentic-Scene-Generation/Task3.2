from types import SimpleNamespace
from unittest.mock import Mock, patch

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
