import asyncio
import math

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call, patch

import numpy as np
import pytest

from agents.exceptions import MaxTurnsExceeded
from pydrake.all import RigidTransform, RollPitchYaw

from scenesmith.agent_utils.room import ObjectType, UniqueID
from scenesmith.agent_utils.base_stateful_agent import BaseStatefulAgent, Runner
from scenesmith.agent_utils.scene_analyzer import FurnitureSelection
from scenesmith.manipuland_agents.cross_stage_inventory import (
    contract_bound_support_object_ids,
    existing_floor_covering_ids,
    redundant_floor_covering_request_indices,
    satisfied_furniture_owned_floor_requirements,
    violates_hard_one_per_support_reparenting,
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


def test_hard_support_contract_exposes_existing_cross_stage_object() -> None:
    surface = SimpleNamespace(surface_id="S_1")
    tv_stand = SimpleNamespace(
        object_id=UniqueID("tv_stand_0"),
        name="tv_stand",
        description="Low media console TV stand",
        object_type=ObjectType.FURNITURE,
        metadata={"semantic_name": "tv_stand"},
        support_surfaces=[surface],
    )
    television = SimpleNamespace(
        object_id=UniqueID("television_0"),
        name="television",
        description="Slim flat-screen television display",
        object_type=ObjectType.FURNITURE,
        metadata={"semantic_name": "television"},
        support_surfaces=[],
        placement_info=SimpleNamespace(parent_surface_id="S_1"),
    )
    scene = SimpleNamespace(
        objects={tv_stand.object_id: tv_stand, television.object_id: television},
        scenebenchmark_intent_contract={
            "constraints": [
                {
                    "relation": "on_top_of",
                    "strength": "hard",
                    "subjects": {"category": "television", "count": 1},
                    "targets": {"category": "tv_stand", "count": 1},
                }
            ]
        },
    )

    assert contract_bound_support_object_ids(scene, tv_stand.object_id) == [
        "television_0"
    ]


def test_cross_stage_support_inventory_is_local_to_actual_surface() -> None:
    desks = [
        SimpleNamespace(
            object_id=UniqueID(f"desk_{index}"),
            name="desk",
            description="office desk",
            object_type=ObjectType.FURNITURE,
            metadata={"semantic_name": "desk"},
            support_surfaces=[SimpleNamespace(surface_id=f"S_{index * 2}")],
        )
        for index in range(4)
    ]
    monitor = SimpleNamespace(
        object_id=UniqueID("computer_monitor_0"),
        name="computer_monitor",
        description="computer monitor",
        object_type=ObjectType.MANIPULAND,
        metadata={"semantic_name": "computer_monitor"},
        support_surfaces=[],
        placement_info=SimpleNamespace(parent_surface_id="S_0"),
    )
    scene = SimpleNamespace(
        objects={
            **{desk.object_id: desk for desk in desks},
            monitor.object_id: monitor,
        },
        scenebenchmark_intent_contract={
            "constraints": [
                {
                    "relation": "on_top_of",
                    "strength": "hard",
                    "subjects": {"category": "computer_monitor", "count": 4},
                    "targets": {"category": "desk", "count": 4},
                }
            ]
        },
    )

    assert contract_bound_support_object_ids(scene, UniqueID("desk_0")) == [
        "computer_monitor_0"
    ]
    for index in range(1, 4):
        assert contract_bound_support_object_ids(scene, UniqueID(f"desk_{index}")) == []


def test_hard_one_per_support_rejects_cross_target_reparenting() -> None:
    desks = [
        SimpleNamespace(
            object_id=UniqueID(f"desk_{index}"),
            name="desk",
            description="office desk",
            object_type=ObjectType.FURNITURE,
            metadata={"semantic_name": "desk"},
            support_surfaces=[SimpleNamespace(surface_id=f"S_{index * 2}")],
        )
        for index in range(2)
    ]
    monitor = SimpleNamespace(
        object_id=UniqueID("computer_monitor_0"),
        name="computer_monitor",
        description="computer monitor",
        object_type=ObjectType.MANIPULAND,
        metadata={"semantic_name": "computer_monitor"},
        support_surfaces=[],
        placement_info=SimpleNamespace(parent_surface_id="S_0"),
    )
    scene = SimpleNamespace(
        objects={
            **{desk.object_id: desk for desk in desks},
            monitor.object_id: monitor,
        },
        scenebenchmark_intent_contract={
            "constraints": [
                {
                    "relation": "one_per_support",
                    "strength": "hard",
                    "subjects": {"category": "computer_monitor", "count": 2},
                    "targets": {"category": "desk", "count": 2},
                }
            ]
        },
    )

    assert not violates_hard_one_per_support_reparenting(
        scene,
        monitor.object_id,
        source_surface_id="S_0",
        target_surface_id="S_0",
    )
    assert violates_hard_one_per_support_reparenting(
        scene,
        monitor.object_id,
        source_surface_id="S_0",
        target_surface_id="S_2",
    )


def test_planner_retries_when_first_turn_has_no_workflow_tool_call() -> None:
    agent = object.__new__(StatefulManipulandAgent)
    agent.planner = SimpleNamespace(instructions="planner instructions")
    agent.planner_session = object()
    agent._planner_initial_design_tool_calls = 0
    agent._planner_budget_exhausted = False
    agent._reasoning_persistence_context_for_session = lambda _session: nullcontext()
    agent._create_run_config = Mock(return_value=None)
    agent._record_module_timing = Mock()
    agent._record_llm_call_debug = Mock()

    calls = []

    async def fake_run(*, input, **_kwargs):
        calls.append(input)
        if len(calls) == 2:
            # Simulate request_initial_design being executed by the recovered
            # planner run.
            agent._planner_initial_design_tool_calls = 1
        return SimpleNamespace(final_output="completed")

    with patch.object(Runner, "run", new=AsyncMock(side_effect=fake_run)):
        result = asyncio.run(
            agent._run_planner_workflow(
                runner_input="start workflow",
                max_turns=3,
            )
        )

    assert result.final_output == "completed"
    assert len(calls) == 2
    assert "request_initial_design()" in calls[1]


def test_planner_recovery_fails_if_second_turn_still_has_no_tool_call() -> None:
    agent = object.__new__(StatefulManipulandAgent)
    agent.planner = SimpleNamespace(instructions="planner instructions")
    agent.planner_session = object()
    agent._planner_initial_design_tool_calls = 0
    agent._planner_budget_exhausted = False
    agent._reasoning_persistence_context_for_session = lambda _session: nullcontext()
    agent._create_run_config = Mock(return_value=None)
    agent._record_module_timing = Mock()
    agent._record_llm_call_debug = Mock()

    with (
        patch.object(
            Runner,
            "run",
            new=AsyncMock(return_value=SimpleNamespace(final_output="acknowledged")),
        ),
        pytest.raises(RuntimeError, match="request_initial_design"),
    ):
        asyncio.run(
            agent._run_planner_workflow(
                runner_input="start workflow",
                max_turns=3,
            )
        )


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
    assert all(selection.is_prompt_required for selection in recovered)


def test_current_target_cardinality_rejects_excess_exact_objects() -> None:
    agent = object.__new__(StatefulManipulandAgent)
    agent.current_furniture_id = UniqueID("sideboard_0")
    agent.scene = SimpleNamespace()
    constraint = {
        "relation": "on_top_of",
        "stage": "manipuland",
        "source": "explicit_prompt",
        "strength": "hard",
        "subjects": {"category": "coaster", "count": 4, "quantifier": "exactly"},
        "targets": {"category": "sideboard", "count": 1, "quantifier": "all"},
    }
    objects = [
        {"id": "sideboard_0", "name": "sideboard", "category": "sideboard"},
        *[
            {"id": f"coaster_{index}", "name": "coaster", "category": "coaster"}
            for index in range(8)
        ],
    ]

    with (
        patch(
            "scenesmith.manipuland_agents.stateful_manipuland_agent.room_scene_to_case_pack",
            return_value={"scene_geometry": {"objects": objects}},
        ),
        patch(
            "scenesmith.manipuland_agents.stateful_manipuland_agent.intent_contract_constraints_for_scene",
            return_value=[constraint],
        ),
    ):
        failures = agent._current_target_cardinality_failures()

    assert failures == [
        "prompt-required exact count for coaster on sideboard_0: expected 4, found 8"
    ]


def test_current_target_cardinality_defers_multi_support_contract() -> None:
    agent = object.__new__(StatefulManipulandAgent)
    agent.current_furniture_id = UniqueID("desk_0")
    agent.scene = SimpleNamespace()
    constraint = {
        "relation": "one_per_support",
        "stage": "manipuland",
        "source": "explicit_prompt",
        "strength": "hard",
        "subjects": {"category": "monitor", "count": 2, "quantifier": "exactly"},
        "targets": {"category": "desk", "count": 2, "quantifier": "all"},
    }
    objects = [
        {"id": "desk_0", "name": "desk", "category": "desk"},
        {"id": "desk_1", "name": "desk", "category": "desk"},
        {"id": "monitor_0", "name": "monitor", "category": "monitor"},
    ]

    with (
        patch(
            "scenesmith.manipuland_agents.stateful_manipuland_agent.room_scene_to_case_pack",
            return_value={"scene_geometry": {"objects": objects}},
        ),
        patch(
            "scenesmith.manipuland_agents.stateful_manipuland_agent.intent_contract_constraints_for_scene",
            return_value=[constraint],
        ),
    ):
        failures = agent._current_target_cardinality_failures()

    assert failures == []


def test_final_dining_alignment_runs_when_planner_ignored_failed_contract() -> None:
    agent = object.__new__(StatefulManipulandAgent)
    agent.scene = Mock()
    agent._remove_duplicate_composite_members = Mock(return_value=[])
    agent.manipuland_tools = Mock()
    agent.manipuland_tools._complete_dining_place_settings_impl.return_value = (
        '{"success": true, "changed": false, "restored": false}'
    )
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
    agent._remove_duplicate_composite_members = Mock(return_value=[])
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


def test_dining_cleanup_removes_only_coincident_composite_member_copy() -> None:
    surface = SimpleNamespace(surface_id="S_5")
    table = SimpleNamespace(
        object_id=UniqueID("dining_table_0"),
        object_type=ObjectType.FURNITURE,
        support_surfaces=[surface],
    )

    def manipuland(object_id: str, x: float):
        return SimpleNamespace(
            object_id=UniqueID(object_id),
            object_type=ObjectType.MANIPULAND,
            metadata={},
            sdf_path="/tmp/shared_vase.sdf",
            transform=RigidTransform(p=[x, 0.0, 0.75]),
            placement_info=SimpleNamespace(parent_surface_id="S_5"),
            immutable=False,
        )

    duplicate = manipuland("vase_0", 0.0)
    separate_setting = manipuland("vase_1", 0.4)
    composite = SimpleNamespace(
        object_id=UniqueID("filled_container_0"),
        object_type=ObjectType.MANIPULAND,
        metadata={
            "composite_type": "filled_container",
            "container_asset": {
                "sdf_path": "/tmp/shared_vase.sdf",
                "transform": {"translation": [0.005, 0.0, 0.75]},
            },
            "fill_assets": [],
        },
        sdf_path=None,
        transform=RigidTransform(p=[0.005, 0.0, 0.75]),
        placement_info=SimpleNamespace(parent_surface_id="S_5"),
        immutable=False,
    )
    objects = {
        obj.object_id: obj for obj in (table, duplicate, separate_setting, composite)
    }
    scene = SimpleNamespace(objects=objects)
    scene.get_object = lambda object_id: scene.objects.get(object_id)

    def remove_object(object_id):
        return scene.objects.pop(object_id, None) is not None

    scene.remove_object = remove_object
    agent = object.__new__(StatefulManipulandAgent)
    agent.scene = scene
    agent.rendering_manager = Mock()
    agent._reset_critic_candidate_cache = Mock()

    removed = agent._remove_duplicate_composite_members(table.object_id)

    assert removed == ["vase_0"]
    assert duplicate.object_id not in scene.objects
    assert separate_setting.object_id in scene.objects
    agent.rendering_manager.clear_cache.assert_called_once_with()


def test_per_furniture_postprocessing_runs_before_final_critique() -> None:
    furniture_id = UniqueID("sofa_0")
    agent = object.__new__(StatefulManipulandAgent)
    agent.cfg = SimpleNamespace(
        agents=SimpleNamespace(planner_agent=SimpleNamespace(max_turns=3))
    )
    agent.prompt_registry = Mock()
    agent._run_planner_workflow = AsyncMock(
        return_value=SimpleNamespace(final_output=None)
    )
    agent._enforce_monitor_work_seat_orientation = Mock()
    agent._enforce_dining_place_setting_alignment = Mock()
    agent.scene = Mock()
    agent.scene.content_hash.return_value = "settled-scene"
    agent._can_skip_final_critique = Mock(return_value=False)

    ordered = Mock()
    agent._apply_per_furniture_postprocessing = Mock(return_value=True)
    agent._repair_dining_alignment_after_physics = Mock()
    agent._request_critique_impl = AsyncMock()
    agent._finalize_scene_and_scores = AsyncMock()
    agent._finalize_dining_joint_contract = Mock()
    ordered.attach_mock(agent._apply_per_furniture_postprocessing, "postprocess")
    ordered.attach_mock(
        agent._repair_dining_alignment_after_physics, "joint_dining_repair"
    )
    ordered.attach_mock(agent._request_critique_impl, "critique")
    ordered.attach_mock(agent._finalize_scene_and_scores, "finalize")
    ordered.attach_mock(
        agent._finalize_dining_joint_contract, "final_joint_dining_contract"
    )

    with patch(
        "scenesmith.manipuland_agents.stateful_manipuland_agent.log_agent_usage"
    ):
        asyncio.run(agent._run_furniture_workflow(furniture_id))

    assert ordered.mock_calls == [
        call.postprocess(furniture_id),
        call.joint_dining_repair(furniture_id),
        call.critique(update_checkpoint=False),
        call.finalize(),
        call.final_joint_dining_contract(furniture_id),
    ]


def test_planner_turn_limit_still_runs_deterministic_finalization() -> None:
    furniture_id = UniqueID("dining_table_0")
    agent = object.__new__(StatefulManipulandAgent)
    agent.cfg = SimpleNamespace(
        agents=SimpleNamespace(planner_agent=SimpleNamespace(max_turns=3))
    )
    agent.prompt_registry = Mock()
    agent._run_planner_workflow = AsyncMock(
        side_effect=MaxTurnsExceeded("Max turns (3) exceeded")
    )
    agent.scene = Mock()
    agent.scene.content_hash.return_value = "settled-scene"
    agent._can_skip_final_critique = Mock(return_value=False)

    ordered = Mock()
    agent._enforce_monitor_work_seat_orientation = Mock()
    agent._enforce_dining_place_setting_alignment = Mock()
    agent._apply_per_furniture_postprocessing = Mock()
    agent._repair_dining_alignment_after_physics = Mock()
    agent._request_critique_impl = AsyncMock()
    agent._finalize_scene_and_scores = AsyncMock()
    agent._finalize_dining_joint_contract = Mock()
    ordered.attach_mock(
        agent._enforce_monitor_work_seat_orientation, "monitor_alignment"
    )
    ordered.attach_mock(
        agent._enforce_dining_place_setting_alignment, "dining_alignment"
    )
    ordered.attach_mock(agent._apply_per_furniture_postprocessing, "postprocess")
    ordered.attach_mock(
        agent._repair_dining_alignment_after_physics, "joint_dining_repair"
    )
    ordered.attach_mock(agent._request_critique_impl, "critique")
    ordered.attach_mock(agent._finalize_scene_and_scores, "finalize")
    ordered.attach_mock(
        agent._finalize_dining_joint_contract, "final_joint_dining_contract"
    )

    asyncio.run(agent._run_furniture_workflow(furniture_id))

    assert ordered.mock_calls == [
        call.monitor_alignment(furniture_id),
        call.dining_alignment(furniture_id),
        call.postprocess(furniture_id),
        call.joint_dining_repair(furniture_id),
        call.critique(update_checkpoint=False),
        call.finalize(),
        call.final_joint_dining_contract(furniture_id),
    ]


def test_final_dining_contract_repairs_score_rollback_candidate() -> None:
    agent = _joint_repair_agent()
    failed = {"primary_object": "dining_table_0", "label": "fail"}
    passed = {
        "primary_object": "dining_table_0",
        "label": "pass",
        "related_objects": ["plate_0", "cutlery_0"],
    }
    complete = {"primary_object": "dining_table_0", "label": "pass"}
    agent._dining_contract_results = Mock(
        side_effect=[(failed, complete), (passed, complete)]
    )
    agent._repair_dining_alignment_after_physics = Mock(return_value=True)

    agent._finalize_dining_joint_contract(UniqueID("dining_table_0"))

    agent._repair_dining_alignment_after_physics.assert_called_once_with(
        UniqueID("dining_table_0")
    )
    agent._dining_support_bindings_valid.assert_called_once_with(
        UniqueID("dining_table_0"), passed
    )
    agent._dining_physics_valid.assert_called_once_with(UniqueID("dining_table_0"))


def test_final_dining_contract_rejects_unresolved_score_rollback() -> None:
    agent = _joint_repair_agent()
    failed = {"primary_object": "dining_table_0", "label": "fail"}
    complete = {"primary_object": "dining_table_0", "label": "pass"}
    agent._dining_contract_results = Mock(
        side_effect=[(failed, complete), (failed, complete)]
    )
    agent._repair_dining_alignment_after_physics = Mock(return_value=False)

    with pytest.raises(RuntimeError, match="unresolved dining"):
        agent._finalize_dining_joint_contract(UniqueID("dining_table_0"))


def test_final_dining_contract_names_failed_predicates() -> None:
    agent = _joint_repair_agent()
    failed_status = {
        "valid": False,
        "failures": [
            "alignment: plate_2 is outside dining_chair_2's front lane",
            "support: aligned place-setting objects are not all bound to dining_table_0",
        ],
    }
    agent._dining_joint_contract_status = Mock(
        side_effect=[failed_status, failed_status]
    )
    agent._repair_dining_alignment_after_physics = Mock(return_value=False)

    with pytest.raises(RuntimeError) as exc_info:
        agent._finalize_dining_joint_contract(UniqueID("dining_table_0"))

    assert "alignment: plate_2" in str(exc_info.value)
    assert "support: aligned place-setting" in str(exc_info.value)


def test_dining_joint_contract_status_reports_failed_predicates() -> None:
    agent = _joint_repair_agent()
    failed_alignment = {
        "primary_object": "dining_table_0",
        "label": "fail",
        "reason": "plate_2 is outside dining_chair_2's front lane",
    }
    complete = {"primary_object": "dining_table_0", "label": "pass"}
    agent._dining_contract_results = Mock(return_value=(failed_alignment, complete))

    status = agent._dining_joint_contract_status(UniqueID("dining_table_0"))

    assert status == {
        "furniture_id": "dining_table_0",
        "alignment": False,
        "completeness": True,
        "support": None,
        "physics": None,
        "valid": False,
        "failures": ["alignment: plate_2 is outside dining_chair_2's front lane"],
        "alignment_result": failed_alignment,
        "completeness_result": complete,
    }
    agent._dining_support_bindings_valid.assert_not_called()
    agent._dining_physics_valid.assert_not_called()


def test_invalid_dining_contract_cannot_become_checkpoint() -> None:
    agent = object.__new__(StatefulManipulandAgent)
    agent.current_furniture_id = UniqueID("dining_table_0")
    agent.scene = SimpleNamespace()
    agent._current_target_cardinality_failures = Mock(return_value=[])
    agent._current_target_dining_contract_failures = Mock(
        return_value=["dining joint contract for dining_table_0: physics: collision"]
    )
    base_state = SimpleNamespace(hard_valid=True, hard_reasons=[])

    with patch.object(
        BaseStatefulAgent,
        "_evaluate_current_hard_state",
        return_value=base_state,
    ):
        hard_state = agent._evaluate_current_hard_state()

    assert not hard_state.hard_valid
    assert hard_state.hard_reasons == [
        "dining joint contract for dining_table_0: physics: collision"
    ]


def test_valid_dining_contract_remains_checkpoint_eligible() -> None:
    agent = object.__new__(StatefulManipulandAgent)
    agent.current_furniture_id = UniqueID("dining_table_0")
    agent.scene = SimpleNamespace()
    agent._current_target_cardinality_failures = Mock(return_value=[])
    agent._current_target_dining_contract_failures = Mock(return_value=[])
    base_state = SimpleNamespace(hard_valid=True, hard_reasons=[])

    with patch.object(
        BaseStatefulAgent,
        "_evaluate_current_hard_state",
        return_value=base_state,
    ):
        hard_state = agent._evaluate_current_hard_state()

    assert hard_state.hard_valid
    assert hard_state.hard_reasons == []


def test_final_dining_contract_repairs_incomplete_layout_without_alignment() -> None:
    agent = _joint_repair_agent()
    passed = {
        "primary_object": "dining_table_0",
        "label": "pass",
        "related_objects": ["plate_0", "cutlery_0"],
    }
    complete = {"primary_object": "dining_table_0", "label": "pass"}
    agent._dining_contract_results = Mock(
        side_effect=[
            (None, {"primary_object": "dining_table_0", "label": "fail"}),
            (passed, complete),
        ]
    )
    agent._repair_dining_alignment_after_physics = Mock(return_value=True)

    agent._finalize_dining_joint_contract(UniqueID("dining_table_0"))

    agent._repair_dining_alignment_after_physics.assert_called_once_with(
        UniqueID("dining_table_0")
    )


def _joint_repair_agent() -> StatefulManipulandAgent:
    agent = object.__new__(StatefulManipulandAgent)
    agent.scene = Mock()
    agent.scene.to_state_dict.return_value = {"objects": {"settled": True}}
    agent.manipuland_tools = Mock()
    agent.manipuland_tools._align_dining_place_settings_impl.return_value = (
        '{"success": true, "restored": false}'
    )
    agent.manipuland_tools._complete_dining_place_settings_impl.return_value = (
        '{"success": true, "changed": false, "restored": false}'
    )
    agent.rendering_manager = Mock()
    agent._reset_critic_candidate_cache = Mock()
    agent._apply_per_furniture_postprocessing = Mock(return_value=True)
    agent._dining_support_bindings_valid = Mock(return_value=True)
    agent._dining_physics_valid = Mock(return_value=True)
    agent._resolve_dining_companion_collisions = Mock(return_value=True)
    return agent


def test_joint_dining_repair_skips_already_passing_five_seat_layout() -> None:
    agent = _joint_repair_agent()
    agent._dining_contract_results = Mock(
        return_value=(
            {"primary_object": "dining_table_0", "label": "pass"},
            {"primary_object": "dining_table_0", "label": "pass"},
        )
    )

    repaired = agent._repair_dining_alignment_after_physics(UniqueID("dining_table_0"))

    assert not repaired
    agent.manipuland_tools._align_dining_place_settings_impl.assert_not_called()
    agent._apply_per_furniture_postprocessing.assert_not_called()


def test_joint_dining_repair_commits_only_after_second_physics_pass() -> None:
    agent = _joint_repair_agent()
    failed = {"primary_object": "dining_table_0", "label": "fail"}
    passed = {
        "primary_object": "dining_table_0",
        "label": "pass",
        "related_objects": ["plate_0", "cutlery_0"],
    }
    complete = {"primary_object": "dining_table_0", "label": "pass"}
    agent._dining_contract_results = Mock(
        side_effect=[(failed, complete), (passed, complete), (passed, complete)]
    )

    repaired = agent._repair_dining_alignment_after_physics(UniqueID("dining_table_0"))

    assert repaired
    agent.manipuland_tools._align_dining_place_settings_impl.assert_called_once_with(
        table_id="dining_table_0"
    )
    agent._apply_per_furniture_postprocessing.assert_called_once_with(
        UniqueID("dining_table_0")
    )
    agent._resolve_dining_companion_collisions.assert_called_once_with(
        UniqueID("dining_table_0"), passed
    )
    agent._dining_physics_valid.assert_called_once_with(UniqueID("dining_table_0"))
    agent.scene.restore_from_state_dict.assert_not_called()


def test_joint_dining_repair_restores_settled_scene_when_physics_fails() -> None:
    agent = _joint_repair_agent()
    failed = {"primary_object": "dining_table_0", "label": "fail"}
    passed = {
        "primary_object": "dining_table_0",
        "label": "pass",
        "related_objects": ["plate_0", "cutlery_0"],
    }
    complete = {"primary_object": "dining_table_0", "label": "pass"}
    agent._dining_contract_results = Mock(
        side_effect=[(failed, complete), (passed, complete), (passed, complete)]
    )
    agent._dining_physics_valid.return_value = False

    repaired = agent._repair_dining_alignment_after_physics(UniqueID("dining_table_0"))

    assert not repaired
    agent.scene.restore_from_state_dict.assert_called_once_with(
        {"objects": {"settled": True}}
    )
    agent.rendering_manager.clear_cache.assert_called_once_with()
    agent._reset_critic_candidate_cache.assert_called_once_with()


def test_joint_dining_repair_completes_incomplete_inventory_before_alignment() -> None:
    agent = _joint_repair_agent()
    passed = {
        "primary_object": "dining_table_0",
        "label": "pass",
        "related_objects": ["plate_0", "cutlery_0"],
    }
    agent._dining_contract_results = Mock(
        side_effect=[
            (None, {"primary_object": "dining_table_0", "label": "fail"}),
            (passed, {"primary_object": "dining_table_0", "label": "pass"}),
            (passed, {"primary_object": "dining_table_0", "label": "pass"}),
        ]
    )

    repaired = agent._repair_dining_alignment_after_physics(UniqueID("dining_table_0"))

    assert repaired
    agent.manipuland_tools._complete_dining_place_settings_impl.assert_called_once_with(
        table_id="dining_table_0"
    )
    agent.manipuland_tools._align_dining_place_settings_impl.assert_called_once_with(
        table_id="dining_table_0"
    )


def test_joint_dining_repair_rejects_failed_second_projection() -> None:
    agent = _joint_repair_agent()
    failed = {"primary_object": "dining_table_0", "label": "fail"}
    passed = {
        "primary_object": "dining_table_0",
        "label": "pass",
        "related_objects": ["plate_0", "cutlery_0"],
    }
    complete = {"primary_object": "dining_table_0", "label": "pass"}
    agent._dining_contract_results = Mock(
        side_effect=[(failed, complete), (passed, complete), (passed, complete)]
    )
    agent._apply_per_furniture_postprocessing.return_value = False

    repaired = agent._repair_dining_alignment_after_physics(UniqueID("dining_table_0"))

    assert not repaired
    agent._dining_physics_valid.assert_not_called()
    agent.scene.restore_from_state_dict.assert_called_once_with(
        {"objects": {"settled": True}}
    )


def test_non_turn_limit_planner_error_still_propagates() -> None:
    furniture_id = UniqueID("dining_table_0")
    agent = object.__new__(StatefulManipulandAgent)
    agent.cfg = SimpleNamespace(
        agents=SimpleNamespace(planner_agent=SimpleNamespace(max_turns=3))
    )
    agent.prompt_registry = Mock()
    agent._run_planner_workflow = AsyncMock(side_effect=RuntimeError("service failed"))

    with pytest.raises(RuntimeError, match="service failed"):
        asyncio.run(agent._run_furniture_workflow(furniture_id))


def test_failed_furniture_workflow_restores_pre_target_scene() -> None:
    furniture_id = UniqueID("sofa_0")
    furniture = _object(str(furniture_id), "sofa")
    selection = FurnitureSelection(
        furniture_id=furniture_id,
        suggested_items="throw blanket",
        prompt_constraints="",
        style_notes="",
    )
    snapshot = {
        "room_geometry": None,
        "objects": {"sofa_0": {"name": "sofa"}},
        "text_description": "living room",
    }
    scene = Mock()
    scene.get_object.return_value = furniture
    scene.to_state_dict.return_value = snapshot
    scene.content_hash.return_value = "restored-scene"
    scene.text_description = "living room"

    agent = object.__new__(StatefulManipulandAgent)
    agent.cfg = SimpleNamespace(
        context_furniture=SimpleNamespace(enabled=False),
        support_surface_extraction={},
        openai=SimpleNamespace(model="test-model"),
    )
    agent.rendering_manager = Mock()
    agent.vlm_service = Mock()
    agent._analyze_furniture_for_placement = AsyncMock(return_value=[selection])
    agent._recover_prompt_required_manipuland_targets = Mock(return_value=[selection])
    agent._route_explicit_floor_selections = Mock(return_value=[selection])
    agent._skip_realized_floor_covering_targets = Mock(return_value=[selection])
    agent._get_max_target_furniture = Mock(return_value=0)
    agent._setup_furniture_context = Mock()
    agent._generate_manipuland_context_image = Mock(return_value=None)
    agent._initialize_checkpoint_state = Mock()
    agent._setup_furniture_agents = Mock()
    agent._run_furniture_workflow = AsyncMock(
        side_effect=RuntimeError("unresolved hard constraints")
    )

    with (
        patch(
            "scenesmith.manipuland_agents.stateful_manipuland_agent.custom_span",
            return_value=nullcontext(),
        ),
        patch(
            "scenesmith.manipuland_agents.stateful_manipuland_agent."
            "SupportSurfaceExtractionConfig.from_config",
            return_value=Mock(),
        ),
        patch(
            "scenesmith.manipuland_agents.stateful_manipuland_agent."
            "extract_and_propagate_support_surfaces",
            return_value=[Mock(surface_id=UniqueID("sofa_surface"))],
        ),
        patch(
            "scenesmith.manipuland_agents.stateful_manipuland_agent."
            "build_manipuland_placement_order_reference",
            return_value="",
        ),
    ):
        asyncio.run(agent.add_manipulands(scene))

    scene.restore_from_state_dict.assert_called_once_with(snapshot)
    assert agent.rendering_manager.clear_cache.call_count == 2
    assert agent._critic_candidate_cache == {"scene_hash": "restored-scene"}


def test_required_furniture_workflow_retries_once_then_raises_root_failure() -> None:
    furniture_id = UniqueID("dining_table_0")
    furniture = _object(str(furniture_id), "dining table")
    selection = FurnitureSelection(
        furniture_id=furniture_id,
        suggested_items="REQUIRED: table settings",
        prompt_constraints="four complete settings",
        style_notes="",
        is_prompt_required=True,
    )
    snapshot = {
        "room_geometry": None,
        "objects": {"dining_table_0": {"name": "dining table"}},
        "text_description": "dining room",
    }
    scene = Mock()
    scene.get_object.return_value = furniture
    scene.to_state_dict.return_value = snapshot
    scene.content_hash.return_value = "restored-scene"
    scene.text_description = "dining room"

    agent = object.__new__(StatefulManipulandAgent)
    agent.cfg = SimpleNamespace(
        context_furniture=SimpleNamespace(enabled=False),
        support_surface_extraction={},
        openai=SimpleNamespace(model="test-model"),
        required_target_retry_attempts=1,
    )
    agent.rendering_manager = Mock()
    agent.vlm_service = Mock()
    agent._analyze_furniture_for_placement = AsyncMock(return_value=[selection])
    agent._recover_prompt_required_manipuland_targets = Mock(return_value=[selection])
    agent._route_explicit_floor_selections = Mock(return_value=[selection])
    agent._skip_realized_floor_covering_targets = Mock(return_value=[selection])
    agent._get_max_target_furniture = Mock(return_value=0)
    agent._setup_furniture_context = Mock()
    agent._generate_manipuland_context_image = Mock(return_value=None)
    agent._initialize_checkpoint_state = Mock()
    agent._setup_furniture_agents = Mock()
    agent._run_furniture_workflow = AsyncMock(
        side_effect=RuntimeError("unresolved hard constraints")
    )
    agent._target_failure_diagnostic = Mock(return_value="7 collisions remain")
    agent._final_hard_validation_enabled = Mock(return_value=True)

    with (
        patch(
            "scenesmith.manipuland_agents.stateful_manipuland_agent.custom_span",
            return_value=nullcontext(),
        ),
        patch(
            "scenesmith.manipuland_agents.stateful_manipuland_agent."
            "SupportSurfaceExtractionConfig.from_config",
            return_value=Mock(),
        ),
        patch(
            "scenesmith.manipuland_agents.stateful_manipuland_agent."
            "extract_and_propagate_support_surfaces",
            return_value=[Mock(surface_id=UniqueID("table_surface"))],
        ),
        patch(
            "scenesmith.manipuland_agents.stateful_manipuland_agent."
            "build_manipuland_placement_order_reference",
            return_value="",
        ),
        pytest.raises(RuntimeError, match="failed after 2 attempt"),
    ):
        asyncio.run(agent.add_manipulands(scene))

    assert agent._run_furniture_workflow.await_count == 2
    assert scene.restore_from_state_dict.call_count >= 2


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


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        (["dining_table_0", "sideboard_0"], {"dining_table_0", "sideboard_0"}),
        ("dining_table_0, sideboard_0", {"dining_table_0", "sideboard_0"}),
        ([], set()),
    ],
)
def test_target_furniture_ids_accept_list_and_csv(configured, expected) -> None:
    agent = object.__new__(StatefulManipulandAgent)
    agent.cfg = SimpleNamespace(target_furniture_ids=configured)

    assert agent._get_target_furniture_ids() == expected


def test_monitor_work_seat_repair_keeps_surface_position_and_fixes_local_yaw() -> None:
    monitor = SimpleNamespace(
        object_id=UniqueID("computer_monitor_0"),
        placement_info=SimpleNamespace(position_2d=np.array([0.1, -0.2])),
    )
    objects = {monitor.object_id: monitor}
    scene = SimpleNamespace(
        room_type="office",
        scene_expert_original_description="An office with a monitor facing a chair.",
        scene_expert_task_spec={},
        scenebenchmark_intent_contract={
            "constraints": [
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


def test_fulfilled_furniture_owned_floor_requirement_skips_single_target() -> None:
    floor_id = UniqueID("floor_office")
    wastebasket_id = UniqueID("trash_can_0")
    floor = SimpleNamespace(object_id=floor_id, object_type=ObjectType.FLOOR)
    wastebasket = SimpleNamespace(
        object_id=wastebasket_id,
        object_type=ObjectType.FURNITURE,
        name="office trash can",
        description="dark gray freestanding wastebasket",
        metadata={"semantic_name": "trash_can"},
    )
    objects = {floor_id: floor, wastebasket_id: wastebasket}
    scene = SimpleNamespace(
        objects=objects,
        scene_expert_task_spec={"required_large_objects": ["wastebasket"]},
        get_object=lambda object_id: objects.get(object_id),
    )
    selection = FurnitureSelection(
        furniture_id=floor_id,
        suggested_items="REQUIRED: wastebasket",
        prompt_constraints="place it in a corner",
        style_notes="",
    )
    agent = object.__new__(StatefulManipulandAgent)

    assert satisfied_furniture_owned_floor_requirements(
        scene, floor_id, selection.suggested_items
    ) == {"wastebasket": ["trash_can_0"]}
    assert (
        agent._skip_satisfied_furniture_owned_floor_targets(
            scene=scene, furniture_data=[selection]
        )
        == []
    )


def test_furniture_owned_floor_requirement_needs_authoritative_count() -> None:
    floor_id = UniqueID("floor_office")
    wastebasket_id = UniqueID("wastebasket_0")
    floor = SimpleNamespace(object_id=floor_id, object_type=ObjectType.FLOOR)
    wastebasket = SimpleNamespace(
        object_id=wastebasket_id,
        object_type=ObjectType.FURNITURE,
        name="wastebasket",
        description="office wastebasket",
        metadata={"semantic_name": "wastebasket"},
    )
    objects = {floor_id: floor, wastebasket_id: wastebasket}
    scene = SimpleNamespace(
        objects=objects,
        scene_expert_task_spec={"required_large_objects": ["wastebasket"] * 2},
        get_object=lambda object_id: objects.get(object_id),
    )
    selection = FurnitureSelection(
        furniture_id=floor_id,
        suggested_items="REQUIRED: wastebasket",
        prompt_constraints="",
        style_notes="",
    )
    agent = object.__new__(StatefulManipulandAgent)

    assert (
        satisfied_furniture_owned_floor_requirements(
            scene, floor_id, selection.suggested_items
        )
        == {}
    )
    assert agent._skip_satisfied_furniture_owned_floor_targets(
        scene=scene, furniture_data=[selection]
    ) == [selection]


def test_mixed_floor_target_keeps_unsatisfied_work_and_exposes_inventory_note() -> None:
    floor_id = UniqueID("floor_office")
    wastebasket_id = UniqueID("wastebasket_0")
    floor = SimpleNamespace(object_id=floor_id, object_type=ObjectType.FLOOR)
    wastebasket = SimpleNamespace(
        object_id=wastebasket_id,
        object_type=ObjectType.FURNITURE,
        name="wastebasket",
        description="office wastebasket",
        metadata={"semantic_name": "wastebasket"},
    )
    objects = {floor_id: floor, wastebasket_id: wastebasket}
    scene = SimpleNamespace(
        objects=objects,
        scene_expert_task_spec={"required_large_objects": ["wastebasket"]},
        get_object=lambda object_id: objects.get(object_id),
    )
    selection = FurnitureSelection(
        furniture_id=floor_id,
        suggested_items="REQUIRED: wastebasket and floor lamp",
        prompt_constraints="place both in a corner",
        style_notes="",
    )
    agent = object.__new__(StatefulManipulandAgent)

    assert agent._skip_satisfied_furniture_owned_floor_targets(
        scene=scene, furniture_data=[selection]
    ) == [selection]
    assert "Cross-stage inventory (do not regenerate): wastebasket" in (
        selection.prompt_constraints
    )


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
