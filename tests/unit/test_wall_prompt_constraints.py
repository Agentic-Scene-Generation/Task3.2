import asyncio

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from agents.exceptions import MaxTurnsExceeded
from scenesmith.wall_agents.prompt_constraints import (
    build_required_wall_object_constraints,
    converge_cross_stage_media_inventory,
)
from scenesmith.wall_agents import stateful_wall_agent
from scenesmith.wall_agents.stateful_wall_agent import StatefulWallAgent


def test_tv_group_on_opposite_wall_does_not_imply_wall_mount() -> None:
    constraints = build_required_wall_object_constraints(
        "A sofa faces a TV stand and television on the opposite wall."
    )

    assert "No explicit wall-object obligations" in constraints


def test_explicit_wall_mounted_tv_requires_window_repair_before_offset() -> None:
    constraints = build_required_wall_object_constraints(
        "A sofa faces a TV stand with a wall-mounted television above it."
    )

    assert "REQUIRED media display" in constraints
    assert "call list_windows" in constraints
    assert "Never leave the display offset" in constraints


def test_normalized_monitor_inventory_preserves_mounted_screen_wall_ownership() -> None:
    constraints = build_required_wall_object_constraints(
        "A mounted screen is opposite the sofa.",
        task_spec={
            "required_large_objects": ["sofa"],
            "required_wall_objects": ["monitor"],
            "required_small_objects": [],
        },
    )

    assert "REQUIRED media display" in constraints
    assert "No explicit wall-object obligations" not in constraints


def test_desktop_monitor_does_not_become_wall_requirement() -> None:
    constraints = build_required_wall_object_constraints(
        "A desk centered against the back wall with a computer monitor on the desk."
    )

    assert "No explicit wall-object obligations" in constraints


def test_task_spec_blocks_stagebrief_from_promoting_later_stage_tv() -> None:
    constraints = build_required_wall_object_constraints(
        """A sofa faces a TV stand and television on the opposite wall.

=== SceneExpert Stage Brief: wall_mounted ===
Designer constraints:
  - Place the television on the wall opposite the sofa.
=== End Stage Brief ===""",
        task_spec={
            "required_large_objects": ["tv_stand"],
            "required_wall_objects": [],
            "required_small_objects": ["television"],
        },
    )

    assert "No explicit wall-object obligations" in constraints
    assert "Do not create" in constraints
    assert "REQUIRED media display" not in constraints


class _Scene:
    def __init__(self, *, requested: list[str]) -> None:
        self.scene_expert_task_spec = {
            "required_large_objects": requested,
            "required_wall_objects": [],
        }
        self.objects = {
            "television_0": SimpleNamespace(
                object_id="television_0",
                object_type="furniture",
                name="television",
                description="television on stand",
                metadata={"semantic_name": "television"},
            ),
            "television_1": SimpleNamespace(
                object_id="television_1",
                object_type="wall_mounted",
                name="television",
                description="wall-mounted display",
                metadata={"semantic_name": "television"},
            ),
        }

    def remove_object(self, object_id: str) -> None:
        self.objects.pop(str(object_id))


def test_cross_stage_media_inventory_keeps_prompt_owned_furniture_tv() -> None:
    scene = _Scene(requested=["television"])

    removed = converge_cross_stage_media_inventory(
        scene, "- No explicit wall-object obligations were extracted from the prompt."
    )

    assert removed == ["television_1"]
    assert set(scene.objects) == {"television_0"}


def test_cross_stage_media_inventory_preserves_explicit_multiple_displays() -> None:
    scene = _Scene(requested=["television", "television"])

    removed = converge_cross_stage_media_inventory(
        scene, "- No explicit wall-object obligations were extracted from the prompt."
    )

    assert removed == []
    assert set(scene.objects) == {"television_0", "television_1"}


def test_wall_prepass_counts_as_initial_design_for_planner_recovery(
    monkeypatch,
) -> None:
    agent = object.__new__(StatefulWallAgent)
    agent.required_wall_object_constraints = "REQUIRED wall object: chalkboard"
    agent.scene = SimpleNamespace(
        get_objects_by_type=lambda _: [],
        content_hash=Mock(side_effect=["before", "after", "after"]),
    )
    agent.cfg = SimpleNamespace(
        agents=SimpleNamespace(planner_agent=SimpleNamespace(max_turns=2))
    )
    agent._planner_initial_design_tool_calls = 0
    agent._planner_successful_designer_mutations = 0
    agent._request_initial_design_impl = AsyncMock(return_value="placed")
    agent.prompt_registry = SimpleNamespace(get_prompt=lambda **_: "start")
    agent._run_planner_workflow = AsyncMock(
        return_value=SimpleNamespace(final_output="finished")
    )
    agent._can_skip_final_critique = lambda _: True
    agent._finalize_scene_and_scores = AsyncMock()
    monkeypatch.setattr(stateful_wall_agent, "log_agent_usage", lambda **_: None)
    monkeypatch.setattr(stateful_wall_agent, "log_agent_response", lambda **_: None)

    asyncio.run(agent._run_wall_workflow())

    assert agent._planner_initial_design_tool_calls == 1
    assert agent._planner_successful_designer_mutations == 1
    agent._run_planner_workflow.assert_awaited_once()


def test_wall_prepass_noop_does_not_suppress_planner_initial_design(
    monkeypatch,
) -> None:
    agent = object.__new__(StatefulWallAgent)
    agent.required_wall_object_constraints = "REQUIRED wall object: chalkboard"
    agent.scene = SimpleNamespace(
        get_objects_by_type=lambda _: [], content_hash=lambda: "unchanged"
    )
    agent.cfg = SimpleNamespace(
        agents=SimpleNamespace(planner_agent=SimpleNamespace(max_turns=2))
    )
    agent._planner_initial_design_tool_calls = 0
    agent._planner_successful_designer_mutations = 0
    agent._request_initial_design_impl = AsyncMock(return_value="no changes")
    agent.prompt_registry = SimpleNamespace(get_prompt=lambda **_: "start")
    agent._run_planner_workflow = AsyncMock(
        return_value=SimpleNamespace(final_output="finished")
    )
    agent._can_skip_final_critique = lambda _: True
    agent._finalize_scene_and_scores = AsyncMock()
    monkeypatch.setattr(stateful_wall_agent, "log_agent_usage", lambda **_: None)
    monkeypatch.setattr(stateful_wall_agent, "log_agent_response", lambda **_: None)

    asyncio.run(agent._run_wall_workflow())

    assert agent._planner_initial_design_tool_calls == 0
    assert agent._planner_successful_designer_mutations == 0
    assert (
        "already run"
        not in agent._run_planner_workflow.await_args.kwargs["runner_input"]
    )


def test_wall_prepass_counts_partial_mutation_before_turn_limit(monkeypatch) -> None:
    agent = object.__new__(StatefulWallAgent)
    agent.required_wall_object_constraints = "REQUIRED wall object: chalkboard"
    agent.scene = SimpleNamespace(
        get_objects_by_type=lambda _: [],
        content_hash=Mock(side_effect=["before", "after", "after"]),
    )
    agent.cfg = SimpleNamespace(
        agents=SimpleNamespace(planner_agent=SimpleNamespace(max_turns=2))
    )
    agent._planner_initial_design_tool_calls = 0
    agent._planner_successful_designer_mutations = 0
    agent._request_initial_design_impl = AsyncMock(
        side_effect=MaxTurnsExceeded("designer turn limit")
    )
    agent.prompt_registry = SimpleNamespace(get_prompt=lambda **_: "start")
    agent._run_planner_workflow = AsyncMock(
        return_value=SimpleNamespace(final_output="finished")
    )
    agent._can_skip_final_critique = lambda _: True
    agent._finalize_scene_and_scores = AsyncMock()
    monkeypatch.setattr(stateful_wall_agent, "log_agent_usage", lambda **_: None)
    monkeypatch.setattr(stateful_wall_agent, "log_agent_response", lambda **_: None)

    asyncio.run(agent._run_wall_workflow())

    assert agent._planner_initial_design_tool_calls == 1
    assert agent._planner_successful_designer_mutations == 1
