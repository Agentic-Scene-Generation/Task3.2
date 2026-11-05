"""Regressions for native geometry drift and persisted failed-target history."""

import asyncio
import random

from types import SimpleNamespace

import pytest

from scenesmith.scene_expert.native_control import (
    install_opening_randomness,
    install_target_retry_sessions,
)


def test_openings_share_a_private_stream_per_instance_and_restore_binding():
    module = SimpleNamespace(random=random)

    class Tools:
        def _add_door_impl(self):
            return module.random.uniform(0.4, 0.8)

        def _add_window_impl(self):
            return module.random.uniform(0.5, 1.6)

    install_opening_randomness(Tools, module, seed=42)
    installed = Tools._add_door_impl
    install_opening_randomness(Tools, module, seed=42)
    assert Tools._add_door_impl is installed
    first, second = Tools(), Tools()
    expected = random.Random(42)
    assert first._add_door_impl() == expected.uniform(0.4, 0.8)
    for _ in range(17):
        random.random()
    assert first._add_window_impl() == expected.uniform(0.5, 1.6)
    another = random.Random(42)
    assert second._add_door_impl() == another.uniform(0.4, 0.8)
    assert second._add_window_impl() == another.uniform(0.5, 1.6)
    assert module.random is random
    with pytest.raises(ValueError, match="different seed"):
        install_opening_randomness(Tools, module, seed=43)


def test_opening_error_restores_module_rng_and_preserves_native_failure():
    module = SimpleNamespace(random=random)

    class Tools:
        def _add_door_impl(self):
            module.random.random()
            raise RuntimeError("native failure")

        def _add_window_impl(self):
            return module.random.random()

    install_opening_randomness(Tools, module, seed=42)
    with pytest.raises(RuntimeError, match="native failure"):
        Tools()._add_door_impl()
    assert module.random is random


@pytest.mark.parametrize("seed", [True, 4.2, "42", None])
def test_opening_seed_must_be_an_integer(seed):
    with pytest.raises(ValueError, match="integer"):
        install_opening_randomness(object, SimpleNamespace(random=random), seed=seed)


def test_retry_factory_is_restored_after_setup_failure():
    class Agent:
        _stage_execution_attempt = 1

        def _create_sessions(self, session_prefix="", *, stage_execution_attempt=None):
            return session_prefix

        def _setup_furniture_agents(self, furniture_id, furniture_description):
            self.prefix = self._create_sessions(session_prefix=f"{furniture_id}_")
            if "target_retry" in self.prefix:
                raise RuntimeError("native setup failure")

    install_target_retry_sessions(Agent)
    installed = Agent._setup_furniture_agents
    install_target_retry_sessions(Agent)
    assert Agent._setup_furniture_agents is installed
    agent = Agent()
    agent._setup_furniture_agents("bed_0", "bed")
    assert agent.prefix == "bed_0_"
    original_factory = lambda **kwargs: kwargs["session_prefix"]
    agent._create_sessions = original_factory
    with pytest.raises(RuntimeError, match="native setup failure"):
        agent._setup_furniture_agents("bed_0", "bed")
    assert agent._create_sessions is original_factory
    assert agent.prefix == "bed_0_target_retry_02_"
    agent._stage_execution_attempt = 2
    agent._setup_furniture_agents("bed_0", "bed")
    assert agent.prefix == "bed_0_"


def test_actual_opening_tools_are_reproducible_across_ambient_rng(monkeypatch):
    pytest.importorskip("pydrake")
    from scenesmith.agent_utils.house import HouseLayout
    from scenesmith.floor_plan_agents.tools import door_window_mixin as opening
    from scenesmith.floor_plan_agents.tools.floor_plan_tools import FloorPlanTools

    for name in ("_add_door_impl", "_add_window_impl"):
        monkeypatch.setattr(
            opening.DoorWindowMixin, name, getattr(opening.DoorWindowMixin, name)
        )
    install_opening_randomness(opening.DoorWindowMixin, opening, seed=42)
    results = []
    for ambient in (1001, 2002):
        random.seed(ambient)
        layout = HouseLayout()
        tools = FloorPlanTools(layout=layout, mode="room")
        assert tools._generate_room_specs_impl(
            '[{"type":"bedroom","width":4.0,"depth":3.5}]'
        ).success
        assert not tools._add_door_impl(
            wall_id="D", position="bottom", width=0.9, height=2.1
        ).success
        assert tools._add_door_impl(
            wall_id="D", position="left", width=0.9, height=2.1
        ).success
        for _ in range(17):
            random.random()
        assert tools._add_window_impl(
            wall_id="C", position="center", width=1.2, height=1.2, sill_height=0.9
        ).success
        results.append(
            (layout.doors[0].position_exact, layout.windows[0].position_along_wall)
        )
    assert results[0] == results[1]
    assert opening.random is random


def test_actual_target_retry_has_empty_all_role_histories_and_keeps_old_db(
    tmp_path, monkeypatch
):
    pytest.importorskip("pydrake")
    from omegaconf import OmegaConf
    from scenesmith.manipuland_agents.stateful_manipuland_agent import (
        StatefulManipulandAgent,
    )

    monkeypatch.setattr(
        StatefulManipulandAgent,
        "_setup_furniture_agents",
        StatefulManipulandAgent._setup_furniture_agents,
    )
    install_target_retry_sessions(StatefulManipulandAgent)
    agent = object.__new__(StatefulManipulandAgent)
    agent.logger = SimpleNamespace(output_dir=tmp_path)
    agent.cfg = OmegaConf.create({"session_memory": {"enabled": False}})
    agent._stage_execution_attempt = 1
    agent._create_tools_for_furniture = lambda furniture_id: ([], [], [])
    agent._create_designer_agent = lambda **kwargs: SimpleNamespace()
    agent._create_critic_agent = lambda **kwargs: SimpleNamespace()
    agent._create_planner_tools = lambda: []
    agent._create_planner_agent = lambda **kwargs: SimpleNamespace()
    stop = {
        "type": "function_call_output",
        "call_id": "failure",
        "output": "STOP: Initial designer failed; do not restart.",
    }

    async def exercise():
        agent._setup_furniture_agents("bed_0", "bed")
        previous = [agent.designer_session, agent.critic_session, agent.planner_session]
        for session in previous:
            await session.add_items([stop])
        agent._setup_furniture_agents("bed_0", "bed")
        current = [agent.designer_session, agent.critic_session, agent.planner_session]
        for old, new in zip(previous, current, strict=True):
            assert old.session_id != new.session_id
            assert await new.get_items() == []
            assert await old.get_items() == [stop]
        assert agent._stage_execution_attempt == 1
        assert "_create_sessions" not in agent.__dict__
        agent._setup_furniture_agents("nightstand_0", "nightstand")
        assert agent.planner_session.session_id == "nightstand_0_planner_attempt_01"
        agent._stage_execution_attempt = 2
        agent._setup_furniture_agents("bed_0", "bed")
        assert agent.planner_session.session_id == "bed_0_planner_attempt_02"
        assert await agent.planner_session.get_items() == []
        for session in previous:
            assert await session.get_items() == [stop]

    asyncio.run(exercise())
    for role in ("designer", "critic", "planner"):
        assert (tmp_path / f"bed_0_{role}.db").is_file()
        assert (tmp_path / f"bed_0_target_retry_02_{role}.db").is_file()
