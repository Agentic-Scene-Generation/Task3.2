from unittest.mock import Mock

from scenesmith.furniture_agents import stateful_furniture_agent
from scenesmith.furniture_agents.stateful_furniture_agent import StatefulFurnitureAgent
from scenesmith.agent_utils.base_stateful_agent import BaseStatefulAgent
from scenesmith.agent_utils.furniture_safety import HardStateEvaluation
from scenesmith.scenebenchmark_critic.config import CriticConfig


def _window_result(label: str, tier: str) -> dict:
    return {
        "check_id": "window_clearance__window_0",
        "metric": "interaction_clearance",
        "label": label,
        "scoring_tier": tier,
        "primary_object": "window_0",
        "blocking_objects": ["wardrobe_0"] if label == "fail" else [],
        "diagnostics": {
            "core_blocking_objects": ["wardrobe_0"] if label == "fail" else [],
            "occlusion_ratio": 0.4 if label == "fail" else 0.0,
        },
    }


def _relation_result(label: str) -> dict:
    return {
        "check_id": "intent_next_to__wardrobe_0__dresser_0",
        "metric": "functional_dependency",
        "label": label,
        "scoring_tier": "core",
        "primary_object": "wardrobe_0",
        "related_objects": ["dresser_0"],
        "evidence": {
            "intent_constraint": {
                "constraint_id": "storage_pair",
                "relation": "next_to",
                "strength": "hard",
            }
        },
    }


def _agent() -> StatefulFurnitureAgent:
    agent = StatefulFurnitureAgent.__new__(StatefulFurnitureAgent)
    agent.scene = Mock()
    agent.cfg = {}
    agent.rendering_manager = Mock()
    agent._reset_critic_candidate_cache = Mock()
    agent._begin_hard_state_transaction = Mock(
        return_value=({"scene": "before"}, set())
    )
    agent._repair_forbidden_zone_conflicts = Mock(return_value=True)
    return agent


def test_substantial_window_repair_accepts_fresh_no_regression_candidate(
    monkeypatch,
) -> None:
    agent = _agent()
    baseline = {"results": [_window_result("fail", "core"), _relation_result("pass")]}
    fresh = {"results": [_window_result("pass", "auxiliary"), _relation_result("pass")]}
    evaluate = Mock(side_effect=[baseline, fresh])
    monkeypatch.setattr(stateful_furniture_agent, "evaluate_room_scene", evaluate)
    monkeypatch.setattr(
        stateful_furniture_agent,
        "critic_config_from_any",
        lambda _cfg: CriticConfig(enabled=True),
    )

    actions = agent._repair_substantial_window_clearance()

    assert actions == [
        "cleared substantial window occlusion for window_0 by moving wardrobe_0"
    ]
    agent._repair_forbidden_zone_conflicts.assert_called_once_with(
        include_windows=True,
        opening_ids={"window_0"},
        blocker_ids={"wardrobe_0"},
    )
    assert evaluate.call_count == 2
    agent.scene.restore_from_state_dict.assert_not_called()


def test_substantial_window_repair_rolls_back_relation_regression(monkeypatch) -> None:
    agent = _agent()
    baseline = {"results": [_window_result("fail", "core"), _relation_result("pass")]}
    fresh = {"results": [_window_result("pass", "auxiliary"), _relation_result("fail")]}
    monkeypatch.setattr(
        stateful_furniture_agent,
        "evaluate_room_scene",
        Mock(side_effect=[baseline, fresh]),
    )
    monkeypatch.setattr(
        stateful_furniture_agent,
        "critic_config_from_any",
        lambda _cfg: CriticConfig(enabled=True),
    )

    actions = agent._repair_substantial_window_clearance()

    assert actions == []
    agent.scene.restore_from_state_dict.assert_called_once_with({"scene": "before"})
    assert agent.rendering_manager.clear_cache.call_count == 2
    assert agent._reset_critic_candidate_cache.call_count == 2


def test_substantial_window_repair_does_not_touch_advisory_overlap(monkeypatch) -> None:
    agent = _agent()
    monkeypatch.setattr(
        stateful_furniture_agent,
        "evaluate_room_scene",
        Mock(return_value={"results": [_window_result("degraded", "auxiliary")]}),
    )
    monkeypatch.setattr(
        stateful_furniture_agent,
        "critic_config_from_any",
        lambda _cfg: CriticConfig(enabled=True),
    )

    assert agent._repair_substantial_window_clearance() == []
    agent._repair_forbidden_zone_conflicts.assert_not_called()


def test_substantial_window_failure_disqualifies_rollback_checkpoint(
    monkeypatch,
) -> None:
    agent = _agent()
    monkeypatch.setattr(
        BaseStatefulAgent,
        "_checkpoint_eligible_furniture_hard_state",
        lambda _self, hard_state: hard_state,
    )
    monkeypatch.setattr(
        stateful_furniture_agent,
        "evaluate_room_scene",
        Mock(return_value={"results": [_window_result("fail", "core")]}),
    )
    monkeypatch.setattr(
        stateful_furniture_agent,
        "critic_config_from_any",
        lambda _cfg: CriticConfig(
            enabled=True,
            metrics=("interaction_clearance",),
        ),
    )

    result = agent._checkpoint_eligible_furniture_hard_state(
        HardStateEvaluation(hard_valid=True)
    )

    assert result is not None
    assert not result.hard_valid
    assert result.hard_reasons == [
        "substantial window occlusion: window_clearance__window_0"
    ]
