import unittest

from types import SimpleNamespace
from unittest.mock import patch

try:
    from scenesmith.agent_utils.base_stateful_agent import BaseStatefulAgent
except ModuleNotFoundError as exc:
    BaseStatefulAgent = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class StageRuntimeReserveTest(unittest.TestCase):
    @unittest.skipIf(
        BaseStatefulAgent is None,
        f"requires stateful agent dependencies: {_IMPORT_ERROR}",
    )
    def test_role_reserves_protect_critic_and_fallback_time(self) -> None:
        budget = {
            "max_wall_clock_seconds": 100.0,
            "critic_reserve_fraction": 0.25,
            "final_critic_reserve_fraction": 0.10,
            "fallback_reserve_fraction": 0.10,
            "finalization_reserve_fraction": 0.05,
        }
        agent = SimpleNamespace(
            _stage_runtime_started_at=100.0,
            _critic_evaluation_started_at=None,
            _stage_runtime_phase="agent",
            _stage_budget_value=lambda key, default: budget.get(key, default),
        )

        with patch(
            "scenesmith.agent_utils.base_stateful_agent.time.monotonic",
            return_value=140.0,
        ):
            designer_remaining = BaseStatefulAgent._remaining_stage_seconds(
                agent, "designer"
            )
            planner_remaining = BaseStatefulAgent._remaining_stage_seconds(
                agent, "planner"
            )
            critic_remaining = BaseStatefulAgent._remaining_stage_seconds(
                agent, "critic"
            )
            agent._stage_runtime_phase = "fallback"
            fallback_critic_remaining = BaseStatefulAgent._remaining_stage_seconds(
                agent, "critic"
            )

        self.assertAlmostEqual(designer_remaining, 20.0)
        self.assertAlmostEqual(planner_remaining, 35.0)
        self.assertAlmostEqual(critic_remaining, 45.0)
        self.assertAlmostEqual(fallback_critic_remaining, 55.0)

    @unittest.skipIf(
        BaseStatefulAgent is None,
        f"requires stateful agent dependencies: {_IMPORT_ERROR}",
    )
    def test_critic_evaluation_has_independent_bounded_window(self) -> None:
        budget = {
            "max_wall_clock_seconds": 100.0,
            "critic_evaluation_max_seconds": 360.0,
        }
        agent = SimpleNamespace(
            _stage_runtime_started_at=100.0,
            _critic_evaluation_started_at=220.0,
            _stage_runtime_phase="agent",
            _stage_budget_value=lambda key, default: budget.get(key, default),
        )

        with patch(
            "scenesmith.agent_utils.base_stateful_agent.time.monotonic",
            return_value=280.0,
        ):
            remaining = BaseStatefulAgent._remaining_stage_seconds(agent, "critic")

        self.assertAlmostEqual(remaining, 300.0)


if __name__ == "__main__":
    unittest.main()
