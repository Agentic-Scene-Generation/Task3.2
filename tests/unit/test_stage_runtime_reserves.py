import asyncio
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
        self.assertAlmostEqual(planner_remaining, 55.0)
        self.assertAlmostEqual(critic_remaining, 45.0)
        self.assertAlmostEqual(fallback_critic_remaining, 55.0)
        self.assertGreater(planner_remaining, designer_remaining)
        self.assertGreater(planner_remaining, critic_remaining)

    @unittest.skipIf(
        BaseStatefulAgent is None,
        f"requires stateful agent dependencies: {_IMPORT_ERROR}",
    )
    def test_critic_evaluation_is_bounded_by_total_stage_window(self) -> None:
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

        self.assertAlmostEqual(remaining, -95.0)

    @unittest.skipIf(
        BaseStatefulAgent is None,
        f"requires stateful agent dependencies: {_IMPORT_ERROR}",
    )
    def test_required_output_rescue_bypasses_planner(self) -> None:
        events: list[tuple[str, object]] = []

        async def request_change(instruction: str) -> str:
            events.append(("designer", instruction))
            return "placed one object"

        async def request_critique(*, update_checkpoint: bool) -> str:
            events.append(("critic", update_checkpoint))
            return "accepted"

        async def finalize() -> None:
            events.append(("finalize", None))

        agent = SimpleNamespace(
            _stage_budget_value=lambda key, default: {
                "min_output_objects": 1,
                "max_output_objects": 3,
            }.get(key, default),
            _request_design_change_impl=request_change,
            _request_critique_impl=request_critique,
            _finalize_scene_and_scores=finalize,
        )

        asyncio.run(
            BaseStatefulAgent.run_required_stage_completion_rescue(
                agent,
                ["missing required stage output"],
            )
        )

        self.assertEqual(
            [event[0] for event in events], ["designer", "critic", "finalize"]
        )
        self.assertIn("between 1 and 3", events[0][1])
        self.assertEqual(agent._stage_runtime_phase, "agent")

    @unittest.skipIf(
        BaseStatefulAgent is None,
        f"requires stateful agent dependencies: {_IMPORT_ERROR}",
    )
    def test_planner_contract_overrides_optional_zero_object_guidance(self) -> None:
        budget = {"min_output_objects": 1, "max_output_objects": 3}
        agent = SimpleNamespace(
            _stage_runtime_budget=budget,
            _stage_budget_value=lambda key, default: budget.get(key, default),
        )

        contract = BaseStatefulAgent._planner_completion_contract(agent)

        self.assertIn("must call request_initial_design", contract)
        self.assertIn("at least 1 and no more than 3", contract)
        self.assertIn("zero-object result is not valid", contract)


if __name__ == "__main__":
    unittest.main()
