import ast
import asyncio
import unittest

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

try:
    from scenesmith.agent_utils.base_stateful_agent import BaseStatefulAgent
except ModuleNotFoundError as exc:
    BaseStatefulAgent = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def _load_budget_compatibility_agent() -> type:
    """Load the two budget methods without importing optional ACP dependencies."""
    source_path = (
        Path(__file__).resolve().parents[2]
        / "scenesmith"
        / "agent_utils"
        / "base_stateful_agent.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    base_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "BaseStatefulAgent"
    )
    method_names = {
        "_refresh_asset_runtime_budget",
        "configure_stage_runtime_budget",
    }
    methods = [
        node
        for node in base_class.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in method_names
    ]
    compatibility_class = ast.ClassDef(
        name="_BudgetCompatibilityAgent",
        bases=[],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            ast.Import(names=[ast.alias(name="time")]),
            ast.ImportFrom(
                module="scenesmith.scene_expert.critic_feedback",
                names=[ast.alias(name="CriticFeedback")],
                level=0,
            ),
            compatibility_class,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace: dict[str, object] = {}
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace["_BudgetCompatibilityAgent"]


BudgetCompatibilityAgent = _load_budget_compatibility_agent()


class StageRuntimeReserveTest(unittest.TestCase):
    def test_floor_plan_budget_configuration_does_not_require_scene(self) -> None:
        agent = BudgetCompatibilityAgent()

        agent.configure_stage_runtime_budget(
            {"max_wall_clock_seconds": 900.0},
        )

        self.assertEqual(
            agent._stage_runtime_budget,
            {"max_wall_clock_seconds": 900.0},
        )
        self.assertEqual(agent._stage_runtime_phase, "agent")
        self.assertFalse(agent._stage_runtime_exhausted)

    def test_placement_budget_configuration_refreshes_asset_manager(self) -> None:
        configure_runtime_budget = Mock()
        agent = BudgetCompatibilityAgent()
        agent.scene = SimpleNamespace(
            scene_expert_stage="furniture",
            scene_expert_required_objects=["bed", "wardrobe"],
        )
        agent.asset_manager = SimpleNamespace(
            configure_runtime_budget=configure_runtime_budget,
        )
        agent.agent_type = SimpleNamespace(value="furniture")
        budget = {"max_asset_requests": 8}

        agent.configure_stage_runtime_budget(budget)

        configure_runtime_budget.assert_called_once_with(
            stage="furniture",
            budget=budget,
            required_objects=["bed", "wardrobe"],
        )

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
            fallback_designer_remaining = (
                BaseStatefulAgent._remaining_stage_seconds(agent, "designer")
            )
            fallback_critic_remaining = BaseStatefulAgent._remaining_stage_seconds(
                agent, "critic"
            )

        self.assertAlmostEqual(designer_remaining, 20.0)
        self.assertAlmostEqual(planner_remaining, 55.0)
        self.assertAlmostEqual(critic_remaining, 45.0)
        self.assertAlmostEqual(fallback_designer_remaining, 30.0)
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

    @unittest.skipIf(
        BaseStatefulAgent is None,
        f"requires stateful agent dependencies: {_IMPORT_ERROR}",
    )
    def test_prompt_requirements_raise_planner_count_contract(self) -> None:
        budget = {"min_output_objects": 1, "max_output_objects": 3}
        agent = SimpleNamespace(
            scene=SimpleNamespace(
                scene_expert_min_output_objects=4,
                scene_expert_max_output_objects=4,
            ),
            _stage_runtime_budget=budget,
            _stage_budget_value=lambda key, default: budget.get(key, default),
        )

        contract = BaseStatefulAgent._planner_completion_contract(agent)

        self.assertIn("at least 4 and no more than 4", contract)


if __name__ == "__main__":
    unittest.main()
