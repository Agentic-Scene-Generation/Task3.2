import ast
import asyncio
import unittest

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


def _load_budget_compatibility_agent() -> type:
    """Load the two budget methods without importing optional ACP dependencies."""
    source_path = (
        Path(__file__).resolve().parents[2]
        / "scenesmith"
        / "agent_utils"
        / "base_stateful_agent.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    cfg_get = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_cfg_get"
    )
    base_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "BaseStatefulAgent"
    )
    method_names = {
        "_refresh_asset_runtime_budget",
        "_expand_critical_retry_budget",
        "_stage_budget_value",
        "_phase_budget_value",
        "_activate_runtime_phase",
        "_current_phase_role_consumption",
        "_begin_mandatory_repair_transaction",
        "_begin_critic_evaluation",
        "_begin_role_timer",
        "_critic_score_call_timeout",
        "_finish_role_timer",
        "_hard_repair_design_change_limit",
        "_planner_completion_contract",
        "_pause_current_role_timer",
        "_remaining_role_active_seconds",
        "_remaining_stage_seconds",
        "_resume_current_role_timer",
        "_stage_output_count_contract",
        "configure_stage_runtime_budget",
        "retry_final_critic_evaluation",
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
            cfg_get,
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
            {
                "execution_control_enabled": True,
                "max_wall_clock_seconds": 900.0,
            },
        )

        self.assertEqual(
            agent._stage_runtime_budget,
            {
                "execution_control_enabled": True,
                "max_wall_clock_seconds": 900.0,
            },
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
        budget = {
            "execution_control_enabled": True,
            "max_asset_requests": 8,
        }

        agent.configure_stage_runtime_budget(budget)

        configure_runtime_budget.assert_called_once_with(
            stage="furniture",
            budget=budget,
            required_objects=["bed", "wardrobe"],
            execution_clock=agent,
        )

    def test_role_reserves_protect_critic_and_fallback_time(self) -> None:
        budget = {
            "max_wall_clock_seconds": 100.0,
            "critic_reserve_fraction": 0.25,
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
            "time.monotonic",
            return_value=140.0,
        ):
            designer_remaining = BudgetCompatibilityAgent._remaining_stage_seconds(
                agent, "designer"
            )
            planner_remaining = BudgetCompatibilityAgent._remaining_stage_seconds(
                agent, "planner"
            )
            critic_remaining = BudgetCompatibilityAgent._remaining_stage_seconds(
                agent, "critic"
            )
            agent._stage_runtime_phase = "fallback"
            fallback_designer_remaining = (
                BudgetCompatibilityAgent._remaining_stage_seconds(agent, "designer")
            )
            fallback_critic_remaining = (
                BudgetCompatibilityAgent._remaining_stage_seconds(agent, "critic")
            )

        self.assertAlmostEqual(designer_remaining, 20.0)
        self.assertAlmostEqual(planner_remaining, 55.0)
        self.assertAlmostEqual(critic_remaining, 45.0)
        self.assertAlmostEqual(fallback_designer_remaining, 30.0)
        self.assertAlmostEqual(fallback_critic_remaining, 55.0)
        self.assertGreater(planner_remaining, designer_remaining)
        self.assertGreater(planner_remaining, critic_remaining)

    def test_nested_role_time_is_not_double_charged(self) -> None:
        agent = BudgetCompatibilityAgent()
        agent._agent_execution_stack = []

        with patch(
            "time.monotonic",
            side_effect=[0.0, 2.0, 2.0, 5.0, 5.0, 7.0],
        ):
            planner_timer = agent._begin_role_timer("planner")
            designer_timer = agent._begin_role_timer("designer")
            designer_seconds = agent._finish_role_timer(designer_timer)
            planner_seconds = agent._finish_role_timer(planner_timer)

        self.assertEqual(3.0, designer_seconds)
        self.assertEqual(4.0, planner_seconds)

    def test_external_asset_time_is_not_charged_to_stage_inference(self) -> None:
        budget = {
            "max_wall_clock_seconds": 100.0,
            "critic_reserve_fraction": 0.25,
            "fallback_reserve_fraction": 0.10,
            "finalization_reserve_fraction": 0.05,
        }
        agent = SimpleNamespace(
            _stage_runtime_started_at=100.0,
            _stage_external_paused_seconds=30.0,
            _critic_evaluation_started_at=None,
            _stage_runtime_phase="agent",
            _stage_budget_value=lambda key, default: budget.get(key, default),
        )

        with patch("time.monotonic", return_value=150.0):
            remaining = BudgetCompatibilityAgent._remaining_stage_seconds(
                agent, "designer"
            )

        # 60-second designer window minus 20 seconds of actual inference.
        self.assertAlmostEqual(remaining, 40.0)

    def test_critic_evaluation_has_an_isolated_quality_window(self) -> None:
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
            "time.monotonic",
            return_value=280.0,
        ):
            remaining = BudgetCompatibilityAgent._remaining_stage_seconds(
                agent, "critic"
            )

        self.assertAlmostEqual(remaining, 300.0)

    def test_sceneexpert_scoring_uses_one_transaction_deadline(self) -> None:
        agent = BudgetCompatibilityAgent()
        agent._stage_runtime_budget = {"critic_evaluation_max_seconds": 240.0}
        agent._stage_role_active_consumed = {"critic": 139.0, "designer": 20.0}

        with patch(
            "time.monotonic",
            return_value=500.0,
        ):
            agent._begin_critic_evaluation()

        self.assertEqual(500.0, agent._critic_evaluation_started_at)
        self.assertNotIn("critic", agent._stage_role_active_consumed)
        self.assertEqual(20.0, agent._stage_role_active_consumed["designer"])
        self.assertIsNone(agent._critic_score_call_timeout(120.0))

        agent._stage_runtime_budget = {}
        self.assertEqual(120.0, agent._critic_score_call_timeout(120.0))

    def test_mandatory_repair_gets_fresh_designer_lease(self) -> None:
        agent = BudgetCompatibilityAgent()
        agent._stage_runtime_budget = {"max_repair_steps": 2}
        agent._stage_runtime_phase = "agent"
        agent._stage_phase_started_at = 100.0
        agent._stage_role_active_consumed = {
            "designer": 359.0,
            "planner": 20.0,
        }
        agent._stage_role_active_consumed_by_phase = {
            "agent": agent._stage_role_active_consumed,
        }

        previous_phase, previous_started_at = (
            agent._begin_mandatory_repair_transaction()
        )

        self.assertEqual("agent", previous_phase)
        self.assertEqual(100.0, previous_started_at)
        self.assertEqual("repair", agent._stage_runtime_phase)
        self.assertNotIn("designer", agent._stage_role_active_consumed)
        self.assertEqual({}, agent._stage_role_active_consumed)
        self.assertEqual(
            20.0,
            agent._stage_role_active_consumed_by_phase["agent"]["planner"],
        )

    def test_recovery_phase_uses_independent_wall_clock_and_role_lease(self) -> None:
        agent = BudgetCompatibilityAgent()
        agent._stage_runtime_budget = {
            "max_wall_clock_seconds": 100.0,
            "designer_active_max_seconds": 80.0,
            "repair_max_wall_clock_seconds": 300.0,
            "repair_designer_active_max_seconds": 240.0,
            "critic_reserve_fraction": 0.0,
            "fallback_reserve_fraction": 0.0,
            "finalization_reserve_fraction": 0.0,
        }
        agent._stage_runtime_started_at = 100.0
        agent._stage_runtime_phase = "repair"
        agent._stage_phase_started_at = 190.0
        agent._stage_external_paused_seconds = 0.0
        agent._stage_role_active_consumed = {"designer": 40.0}
        agent._stage_role_active_consumed_by_phase = {
            "agent": {"designer": 75.0},
            "repair": agent._stage_role_active_consumed,
        }

        with patch("time.monotonic", return_value=200.0):
            wall_remaining = agent._remaining_stage_seconds("designer")
            role_remaining = agent._remaining_role_active_seconds("designer")

        self.assertAlmostEqual(290.0, wall_remaining)
        self.assertAlmostEqual(200.0, role_remaining)

    def test_role_consumption_is_restored_with_runtime_phase(self) -> None:
        agent = BudgetCompatibilityAgent()
        agent._stage_runtime_phase = "agent"
        agent._stage_role_active_consumed = {"designer": 70.0}
        agent._stage_role_active_consumed_by_phase = {
            "agent": agent._stage_role_active_consumed,
        }

        agent._activate_runtime_phase("repair", reset_role_consumption=True)
        agent._stage_role_active_consumed["designer"] = 25.0
        agent._activate_runtime_phase("agent")

        self.assertEqual({"designer": 70.0}, agent._stage_role_active_consumed)
        self.assertEqual(
            {"designer": 25.0},
            agent._stage_role_active_consumed_by_phase["repair"],
        )

    def test_critical_retry_expansion_does_not_mutate_normal_budget(self) -> None:
        agent = BudgetCompatibilityAgent()
        agent._stage_runtime_budget = {
            "max_wall_clock_seconds": 100.0,
            "critic_active_max_seconds": 80.0,
            "critic_evaluation_max_seconds": 120.0,
            "critic_retry_evaluation_max_seconds": 300.0,
            "critical_retry_budget_multiplier": 2.0,
        }
        agent._critical_retry_budget_expanded = False

        agent._expand_critical_retry_budget()

        self.assertEqual(100.0, agent._stage_runtime_budget["max_wall_clock_seconds"])
        self.assertEqual(
            200.0,
            agent._stage_runtime_budget["critic_retry_max_wall_clock_seconds"],
        )
        self.assertEqual(
            160.0,
            agent._stage_runtime_budget["critic_retry_critic_active_max_seconds"],
        )
        self.assertEqual(
            300.0,
            agent._stage_runtime_budget["critic_retry_evaluation_max_seconds"],
        )

    def test_critic_retry_restores_parent_runtime_clock(self) -> None:
        agent = BudgetCompatibilityAgent()
        agent._stage_runtime_budget = {
            "critic_retry_evaluation_max_seconds": 300.0,
        }
        agent._stage_runtime_phase = "regeneration"
        agent._stage_runtime_started_at = 10.0
        agent._stage_phase_started_at = 20.0
        agent._stage_external_paused_seconds = 7.0
        agent._external_operation_depth = 0
        agent._external_operation_started_at = None
        agent._external_paused_lease = None
        agent._critic_evaluation_started_at = 30.0
        agent._stage_runtime_exhausted = True
        agent._critical_retry_budget_expanded = False
        agent._critical_retry_compact_context = False
        agent._stage_role_active_consumed = {"critic": 90.0}
        agent._stage_role_active_consumed_by_phase = {
            "regeneration": agent._stage_role_active_consumed,
        }

        async def request_critique(*, update_checkpoint: bool) -> None:
            self.assertFalse(update_checkpoint)

        async def finalize() -> None:
            return None

        agent._request_critique_impl = request_critique
        agent._finalize_scene_and_scores = finalize

        asyncio.run(agent.retry_final_critic_evaluation())

        self.assertEqual("regeneration", agent._stage_runtime_phase)
        self.assertEqual(10.0, agent._stage_runtime_started_at)
        self.assertEqual(20.0, agent._stage_phase_started_at)
        self.assertEqual(7.0, agent._stage_external_paused_seconds)
        self.assertEqual(30.0, agent._critic_evaluation_started_at)
        self.assertTrue(agent._stage_runtime_exhausted)
        self.assertEqual(
            {"critic": 90.0},
            agent._stage_role_active_consumed,
        )

    def test_harness_repair_budget_overrides_smaller_legacy_limit(self) -> None:
        budget = {"max_repair_steps": 2}
        agent = BudgetCompatibilityAgent()
        agent._stage_runtime_budget = budget
        agent._critic_fast_path_cfg = lambda: {
            "max_hard_repair_design_changes": 1,
        }

        limit = agent._hard_repair_design_change_limit()

        self.assertEqual(2, limit)

    def test_optional_target_is_separate_from_required_output_minimum(self) -> None:
        agent = BudgetCompatibilityAgent()
        agent.agent_type = SimpleNamespace(
            to_object_type=lambda: "wall_mounted",
        )
        agent.scene = SimpleNamespace(
            scene_expert_min_output_objects=1,
            scene_expert_required_min_output_objects=0,
            get_objects_by_type=lambda object_type: [],
        )

        required, target, current = agent._stage_output_count_contract()

        self.assertEqual((required, target, current), (0, 1, 0))

    def test_planner_contract_preserves_optional_target_semantics(self) -> None:
        budget = {"min_output_objects": 1, "max_output_objects": 3}
        agent = BudgetCompatibilityAgent()
        agent.scene = SimpleNamespace(
            scene_expert_min_output_objects=1,
            scene_expert_required_min_output_objects=0,
            scene_expert_max_output_objects=3,
            get_objects_by_type=lambda object_type: [],
        )
        agent.agent_type = SimpleNamespace(
            to_object_type=lambda: "wall_mounted",
        )
        agent._stage_runtime_budget = budget

        contract = agent._planner_completion_contract()

        self.assertIn("must call request_initial_design", contract)
        self.assertIn("at least 1 and no more than 3", contract)
        self.assertIn("preferred quality target", contract)
        self.assertNotIn("not valid", contract)

    def test_prompt_requirements_raise_planner_count_contract(self) -> None:
        budget = {"min_output_objects": 1, "max_output_objects": 3}
        agent = BudgetCompatibilityAgent()
        agent.scene = SimpleNamespace(
            scene_expert_min_output_objects=4,
            scene_expert_required_min_output_objects=4,
            scene_expert_max_output_objects=4,
            get_objects_by_type=lambda object_type: [],
        )
        agent.agent_type = SimpleNamespace(
            to_object_type=lambda: "wall_mounted",
        )
        agent._stage_runtime_budget = budget

        contract = agent._planner_completion_contract()

        self.assertIn("at least 4 and no more than 4", contract)


if __name__ == "__main__":
    unittest.main()
