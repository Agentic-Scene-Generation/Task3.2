import unittest

from types import SimpleNamespace

from scenesmith.scene_expert.global_planner import GlobalPlanner
from scenesmith.scene_expert.harness import Harness
from scenesmith.scene_expert.schemas import (
    HarnessContext,
    MemoryPack,
    SceneTaskSpec,
    StageBudget,
    StageVerifyReport,
)


class SceneExpertHarnessBudgetTest(unittest.TestCase):
    def test_stage_budget_only_controls_outer_harness_iterations(self) -> None:
        cfg = SimpleNamespace(
            stage_budget=SimpleNamespace(
                default=SimpleNamespace(
                    max_designer_iterations=2,
                    max_repair_steps=1,
                ),
                manipuland=SimpleNamespace(
                    max_designer_iterations=3,
                    max_repair_steps=2,
                ),
            )
        )

        default_budget = Harness(cfg)._get_stage_budget("furniture")
        manipuland_budget = Harness(cfg)._get_stage_budget("manipuland")

        self.assertEqual(default_budget.max_designer_iterations, 2)
        self.assertEqual(default_budget.max_repair_steps, 1)
        self.assertEqual(manipuland_budget.max_designer_iterations, 3)
        self.assertEqual(manipuland_budget.max_repair_steps, 2)
        self.assertEqual(
            set(manipuland_budget.model_dump()),
            {"max_designer_iterations", "max_repair_steps"},
        )

    def test_disabled_budget_is_not_injected_or_enforced(self) -> None:
        cfg = SimpleNamespace(
            stage_budget=SimpleNamespace(
                default=SimpleNamespace(
                    max_designer_iterations=2,
                    max_repair_steps=1,
                )
            )
        )
        harness = Harness(cfg, budget_enabled=False)

        budget = harness._get_stage_budget("furniture")
        first = harness.decide_repair(
            "furniture", StageVerifyReport(stage="furniture", pass_stage=False)
        )
        second = harness.decide_repair(
            "furniture", StageVerifyReport(stage="furniture", pass_stage=False)
        )

        self.assertEqual(budget.max_designer_iterations, 0)
        self.assertEqual(budget.max_repair_steps, 0)
        self.assertTrue(first.should_repair)
        self.assertTrue(second.should_repair)
        self.assertIn("unbounded", second.reason)

        context = HarnessContext(
            stage="furniture",
            task_spec=SceneTaskSpec(room_type="office", style="modern"),
            memory_pack=MemoryPack(),
            stage_budget=StageBudget(
                max_designer_iterations=0,
                max_repair_steps=0,
            ),
        )
        prompt = object.__new__(GlobalPlanner)._build_user_message(
            context,
            scene_state_summary="",
        )
        self.assertNotIn("## Budget", prompt)


if __name__ == "__main__":
    unittest.main()
