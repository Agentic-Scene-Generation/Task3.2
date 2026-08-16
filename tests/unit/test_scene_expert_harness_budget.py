import unittest

from types import SimpleNamespace

from scenesmith.scene_expert.harness import Harness


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


if __name__ == "__main__":
    unittest.main()
