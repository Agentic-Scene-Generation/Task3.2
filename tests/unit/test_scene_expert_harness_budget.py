import unittest

from types import SimpleNamespace

from scenesmith.scene_expert.harness import Harness


class SceneExpertHarnessBudgetTest(unittest.TestCase):
    def test_stage_override_inherits_default_execution_limits(self) -> None:
        cfg = SimpleNamespace(
            stage_budget=SimpleNamespace(
                default=SimpleNamespace(
                    max_designer_iterations=2,
                    max_repair_steps=1,
                    max_planner_turns=4,
                    max_designer_turns=12,
                    max_critic_turns=6,
                    max_wall_clock_seconds=600,
                    max_asset_requests=4,
                    max_optional_object_families=3,
                    max_assets_per_request=6,
                    max_semantic_retries_per_family=2,
                ),
                wall_mounted=SimpleNamespace(
                    max_designer_iterations=1,
                    max_wall_clock_seconds=300,
                    max_asset_requests=2,
                ),
            )
        )

        budget = Harness(cfg)._get_stage_budget("wall_mounted")

        self.assertEqual(budget.max_designer_iterations, 1)
        self.assertEqual(budget.max_wall_clock_seconds, 300)
        self.assertEqual(budget.max_asset_requests, 2)
        self.assertEqual(budget.max_designer_turns, 12)
        self.assertEqual(budget.max_semantic_retries_per_family, 2)


if __name__ == "__main__":
    unittest.main()

