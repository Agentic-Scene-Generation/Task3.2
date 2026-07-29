import unittest

from scenesmith.scene_expert.config_utils import (
    resolve_scene_expert_config,
    resolve_scene_expert_stage_budget,
    scene_expert_execution_control_enabled,
)


class SceneExpertConfigUtilsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = {
            "scene_expert": {
                "enabled": False,
                "mode": "disabled",
                "execution_control": {"enabled": True, "profile": "quality"},
                "stage_budget": {
                    "default": {
                        "max_designer_iterations": 2,
                        "max_planner_turns": 4,
                        "max_wall_clock_seconds": 600,
                    },
                    "floor_plan": {
                        "max_planner_turns": 8,
                        "max_critic_turns": 4,
                    },
                },
            },
            "experiment": {
                "scene_expert": {
                    "enabled": True,
                    "mode": "harness_memory",
                    "stage_budget": {"default": {"max_designer_iterations": 1}},
                }
            },
        }

    def test_experiment_overrides_inherit_root_defaults(self) -> None:
        resolved = resolve_scene_expert_config(self.cfg)

        self.assertTrue(resolved["enabled"])
        self.assertEqual(resolved["mode"], "harness_memory")
        self.assertEqual(
            resolved["stage_budget"]["default"]["max_designer_iterations"], 1
        )
        self.assertEqual(
            resolved["stage_budget"]["default"]["max_wall_clock_seconds"], 600
        )

    def test_floor_plan_budget_keeps_wall_clock_and_stage_override(self) -> None:
        budget = resolve_scene_expert_stage_budget(self.cfg, "floor_plan")

        self.assertEqual(budget["max_designer_iterations"], 1)
        self.assertEqual(budget["max_wall_clock_seconds"], 600)
        self.assertEqual(budget["max_planner_turns"], 8)
        self.assertEqual(budget["max_critic_turns"], 4)
        self.assertTrue(budget["execution_control_enabled"])
        self.assertEqual(budget["execution_control_profile"], "quality")

    def test_master_execution_switch_restores_native_budget(self) -> None:
        self.cfg["experiment"]["scene_expert"]["execution_control"] = {"enabled": False}

        self.assertFalse(scene_expert_execution_control_enabled(self.cfg))
        self.assertEqual(
            resolve_scene_expert_stage_budget(self.cfg, "furniture"),
            {},
        )


if __name__ == "__main__":
    unittest.main()
