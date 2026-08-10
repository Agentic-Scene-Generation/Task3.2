import unittest

from scenesmith.scene_expert.config_utils import (
    resolve_component_flags,
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

    def test_mode_preset_can_be_overridden_per_component(self) -> None:
        self.cfg["experiment"]["scene_expert"]["components"] = {
            "memory_writer": {"enabled": False},
            "verifier": {"enabled": True},
        }

        flags = resolve_component_flags(self.cfg)

        self.assertTrue(flags["fast_memory_retrieval"])
        self.assertFalse(flags["memory_writer"])
        self.assertTrue(flags["verifier"])
        self.assertTrue(flags["critic_bridge"])

    def test_execution_subswitches_filter_only_their_budget_group(self) -> None:
        self.cfg["scene_expert"]["stage_budget"]["default"].update(
            {
                "planner_max_output_tokens": 768,
                "max_asset_requests": 4,
            }
        )
        self.cfg["experiment"]["scene_expert"]["execution_control"] = {
            "enabled": True,
            "token_budget_enabled": False,
            "asset_budget_enabled": False,
        }

        budget = resolve_scene_expert_stage_budget(self.cfg, "furniture")

        self.assertNotIn("planner_max_output_tokens", budget)
        self.assertNotIn("max_asset_requests", budget)
        self.assertEqual(budget["max_wall_clock_seconds"], 600)


if __name__ == "__main__":
    unittest.main()
