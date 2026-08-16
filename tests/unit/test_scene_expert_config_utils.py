import unittest

from scenesmith.scene_expert.config_utils import (
    resolve_component_flags,
    resolve_scene_expert_config,
)


class SceneExpertConfigUtilsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = {
            "scene_expert": {
                "enabled": False,
                "mode": "disabled",
                "stage_budget": {
                    "default": {
                        "max_designer_iterations": 2,
                        "max_repair_steps": 1,
                    }
                },
            },
            "experiment": {
                "scene_expert": {
                    "enabled": True,
                    "mode": "harness_memory",
                    "stage_budget": {
                        "default": {"max_designer_iterations": 1}
                    },
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
            resolved["stage_budget"]["default"]["max_repair_steps"], 1
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


if __name__ == "__main__":
    unittest.main()
