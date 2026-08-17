import unittest

from scenesmith.scene_expert.config_utils import (
    intent_contract_is_authoritative,
    resolve_component_flags,
    resolve_scene_expert_config,
    should_run_sceneexpert_task_compiler,
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

    def test_only_successful_intent_contract_is_authoritative(self) -> None:
        contract = {"constraints": [{"relation": "on_top_of"}]}

        self.assertTrue(
            intent_contract_is_authoritative(contract, {"status": "ok"})
        )
        self.assertFalse(
            intent_contract_is_authoritative(contract, {"status": "fallback"})
        )
        self.assertFalse(
            intent_contract_is_authoritative(
                contract,
                {"status": "deterministic_fallback"},
            )
        )
        self.assertFalse(intent_contract_is_authoritative({}, {"status": "ok"}))

    def test_auto_task_compiler_takes_over_critic_fallback(self) -> None:
        contract = {"constraints": [{"relation": "on_top_of"}]}

        self.assertTrue(
            should_run_sceneexpert_task_compiler(
                component_enabled=True,
                source="auto",
                intent_contract=contract,
                intent_trace={"status": "fallback"},
            )
        )
        self.assertFalse(
            should_run_sceneexpert_task_compiler(
                component_enabled=True,
                source="auto",
                intent_contract=contract,
                intent_trace={"status": "ok"},
            )
        )
        self.assertFalse(
            should_run_sceneexpert_task_compiler(
                component_enabled=False,
                source="sceneexpert",
                intent_contract={},
                intent_trace={},
            )
        )


if __name__ == "__main__":
    unittest.main()
