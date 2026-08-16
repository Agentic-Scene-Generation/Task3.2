import unittest

from pathlib import Path

from scenesmith.scene_expert.schemas import StageBudget


_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class SceneExpertRuntimeBoundaryTest(unittest.TestCase):
    def _source(self, relative_path: str) -> str:
        return (_PROJECT_ROOT / relative_path).read_text(encoding="utf-8")

    def test_harness_budget_does_not_override_scenesmith_runtime(self) -> None:
        self.assertEqual(
            set(StageBudget.model_fields),
            {"max_designer_iterations", "max_repair_steps"},
        )

        hooks_source = self._source("scenesmith/scene_expert/hooks.py")
        self.assertNotIn("scene_expert_stage_budget", hooks_source)
        self.assertNotIn("scene_expert_min_output_objects", hooks_source)
        self.assertNotIn("scene_expert_max_output_objects", hooks_source)

    def test_asset_and_critic_ownership_remain_in_scenesmith(self) -> None:
        asset_source = self._source("scenesmith/agent_utils/asset_manager.py")
        scoring_source = self._source("scenesmith/agent_utils/scoring.py")
        verifier_source = self._source("scenesmith/scene_expert/verifier.py")

        self.assertNotIn("AssetRuntimeGate", asset_source)
        self.assertNotIn("scene_expert_stage_budget", asset_source)
        self.assertNotIn("scenesmith.scene_expert", scoring_source)
        self.assertNotIn("room_size_policy", verifier_source)
        self.assertNotIn("placeholder_asset", verifier_source)

    def test_sceneexpert_verifier_only_bridges_existing_critic_output(self) -> None:
        verifier_source = self._source("scenesmith/scene_expert/verifier.py")

        self.assertIn("_find_scores_yaml", verifier_source)
        self.assertIn("critic_bridge_enabled", verifier_source)
        self.assertNotIn("chat.completions.create", verifier_source)


if __name__ == "__main__":
    unittest.main()
