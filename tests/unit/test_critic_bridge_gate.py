import tempfile
import unittest

from pathlib import Path

from scenesmith.scene_expert.schemas import SceneTaskSpec
from scenesmith.scene_expert.verifier import FullVerifier, StageVerifier


class CriticBridgeGateTest(unittest.TestCase):
    def test_disabled_bridge_ignores_authoritative_scores_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scores_path = Path(tmp) / "scene_states" / "furniture" / "scores.yaml"
            scores_path.parent.mkdir(parents=True)
            scores_path.write_text(
                "aesthetic: 9\nfunctionality: 8\nsummary: critic evidence\n",
                encoding="utf-8",
            )
            task_spec = SceneTaskSpec(room_type="office", style="modern")

            bridged = StageVerifier(critic_bridge_enabled=True).verify(
                stage="furniture",
                stage_output_dir=tmp,
                task_spec=task_spec,
            )
            isolated = StageVerifier(critic_bridge_enabled=False).verify(
                stage="furniture",
                stage_output_dir=tmp,
                task_spec=task_spec,
            )

        self.assertEqual(bridged.critique_summary, "critic evidence")
        self.assertNotEqual(bridged.scores, isolated.scores)
        self.assertEqual(isolated.critique_summary, "")
        self.assertEqual(isolated.scores["semantic"], 0.5)

        isolated_full = FullVerifier().verify([isolated])
        self.assertEqual(isolated_full.overall_score, 0.0)


if __name__ == "__main__":
    unittest.main()
