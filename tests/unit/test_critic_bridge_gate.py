import tempfile
import unittest

from pathlib import Path

from scenesmith.scene_expert.schemas import SceneTaskSpec
from scenesmith.scene_expert.verifier import (
    FullVerifier,
    StageVerifier,
    _critique_is_final,
)


class CriticBridgeGateTest(unittest.TestCase):
    def test_literal_placeholder_is_not_a_completed_critique(self) -> None:
        self.assertFalse(_critique_is_final("placeholder"))
        self.assertFalse(_critique_is_final("  PLACEHOLDER\n"))
        self.assertTrue(_critique_is_final("A placeholder mesh blocks the doorway."))

    def test_placeholder_grades_remain_unavailable_visual_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scores_path = Path(tmp) / "scene_states" / "furniture" / "scores.yaml"
            scores_path.parent.mkdir(parents=True)
            original = (
                "realism: {grade: 0, comment: placeholder}\n"
                "functionality: {grade: 0, comment: placeholder}\n"
                "summary: placeholder\n"
            )
            scores_path.write_text(original, encoding="utf-8")
            report = StageVerifier().verify(
                stage="furniture",
                stage_output_dir=tmp,
                task_spec=SceneTaskSpec(room_type="kitchen", style="country"),
            )
            self.assertEqual(scores_path.read_text(encoding="utf-8"), original)
        self.assertEqual(report.visual_scores, {})
        self.assertEqual(report.critique_summary, "")

    def test_real_zero_grades_are_retained(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scores_path = Path(tmp) / "scene_states" / "furniture" / "scores.yaml"
            scores_path.parent.mkdir(parents=True)
            scores_path.write_text(
                "realism: {grade: 0, comment: Severe geometry failures}\n"
                "functionality: {grade: 0, comment: Door and storage blocked}\n"
                "summary: Severe geometry and access failures; unusable scene.\n",
                encoding="utf-8",
            )
            report = StageVerifier().verify(
                stage="furniture",
                stage_output_dir=tmp,
                task_spec=SceneTaskSpec(room_type="kitchen", style="country"),
            )
        self.assertTrue(report.visual_scores)
        self.assertTrue(all(value == 0.0 for value in report.visual_scores.values()))
        self.assertIn("unusable scene", report.critique_summary)

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
