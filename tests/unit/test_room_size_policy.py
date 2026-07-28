"""Regression tests for deterministic professional room sizing."""

import unittest

from tempfile import TemporaryDirectory

from scenesmith.agent_utils.room_size_policy import (
    normalize_room_dimensions,
    prompt_has_explicit_room_dimensions,
)
from scenesmith.scene_expert.schemas import SceneTaskSpec
from scenesmith.scene_expert.verifier import StageVerifier, _check_floor_plan_layout


class TestRoomSizePolicy(unittest.TestCase):
    def test_unqualified_bedroom_is_reduced_from_global_limit(self) -> None:
        result = normalize_room_dimensions(
            room_type="bedroom",
            width=20.0,
            depth=20.0,
            prompt="A simple bedroom with a bed, two nightstands, and a wardrobe.",
        )

        self.assertTrue(result.changed)
        self.assertLessEqual(result.width * result.depth, 24.01)
        self.assertLessEqual(max(result.width, result.depth), 5.5)

    def test_explicit_room_dimensions_are_preserved(self) -> None:
        prompt = "Create a 19m x 18m open-plan coworking space."
        result = normalize_room_dimensions(
            room_type="office",
            width=19.0,
            depth=18.0,
            prompt=prompt,
        )

        self.assertTrue(prompt_has_explicit_room_dimensions(prompt))
        self.assertFalse(result.changed)
        self.assertEqual((result.width, result.depth), (19.0, 18.0))

    def test_explicit_chinese_room_dimensions_are_preserved(self) -> None:
        prompt = "创建一间尺寸为19米×18米的卧室。"
        result = normalize_room_dimensions(
            room_type="卧室",
            width=19.0,
            depth=18.0,
            prompt=prompt,
        )

        self.assertTrue(prompt_has_explicit_room_dimensions(prompt))
        self.assertFalse(result.changed)

    def test_object_dimensions_do_not_disable_room_safeguard(self) -> None:
        prompt = "A bedroom with a 2m x 3m rug beneath the bed."
        result = normalize_room_dimensions(
            room_type="bedroom",
            width=20.0,
            depth=20.0,
            prompt=prompt,
        )

        self.assertFalse(prompt_has_explicit_room_dimensions(prompt))
        self.assertTrue(result.changed)

    def test_house_mode_is_unchanged_by_default(self) -> None:
        result = normalize_room_dimensions(
            room_type="living_room",
            width=10.0,
            depth=8.0,
            prompt="A living room in a large house.",
            mode="house",
        )

        self.assertFalse(result.changed)
        self.assertEqual((result.width, result.depth), (10.0, 8.0))

    def test_extreme_aspect_ratio_keeps_a_habitable_minimum_side(self) -> None:
        result = normalize_room_dimensions(
            room_type="bedroom",
            width=20.0,
            depth=1.5,
            prompt="A compact bedroom.",
        )

        self.assertTrue(result.changed)
        self.assertGreaterEqual(min(result.width, result.depth), 2.8)
        self.assertLessEqual(max(result.width, result.depth), 5.5)

    def test_verifier_reads_serialized_length_and_rejects_oversized_room(self) -> None:
        issues = _check_floor_plan_layout(
            {
                "layout_exists": True,
                "room_count": 1,
                "rooms": [
                    {
                        "id": "bedroom",
                        "type": "bedroom",
                        "width": 20.0,
                        "length": 20.0,
                        "prompt": "A simple bedroom.",
                    }
                ],
            }
        )

        self.assertIn("implausible_room_scale", {issue.issue_type for issue in issues})

    def test_verifier_accepts_explicit_large_room(self) -> None:
        issues = _check_floor_plan_layout(
            {
                "layout_exists": True,
                "room_count": 1,
                "rooms": [
                    {
                        "id": "bedroom",
                        "type": "bedroom",
                        "width": 18.0,
                        "length": 19.0,
                        "prompt": "A 19m x 18m bedroom for a hotel suite.",
                    }
                ],
            }
        )

        self.assertNotIn(
            "implausible_room_scale", {issue.issue_type for issue in issues}
        )

    def test_furniture_verifier_rejects_repair_placeholders(self) -> None:
        with TemporaryDirectory() as output_dir:
            report = StageVerifier(pass_threshold=0.4).verify(
                stage="furniture",
                stage_output_dir=output_dir,
                task_spec=SceneTaskSpec(
                    room_type="bedroom",
                    style="standard",
                    required_large_objects=["bed"],
                ),
                scene_state_info={
                    "object_names": ["bed"],
                    "placeholder_names": ["bed"],
                },
            )

        self.assertFalse(report.pass_stage)
        self.assertIn(
            "placeholder_asset", {issue.issue_type for issue in report.issues}
        )


if __name__ == "__main__":
    unittest.main()
