import unittest

from scenesmith.experiments.indoor_scene_generation import (
    _is_repairable_stage_validation,
)
from scenesmith.scene_expert.exceptions import StageValidationError


class StageFailureRecoveryTest(unittest.TestCase):
    def test_collision_and_missing_content_are_repairable(self) -> None:
        error = StageValidationError(
            stage="furniture",
            reasons=[
                "physics hard violation: collisions",
                "missing required wardrobe: expected 1, found 0",
            ],
        )

        self.assertTrue(_is_repairable_stage_validation(error))

    def test_invalid_room_geometry_remains_terminal(self) -> None:
        error = StageValidationError(
            stage="furniture",
            reasons=["invalid room geometry: floor polygon is empty"],
        )

        self.assertFalse(_is_repairable_stage_validation(error))


if __name__ == "__main__":
    unittest.main()
