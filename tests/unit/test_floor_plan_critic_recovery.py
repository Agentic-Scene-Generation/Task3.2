import unittest

from scenesmith.agent_utils.scoring import parse_floor_plan_critique_text


class FloorPlanCriticRecoveryTest(unittest.TestCase):
    def test_recovers_legacy_markdown_from_model_behavior_error(self) -> None:
        error_text = """Invalid JSON when parsing

## 1. Summary
The layout is usable and follows the requested single-room brief.

## 2. Category Scores
- **Room Proportions:** 8/10 - Dimensions are appropriate.
- **Spatial Flow:** 9/10 - The entry path is clear.
- **Natural Lighting:** 7/10 - One exterior window is present.
- **Material Consistency:** 8/10 - Finishes suit the room.
- **Prompt Following:** 10/10 - All architectural requirements are met.

## 3. Detailed Critique
Proceed to furniture placement.
 for TypeAdapter(OutputType); 1 validation error for OutputType"""

        recovered = parse_floor_plan_critique_text(error_text)

        self.assertIsNotNone(recovered)
        assert recovered is not None
        self.assertEqual(recovered.room_proportions.grade, 8)
        self.assertEqual(recovered.spatial_flow.grade, 9)
        self.assertEqual(recovered.natural_lighting.grade, 7)
        self.assertEqual(recovered.material_consistency.grade, 8)
        self.assertEqual(recovered.prompt_following.grade, 10)
        self.assertNotIn("Invalid JSON when parsing", recovered.critique)
        self.assertNotIn("TypeAdapter", recovered.critique)

    def test_requires_all_five_categories(self) -> None:
        partial = """
Room Proportions: 8/10 - Good.
Spatial Flow: 8/10 - Good.
Natural Lighting: 8/10 - Good.
Material Consistency: 8/10 - Good.
"""

        self.assertIsNone(parse_floor_plan_critique_text(partial))

    def test_recovers_unbulleted_score_block(self) -> None:
        score_block = """
Room Proportions: 8/10 - Dimensions are appropriate.
Spatial Flow: 9/10 - Circulation is clear.
Natural Lighting: 7/10 - Daylight is adequate.
Material Consistency: 8/10 - Materials are coherent.
Prompt Following: 10/10 - Requirements are satisfied.
"""

        recovered = parse_floor_plan_critique_text(score_block)

        self.assertIsNotNone(recovered)
        assert recovered is not None
        self.assertEqual([8, 9, 7, 8, 10], [s.grade for s in recovered.get_scores()])


if __name__ == "__main__":
    unittest.main()
