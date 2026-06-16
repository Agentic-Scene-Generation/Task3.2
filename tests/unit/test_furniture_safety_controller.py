import unittest

from dataclasses import dataclass

from scenesmith.agent_utils.furniture_safety import FurnitureSafetyController
from scenesmith.agent_utils.scoring import CategoryScore, CritiqueWithScores


@dataclass
class DummyFurnitureScores(CritiqueWithScores):
    realism: CategoryScore
    functionality: CategoryScore
    layout: CategoryScore
    layout_plausibility: CategoryScore
    holistic_completeness: CategoryScore
    prompt_following: CategoryScore
    reachability: CategoryScore

    def get_scores(self) -> list[CategoryScore]:
        return [
            self.realism,
            self.functionality,
            self.layout,
            self.layout_plausibility,
            self.holistic_completeness,
            self.prompt_following,
            self.reachability,
        ]


def make_scores(
    *,
    critique: str = "Window is partly blocked but the room is usable.",
    realism: int = 8,
    functionality: int = 8,
    layout: int = 8,
    layout_plausibility: int = 8,
    holistic_completeness: int = 8,
    prompt_following: int = 10,
    reachability: int = 8,
) -> DummyFurnitureScores:
    return DummyFurnitureScores(
        critique=critique,
        realism=CategoryScore("realism", realism, "Looks plausible."),
        functionality=CategoryScore("functionality", functionality, "Usable."),
        layout=CategoryScore("layout", layout, "Logical."),
        layout_plausibility=CategoryScore(
            "layout_plausibility",
            layout_plausibility,
            "Human-like enough.",
        ),
        holistic_completeness=CategoryScore(
            "holistic_completeness",
            holistic_completeness,
            "Complete.",
        ),
        prompt_following=CategoryScore(
            "prompt_following",
            prompt_following,
            "All required objects are present.",
        ),
        reachability=CategoryScore("reachability", reachability, "Reachable."),
    )


class FurnitureSafetyControllerTest(unittest.TestCase):
    def test_window_only_issue_is_soft(self) -> None:
        controller = FurnitureSafetyController({"enabled": True})
        evaluation = controller.evaluate_scores(make_scores())

        self.assertTrue(evaluation.hard_valid)
        self.assertTrue(evaluation.soft_reasons)

    def test_collision_issue_is_hard_but_negated_collision_is_not(self) -> None:
        controller = FurnitureSafetyController({"enabled": True})

        collision_eval = controller.evaluate_scores(
            make_scores(critique="A physics-validated collision remains.")
        )
        clean_eval = controller.evaluate_scores(
            make_scores(critique="No collisions remain after the fix.")
        )

        self.assertFalse(collision_eval.hard_valid)
        self.assertTrue(clean_eval.hard_valid)

    def test_required_object_removal_is_blocked(self) -> None:
        controller = FurnitureSafetyController({"enabled": True})
        controller.reset_for_scene("A bedroom with a bed, two nightstands, and a wardrobe.")

        allowed, message = controller.record_remove(
            object_id="nightstand_0",
            object_text="wooden nightstand",
        )

        self.assertFalse(allowed)
        self.assertIn("required", message)

    def test_candidate_must_clearly_improve_best(self) -> None:
        controller = FurnitureSafetyController(
            {
                "enabled": True,
                "min_accept_delta": 0.05,
                "accept_score_threshold": 1.0,
            }
        )

        first = controller.consider_candidate(make_scores(layout=7), {"state": 1}, None)
        second = controller.consider_candidate(make_scores(layout=7), {"state": 2}, None)

        self.assertTrue(first.accepted)
        self.assertFalse(second.accepted)
        self.assertTrue(second.rollback_to_best)
        self.assertEqual(controller.best_scene_state, {"state": 1})


if __name__ == "__main__":
    unittest.main()
