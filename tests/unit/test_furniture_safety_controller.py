import unittest

from dataclasses import dataclass
from types import SimpleNamespace

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
        self.assertTrue(second.should_finish)
        self.assertEqual(controller.best_scene_state, {"state": 1})

    def test_low_functionality_score_is_not_hard_by_default(self) -> None:
        controller = FurnitureSafetyController({"enabled": True})

        evaluation = controller.evaluate_scores(make_scores(functionality=3))

        self.assertTrue(evaluation.hard_valid)

    def test_low_functionality_score_can_be_configured_as_hard(self) -> None:
        controller = FurnitureSafetyController(
            {"enabled": True, "score_thresholds_are_hard": True}
        )

        evaluation = controller.evaluate_scores(make_scores(functionality=3))

        self.assertFalse(evaluation.hard_valid)

    def test_required_counts_parse_two_nightstands(self) -> None:
        controller = FurnitureSafetyController({"enabled": True})
        controller.reset_for_scene("A bedroom with a bed, two nightstands, and a wardrobe.")

        self.assertEqual(controller.required_counts.get("nightstand"), 2)
        self.assertEqual(controller.required_counts.get("bed"), 1)
        self.assertEqual(controller.required_counts.get("wardrobe"), 1)

    def test_add_required_object_is_blocked_after_requested_count(self) -> None:
        controller = FurnitureSafetyController({"enabled": True})
        controller.reset_for_scene("A bedroom with a bed, two nightstands, and a wardrobe.")
        scene = SimpleNamespace(
            objects={
                "nightstand_0": SimpleNamespace(
                    name="nightstand",
                    description="wooden nightstand",
                    immutable=False,
                ),
                "nightstand_1": SimpleNamespace(
                    name="nightstand",
                    description="wooden nightstand",
                    immutable=False,
                ),
            }
        )

        allowed, message = controller.record_add(
            scene=scene,
            asset_text="wooden nightstand with drawer",
        )

        self.assertFalse(allowed)
        self.assertIn("requires 2 nightstand", message)

    def test_extra_required_object_can_be_removed_but_last_required_is_blocked(
        self,
    ) -> None:
        controller = FurnitureSafetyController({"enabled": True})
        controller.reset_for_scene("A bedroom with a bed, two nightstands, and a wardrobe.")
        scene = SimpleNamespace(
            objects={
                "nightstand_0": SimpleNamespace(
                    name="nightstand",
                    description="wooden nightstand",
                    immutable=False,
                ),
                "nightstand_1": SimpleNamespace(
                    name="nightstand",
                    description="wooden nightstand",
                    immutable=False,
                ),
                "nightstand_2": SimpleNamespace(
                    name="nightstand",
                    description="wooden nightstand",
                    immutable=False,
                ),
            }
        )

        allowed_extra, _ = controller.record_remove(
            "nightstand_2",
            "wooden nightstand",
            scene=scene,
        )
        scene.objects.pop("nightstand_2")
        allowed_required, message = controller.record_remove(
            "nightstand_1",
            "wooden nightstand",
            scene=scene,
        )

        self.assertTrue(allowed_extra)
        self.assertFalse(allowed_required)
        self.assertIn("below the requested count", message)

    def test_per_designer_call_move_budget_is_enforced(self) -> None:
        controller = FurnitureSafetyController(
            {
                "enabled": True,
                "max_moves_design_change": 2,
                "max_moves_per_object_per_call": 2,
            }
        )
        controller.begin_designer_call("change")

        self.assertTrue(controller.record_move("bed_0")[0])
        self.assertTrue(controller.record_move("nightstand_0")[0])
        allowed, message = controller.record_move("wardrobe_0")

        self.assertFalse(allowed)
        self.assertIn("already used 2 move", message)

    def test_per_object_move_budget_is_enforced(self) -> None:
        controller = FurnitureSafetyController(
            {
                "enabled": True,
                "max_moves_design_change": 10,
                "max_moves_per_object_per_call": 1,
            }
        )
        controller.begin_designer_call("change")

        self.assertTrue(controller.record_move("bed_0")[0])
        allowed, message = controller.record_move("bed_0")

        self.assertFalse(allowed)
        self.assertIn("already been moved 1", message)

    def test_physics_check_budget_is_enforced(self) -> None:
        controller = FurnitureSafetyController(
            {"enabled": True, "max_physics_checks_per_designer_call": 1}
        )
        controller.begin_designer_call("change")

        self.assertTrue(controller.record_physics_check()[0])
        allowed, message = controller.record_physics_check()

        self.assertFalse(allowed)
        self.assertIn("already used 1", message)


if __name__ == "__main__":
    unittest.main()
