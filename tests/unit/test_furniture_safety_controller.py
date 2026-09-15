import unittest

from dataclasses import dataclass
from types import SimpleNamespace

from scenesmith.agent_utils.furniture_safety import (
    FurnitureSafetyController,
    HardStateEvaluation,
    furniture_object_category_matches,
)
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


class BoundedFurniture:
    def __init__(
        self,
        *,
        name: str,
        description: str,
        world_min: tuple[float, float, float],
        world_max: tuple[float, float, float],
        rotation_matrix: tuple[tuple[float, float, float], ...] | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.object_type = SimpleNamespace(value="furniture")
        self.immutable = False
        self._world_min = world_min
        self._world_max = world_max
        if rotation_matrix is not None:
            self.transform = SimpleNamespace(
                rotation=lambda: SimpleNamespace(matrix=lambda: rotation_matrix)
            )

    def compute_world_bounds(
        self,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        return self._world_min, self._world_max


class FurnitureSafetyControllerTest(unittest.TestCase):
    def test_asset_generation_recovery_is_allowed_after_partial_failure(self) -> None:
        controller = FurnitureSafetyController({"enabled": True})

        self.assertTrue(controller.record_generate_assets()[0])
        controller.record_asset_generation_result(had_failures=True)
        self.assertTrue(controller.record_generate_assets()[0])
        controller.record_asset_generation_result(had_failures=True)

        allowed, message = controller.record_generate_assets()

        self.assertFalse(allowed)
        self.assertIn("recovery allowance", message)

    def test_asset_generation_success_does_not_open_recovery_retry(self) -> None:
        controller = FurnitureSafetyController({"enabled": True})

        self.assertTrue(controller.record_generate_assets()[0])
        controller.record_asset_generation_result(had_failures=False)

        allowed, message = controller.record_generate_assets()

        self.assertFalse(allowed)
        self.assertIn("already succeeded", message)

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

    def test_collision_free_and_clear_door_language_is_not_hard(self) -> None:
        controller = FurnitureSafetyController({"enabled": True})

        evaluation = controller.evaluate_scores(
            make_scores(
                critique=(
                    "The bedside group is collision-free with 0 collisions. No "
                    "blocked doorways or windows remain, and door clearance is "
                    "sufficient."
                )
            )
        )

        self.assertTrue(evaluation.hard_valid)
        self.assertEqual(evaluation.hard_reasons, [])

    def test_room_bounds_guard_emits_typed_containment_failure(self) -> None:
        controller = FurnitureSafetyController({"enabled": True})
        scene = SimpleNamespace(
            room_type="living_room",
            room_geometry=SimpleNamespace(length=4.0, width=4.0),
            objects={
                "cabinet_0": BoundedFurniture(
                    name="cabinet_0",
                    description="media cabinet",
                    world_min=(1.8, -0.4, 0.0),
                    world_max=(2.4, 0.4, 0.9),
                )
            },
        )

        evaluation = controller.evaluate_scene_state(scene)

        self.assertFalse(evaluation.hard_valid)
        self.assertEqual(evaluation.hard_reasons, [])
        self.assertEqual(
            evaluation.typed_failures[0]["relation_type"], "room_containment"
        )
        self.assertEqual(evaluation.typed_failures[0]["primary_object"], "cabinet_0")

    def test_asserted_door_blockage_is_hard_without_deterministic_state(self) -> None:
        controller = FurnitureSafetyController({"enabled": True})

        evaluation = controller.evaluate_scores(
            make_scores(critique="The wardrobe blocks the west door.")
        )

        self.assertFalse(evaluation.hard_valid)
        self.assertIn(
            "door or open-connection blockage indicated by critique",
            evaluation.hard_reasons,
        )

        clearance_evaluation = controller.evaluate_scores(
            make_scores(critique="A door clearance violation remains.")
        )
        self.assertFalse(clearance_evaluation.hard_valid)

    def test_deterministic_hard_state_overrides_free_form_issue_language(self) -> None:
        controller = FurnitureSafetyController({"enabled": True})

        decision = controller.consider_candidate(
            make_scores(
                critique="A collision remains and the wardrobe blocks the west door."
            ),
            {"state": "complete-bedroom"},
            None,
            hard_state_evaluation=HardStateEvaluation(hard_valid=True),
        )

        self.assertTrue(decision.accepted)
        self.assertTrue(decision.evaluation.hard_valid)
        self.assertEqual(controller.best_scene_state, {"state": "complete-bedroom"})

    def test_deterministic_hard_failure_overrides_clean_free_form_language(
        self,
    ) -> None:
        controller = FurnitureSafetyController({"enabled": True})

        decision = controller.consider_candidate(
            make_scores(
                critique="No collisions remain and all doorways are unobstructed."
            ),
            {"state": "blocked-bedroom"},
            None,
            hard_state_evaluation=HardStateEvaluation(
                hard_valid=False,
                hard_reasons=["door swing clearance violation"],
            ),
        )

        self.assertFalse(decision.accepted)
        self.assertFalse(decision.evaluation.hard_valid)
        self.assertIn(
            "door swing clearance violation",
            decision.evaluation.hard_reasons,
        )

    def test_required_object_removal_is_blocked(self) -> None:
        controller = FurnitureSafetyController({"enabled": True})
        controller.reset_for_scene(
            "A bedroom with a bed, two nightstands, and a wardrobe."
        )

        allowed, message = controller.record_remove(
            object_id="nightstand_0",
            object_text="wooden nightstand",
        )

        self.assertFalse(allowed)
        self.assertIn("required", message)

    def test_media_room_infers_all_required_major_furniture(self) -> None:
        controller = FurnitureSafetyController({"enabled": True})
        controller.reset_for_scene(
            "A living room with a sofa, a TV stand, two armchairs, and a floor lamp."
        )

        self.assertEqual(controller.required_counts["sofa"], 1)
        self.assertEqual(controller.required_counts["tv_stand"], 1)
        self.assertEqual(controller.required_counts["armchair"], 2)
        self.assertEqual(controller.required_counts["floor_lamp"], 1)

    def test_structured_identity_satisfies_specific_contract_categories(self) -> None:
        self.assertTrue(
            furniture_object_category_matches(
                "fridge_0",
                "fridge",
                "Stainless steel French-door refrigerator, tall",
                "fridge",
            )
        )
        self.assertTrue(
            furniture_object_category_matches(
                "pantry_shelf_0",
                "pantry_shelf",
                "Sleek slim pantry shelving unit",
                "pantry_shelf",
            )
        )
        self.assertTrue(
            furniture_object_category_matches(
                "bar_table_0",
                "bar_table",
                "Large modern bar table",
                "bar_table",
            )
        )

    def test_display_wording_and_negated_tv_do_not_require_television(self) -> None:
        for prompt in (
            "A living room with a display shelf and a display cabinet.",
            "A living room with a display zone.",
            "A living room with no TV and a sofa.",
            "A living room without television and a sofa.",
            "A living room with no TV or television.",
            "A living room without a TV or television.",
            "A living room with no wall-mounted TV and a sofa.",
            "A living room with a sofa; do not include a television.",
            "A living room with a TV stand and a sofa.",
        ):
            controller = FurnitureSafetyController({"enabled": True})
            controller.reset_for_scene(prompt)

            self.assertNotIn("television", controller.required_terms, prompt)
            self.assertNotIn("television", controller.required_counts, prompt)

    def test_explicit_tv_or_television_still_requires_display_inventory(self) -> None:
        for prompt in (
            "A living room with a TV and a sofa.",
            "A living room with one television and a sofa.",
            "A media room with not only a TV but also a sofa.",
            "A living room with no sofa and a TV.",
            "A living room with a TV stand and a separate television.",
        ):
            controller = FurnitureSafetyController({"enabled": True})
            controller.reset_for_scene(prompt)

            self.assertIn("television", controller.required_terms, prompt)
            self.assertEqual(controller.required_counts["television"], 1)

    def test_sofa_chair_is_one_atomic_inventory_category(self) -> None:
        controller = FurnitureSafetyController({"enabled": True})
        controller.reset_for_scene("A lounge with four sofa chairs around a rug.")

        self.assertEqual(controller.required_counts["sofa_chair"], 4)
        self.assertNotIn("sofa", controller.required_counts)
        self.assertNotIn("chair", controller.required_counts)

    def test_sofa_chair_does_not_hide_independent_sofa(self) -> None:
        controller = FurnitureSafetyController({"enabled": True})
        controller.reset_for_scene(
            "A lounge with four sofa chairs around a rug and one sofa."
        )

        self.assertEqual(controller.required_counts["sofa_chair"], 4)
        self.assertEqual(controller.required_counts["sofa"], 1)
        self.assertNotIn("chair", controller.required_counts)

    def test_dressing_table_is_not_a_second_generic_table_requirement(self) -> None:
        controller = FurnitureSafetyController({"enabled": True})
        controller.reset_for_scene(
            "A bedroom with one low dressing table and one stool."
        )

        self.assertEqual(controller.required_counts["dressing_table"], 1)
        self.assertNotIn("table", controller.required_counts)

    def test_dressing_table_does_not_hide_an_explicit_generic_table(self) -> None:
        controller = FurnitureSafetyController({"enabled": True})
        controller.reset_for_scene(
            "A bedroom with one dressing table and one utility table."
        )

        self.assertEqual(controller.required_counts["dressing_table"], 1)
        self.assertEqual(controller.required_counts["table"], 1)

    def test_tv_console_satisfies_tv_stand_inventory_role(self) -> None:
        self.assertTrue(
            furniture_object_category_matches(
                "tv_console_0",
                "tv_console",
                "Modern TV console stand with an integrated television",
                "tv_stand",
            )
        )

    def test_generic_cabinet_satisfies_storage_cabinet_inventory_role(self) -> None:
        self.assertTrue(
            furniture_object_category_matches(
                "cabinet_0",
                "cabinet",
                "Compact freestanding storage cabinet",
                "storage_cabinet",
            )
        )

    def test_named_filing_cabinet_does_not_satisfy_storage_cabinet_role(self) -> None:
        self.assertFalse(
            furniture_object_category_matches(
                "filing_cabinet_0",
                "filing_cabinet",
                "Steel filing cabinet with drawers",
                "storage_cabinet",
            )
        )

    def test_role_specific_desks_satisfy_generic_desk_inventory(self) -> None:
        for object_id, name in (
            ("student_desk_0", "student_desk"),
            ("teacher_desk_0", "teacher_desk"),
        ):
            self.assertTrue(
                furniture_object_category_matches(
                    object_id,
                    name,
                    "classroom desk",
                    "desk",
                )
            )

    def test_plural_teacher_asset_name_satisfies_teacher_desk_inventory(self) -> None:
        controller = FurnitureSafetyController({"enabled": True})
        controller.reset_for_scene("A classroom with a teacher's desk.")
        scene = SimpleNamespace(
            room_type="classroom",
            text_description=controller.scene_description,
            room_geometry=None,
            objects={
                "teachers_desk_0": SimpleNamespace(
                    name="teachers_desk",
                    description="wooden instructor desk with drawer storage",
                    immutable=False,
                )
            },
        )

        evaluation = controller.evaluate_scene_state(scene)

        self.assertTrue(evaluation.hard_valid)
        self.assertNotIn(
            "missing required teacher_desk: expected 1, found 0",
            evaluation.hard_reasons,
        )

    def test_role_specific_chair_satisfies_generic_chair_inventory(self) -> None:
        self.assertTrue(
            furniture_object_category_matches(
                "student_chair_0",
                "student_chair",
                "classroom chair",
                "chair",
            )
        )

    def test_floor_speaker_satisfies_generic_speaker_inventory(self) -> None:
        self.assertTrue(
            furniture_object_category_matches(
                "floor_speaker_0",
                "floor_speaker",
                "Tall floor-standing speaker tower",
                "speaker",
            )
        )

    def test_dining_room_infers_sideboard(self) -> None:
        controller = FurnitureSafetyController({"enabled": True})
        controller.reset_for_scene(
            "A dining room with a sideboard and four dining chairs."
        )

        self.assertEqual(controller.required_counts["sideboard"], 1)
        self.assertEqual(controller.required_counts["dining_chair"], 4)
        self.assertEqual(controller.required_counts["chair"], 4)

    def test_sideboard_identity_wins_over_cabinet_door_description(self) -> None:
        controller = FurnitureSafetyController({"enabled": True})
        controller.reset_for_scene("A dining room with a sideboard.")
        scene = SimpleNamespace(
            room_type="dining_room",
            text_description=controller.scene_description,
            room_geometry=None,
            objects={
                "sideboard_0": SimpleNamespace(
                    name="sideboard",
                    description=(
                        "Traditional wooden sideboard with drawers and cabinet doors "
                        "for dining room storage"
                    ),
                    immutable=False,
                )
            },
        )

        evaluation = controller.evaluate_scene_state(scene)

        self.assertTrue(evaluation.hard_valid)
        self.assertNotIn(
            "missing required sideboard: expected 1, found 0",
            evaluation.hard_reasons,
        )

    def test_water_dispenser_identity_satisfies_contract_inventory(self) -> None:
        controller = FurnitureSafetyController({"enabled": True})
        controller.required_counts = {"water_dispenser": 1}
        controller.required_terms = {"water_dispenser"}
        scene = SimpleNamespace(
            room_type="office",
            text_description="An office with a water dispenser.",
            room_geometry=None,
            objects={
                "water_dispenser_0": SimpleNamespace(
                    name="water_dispenser",
                    description=(
                        "Freestanding water cooler with bottle on top and "
                        "dispensing spigots"
                    ),
                    immutable=False,
                )
            },
        )

        evaluation = controller.evaluate_scene_state(scene)

        self.assertTrue(evaluation.hard_valid)
        self.assertNotIn(
            "missing required water_dispenser: expected 1, found 0",
            evaluation.hard_reasons,
        )

    def test_chair_requirements_preserve_explicit_subtype_counts(self) -> None:
        controller = FurnitureSafetyController({"enabled": True})
        controller.reset_for_scene(
            "A study with one office chair and two guest chairs around a desk."
        )
        scene = SimpleNamespace(
            room_type="study",
            text_description=controller.scene_description,
            room_geometry=None,
            objects={
                "desk_0": SimpleNamespace(
                    name="desk",
                    description="work desk",
                    immutable=False,
                ),
                "office_chair_0": SimpleNamespace(
                    name="office_chair",
                    description="ergonomic task chair",
                    immutable=False,
                ),
                "guest_armchair_0": SimpleNamespace(
                    name="guest_armchair",
                    description="upholstered armchair",
                    immutable=False,
                ),
                "guest_armchair_1": SimpleNamespace(
                    name="guest_armchair",
                    description="upholstered armchair",
                    immutable=False,
                ),
            },
        )

        self.assertEqual(
            controller.required_counts,
            {"office_chair": 1, "guest_chair": 2, "desk": 1, "chair": 3},
        )
        self.assertTrue(controller.evaluate_scene_state(scene).hard_valid)

        allowed, message = controller.record_add(
            scene=scene,
            asset_text="another upholstered armchair",
        )
        self.assertFalse(allowed)
        self.assertIn("requires 3 chair", message)

    def test_extra_office_chair_does_not_satisfy_guest_chair_count(self) -> None:
        controller = FurnitureSafetyController({"enabled": True})
        controller.reset_for_scene(
            "A study with one office chair and two guest chairs around a desk."
        )
        scene = SimpleNamespace(
            room_type="study",
            text_description=controller.scene_description,
            room_geometry=None,
            objects={
                "desk_0": SimpleNamespace(
                    name="desk",
                    description="work desk",
                    immutable=False,
                ),
                "office_chair_0": SimpleNamespace(
                    name="office_chair",
                    description="ergonomic task chair",
                    immutable=False,
                ),
                "office_chair_1": SimpleNamespace(
                    name="office_chair",
                    description="ergonomic task chair",
                    immutable=False,
                ),
                "guest_chair_0": SimpleNamespace(
                    name="guest_chair",
                    description="upholstered visitor chair",
                    immutable=False,
                ),
            },
        )

        evaluation = controller.evaluate_scene_state(scene)

        self.assertFalse(evaluation.hard_valid)
        self.assertIn(
            "missing required guest_chair: expected 2, found 1",
            evaluation.hard_reasons,
        )

    def test_table_lamp_is_not_required_furniture_table(self) -> None:
        controller = FurnitureSafetyController({"enabled": True})
        controller.reset_for_scene(
            "A bedroom with a bed and a table lamp on each nightstand."
        )

        self.assertNotIn("table", controller.required_counts)
        self.assertIsNone(controller._infer_category("table_lamp_0 table lamp"))

    def test_lamp_on_each_bed_side_requires_two_nightstands(self) -> None:
        controller = FurnitureSafetyController({"enabled": True})
        controller.reset_for_scene(
            "A bedroom with a bed centered on the main wall, a nightstand with "
            "a table lamp on each side of the bed."
        )

        self.assertEqual(controller.required_counts["nightstand"], 2)

    def test_candidate_must_clearly_improve_best(self) -> None:
        controller = FurnitureSafetyController(
            {
                "enabled": True,
                "min_accept_delta": 0.05,
                "accept_score_threshold": 1.0,
            }
        )

        first = controller.consider_candidate(make_scores(layout=7), {"state": 1}, None)
        second = controller.consider_candidate(
            make_scores(layout=7), {"state": 2}, None
        )

        self.assertTrue(first.accepted)
        self.assertFalse(second.accepted)
        self.assertTrue(second.rollback_to_best)
        self.assertTrue(second.should_finish)
        self.assertEqual(controller.best_scene_state, {"state": 1})

    def test_unscored_baseline_allows_first_critic_guided_repair(self) -> None:
        controller = FurnitureSafetyController(
            {
                "enabled": True,
                "score_thresholds_are_hard": True,
                "max_critique_design_cycles": 1,
            }
        )
        controller.remember_hard_valid_scene_state(
            {"state": "deterministic-baseline"},
            source="pre-initial",
        )

        decision = controller.consider_candidate(
            make_scores(prompt_following=5),
            {"state": "deterministic-baseline"},
            None,
        )

        self.assertFalse(decision.accepted)
        self.assertFalse(decision.rollback_to_best)
        self.assertFalse(decision.should_finish)
        self.assertEqual(
            controller.best_scene_state, {"state": "deterministic-baseline"}
        )
        self.assertTrue(controller.record_design_change(has_prior_critique=True)[0])

    def test_low_functionality_score_is_not_hard_by_default(self) -> None:
        controller = FurnitureSafetyController({"enabled": True})

        evaluation = controller.evaluate_scores(make_scores(functionality=3))

        self.assertTrue(evaluation.hard_valid)

    def test_low_prompt_following_score_is_not_hard_by_default(self) -> None:
        controller = FurnitureSafetyController({"enabled": True})

        evaluation = controller.evaluate_scores(make_scores(prompt_following=5))

        self.assertTrue(evaluation.hard_valid)

    def test_low_prompt_following_score_can_be_configured_as_hard(self) -> None:
        controller = FurnitureSafetyController(
            {"enabled": True, "score_thresholds_are_hard": True}
        )

        evaluation = controller.evaluate_scores(make_scores(prompt_following=5))

        self.assertFalse(evaluation.hard_valid)
        self.assertIn("prompt_following=5 below 8", evaluation.hard_reasons)

    def test_low_functionality_score_can_be_configured_as_hard(self) -> None:
        controller = FurnitureSafetyController(
            {"enabled": True, "score_thresholds_are_hard": True}
        )

        evaluation = controller.evaluate_scores(make_scores(functionality=3))

        self.assertFalse(evaluation.hard_valid)

    def test_window_access_warning_can_be_configured_as_hard(self) -> None:
        controller = FurnitureSafetyController(
            {"enabled": True, "bedroom_layout": {"window_blocking_is_hard": True}}
        )

        evaluation = controller.evaluate_scene_state(
            SimpleNamespace(
                room_type="bedroom",
                text_description="A bedroom.",
                objects={},
                room_geometry=None,
            ),
            physics_context=(
                "Physics violations detected:\n"
                "Window access warnings (1): wardrobe_0 blocks window_2"
            ),
        )

        self.assertFalse(evaluation.hard_valid)
        self.assertIn("window access warning", evaluation.hard_reasons)

    def test_window_access_warning_is_soft_outside_bedroom(self) -> None:
        controller = FurnitureSafetyController(
            {"enabled": True, "bedroom_layout": {"window_blocking_is_hard": True}}
        )

        evaluation = controller.evaluate_scene_state(
            SimpleNamespace(
                room_type="living_room",
                text_description="A living room.",
                objects={},
                room_geometry=None,
            ),
            physics_context=(
                "Physics violations detected:\n"
                "Window access warnings (1): sofa_0 blocks window_2"
            ),
        )

        self.assertTrue(evaluation.hard_valid)
        self.assertIn("window access warning", evaluation.soft_reasons)

    def test_nightstand_overlapping_bed_is_hard(self) -> None:
        controller = FurnitureSafetyController({"enabled": True})
        scene = SimpleNamespace(
            room_type="bedroom",
            text_description="A bedroom with a bed and nightstand.",
            objects={
                "bed_0": BoundedFurniture(
                    name="bed",
                    description="compact bed",
                    world_min=(-1.0, -1.0, 0.0),
                    world_max=(1.0, 1.0, 0.8),
                ),
                "nightstand_0": BoundedFurniture(
                    name="nightstand",
                    description="bedside table",
                    world_min=(0.80, -0.20, 0.0),
                    world_max=(1.20, 0.40, 0.6),
                ),
            },
        )

        evaluation = controller.evaluate_scene_state(scene)

        self.assertFalse(evaluation.hard_valid)
        self.assertTrue(
            any(
                "overlaps bed_0 footprint" in reason
                for reason in evaluation.hard_reasons
            )
        )

    def test_unrequested_nightstands_are_advisory(self) -> None:
        controller = FurnitureSafetyController({"enabled": True})
        yaw_0 = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        scene = SimpleNamespace(
            room_type="bedroom",
            text_description="A bedroom featuring rustic farmhouse decor.",
            objects={
                "bed_0": BoundedFurniture(
                    name="bed",
                    description="farmhouse bed",
                    world_min=(-1.0, -1.0, 0.0),
                    world_max=(1.0, 1.0, 0.8),
                    rotation_matrix=yaw_0,
                ),
                "nightstand_0": BoundedFurniture(
                    name="nightstand",
                    description="farmhouse nightstand",
                    world_min=(1.2, -0.2, 0.0),
                    world_max=(1.6, 0.2, 0.6),
                ),
                "nightstand_1": BoundedFurniture(
                    name="nightstand",
                    description="farmhouse nightstand",
                    world_min=(1.7, -0.2, 0.0),
                    world_max=(2.1, 0.2, 0.6),
                ),
            },
        )

        evaluation = controller.evaluate_scene_state(scene)
        hard_issues, soft_issues = controller._classify_bedroom_plausibility_issues(
            ["bedroom plausibility: nightstands are not on opposite bed sides"],
            scene=scene,
        )

        self.assertTrue(evaluation.hard_valid)
        self.assertEqual([], hard_issues)
        self.assertEqual(
            ["bedroom plausibility: nightstands are not on opposite bed sides"],
            soft_issues,
        )

    def test_required_counts_parse_two_nightstands(self) -> None:
        controller = FurnitureSafetyController({"enabled": True})
        controller.reset_for_scene(
            "A bedroom with a bed, two nightstands, and a wardrobe."
        )

        self.assertEqual(controller.required_counts.get("nightstand"), 2)
        self.assertEqual(controller.required_counts.get("bed"), 1)
        self.assertEqual(controller.required_counts.get("wardrobe"), 1)

    def test_dresser_uses_opposite_wall_and_nightstand_matches_bed_yaw(self) -> None:
        controller = FurnitureSafetyController({"enabled": True})
        controller.reset_for_scene(
            "A bedroom with a bed, two nightstands, and a dresser against the "
            "opposite wall directly facing the bed, with a wardrobe next to the dresser."
        )
        yaw_180 = ((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0))
        yaw_0 = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        scene = SimpleNamespace(
            room_type="bedroom",
            text_description=controller.scene_description,
            scene_expert_original_description=controller.scene_description,
            room_geometry=SimpleNamespace(length=5.0, width=4.5),
            objects={
                "bed_0": BoundedFurniture(
                    name="bed",
                    description="bed",
                    world_min=(-0.8, 0.1, 0.0),
                    world_max=(0.8, 2.1, 0.8),
                    rotation_matrix=yaw_180,
                ),
                "nightstand_0": BoundedFurniture(
                    name="nightstand",
                    description="nightstand",
                    world_min=(-1.3, 1.3, 0.0),
                    world_max=(-0.9, 1.7, 0.6),
                    rotation_matrix=yaw_0,
                ),
                "nightstand_1": BoundedFurniture(
                    name="nightstand",
                    description="nightstand",
                    world_min=(0.9, 1.3, 0.0),
                    world_max=(1.3, 1.7, 0.6),
                    rotation_matrix=yaw_180,
                ),
                "dresser_0": BoundedFurniture(
                    name="dresser",
                    description="dresser",
                    world_min=(-2.45, -0.3, 0.0),
                    world_max=(-1.45, 0.3, 0.9),
                    rotation_matrix=yaw_0,
                ),
                "wardrobe_0": BoundedFurniture(
                    name="wardrobe",
                    description="wardrobe",
                    world_min=(1.4, -0.3, 0.0),
                    world_max=(2.3, 0.3, 2.0),
                    rotation_matrix=yaw_0,
                ),
            },
        )

        evaluation = controller.evaluate_scene_state(scene)

        self.assertFalse(evaluation.hard_valid)
        self.assertTrue(
            any(
                "dresser_0 is not backed against the south wall" in reason
                for reason in evaluation.hard_reasons
            )
        )
        self.assertTrue(
            any(
                "nightstand_0 use direction is not aligned" in reason
                for reason in evaluation.hard_reasons
            )
        )
        self.assertTrue(
            any(
                "wardrobe_0 is 2.85m from dresser_0" in reason
                for reason in evaluation.hard_reasons
            )
        )

    def test_style_only_bedroom_still_requires_a_bed(self) -> None:
        controller = FurnitureSafetyController({"enabled": True})

        controller.reset_for_scene(
            "A bedroom featuring rustic farmhouse decor with exposed beams."
        )

        self.assertEqual(controller.required_counts, {"bed": 1})

    def test_each_relation_propagates_required_count(self) -> None:
        controller = FurnitureSafetyController({"enabled": True})

        controller.reset_for_scene(
            "A classroom with six student desks, each with a chair. "
            "A teacher's desk sits at the front near the chalkboard."
        )

        self.assertEqual(controller.required_counts.get("desk"), 7)
        self.assertEqual(controller.required_counts.get("chair"), 6)
        self.assertEqual(controller.required_counts.get("student_chair"), 6)
        self.assertNotIn("table", controller.required_counts)

    def test_teacher_chair_cannot_satisfy_role_specific_student_inventory(self) -> None:
        controller = FurnitureSafetyController({"enabled": True})
        controller.reset_for_scene(
            "A classroom with six student desks, each with a chair. A teacher's "
            "desk sits at the front."
        )
        objects = {
            "teacher_chair_0": SimpleNamespace(
                name="teacher_chair", description="teacher armchair", immutable=False
            ),
            **{
                f"student_chair_{index}": SimpleNamespace(
                    name="student_chair",
                    description="student classroom chair",
                    immutable=False,
                )
                for index in range(5)
            },
        }

        evaluation = controller.evaluate_scene_state(
            SimpleNamespace(objects=objects, room_geometry=None)
        )

        self.assertIn(
            "missing required student_chair: expected 6, found 5",
            evaluation.hard_reasons,
        )

    def test_living_room_prompt_tracks_rug_and_plant_counts(self) -> None:
        controller = FurnitureSafetyController({"enabled": True})

        controller.reset_for_scene(
            "A living room with a two-seater sofa, a square rug, and two "
            "large plants near the sofa."
        )

        self.assertEqual(controller.required_counts.get("sofa"), 1)
        self.assertEqual(controller.required_counts.get("rug"), 1)
        self.assertEqual(controller.required_counts.get("plant"), 2)

    def test_scene_expert_injection_does_not_activate_bedroom_checks(self) -> None:
        controller = FurnitureSafetyController({"enabled": True})
        controller.reset_for_scene("A living room with a two-seater sofa.")
        scene = SimpleNamespace(
            room_type="living_room",
            scene_expert_original_description=("A living room with a two-seater sofa."),
            text_description=(
                "A living room with a two-seater sofa.\n\n"
                "Retrieved failure memory: a bed and wardrobe blocked a window."
            ),
            objects={},
            room_geometry=None,
        )

        evaluation = controller.evaluate_scene_state(scene)

        self.assertIn(
            "missing required sofa: expected 1, found 0",
            evaluation.hard_reasons,
        )
        self.assertFalse(
            any("bedroom plausibility" in reason for reason in evaluation.hard_reasons)
        )

    def test_add_required_object_is_blocked_after_requested_count(self) -> None:
        controller = FurnitureSafetyController({"enabled": True})
        controller.reset_for_scene(
            "A bedroom with a bed, two nightstands, and a wardrobe."
        )
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

    def test_coffee_and_dining_tables_have_distinct_inventory_limits(self) -> None:
        controller = FurnitureSafetyController({"enabled": True})
        controller.reset_for_scene(
            "A living room with one coffee table and one rectangular dining table."
        )
        scene = SimpleNamespace(
            objects={
                "coffee_table_0": SimpleNamespace(
                    name="coffee_table",
                    description="low coffee table",
                    immutable=False,
                )
            }
        )

        self.assertEqual(controller.required_counts["coffee_table"], 1)
        self.assertEqual(controller.required_counts["dining_table"], 1)
        allowed, message = controller.record_add(
            scene=scene,
            asset_text="rectangular dining table",
        )
        self.assertTrue(allowed, message)

        scene.objects["dining_table_0"] = SimpleNamespace(
            name="dining_table",
            description="rectangular dining table",
            immutable=False,
        )
        allowed, message = controller.record_add(
            scene=scene,
            asset_text="another dining table",
        )
        self.assertFalse(allowed)
        self.assertIn("requires 1 dining_table", message)

    def test_extra_required_object_can_be_removed_but_last_required_is_blocked(
        self,
    ) -> None:
        controller = FurnitureSafetyController({"enabled": True})
        controller.reset_for_scene(
            "A bedroom with a bed, two nightstands, and a wardrobe."
        )
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

    def test_design_change_allows_a_bounded_coordinated_group(self) -> None:
        controller = FurnitureSafetyController(
            {
                "enabled": True,
                "max_moves_design_change": 16,
            }
        )
        controller.begin_designer_call("change")

        for index in range(13):
            self.assertTrue(controller.record_move(f"furniture_{index}")[0])

        for index in range(3):
            self.assertTrue(controller.record_move(f"reserve_{index}")[0])

        allowed, message = controller.record_move("over_budget")

        self.assertFalse(allowed)
        self.assertIn("already used 16 move", message)

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

    def test_repeated_blocked_tools_leave_incomplete_scene_recoverable(self) -> None:
        controller = FurnitureSafetyController(
            {
                "enabled": True,
                "max_moves_per_object_per_call": 1,
                "max_blocked_tool_calls_per_designer_call": 2,
            }
        )
        controller.begin_designer_call("change")

        self.assertTrue(controller.record_move("bed_0")[0])
        self.assertFalse(controller.record_move("bed_0")[0])
        allowed, message = controller.record_move("bed_0")

        self.assertFalse(allowed)
        self.assertFalse(controller.should_finish)
        self.assertIn("STOP:", message)
        self.assertIn("next critique-guided repair", message)
        self.assertTrue(controller.record_move("nightstand_0")[0])


if __name__ == "__main__":
    unittest.main()
