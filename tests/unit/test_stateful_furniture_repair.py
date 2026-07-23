import unittest

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np

try:
    from pydrake.all import RigidTransform
    from scenesmith.agent_utils.scoring import (
        CategoryScore,
        FurnitureCritiqueWithScores,
    )
    from scenesmith.furniture_agents.stateful_furniture_agent import (
        StatefulFurnitureAgent,
    )
except ModuleNotFoundError as exc:
    RigidTransform = None
    StatefulFurnitureAgent = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class ReadOnlyTranslationTransform:
    def __init__(self, translation: tuple[float, float, float]) -> None:
        self._translation = np.asarray(translation, dtype=float)
        self._translation.setflags(write=False)
        self._rotation = RigidTransform().rotation()

    def translation(self) -> np.ndarray:
        return self._translation

    def rotation(self):
        return self._rotation


class StatefulFurnitureRepairTest(unittest.TestCase):
    @unittest.skipIf(
        StatefulFurnitureAgent is None,
        f"requires pydrake/stateful furniture imports: {_IMPORT_ERROR}",
    )
    def test_deterministic_candidate_requires_trusted_score_delta(self) -> None:
        agent_candidate = {
            "score_source": "vlm_critic",
            "weighted_score": 0.72,
        }
        deterministic_candidate = {
            "score_source": "vlm_critic",
            "weighted_score": 0.76,
        }

        self.assertFalse(
            StatefulFurnitureAgent.deterministic_candidate_improves(
                agent_candidate,
                deterministic_candidate,
                minimum_delta=0.05,
            )
        )
        deterministic_candidate["weighted_score"] = 0.78
        self.assertTrue(
            StatefulFurnitureAgent.deterministic_candidate_improves(
                agent_candidate,
                deterministic_candidate,
                minimum_delta=0.05,
            )
        )
        agent_candidate["score_source"] = "deterministic_hard_check"
        self.assertFalse(
            StatefulFurnitureAgent.deterministic_candidate_improves(
                agent_candidate,
                deterministic_candidate,
                minimum_delta=0.05,
            )
        )

    @unittest.skipIf(
        StatefulFurnitureAgent is None,
        f"requires pydrake/stateful furniture imports: {_IMPORT_ERROR}",
    )
    def test_quality_regeneration_requires_scene_expert_and_trusted_score(self) -> None:
        agent = object.__new__(StatefulFurnitureAgent)
        agent.scene = SimpleNamespace(scene_expert_stage_budget={"enabled": True})
        agent.furniture_safety_controller = SimpleNamespace(accept_score_threshold=0.75)
        trusted = {
            "score_source": "vlm_critic",
            "weighted_score": 0.6,
            "scores": SimpleNamespace(critique="Sofa faces the wall."),
        }

        should_regenerate, reason = agent.should_regenerate_for_quality(trusted)

        self.assertTrue(should_regenerate)
        self.assertIn("Sofa faces the wall", reason)
        trusted["score_source"] = "critic_fallback"
        self.assertFalse(agent.should_regenerate_for_quality(trusted)[0])
        agent.scene.scene_expert_stage_budget = {}
        trusted["score_source"] = "vlm_critic"
        self.assertFalse(agent.should_regenerate_for_quality(trusted)[0])

    @unittest.skipIf(
        StatefulFurnitureAgent is None,
        f"requires pydrake/stateful furniture imports: {_IMPORT_ERROR}",
    )
    def test_untrusted_or_hard_exhausted_candidate_triggers_fallback(self) -> None:
        agent = object.__new__(StatefulFurnitureAgent)
        agent.scene = SimpleNamespace(scene_expert_stage_budget={"enabled": True})
        agent.furniture_safety_controller = SimpleNamespace(accept_score_threshold=0.75)
        untrusted = {
            "score_source": "deterministic_hard_check",
            "weighted_score": None,
            "hard_valid": True,
        }

        should_fallback, reason = agent.should_generate_deterministic_fallback(
            untrusted,
            regeneration_attempts=0,
            max_stage_regenerations=1,
            repairable_hard_exhausted=False,
        )

        self.assertTrue(should_fallback)
        self.assertIn("without a trustworthy visual critic", reason)
        untrusted["hard_valid"] = False
        untrusted["hard_reasons"] = ["physics hard violation: collisions"]
        should_fallback, reason = agent.should_generate_deterministic_fallback(
            untrusted,
            regeneration_attempts=1,
            max_stage_regenerations=1,
            repairable_hard_exhausted=True,
        )
        self.assertTrue(should_fallback)
        self.assertIn("hard-constraint recovery exhausted", reason)

    @unittest.skipIf(
        StatefulFurnitureAgent is None,
        f"requires pydrake/stateful furniture imports: {_IMPORT_ERROR}",
    )
    def test_capture_does_not_promote_fallback_score_to_trusted_critic(self) -> None:
        agent = object.__new__(StatefulFurnitureAgent)
        fallback_scores = SimpleNamespace(critique="critic timed out")
        agent.furniture_safety_controller = SimpleNamespace(
            best_scene_state={"objects": []},
            best_scores=fallback_scores,
            best_score_source="critic_fallback",
            best_render_dir=None,
            best_weighted_score=0.575,
        )

        candidate = agent.capture_agent_candidate()

        self.assertIsNotNone(candidate)
        self.assertEqual(candidate["score_source"], "critic_fallback")
        self.assertIsNone(candidate["scores"])
        self.assertIsNone(candidate["weighted_score"])

    @unittest.skipIf(
        StatefulFurnitureAgent is None,
        f"requires pydrake/stateful furniture imports: {_IMPORT_ERROR}",
    )
    def test_fallback_provenance_is_recorded_without_render_directory(self) -> None:
        agent = object.__new__(StatefulFurnitureAgent)
        agent._last_score_provenance = {}
        neutral = CategoryScore(name="test", grade=5, comment="transport fallback")
        response = FurnitureCritiqueWithScores(
            critique="TRANSIENT LOCAL VLM TIMEOUT DURING VISUAL CRITIC SCORING.",
            realism=neutral,
            functionality=neutral,
            layout=neutral,
            layout_plausibility=neutral,
            holistic_completeness=neutral,
            prompt_following=neutral,
            reachability=neutral,
        )

        agent._write_scores_and_memory(response=response, images_dir=None)

        self.assertEqual(
            agent._last_score_provenance["score_source"], "critic_fallback"
        )
        self.assertFalse(agent._last_score_provenance["vlm_scoring_performed"])

    @unittest.skipIf(
        StatefulFurnitureAgent is None,
        f"requires pydrake/stateful furniture imports: {_IMPORT_ERROR}",
    )
    def test_fallback_artifact_has_no_placeholder_numeric_scores(self) -> None:
        agent = object.__new__(StatefulFurnitureAgent)
        neutral = CategoryScore(name="test", grade=5, comment="transport fallback")
        response = FurnitureCritiqueWithScores(
            critique="TRANSIENT LOCAL VLM TIMEOUT DURING VISUAL CRITIC SCORING.",
            realism=neutral,
            functionality=neutral,
            layout=neutral,
            layout_plausibility=neutral,
            holistic_completeness=neutral,
            prompt_following=neutral,
            reachability=neutral,
        )

        with TemporaryDirectory() as tmp:
            render_dir = Path(tmp)
            provenance = agent._write_score_artifacts(
                response=response,
                images_dir=render_dir,
                physics_context="No physics violations detected.",
            )

            self.assertEqual("critic_fallback", provenance["score_source"])
            self.assertFalse((render_dir / "scores.yaml").exists())
            self.assertFalse((render_dir / "critic_fallback_scores.yaml").exists())
            self.assertTrue((render_dir / "critic_unavailable.yaml").exists())

    @unittest.skipIf(
        StatefulFurnitureAgent is None,
        f"requires pydrake/stateful furniture imports: {_IMPORT_ERROR}",
    )
    def test_functional_reorder_is_disabled_during_normal_hard_repair(self) -> None:
        agent = object.__new__(StatefulFurnitureAgent)
        agent.scene = SimpleNamespace(
            room_type="living_room",
            text_description="A living room with a sofa.",
            scene_expert_original_description="A living room with a sofa.",
        )
        agent.furniture_safety_controller = SimpleNamespace(required_counts={})
        agent._replace_invalid_furniture_assets = lambda _state: 0
        agent._repair_forbidden_zone_conflicts = lambda include_windows=False: False
        agent._repair_structured_collisions = lambda _state: 0
        calls: list[str] = []
        agent._repair_functional_layout = lambda: calls.append("reordered") or "changed"

        repaired, _ = agent._attempt_deterministic_repair(
            SimpleNamespace(hard_valid=False, hard_reasons=[], issues=[])
        )

        self.assertFalse(repaired)
        self.assertEqual(calls, [])

    @unittest.skipIf(
        StatefulFurnitureAgent is None,
        f"requires pydrake/stateful furniture imports: {_IMPORT_ERROR}",
    )
    def test_bedroom_is_dispatched_to_deterministic_fallback_operator(self) -> None:
        agent = object.__new__(StatefulFurnitureAgent)
        agent.scene = SimpleNamespace(
            room_type="bedroom",
            text_description="A bedroom with a bed.",
            scene_expert_original_description="A bedroom with a bed.",
            objects={"bed_0": object()},
        )
        agent._repair_bedroom_layout = lambda: ["anchored bed"]

        action = agent._repair_functional_layout()

        self.assertIn("normalized bedroom fallback", action)
        self.assertIn("anchored bed", action)

    @unittest.skipIf(
        StatefulFurnitureAgent is None,
        f"requires pydrake/stateful furniture imports: {_IMPORT_ERROR}",
    )
    def test_bedroom_fallback_moves_dependent_furniture_as_one_candidate(self) -> None:
        agent = object.__new__(StatefulFurnitureAgent)
        agent.scene = SimpleNamespace(objects={"bed_0": object()})
        agent._bedroom_layout_cfg = lambda: SimpleNamespace()
        calls: list[str] = []
        agent._anchor_existing_bed = lambda: calls.append("bed") or True
        agent._repair_bedside_nightstands = (
            lambda: calls.append("nightstands") or True
        )
        agent._repair_wardrobe_wall_anchor = (
            lambda: calls.append("wardrobe") or True
        )

        with patch(
            "scenesmith.furniture_agents.stateful_furniture_agent."
            "evaluate_bedroom_layout_plausibility",
            return_value=SimpleNamespace(
                issues=[
                    "bedroom plausibility: bed headboard faces west_wall, "
                    "expected north_wall"
                ]
            ),
        ):
            actions = agent._repair_bedroom_layout()

        self.assertEqual(calls, ["bed", "nightstands", "wardrobe"])
        self.assertEqual(len(actions), 3)

    @unittest.skipIf(
        StatefulFurnitureAgent is None,
        f"requires pydrake/stateful furniture imports: {_IMPORT_ERROR}",
    )
    def test_bedroom_fallback_does_not_rearrange_unrelated_low_score(self) -> None:
        agent = object.__new__(StatefulFurnitureAgent)
        agent.scene = SimpleNamespace(objects={"bed_0": object()})
        agent._bedroom_layout_cfg = lambda: SimpleNamespace()
        calls: list[str] = []
        agent._anchor_existing_bed = lambda: calls.append("bed") or True
        agent._repair_bedside_nightstands = (
            lambda: calls.append("nightstands") or True
        )
        agent._repair_wardrobe_wall_anchor = (
            lambda: calls.append("wardrobe") or True
        )

        with patch(
            "scenesmith.furniture_agents.stateful_furniture_agent."
            "evaluate_bedroom_layout_plausibility",
            return_value=SimpleNamespace(issues=[]),
        ):
            actions = agent._repair_bedroom_layout()

        self.assertEqual(actions, [])
        self.assertEqual(calls, [])

    @unittest.skipIf(
        StatefulFurnitureAgent is None,
        f"requires pydrake/stateful furniture imports: {_IMPORT_ERROR}",
    )
    def test_non_bedroom_missing_required_asset_uses_generic_repair(self) -> None:
        agent = object.__new__(StatefulFurnitureAgent)
        agent.scene = SimpleNamespace(
            room_type="living_room",
            text_description="A living room with a sofa.",
            scene_expert_original_description="A living room with a sofa.",
        )
        agent.furniture_safety_controller = SimpleNamespace(required_counts={"sofa": 1})
        repaired_categories: list[str] = []
        agent._ensure_required_furniture_asset = lambda category: (
            repaired_categories.append(category) or 1
        )
        agent._repair_forbidden_zone_conflicts = lambda include_windows=False: False

        repaired, actions = agent._attempt_deterministic_repair(
            SimpleNamespace(
                hard_valid=False,
                hard_reasons=["missing required sofa: expected 1, found 0"],
            )
        )

        self.assertTrue(repaired)
        self.assertEqual(repaired_categories, ["sofa"])
        self.assertTrue(any("missing sofa" in action for action in actions))

    @unittest.skipIf(
        StatefulFurnitureAgent is None,
        f"requires pydrake/stateful furniture imports: {_IMPORT_ERROR}",
    )
    def test_window_warning_invokes_window_aware_repair(self) -> None:
        agent = object.__new__(StatefulFurnitureAgent)
        agent.scene = SimpleNamespace(
            room_type="bedroom",
            text_description="A bedroom with a wardrobe.",
            scene_expert_original_description="A bedroom with a wardrobe.",
        )
        agent.furniture_safety_controller = SimpleNamespace(required_counts={})
        agent._replace_invalid_furniture_assets = lambda _state: 0
        repair_calls: list[bool] = []
        agent._repair_forbidden_zone_conflicts = lambda include_windows=False: (
            repair_calls.append(include_windows) or include_windows
        )
        agent._anchor_existing_bed = lambda: False
        agent._repair_bedside_nightstands = lambda: False
        agent._repair_wardrobe_wall_anchor = lambda: False
        agent._repair_functional_layout = lambda: None
        agent._repair_structured_collisions = lambda _state: 0

        repaired, actions = agent._attempt_deterministic_repair(
            SimpleNamespace(
                hard_valid=False,
                hard_reasons=["window access warning"],
                issues=[],
            )
        )

        self.assertTrue(repaired)
        self.assertIn(True, repair_calls)
        self.assertIn("cleared deterministic window forbidden zones", actions)

    @unittest.skipIf(
        StatefulFurnitureAgent is None,
        f"requires pydrake/stateful furniture imports: {_IMPORT_ERROR}",
    )
    def test_unrelated_hard_repair_does_not_normalize_bedroom_relations(self) -> None:
        agent = object.__new__(StatefulFurnitureAgent)
        agent.scene = SimpleNamespace(
            room_type="bedroom",
            text_description="A bedroom with a bed.",
            scene_expert_original_description="A bedroom with a bed.",
        )
        agent.furniture_safety_controller = SimpleNamespace(required_counts={})
        agent._replace_geometry_failed_furniture_assets = lambda _reasons: 0
        agent._replace_invalid_furniture_assets = lambda _state: 0
        agent._repair_forbidden_zone_conflicts = lambda include_windows=False: False
        calls: list[str] = []
        agent._anchor_existing_bed = lambda: calls.append("bed") or True
        agent._repair_bedside_nightstands = (
            lambda: calls.append("nightstands") or True
        )
        agent._repair_wardrobe_wall_anchor = (
            lambda: calls.append("wardrobe") or True
        )
        agent._repair_structured_collisions = lambda _state: 0

        repaired, _ = agent._attempt_deterministic_repair(
            SimpleNamespace(
                hard_valid=False,
                hard_reasons=["geometry construction failed for chair_0"],
                issues=[],
            )
        )

        self.assertFalse(repaired)
        self.assertEqual(calls, [])

    @unittest.skipIf(
        StatefulFurnitureAgent is None,
        f"requires pydrake/stateful furniture imports: {_IMPORT_ERROR}",
    )
    def test_snap_transform_to_wall_copies_readonly_translation(self) -> None:
        agent = self._make_agent()
        agent._bounds_for_transform = lambda _obj, _transform: (
            np.asarray([-2.7, -0.2, 0.0]),
            np.asarray([-1.7, 0.2, 1.0]),
        )

        transform = ReadOnlyTranslationTransform((0.0, 0.0, 0.0))

        snapped = agent._snap_transform_to_wall(SimpleNamespace(), transform, "west")

        self.assertIsInstance(snapped, RigidTransform)
        self.assertLess(snapped.translation()[0], 0.0)

    @unittest.skipIf(
        StatefulFurnitureAgent is None,
        f"requires pydrake/stateful furniture imports: {_IMPORT_ERROR}",
    )
    def test_fit_transform_inside_room_copies_readonly_translation(self) -> None:
        agent = self._make_agent()
        agent._bounds_for_transform = lambda _obj, _transform: (
            np.asarray([-2.7, -2.2, 0.0]),
            np.asarray([-1.7, -1.2, 1.0]),
        )

        transform = ReadOnlyTranslationTransform((0.0, 0.0, 0.0))

        fitted = agent._fit_transform_inside_room(SimpleNamespace(), transform)

        self.assertIsInstance(fitted, RigidTransform)
        self.assertGreater(fitted.translation()[0], 0.0)
        self.assertGreater(fitted.translation()[1], 0.0)

    @unittest.skipIf(
        StatefulFurnitureAgent is None,
        f"requires pydrake/stateful furniture imports: {_IMPORT_ERROR}",
    )
    def test_wall_collision_repair_moves_object_inward_by_penetration(self) -> None:
        agent = self._make_agent()
        agent._fit_transform_inside_room = lambda _obj, transform: transform
        obj = SimpleNamespace(
            object_id="bed_0",
            transform=RigidTransform(np.array([0.0, 1.75, 0.0])),
        )

        repaired = agent._move_away_from_room_boundary_transform(
            obj,
            room_boundary_id="room_geometry::north_wall",
            penetration_depth_m=0.05,
        )

        self.assertIsNotNone(repaired)
        self.assertAlmostEqual(repaired.translation()[0], 0.0)
        self.assertAlmostEqual(repaired.translation()[1], 1.67)

    def _make_agent(self) -> Any:
        agent = object.__new__(StatefulFurnitureAgent)
        agent._room_bounds_xy = lambda: (-2.5, -2.0, 2.5, 2.0)
        agent._repair_cfg_value = lambda _name, default: default
        return agent


if __name__ == "__main__":
    unittest.main()
