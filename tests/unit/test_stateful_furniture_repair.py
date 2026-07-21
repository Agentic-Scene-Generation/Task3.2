import unittest
from typing import Any
from types import SimpleNamespace

import numpy as np

try:
    from pydrake.all import RigidTransform
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
