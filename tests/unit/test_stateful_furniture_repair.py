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
    def test_transaction_repairs_hard_failure_before_rollback(self) -> None:
        agent = object.__new__(StatefulFurnitureAgent)
        remembered: list[tuple[dict[str, Any], str]] = []
        ended: list[bool] = []
        agent.scene = SimpleNamespace(to_state_dict=lambda: {"objects": ["repaired"]})
        agent.furniture_safety_controller = SimpleNamespace(
            enabled=True,
            best_scene_state={"objects": ["old"]},
            remember_hard_valid_scene_state=lambda scene_state, source: remembered.append(
                (scene_state, source)
            ),
            end_designer_call=lambda: ended.append(True),
        )
        agent._evaluate_current_furniture_hard_state = lambda: SimpleNamespace(
            hard_valid=False,
            hard_reasons=["wardrobe collision"],
        )
        repair_sources: list[str] = []

        def repair(hard_state, *, source):
            repair_sources.append(source)
            return (
                SimpleNamespace(hard_valid=True, hard_reasons=[]),
                None,
                ["moved wardrobe to wall anchor"],
            )

        agent._try_deterministic_repair_for_hard_state = repair

        result = agent._end_furniture_design_transaction(
            {
                "call_kind": "change",
                "pre_state": {"objects": ["old"]},
                "pre_hard_valid": True,
            }
        )

        self.assertEqual(repair_sources, ["post-change-transaction"])
        self.assertEqual(remembered, [({"objects": ["repaired"]}, "post-change")])
        self.assertEqual(ended, [True])
        self.assertIn("hard checks passed", result)

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
        agent.furniture_safety_controller = SimpleNamespace(
            required_counts={"sofa": 1}
        )
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
    def test_snap_transform_to_wall_copies_readonly_translation(self) -> None:
        agent = self._make_agent()
        agent._bounds_for_transform = lambda _obj, _transform: (
            np.asarray([-2.7, -0.2, 0.0]),
            np.asarray([-1.7, 0.2, 1.0]),
        )

        transform = ReadOnlyTranslationTransform((0.0, 0.0, 0.0))

        snapped = agent._snap_transform_to_wall(SimpleNamespace(), transform, "west")

        self.assertIsInstance(snapped, RigidTransform)
        # The object bounds start outside the west boundary, so moving the
        # object's origin by a positive offset is the correct inward snap.
        self.assertGreater(snapped.translation()[0], 0.0)

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

    def _make_agent(self) -> Any:
        agent = object.__new__(StatefulFurnitureAgent)
        agent._room_bounds_xy = lambda: (-2.5, -2.0, 2.5, 2.0)
        agent._repair_cfg_value = lambda _name, default: default
        return agent


if __name__ == "__main__":
    unittest.main()
