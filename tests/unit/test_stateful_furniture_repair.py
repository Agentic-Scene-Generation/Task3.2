import unittest

from types import SimpleNamespace
from typing import Any

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
    def test_head_wall_yaw_points_bed_arrow_inward(self) -> None:
        agent = object.__new__(StatefulFurnitureAgent)

        self.assertEqual(agent._yaw_for_head_wall("east"), 90.0)
        self.assertEqual(agent._yaw_for_head_wall("west"), -90.0)
        self.assertEqual(agent._yaw_for_head_wall("north"), 180.0)
        self.assertEqual(agent._yaw_for_head_wall("south"), 0.0)

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
    def test_broad_chair_repair_count_includes_armchairs(self) -> None:
        agent = object.__new__(StatefulFurnitureAgent)
        furniture_type = SimpleNamespace(value="furniture")
        agent.scene = SimpleNamespace(
            objects={
                "office_chair_0": SimpleNamespace(
                    name="office_chair",
                    description="task chair",
                    object_type=furniture_type,
                    immutable=False,
                ),
                "guest_armchair_0": SimpleNamespace(
                    name="guest_armchair",
                    description="armchair",
                    object_type=furniture_type,
                    immutable=False,
                ),
                "guest_armchair_1": SimpleNamespace(
                    name="guest_armchair",
                    description="armchair",
                    object_type=furniture_type,
                    immutable=False,
                ),
            }
        )

        self.assertEqual(3, len(agent._furniture_by_category("chair")))

    @unittest.skipIf(
        StatefulFurnitureAgent is None,
        f"requires pydrake/stateful furniture imports: {_IMPORT_ERROR}",
    )
    def test_counted_inventory_removes_placeholder_duplicate(self) -> None:
        agent = object.__new__(StatefulFurnitureAgent)
        furniture_type = SimpleNamespace(value="furniture")
        normal = SimpleNamespace(
            object_id="sideboard_0",
            name="sideboard",
            description="solid wood sideboard",
            object_type=furniture_type,
            immutable=False,
            metadata={},
        )
        placeholder = SimpleNamespace(
            object_id="sideboard_repair_placeholder_0",
            name="sideboard",
            description="deterministic placeholder sideboard",
            object_type=furniture_type,
            immutable=False,
            metadata={"repair_placeholder": True},
        )
        agent.scene = SimpleNamespace(
            objects={normal.object_id: normal, placeholder.object_id: placeholder}
        )
        agent.scene.remove_object = lambda object_id: agent.scene.objects.pop(object_id)
        agent.furniture_safety_controller = SimpleNamespace(
            infer_object_category=lambda _text: "sideboard"
        )
        agent._nearest_room_boundary_distance = lambda obj: (
            0.02 if obj is normal else 0.5
        )

        removed = agent._remove_excess_required_furniture({"sideboard": 1})

        self.assertEqual(removed, 1)
        self.assertEqual(set(agent.scene.objects), {"sideboard_0"})

    @unittest.skipIf(
        StatefulFurnitureAgent is None,
        f"requires pydrake/stateful furniture imports: {_IMPORT_ERROR}",
    )
    def test_inventory_convergence_removes_normal_duplicate_without_hard_failure(
        self,
    ) -> None:
        agent = object.__new__(StatefulFurnitureAgent)
        furniture_type = SimpleNamespace(value="furniture")
        wall_backed = SimpleNamespace(
            object_id="sideboard_0",
            name="sideboard",
            description="compact sideboard",
            object_type=furniture_type,
            immutable=False,
            metadata={},
        )
        duplicate = SimpleNamespace(
            object_id="sideboard_1",
            name="sideboard",
            description="second compact sideboard",
            object_type=furniture_type,
            immutable=False,
            metadata={},
        )
        agent.scene = SimpleNamespace(
            objects={
                wall_backed.object_id: wall_backed,
                duplicate.object_id: duplicate,
            }
        )
        agent.scene.remove_object = lambda object_id: agent.scene.objects.pop(object_id)
        agent.furniture_safety_controller = SimpleNamespace(
            required_counts={"sideboard": 1},
            infer_object_category=lambda _text: "sideboard",
        )
        agent._nearest_room_boundary_distance = lambda obj: (
            0.02 if obj is wall_backed else 0.7
        )
        agent.rendering_manager = SimpleNamespace(clear_cache=lambda: None)
        agent._reset_critic_candidate_cache = lambda: None

        removed = agent._converge_prompt_required_inventory(source="test")

        self.assertEqual(removed, 1)
        self.assertEqual(set(agent.scene.objects), {"sideboard_0"})

    @unittest.skipIf(
        StatefulFurnitureAgent is None,
        f"requires pydrake/stateful furniture imports: {_IMPORT_ERROR}",
    )
    def test_dining_inventory_prefers_sideboard_on_table_long_side_wall(self) -> None:
        agent = object.__new__(StatefulFurnitureAgent)
        table = SimpleNamespace(
            object_id="dining_table_0",
            name="dining_table",
            description="rectangular dining table",
            bbox_min=np.array([-0.9, -0.45, 0.0]),
            bbox_max=np.array([0.9, 0.45, 0.75]),
            transform=RigidTransform(),
        )
        north_sideboard = SimpleNamespace(
            compute_world_bounds=lambda: (
                np.array([-0.7, 1.70, 0.0]),
                np.array([0.7, 2.17, 0.8]),
            )
        )
        west_sideboard = SimpleNamespace(
            compute_world_bounds=lambda: (
                np.array([-2.43, -0.7, 0.0]),
                np.array([-1.96, 0.7, 0.8]),
            )
        )
        agent.scene = SimpleNamespace(
            room_geometry=SimpleNamespace(length=5.0, width=4.5),
            objects={table.object_id: table},
        )

        north_penalty = agent._dining_sideboard_wall_penalty(north_sideboard)
        west_penalty = agent._dining_sideboard_wall_penalty(west_sideboard)

        self.assertEqual(north_penalty, 0.0)
        self.assertEqual(west_penalty, 1.0)

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
