import asyncio
import unittest

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

try:
    from pydrake.all import RigidTransform

    from scenesmith.agent_utils.base_stateful_agent import BaseStatefulAgent
    from scenesmith.agent_utils.room import ObjectType
    from scenesmith.furniture_agents.stateful_furniture_agent import (
        REPAIR_ASSET_SPECS,
        StatefulFurnitureAgent,
    )
except ModuleNotFoundError as exc:
    RigidTransform = None
    BaseStatefulAgent = None
    REPAIR_ASSET_SPECS = {}
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


class _FakeFurniture:
    def __init__(
        self,
        object_id: str,
        translation: tuple[float, float, float],
        size: tuple[float, float, float],
    ) -> None:
        self.object_id = object_id
        self.name = object_id
        self.immutable = False
        self.object_type = ObjectType.FURNITURE
        self.transform = RigidTransform(p=translation)
        self._size = np.asarray(size, dtype=float)
        self.bbox_min = -self._size / 2.0
        self.bbox_max = self._size / 2.0

    def compute_world_bounds(self):
        center = np.asarray(self.transform.translation(), dtype=float)
        return center - self._size / 2.0, center + self._size / 2.0


class _FakeCollisionScene:
    def __init__(self, *objects: _FakeFurniture) -> None:
        self.objects = {obj.object_id: obj for obj in objects}
        self.room_geometry = SimpleNamespace(length=4.0, width=4.0, wall_height=2.7)

    def move_object(self, object_id, transform) -> bool:
        self.objects[object_id].transform = transform
        return True


class StatefulFurnitureRepairTest(unittest.TestCase):
    def test_repair_asset_specs_cover_specialized_scene_furniture(self) -> None:
        expected_categories = {
            "coffee_table",
            "dining_table",
            "dressing_table",
            "stool",
            "storage_cabinet",
            "water_dispenser",
        }

        self.assertTrue(expected_categories.issubset(REPAIR_ASSET_SPECS))
        self.assertIn("no integrated mirror", REPAIR_ASSET_SPECS["dressing_table"][0])

    @unittest.skipIf(
        StatefulFurnitureAgent is None,
        f"requires pydrake/stateful furniture imports: {_IMPORT_ERROR}",
    )
    def test_bed_anchor_detects_only_openings_on_its_head_wall(self) -> None:
        agent = object.__new__(StatefulFurnitureAgent)
        agent.scene = SimpleNamespace(
            room_geometry=SimpleNamespace(
                openings=[
                    SimpleNamespace(
                        opening_type="window",
                        wall_direction="north",
                        center_world=(0.5, 2.0, 1.5),
                        width=1.0,
                    ),
                    SimpleNamespace(
                        opening_type="window",
                        wall_direction="east",
                        center_world=(2.0, 0.0, 1.5),
                        width=1.0,
                    ),
                ]
            )
        )

        blocked = (
            np.array([0.1, 1.6, 0.0]),
            np.array([0.9, 2.0, 1.0]),
        )
        clear = (
            np.array([-1.8, 1.6, 0.0]),
            np.array([-0.8, 2.0, 1.0]),
        )

        self.assertTrue(agent._bed_anchor_overlaps_opening(blocked, "north"))
        self.assertFalse(agent._bed_anchor_overlaps_opening(clear, "north"))

    @unittest.skipIf(
        StatefulFurnitureAgent is None,
        f"requires pydrake/stateful furniture imports: {_IMPORT_ERROR}",
    )
    def test_window_safe_bed_anchor_shifts_sideways_without_leaving_wall(self) -> None:
        bed = _FakeFurniture("bed_0", (-0.05, 1.42, 0.0), (1.6, 1.8, 1.0))
        agent = object.__new__(StatefulFurnitureAgent)
        agent.cfg = SimpleNamespace(furniture_safety_controller=None)
        agent.scene = SimpleNamespace(
            room_geometry=SimpleNamespace(
                length=5.5,
                width=4.8,
                openings=[
                    SimpleNamespace(
                        opening_type="window",
                        wall_direction="north",
                        center_world=(1.0, 2.4, 1.6),
                        width=1.5,
                    )
                ],
            )
        )

        anchored = RigidTransform(p=(-0.05, 1.42, 0.0))
        repaired = agent._best_window_safe_bed_anchor_transform(
            bed=bed,
            transform=anchored,
            wall="north",
        )
        repaired_bounds = agent._bounds_for_transform(bed, repaired)

        self.assertIsNotNone(repaired_bounds)
        self.assertNotAlmostEqual(
            float(repaired.translation()[0]), float(anchored.translation()[0])
        )
        self.assertAlmostEqual(
            float(repaired.translation()[1]), float(anchored.translation()[1])
        )
        self.assertFalse(agent._bed_anchor_overlaps_opening(repaired_bounds, "north"))

    @unittest.skipIf(
        StatefulFurnitureAgent is None,
        f"requires pydrake/stateful furniture imports: {_IMPORT_ERROR}",
    )
    def test_window_safe_bed_anchor_uses_minimum_safe_lateral_margin(self) -> None:
        bed = _FakeFurniture("bed_0", (0.0, 1.1, 0.0), (1.6, 1.8, 1.0))
        agent = object.__new__(StatefulFurnitureAgent)
        agent.cfg = SimpleNamespace(
            furniture_safety_controller=SimpleNamespace(
                deterministic_repair=SimpleNamespace(wall_margin_m=0.08)
            )
        )
        agent.scene = SimpleNamespace(
            room_geometry=SimpleNamespace(
                length=4.5,
                width=4.0,
                openings=[
                    SimpleNamespace(
                        opening_type="window",
                        wall_direction="north",
                        center_world=(-0.025, 2.0, 1.5),
                        width=1.2,
                    )
                ],
            )
        )

        repaired = agent._best_window_safe_bed_anchor_transform(
            bed=bed,
            transform=RigidTransform(p=(0.0, 1.1, 0.0)),
            wall="north",
        )
        repaired_bounds = agent._bounds_for_transform(bed, repaired)

        self.assertIsNotNone(repaired_bounds)
        self.assertGreater(float(repaired.translation()[0]), 1.4)
        self.assertLessEqual(float(repaired_bounds[1][0]), 2.22 + 1e-6)
        self.assertFalse(agent._bed_anchor_overlaps_opening(repaired_bounds, "north"))

    @unittest.skipIf(
        StatefulFurnitureAgent is None,
        f"requires pydrake/stateful furniture imports: {_IMPORT_ERROR}",
    )
    def test_bed_uses_interior_pose_when_every_wall_slot_blocks_an_opening(
        self,
    ) -> None:
        bed = _FakeFurniture("bed_0", (0.0, 1.0, 0.0), (1.6, 1.816, 1.0))
        agent = object.__new__(StatefulFurnitureAgent)
        agent.cfg = SimpleNamespace(
            furniture_safety_controller=SimpleNamespace(
                deterministic_repair=SimpleNamespace(wall_margin_m=0.08)
            )
        )
        agent.scene = _FakeCollisionScene(bed)
        agent.scene.room_type = "bedroom"
        agent.scene.room_geometry = SimpleNamespace(
            length=4.5,
            width=4.0,
            wall_thickness=0.05,
            openings=[
                SimpleNamespace(
                    opening_id="north_window",
                    opening_type="window",
                    wall_direction="north",
                    center_world=(0.0, 2.0, 1.5),
                    width=1.2,
                    clearance_bbox_min=(-0.6, 1.5, 0.0),
                    clearance_bbox_max=(0.6, 2.0, 2.1),
                ),
                SimpleNamespace(
                    opening_id="south_window",
                    opening_type="window",
                    wall_direction="south",
                    center_world=(0.54, -2.0, 1.5),
                    width=1.2,
                    clearance_bbox_min=(-0.06, -2.0, 0.0),
                    clearance_bbox_max=(1.14, -1.5, 2.1),
                ),
                SimpleNamespace(
                    opening_id="east_window",
                    opening_type="window",
                    wall_direction="east",
                    center_world=(2.25, -0.08, 1.5),
                    width=1.2,
                    clearance_bbox_min=(1.75, -0.68, 0.0),
                    clearance_bbox_max=(2.25, 0.52, 2.1),
                ),
                SimpleNamespace(
                    opening_id="west_door",
                    opening_type="door",
                    wall_direction="west",
                    center_world=(-2.25, 0.0, 1.05),
                    width=0.9,
                    clearance_bbox_min=(-2.25, -0.45, 0.0),
                    clearance_bbox_max=(-1.45, 0.45, 2.1),
                ),
            ],
        )

        self.assertTrue(agent._anchor_existing_bed())
        self.assertLess(float(bed.transform.translation()[1]), 0.7)
        self.assertTrue(agent._bed_transform_clears_openings(bed, bed.transform))
        bed_bounds = bed.compute_world_bounds()
        self.assertGreater(float(bed_bounds[0][0]), -1.35)

    @unittest.skipIf(
        StatefulFurnitureAgent is None,
        f"requires pydrake/stateful furniture imports: {_IMPORT_ERROR}",
    )
    def test_bed_anchor_preserves_in_room_fallback_when_no_safe_slot_exists(
        self,
    ) -> None:
        bed = _FakeFurniture("bed_0", (0.0, 0.85, 0.0), (1.6, 1.816, 1.0))
        bench = _FakeFurniture("bench_0", (0.0, -0.57, 0.0), (1.0, 0.41, 0.4))
        agent = object.__new__(StatefulFurnitureAgent)
        agent.cfg = SimpleNamespace(
            furniture_safety_controller=SimpleNamespace(
                deterministic_repair=SimpleNamespace(wall_margin_m=0.08)
            )
        )
        agent.scene = _FakeCollisionScene(bed, bench)
        agent.scene.room_type = "bedroom"
        agent.scene.room_geometry = SimpleNamespace(
            length=4.5,
            width=4.0,
            wall_thickness=0.05,
            openings=[
                SimpleNamespace(
                    opening_id="north_window",
                    opening_type="window",
                    wall_direction="north",
                    center_world=(0.0, 2.0, 1.5),
                    width=1.2,
                    clearance_bbox_min=(-0.6, 1.5, 0.0),
                    clearance_bbox_max=(0.6, 2.0, 2.1),
                ),
                SimpleNamespace(
                    opening_id="south_window",
                    opening_type="window",
                    wall_direction="south",
                    center_world=(0.54, -2.0, 1.5),
                    width=1.2,
                    clearance_bbox_min=(-0.06, -2.0, 0.0),
                    clearance_bbox_max=(1.14, -1.5, 2.1),
                ),
                SimpleNamespace(
                    opening_id="east_window",
                    opening_type="window",
                    wall_direction="east",
                    center_world=(2.25, -0.08, 1.5),
                    width=1.2,
                    clearance_bbox_min=(1.75, -0.68, 0.0),
                    clearance_bbox_max=(2.25, 0.52, 2.1),
                ),
                SimpleNamespace(
                    opening_id="west_door",
                    opening_type="door",
                    wall_direction="west",
                    center_world=(-2.25, 0.0, 1.05),
                    width=0.9,
                    clearance_bbox_min=(-2.25, -0.45, 0.0),
                    clearance_bbox_max=(-1.45, 0.45, 2.1),
                ),
            ],
        )

        agent._anchor_existing_bed()
        self.assertAlmostEqual(float(bed.transform.translation()[0]), 0.0)
        self.assertAlmostEqual(float(bed.transform.translation()[1]), 0.85)

    @unittest.skipIf(
        StatefulFurnitureAgent is None,
        f"requires pydrake/stateful furniture imports: {_IMPORT_ERROR}",
    )
    def test_contract_relation_failure_only_disqualifies_checkpoint(self) -> None:
        agent = object.__new__(StatefulFurnitureAgent)
        agent.scene = SimpleNamespace()
        agent.cfg = {
            "scenebenchmark_critic": {
                "enabled": True,
                "metrics": ["functional_dependency"],
            }
        }
        agent.furniture_safety_controller = SimpleNamespace(
            enabled=True,
            evaluate_scene_state=lambda **_kwargs: SimpleNamespace(
                hard_valid=True,
                hard_reasons=[],
                soft_reasons=[],
            ),
        )

        module = "scenesmith.agent_utils.base_stateful_agent"
        with patch(f"{module}.unresolved_furniture_relation_failures") as unresolved:
            unresolved.return_value = [
                {
                    "relation_type": "centered_between_alignment",
                    "primary_object": "coffee_table_0",
                }
            ]

            physical_evaluation = agent._evaluate_current_furniture_hard_state(
                physics_context="No physics violations detected."
            )
            evaluation = agent._checkpoint_eligible_furniture_hard_state(
                physical_evaluation
            )

        self.assertTrue(physical_evaluation.hard_valid)
        self.assertFalse(evaluation.hard_valid)
        self.assertEqual(
            evaluation.hard_reasons,
            [
                "unresolved prompt-core furniture relation: "
                "centered_between_alignment:coffee_table_0"
            ],
        )

    @unittest.skipIf(
        StatefulFurnitureAgent is None,
        f"requires pydrake/stateful furniture imports: {_IMPORT_ERROR}",
    )
    def test_wall_height_repair_grounds_accidentally_elevated_furniture(self) -> None:
        agent = object.__new__(StatefulFurnitureAgent)
        wardrobe = _FakeFurniture("wardrobe_0", (0.0, 0.0, 2.05), (0.9, 0.6, 1.8))
        agent.scene = _FakeCollisionScene(wardrobe)

        grounded = agent._ground_elevated_floor_furniture()

        self.assertEqual(grounded, 1)
        world_bounds = wardrobe.compute_world_bounds()
        self.assertAlmostEqual(float(world_bounds[0][2]), 0.0)
        self.assertLess(
            float(world_bounds[1][2]), agent.scene.room_geometry.wall_height
        )

    @unittest.skipIf(
        StatefulFurnitureAgent is None,
        f"requires pydrake/stateful furniture imports: {_IMPORT_ERROR}",
    )
    def test_wall_height_repair_grounds_just_over_tolerance(self) -> None:
        agent = object.__new__(StatefulFurnitureAgent)
        wardrobe = _FakeFurniture("wardrobe_0", (0.0, 0.0, 1.351), (0.9, 0.6, 2.7))
        agent.scene = _FakeCollisionScene(wardrobe)

        grounded = agent._ground_elevated_floor_furniture()

        self.assertEqual(grounded, 1)
        self.assertAlmostEqual(float(wardrobe.compute_world_bounds()[0][2]), 0.0)

    @unittest.skipIf(
        StatefulFurnitureAgent is None,
        f"requires pydrake/stateful furniture imports: {_IMPORT_ERROR}",
    )
    def test_generic_wall_repair_applies_to_bedroom_furniture(self) -> None:
        agent = object.__new__(StatefulFurnitureAgent)
        nightstand = _FakeFurniture("nightstand_0", (0.0, -1.98, 0.3), (0.5, 0.45, 0.6))
        agent.scene = _FakeCollisionScene(nightstand)
        agent.cfg = SimpleNamespace(
            furniture_safety_controller=SimpleNamespace(
                deterministic_repair=SimpleNamespace(wall_margin_m=0.08)
            )
        )

        self.assertTrue(agent._repair_generic_wall_collisions())
        lower, _ = nightstand.compute_world_bounds()
        self.assertGreaterEqual(float(lower[1]), -1.92 - 1e-6)

    @unittest.skipIf(
        StatefulFurnitureAgent is None,
        f"requires pydrake/stateful furniture imports: {_IMPORT_ERROR}",
    )
    def test_generic_wall_repair_preserves_bed_window_clearance(self) -> None:
        bed = _FakeFurniture("bed_0", (1.414551, 1.1, 0.0), (1.6, 1.8, 1.0))
        agent = object.__new__(StatefulFurnitureAgent)
        agent.cfg = SimpleNamespace(
            furniture_safety_controller=SimpleNamespace(
                deterministic_repair=SimpleNamespace(wall_margin_m=0.08)
            )
        )
        agent.scene = _FakeCollisionScene(bed)
        agent.scene.room_type = "bedroom"
        agent.scene.room_geometry = SimpleNamespace(
            length=4.5,
            width=4.0,
            openings=[
                SimpleNamespace(
                    opening_type="window",
                    wall_direction="north",
                    center_world=(-0.025, 2.0, 1.5),
                    width=1.2,
                ),
                SimpleNamespace(
                    opening_type="door",
                    wall_direction="south",
                    center_world=(0.0, -2.0, 1.0),
                    width=0.9,
                ),
                SimpleNamespace(
                    opening_type="door",
                    wall_direction="east",
                    center_world=(2.25, 0.0, 1.0),
                    width=0.9,
                ),
                SimpleNamespace(
                    opening_type="door",
                    wall_direction="west",
                    center_world=(-2.25, 0.0, 1.0),
                    width=0.9,
                ),
            ],
        )

        self.assertTrue(agent._repair_generic_wall_collisions())
        repaired_bounds = bed.compute_world_bounds()

        self.assertGreater(float(bed.transform.translation()[0]), 1.4)
        self.assertFalse(agent._bed_anchor_overlaps_opening(repaired_bounds, "north"))

    @unittest.skipIf(
        StatefulFurnitureAgent is None,
        f"requires pydrake/stateful furniture imports: {_IMPORT_ERROR}",
    )
    def test_shallow_collision_repair_preserves_wall_anchor_axis(self) -> None:
        agent = object.__new__(StatefulFurnitureAgent)
        chair = _FakeFurniture("guest_chair_0", (-1.57, 0.0, 0.45), (0.70, 0.70, 0.90))
        shelf = _FakeFurniture("bookshelf_0", (-1.50, 0.65, 0.90), (0.80, 0.70, 1.80))
        agent.scene = _FakeCollisionScene(chair, shelf)
        agent.cfg = SimpleNamespace(
            furniture_safety_controller=SimpleNamespace(
                deterministic_repair=SimpleNamespace(
                    collision_separation_max_penetration_m=0.08,
                    collision_separation_margin_m=0.025,
                    wall_anchor_preservation_distance_m=0.16,
                    wall_margin_m=0.08,
                )
            )
        )
        agent._get_cached_physics_context = lambda: (
            "Physics violations detected (1 issue(s)):\n"
            "Collisions (1):\n"
            "- guest_chair_0 collides with bookshelf_0 (2.5cm penetration)"
        )

        old_translation = chair.transform.translation().copy()
        actions = agent._repair_shallow_furniture_collisions()

        self.assertEqual(len(actions), 1)
        self.assertAlmostEqual(chair.transform.translation()[0], old_translation[0])
        self.assertNotAlmostEqual(chair.transform.translation()[1], old_translation[1])
        self.assertEqual(agent._furniture_aabb_overlap_pairs(), set())

    @unittest.skipIf(
        StatefulFurnitureAgent is None,
        f"requires pydrake/stateful furniture imports: {_IMPORT_ERROR}",
    )
    def test_shallow_collision_repair_uses_aabb_depth_when_larger_than_report(
        self,
    ) -> None:
        agent = object.__new__(StatefulFurnitureAgent)
        chair = _FakeFurniture("guest_chair_1", (-1.57, 0.0, 0.45), (0.70, 0.70, 0.90))
        shelf = _FakeFurniture("bookshelf_0", (-1.50, 0.55, 0.90), (0.80, 0.70, 1.80))
        agent.scene = _FakeCollisionScene(chair, shelf)
        agent.cfg = SimpleNamespace(
            furniture_safety_controller=SimpleNamespace(
                deterministic_repair=SimpleNamespace(
                    collision_separation_max_penetration_m=0.08,
                    collision_separation_margin_m=0.025,
                    wall_anchor_preservation_distance_m=0.16,
                    wall_margin_m=0.08,
                )
            )
        )
        agent._get_cached_physics_context = lambda: (
            "- guest_chair_1 collides with bookshelf_0 (3.2cm penetration)"
        )

        self.assertEqual(len(agent._repair_shallow_furniture_collisions()), 1)
        self.assertEqual(agent._furniture_aabb_overlap_pairs(), set())

    @unittest.skipIf(
        StatefulFurnitureAgent is None,
        f"requires pydrake/stateful furniture imports: {_IMPORT_ERROR}",
    )
    def test_deep_collision_is_left_for_the_planner(self) -> None:
        agent = object.__new__(StatefulFurnitureAgent)
        chair = _FakeFurniture("chair_0", (0.0, 0.0, 0.45), (0.70, 0.70, 0.90))
        shelf = _FakeFurniture("shelf_0", (0.0, 0.65, 0.90), (0.80, 0.70, 1.80))
        agent.scene = _FakeCollisionScene(chair, shelf)
        agent.cfg = SimpleNamespace(
            furniture_safety_controller=SimpleNamespace(
                deterministic_repair=SimpleNamespace(
                    collision_separation_max_penetration_m=0.08,
                    collision_separation_margin_m=0.025,
                    wall_anchor_preservation_distance_m=0.16,
                    wall_margin_m=0.08,
                )
            )
        )
        agent._get_cached_physics_context = lambda: (
            "- chair_0 collides with shelf_0 (20cm penetration)"
        )

        old_translation = chair.transform.translation().copy()
        self.assertEqual(agent._repair_shallow_furniture_collisions(), [])
        np.testing.assert_allclose(chair.transform.translation(), old_translation)

    @unittest.skipIf(
        StatefulFurnitureAgent is None,
        f"requires pydrake/stateful furniture imports: {_IMPORT_ERROR}",
    )
    def test_planner_turn_limit_recovers_only_after_hard_repair(self) -> None:
        from agents.exceptions import MaxTurnsExceeded

        agent = object.__new__(StatefulFurnitureAgent)
        agent._evaluate_current_hard_state = lambda: SimpleNamespace(hard_valid=False)
        agent._try_deterministic_repair_for_hard_state = lambda _state, source: (
            SimpleNamespace(hard_valid=True),
            None,
            ["separated shallow collision guest_chair_1<->bookshelf_0"],
        )

        actions = agent._recover_from_planner_turn_limit(MaxTurnsExceeded("limit"))

        self.assertEqual(
            actions, ["separated shallow collision guest_chair_1<->bookshelf_0"]
        )

    @unittest.skipIf(
        StatefulFurnitureAgent is None,
        f"requires pydrake/stateful furniture imports: {_IMPORT_ERROR}",
    )
    def test_wardrobe_wall_anchor_cannot_reintroduce_door_blockage(self) -> None:
        agent = object.__new__(StatefulFurnitureAgent)
        wardrobe = _FakeFurniture("wardrobe_0", (0.0, 0.0, 1.0), (0.8, 0.6, 2.0))
        bed = _FakeFurniture("bed_0", (2.0, 0.0, 0.4), (0.5, 0.5, 0.8))
        agent.scene = _FakeCollisionScene(wardrobe, bed)
        blocked = RigidTransform(p=[-2.0, 0.0, 1.0])
        clear = RigidTransform(p=[0.0, 1.5, 1.0])
        agent._wardrobe_candidate_transforms = lambda _obj: [
            (blocked, 0.0),
            (clear, 0.0),
        ]
        agent._opening_forbidden_zones = lambda include_windows=False: [
            (
                "door_1",
                "door",
                np.asarray([-2.5, -0.5, 0.0]),
                np.asarray([-1.5, 0.5, 2.5]),
            )
        ]
        agent._furniture_by_category = lambda category: {
            "wardrobe": [wardrobe],
            "bed": [bed],
            "nightstand": [],
        }.get(category, [])

        self.assertTrue(agent._repair_wardrobe_wall_anchor())
        np.testing.assert_allclose(
            wardrobe.transform.translation(), clear.translation()
        )

    @unittest.skipIf(
        StatefulFurnitureAgent is None,
        f"requires pydrake/stateful furniture imports: {_IMPORT_ERROR}",
    )
    def test_wardrobe_wall_anchor_cannot_reintroduce_dresser_collision(self) -> None:
        agent = object.__new__(StatefulFurnitureAgent)
        wardrobe = _FakeFurniture("wardrobe_0", (0.0, 0.0, 1.0), (0.8, 0.6, 2.0))
        bed = _FakeFurniture("bed_0", (1.8, 0.0, 0.4), (0.5, 0.5, 0.8))
        dresser = _FakeFurniture("dresser_0", (0.0, 1.5, 0.5), (1.0, 0.7, 1.0))
        agent.scene = _FakeCollisionScene(wardrobe, bed, dresser)
        colliding = RigidTransform(p=[0.0, 1.5, 1.0])
        clear = RigidTransform(p=[1.3, -1.0, 1.0])
        agent._wardrobe_candidate_transforms = lambda _obj: [
            (colliding, 0.0),
            (clear, 0.0),
        ]
        agent._opening_forbidden_zones = lambda include_windows=False: []
        agent._furniture_by_category = lambda category: {
            "wardrobe": [wardrobe],
            "bed": [bed],
            "nightstand": [],
            "dresser": [dresser],
        }.get(category, [])

        self.assertTrue(agent._repair_wardrobe_wall_anchor())
        np.testing.assert_allclose(
            wardrobe.transform.translation(), clear.translation()
        )

    @unittest.skipIf(
        StatefulFurnitureAgent is None,
        f"requires pydrake/stateful furniture imports: {_IMPORT_ERROR}",
    )
    def test_wardrobe_wall_anchor_yields_a_prompt_reserved_free_wall(self) -> None:
        agent = object.__new__(StatefulFurnitureAgent)
        wardrobe = _FakeFurniture("wardrobe_0", (0.0, 0.0, 1.0), (0.8, 0.6, 2.0))
        bed = _FakeFurniture("bed_0", (0.0, 0.0, 0.4), (0.5, 0.5, 0.8))
        dressing_table = _FakeFurniture(
            "dressing_table_0", (-1.5, 0.0, 0.5), (1.0, 0.5, 0.8)
        )
        agent.scene = _FakeCollisionScene(wardrobe, bed, dressing_table)
        agent.scene.scenebenchmark_intent_contract = {
            "constraints": [
                {
                    "relation": "against_wall",
                    "subjects": {"category": "dressing_table", "count": 1},
                    "targets": {"category": "wall", "role": "free"},
                    "strength": "hard",
                }
            ]
        }
        west = RigidTransform(p=[-1.5, 1.0, 1.0])
        east = RigidTransform(p=[1.5, 1.0, 1.0])
        agent._wardrobe_candidate_transforms = lambda _obj: [(west, 0.0), (east, 0.0)]
        agent._opening_forbidden_zones = lambda include_windows=False: []
        agent._furniture_by_category = lambda category: {
            "wardrobe": [wardrobe],
            "bed": [bed],
            "nightstand": [],
            "dressing_table": [dressing_table],
        }.get(category, [])

        self.assertTrue(agent._repair_wardrobe_wall_anchor())
        np.testing.assert_allclose(wardrobe.transform.translation(), east.translation())

    @unittest.skipIf(
        StatefulFurnitureAgent is None,
        f"requires pydrake/stateful furniture imports: {_IMPORT_ERROR}",
    )
    def test_initial_design_repairs_before_planner_can_request_critique(self) -> None:
        """The first automatic critic must receive the post-repair scene."""
        agent = object.__new__(StatefulFurnitureAgent)
        events: list[str] = []

        async def initial_design(*_args: Any, **_kwargs: Any) -> str:
            events.append("designer")
            return "designer report"

        agent._repair_initial_contract_layout = lambda: events.append("repair") or []
        with patch.object(
            BaseStatefulAgent,
            "_request_initial_design_impl",
            AsyncMock(side_effect=initial_design),
        ):
            report = asyncio.run(agent._request_initial_design_impl())

        self.assertEqual(report, "designer report")
        self.assertEqual(events, ["designer", "repair"])

    @unittest.skipIf(
        StatefulFurnitureAgent is None,
        f"requires pydrake/stateful furniture imports: {_IMPORT_ERROR}",
    )
    def test_critic_uses_immutable_prompt_not_mutable_stage_brief(self) -> None:
        agent = object.__new__(StatefulFurnitureAgent)
        agent.cfg = SimpleNamespace(
            agents=SimpleNamespace(
                critic_agent=SimpleNamespace(prompt="STATEFUL_CRITIC_AGENT")
            )
        )
        scene = SimpleNamespace(
            scene_expert_original_description=(
                "A study with guest chairs facing into the room."
            ),
            text_description="Mutable StageBrief: guest chairs face the desk.",
        )

        with patch.object(
            BaseStatefulAgent,
            "_create_critic_agent",
            return_value=MagicMock(),
        ) as create:
            agent._create_critic_agent(scene, [])

        self.assertEqual(
            create.call_args.kwargs["scene_description"],
            scene.scene_expert_original_description,
        )

    @unittest.skipIf(
        StatefulFurnitureAgent is None,
        f"requires pydrake/stateful furniture imports: {_IMPORT_ERROR}",
    )
    def test_initial_contract_repair_is_geometry_only_and_clears_caches(self) -> None:
        agent = object.__new__(StatefulFurnitureAgent)
        agent.cfg = object()
        agent.scene = object()
        agent.rendering_manager = SimpleNamespace(clear_cache=MagicMock())
        agent._reset_critic_candidate_cache = MagicMock()
        critic_config = SimpleNamespace(
            enabled=True,
            metric_enabled=lambda metric: metric == "functional_dependency",
        )
        relation_fix = SimpleNamespace(
            object_id="desk_0", relation_type="room_center_alignment"
        )
        seating_fix = SimpleNamespace(subject_id="office_chair_0", target_id="desk_0")
        module = "scenesmith.furniture_agents.stateful_furniture_agent"
        targets = {"office_chair_0": {"desk_0"}}

        with (
            patch(f"{module}.critic_config_from_any", return_value=critic_config),
            patch(
                f"{module}.improve_furniture_relations", return_value=[relation_fix]
            ) as improve,
            patch(
                f"{module}.seating_orientation_targets", return_value=targets
            ) as seating_targets,
            patch(
                f"{module}.align_seating_to_nearest_surface", return_value=[seating_fix]
            ) as align,
        ):
            actions = agent._repair_initial_contract_layout()

        improve.assert_called_once_with(
            agent.scene,
            config=critic_config,
            candidate_validator=agent._is_furniture_relation_candidate_hard_valid,
        )
        seating_targets.assert_called_once_with(agent.scene, config=critic_config)
        align.assert_called_once_with(
            agent.scene,
            allowed_targets_by_seat=targets,
        )
        agent.rendering_manager.clear_cache.assert_called_once_with()
        agent._reset_critic_candidate_cache.assert_called_once_with()
        self.assertEqual(
            actions,
            [
                "desk_0:room_center_alignment",
                "office_chair_0->desk_0:seating_orientation",
            ],
        )

    @unittest.skipIf(
        StatefulFurnitureAgent is None,
        f"requires pydrake/stateful furniture imports: {_IMPORT_ERROR}",
    )
    def test_initial_contract_repair_is_noop_when_critic_disabled(
        self,
    ) -> None:
        agent = object.__new__(StatefulFurnitureAgent)
        agent.cfg = object()
        agent.scene = object()
        agent.rendering_manager = SimpleNamespace(clear_cache=MagicMock())
        agent._reset_critic_candidate_cache = MagicMock()
        critic_config = SimpleNamespace(
            enabled=False,
            metric_enabled=lambda _metric: True,
        )
        module = "scenesmith.furniture_agents.stateful_furniture_agent"

        with (
            patch(f"{module}.critic_config_from_any", return_value=critic_config),
            patch(f"{module}.improve_furniture_relations") as improve,
            patch(f"{module}.seating_orientation_targets") as seating_targets,
            patch(f"{module}.align_seating_to_nearest_surface") as align,
        ):
            self.assertEqual(agent._repair_initial_contract_layout(), [])

        improve.assert_not_called()
        seating_targets.assert_not_called()
        align.assert_not_called()
        agent.rendering_manager.clear_cache.assert_not_called()
        agent._reset_critic_candidate_cache.assert_not_called()

    @unittest.skipIf(
        StatefulFurnitureAgent is None,
        f"requires pydrake/stateful furniture imports: {_IMPORT_ERROR}",
    )
    def test_inventory_repair_binds_new_furniture_to_contract_relations(self) -> None:
        agent = object.__new__(StatefulFurnitureAgent)
        agent.cfg = object()
        agent.scene = object()
        critic_config = SimpleNamespace(
            enabled=True,
            metric_enabled=lambda metric: metric == "functional_dependency",
        )
        relation_fix = SimpleNamespace(
            object_id="student_chair_0",
            relation_type="seating_to_work_surface",
        )
        seating_fix = SimpleNamespace(
            subject_id="student_chair_0",
            target_id="student_desk_0",
        )
        module = "scenesmith.furniture_agents.stateful_furniture_agent"
        targets = {"student_chair_0": {"student_desk_0"}}

        with (
            patch(f"{module}.critic_config_from_any", return_value=critic_config),
            patch(
                f"{module}.improve_furniture_relations",
                return_value=[relation_fix],
            ) as improve,
            patch(
                f"{module}.seating_orientation_targets",
                return_value=targets,
            ) as seating_targets,
            patch(
                f"{module}.align_seating_to_nearest_surface",
                return_value=[seating_fix],
            ) as align,
        ):
            actions = agent._repair_relations_after_inventory_change()

        improve.assert_called_once_with(
            agent.scene,
            config=critic_config,
            candidate_validator=agent._is_furniture_relation_candidate_hard_valid,
        )
        seating_targets.assert_called_once_with(agent.scene, config=critic_config)
        align.assert_called_once_with(
            agent.scene,
            allowed_targets_by_seat=targets,
        )
        self.assertEqual(
            actions,
            [
                "bound student_chair_0 via seating_to_work_surface after inventory repair",
                "aligned student_chair_0 toward student_desk_0 after inventory repair",
            ],
        )

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
    def test_transaction_repairs_checkpoint_only_contract_failure(self) -> None:
        agent = object.__new__(StatefulFurnitureAgent)
        remembered: list[tuple[dict[str, Any], str]] = []
        ended: list[bool] = []
        agent.scene = SimpleNamespace(to_state_dict=lambda: {"objects": ["repaired"]})
        agent.furniture_safety_controller = SimpleNamespace(
            enabled=True,
            best_scene_state=None,
            remember_hard_valid_scene_state=lambda scene_state, source: remembered.append(
                (scene_state, source)
            ),
            end_designer_call=lambda: ended.append(True),
        )
        physical_state = SimpleNamespace(hard_valid=True, hard_reasons=[])
        relation_state = SimpleNamespace(
            hard_valid=False,
            hard_reasons=[
                "unresolved prompt-core furniture relation: edge_distribution:table_0"
            ],
        )
        agent._evaluate_current_furniture_hard_state = lambda: physical_state
        checkpoint_states = iter((relation_state, physical_state))
        agent._checkpoint_eligible_furniture_hard_state = lambda _state: next(
            checkpoint_states
        )
        repair_calls: list[tuple[str, list[str]]] = []

        def repair(hard_state, *, source):
            repair_calls.append((source, list(hard_state.hard_reasons)))
            return (
                hard_state,
                None,
                (
                    ["bound dining chairs via edge_distribution"]
                    if source.endswith("contract-transaction")
                    else []
                ),
            )

        agent._try_deterministic_repair_for_hard_state = repair

        result = agent._end_furniture_design_transaction(
            {
                "call_kind": "change",
                "pre_state": {"objects": ["old"]},
                "pre_hard_valid": False,
            }
        )

        self.assertEqual(
            repair_calls,
            [
                ("post-change-transaction", []),
                (
                    "post-change-contract-transaction",
                    [
                        "unresolved prompt-core furniture relation: "
                        "edge_distribution:table_0"
                    ],
                ),
            ],
        )
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
        relation_repairs: list[bool] = []
        agent._repair_relations_after_inventory_change = lambda: (
            relation_repairs.append(True)
            or ["bound sofa_0 via back_against_wall after inventory repair"]
        )

        repaired, actions = agent._attempt_deterministic_repair(
            SimpleNamespace(
                hard_valid=False,
                hard_reasons=["missing required sofa: expected 1, found 0"],
            )
        )

        self.assertTrue(repaired)
        self.assertEqual(repaired_categories, ["sofa"])
        self.assertEqual(relation_repairs, [True])
        self.assertTrue(any("missing sofa" in action for action in actions))
        self.assertTrue(any("bound sofa_0" in action for action in actions))

    @unittest.skipIf(
        StatefulFurnitureAgent is None,
        f"requires pydrake/stateful furniture imports: {_IMPORT_ERROR}",
    )
    def test_inventory_repair_rechecks_door_clearance_before_relations(self) -> None:
        agent = object.__new__(StatefulFurnitureAgent)
        agent.scene = SimpleNamespace(
            room_type="classroom",
            text_description="A classroom with student desks.",
            scene_expert_original_description="A classroom with student desks.",
        )
        agent.furniture_safety_controller = SimpleNamespace(
            required_counts={"student_desk": 1}
        )
        agent._ensure_required_furniture_asset = lambda _category: 1
        clearance_calls: list[bool] = []

        def repair_forbidden_zone_conflicts(*, include_windows: bool = False) -> bool:
            clearance_calls.append(include_windows)
            return True

        agent._repair_forbidden_zone_conflicts = repair_forbidden_zone_conflicts
        agent._repair_relations_after_inventory_change = lambda: [
            "bound student_desk_0 via edge_distribution after inventory repair"
        ]
        agent._remove_excess_required_furniture = lambda _counts: 0

        repaired, actions = agent._attempt_deterministic_repair(
            SimpleNamespace(
                hard_valid=False,
                hard_reasons=["missing required student_desk: expected 1, found 0"],
            )
        )

        self.assertTrue(repaired)
        self.assertEqual(clearance_calls, [False])
        self.assertIn("cleared deterministic door/opening forbidden zones", actions)

    @unittest.skipIf(
        StatefulFurnitureAgent is None,
        f"requires pydrake/stateful furniture imports: {_IMPORT_ERROR}",
    )
    def test_forbidden_zone_repair_rejects_all_category_furniture_collisions(
        self,
    ) -> None:
        moving = _FakeFurniture("teacher_desk_0", (0.0, 0.0, 0.4), (1.0, 1.0, 0.8))
        blocker = _FakeFurniture("chair_0", (1.0, 0.0, 0.4), (1.0, 1.0, 0.8))
        agent = object.__new__(StatefulFurnitureAgent)
        agent.scene = _FakeCollisionScene(moving, blocker)
        agent.cfg = SimpleNamespace(furniture_safety_controller=None)
        bad = RigidTransform(p=(1.0, 0.0, 0.4))
        safe = RigidTransform(p=(-1.0, 0.0, 0.4))
        agent._generic_wall_candidate_transforms = lambda _obj: [bad, safe]
        zones = [
            ("door_1", "door", np.array([0.8, -0.5, 0.0]), np.array([1.2, 0.5, 2.0]))
        ]

        repaired = agent._best_forbidden_zone_repair_transform(moving, zones)

        self.assertIsNotNone(repaired)
        repaired_bounds = agent._bounds_for_transform(moving, repaired)
        self.assertIsNotNone(repaired_bounds)
        self.assertEqual(agent._zone_overlap_penalty(repaired_bounds, zones), 0.0)
        blocker_bounds = blocker.compute_world_bounds()
        overlap_x, overlap_y = agent._xy_overlap_depths(repaired_bounds, blocker_bounds)
        overlap_z = max(
            0.0,
            float(
                min(repaired_bounds[1][2], blocker_bounds[1][2])
                - max(repaired_bounds[0][2], blocker_bounds[0][2])
            ),
        )
        self.assertFalse(overlap_x > 1e-4 and overlap_y > 1e-4 and overlap_z > 1e-4)

    @unittest.skipIf(
        StatefulFurnitureAgent is None,
        f"requires pydrake/stateful furniture imports: {_IMPORT_ERROR}",
    )
    def test_non_bedroom_relation_failure_repairs_without_inventory_change(
        self,
    ) -> None:
        agent = object.__new__(StatefulFurnitureAgent)
        agent.scene = SimpleNamespace(
            room_type="dining_room",
            text_description="A dining room with a table and chairs.",
            scene_expert_original_description="A dining room with a table and chairs.",
        )
        agent.furniture_safety_controller = SimpleNamespace(required_counts={})
        agent._repair_forbidden_zone_conflicts = lambda include_windows=False: False
        repair_calls: list[bool] = []
        agent._repair_unresolved_prompt_contract_relations = lambda: (
            repair_calls.append(True)
            or [
                "bound dining_chair_0 via edge_distribution after hard constraint failure"
            ]
        )

        repaired, actions = agent._attempt_deterministic_repair(
            SimpleNamespace(
                hard_valid=False,
                hard_reasons=[
                    "unresolved prompt-core furniture relation: "
                    "edge_distribution:table_0"
                ],
            )
        )

        self.assertTrue(repaired)
        self.assertEqual(repair_calls, [True])
        self.assertEqual(
            actions,
            [
                "bound dining_chair_0 via edge_distribution "
                "after hard constraint failure"
            ],
        )

    @unittest.skipIf(
        StatefulFurnitureAgent is None,
        f"requires pydrake/stateful furniture imports: {_IMPORT_ERROR}",
    )
    def test_non_bedroom_collision_repairs_prompt_support_before_separation(
        self,
    ) -> None:
        agent = object.__new__(StatefulFurnitureAgent)
        agent.scene = SimpleNamespace(
            room_type="living_room",
            text_description="A living room with a television on a TV stand.",
            scene_expert_original_description=(
                "A living room with a television on a TV stand."
            ),
        )
        agent.furniture_safety_controller = SimpleNamespace(required_counts={})
        agent._repair_forbidden_zone_conflicts = lambda include_windows=False: False
        agent._repair_generic_wall_collisions = lambda: False
        agent._remove_excess_required_furniture = lambda _counts: 0
        agent._repair_shallow_furniture_collisions = lambda: []
        repair_contexts: list[str] = []
        agent._repair_prompt_contract_relations = lambda context: (
            repair_contexts.append(context)
            or [
                "bound television_0 via object_on_support after physical collision repair"
            ]
        )

        repaired, actions = agent._attempt_deterministic_repair(
            SimpleNamespace(
                hard_valid=False,
                hard_reasons=["physics hard violation: collisions"],
            )
        )

        self.assertTrue(repaired)
        self.assertEqual(repair_contexts, ["after physical collision repair"])
        self.assertEqual(
            actions,
            [
                "bound television_0 via object_on_support "
                "after physical collision repair"
            ],
        )

    @unittest.skipIf(
        StatefulFurnitureAgent is None,
        f"requires pydrake/stateful furniture imports: {_IMPORT_ERROR}",
    )
    def test_bedroom_relation_failure_repairs_without_inventory_change(self) -> None:
        agent = object.__new__(StatefulFurnitureAgent)
        agent.scene = SimpleNamespace(
            room_type="bedroom",
            text_description="A bedroom with a bed and two nightstands.",
            scene_expert_original_description="A bedroom with a bed and two nightstands.",
        )
        agent.furniture_safety_controller = SimpleNamespace(required_counts={})
        agent._repair_forbidden_zone_conflicts = lambda include_windows=False: False
        agent._anchor_existing_bed = lambda: False
        agent._repair_bedside_nightstands = lambda: False
        agent._prompt_requires_wardrobe_next_to_dresser = lambda: False
        agent._remove_excess_required_furniture = lambda _counts: 0
        repair_calls: list[bool] = []
        agent._repair_unresolved_prompt_contract_relations = lambda: (
            repair_calls.append(True)
            or [
                "bound nightstand_0 via paired_surface_facing after hard constraint failure"
            ]
        )

        repaired, actions = agent._attempt_deterministic_repair(
            SimpleNamespace(
                hard_valid=False,
                hard_reasons=[
                    "unresolved prompt-core furniture relation: "
                    "paired_surface_facing:nightstand_0"
                ],
            )
        )

        self.assertTrue(repaired)
        self.assertEqual(repair_calls, [True])
        self.assertEqual(
            actions,
            [
                "bound nightstand_0 via paired_surface_facing "
                "after hard constraint failure"
            ],
        )

    @unittest.skipIf(
        StatefulFurnitureAgent is None,
        f"requires pydrake/stateful furniture imports: {_IMPORT_ERROR}",
    )
    def test_bedroom_revalidates_opening_clearance_after_layout_repair(self) -> None:
        agent = object.__new__(StatefulFurnitureAgent)
        agent.scene = SimpleNamespace(
            room_type="bedroom",
            text_description="A bedroom with a bed and two nightstands.",
            scene_expert_original_description="A bedroom with a bed and two nightstands.",
        )
        agent.furniture_safety_controller = SimpleNamespace(required_counts={})
        forbidden_zone_calls: list[bool] = []

        def repair_forbidden_zone_conflicts(*, include_windows: bool = False) -> bool:
            forbidden_zone_calls.append(include_windows)
            return len(forbidden_zone_calls) == 2

        agent._repair_forbidden_zone_conflicts = repair_forbidden_zone_conflicts
        agent._anchor_existing_bed = lambda: False
        agent._repair_bedside_nightstands = lambda: True
        agent._prompt_requires_wardrobe_next_to_dresser = lambda: False
        agent._remove_excess_required_furniture = lambda _counts: 0

        repaired, actions = agent._attempt_deterministic_repair(
            SimpleNamespace(
                hard_valid=False,
                hard_reasons=[
                    "physics hard violation: door clearance violations",
                ],
            )
        )

        self.assertTrue(repaired)
        self.assertEqual(forbidden_zone_calls, [False, False])
        self.assertIn("revalidated deterministic door/opening forbidden zones", actions)

    @unittest.skipIf(
        StatefulFurnitureAgent is None,
        f"requires pydrake/stateful furniture imports: {_IMPORT_ERROR}",
    )
    def test_bedroom_collision_repairs_shallow_optional_furniture(self) -> None:
        agent = object.__new__(StatefulFurnitureAgent)
        agent.scene = SimpleNamespace(
            room_type="bedroom",
            text_description="A bedroom with a bed and two nightstands.",
            scene_expert_original_description="A bedroom with a bed and two nightstands.",
        )
        agent.furniture_safety_controller = SimpleNamespace(required_counts={})
        agent._repair_forbidden_zone_conflicts = lambda include_windows=False: False
        agent._anchor_existing_bed = lambda: False
        agent._repair_bedside_nightstands = lambda: False
        agent._prompt_requires_wardrobe_next_to_dresser = lambda: False
        agent._repair_wardrobe_wall_anchor = lambda: False
        agent._remove_excess_required_furniture = lambda _counts: 0
        agent._repair_generic_wall_collisions = lambda: False
        agent._repair_shallow_furniture_collisions = lambda: [
            "separated shallow collision bed_0<->bench_0 by moving bench_0"
        ]

        repaired, actions = agent._attempt_deterministic_repair(
            SimpleNamespace(
                hard_valid=False,
                hard_reasons=["physics hard violation: collisions"],
            )
        )

        self.assertTrue(repaired)
        self.assertEqual(
            actions,
            ["separated shallow collision bed_0<->bench_0 by moving bench_0"],
        )

    @unittest.skipIf(
        StatefulFurnitureAgent is None,
        f"requires pydrake/stateful furniture imports: {_IMPORT_ERROR}",
    )
    def test_bedside_anchor_search_keeps_nightstand_out_of_door_zone(self) -> None:
        bed = _FakeFurniture("bed_0", (0.0, -1.145, 0.4), (1.6, 2.05, 0.8))
        left = _FakeFurniture(
            "nightstand_left", (-1.22, -1.948, 0.325), (0.614, 0.443, 0.65)
        )
        right = _FakeFurniture(
            "nightstand_right", (1.22, -1.948, 0.325), (0.614, 0.443, 0.65)
        )
        scene = _FakeCollisionScene(bed, left, right)
        scene.room_geometry.width = 4.5
        scene.room_geometry.openings = [
            SimpleNamespace(
                opening_id="door_1",
                opening_type="door",
                clearance_bbox_min=np.array([-2.0, -1.739, 0.0]),
                clearance_bbox_max=np.array([-1.2, -0.839, 2.1]),
            )
        ]
        agent = object.__new__(StatefulFurnitureAgent)
        agent.scene = scene
        agent.cfg = SimpleNamespace(furniture_safety_controller=None)
        agent.furniture_safety_controller = SimpleNamespace(
            required_counts={"nightstand": 2}
        )

        self.assertTrue(agent._repair_bedside_nightstands())
        zones = agent._opening_forbidden_zones(include_windows=False)
        for nightstand in (left, right):
            bounds = nightstand.compute_world_bounds()
            self.assertEqual(agent._zone_overlap_penalty(bounds, zones), 0.0)
        self.assertGreater(float(left.transform.translation()[0]), -1.22)

    @unittest.skipIf(
        StatefulFurnitureAgent is None,
        f"requires pydrake/stateful furniture imports: {_IMPORT_ERROR}",
    )
    def test_generic_desk_hard_failure_repairs_task_roles(self) -> None:
        agent = object.__new__(StatefulFurnitureAgent)
        agent.scene = SimpleNamespace(
            room_type="classroom",
            text_description="A classroom with student and teacher desks.",
            scene_expert_original_description="A classroom with student and teacher desks.",
            scene_expert_task_spec={
                "required_large_objects": [
                    *["student desk"] * 6,
                    "teacher's desk",
                    *["chair"] * 6,
                ],
            },
            objects={},
        )
        agent.furniture_safety_controller = SimpleNamespace(
            required_counts={"desk": 7, "chair": 6}
        )
        repaired_categories: list[str] = []
        agent._ensure_required_furniture_asset = lambda category: (
            repaired_categories.append(category) or 1
        )
        agent._repair_forbidden_zone_conflicts = lambda include_windows=False: False
        agent._repair_relations_after_inventory_change = lambda: []

        repaired, _ = agent._attempt_deterministic_repair(
            SimpleNamespace(
                hard_valid=False,
                hard_reasons=["missing required desk: expected 7, found 0"],
            )
        )

        self.assertTrue(repaired)
        self.assertEqual(repaired_categories, ["student_desk", "teacher_desk"])

    @unittest.skipIf(
        StatefulFurnitureAgent is None,
        f"requires pydrake/stateful furniture imports: {_IMPORT_ERROR}",
    )
    def test_role_specific_counts_replace_covered_generic_inventory(self) -> None:
        agent = object.__new__(StatefulFurnitureAgent)
        agent.scene = SimpleNamespace(objects={})
        agent.furniture_safety_controller = SimpleNamespace(
            required_counts={
                "desk": 7,
                "student_desk": 6,
                "teacher_desk": 1,
                "chair": 6,
                "student_chair": 6,
            }
        )

        self.assertEqual(
            agent._repair_required_counts(),
            {"student_desk": 6, "teacher_desk": 1, "student_chair": 6},
        )

    @unittest.skipIf(
        StatefulFurnitureAgent is None,
        f"requires pydrake/stateful furniture imports: {_IMPORT_ERROR}",
    )
    def test_task_counts_override_prompt_safety_counts(self) -> None:
        agent = object.__new__(StatefulFurnitureAgent)
        agent.scene = SimpleNamespace(
            objects={},
            scene_expert_task_spec={
                "required_large_objects": [
                    "desk",
                    "desk",
                    "office_chair",
                    "office_chair",
                ],
            },
        )
        agent.furniture_safety_controller = SimpleNamespace(
            enabled=True,
            required_counts={"desk": 2, "office_chair": 1},
            required_terms={"desk", "office_chair"},
        )
        agent.stage_working_memory = MagicMock()

        agent._synchronize_task_required_counts()

        self.assertEqual(
            agent.furniture_safety_controller.required_counts,
            {"desk": 2, "office_chair": 2},
        )
        agent.stage_working_memory.set_required_counts.assert_called_once_with(
            {"desk": 2, "office_chair": 2}
        )

    @unittest.skipIf(
        StatefulFurnitureAgent is None,
        f"requires pydrake/stateful furniture imports: {_IMPORT_ERROR}",
    )
    def test_intent_contract_subtype_replaces_generic_task_count(self) -> None:
        agent = object.__new__(StatefulFurnitureAgent)
        agent.scene = SimpleNamespace(
            objects={},
            scene_expert_task_spec={
                "required_large_objects": ["chair"] * 9,
            },
            scenebenchmark_intent_contract={
                "constraints": [
                    {
                        "relation": "edge_distribution",
                        "subjects": {
                            "category": "office_chair",
                            "count": 7,
                        },
                    }
                ]
            },
        )
        agent.furniture_safety_controller = SimpleNamespace(
            required_counts={"chair": 9}
        )

        self.assertEqual(agent._repair_required_counts(), {"office_chair": 7})

    @unittest.skipIf(
        StatefulFurnitureAgent is None,
        f"requires pydrake/stateful furniture imports: {_IMPORT_ERROR}",
    )
    def test_dressing_table_contract_replaces_generic_table_task_count(self) -> None:
        agent = object.__new__(StatefulFurnitureAgent)
        agent.scene = SimpleNamespace(
            objects={},
            scene_expert_task_spec={"required_large_objects": ["table"]},
            scenebenchmark_intent_contract={
                "constraints": [
                    {
                        "relation": "required_count",
                        "subjects": {"category": "dressing_table", "count": 1},
                        "stage": "furniture",
                        "strength": "hard",
                    }
                ]
            },
        )
        agent.furniture_safety_controller = SimpleNamespace(
            required_counts={"table": 1}
        )

        self.assertEqual(agent._repair_required_counts(), {"dressing_table": 1})

    @unittest.skipIf(
        StatefulFurnitureAgent is None,
        f"requires pydrake/stateful furniture imports: {_IMPORT_ERROR}",
    )
    def test_task_count_sync_removes_replaced_generic_table_term(self) -> None:
        agent = object.__new__(StatefulFurnitureAgent)
        agent.scene = SimpleNamespace(
            objects={},
            scene_expert_task_spec={"required_large_objects": ["table"]},
            scenebenchmark_intent_contract={
                "constraints": [
                    {
                        "relation": "required_count",
                        "subjects": {"category": "dressing_table", "count": 1},
                        "stage": "furniture",
                        "strength": "hard",
                    }
                ]
            },
        )
        agent.furniture_safety_controller = SimpleNamespace(
            enabled=True,
            required_counts={"table": 1},
            required_terms={"table"},
        )
        agent.stage_working_memory = MagicMock()

        agent._synchronize_task_required_counts()

        self.assertEqual(
            agent.furniture_safety_controller.required_counts,
            {"dressing_table": 1},
        )
        self.assertEqual(
            agent.furniture_safety_controller.required_terms,
            {"dressing_table"},
        )
        agent.stage_working_memory.set_required_counts.assert_called_once_with(
            {"dressing_table": 1}
        )

    @unittest.skipIf(
        StatefulFurnitureAgent is None,
        f"requires pydrake/stateful furniture imports: {_IMPORT_ERROR}",
    )
    def test_hard_media_contract_includes_repairable_television_inventory(self) -> None:
        agent = object.__new__(StatefulFurnitureAgent)
        agent.scene = SimpleNamespace(
            objects={},
            scenebenchmark_intent_contract={
                "constraints": [
                    {
                        "relation": "on_top_of",
                        "stage": "furniture",
                        "strength": "hard",
                        "subjects": {"category": "television", "count": 1},
                        "targets": {"category": "tv_stand", "count": 1},
                    }
                ]
            },
        )
        agent.furniture_safety_controller = SimpleNamespace(required_counts={})

        self.assertEqual(
            agent._repair_required_counts(), {"television": 1, "tv_stand": 1}
        )

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
    def test_inventory_prefers_explicit_sideboard_name_over_cabinet_description(
        self,
    ) -> None:
        agent = object.__new__(StatefulFurnitureAgent)
        furniture_type = SimpleNamespace(value="furniture")
        original = SimpleNamespace(
            object_id="sideboard_0",
            name="sideboard",
            description=(
                "Traditional wooden sideboard with drawers and cabinet doors for "
                "dining room storage"
            ),
            object_type=furniture_type,
            immutable=False,
            metadata={},
        )
        repair = SimpleNamespace(
            object_id="sideboard_1",
            name="sideboard",
            description="Compact dining room sideboard",
            object_type=furniture_type,
            immutable=False,
            metadata={},
        )
        agent.scene = SimpleNamespace(
            objects={original.object_id: original, repair.object_id: repair}
        )
        agent.scene.remove_object = lambda object_id: agent.scene.objects.pop(object_id)
        agent.furniture_safety_controller = SimpleNamespace(
            required_counts={"sideboard": 1}
        )
        agent._nearest_room_boundary_distance = lambda obj: (
            0.02 if obj is original else 0.7
        )

        self.assertEqual(
            agent._category_for_object(original.object_id, original), "sideboard"
        )
        self.assertEqual(
            {obj.object_id for obj in agent._furniture_by_category("sideboard")},
            {"sideboard_0", "sideboard_1"},
        )
        self.assertEqual(agent._remove_excess_required_furniture({"sideboard": 1}), 1)
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
