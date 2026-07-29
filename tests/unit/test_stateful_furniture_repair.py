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
        StatefulFurnitureAgent,
    )
except ModuleNotFoundError as exc:
    RigidTransform = None
    BaseStatefulAgent = None
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

    def compute_world_bounds(self):
        center = np.asarray(self.transform.translation(), dtype=float)
        return center - self._size / 2.0, center + self._size / 2.0


class _FakeCollisionScene:
    def __init__(self, *objects: _FakeFurniture) -> None:
        self.objects = {obj.object_id: obj for obj in objects}
        self.room_geometry = SimpleNamespace(length=4.0, width=4.0)

    def move_object(self, object_id, transform) -> bool:
        self.objects[object_id].transform = transform
        return True


class StatefulFurnitureRepairTest(unittest.TestCase):
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
            constraint_mode="contract",
            metric_enabled=lambda metric: metric == "functional_dependency",
        )
        relation_fix = SimpleNamespace(
            object_id="desk_0", relation_type="study_furniture_layout"
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

        improve.assert_called_once_with(agent.scene, config=critic_config)
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
                "desk_0:study_furniture_layout",
                "office_chair_0->desk_0:seating_orientation",
            ],
        )

    @unittest.skipIf(
        StatefulFurnitureAgent is None,
        f"requires pydrake/stateful furniture imports: {_IMPORT_ERROR}",
    )
    def test_initial_contract_repair_does_not_change_shadow_or_legacy_rollout(
        self,
    ) -> None:
        agent = object.__new__(StatefulFurnitureAgent)
        agent.cfg = object()
        agent.scene = object()
        agent.rendering_manager = SimpleNamespace(clear_cache=MagicMock())
        agent._reset_critic_candidate_cache = MagicMock()
        critic_config = SimpleNamespace(
            enabled=True,
            constraint_mode="shadow",
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
                "intent_constraints": [],
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
    def test_polygon_room_skips_rectangle_only_bedroom_anchor_repairs(self) -> None:
        agent = object.__new__(StatefulFurnitureAgent)
        agent.scene = SimpleNamespace(
            room_id="polygon_bedroom",
            room_type="bedroom",
            text_description="A bedroom with a bed, nightstands, and wardrobe.",
            scene_expert_original_description=(
                "A bedroom with a bed, nightstands, and wardrobe."
            ),
            room_geometry=SimpleNamespace(
                footprint_vertices=[
                    (-3.0, -2.0),
                    (3.0, -2.0),
                    (3.0, 0.0),
                    (0.0, 0.0),
                    (0.0, 2.0),
                    (-3.0, 2.0),
                ]
            ),
        )
        agent.furniture_safety_controller = SimpleNamespace(required_counts={})
        agent._repair_forbidden_zone_conflicts = lambda include_windows=False: False
        called: list[str] = []
        agent._anchor_existing_bed = lambda: called.append("bed") or True
        agent._repair_bedside_nightstands = (
            lambda: called.append("nightstand") or True
        )
        agent._repair_wardrobe_wall_anchor = (
            lambda: called.append("wardrobe") or True
        )

        repaired, actions = agent._attempt_deterministic_repair(
            SimpleNamespace(
                hard_valid=False,
                hard_reasons=["collisions"],
            )
        )

        self.assertFalse(repaired)
        self.assertEqual(actions, [])
        self.assertEqual(called, [])

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
