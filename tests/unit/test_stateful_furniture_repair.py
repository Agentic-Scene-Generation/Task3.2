import unittest
from typing import Any
from types import SimpleNamespace
from pathlib import Path

import numpy as np

try:
    from pydrake.all import RigidTransform
    from scenesmith.furniture_agents.stateful_furniture_agent import (
        StatefulFurnitureAgent,
    )
    from scenesmith.agent_utils.room import (
        ObjectType,
        RoomScene,
        SceneObject,
        UniqueID,
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

    @unittest.skipIf(
        StatefulFurnitureAgent is None,
        f"requires pydrake/stateful furniture imports: {_IMPORT_ERROR}",
    )
    def test_explicit_study_contract_restores_wall_anchored_workstation(self) -> None:
        agent = self._make_layout_agent(
            "A study with a desk centered against the back wall, an office chair "
            "tucked under the desk, two guest chairs against the side wall facing "
            "the desk, and a bookshelf on the adjacent wall."
        )
        self._add_furniture(
            agent.scene, "desk_0", "desk", [1.0, 0.0, 0.0], [1.4, 0.8, 0.75]
        )
        self._add_furniture(
            agent.scene, "bookshelf_0", "bookshelf", [-0.5, 0.0, 0.0], [0.9, 0.35, 1.8]
        )
        for index in range(3):
            self._add_furniture(
                agent.scene,
                f"office_chair_{index}",
                "office chair",
                [-0.5 + index * 0.3, 0.2, 0.0],
                [0.5, 0.5, 0.9],
            )

        actions = agent.enforce_prompt_layout_contracts()

        desk = agent.scene.objects[UniqueID("desk_0")]
        shelf = agent.scene.objects[UniqueID("bookshelf_0")]
        primary = agent.scene.objects[UniqueID("office_chair_0")]
        guests = [
            agent.scene.objects[UniqueID("office_chair_1")],
            agent.scene.objects[UniqueID("office_chair_2")],
        ]
        self.assertTrue(actions)
        self.assertAlmostEqual(float(desk.transform.translation()[0]), 0.0, places=3)
        self.assertGreater(float(desk.transform.translation()[1]), 1.5)
        self.assertGreater(float(shelf.transform.translation()[0]), 2.0)
        self.assertAlmostEqual(
            float(primary.transform.translation()[0]),
            float(desk.transform.translation()[0]),
            places=3,
        )
        self.assertLess(
            float(primary.transform.translation()[1]),
            float(desk.transform.translation()[1]),
        )
        self.assertTrue(
            all(float(chair.transform.translation()[0]) < -2.0 for chair in guests)
        )
        self.assertNotAlmostEqual(
            float(guests[0].transform.translation()[1]),
            float(guests[1].transform.translation()[1]),
            places=3,
        )

    @unittest.skipIf(
        StatefulFurnitureAgent is None,
        f"requires pydrake/stateful furniture imports: {_IMPORT_ERROR}",
    )
    def test_explicit_scene_replaces_stale_agent_scene(self) -> None:
        current_scene_agent = self._make_layout_agent(
            "A study with a desk centered against the back wall, an office chair "
            "tucked under the desk, two guest chairs against the side wall facing "
            "the desk, and a bookshelf on the adjacent wall."
        )
        for object_id, name, center, size in (
            ("desk_0", "desk", [1.0, 0.0, 0.0], [1.4, 0.8, 0.75]),
            ("bookshelf_0", "bookshelf", [-0.5, 0.0, 0.0], [0.9, 0.35, 1.8]),
            ("office_chair_0", "office chair", [-0.5, 0.2, 0.0], [0.5, 0.5, 0.9]),
            ("office_chair_1", "office chair", [-0.2, 0.2, 0.0], [0.5, 0.5, 0.9]),
            ("office_chair_2", "office chair", [0.1, 0.2, 0.0], [0.5, 0.5, 0.9]),
        ):
            self._add_furniture(
                current_scene_agent.scene, object_id, name, center, size
            )

        stale_scene_agent = self._make_layout_agent("A room with a sofa.")
        current_scene = current_scene_agent.scene
        current_scene_agent.scene = stale_scene_agent.scene

        actions = current_scene_agent.enforce_prompt_layout_contracts(
            scene=current_scene
        )

        self.assertTrue(actions)
        self.assertIs(current_scene_agent.scene, current_scene)
        self.assertGreater(
            float(current_scene.objects[UniqueID("desk_0")].transform.translation()[1]),
            1.5,
        )

    @unittest.skipIf(
        StatefulFurnitureAgent is None,
        f"requires pydrake/stateful furniture imports: {_IMPORT_ERROR}",
    )
    def test_study_contract_prefers_semantic_chairs_and_removes_duplicates(
        self,
    ) -> None:
        agent = self._make_layout_agent(
            "A study with a desk centered against the back wall, an office chair "
            "tucked under the desk, two guest chairs against the side wall facing "
            "the desk, and a bookshelf on the adjacent wall."
        )
        self._add_furniture(
            agent.scene, "desk_0", "desk", [1.0, 0.0, 0.0], [1.4, 0.8, 0.75]
        )
        self._add_furniture(
            agent.scene,
            "bookshelf_0",
            "bookshelf",
            [-0.5, 0.0, 0.0],
            [0.9, 0.35, 1.8],
        )
        self._add_furniture(
            agent.scene,
            "office_chair_0",
            "office chair",
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.9],
        )
        for index in range(2):
            self._add_furniture(
                agent.scene,
                f"armchair_{index}",
                "guest armchair",
                [1.0, index * 0.4, 0.0],
                [0.6, 0.6, 0.9],
            )
        for index in range(1, 3):
            self._add_furniture(
                agent.scene,
                f"office_chair_{index}",
                "office chair",
                [-1.0, index * 0.3, 0.0],
                [0.5, 0.5, 0.9],
            )

        actions = agent.enforce_prompt_layout_contracts()

        self.assertTrue(actions)
        self.assertAlmostEqual(
            float(
                agent.scene.objects[UniqueID("office_chair_0")].transform.translation()[
                    0
                ]
            ),
            0.0,
            places=3,
        )
        self.assertTrue(
            all(
                float(
                    agent.scene.objects[
                        UniqueID(f"armchair_{index}")
                    ].transform.translation()[0]
                )
                < -2.0
                for index in range(2)
            )
        )
        self.assertNotIn(UniqueID("office_chair_1"), agent.scene.objects)
        self.assertNotIn(UniqueID("office_chair_2"), agent.scene.objects)

    @unittest.skipIf(
        StatefulFurnitureAgent is None,
        f"requires pydrake/stateful furniture imports: {_IMPORT_ERROR}",
    )
    def test_explicit_four_chair_dining_contract_centers_each_edge(self) -> None:
        agent = self._make_layout_agent(
            "A dining room with a dining table in the center and four dining chairs "
            "arranged around it with one on each side."
        )
        self._add_furniture(
            agent.scene,
            "dining_table_0",
            "dining table",
            [0.0, 0.0, 0.0],
            [1.55, 0.95, 0.75],
        )
        for index, center in enumerate(
            ((0.5, 0.85), (-0.5, -0.85), (1.05, -0.03), (-1.05, 0.03))
        ):
            self._add_furniture(
                agent.scene,
                f"dining_chair_{index}",
                "dining chair",
                [*center, 0.0],
                [0.45, 0.59, 0.63],
            )

        actions = agent.enforce_prompt_layout_contracts()

        centers = [
            np.asarray(
                agent.scene.objects[
                    UniqueID(f"dining_chair_{index}")
                ].transform.translation()
            )
            for index in range(4)
        ]
        self.assertTrue(actions)
        self.assertEqual(sum(abs(float(center[0])) < 0.01 for center in centers), 2)
        self.assertEqual(sum(abs(float(center[1])) < 0.01 for center in centers), 2)
        self.assertTrue(any(float(center[1]) > 0.85 for center in centers))
        self.assertTrue(any(float(center[1]) < -0.85 for center in centers))
        self.assertTrue(any(float(center[0]) > 1.0 for center in centers))
        self.assertTrue(any(float(center[0]) < -1.0 for center in centers))

    def _make_agent(self) -> Any:
        agent = object.__new__(StatefulFurnitureAgent)
        agent._room_bounds_xy = lambda: (-2.5, -2.0, 2.5, 2.0)
        agent._repair_cfg_value = lambda _name, default: default
        return agent

    def _make_layout_agent(self, description: str) -> Any:
        agent = object.__new__(StatefulFurnitureAgent)
        geometry = SimpleNamespace(
            length=5.0,
            width=4.5,
            openings=[SimpleNamespace(wall_direction="south")],
        )
        scene = RoomScene(
            room_geometry=geometry,
            scene_dir=Path("."),
            text_description=description,
        )
        scene.scene_expert_original_description = description
        agent.scene = scene
        agent._repair_cfg_value = lambda _name, default: default
        return agent

    def _add_furniture(
        self,
        scene: RoomScene,
        object_id: str,
        name: str,
        center: list[float],
        size: list[float],
    ) -> None:
        half = np.asarray(size, dtype=float) / 2.0
        scene.add_object(
            SceneObject(
                object_id=UniqueID(object_id),
                object_type=ObjectType.FURNITURE,
                name=name,
                description=name,
                transform=RigidTransform(p=center),
                bbox_min=-half,
                bbox_max=half,
            )
        )


if __name__ == "__main__":
    unittest.main()
