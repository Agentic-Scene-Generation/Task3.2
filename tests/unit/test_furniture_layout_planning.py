import unittest

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from scenesmith.agent_utils.furniture_layout_planning import (
    apply_bedroom_asset_size_policy,
    build_bedroom_anchor_plan,
    evaluate_bedroom_layout_plausibility,
    is_bedroom_scene,
)


@dataclass
class DummyRoomGeometry:
    length: float = 4.5
    width: float = 4.0
    openings: list[dict[str, Any]] = field(default_factory=list)


class DummyRotation:
    def matrix(self) -> np.ndarray:
        return np.eye(3)


class DummyTransform:
    def __init__(self, translation: tuple[float, float, float] = (0.0, 0.0, 0.0)):
        self._translation = np.array(translation, dtype=float)

    def translation(self) -> np.ndarray:
        return self._translation

    def rotation(self) -> DummyRotation:
        return DummyRotation()


@dataclass
class DummyObject:
    object_type: str
    name: str
    description: str
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    transform: DummyTransform = field(default_factory=DummyTransform)
    immutable: bool = False

    def compute_world_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        translation = self.transform.translation()
        return self.bbox_min + translation, self.bbox_max + translation


@dataclass
class DummyScene:
    room_geometry: DummyRoomGeometry
    text_description: str
    room_type: str = "bedroom"
    scene_dir: Path = Path(".")
    objects: dict[str, DummyObject] = field(default_factory=dict)


def make_bedroom_scene() -> DummyScene:
    return DummyScene(
        room_geometry=DummyRoomGeometry(
            openings=[
                {
                    "opening_type": "window",
                    "wall_direction": "north",
                    "center_world": [0.6, 2.0, 1.5],
                    "width": 1.2,
                },
                {
                    "opening_type": "window",
                    "wall_direction": "south",
                    "center_world": [-0.4, -2.0, 1.5],
                    "width": 1.2,
                },
                {
                    "opening_type": "door",
                    "wall_direction": "west",
                    "center_world": [-2.25, -0.5, 1.05],
                    "width": 0.9,
                },
            ]
        ),
        text_description=(
            "A bedroom with a bed, two nightstands, and a wardrobe in the corner."
        ),
    )


class FurnitureLayoutPlanningTest(unittest.TestCase):
    def test_injected_memory_cannot_turn_living_room_into_bedroom(self) -> None:
        scene = DummyScene(
            room_geometry=DummyRoomGeometry(),
            room_type="living_room",
            text_description="A living room. Retrieved memory mentions a bed.",
        )
        scene.scene_expert_original_description = "A living room with a sofa."

        self.assertFalse(is_bedroom_scene(scene))

    def test_anchor_plan_prefers_solid_wall_without_openings(self) -> None:
        scene = make_bedroom_scene()

        plan = build_bedroom_anchor_plan(scene)

        self.assertIsNotNone(plan)
        self.assertEqual(plan.bed_head_wall, "east")

    def test_asset_size_policy_rewrites_unqualified_large_bed(self) -> None:
        scene = make_bedroom_scene()

        result = apply_bedroom_asset_size_policy(
            scene=scene,
            object_descriptions=[
                "Queen bed with mattress",
                "Compact wooden nightstand",
            ],
            short_names=["queen_bed", "nightstand"],
            desired_dimensions=[[2.2, 2.0, 0.7], [0.5, 0.5, 0.5]],
        )

        self.assertEqual(result.short_names[0], "bed")
        self.assertIn("Compact standard double bed", result.object_descriptions[0])
        self.assertEqual(result.desired_dimensions[0], [1.6, 2.05, 0.8])
        self.assertTrue(result.notes)

    def test_plausibility_penalizes_bed_head_facing_window_wall(self) -> None:
        scene = make_bedroom_scene()
        scene.objects["bed_0"] = DummyObject(
            object_type="furniture",
            name="bed",
            description="Bed with headboard",
            bbox_min=np.array([-1.1, -1.12, 0.0]),
            bbox_max=np.array([1.1, 1.12, 1.0]),
            transform=DummyTransform((0.0, 0.25, 0.0)),
        )
        scene.objects["nightstand_0"] = DummyObject(
            object_type="furniture",
            name="nightstand",
            description="Nightstand",
            bbox_min=np.array([-0.2, -0.2, 0.0]),
            bbox_max=np.array([0.2, 0.2, 0.5]),
            transform=DummyTransform((-1.4, 0.25, 0.0)),
        )
        scene.objects["nightstand_1"] = DummyObject(
            object_type="furniture",
            name="nightstand",
            description="Nightstand",
            bbox_min=np.array([-0.2, -0.2, 0.0]),
            bbox_max=np.array([0.2, 0.2, 0.5]),
            transform=DummyTransform((1.4, 0.25, 0.0)),
        )
        scene.objects["wardrobe_0"] = DummyObject(
            object_type="furniture",
            name="wardrobe",
            description="Wardrobe",
            bbox_min=np.array([-0.45, -0.25, 0.0]),
            bbox_max=np.array([0.45, 0.25, 2.0]),
            transform=DummyTransform((1.75, 1.5, 0.0)),
        )

        report = evaluate_bedroom_layout_plausibility(scene)

        self.assertGreater(report.penalty, 0.0)
        self.assertLess(report.score, 1.0)
        self.assertTrue(any("expected east_wall" in issue for issue in report.issues))


if __name__ == "__main__":
    unittest.main()
