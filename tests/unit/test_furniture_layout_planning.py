import unittest

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from scenesmith.agent_utils.furniture_layout_planning import (
    apply_bedroom_asset_size_policy,
    build_bedroom_anchor_plan,
    build_opening_aware_reservation_plan,
    evaluate_bedroom_layout_plausibility,
    format_opening_aware_reservation_guidance,
    is_bedroom_scene,
)


@dataclass
class DummyRoomGeometry:
    length: float = 4.5
    width: float = 4.0
    openings: list[dict[str, Any]] = field(default_factory=list)


class DummyRotation:
    def __init__(self, matrix: np.ndarray | None = None):
        self._matrix = np.eye(3) if matrix is None else np.asarray(matrix, dtype=float)

    def matrix(self) -> np.ndarray:
        return self._matrix


class DummyTransform:
    def __init__(
        self,
        translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
        rotation: np.ndarray | None = None,
    ):
        self._translation = np.array(translation, dtype=float)
        self._rotation = DummyRotation(rotation)

    def translation(self) -> np.ndarray:
        return self._translation

    def rotation(self) -> DummyRotation:
        return self._rotation


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
    def test_opening_reservations_expose_safe_walls_and_split_opening_walls(
        self,
    ) -> None:
        scene = DummyScene(
            room_geometry=DummyRoomGeometry(
                length=6.0,
                width=5.0,
                openings=[
                    {
                        "opening_id": "door_1",
                        "opening_type": "door",
                        "wall_direction": "west",
                        "center_world": [-3.0, 0.75, 1.05],
                        "width": 0.9,
                        "clearance_bbox_min": [-3.0, 0.30, 0.0],
                        "clearance_bbox_max": [-2.1, 1.20, 2.1],
                    },
                    {
                        "opening_id": "window_1",
                        "opening_type": "window",
                        "wall_direction": "east",
                        "center_world": [3.0, 0.0, 1.5],
                        "width": 1.5,
                        "clearance_bbox_min": [2.5, -0.75, 0.9],
                        "clearance_bbox_max": [3.0, 0.75, 2.1],
                    },
                ],
            ),
            room_type="living_room",
            text_description="A living room with a media area.",
        )

        plan = build_opening_aware_reservation_plan(scene)

        self.assertEqual(plan.fully_opening_free_walls, ["north", "south"])
        self.assertEqual(
            {zone.zone_id: zone.severity for zone in plan.zones},
            {"door_1": "hard", "window_1": "advisory"},
        )
        west_segments = [
            segment for segment in plan.usable_wall_segments if segment.wall == "west"
        ]
        self.assertEqual(len(west_segments), 2)
        self.assertTrue(
            all(not (segment.start <= 0.75 <= segment.end) for segment in west_segments)
        )
        guidance = format_opening_aware_reservation_guidance(scene)
        self.assertIn(
            "Fully opening-free anchor walls: north_wall, south_wall", guidance
        )
        self.assertIn("door_1 on west_wall", guidance)
        self.assertIn("window_1 on east_wall", guidance)

    def test_open_connection_without_stored_bbox_gets_hard_clearance(self) -> None:
        scene = DummyScene(
            room_geometry=DummyRoomGeometry(
                length=6.0,
                width=5.0,
                openings=[
                    {
                        "opening_id": "open_1",
                        "opening_type": "open",
                        "wall_direction": "south",
                        "center_world": [0.5, -2.5, 1.25],
                        "width": 1.4,
                        "clearance_bbox_min": None,
                        "clearance_bbox_max": None,
                    }
                ],
            ),
            room_type="living_room",
            text_description="An open-plan living room.",
        )

        plan = build_opening_aware_reservation_plan(scene)

        self.assertEqual(len(plan.zones), 1)
        self.assertEqual(plan.zones[0].severity, "hard")
        np.testing.assert_allclose(plan.zones[0].bounds_min, (-0.2, -2.5, 0.0))
        np.testing.assert_allclose(plan.zones[0].bounds_max, (1.2, -1.5, 2.5))

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

    def test_opening_limited_bedroom_allows_verified_interior_bed_layout(self) -> None:
        scene = DummyScene(
            room_geometry=DummyRoomGeometry(
                openings=[
                    {
                        "opening_type": "window",
                        "wall_direction": "north",
                        "center_world": [0.0, 2.0, 1.5],
                        "width": 1.2,
                        "clearance_bbox_min": [-0.6, 1.5, 0.0],
                        "clearance_bbox_max": [0.6, 2.0, 2.1],
                    },
                    {
                        "opening_type": "window",
                        "wall_direction": "south",
                        "center_world": [0.54, -2.0, 1.5],
                        "width": 1.2,
                        "clearance_bbox_min": [-0.06, -2.0, 0.0],
                        "clearance_bbox_max": [1.14, -1.5, 2.1],
                    },
                    {
                        "opening_type": "window",
                        "wall_direction": "east",
                        "center_world": [2.25, -0.08, 1.5],
                        "width": 1.2,
                        "clearance_bbox_min": [1.75, -0.68, 0.0],
                        "clearance_bbox_max": [2.25, 0.52, 2.1],
                    },
                    {
                        "opening_type": "door",
                        "wall_direction": "west",
                        "center_world": [-2.25, 0.0, 1.05],
                        "width": 0.9,
                        "clearance_bbox_min": [-2.25, -0.45, 0.0],
                        "clearance_bbox_max": [-1.45, 0.45, 2.1],
                    },
                ]
            ),
            text_description="A compact bedroom with a bed.",
        )
        yaw_north = np.diag([-1.0, -1.0, 1.0])
        scene.objects["bed_0"] = DummyObject(
            object_type="furniture",
            name="bed",
            description="Compact standard double bed",
            bbox_min=np.array([-0.8, -0.908, 0.0]),
            bbox_max=np.array([0.8, 0.908, 1.0]),
            transform=DummyTransform((0.0, 0.51, 0.0), rotation=yaw_north),
        )

        report = evaluate_bedroom_layout_plausibility(scene)

        self.assertFalse(any("is not anchored" in issue for issue in report.issues))
        self.assertTrue(
            any("interior opening-safe" in issue for issue in report.issues)
        )
        self.assertFalse(
            any("headboard overlaps/targets" in issue for issue in report.issues)
        )

    def test_bed_head_is_opposite_asset_facing_arrow(self) -> None:
        scene = make_bedroom_scene()
        # A -90 degree yaw points the asset's +Y arrow east, while its
        # headboard points west and therefore misses the planned east wall.
        quarter_turn = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        scene.objects["bed_0"] = DummyObject(
            object_type="furniture",
            name="bed",
            description="Bed with headboard",
            bbox_min=np.array([-0.8, -1.0, 0.0]),
            bbox_max=np.array([0.8, 1.0, 1.0]),
            transform=DummyTransform((1.4, 0.0, 0.0), quarter_turn),
        )

        report = evaluate_bedroom_layout_plausibility(scene)

        self.assertTrue(any("expected east_wall" in issue for issue in report.issues))


if __name__ == "__main__":
    unittest.main()
