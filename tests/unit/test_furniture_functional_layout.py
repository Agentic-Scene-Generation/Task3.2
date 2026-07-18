import math
import unittest

from types import SimpleNamespace

import numpy as np

from scenesmith.agent_utils.furniture_functional_layout import (
    choose_functional_anchor_wall,
    evaluate_functional_layout,
    format_functional_layout_guidance,
)


class _Rotation:
    def __init__(self, yaw_degrees: float) -> None:
        yaw = math.radians(yaw_degrees)
        self._matrix = np.asarray(
            [
                [math.cos(yaw), -math.sin(yaw), 0.0],
                [math.sin(yaw), math.cos(yaw), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

    def matrix(self) -> np.ndarray:
        return self._matrix


class _Transform:
    def __init__(self, x: float, y: float, yaw_degrees: float) -> None:
        self._translation = np.asarray([x, y, 0.0])
        self._rotation = _Rotation(yaw_degrees)

    def translation(self) -> np.ndarray:
        return self._translation

    def rotation(self) -> _Rotation:
        return self._rotation


class _Furniture:
    def __init__(
        self,
        name: str,
        x: float,
        y: float,
        yaw_degrees: float,
        width: float,
        depth: float,
        height: float = 0.8,
    ) -> None:
        self.name = name
        self.description = name.replace("_", " ")
        self.object_type = SimpleNamespace(value="furniture")
        self.immutable = False
        self.transform = _Transform(x, y, yaw_degrees)
        self._half = np.asarray([width / 2.0, depth / 2.0, height / 2.0])

    def compute_world_bounds(self):
        center = self.transform.translation() + np.asarray([0.0, 0.0, self._half[2]])
        return center - self._half, center + self._half


def _category(text: str) -> str | None:
    normalized = text.lower().replace("_", " ")
    for category in (
        "student desk",
        "teacher desk",
        "chair",
        "sofa",
        "rug",
        "plant",
    ):
        if category in normalized:
            return category.replace(" ", "_")
    return None


def _room(room_type: str, text: str, objects: dict, openings: list) -> SimpleNamespace:
    return SimpleNamespace(
        room_type=room_type,
        text_description=text,
        scene_expert_original_description=text,
        room_geometry=SimpleNamespace(
            length=8.0 if room_type == "classroom" else 5.0,
            width=6.0 if room_type == "classroom" else 4.5,
            openings=openings,
        ),
        objects=objects,
    )


class FurnitureFunctionalLayoutTest(unittest.TestCase):
    def test_living_room_rejects_disconnected_rug_and_same_side_plants(self) -> None:
        scene = _room(
            "living_room",
            "A living room with a sofa, rug, and two plants.",
            {
                "sofa_0": _Furniture("sofa", -1.56, -1.71, 0.0, 1.7, 0.9),
                "rug_0": _Furniture("rug", -1.67, 1.42, 0.0, 1.8, 1.8, 0.03),
                "plant_0": _Furniture("plant", -1.9, -0.9, 0.0, 0.5, 0.5, 1.2),
                "plant_1": _Furniture("plant", -1.9, 0.0, 0.0, 0.5, 0.5, 1.2),
            },
            [
                SimpleNamespace(opening_type="door", wall_direction="north"),
                SimpleNamespace(opening_type="window", wall_direction="south"),
            ],
        )

        report = evaluate_functional_layout(scene, _category)

        self.assertIsNotNone(report)
        self.assertTrue(any("not centered in front" in issue for issue in report.issues))
        self.assertTrue(any("not flanking opposite" in issue for issue in report.issues))

    def test_classroom_rejects_unpaired_rows_and_teacher_behind_students(self) -> None:
        scene = _room(
            "classroom",
            "A classroom with student desks and a teacher's desk.",
            {
                "student_desk_0": _Furniture(
                    "student_desk", -1.0, 0.5, 0.0, 0.7, 0.55
                ),
                "student_desk_1": _Furniture(
                    "student_desk", 1.0, 0.5, 90.0, 0.7, 0.55
                ),
                "chair_0": _Furniture("chair", 2.5, -2.0, 0.0, 0.48, 0.5),
                "chair_1": _Furniture("chair", -2.5, -2.0, 0.0, 0.48, 0.5),
                "teacher_desk_0": _Furniture(
                    "teacher_desk", 0.0, -1.5, 0.0, 1.4, 0.7
                ),
            },
            [
                SimpleNamespace(opening_type="door", wall_direction="west"),
                SimpleNamespace(opening_type="window", wall_direction="south"),
                SimpleNamespace(opening_type="window", wall_direction="east"),
            ],
        )

        report = evaluate_functional_layout(scene, _category)

        self.assertIsNotNone(report)
        self.assertEqual(report.anchor_wall, "north")
        self.assertTrue(any("inconsistent orientation" in issue for issue in report.issues))
        self.assertTrue(any("correctly aligned chair" in issue for issue in report.issues))
        self.assertTrue(any("not ahead" in issue for issue in report.issues))

    def test_guidance_chooses_only_solid_classroom_front_wall(self) -> None:
        scene = _room(
            "classroom",
            "A classroom with six student desks.",
            {},
            [
                SimpleNamespace(opening_type="door", wall_direction="west"),
                SimpleNamespace(opening_type="window", wall_direction="south"),
                SimpleNamespace(opening_type="window", wall_direction="east"),
            ],
        )

        self.assertEqual(choose_functional_anchor_wall(scene, "classroom"), "north")
        self.assertIn(
            "north_wall as the front/chalkboard wall",
            format_functional_layout_guidance(scene),
        )


if __name__ == "__main__":
    unittest.main()
