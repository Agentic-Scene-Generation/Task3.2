import unittest

from types import SimpleNamespace

from scenesmith.agent_utils.furniture_functional_layout import (
    furnishable_room_bounds_xy,
)


class FurnishableRoomBoundsTest(unittest.TestCase):
    def test_bounds_use_wall_inner_faces(self) -> None:
        scene = SimpleNamespace(
            room_geometry=SimpleNamespace(
                length=5.0,
                width=4.0,
                wall_thickness=0.05,
            )
        )

        self.assertEqual(
            furnishable_room_bounds_xy(scene),
            (-2.45, -1.95, 2.45, 1.95),
        )

    def test_degenerate_inner_room_is_rejected(self) -> None:
        scene = SimpleNamespace(
            room_geometry=SimpleNamespace(
                length=0.08,
                width=1.0,
                wall_thickness=0.05,
            )
        )

        self.assertIsNone(furnishable_room_bounds_xy(scene))


if __name__ == "__main__":
    unittest.main()
