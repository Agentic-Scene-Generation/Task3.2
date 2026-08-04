import unittest

from types import SimpleNamespace

import numpy as np

from scenesmith.scenebenchmark_critic.adapter import (
    _opening_record,
    _room_geometry_record,
)


class SceneBenchmarkPolygonAdapterTest(unittest.TestCase):
    def test_polygon_room_uses_exact_floor_polygon_with_room_offset(self) -> None:
        footprint = [
            (-3.0, -2.0),
            (3.0, -2.0),
            (3.0, 0.0),
            (0.0, 0.0),
            (0.0, 2.0),
            (-3.0, 2.0),
        ]
        geometry = SimpleNamespace(
            length=6.0,
            width=4.0,
            wall_height=2.8,
            footprint_vertices=footprint,
            room_local_footprint_vertices=footprint,
        )
        scene = SimpleNamespace(
            room_id="polygon_room",
            room_type="bedroom",
            room_geometry=geometry,
        )

        record = _room_geometry_record(scene, np.array([10.0, -4.0, 0.0]))

        self.assertEqual(
            record["floor_polygon"],
            [[x + 10.0, y - 4.0] for x, y in footprint],
        )
        self.assertNotIn([13.0, -2.0], record["floor_polygon"])

    def test_rectangle_room_keeps_legacy_floor_polygon(self) -> None:
        geometry = SimpleNamespace(
            length=6.0,
            width=4.0,
            wall_height=2.8,
            footprint_vertices=None,
        )
        scene = SimpleNamespace(
            room_id="rectangle_room",
            room_type="bedroom",
            room_geometry=geometry,
        )

        record = _room_geometry_record(scene, np.zeros(3))

        self.assertEqual(
            record["floor_polygon"],
            [[-3.0, -2.0], [3.0, -2.0], [3.0, 2.0], [-3.0, 2.0]],
        )

    def test_polygon_opening_preserves_stable_wall_and_oriented_clearance(self) -> None:
        opening = SimpleNamespace(
            opening_id="door_W02",
            opening_type="door",
            center_world=[1.0, 2.0, 1.0],
            width=0.9,
            height=2.1,
            sill_height=0.0,
            wall_direction=None,
            wall_id="polygon_room_wall_02",
            wall_inward_normal=[-0.6, 0.8],
            clearance_polygon=[
                [0.0, 0.0],
                [0.9, 0.0],
                [0.3, 0.8],
                [-0.6, 0.8],
            ],
            clearance_bbox_min=[-0.6, 0.0, 0.0],
            clearance_bbox_max=[0.9, 0.8, 2.1],
        )

        record = _opening_record(opening, np.array([4.0, 5.0, 0.0]))

        self.assertIsNone(record["wall_direction"])
        self.assertEqual(record["wall_id"], "polygon_room_wall_02")
        self.assertEqual(record["wall_inward_normal"], [-0.6, 0.8])
        self.assertEqual(record["clearance_polygon"][0], [4.0, 5.0])


if __name__ == "__main__":
    unittest.main()
