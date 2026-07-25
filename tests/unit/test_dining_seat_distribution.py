from __future__ import annotations

import math
import unittest

from scenesmith.scenebenchmark_critic.metrics.functional_dependency.extensions.dining_seat import (
    _equal_edge_segment_slots,
    _evaluate_table,
    _requests_one_seat_per_edge,
    evaluate_dining_seat_distribution,
)


def _bbox(center: tuple[float, float], size: tuple[float, float, float]) -> dict:
    x, y = center
    width, depth, height = size
    return {
        "center": [x, y, 0.0],
        "size": list(size),
        "min": [x - width / 2, y - depth / 2, 0.0],
        "max": [x + width / 2, y + depth / 2, height],
    }


def _table() -> dict:
    return {
        "id": "dining_table",
        "category": "dining_table",
        "functional_hints": {"scene_object_type": "furniture"},
        "yaw_deg": 0.0,
        "bbox_world": _bbox((0.0, 0.0), (3.3, 0.79, 0.79)),
    }


def _chair(identifier: str, x: float, y: float) -> dict:
    return {
        "id": identifier,
        "category": "dining_chair",
        "functional_hints": {"scene_object_type": "furniture"},
        "bbox_world": _bbox((x, y), (0.713, 0.467, 0.489)),
    }


class TestDiningSeatDistribution(unittest.TestCase):
    def test_equal_segments_generalize_to_multiple_chairs(self):
        self.assertEqual(_equal_edge_segment_slots(3.3, 1), [0.0])
        for actual, expected in zip(_equal_edge_segment_slots(3.3, 2), [-0.825, 0.825]):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(
            _equal_edge_segment_slots(3.3, 3), [-1.1, 0.0, 1.1]
        ):
            self.assertAlmostEqual(actual, expected)

    def test_two_chairs_are_centered_in_the_two_long_edge_segments(self):
        result = _evaluate_table(
            _table(),
            [_chair("chair_left", -0.825, -0.95), _chair("chair_right", 0.825, -0.95)],
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["label"], "pass")
        slots = result["diagnostics"]["seat_slots"]
        self.assertEqual([item["target_position_m"] for item in slots], [-0.825, 0.825])
        self.assertTrue(all(item["aligned"] for item in slots))

    def test_end_biased_two_chair_layout_is_not_equal_segmented(self):
        result = _evaluate_table(
            _table(),
            [_chair("chair_left", -1.255, -0.95), _chair("chair_right", 1.255, -0.95)],
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["label"], "fail")
        slots = result["diagnostics"]["seat_slots"]
        self.assertTrue(all(not item["aligned"] for item in slots))

    def test_three_chairs_use_three_equal_segment_centers(self):
        result = _evaluate_table(
            _table(),
            [
                _chair("chair_left", -1.1, -0.95),
                _chair("chair_center", 0.0, -0.95),
                _chair("chair_right", 1.1, -0.95),
            ],
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["label"], "pass")
        slots = result["diagnostics"]["seat_slots"]
        self.assertEqual([item["segment_count"] for item in slots], [3, 3, 3])
        self.assertEqual(
            [item["target_position_m"] for item in slots], [-1.1, 0.0, 1.1]
        )

    def test_rotated_table_reports_world_target_without_changing_normal_gap(self):
        yaw = math.radians(30.0)
        tangent_x = (math.cos(yaw), math.sin(yaw))
        tangent_y = (-math.sin(yaw), math.cos(yaw))
        table = _table()
        table["yaw_deg"] = 30.0
        table["footprint_world"] = [
            [
                du * tangent_x[0] + dv * tangent_y[0],
                du * tangent_x[1] + dv * tangent_y[1],
            ]
            for du, dv in (
                (-1.65, -0.395),
                (1.65, -0.395),
                (1.65, 0.395),
                (-1.65, 0.395),
            )
        ]
        chair_center = (
            0.55 * tangent_x[0] - 0.95 * tangent_y[0],
            0.55 * tangent_x[1] - 0.95 * tangent_y[1],
        )
        chair = _chair("seat_any_name", *chair_center)

        result = _evaluate_table(table, [chair])

        self.assertIsNotNone(result)
        slot = result["diagnostics"]["seat_slots"][0]
        self.assertEqual(slot["edge"], "front")
        self.assertAlmostEqual(
            slot["target_center_xy_m"][0], -0.95 * tangent_y[0], places=5
        )
        self.assertAlmostEqual(
            slot["target_center_xy_m"][1], -0.95 * tangent_y[1], places=5
        )

    def test_plain_table_is_dining_table_only_in_dining_context(self):
        table = {
            "id": "table_0",
            "category": "table",
            "functional_hints": {"scene_object_type": "furniture"},
            "yaw_deg": 0.0,
            "bbox_world": _bbox((0.0, 0.0), (2.0, 0.8, 0.8)),
        }
        result = evaluate_dining_seat_distribution(
            {
                "task_instruction": "A dining room with a table and two chairs",
                "room_type": "dining_room",
                "scene_geometry": {
                    "objects": [
                        table,
                        _chair("task_chair_0", -0.5, -0.95),
                        _chair("task_chair_1", 0.5, -0.95),
                    ]
                },
            }
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["primary_object"], "table_0")

    def test_four_sides_prompt_includes_a_remote_fourth_chair(self):
        self.assertTrue(
            _requests_one_seat_per_edge(
                "four dining chairs arranged around it with one on each of the four sides"
            )
        )
        chairs = [
            _chair("chair_front_left", -0.825, -0.95),
            _chair("chair_front_right", 0.825, -0.95),
            _chair("chair_right", 1.2, 0.0),
            _chair("chair_remote", -1.9, -1.92),
        ]
        result = evaluate_dining_seat_distribution(
            {
                "task_instruction": (
                    "A dining room with four dining chairs arranged around it "
                    "with one on each of the four sides"
                ),
                "room_type": "dining_room",
                "scene_geometry": {"objects": [_table(), *chairs]},
            }
        )

        self.assertEqual(len(result), 1)
        slots = result[0]["diagnostics"]["seat_slots"]
        self.assertEqual(
            {slot["seat_id"] for slot in slots},
            {chair["id"] for chair in chairs},
        )
        self.assertEqual({slot["segment_count"] for slot in slots}, {1})

    def test_four_sides_prompt_reports_complete_outside_edge_target(self):
        chairs = [
            _chair("chair_front", 0.0, -0.75),
            _chair("chair_back", 0.0, 0.75),
            _chair("chair_right", 1.2, 0.0),
            _chair("chair_remote", -1.9, -1.92),
        ]
        result = evaluate_dining_seat_distribution(
            {
                "task_instruction": (
                    "A dining room with four dining chairs arranged around it "
                    "with one on each of the four sides"
                ),
                "room_type": "dining_room",
                "scene_geometry": {"objects": [_table(), *chairs]},
            }
        )[0]

        remote = next(
            slot
            for slot in result["diagnostics"]["seat_slots"]
            if slot["seat_id"] == "chair_remote"
        )
        self.assertEqual(remote["edge"], "left")
        self.assertFalse(remote["aligned"])
        self.assertGreater(remote["normal_deviation_m"], 0.1)
        self.assertAlmostEqual(remote["target_center_xy_m"][0], -2.0565, places=5)
        self.assertAlmostEqual(remote["target_center_xy_m"][1], 0.0, places=5)
