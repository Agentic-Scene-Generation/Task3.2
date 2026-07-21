from __future__ import annotations

import unittest

from scenesmith.scenebenchmark_critic.metrics.functional_dependency.extensions.dining_seat import (
    _equal_edge_segment_slots,
    _evaluate_table,
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
        for actual, expected in zip(
            _equal_edge_segment_slots(3.3, 2), [-0.825, 0.825]
        ):
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
        self.assertEqual([item["target_position_m"] for item in slots], [-1.1, 0.0, 1.1])
