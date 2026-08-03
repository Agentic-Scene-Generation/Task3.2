from __future__ import annotations

import math
import unittest

from scenesmith.scenebenchmark_critic.metrics.functional_dependency.extensions.table_seat import (
    _equal_edge_segment_slots,
    _evaluate_table,
    _requests_one_seat_per_edge,
    evaluate_table_seat_distribution,
)
from scenesmith.scenebenchmark_critic.intent_contract import (
    build_intent_contract,
    bound_ids,
)
from scenesmith.scenebenchmark_critic.config import CriticConfig
from scenesmith.scenebenchmark_critic.evaluator import run_case_pack_checks
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.extensions.intent_contract import (
    evaluate_intent_contract_extensions,
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


class TestTableSeatDistribution(unittest.TestCase):
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

    def test_conference_long_side_prompt_enforces_equal_groups_and_slots(self):
        table = _table()
        table["id"] = "conference_table"
        table["category"] = "conference_table"
        chairs = [
            _chair("front_left", -1.1, -0.95),
            _chair("front_center", 0.0, -0.95),
            _chair("front_right", 1.1, -0.95),
            _chair("back_left", -1.1, 0.95),
            _chair("back_center", 0.0, 0.95),
            _chair("back_right", 1.1, 0.95),
        ]
        prompt = (
            "A meeting room with one rectangular conference table and six office "
            "chairs. Arrange exactly three chairs evenly spaced along each of the "
            "table's two long sides."
        )
        result = evaluate_table_seat_distribution(
            {
                "task_instruction": prompt,
                "room_type": "meeting_room",
                "intent_contract": build_intent_contract(prompt),
                "scene_geometry": {"objects": [table, *chairs]},
            }
        )[0]

        self.assertEqual(result["label"], "pass")
        self.assertTrue(result["diagnostics"]["long_side_distribution"])
        self.assertEqual(result["diagnostics"]["long_edges"], ["back", "front"])
        self.assertEqual(
            [slot["target_position_m"] for slot in result["diagnostics"]["seat_slots"]],
            [-1.1, 0.0, 1.1, -1.1, 0.0, 1.1],
        )

    def test_conference_long_side_prompt_rejects_short_edge_or_uneven_groups(self):
        table = _table()
        table["id"] = "conference_table"
        table["category"] = "conference_table"
        chairs = [
            _chair("front_left", -1.1, -0.95),
            _chair("front_center", 0.0, -0.95),
            _chair("front_right", 1.1, -0.95),
            _chair("back_left", -1.1, 0.95),
            _chair("back_center", 0.0, 0.95),
            _chair("short_edge", 1.95, 0.0),
        ]
        prompt = (
            "A meeting room with one rectangular conference table and six office "
            "chairs. Arrange exactly three chairs evenly spaced along each of the "
            "table's two long sides."
        )
        result = evaluate_table_seat_distribution(
            {
                "task_instruction": prompt,
                "room_type": "meeting_room",
                "intent_contract": build_intent_contract(prompt),
                "scene_geometry": {"objects": [table, *chairs]},
            }
        )[0]

        self.assertEqual(result["label"], "fail")
        self.assertIn("only the two long table edges", result["reason"])
        self.assertIn("equal nonzero groups", result["reason"])

    def test_conference_prompt_can_require_a_single_occupied_long_side(self):
        table = _table()
        table["id"] = "conference_table"
        table["category"] = "conference_table"
        chairs = [
            _chair("left", -1.1, -0.95),
            _chair("center", 0.0, -0.95),
            _chair("right", 1.1, -0.95),
        ]
        prompt = (
            "A meeting room with one rectangular conference table and three office "
            "chairs. Arrange all three chairs evenly spaced along one long side of "
            "the table. Keep the opposite long side and both short sides free of chairs."
        )
        result = evaluate_table_seat_distribution(
            {
                "task_instruction": prompt,
                "room_type": "meeting_room",
                "intent_contract": build_intent_contract(prompt),
                "scene_geometry": {"objects": [table, *chairs]},
            }
        )[0]

        self.assertEqual(result["label"], "pass")
        self.assertTrue(result["diagnostics"]["single_long_side_distribution"])
        self.assertEqual(
            [slot["target_position_m"] for slot in result["diagnostics"]["seat_slots"]],
            [-1.1, 0.0, 1.1],
        )

    def test_conference_prompt_can_mix_equal_long_edges_and_one_short_edge_seat(self):
        table = _table()
        table["id"] = "conference_table"
        table["category"] = "conference_table"
        short_side_chair = _chair("short_side", -1.95, 0.0)
        short_side_chair["yaw_deg"] = -90.0
        chairs = [
            _chair("front_left", -1.1, -0.95),
            _chair("front_center", 0.0, -0.95),
            _chair("front_right", 1.1, -0.95),
            _chair("back_left", -1.1, 0.95),
            _chair("back_center", 0.0, 0.95),
            _chair("back_right", 1.1, 0.95),
            short_side_chair,
        ]
        for chair in chairs:
            chair["category"] = "office_chair"
        prompt = (
            "A meeting room with one rectangular conference table and seven office "
            "chairs. Arrange six office chairs in two equal groups of three, evenly "
            "spaced along the table's two long sides. Place one remaining office chair "
            "centered along one short side, facing the table. Keep the opposite short "
            "side free of chairs."
        )
        result = evaluate_table_seat_distribution(
            {
                "task_instruction": prompt,
                "room_type": "meeting_room",
                "intent_contract": build_intent_contract(prompt),
                "scene_geometry": {"objects": [table, *chairs]},
            }
        )[0]

        self.assertEqual(result["label"], "pass")
        self.assertTrue(result["diagnostics"]["long_side_distribution"])
        self.assertTrue(result["diagnostics"]["single_short_side_seat_distribution"])
        self.assertEqual(
            {slot["edge"] for slot in result["diagnostics"]["seat_slots"]},
            {"front", "back", "left"},
        )

    def test_mixed_edge_contract_bypasses_ambiguous_six_chair_subset_binding(self):
        """The table topology evaluator owns all seven chairs atomically.

        The compiler accurately records six chairs in the two long-side groups,
        but there is no stable ID selector for that subset once the seventh
        chair occupies a short side.  Generic binding must therefore not fail
        before the table-local evaluator can classify the seven poses.
        """
        table = _table()
        table.update({"id": "conference_table", "category": "conference_table"})
        short_side_chair = _chair("short_side", -1.95, 0.0)
        short_side_chair["yaw_deg"] = -90.0
        chairs = [
            _chair("front_left", -1.1, -0.95),
            _chair("front_center", 0.0, -0.95),
            _chair("front_right", 1.1, -0.95),
            _chair("back_left", -1.1, 0.95),
            _chair("back_center", 0.0, 0.95),
            _chair("back_right", 1.1, 0.95),
            short_side_chair,
        ]
        for chair in chairs:
            chair["category"] = "office_chair"
        prompt = (
            "A meeting room with one rectangular conference table and seven office "
            "chairs. Arrange six office chairs in two equal groups of three, evenly "
            "spaced along the table's two long sides. Place one remaining office chair "
            "centered along one short side, facing the table. Keep the opposite short "
            "side free of chairs."
        )
        case_pack = {
            "stage": "furniture",
            "task_instruction": prompt,
            "room_type": "meeting_room",
            "intent_contract": build_intent_contract(prompt),
            "scene_geometry": {"objects": [table, *chairs], "relations": []},
            "checks": [],
        }

        results = run_case_pack_checks(
            case_pack,
            CriticConfig(enabled=True, metrics=("functional_dependency",)),
        )

        table_result = next(
            result
            for result in results
            if result["relation_type"] == "table_seat_distribution"
        )
        self.assertEqual(table_result["label"], "pass")
        self.assertEqual(table_result["contract_state"], "passed")
        self.assertFalse(
            any(result["relation_type"] == "one_per_side" for result in results)
        )

    def test_mixed_edge_prompt_rejects_chairs_on_both_short_edges(self):
        table = _table()
        table["id"] = "conference_table"
        table["category"] = "conference_table"
        left_chair = _chair("short_left", -1.95, 0.0)
        left_chair["yaw_deg"] = -90.0
        right_chair = _chair("short_right", 1.95, 0.0)
        right_chair["yaw_deg"] = 90.0
        chairs = [
            _chair("front_left", -1.1, -0.95),
            _chair("front_center", 0.0, -0.95),
            _chair("front_right", 1.1, -0.95),
            _chair("back_left", -1.1, 0.95),
            _chair("back_center", 0.0, 0.95),
            _chair("back_right", 1.1, 0.95),
            left_chair,
            right_chair,
        ]
        prompt = (
            "A meeting room with one rectangular conference table and seven office "
            "chairs. Arrange six office chairs in two equal groups of three, evenly "
            "spaced along the table's two long sides. Place one remaining office chair "
            "centered along one short side, facing the table. Keep the opposite short "
            "side free of chairs."
        )
        result = evaluate_table_seat_distribution(
            {
                "task_instruction": prompt,
                "room_type": "meeting_room",
                "intent_contract": build_intent_contract(prompt),
                "scene_geometry": {"objects": [table, *chairs]},
            }
        )[0]

        self.assertEqual(result["label"], "fail")
        self.assertIn("exactly one short table edge", result["reason"])

    def test_mixed_edge_prompt_exposes_atomic_repair_slots_for_reversed_layout(self):
        table = _table()
        table["id"] = "conference_table"
        table["category"] = "conference_table"
        chairs = (
            [
                _chair(f"left_{index}", -1.95, position)
                for index, position in enumerate((-0.95, 0.0, 0.95))
            ]
            + [
                _chair(f"right_{index}", 1.95, position)
                for index, position in enumerate((-0.95, 0.0, 0.95))
            ]
            + [_chair("wrong_long_edge", 0.0, 0.95)]
        )
        prompt = (
            "A meeting room with one rectangular conference table and seven office "
            "chairs. Arrange six office chairs in two equal groups of three, evenly "
            "spaced along the table's two long sides. Place one remaining office chair "
            "centered along one short side, facing the table. Keep the opposite short "
            "side free of chairs."
        )
        result = evaluate_table_seat_distribution(
            {
                "task_instruction": prompt,
                "room_type": "meeting_room",
                "intent_contract": build_intent_contract(prompt),
                "scene_geometry": {"objects": [table, *chairs]},
            }
        )[0]

        self.assertEqual(result["label"], "fail")
        repair_slots = result["diagnostics"]["topology_repair_slots"]
        self.assertEqual(len(repair_slots), 7)
        repair_edges = [slot["edge"] for slot in repair_slots]
        self.assertEqual(set(repair_edges) & {"front", "back"}, {"front", "back"})
        self.assertEqual(sum(edge in {"left", "right"} for edge in repair_edges), 1)
        self.assertEqual(sum(edge == "front" for edge in repair_edges), 3)
        self.assertEqual(sum(edge == "back" for edge in repair_edges), 3)

    def test_conference_long_side_contract_ignores_auxiliary_table(self):
        prompt = (
            "A meeting room with one rectangular conference table and three office "
            "chairs. Arrange all three chairs evenly spaced along one long side of "
            "the table. Keep the opposite long side and both short sides free of chairs."
        )
        contract = build_intent_contract(prompt)
        constraint = next(
            item
            for item in contract["constraints"]
            if item["relation"] == "one_per_side"
        )
        objects = [
            {
                "id": "conference_table_0",
                "category": "table",
                "metadata": {"semantic_name": "conference_table"},
            },
            {
                "id": "auxiliary_table_0",
                "category": "table",
                "metadata": {"semantic_name": "auxiliary_table"},
            },
        ]

        self.assertEqual(constraint["targets"]["category"], "conference_table")
        self.assertEqual(
            bound_ids(constraint["targets"], objects), ["conference_table_0"]
        )

    def test_long_side_contract_defers_topology_to_table_seat_evaluator(self):
        prompt = (
            "A meeting room with one rectangular conference table and three office "
            "chairs. Arrange all three chairs evenly spaced along one long side of "
            "the table. Keep the opposite long side and both short sides free of chairs."
        )
        table = _table()
        table.update(
            {
                "id": "conference_table_0",
                "category": "table",
                "metadata": {"semantic_name": "conference_table"},
            }
        )
        auxiliary_table = {
            **_table(),
            "id": "auxiliary_table_0",
            "category": "table",
            "metadata": {"semantic_name": "auxiliary_table"},
            "bbox_world": _bbox((2.2, 1.8), (0.8, 0.6, 0.75)),
        }
        chairs = [
            _chair("office_chair_0", -1.1, -0.95),
            _chair("office_chair_1", 0.0, -0.95),
            _chair("office_chair_2", 1.1, -0.95),
        ]
        for chair in chairs:
            chair["category"] = "office_chair"
        case_pack = {
            "stage": "furniture",
            "task_instruction": prompt,
            "intent_contract": build_intent_contract(prompt),
            "scene_geometry": {"objects": [table, auxiliary_table, *chairs]},
        }

        table_result = evaluate_table_seat_distribution(case_pack)[0]
        contract_results = evaluate_intent_contract_extensions(case_pack)

        self.assertEqual(table_result["label"], "pass")
        self.assertEqual(table_result["primary_object"], "conference_table_0")
        self.assertFalse(
            any(
                result["relation_type"] == "one_per_side" for result in contract_results
            )
        )

    def test_generic_rectangular_table_long_side_prompt_does_not_require_dining(self):
        table = _table()
        table["id"] = "table_0"
        table["category"] = "table"
        chairs = [
            _chair("left", -1.1, -0.95),
            _chair("center", 0.0, -0.95),
            _chair("right", 1.1, -0.95),
        ]
        prompt = (
            "A training room with one rectangular table and three chairs. Arrange all "
            "three chairs evenly spaced along one long side of the table. Keep the "
            "opposite long side and both short sides clear of chairs."
        )
        result = evaluate_table_seat_distribution(
            {
                "task_instruction": prompt,
                "room_type": "training_room",
                "intent_contract": build_intent_contract(prompt),
                "scene_geometry": {"objects": [table, *chairs]},
            }
        )[0]

        self.assertEqual(result["label"], "pass")
        self.assertTrue(result["diagnostics"]["long_side_distribution"])
        self.assertTrue(result["diagnostics"]["single_long_side_distribution"])

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
        result = evaluate_table_seat_distribution(
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
        result = evaluate_table_seat_distribution(
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
        result = evaluate_table_seat_distribution(
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
