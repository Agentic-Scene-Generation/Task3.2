from __future__ import annotations

from copy import deepcopy

from scenesmith.scenebenchmark_critic.core.geometry import load_geometry
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.builder import (
    build_functional_dependency_checks,
)
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.relations import (
    evaluate_functional_dependency,
)
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.seat_surface_assignment import (
    ASSIGNMENT_SOURCE,
    assign_work_seats_to_surfaces,
)


def _object(
    object_id: str,
    category: str,
    center: tuple[float, float],
    size: tuple[float, float],
    *,
    yaw: float = 0.0,
    target_relations: list[str] | None = None,
) -> dict:
    is_seating = category in {"chair", "office_chair"} or category.endswith("_chair")
    is_surface = category in {"desk", "table"} or category.endswith("_desk")
    cx, cy = center
    sx, sy = size
    hints = {
        "category_group": "seating" if is_seating else "work_surface",
        "functional_categories": ["sittable" if is_seating else "supportable"],
        "scene_object_type": "furniture",
        "front_hint": "front",
    }
    if target_relations:
        hints["explicit_target_relation"] = target_relations
    return {
        "id": object_id,
        "name": object_id,
        "category": category,
        "category_norm": category,
        "yaw_deg": yaw,
        "bbox_world": {
            "center": [cx, cy, 0.5],
            "size": [sx, sy, 1.0],
            "min": [cx - sx / 2.0, cy - sy / 2.0, 0.0],
            "max": [cx + sx / 2.0, cy + sy / 2.0, 1.0],
        },
        "functional_hints": hints,
        "object_function_profile": {
            "can_support_top": is_surface,
            "has_internal_shelf": False,
            "is_small_placeable": False,
            "is_seating": is_seating,
            "is_work_surface": is_surface,
            "is_media_target": False,
            "is_bedside_surface": False,
            "is_sleeping_surface": False,
        },
    }


def _case(
    objects: list[dict], task: str = "A classroom with work desks and chairs"
) -> dict:
    return {
        "task_instruction": task,
        "room_type": "classroom" if "classroom" in task else "office",
        "scene_geometry": {"objects": objects, "relations": []},
        "checks": [],
    }


def test_repeated_mislabeled_surfaces_still_pair_one_to_one() -> None:
    objects = [
        _object("teacher_desk_0", "desk", (-2.0, 0.0), (1.2, 0.7)),
        _object("teacher_desk_1", "desk", (2.0, 0.0), (1.2, 0.7)),
        _object("work_chair_a", "chair", (-4.0, -2.0), (0.5, 0.5)),
        _object("work_chair_b", "chair", (-4.0, 2.0), (0.5, 0.5)),
    ]

    assignments = assign_work_seats_to_surfaces(
        objects, task_instruction="A classroom with two desks and chairs"
    )

    assert len(assignments) == 2
    assert {item.surface_id for item in assignments} == {
        "teacher_desk_0",
        "teacher_desk_1",
    }


def test_singleton_teacher_surface_does_not_take_repeated_student_cohort_seat() -> None:
    objects = [
        _object("teacher_desk_0", "desk", (0.0, 4.0), (1.6, 0.8)),
        _object("learner_table_0", "desk", (-1.0, 0.0), (1.2, 0.7)),
        _object("learner_table_1", "desk", (1.0, 0.0), (1.2, 0.7)),
        _object("task_chair_0", "chair", (-3.0, -2.0), (0.5, 0.5)),
        _object("task_chair_1", "chair", (3.0, -2.0), (0.5, 0.5)),
    ]

    assignments = assign_work_seats_to_surfaces(
        objects, task_instruction="A classroom with student work areas"
    )

    assert len(assignments) == 2
    assert {item.surface_id for item in assignments} == {
        "learner_table_0",
        "learner_table_1",
    }


def test_generic_classroom_family_excludes_front_center_teacher_desk() -> None:
    desk_positions = [
        (-3.04, -3.8),
        (-3.04, -2.0),
        (-3.04, 0.0),
        (-3.04, 2.0),
        (-1.6, -3.8),
        (-1.6, -2.0),
        (-1.6, 0.0),
    ]
    chair_positions = [
        (-1.6, 2.0),
        (-1.6, 3.8),
        (0.0, -3.8),
        (0.0, -2.0),
        (0.0, 0.0),
        (0.0, 2.0),
    ]
    objects = [
        *[
            _object(f"desk_{index}", "desk", center, (1.13, 0.6))
            for index, center in enumerate(desk_positions)
        ],
        *[
            _object(f"chair_{index}", "chair", center, (0.51, 0.5))
            for index, center in enumerate(chair_positions)
        ],
    ]

    assignments = assign_work_seats_to_surfaces(
        objects,
        task_instruction=(
            "A classroom with six student desks, each with a chair, and a "
            "teacher's desk at the front."
        ),
        room_type="classroom",
        room_bounds=(-4.0, -5.0, 4.0, 5.0),
    )

    assert len(assignments) == 6
    assert "desk_2" not in {assignment.surface_id for assignment in assignments}
    assert "desk_6" in {assignment.surface_id for assignment in assignments}


def test_classroom_roles_never_use_teacher_chair_as_missing_student_seat() -> None:
    student_desks = [
        _object(
            f"student_desk_{index}",
            "student_desk",
            (-2.0 + (index % 3) * 2.0, 1.0 - (index // 3) * 2.0),
            (1.1, 0.6),
        )
        for index in range(6)
    ]
    student_chairs = [
        _object(
            f"student_chair_{index}",
            "student_chair",
            (-2.0 + (index % 3) * 2.0, 0.3 - (index // 3) * 2.0),
            (0.5, 0.5),
        )
        for index in range(5)
    ]
    objects = [
        _object("teacher_desk_0", "teacher_desk", (0.0, 4.0), (1.5, 0.8)),
        _object("teacher_chair_0", "teacher_chair", (2.0, -2.0), (0.6, 0.6)),
        *student_desks,
        *student_chairs,
    ]
    task = (
        "A classroom with six student desks, each with a chair. A teacher's desk "
        "sits at the front."
    )

    assignments = assign_work_seats_to_surfaces(
        objects,
        task_instruction=task,
        room_type="classroom",
        room_bounds=(-4.0, -5.0, 4.0, 5.0),
    )
    pairs = {item.seat_id: item.surface_id for item in assignments}

    assert pairs["teacher_chair_0"] == "teacher_desk_0"
    assert all(
        pairs[f"student_chair_{index}"].startswith("student_desk_")
        for index in range(5)
    )
    assert len(set(pairs.values()) & {desk["id"] for desk in student_desks}) == 5

    objects.append(
        _object(
            "student_chair_5",
            "student_chair",
            (2.0, -2.7),
            (0.5, 0.5),
        )
    )
    complete = assign_work_seats_to_surfaces(
        objects,
        task_instruction=task,
        room_type="classroom",
        room_bounds=(-4.0, -5.0, 4.0, 5.0),
    )
    complete_pairs = {item.seat_id: item.surface_id for item in complete}

    assert len(complete_pairs) == 7
    assert complete_pairs["teacher_chair_0"] == "teacher_desk_0"
    assert {complete_pairs[f"student_chair_{index}"] for index in range(6)} == {
        desk["id"] for desk in student_desks
    }


def test_plain_office_chair_uses_function_annotations_without_student_label() -> None:
    objects = [
        _object("writing_surface", "desk", (0.0, 0.0), (1.4, 0.8)),
        _object(
            "ergonomic_seat",
            "office_chair",
            (0.0, -1.0),
            (0.6, 0.6),
            target_relations=["desk"],
        ),
    ]

    assignments = assign_work_seats_to_surfaces(
        objects, task_instruction="A quiet office workstation", room_type="office"
    )

    assert [(item.seat_id, item.surface_id) for item in assignments] == [
        ("ergonomic_seat", "writing_surface")
    ]


def test_guest_chair_is_not_bound_when_office_chair_owns_only_desk() -> None:
    objects = [
        _object("desk_alpha", "desk", (0.0, 0.0), (1.4, 0.8)),
        _object("office_seat", "office_chair", (0.0, -1.0), (0.6, 0.6)),
        _object("visitor_seat", "chair", (4.0, 4.0), (0.6, 0.6)),
    ]

    assignments = assign_work_seats_to_surfaces(
        objects, task_instruction="An office with one desk", room_type="office"
    )

    assert [item.seat_id for item in assignments] == ["office_seat"]


def test_fixed_pairs_remain_stable_after_geometry_crosses() -> None:
    objects = [
        _object("desk_left", "desk", (-2.0, 0.0), (1.2, 0.7)),
        _object("desk_right", "desk", (2.0, 0.0), (1.2, 0.7)),
        _object("work_chair_left", "chair", (-2.0, -1.0), (0.5, 0.5)),
        _object("work_chair_right", "chair", (2.0, -1.0), (0.5, 0.5)),
    ]
    initial = assign_work_seats_to_surfaces(
        objects, task_instruction="A classroom with two work desks"
    )
    fixed = {item.seat_id: item.surface_id for item in initial}
    moved = deepcopy(objects)
    moved[2]["bbox_world"]["center"][:2] = [2.0, -1.0]
    moved[3]["bbox_world"]["center"][:2] = [-2.0, -1.0]

    reassigned = assign_work_seats_to_surfaces(
        moved,
        task_instruction="A classroom with two work desks",
        fixed_pairs=fixed,
    )

    assert {item.seat_id: item.surface_id for item in reassigned} == fixed


def test_desk_slot_side_comes_from_semantic_front_not_bad_chair_position() -> None:
    objects = [
        _object("desk_alpha", "desk", (0.0, 0.0), (1.2, 0.7), yaw=0.0),
        _object("office_seat", "office_chair", (0.0, 3.0), (0.5, 0.5), yaw=180.0),
    ]

    assignment = assign_work_seats_to_surfaces(
        objects, task_instruction="An office workstation", room_type="office"
    )[0]

    assert assignment.side == "back"
    assert assignment.target_center_xy[1] < 0.0
    assert assignment.target_yaw_deg == 0.0


def test_desk_slot_respects_room_boundary_when_front_side_is_outside() -> None:
    objects = [
        _object("desk_north", "desk", (0.0, 1.6), (1.2, 0.7), yaw=0.0),
        _object("office_seat", "office_chair", (0.0, 1.0), (0.5, 0.5), yaw=0.0),
        {
            "id": "wall_west",
            "category": "wall",
            "category_norm": "wall",
            "bbox_world": {
                "center": [-2.4, 0.0, 1.25],
                "size": [0.2, 4.8, 2.5],
            },
        },
        {
            "id": "wall_east",
            "category": "wall",
            "category_norm": "wall",
            "bbox_world": {
                "center": [2.4, 0.0, 1.25],
                "size": [0.2, 4.8, 2.5],
            },
        },
        {
            "id": "wall_south",
            "category": "wall",
            "category_norm": "wall",
            "bbox_world": {
                "center": [0.0, -2.4, 1.25],
                "size": [4.8, 0.2, 2.5],
            },
        },
        {
            "id": "wall_north",
            "category": "wall",
            "category_norm": "wall",
            "bbox_world": {
                "center": [0.0, 2.4, 1.25],
                "size": [4.8, 0.2, 2.5],
            },
        },
    ]

    assignment = assign_work_seats_to_surfaces(
        objects,
        task_instruction="An office workstation",
        room_type="office",
    )[0]

    assert assignment.side == "back"
    assert assignment.target_center_xy[1] < 1.6
    assert assignment.target_yaw_deg == 0.0


def test_assignment_check_fails_far_pair_and_passes_exact_slot() -> None:
    objects = [
        _object("desk_alpha", "desk", (0.0, 0.0), (1.2, 0.7)),
        _object("work_chair_alpha", "chair", (4.0, 4.0), (0.5, 0.5)),
    ]
    case = _case(objects)
    checks = build_functional_dependency_checks(case)
    check = next(
        item for item in checks if item.get("check_source") == ASSIGNMENT_SOURCE
    )
    case["checks"] = checks
    store = load_geometry(case)
    assert store is not None
    assert evaluate_functional_dependency(store, check)["label"] == "fail"

    slot = check["evidence"]["target_slot"]
    chair = objects[1]
    center = slot["center_xy"]
    size = chair["bbox_world"]["size"]
    chair["bbox_world"]["center"][:2] = center
    chair["bbox_world"]["min"][:2] = [
        center[0] - size[0] / 2.0,
        center[1] - size[1] / 2.0,
    ]
    chair["bbox_world"]["max"][:2] = [
        center[0] + size[0] / 2.0,
        center[1] + size[1] / 2.0,
    ]
    chair["yaw_deg"] = slot["yaw_deg"]

    store = load_geometry(case)
    assert store is not None
    assert evaluate_functional_dependency(store, check)["label"] == "pass"


def test_unmatched_work_seat_does_not_reuse_an_assigned_surface() -> None:
    objects = [
        _object("learner_desk_0", "desk", (-1.0, 0.0), (1.2, 0.7)),
        _object("learner_desk_1", "desk", (1.0, 0.0), (1.2, 0.7)),
        _object(
            "task_chair_0", "chair", (-2.0, -2.0), (0.5, 0.5), target_relations=["desk"]
        ),
        _object(
            "task_chair_1", "chair", (0.0, -2.0), (0.5, 0.5), target_relations=["desk"]
        ),
        _object(
            "task_chair_2", "chair", (2.0, -2.0), (0.5, 0.5), target_relations=["desk"]
        ),
    ]
    case = _case(objects)

    checks = build_functional_dependency_checks(case)
    seating_checks = [
        check
        for check in checks
        if check.get("relation_type") == "seating_to_work_surface"
    ]

    assert len(seating_checks) == 2
    assert len({check["target_ids"][0] for check in seating_checks}) == 2


def test_dining_task_chairs_are_not_work_seats() -> None:
    objects = [
        _object("table_0", "table", (0.0, 0.0), (1.4, 0.8)),
        _object(
            "task_chair_0",
            "chair",
            (0.0, -1.0),
            (0.5, 0.5),
            target_relations=["table"],
        ),
    ]
    case = _case(objects, task="A dining room with one table and one chair")
    case["room_type"] = "dining_room"

    checks = build_functional_dependency_checks(case)

    assert not any(
        check.get("relation_type") == "seating_to_work_surface" for check in checks
    )
