from scenesmith.scenebenchmark_critic.metrics.spatial_accessibility.companions import (
    attach_expected_clearance_companions,
)


def _obj(object_id: str, category: str, object_type: str = "furniture") -> dict:
    return {
        "id": object_id,
        "category": category,
        "category_norm": category,
        "object_type": object_type,
        "bbox_world": {
            "min": [-0.4, -0.4, 0.0],
            "max": [0.4, 0.4, 0.8],
            "center": [0.0, 0.0, 0.4],
            "size": [0.8, 0.8, 0.8],
        },
    }


def test_bound_seat_is_removed_only_from_anchor_clearance() -> None:
    objects = {
        "chair_0": _obj("chair_0", "dining_chair"),
        "table_0": _obj("table_0", "dining_table"),
    }
    case_pack = {
        "checks": [
            {
                "check_id": "clearance__table_0",
                "metric": "interaction_clearance",
                "subject_id": "table_0",
                "clearance_result": {
                    "label": "fail",
                    "blocking_objects": ["chair_0", "other_0"],
                    "intrusions": [
                        {"object_id": "chair_0"},
                        {"object_id": "other_0"},
                    ],
                },
            },
            {
                "check_id": "clearance__chair_0",
                "metric": "interaction_clearance",
                "subject_id": "chair_0",
                "clearance_result": {
                    "label": "fail",
                    "blocking_objects": ["table_0"],
                    "intrusions": [{"object_id": "table_0"}],
                },
            },
        ],
        "intent_contract": {
            "constraints": [
                {
                    "constraint_id": "pair_0",
                    "relation": "paired_with",
                    "subjects": {"category": "dining_chair", "count": 1},
                    "targets": {"category": "dining_table", "count": 1},
                    "source": "explicit_prompt",
                    "strength": "hard",
                }
            ]
        },
    }

    attach_expected_clearance_companions(case_pack, objects)

    table_clearance = case_pack["checks"][0]["clearance_result"]
    chair_clearance = case_pack["checks"][1]["clearance_result"]
    assert table_clearance["label"] == "fail"
    assert table_clearance["blocking_objects"] == ["other_0"]
    assert chair_clearance["blocking_objects"] == ["table_0"]


def test_surround_seats_are_removed_only_from_anchor_clearance() -> None:
    objects = {
        "chair_0": _obj("chair_0", "dining_chair"),
        "chair_1": _obj("chair_1", "dining_chair"),
        "table_0": _obj("table_0", "dining_table"),
    }
    case_pack = {
        "checks": [
            {
                "check_id": "clearance__table_0",
                "metric": "interaction_clearance",
                "subject_id": "table_0",
                "clearance_result": {
                    "label": "fail",
                    "blocking_objects": ["chair_0", "chair_1", "other_0"],
                    "intrusions": [
                        {"object_id": "chair_0"},
                        {"object_id": "chair_1"},
                        {"object_id": "other_0"},
                    ],
                },
            }
        ],
        "intent_contract": {
            "constraints": [
                {
                    "constraint_id": "surround_0",
                    "relation": "surround",
                    "subjects": {"category": "dining_chair", "count": 2},
                    "targets": {"category": "dining_table", "count": 1},
                    "source": "explicit_prompt",
                    "strength": "hard",
                }
            ]
        },
    }

    attach_expected_clearance_companions(case_pack, objects)

    clearance = case_pack["checks"][0]["clearance_result"]
    assert clearance["label"] == "fail"
    assert clearance["blocking_objects"] == ["other_0"]


def test_rug_and_supported_small_object_are_ignored_without_direct_access_contract() -> (
    None
):
    rug = _obj("rug_0", "rug")
    book = _obj("book_0", "book", "manipuland")
    book["bbox_world"]["min"] = [-0.1, -0.1, 0.7]
    book["bbox_world"]["max"] = [0.1, 0.1, 0.9]
    book["bbox_world"]["center"] = [0.0, 0.0, 0.8]
    book["bbox_world"]["size"] = [0.2, 0.2, 0.2]
    book["placement_info"] = {"parent_surface_id": "desk_surface_0"}
    objects = {"rug_0": rug, "book_0": book}
    case_pack = {
        "checks": [
            {
                "check_id": "clearance__rug_0",
                "metric": "interaction_clearance",
                "subject_id": "rug_0",
                "clearance_result": {
                    "label": "fail",
                    "blocking_objects": ["chair_0"],
                    "intrusions": [{"object_id": "chair_0"}],
                },
            },
            {
                "check_id": "clearance__book_0",
                "metric": "interaction_clearance",
                "subject_id": "book_0",
                "clearance_result": {
                    "label": "fail",
                    "blocking_objects": ["chair_0"],
                    "intrusions": [{"object_id": "chair_0"}],
                },
            },
        ],
        "intent_contract": {"constraints": []},
    }

    attach_expected_clearance_companions(case_pack, objects)

    assert [check["scoring_tier"] for check in case_pack["checks"]] == [
        "ignored",
        "ignored",
    ]
    assert all(
        check["clearance_result"]["label"] == "pass" for check in case_pack["checks"]
    )


def test_explicit_independent_access_keeps_supported_small_object_check() -> None:
    book = _obj("book_0", "book", "manipuland")
    book["placement_info"] = {"parent_surface_id": "desk_surface_0"}
    book["functional_hints"] = {"independent_access_required": True}
    objects = {"book_0": book}
    case_pack = {
        "checks": [
            {
                "check_id": "clearance__book_0",
                "metric": "interaction_clearance",
                "subject_id": "book_0",
                "clearance_result": {
                    "label": "fail",
                    "blocking_objects": ["chair_0"],
                    "intrusions": [{"object_id": "chair_0"}],
                },
            }
        ],
        "intent_contract": {"constraints": []},
    }

    attach_expected_clearance_companions(case_pack, objects)

    check = case_pack["checks"][0]
    assert check.get("scoring_tier", "core") != "ignored"
    assert check["clearance_result"]["label"] == "fail"
