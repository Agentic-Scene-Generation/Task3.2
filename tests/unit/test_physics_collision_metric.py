from scenesmith.scenebenchmark_critic.evaluator import evaluate_case_pack


def _case_pack(evidence, *, objects=None):
    return {
        "scene_geometry": {"objects": list(objects or [])},
        "physics_evidence": evidence,
        "checks": [],
    }


def test_hard_physics_penetration_is_canonical_scoring_failure() -> None:
    result = evaluate_case_pack(
        _case_pack(
            {
                "available": True,
                "collisions": [
                    {
                        "object_a_id": "stool_0",
                        "object_b_id": "stool_1",
                        "penetration_depth_m": 0.1764,
                        "classification": "hard",
                    }
                ],
            }
        ),
        config={
            "scenebenchmark_critic": {"enabled": True, "metrics": ["physics_collision"]}
        },
    )

    assert result["results"][0]["label"] == "fail"
    assert result["summary"]["scene_summary"]["total_checks"] == 1
    assert result["summary"]["scene_summary"]["fail"] == 1


def test_support_contact_and_shallow_contact_do_not_fail_collision_metric() -> None:
    result = evaluate_case_pack(
        _case_pack(
            {
                "available": True,
                "penetration_tolerance_m": 0.01,
                "collisions": [
                    {
                        "object_a_id": "book_0",
                        "object_b_id": "shelf_0",
                        "penetration_depth_m": 0.02,
                        "classification": "expected_support",
                    },
                    {
                        "object_a_id": "pillow_0",
                        "object_b_id": "sofa_0",
                        "penetration_depth_m": 0.02,
                        "classification": "soft_furnishing",
                    },
                    {
                        "object_a_id": "tv_0",
                        "object_b_id": "stand_0",
                        "penetration_depth_m": 0.005,
                        "classification": "hard",
                    },
                ],
            }
        ),
        config={
            "scenebenchmark_critic": {"enabled": True, "metrics": ["physics_collision"]}
        },
    )

    assert result["results"][0]["label"] == "pass"


def test_unavailable_physics_is_unknown_not_pass() -> None:
    result = evaluate_case_pack(
        _case_pack({"available": False, "error": "validator unavailable"}),
        config={
            "scenebenchmark_critic": {"enabled": True, "metrics": ["physics_collision"]}
        },
    )

    assert result["results"][0]["label"] == "unknown"


def _collision_case(*, first, second):
    return _case_pack(
        {
            "available": True,
            "collisions": [
                {
                    "object_a_id": first["id"],
                    "object_b_id": second["id"],
                    "penetration_depth_m": 0.05,
                    "classification": "hard",
                }
            ],
        },
        objects=[first, second],
    )


def _evaluate_collision_owner(*, first, second):
    result = evaluate_case_pack(
        _collision_case(first=first, second=second),
        config={
            "scenebenchmark_critic": {
                "enabled": True,
                "metrics": ["physics_collision"],
            }
        },
    )
    return result["results"][0]["diagnostics"]


def test_furniture_collision_is_owned_by_furniture_stage() -> None:
    diagnostics = _evaluate_collision_owner(
        first={"id": "sofa_0", "category_norm": "sofa", "object_type": "furniture"},
        second={
            "id": "table_0",
            "category_norm": "coffee_table",
            "object_type": "furniture",
        },
    )

    assert diagnostics["earliest_stage"] == "furniture"
    assert diagnostics["endpoint_generation_owners"] == {
        "sofa_0": "furniture",
        "table_0": "furniture",
    }


def test_cross_stage_collision_is_owned_when_latest_movable_endpoint_exists() -> None:
    furniture = {
        "id": "sofa_0",
        "category_norm": "sofa",
        "object_type": "furniture",
    }
    wall_mounted = {
        "id": "mirror_0",
        "category_norm": "mirror",
        "object_type": "wall_mounted",
    }
    manipuland = {
        "id": "vase_0",
        "category_norm": "vase",
        "object_type": "manipuland",
    }

    wall_diagnostics = _evaluate_collision_owner(first=furniture, second=wall_mounted)
    manipuland_diagnostics = _evaluate_collision_owner(
        first=furniture, second=manipuland
    )

    assert wall_diagnostics["earliest_stage"] == "wall_mounted"
    assert manipuland_diagnostics["earliest_stage"] == "manipuland"


def test_structural_collision_anchor_does_not_become_repair_owner() -> None:
    diagnostics = _evaluate_collision_owner(
        first={
            "id": "cabinet_0",
            "category_norm": "storage_cabinet",
            "object_type": "furniture",
        },
        second={
            "id": "north_wall",
            "category_norm": "wall",
            "object_type": "structural",
        },
    )

    assert diagnostics["earliest_stage"] == "furniture"
    assert diagnostics["endpoint_generation_owners"] == {
        "cabinet_0": "furniture",
        "north_wall": "",
    }
