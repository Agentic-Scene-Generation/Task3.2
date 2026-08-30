from scenesmith.scenebenchmark_critic.evaluator import evaluate_case_pack


def _case_pack(evidence):
    return {
        "scene_geometry": {"objects": []},
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
