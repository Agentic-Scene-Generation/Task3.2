from scenesmith.scenebenchmark_critic.metrics.functional_dependency.extensions.intent_contract import (
    evaluate_intent_contract_extensions,
)


def _bbox(z_top: float = 0.8) -> dict:
    return {
        "min": [-0.5, -0.5, 0.0],
        "max": [0.5, 0.5, z_top],
        "center": [0.0, 0.0, z_top / 2.0],
        "size": [1.0, 1.0, z_top],
    }


def _case(target: dict) -> dict:
    return {
        "stage": "furniture",
        "scene_geometry": {"objects": [target]},
        "intent_contract": {
            "constraints": [
                {
                    "constraint_id": "intent_support",
                    "relation": "on_top_of",
                    "subjects": {
                        "category": "blanket",
                        "count": 1,
                        "stage": "manipuland",
                    },
                    "targets": {"category": "armchair", "count": 1},
                    "source": "explicit_prompt",
                    "strength": "hard",
                    "stage": "manipuland",
                    "evidence_span": "a throw blanket on the armchair",
                }
            ]
        },
    }


def test_support_readiness_fails_present_target_without_verified_surface() -> None:
    target = {
        "id": "armchair_0",
        "category": "armchair",
        "object_type": "furniture",
        "bbox_world": _bbox(),
        "support_surfaces": [],
    }

    results = evaluate_intent_contract_extensions(_case(target))

    assert [(row["relation_type"], row["label"]) for row in results] == [
        ("support_readiness", "fail")
    ]
    assert results[0]["primary_object"] == "armchair_0"


def test_support_readiness_passes_target_with_declared_surface() -> None:
    target = {
        "id": "desk_0",
        "category": "desk",
        "object_type": "furniture",
        "bbox_world": _bbox(0.75),
        "support_regions": [
            {
                "region_id": "desk_top",
                "support_kind": "top_surface",
                "polygon_xy": [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]],
                "height_world_z": 0.75,
            }
        ],
    }
    case_pack = _case(target)
    case_pack["intent_contract"]["constraints"][0]["targets"] = {
        "category": "desk",
        "count": 1,
    }

    results = evaluate_intent_contract_extensions(case_pack)

    assert results[0]["relation_type"] == "support_readiness"
    assert results[0]["label"] == "pass"


def test_support_readiness_does_not_replace_present_subject_evaluation() -> None:
    target = {
        "id": "desk_0",
        "category": "desk",
        "object_type": "furniture",
        "bbox_world": _bbox(0.75),
        "support_regions": [],
    }
    subject = {
        "id": "blanket_0",
        "category": "blanket",
        "object_type": "manipuland",
        "bbox_world": _bbox(0.8),
    }
    case_pack = _case(target)
    case_pack["scene_geometry"]["objects"].append(subject)
    case_pack["intent_contract"]["constraints"][0]["targets"] = {
        "category": "desk",
        "count": 1,
    }

    results = evaluate_intent_contract_extensions(case_pack)

    assert not any(row["relation_type"] == "support_readiness" for row in results)
