from scenesmith.scenebenchmark_critic.metrics.functional_dependency.extensions.intent_contract import (
    _evaluate_corner_of_room,
)


def _constraint():
    return {"constraint_id": "corner_sofa", "subjects": {"category": "sofa"}}


def _geometry():
    return {"rooms": [{"bbox": {"min": [-3.0, -2.0, 0.0], "max": [3.0, 2.0, 2.7]}}]}


def test_large_object_footprint_can_satisfy_corner_even_when_center_is_far() -> None:
    result = _evaluate_corner_of_room(
        _constraint(),
        _geometry(),
        [
            {
                "id": "sofa_0",
                "category_norm": "sofa",
                "footprint_world": [
                    [-3.0, -2.0],
                    [-0.6, -2.0],
                    [-0.6, -1.0],
                    [-3.0, -1.0],
                ],
                "bbox_world": {"center": [-1.8, -1.5, 0.5]},
            }
        ],
        "core",
    )

    assert result[0]["label"] == "pass"


def test_footprint_near_only_one_wall_does_not_satisfy_corner() -> None:
    result = _evaluate_corner_of_room(
        _constraint(),
        _geometry(),
        [
            {
                "id": "sofa_0",
                "category_norm": "sofa",
                "footprint_world": [
                    [-3.0, -0.4],
                    [-1.0, -0.4],
                    [-1.0, 0.6],
                    [-3.0, 0.6],
                ],
                "bbox_world": {"center": [-2.0, 0.1, 0.5]},
            }
        ],
        "core",
    )

    assert result[0]["label"] == "fail"
