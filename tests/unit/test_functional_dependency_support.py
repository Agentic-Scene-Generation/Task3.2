from scenesmith.scenebenchmark_critic.metrics.functional_dependency.support import (
    evaluate_support_relation,
)


def _object(
    object_id: str,
    category: str,
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> dict:
    lower, upper = bounds
    return {
        "id": object_id,
        "category": category,
        "category_norm": category,
        "object_type": "furniture",
        "bbox_world": {
            "min": list(lower),
            "max": list(upper),
            "center": [(low + high) / 2.0 for low, high in zip(lower, upper)],
            "size": [high - low for low, high in zip(lower, upper)],
        },
    }


def test_floor_furniture_on_rug_uses_footprint_coverage() -> None:
    rug = _object("rug_0", "rug", ((-2.0, -1.4, 0.0), (2.0, 1.4, 0.006)))
    chair = _object(
        "chair_0",
        "sofa_chair",
        ((-0.42, 0.63, 0.0), (0.42, 1.43, 0.84)),
    )

    result = evaluate_support_relation(chair, rug, "object_on_support")

    assert result.label == "pass"
    assert result.evaluation_path == "floor_covering_footprint"


def test_floor_furniture_outside_rug_fails_footprint_coverage() -> None:
    rug = _object("rug_0", "rug", ((-2.0, -1.4, 0.0), (2.0, 1.4, 0.006)))
    chair = _object(
        "chair_0",
        "sofa_chair",
        ((2.1, 0.6, 0.0), (2.9, 1.4, 0.84)),
    )

    result = evaluate_support_relation(chair, rug, "object_on_support")

    assert result.label == "fail"
    assert result.evaluation_path == "floor_covering_footprint"
