from scenesmith.scenebenchmark_critic.metrics.functional_dependency.support import (
    evaluate_support_relation,
)
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.relations import (
    evaluate_functional_dependency,
)
from scenesmith.scenebenchmark_critic.core.geometry import load_geometry


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


def test_explicit_support_relation_uses_geometry_for_open_vocabulary_objects() -> None:
    plinth = _object(
        "display_plinth_0",
        "display_plinth",
        ((-0.6, -0.4, 0.0), (0.6, 0.4, 0.8)),
    )
    framed_photo = _object(
        "framed_photo_0",
        "framed_photo",
        ((-0.1, -0.08, 0.8), (0.1, 0.08, 1.0)),
    )
    framed_photo["object_type"] = "manipuland"
    case_pack = {"scene_geometry": {"objects": [plinth, framed_photo]}}
    check = {
        "check_id": "photo_on_plinth",
        "metric": "functional_dependency",
        "subject_id": "framed_photo_0",
        "target_ids": ["display_plinth_0"],
        "relation_type": "object_on_support",
    }

    result = evaluate_functional_dependency(load_geometry(case_pack), check)

    assert result["label"] == "pass"
    assert "target category is not compatible" not in result["reason"]
