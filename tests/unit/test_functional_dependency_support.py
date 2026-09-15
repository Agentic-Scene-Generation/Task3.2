import json

import numpy as np

from pydrake.all import RigidTransform

from scenesmith.agent_utils.room import (
    ObjectType,
    RoomScene,
    SceneObject,
    SupportSurface,
    UniqueID,
)
from scenesmith.scenebenchmark_critic.adapter import room_scene_to_case_pack
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


def test_adapter_preserves_support_clearance_and_explicit_openness(tmp_path) -> None:
    surface = SupportSurface(
        surface_id=UniqueID("S_0"),
        bounding_box_min=np.array([-0.6, -0.3, 0.01]),
        bounding_box_max=np.array([0.6, 0.3, 0.21]),
        transform=RigidTransform(p=[0.0, 0.0, 0.5]),
        open_above=True,
    )
    stand = SceneObject(
        object_id=UniqueID("tv_stand_0"),
        object_type=ObjectType.FURNITURE,
        name="TV stand",
        description="A low media stand",
        transform=RigidTransform(),
        support_surfaces=[surface],
        metadata={"semantic_name": "tv_stand"},
        bbox_min=np.array([-0.6, -0.3, 0.0]),
        bbox_max=np.array([0.6, 0.3, 0.5]),
    )
    scene = RoomScene(
        room_geometry=None,
        scene_dir=tmp_path,
        objects={stand.object_id: stand},
    )

    case_pack = room_scene_to_case_pack(scene, metrics=())
    stand_geometry = case_pack["scene_geometry"]["objects"][0]
    region = stand_geometry["support_regions"][0]

    assert np.isclose(region["clearance_above_m"], 0.2)
    assert region["open_above"] is True
    json.dumps(region, allow_nan=False)


def _television_support_pair(*, open_above: bool | None) -> tuple[dict, dict]:
    stand = _object(
        "tv_stand_0",
        "tv_stand",
        ((-0.6, -0.3, 0.0), (0.6, 0.3, 0.5)),
    )
    region = {
        "region_id": "S_0",
        "support_kind": "top_surface",
        "polygon_world_xy": [
            [-0.6, -0.3],
            [0.6, -0.3],
            [0.6, 0.3],
            [-0.6, 0.3],
        ],
        "height_world_z": 0.5,
        "clearance_above_m": 0.2,
    }
    if open_above is not None:
        region["open_above"] = open_above
    stand["support_regions"] = [region]
    television = _object(
        "television_0",
        "television",
        ((-0.4, -0.1, 0.5), (0.4, 0.1, 1.5)),
    )
    television["object_type"] = "manipuland"
    return television, stand


def test_open_support_surface_accepts_tall_television() -> None:
    television, stand = _television_support_pair(open_above=True)

    result = evaluate_support_relation(television, stand, "object_on_support")

    assert result.label == "pass"
    assert result.evaluation_path == "support_region"
    assert result.evidence["support_open_above"] is True
    assert "open above" in result.reason


def test_bounded_support_surface_rejects_tall_television() -> None:
    television, stand = _television_support_pair(open_above=False)

    result = evaluate_support_relation(television, stand, "object_on_support")

    assert result.label == "degraded"
    assert result.evidence["support_open_above"] is False


def test_legacy_support_surface_without_openness_remains_bounded() -> None:
    television, stand = _television_support_pair(open_above=None)

    result = evaluate_support_relation(television, stand, "object_on_support")

    assert result.label == "degraded"
    assert result.evidence["support_open_above"] is False
