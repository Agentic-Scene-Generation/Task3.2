from pathlib import Path
from types import SimpleNamespace

import numpy as np

from pydrake.math import RigidTransform

from scenesmith.agent_utils.room import ObjectType, RoomScene, SceneObject, UniqueID
from scenesmith.scenebenchmark_critic.furniture_relation_repair import (
    _RepairHandlerContext,
    _activity_zone_repair_targets,
)
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.extensions.activity_zone_separation import (
    evaluate_activity_zone_separation,
)


def _geometry_object(
    object_id: str,
    category: str,
    center: tuple[float, float],
    size: tuple[float, float],
    *,
    object_type: str = "furniture",
) -> dict:
    cx, cy = center
    sx, sy = size
    return {
        "id": object_id,
        "category": category,
        "category_norm": category,
        "object_type": object_type,
        "bbox_world": {
            "center": [cx, cy, 0.5],
            "size": [sx, sy, 1.0],
            "min": [cx - sx / 2.0, cy - sy / 2.0, 0.0],
            "max": [cx + sx / 2.0, cy + sy / 2.0, 1.0],
        },
        "footprint_world": [
            [cx - sx / 2.0, cy - sy / 2.0],
            [cx + sx / 2.0, cy - sy / 2.0],
            [cx + sx / 2.0, cy + sy / 2.0],
            [cx - sx / 2.0, cy + sy / 2.0],
        ],
    }


def _case_pack(*, dining_x: float, include_dining_relation: bool = True) -> dict:
    objects = [
        _geometry_object("sofa_0", "sofa", (0.0, -2.0), (2.0, 0.8)),
        _geometry_object(
            "television_0",
            "television",
            (0.0, 2.0),
            (1.4, 0.1),
            object_type="wall_mounted",
        ),
        _geometry_object("dining_table_0", "dining_table", (dining_x, 0.0), (1.2, 0.8)),
        _geometry_object(
            "dining_chair_0", "dining_chair", (dining_x, -0.75), (0.5, 0.5)
        ),
        _geometry_object(
            "dining_chair_1", "dining_chair", (dining_x, 0.75), (0.5, 0.5)
        ),
    ]
    checks = [
        {
            "check_id": "media",
            "metric": "functional_dependency",
            "subject_id": "sofa_0",
            "target_ids": ["television_0"],
            "relation_type": "seating_to_media",
            "scoring_tier": "core",
        }
    ]
    if include_dining_relation:
        checks.append(
            {
                "check_id": "dining",
                "metric": "functional_dependency",
                "subject_id": "dining_table_0",
                "target_ids": ["dining_chair_0", "dining_chair_1"],
                "relation_type": "edge_distribution",
                "scoring_tier": "core",
            }
        )
    return {
        "scene_geometry": {"objects": objects, "rooms": []},
        "checks": checks,
    }


def test_dining_group_between_sofa_and_media_fails_view_corridor() -> None:
    results = evaluate_activity_zone_separation(_case_pack(dining_x=0.0))

    assert len(results) == 1
    assert results[0]["label"] == "fail"
    assert results[0]["relation_type"] == "activity_zone_separation"
    assert results[0]["diagnostics"]["intersection_area_m2"] > 0.0


def test_dining_group_beside_media_axis_passes() -> None:
    results = evaluate_activity_zone_separation(_case_pack(dining_x=2.0))

    assert len(results) == 1
    assert results[0]["label"] == "pass"


def test_single_functional_group_does_not_enable_zone_rule() -> None:
    results = evaluate_activity_zone_separation(
        _case_pack(dining_x=0.0, include_dining_relation=False)
    )

    assert results == []


def test_hard_contract_media_axis_does_not_depend_on_prior_check_order() -> None:
    case_pack = _case_pack(dining_x=0.0)
    case_pack["checks"] = [
        check
        for check in case_pack["checks"]
        if check["relation_type"] != "seating_to_media"
    ]
    case_pack["intent_contract"] = {
        "constraints": [
            {
                "relation": "faces",
                "stage": "furniture",
                "strength": "hard",
                "subjects": {"category": "sofa", "count": 1},
                "targets": {"category": "television", "count": 1},
            }
        ]
    }

    results = evaluate_activity_zone_separation(case_pack)

    assert len(results) == 1
    assert results[0]["label"] == "fail"
    assert results[0]["diagnostics"]["media_object_id"] == "television_0"


def test_future_media_can_use_explicit_present_support_as_axis_proxy() -> None:
    case_pack = _case_pack(dining_x=0.0)
    objects = case_pack["scene_geometry"]["objects"]
    objects[:] = [obj for obj in objects if obj["id"] != "television_0"]
    objects.append(_geometry_object("tv_stand_0", "tv_stand", (0.0, 2.0), (1.4, 0.5)))
    case_pack["checks"] = [
        check
        for check in case_pack["checks"]
        if check["relation_type"] != "seating_to_media"
    ]
    case_pack["intent_contract"] = {
        "constraints": [
            {
                "relation": "faces",
                "stage": "wall_mounted",
                "strength": "hard",
                "subjects": {"category": "sofa", "count": 1},
                "targets": {"category": "television", "count": 1},
            },
            {
                "relation": "on_top_of",
                "stage": "wall_mounted",
                "strength": "hard",
                "subjects": {"category": "television", "count": 1},
                "targets": {"category": "tv_stand", "count": 1},
            },
        ]
    }

    results = evaluate_activity_zone_separation(case_pack)

    assert len(results) == 1
    diagnostics = results[0]["diagnostics"]
    assert diagnostics["media_object_id"] == "tv_stand_0"
    assert diagnostics["media_proxy_for_category"] == "television"


def _scene_object(object_id: str, center: tuple[float, float]) -> SceneObject:
    return SceneObject(
        object_id=UniqueID(object_id),
        object_type=ObjectType.FURNITURE,
        name=object_id,
        description=object_id,
        transform=RigidTransform(p=[center[0], center[1], 0.5]),
        bbox_min=np.array([-0.25, -0.25, -0.5]),
        bbox_max=np.array([0.25, 0.25, 0.5]),
    )


def test_zone_repair_targets_rigidly_translate_every_dining_member() -> None:
    objects = [
        _scene_object("dining_table_0", (0.0, 0.0)),
        _scene_object("dining_chair_0", (0.0, -0.8)),
        _scene_object("dining_chair_1", (0.0, 0.8)),
    ]
    scene = RoomScene(
        room_geometry=SimpleNamespace(length=6.0, width=5.0),
        scene_dir=Path("."),
        objects={obj.object_id: obj for obj in objects},
    )
    context = _RepairHandlerContext(
        scene=scene,
        payload={},
        result={"primary_object": "dining_table_0"},
        check_id="activity_zone",
        relation="activity_zone_separation",
        diagnostics={
            "group_object_ids": [
                "dining_table_0",
                "dining_chair_0",
                "dining_chair_1",
            ],
            "view_normal_unit": [1.0, 0.0],
        },
        coordinated_front_checks=set(),
        claimed_near_checks=set(),
    )

    targets = _activity_zone_repair_targets(context)

    assert targets
    first = targets[0]
    deltas = []
    for pose in first.member_poses:
        original = scene.objects[UniqueID(pose.object_id)].transform.translation()[:2]
        deltas.append(np.asarray(pose.target_center_xy) - original)
    for delta in deltas[1:]:
        np.testing.assert_allclose(delta, deltas[0])
    np.testing.assert_allclose(deltas[0], [-0.25, 0.0])
