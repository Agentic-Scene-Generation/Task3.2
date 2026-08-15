from scenesmith.scenebenchmark_critic.asset_library_annotations import (
    build_scenebenchmark_annotation,
    get_hssd_asset_annotations,
)
from scenesmith.scenebenchmark_critic.core.geometry import load_geometry
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.builder import (
    build_functional_dependency_checks,
)
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.relations import (
    evaluate_functional_dependency,
)


def test_bundled_hssd_physics_and_quality_record():
    record = get_hssd_asset_annotations("0001fb06b075a743e6289236cf049df3ad5dfa9c")
    assert record is not None
    physics = record["asset_physics"]
    quality = record["asset_quality"]
    assert physics["material"] == "steel"
    assert (
        physics["mass_range_kg"][0] <= physics["mass_kg"] <= physics["mass_range_kg"][1]
    )
    assert 0.0 <= physics["friction_coefficient"] <= 2.0
    assert quality["is_acceptable"] is True
    assert quality["watertight"] is False
    assert record["scenebenchmark_fd_sa"]


def _object(
    object_id: str, category: str, y: float, yaw_deg: float
) -> dict[str, object]:
    return {
        "id": object_id,
        "category": category,
        "category_norm": category,
        "name": category,
        "yaw_deg": yaw_deg,
        "bbox_world": {
            "center": [0.0, y, 0.4],
            "size": [1.2, 0.5, 0.8],
            "min": [-0.6, y - 0.25, 0.0],
            "max": [0.6, y + 0.25, 0.8],
        },
    }


def _hssd_front_faces_annotation(distance_range_m: object = None) -> dict:
    relation = {
        "target_kind": "asset_category",
        "target_category": "bed",
        "relation_type": "faces",
        "relative_facing": "front_faces_target",
    }
    if distance_range_m is not None:
        relation["distance_range_m"] = distance_range_m
    return build_scenebenchmark_annotation({"relation_priors": [relation]})


def test_hssd_front_faces_without_distance_is_orientation_only_across_room():
    annotation = _hssd_front_faces_annotation()
    dependency = annotation["functional_hints"]["orientation_dependencies"][0]

    assert dependency["distance_required"] is False
    assert "max_distance_m" not in dependency

    dresser = _object("dresser_0", "dresser", -3.0, 0.0)
    dresser["functional_hints"] = annotation["functional_hints"]
    bed = _object("bed_0", "bed", 3.0, 180.0)
    case_pack = {"scene_geometry": {"objects": [dresser, bed]}}
    checks = build_functional_dependency_checks(case_pack)

    assert len(checks) == 1
    result = evaluate_functional_dependency(load_geometry(case_pack), checks[0])

    assert result["label"] == "pass"
    assert "gap" not in result["reason"]

    # Replays can restore annotations created before ``distance_required`` was
    # serialized.  The legacy HSSD provenance must retain the same semantics.
    legacy_dependency = {
        key: value for key, value in dependency.items() if key != "distance_required"
    }
    legacy_dependency["max_distance_m"] = None
    dresser["functional_hints"] = {"orientation_dependencies": [legacy_dependency]}
    legacy_checks = build_functional_dependency_checks(case_pack)
    legacy_result = evaluate_functional_dependency(
        load_geometry(case_pack), legacy_checks[0]
    )

    assert legacy_result["label"] == "pass"
    assert "gap" not in legacy_result["reason"]


def test_hssd_front_faces_with_explicit_distance_still_fails_when_far():
    annotation = _hssd_front_faces_annotation([0.2, 1.8])
    dependency = annotation["functional_hints"]["orientation_dependencies"][0]

    assert dependency["distance_required"] is True
    assert dependency["max_distance_m"] == 1.8

    dresser = _object("dresser_0", "dresser", -3.0, 0.0)
    bed = _object("bed_0", "bed", 3.0, 180.0)
    case_pack = {"scene_geometry": {"objects": [dresser, bed]}}
    check = {
        "check_id": "dresser_faces_bed",
        "subject_id": "dresser_0",
        "target_ids": ["bed_0"],
        "relation_type": "furniture_faces_furniture",
        "evidence": {"dependency": dependency},
    }

    result = evaluate_functional_dependency(load_geometry(case_pack), check)

    assert result["label"] == "fail"
    assert "gap" in result["reason"]


def test_oriented_edge_distribution_owns_seat_facing_over_hssd_priors():
    chair = _object("dining_chair_0", "dining_chair", 1.5, 180.0)
    chair["functional_hints"] = {
        "orientation_dependencies": [
            {
                "target_kind": "asset_category",
                "target_category": "dining_table",
                "relation_type": "front_faces",
            },
            {
                "target_kind": "asset_category",
                "target_category": "coffee_table",
                "relation_type": "front_faces",
                "distance_required": False,
            },
        ]
    }
    dining_table = _object("dining_table_0", "dining_table", 0.0, 0.0)
    coffee_table = _object("coffee_table_0", "coffee_table", 4.0, 0.0)
    case_pack = {
        "scene_geometry": {"objects": [chair, dining_table, coffee_table]},
        "intent_contract": {
            "constraints": [
                {
                    "relation": "edge_distribution",
                    "subjects": {
                        "category": "dining_chair",
                        "count": 1,
                        "quantifier": "exactly",
                    },
                    "targets": {
                        "category": "dining_table",
                        "count": 1,
                        "quantifier": "exactly",
                    },
                    "orientation": "toward_target",
                    "source": "explicit_prompt",
                }
            ]
        },
    }

    checks = build_functional_dependency_checks(case_pack)

    assert not [
        check
        for check in checks
        if check.get("subject_id") == "dining_chair_0"
        and check.get("relation_type") == "furniture_faces_furniture"
    ]


def test_unoriented_edge_distribution_keeps_hssd_facing_priors():
    chair = _object("dining_chair_0", "dining_chair", 1.5, 180.0)
    chair["functional_hints"] = {
        "orientation_dependencies": [
            {
                "target_kind": "asset_category",
                "target_category": "dining_table",
                "relation_type": "front_faces",
            }
        ]
    }
    dining_table = _object("dining_table_0", "dining_table", 0.0, 0.0)
    case_pack = {
        "scene_geometry": {"objects": [chair, dining_table]},
        "intent_contract": {
            "constraints": [
                {
                    "relation": "edge_distribution",
                    "subjects": {
                        "category": "dining_chair",
                        "count": 1,
                        "quantifier": "exactly",
                    },
                    "targets": {
                        "category": "dining_table",
                        "count": 1,
                        "quantifier": "exactly",
                    },
                    "orientation": "unconstrained",
                    "source": "explicit_prompt",
                }
            ]
        },
    }

    checks = build_functional_dependency_checks(case_pack)

    assert [
        check
        for check in checks
        if check.get("subject_id") == "dining_chair_0"
        and check.get("relation_type") == "furniture_faces_furniture"
    ]


def test_hard_across_and_wall_backed_contracts_own_subject_facing():
    sofa = _object("sofa_0", "sofa", -1.5, 0.0)
    cabinet = _object("cabinet_0", "storage_cabinet", 1.5, 0.0)
    for subject in (sofa, cabinet):
        subject["functional_hints"] = {
            "orientation_dependencies": [
                {
                    "target_kind": "asset_category",
                    "target_category": "coffee_table",
                    "relation_type": "front_faces",
                }
            ]
        }
    tv_stand = _object("tv_stand_0", "tv_stand", 1.5, 180.0)
    coffee_table = _object("coffee_table_0", "coffee_table", 0.0, 0.0)
    wall = _object("wall_0", "wall", 3.0, 0.0)
    case_pack = {
        "scene_geometry": {"objects": [sofa, cabinet, tv_stand, coffee_table, wall]},
        "intent_contract": {
            "constraints": [
                {
                    "relation": "across_from",
                    "subjects": {"category": "sofa", "count": 1},
                    "targets": {"category": "tv_stand", "count": 1},
                    "source": "explicit_prompt",
                },
                {
                    "relation": "against_wall",
                    "subjects": {"category": "storage_cabinet", "count": 1},
                    "targets": {"category": "wall", "count": 1},
                    "source": "explicit_prompt",
                },
            ]
        },
    }

    checks = build_functional_dependency_checks(case_pack)

    assert not [
        check
        for check in checks
        if check.get("subject_id") in {"sofa_0", "cabinet_0"}
        and check.get("relation_type") == "furniture_faces_furniture"
    ]
