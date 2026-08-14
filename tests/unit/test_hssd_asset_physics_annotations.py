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
