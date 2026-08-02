from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "scenesmith" / "scenebenchmark_critic" / "asset_annotation_data"
LOOKUP = DATA / "hssd_annotation_lookup.json.gz"
AUDIT = DATA / "HSSD_OPEN_MESH_PHYSICS_AUDIT.json"
FULL_SUMMARY = DATA / "HSSD_MESH_PHYSICS_FULL_SUMMARY.json"
PLANT_ID = "cb9b5a9ee8e0eb6cacd1eb98cfe65cced77ad54f"
HIGH_RISK_IDS = {
    PLANT_ID,
    "d160e323f1d5a92394de74865aef8938f986d344",
    "19c7d98ce8c620cd916b4fe978c565b065edcea1",
    "3b63f000011404011c4aeb3f19bb04a112535634",
    "xxxx82261fccxf2d8x4aacxbf4axb3a0a222dbad",
    "c717fbe8f10ad354c83f9cfc7ce154ae2fcdf96c",
}

# Load the standalone annotation module without importing the package-level
# Drake adapter. This keeps data-contract tests runnable in lightweight CI.
MODULE_PATH = (
    REPO / "scenesmith" / "scenebenchmark_critic" / "asset_library_annotations.py"
)
SPEC = importlib.util.spec_from_file_location(
    "hssd_asset_library_annotations", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
ANNOTATIONS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANNOTATIONS)
AssetLibraryAnnotationStore = ANNOTATIONS.AssetLibraryAnnotationStore
build_scenebenchmark_annotation = ANNOTATIONS.build_scenebenchmark_annotation


def test_high_risk_open_mesh_assets_are_merged_into_lookup():
    with gzip.open(LOOKUP, "rt", encoding="utf-8") as handle:
        lookup = json.load(handle)
    assert len(lookup) == 10963
    for asset_id in HIGH_RISK_IDS:
        quality = lookup[asset_id]["asset_quality"]
        assert quality["mesh_topology"]["measured"] is True
        assert quality["mesh_topology"]["runtime_watertight_observation"] is False
        assert quality["mesh_topology"]["topology_detail_status"] == "resolved_glb_scan_complete"
        assert quality["physics_proxy"]["policy"] in {
            "bbox_inertia",
            "weld_or_static",
        }
        assert quality["is_acceptable"] is True
        assert "watertight_not_measured" not in quality["warning_tags"]


def test_falling_plant_uses_bbox_inertia_without_rejection():
    store = AssetLibraryAnnotationStore(lookup_path=LOOKUP)
    record = store.require(PLANT_ID)
    quality = record["asset_quality"]
    topology = quality["mesh_topology"]
    proxy = store.get_physics_proxy_policy(PLANT_ID)
    stability = quality["support_stability"]

    assert topology["is_open_by_design"] is True
    assert topology["open_mesh_reason"] == "thin_leaf_surfaces"
    assert topology["boundary_edge_count"] == 16796
    assert topology["topology_detail_status"] == "resolved_glb_scan_complete"
    assert proxy["policy"] == "bbox_inertia"
    assert proxy["collision_proxy_policy"] == "convex_decomposition"
    assert proxy["is_usable_in_physics"] is True
    assert stability["stable_with_recommended_proxy"] is True
    assert stability["validation_status"] == "week31_scene_replay_verified"


def test_scenebenchmark_hints_expose_open_mesh_policy():
    store = AssetLibraryAnnotationStore(lookup_path=LOOKUP)
    record = store.require(PLANT_ID)
    hints = build_scenebenchmark_annotation(record)["functional_hints"]
    assert hints["mesh_topology"]["watertight"] is False
    assert hints["physics_proxy"]["policy"] == "bbox_inertia"
    assert hints["support_stability"]["stable_with_recommended_proxy"] is True


def test_audit_is_explicit_about_unavailable_measurements():
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    assert audit["scope"]["asset_count"] == 6
    assert audit["scope"]["full_topology_scan_count"] == 0
    assert set(audit["assets"]) == HIGH_RISK_IDS
    for annotation in audit["assets"].values():
        topology = annotation["mesh_topology"]
        assert topology["measurement_method"] == "trimesh.is_watertight"
        assert topology["boundary_edge_count"] is None
        assert topology["non_manifold_edge_count"] is None


def test_full_library_mesh_physics_audit_is_complete():
    summary = json.loads(FULL_SUMMARY.read_text(encoding="utf-8"))
    assert summary["asset_count"] == 10963
    assert summary["scan_status_counts"] == {"complete": 10962, "error": 1}
    assert summary["watertight_counts"] == {"False": 10256, "True": 706}
    assert summary["physics_proxy_policy_counts"] == {
        "bbox_inertia": 7889,
        "mesh_mass_properties": 691,
        "weld_or_static": 2383,
    }


def test_zero_thickness_wall_art_is_attached_instead_of_rejected():
    store = AssetLibraryAnnotationStore(lookup_path=LOOKUP)
    policy = store.get_physics_proxy_policy(
        "775a3ca949c41b0f143a1c919efa402817172e1f"
    )
    assert policy["policy"] == "weld_or_static"
    assert policy["is_usable_in_physics"] is True
