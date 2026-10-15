"""Optional planner hints must respect positive native capability evidence."""

import json

import pytest

from scenesmith.scene_expert.global_planner import GlobalPlanner
from scenesmith.scene_expert.retrieval_capabilities import (
    native_capabilities_from_config,
)
from scenesmith.scene_expert.schemas import (
    HarnessContext,
    MemoryPack,
    OptionalAssetRecommendation,
    SceneTaskSpec,
    StageBrief,
)


def config(tmp_path, *, bucket="large_objects"):
    data = {
        "large_objects": ["sink.n.01", "bathtub.n.01", "table.n.02", "floor_lamp.n.01"],
        "wall_objects": ["toilet.n.02", "table_lamp.n.01", "sink.n.01"],
        "small_objects": ["book.n.01"],
        "ceiling_objects": ["chandelier.n.01"],
    }
    (tmp_path / "object_categories.json").write_text(json.dumps(data))
    return {
        "furniture_agent": {
            "asset_manager": {
                "general_asset_source": "hssd",
                "hssd": {
                    "preprocessed_path": str(tmp_path),
                    "object_type_mapping": {"FURNITURE": bucket},
                },
            }
        }
    }


def proposal(*names):
    return StageBrief(
        stage="furniture",
        stage_objective="Complete the room",
        required_objects=["sink", "floor_rack"],
        optional_asset_recommendations=[
            OptionalAssetRecommendation(name=n) for n in names
        ],
    )


def test_native173_optional_eligibility_preserves_required_and_other_choices(tmp_path):
    cfg = config(tmp_path)
    cap = native_capabilities_from_config(cfg, enabled=True)["furniture"]
    original = proposal("toilet", "bathtub", "custom_floor_rack")
    snapshot = original.model_dump()
    result, audit = cap.filter_optional(original, ["sink", "floor_rack"])
    assert [x.name for x in result.optional_asset_recommendations] == [
        "bathtub",
        "custom_floor_rack",
    ]
    assert result.required_objects == original.required_objects
    assert original.model_dump() == snapshot
    assert audit["omitted_optional_names"] == ["toilet"]
    assert audit["optional_observations"][-1]["status"] == "unknown"
    assert len(audit["metadata_sha256"]) == 64
    assert (
        "OPTIONAL" in result.to_injection_text()
        and "toilet" in result.to_injection_text()
    )


def test_most_specific_category_and_qualified_head_do_not_use_interior_nouns(tmp_path):
    cap = native_capabilities_from_config(config(tmp_path), enabled=True)["furniture"]
    assert cap.classify("brass_table_lamp")[0] == "known_unavailable"
    assert cap.classify("vintage_toilet")[0] == "known_unavailable"
    assert cap.classify("floor_lamp")[0] == "eligible"
    assert cap.classify("toilet_brush")[0] == "unknown"
    assert cap.classify("nightstand_with_table_lamp")[0] == "unknown"


def test_native_multiple_bucket_membership_keeps_current_eligibility(tmp_path):
    cap = native_capabilities_from_config(config(tmp_path), enabled=True)["furniture"]
    status, _, buckets = cap.classify("sink")
    assert status == "eligible" and buckets == ["large_objects", "wall_objects"]


def test_required_capability_gap_is_visible_and_never_removed(tmp_path):
    cap = native_capabilities_from_config(config(tmp_path), enabled=True)["furniture"]
    brief = proposal("toilet").model_copy(update={"required_objects": ["toilet"]})
    result, audit = cap.filter_optional(brief, ["toilet"])
    assert result.required_objects == ["toilet"]
    assert [x.name for x in result.optional_asset_recommendations] == ["toilet"]
    assert audit["optional_observations"][0]["status"] == "known_unavailable"
    assert audit["optional_observations"][0]["required_inventory_protected"]


def test_actual_object_type_mapping_controls_eligibility(tmp_path):
    cap = native_capabilities_from_config(
        config(tmp_path, bucket="wall_objects"), enabled=True
    )["furniture"]
    result, _ = cap.filter_optional(proposal("toilet", "bathtub"), [])
    assert [x.name for x in result.optional_asset_recommendations] == ["toilet"]


def test_disabled_guard_never_opens_native_metadata(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    monkeypatch.setattr(
        type(tmp_path), "read_bytes", lambda _: pytest.fail("unexpected metadata read")
    )
    assert native_capabilities_from_config(cfg) == {}


@pytest.mark.parametrize("data", [None, {}, {"large_objects": "not a category list"}])
def test_missing_or_malformed_catalog_does_not_invent_a_negative_capability(
    tmp_path, data
):
    cfg = config(tmp_path)
    path = tmp_path / "object_categories.json"
    if data is None:
        path.unlink()
    else:
        path.write_text(json.dumps(data))
    assert native_capabilities_from_config(cfg, enabled=True) == {}


def test_non_hssd_provider_and_unknown_mapping_are_not_filtered(tmp_path):
    cfg = config(tmp_path)
    cfg["furniture_agent"]["asset_manager"]["general_asset_source"] = "generated"
    assert native_capabilities_from_config(cfg, enabled=True) == {}
    assert (
        native_capabilities_from_config(config(tmp_path, bucket="custom"), enabled=True)
        == {}
    )


def test_planner_filters_exactly_one_native_proposal_and_records_audit(tmp_path):
    planner = GlobalPlanner.__new__(GlobalPlanner)
    planner._native_retrieval_capabilities = native_capabilities_from_config(
        config(tmp_path), enabled=True
    )
    planner.last_trace = {"status": "ok"}
    calls = []
    planner._generate_stage_brief = lambda *args: calls.append(args) or proposal(
        "toilet", "bathtub"
    )
    context = HarnessContext(
        stage="furniture",
        task_spec=SceneTaskSpec(
            room_type="bathroom",
            style="oldschool",
            required_large_objects=["sink", "floor_rack"],
        ),
        memory_pack=MemoryPack(),
    )
    result = planner.generate_stage_brief(context)
    assert len(calls) == 1
    assert [x.name for x in result.optional_asset_recommendations] == ["bathtub"]
    assert planner.last_trace["native_retrieval_capability"][
        "omitted_optional_names"
    ] == ["toilet"]
    assert "Native Retrieval Capability" in planner._build_user_message(context, "")


def test_legacy_planner_without_capability_configuration_preserves_proposal():
    planner = GlobalPlanner.__new__(GlobalPlanner)
    original = proposal("toilet")
    planner._generate_stage_brief = lambda *args: original
    context = HarnessContext(
        stage="furniture",
        task_spec=SceneTaskSpec(room_type="bathroom", style="oldschool"),
        memory_pack=MemoryPack(),
    )
    assert planner.generate_stage_brief(context) is original
