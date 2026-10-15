"""Declared stage ownership must survive optional planner recommendations."""

import pytest

from scenesmith.scene_expert.global_planner import (
    _apply_stage_policy,
    _format_task_spec,
)
from scenesmith.scene_expert.schemas import (
    HarnessContext,
    MemoryPack,
    OptionalAssetRecommendation,
    SceneTaskSpec,
    StageBrief,
)


def resolve(spec, stage, names, policy="auto"):
    proposal = StageBrief(
        stage=stage,
        stage_objective="Complete the current stage",
        optional_asset_recommendations=[
            OptionalAssetRecommendation(name=name) for name in names
        ],
    )
    original = proposal.model_dump()
    result = _apply_stage_policy(
        proposal,
        HarnessContext(
            stage=stage, task_spec=spec, memory_pack=MemoryPack(), stage_policy=policy
        ),
    )
    assert proposal.model_dump() == original
    return result


def test_native_bathroom_future_assets_cannot_be_optional_furniture():
    spec = SceneTaskSpec(
        room_type="bathroom",
        style="modern",
        required_large_objects=["towel_rack", "shower_cabin"],
        required_wall_objects=["toilet", "floating_shelf"],
        required_small_objects=["scented_candle", "scented_candle"],
    )
    result = resolve(
        spec,
        "furniture",
        ["wall_hung_toilet", "floating_shelf", "scented_candle", "bath_mat"],
    )
    assert [x.name for x in result.optional_asset_recommendations] == ["bath_mat"]
    assert result.required_objects == ["towel_rack", "shower_cabin"]
    assert result.optional_assets_allowed
    assert "wall_mounted: toilet, floating_shelf" in result.to_injection_text()
    assert "manipuland: scented_candle, scented_candle" in result.to_injection_text()
    assert "Context-only inventory owned by wall_mounted" in _format_task_spec(
        spec, "furniture"
    )


@pytest.mark.parametrize(
    "stage,foreign,current",
    [
        ("floor_plan", "bed", "door"),
        ("furniture", "ceiling_light", "desk"),
        ("wall_mounted", "desk", "wall_mirror"),
        ("ceiling_mounted", "wall_mirror", "ceiling_light"),
        ("manipuland", "desk", "book"),
    ],
)
def test_ownership_guard_covers_all_native_stages(stage, foreign, current):
    spec = SceneTaskSpec(
        room_type="office",
        style="modern",
        required_architectural_features=["door"],
        required_large_objects=["desk", "bed"],
        required_wall_objects=["wall_mirror"],
        required_ceiling_objects=["ceiling_light"],
        required_small_objects=["book"],
    )
    result = resolve(spec, stage, [foreign, current, "unassigned_custom_asset"])
    assert [x.name for x in result.optional_asset_recommendations] == [
        current,
        "unassigned_custom_asset",
    ]


def test_matching_uses_head_nouns_and_keeps_same_stage_authority():
    spec = SceneTaskSpec(
        room_type="bathroom",
        style="modern",
        required_large_objects=["toilet_brush"],
        required_wall_objects=["toilet"],
        required_small_objects=["table_lamp"],
    )
    result = resolve(
        spec,
        "furniture",
        ["toilet_brush", "toilet_paper_holder", "brass_table_lamp", "coffee_table"],
    )
    assert [x.name for x in result.optional_asset_recommendations] == [
        "toilet_brush",
        "toilet_paper_holder",
        "coffee_table",
    ]


def test_empty_current_inventory_and_required_only_keep_native_stage_active():
    spec = SceneTaskSpec(
        room_type="office", style="modern", required_large_objects=["desk"]
    )
    for policy in ["auto", "required_only"]:
        result = resolve(spec, "wall_mounted", ["desk"], policy)
        assert not result.optional_asset_recommendations
        assert "does NOT disable or skip the stage" in result.to_injection_text()
        assert result.optional_assets_allowed == (policy == "auto")
