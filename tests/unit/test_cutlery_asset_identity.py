"""Reject known utensil substitutions before any name or material fallback."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from scenesmith.agent_utils.asset_manager import AssetManager
from scenesmith.agent_utils.asset_router.rendered_asset_choice import (
    choose_hssd_candidate_from_iso_renders,
    cutlery_asset_identity_mismatch,
)
from scenesmith.agent_utils.hssd_retrieval_server.dataclasses import HssdRetrievalResult


SPOON208 = "3b63f000011404011c4aeb3f19bb04a112535634"
LOOKUP = "scenesmith.scenebenchmark_critic.current_asset_annotations.get_current_asset_annotation"
CHOICE = "scenesmith.agent_utils.asset_router.rendered_asset_choice"


def candidate(asset_id: str) -> HssdRetrievalResult:
    # The real retrieval response echoed the requested name for every geometry.
    return HssdRetrievalResult(
        mesh_path="/unopened/mesh.gltf",
        hssd_id=asset_id,
        object_name="Stainless steel dinner knife",
        category="small_objects",
        similarity_score=0.9,
        size=(0.05, 0.25, 0.01),
    )


def choose(candidates, **kwargs):
    return choose_hssd_candidate_from_iso_renders(
        candidates=candidates,
        object_description="Stainless steel dinner knife",
        object_short_name="dinner_knife",
        scene_context=None,
        vlm_service=kwargs.pop("vlm_service", MagicMock()),
        model="fixture",
        reasoning_effort="low",
        verbosity="low",
        vision_detail="low",
        rendered_assets_dir=Path("/unopened/renders"),
        top_n=kwargs.pop("top_n", 4),
        **kwargs,
    )


@pytest.mark.parametrize("requested", ["dinner_fork", "dinner_knife", "knives_0"])
def test_actual208_spoon_annotation_cannot_be_renamed_as_requested_cutlery(requested):
    reason = cutlery_asset_identity_mismatch(SPOON208, requested)
    assert reason is not None and "'spoon'" in reason


def test_one_wrong_candidate_is_rejected_before_skip_or_model_call(tmp_path):
    vlm = MagicMock()
    audit = tmp_path / "choice.jsonl"
    choice = choose([candidate(SPOON208)], top_n=1, vlm_service=vlm, audit_path=audit)
    assert choice.candidates == []
    assert choice.selected_hssd_id is None
    vlm.create_completion.assert_not_called()
    event = json.loads(audit.read_text())
    assert event["status"] == "rejected_known_cutlery_identity"
    assert SPOON208 in event["identity_rejections"]


def test_material_fallback_preserves_cutlery_identity_and_bound_label():
    assets = [
        candidate("knife_dark"),
        candidate("spoon_bright"),
        candidate("knife_clear"),
    ]
    annotations = {
        "knife_dark": {"category": "knife"},
        "spoon_bright": {"category": "spoon"},
        "knife_clear": {"category": "knife"},
    }
    vlm = MagicMock()
    vlm.create_completion.return_value = json.dumps(
        {
            "selected_hssd_id": "knife_dark",
            "selected_index": 1,
            "semantic_name": "glass",
            "reason": "knife shape",
        }
    )
    with (
        patch(LOOKUP, side_effect=lambda uid: annotations[uid]),
        patch(
            CHOICE + "._candidate_render_views",
            return_value=(("iso", Path("/unused.png")),),
        ),
        patch(CHOICE + "._encode_candidate_views_as_one_image", return_value="fixture"),
        patch(
            CHOICE + "._render_material_quality",
            side_effect=lambda evidence: {
                "score": 0.1 if evidence.candidate.hssd_id == "knife_dark" else 1.0,
                "issues": [],
            },
        ),
    ):
        result = choose(
            assets, vlm_service=vlm, semantic_name_candidates=["glass", "dinner_knife"]
        )
    assert result.selected_hssd_id == "knife_clear"
    assert result.semantic_name == "dinner_knife"
    assert "spoon_bright" not in {item.hssd_id for item in result.candidates}
    assert "material-evidence guard" in result.reason


@pytest.mark.parametrize(
    "failure_mode", ["no_renders", "model_error", "invalid_selection"]
)
def test_fallbacks_never_restore_a_rejected_spoon(failure_mode):
    assets = [candidate("knife_a"), candidate("spoon_bright"), candidate("knife_b")]
    annotations = {
        "knife_a": {"category": "knife"},
        "spoon_bright": {"category": "spoon"},
        "knife_b": {"category": "knife"},
    }
    vlm = MagicMock()
    if failure_mode == "model_error":
        vlm.create_completion.side_effect = RuntimeError("fixture transport failure")
    else:
        vlm.create_completion.return_value = '{"selected_hssd_id":"absent"}'
    with (
        patch(LOOKUP, side_effect=lambda uid: annotations[uid]),
        patch(
            CHOICE + "._candidate_render_views",
            return_value=(
                () if failure_mode == "no_renders" else (("iso", Path("/unused.png")),)
            ),
        ),
        patch(CHOICE + "._encode_candidate_views_as_one_image", return_value="fixture"),
        patch(
            CHOICE + "._render_material_quality",
            return_value={"score": 1.0, "issues": []},
        ),
    ):
        result = choose(assets, vlm_service=vlm)
    assert {item.hssd_id for item in result.candidates} == {"knife_a", "knife_b"}


def test_direct_processing_rejects_before_geometry_and_registration():
    manager = AssetManager.__new__(AssetManager)
    request = SimpleNamespace(short_names=["dinner_fork"])
    with pytest.raises(ValueError, match="source category 'spoon'"):
        manager._process_direct_hssd_candidate(
            request=request,
            index=0,
            config=None,
            candidate=candidate(SPOON208),
            semantic_name="dinner_fork",
        )


@pytest.mark.parametrize("requested", ["spoon", "spoon_holder", "knife_block", "table"])
def test_matching_spoon_or_non_utensil_requests_are_not_rejected(requested):
    assert cutlery_asset_identity_mismatch(SPOON208, requested) is None


def test_missing_or_generic_annotation_does_not_invent_an_identity_failure():
    with patch(LOOKUP, side_effect=KeyError("unknown")):
        assert cutlery_asset_identity_mismatch("unknown", "dinner_fork") is None
    with patch(LOOKUP, return_value={"category": "cutlery"}):
        assert cutlery_asset_identity_mismatch("generic", "dinner_fork") is None
