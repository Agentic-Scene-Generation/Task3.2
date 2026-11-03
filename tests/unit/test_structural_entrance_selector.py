"""Entrance relations must bind physical openings without bypassing geometry gates."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from scenesmith.scenebenchmark_critic.adapter import _opening_geometry_proxy
from scenesmith.scenebenchmark_critic.core.geometry import load_geometry
from scenesmith.scenebenchmark_critic.intent_contract import (
    augment_contract_checks,
    bound_ids,
    selected_ids,
    selector_match_count,
    semantic_selector_matches,
)
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.extensions.intent_contract import (
    _binding_state_result,
)
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.relations import (
    evaluate_functional_dependency,
)


def _door(opening_id: str = "door_1", x: float = 0.0) -> dict:
    opening = SimpleNamespace(
        opening_id=opening_id,
        opening_type="door",
        wall_direction="south",
        center_world=[x, -3.0, 1.05],
        width=1.0,
        height=2.1,
        sill_height=0.0,
    )
    scene = SimpleNamespace(
        room_id="room_0", room_geometry=SimpleNamespace(wall_thickness=0.05)
    )
    proxy = _opening_geometry_proxy(opening, scene, np.zeros(3))
    assert proxy is not None
    return proxy


def _counter(y: float) -> dict:
    return {
        "id": "checkout_counter_0",
        "category": "counter",
        "category_norm": "counter",
        "object_type": "furniture",
        "metadata": {"semantic_name": "checkout_counter"},
        "bbox_world": {
            "center": [0.0, y, 0.45],
            "size": [0.8, 0.5, 0.9],
            "min": [-0.4, y - 0.25, 0.0],
            "max": [0.4, y + 0.25, 0.9],
        },
    }


def _constraint(category: str = "entrance") -> dict:
    return {
        "constraint_id": "counter_near_entrance",
        "relation": "near",
        "stage": "furniture",
        "strength": "hard",
        "source": "explicit_prompt",
        "evidence_span": "a checkout counter near the entrance",
        "subjects": {
            "category": "checkout_counter",
            "count": 1,
            "quantifier": "exactly",
        },
        "targets": {"category": category, "count": 1, "quantifier": "exactly"},
    }


@pytest.mark.parametrize("category", ["entrance", "entry"])
def test_entrance_selector_binds_the_actual_structural_door(category: str) -> None:
    objects = [_counter(-2.3), _door()]
    before = copy.deepcopy(objects)
    selector = _constraint(category)["targets"]
    assert selected_ids(selector, objects) == ["door_1"]
    assert bound_ids(selector, objects) == ["door_1"]
    assert selector_match_count(selector, objects) == 1
    matches = semantic_selector_matches(selector, objects)
    assert len(matches) == 1
    assert matches[0]["match_kind"] == "structural_anchor_alias"
    assert matches[0]["match_path"] == ["door", category]
    assert matches[0]["context_matched"] is True
    assert (
        _binding_state_result({"stage": "furniture"}, _constraint(category), objects)
        is None
    )
    assert objects == before


@pytest.mark.parametrize("y, expected", [(-2.3, "pass"), (2.4, "fail")])
def test_resolved_entrance_still_uses_the_existing_near_geometry_gate(
    y: float, expected: str
) -> None:
    objects = [_counter(y), _door()]
    case_pack = {
        "stage": "furniture",
        "scene_geometry": {"objects": objects},
        "intent_contract": {"constraints": [_constraint()]},
    }
    assert augment_contract_checks(case_pack)
    checks = [
        x for x in case_pack["checks"] if x["relation_type"] == "generic_near_relation"
    ]
    assert len(checks) == 1
    assert checks[0]["target_ids"] == ["door_1"]
    result = evaluate_functional_dependency(load_geometry(case_pack), checks[0])
    assert result["label"] == expected


def test_multiple_doors_keep_singular_entrance_binding_ambiguous() -> None:
    objects = [_counter(-2.3), _door("door_1", -1.5), _door("door_2", 1.5)]
    selector = _constraint()["targets"]
    assert selected_ids(selector, objects) == ["door_1", "door_2"]
    assert bound_ids(selector, objects) == []
    result = _binding_state_result({"stage": "furniture"}, _constraint(), objects)
    assert result is not None and result["label"] == "fail"
    assert result["diagnostics"]["binding_issue"] == "ambiguous"


@pytest.mark.parametrize(
    "change",
    ["furniture", "no_proxy", "different_id", "window", "conflicting_category"],
)
def test_entrance_alias_does_not_accept_unverified_or_non_door_assets(
    change: str,
) -> None:
    obj = _door()
    if change == "furniture":
        obj["object_type"] = "furniture"
    elif change == "no_proxy":
        obj["metadata"].pop("structural_proxy")
    elif change == "different_id":
        obj["metadata"]["opening_id"] = "different"
    elif change == "window":
        obj["category"] = obj["category_norm"] = "window"
    else:
        obj["category_norm"] = "window"
    obj["name"] = "entrance_doorway"
    selector = _constraint()["targets"]
    assert selected_ids(selector, [obj]) == []
    assert selector_match_count(selector, [obj]) == 0
    assert semantic_selector_matches(selector, [obj]) == []


@pytest.mark.parametrize(
    "qualifier",
    [{"stage": "furniture"}, {"cohort": "unmatched"}, {"capabilities": ["storage"]}],
)
def test_structural_alias_does_not_bypass_selector_qualifiers(qualifier: dict) -> None:
    obj = _door()
    obj["metadata"]["cohort"] = "actual_openings"
    selector = {**_constraint()["targets"], **qualifier}
    assert selected_ids(selector, [obj]) == []
    assert selector_match_count(selector, [obj]) == 0
    assert semantic_selector_matches(selector, [obj])[0]["context_matched"] is False


def test_missing_door_and_unrelated_furniture_names_remain_unbound() -> None:
    obj = _counter(-2.3)
    obj["name"] = "entrance_display_cabinet"
    result = _binding_state_result({"stage": "furniture"}, _constraint(), [obj])
    assert result is not None and result["diagnostics"]["target_matches"] == []
    assert result["label"] == "fail"


def test_structural_alias_does_not_guess_entrance_from_pose_or_reclassify_inventory() -> (
    None
):
    obj = _door()
    selector = _constraint()["targets"]
    original = copy.deepcopy(obj)
    obj["bbox_world"]["center"] = [100.0, 100.0, 1.05]
    assert selected_ids(selector, [obj]) == ["door_1"]
    assert selected_ids({"category": "aquarium"}, [obj]) == []
    assert selected_ids({"category": "shelf_unit"}, [obj]) == []
    assert obj["category"] == original["category"] == "door"
    assert obj["metadata"] == original["metadata"]
