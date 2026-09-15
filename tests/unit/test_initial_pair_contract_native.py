"""Real deterministic parser/rule regressions for candidate label calibration."""

from __future__ import annotations

from copy import deepcopy

import pytest

pytest.importorskip("pydrake")

from scenesmith.scene_expert.slow_memory.paired_contract import (
    calibrate_asset_checks,
    calibrate_contract,
)
from scenesmith.scenebenchmark_critic.config import CriticConfig
from scenesmith.scenebenchmark_critic.evaluator import (
    build_all_checks,
    run_case_pack_checks,
)
from scenesmith.scenebenchmark_critic.intent_contract import build_intent_contract


def _object(name, category, x, y, sx, sy, height, *, z=0.0, yaw=0.0):
    return {
        "id": name,
        "category": category,
        "category_norm": category,
        "name": category,
        "yaw_deg": yaw,
        "bbox_world": {
            "center": [x, y, z + height / 2],
            "size": [sx, sy, height],
            "min": [x - sx / 2, y - sy / 2, z],
            "max": [x + sx / 2, y + sy / 2, z + height],
        },
    }


def _evaluate(case):
    result = deepcopy(case)
    calibrate_contract(result)
    result["checks"] = build_all_checks(result, ["functional_dependency"])
    calibrate_asset_checks(result)
    rows = run_case_pack_checks(result, CriticConfig(metrics=["functional_dependency"]))
    return result, [r for r in rows if r.get("scoring_tier") != "auxiliary"]


@pytest.mark.parametrize("lifted", [False, True])
def test_prompt_floor_support_overrides_asset_prior_but_still_checks_height(lifted):
    plant = _object("plant_0", "plant", 1.0, 0, 0.4, 0.4, 0.9, z=0.8 if lifted else 0)
    plant["object_type"] = "furniture"
    plant["object_function_profile"] = {"is_small_placeable": True}
    plant["functional_hints"] = {
        "explicit_target_relation": ["table"],
        "scene_object_type": "manipuland",
    }
    table = _object("table_0", "coffee_table", 0, 0, 1, 1, 0.8)
    table["functional_hints"] = {"candidate_affordances": ["supportable"]}
    case = {
        "stage": "furniture",
        "original_task_instruction": "A plant on the floor.",
        "intent_contract": {"constraints": []},
        "scene_geometry": {
            "objects": [
                plant,
                table,
                _object("floor_0", "floor", 0, 0, 6, 6, 0.05, z=-0.05),
            ]
        },
    }
    before = deepcopy(case)
    calibrated, rows = _evaluate(case)
    assert case == before
    assert calibrated["sceneexpert_contract_calibration"]["superseded_asset_checks"]
    floor = [r for r in rows if r.get("relation_type") == "object_on_floor"]
    assert floor and any(r["label"] == "fail" for r in floor) is lifted


def test_explicit_table_support_keeps_failure_for_floor_plant():
    plant = _object("plant_0", "plant", 1, 0, 0.4, 0.4, 1.6)
    plant["functional_hints"] = {"explicit_target_relation": ["table"]}
    case = {
        "stage": "furniture",
        "original_task_instruction": "A plant on the table.",
        "intent_contract": build_intent_contract("A plant on the table."),
        "scene_geometry": {
            "objects": [plant, _object("table_0", "table", 0, 0, 1, 1, 0.8)]
        },
    }
    calibrated, rows = _evaluate(case)
    assert not calibrated["sceneexpert_contract_calibration"]["superseded_asset_checks"]
    assert any(r["label"] == "fail" for r in rows)


def test_asset_prior_is_not_suppressed_without_an_enforced_floor_check(monkeypatch):
    monkeypatch.setattr(
        "scenesmith.scenebenchmark_critic.intent_contract.augment_contract_checks",
        lambda case: False,
    )
    case = {
        "original_task_instruction": "A plant on the floor.",
        "intent_contract": {"constraints": []},
        "scene_geometry": {
            "objects": [
                _object("plant_0", "plant", 1, 0, 0.4, 0.4, 0.9),
                _object("floor_0", "floor", 0, 0, 6, 6, 0.05, z=-0.05),
            ]
        },
        "checks": [
            {
                "check_id": "generic_support",
                "subject_id": "plant_0",
                "relation_type": "object_on_support",
                "check_source": "asset_explicit_target_relation",
                "scoring_tier": "core",
            }
        ],
    }
    calibrate_contract(case)
    calibrate_asset_checks(case)
    assert case["checks"][0]["scoring_tier"] == "core"
    assert not case["sceneexpert_contract_calibration"]["superseded_asset_checks"]


@pytest.mark.parametrize("far", [False, True])
def test_tucked_seating_retains_functional_distance_and_orientation_checks(far):
    prompt = "An office chair tucked under the desk."
    contract = build_intent_contract(prompt)
    aligned = next(
        r for r in contract["constraints"] if r["relation"] == "aligned_with"
    )
    aligned["relation"] = "in_front_of"
    case = {
        "stage": "furniture",
        "original_task_instruction": prompt,
        "intent_contract": contract,
        "scene_geometry": {
            "objects": [
                _object(
                    "office_chair_0",
                    "office_chair",
                    0,
                    -4 if far else -0.4,
                    0.5,
                    0.5,
                    0.9,
                ),
                _object("desk_0", "desk", 0, 0, 1.2, 0.6, 0.75),
            ]
        },
    }
    calibrated, rows = _evaluate(case)
    changes = calibrated["sceneexpert_contract_calibration"]["changes"]
    assert any(r["kind"] == "restore_tucked_seat_alignment" for r in changes)
    assert rows and any(r["label"] == "fail" for r in rows) is far


def test_explicit_in_front_requirement_is_not_relaxed():
    prompt = "An office chair in front of the desk."
    case = {
        "original_task_instruction": prompt,
        "intent_contract": build_intent_contract(prompt),
    }
    original = deepcopy(case["intent_contract"])
    calibrate_contract(case)
    assert case["intent_contract"] == original
    assert not case["sceneexpert_contract_calibration"]["changes"]


@pytest.mark.parametrize("wrong_facing", [False, True])
def test_combined_edge_partition_checks_all_seven_chairs(wrong_facing):
    prompt = (
        "A meeting room with one rectangular conference table and seven office chairs. "
        "Arrange six office chairs in two equal groups of three along the table's two long sides, "
        "all facing the table. Place one remaining office chair centered along one short side, "
        "facing the table. Keep the opposite short side free of chairs."
    )
    contract = build_intent_contract(prompt)
    edge = next(
        r for r in contract["constraints"] if r["relation"] == "edge_distribution"
    )
    parts = []
    for i, group in enumerate(edge["groups"]):
        part = deepcopy(edge)
        part["subjects"].update(
            count=sum(group["counts_per_edge"]), cohort=f"cohort_{i}"
        )
        part["groups"] = [group]
        parts.append(part)
    contract["constraints"] = parts
    objects = [_object("conference_table_0", "conference_table", 0, 0, 4, 1.2, 0.75)]
    for i, (x, y, yaw) in enumerate(
        [
            (-4 / 3, -0.95, 0),
            (0, -0.95, 0),
            (4 / 3, -0.95, 0),
            (-4 / 3, 0.95, 180),
            (0, 0.95, 180),
            (4 / 3, 0.95, 180),
            (-2.35, 0, -90),
        ]
    ):
        objects.append(
            _object(
                f"office_chair_{i}",
                "office_chair",
                x,
                y,
                0.6,
                0.6,
                0.9,
                yaw=90 if wrong_facing and i == 0 else yaw,
            )
        )
    calibrated, rows = _evaluate(
        {
            "stage": "furniture",
            "original_task_instruction": prompt,
            "intent_contract": contract,
            "scene_geometry": {"objects": objects},
        }
    )
    assert (
        calibrated["sceneexpert_contract_calibration"]["changes"][0]["kind"]
        == "restore_complete_edge_partition"
    )
    assert rows and any(r["label"] == "fail" for r in rows) is wrong_facing
