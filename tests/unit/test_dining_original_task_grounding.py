"""Keep dining obligations tied to the user task and physical object identity."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from scenesmith.scenebenchmark_critic import adapter
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.extensions.manipuland_completeness import (
    _required_groups,
    evaluate_manipuland_completeness,
)
from scenesmith.scene_expert.schemas import SceneTaskSpec
from scenesmith.scene_expert.verifier import evaluate_required_asset_coverage


TASK = (
    "A dining room with a table set for 8 people. Each setting has a plate, "
    "a glass, a fork, and a knife."
)


def _geometry(place_count: int = 8, *, include_glasses: bool = True) -> dict:
    objects = [{
        "id": "dining_table_0", "name": "dining_table", "object_type": "furniture",
        "support_surfaces": [{"surface_id": "table.top"}],
    }]
    groups = ["dinner_plate", "dinner_fork", "dinner_knife"]
    if include_glasses:
        groups.append("water_glass")
    for name in groups:
        for index in range(place_count):
            objects.append({
                "id": f"{name}_{index}", "name": name,
                "object_type": "manipuland",
                "placement_info": {"parent_surface_id": "table.top"},
            })
    return {"objects": objects}


@pytest.mark.parametrize("live_original_attribute", [True, False])
def test_case_pack_uses_original_task_before_dining_hard_checks(
    monkeypatch: pytest.MonkeyPatch, live_original_attribute: bool,
) -> None:
    injected = TASK + (
        "\n\n=== SceneExpert Stage Brief: manipuland ===\n"
        "Optional asset recommendation: one central serving bowl."
    )
    scene = SimpleNamespace(
        text_description=injected, scene_expert_original_description=(
            TASK if live_original_attribute else ""
        ), room_id="dining_room", room_type="dining_room", metadata={},
    )
    monkeypatch.setattr(adapter, "_room_scene_geometry", lambda _: _geometry())
    monkeypatch.setattr(adapter, "_physics_evidence_for_scene", lambda _: None)
    monkeypatch.setattr(
        adapter, "build_all_checks",
        lambda pack, **_: evaluate_manipuland_completeness(pack),
    )
    pack = adapter.room_scene_to_case_pack(scene, stage="manipuland")
    assert pack["task_instruction"] == pack["original_task_instruction"] == TASK
    assert scene.text_description == injected  # Designer hints stay available.
    assert pack["checks"][0]["label"] == "pass"
    assert pack["checks"][0]["diagnostics"]["required_groups"] == [
        "drinkware", "fork", "knife", "plate",
    ]


def test_explicit_bowls_remain_required() -> None:
    prompt = TASK.replace("a plate,", "a bowl, a plate,")
    result = evaluate_manipuland_completeness({
        "task_instruction": prompt, "scene_geometry": _geometry(),
    })[0]
    assert result["label"] == "fail"
    assert result["diagnostics"]["missing"] == {"bowl": 8}


def test_original_eight_people_count_is_not_reduced_to_six_observed_settings() -> None:
    result = evaluate_manipuland_completeness({
        "task_instruction": TASK, "scene_geometry": _geometry(6),
    })[0]
    assert result["label"] == "fail"
    assert result["diagnostics"]["place_count"] == 8
    assert result["diagnostics"]["missing"] == {
        "drinkware": 2, "fork": 2, "knife": 2, "plate": 2,
    }


def test_singular_drinking_glass_requirement_remains_enforced() -> None:
    result = evaluate_manipuland_completeness({
        "task_instruction": TASK, "scene_geometry": _geometry(include_glasses=False),
    })[0]
    assert result["diagnostics"]["missing"] == {"drinkware": 8}
    assert "drinkware" not in _required_groups("A dining room with a glass-fronted cabinet.")


def _glass_task(count: int = 1) -> SceneTaskSpec:
    return SceneTaskSpec(room_type="dining_room", style="modern",
                         required_small_objects=["glass"] * count)


def test_glass_materials_in_actual_cabinet_and_light_descriptions_are_not_cups() -> None:
    report = evaluate_required_asset_coverage(_glass_task(8), "manipuland", {
        "object_records": [
            {"name": "china_cabinet", "description":
             "Tall wooden china cabinet / sideboard with glass-fronted upper doors and drawers"},
            {"name": "dining_chandelier", "description":
             "Elegant multi-arm chandelier with warm brass frame and frosted glass globes"},
        ],
    })
    assert report["required_satisfied_objects"] == []
    assert report["required_missing_objects"] == ["glass"] * 8
    assert report["required_coverage"] == 0.0


@pytest.mark.parametrize("name", ["glass_vase", "glass_table", "glass_fronted_cabinet"])
def test_material_words_in_asset_names_do_not_create_drinkware(name: str) -> None:
    report = evaluate_required_asset_coverage(
        _glass_task(), "manipuland", {"object_names": [name]},
    )
    assert report["required_coverage"] == 0.0


def test_eight_real_glass_identities_consume_eight_distinct_objects() -> None:
    records = [{"name": f"small_glass_{index}"} for index in range(8)]
    full = evaluate_required_asset_coverage(
        _glass_task(8), "manipuland", {"object_records": records},
    )
    partial = evaluate_required_asset_coverage(
        _glass_task(8), "manipuland", {"object_records": records[:7]},
    )
    assert full["required_coverage"] == 1.0
    assert partial["required_missing_objects"] == ["glass"]


def test_explicit_drinkware_description_can_identify_an_opaque_asset_name() -> None:
    report = evaluate_required_asset_coverage(_glass_task(), "manipuland", {
        "object_records": [{"name": "asset_42", "description": "A clear water glass."}],
    })
    assert report["required_coverage"] == 1.0
