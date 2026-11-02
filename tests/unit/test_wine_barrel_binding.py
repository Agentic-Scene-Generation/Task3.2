"""Bind declared wine-barrel labels without certifying the selected asset's function."""

import pytest

from scenesmith.scenebenchmark_critic.object_taxonomy import (
    canonical_object_category,
    generation_owner,
    semantic_category_match,
)


@pytest.mark.parametrize(
    "label",
    [
        "wine barrel",
        "wine_barrels",
        "wine_barrel_0",
        "wine_barrel2",
        "wooden wine barrel",
    ],
)
def test_wine_barrel_labels_satisfy_required_barrel(label: str) -> None:
    assert canonical_object_category(label) == "barrel"
    assert semantic_category_match("barrel", label)["matched"]
    assert generation_owner(label) == "furniture"


@pytest.mark.parametrize(
    "label",
    [
        "wine_barrel_chair",
        "wine_barrel_planter",
        "wine_barrel_table",
        "wine_barrel_holder",
        "wine_barrel_cover",
    ],
)
def test_barrel_shaped_compounds_do_not_gain_barrel_identity(label: str) -> None:
    assert not semantic_category_match("barrel", label)["matched"]


def _record(name: str, kind: str = "furniture") -> dict:
    category = canonical_object_category(name)
    return {
        "id": name + "_0",
        "category": category,
        "category_norm": category,
        "metadata": {"semantic_name": name},
        "object_type": kind,
        "functional_hints": {"scene_object_type": kind},
    }


def test_native_count_and_verifier_keep_distinct_instances_and_exact_limits() -> None:
    from scenesmith.scene_expert.schemas import SceneTaskSpec
    from scenesmith.scene_expert.verifier import _check_required_objects
    from scenesmith.scenebenchmark_critic.intent_contract import selected_ids
    from scenesmith.scenebenchmark_critic.metrics.functional_dependency.extensions.intent_contract import (
        _evaluate_required_count,
    )

    objects = [_record("barrel"), _record("wine_barrel"), _record("wine_barrel_chair")]
    constraint = {
        "constraint_id": "synthetic_barrel_count",
        "relation": "required_count",
        "subjects": {"category": "barrel", "count": 2, "stage": "furniture"},
    }
    assert selected_ids(constraint["subjects"], objects) == [
        "barrel_0",
        "wine_barrel_0",
    ]
    report = _evaluate_required_count(constraint, objects, "core")
    assert report["label"] == "pass"
    assert report["diagnostics"]["observed_count"] == 2
    assert _evaluate_required_count(constraint, objects[:1], "core")["label"] == "fail"
    exact = {
        **constraint,
        "subjects": {**constraint["subjects"], "count": 1, "quantifier": "exactly"},
    }
    assert _evaluate_required_count(exact, objects, "core")["label"] == "fail"
    task = SceneTaskSpec(
        room_type="wine_cellar",
        style="standard",
        required_large_objects=["barrel", "barrel"],
    )
    assert not _check_required_objects(
        task, "furniture", {"object_names": ["barrel", "wine barrel"]}
    )
    assert (
        len(
            _check_required_objects(
                task, "furniture", {"object_names": ["wine barrel"]}
            )
        )
        == 1
    )


@pytest.mark.parametrize("stage", ["wall_mounted", "ceiling_mounted", "manipuland"])
def test_barrel_alias_preserves_explicit_furniture_stage_guard(stage: str) -> None:
    from scenesmith.scenebenchmark_critic.intent_contract import selected_ids

    assert (
        selected_ids(
            {"category": "barrel", "stage": "furniture"},
            [_record("wine_barrel", stage)],
        )
        == []
    )
