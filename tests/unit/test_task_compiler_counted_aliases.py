"""Preserve supplied units when distinct counted names share a category."""

from copy import deepcopy

import pytest

from scenesmith.scene_expert.schemas import SceneTaskSpec
from scenesmith.scene_expert.task_compiler import _normalize_stage_ownership


@pytest.mark.parametrize(
    "prompt,names,category,expected",
    [
        (
            "A cart holds one olive oil bottle, and one vinegar bottle.",
            ["olive oil bottle", "vinegar bottle"],
            "bottle",
            2,
        ),
        (
            "Place two olive oil bottles and two vinegar bottles on the cart.",
            ["olive oil bottle"] * 2 + ["vinegar bottle"] * 2,
            "bottle",
            4,
        ),
        (
            "Place 2 vinegar bottles, 1 olive oil bottle on the cart.",
            ["olive oil bottle"] + ["vinegar bottle"] * 2,
            "bottle",
            3,
        ),
        (
            "Place one blue mug and one red mug on the desk.",
            ["blue mug", "red mug"],
            "mug",
            2,
        ),
        (
            "Place one oil bottle, one vinegar bottle and one water bottle.",
            ["oil bottle", "vinegar bottle", "water bottle"],
            "bottle",
            3,
        ),
    ],
)
def test_counted_conjunction_preserves_model_supplied_units(
    prompt: str, names: list[str], category: str, expected: int
) -> None:
    raw = SceneTaskSpec(room_type="kitchen", style="", required_small_objects=names)
    before = raw.model_dump()

    actual = _normalize_stage_ownership(raw, prompt=prompt)

    assert actual.required_small_objects.count(category) == expected
    assert raw.model_dump() == before


@pytest.mark.parametrize(
    "prompt",
    [
        "",
        "Place an olive oil bottle and a vinegar bottle.",
        "Place one olive oil bottle or one vinegar bottle.",
        "Do not place one olive oil bottle and one vinegar bottle.",
        "If needed, place one olive oil bottle and one vinegar bottle.",
        "Place one olive oil bottle (also called one vinegar bottle).",
        "Place one olive oil bottle. Place one vinegar bottle.",
        "Place two olive oil bottles and one vinegar bottle.",
        "Place one olive oil bottle and one vinegar bottle. Remove one vinegar bottle.",
    ],
)
def test_unconfirmed_alias_multiplicity_keeps_existing_cap(prompt: str) -> None:
    raw = SceneTaskSpec(
        room_type="kitchen",
        style="",
        required_small_objects=["olive oil bottle", "vinegar bottle"],
    )

    actual = _normalize_stage_ownership(raw, prompt=prompt)

    assert actual.required_small_objects.count("bottle") == 1


def test_overlapping_generic_alias_mentions_still_count_once() -> None:
    raw = SceneTaskSpec(
        room_type="office",
        style="",
        required_large_objects=["large floor plant"] * 4 + ["plant"] * 4,
        required_small_objects=["computer monitor"] * 4 + ["monitor"] * 4,
    )

    actual = _normalize_stage_ownership(
        raw, prompt="Place four large floor plants and four computer monitors."
    )

    assert actual.required_large_objects.count("plant") == 4
    assert actual.required_small_objects.count("monitor") == 4


def test_count_conservation_does_not_promote_suggested_counter() -> None:
    raw = SceneTaskSpec(
        room_type="kitchen",
        style="",
        required_small_objects=["olive oil bottle", "vinegar bottle"],
        suggested_large_objects=["counter"],
    )

    actual = _normalize_stage_ownership(
        raw, prompt="One olive oil bottle and one vinegar bottle sit on a counter."
    )

    assert actual.required_small_objects.count("bottle") == 2
    assert actual.required_large_objects == []
    assert actual.suggested_large_objects == ["counter"]


def test_count_reconciliation_keeps_true_excess_capped() -> None:
    from scenesmith.scene_expert.hooks import (
        _reconcile_relation_counts_with_inventory,
        _task_spec_inventory_counts,
    )

    raw = SceneTaskSpec(
        room_type="kitchen",
        style="",
        required_large_objects=["rolling cart"],
        required_small_objects=["olive oil bottle", "vinegar bottle"],
    )
    normalized = _normalize_stage_ownership(
        raw, prompt="One olive oil bottle and one vinegar bottle sit on the cart."
    )
    relation = {
        "relation": "on_top_of",
        "source": "explicit_prompt",
        "subjects": {"category": "bottle", "count": 2},
        "targets": {"category": "rolling_cart", "count": 1},
    }
    excessive = deepcopy(relation)
    excessive["subjects"]["count"] = 3
    constraints = [relation, excessive]

    _reconcile_relation_counts_with_inventory(
        {"constraints": constraints}, _task_spec_inventory_counts(normalized)
    )

    assert relation["subjects"]["count"] == 2
    assert "reconciliation_reason" not in relation
    assert excessive["subjects"]["count"] == 2
    assert excessive["reconciliation_reason"] == "task_inventory_count_cap"
