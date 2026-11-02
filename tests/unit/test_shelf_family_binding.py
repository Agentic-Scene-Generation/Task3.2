"""Count actual shelving consistently while retaining typed and stage boundaries."""

import pytest

from scenesmith.scenebenchmark_critic.object_taxonomy import (
    canonical_object_category,
    generation_owner,
    semantic_category_match,
)


@pytest.mark.parametrize(
    "label",
    [
        "bookshelf",
        "bookcase",
        "shelving unit",
        "bookshelf_0",
        "bookcase2",
        "wooden_bookcase",
    ],
)
def test_bookcase_family_satisfies_generic_shelf_without_losing_identity(
    label: str,
) -> None:
    assert canonical_object_category(label) == "bookshelf"
    assert generation_owner(label) == "furniture"
    assert semantic_category_match("shelf", label)["matched"]
    assert not semantic_category_match("bookshelf", "shelf")["matched"]


@pytest.mark.parametrize(
    "label",
    [
        "wall_shelf",
        "wall_bookshelf",
        "bookshelf_cover",
        "bookcase_holder",
        "desk_bookshelf",
    ],
)
def test_storage_compounds_do_not_gain_generic_shelf_identity(label: str) -> None:
    assert not semantic_category_match("shelf", label)["matched"]


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


def test_native_count_and_verifier_agree_on_distinct_shelf_family_instances() -> None:
    from scenesmith.scene_expert.schemas import SceneTaskSpec
    from scenesmith.scene_expert.verifier import _check_required_objects
    from scenesmith.scenebenchmark_critic.intent_contract import selected_ids
    from scenesmith.scenebenchmark_critic.metrics.functional_dependency.extensions.intent_contract import (
        _evaluate_required_count,
    )

    objects = [_record("shelf"), _record("bookcase"), _record("bookshelf_cover")]
    constraint = {
        "constraint_id": "synthetic_shelving_count",
        "relation": "required_count",
        "subjects": {"category": "shelf", "count": 2, "stage": "furniture"},
    }
    assert selected_ids(constraint["subjects"], objects) == ["bookcase_0", "shelf_0"]
    result = _evaluate_required_count(constraint, objects, "core")
    assert result["label"] == "pass"
    assert result["diagnostics"]["observed_count"] == 2
    assert _evaluate_required_count(constraint, objects[:1], "core")["label"] == "fail"
    exact = {
        **constraint,
        "subjects": {**constraint["subjects"], "count": 1, "quantifier": "exactly"},
    }
    assert _evaluate_required_count(exact, objects, "core")["label"] == "fail"
    task = SceneTaskSpec(
        room_type="office", style="standard", required_large_objects=["shelf", "shelf"]
    )
    assert not _check_required_objects(
        task, "furniture", {"object_names": ["shelf", "bookcase"]}
    )
    assert (
        len(_check_required_objects(task, "furniture", {"object_names": ["bookcase"]}))
        == 1
    )


def test_native_selector_keeps_bookcase_specialization_and_stage_guards() -> None:
    from scenesmith.scenebenchmark_critic.intent_contract import selected_ids

    objects = [_record("shelf"), _record("bookcase")]
    assert selected_ids({"category": "bookshelf"}, objects) == ["bookcase_0"]
    assert selected_ids({"category": "bookshelf"}, objects[:1]) == []
    for stage in ("wall_mounted", "ceiling_mounted", "manipuland"):
        assert (
            selected_ids(
                {"category": "shelf", "stage": "furniture"},
                [_record("bookcase", stage)],
            )
            == []
        )
