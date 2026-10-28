"""Preserve explicit asset identity across stage ownership guards."""

import pytest

from scenesmith.scenebenchmark_critic.intent_contract import selected_ids


def _object(category: str, semantic_name: str) -> dict:
    return {
        "id": "candidate_0",
        "name": semantic_name,
        "object_type": "manipuland",
        "category": category,
        "category_norm": category,
        "metadata": {"semantic_name": semantic_name},
    }


@pytest.mark.parametrize("semantic_name", ["plant", "succulent_plant", "ivy_plant"])
def test_explicit_plant_identity_survives_manipuland_ownership(semantic_name: str) -> None:
    """A surface-generated plant retains its existing semantic category."""
    assert selected_ids({"category": "plant"}, [_object("plant", semantic_name)]) == [
        "candidate_0"
    ]


@pytest.mark.parametrize(
    ("category", "semantic_name"),
    [("plant", ""), ("plant", "plant_holder"), ("dresser", "dresser_mirror_tabletop")],
)
def test_broad_retrieval_category_cannot_replace_asset_identity(
    category: str, semantic_name: str
) -> None:
    """An adapter parent noun alone cannot turn decor into a contract endpoint."""
    assert selected_ids({"category": category}, [_object(category, semantic_name)]) == []


def test_explicit_selector_stage_remains_authoritative() -> None:
    """Matching identity does not erase a requested furniture-stage qualifier."""
    assert selected_ids(
        {"category": "plant", "stage": "furniture"},
        [_object("plant", "succulent_plant")],
    ) == []


def test_unknown_cushion_compound_is_not_given_new_taxonomy_equivalence() -> None:
    """This fix does not introduce fuzzy matching for an unresolved identity."""
    assert selected_ids(
        {"category": "cushion"}, [_object("plush_cushion", "plush_cushion")]
    ) == []
