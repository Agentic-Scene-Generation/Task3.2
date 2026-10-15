"""Preserve semantic nouns without turning arbitrary compounds into matches."""

from types import SimpleNamespace

import pytest

from scenesmith.agent_utils.room import ObjectType
from scenesmith.scenebenchmark_critic.adapter import _category_for_object
from scenesmith.scenebenchmark_critic.intent_contract import (
    selected_ids,
    semantic_selector_matches,
)
from scenesmith.scenebenchmark_critic.object_taxonomy import (
    canonical_object_category,
    semantic_category_match,
)


def record(name: str, category: str | None = None, kind: str = "furniture") -> dict:
    return {
        "id": name + "_0",
        "category": category or name,
        "category_norm": category or name,
        "metadata": {"semantic_name": name},
        "functional_hints": {"scene_object_type": kind},
    }


@pytest.mark.parametrize(
    "category,name",
    [
        ("sink", "oldschool_sink"),
        ("floor_rack", "plastic_floor_rack"),
        ("music_stand", "modern_wooden_music_stand"),
    ],
)
def test_appearance_prefix_preserves_open_category_binding(category, name):
    obj = record(name)
    assert selected_ids({"category": category}, [obj]) == [obj["id"]]
    matches = semantic_selector_matches({"category": category}, [obj])
    assert matches[0]["match_kind"] == "appearance_qualified"
    assert matches[0]["match_path"] == [name, category]
    assert canonical_object_category(name) == name  # No IR/inventory rewrite.


@pytest.mark.parametrize(
    "category,name",
    [
        ("sink", "sink_cabinet"),
        ("sink", "cabinet_with_sink"),
        ("sink", "countertop_sink"),
        ("sink", "custom_sink"),
        ("detector", "metal_detector"),
        ("floor_rack", "wall_floor_rack"),
        ("music_stand", "plastic_music_stand_holder"),
    ],
)
def test_nonappearance_compounds_do_not_gain_category_identity(category, name):
    assert not semantic_category_match(category, name)["matched"]


def test_prefixed_furniture_does_not_bypass_manipuland_stage_guard():
    obj = record("plastic_floor_rack", kind="manipuland")
    assert selected_ids({"category": "floor_rack"}, [obj]) == []


def test_known_specialization_does_not_join_existing_exact_cohort():
    objects = [record("table"), record("side_table"), record("modern_side_table")]
    assert selected_ids({"category": "table"}, objects) == ["table_0"]


@pytest.mark.parametrize("name", ["floor_rack", "plastic_floor_rack", "door_stop"])
def test_furniture_compound_is_not_a_structural_anchor(name):
    obj = SimpleNamespace(
        object_id=name + "_0",
        object_type=ObjectType.FURNITURE,
        name=name,
        description=name,
        metadata={"semantic_name": name},
    )
    assert _category_for_object(obj) == name


def test_real_structure_and_known_furniture_keep_categories():
    for name, kind, expected in [
        ("north_wall", "wall", "wall"),
        ("floor_lamp", "furniture", "floor_lamp"),
        ("wall_cabinet", "furniture", "wall_cabinet"),
    ]:
        obj = SimpleNamespace(
            object_id=name + "_0",
            object_type=ObjectType(kind),
            name=name,
            description=name,
            metadata={"semantic_name": name},
        )
        assert _category_for_object(obj) == expected


def furniture_category(name: str) -> str:
    obj = SimpleNamespace(
        object_id=name + "_0",
        object_type=ObjectType.FURNITURE,
        name=name,
        description=name,
        metadata={"semantic_name": name},
    )
    return _category_for_object(obj)


@pytest.mark.parametrize(
    "name", ["desk_chair", "modern_desk_chair", "compact_modern_desk_chair"]
)
def test_known_desk_chair_uses_existing_seating_identity(name):
    assert canonical_object_category(name) == "office_chair"
    assert furniture_category(name) == "office_chair"


@pytest.mark.parametrize(
    "name", ["desk_chair_cover", "desk_chair_holder", "desk_chair_table"]
)
def test_unknown_desk_chair_compounds_do_not_gain_seating_identity(name):
    assert canonical_object_category(name) != "office_chair"
    assert furniture_category(name) != "office_chair"


def test_optional_desk_chair_does_not_ambiguous_bind_required_desk():
    objects = [
        record("writing_desk", furniture_category("writing_desk")),
        record("desk_chair", furniture_category("desk_chair")),
    ]
    assert selected_ids({"category": "desk"}, objects) == ["writing_desk_0"]
    assert selected_ids({"category": "office_chair"}, objects) == ["desk_chair_0"]


def test_known_desk_chair_identity_preserves_stage_ownership():
    obj = record("desk_chair", furniture_category("desk_chair"), kind="manipuland")
    assert selected_ids({"category": "office_chair"}, [obj]) == []


def test_desk_chair_precedence_keeps_neighboring_known_categories():
    for name, expected in [
        ("desk", "desk"),
        ("writing_desk", "desk"),
        ("desk_lamp", "desk_lamp"),
        ("office_chair", "chair"),  # Legacy room-prefix normalization is retained.
        ("floor_lamp", "floor_lamp"),
    ]:
        assert furniture_category(name) == expected
