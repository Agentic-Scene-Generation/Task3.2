"""Synthetic category regressions; these never establish real asset function."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from scenesmith.scenebenchmark_critic.adapter import ObjectType, _category_for_object
from scenesmith.scenebenchmark_critic.intent_contract import selector_match_count
from scenesmith.scenebenchmark_critic.object_taxonomy import canonical_object_category


def _object(name: str, *, kind: ObjectType = ObjectType.FURNITURE) -> SimpleNamespace:
    return SimpleNamespace(
        object_type=kind, object_id=f"synthetic_{name}_0", name=name,
        description="", metadata={"semantic_name": name},
    )


def _record(obj: SimpleNamespace) -> dict:
    category = _category_for_object(obj)
    return {
        "id": obj.object_id, "name": obj.name, "category": category,
        "category_norm": category, "object_type": obj.object_type.value,
        "metadata": obj.metadata,
    }


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("display_table", "table"), ("display_table_2", "table"),
        ("laptop_table", "table"), ("tablet_table", "table"),
        ("desk_chair", "office_chair"),
        ("display_shelf", "display_shelf"), ("display_cabinet", "display_cabinet"),
    ],
)
def test_adapter_semantic_category_outranks_interior_device_token(name, expected):
    obj = _object(name)
    before = deepcopy(obj.metadata)
    assert canonical_object_category(name) == expected
    assert _category_for_object(obj) == expected
    assert obj.metadata == before


@pytest.mark.parametrize(
    "name", ["display_table_laptop", "display_table_tablet", "headphone_display_stand", "laptop_stand"],
)
def test_adapter_unknown_furniture_compound_is_not_a_device(name):
    obj = _object(name)
    assert _category_for_object(obj) == name
    for device in ["laptop", "television", "monitor"]:
        assert selector_match_count({"category": device}, [_record(obj)]) == 0


@pytest.mark.parametrize(
    "names", [("display_table", "display_table_2"), ("laptop_table", "tablet_table")],
)
def test_adapter_two_named_table_units_count_without_creating_devices(names):
    rows = [_record(_object(name)) for name in names]
    assert selector_match_count({"category": "table"}, rows) == 2
    assert selector_match_count({"category": "laptop"}, rows) == 0
    assert selector_match_count({"category": "tablet"}, rows) == 0


def test_adapter_table_lamp_and_real_laptop_keep_their_own_identity():
    lamp = _record(_object("table_lamp", kind=ObjectType.MANIPULAND))
    laptop = _record(_object("laptop", kind=ObjectType.MANIPULAND))
    assert selector_match_count({"category": "table"}, [lamp, laptop]) == 0
    assert selector_match_count({"category": "laptop"}, [lamp, laptop]) == 1


def test_adapter_unregistered_nondevice_and_structural_compounds_stay_unchanged():
    assert _category_for_object(_object("desk_chair_holder")) == "desk"
    assert _category_for_object(_object("floor_rack")) == "floor_rack"


def test_adapter_requested_stand_label_is_not_generic_display_stand_proof():
    stand = _record(_object("headphone_display_stand"))
    assert selector_match_count({"category": "display_stand"}, [stand]) == 0


def test_adapter_nondevice_legacy_room_prefix_categories_stay_unchanged():
    for name, expected in [
        ("desk", "desk"), ("writing_desk", "desk"),
        ("desk_lamp", "desk_lamp"), ("office_chair", "chair"),
        ("floor_lamp", "floor_lamp"), ("vanity_table", "table"),
    ]:
        assert _category_for_object(_object(name)) == expected
