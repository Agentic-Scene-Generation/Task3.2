"""Keep shelf spelling and instance labels within their original semantic owner."""

import pytest

from scenesmith.scenebenchmark_critic.object_taxonomy import (
    canonical_object_category,
    generation_owner,
    semantic_category_match,
)


@pytest.mark.parametrize(
    "label",
    [
        "shelf",
        "shelve",
        "shelves",
        "tall_shelve",
        "wooden_shelves",
        "shelf2",
        "shelve2",
        "shelf_0",
    ],
)
def test_shelf_spelling_and_instance_labels_preserve_identity(label: str) -> None:
    assert canonical_object_category(label) == "shelf"
    assert semantic_category_match("shelf", label)["matched"]
    assert generation_owner(label) == "furniture"


@pytest.mark.parametrize("label", ["wall_shelf", "wall_shelf2", "wall_shelf_0"])
def test_wall_storage_keeps_its_category_and_execution_stage(label: str) -> None:
    assert canonical_object_category(label) == "wall_shelf"
    assert generation_owner(label) == "wall_mounted"
    assert not semantic_category_match("shelf", label)["matched"]


@pytest.mark.parametrize(
    "label",
    [
        "shelf_cover",
        "shelve_holder",
        "shelf_support2",
        "desk_shelve",
        "wall_shelve",
        "wall_shelve2",
        "vase_shelves",
    ],
)
def test_compound_objects_do_not_gain_plain_shelf_identity(label: str) -> None:
    assert canonical_object_category(label) != "shelf"
    assert not semantic_category_match("shelf", label)["matched"]


@pytest.mark.parametrize(
    "label,category,owner",
    [
        ("side_table2", "side_table", "furniture"),
        ("table_lamp2", "table_lamp", "manipuland"),
        ("ceiling_fan2", "ceiling_fan", "ceiling_mounted"),
        ("monitor_0", "monitor", "manipuland"),
    ],
)
def test_registered_indexed_names_keep_existing_specialization_and_owner(
    label: str, category: str, owner: str
) -> None:
    assert canonical_object_category(label) == category
    assert generation_owner(label) == owner


@pytest.mark.parametrize(
    "label", ["r2d2", "area51", "custom_storage2", "shelf_support2"]
)
def test_unregistered_numbered_names_remain_open_vocabulary(label: str) -> None:
    assert canonical_object_category(label) == label


def test_numbered_wall_storage_can_only_change_owner_with_explicit_support() -> None:
    assert generation_owner("wall_shelf2", declared_owner="furniture") == "wall_mounted"
    assert (
        generation_owner("wall_shelf2", relation="on_top_of", target_category="table")
        == "manipuland"
    )
