"""Shared object naming and execution-owner rules for critic integrations.

Scene generation has several consumers of object names: task inventory,
intent selectors, stage verification, and critic case-pack adapters.  This
module keeps their semantic identity and stage ownership rules in one place.
"""

from __future__ import annotations

import re

from typing import Any

from scenesmith.scenebenchmark_critic.relation_registry import (
    CEILING_MOUNTED_CATEGORIES,
    MANIPULAND_CATEGORIES,
    WALL_MOUNTED_CATEGORIES,
)


_CATEGORY_ALIASES = {
    "entrance_route": "entrance",
    "entrance_path": "entrance",
    "entry_route": "entrance",
    "entry_path": "entrance",
    "vanity": "dressing_table",
    "vanity_table": "dressing_table",
    "makeup_table": "dressing_table",
    "computer_display": "monitor",
    "computer_monitor": "monitor",
    "display_monitor": "monitor",
    "water_cooler": "water_dispenser",
    "drinking_water_dispenser": "water_dispenser",
    "storage_cupboard": "storage_cabinet",
    "dining_chairs": "dining_chair",
    "large_plants": "large_plant",
    "floor_plant": "plant",
    "floor_plants": "plant",
    "large_floor_plant": "plant",
    "large_floor_plants": "plant",
    "two_seater_sofas": "two_seater_sofa",
    "centerpiece_vase": "vase",
    "centerpiece_vases": "vase",
    "vases": "vase",
    "coasters": "coaster",
    "plates": "plate",
    "glasses": "glass",
    "wine_glasses": "glass",
    "drinking_glasses": "glass",
    "cutleries": "cutlery",
    "flatware": "cutlery",
    "silverware": "cutlery",
    "fork": "cutlery",
    "forks": "cutlery",
    "knife": "cutlery",
    "knives": "cutlery",
    "spoon": "cutlery",
    "spoons": "cutlery",
    "wine_glass": "glass",
    "drinking_glass": "glass",
    "water_glass": "glass",
    "tumbler": "glass",
    "tv": "television",
    "tv_display": "television",
    "television_display": "television",
    "table_settings": "table_setting",
    "place_settings": "table_setting",
    "place_setting": "table_setting",
    "flowers": "flower",
    "chalkboard": "instructional_surface",
    "blackboard": "instructional_surface",
    "whiteboard": "instructional_surface",
    "projection_screen": "instructional_surface",
    "projector_screen": "instructional_surface",
    "teaching_screen": "instructional_surface",
    "presentation_screen": "instructional_surface",
}

# These are equivalence classes, not aliases.  Keeping the concrete category
# preserves useful role data (for example, a desk lamp is not renamed in the
# scene), while inventory verification can still satisfy the requested concept.
_EQUIVALENT_CATEGORY_GROUPS = (
    frozenset({"table_lamp", "desk_lamp", "reading_lamp", "bedside_lamp"}),
    frozenset({"wastebasket", "trash_can", "trash_bin"}),
)

# These objects normally stand on the floor unless an explicit support relation
# says otherwise.  The set is intentionally semantic rather than prompt- or
# room-specific, so it applies to future office, bedroom, and public-room tasks.
_FLOOR_STANDING_DEFAULTS = frozenset(
    {
        "wastebasket",
        "trash_can",
        "trash_bin",
        "water_dispenser",
        "plant",
        "tv_stand",
        "television_stand",
        "media_console",
        "media_cabinet",
        "entertainment_center",
    }
)
_SURFACE_SUPPORT_RELATIONS = frozenset({"on_top_of"})
_FLOOR_SUPPORT_RELATIONS = frozenset({"on_floor", "object_on_floor"})
_WALL_MOUNT_RELATIONS = frozenset({"mounted_on_wall", "hung_on_wall"})
_CEILING_MOUNT_RELATIONS = frozenset({"mounted_on_ceiling", "hung_from_ceiling"})
_EXECUTION_STAGES = frozenset(
    {"furniture", "wall_mounted", "ceiling_mounted", "manipuland"}
)


def _singularize_category(normalized: str) -> str:
    if normalized.endswith("ies") and len(normalized) > 3:
        return f"{normalized[:-3]}y"
    if normalized.endswith(("ches", "shes", "xes", "zes", "sses")):
        return normalized[:-2]
    if normalized.endswith("s") and not normalized.endswith(("ss", "us", "is")):
        return normalized[:-1]
    return normalized


def canonical_object_category(value: Any) -> str:
    """Normalize prompt, selector, and asset labels to a stable category."""
    normalized = str(value or "").strip().lower()
    normalized = re.sub(r"(?<=[a-z])['\u2019]s\b", "", normalized)
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
    return _CATEGORY_ALIASES.get(normalized, _singularize_category(normalized))


def categories_are_equivalent(first: Any, second: Any) -> bool:
    """Return whether two labels satisfy the same requested object concept."""
    first_category = canonical_object_category(first)
    second_category = canonical_object_category(second)
    if not first_category or not second_category:
        return False
    if first_category == second_category:
        return True
    return any(
        first_category in group and second_category in group
        for group in _EQUIVALENT_CATEGORY_GROUPS
    )


def execution_owner(
    category: Any,
    *,
    relation: str = "",
    endpoint: str = "subject",
    existing_owner: str = "",
) -> str:
    """Return the pipeline stage that must create an object category.

    Explicit support/mounting relations take precedence over category defaults.
    An existing inventory owner is only a fallback; it must not make a normally
    floor-standing object become a manipuland merely because an upstream model
    emitted it in the wrong inventory list.
    """
    normalized_category = canonical_object_category(category)
    normalized_relation = str(relation or "").strip().lower().replace("-", "_")
    normalized_endpoint = str(endpoint or "subject").strip().lower()

    if normalized_endpoint == "subject":
        if normalized_relation in _SURFACE_SUPPORT_RELATIONS:
            return "manipuland"
        if normalized_relation in _FLOOR_SUPPORT_RELATIONS:
            return "furniture"
        if normalized_relation in _WALL_MOUNT_RELATIONS:
            return "wall_mounted"
        if normalized_relation in _CEILING_MOUNT_RELATIONS:
            return "ceiling_mounted"

    if normalized_category in _FLOOR_STANDING_DEFAULTS:
        return "furniture"
    if normalized_category in WALL_MOUNTED_CATEGORIES:
        return "wall_mounted"
    if normalized_category in CEILING_MOUNTED_CATEGORIES:
        return "ceiling_mounted"
    if normalized_category in MANIPULAND_CATEGORIES:
        return "manipuland"
    if existing_owner in _EXECUTION_STAGES:
        return existing_owner
    return "furniture"
