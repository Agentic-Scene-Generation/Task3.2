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
    STAGE_ORDER,
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
    "loudspeaker": "speaker",
    "loudspeakers": "speaker",
}

# Ordered from specialized to generic so prompt parsing and descriptor
# stripping resolve the same noun phrase. This is the shared semantic
# vocabulary for TaskCompiler inventory and intent relation selectors.
OBJECT_CATEGORY_PHRASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("window", ("window",)),
    ("door", ("door",)),
    ("opening", ("opening", "open connection")),
    (
        "instructional_surface",
        (
            "chalkboard",
            "blackboard",
            "whiteboard",
            "projection screen",
            "projector screen",
            "teaching screen",
        ),
    ),
    ("student_desk", ("student desk",)),
    ("teacher_desk", ("teacher desk", "instructor desk")),
    ("reception_desk", ("reception desk", "reception counter")),
    ("office_chair", ("office chair", "desk chair", "task chair")),
    (
        "guest_chair",
        ("guest chair", "visitor chair", "guest armchair", "visitor armchair"),
    ),
    ("dining_chair", ("dining chair",)),
    ("sofa_chair", ("sofa chair",)),
    ("rocking_chair", ("rocking chair",)),
    ("armchair", ("armchair", "arm chair")),
    ("dining_table", ("dining table",)),
    (
        "conference_table",
        ("conference table", "meeting table", "boardroom table"),
    ),
    ("coffee_table", ("coffee table",)),
    ("side_table", ("side table", "end table", "accent table")),
    (
        "dressing_table",
        ("dressing table", "vanity table", "makeup table", "vanity"),
    ),
    ("filing_cabinet", ("filing cabinet", "file cabinet")),
    ("storage_cabinet", ("storage cabinet", "storage cupboard")),
    (
        "water_dispenser",
        ("water dispenser", "water cooler", "drinking water dispenser"),
    ),
    ("tv_stand", ("tv stand", "television stand", "media console")),
    (
        "media_cabinet",
        (
            "media cabinet",
            "floating media cabinet",
            "media unit",
            "entertainment cabinet",
        ),
    ),
    ("television", ("television", "tv")),
    (
        "monitor",
        ("computer monitor", "computer display", "monitor", "display", "screen"),
    ),
    ("brochure_holder", ("brochure holder", "leaflet holder", "brochure stand")),
    ("printer", ("printer",)),
    ("nightstand", ("nightstand", "bedside table")),
    ("stool", ("stool", "vanity stool", "dressing stool")),
    ("bookshelf", ("bookshelf", "bookcase", "shelving unit")),
    ("sideboard", ("sideboard", "buffet")),
    ("wardrobe", ("wardrobe", "closet", "armoire")),
    ("dresser", ("dresser", "chest of drawers", "chest of drawer", "bureau")),
    ("floor_lamp", ("floor lamp",)),
    (
        "floor_speaker",
        ("floor speaker", "floor-standing speaker", "speaker tower"),
    ),
    ("speaker", ("speaker", "loudspeaker")),
    ("table_lamp", ("table lamp", "desk lamp")),
    ("vase", ("vase",)),
    ("flower", ("flower", "flowers")),
    ("alarm_clock", ("alarm clock",)),
    ("plate", ("plate",)),
    ("cutlery", ("cutlery", "fork", "knife", "spoon")),
    ("glass_bowl", ("glass bowl",)),
    ("glass", ("glass", "drinking glass", "wine glass")),
    ("coaster", ("coaster",)),
    ("book", ("book",)),
    ("wastebasket", ("wastebasket", "waste basket", "trash can", "trash bin")),
    ("bottle", ("bottle",)),
    ("bowl", ("bowl",)),
    ("cup", ("cup",)),
    ("mug", ("mug",)),
    ("keyboard", ("keyboard",)),
    ("laptop", ("laptop",)),
    ("remote", ("remote", "remote control")),
    ("rug", ("rug", "carpet", "area rug")),
    ("mirror", ("mirror",)),
    ("table_setting", ("table setting", "place setting")),
    ("plant", ("plant",)),
    ("bed", ("bed",)),
    ("sofa", ("sofa", "couch", "settee")),
    ("desk", ("desk",)),
    ("chair", ("chair", "seat")),
    ("table", ("table",)),
    ("floor", ("floor",)),
)

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
_SUPPORT_SENSITIVE_CATEGORIES = frozenset({"plant"})
_FLOOR_SUPPORT_RELATIONS = frozenset({"on_floor", "object_on_floor"})
_WALL_MOUNT_RELATIONS = frozenset({"mounted_on_wall", "hung_on_wall"})
_CEILING_MOUNT_RELATIONS = frozenset({"mounted_on_ceiling", "hung_from_ceiling"})
_EXECUTION_STAGES = frozenset(
    {"furniture", "wall_mounted", "ceiling_mounted", "manipuland"}
)

STRUCTURAL_ANCHOR_CATEGORIES = frozenset({"door", "opening", "window"})


def is_structural_anchor(category: Any) -> bool:
    """Return whether a category is generated by the floor-plan stage."""
    return canonical_object_category(category) in STRUCTURAL_ANCHOR_CATEGORIES


def _singularize_category(normalized: str) -> str:
    if normalized.endswith("ies") and len(normalized) > 3:
        return f"{normalized[:-3]}y"
    if normalized.endswith(("ches", "shes", "xes", "zes", "sses")):
        return normalized[:-2]
    if normalized.endswith("s") and not normalized.endswith(("ss", "us", "is")):
        return normalized[:-1]
    return normalized


def _phrase_category_map() -> dict[str, str]:
    result: dict[str, str] = {}
    for category, aliases in OBJECT_CATEGORY_PHRASES:
        for value in (category, *aliases):
            normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
            result[_singularize_category(normalized)] = category
    return result


_PHRASE_CATEGORY_MAP = _phrase_category_map()
_KNOWN_CATEGORIES = frozenset(
    set(_CATEGORY_ALIASES.values())
    | set(_PHRASE_CATEGORY_MAP.values())
    | set(WALL_MOUNTED_CATEGORIES)
    | set(CEILING_MOUNTED_CATEGORIES)
    | set(MANIPULAND_CATEGORIES)
    | {"door", "opening", "room", "wall", "window"}
)


def is_known_object_category(category: Any) -> bool:
    """Return whether taxonomy has an intrinsic policy for this category."""
    return canonical_object_category(category) in _KNOWN_CATEGORIES


def _token_is_known_category(token: str) -> bool:
    singular = _singularize_category(token)
    return (
        singular in _CATEGORY_ALIASES
        or singular in _PHRASE_CATEGORY_MAP
        or singular in _KNOWN_CATEGORIES
    )


def canonical_object_category(value: Any) -> str:
    """Resolve a label to the longest known semantic object category."""
    normalized = str(value or "").strip().lower()
    normalized = re.sub(r"(?<=[a-z])['\u2019]s\b", "", normalized)
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
    if not normalized:
        return ""
    alias = _CATEGORY_ALIASES.get(normalized)
    if alias is not None:
        return alias

    singular = _singularize_category(normalized)
    alias = _CATEGORY_ALIASES.get(singular)
    if alias is not None:
        return alias
    phrase_category = _PHRASE_CATEGORY_MAP.get(singular)
    if phrase_category is not None:
        return phrase_category
    if singular in _KNOWN_CATEGORIES:
        return singular

    # Descriptor-heavy inventory labels put shape, size, material, or style
    # before the semantic noun (for example ``circular_ceramic_table``). Match
    # only a known suffix. Matching arbitrary interior tokens would collapse
    # real compound categories such as ``wall_cabinet`` or ``vase_flowers``.
    tokens = singular.split("_")
    for width in range(len(tokens), 0, -1):
        start = len(tokens) - width
        phrase = "_".join(tokens[start:])
        phrase_category = _PHRASE_CATEGORY_MAP.get(phrase)
        if phrase_category is None and phrase in _KNOWN_CATEGORIES:
            phrase_category = phrase
        if phrase_category is None:
            continue
        if any(_token_is_known_category(token) for token in tokens[:start]):
            continue
        return phrase_category
    return singular


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


def generation_owner(
    category: Any,
    *,
    relation: str = "",
    endpoint: str = "subject",
    declared_owner: str = "",
) -> str:
    """Return the stage that creates an object, independently of check timing.

    Explicit mounting relations take precedence over category defaults. Surface
    support only overrides a declared inventory for categories with a reliable
    intrinsic policy. Open-vocabulary objects otherwise retain their typed
    TaskCompiler inventory, so an unknown small object cannot be promoted to
    furniture merely because its support already exists there.
    """
    normalized_category = canonical_object_category(category)
    normalized_relation = str(relation or "").strip().lower().replace("-", "_")
    normalized_endpoint = str(endpoint or "subject").strip().lower()

    if normalized_category in STRUCTURAL_ANCHOR_CATEGORIES:
        return "floor_plan"

    if normalized_endpoint == "subject":
        if normalized_relation in _SURFACE_SUPPORT_RELATIONS:
            # ``on_top_of`` describes both manipulands on furniture (a lamp on
            # a desk) and structural furniture support (a television on a TV
            # stand). The latter must remain at the furniture stage so the
            # support-pose repair can place it in XYZ, not as a loose floor
            # object deferred to the manipuland agent.
            if normalized_category in (
                MANIPULAND_CATEGORIES | _SUPPORT_SENSITIVE_CATEGORIES
            ):
                return "manipuland"
            if (
                declared_owner == "manipuland"
                and normalized_category not in _KNOWN_CATEGORIES
            ):
                return "manipuland"
            return "furniture"
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
    if declared_owner in _EXECUTION_STAGES:
        return declared_owner
    return "furniture"


def constraint_evaluation_stage(*endpoint_owners: str) -> str:
    """Return the first stage at which every relation endpoint can exist."""
    owners = [owner for owner in endpoint_owners if owner in STAGE_ORDER]
    if not owners:
        return "furniture"
    return max(owners, key=STAGE_ORDER.index)


def execution_owner(
    category: Any,
    *,
    relation: str = "",
    endpoint: str = "subject",
    existing_owner: str = "",
) -> str:
    """Compatibility wrapper for the retired combined ownership helper."""
    return generation_owner(
        category,
        relation=relation,
        endpoint=endpoint,
        declared_owner=existing_owner,
    )
