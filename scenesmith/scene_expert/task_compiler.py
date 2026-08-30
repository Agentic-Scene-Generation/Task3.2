"""TaskCompiler: converts a raw text prompt into a structured SceneTaskSpec.

Single Qwen3 call with role=task_compiler. Uses JSON output mode for reliability
with smaller open models.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time

from pathlib import Path
from typing import Any

from scenesmith.agent_utils.thinking import (
    chat_template_kwargs_from_effort,
    prepend_text_thinking_directive,
    thinking_directive_from_effort,
)
from scenesmith.scene_expert.context_bundle import build_llm_call_debug_record
from scenesmith.scene_expert.schemas import SceneTaskSpec
from scenesmith.scenebenchmark_critic.object_taxonomy import (
    canonical_object_category,
    generation_owner,
    is_structural_anchor,
)
from scenesmith.utils.llm_json import json_response_format, parse_llm_json_object

console_logger = logging.getLogger(__name__)


def _append_llm_debug(record: dict) -> None:
    path = os.environ.get("SCENEEXPERT_LLM_DEBUG_PATH", "")
    if not path:
        return
    try:
        debug_path = Path(path)
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        with debug_path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception as e:
        console_logger.warning("TaskCompiler failed to write LLM debug record: %s", e)


_SYSTEM_PROMPT = """\
/no_think
You are the task_compiler for SceneExpert, a 3D indoor scene generation system.
Your job is to extract structured scene requirements from a natural-language prompt.

You MUST output valid JSON matching this exact schema:
{
  "room_type": "string — primary room type (e.g. bedroom, kitchen, living room, office)",
  "style": "string — aesthetic style (e.g. cozy modern, industrial, minimalist, farmhouse)",
  "required_large_objects": ["list of furniture-scale objects that must be in the room"],
  "required_wall_objects": ["list of wall-mounted objects (paintings, mirrors, shelves, lights)"],
  "required_ceiling_objects": ["list of ceiling-mounted objects (lights, fans, sprinklers)"],
  "required_small_objects": ["list of small manipulable objects (books, cups, tools)"],
  "functional_zones": ["list of spatial zones within the room (e.g. sleeping_zone, working_zone)"],
  "interaction_constraints": [
    "constraints about robot reachability, clearance, support surfaces",
    "e.g. 'nightstand should be reachable from the accessible side of the bed'"
  ],
  "aesthetic_constraints": [
    "visual and style constraints",
    "e.g. 'modern material palette', 'balanced visual density', 'avoid overcrowding'"
  ],
}

Rules:
- Be comprehensive — extract ALL objects mentioned in the prompt.
- Classify floor-standing objects as required_large_objects, even when they are
  decorative or compact. Objects explicitly supported by the floor must never
  also appear in required_small_objects.
- Use on_wall only for objects explicitly described as mounted or hung. A
  floor-standing item that is on, against, or opposite a wall uses against_wall;
  a TV stand/media console is always floor-standing. Room-relative wall labels
  such as back_wall and opposite_wall are relation targets, never inventory.
- Infer reasonable functional zones based on the room type and objects.
- Infer reachability constraints for any small objects placed on furniture surfaces.
- Keep object names concise (e.g. "bed" not "a large king-sized bed").
- Windows, doors, and open connections are structural anchors owned by the
  floor plan. Do not put them in any required_* object array.
- Do not invent coordinates, room sides, or nearest-object identities.
- Output ONLY the JSON object, no other text.

Example input: "A bedroom with a bed, two nightstands, and a wardrobe."
Example output:
{
  "room_type": "bedroom",
  "style": "standard",
  "required_large_objects": ["bed", "nightstand", "nightstand", "wardrobe"],
  "required_wall_objects": [],
  "required_ceiling_objects": [],
  "required_small_objects": [],
  "functional_zones": ["sleeping_zone", "storage_zone"],
  "interaction_constraints": ["nightstands should be accessible from both sides of the bed"],
  "aesthetic_constraints": ["balanced furniture placement", "clear walking paths"],
}
"""


_TASK_COMPILER_WIRE_FIELDS = (
    "room_type",
    "style",
    "required_large_objects",
    "required_wall_objects",
    "required_ceiling_objects",
    "required_small_objects",
    "functional_zones",
    "interaction_constraints",
    "aesthetic_constraints",
)


def _task_compiler_wire_schema() -> dict[str, Any]:
    """Return only the fields the model is responsible for generating."""
    full = SceneTaskSpec.model_json_schema()
    properties = full.get("properties") or {}
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            field: properties[field]
            for field in _TASK_COMPILER_WIRE_FIELDS
            if field in properties
        },
        "required": list(_TASK_COMPILER_WIRE_FIELDS),
    }


_ROOM_TYPE_KEYWORDS: dict[str, list[str]] = {
    "bedroom": ["bedroom", "bed", "nightstand", "wardrobe", "sleeping"],
    "living room": ["living room", "living", "sofa", "couch", "tv", "coffee table"],
    "classroom": [
        "classroom",
        "chalkboard",
        "blackboard",
        "whiteboard",
        "student desk",
        "teacher's desk",
    ],
    "kitchen": ["kitchen", "stove", "oven", "fridge", "sink", "counter"],
    "bathroom": ["bathroom", "toilet", "bathtub", "shower", "sink"],
    "office": ["office", "desk", "chair", "computer", "monitor", "study"],
    "dining room": ["dining room", "dining", "dining table", "chairs"],
    "garage": ["garage", "car", "workbench", "tools"],
    "basement": ["basement", "laundry", "storage"],
}

_STYLE_KEYWORDS: dict[str, list[str]] = {
    "modern": ["modern", "contemporary", "sleek", "minimalist"],
    "cozy": ["cozy", "warm", "comfortable", "homey"],
    "industrial": ["industrial", "metal", "raw", "exposed"],
    "farmhouse": ["farmhouse", "rustic", "country", "wooden"],
    "scandinavian": ["scandinavian", "nordic", "simple", "functional"],
    "luxury": ["luxury", "elegant", "upscale", "premium"],
}

_NUMBER_WORDS: dict[str, int] = {
    "one": 1,
    "a": 1,
    "an": 1,
    "single": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
}

_OBJECT_ALIASES: dict[str, tuple[str, list[str], str]] = {
    "reception desk": (
        "large",
        ["reception desk", "reception desks", "reception counter"],
        "reception_desk",
    ),
    "side table": (
        "large",
        ["side table", "side tables", "end table", "end tables"],
        "side_table",
    ),
    "conference table": (
        "large",
        [
            "conference table",
            "conference tables",
            "meeting table",
            "meeting tables",
            "boardroom table",
            "boardroom tables",
        ],
        "conference_table",
    ),
    "dining table": (
        "large",
        ["dining table", "dining tables"],
        "dining_table",
    ),
    "coffee table": (
        "large",
        ["coffee table", "coffee tables"],
        "coffee_table",
    ),
    "dressing table": (
        "large",
        ["dressing table", "dressing tables", "vanity table", "makeup table"],
        "dressing_table",
    ),
    "filing cabinet": (
        "large",
        ["filing cabinet", "filing cabinets", "file cabinet"],
        "filing_cabinet",
    ),
    "office chair": (
        "large",
        ["office chair", "office chairs", "desk chair", "desk chairs"],
        "office_chair",
    ),
    "dining chair": (
        "large",
        ["dining chair", "dining chairs"],
        "dining_chair",
    ),
    "stool": ("large", ["stool", "stools", "vanity stool"], "stool"),
    "water dispenser": (
        "large",
        ["water dispenser", "water dispensers", "water cooler"],
        "water_dispenser",
    ),
    "storage cabinet": (
        "large",
        ["storage cabinet", "storage cabinets", "storage cupboard"],
        "storage_cabinet",
    ),
    "tv stand": (
        "large",
        ["tv stand", "tv stands", "television stand", "media console"],
        "tv_stand",
    ),
    "television": ("large", ["television", "televisions"], "television"),
    "rocking chair": (
        "large",
        ["rocking chair", "rocking chairs"],
        "rocking_chair",
    ),
    "guest chair": (
        "large",
        ["guest chair", "guest chairs", "visitor chair", "visitor chairs"],
        "guest_chair",
    ),
    "chalkboard": (
        "wall",
        [
            "chalkboard",
            "blackboard",
            "whiteboard",
            "projection screen",
            "projector screen",
            "teaching screen",
        ],
        "chalkboard",
    ),
    "bed": ("large", ["bed", "beds"], "bed"),
    "nightstand": (
        "large",
        ["nightstand", "nightstands", "bedside table", "bedside tables"],
        "nightstand",
    ),
    "wardrobe": ("large", ["wardrobe", "wardrobes", "closet", "closets"], "wardrobe"),
    "sofa": ("large", ["sofa", "sofas", "couch", "couches"], "sofa"),
    "desk": ("large", ["desk", "desks"], "desk"),
    "table": ("large", ["table", "tables"], "table"),
    "chair": ("large", ["chair", "chairs"], "chair"),
    "painting": ("wall", ["painting", "paintings", "artwork", "artworks"], "painting"),
    "mirror": ("wall", ["mirror", "mirrors"], "mirror"),
    "shelf": (
        "wall",
        ["shelf", "shelves", "floating shelf", "floating shelves"],
        "shelf",
    ),
    "ceiling light": (
        "ceiling",
        ["ceiling light", "ceiling lights", "pendant light", "pendant lights", "lamp"],
        "ceiling light",
    ),
    "book": ("small", ["book", "books"], "book"),
    "plant": ("large", ["plant", "plants", "floor plant", "floor plants"], "plant"),
    "monitor": (
        "small",
        ["computer monitor", "computer monitors", "monitor", "monitors"],
        "monitor",
    ),
    "printer": ("small", ["printer", "printers"], "printer"),
    "wastebasket": (
        "small",
        ["wastebasket", "wastebaskets", "trash can", "trash bin"],
        "wastebasket",
    ),
    "plate": ("small", ["plate", "plates"], "plate"),
    "cutlery": ("small", ["cutlery", "flatware", "silverware"], "cutlery"),
    "drinking glass": (
        "small",
        ["drinking glass", "drinking glasses", "glass", "glasses"],
        "glass",
    ),
    "brochure holder": (
        "small",
        ["brochure holder", "brochure holders", "leaflet holder"],
        "brochure_holder",
    ),
}

_SPECIFIC_INVENTORY_FAMILIES = {
    "reception_desk": "desk",
    "student_desk": "desk",
    "teacher_desk": "desk",
    "side_table": "table",
    "coffee_table": "table",
    "dining_table": "table",
    "conference_table": "table",
    "dressing_table": "table",
    "office_chair": "chair",
    "guest_chair": "chair",
    "student_chair": "chair",
    "dining_chair": "chair",
    "rocking_chair": "chair",
}

_INVENTORY_CATEGORY_ALIASES = {
    "computer_monitor": "monitor",
    "computer_display": "monitor",
    "vanity": "dressing_table",
    "vanity_table": "dressing_table",
    "makeup_table": "dressing_table",
    "water_cooler": "water_dispenser",
    "storage_cupboard": "storage_cabinet",
    "chalkboard": "instructional_surface",
    "blackboard": "instructional_surface",
    "whiteboard": "instructional_surface",
    "projection_screen": "instructional_surface",
    "projector_screen": "instructional_surface",
    "teaching_screen": "instructional_surface",
    "presentation_screen": "instructional_surface",
    "floor_plant": "plant",
    "large_floor_plant": "plant",
    "large_plant": "plant",
    "potted_plant": "plant",
    "table_settings": "table_setting",
    "place_setting": "table_setting",
    "place_settings": "table_setting",
}

_VIRTUAL_CATEGORIES = {
    "room",
    "wall",
    "floor",
    "ceiling",
    "entrance",
    "entry",
    "door",
    "opening",
    "window",
    "back_wall",
    "front_wall",
    "side_wall",
    "main_wall",
    "opposite_wall",
}

_WALL_STAGE_CATEGORIES = {
    "instructional_surface",
    "painting",
    "mirror",
    "wall_shelf",
    "wall_light",
}

_FURNITURE_STAGE_CATEGORIES = {
    "dressing_table",
    "plant",
    "stool",
    "storage_cabinet",
    "water_dispenser",
}

_MANIPULAND_STAGE_CATEGORIES = {
    "cutlery",
    "glass",
    "monitor",
    "plate",
}

_TABLE_SETTING_COMPONENTS = frozenset({"plate", "cutlery", "glass"})
# Media supports are furniture even when an LLM mistakes a phrase such as
# "TV stand on the opposite wall" for a wall-mounted placement.
_FLOOR_STANDING_MEDIA_SUPPORT_CATEGORIES = frozenset(
    {
        "tv_stand",
        "television_stand",
        "media_console",
        "media_cabinet",
        "entertainment_center",
    }
)


def _extract_count_before_alias(text: str, alias: str) -> int:
    """Return a conservative count for an object mention in fallback parsing."""
    alias_pattern = re.escape(alias.lower()).replace(r"\ ", r"\s+")
    if alias.lower() == "table":
        alias_pattern += r"(?!\s+(?:setting|settings)\b)"
    number_pattern = "|".join([r"\d+", *map(re.escape, _NUMBER_WORDS)])
    best = 0
    counted_pattern = (
        rf"(?<![a-z0-9])(?P<count>{number_pattern})\s+"
        rf"(?:[a-z][a-z0-9-]*\s+){{0,2}}?{alias_pattern}\b"
    )
    for match in re.finditer(counted_pattern, text):
        count_text = match.group("count")
        if count_text.isdigit():
            count = int(count_text)
        else:
            count = _NUMBER_WORDS.get(count_text, 1)
        best = max(best, count)
    if best:
        return best
    return int(re.search(rf"(?<![a-z0-9]){alias_pattern}\b", text) is not None)


def _extract_required_objects_from_prompt(prompt_lower: str) -> dict[str, list[str]]:
    """Small deterministic parser used only when the model compiler fails."""
    required = {
        "large": [],
        "wall": [],
        "ceiling": [],
        "small": [],
    }
    for _, (bucket, aliases, canonical) in _OBJECT_ALIASES.items():
        count = 0
        for alias in aliases:
            count = max(count, _extract_count_before_alias(prompt_lower, alias))
        if count > 0:
            required[bucket].extend([canonical] * count)
    for bucket in required:
        for specific, generic in _SPECIFIC_INVENTORY_FAMILIES.items():
            specific_count = required[bucket].count(specific)
            for _ in range(min(specific_count, required[bucket].count(generic))):
                required[bucket].remove(generic)
    return required


def _aliases_for_canonical(canonical: str) -> list[str]:
    """Return prompt aliases for an inventory category."""
    for _name, (_bucket, aliases, value) in _OBJECT_ALIASES.items():
        if value == canonical:
            return aliases
    return [canonical.replace("_", " ")]


def _prompt_mentions_standalone_generic(
    prompt: str, *, specific: str, generic: str
) -> bool:
    """Whether a generic item remains after removing a specific family phrase."""
    remainder = str(prompt or "").lower().replace("_", " ")
    for alias in _aliases_for_canonical(specific):
        alias_pattern = re.escape(alias.lower()).replace(r"\ ", r"\s+")
        remainder = re.sub(
            rf"(?<![a-z0-9]){alias_pattern}(?:s|es)?(?![a-z0-9])",
            " ",
            remainder,
        )
    for alias in _aliases_for_canonical(generic):
        alias_pattern = re.escape(alias.lower()).replace(r"\ ", r"\s+")
        for match in re.finditer(
            rf"(?<![a-z0-9]){alias_pattern}(?:s|es)?(?![a-z0-9])", remainder
        ):
            # Later references such as ``the table`` and ``this chair`` name
            # an already-introduced specific object. They must not preserve a
            # second generic inventory entry beside ``conference_table`` or
            # another typed family member.
            before = remainder[: match.start()].rstrip()
            if re.search(
                r"\b(?:the|this|that|these|those|its|their)(?:\s+[a-z]+){0,2}$",
                before,
            ):
                continue
            if _extract_count_before_alias(remainder, alias) > 0:
                return True
    return False


def _remove_spurious_generic_inventory_entries(
    inventories: dict[str, list[str]], *, prompt: str
) -> None:
    """Undo model inventory overlap such as ``rocking_chair`` plus ``chair``.

    A broad family item remains valid when the immutable prompt also mentions a
    standalone generic item (for example, "a rocking chair and a chair").
    """
    if not prompt:
        return
    for values in inventories.values():
        normalized_values = [
            "_".join(str(value or "").strip().lower().split()) for value in values
        ]
        for specific, generic in _SPECIFIC_INVENTORY_FAMILIES.items():
            specific_count = normalized_values.count(specific)
            if (
                specific_count == 0
                or generic not in normalized_values
                or _prompt_mentions_standalone_generic(
                    prompt, specific=specific, generic=generic
                )
            ):
                continue
            remaining = specific_count
            retained: list[str] = []
            for value, normalized in zip(values, normalized_values, strict=True):
                if normalized == generic and remaining > 0:
                    remaining -= 1
                    continue
                retained.append(value)
            values[:] = retained
            normalized_values = [
                "_".join(str(value or "").strip().lower().split()) for value in values
            ]


def _extract_json_from_text(text: str) -> dict:
    """Extract a top-level object and repair common local-model JSON drift.

    TaskCompiler must use the shared parser rather than strict ``json.loads``:
    local model output is occasionally fenced, has a trailing comma, or is
    cut off after a complete nested object.  Schema validation below remains
    the authority on whether the recovered payload is usable.
    """
    return parse_llm_json_object(text)


def _repair_zero_target_relation_payloads(data: dict) -> dict:
    """Return a copied inventory payload for callers of the retired helper.

    Hard relation repair was removed from TaskCompiler in v3.  The name remains
    as a no-op import compatibility shim for downstream tooling; no relation
    data is read or written here.
    """
    return dict(data)


def _normalize_stage_ownership(
    task_spec: SceneTaskSpec, *, prompt: str = ""
) -> SceneTaskSpec:
    """Keep each inventory category in one structurally appropriate stage.

    Functional relations are intentionally absent here.  Their authoritative
    representation is the independent critic contract, not TaskCompiler data.
    """

    def inventory_key(value: str) -> str:
        key = canonical_object_category(value)
        return _INVENTORY_CATEGORY_ALIASES.get(key, key)

    inventories = {
        "large": list(task_spec.required_large_objects),
        "wall": list(task_spec.required_wall_objects),
        "ceiling": list(task_spec.required_ceiling_objects),
        "small": list(task_spec.required_small_objects),
    }
    _remove_spurious_generic_inventory_entries(inventories, prompt=prompt)
    for values in inventories.values():
        values[:] = [
            value
            for value in values
            if inventory_key(value) not in _VIRTUAL_CATEGORIES
            and not is_structural_anchor(inventory_key(value))
        ]

    # The model may emit aliases as separate inventories (for example four
    # ``floor plant`` rows plus four ``plant`` rows). Repeated identical names
    # are real cardinality, while alias-equivalent groups describe the same set.
    alias_counts: dict[str, dict[str, int]] = {}
    for values in inventories.values():
        for value in values:
            raw_key = "_".join(str(value or "").strip().lower().split())
            category = inventory_key(value)
            aliases = alias_counts.setdefault(category, {})
            aliases[raw_key] = aliases.get(raw_key, 0) + 1
    for category, counts in alias_counts.items():
        if len(counts) <= 1:
            continue
        desired_count = max(counts.values())
        owning_stage = next(
            stage
            for stage, values in inventories.items()
            if any(inventory_key(value) == category for value in values)
        )
        for stage, values in inventories.items():
            inventories[stage] = [
                value for value in values if inventory_key(value) != category
            ]
        inventories[owning_stage].extend([category] * desired_count)

    for values in inventories.values():
        values[:] = [inventory_key(value) for value in values]

    physical_categories = {
        inventory_key(value) for values in inventories.values() for value in values
    }
    if "table_setting" in physical_categories and (
        physical_categories & _TABLE_SETTING_COMPONENTS
    ):
        for stage, values in inventories.items():
            inventories[stage] = [
                value for value in values if inventory_key(value) != "table_setting"
            ]
    existing_stage: dict[str, str] = {}
    for stage, values in inventories.items():
        for value in values:
            existing_stage.setdefault(inventory_key(value), stage)

    category_stages: dict[str, str] = dict(existing_stage)
    category_counts: dict[str, int] = {}
    for values in inventories.values():
        for value in values:
            category = inventory_key(value)
            category_counts[category] = category_counts.get(category, 0) + 1
    for category in _WALL_STAGE_CATEGORIES:
        if category in category_stages:
            category_stages[category] = "wall"
    for category in _FURNITURE_STAGE_CATEGORIES:
        if category in category_stages:
            category_stages[category] = "large"
    for category in _MANIPULAND_STAGE_CATEGORIES:
        if category in category_stages and category_stages[category] not in {
            "wall",
            "ceiling",
        }:
            category_stages[category] = "small"
    for category in _FLOOR_STANDING_MEDIA_SUPPORT_CATEGORIES:
        if category in category_stages:
            category_stages[category] = "large"
    task_stage_to_owner = {
        "large": "furniture",
        "wall": "wall_mounted",
        "ceiling": "ceiling_mounted",
        "small": "manipuland",
    }
    owner_to_task_stage = {owner: stage for stage, owner in task_stage_to_owner.items()}
    for category, stage in list(category_stages.items()):
        category_stages[category] = owner_to_task_stage[
            generation_owner(
                category,
                declared_owner=task_stage_to_owner.get(stage, ""),
            )
        ]
    desired_counts: dict[str, int] = {}
    for category in (
        _WALL_STAGE_CATEGORIES
        | _FURNITURE_STAGE_CATEGORIES
        | _MANIPULAND_STAGE_CATEGORIES
        | {"wastebasket"}
    ):
        inventory_count = sum(
            inventory_key(value) == category
            for values in inventories.values()
            for value in values
        )
        if inventory_count:
            desired_counts[category] = max(
                desired_counts.get(category, 0), inventory_count
            )
    for category in _FLOOR_STANDING_MEDIA_SUPPORT_CATEGORIES:
        inventory_count = sum(
            inventory_key(value) == category
            for values in inventories.values()
            for value in values
        )
        if inventory_count:
            desired_counts[category] = max(
                desired_counts.get(category, 0), inventory_count
            )

    for category in desired_counts:
        category_stages.setdefault(category, existing_stage.get(category, "large"))

    # Move each known category to its single owning stage before filling counts.
    for stage, values in inventories.items():
        inventories[stage] = [
            value
            for value in values
            if category_stages.get(inventory_key(value), stage) == stage
        ]

    for category, desired_count in desired_counts.items():
        stage = category_stages[category]
        values = inventories[stage]
        current = sum(
            inventory_key(value) == category
            or _SPECIFIC_INVENTORY_FAMILIES.get(inventory_key(value)) == category
            for value in values
        )
        family = _SPECIFIC_INVENTORY_FAMILIES.get(category)
        while current < desired_count and family:
            generic_index = next(
                (
                    index
                    for index, value in enumerate(values)
                    if inventory_key(value) == family
                ),
                None,
            )
            if generic_index is None:
                break
            values[generic_index] = category
            current += 1
        values.extend([category] * max(0, desired_count - current))

    return task_spec.model_copy(
        update={
            "required_large_objects": inventories["large"],
            "required_wall_objects": inventories["wall"],
            "required_ceiling_objects": inventories["ceiling"],
            "required_small_objects": inventories["small"],
        }
    )


def _task_spec_normalization_warnings(
    raw_spec: SceneTaskSpec, normalized_spec: SceneTaskSpec
) -> list[str]:
    warnings: list[str] = []
    for field in (
        "required_large_objects",
        "required_wall_objects",
        "required_ceiling_objects",
        "required_small_objects",
    ):
        before = list(getattr(raw_spec, field))
        after = list(getattr(normalized_spec, field))
        if before != after:
            warnings.append(f"Normalized {field}: {before!r} -> {after!r}")
    return warnings


def _fallback_spec_from_prompt(prompt: str) -> SceneTaskSpec:
    """Parse room_type and style from prompt text when model call fails."""
    prompt_lower = prompt.lower()

    room_type = "room"
    for rtype, keywords in _ROOM_TYPE_KEYWORDS.items():
        if any(kw in prompt_lower for kw in keywords):
            room_type = rtype
            break

    style = "standard"
    for stype, keywords in _STYLE_KEYWORDS.items():
        if any(kw in prompt_lower for kw in keywords):
            style = stype
            break

    required = _extract_required_objects_from_prompt(prompt_lower)
    functional_zones: list[str] = []
    if room_type == "bedroom" or any(
        obj in required["large"] for obj in ("bed", "nightstand", "wardrobe")
    ):
        functional_zones.extend(["sleeping_zone", "storage_zone"])
    if any(obj in required["large"] for obj in ("table", "chair")):
        functional_zones.append("working_or_dining_zone")

    interaction_constraints: list[str] = []
    if "bed" in required["large"] and "nightstand" in required["large"]:
        interaction_constraints.append(
            "nightstands should flank the bed and remain reachable from the bed"
        )
    if "wardrobe" in required["large"]:
        interaction_constraints.append(
            "wardrobe doors should have clear access and should not block the room door"
        )

    console_logger.info(
        "TaskCompiler: fallback spec from prompt text: room_type=%s, style=%s, "
        "large_objects=%s",
        room_type,
        style,
        required["large"],
    )
    return _normalize_stage_ownership(
        SceneTaskSpec(
            room_type=room_type,
            style=style,
            required_large_objects=required["large"],
            required_wall_objects=required["wall"],
            required_ceiling_objects=required["ceiling"],
            required_small_objects=required["small"],
            functional_zones=functional_zones,
            interaction_constraints=interaction_constraints,
            aesthetic_constraints=["balanced placement", "clear walking paths"],
            compiler_status="degraded",
        ),
        prompt=prompt,
    )


class TaskCompiler:
    """Converts a raw text prompt to a structured SceneTaskSpec via Qwen3."""

    def __init__(
        self,
        model: str,
        api_base_url: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 1536,
        temperature: float = 0.0,
        llm_client: Any | None = None,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._structured_llm = llm_client
        self._client = None
        if self._structured_llm is None:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=api_base_url
                or os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"),
                api_key=api_key or os.environ.get("OPENAI_API_KEY", "dummy"),
            )
        self.last_trace: dict = {}

    def compile(self, prompt: str) -> SceneTaskSpec:
        """Parse a raw text prompt into a SceneTaskSpec.

        Args:
            prompt: Natural-language scene description.

        Returns:
            Structured SceneTaskSpec.

        Raises:
            ValueError: If the model response cannot be parsed.
        """
        console_logger.info(f"TaskCompiler: compiling prompt: {prompt[:100]}...")
        user_message = f"Extract scene requirements from: {prompt}"
        base_messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
        structured_llm = getattr(self, "_structured_llm", None)
        if structured_llm is not None:
            result = structured_llm.complete(
                role="task_compiler",
                stage="task_compiler",
                event="compile",
                messages=base_messages,
                response_model=SceneTaskSpec,
            )
            if result.value is None:
                reason = (f"{result.final_error_kind}: {result.final_error}").strip(
                    ": "
                )
                task_spec = _fallback_spec_from_prompt(prompt).model_copy(
                    update={"compiler_failure_reason": reason}
                )
                self.last_trace = {
                    "status": "fallback",
                    "attempts": [attempt.model_dump() for attempt in result.attempts],
                    "retry_count": max(0, len(result.attempts) - 1),
                    "failure_reason": reason,
                    "normalized_task_spec": task_spec.model_dump(
                        mode="json", exclude_none=True
                    ),
                    "normalization_warnings": [],
                }
                console_logger.warning(
                    "Structured TaskCompiler failed; using deterministic contract: %s",
                    reason,
                )
                return task_spec

            raw_task_spec = result.value
            task_spec = _normalize_stage_ownership(
                raw_task_spec, prompt=prompt
            ).model_copy(
                update={"compiler_status": "ok", "compiler_failure_reason": ""}
            )
            normalization_warnings = _task_spec_normalization_warnings(
                raw_task_spec, task_spec
            )
            self.last_trace = {
                "status": "ok",
                "attempts": [attempt.model_dump() for attempt in result.attempts],
                "retry_count": max(0, len(result.attempts) - 1),
                "failure_reason": "",
                "raw_task_spec": raw_task_spec.model_dump(
                    mode="json", exclude_none=True
                ),
                "normalized_task_spec": task_spec.model_dump(
                    mode="json", exclude_none=True
                ),
                "normalization_warnings": normalization_warnings,
            }
            return task_spec

        attempts: list[dict] = []
        previous_output = ""
        validation_error = ""
        for attempt in range(2):
            messages = [
                {
                    "role": "system",
                    "content": prepend_text_thinking_directive(
                        _SYSTEM_PROMPT,
                        thinking_directive_from_effort("none", model=self._model),
                    ),
                },
                {"role": "user", "content": user_message},
            ]
            if attempt:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "The previous candidate failed validation. Return a corrected "
                            "JSON object only.\nValidation error: "
                            f"{validation_error}\nPrevious candidate:\n{previous_output}"
                        ),
                    }
                )
            started_at = time.perf_counter()
            raw = ""
            response = None
            response_elapsed_sec: float | None = None
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                    response_format=json_response_format(
                        model=self._model,
                        name="scene_task_spec",
                        schema=_task_compiler_wire_schema(),
                    ),
                    extra_body=chat_template_kwargs_from_effort(
                        "none", model=self._model
                    ),
                )
                response_elapsed_sec = round(time.perf_counter() - started_at, 6)
                message = response.choices[0].message
                raw = message.content
                if not raw:
                    raw = getattr(message, "reasoning_content", None)
                if not raw:
                    extra = getattr(message, "model_extra", None)
                    if isinstance(extra, dict):
                        raw = extra.get("reasoning_content")
                data = _extract_json_from_text(raw)
                raw_task_spec = SceneTaskSpec.model_validate(data)
                task_spec = _normalize_stage_ownership(
                    raw_task_spec, prompt=prompt
                ).model_copy(
                    update={"compiler_status": "ok", "compiler_failure_reason": ""}
                )
                normalization_warnings = _task_spec_normalization_warnings(
                    raw_task_spec, task_spec
                )
                attempts.append(
                    {
                        "attempt": attempt,
                        "status": "ok",
                        "elapsed_sec": round(time.perf_counter() - started_at, 6),
                    }
                )
                self.last_trace = {
                    "status": "ok",
                    "attempts": attempts,
                    "retry_count": attempt,
                    "failure_reason": "",
                    "raw_task_spec": raw_task_spec.model_dump(
                        mode="json", exclude_none=True
                    ),
                    "normalized_task_spec": task_spec.model_dump(
                        mode="json", exclude_none=True
                    ),
                    "normalization_warnings": normalization_warnings,
                }
                record = build_llm_call_debug_record(
                    stage="task_compiler",
                    agent_role="task_compiler",
                    event="compile",
                    prompt=messages,
                    output=raw or "",
                    raw_response=response,
                ).model_dump()
                record.update(
                    {
                        "input": messages,
                        "output": raw or "",
                        "elapsed_sec": response_elapsed_sec,
                        "status": "ok",
                        "attempt": attempt,
                        "raw_task_spec": raw_task_spec.model_dump(
                            mode="json", exclude_none=True
                        ),
                        "normalized_task_spec": task_spec.model_dump(
                            mode="json", exclude_none=True
                        ),
                        "normalization_warnings": normalization_warnings,
                    }
                )
                _append_llm_debug(record)
                return task_spec
            except Exception as exc:
                validation_error = f"{type(exc).__name__}: {exc}"
                previous_output = raw
                elapsed = round(time.perf_counter() - started_at, 6)
                attempts.append(
                    {
                        "attempt": attempt,
                        "status": "error",
                        "error": validation_error,
                        "elapsed_sec": elapsed,
                    }
                )
                record = build_llm_call_debug_record(
                    stage="task_compiler",
                    agent_role="task_compiler",
                    event="compile",
                    prompt=messages,
                    output=raw,
                    raw_response=response,
                    error=validation_error,
                ).model_dump()
                record.update(
                    {
                        "input": messages,
                        "output": raw,
                        "elapsed_sec": (
                            response_elapsed_sec
                            if response_elapsed_sec is not None
                            else elapsed
                        ),
                        "status": "error",
                        "attempt": attempt,
                    }
                )
                _append_llm_debug(record)

        attempts.append(
            {"attempt": 2, "status": "deterministic_fallback", "elapsed_sec": 0.0}
        )
        self.last_trace = {
            "status": "fallback",
            "attempts": attempts,
            "retry_count": 1,
            "failure_reason": validation_error,
        }
        fallback = _fallback_spec_from_prompt(prompt).model_copy(
            update={"compiler_failure_reason": validation_error}
        )
        console_logger.warning(
            "TaskCompiler failed twice; using deterministic contract: %s",
            validation_error,
        )
        return fallback
