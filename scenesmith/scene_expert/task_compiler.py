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

from scenesmith.scene_expert.context_bundle import build_llm_call_debug_record
from scenesmith.scene_expert.schemas import IntentConstraintSpec, SceneTaskSpec
from scenesmith.scenebenchmark_critic.relation_registry import (
    PUBLIC_RELATIONS,
    ROOM_RELATIVE_WALL_CATEGORIES,
    WALL_MOUNTED_CATEGORIES,
)
from scenesmith.agent_utils.thinking import chat_template_kwargs_from_effort
from scenesmith.utils.llm_json import parse_llm_json_object

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
  "intent_constraints": [
    {
      "relation": "one of the registered relation names listed below",
      "subjects": {"category": "canonical object category", "count": 1},
      "targets": {"category": "canonical target category", "secondary_category": "second target category for between relations only"},
      "source": "explicit_prompt | model_inferred",
      "confidence": 0.0,
      "evidence_span": "exact prompt words for explicit_prompt, otherwise empty",
      "inference_reason": "short functional reason for model_inferred, otherwise empty"
    }
  ]
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
- Always include "intent_constraints". Emit every atomic relation explicitly
  stated in the input plus reasonable common-sense functional relations needed
  for the named objects to work. All emitted relations become hard constraints.
- Use source "explicit_prompt" and a verbatim evidence_span for directly stated
  relations. Use source "model_inferred" and a concrete inference_reason for
  common-sense relations. Confidence is audit metadata only.
- Use only these registered relations: __REGISTERED_RELATIONS__.
- Do not invent coordinates, room sides, or nearest-object identities.
- For explicit wording such as "the rug in front of the sofa", emit
  "in_front_of" with the rug as subject and sofa as target. Do not add this
  relation for merely nearby objects.
- For explicit "X behind Y", emit "behind" with X as subject and Y as target.
- For "X centered between A and B", emit "centered_between" with A as
  targets.category and B as targets.secondary_category. For ordinary
  "X between A and B", emit "between" using the same two-target shape.
- For "X between the two As", repeat A in targets.category and
  targets.secondary_category; the contract binder resolves the two instances.
- For a clear route from an entrance to an object, `entrance` is a virtual
  clear_access subject and the destination object is the target.
- Keep every selector minimal: use category and count only when a count is
  explicit. Omit role and stage fields. When a relational target means one or
  any member of a repeated category, set count to 1 and quantifier to "minimum".
- Emit each relation atomically. Do not cap the number of constraints and do not
  combine independent relations into one row.
- Copy evidence_span verbatim from the input prompt. Do not use StageBrief,
  retrieved memory, or current object positions as evidence.
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
  "intent_constraints": []
}
"""

_SYSTEM_PROMPT = _SYSTEM_PROMPT.replace(
    "__REGISTERED_RELATIONS__", ", ".join(sorted(PUBLIC_RELATIONS))
)

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
    "plant": ("small", ["plant", "plants"], "plant"),
    "monitor": (
        "small",
        ["computer monitor", "computer monitors", "monitor", "monitors"],
        "monitor",
    ),
    "printer": ("small", ["printer", "printers"], "printer"),
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
    "office_chair": "chair",
    "guest_chair": "chair",
    "student_chair": "chair",
    "dining_chair": "chair",
    "rocking_chair": "chair",
}

_INVENTORY_CATEGORY_ALIASES = {
    "computer_monitor": "monitor",
    "computer_display": "monitor",
    "chalkboard": "instructional_surface",
    "blackboard": "instructional_surface",
    "whiteboard": "instructional_surface",
    "projection_screen": "instructional_surface",
    "projector_screen": "instructional_surface",
    "teaching_screen": "instructional_surface",
}

_VIRTUAL_CATEGORIES = {
    "room",
    "wall",
    "floor",
    "ceiling",
    "entrance",
    "entry",
    *ROOM_RELATIVE_WALL_CATEGORIES,
}

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

_FURNITURE_RELATIONS = {
    "across_from",
    "against_wall",
    "aligned_with",
    "between",
    "behind",
    "centered_between",
    "centered_in_room",
    "centered_on_wall",
    "clear_access",
    "corner_of_room",
    "distributed_evenly",
    "faces",
    "flanking",
    "in_front_of",
    "near",
    "next_to",
    "one_per_side",
    "operation_zone_at_wall",
    "paired_with",
    "surround",
}


def _extract_count_before_alias(text: str, alias: str) -> int:
    """Return a conservative count for an object mention in fallback parsing."""
    alias_pattern = re.escape(alias.lower()).replace(r"\ ", r"\s+")
    number_pattern = "|".join([r"\d+", *map(re.escape, _NUMBER_WORDS)])
    pattern = (
        rf"(?:(?P<count>{number_pattern})\s+)?" rf"(?:\w+\s+){{0,2}}{alias_pattern}\b"
    )
    best = 0
    for match in re.finditer(pattern, text):
        count_text = match.groupdict().get("count")
        if not count_text:
            count = 1
        elif count_text.isdigit():
            count = int(count_text)
        else:
            count = _NUMBER_WORDS.get(count_text, 1)
        best = max(best, count)
    return best


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
    return any(
        _extract_count_before_alias(remainder, alias) > 0
        for alias in _aliases_for_canonical(generic)
    )


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


def _normalize_stage_ownership(
    task_spec: SceneTaskSpec, *, prompt: str = ""
) -> SceneTaskSpec:
    """Keep structurally floor-positioned objects in furniture inventory only."""

    authored_constraints = list(task_spec.intent_constraints)
    fallback_constraints: list[IntentConstraintSpec] = []
    if prompt:
        fallback_constraints = [
            IntentConstraintSpec.model_validate(row)
            for row in _fallback_intent_constraints(prompt)
        ]
    constraints = [*authored_constraints, *fallback_constraints]

    def inventory_key(value: str) -> str:
        key = "_".join(str(value or "").strip().lower().split())
        return _INVENTORY_CATEGORY_ALIASES.get(key, key)

    explicit_required_counts: dict[str, int] = {}
    for constraint in constraints:
        if constraint.relation != "required_count":
            continue
        category = inventory_key(constraint.subjects.category)
        explicit_required_counts[category] = max(
            explicit_required_counts.get(category, 0), constraint.subjects.count or 1
        )

    inventories = {
        "large": list(task_spec.required_large_objects),
        "wall": list(task_spec.required_wall_objects),
        "ceiling": list(task_spec.required_ceiling_objects),
        "small": list(task_spec.required_small_objects),
    }
    _remove_spurious_generic_inventory_entries(inventories, prompt=prompt)
    for values in inventories.values():
        values[:] = [
            value for value in values if inventory_key(value) not in _VIRTUAL_CATEGORIES
        ]
    existing_stage: dict[str, str] = {}
    for stage, values in inventories.items():
        for value in values:
            existing_stage.setdefault(inventory_key(value), stage)

    # Repair an LLM's stage error without changing a valid mounted-object
    # relation. The support category is semantic knowledge, not prompt wording:
    # a TV stand remains furniture even when the prompt says it is "on" a wall.
    normalized_authored_constraints: list[IntentConstraintSpec] = []
    for constraint in authored_constraints:
        subject_category = inventory_key(constraint.subjects.category)
        if (
            constraint.relation == "on_wall"
            and subject_category in _FLOOR_STANDING_MEDIA_SUPPORT_CATEGORIES
        ):
            normalized_authored_constraints.append(
                constraint.model_copy(update={"relation": "against_wall"})
            )
        else:
            normalized_authored_constraints.append(constraint)
    constraints = [*normalized_authored_constraints, *fallback_constraints]

    category_stages: dict[str, str] = dict(existing_stage)
    category_counts: dict[str, int] = {}
    for values in inventories.values():
        for value in values:
            category = inventory_key(value)
            category_counts[category] = category_counts.get(category, 0) + 1
    for category in WALL_MOUNTED_CATEGORIES:
        if category in category_stages:
            category_stages[category] = "wall"
    for category in _FLOOR_STANDING_MEDIA_SUPPORT_CATEGORIES:
        if category in category_stages:
            category_stages[category] = "large"
    desired_counts: dict[str, int] = dict(explicit_required_counts)
    for category in WALL_MOUNTED_CATEGORIES:
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

    for constraint in constraints:
        target = constraint.targets
        subject_category = inventory_key(constraint.subjects.category)
        target_category = inventory_key(target.category) if target else ""
        evidence = str(constraint.evidence_span or "").lower()
        target_family = _SPECIFIC_INVENTORY_FAMILIES.get(target_category, "")
        target_words = {
            target_category.replace("_", " "),
            target_family.replace("_", " "),
        } - {""}
        target_is_explicit_plural = any(
            re.search(
                rf"\b{re.escape(word)}(?:s|es)\b",
                evidence,
            )
            for word in target_words
        )
        if (
            target is not None
            and category_counts.get(target_category, 0) > 1
            and target_is_explicit_plural
        ):
            target.count = category_counts[target_category]
            target.quantifier = "all"
        secondary_category = (
            inventory_key(target.secondary_category) if target else ""
        )
        if (
            target is not None
            and constraint.relation in {"between", "centered_between"}
            and target_category
            and target_category == secondary_category
            and category_counts.get(target_category, 0) == 2
        ):
            target.count = 2
            target.quantifier = "all"
            target.secondary_count = 2
        if subject_category in _VIRTUAL_CATEGORIES:
            continue
        subject_stage = ""
        if constraint.relation == "on_wall":
            subject_stage = "wall"
        elif constraint.relation == "hang_from_ceiling":
            subject_stage = "ceiling"
        elif constraint.relation == "on_top_of":
            subject_stage = "large" if target_category == "floor" else "small"
            if target_category not in _VIRTUAL_CATEGORIES:
                category_stages.setdefault(target_category, "large")
        elif constraint.relation in _FURNITURE_RELATIONS:
            subject_stage = "large"
            if target_category not in _VIRTUAL_CATEGORIES:
                category_stages.setdefault(target_category, "large")
        if subject_stage:
            if constraint.relation in {
                "centered_in_room",
                "corner_of_room",
                "hang_from_ceiling",
                "on_top_of",
                "on_wall",
            }:
                category_stages[subject_category] = subject_stage
            else:
                category_stages.setdefault(subject_category, subject_stage)

        subject_desired_count = max(
            constraint.subjects.count or 1,
            explicit_required_counts.get(subject_category, 0),
        )
        target_desired_count = (
            max(
                target.count or 1,
                explicit_required_counts.get(target_category, 0),
            )
            if target_category and target_category not in _VIRTUAL_CATEGORIES
            else 0
        )
        if (
            constraint.relation == "paired_with"
            and target is not None
            and constraint.subjects.quantifier == "all"
            and target.quantifier == "all"
        ):
            # ``paired_with`` is one-to-one. Models often preserve an explicit
            # count on only one endpoint (for example, six student desks but an
            # uncounted ``all`` selector for student chairs). Propagate the
            # known common cardinality before upgrading generic family items.
            # Conflicting endpoint counts remain untouched so binding reports
            # the inconsistency instead of silently rewriting user intent.
            paired_counts = {
                int(value)
                for value in (
                    constraint.subjects.count,
                    explicit_required_counts.get(subject_category),
                    target.count,
                    explicit_required_counts.get(target_category),
                )
                if value is not None and int(value) > 0
            }
            if len(paired_counts) == 1:
                paired_count = next(iter(paired_counts))
                subject_desired_count = max(subject_desired_count, paired_count)
                target_desired_count = max(target_desired_count, paired_count)

        desired_counts[subject_category] = max(
            desired_counts.get(subject_category, 0), subject_desired_count
        )
        if target_category and target_category not in _VIRTUAL_CATEGORIES:
            desired_counts[target_category] = max(
                desired_counts.get(target_category, 0), target_desired_count
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
            "intent_constraints": normalized_authored_constraints,
        }
    )


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
    intent_constraints = _fallback_intent_constraints(prompt)
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
            intent_constraints=intent_constraints,
            compiler_status="degraded",
        ),
        prompt=prompt,
    )


def _fallback_intent_constraints(prompt: str) -> list[dict[str, object]]:
    """Build a small deterministic v2 contract without model-authored data."""
    from scenesmith.scenebenchmark_critic.intent_contract import (
        _explicit_prompt_constraints,
        _explicit_required_count_constraints,
        _normalize_room_type,
        _room_ontology_constraints,
    )

    normalized = " ".join(str(prompt or "").split())
    lowered = normalized.lower()
    room_type = _normalize_room_type("", lowered)
    rows = [
        *_explicit_required_count_constraints(normalized),
        *_explicit_prompt_constraints(normalized, lowered),
        *_room_ontology_constraints(room_type, lowered),
    ]
    result: list[dict[str, object]] = []
    for row in rows:
        source = str(row.get("source") or "explicit_prompt")
        if source not in {"explicit_prompt", "model_inferred"}:
            source = "model_inferred"
        result.append(
            {
                "relation": row["relation"],
                "subjects": row["subjects"],
                "targets": row.get("targets") or None,
                "source": source,
                "confidence": row.get("confidence", 1.0),
                "evidence_span": (
                    row.get("evidence_span", "") if source == "explicit_prompt" else ""
                ),
                "inference_reason": (
                    str(row.get("evidence_span") or "deterministic room ontology")
                    if source == "model_inferred"
                    else ""
                ),
            }
        )
    return result


class TaskCompiler:
    """Converts a raw text prompt to a structured SceneTaskSpec via Qwen3."""

    def __init__(
        self,
        model: str,
        api_base_url: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 1536,
        temperature: float = 0.0,
    ) -> None:
        from openai import OpenAI

        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._client = OpenAI(
            base_url=api_base_url
            or os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"),
            api_key=api_key or os.environ.get("OPENAI_API_KEY", "dummy"),
        )

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
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
        started_at = time.perf_counter()

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                extra_body=chat_template_kwargs_from_effort("none"),
            )
        except Exception as exc:
            record = build_llm_call_debug_record(
                stage="task_compiler",
                agent_role="task_compiler",
                event="compile",
                prompt=messages,
                error=f"{type(exc).__name__}: {exc}",
            ).model_dump()
            record.update(
                {
                    "input": messages,
                    "output": "",
                    "elapsed_sec": round(time.perf_counter() - started_at, 6),
                    "status": "error",
                }
            )
            _append_llm_debug(record)
            fallback = _fallback_spec_from_prompt(prompt).model_copy(
                update={"compiler_failure_reason": f"{type(exc).__name__}: {exc}"}
            )
            console_logger.warning(
                "TaskCompiler model call failed; using deterministic contract: %s", exc
            )
            return fallback

        message = response.choices[0].message
        raw = message.content
        # Qwen3 with --reasoning-parser may put output in reasoning_content.
        if not raw:
            raw = getattr(message, "reasoning_content", None)
        if not raw:
            extra = getattr(message, "model_extra", None)
            if isinstance(extra, dict):
                raw = extra.get("reasoning_content")
        console_logger.debug(f"TaskCompiler raw response: {raw}")
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
                "elapsed_sec": round(time.perf_counter() - started_at, 6),
                "status": "ok",
            }
        )
        usage = getattr(response, "usage", None)
        if usage is not None:
            usage_payload = (
                usage.model_dump() if hasattr(usage, "model_dump") else vars(usage)
            )
            record["token_usage"] = {
                str(key): int(value)
                for key, value in usage_payload.items()
                if isinstance(value, int)
            }

        try:
            data = _extract_json_from_text(raw)
            task_spec = _normalize_stage_ownership(
                SceneTaskSpec.model_validate(data), prompt=prompt
            ).model_copy(
                update={"compiler_status": "ok", "compiler_failure_reason": ""}
            )
            _append_llm_debug(record)
            console_logger.info(
                f"TaskCompiler: room_type={task_spec.room_type}, style={task_spec.style}, "
                f"large_objects={task_spec.required_large_objects}"
            )
            return task_spec
        except Exception as e:
            record["status"] = "error"
            record["error"] = f"{type(e).__name__}: {e}"
            _append_llm_debug(record)
            fallback = _fallback_spec_from_prompt(prompt).model_copy(
                update={"compiler_failure_reason": f"{type(e).__name__}: {e}"}
            )
            console_logger.warning(
                "TaskCompiler output failed v2 validation; using deterministic "
                "contract: %s",
                e,
            )
            return fallback
