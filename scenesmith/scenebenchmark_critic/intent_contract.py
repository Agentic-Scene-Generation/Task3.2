"""Prompt-originated, geometry-grounded critic intent contracts.

The critic used to infer hard layout requirements from the current scene and
from the (possibly memory-augmented) agent prompt.  That makes an evaluator
both invent and judge a requirement, which is especially fragile during scene
replay.  This module keeps the requirement source explicit and serializable:

``original prompt -> semantic selector -> object-id binding -> geometry check``.

No function in this module returns a pose.  LLM/VLM observations may enrich a
contract's evidence, but only explicit prompt clauses and the small room
ontology below are allowed to be hard constraints.
"""

from __future__ import annotations

import hashlib
import math
import re

from typing import Any, Iterable

from scenesmith.scenebenchmark_critic.core.geometry import (
    bbox_center_xy,
    object_category,
)


SCHEMA_VERSION = "scenesmith.intent_contract.v1"
VALID_MODES = frozenset({"legacy", "shadow", "contract"})
VALID_RELATIONS = frozenset(
    {
        "required_count",
        "against_wall",
        "centered_on_wall",
        "centered_in_room",
        "centered_between",
        "between",
        "in_front_of",
        "flanking",
        "faces",
        "on_top_of",
        "near",
        "aligned_with",
        "paired_with",
        "distributed_evenly",
        "one_per_side",
        "clear_access",
    }
)
HARD_SOURCES = frozenset({"explicit_prompt", "room_ontology"})

_CATEGORY_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("student_desk", ("student desk",)),
    ("teacher_desk", ("teacher desk", "instructor desk")),
    ("office_chair", ("office chair", "desk chair", "task chair")),
    (
        "guest_chair",
        ("guest chair", "visitor chair", "guest armchair", "visitor armchair"),
    ),
    ("dining_chair", ("dining chair",)),
    ("armchair", ("armchair", "arm chair")),
    ("dining_table", ("dining table",)),
    ("coffee_table", ("coffee table",)),
    ("tv_stand", ("tv stand", "television stand", "media console")),
    ("television", ("television", "tv")),
    ("monitor", ("computer monitor", "monitor", "display", "screen")),
    ("nightstand", ("nightstand", "bedside table")),
    ("bookshelf", ("bookshelf", "bookcase", "shelving unit")),
    ("sideboard", ("sideboard", "buffet")),
    ("wardrobe", ("wardrobe", "closet", "armoire")),
    ("dresser", ("dresser", "chest of drawers", "bureau")),
    ("floor_lamp", ("floor lamp",)),
    ("table_lamp", ("table lamp", "desk lamp")),
    ("rug", ("rug", "carpet", "area rug")),
    ("plant", ("plant",)),
    ("bed", ("bed",)),
    ("sofa", ("sofa", "couch", "settee")),
    ("desk", ("desk",)),
    ("chair", ("chair", "seat")),
    ("table", ("table",)),
)

_NUMBER_WORDS = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
}

_SCENE_EXPERT_INJECTION_MARKERS = (
    "=== SceneExpert Stage Brief:",
    "=== SceneExpert Retrieved Memory Directives ===",
    "=== Reference Layout (",
)

_MEDIA_SUPPORT_PATTERN = re.compile(
    r"\b(?:tv stand|television stand|media console|media cabinet|entertainment center)\b",
    re.IGNORECASE,
)
_TELEVISION_PATTERN = re.compile(
    r"\b(?:tv|television)\b(?!\s+(?:stand|console|cabinet))", re.IGNORECASE
)
_EXPLICIT_WALL_MOUNTED_TELEVISION_PATTERN = re.compile(
    r"(?:\b(?:wall[- ]mounted|mounted|hung|hanging)\s+"
    r"(?:flat[- ]screen\s+)?(?:tv|television)\b)"
    r"|(?:\b(?:tv|television)\s+(?:is\s+)?(?:mounted|hung|hanging)\s+"
    r"(?:on|against)\s+(?:the\s+)?(?:opposite\s+)?wall\b)"
    r"|(?:\b(?:tv|television)\s+(?:is\s+)?(?:on|against)\s+"
    r"(?:the\s+)?(?:opposite\s+)?wall\b)",
    re.IGNORECASE,
)
_MEDIA_GROUP_ON_WALL_PATTERN = re.compile(
    r"\b(?:tv stand|television stand|media console|media cabinet|entertainment center)"
    r"\s+and\s+(?:a\s+|an\s+|the\s+)?(?:tv|television)\s+"
    r"(?:is\s+)?(?:on|against)\s+(?:the\s+)?(?:opposite\s+)?wall\b",
    re.IGNORECASE,
)


def constraint_mode(config: Any | None) -> str:
    """Return the configured rollout mode without accepting unknown values."""
    if config is not None and hasattr(config, "constraint_mode"):
        value = getattr(config, "constraint_mode")
    else:
        extra = getattr(config, "extra", None)
        if isinstance(extra, dict):
            value = extra.get("constraint_mode", "legacy")
        elif isinstance(config, dict):
            value = config.get("constraint_mode", "legacy")
        else:
            value = "legacy"
    mode = str(value or "legacy").strip().lower()
    return mode if mode in VALID_MODES else "legacy"


def original_prompt_for_scene(scene: Any) -> str:
    """Read the immutable prompt before SceneExpert/memory prompt injection."""
    original = getattr(scene, "scene_expert_original_description", "")
    if original:
        return str(original).strip()

    # RoomScene checkpoints serialize text_description but not SceneExpert's
    # dynamic provenance attributes.  Only remove blocks with explicit system
    # markers so ordinary prompt wording cannot be mistaken for injected text.
    text = str(getattr(scene, "text_description", "") or "")
    marker_offsets = [
        match.start()
        for marker in _SCENE_EXPERT_INJECTION_MARKERS
        if (match := re.search(rf"(?m)^\s*{re.escape(marker)}", text)) is not None
    ]
    if marker_offsets:
        text = text[: min(marker_offsets)]
    return text.strip()


def build_intent_contract(
    prompt: str,
    *,
    room_type: str = "",
    task_spec: Any | None = None,
) -> dict[str, Any]:
    """Compile a conservative semantic contract from the original prompt.

    The deterministic parser is deliberately recall-limited: an unrecognised
    wording produces no hard relation instead of silently inventing a layout.
    A SceneExpert TaskCompiler result may add structured clauses, but only after
    schema/provenance validation.
    """
    normalized_prompt = " ".join(str(prompt or "").split())
    lowered = normalized_prompt.lower()
    normalized_room = _normalize_room_type(room_type, lowered)
    constraints: list[dict[str, Any]] = []

    for raw in _task_spec_constraints(task_spec):
        normalized = _normalize_external_constraint(raw)
        if normalized is not None:
            constraints.append(normalized)

    constraints.extend(_explicit_required_count_constraints(normalized_prompt))
    constraints.extend(_explicit_prompt_constraints(normalized_prompt, lowered))
    constraints.extend(_room_ontology_constraints(normalized_room, lowered))
    constraints = _deduplicate_constraints(constraints)
    return {
        "schema_version": SCHEMA_VERSION,
        "prompt": normalized_prompt,
        "prompt_sha256": hashlib.sha256(normalized_prompt.encode("utf-8")).hexdigest(),
        "room_type": normalized_room,
        "constraints": constraints,
    }


def attach_intent_contract_to_case_pack(
    scene: Any, case_pack: dict[str, Any]
) -> dict[str, Any]:
    """Attach a cached contract to a room case pack and preserve original text."""
    prompt = original_prompt_for_scene(scene)
    task_spec = getattr(scene, "scene_expert_task_spec", None)
    room_type = str(getattr(scene, "room_type", "") or case_pack.get("room_type") or "")
    prompt_hash = hashlib.sha256(" ".join(prompt.split()).encode("utf-8")).hexdigest()
    cached = getattr(scene, "scenebenchmark_intent_contract", None)
    if not (
        isinstance(cached, dict)
        and cached.get("schema_version") == SCHEMA_VERSION
        and cached.get("prompt_sha256") == prompt_hash
    ):
        cached = build_intent_contract(prompt, room_type=room_type, task_spec=task_spec)
        setattr(scene, "scenebenchmark_intent_contract", cached)
    case_pack["original_task_instruction"] = prompt
    case_pack["intent_contract"] = _copy_contract(cached)
    return case_pack["intent_contract"]


def set_contract_mode(case_pack: dict[str, Any], mode: str) -> None:
    resolved_mode = str(mode or "legacy").strip().lower()
    case_pack["intent_contract_mode"] = (
        resolved_mode if resolved_mode in VALID_MODES else "legacy"
    )
    contract = case_pack.get("intent_contract")
    if isinstance(contract, dict):
        contract["mode"] = case_pack["intent_contract_mode"]


def contract_constraints(
    case_pack: dict[str, Any],
    *,
    relations: Iterable[str] | None = None,
    include_auxiliary: bool = True,
) -> list[dict[str, Any]]:
    contract = case_pack.get("intent_contract") or {}
    rows = contract.get("constraints") if isinstance(contract, dict) else []
    allowed = {str(item) for item in relations} if relations is not None else None
    result: list[dict[str, Any]] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        relation = str(row.get("relation") or "")
        if allowed is not None and relation not in allowed:
            continue
        if not include_auxiliary and not is_hard_constraint(row):
            continue
        result.append(row)
    return result


def is_hard_constraint(constraint: dict[str, Any]) -> bool:
    return (
        str(constraint.get("strength") or "auxiliary").lower() == "hard"
        and str(constraint.get("source") or "").lower() in HARD_SOURCES
    )


def contract_relation_requested(case_pack: dict[str, Any], *relations: str) -> bool:
    """Whether a relation is authorized to enable a contract-mode hard rule.

    Auxiliary model/VLM observations can be evaluated and reported, but must
    not activate a legacy layout rule or grant a repair target.
    """
    return bool(
        contract_constraints(
            case_pack,
            relations=relations,
            include_auxiliary=False,
        )
    )


def selected_ids(
    selector: dict[str, Any] | None,
    objects: Iterable[dict[str, Any]],
) -> list[str]:
    """Resolve a semantic selector only from object identity/annotations.

    Resolution intentionally does not use current XY pose.  If a single object
    is requested and several indistinguishable candidates exist, callers retain
    all candidates or downgrade the relation instead of guessing a pose-derived
    identity.
    """
    if not isinstance(selector, dict):
        return []
    category = str(selector.get("category") or "").lower()
    role = str(selector.get("role") or "").lower()
    ids: list[str] = []
    for obj in objects:
        if not isinstance(obj, dict) or not obj.get("id"):
            continue
        if _selector_matches_object(category, role, obj):
            ids.append(str(obj["id"]))
    return sorted(dict.fromkeys(ids))


def bound_ids(
    selector: dict[str, Any] | None,
    objects: Iterable[dict[str, Any]],
) -> list[str]:
    """Resolve a relation endpoint only when its requested cardinality is safe.

    A prompt saying ``an office chair`` must not silently bind every chair-like
    asset in a room.  Group/count evaluators can use :func:`selected_ids`, but
    relation binding deliberately degrades to no hard check when the semantic
    selector cannot be matched at the requested cardinality.
    """
    ids = selected_ids(selector, objects)
    if not isinstance(selector, dict):
        return ids
    try:
        count = int(selector.get("count"))
    except (TypeError, ValueError):
        count = 0
    quantifier = str(selector.get("quantifier") or "all").lower()
    if count > 0 and quantifier not in {"at_least", "minimum"} and len(ids) != count:
        return []
    return ids


def selector_for_phrase(
    value: str, *, count: int | None = None
) -> dict[str, Any] | None:
    text = " ".join(str(value or "").lower().replace("_", " ").split())
    if not text:
        return None
    category = ""
    # Role-bearing plural phrases are frequent in prompts and should not rely
    # on exact singular aliases (``two guest chairs`` vs ``guest chair``).
    if ("guest" in text or "visitor" in text) and any(
        token in text for token in ("chair", "seat", "armchair")
    ):
        category = "guest_chair"
    elif "student" in text and "desk" in text:
        category = "student_desk"
    elif "student" in text and any(token in text for token in ("chair", "seat")):
        category = "student_chair"
    elif ("teacher" in text or "instructor" in text) and "desk" in text:
        category = "teacher_desk"
    else:
        for candidate, aliases in _CATEGORY_PATTERNS:
            if any(_contains_phrase(text, alias) for alias in aliases):
                category = candidate
                break
    if not category:
        return None
    role = ""
    if "student" in text:
        role = "student"
    elif "teacher" in text or "instructor" in text:
        role = "teacher"
    elif "guest" in text or "visitor" in text:
        role = "guest"
    inferred_count = count if count is not None else _leading_count(text)
    if inferred_count is None and re.match(r"\s*(?:a|an)\b", text):
        inferred_count = 1
    if inferred_count is None and not _phrase_mentions_plural(text, category):
        # Object references without a number/article are normally singular
        # (for example ``faces desk`` after article stripping).  Recording the
        # cardinality lets binding decline an ambiguous multi-desk scene rather
        # than selecting a target based on current geometry or ID order.
        inferred_count = 1
    selector: dict[str, Any] = {"category": category, "quantifier": "all"}
    if role:
        selector["role"] = role
    if inferred_count is not None and inferred_count > 0:
        selector["count"] = int(inferred_count)
    return selector


def augment_contract_checks(case_pack: dict[str, Any]) -> bool:
    """Add contract-grounded FD checks in shadow/contract modes.

    Shadow checks use the ignored tier.  They remain in the JSON report for
    comparison but cannot alter an agent prompt, gate, or repair acceptance.
    """
    mode = str(case_pack.get("intent_contract_mode") or "legacy")
    if mode == "legacy":
        return False
    geometry = case_pack.get("scene_geometry") or {}
    objects = [
        item
        for item in geometry.get("objects") or []
        if isinstance(item, dict) and item.get("id")
    ]
    if not objects:
        return False
    existing = {
        str(check.get("check_id") or "")
        for check in case_pack.get("checks") or []
        if isinstance(check, dict)
    }
    added = False
    checks = list(case_pack.get("checks") or [])
    shadow_ids: list[str] = []
    for constraint in contract_constraints(case_pack):
        if str(constraint.get("relation") or "") == "paired_with":
            paired_checks = _paired_seating_checks(
                constraint,
                case_pack=case_pack,
                objects=objects,
                mode=mode,
            )
            for check in paired_checks:
                check_id = str(check["check_id"])
                if check_id in existing:
                    continue
                checks.append(check)
                existing.add(check_id)
                added = True
                if mode == "shadow":
                    shadow_ids.append(check_id)
            continue
        relation_type = _fd_relation_for_constraint(constraint, objects)
        if relation_type is None:
            continue
        subject_ids = bound_ids(constraint.get("subjects"), objects)
        target_ids = bound_ids(constraint.get("targets"), objects)
        if relation_type == "back_against_wall":
            # Direction words such as "back" and "side" are room-relative,
            # not reliable generated IDs.  Bind each subject to its present
            # closest wall instead of choosing an arbitrary lexicographic wall.
            target_ids = []
        if not subject_ids or (relation_type != "back_against_wall" and not target_ids):
            continue
        for subject_id in subject_ids:
            candidate_targets = (
                _nearest_wall_ids([subject_id], objects)
                if relation_type == "back_against_wall"
                else target_ids
            )
            compatible_targets = [
                target for target in candidate_targets if target != subject_id
            ]
            if not compatible_targets:
                continue
            # A relation can legitimately point to one work surface/media item.
            # Keep the semantic binding stable instead of choosing by current pose.
            target_id = compatible_targets[0]
            check_id = (
                f"intent_contract__{constraint['constraint_id']}__"
                f"{subject_id}__{target_id}"
            )
            if check_id in existing:
                continue
            tier = _constraint_tier(constraint, mode)
            checks.append(
                {
                    "check_id": check_id,
                    "metric": "functional_dependency",
                    "subject_id": subject_id,
                    "target_ids": [target_id],
                    "relation_type": relation_type,
                    "expected_use": _expected_use(constraint, relation_type),
                    "check_source": "intent_contract",
                    "scoring_tier": tier,
                    "evidence": {
                        "intent_constraint": constraint,
                        "constraint_mode": mode,
                    },
                }
            )
            existing.add(check_id)
            added = True
            if mode == "shadow":
                shadow_ids.append(check_id)
    if added:
        case_pack["checks"] = checks
    contract = case_pack.get("intent_contract")
    if isinstance(contract, dict):
        contract["shadow_check_ids"] = shadow_ids
    return added


def contract_seating_targets(case_pack: dict[str, Any]) -> dict[str, set[str]]:
    """Return target ids allowed for direct seating-orientation repair.

    This is used before the final critic runs, where the old nearest-surface
    guard otherwise had no way to know whether a chair was supposed to face a
    desk, television, or simply face into the room from a wall.
    """
    geometry = case_pack.get("scene_geometry") or {}
    objects = [item for item in geometry.get("objects") or [] if isinstance(item, dict)]
    allowed: dict[str, set[str]] = {}
    for constraint in contract_constraints(
        case_pack,
        relations=(
            "faces",
            "aligned_with",
            "paired_with",
            "against_wall",
            "centered_on_wall",
        ),
        include_auxiliary=False,
    ):
        if str(constraint.get("relation") or "") == "paired_with":
            for check in _paired_seating_checks(
                constraint,
                case_pack=case_pack,
                objects=objects,
                mode="contract",
            ):
                if check.get("relation_type") != "seating_to_work_surface":
                    continue
                allowed.setdefault(str(check["subject_id"]), set()).update(
                    str(target_id) for target_id in check.get("target_ids") or []
                )
            continue
        subjects = bound_ids(constraint.get("subjects"), objects)
        targets = bound_ids(constraint.get("targets"), objects)
        relation = str(constraint.get("relation") or "")
        for subject_id in subjects:
            targets_for_subject = (
                _nearest_wall_ids([subject_id], objects)
                if relation in {"against_wall", "centered_on_wall"}
                else targets
            )
            allowed.setdefault(subject_id, set()).update(targets_for_subject)
    return allowed


def _explicit_prompt_constraints(prompt: str, lowered: str) -> list[dict[str, Any]]:
    constraints: list[dict[str, Any]] = []
    clauses = _clauses(prompt)
    for clause in clauses:
        normalized = clause.lower()
        # Keep the subject phrase short.  This accepts paraphrases while
        # requiring the relation words to occur in the same clause.
        for match in re.finditer(
            r"(?P<subject>[a-z0-9_\- ,']{1,70}?)\s+"
            r"(?:is |sits |placed |positioned |)?"
            r"(?P<centered>centered|centred)?\s*"
            r"(?:against|on)\s+(?:the\s+)?(?P<wall>(?:[a-z]+\s+){0,2})?wall\b",
            normalized,
        ):
            subject = selector_for_phrase(match.group("subject"))
            if subject is None:
                continue
            relation = "centered_on_wall" if match.group("centered") else "against_wall"
            constraints.append(
                _constraint(
                    relation,
                    subject,
                    {"category": "wall", "role": (match.group("wall") or "").strip()},
                    source="explicit_prompt",
                    evidence_span=clause,
                )
            )

        for match in re.finditer(
            r"(?P<subject>[a-z0-9_\- ,']{1,70}?)\s+"
            r"(?:is |sits |placed |positioned |)?(?:centered|centred|central|centrally positioned)"
            r"(?:\s+in|\s+at|\s+of)?\s+(?:the\s+)?(?:center|centre|middle)"
            r"(?:\s+of\s+(?:the\s+)?)?(?:room)?\b",
            normalized,
        ):
            subject = selector_for_phrase(match.group("subject"))
            if subject is not None:
                constraints.append(
                    _constraint(
                        "centered_in_room",
                        subject,
                        {"category": "room"},
                        source="explicit_prompt",
                        evidence_span=clause,
                    )
                )

        for match in re.finditer(
            r"(?P<subject>[a-z0-9_\- ,']{1,70}?)\s+"
            r"(?:is |sits |lies |placed |positioned )?"
            r"(?P<centered>centered|centred)?\s*between\s+"
            r"(?:the\s+|a\s+|an\s+)?(?P<first>[a-z0-9_\- ']{1,50}?)\s+"
            r"and\s+(?:the\s+|a\s+|an\s+)?"
            r"(?P<second>[a-z0-9_\- ']{1,50}?)(?:[,.;]|$)",
            normalized,
        ):
            subject = selector_for_phrase(match.group("subject"))
            first = selector_for_phrase(match.group("first"))
            second = selector_for_phrase(match.group("second"))
            if subject is None or first is None or second is None:
                continue
            targets = dict(first)
            targets["secondary_category"] = second["category"]
            if "role" in second:
                targets["secondary_role"] = second["role"]
            if "count" in second:
                targets["secondary_count"] = second["count"]
            constraints.append(
                _constraint(
                    "centered_between" if match.group("centered") else "between",
                    subject,
                    targets,
                    source="explicit_prompt",
                    evidence_span=clause,
                )
            )

        for match in re.finditer(
            r"(?P<subject>[a-z0-9_\- ,']{1,70}?)\s+"
            r"(?:is |sits |placed |positioned )?in\s+(?:the\s+)?(?:center|centre|middle)\b"
            r"(?:\s+of\s+(?:the\s+)?room)?",
            normalized,
        ):
            subject = selector_for_phrase(match.group("subject"))
            if subject is not None:
                constraints.append(
                    _constraint(
                        "centered_in_room",
                        subject,
                        {"category": "room"},
                        source="explicit_prompt",
                        evidence_span=clause,
                    )
                )

        for match in re.finditer(
            r"(?P<subject>[a-z0-9_\- ,']{1,70}?)\s+"
            r"(?:is |sits |placed |positioned )?(?:directly\s+)?"
            r"in\s+front\s+of\s+(?:the\s+|a\s+|an\s+)?"
            r"(?P<target>[a-z0-9_\- ,']{1,70}?)(?:[,.;]|$)",
            normalized,
        ):
            subject = selector_for_phrase(match.group("subject"))
            target = selector_for_phrase(match.group("target"))
            if subject is not None and target is not None:
                constraints.append(
                    _constraint(
                        "in_front_of",
                        subject,
                        target,
                        source="explicit_prompt",
                        evidence_span=clause,
                    )
                )

        for match in re.finditer(
            r"(?P<subject>[a-z0-9_\- ,']{1,70}?)\s+"
            r"(?:facing|faces|face)\s+(?:the\s+|a\s+|an\s+)?"
            r"(?P<target>[a-z0-9_\- ,']{1,70}?)(?:[,.;]|$)",
            normalized,
        ):
            subject = selector_for_phrase(match.group("subject"))
            target = selector_for_phrase(match.group("target"))
            if subject is not None and target is not None:
                constraints.append(
                    _constraint(
                        "faces",
                        subject,
                        target,
                        source="explicit_prompt",
                        evidence_span=clause,
                    )
                )

        for match in re.finditer(
            r"(?P<subject>[a-z0-9_\- ,']{1,70}?)\s+"
            r"(?:tucked\s+under|at|beside|next\s+to)\s+(?:the\s+|a\s+|an\s+)?"
            r"(?P<target>(?:[a-z]+\s+){0,2}(?:desk|table|monitor|screen))\b",
            normalized,
        ):
            subject = selector_for_phrase(match.group("subject"))
            target = selector_for_phrase(match.group("target"))
            if subject is not None and target is not None:
                relation = "aligned_with" if "tucked" in match.group(0) else "near"
                constraints.append(
                    _constraint(
                        relation,
                        subject,
                        target,
                        source="explicit_prompt",
                        evidence_span=clause,
                    )
                )

        # Keep ordinary adjacency separate from workstation alignment.  The
        # wording is explicit, but the resulting relation remains geometric
        # and category-agnostic (for example, wardrobe -> dresser or plant ->
        # sofa); it does not rely on a room profile or a nearest-object guess.
        for match in re.finditer(
            r"(?P<subject>[a-z0-9_\- ,']{1,70}?)\s+"
            r"(?:is |sits |placed |positioned )?"
            r"(?:beside|next\s+to|adjacent\s+to|near)\s+(?:the\s+|a\s+|an\s+)?"
            r"(?P<target>[a-z0-9_\- ,']{1,70}?)(?:[,.;]|$)",
            normalized,
        ):
            subject = selector_for_phrase(match.group("subject"))
            target = selector_for_phrase(match.group("target"))
            if subject is not None and target is not None:
                constraints.append(
                    _constraint(
                        "near",
                        subject,
                        target,
                        source="explicit_prompt",
                        evidence_span=clause,
                    )
                )

        for match in re.finditer(
            r"(?P<subject>[a-z0-9_\- ,']{1,70}?)\s+"
            r"(?:on|sits\s+on|resting\s+on|placed\s+on)\s+(?:the\s+|a\s+|an\s+)?"
            r"(?P<target>[a-z0-9_\- ,']{1,70}?)(?:[,.;]|$)",
            normalized,
        ):
            # "a nightstand with a lamp on each side of the bed" describes
            # a lateral placement, not a support relation.  Do not turn it
            # into the nonsensical `nightstand on_top_of bed` contract.
            if re.search(r"\bon\s+(?:each|either|both)\s+side", match.group(0)):
                continue
            subject = selector_for_phrase(match.group("subject"))
            target = selector_for_phrase(match.group("target"))
            if subject is not None and target is not None:
                constraints.append(
                    _constraint(
                        "on_top_of",
                        subject,
                        target,
                        source="explicit_prompt",
                        evidence_span=clause,
                    )
                )

        # ``on each side of`` is lateral placement, not support.  Compile it
        # as a reusable flanking relation so beds, tables, and other anchors
        # do not need scenario-specific bedside rules.
        for match in re.finditer(
            r"(?P<subject>[a-z0-9_\- ,']{1,90}?)\s+"
            r"(?:on|at)\s+(?:each|either|both)\s+sides?\s+of\s+"
            r"(?:the\s+|a\s+|an\s+)?(?P<target>[a-z0-9_\- ,']{1,70}?)(?:[,.;]|$)",
            normalized,
        ):
            subject = selector_for_phrase(match.group("subject"), count=2)
            target = selector_for_phrase(match.group("target"))
            if subject is not None and target is not None:
                constraints.append(
                    _constraint(
                        "flanking",
                        subject,
                        target,
                        source="explicit_prompt",
                        evidence_span=clause,
                    )
                )

        flank = re.search(
            r"(?P<subject>(?:one|two|three|four|five|six)?\s*[a-z_\- ]*(?:armchairs?|chairs?|seats?))"
            r"\s+flank(?:ing)?\s+(?:the\s+)?(?P<target>[a-z_\- ]{1,60}?)(?:[,.;]|$)",
            normalized,
        )
        if flank:
            count = _leading_count(flank.group("subject"))
            subject = selector_for_phrase(flank.group("subject"), count=count)
            target = selector_for_phrase(flank.group("target"))
            if subject is not None and target is not None:
                constraints.append(
                    _constraint(
                        "flanking",
                        subject,
                        target,
                        source="explicit_prompt",
                        evidence_span=clause,
                    )
                )

    # Group constraints often span a full sentence and are easier to recognise
    # independently from individual relation clauses.
    if re.search(r"\b(?:student\s+desks?|desks?)\b", lowered) and re.search(
        r"\beach\s+with\s+(?:a\s+)?chair\b|\bstudent\s+chairs?\b", lowered
    ):
        constraints.append(
            _constraint(
                "paired_with",
                {"category": "student_chair", "role": "student", "quantifier": "all"},
                {"category": "student_desk", "role": "student", "quantifier": "all"},
                source="explicit_prompt",
                evidence_span=_first_sentence_with(lowered, "student"),
            )
        )
        if re.search(
            r"\b(?:rows?|grid|evenly|evenly\s+distributed|distributed\s+evenly)\b",
            lowered,
        ):
            constraints.append(
                _constraint(
                    "distributed_evenly",
                    {
                        "category": "student_desk",
                        "role": "student",
                        "quantifier": "all",
                    },
                    {
                        "category": "student_chair",
                        "role": "student",
                        "quantifier": "all",
                    },
                    source="explicit_prompt",
                    evidence_span=_first_sentence_with(lowered, "student"),
                )
            )
    if (
        re.search(
            r"\b(?:one\s+(?:on|at)\s+each\s+(?:side|edge)|one\s+on\s+each)\b", lowered
        )
        and "dining" in lowered
        and re.search(r"\bchairs?\b", lowered)
    ):
        constraints.append(
            _constraint(
                "one_per_side",
                {"category": "dining_chair", "quantifier": "all"},
                {"category": "dining_table", "quantifier": "all"},
                source="explicit_prompt",
                evidence_span=_first_sentence_with(lowered, "each"),
            )
        )
    return constraints


def _explicit_required_count_constraints(prompt: str) -> list[dict[str, Any]]:
    """Compile only quantities stated directly in the immutable prompt.

    Prefer the most specific semantic category for an overlapping phrase, so
    ``six student desks`` does not degrade to the broader ``six desks``.  The
    second pass propagates explicit ``N objects, each with a ...`` cardinality
    without inventing a count from room conventions.
    """
    explicit_numbers = {
        key: value for key, value in _NUMBER_WORDS.items() if key not in {"a", "an"}
    }
    number_pattern = r"\d+|" + "|".join(map(re.escape, explicit_numbers))
    candidates: dict[tuple[int, int], tuple[str, int, str]] = {}
    generic_categories = {"desk", "chair", "table"}

    for category, aliases in _CATEGORY_PATTERNS:
        for alias in aliases:
            alias_pattern = re.escape(alias).replace(r"\ ", r"\s+")
            pattern = re.compile(
                rf"(?<![a-z0-9])(?P<count>{number_pattern})\s+"
                rf"(?:(?:[a-z][a-z0-9-]*)\s+){{0,2}}?"
                rf"{alias_pattern}(?:s|es)?(?![a-z0-9])",
                re.IGNORECASE,
            )
            for match in pattern.finditer(prompt):
                count_text = match.group("count").lower()
                count = (
                    int(count_text)
                    if count_text.isdigit()
                    else explicit_numbers.get(count_text, 0)
                )
                if count <= 0:
                    continue
                key = match.span()
                previous = candidates.get(key)
                if previous is None or (
                    previous[0] in generic_categories
                    and category not in generic_categories
                ):
                    candidates[key] = (category, count, match.group(0))

    constraints: list[dict[str, Any]] = []
    for category, count, evidence in candidates.values():
        constraints.append(
            _constraint(
                "required_count",
                {"category": category, "count": count, "quantifier": "at_least"},
                None,
                source="explicit_prompt",
                evidence_span=evidence,
            )
        )

    each_pattern = re.compile(
        rf"(?<![a-z0-9])(?P<count>{number_pattern})\s+"
        r"(?P<source>[a-z][a-z0-9' -]{0,40}?)\s*,?\s+each\s+"
        r"(?:with|having)\s+(?:(?:a|an|one)\s+)?"
        r"(?P<target>[a-z][a-z0-9' -]{0,40}?)(?=[,.;]|$)",
        re.IGNORECASE,
    )
    for match in each_pattern.finditer(prompt):
        count_text = match.group("count").lower()
        count = (
            int(count_text)
            if count_text.isdigit()
            else explicit_numbers.get(count_text, 0)
        )
        source = selector_for_phrase(match.group("source"), count=count)
        target = selector_for_phrase(match.group("target"), count=count)
        if count <= 0 or source is None or target is None:
            continue
        constraints.append(
            _constraint(
                "required_count",
                {
                    "category": str(target["category"]),
                    "count": count,
                    "quantifier": "at_least",
                    **({"role": target["role"]} if target.get("role") else {}),
                },
                None,
                source="explicit_prompt",
                evidence_span=match.group(0),
            )
        )
    return constraints


def _room_ontology_constraints(room_type: str, lowered: str) -> list[dict[str, Any]]:
    constraints: list[dict[str, Any]] = []
    if room_type == "bedroom":
        constraints.append(
            _constraint(
                "required_count",
                {"category": "bed", "count": 1, "quantifier": "at_least"},
                None,
                source="room_ontology",
                evidence_span="bedroom room-type ontology",
            )
        )
        # The anchor is room-function knowledge, not a case-specific recipe.
        constraints.append(
            _constraint(
                "against_wall",
                {"category": "bed", "quantifier": "all"},
                {"category": "wall"},
                source="room_ontology",
                evidence_span="bedroom sleeping-surface anchor",
            )
        )
    if (
        _MEDIA_SUPPORT_PATTERN.search(lowered)
        and _TELEVISION_PATTERN.search(lowered)
        and not _prompt_explicitly_wall_mounts_television(lowered)
    ):
        constraints.append(
            _constraint(
                "on_top_of",
                {"category": "television", "count": 1, "quantifier": "all"},
                {"category": "tv_stand", "count": 1, "quantifier": "all"},
                source="model_inferred",
                confidence=0.7,
                evidence_span="possible freestanding television and media-support pairing",
            )
        )
    return constraints


def _prompt_explicitly_wall_mounts_television(prompt: str) -> bool:
    """Distinguish a mounted display from an entertainment group at a wall."""
    if not _EXPLICIT_WALL_MOUNTED_TELEVISION_PATTERN.search(prompt):
        return False
    if _MEDIA_GROUP_ON_WALL_PATTERN.search(prompt) and not re.search(
        r"\b(?:wall[- ]mounted|mounted|hung|hanging)\b", prompt, re.IGNORECASE
    ):
        return False
    return True


def _task_spec_constraints(task_spec: Any | None) -> list[dict[str, Any]]:
    if task_spec is None:
        return []
    if isinstance(task_spec, dict):
        values = task_spec.get("intent_constraints") or []
    else:
        values = getattr(task_spec, "intent_constraints", []) or []
    return [item for item in values if isinstance(item, dict)]


def _normalize_external_constraint(raw: dict[str, Any]) -> dict[str, Any] | None:
    relation = str(raw.get("relation") or "").strip().lower()
    if relation not in VALID_RELATIONS:
        return None
    subjects = _normalize_selector(raw.get("subjects") or raw.get("subject"))
    targets = _normalize_selector(raw.get("targets") or raw.get("target"))
    if subjects is None:
        return None
    supplied_source = str(raw.get("source") or "model_inferred").strip().lower()
    # A model can quote a real span yet still hallucinate its relation.  It is
    # therefore never allowed to self-authorize an ``explicit_prompt`` or
    # ``room_ontology`` hard constraint.  The deterministic prompt parser and
    # this module's controlled ontology are the only hard-constraint authors;
    # model/VLM output remains useful as labelled auxiliary evidence.
    source = (
        supplied_source
        if supplied_source in {"model_inferred", "vlm_observation"}
        else "model_inferred"
    )
    evidence = str(raw.get("evidence_span") or raw.get("evidence") or "").strip()
    return _constraint(
        relation,
        subjects,
        targets,
        source=source,
        evidence_span=evidence,
        confidence=_bounded_float(raw.get("confidence"), default=0.7),
    )


def _normalize_selector(value: Any) -> dict[str, Any] | None:
    if isinstance(value, str):
        return selector_for_phrase(value)
    if not isinstance(value, dict):
        return None
    category = str(value.get("category") or "").strip().lower().replace(" ", "_")
    if not category:
        return None
    normalized: dict[str, Any] = {
        "category": category,
        "quantifier": str(value.get("quantifier") or "all"),
    }
    role = str(value.get("role") or "").strip().lower()
    if role:
        normalized["role"] = role
    count = value.get("count")
    if isinstance(count, (int, float)) and int(count) > 0:
        normalized["count"] = int(count)
    secondary_category = (
        str(value.get("secondary_category") or "").strip().lower().replace(" ", "_")
    )
    if secondary_category:
        normalized["secondary_category"] = secondary_category
        secondary_role = str(value.get("secondary_role") or "").strip().lower()
        if secondary_role:
            normalized["secondary_role"] = secondary_role
        secondary_count = value.get("secondary_count")
        if isinstance(secondary_count, (int, float)) and int(secondary_count) > 0:
            normalized["secondary_count"] = int(secondary_count)
    return normalized


def _constraint(
    relation: str,
    subjects: dict[str, Any],
    targets: dict[str, Any] | None,
    *,
    source: str,
    evidence_span: str,
    confidence: float = 1.0,
) -> dict[str, Any]:
    normalized_evidence = " ".join(str(evidence_span or "").split())
    digest = hashlib.sha1(
        repr((relation, subjects, targets, source, normalized_evidence)).encode("utf-8")
    ).hexdigest()[:12]
    return {
        "constraint_id": f"intent_{digest}",
        "relation": relation,
        "subjects": subjects,
        "targets": targets or {},
        "strength": "hard" if source in HARD_SOURCES else "auxiliary",
        "source": source,
        "confidence": _bounded_float(confidence, default=1.0),
        "evidence_span": normalized_evidence,
        "stage": "furniture",
    }


def _deduplicate_constraints(constraints: list[dict[str, Any]]) -> list[dict[str, Any]]:
    priority = {
        "explicit_prompt": 0,
        "room_ontology": 1,
        "model_inferred": 2,
        "vlm_observation": 3,
    }
    keyed: dict[tuple[str, str, str], dict[str, Any]] = {}
    for constraint in constraints:
        subjects = constraint.get("subjects") or {}
        targets = constraint.get("targets") or {}
        key = (
            str(constraint.get("relation") or ""),
            repr(subjects),
            repr(targets),
        )
        previous = keyed.get(key)
        if previous is None or priority.get(
            str(constraint.get("source")), 9
        ) < priority.get(str(previous.get("source")), 9):
            keyed[key] = constraint
    return sorted(
        keyed.values(),
        key=lambda item: (str(item.get("relation")), str(item.get("constraint_id"))),
    )


def _normalize_room_type(room_type: str, prompt: str) -> str:
    value = str(room_type or "").strip().lower().replace(" ", "_")
    if value in {"bedroom", "dining_room", "classroom", "living_room", "study"}:
        return value
    if "bedroom" in prompt:
        return "bedroom"
    if "classroom" in prompt:
        return "classroom"
    if "dining" in prompt:
        return "dining_room"
    if "living room" in prompt:
        return "living_room"
    if "study" in prompt or "home office" in prompt:
        return "study"
    return value or "room"


def _clauses(prompt: str) -> list[str]:
    return [
        item.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", prompt)
        for item in re.split(r"\s*,\s*(?:and\s+)?", sentence)
        if item.strip()
    ]


def _leading_count(text: str) -> int | None:
    match = re.match(r"\s*(\d+|" + "|".join(_NUMBER_WORDS) + r")\b", text.lower())
    if not match:
        return None
    value = match.group(1)
    return int(value) if value.isdigit() else _NUMBER_WORDS.get(value)


def _first_sentence_with(text: str, needle: str) -> str:
    for sentence in _clauses(text):
        if needle in sentence:
            return sentence
    return text[:180]


def _contains_phrase(text: str, phrase: str) -> bool:
    # Prompt nouns are commonly plural while the semantic vocabulary stores a
    # canonical singular form.  Matching the regular plural here keeps object
    # binding independent of generated IDs without accepting arbitrary stems.
    plural = r"(?:s|es)?" if phrase and phrase[-1:].isalpha() else ""
    return (
        re.search(rf"(?<![a-z0-9]){re.escape(phrase)}{plural}(?![a-z0-9])", text)
        is not None
    )


def _phrase_mentions_plural(text: str, category: str) -> bool:
    candidates = [category.replace("_", " ")]
    for candidate, aliases in _CATEGORY_PATTERNS:
        if candidate == category:
            candidates.extend(aliases)
            break
    for candidate in candidates:
        escaped = re.escape(candidate)
        if re.search(rf"(?<![a-z0-9]){escaped}(?:s|es)(?![a-z0-9])", text):
            return True
    return False


def _role_matches_object(role: str, obj: dict[str, Any]) -> bool:
    if not role:
        return True
    metadata = obj.get("metadata") if isinstance(obj.get("metadata"), dict) else {}
    identity = " ".join(
        str(value or "").lower().replace("_", " ")
        for value in (
            metadata.get("semantic_name"),
            obj.get("id"),
            obj.get("name"),
            obj.get("description"),
        )
    )
    if role in {
        "back",
        "main",
        "side",
        "opposite",
        "front",
        "north",
        "south",
        "east",
        "west",
    }:
        return (
            str(obj.get("category_norm") or obj.get("category") or "").lower() == "wall"
        )
    return role in identity


def _selector_matches_object(category: str, role: str, obj: dict[str, Any]) -> bool:
    object_cat = object_category(obj)
    metadata = obj.get("metadata") if isinstance(obj.get("metadata"), dict) else {}
    semantic_name = str(metadata.get("semantic_name") or "").lower()
    if semantic_name == category:
        return _role_matches_object(role, obj)
    base_category = str(obj.get("category_norm") or obj.get("category") or "").lower()
    identity = " ".join(
        str(obj.get(key) or "").lower().replace("_", " ")
        for key in ("id", "name", "description")
    )
    identity = " ".join(
        value
        for value in (semantic_name.replace("_", " "), base_category, identity)
        if value
    )
    normalized_category = category.replace("_", " ")
    category_matches = {
        "student_desk": base_category == "desk" and "student" in identity,
        "teacher_desk": base_category == "desk"
        and ("teacher" in identity or "instructor" in identity),
        "guest_chair": base_category in {"chair", "armchair", "office_chair"}
        and ("guest" in identity or "visitor" in identity),
        "student_chair": base_category in {"chair", "office_chair", "dining_chair"}
        and "student" in identity,
        "dining_chair": base_category in {"dining_chair", "chair"}
        and "dining" in identity,
        "dining_table": base_category in {"dining_table", "table"}
        and "dining" in identity,
        "coffee_table": base_category in {"coffee_table", "table"}
        and "coffee" in identity,
        "tv_stand": base_category
        in {"tv_stand", "media_console", "entertainment_center"}
        or "tv stand" in identity,
        "television": base_category in {"television", "tv", "screen", "display"}
        or (
            not _MEDIA_SUPPORT_PATTERN.search(identity)
            and ("television" in identity or re.search(r"\btv\b", identity) is not None)
        ),
        "office_chair": base_category in {"office_chair", "chair"}
        and (
            "office" in identity
            or "desk chair" in identity
            or base_category == "office_chair"
        ),
        "table": base_category in {"table", "dining_table", "coffee_table", "desk"},
        "chair": base_category
        in {"chair", "office_chair", "dining_chair", "armchair", "stool", "bench"},
        "wall": base_category == "wall",
        "room": False,
    }.get(category)
    if category_matches is None:
        category_matches = (
            object_cat == category
            or base_category == category
            or _contains_phrase(identity, normalized_category)
        )
    if not category_matches:
        return False
    return _role_matches_object(role, obj)


def _fd_relation_for_constraint(
    constraint: dict[str, Any], objects: list[dict[str, Any]]
) -> str | None:
    relation = str(constraint.get("relation") or "")
    if relation == "against_wall":
        return "back_against_wall"
    if relation == "paired_with":
        return "seating_to_work_surface"
    if relation in {"faces", "aligned_with"}:
        subject_category = str((constraint.get("subjects") or {}).get("category") or "")
        target_category = str((constraint.get("targets") or {}).get("category") or "")
        if subject_category in {
            "chair",
            "office_chair",
            "guest_chair",
            "student_chair",
            "dining_chair",
            "armchair",
            "sofa",
        }:
            if target_category in {
                "desk",
                "table",
                "dining_table",
                "coffee_table",
                "student_desk",
                "teacher_desk",
            }:
                return "seating_to_work_surface"
            if target_category in {"television", "monitor", "tv_stand"}:
                return "seating_to_media"
        return "furniture_faces_furniture"
    if relation == "on_top_of":
        return "object_on_support"
    if relation == "near":
        return "generic_near_relation"
    return None


def _paired_seating_checks(
    constraint: dict[str, Any],
    *,
    case_pack: dict[str, Any],
    objects: list[dict[str, Any]],
    mode: str,
) -> list[dict[str, Any]]:
    """Materialize prompt-authorized one-to-one seat/surface checks.

    Geometry may select which interchangeable generated instances form a pair,
    but it cannot create the pairing requirement: that authority comes only
    from the immutable prompt constraint.
    """
    from scenesmith.scenebenchmark_critic.metrics.functional_dependency.seat_surface_assignment import (
        assign_work_seats_to_surfaces,
        room_bounds_from_case_pack,
    )

    subject_ids = set(bound_ids(constraint.get("subjects"), objects))
    target_ids = set(bound_ids(constraint.get("targets"), objects))
    if not subject_ids or not target_ids or len(subject_ids) != len(target_ids):
        return []
    assignments = assign_work_seats_to_surfaces(
        objects,
        task_instruction=str(case_pack.get("task_instruction") or ""),
        room_type=str(case_pack.get("room_type") or ""),
        room_bounds=room_bounds_from_case_pack(case_pack),
    )
    selected = [
        assignment
        for assignment in assignments
        if assignment.seat_id in subject_ids and assignment.surface_id in target_ids
    ]
    if len(selected) != len(subject_ids):
        return []
    tier = _constraint_tier(constraint, mode)
    checks: list[dict[str, Any]] = []
    for assignment in selected:
        seat_check_id = (
            f"intent_contract__{constraint['constraint_id']}__"
            f"{assignment.seat_id}__{assignment.surface_id}"
        )
        common_evidence = {
            "intent_constraint": constraint,
            "constraint_mode": mode,
            **assignment.evidence(),
        }
        checks.append(
            {
                "check_id": seat_check_id,
                "metric": "functional_dependency",
                "subject_id": assignment.seat_id,
                "target_ids": [assignment.surface_id],
                "relation_type": "seating_to_work_surface",
                "expected_use": _expected_use(constraint, "seating_to_work_surface"),
                "check_source": "intent_contract",
                "scoring_tier": tier,
                "evidence": common_evidence,
            }
        )
        checks.append(
            {
                "check_id": f"{seat_check_id}__surface_faces_seat",
                "metric": "functional_dependency",
                "subject_id": assignment.surface_id,
                "target_ids": [assignment.seat_id],
                "relation_type": "furniture_faces_furniture",
                "expected_use": (
                    "the paired work surface's usable front faces its assigned seat"
                ),
                "check_source": "intent_contract",
                "scoring_tier": tier,
                "evidence": {
                    **common_evidence,
                    "paired_surface_facing": True,
                    "dependency": {
                        "subject_face": "front",
                        "target_face": "any",
                        "max_angle_deg": 60.0,
                        "max_distance_m": 1.8,
                    },
                },
            }
        )
    return checks


def _nearest_wall_ids(
    subject_ids: list[str], objects: list[dict[str, Any]]
) -> list[str]:
    by_id = {str(obj.get("id") or ""): obj for obj in objects}
    walls = [obj for obj in objects if object_category(obj) == "wall"]
    selected: list[str] = []
    for subject_id in subject_ids:
        subject = by_id.get(subject_id)
        center = bbox_center_xy(subject) if subject is not None else None
        if center is None or not walls:
            continue
        wall = min(
            walls,
            key=lambda item: _center_distance_sq(center, bbox_center_xy(item)),
        )
        selected.append(str(wall.get("id") or ""))
    return selected


def _center_distance_sq(
    first: tuple[float, float], second: tuple[float, float] | None
) -> float:
    if second is None:
        return math.inf
    return (first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2


def _constraint_tier(constraint: dict[str, Any], mode: str) -> str:
    if mode == "shadow":
        return "ignored"
    return "core" if is_hard_constraint(constraint) else "auxiliary"


def _expected_use(constraint: dict[str, Any], relation_type: str) -> str:
    evidence = str(constraint.get("evidence_span") or "")
    return {
        "back_against_wall": "keep the named furniture backed against its requested wall",
        "seating_to_work_surface": "use the explicitly paired desk or work surface",
        "seating_to_media": "face the explicitly named media target",
        "furniture_faces_furniture": "face the explicitly named furniture target",
        "object_on_support": "rest on the explicitly named support surface",
        "generic_near_relation": "remain near the explicitly named related object",
    }.get(relation_type, evidence or "satisfy the prompt-originated relation")


def _bounded_float(value: Any, *, default: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _copy_contract(contract: dict[str, Any]) -> dict[str, Any]:
    # Contract payloads are shallow scalar/list/dict data; copying the nested
    # rows prevents evaluators from mutating the scene-level cache.
    return {
        **contract,
        "constraints": [
            {
                **constraint,
                "subjects": dict(constraint.get("subjects") or {}),
                "targets": dict(constraint.get("targets") or {}),
            }
            for constraint in contract.get("constraints") or []
            if isinstance(constraint, dict)
        ],
    }
