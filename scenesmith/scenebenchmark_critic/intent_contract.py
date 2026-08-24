"""Prompt-originated, geometry-grounded critic intent contracts.

The critic used to infer hard layout requirements from the current scene and
from the (possibly memory-augmented) agent prompt.  That makes an evaluator
both invent and judge a requirement, which is especially fragile during scene
replay.  This module keeps the requirement source explicit and serializable:

``original prompt -> semantic selector -> object-id binding -> geometry check``.

No function in this module returns a pose.  LLM/VLM observations may enrich a
contract's evidence, while schema-validated explicit and model-inferred
relations are evaluated by the same hard contract path.
"""

from __future__ import annotations

import hashlib
import json
import math
import re

from typing import Any, Iterable

from scenesmith.scenebenchmark_critic.core.geometry import (
    bbox_center_xy,
    bbox_gap_xy,
    object_category,
)
from scenesmith.scenebenchmark_critic.relation_registry import (
    CEILING_MOUNTED_CATEGORIES,
    MANIPULAND_CATEGORIES,
    PUBLIC_RELATIONS,
    ROOM_RELATIVE_WALL_CATEGORIES,
    STAGE_ORDER,
    WALL_MOUNTED_CATEGORIES,
    relation_spec,
)
from scenesmith.scenebenchmark_critic.intent_schema import (
    INTENT_COMPILER_SPEC_VERSION,
    INTENT_CONTRACT_SCHEMA_VERSION,
    canonical_selector_category,
    migrate_intent_contract_payload,
    validate_intent_contract,
)
from scenesmith.scenebenchmark_critic.object_taxonomy import OBJECT_CATEGORY_PHRASES
from scenesmith.scenebenchmark_critic.object_taxonomy import (
    is_known_object_category,
    is_structural_anchor,
)


SCHEMA_VERSION = INTENT_CONTRACT_SCHEMA_VERSION
VALID_RELATIONS = PUBLIC_RELATIONS
HARD_SOURCES = frozenset(
    {
        "explicit_prompt",
        "task_compiler_inventory",
        "model_inferred",
        "room_ontology",
        "deterministic_fallback",
    }
)
# A SceneTaskSpec records one inventory entry per required place setting, but
# these labels can expand into several physical manipulands.  Their count is a
# coverage minimum, not an upper bound on independently generated instances.
_NON_ATOMIC_INVENTORY_CATEGORIES = frozenset({"cutlery", "table_setting"})
_WALL_TARGET_RELATIONS = frozenset({"against_wall", "centered_on_wall"})
_WALL_TARGET_CATEGORIES = frozenset({"wall", *ROOM_RELATIVE_WALL_CATEGORIES})
_DIRECT_FD_EVALUATORS = frozenset(
    {
        "back_against_wall",
        "generic_near_relation",
        "mounted_to_wall",
        "object_on_support",
    }
)

_CATEGORY_PATTERNS = OBJECT_CATEGORY_PHRASES

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
    "=== SceneExpert Hard Intent Contract",
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


def original_prompt_for_scene(scene: Any) -> str:
    """Read the immutable prompt before SceneExpert/memory prompt injection."""
    original = getattr(scene, "scene_expert_original_description", "")
    if original:
        return _strip_scene_expert_injection(str(original))

    # RoomScene checkpoints serialize text_description but not SceneExpert's
    # dynamic provenance attributes.  Only remove blocks with explicit system
    # markers so ordinary prompt wording cannot be mistaken for injected text.
    return _strip_scene_expert_injection(
        str(getattr(scene, "text_description", "") or "")
    )


def intent_contract_for_scene(scene: Any) -> dict[str, Any]:
    """Return the validated v4 contract attached to a live/checkpoint scene."""
    cached = getattr(scene, "scenebenchmark_intent_contract", None)
    if not isinstance(cached, dict):
        metadata = getattr(scene, "metadata", None)
        cached = (
            metadata.get("scenebenchmark_intent_contract")
            if isinstance(metadata, dict)
            else None
        )
    return cached if isinstance(cached, dict) else {}


def intent_contract_constraints_for_scene(scene: Any) -> list[dict[str, Any]]:
    """Return relation rows without consulting the inventory TaskSpec."""
    contract = intent_contract_for_scene(scene)
    rows = contract.get("constraints") if isinstance(contract, dict) else []
    return [row for row in rows or [] if isinstance(row, dict)]


def intent_contract_required_counts(
    scene: Any, *, stage: str | None = None
) -> dict[str, int]:
    """Return hard object counts owned by the independent contract.

    ``stage=None`` preserves the historical all-stage inventory view. A stage
    filter lets generation agents consume only obligations they own.
    """
    counts: dict[str, int] = {}
    for row in intent_contract_constraints_for_scene(scene):
        row_stage = str(row.get("stage") or "")
        if stage is not None and row_stage not in {"", stage}:
            continue
        relation = str(row.get("relation") or "")
        if relation in {"required_count", "edge_distribution"}:
            _record_contract_inventory_count(counts, row.get("subjects") or {})

        # Furniture-stage endpoints are explicit inventory claims. Without
        # this, a monitor requested on a desk can be judged as missing but
        # never be eligible for inventory repair.
        if (
            row_stage in {"", stage or "furniture"}
            and str(row.get("strength") or "hard").lower() == "hard"
        ):
            _record_contract_inventory_count(counts, row.get("subjects") or {})
            _record_contract_inventory_count(counts, row.get("targets") or {})
    return counts


def _record_contract_inventory_count(counts: dict[str, int], selector: Any) -> None:
    if not isinstance(selector, dict):
        return
    category = _normalize_selector_category(selector.get("category"))
    if (
        not category
        or is_structural_anchor(category)
        or category in _NON_FURNITURE_INVENTORY_CATEGORIES
    ):
        return
    try:
        count = int(selector.get("count") or 0)
    except (TypeError, ValueError):
        return
    if count > 0:
        counts[category] = max(counts.get(category, 0), count)


_NON_FURNITURE_INVENTORY_CATEGORIES = frozenset(
    {
        "",
        "room",
        "floor",
        "ceiling",
        "wall",
        "door",
        "opening",
        "window",
        *ROOM_RELATIVE_WALL_CATEGORIES,
        *WALL_MOUNTED_CATEGORIES,
        *CEILING_MOUNTED_CATEGORIES,
        *MANIPULAND_CATEGORIES,
    }
)


def _strip_scene_expert_injection(text: str) -> str:
    """Remove StageBrief/memory suffixes from both live and resumed scenes."""
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

    prompt_inventory = _explicit_required_count_constraints(normalized_prompt)
    task_inventory = _task_spec_inventory_constraints(task_spec)
    constraints.extend(prompt_inventory)
    constraints.extend(task_inventory)
    constraints.extend(_explicit_prompt_constraints(normalized_prompt, lowered))
    constraints.extend(
        _room_ontology_constraints(normalized_room, lowered, task_spec=task_spec)
    )
    constraints = _remove_expanded_table_setting_counts(constraints, task_spec)
    constraints = _normalize_group_required_counts(constraints)
    constraints = _remove_redundant_edge_facing_constraints(constraints)
    constraints = _deduplicate_constraints(constraints)
    constraints = _apply_task_spec_contract_metadata(constraints, task_spec)
    constraints = _deduplicate_constraints(constraints)
    warnings = _task_spec_inventory_conflict_warnings(prompt_inventory, task_inventory)
    contract = {
        "schema_version": SCHEMA_VERSION,
        "prompt": normalized_prompt,
        "prompt_sha256": hashlib.sha256(normalized_prompt.encode("utf-8")).hexdigest(),
        "room_type": normalized_room,
        "intent_compiler_spec_version": INTENT_COMPILER_SPEC_VERSION,
        "retry_count": 0,
        "warnings": warnings,
        "constraints": constraints,
    }
    return ensure_coverage_requirements(contract, task_spec=task_spec)


def ensure_coverage_requirements(
    contract: dict[str, Any], *, task_spec: Any | None = None
) -> dict[str, Any]:
    """Add deterministic audit rows without changing relation semantics.

    Coverage is intentionally conservative: only explicit architectural or
    functional-zone wording and explicit containment/overflow wording become
    rows.  Generic style and layout prose remains outside critic scope.
    """

    result = migrate_intent_contract_payload(dict(contract))
    prompt = " ".join(str(result.get("prompt") or "").split())
    rows = [
        row
        for row in result.get("coverage_requirements") or []
        if isinstance(row, dict)
    ]
    existing_ids = {str(row.get("requirement_id") or "") for row in rows}

    def add_row(
        *,
        kind: str,
        normalized: str,
        evidence_span: str,
        source: str = "explicit_prompt",
        earliest_stage: str = "floor_plan",
        final_stage: str = "final",
        relation: str = "",
    ) -> None:
        normalized_value = "_".join(str(normalized).strip().lower().split())
        evidence = " ".join(str(evidence_span or "").split())
        digest = hashlib.sha1(
            "|".join((kind, normalized_value, source, evidence)).encode("utf-8")
        ).hexdigest()[:12]
        requirement_id = f"coverage_{digest}"
        if requirement_id in existing_ids:
            return
        rows.append(
            {
                "requirement_id": requirement_id,
                "kind": kind,
                "normalized": normalized_value,
                "earliest_stage": earliest_stage,
                "final_stage": final_stage,
                "source": source,
                "evidence_span": evidence,
                "relation": relation,
            }
        )
        existing_ids.add(requirement_id)

    for match in re.finditer(r"\bwalk[- ]in\s+closet\b", prompt, re.IGNORECASE):
        add_row(
            kind="functional_zone",
            normalized="walk_in_closet",
            evidence_span=match.group(0),
        )

    task_payload = _task_spec_payload(task_spec)
    for value in task_payload.get("required_architectural_features") or []:
        text = " ".join(str(value or "").split())
        if text:
            add_row(
                kind="architectural_feature",
                normalized=text,
                evidence_span=text,
                source="task_spec_architectural_feature",
            )
    for value in task_payload.get("functional_zones") or []:
        text = " ".join(str(value or "").replace("_", " ").split())
        normalized = text.casefold()
        if text and any(
            token in normalized for token in ("closet", "enclos", "partition")
        ):
            add_row(
                kind="functional_zone",
                normalized=text,
                evidence_span=text,
                source="task_spec_functional_zone",
            )

    unsupported_patterns = (
        r"\b(?:a|an|the)\s+[a-z0-9_-]+"
        r"(?:\s+(?!(?:with|a|an|the)\b)[a-z0-9_-]+){0,4}\s+"
        r"overflowing\s+with\s+[^.;!?]+",
        r"\b(?:a|an|the)\s+[a-z0-9_-]+"
        r"(?:\s+(?!(?:with|a|an|the)\b)[a-z0-9_-]+){0,4}\s+"
        r"(?:contained|stored|stacked)\s+(?:in|inside)\s+[^.;!?]+",
        r"\b(?:inside|within)\s+(?:the|a|an)\s+[^.;!?]+",
    )
    relation_rows = result.get("constraints") or []
    relation_evidence = " ".join(
        str(row.get("evidence_span") or "")
        for row in relation_rows
        if isinstance(row, dict)
    ).casefold()
    for pattern in unsupported_patterns:
        for match in re.finditer(pattern, prompt, re.IGNORECASE):
            evidence = " ".join(match.group(0).split())
            if evidence.casefold() in relation_evidence:
                continue
            add_row(
                kind="unsupported_relation",
                normalized="containment",
                evidence_span=evidence,
                earliest_stage="manipuland",
                relation="contains",
            )

    result["schema_version"] = INTENT_CONTRACT_SCHEMA_VERSION
    result["intent_compiler_spec_version"] = INTENT_COMPILER_SPEC_VERSION
    result["coverage_requirements"] = rows
    return result


def _normalize_group_required_counts(
    constraints: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Drop generic sub-counts that only describe a typed group layout."""
    group_rows = [
        row
        for row in constraints
        if str(row.get("relation") or "")
        in {"edge_distribution", "one_per_support", "corner_distribution"}
    ]
    normalized: list[dict[str, Any]] = []
    for row in constraints:
        relation = str(row.get("relation") or "")
        if relation == "on_top_of" and any(
            str(group.get("relation") or "") == "one_per_support"
            and _selectors_can_overlap(
                row.get("subjects") or {}, group.get("subjects") or {}
            )
            and _selectors_can_overlap(
                row.get("targets") or {}, group.get("targets") or {}
            )
            for group in group_rows
        ):
            continue
        if relation == "faces" and any(
            str(group.get("relation") or "") == "paired_with"
            and _selectors_can_overlap(
                row.get("subjects") or {}, group.get("subjects") or {}
            )
            and _selectors_can_overlap(
                row.get("targets") or {}, group.get("targets") or {}
            )
            for group in constraints
        ):
            continue
        if relation != "required_count":
            normalized.append(row)
            continue
        selector = row.get("subjects") or {}
        category = str(selector.get("category") or "")
        evidence = str(row.get("evidence_span") or "").lower()
        redundant = any(
            category != str((group.get("subjects") or {}).get("category") or "")
            and _selectors_can_overlap(selector, group.get("subjects") or {})
            and evidence
            and evidence in str(group.get("evidence_span") or "").lower()
            for group in group_rows
        )
        if not redundant:
            normalized.append(row)
    return normalized


def _remove_expanded_table_setting_counts(
    constraints: list[dict[str, Any]], task_spec: Any | None
) -> list[dict[str, Any]]:
    """Treat table_setting as semantic when concrete components are inventory."""

    payload = _task_spec_payload(task_spec)
    physical_categories = {
        _normalize_selector_category(value)
        for field in (
            "required_large_objects",
            "required_wall_objects",
            "required_ceiling_objects",
            "required_small_objects",
        )
        for value in payload.get(field) or []
    }
    if not {"plate", "cutlery", "glass"}.issubset(physical_categories):
        return constraints
    return [
        row
        for row in constraints
        if not (
            str(row.get("relation") or "") == "required_count"
            and _normalize_selector_category(
                (row.get("subjects") or {}).get("category")
            )
            == "table_setting"
        )
    ]


def _remove_redundant_edge_facing_constraints(
    constraints: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Drop pairwise facing rows covered by an inward edge distribution.

    A rectangular seating instruction already carries the complete subject
    group, target binding, and inward-facing requirement.  Keeping a generic
    ``faces(chair -> table)`` row alongside it would duplicate the same hard
    requirement and can make schema validation reject an otherwise valid
    contract.  Endpoint overlap is semantic, so typed selectors such as
    ``office_chair -> conference_table`` also cover a generic pairwise row.
    """
    edge_rows = [
        row
        for row in constraints
        if isinstance(row, dict)
        and str(row.get("relation") or "") == "edge_distribution"
        and str(row.get("orientation") or "") == "toward_target"
    ]
    if not edge_rows:
        return constraints

    result: list[dict[str, Any]] = []
    for row in constraints:
        if str(row.get("relation") or "") != "faces":
            result.append(row)
            continue
        subjects = row.get("subjects")
        targets = row.get("targets")
        if not isinstance(subjects, dict) or not isinstance(targets, dict):
            result.append(row)
            continue
        redundant = any(
            _selectors_can_overlap(subjects, edge_row.get("subjects") or {})
            and _selectors_can_overlap(targets, edge_row.get("targets") or {})
            for edge_row in edge_rows
        )
        if not redundant:
            result.append(row)
    return result


def attach_intent_contract_to_case_pack(
    scene: Any, case_pack: dict[str, Any]
) -> dict[str, Any]:
    """Validate/copy the independent contract without recompiling it."""
    prompt = original_prompt_for_scene(scene)
    prompt_hash = hashlib.sha256(" ".join(prompt.split()).encode("utf-8")).hexdigest()
    cached = getattr(scene, "scenebenchmark_intent_contract", None)
    if not isinstance(cached, dict):
        metadata = getattr(scene, "metadata", None)
        cached = (
            metadata.get("scenebenchmark_intent_contract")
            if isinstance(metadata, dict)
            else None
        )
    if cached is None:
        case_pack["original_task_instruction"] = prompt
        case_pack.pop("intent_contract", None)
        return {}
    if not isinstance(cached, dict):
        raise ValueError("scenebenchmark_intent_contract must be an object")
    if cached.get("schema_version") == "scenesmith.intent_contract.v4":
        cached = dict(cached)
        cached["schema_version"] = SCHEMA_VERSION
    cached = migrate_intent_contract_payload(cached)
    if cached.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            "furniture checkpoint requires a compatible intent contract; "
            f"got {cached.get('schema_version')!r}"
        )
    if cached.get("prompt_sha256") != prompt_hash:
        raise ValueError("intent contract prompt hash does not match original prompt")
    cached = validate_intent_contract(cached)
    setattr(scene, "scenebenchmark_intent_contract", cached)
    metadata = getattr(scene, "metadata", None)
    if isinstance(metadata, dict):
        metadata["scenebenchmark_intent_contract"] = cached
    case_pack["original_task_instruction"] = prompt
    case_pack["intent_contract"] = _copy_contract(cached)
    return case_pack["intent_contract"]


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
    return str(constraint.get("source") or "").lower() in HARD_SOURCES


def contract_relation_requested(case_pack: dict[str, Any], *relations: str) -> bool:
    """Whether a relation is authorized to enable a contract-mode hard rule.

    Only schema-validated compiled relations may activate a hard rule or grant
    a repair target.
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
    category = _normalize_selector_category(selector.get("category"))
    role = str(selector.get("role") or "").lower()
    object_rows = [obj for obj in objects if isinstance(obj, dict) and obj.get("id")]
    ids = [
        str(obj["id"])
        for obj in object_rows
        if _selector_matches_object(category, role, obj)
        and _selector_context_matches(selector, obj)
    ]
    if not ids:
        # Rendered asset retrieval can preserve only a broad semantic label
        # (for example ``table``) even when the prompt names a specialized
        # role (for example ``conference_table``).  Use that label only when
        # no exact specialized candidate exists; returning every fallback
        # candidate keeps the normal uniqueness gate conservative.
        ids = [
            str(obj["id"])
            for obj in object_rows
            if _selector_matches_generic_fallback(category, role, obj)
            and _selector_context_matches(selector, obj)
        ]
    return sorted(dict.fromkeys(ids))


def _selector_context_matches(selector: dict[str, Any], obj: dict[str, Any]) -> bool:
    """Apply stage/cohort/support/capability qualifiers after taxonomy matching."""
    hints = (
        obj.get("functional_hints")
        if isinstance(obj.get("functional_hints"), dict)
        else {}
    )
    metadata = obj.get("metadata") if isinstance(obj.get("metadata"), dict) else {}
    expected_stage = str(selector.get("stage") or "").strip().lower().replace("-", "_")
    if expected_stage:
        object_type = (
            str(obj.get("object_type") or hints.get("scene_object_type") or "")
            .lower()
            .replace("-", "_")
        )
        if object_type and object_type != expected_stage:
            return False
    expected_cohort = str(selector.get("cohort") or "").strip().lower()
    if expected_cohort:
        observed_cohort = (
            str(
                metadata.get("prompt_cohort")
                or metadata.get("cohort")
                or hints.get("prompt_cohort")
                or hints.get("cohort")
                or ""
            )
            .strip()
            .lower()
        )
        if observed_cohort and observed_cohort != expected_cohort:
            return False
    expected_support = str(
        selector.get("support_target") or selector.get("support_target_id") or ""
    ).strip()
    if expected_support:
        placement = (
            obj.get("placement_info")
            if isinstance(obj.get("placement_info"), dict)
            else {}
        )
        observed_support = str(
            placement.get("parent_surface_id")
            or metadata.get("support_target_id")
            or metadata.get("support_target")
            or hints.get("support_target_id")
            or ""
        ).strip()
        if observed_support != expected_support:
            return False
    capabilities = (
        selector.get("capabilities") or selector.get("required_capabilities") or []
    )
    if isinstance(capabilities, str):
        capabilities = [capabilities]
    if capabilities:
        declared = {
            str(value or "").strip().lower().replace("-", "_")
            for key in ("functional_categories", "candidate_affordances", "affordances")
            for value in (hints.get(key) or [])
            if str(value or "").strip()
        }
        if hints.get("is_media_target"):
            declared.add("media_target")
        if not {
            str(value or "").strip().lower().replace("-", "_") for value in capabilities
        }.issubset(declared):
            return False
    return True


def selector_match_count(
    selector: dict[str, Any] | None,
    objects: Iterable[dict[str, Any]],
) -> int:
    """Count semantic matches, including members represented by stack containers.

    Stacked manipulands are serialized as one physical scene object so they can
    be simulated and supported as a unit. Their ``member_assets`` still record
    the individual prompt objects, which must count toward a selector such as
    "three magazines". Relation binding continues to use the physical stack
    object returned by :func:`selected_ids`.
    """
    if not isinstance(selector, dict):
        return 0
    category = _normalize_selector_category(selector.get("category"))
    role = str(selector.get("role") or "").lower()
    object_rows = [obj for obj in objects if isinstance(obj, dict) and obj.get("id")]
    matcher = _selector_matches_object
    if not any(matcher(category, role, obj) for obj in object_rows):
        matcher = _selector_matches_generic_fallback
    return sum(
        _selector_match_multiplicity(category, role, obj, matcher=matcher)
        for obj in object_rows
    )


def _selector_match_multiplicity(
    category: str,
    role: str,
    obj: dict[str, Any],
    *,
    matcher: Any | None = None,
) -> int:
    if matcher is None:
        matcher = _selector_matches_object
    metadata = obj.get("metadata") if isinstance(obj.get("metadata"), dict) else {}
    members = metadata.get("member_assets")
    if metadata.get("composite_type") == "stack" and isinstance(members, list):
        matched_members = sum(
            matcher(
                category,
                role,
                {
                    "id": f"{obj['id']}::member::{index}",
                    "name": member.get("name") or member.get("asset_id") or "",
                    "description": member.get("description") or "",
                    "category": member.get("category") or "",
                    "category_norm": member.get("category_norm") or "",
                    "metadata": (
                        member.get("metadata")
                        if isinstance(member.get("metadata"), dict)
                        else {}
                    ),
                },
            )
            for index, member in enumerate(members)
            if isinstance(member, dict)
        )
        if matched_members:
            return matched_members
    return int(matcher(category, role, obj))


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
    object_rows = list(objects)
    ids = selected_ids(selector, object_rows)
    if not isinstance(selector, dict):
        return ids
    role = str(selector.get("role") or "").strip()
    if not ids and role:
        # A compiler can use a relation-local label such as "anchor" as a
        # selector role.  Generated assets normally do not expose that label
        # as metadata.  Fall back only when the category-only endpoint is
        # itself cardinality-safe, so a missing descriptive role cannot bind
        # an arbitrary member of a repeated object category.
        fallback_selector = dict(selector)
        fallback_selector.pop("role", None)
        fallback_ids = selected_ids(fallback_selector, object_rows)
        try:
            fallback_count = selector_match_count(fallback_selector, object_rows)
            requested_count = int(selector.get("count"))
        except (TypeError, ValueError):
            fallback_count = 0
            requested_count = 0
        quantifier = str(selector.get("quantifier") or "all").lower()
        if requested_count > 0 and quantifier not in {"at_least", "minimum"}:
            fallback_is_safe = fallback_count == requested_count
        else:
            fallback_is_safe = len(fallback_ids) == 1
        if fallback_is_safe:
            selector = fallback_selector
            ids = fallback_ids
    try:
        count = int(selector.get("count"))
    except (TypeError, ValueError):
        count = 0
    quantifier = str(selector.get("quantifier") or "all").lower()
    if (
        count > 0
        and quantifier not in {"at_least", "minimum"}
        and selector_match_count(selector, object_rows) != count
    ):
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


def _wall_anchor_subject_selector(value: str) -> dict[str, Any] | None:
    """Resolve the item immediately before a wall-anchor phrase.

    A sentence can introduce inventory before describing its final item, as in
    ``four chairs and a sideboard against the wall``. The generic selector
    intentionally recognises the first known category in a phrase, so use the
    last coordination segment for this syntactically local wall relation.
    """
    segments = re.split(r"\s*,\s*|\s+and\s+", str(value or ""))
    for segment in reversed(segments):
        selector = selector_for_phrase(segment)
        if selector is not None:
            return selector
    return selector_for_phrase(value)


def augment_contract_checks(case_pack: dict[str, Any]) -> bool:
    """Add pairwise contract checks to the hard functional-dependency core."""
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
    floor_supported_subject_ids: set[str] = set()
    for floor_constraint in contract_constraints(case_pack, relations=("on_top_of",)):
        target_category = _normalize_selector_category(
            (floor_constraint.get("targets") or {}).get("category")
        )
        if target_category == "floor":
            floor_supported_subject_ids.update(
                bound_ids(floor_constraint.get("subjects"), objects)
            )

    def floor_near_target_is_furniture(
        check: dict[str, Any], constraint: dict[str, Any]
    ) -> bool:
        """Scope floor-object external-adjacency guards to furniture targets.

        A floor-supported object may also be near a wall, opening, or another
        structural object.  The containment rejection is intended for the
        original failure mode (for example, a plant hidden inside a sofa), so
        do not apply it to non-furniture targets.  Older synthetic geometry
        records omit ``object_type``; in that case use the semantic selector
        and conservatively treat structural categories as non-furniture.
        """
        target_ids = [
            str(item) for item in (check.get("target_ids") or []) if str(item)
        ]
        records = {
            str(item.get("id")): item
            for item in objects
            if isinstance(item, dict) and item.get("id")
        }
        resolved = [records[item] for item in target_ids if item in records]
        typed = [str(item.get("object_type") or "").lower() for item in resolved]
        if any(typed):
            return bool(typed) and all(item == "furniture" for item in typed)
        category = _normalize_selector_category(
            (constraint.get("targets") or {}).get("category")
        )
        return category not in {
            "wall",
            "floor",
            "ceiling",
            "door",
            "window",
            "opening",
        }

    # A valid LLM contract may already have emitted the pairwise near check.
    # Apply the floor-support interpretation to those existing checks too;
    # otherwise the safety rule would depend on whether deterministic
    # enrichment happened to create the check first.
    for check in checks:
        if not isinstance(check, dict):
            continue
        if (
            str(check.get("relation_type") or "") != "generic_near_relation"
            or str(check.get("subject_id") or "") not in floor_supported_subject_ids
        ):
            continue
        evidence = check.setdefault("evidence", {})
        if not isinstance(evidence, dict):
            evidence = {}
            check["evidence"] = evidence
        constraint = evidence.get("intent_constraint") or {}
        target_category = _normalize_selector_category(
            (constraint.get("targets") or {}).get("category")
        )
        if target_category == "floor" or not floor_near_target_is_furniture(
            check, constraint
        ):
            continue
        dependency = evidence.setdefault("dependency", {})
        if not isinstance(dependency, dict):
            dependency = {}
            evidence["dependency"] = dependency
        if not dependency.get("requires_external_adjacency"):
            dependency["requires_external_adjacency"] = True
            added = True
    for constraint in contract_constraints(case_pack):
        if str(constraint.get("relation") or "") == "edge_distribution":
            # The dedicated extension binds the complete subject group and
            # generates atomic repair slots; generic pairwise checks cannot do
            # that without selecting a pose-dependent subset.
            continue
        if str(constraint.get("relation") or "") == "paired_with":
            paired_checks = _paired_seating_checks(
                constraint,
                case_pack=case_pack,
                objects=objects,
            )
            for check in paired_checks:
                check_id = str(check["check_id"])
                if check_id in existing:
                    continue
                checks.append(check)
                existing.add(check_id)
                added = True
            continue
        relation_type = _fd_relation_for_constraint(constraint)
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
            target_selector = constraint.get("targets") or {}
            existential_target = str(target_selector.get("quantifier") or "") in {
                "at_least",
                "minimum",
            }
            try:
                subject_count = int((constraint.get("subjects") or {}).get("count"))
                target_count = int(target_selector.get("count"))
            except (TypeError, ValueError):
                subject_count = target_count = 0
            group_targets = (
                subject_count > 1
                and subject_count == target_count
                and len(subject_ids) == len(target_ids)
            )
            target_id = compatible_targets[0]
            check_target_ids = (
                compatible_targets
                if existential_target or group_targets
                else [target_id]
            )
            check_id = (
                f"intent_contract__{constraint['constraint_id']}__"
                f"{subject_id}__{target_id}"
            )
            if check_id in existing:
                continue
            dependency = _relation_threshold_dependency(constraint)
            if (
                relation_type == "generic_near_relation"
                and subject_id in floor_supported_subject_ids
                and floor_near_target_is_furniture(
                    {"target_ids": check_target_ids}, constraint
                )
            ):
                # "On the floor near X" requires exterior adjacency.  Without
                # this flag, a footprint contained by X reports a 0m gap and is
                # incorrectly accepted as a successful near relation.
                dependency["requires_external_adjacency"] = True
            checks.append(
                {
                    "check_id": check_id,
                    "metric": "functional_dependency",
                    "subject_id": subject_id,
                    "target_ids": check_target_ids,
                    "relation_type": relation_type,
                    "expected_use": _expected_use(constraint, relation_type),
                    "check_source": "intent_contract",
                    "scoring_tier": "core",
                    "evidence": {
                        "intent_constraint": constraint,
                        "dependency": dependency,
                    },
                }
            )
            existing.add(check_id)
            added = True
    if added:
        case_pack["checks"] = checks
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
            "across_from",
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


def apply_contract_execution_states(
    case_pack: dict[str, Any], results: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Apply dependency ordering and persist one audit row per constraint."""
    constraints = contract_constraints(case_pack)
    if not constraints:
        return results
    stage = _execution_stage(str(case_pack.get("stage") or "adhoc"))
    final_stage = stage == "final"
    by_constraint: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        constraint = (result.get("evidence") or {}).get("intent_constraint") or {}
        constraint_id = str(constraint.get("constraint_id") or "")
        if constraint_id:
            by_constraint.setdefault(constraint_id, []).append(result)

    constraint_by_id = {str(row.get("constraint_id") or ""): row for row in constraints}
    status: dict[str, str] = {}
    for constraint_id, constraint in constraint_by_id.items():
        rows = by_constraint.get(constraint_id, [])
        expected = str(
            constraint.get("stage")
            or relation_spec(str(constraint.get("relation") or "")).earliest_stage
        )
        before_expected = STAGE_ORDER.index(stage) < STAGE_ORDER.index(expected)
        if not rows:
            status[constraint_id] = "pending" if before_expected else "failed"
        elif before_expected and not all(
            str(row.get("label") or "") == "pass" for row in rows
        ):
            status[constraint_id] = "pending"
        elif any(str(row.get("contract_state") or "") == "pending" for row in rows):
            status[constraint_id] = "pending"
        elif any(str(row.get("label") or "") == "fail" for row in rows):
            status[constraint_id] = "failed"
        elif rows and all(str(row.get("label") or "") == "pass" for row in rows):
            status[constraint_id] = "passed"
        else:
            status[constraint_id] = "failed"

    dependency_ids: dict[str, list[str]] = {}
    for constraint_id, constraint in constraint_by_id.items():
        relation = relation_spec(str(constraint.get("relation") or ""))
        dependency_relations = set(relation.dependencies)
        dependencies = [
            candidate_id
            for candidate_id, candidate in constraint_by_id.items()
            if candidate_id != constraint_id
            and str(candidate.get("relation") or "") in dependency_relations
            and _dependency_selectors_overlap(
                constraint,
                candidate,
                relation.dependency_binding,
            )
        ]
        dependency_ids[constraint_id] = dependencies
        if dependencies and any(status.get(item) == "failed" for item in dependencies):
            status[constraint_id] = "blocked"

    if final_stage:
        for constraint_id, value in list(status.items()):
            if value in {"pending", "blocked"}:
                status[constraint_id] = "failed"

    execution: list[dict[str, Any]] = []
    for constraint_id, constraint in constraint_by_id.items():
        state = status[constraint_id]
        rows = by_constraint.get(constraint_id, [])
        if not rows:
            synthetic = {
                "check_id": f"intent_execution__{constraint_id}",
                "metric": "functional_dependency",
                "label": "fail" if state == "failed" else "unknown",
                "confidence": 1.0,
                "primary_object": str(
                    (constraint.get("subjects") or {}).get("category") or "unbound"
                ),
                "related_objects": [],
                "selected_related_objects": [],
                "blocking_objects": [],
                "relation_type": str(constraint.get("relation") or ""),
                "reason": (
                    "No executable geometry result was produced for this hard "
                    f"constraint at stage `{stage}`."
                ),
                "diagnostics": {"contract_state": state, "stage": stage},
                "evidence": {"intent_constraint": constraint},
                "evaluation_source": "scenesmith_intent_contract",
                "scoring_tier": "core" if state == "failed" else "auxiliary",
                "contract_state": state,
            }
            results.append(synthetic)
            rows = [synthetic]
            by_constraint[constraint_id] = rows
        for row in rows:
            row["contract_state"] = state
            diagnostics = row.setdefault("diagnostics", {})
            diagnostics["contract_state"] = state
            diagnostics["dependency_constraint_ids"] = dependency_ids[constraint_id]
            if state == "pending" and not final_stage:
                row["label"] = "unknown"
                row["scoring_tier"] = "auxiliary"
                row["reason"] = (
                    f"Constraint is pending until its `{constraint.get('stage')}` stage."
                )
            elif state == "blocked" and not final_stage:
                row["label"] = "unknown"
                row["scoring_tier"] = "auxiliary"
                row["reason"] = (
                    "Blocked by failed upstream contract constraint(s): "
                    + ", ".join(dependency_ids[constraint_id])
                )
        if state == "failed" and not any(
            str(row.get("label") or "") == "fail" for row in rows
        ):
            rows[0]["label"] = "fail"
            rows[0]["scoring_tier"] = "core"
        execution.append(
            {
                "constraint_id": constraint_id,
                "relation": str(constraint.get("relation") or ""),
                "source": str(constraint.get("source") or ""),
                "evidence_span": str(constraint.get("evidence_span") or ""),
                "inference_reason": str(constraint.get("inference_reason") or ""),
                "state": state,
                "subject_ids": sorted(
                    {
                        str(row.get("primary_object") or "")
                        for row in rows
                        if row.get("primary_object")
                    }
                ),
                "target_ids": sorted(
                    {
                        str(object_id)
                        for row in rows
                        for object_id in row.get("related_objects") or []
                    }
                ),
                "dependency_constraint_ids": dependency_ids[constraint_id],
                "repair_strategy": relation_spec(
                    str(constraint.get("relation") or "")
                ).repair_strategy,
            }
        )
    contract = case_pack.get("intent_contract")
    if isinstance(contract, dict):
        contract["execution"] = execution
        contract["resolution_rate"] = round(
            sum(row["state"] == "passed" for row in execution) / len(execution), 6
        )
    return results


def _dependency_selectors_overlap(
    constraint: dict[str, Any],
    candidate: dict[str, Any],
    binding: str,
) -> bool:
    """Return whether a declared relation dependency concerns the same objects.

    Dependency relation names alone are too broad: a failed ``near`` check for
    a plant and sofa must not block a chair that faces a desk.  The registry
    declares whether the prerequisite applies to the same subject, to the full
    endpoint pair, or to any shared inventory endpoint.
    """
    subject = _constraint_endpoint_selectors(constraint, include_subject=True)
    candidate_subject = _constraint_endpoint_selectors(
        candidate, include_subject=True, include_targets=False
    )
    candidate_endpoints = _constraint_endpoint_selectors(candidate)
    if binding == "subject":
        return _selector_groups_overlap(subject, candidate_subject)
    if binding == "any_endpoint":
        return _selector_groups_overlap(subject, candidate_endpoints)

    targets = _constraint_endpoint_selectors(
        constraint, include_subject=False, include_targets=True
    )
    candidate_targets = _constraint_endpoint_selectors(
        candidate, include_subject=False, include_targets=True
    )
    return (
        _selector_groups_overlap(subject, candidate_subject)
        and _selector_groups_overlap(targets, candidate_targets)
    ) or (
        _selector_groups_overlap(subject, candidate_targets)
        and _selector_groups_overlap(targets, candidate_subject)
    )


def _constraint_endpoint_selectors(
    constraint: dict[str, Any],
    *,
    include_subject: bool = True,
    include_targets: bool = True,
) -> list[dict[str, Any]]:
    selectors: list[dict[str, Any]] = []
    if include_subject and isinstance(constraint.get("subjects"), dict):
        selectors.append(constraint["subjects"])
    targets = constraint.get("targets")
    if include_targets and isinstance(targets, dict):
        if targets.get("category"):
            selectors.append(targets)
        if targets.get("secondary_category"):
            selectors.append(
                {
                    "category": targets["secondary_category"],
                    "role": targets.get("secondary_role", ""),
                }
            )
    return selectors


def _selector_groups_overlap(
    first: Iterable[dict[str, Any]], second: Iterable[dict[str, Any]]
) -> bool:
    return any(
        _selectors_can_overlap(left, right) for left in first for right in second
    )


def _selectors_can_overlap(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_category = _normalize_selector_category(left.get("category"))
    right_category = _normalize_selector_category(right.get("category"))
    if not left_category or not right_category:
        return False
    left_role = str(left.get("role") or "").strip().lower()
    right_role = str(right.get("role") or "").strip().lower()
    if left_role and right_role and left_role != right_role:
        return False
    return bool(
        _selector_category_family(left_category)
        & _selector_category_family(right_category)
    )


def _selector_category_family(category: str) -> set[str]:
    """Return semantic families which an endpoint selector may bind."""
    category = _normalize_selector_category(category)
    families = {
        "chair": {
            "chair",
            "office_chair",
            "dining_chair",
            "student_chair",
            "guest_chair",
            "teacher_chair",
            "armchair",
            "stool",
            "bench",
        },
        "table": {
            "table",
            "dining_table",
            "conference_table",
            "coffee_table",
            "side_table",
            "desk",
        },
        "desk": {"desk", "teacher_desk", "student_desk", "reception_desk"},
        "wall_light": {"wall_light", "wall_lamp", "wall_sconce", "sconce"},
        "clock": {"clock", "alarm_clock", "bedside_clock"},
        "cup": {"cup", "cup_of_tea", "tea_cup"},
        "sculpture": {"sculpture", "wood_sculpture", "stone_sculpture"},
        "tray": {"tray", "serving_tray", "cafeteria_tray"},
        "glass": {"glass", "glass_bowl", "wine_glass", "drinking_glass"},
    }
    if category in families:
        return families[category]
    for broad, members in families.items():
        if category in members:
            return {category, broad}
    return {category}


def _execution_stage(stage: str) -> str:
    normalized = str(stage or "").strip().lower()
    aliases = {
        "scene_after_furniture": "furniture",
        "furniture_relation_repair": "furniture",
        "scene_after_wall_objects": "wall_mounted",
        "wall_visual_clearance_repair": "wall_mounted",
        "scene_after_ceiling_objects": "ceiling_mounted",
        "scene_after_manipulands": "manipuland",
        "final_scene": "final",
    }
    if normalized in aliases:
        return aliases[normalized]
    if normalized in STAGE_ORDER:
        return normalized
    for token, resolved in (
        ("manipuland", "manipuland"),
        ("ceiling", "ceiling_mounted"),
        ("wall", "wall_mounted"),
        ("furniture", "furniture"),
    ):
        if token in normalized:
            return resolved
    return "final"


def _explicit_prompt_constraints(prompt: str, lowered: str) -> list[dict[str, Any]]:
    constraints: list[dict[str, Any]] = []
    clauses = _clauses(prompt)
    previous_selector: dict[str, Any] | None = None
    for clause in clauses:
        normalized = clause.lower()
        # Failure examples and negative advice describe poses to avoid, not
        # user requirements. Do not let them create hard intent contracts.
        if re.search(
            r"\b(?:avoid|avoiding|failure\s+patterns?|known\s+failure|"
            r"do\s+not|don't|must\s+not|should\s+not|off[-\s]?center)\b",
            normalized,
        ):
            continue
        # Keep the subject phrase short.  This accepts paraphrases while
        # requiring the relation words to occur in the same clause.
        for match in re.finditer(
            r"(?P<subject>[a-z0-9_\- ,']{1,70}?)\s+"
            r"(?:is |sits |placed |positioned |)?"
            r"(?P<centered>centered|centred)?\s*"
            r"(?:against|on)\s+(?:the\s+)?(?P<wall>(?:[a-z]+\s+){0,2})?wall\b",
            normalized,
        ):
            subject = _wall_anchor_subject_selector(match.group("subject"))
            if subject is None:
                continue
            mounted_subject = subject.get(
                "category"
            ) in WALL_MOUNTED_CATEGORIES and bool(
                re.search(r"\b(?:mount|mounted|hang|hung|hanging)\b", normalized)
            )
            relation = (
                "on_wall"
                if mounted_subject
                else ("centered_on_wall" if match.group("centered") else "against_wall")
            )
            wall_role = (match.group("wall") or "").strip()
            if wall_role in {"a", "an", "one", "the"}:
                wall_role = ""
            constraints.append(
                _constraint(
                    relation,
                    subject,
                    {"category": "wall", "role": wall_role},
                    source="explicit_prompt",
                    evidence_span=clause,
                )
            )

        # ``centered above`` is an object-to-object alignment. A mirror above
        # a dressing table is usually off-center on the wall, so it must not
        # become ``centered_on_wall``.
        for match in re.finditer(
            r"(?P<subject>[a-z0-9_\- ,']{1,100}?)\s*,?\s*"
            r"(?:is |sits |stands |placed |positioned )?"
            r"(?:centered|centred)(?:\s+(?:directly|horizontally))?\s+above\s+"
            r"(?:the\s+|a\s+|an\s+)?"
            r"(?P<target>[a-z0-9_\- ,']{1,70}?)(?:[,.;]|$)",
            normalized,
        ):
            subject = selector_for_phrase(match.group("subject"))
            target = selector_for_phrase(match.group("target"))
            if subject is not None and target is not None:
                constraints.append(
                    _constraint(
                        "centered_above",
                        subject,
                        target,
                        source="explicit_prompt",
                        evidence_span=clause,
                    )
                )

        # Commas split "X on the wall, centered above Y" into two clauses.
        # Preserve the subject of the immediately preceding clause for this
        # compact anaphoric form.
        for match in re.finditer(
            r"^(?:it\s+)?(?:is |sits |stands |placed |positioned )?"
            r"(?:centered|centred)(?:\s+(?:directly|horizontally))?\s+above\s+"
            r"(?:the\s+|a\s+|an\s+)?"
            r"(?P<target>[a-z0-9_\- ,']{1,70}?)(?:[,.;]|$)",
            normalized,
        ):
            target = selector_for_phrase(match.group("target"))
            if previous_selector is not None and target is not None:
                constraints.append(
                    _constraint(
                        "centered_above",
                        dict(previous_selector),
                        target,
                        source="explicit_prompt",
                        evidence_span=clause,
                    )
                )

        # A center/middle phrase with a concrete object target is local to that
        # support, not a room-center or wall-center instruction.  Keep this
        # parser before the room-center variants so their shorter matches do
        # not consume the prefix of "middle of the table".
        for match in re.finditer(
            r"(?P<subject>[a-z0-9_\- ,']{1,70}?)\s+"
            r"(?:is |sits |rests |stands |placed |positioned )?"
            r"(?:in|at)\s+(?:the\s+)?(?:center|centre|middle)\s+of\s+"
            r"(?:the\s+|a\s+|an\s+)?"
            r"(?P<target>[a-z0-9_\- ,']{1,70}?)(?:[,.;]|$)",
            normalized,
        ):
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

        for match in re.finditer(
            r"(?P<subject>[a-z0-9_\- ,']{1,70}?)\s+"
            r"(?:is |sits |placed |positioned |)?(?:centered|centred|central|centrally positioned)"
            r"(?:\s+in|\s+at|\s+of)?\s+(?:the\s+)?(?:center|centre|middle)"
            r"(?:\s+of\s+(?:the\s+)?room)?\b"
            r"(?!\s+of\s+(?:the\s+)?(?!room\b)[a-z])",
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

        # Preserve the legacy natural-language form where the object follows
        # the room-center phrase, for example "the center of the room contains
        # a bed". This must become a prompt-originated v4 contract as well.
        for match in re.finditer(
            r"(?:the\s+)?(?:center|centre|middle)\s+of\s+"
            r"(?:the\s+)?room\s+"
            r"(?:contains?|holds?|features?|includes?|has|houses?|"
            r"is\s+(?:occupied|anchored)\s+by)\s+"
            r"(?:a|an|the|one)\s+(?P<subject>[a-z0-9_\- ,']{1,70}?)(?:[,.;]|$)",
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
            r"(?:is |sits |stands |placed |positioned )?"
            r"(?:in|at)\s+(?:a|the|one)\s+(?:room\s+)?corner\b",
            normalized,
        ):
            subject = selector_for_phrase(match.group("subject"))
            if subject is not None:
                constraints.append(
                    _constraint(
                        "corner_of_room",
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
                    (
                        "centered_between"
                        if match.group("centered")
                        or re.match(r"\s*cent(?:er|re)\b", normalized)
                        else "between"
                    ),
                    subject,
                    targets,
                    source="explicit_prompt",
                    evidence_span=clause,
                )
            )

        for match in re.finditer(
            r"(?P<subject>[a-z0-9_\- ,']{1,70}?)\s+"
            r"(?:is |sits |placed |positioned )?in\s+(?:the\s+)?(?:center|centre|middle)\b"
            r"(?:\s+of\s+(?:the\s+)?room)?"
            r"(?!\s+of\s+(?:the\s+)?(?!room\b)[a-z])",
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
            subject_phrase = re.sub(
                r"\s+on\s+(?:the\s+)?floor\b.*$", "", match.group("subject")
            )
            subject = selector_for_phrase(subject_phrase)
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
            r"(?:is |sits |stands |placed |positioned )?(?:directly\s+)?"
            r"behind\s+(?:the\s+|a\s+|an\s+)?"
            r"(?P<target>it|[a-z0-9_\- ,']{1,70}?)(?:[,.;]|$)",
            normalized,
        ):
            # "X against the wall behind Y" locates the wall relative to Y;
            # it is not a strict X-behind-Y placement relation.
            if re.search(
                r"\bagainst\s+(?:the\s+|a\s+|an\s+)?" r"(?:[a-z]+\s+){0,3}?wall\s*$",
                match.group("subject"),
            ):
                continue
            subject = selector_for_phrase(match.group("subject"))
            target = (
                previous_selector
                if match.group("target") == "it"
                else selector_for_phrase(match.group("target"))
            )
            if subject is not None and target is not None:
                constraints.append(
                    _constraint(
                        "behind",
                        subject,
                        dict(target),
                        source="explicit_prompt",
                        evidence_span=clause,
                    )
                )

        for match in re.finditer(
            r"(?P<subject>[a-z0-9_\- ,']{1,70}?)\s+"
            r"(?:is |sits |lies |placed |positioned )?"
            r"(?P<centered>centered|centred)?\s*between\s+"
            r"(?:the\s+)?(?P<anchors>[a-z0-9_\- ']{1,50}?)(?:[,.;]|$)",
            normalized,
        ):
            subject = selector_for_phrase(match.group("subject"))
            anchors = selector_for_phrase(match.group("anchors"), count=2)
            if (
                subject is None
                or anchors is None
                or not _phrase_mentions_plural(
                    match.group("anchors"), str(anchors.get("category") or "")
                )
            ):
                continue
            targets = dict(anchors)
            targets["count"] = 2
            targets["quantifier"] = "all"
            targets["secondary_category"] = anchors["category"]
            targets["secondary_count"] = 2
            if anchors.get("role"):
                targets["secondary_role"] = anchors["role"]
            constraints.append(
                _constraint(
                    (
                        "centered_between"
                        if match.group("centered")
                        or re.match(r"\s*cent(?:er|re)\b", normalized)
                        else "between"
                    ),
                    subject,
                    targets,
                    source="explicit_prompt",
                    evidence_span=clause,
                )
            )

        for match in re.finditer(
            r"(?P<anchors>(?:two|2)\s+[a-z0-9_\- ']{1,50}?)\b.*?"
            r"(?:with|and)\s+(?:a\s+|an\s+|the\s+)?"
            r"(?P<subject>[a-z0-9_\- ']{1,50}?)\s+"
            r"(?P<centered>centered\s+)?between\s+them\b",
            normalized,
        ):
            anchors = selector_for_phrase(match.group("anchors"), count=2)
            subject = selector_for_phrase(match.group("subject"))
            if anchors is None or subject is None:
                continue
            targets = dict(anchors)
            targets["count"] = 2
            targets["quantifier"] = "all"
            targets["secondary_category"] = anchors["category"]
            targets["secondary_count"] = 2
            if anchors.get("role"):
                targets["secondary_role"] = anchors["role"]
            constraints.append(
                _constraint(
                    (
                        "centered_between"
                        if match.group("centered")
                        or re.match(r"\s*cent(?:er|re)\b", normalized)
                        else "between"
                    ),
                    subject,
                    targets,
                    source="explicit_prompt",
                    evidence_span=clause,
                )
            )

        for match in re.finditer(
            r"(?P<subject>[a-z0-9_\- ,']{1,70}?)\s+"
            r"(?:facing|faces|face)\s+(?:the\s+|a\s+|an\s+)?"
            r"(?P<target>[a-z0-9_\- ,']{1,70}?)(?=\s+with\b|[,.;]|$)",
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
            r"(?P<adjacency>tucked\s+under|at|beside|next\s+to)\s+"
            r"(?:the\s+|a\s+|an\s+)?"
            r"(?P<target>(?:[a-z]+\s+){0,2}(?:desk|table|monitor|screen))\b",
            normalized,
        ):
            subject = selector_for_phrase(match.group("subject"))
            target = selector_for_phrase(match.group("target"))
            if subject is not None and target is not None:
                adjacency = match.group("adjacency")
                relation = (
                    "aligned_with"
                    if adjacency.startswith("tucked")
                    else "next_to" if adjacency in {"beside", "next to"} else "near"
                )
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
            r"(?P<adjacency>beside|next\s+to|adjacent\s+to|near)\s+"
            r"(?:the\s+|a\s+|an\s+)?"
            r"(?P<target>[a-z0-9_\- ,']{1,70}?)(?:[,.;]|$)",
            normalized,
        ):
            subject_phrase = re.sub(
                r"\s+on\s+(?:the\s+)?floor\b.*$", "", match.group("subject")
            )
            subject = selector_for_phrase(subject_phrase)
            target = selector_for_phrase(match.group("target"))
            if subject is not None and target is not None:
                constraints.append(
                    _constraint(
                        ("near" if match.group("adjacency") == "near" else "next_to"),
                        subject,
                        target,
                        source="explicit_prompt",
                        evidence_span=clause,
                    )
                )

        for match in re.finditer(
            r"(?P<subject>[a-z0-9_\- ,']{1,70}?)\s+"
            r"(?:on|sits\s+on|resting\s+on|placed\s+on)\s+(?:the\s+|a\s+|an\s+)?"
            r"(?P<target>[a-z0-9_\- ,']{1,70}?)(?=\s+near\b|[,.;]|$)",
            normalized,
        ):
            # "a nightstand with a lamp on each side of the bed" describes
            # a lateral placement, not a support relation.  Do not turn it
            # into the nonsensical `nightstand on_top_of bed` contract.
            if re.search(r"\bon\s+(?:each|either|both)\s+side", match.group(0)):
                continue
            if re.search(r"\bwall\b", match.group("target")) or re.search(
                r"\b(?:mount|mounted|hang|hung|hanging)\b", match.group("subject")
            ):
                continue
            subject_phrase = re.split(r"\band\b", match.group("subject"))[-1]
            subject = selector_for_phrase(subject_phrase)
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

        appositive_flank = re.search(
            r"(?P<subject>(?:two|2)\s+[a-z0-9_\- ']{1,50}?),\s*"
            r"(?:one\s+)?(?:on|at)\s+each\s+side\s+of\s+"
            r"(?:the\s+|a\s+|an\s+)?(?P<target>[a-z0-9_\- ']{1,50}?)(?:[,.;]|$)",
            normalized,
        )
        if appositive_flank:
            subject = selector_for_phrase(appositive_flank.group("subject"), count=2)
            target = selector_for_phrase(appositive_flank.group("target"))
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

        clause_selector = selector_for_phrase(clause)
        if clause_selector is not None:
            previous_selector = clause_selector

    number_pattern = r"\d+|" + "|".join(
        re.escape(value) for value in _NUMBER_WORDS if value not in {"a", "an"}
    )
    for match in re.finditer(
        rf"(?P<subject>(?:{number_pattern})\s+[a-z0-9_\- ]{{1,60}}?)\s+"
        r"(?:are\s+)?(?:facing|face)\s+each\s+other\b",
        lowered,
    ):
        subject = selector_for_phrase(match.group("subject"))
        if subject is None or int(subject.get("count") or 0) < 2:
            continue
        constraints.append(
            _constraint(
                "across_from",
                subject,
                dict(subject),
                source="explicit_prompt",
                evidence_span=_first_sentence_with(lowered, "each other"),
            )
        )

    for match in re.finditer(
        r"(?P<subject>(?:two|2)\s+[a-z0-9_\- ']{1,50}?),\s*"
        r"(?:one\s+)?(?:on|at)\s+each\s+side\s+of\s+"
        r"(?:the\s+|a\s+|an\s+)?(?P<target>[a-z0-9_\- ']{1,50}?)(?:[,.;]|$)",
        lowered,
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
                    evidence_span=match.group(0),
                )
            )

    required_counts: dict[str, int] = {}
    for row in _explicit_required_count_constraints(prompt):
        category = str((row.get("subjects") or {}).get("category") or "")
        count = int((row.get("subjects") or {}).get("count") or 1)
        if category:
            required_counts[category] = max(required_counts.get(category, 0), count)
    route_pattern = re.compile(
        r"(?P<evidence>(?:keep|maintain|leave)?\s*(?:an?\s+|the\s+)?"
        r"(?:open|clear|unobstructed)\s+(?:walking\s+)?(?:route|path|access)\s+"
        r"from\s+(?:the\s+)?(?:entrance|entry|door)\s+to\s+"
        r"(?P<targets>[a-z0-9_\- ',]+?))(?=[.;]|$)",
        re.IGNORECASE,
    )
    for route in route_pattern.finditer(lowered):
        for phrase in re.split(r"\s+and\s+|\s*,\s*", route.group("targets")):
            target = selector_for_phrase(phrase)
            if target is None:
                continue
            category = str(target.get("category") or "")
            if _phrase_mentions_plural(phrase, category):
                specific = [
                    candidate
                    for candidate, count in required_counts.items()
                    if count > 1 and category in _selector_category_family(candidate)
                ]
                if len(specific) == 1:
                    category = specific[0]
                    target["category"] = category
                    target["count"] = required_counts[category]
                    target["quantifier"] = "all"
            constraints.append(
                _constraint(
                    "clear_access",
                    {"category": "entrance", "count": 1, "quantifier": "all"},
                    target,
                    source="explicit_prompt",
                    evidence_span=route.group("evidence"),
                )
            )

    # These phrases reserve the usable zone immediately in front of a named
    # item.  They are deliberately limited to explicit local-access wording:
    # a generic statement that a room feels open must not invent a hard check.
    local_access_patterns = (
        re.compile(
            r"\b(?:keep|leave|ensure)\s+(?:its\s+|the\s+)?"
            r"(?:front|front\s+side|access\s+area)\s+"
            r"(?:clear|accessible)\b",
            re.IGNORECASE,
        ),
    )
    for clause in clauses:
        if not any(pattern.search(clause) for pattern in local_access_patterns):
            continue
        subject = selector_for_phrase(clause)
        if subject is None:
            continue
        constraints.append(
            _constraint(
                "clear_access",
                subject,
                {"category": "room", "count": 1, "quantifier": "all"},
                source="explicit_prompt",
                evidence_span=clause,
            )
        )
    global_circulation_pattern = re.compile(
        r"\b(?:without\s+(?:blocking|obstructing)|while\s+(?:preserving|maintaining))\s+"
        r"(?:the\s+)?(?:circulation|traffic(?:\s+flow)?|walkway|walking\s+path)\b",
        re.IGNORECASE,
    )
    for clause in clauses:
        if not global_circulation_pattern.search(clause):
            continue
        destination = selector_for_phrase(clause)
        if destination is None:
            continue
        constraints.append(
            _constraint(
                "clear_access",
                {"category": "entrance", "count": 1, "quantifier": "all"},
                destination,
                source="explicit_prompt",
                evidence_span=clause,
            )
        )
    for match in re.finditer(
        r"(?P<items>[^,.;]{1,100}?)\s+at\s+each\s+"
        r"(?P<target>[a-z][a-z0-9' -]{0,40}?)(?=[,.;]|$)",
        lowered,
    ):
        target = selector_for_phrase(match.group("target"))
        target_count = required_counts.get(str((target or {}).get("category") or ""), 0)
        if target is None or target_count <= 1:
            continue
        target["count"] = target_count
        for item_phrase in re.split(r"\s+and\s+", match.group("items")):
            subject = selector_for_phrase(item_phrase, count=target_count)
            if subject is None:
                continue
            constraints.append(
                _constraint(
                    "near",
                    subject,
                    dict(target),
                    source="explicit_prompt",
                    evidence_span=match.group(0),
                )
            )

    constraints.extend(_explicit_per_target_constraints(lowered, required_counts))
    constraints.extend(_explicit_corner_distribution_constraints(lowered))

    # Group constraints often span a full sentence and are easier to recognise
    # independently from individual relation clauses.
    if re.search(r"\b(?:student\s+desks?|desks?)\b", lowered) and re.search(
        r"\beach\s+with\s+(?:a\s+)?chair\b|\bstudent\s+chairs?\b", lowered
    ):
        desk_count = required_counts.get("student_desk", 0)
        chair_count = max(
            required_counts.get("student_chair", 0),
            required_counts.get("chair", 0),
        )
        paired_count = desk_count if desk_count == chair_count else 0
        chair_selector: dict[str, Any] = {
            "category": "student_chair",
            "role": "student",
            "quantifier": "all",
        }
        desk_selector: dict[str, Any] = {
            "category": "student_desk",
            "role": "student",
            "quantifier": "all",
        }
        if paired_count > 0:
            chair_selector["count"] = paired_count
            desk_selector["count"] = paired_count
        constraints.append(
            _constraint(
                "paired_with",
                chair_selector,
                desk_selector,
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
    edge_constraint = _explicit_edge_distribution_constraint(
        lowered, required_counts=required_counts
    )
    if edge_constraint is not None:
        constraints.append(edge_constraint)
    return constraints


def _explicit_per_target_constraints(
    prompt: str, required_counts: dict[str, int]
) -> list[dict[str, Any]]:
    """Compile one-to-one pair/support wording without room-specific rules."""
    constraints: list[dict[str, Any]] = []
    pair_pattern = re.compile(
        r"(?:pair\s+)?each\s+(?P<target>[a-z][a-z0-9' -]{0,40}?)\s+"
        r"(?:with|to)\s+(?:exactly\s+)?(?:one|a|an)\s+"
        r"(?P<subject>[a-z][a-z0-9' -]{0,40}?)(?=\s+(?:positioned|placed|"
        r"facing|at|beside|near|on|for)\b|[,.;]|$)",
        re.IGNORECASE,
    )
    for match in pair_pattern.finditer(prompt):
        target = selector_for_phrase(match.group("target"))
        subject = selector_for_phrase(match.group("subject"))
        target_category = str((target or {}).get("category") or "")
        count = required_counts.get(target_category, 0)
        if target is None or subject is None or count <= 1:
            continue
        subject = dict(subject, count=count, quantifier="all")
        target = dict(target, count=count, quantifier="all")
        evidence = match.group(0)
        constraints.extend(
            [
                _constraint(
                    "required_count",
                    dict(subject, quantifier="at_least"),
                    None,
                    source="explicit_prompt",
                    evidence_span=evidence,
                ),
                _constraint(
                    "paired_with",
                    subject,
                    target,
                    source="explicit_prompt",
                    evidence_span=evidence,
                ),
            ]
        )

    support_pattern = re.compile(
        r"(?:exactly\s+)?(?:one|a|an)\s+"
        r"(?P<subject>[a-z][a-z0-9' -]{0,50}?)\s+"
        r"(?:placed\s+|supported\s+)?(?:on\s+top\s+of|on|at)\s+each\s+"
        r"(?P<target>[a-z][a-z0-9' -]{0,40}?)(?=[,.;]|$)",
        re.IGNORECASE,
    )
    for match in support_pattern.finditer(prompt):
        target = selector_for_phrase(match.group("target"))
        subject = selector_for_phrase(match.group("subject"))
        target_category = str((target or {}).get("category") or "")
        count = required_counts.get(target_category, 0)
        if target is None or subject is None or count <= 1:
            continue
        subject = dict(subject, count=count, quantifier="all")
        target = dict(target, count=count, quantifier="all")
        evidence = match.group(0)
        constraints.extend(
            [
                _constraint(
                    "required_count",
                    dict(subject, quantifier="at_least"),
                    None,
                    source="explicit_prompt",
                    evidence_span=evidence,
                ),
                _constraint(
                    "one_per_support",
                    subject,
                    target,
                    source="explicit_prompt",
                    evidence_span=evidence,
                ),
            ]
        )
    return constraints


def _explicit_corner_distribution_constraints(prompt: str) -> list[dict[str, Any]]:
    """Compile explicit multi-object, distinct-corner assignments."""
    number_pattern = r"\d+|" + "|".join(
        re.escape(value) for value in _NUMBER_WORDS if value not in {"a", "an"}
    )
    pattern = re.compile(
        rf"(?P<count>{number_pattern})\s+"
        r"(?P<subject>[a-z][a-z0-9' -]{0,60}?)\s+in\s+"
        rf"(?:(?:the\s+)?room['’]?s?\s+)?(?P<corners>{number_pattern})\s+"
        r"(?:distinct\s+)?(?:room\s+)?corners\b",
        re.IGNORECASE,
    )
    constraints: list[dict[str, Any]] = []
    for match in pattern.finditer(prompt):
        count = _leading_count(match.group("count")) or 0
        corner_count = _leading_count(match.group("corners")) or 0
        subject = selector_for_phrase(match.group("subject"), count=count)
        if subject is None or count < 2 or corner_count != count:
            continue
        subject = dict(subject, count=count, quantifier="all")
        evidence = match.group(0)
        constraints.extend(
            [
                _constraint(
                    "required_count",
                    dict(subject, quantifier="at_least"),
                    None,
                    source="explicit_prompt",
                    evidence_span=evidence,
                ),
                _constraint(
                    "corner_distribution",
                    subject,
                    {"category": "room", "count": 1, "quantifier": "all"},
                    source="explicit_prompt",
                    evidence_span=evidence,
                ),
            ]
        )
    return constraints


def _long_side_layout_table_selector(prompt: str) -> dict[str, Any]:
    """Resolve the named table type without broadening to incidental tables.

    A long-side seating instruction refers back to the table introduced in the
    prompt.  Keep a typed noun such as ``conference table`` as a typed
    selector, otherwise a generated side/auxiliary table can make the endpoint
    binding ambiguous.  Bare ``table`` remains supported for prompts that do
    not provide a more specific noun.
    """
    table_categories = {
        "dining_table",
        "conference_table",
        "coffee_table",
        "side_table",
    }
    matches: list[tuple[int, str]] = []
    for category, aliases in _CATEGORY_PATTERNS:
        if category not in table_categories:
            continue
        for alias in aliases:
            match = re.search(r"\b" + re.escape(alias) + r"\b", prompt)
            if match is not None:
                matches.append((match.start(), category))
    if matches:
        edge_reference = min(
            (
                position
                for token in ("long side", "long edge", "short side", "short edge")
                if (position := prompt.find(token)) >= 0
            ),
            default=len(prompt),
        )
        preceding = [match for match in matches if match[0] < edge_reference]
        _, category = max(preceding or matches)
        return {"category": category, "quantifier": "all"}
    return {"category": "table", "quantifier": "all"}


def _explicit_edge_distribution_constraint(
    prompt: str, *, required_counts: dict[str, int]
) -> dict[str, Any] | None:
    """Compile a complete rectangular-edge row for the legacy deterministic parser.

    The independent IntentCompiler is authoritative during critic-enabled runs.
    This helper remains useful for fixtures and degraded tooling, so it must emit
    a complete v4 row or emit nothing; partial topology rows are unsafe.
    """
    if not re.search(r"\b(?:chair|seat)s?\b", prompt):
        return None
    target = _long_side_layout_table_selector(prompt)
    if target.get("category") == "table":
        if not re.search(r"\btable\b", prompt):
            return None
    target["count"] = 1
    target["quantifier"] = "all"

    subject_category = _edge_subject_category(prompt, required_counts)
    if not subject_category:
        return None
    subject_count = max(
        (
            count
            for category, count in required_counts.items()
            if category in _selector_category_family(subject_category)
            or subject_category in _selector_category_family(category)
        ),
        default=0,
    )
    if subject_count <= 0:
        for match in re.finditer(
            r"\b(\d+|" + "|".join(_NUMBER_WORDS) + r")\s+"
            r"(?:[a-z0-9_-]+\s+){0,2}(?:chairs?|seats?)\b",
            prompt,
        ):
            count = _leading_count(match.group(0)) or 0
            subject_count = max(subject_count, count)
    if subject_count <= 0:
        if "dining" in prompt and re.search(
            r"\bone\s+(?:on|at)\s+each\s+(?:side|edge)\b", prompt
        ):
            subject_count = 4
        else:
            return None

    groups: list[dict[str, Any]] = []
    long_group: list[int] | None = None
    short_group: list[int] | None = None
    equal_group = re.search(
        r"\b(?:two\s+)?equal\s+groups?\s+of\s+"
        r"(\d+|" + "|".join(_NUMBER_WORDS) + r")\b",
        prompt,
    )
    long_each = re.search(
        r"\b(?P<count>\d+|"
        + "|".join(_NUMBER_WORDS)
        + r")\s+(?:(?:chairs?|seats?)\s+)?(?:evenly\s+spaced\s+)?"
        r"(?:on|along)\s+(?:each|both)\b[^.]{0,40}\b"
        r"(?:two\s+)?long\s+(?:side|edge)s?\b",
        prompt,
    )
    if long_each is None:
        long_each = re.search(
            r"\b(?P<count>\d+|"
            + "|".join(_NUMBER_WORDS)
            + r")\s+(?:[a-z0-9_-]+\s+){0,2}(?:chairs?|seats?)\b"
            r"[^.]{0,100}\b(?:each|both)\b[^.]{0,40}\b"
            r"(?:two\s+)?long\s+(?:side|edge)s?\b",
            prompt,
        )
    if equal_group and re.search(r"\blong\s+(?:side|edge)s?\b", prompt):
        per_edge = _leading_count(equal_group.group(1)) or 0
        if per_edge > 0:
            long_group = [per_edge, per_edge]
    elif long_each:
        per_edge = _leading_count(long_each.group("count")) or 0
        if per_edge > 0:
            long_group = [per_edge, per_edge]
    elif re.search(r"\btwo\s+long\s+(?:side|edge)s?\b", prompt):
        if subject_count % 2 == 0:
            long_group = [subject_count // 2, subject_count // 2]
    elif re.search(r"\b(?:one|a|single)\s+long\s+(?:side|edge)\b", prompt):
        long_group = [subject_count, 0]

    each_long = re.search(r"\bone\s+(?:on|at)\s+each\s+long\s+(?:side|edge)\b", prompt)
    each_short = re.search(
        r"\bone\s+(?:on|at)\s+each\s+short\s+(?:side|edge)\b", prompt
    )
    each_any = re.search(r"\bone\s+(?:on|at)\s+each\s+(?:side|edge)\b", prompt)
    if each_long:
        long_group = [1, 1]
    elif each_any and "dining" in prompt and not each_short:
        long_group = [1, 1]
        short_group = [1, 1]
    if each_short:
        short_group = [1, 1]

    one_short = re.search(
        r"\b(?:one|a|single|one\s+remaining)\b[^.]{0,80}?"
        r"(?:short\s+(?:side|edge))\b",
        prompt,
    )
    if one_short:
        short_group = [1, 0]

    if long_group is None and short_group is None:
        return None
    occupied = sum(long_group or []) + sum(short_group or [])
    if occupied != subject_count:
        if long_group is not None and short_group is None:
            remainder = subject_count - sum(long_group)
            if remainder == 1 and re.search(r"\bremaining\b", prompt):
                short_group = [1, 0]
                occupied += 1
        if occupied != subject_count:
            return None
    if long_group is not None:
        groups.append(
            {
                "edge_class": "long",
                "counts_per_edge": sorted(long_group, reverse=True),
                "spacing": "equal_segments",
            }
        )
    if short_group is not None:
        groups.append(
            {
                "edge_class": "short",
                "counts_per_edge": sorted(short_group, reverse=True),
                "spacing": "equal_segments",
            }
        )
    all_subjects_face_target = bool(
        re.search(
            r"\b(?:all|every)\s+(?:the\s+)?(?:chairs?|seats?)\b"
            r"[^.]{0,60}\bfac(?:e|es|ing)\s+(?:the\s+)?(?:table|target)\b",
            prompt,
        )
        or re.search(
            r"\ball\s+facing\s+(?:the\s+)?(?:table|target)\b",
            prompt,
        )
        or re.search(r"\b(?:all|every)\s+the\s+way\s+facing\b", prompt)
        or re.search(r"\bface\s+inward\b", prompt)
    )
    return _constraint(
        "edge_distribution",
        {
            "category": subject_category,
            "count": subject_count,
            "quantifier": "all",
        },
        target,
        source="explicit_prompt",
        evidence_span=_first_sentence_with(prompt, "long side"),
        edge_frame="target_local_rectangle",
        groups=groups,
        orientation="toward_target" if all_subjects_face_target else "unconstrained",
    )


def _edge_subject_category(prompt: str, required_counts: dict[str, int]) -> str:
    candidates: list[tuple[int, int, str]] = []
    for category, aliases in _CATEGORY_PATTERNS:
        if "chair" not in category and "seat" not in category:
            continue
        for alias in aliases:
            match = re.search(r"\b" + re.escape(alias) + r"s?\b", prompt)
            if match is not None:
                candidates.append((match.start(), -len(alias), category))
    if candidates:
        return min(candidates)[2]
    for category in required_counts:
        if "chair" in category or "seat" in category:
            return category
    return ""


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
        if category in {"floor", "room", "wall", "ceiling"}:
            continue
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
    filtered_candidates = []
    for span, candidate in candidates.items():
        if any(
            other_span != span and other_span[0] <= span[0] and other_span[1] >= span[1]
            for other_span in candidates
        ):
            continue
        filtered_candidates.append(candidate)

    for category, count, evidence in filtered_candidates:
        quantifier = _prompt_count_quantifier(evidence)
        constraints.append(
            _constraint(
                "required_count",
                {"category": category, "count": count, "quantifier": quantifier},
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

    explicit_counts = {
        str((row.get("subjects") or {}).get("category") or ""): int(
            (row.get("subjects") or {}).get("count") or 1
        )
        for row in constraints
        if row.get("relation") == "required_count"
    }
    distributed_pattern = re.compile(
        r"(?P<items>[^,.;]{1,100}?)\s+(?:at|on)\s+each\s+"
        r"(?P<target>[a-z][a-z0-9' -]{0,40}?)(?=[,.;]|$)",
        re.IGNORECASE,
    )
    for match in distributed_pattern.finditer(prompt):
        target = selector_for_phrase(match.group("target"))
        if target is None:
            continue
        count = explicit_counts.get(str(target.get("category") or ""), 0)
        if count <= 1:
            continue
        for item_phrase in re.split(r"\s+and\s+", match.group("items")):
            item = selector_for_phrase(item_phrase)
            if item is None:
                continue
            category = str(item.get("category") or "")
            if not category or explicit_counts.get(category, 0) >= count:
                continue
            constraints.append(
                _constraint(
                    "required_count",
                    {
                        "category": category,
                        "count": count,
                        "quantifier": "at_least",
                        **({"role": item["role"]} if item.get("role") else {}),
                    },
                    None,
                    source="explicit_prompt",
                    evidence_span=match.group(0),
                )
            )
            explicit_counts[category] = count

    setting_pattern = re.compile(
        rf"(?P<count>{number_pattern})\s+(?:complete\s+)?"
        r"(?:table|place)\s+settings?\b[^.]{0,120}?"
        r"(?:including|with|contains?)\s+(?P<items>[^.;]+)",
        re.IGNORECASE,
    )
    for match in setting_pattern.finditer(prompt):
        count = _leading_count(match.group("count")) or 0
        if count <= 0:
            continue
        for item_phrase in re.split(r"\s*,\s*|\s+and\s+", match.group("items")):
            item = selector_for_phrase(item_phrase, count=count)
            category = str((item or {}).get("category") or "")
            if item is None or category not in {"plate", "cutlery", "glass"}:
                continue
            constraints.append(
                _constraint(
                    "required_count",
                    dict(item, count=count, quantifier="at_least"),
                    None,
                    source="explicit_prompt",
                    evidence_span=match.group(0),
                )
            )
    return constraints


def _prompt_count_quantifier(evidence: str) -> str:
    """Return an exact bound only when prompt wording explicitly requires it."""
    text = " ".join(str(evidence or "").lower().split())
    if re.search(
        r"\b(?:exactly|precisely|only|no\s+more\s+than|at\s+most|a\s+total\s+of)\b",
        text,
    ):
        return "exactly"
    return "at_least"


def _room_ontology_constraints(
    room_type: str,
    lowered: str,
    *,
    task_spec: Any | None = None,
) -> list[dict[str, Any]]:
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
    if _instructional_workstation_topology(room_type, lowered, task_spec):
        # This is a role-level functional profile, not a classroom coordinate
        # recipe. The furniture-stage relation reserves the operator's side of
        # a presenter work surface against whichever wall becomes the teaching
        # front. The wall-stage relation then binds a later-created board or
        # screen to that same presenter centerline.
        constraints.extend(
            (
                _constraint(
                    "operation_zone_at_wall",
                    {
                        "category": "teacher_desk",
                        "count": 1,
                        "quantifier": "all",
                    },
                    {"category": "wall"},
                    source="room_ontology",
                    evidence_span="presenter workstation operation-zone ontology",
                ),
                _constraint(
                    "instructional_surface_alignment",
                    {
                        "category": "instructional_surface",
                        "count": 1,
                        "quantifier": "all",
                    },
                    {
                        "category": "teacher_desk",
                        "count": 1,
                        "quantifier": "all",
                    },
                    source="room_ontology",
                    evidence_span="instructional focal-surface alignment ontology",
                ),
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
                inference_reason=(
                    "The prompt names a television with a media support and does "
                    "not explicitly require wall mounting."
                ),
            )
        )
    return constraints


def _instructional_workstation_topology(
    room_type: str,
    prompt: str,
    task_spec: Any | None,
) -> bool:
    """Recognize a presenter/audience/focal-surface topology without poses."""

    def values(field: str) -> list[str]:
        if isinstance(task_spec, dict):
            raw = task_spec.get(field) or []
        else:
            raw = getattr(task_spec, field, []) or [] if task_spec is not None else []
        return [str(item).lower().replace("_", " ") for item in raw]

    large_objects = values("required_large_objects")
    wall_objects = values("required_wall_objects")
    inventory = " ".join((prompt, *large_objects, *wall_objects))
    zones = " ".join(values("functional_zones"))
    presenter_surface_pattern = re.compile(
        r"\b(?:teacher(?:'s|s')?|instructor(?:'s|s')?)\s+"
        r"(?:desk|table|workstation)\b"
    )
    has_presenter_surface = any(
        presenter_surface_pattern.search(item) is not None
        for item in (prompt, *large_objects)
    )
    has_focal_surface = any(
        token in inventory
        for token in (
            "chalkboard",
            "blackboard",
            "whiteboard",
            "projection screen",
            "projector screen",
            "teaching screen",
        )
    )
    has_audience = any(token in inventory for token in ("student", "audience"))
    has_teaching_zone = any(
        token in zones
        for token in ("teaching zone", "instruction zone", "lecture zone")
    )
    return (
        has_presenter_surface
        and has_focal_surface
        and has_audience
        and (room_type == "classroom" or has_teaching_zone)
    )


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
    """TaskCompiler no longer owns hard relations."""
    return []


def _task_spec_inventory_constraints(task_spec: Any | None) -> list[dict[str, Any]]:
    """Compile normalized SceneTaskSpec inventory into authoritative counts."""

    payload = _task_spec_payload(task_spec)
    counts: dict[str, int] = {}
    fields_by_category: dict[str, str] = {}
    for field in (
        "required_large_objects",
        "required_wall_objects",
        "required_ceiling_objects",
        "required_small_objects",
    ):
        for value in payload.get(field) or []:
            category = _normalize_selector_category(value)
            if (
                not category
                or is_structural_anchor(category)
                or category in {"room", "wall", "floor", "ceiling"}
            ):
                continue
            counts[category] = counts.get(category, 0) + 1
            fields_by_category.setdefault(category, field)

    return [
        _constraint(
            "required_count",
            {
                "category": category,
                "count": count,
                # TaskSpec inventory records coverage, not an exact upper
                # bound. Explicit prompt wording is reduced separately and
                # wins conflicts in _deduplicate_constraints().
                "quantifier": "at_least",
            },
            None,
            source="task_compiler_inventory",
            evidence_span="",
            inference_reason=f"SceneTaskSpec {fields_by_category[category]}",
        )
        for category, count in sorted(counts.items())
    ]


def _task_spec_inventory_conflict_warnings(
    prompt_inventory: list[dict[str, Any]],
    task_inventory: list[dict[str, Any]],
) -> list[str]:
    prompt_counts = {
        str((row.get("subjects") or {}).get("category") or ""): int(
            (row.get("subjects") or {}).get("count") or 0
        )
        for row in prompt_inventory
    }
    warnings: list[str] = []
    for row in task_inventory:
        subjects = row.get("subjects") or {}
        category = str(subjects.get("category") or "")
        task_count = int(subjects.get("count") or 0)
        prompt_count = prompt_counts.get(category)
        if prompt_count is not None and prompt_count != task_count:
            warnings.append(
                "Inventory count conflict for "
                f"{category}: prompt={prompt_count}, SceneTaskSpec={task_count}; "
                "explicit prompt cardinality wins; inventory remains a minimum"
            )
    return warnings


def _task_spec_payload(task_spec: Any | None) -> dict[str, Any]:
    if task_spec is None:
        return {}
    if isinstance(task_spec, dict):
        return dict(task_spec)
    if hasattr(task_spec, "model_dump"):
        return task_spec.model_dump(mode="json", exclude_none=True)
    return {}


def _apply_task_spec_contract_metadata(
    constraints: list[dict[str, Any]], task_spec: Any | None
) -> list[dict[str, Any]]:
    """Derive endpoint stage and existential selectors from typed inventory.

    When a TaskSpec supplies inventory, its generated ``required_count`` rows
    are authoritative. Prompt parsing can find overlapping noun fragments in a
    compound label (for example ``sofa`` and ``chair`` inside
    ``sofa_chair``); retaining those fragments would require extra physical
    objects that the TaskSpec never requested.
    """

    payload = _task_spec_payload(task_spec)
    category_stages: dict[str, str] = {}
    category_counts: dict[str, int] = {}
    for field, stage in (
        ("required_large_objects", "furniture"),
        ("required_wall_objects", "wall_mounted"),
        ("required_ceiling_objects", "ceiling_mounted"),
        ("required_small_objects", "manipuland"),
    ):
        for value in payload.get(field) or []:
            category = _normalize_selector_category(value)
            if not category:
                continue
            category_stages[category] = stage
            category_counts[category] = category_counts.get(category, 0) + 1

    result: list[dict[str, Any]] = []
    for original in constraints:
        constraint = dict(original)
        # Keep explicit prompt rows. Their evidence and quantifier are merged
        # with inventory coverage after typed selector normalization.
        subjects = _canonicalize_selector_to_typed_inventory(
            constraint.get("subjects"), category_counts
        )
        constraint["subjects"] = subjects
        subject_category = _normalize_selector_category(subjects.get("category"))
        subject_stage = _typed_inventory_stage(subject_category, category_stages)
        subject_count = _typed_inventory_count(subject_category, category_counts)
        if subject_stage:
            constraint["stage"] = subject_stage
        if subject_count == 1 and not subjects.get("count"):
            subjects["count"] = 1
            constraint["subjects"] = subjects
        evidence = str(constraint.get("evidence_span") or "").lower()
        if subjects.get("count") == 1 and (
            re.search(r"\b(?:few|several|multiple)\b", evidence)
            or _phrase_mentions_plural(evidence, subject_category)
        ):
            subjects["quantifier"] = "minimum"
            constraint["subjects"] = subjects

        # "Table/place settings for N" specifies at least N usable setting
        # components.  A setting may legitimately contain both a fork and a
        # knife, so treating N as the exact number of generated ``cutlery``
        # assets makes the relation unbindable.  Minimum binding still emits a
        # support check for every matching asset present in the scene.
        if _is_place_setting_support_constraint(constraint, evidence):
            if not subjects.get("count") and subject_count > 0:
                subjects["count"] = subject_count
            subjects["quantifier"] = "minimum"
            constraint["subjects"] = subjects

        raw_targets = constraint.get("targets")
        targets = _canonicalize_selector_to_typed_inventory(
            raw_targets, category_counts
        )
        if raw_targets is not None:
            constraint["targets"] = targets
        target_category = _normalize_selector_category(targets.get("category"))
        target_count = _typed_inventory_count(target_category, category_counts)
        if target_count == 1 and not targets.get("count"):
            targets["count"] = 1
            constraint["targets"] = targets
        endpoint_stages = [
            endpoint_stage
            for category in (
                subject_category,
                target_category,
                _normalize_selector_category(targets.get("secondary_category")),
            )
            if (endpoint_stage := _typed_inventory_stage(category, category_stages))
        ]
        if endpoint_stages:
            current_stage = str(
                constraint.get("stage")
                or relation_spec(str(constraint.get("relation") or "")).earliest_stage
            )
            constraint["stage"] = max(
                [current_stage, *endpoint_stages], key=STAGE_ORDER.index
            )
        evidence_has_each = re.search(r"\b(?:each|every)\b", evidence) is not None
        try:
            requested_subject_count = int(subjects.get("count") or 0)
        except (TypeError, ValueError):
            requested_subject_count = 0
        inferred_shared_target = (
            constraint.get("source") == "model_inferred"
            and relation_spec(str(constraint.get("relation") or "")).target_arity == 1
            and subject_count > 1
            and target_count == 1
            and requested_subject_count <= 1
        )
        if inferred_shared_target:
            # A compiler often represents a shared target as one singular
            # endpoint (for example, nightstands next to one bed).  The typed
            # inventory determines the subject group cardinality, preventing
            # an otherwise valid relation from becoming an ambiguous binding.
            subjects["count"] = subject_count
            subjects["quantifier"] = "all"
            targets["count"] = 1
            targets["quantifier"] = "all"
            constraint["subjects"] = subjects
            constraint["targets"] = targets
        inferred_group = (
            constraint.get("source") == "model_inferred"
            and subject_count > 1
            and subject_count == target_count
        )
        if (
            relation_spec(str(constraint.get("relation") or "")).target_arity == 1
            and subject_count > 1
            and subject_count == target_count
            and (evidence_has_each or inferred_group)
        ):
            subjects["count"] = subject_count
            subjects["quantifier"] = "all"
            targets["count"] = target_count
            targets["quantifier"] = "all"
            constraint["subjects"] = subjects
            constraint["targets"] = targets

        if (
            constraint.get("relation") in {"across_from", "clear_access"}
            and subject_category == target_category
            and subject_count == 2
        ):
            subjects["count"] = 2
            subjects["quantifier"] = "all"
            targets["count"] = 2
            targets["quantifier"] = "all"
            constraint["subjects"] = subjects
            constraint["targets"] = targets

        if (
            constraint.get("relation") == "centered_in_room"
            and subject_count > 1
            and not re.search(r"\b(?:each|every)\b", evidence)
            and not re.search(
                rf"\b(?:a|an|one|single)\s+(?:[a-z0-9_-]+\s+){{0,2}}"
                rf"{re.escape(subject_category.replace('_', ' '))}\b",
                evidence,
            )
        ):
            continue
        if target_count > 1:
            phrase = re.escape(target_category.replace("_", " "))
            target_nouns = _selector_evidence_noun_pattern(target_category)
            explicitly_one = (
                targets.get("count") == 1
                or re.search(
                    rf"\b(?:one|any|other|another)\s+(?:[a-z0-9_-]+\s+){{0,2}}{phrase}s?\b",
                    evidence,
                )
                # A relation may elide the repeated noun entirely: "the
                # other" resolves to one member of its already typed target
                # group, rather than every matching object in the room.
                or re.search(r"\b(?:the\s+)?(?:another|(?<!each\s)other)\b", evidence)
                # Scope "on one side" to this relation's target noun.  An
                # unrelated clause in a wider evidence span must not weaken a
                # plural target into an existential one.  The compiler can
                # retain a typed selector (``dining_chair``) while its exact
                # prompt evidence uses a broad noun (``chairs``), so accept
                # the selector's safe family aliases here.
                or re.search(
                    rf"\b(?:{target_nouns})(?:s|es)?\s+on\s+"
                    r"(?:the\s+)?one\s+side\b",
                    evidence,
                )
            )
            if explicitly_one:
                targets["count"] = 1
                targets["quantifier"] = "minimum"
                constraint["targets"] = targets
        result.append(constraint)
    return result


def _canonicalize_selector_to_typed_inventory(
    selector: Any, category_counts: dict[str, int]
) -> dict[str, Any]:
    """Replace an unambiguous broad selector with its typed inventory category.

    Task compilers can emit a family noun (for example ``chair``) while the
    required inventory and the deterministic prompt parser retain a concrete
    category (for example ``dining_chair``).  Canonicalizing only a unique
    compatible category lets the ordinary contract de-duplication remove that
    duplicate relation without losing ambiguity when multiple chair types are
    requested.
    """
    normalized = dict(selector or {}) if isinstance(selector, dict) else {}
    category = _normalize_selector_category(normalized.get("category"))
    typed_categories = _typed_inventory_categories(category, category_counts)
    if len(typed_categories) == 1 and typed_categories[0] != category:
        normalized["category"] = typed_categories[0]

    secondary_category = _normalize_selector_category(
        normalized.get("secondary_category")
    )
    secondary_typed_categories = _typed_inventory_categories(
        secondary_category, category_counts
    )
    if (
        len(secondary_typed_categories) == 1
        and secondary_typed_categories[0] != secondary_category
    ):
        normalized["secondary_category"] = secondary_typed_categories[0]
    return normalized


def _typed_inventory_categories(
    selector_category: str, category_values: dict[str, Any]
) -> list[str]:
    """Resolve broad selectors against compatible typed inventory categories."""
    category = _normalize_selector_category(selector_category)
    family = _selector_category_family(category)
    if len(family) <= 2:
        return [category] if category in category_values else []
    return [candidate for candidate in category_values if candidate in family]


def _typed_inventory_count(
    selector_category: str, category_counts: dict[str, int]
) -> int:
    return sum(
        category_counts[category]
        for category in _typed_inventory_categories(selector_category, category_counts)
    )


def _typed_inventory_stage(
    selector_category: str, category_stages: dict[str, str]
) -> str:
    stages = [
        category_stages[category]
        for category in _typed_inventory_categories(selector_category, category_stages)
    ]
    return max(stages, key=STAGE_ORDER.index) if stages else ""


def _selector_evidence_noun_pattern(selector_category: str) -> str:
    """Return prompt nouns that safely refer to a typed selector family."""
    categories = _selector_category_family(selector_category)
    nouns = {category.replace("_", " ") for category in categories if category}
    for category, aliases in _CATEGORY_PATTERNS:
        if category in categories:
            nouns.update(alias for alias in aliases if alias)
    return "|".join(re.escape(noun) for noun in sorted(nouns, key=len, reverse=True))


def _is_place_setting_support_constraint(
    constraint: dict[str, Any], evidence: str
) -> bool:
    """Whether a support relation describes a minimum number of place settings."""
    if str(constraint.get("relation") or "") != "on_top_of":
        return False
    subject_category = _normalize_selector_category(
        (constraint.get("subjects") or {}).get("category")
    )
    if subject_category not in {"cutlery", "flatware", "silverware"}:
        return False
    target_category = _normalize_selector_category(
        (constraint.get("targets") or {}).get("category")
    )
    if target_category not in _selector_category_family("table"):
        return False
    return (
        re.search(r"\b(?:cutlery|flatware|silverware)\b", evidence) is not None
        and re.search(
            r"\b(?:place|table)\s+settings?\s+(?:for\s+)?"
            r"(?:\d+|one|two|three|four|five|six|seven|eight)\b",
            evidence,
        )
        is not None
    )


def _normalize_external_constraint(raw: dict[str, Any]) -> dict[str, Any] | None:
    relation = str(raw.get("relation") or "").strip().lower()
    if relation not in VALID_RELATIONS:
        return None
    subjects = _normalize_selector(raw.get("subjects") or raw.get("subject"))
    targets = _normalize_selector(raw.get("targets") or raw.get("target"))
    if targets is None and relation in _WALL_TARGET_RELATIONS:
        raw_target = raw.get("targets") or raw.get("target")
        target_text = " ".join(str(raw_target or "").lower().split())
        if target_text in _WALL_TARGET_CATEGORIES:
            targets = {"category": target_text, "quantifier": "all"}
    if subjects is None:
        return None
    if relation == "one_per_support" and _is_seating_surface_pair(subjects, targets):
        # A chair is paired *at* a desk, not supported by the desk's top
        # surface.  Compilers occasionally map "each desk with a chair" to
        # one_per_support, whose geometry evaluator correctly expects an
        # actual parent support surface.  Preserve the one-to-one intent while
        # selecting the seating/work-surface evaluator instead.
        relation = "paired_with"
    if (
        relation in _WALL_TARGET_RELATIONS
        and _normalize_selector_category((targets or {}).get("category"))
        not in _WALL_TARGET_CATEGORIES
    ):
        # Wall-local evaluators intentionally select a physical wall.  A
        # malformed compiler target such as a table would otherwise move or
        # score the subject against an unrelated nearest wall.
        return None
    supplied_source = str(raw.get("source") or "model_inferred").strip().lower()
    if supplied_source not in {"explicit_prompt", "model_inferred"}:
        return None
    source = supplied_source
    evidence = str(raw.get("evidence_span") or raw.get("evidence") or "").strip()
    inference_reason = str(raw.get("inference_reason") or "").strip()
    if _is_unsupported_composite_direction_inference(
        relation, source, evidence, inference_reason
    ):
        return None
    edge_fields: dict[str, Any] = {}
    if relation == "edge_distribution":
        groups = raw.get("groups")
        if not isinstance(groups, list) or not groups:
            return None
        edge_fields = {
            "edge_frame": raw.get("edge_frame"),
            "groups": groups,
            "orientation": raw.get("orientation"),
        }
    return _constraint(
        relation,
        subjects,
        targets,
        source=source,
        evidence_span=evidence,
        inference_reason=inference_reason,
        confidence=_bounded_float(raw.get("confidence"), default=0.7),
        **edge_fields,
    )


def _is_seating_surface_pair(
    subjects: dict[str, Any], targets: dict[str, Any] | None
) -> bool:
    subject_category = _normalize_selector_category(subjects.get("category"))
    target_category = _normalize_selector_category((targets or {}).get("category"))
    return subject_category in {
        "chair",
        "office_chair",
        "guest_chair",
        "student_chair",
        "teacher_chair",
        "dining_chair",
        "armchair",
        "stool",
        "bench",
    } and target_category in {
        "desk",
        "student_desk",
        "teacher_desk",
        "reception_desk",
        "table",
        "dining_table",
        "conference_table",
    }


def _normalize_selector(value: Any) -> dict[str, Any] | None:
    if isinstance(value, str):
        return selector_for_phrase(value)
    if not isinstance(value, dict):
        return None
    category = _normalize_selector_category(value.get("category"))
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
    raw_selector = value
    for field in ("cohort", "support_target", "support_target_id", "stage"):
        context_value = raw_selector.get(field)
        if context_value:
            normalized[field] = "_".join(str(context_value).strip().lower().split())
    capabilities = raw_selector.get("capabilities")
    if capabilities is None:
        capabilities = raw_selector.get("required_capabilities")
    if capabilities:
        if isinstance(capabilities, str):
            capabilities = [capabilities]
        normalized["capabilities"] = sorted(
            {
                "_".join(str(item).strip().lower().split())
                for item in capabilities
                if str(item).strip()
            }
        )
    return normalized


def _is_unsupported_composite_direction_inference(
    relation: str, source: str, evidence: str, inference_reason: str
) -> bool:
    """Reject directional guesses that describe an asset's internal components.

    A retrieved ``vase_flowers``-style asset is one physical object.  Without
    prompt evidence its flower component cannot truthfully be evaluated as in
    front of, behind, or facing its vase component.  Explicit directions remain
    valid, as do model inferences whose rationale describes a real spatial use.
    """
    if (
        source != "model_inferred"
        or evidence
        or relation
        not in {
            "in_front_of",
            "behind",
            "faces",
            "aligned_with",
        }
    ):
        return False
    normalized_reason = " ".join(inference_reason.lower().split())
    return any(
        phrase in normalized_reason
        for phrase in (
            "inside",
            "within",
            "emerge from",
            "emerging from",
            "part of",
            "component of",
            "contained in",
            "integrated in",
        )
    )


def _constraint(
    relation: str,
    subjects: dict[str, Any],
    targets: dict[str, Any] | None,
    *,
    source: str,
    evidence_span: str,
    inference_reason: str = "",
    confidence: float = 1.0,
    edge_frame: str | None = None,
    groups: list[dict[str, Any]] | None = None,
    orientation: str | None = None,
) -> dict[str, Any]:
    normalized_evidence = " ".join(str(evidence_span or "").split())
    normalized_inference_reason = " ".join(str(inference_reason or "").split())
    digest = hashlib.sha1(
        _stable_json(
            [
                relation,
                subjects,
                targets,
                source,
                normalized_evidence,
                normalized_inference_reason,
                edge_frame,
                groups,
                orientation,
            ]
        ).encode("utf-8")
    ).hexdigest()[:12]
    result = {
        "constraint_id": f"intent_{digest}",
        "relation": relation,
        "subjects": subjects,
        "strength": "hard",
        "source": source,
        "confidence": _bounded_float(confidence, default=1.0),
        "evidence_span": normalized_evidence,
        "inference_reason": normalized_inference_reason,
        "stage": relation_spec(relation).earliest_stage,
    }
    if targets is not None:
        result["targets"] = targets
    if edge_frame is not None:
        result["edge_frame"] = edge_frame
    if groups is not None:
        result["groups"] = groups
    if orientation is not None:
        result["orientation"] = orientation
    return result


def _merge_provenance_text(*values: Any) -> str:
    """Join source explanations without losing conflict provenance."""
    parts: list[str] = []
    for value in values:
        text = " ".join(str(value or "").split())
        if text and text not in parts:
            parts.append(text)
    return " | ".join(parts)


def _deduplicate_constraints(constraints: list[dict[str, Any]]) -> list[dict[str, Any]]:
    priority = {
        "explicit_prompt": 0,
        "room_ontology": 2,
        "model_inferred": 3,
        "task_compiler_inventory": 4,
        "vlm_observation": 5,
    }
    keyed: dict[tuple[str, str, str, str, str, str], dict[str, Any]] = {}
    required_by_category: dict[tuple[str, str], dict[str, Any]] = {}
    for constraint in constraints:
        relation = str(constraint.get("relation") or "")
        subjects = _normalize_selector(constraint.get("subjects")) or {}
        targets = _normalize_selector(constraint.get("targets")) or {}
        if relation == "required_count":
            category = str(subjects.get("category") or "")
            role = str(subjects.get("role") or "")
            key = (category, role)
            previous = required_by_category.get(key)
            if previous is None:
                required_by_category[key] = constraint
            else:
                previous_source = str(previous.get("source") or "")
                current_source = str(constraint.get("source") or "")
                previous_count = int((previous.get("subjects") or {}).get("count") or 0)
                current_count = int(subjects.get("count") or 0)
                previous_quantifier = str(
                    (previous.get("subjects") or {}).get("quantifier") or "at_least"
                )
                current_quantifier = str(subjects.get("quantifier") or "at_least")
                previous_exact = previous_quantifier == "exactly"
                current_exact = current_quantifier == "exactly"
                # Explicit prompt evidence owns an exact bound. Otherwise
                # merge lower bounds so inventory cannot turn a natural plural
                # into an exact upper bound or override an explicit number.
                if (
                    current_source == "explicit_prompt"
                    and previous_source != "explicit_prompt"
                ):
                    required_by_category[key] = constraint
                elif (
                    previous_source == "explicit_prompt"
                    and current_source != "explicit_prompt"
                ):
                    if current_count > previous_count and not previous_exact:
                        merged = dict(previous)
                        merged_subjects = dict(merged.get("subjects") or {})
                        merged_subjects["count"] = current_count
                        merged_subjects["quantifier"] = "at_least"
                        merged["subjects"] = merged_subjects
                        merged["inference_reason"] = _merge_provenance_text(
                            merged.get("inference_reason"),
                            constraint.get("inference_reason"),
                        )
                        required_by_category[key] = merged
                elif current_exact and not previous_exact:
                    required_by_category[key] = constraint
                elif not current_exact and previous_exact:
                    pass
                elif current_count > previous_count:
                    required_by_category[key] = constraint
                elif current_count == previous_count and priority.get(
                    current_source, 9
                ) < priority.get(previous_source, 9):
                    required_by_category[key] = constraint
            continue
        key = (
            relation,
            _stable_json(subjects),
            _stable_json(targets),
            _stable_json(constraint.get("edge_frame")),
            _stable_json(constraint.get("groups")),
            str(constraint.get("orientation") or ""),
        )
        previous = keyed.get(key)
        if previous is None or priority.get(
            str(constraint.get("source")), 9
        ) < priority.get(str(previous.get("source")), 9):
            keyed[key] = constraint
    for (category, role), constraint in required_by_category.items():
        keyed[("required_count", category, role, "", "", "")] = constraint
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
        for item in re.split(r"\s*,\s*(?:and\s+)?|\s*;\s*", sentence)
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
        "adjacent",
    }:
        return (
            str(obj.get("category_norm") or obj.get("category") or "").lower() == "wall"
        )
    return role in identity


_GENERIC_SELECTOR_PARENTS = {
    "conference_table": "table",
    "dining_table": "table",
    "coffee_table": "table",
    "side_table": "table",
    "office_chair": "chair",
    "guest_chair": "chair",
    "dining_chair": "chair",
    "student_chair": "chair",
    "teacher_chair": "chair",
    "armchair": "chair",
    "stool": "chair",
    "bench": "chair",
    "student_desk": "desk",
    "teacher_desk": "desk",
    "reception_desk": "desk",
    # HSSD retrieval commonly keeps the stable asset label ``cabinet`` for a
    # prompt's more descriptive ``storage_cabinet`` request.  This fallback is
    # deliberately one-way and runs only after exact selector matching, so a
    # retrieved storage-cabinet asset still wins whenever it is available.
    "storage_cabinet": "cabinet",
}

_GENERIC_SELECTOR_SUFFIX_PARENTS = frozenset(
    {
        "bed",
        "chair",
        "desk",
        "lamp",
        "plant",
        "rug",
        "sofa",
        "table",
    }
)


def _generic_selector_parent(category: str) -> str | None:
    """Return a broad asset class only when a specialized selector has none.

    Asset retrieval often preserves a descriptive modifier in a prompt selector
    (for example, ``square_rug``) while the generated asset is labeled with its
    stable broad class (``rug``).  The fallback is deliberately directional and
    only runs after exact matching, so a specialized generated asset remains
    preferred whenever it is available.
    """
    normalized = _normalize_selector_category(category)
    parent = _GENERIC_SELECTOR_PARENTS.get(normalized)
    if parent is not None:
        return parent
    for candidate in _GENERIC_SELECTOR_SUFFIX_PARENTS:
        if normalized.endswith(f"_{candidate}"):
            return candidate
    return None


def _matches_category_or_instance_suffix(expected: str, observed: str) -> bool:
    """Match a semantic category or a generator-added numeric instance suffix."""
    if observed == expected:
        return True
    suffix = observed.removeprefix(f"{expected}_")
    return bool(suffix) and suffix.isdigit()


def _selector_matches_generic_fallback(
    category: str, role: str, obj: dict[str, Any]
) -> bool:
    """Match a broad asset label to a specialized selector conservatively."""
    parent = _generic_selector_parent(category)
    if parent is None:
        return False
    object_category_value = _normalize_selector_category(object_category(obj))
    base_category = _normalize_selector_category(
        obj.get("category_norm") or obj.get("category")
    )
    if object_category_value != parent and base_category != parent:
        return False
    return _role_matches_object(role, obj)


def _selector_matches_object(category: str, role: str, obj: dict[str, Any]) -> bool:
    category = _normalize_selector_category(category)
    normalized_category = category.replace("_", " ")
    object_cat = _normalize_selector_category(object_category(obj))
    metadata = obj.get("metadata") if isinstance(obj.get("metadata"), dict) else {}
    semantic_name = _normalize_selector_category(metadata.get("semantic_name"))
    hints = obj.get("functional_hints") or {}
    scene_object_type = (
        str(obj.get("object_type") or hints.get("scene_object_type") or "")
        .strip()
        .lower()
        .replace("-", "_")
    )
    base_category = _normalize_selector_category(
        obj.get("category_norm") or obj.get("category")
    )
    exact_adapter_category = _matches_category_or_instance_suffix(
        category, base_category
    )
    open_vocabulary_adapter_category = (
        exact_adapter_category and not is_known_object_category(category)
    )
    # Generated decor can retain a furniture category for retrieval, such as
    # ``dresser_mirror_tabletop``. It is still a manipuland and must not bind
    # a furniture contract endpoint solely because its parent noun appears in
    # the asset category.
    if (
        scene_object_type == "manipuland"
        and category not in MANIPULAND_CATEGORIES
        and not open_vocabulary_adapter_category
    ):
        return False
    if (
        scene_object_type == "wall_mounted"
        and category not in WALL_MOUNTED_CATEGORIES
        and not exact_adapter_category
    ):
        return False
    if (
        scene_object_type == "ceiling_mounted"
        and category not in CEILING_MOUNTED_CATEGORIES
        and not exact_adapter_category
    ):
        return False
    if _matches_category_or_instance_suffix(category, semantic_name):
        return _role_matches_object(role, obj)
    if _matches_category_or_instance_suffix(
        category, object_cat
    ) or _matches_category_or_instance_suffix(category, base_category):
        return _role_matches_object(role, obj)

    # Generated wall/ceiling decor often includes the furniture it accompanies
    # in its free-form identity (for example ``mirror_dresser`` or ``art_bed``).
    # Exact semantic categories above remain authoritative, but identity-token
    # fallback must not let later-stage decoration become a furniture endpoint.
    identity = " ".join(
        str(obj.get(key) or "").lower().replace("_", " ")
        for key in ("id", "name", "description")
    )
    identity = " ".join(
        value
        for value in (semantic_name.replace("_", " "), base_category, identity)
        if value
    )
    # A generated floral arrangement can be a single physical manipuland that
    # already includes its container.  Preserve that composite asset as one
    # object, while allowing it to satisfy both prompt endpoints when the
    # semantic identity explicitly establishes both parts.  A bare flower
    # arrangement must never stand in for a vase.
    is_floral_arrangement = any(
        phrase in identity for phrase in ("flower arrangement", "floral arrangement")
    )
    composite_type = _normalize_selector_category(metadata.get("composite_type"))
    container_asset = metadata.get("container_asset")
    fill_assets = metadata.get("fill_assets")
    container_identity = (
        " ".join(
            str(container_asset.get(key) or "").lower().replace("_", " ")
            for key in ("name", "description", "semantic_name")
        )
        if isinstance(container_asset, dict)
        else ""
    )
    fill_identity = (
        " ".join(
            " ".join(
                str(fill.get(key) or "").lower().replace("_", " ")
                for key in ("name", "description", "semantic_name")
            )
            for fill in fill_assets
            if isinstance(fill, dict)
        )
        if isinstance(fill_assets, list)
        else ""
    )
    composite_identity = " ".join(
        value for value in (identity, container_identity, fill_identity) if value
    )
    declared_capabilities = {
        str(value or "").strip().lower().replace("-", "_")
        for key in ("functional_categories", "candidate_affordances", "affordances")
        for value in (hints.get(key) or [])
        if str(value or "").strip()
    }
    if category in {"television", "monitor", "instructional_surface"} and (
        bool(hints.get("is_media_target")) or "media" in declared_capabilities
    ):
        return _role_matches_object(role, obj)
    is_filled_floral_vase = (
        composite_type in {"filled_container", "filled_vase"}
        and "vase" in composite_identity
        and re.search(r"\b(?:flower|flowers|floral)\b", composite_identity) is not None
    )
    is_floral_vase_composite = (
        is_floral_arrangement and "vase" in identity
    ) or is_filled_floral_vase
    category_matches = {
        "instructional_surface": any(
            token in identity
            for token in (
                "chalkboard",
                "blackboard",
                "whiteboard",
                "projection screen",
                "projector screen",
                "teaching screen",
            )
        ),
        "student_desk": base_category == "desk" and "student" in identity,
        "teacher_desk": base_category == "teacher_desk"
        or (
            base_category == "desk"
            and ("teacher" in identity or "instructor" in identity)
        ),
        "guest_chair": base_category in {"chair", "armchair", "office_chair"}
        and ("guest" in identity or "visitor" in identity),
        "rocking_chair": base_category == "rocking_chair"
        or semantic_name == "rocking_chair"
        or semantic_name.endswith("_rocking_chair"),
        "student_chair": base_category in {"chair", "office_chair", "dining_chair"}
        and "student" in identity,
        "dining_chair": base_category in {"dining_chair", "chair"}
        and "dining" in identity,
        "dining_table": base_category in {"dining_table", "table"}
        and "dining" in identity,
        "conference_table": base_category in {"conference_table", "table"}
        and any(
            token in identity
            for token in ("conference table", "meeting table", "boardroom table")
        ),
        "coffee_table": base_category in {"coffee_table", "table"}
        and "coffee" in identity,
        "tv_stand": base_category
        in {"tv_stand", "media_console", "entertainment_center"}
        or "tv stand" in identity,
        "television": base_category in {"television", "tv", "screen", "display"}
        or (
            base_category in {"", "unknown", "object"}
            and not _MEDIA_SUPPORT_PATTERN.search(identity)
            and ("television" in identity or re.search(r"\btv\b", identity) is not None)
        ),
        "office_chair": base_category in {"office_chair", "chair"}
        and (
            "office" in identity
            or "desk chair" in identity
            or base_category == "office_chair"
        ),
        # Retrieval preserves useful asset specificity in semantic names (for
        # example ``two_seater_sofa``). Prompt contracts use the family noun,
        # so bind that stable token without relying on free-form descriptions.
        "sofa": base_category == "sofa"
        or semantic_name == "sofa"
        or semantic_name.endswith("_sofa"),
        "desk": base_category == "desk"
        or base_category.endswith("_desk")
        or semantic_name.endswith("_desk"),
        "table_lamp": base_category.startswith("table_lamp")
        or base_category.startswith("desk_lamp")
        or semantic_name.startswith("table_lamp")
        or semantic_name.startswith("desk_lamp"),
        "plant": base_category == "plant"
        or base_category.endswith("_plant")
        or semantic_name.endswith("_plant"),
        "rug": base_category == "rug"
        or base_category.endswith("_rug")
        or semantic_name.endswith("_rug"),
        "vase": base_category in {"vase", "vase_flowers"}
        or semantic_name in {"vase", "vase_flowers"}
        or semantic_name.endswith("_vase")
        or is_floral_vase_composite,
        "coaster": base_category == "coaster" or semantic_name.endswith("_coaster"),
        "plate": base_category == "plate" or semantic_name.endswith("_plate"),
        "cutlery": base_category
        in {"cutlery", "flatware", "silverware", "fork", "knife", "spoon"}
        or semantic_name
        in {"cutlery", "flatware", "silverware", "fork", "knife", "spoon"},
        "glass": base_category in {"glass", "wine_glass", "drinking_glass", "tumbler"}
        or semantic_name in {"glass", "wine_glass", "drinking_glass", "tumbler"},
        "flower": base_category in {"flower", "flowers"}
        or semantic_name in {"flower", "flowers"}
        or is_floral_vase_composite,
        "table": base_category
        in {
            "table",
            "dining_table",
            "conference_table",
            "coffee_table",
            "side_table",
            "desk",
        },
        "chair": base_category
        in {
            "chair",
            "office_chair",
            "dining_chair",
            "student_chair",
            "guest_chair",
            "teacher_chair",
            "armchair",
            "stool",
            "bench",
        },
        "speaker": base_category in {"speaker", "floor_speaker"}
        or semantic_name in {"speaker", "floor_speaker"}
        or semantic_name.endswith("_speaker"),
        "wall_light": base_category
        in {"wall_light", "wall_lamp", "wall_sconce", "sconce"}
        or semantic_name in {"wall_light", "wall_lamp", "wall_sconce", "sconce"}
        or "wall light" in identity
        or "wall lamp" in identity
        or "sconce" in identity,
        "tray": base_category in {"tray", "serving_tray", "cafeteria_tray"}
        or semantic_name in {"tray", "serving_tray", "cafeteria_tray"}
        or re.search(r"\b(?:serving|cafeteria)?\s*tray\b", identity) is not None,
        "clock": base_category in {"clock", "alarm_clock", "bedside_clock"}
        or semantic_name in {"clock", "alarm_clock", "bedside_clock"}
        or re.search(r"\b(?:bedside|alarm|mantel)?\s*clock\b", identity) is not None,
        "cup": base_category in {"cup", "cup_of_tea", "tea_cup"}
        or semantic_name in {"cup", "cup_of_tea", "tea_cup"}
        or re.search(r"\b(?:cup|tea cup)\b", identity) is not None,
        "recliner": base_category in {"recliner", "armchair"}
        or semantic_name in {"recliner", "armchair"}
        or "recliner" in identity,
        "sculpture": base_category in {"sculpture", "wood_sculpture", "stone_sculpture"}
        or semantic_name in {"sculpture", "wood_sculpture", "stone_sculpture"}
        or re.search(r"\b(?:wood|stone|metal)?\s*sculpture\b", identity) is not None,
        "wall": base_category == "wall",
        "room": False,
    }.get(category)
    if category_matches is None:
        category_matches = (
            object_cat == category
            or base_category == category
            or (
                base_category in {"", "unknown", "object"}
                and _contains_phrase(identity, normalized_category)
            )
        )
    if not category_matches:
        return False
    return _role_matches_object(role, obj)


def _normalize_selector_category(value: Any) -> str:
    """Canonicalize known asset/prompt spellings used by semantic selectors."""
    normalized = re.sub(
        r"[^a-z0-9]+",
        "_",
        str(value or "").strip().lower().replace("\u2019", "'"),
    ).strip("_")
    if normalized in {
        "teacher_desk",
        "teachers_desk",
        "teacher_s_desk",
        "instructor_desk",
    }:
        return "teacher_desk"
    if normalized in {"computer_monitor", "computer_display", "display_monitor"}:
        return "monitor"
    if normalized in {
        "chalkboard",
        "blackboard",
        "whiteboard",
        "projection_screen",
        "projector_screen",
        "teaching_screen",
    }:
        return "instructional_surface"
    return canonical_selector_category(normalized)


def _fd_relation_for_constraint(constraint: dict[str, Any]) -> str | None:
    relation = str(constraint.get("relation") or "")
    evaluator = relation_spec(relation).evaluator
    if (
        evaluator == "object_on_support"
        and str((constraint.get("targets") or {}).get("category") or "") == "floor"
    ):
        return "object_on_floor"
    if evaluator == "paired_with":
        return "seating_to_work_surface"
    if evaluator in {"faces", "aligned_with"}:
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
    return evaluator if evaluator in _DIRECT_FD_EVALUATORS else None


def _relation_threshold_dependency(
    constraint: dict[str, Any],
) -> dict[str, Any]:
    relation = str(constraint.get("relation") or "")
    thresholds = relation_spec(str(constraint.get("relation") or "")).thresholds
    max_gap = thresholds.get("max_gap_m")
    dependency = {"max_distance_m": float(max_gap)} if max_gap is not None else {}
    for key in ("max_angle_deg", "max_degraded_angle_deg"):
        if key in thresholds:
            dependency[key] = float(thresholds[key])
    if relation == "faces":
        # Prompt-level facing is directional.  Pairing/near constraints own
        # interaction distance, while a classroom desk may legitimately face
        # a board several metres away.
        dependency["distance_required"] = False
        subject_category = _normalize_selector_category(
            (constraint.get("subjects") or {}).get("category")
        )
        if subject_category in {
            "desk",
            "student_desk",
            "teacher_desk",
            "reception_desk",
        }:
            # A desk's physical front faces its user.  "The desk faces the
            # board" describes the seated user's viewing direction, which is
            # the desk's back axis.
            dependency["subject_face"] = "back"
    return dependency


def _paired_seating_checks(
    constraint: dict[str, Any],
    *,
    case_pack: dict[str, Any],
    objects: list[dict[str, Any]],
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
    checks: list[dict[str, Any]] = []
    for assignment in selected:
        seat_check_id = (
            f"intent_contract__{constraint['constraint_id']}__"
            f"{assignment.seat_id}__{assignment.surface_id}"
        )
        common_evidence = {
            "intent_constraint": constraint,
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
                "scoring_tier": "core",
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
                "scoring_tier": "core",
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
            key=lambda item: (
                # A long wall's center can be farther than the subject even
                # when it is the wall the object physically touches.
                _wall_gap_or_infinity(subject, item),
                _center_distance_sq(center, bbox_center_xy(item)),
                str(item.get("id") or ""),
            ),
        )
        selected.append(str(wall.get("id") or ""))
    return selected


def _wall_gap_or_infinity(subject: dict[str, Any], wall: dict[str, Any]) -> float:
    gap = bbox_gap_xy(subject, wall)
    return float(gap) if gap is not None else math.inf


def _center_distance_sq(
    first: tuple[float, float], second: tuple[float, float] | None
) -> float:
    if second is None:
        return math.inf
    return (first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2


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
    rows: list[dict[str, Any]] = []
    for constraint in contract.get("constraints") or []:
        if not isinstance(constraint, dict):
            continue
        copied = {
            **constraint,
            "subjects": dict(constraint.get("subjects") or {}),
        }
        if isinstance(constraint.get("targets"), dict):
            copied["targets"] = dict(constraint["targets"])
        if isinstance(constraint.get("groups"), list):
            copied["groups"] = [
                dict(group) for group in constraint["groups"] if isinstance(group, dict)
            ]
        rows.append(copied)
    return {**contract, "constraints": rows}


def _stable_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    )
