"""Independent LLM compiler for hard SceneBenchmark intent contracts."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

from scenesmith.agent_utils.thinking import (
    chat_template_kwargs_from_effort,
    openrouter_extra_body,
    prepend_text_thinking_directive,
    thinking_directive_from_effort,
)
from scenesmith.scene_expert.context_bundle import build_llm_call_debug_record
from scenesmith.scenebenchmark_critic.intent_schema import (
    INTENT_COMPILER_SPEC_VERSION,
    INTENT_COMPILER_SEMANTIC_IR_VERSION,
    INTENT_CONTRACT_SCHEMA_VERSION,
    canonical_selector_category,
    intent_compiler_wire_json_schema,
    selector_categories_overlap,
    validate_intent_contract,
)
from scenesmith.scenebenchmark_critic.intent_contract import (
    HARD_SOURCES,
    _apply_task_spec_contract_metadata,
    _task_spec_inventory_constraints,
    build_intent_contract,
    ensure_coverage_requirements,
)
from scenesmith.scenebenchmark_critic.object_taxonomy import (
    execution_owner,
    is_known_object_category,
)
from scenesmith.scenebenchmark_critic.relation_registry import (
    RELATION_REGISTRY,
    ROOM_RELATIVE_WALL_CATEGORIES,
    relation_spec,
    relations_are_exclusive,
)
from scenesmith.utils.llm_json import json_response_format, parse_llm_json_object

logger = logging.getLogger(__name__)


class IntentCompilationError(RuntimeError):
    """Raised when both independent intent compilation attempts fail."""

    def __init__(self, message: str, *, trace: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.trace = trace or {}


class IncompleteIntentContractError(ValueError):
    """Raised when parseable compiler output omits required contract content."""


_INVENTORY_FIELDS = (
    "required_large_objects",
    "required_wall_objects",
    "required_ceiling_objects",
    "required_small_objects",
)
_LEGAL_ENVIRONMENT_ANCHORS = frozenset(
    {
        "room",
        "wall",
        "floor",
        "ceiling",
        "entrance",
        "entry",
        "door",
        "opening",
        "window",
        *ROOM_RELATIVE_WALL_CATEGORIES,
    }
)


def _entity_catalog(task_spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Build the only endpoint namespace admitted by the live compiler."""

    catalog: dict[str, dict[str, Any]] = {}
    for row in _task_spec_inventory_constraints(task_spec):
        subjects = row.get("subjects") or {}
        category = canonical_selector_category(subjects.get("category"))
        if not category:
            continue
        entity_ref = f"inventory:{category}"
        catalog[entity_ref] = {
            "entity_ref": entity_ref,
            "canonical_category": category,
            "required_count": int(subjects.get("count") or 1),
            "generation_stage": execution_owner(category),
            "source": str(row.get("inference_reason") or "SceneTaskSpec inventory"),
        }
    for category in sorted(_LEGAL_ENVIRONMENT_ANCHORS):
        entity_ref = f"anchor:{category}"
        catalog[entity_ref] = {
            "entity_ref": entity_ref,
            "canonical_category": category,
            "required_count": 1,
            "generation_stage": "floor_plan",
            "source": "legal_environment_anchor",
        }
    return catalog


def _render_entity_catalog(catalog: dict[str, dict[str, Any]]) -> str:
    return json.dumps(
        list(catalog.values()), ensure_ascii=False, indent=2, sort_keys=True
    )


def _rejected_entity_refs(
    payload: Any, catalog: dict[str, dict[str, Any]]
) -> list[dict[str, str]]:
    """Project rejected candidate refs into the persisted attempt trace."""

    if not isinstance(payload, dict):
        return []
    rows = payload.get("requirements")
    if not isinstance(rows, list):
        return []
    rejected: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        for field in ("subject_ref", "target_ref", "secondary_target_ref"):
            value = row.get(field)
            if value is None or not str(value).strip() or str(value) in catalog:
                continue
            rejected.append(
                {
                    "requirement_id": str(row.get("requirement_id") or ""),
                    "grounding": str(row.get("grounding") or ""),
                    "entity_ref": str(value),
                    "reason": "unknown_or_unbound_entity_ref",
                }
            )
    return rejected


def _semantic_source(
    grounding: str, grounding_catalog: dict[str, str]
) -> dict[str, str]:
    text = grounding_catalog[grounding]
    if grounding.startswith("prompt:"):
        return {
            "source": "explicit_prompt",
            "evidence_span": text,
            "inference_reason": "",
        }
    return {
        "source": "model_inferred",
        "evidence_span": "",
        "inference_reason": f"SceneTaskSpec {grounding}: {text}",
    }


def _catalog_selector(
    entity_ref: Any,
    catalog: dict[str, dict[str, Any]],
    *,
    count: Any = None,
    quantifier: Any = None,
    role: Any = None,
    cohort: Any = None,
) -> dict[str, Any]:
    ref = str(entity_ref or "")
    entry = catalog.get(ref)
    if entry is None:
        raise ValueError(f"unknown or unbound entity_ref {ref!r}")
    selector: dict[str, Any] = {"category": entry["canonical_category"]}
    if count is not None:
        selector["count"] = int(count)
    elif ref.startswith("inventory:"):
        selector["count"] = int(entry["required_count"])
    else:
        selector["count"] = 1
    if quantifier:
        selector["quantifier"] = str(quantifier)
    if role:
        selector["role"] = str(role)
    if cohort:
        selector["cohort"] = str(cohort)
    return selector


def _coverage_row(
    requirement: dict[str, Any],
    *,
    kind: str,
    disposition: str,
    normalized: str,
    source: dict[str, str],
    earliest_stage: str = "floor_plan",
    relation: str = "",
) -> dict[str, Any]:
    return {
        "requirement_id": str(requirement["requirement_id"]),
        "kind": kind,
        "disposition": disposition,
        "normalized": normalized,
        "earliest_stage": earliest_stage,
        "final_stage": "final",
        "source": source["source"],
        "evidence_span": source["evidence_span"] or source["inference_reason"],
        "relation": relation,
    }


def _admit_semantic_ir(
    payload: dict[str, Any],
    *,
    prompt: str,
    prompt_hash: str,
    task_spec: dict[str, Any],
    grounding_catalog: dict[str, str],
    catalog: dict[str, dict[str, Any]],
    retry_count: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, str]]]:
    """Validate LLM semantics without interpreting prompt wording in code."""

    if payload.get("schema_version") != INTENT_COMPILER_SEMANTIC_IR_VERSION:
        raise ValueError("semantic IR schema_version is missing or unsupported")
    unexpected_top_level = set(payload) - {"schema_version", "requirements"}
    if unexpected_top_level:
        raise ValueError(
            "semantic IR has unexpected fields: "
            + ", ".join(sorted(unexpected_top_level))
        )
    requirements = payload.get("requirements")
    if not isinstance(requirements, list):
        raise ValueError("semantic IR requirements must be a list")
    seen_ids: set[str] = set()
    covered_groundings: set[str] = set()
    constraints = _task_spec_inventory_constraints(task_spec)
    coverage_requirements: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []

    for raw in requirements:
        if not isinstance(raw, dict):
            raise ValueError("semantic IR requirement must be an object")
        unexpected_fields = set(raw) - {
            "requirement_id",
            "kind",
            "grounding",
            "relation",
            "subject_ref",
            "target_ref",
            "secondary_target_ref",
            "subject_count",
            "target_count",
            "subject_quantifier",
            "target_quantifier",
            "subject_role",
            "target_role",
            "subject_cohort",
            "target_cohort",
            "edge_frame",
            "groups",
            "orientation",
            "forbidden_category",
            "reason",
            "surface_mentions",
        }
        if unexpected_fields:
            raise ValueError(
                "semantic IR requirement has unexpected fields: "
                + ", ".join(sorted(unexpected_fields))
            )
        requirement_id = str(raw.get("requirement_id") or "")
        grounding = str(raw.get("grounding") or "")
        kind = str(raw.get("kind") or "")
        if not requirement_id or requirement_id in seen_ids:
            raise ValueError("semantic IR requirement_id must be unique and non-empty")
        if grounding not in grounding_catalog:
            raise ValueError(f"unknown grounding id {grounding!r}")
        if kind not in {
            "inventory",
            "forbidden_inventory",
            "relation",
            "unsupported",
            "unresolved",
            "soft_scope",
        }:
            raise ValueError(f"unsupported semantic IR requirement kind {kind!r}")
        seen_ids.add(requirement_id)
        covered_groundings.add(grounding)
        source = _semantic_source(grounding, grounding_catalog)
        refs = [
            str(raw.get(field) or "")
            for field in ("subject_ref", "target_ref", "secondary_target_ref")
            if raw.get(field) is not None
        ]
        ledger_row = {
            "requirement_id": requirement_id,
            "grounding": grounding,
            "kind": kind,
            "entity_refs": refs,
            "surface_mentions": list(raw.get("surface_mentions") or []),
            "reason": str(raw.get("reason") or ""),
        }
        if kind == "relation":
            relation = str(raw.get("relation") or "")
            spec = relation_spec(relation)
            if relation == "required_count":
                raise ValueError("semantic IR may not create required_count")
            subject_ref = raw.get("subject_ref")
            if not subject_ref:
                raise ValueError(f"relation {relation!r} omitted subject_ref")
            subjects = _catalog_selector(
                subject_ref,
                catalog,
                count=raw.get("subject_count"),
                quantifier=raw.get("subject_quantifier"),
                role=raw.get("subject_role"),
                cohort=raw.get("subject_cohort"),
            )
            target_ref = raw.get("target_ref")
            secondary_ref = raw.get("secondary_target_ref")
            if spec.target_arity == 0:
                if target_ref is not None or secondary_ref is not None:
                    raise ValueError(
                        f"relation {relation!r} must not have endpoint refs"
                    )
                targets = None
            else:
                if not target_ref:
                    raise ValueError(f"relation {relation!r} omitted target_ref")
                targets = _catalog_selector(
                    target_ref,
                    catalog,
                    count=raw.get("target_count"),
                    quantifier=raw.get("target_quantifier"),
                    role=raw.get("target_role"),
                    cohort=raw.get("target_cohort"),
                )
                if spec.target_arity == 2:
                    if not secondary_ref:
                        raise ValueError(
                            f"relation {relation!r} omitted secondary_target_ref"
                        )
                    secondary = _catalog_selector(secondary_ref, catalog)
                    targets["secondary_category"] = secondary["category"]
                    targets["secondary_count"] = secondary["count"]
                elif secondary_ref is not None:
                    raise ValueError(
                        f"relation {relation!r} has unexpected secondary_target_ref"
                    )
            row = {
                "relation": relation,
                "subjects": subjects,
                "targets": targets,
                "edge_frame": raw.get("edge_frame"),
                "groups": raw.get("groups") or [],
                "orientation": raw.get("orientation"),
                **source,
            }
            constraints.append(row)
            ledger_row["disposition"] = "compiled"
        elif kind == "inventory":
            subject_ref = raw.get("subject_ref")
            if not subject_ref or not str(subject_ref).startswith("inventory:"):
                raise ValueError("inventory requires an inventory subject_ref")
            _catalog_selector(subject_ref, catalog)
            if (
                raw.get("target_ref") is not None
                or raw.get("secondary_target_ref") is not None
            ):
                raise ValueError("inventory must not have target entity refs")
            ledger_row["disposition"] = "compiled"
        elif kind == "forbidden_inventory":
            category = canonical_selector_category(raw.get("forbidden_category"))
            if not category or not is_known_object_category(category):
                raise ValueError(
                    "forbidden_inventory requires a known forbidden_category"
                )
            coverage_requirements.append(
                _coverage_row(
                    raw,
                    kind="forbidden_inventory",
                    disposition="compiled",
                    normalized=category,
                    source=source,
                    earliest_stage=execution_owner(category),
                )
            )
            ledger_row["disposition"] = "compiled"
        elif kind == "unsupported":
            normalized = "_".join(
                str(raw.get("reason") or "unsupported").lower().split()
            )
            coverage_requirements.append(
                _coverage_row(
                    raw,
                    kind="unsupported_relation",
                    disposition="unsupported",
                    normalized=normalized,
                    source=source,
                )
            )
            ledger_row["disposition"] = "unsupported"
        elif kind == "unresolved":
            normalized = "_".join(
                str(raw.get("reason") or "unresolved").lower().split()
            )
            coverage_requirements.append(
                _coverage_row(
                    raw,
                    kind="unresolved",
                    disposition="unresolved",
                    normalized=normalized,
                    source=source,
                )
            )
            ledger_row["disposition"] = "unresolved"
        elif kind == "soft_scope":
            normalized = "_".join(
                str(raw.get("reason") or "soft_scope").lower().split()
            )
            coverage_requirements.append(
                _coverage_row(
                    raw,
                    kind="soft_scope",
                    disposition="soft_scope",
                    normalized=normalized,
                    source=source,
                )
            )
            ledger_row["disposition"] = "soft_scope"
        ledger.append(ledger_row)

    missing = sorted(set(grounding_catalog) - covered_groundings)
    if missing:
        raise ValueError("semantic IR coverage gap for " + ", ".join(missing))
    contract = {
        "schema_version": INTENT_CONTRACT_SCHEMA_VERSION,
        "prompt": prompt,
        "prompt_sha256": prompt_hash,
        "intent_compiler_spec_version": INTENT_COMPILER_SPEC_VERSION,
        "room_type": str(task_spec.get("room_type") or ""),
        "constraints": constraints,
        "coverage_requirements": coverage_requirements,
        "coverage_ledger": ledger,
        "retry_count": retry_count,
        "warnings": [],
    }
    return contract, ledger, rejected


def _merge_warnings(*groups: Any) -> list[str]:
    """Combine warning lists without losing order or repeating text."""

    merged: list[str] = []
    seen: set[str] = set()
    for group in groups:
        if not isinstance(group, (list, tuple)):
            continue
        for value in group:
            warning = " ".join(str(value or "").split())
            if warning and warning not in seen:
                merged.append(warning)
                seen.add(warning)
    return merged


def _validate_contract_completeness(
    contract: dict[str, Any], task_spec: dict[str, Any]
) -> None:
    """Require authoritative inventory counts and resolvable relation endpoints."""

    expected_counts: dict[str, int] = {}
    for field in _INVENTORY_FIELDS:
        for value in task_spec.get(field) or []:
            category = canonical_selector_category(value)
            if category and category not in _LEGAL_ENVIRONMENT_ANCHORS:
                expected_counts[category] = expected_counts.get(category, 0) + 1

    count_rows: dict[str, list[dict[str, Any]]] = {}
    constraints = contract.get("constraints")
    if not isinstance(constraints, list):
        raise IncompleteIntentContractError("contract constraints must be a list")
    # Legacy callers may compile a raw prompt without a TaskSpec. In that mode
    # there is no authoritative inventory against which endpoints can resolve.
    if not expected_counts:
        return
    for row in constraints:
        if not isinstance(row, dict) or row.get("relation") != "required_count":
            continue
        category = canonical_selector_category(
            (row.get("subjects") or {}).get("category")
        )
        count_rows.setdefault(category, []).append(row)

    for category, expected_count in expected_counts.items():
        rows = count_rows.get(category, [])
        if not rows:
            raise IncompleteIntentContractError(
                f"missing authoritative required_count for {category!r}"
            )
        # TaskSpec inventory is a minimum coverage obligation.  A validated
        # explicit prompt row may intentionally win a conflicting inventory
        # count (for example, ``exactly two bowls`` versus an inferred list of
        # three).  Keep the exact prompt cardinality instead of rejecting the
        # whole LLM attempt and consuming the retry response.
        accepted = False
        for row in rows:
            subjects = row.get("subjects") or {}
            actual_count = int(subjects.get("count") or 0)
            source = str(row.get("source") or "")
            if source == "explicit_prompt":
                accepted = actual_count > 0
                if accepted:
                    break
            elif actual_count == expected_count and source in HARD_SOURCES:
                accepted = True
                break
        if not accepted:
            raise IncompleteIntentContractError(
                f"required_count for {category!r} has no accepted count for expected {expected_count}"
            )

    # Compound TaskSpec labels are retained as stable inventory identities
    # (for example ``bowl_of_fruit``), while prompt relations may naturally
    # refer to one of their concrete noun endpoints (``bowl``).  Admit only
    # noun tokens that have an established taxonomy entry; this keeps the
    # completeness gate strict for arbitrary hallucinated endpoints such as
    # ``phantom_object``.
    compound_nouns: set[str] = set()
    for inventory_category in expected_counts:
        tokens = inventory_category.split("_")
        if len(tokens) < 2:
            continue
        for token in tokens:
            noun = canonical_selector_category(token)
            if noun and is_known_object_category(noun):
                compound_nouns.add(noun)

    resolvable = set(count_rows) | compound_nouns | set(_LEGAL_ENVIRONMENT_ANCHORS)
    for row in constraints:
        if not isinstance(row, dict) or row.get("relation") == "required_count":
            continue
        selectors = [row.get("subjects"), row.get("targets")]
        for selector in selectors:
            if not isinstance(selector, dict):
                continue
            for field in ("category", "secondary_category"):
                category = canonical_selector_category(selector.get(field))
                if category and not any(
                    selector_categories_overlap(category, candidate)
                    for candidate in resolvable
                ):
                    raise IncompleteIntentContractError(
                        f"relation {row.get('relation')!r} endpoint {category!r} "
                        "has no required_count or legal environment anchor"
                    )


def _append_llm_debug(record: dict[str, Any]) -> None:
    path = os.environ.get("SCENEEXPERT_LLM_DEBUG_PATH", "")
    if not path:
        return
    try:
        debug_path = Path(path)
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        with debug_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception as exc:  # pragma: no cover - debug output must not fail a run
        logger.warning("IntentCompiler failed to write LLM debug record: %s", exc)


def _normalize_side_distribution(payload: dict[str, Any]) -> dict[str, Any]:
    """Use the reusable flanking relation for an unqualified two-object pair."""
    constraints = payload.get("constraints")
    if not isinstance(constraints, list):
        return payload

    for index, constraint in enumerate(constraints):
        if not isinstance(constraint, dict):
            continue
        if str(constraint.get("relation") or "") != "edge_distribution":
            continue
        subjects = constraint.get("subjects") or {}
        targets = constraint.get("targets") or {}
        if subjects.get("count") != 2 or targets.get("count") != 1:
            continue
        evidence = " ".join(
            str(value or "")
            for value in (constraint.get("evidence_span"), payload.get("prompt"))
        )
        if not re.search(
            r"\b(?:on|at)\s+(?:each|either|both)\s+sides?\s+of\b",
            evidence,
            re.IGNORECASE,
        ):
            continue
        if re.search(
            r"\b(?:long|short)\s+(?:sides?|edges?)\b",
            evidence,
            re.IGNORECASE,
        ):
            continue
        normalized = dict(constraint)
        normalized["relation"] = "flanking"
        for field in ("edge_frame", "groups", "orientation"):
            normalized.pop(field, None)
        constraints[index] = normalized
    return payload


def _normalize_seating_support_relations(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Interpret chair-at-desk cardinality as pairing, not top support."""
    constraints = payload.get("constraints")
    if not isinstance(constraints, list):
        return payload
    seating = {
        "chair",
        "office_chair",
        "guest_chair",
        "student_chair",
        "teacher_chair",
        "dining_chair",
        "armchair",
        "stool",
        "bench",
    }
    work_surfaces = {
        "desk",
        "student_desk",
        "teacher_desk",
        "reception_desk",
        "table",
        "dining_table",
        "conference_table",
    }
    normalized = dict(payload)
    normalized_constraints: list[Any] = []
    changed = False
    for constraint in constraints:
        if not isinstance(constraint, dict):
            normalized_constraints.append(constraint)
            continue
        subject_category = canonical_selector_category(
            (constraint.get("subjects") or {}).get("category")
        )
        target_category = canonical_selector_category(
            (constraint.get("targets") or {}).get("category")
        )
        if (
            str(constraint.get("relation") or "") == "one_per_support"
            and subject_category in seating
            and target_category in work_surfaces
        ):
            constraint = dict(constraint, relation="paired_with")
            changed = True
        normalized_constraints.append(constraint)
    if not changed:
        return payload
    normalized["constraints"] = normalized_constraints
    return normalized


def _remove_redundant_edge_faces(payload: dict[str, Any]) -> dict[str, Any]:
    """Drop faces rows already encoded by an inward edge distribution."""

    constraints = payload.get("constraints")
    if not isinstance(constraints, list):
        return payload
    inward_edges = [
        row
        for row in constraints
        if isinstance(row, dict)
        and row.get("relation") == "edge_distribution"
        and row.get("orientation") == "toward_target"
    ]
    if not inward_edges:
        return payload
    filtered = [
        row
        for row in constraints
        if not (
            isinstance(row, dict)
            and row.get("relation") == "faces"
            and any(
                _selectors_semantically_overlap(
                    row.get("subjects"), edge.get("subjects")
                )
                and _selectors_semantically_overlap(
                    row.get("targets"), edge.get("targets")
                )
                for edge in inward_edges
            )
        )
    ]
    if len(filtered) == len(constraints):
        return payload
    normalized = dict(payload)
    normalized["constraints"] = filtered
    return normalized


def _normalize_centered_above_relations(
    payload: dict[str, Any], prompt: str
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Correct centered relations only when deterministic prompt grounding agrees.

    ``centered_on_wall`` is only valid when the wall itself is the alignment
    anchor. Language such as "a mirror on the wall, centered above the dressing
    table" is instead a local object-to-object relation. Use the deterministic
    parser only when it found one unambiguous ``centered_above`` counterpart.

    LLMs can also confuse the word "centered" in a horizontal placement phrase
    such as "centered in front of the desk" with vertical ``centered_above``.
    Correct that mistake only when the same evidence explicitly names a front or
    rear axis and the deterministic parser found one matching relation for the
    same endpoints. This preserves legitimate vertical alignment constraints.
    """
    constraints = payload.get("constraints")
    if not isinstance(constraints, list):
        return payload, []

    deterministic = build_intent_contract(prompt).get("constraints") or []
    normalized_constraints = list(constraints)
    normalized: list[dict[str, str]] = []
    for index, constraint in enumerate(normalized_constraints):
        if not isinstance(constraint, dict):
            continue
        relation = str(constraint.get("relation") or "")
        evidence = str(constraint.get("evidence_span") or "").lower()
        replacement_relation: str | None = None
        require_target_overlap = False
        if relation == "centered_on_wall" and "above" in evidence:
            replacement_relation = "centered_above"
        elif relation == "centered_above":
            if re.search(r"\bin\s+front\s+of\b", evidence):
                replacement_relation = "in_front_of"
            elif re.search(r"\bbehind\b", evidence):
                replacement_relation = "behind"
            require_target_overlap = replacement_relation is not None
        if replacement_relation is None:
            continue

        matches = []
        for candidate in deterministic:
            if (
                not isinstance(candidate, dict)
                or str(candidate.get("relation") or "") != replacement_relation
                or not _selectors_semantically_overlap(
                    constraint.get("subjects"), candidate.get("subjects")
                )
            ):
                continue
            if require_target_overlap and not _selectors_semantically_overlap(
                constraint.get("targets"), candidate.get("targets")
            ):
                continue
            matches.append(candidate)
        if len(matches) != 1:
            continue
        candidate = matches[0]
        replacement = dict(constraint)
        replacement["relation"] = replacement_relation
        replacement["targets"] = dict(candidate.get("targets") or {})
        if not replacement.get("evidence_span"):
            replacement["evidence_span"] = str(candidate.get("evidence_span") or "")
        normalized_constraints[index] = replacement
        normalized.append(
            {
                "subject_category": canonical_selector_category(
                    (replacement.get("subjects") or {}).get("category")
                ),
                "target_category": canonical_selector_category(
                    (replacement.get("targets") or {}).get("category")
                ),
            }
        )

    if not normalized:
        return payload, normalized
    result = dict(payload)
    result["constraints"] = normalized_constraints
    warnings = list(result.get("warnings") or [])
    warnings.append(
        "Wall-center relation corrected to prompt-grounded centered_above relation"
    )
    result["warnings"] = warnings
    return result, normalized


_DETERMINISTIC_ENRICHMENT_SOURCES = frozenset(
    {"explicit_prompt", "task_compiler_inventory", "room_ontology"}
)
_HIGH_CONFIDENCE_EXPLICIT_RELATIONS = frozenset(
    {
        "required_count",
        "flanking",
        "paired_with",
        "one_per_support",
        "edge_distribution",
        "corner_distribution",
        "on_top_of",
        "against_wall",
        "centered_above",
        "on_wall",
        "mounted_to_ceiling",
        "clear_access",
    }
)
_FLANKING_GROUNDING_PATTERNS = (
    re.compile(r"\bflank(?:s|ed|ing)?\b", re.IGNORECASE),
    re.compile(
        r"\b(?:on|at)\s+(?:each|either|both|opposite)\s+sides?\s+of\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:one|a|an)\b[^.;]{0,80}\b(?:left|one)\s+side\b"
        r"[^.;]{0,120}\b(?:one|a|an)\b[^.;]{0,80}\b(?:right|other)\s+side\b",
        re.IGNORECASE,
    ),
)


def _selector_phrase_pattern(selector: dict[str, Any] | None) -> str:
    category = str((selector or {}).get("category") or "").strip().replace("_", " ")
    if not category:
        return ""
    words = [re.escape(word) for word in category.split()]
    return rf"(?<![a-z0-9]){'[ -]+'.join(words)}(?:s|es)?(?![a-z0-9])"


def _on_top_relation_uses_containment_wording(
    constraint: dict[str, Any], evidence: str
) -> bool:
    """Whether grounded endpoints are described as contained, not supported."""
    subject = _selector_phrase_pattern(constraint.get("subjects"))
    target = _selector_phrase_pattern(constraint.get("targets"))
    if not subject or not target:
        return False
    bounded = r"[^.;!?]{0,100}"
    patterns = (
        rf"{target}{bounded}\b(?:holding|holds|containing|contains)\b{bounded}{subject}",
        rf"{target}{bounded}\bwith\b{bounded}{subject}{bounded}\binside\b",
        rf"{subject}{bounded}\b(?:inside|within|contained\s+(?:in|inside))\b{bounded}{target}",
    )
    return any(re.search(pattern, evidence, re.IGNORECASE) for pattern in patterns)


def _validate_prompt_grounded_relations(
    payload: dict[str, Any], prompt: str
) -> dict[str, Any]:
    """Reject relation-specific conventions that the prompt never states.

    An inventory such as ``bed, two nightstands`` establishes counts, not a
    bilateral layout.  TaskCompiler or HSSD may recommend that layout as soft
    guidance, but the independent hard contract may emit ``flanking`` only
    when either the original prompt or a TaskCompiler constraint contains an
    explicit bilateral cue.
    """
    for constraint in payload.get("constraints") or []:
        if not isinstance(constraint, dict):
            continue
        relation = str(constraint.get("relation") or "")
        source = str(constraint.get("source") or "")
        evidence = (
            str(constraint.get("inference_reason") or "")
            if source == "model_inferred"
            else str(constraint.get("evidence_span") or "")
        )
        if relation == "centered_on_wall" and not re.search(
            r"\bwall\b", evidence, re.IGNORECASE
        ):
            raise ValueError(
                "centered_on_wall hard intent requires explicit wall wording in "
                "its grounded prompt or TaskCompiler constraint"
            )
        if (
            relation == "on_top_of"
            and source == "explicit_prompt"
            and _on_top_relation_uses_containment_wording(constraint, evidence)
        ):
            raise ValueError(
                "on_top_of hard intent cannot replace containment wording for "
                "the same grounded endpoints"
            )
        if relation != "flanking":
            continue
        if source == "model_inferred":
            flanking_evidence = evidence
        else:
            flanking_evidence = " ".join(
                str(value or "") for value in (evidence, prompt)
            )
        if any(
            pattern.search(flanking_evidence)
            for pattern in _FLANKING_GROUNDING_PATTERNS
        ):
            continue
        subject = str((constraint.get("subjects") or {}).get("category") or "object")
        raise ValueError(
            "flanking hard intent requires explicit bilateral wording in the "
            f"original prompt or TaskCompiler constraints for subject {subject!r}"
        )
    return payload


def _same_subject(first: dict[str, Any], second: dict[str, Any]) -> bool:
    return _selectors_semantically_overlap(
        first.get("subjects"), second.get("subjects")
    )


def _task_constraint_rejection_reason(
    candidate: dict[str, Any], accepted: list[dict[str, Any]]
) -> str:
    """Return why a model-inferred row loses to an explicit prompt row."""
    relation = str(candidate.get("relation") or "")
    for existing in accepted:
        if str(existing.get("source") or "") != "explicit_prompt":
            continue
        existing_relation = str(existing.get("relation") or "")
        if _constraints_semantically_overlap(candidate, existing):
            return "duplicate_of_explicit_prompt"
        if _same_subject(candidate, existing) and relations_are_exclusive(
            relation, existing_relation
        ):
            return "conflicts_with_explicit_prompt"
        if (
            _same_subject(candidate, existing)
            and _selectors_semantically_overlap(
                candidate.get("targets"), existing.get("targets")
            )
            and {relation, existing_relation} == {"in_front_of", "behind"}
        ):
            return "conflicts_with_explicit_prompt"
        if relation == existing_relation == "required_count" and _same_subject(
            candidate, existing
        ):
            candidate_count = (candidate.get("subjects") or {}).get("count")
            existing_count = (existing.get("subjects") or {}).get("count")
            if candidate_count != existing_count:
                return "conflicts_with_explicit_prompt"
    return ""


def _prefer_explicit_prompt_constraints(
    payload: dict[str, Any], *, allowed_inference_reasons: tuple[str, ...]
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Drop ungrounded or conflicting TaskCompiler-derived hard relations."""
    constraints = payload.get("constraints")
    if not isinstance(constraints, list):
        return payload, []

    accepted = [
        row
        for row in constraints
        if isinstance(row, dict) and str(row.get("source") or "") != "model_inferred"
    ]
    rejected: list[dict[str, str]] = []
    for row in constraints:
        if (
            not isinstance(row, dict)
            or str(row.get("source") or "") != "model_inferred"
        ):
            continue
        reason = ""
        inference_reason = " ".join(str(row.get("inference_reason") or "").split())
        if not allowed_inference_reasons:
            reason = "missing_task_compiler_context"
        elif not inference_reason:
            reason = "missing_task_compiler_provenance"
        elif inference_reason.casefold() not in {
            value.casefold() for value in allowed_inference_reasons
        }:
            reason = "not_grounded_in_task_compiler_constraint"
        else:
            reason = _task_constraint_rejection_reason(row, accepted)
        if reason:
            rejected.append(
                {
                    "relation": str(row.get("relation") or ""),
                    "subject_category": canonical_selector_category(
                        (row.get("subjects") or {}).get("category")
                    ),
                    "reason": reason,
                    "inference_reason": inference_reason,
                }
            )
            continue
        accepted.append(row)

    if not rejected:
        return payload, rejected
    filtered = dict(payload)
    filtered["constraints"] = accepted
    warnings = list(filtered.get("warnings") or [])
    warnings.append(
        "TaskCompiler-derived constraints filtered by explicit-prompt precedence"
    )
    filtered["warnings"] = warnings
    return filtered, rejected


def _normalize_provenance_fields(payload: dict[str, Any]) -> dict[str, Any]:
    """Prevent explanatory model text from contaminating prompt provenance."""

    constraints = payload.get("constraints")
    if not isinstance(constraints, list):
        return payload
    normalized_rows: list[Any] = []
    for row in constraints:
        if not isinstance(row, dict):
            normalized_rows.append(row)
            continue
        normalized = dict(row)
        if str(normalized.get("source") or "") == "explicit_prompt":
            normalized["inference_reason"] = ""
        normalized_rows.append(normalized)
    result = dict(payload)
    result["constraints"] = normalized_rows
    return result


def _grounding_catalog(
    prompt: str, task_spec: dict[str, Any]
) -> tuple[dict[str, str], str]:
    """Build stable short IDs for all text that may authorize a hard relation."""

    catalog: dict[str, str] = {}
    prompt_clauses = [
        " ".join(value.split())
        for value in re.split(r"(?<=[.!?])\s+", prompt)
        if value.strip()
    ]
    for index, value in enumerate(prompt_clauses):
        catalog[f"prompt:{index}"] = value
    for field, prefix in (
        ("interaction_constraints", "interaction"),
        ("aesthetic_constraints", "aesthetic"),
    ):
        for index, value in enumerate(task_spec.get(field) or []):
            text = " ".join(str(value or "").split())
            if text:
                catalog[f"{prefix}:{index}"] = text
    rendered = "\n".join(
        f"- {key}: {json.dumps(value)}" for key, value in catalog.items()
    )
    return catalog, rendered


def _attach_grounding_provenance(
    payload: dict[str, Any], catalog: dict[str, str]
) -> dict[str, Any]:
    """Expand model-selected grounding IDs into validated contract provenance."""

    constraints = payload.get("constraints")
    if not isinstance(constraints, list):
        return payload
    normalized_rows: list[Any] = []
    for row in constraints:
        if not isinstance(row, dict):
            normalized_rows.append(row)
            continue
        normalized = dict(row)
        grounding = str(normalized.pop("grounding", "") or "")
        if grounding:
            text = catalog.get(grounding)
            if text is None:
                raise ValueError(f"unknown grounding id {grounding!r}")
            prefix = grounding.split(":", 1)[0]
            if prefix == "prompt":
                normalized.update(
                    source="explicit_prompt",
                    evidence_span=text,
                    inference_reason="",
                )
            else:
                field = f"{prefix}_constraints"
                normalized.update(
                    source="model_inferred",
                    evidence_span="",
                    inference_reason=f"TaskCompiler {field}: {text}",
                )
        elif not any(
            normalized.get(field)
            for field in ("source", "evidence_span", "inference_reason")
        ):
            raise ValueError("relation omitted required grounding id")
        normalized_rows.append(normalized)
    result = dict(payload)
    result["constraints"] = normalized_rows
    return result


def _selectors_semantically_overlap(
    first: dict[str, Any] | None, second: dict[str, Any] | None
) -> bool:
    """Compare contract endpoints without discarding role-qualified intent."""
    if not isinstance(first, dict) or not isinstance(second, dict):
        return first is None and second is None
    first_category = canonical_selector_category(first.get("category"))
    second_category = canonical_selector_category(second.get("category"))
    if not selector_categories_overlap(first_category, second_category):
        return False
    first_role = str(first.get("role") or "").strip().lower()
    second_role = str(second.get("role") or "").strip().lower()
    return not first_role or not second_role or first_role == second_role


def _constraints_semantically_overlap(
    first: dict[str, Any], second: dict[str, Any]
) -> bool:
    """Identify duplicate hard relations despite generic/specific selectors."""
    if str(first.get("relation") or "") != str(second.get("relation") or ""):
        return False
    if not _selectors_semantically_overlap(
        first.get("subjects"), second.get("subjects")
    ):
        return False
    return _selectors_semantically_overlap(first.get("targets"), second.get("targets"))


def _restore_missing_target_selectors(
    payload: dict[str, Any], prompt: str
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Fill an omitted target only when the deterministic parser agrees.

    llama.cpp's JSON grammar guarantees object syntax but does not enforce the
    relation-dependent ``if/then`` clauses emitted by Pydantic.  The model can
    therefore correctly identify a relation such as ``flanking`` while omit
    its target selector.  Reuse a target only when the deterministic parser
    extracted the same relation for the same subject from the original prompt.
    Recovery is relation-agnostic but remains conservative: an endpoint is
    restored only when the deterministic parser produced exactly one relation
    with the same name and overlapping subject selector.
    """

    constraints = payload.get("constraints")
    if not isinstance(constraints, list):
        return payload, []

    deterministic_constraints = build_intent_contract(prompt).get("constraints") or []
    restored: list[dict[str, str]] = []
    normalized_constraints = list(constraints)
    for index, constraint in enumerate(normalized_constraints):
        if not isinstance(constraint, dict):
            continue
        try:
            spec = relation_spec(str(constraint.get("relation") or ""))
        except ValueError:
            continue
        target = constraint.get("targets")
        has_target = isinstance(target, dict) and bool(target.get("category"))
        if spec.target_arity == 0 or has_target:
            continue

        matches = [
            candidate
            for candidate in deterministic_constraints
            if isinstance(candidate, dict)
            and str(candidate.get("relation") or "") == spec.name
            and _selectors_semantically_overlap(
                constraint.get("subjects"), candidate.get("subjects")
            )
            and isinstance(candidate.get("targets"), dict)
            and candidate["targets"].get("category")
        ]
        if len(matches) != 1:
            continue
        repaired = dict(constraint)
        repaired["targets"] = dict(matches[0]["targets"])
        normalized_constraints[index] = repaired
        restored.append(
            {
                "relation": spec.name,
                "subject_category": str(
                    (repaired.get("subjects") or {}).get("category") or ""
                ),
                "target_category": str(repaired["targets"].get("category") or ""),
            }
        )

    if not restored:
        return payload, restored
    normalized = dict(payload)
    normalized["constraints"] = normalized_constraints
    warnings = list(normalized.get("warnings") or [])
    warnings.append(
        "Missing relation targets restored from the deterministic prompt parser"
    )
    normalized["warnings"] = warnings
    return normalized, restored


def _restore_missing_relation_fields(
    payload: dict[str, Any], prompt: str, task_spec: dict[str, Any] | None = None
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Restore omitted schema fields only from one deterministic relation match.

    Some llama.cpp builds enforce the outer JSON shape but do not consistently
    apply required fields below a ``$ref`` or relation-dependent ``if/then``.
    Treat that as a transport limitation, not permission to guess: metadata
    and edge-distribution fields are restored only when the prompt parser found
    exactly one relation with the same type and overlapping subject selector.
    """

    constraints = payload.get("constraints")
    if not isinstance(constraints, list):
        return payload, []

    deterministic_constraints = (
        build_intent_contract(
            prompt,
            room_type=str((task_spec or {}).get("room_type") or ""),
            task_spec=task_spec,
        ).get("constraints")
        or []
    )
    restored: list[dict[str, Any]] = []
    normalized_constraints = list(constraints)
    for index, constraint in enumerate(normalized_constraints):
        if not isinstance(constraint, dict):
            continue
        relation = str(constraint.get("relation") or "")
        matches = [
            candidate
            for candidate in deterministic_constraints
            if isinstance(candidate, dict)
            and str(candidate.get("relation") or "") == relation
            and _selectors_semantically_overlap(
                constraint.get("subjects"), candidate.get("subjects")
            )
        ]
        if len(matches) != 1:
            continue

        repaired = dict(constraint)
        restored_fields: list[str] = []
        candidate = matches[0]
        for field in ("source", "evidence_span"):
            if repaired.get(field) or not candidate.get(field):
                continue
            repaired[field] = candidate[field]
            restored_fields.append(field)
        if relation == "edge_distribution":
            repaired_subjects = dict(repaired.get("subjects") or {})
            candidate_subjects = candidate.get("subjects") or {}
            if candidate_subjects.get("count") is not None and repaired_subjects.get(
                "count"
            ) != candidate_subjects.get("count"):
                repaired_subjects["count"] = candidate_subjects["count"]
                repaired["subjects"] = repaired_subjects
                restored_fields.append("subjects.count")
            for field in ("edge_frame", "groups", "orientation"):
                if not candidate.get(field) or repaired.get(field) == candidate.get(
                    field
                ):
                    continue
                repaired[field] = candidate[field]
                restored_fields.append(field)
        if not restored_fields:
            continue
        normalized_constraints[index] = repaired
        restored.append(
            {
                "relation": relation,
                "subject_category": canonical_selector_category(
                    (repaired.get("subjects") or {}).get("category")
                ),
                "fields": restored_fields,
            }
        )

    if not restored:
        return payload, restored
    normalized = dict(payload)
    normalized["constraints"] = normalized_constraints
    warnings = list(normalized.get("warnings") or [])
    warnings.append(
        "Missing relation fields restored from the deterministic prompt parser"
    )
    normalized["warnings"] = warnings
    return normalized, restored


def _is_high_confidence_deterministic_constraint(candidate: dict[str, Any]) -> bool:
    """Limit enrichment to deterministic facts whose parser has no direction ambiguity."""
    source = str(candidate.get("source") or "")
    if source in {"task_compiler_inventory", "room_ontology"}:
        return True
    return (
        source == "explicit_prompt"
        and str(candidate.get("relation") or "") in _HIGH_CONFIDENCE_EXPLICIT_RELATIONS
    )


def _enrich_with_deterministic_constraints(
    llm_contract: dict[str, Any],
    prompt: str,
    allowed_inference_reasons: tuple[str, ...],
    task_spec: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, str]]]:
    """Restore explicit and ontology constraints omitted by a valid LLM response.

    The independent compiler has higher recall for nuanced prompt relations, but
    the deterministic parser owns a small set of high-confidence facts.  A valid
    LLM response must therefore supplement rather than replace those facts.
    """
    deterministic = build_intent_contract(
        prompt,
        room_type=str((task_spec or {}).get("room_type") or ""),
        task_spec=task_spec,
    )
    constraints = list(llm_contract.get("constraints") or [])
    added: list[dict[str, str]] = []
    for candidate in deterministic.get("constraints") or []:
        if not isinstance(candidate, dict):
            continue
        source = str(candidate.get("source") or "")
        if (
            source not in _DETERMINISTIC_ENRICHMENT_SOURCES
            or not _is_high_confidence_deterministic_constraint(candidate)
        ):
            continue
        if source == "task_compiler_inventory":
            category = canonical_selector_category(
                (candidate.get("subjects") or {}).get("category")
            )
            constraints = [
                existing
                for existing in constraints
                if not (
                    isinstance(existing, dict)
                    and str(existing.get("relation") or "") == "required_count"
                    and canonical_selector_category(
                        (existing.get("subjects") or {}).get("category")
                    )
                    == category
                )
            ]
        elif any(
            isinstance(existing, dict)
            and str(existing.get("source") or "") != "model_inferred"
            and _constraints_semantically_overlap(existing, candidate)
            for existing in constraints
        ):
            continue
        constraints.append(candidate)
        added.append(
            {
                "constraint_id": str(candidate.get("constraint_id") or ""),
                "relation": str(candidate.get("relation") or ""),
                "source": source,
            }
        )

    if not added:
        return llm_contract, added, []
    enriched = dict(llm_contract)
    enriched["constraints"] = constraints
    warnings = list(enriched.get("warnings") or [])
    warnings.extend(
        warning
        for warning in deterministic.get("warnings") or []
        if warning not in warnings
    )
    warnings.append(
        "LLM contract enriched with deterministic explicit/room-ontology constraints"
    )
    enriched["warnings"] = warnings
    enriched, rejected = _prefer_explicit_prompt_constraints(
        enriched,
        allowed_inference_reasons=allowed_inference_reasons,
    )
    return validate_intent_contract(enriched), added, rejected


def _system_prompt() -> str:
    relation_lines = "\n".join(
        f"- {name} (target arity {spec.target_arity}): {spec.prompt_description}"
        for name, spec in sorted(RELATION_REGISTRY.items())
    )
    unary_relations = ", ".join(
        sorted(
            name for name, spec in RELATION_REGISTRY.items() if spec.target_arity == 1
        )
    )
    binary_relations = ", ".join(
        sorted(
            name for name, spec in RELATION_REGISTRY.items() if spec.target_arity == 2
        )
    )
    schema = json.dumps(
        intent_compiler_wire_json_schema(), ensure_ascii=False, sort_keys=True
    )
    return f"""\
/no_think
You are the intent_compiler for a 3D indoor scene critic. Extract hard,
geometry-verifiable relations from the original scene prompt and the optional
normalized SceneTaskSpec supplied below. Do not use inventory object positions,
memory, StageBrief, current scene state, or knowledge not present in these inputs.

The normalized SceneTaskSpec required_* arrays are authoritative hard inventory,
but the compiler adds their required_count rows deterministically. NEVER emit a
required_count relation. The original prompt remains authoritative for spatial
relations, negation, inventory mentions, and scope decisions. SceneTaskSpec
interaction_constraints and geometrically verifiable aesthetic_constraints are
expert inferences with lower priority.
A supplemental constraint may become a hard
relation only when it maps precisely to one of the registered relations and
can be checked by that relation's existing geometry evaluator. Material palette, style,
visual harmony, density, and other advice with no exact registered geometric
relation remain context only and MUST NOT produce a relation. Never invent a
new relation name. If a supplemental constraint duplicates or conflicts with
an explicit prompt relation, emit only the explicit prompt relation.
SceneTaskSpec room_type, style, and functional_zones are context only and do
not independently authorize hard relations.

Return a CompilerSemanticIR requirement for every supplied grounding ID. Use
prompt:N for an original-prompt requirement, interaction:N for an interaction
constraint, or aesthetic:N for an aesthetic constraint. A spatial requirement
uses kind="relation"; an inventory mention uses kind="inventory"; a prohibition
such as "no TV" uses kind="forbidden_inventory" and forbidden_category; and a
requirement the critic cannot safely enforce uses unsupported, unresolved, or
soft_scope with a brief reason. Return the ID only. NEVER copy source text and
NEVER emit source, evidence_span, inference_reason, warnings, explanations, or
other provenance text. The compiler expands the ID deterministically after
generation.

Every relation endpoint MUST be one of the exact entity_ref values in the entity
catalog. Use subject_ref, target_ref, and (only for a binary relation)
secondary_target_ref; do not emit targets, subjects, category, or any free-form
endpoint. An inventory row requires subject_ref="inventory:<category>" and no
target ref. A forbidden_inventory row has no endpoint refs because the forbidden
object is intentionally absent from the catalog.

Map accessibility language according to what the geometry evaluator can
actually measure. "X reachable/accessible from Y" may emit near(X, Y); it does
not imply flanking, a side assignment, or symmetry. "Keep X's front/operation
area clear" may emit clear_access(X, room). General phrases such as "clear
walking paths around the room" do not identify an exact registered endpoint
relation and must remain context only.

Preserve explicit adjacency wording: "next to", "beside", and "adjacent to"
must emit next_to. Emit the looser near relation only for literal "near" wording
or the accessibility mapping described above. Never downgrade next_to to near.

Registered relations:
{relation_lines}

Target rule is mandatory. Relations with target arity 1 ({unary_relations})
MUST include a non-null target_ref. Target arity is the number of entity
endpoints, not the number of physical target objects: group relations such as
paired_with and one_per_support may use target_count greater than one.
``corner_of_room``, ``corner_distribution``, and ``centered_in_room`` use
target_ref="anchor:room". For example:
``{{"requirement_id": "nightstands_by_bed", "kind": "relation", "grounding": "prompt:0", "relation": "flanking", "subject_ref": "inventory:nightstand", "subject_count": 2, "target_ref": "inventory:bed", "target_count": 1}}``.
Relations with target arity 2 ({binary_relations}) must also use
secondary_target_ref for the second endpoint.

For a rectangular target with edge-specific distribution, emit exactly one
edge_distribution relation. It must contain subject_count, target_ref,
target_count=1, edge_frame target_local_rectangle, and groups. Each edge class
has one counts_per_edge pair, sorted descending. The sum of all counts must
equal subject_count. Use [3, 3] for three objects on each long edge and [1, 0]
for one object on either short edge. Use toward_target only when the prompt
requires all subjects to face inward; do not also emit a duplicate faces row.
Do not emit one_per_side; that relation was removed.
For wording such as "two objects on each side of a target", when no long/short
edge is explicitly named, emit the reusable flanking relation instead of
edge_distribution. Reserve edge_distribution for explicit finite rectangular
edge layouts or unambiguous long/short edge wording.

Target shape is strict: a relation with registered target arity 1 must have
exactly one target_ref and must omit secondary_target_ref. Only a relation
whose registered target arity is 2 may use secondary_target_ref. Never combine
two target nouns into one unary relation; emit separate relation rows when the
prompt states separate relations.

Use centered_on_wall only when the prompt says the subject is centered on the
wall itself. For "X on the wall, centered directly above Y", emit
centered_above(X, Y) plus on_wall(X, wall); the centerline belongs to Y, not to
the full wall.

Use one_per_support for explicit "one X on/at each Y" requirements. Give
subject_count and target_count the same explicit value and use
subject_quantifier="all" and target_quantifier="all". Use corner_distribution
only when the prompt explicitly assigns multiple subjects to distinct room
corners or says one subject per corner; ordinary singular corner wording uses
corner_of_room.

For a unary relation that says an object is on/near one, another, or the other
member of a target category that the prompt explicitly repeats, keep one
target_ref with target_count=1 but use target_quantifier="at_least". This is
an existential relation: any matching target may satisfy it, without selecting
an arbitrary generated object ID. Use target_quantifier="exactly" only when the
prompt identifies a unique target instance.

For collective subject wording such as a set, collection, assortment, or
several objects, use subject_quantifier="at_least". The subject_count is a
minimum; do not turn an unspecified collection into an exactly-one hard
constraint just because the collection itself is singular.

Treat wall-relative phrases precisely. In "X against the wall behind Y" (and
the equivalent "X against the wall in front of Y"), "behind Y" locates the
wall; it does not state that X is behind Y. Emit against_wall(X, wall) for
that clause and do not emit behind(X, Y) or in_front_of(X, Y), unless a
separate clause explicitly states the object-to-object directional relation.

Return only one JSON object matching this schema:
{schema}

Do not fill provenance or runtime fields; the compiler derives them
deterministically after validation.
"""


class IntentCompiler:
    """Compile LLM semantic IR into an admitted, versioned intent contract."""

    SPEC_VERSION = INTENT_COMPILER_SPEC_VERSION
    SCHEMA_VERSION = INTENT_CONTRACT_SCHEMA_VERSION

    def __init__(
        self,
        model: str,
        api_base_url: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 8192,
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
        self.last_trace: dict[str, Any] = {}

    @staticmethod
    def _prompt_metadata(prompt: str) -> tuple[str, str]:
        normalized = " ".join(str(prompt or "").split())
        return normalized, hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    @staticmethod
    def _normalize_constraints(values: Any) -> tuple[str, ...]:
        if not isinstance(values, (list, tuple)):
            return ()
        normalized: list[str] = []
        seen: set[str] = set()
        for value in values:
            text = " ".join(str(value or "").split())
            if text and text not in seen:
                normalized.append(text)
                seen.add(text)
        return tuple(normalized)

    @classmethod
    def _normalize_task_spec(
        cls,
        task_spec: Any | None,
        *,
        interaction_constraints: Any = None,
        aesthetic_constraints: Any = None,
    ) -> dict[str, Any]:
        if (
            task_spec is None
            and interaction_constraints is None
            and aesthetic_constraints is None
        ):
            return {}
        if isinstance(task_spec, dict):
            payload = dict(task_spec)
        elif hasattr(task_spec, "model_dump"):
            payload = task_spec.model_dump(mode="json", exclude_none=True)
        else:
            payload = {}
        if interaction_constraints is not None:
            payload["interaction_constraints"] = list(interaction_constraints)
        if aesthetic_constraints is not None:
            payload["aesthetic_constraints"] = list(aesthetic_constraints)
        for field in (
            "required_large_objects",
            "required_wall_objects",
            "required_ceiling_objects",
            "required_small_objects",
        ):
            values = payload.get(field)
            payload[field] = (
                [
                    text
                    for value in values
                    if (text := " ".join(str(value or "").split()))
                ]
                if isinstance(values, (list, tuple))
                else []
            )
        for field in (
            "functional_zones",
            "interaction_constraints",
            "aesthetic_constraints",
        ):
            payload[field] = list(cls._normalize_constraints(payload.get(field)))
        for field in ("room_type", "style"):
            payload[field] = " ".join(str(payload.get(field) or "").split())
        return payload

    @staticmethod
    def _finish_reason(response: Any) -> str:
        choices = getattr(response, "choices", None) or []
        return str(getattr(choices[0], "finish_reason", "") or "") if choices else ""

    @staticmethod
    def _token_usage(response: Any) -> dict[str, int]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return {}
        payload = usage.model_dump() if hasattr(usage, "model_dump") else vars(usage)
        return {
            str(key): int(value)
            for key, value in payload.items()
            if isinstance(value, int)
        }

    @staticmethod
    def _task_compiler_trace_fields(
        constraints: list[dict[str, Any]],
        interaction_constraints: tuple[str, ...],
        aesthetic_constraints: tuple[str, ...],
        rejected: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        injected = {
            "interaction_constraints": list(interaction_constraints),
            "aesthetic_constraints": list(aesthetic_constraints),
        }
        accepted = [
            {
                "constraint_id": str(row.get("constraint_id") or ""),
                "relation": str(row.get("relation") or ""),
                "subject_category": canonical_selector_category(
                    (row.get("subjects") or {}).get("category")
                ),
                "inference_reason": str(row.get("inference_reason") or ""),
            }
            for row in constraints
            if isinstance(row, dict)
            and str(row.get("source") or "") == "model_inferred"
        ]
        accepted_reasons = {
            " ".join(str(row.get("inference_reason") or "").split()).casefold()
            for row in constraints
            if isinstance(row, dict)
            and str(row.get("source") or "") == "model_inferred"
        }
        rejected_by_reason = {
            " ".join(str(row.get("inference_reason") or "").split()).casefold(): str(
                row.get("reason") or "invalid"
            )
            for row in (rejected or [])
            if isinstance(row, dict) and row.get("inference_reason")
        }
        dispositions: list[dict[str, str]] = []
        soft_terms = re.compile(
            r"\b(?:aesthetic|appearance|atmosphere|circulation|cozy|density|"
            r"functional layout|harmony|material|overcrowd|palette|style|visual)\b",
            re.IGNORECASE,
        )
        for field, values in (
            ("interaction_constraints", interaction_constraints),
            ("aesthetic_constraints", aesthetic_constraints),
        ):
            for value in values:
                reason = f"TaskCompiler {field}: {value}".casefold()
                rejected_reason = rejected_by_reason.get(reason, "")
                if reason in accepted_reasons:
                    disposition = "accepted"
                elif rejected_reason == "duplicate_of_explicit_prompt":
                    disposition = "duplicate-explicit"
                elif rejected_reason == "conflicts_with_explicit_prompt":
                    disposition = "conflict"
                elif rejected_reason:
                    disposition = "invalid"
                elif field == "aesthetic_constraints" or soft_terms.search(value):
                    disposition = "unsupported-soft"
                else:
                    disposition = "invalid"
                dispositions.append(
                    {"field": field, "text": value, "disposition": disposition}
                )
        unmapped = {
            field: [
                row["text"]
                for row in dispositions
                if row["field"] == field and row["disposition"] == "invalid"
            ]
            for field, values in (
                ("interaction_constraints", interaction_constraints),
                ("aesthetic_constraints", aesthetic_constraints),
            )
        }
        return {
            "task_compiler_constraints": injected,
            "accepted_task_compiler_constraints": accepted,
            "rejected_task_compiler_constraints": list(rejected or []),
            "task_compiler_constraint_dispositions": dispositions,
            "unmapped_task_compiler_constraints": unmapped,
        }

    @staticmethod
    def _raw_message(response: Any) -> str:
        message = response.choices[0].message
        raw = getattr(message, "content", None)
        if not raw:
            raw = getattr(message, "reasoning_content", None)
        if not raw:
            extra = getattr(message, "model_extra", None)
            if isinstance(extra, dict):
                raw = extra.get("reasoning_content")
        if isinstance(raw, list):
            raw = "".join(
                str(item.get("text") or item) if isinstance(item, dict) else str(item)
                for item in raw
            )
        return str(raw or "")

    def _messages(
        self,
        prompt: str,
        *,
        task_spec: dict[str, Any] | None = None,
        grounding_catalog: str = "",
        entity_catalog: str = "",
        previous_output: str = "",
        validation_error: str = "",
    ) -> list[dict[str, str]]:
        user = f"Original scene prompt:\n{prompt}"
        if task_spec:
            user += (
                "\n\nNormalized SceneTaskSpec. Its required_* inventory is "
                "authoritative; its spatial constraints are lower priority than "
                "explicit prompt relations:\n"
                + json.dumps(task_spec, ensure_ascii=False, indent=2, sort_keys=True)
            )
        if grounding_catalog:
            user += (
                "\n\nGrounding catalog. Every explicit input requirement must "
                "appear in requirements[] with one ID from this list:\n"
                + grounding_catalog
            )
        if entity_catalog:
            user += (
                "\n\nEntity catalog. Every relation endpoint must use these exact "
                "entity_ref values; never write a free-form category:\n"
                + entity_catalog
            )
        user += (
            "\n\nReturn CompilerSemanticIR, not an IntentContract. Each prompt or "
            "TaskSpec grounding needs a requirement disposition: relation/inventory "
            "for compiled semantics, forbidden_inventory, unsupported, unresolved, "
            "or soft_scope. required_count is generated from SceneTaskSpec only; do "
            "not emit it. Relations require subject_ref and catalog target refs."
        )
        if validation_error:
            user += (
                "\n\nThe previous candidate was invalid. Correct it and return a "
                "complete replacement JSON object. Validation error:\n"
                f"{validation_error}"
            )
            if previous_output:
                user += f"\nPrevious candidate:\n{previous_output}"
            if "finish_reason=length" in validation_error:
                user += (
                    "\nThe previous response was truncated. Return a concise but "
                    "complete contract; do not repeat explanations or context."
                )
            if "omitted subject_ref" in validation_error:
                user += (
                    "\nThe reported relation is missing subject_ref. Use the exact "
                    "entity_ref from the entity catalog for its subject; a "
                    "subject_role or subject_cohort is optional metadata, never an "
                    "endpoint."
                )
            if "omitted target_ref" in validation_error:
                user += (
                    "\nThe reported relation is missing target_ref. Every relation "
                    "with a target must use the exact entity_ref from the entity "
                    "catalog for that target. target_role and target_cohort are "
                    "optional metadata only; they never replace target_ref. For "
                    "example, a relation targeting a window uses "
                    'target_ref="anchor:window".'
                )
            if "omitted required constraints field" in validation_error:
                user += (
                    "\nThe top-level constraints field is mandatory, even when it "
                    "is an empty list. Return all hard relations in that field; "
                    "warnings alone are not a complete contract."
                )
            if "requires 1 target(s), got 2" in validation_error:
                user += (
                    "\nFor every reported unary relation, remove its "
                    "secondary_category, secondary_count, and secondary_role; "
                    "keep exactly one primary target selector."
                )
            if "requires 1 target(s), got 0" in validation_error or (
                "requires 2 target(s), got 0" in validation_error
            ):
                user += (
                    "\nEvery non-count relation must include its prompt-grounded "
                    "targets object. Do not explain the target in inference_reason; "
                    "put the endpoint category in targets."
                )
            if "wall-relative directional relation" in validation_error:
                user += (
                    "\nDo not convert 'X against the wall behind/in front of Y' "
                    "into a directional X-to-Y relation. Keep only the wall "
                    "relation unless a separate clause explicitly gives that "
                    "object-to-object direction."
                )
            if (
                "centered_on_wall hard intent requires explicit wall wording"
                in validation_error
            ):
                user += (
                    "\nUse centered_on_wall only when the selected grounding text "
                    "explicitly says wall. Centering on a table side or edge belongs "
                    "in edge_distribution, not centered_on_wall."
                )
            if (
                "flanking hard intent requires explicit bilateral wording"
                in validation_error
            ):
                user += (
                    "\nDo not infer flanking from reachability or accessibility. "
                    "If the supplied TaskCompiler constraint says X is reachable "
                    "or accessible from Y, use near(X, Y) with that exact "
                    "TaskCompiler provenance, provided the relation is otherwise "
                    "geometry-verifiable."
                )
        return [
            {
                "role": "system",
                "content": prepend_text_thinking_directive(
                    _system_prompt(),
                    thinking_directive_from_effort("none", model=self._model),
                ),
            },
            {"role": "user", "content": user},
        ]

    def compile(
        self,
        prompt: str,
        interaction_constraints: list[str] | tuple[str, ...] | None = None,
        aesthetic_constraints: list[str] | tuple[str, ...] | None = None,
        task_spec: Any | None = None,
    ) -> dict[str, Any]:
        """Compile live Prompt semantics only from a grounded LLM IR.

        Structured TaskSpec inventory is projected deterministically. All prompt
        relations, negation, unsupported requirements, and coverage
        dispositions must be present in the model response; failures remain
        runtime failures after the bounded corrective retry.
        """

        normalized_prompt, prompt_hash = self._prompt_metadata(prompt)
        normalized_task_spec = self._normalize_task_spec(
            task_spec,
            interaction_constraints=interaction_constraints,
            aesthetic_constraints=aesthetic_constraints,
        )
        grounding_catalog, rendered_grounding_catalog = _grounding_catalog(
            normalized_prompt, normalized_task_spec
        )
        entity_catalog = _entity_catalog(normalized_task_spec)
        rendered_entity_catalog = _render_entity_catalog(entity_catalog)
        previous_output = ""
        last_error = ""
        attempts: list[dict[str, Any]] = []

        for attempt in range(2):
            messages = self._messages(
                normalized_prompt,
                task_spec=normalized_task_spec,
                grounding_catalog=rendered_grounding_catalog,
                entity_catalog=rendered_entity_catalog,
                previous_output=previous_output,
                validation_error=last_error,
            )
            started_at = time.perf_counter()
            response = None
            raw = ""
            semantic_ir: Any = None
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                    response_format=json_response_format(
                        model=self._model,
                        name="compiler_semantic_ir",
                        schema=intent_compiler_wire_json_schema(),
                    ),
                    extra_body=chat_template_kwargs_from_effort(
                        "none", model=self._model
                    ),
                )
                raw = self._raw_message(response)
                finish_reason = self._finish_reason(response)
                if finish_reason == "length":
                    raise ValueError("finish_reason=length: semantic IR was truncated")
                semantic_ir = parse_llm_json_object(raw)
                contract, ledger, rejected = _admit_semantic_ir(
                    semantic_ir,
                    prompt=normalized_prompt,
                    prompt_hash=prompt_hash,
                    task_spec=normalized_task_spec,
                    grounding_catalog=grounding_catalog,
                    catalog=entity_catalog,
                    retry_count=attempt,
                )
                result = validate_intent_contract(
                    contract, validate_prompt_semantics=False
                )
                status = "retry_ok" if attempt else "ok"
                attempt_record = {
                    "attempt": attempt,
                    "status": status,
                    "finish_reason": finish_reason,
                    "token_usage": self._token_usage(response),
                    "elapsed_sec": round(time.perf_counter() - started_at, 6),
                    "accepted_entity_refs": sorted(
                        {ref for row in ledger for ref in row.get("entity_refs") or []}
                    ),
                    "rejected_entity_refs": rejected,
                }
                attempts.append(attempt_record)
                self.last_trace = {
                    "status": status,
                    "spec_version": self.SPEC_VERSION,
                    "semantic_ir_version": INTENT_COMPILER_SEMANTIC_IR_VERSION,
                    "prompt_sha256": prompt_hash,
                    "normalized_task_spec": normalized_task_spec,
                    "entity_catalog": list(entity_catalog.values()),
                    "constraints": result.get("constraints", []),
                    "coverage_ledger": result.get("coverage_ledger", []),
                    "coverage_requirements": result.get("coverage_requirements", []),
                    "retry_count": attempt,
                    "failure_reason": "",
                    "attempts": attempts,
                    "rejected_entity_refs": [
                        row
                        for attempt_row in attempts
                        for row in attempt_row.get("rejected_entity_refs") or []
                    ],
                }
                _append_llm_debug(
                    build_llm_call_debug_record(
                        stage="intent_compiler",
                        agent_role="intent_compiler",
                        event="compile",
                        prompt=messages,
                        output=raw,
                        raw_response=response,
                    ).model_dump()
                    | {
                        "input": messages,
                        "output": raw,
                        "status": status,
                        "attempt": attempt,
                        "semantic_ir_version": INTENT_COMPILER_SEMANTIC_IR_VERSION,
                        "entity_catalog": list(entity_catalog.values()),
                        "coverage_ledger": result.get("coverage_ledger", []),
                        "accepted_entity_refs": attempt_record["accepted_entity_refs"],
                        "rejected_entity_refs": rejected,
                    }
                )
                return result
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                previous_output = (
                    "" if self._finish_reason(response) == "length" else raw
                )
                rejected_entity_refs = _rejected_entity_refs(
                    semantic_ir, entity_catalog
                )
                attempts.append(
                    {
                        "attempt": attempt,
                        "status": "error",
                        "error": last_error,
                        "finish_reason": self._finish_reason(response),
                        "token_usage": self._token_usage(response),
                        "elapsed_sec": round(time.perf_counter() - started_at, 6),
                        "rejected_entity_refs": rejected_entity_refs,
                    }
                )
                _append_llm_debug(
                    build_llm_call_debug_record(
                        stage="intent_compiler",
                        agent_role="intent_compiler",
                        event="compile",
                        prompt=messages,
                        output=raw,
                        raw_response=response,
                        error=last_error,
                    ).model_dump()
                    | {
                        "input": messages,
                        "output": raw,
                        "status": "error",
                        "attempt": attempt,
                        "semantic_ir_version": INTENT_COMPILER_SEMANTIC_IR_VERSION,
                        "entity_catalog": list(entity_catalog.values()),
                        "error": last_error,
                        "rejected_entity_refs": rejected_entity_refs,
                    }
                )
                logger.warning(
                    "IntentCompiler attempt %d failed: %s", attempt + 1, last_error
                )

        self.last_trace = {
            "status": "error",
            "spec_version": self.SPEC_VERSION,
            "semantic_ir_version": INTENT_COMPILER_SEMANTIC_IR_VERSION,
            "prompt_sha256": prompt_hash,
            "normalized_task_spec": normalized_task_spec,
            "entity_catalog": list(entity_catalog.values()),
            "constraints": [],
            "coverage_ledger": [],
            "retry_count": 1,
            "failure_reason": last_error,
            "attempts": attempts,
            "rejected_entity_refs": [
                row
                for attempt in attempts
                for row in attempt.get("rejected_entity_refs") or []
            ],
        }
        raise IntentCompilationError(
            f"IntentCompiler failed after two attempts: {last_error}",
            trace=self.last_trace,
        )

    def _compile_legacy_semantics(
        self,
        prompt: str,
        interaction_constraints: list[str] | tuple[str, ...] | None = None,
        aesthetic_constraints: list[str] | tuple[str, ...] | None = None,
        task_spec: Any | None = None,
    ) -> dict[str, Any]:
        normalized_prompt, prompt_hash = self._prompt_metadata(prompt)
        normalized_task_spec = self._normalize_task_spec(
            task_spec,
            interaction_constraints=interaction_constraints,
            aesthetic_constraints=aesthetic_constraints,
        )
        normalized_interaction = self._normalize_constraints(
            normalized_task_spec.get("interaction_constraints")
        )
        normalized_aesthetic = self._normalize_constraints(
            normalized_task_spec.get("aesthetic_constraints")
        )
        allowed_inference_reasons = tuple(
            f"TaskCompiler {field}: {text}"
            for field, values in (
                ("interaction_constraints", normalized_interaction),
                ("aesthetic_constraints", normalized_aesthetic),
            )
            for text in values
        )
        grounding_catalog, rendered_grounding_catalog = _grounding_catalog(
            normalized_prompt, normalized_task_spec
        )
        previous_output = ""
        last_error = ""
        attempts: list[dict[str, Any]] = []
        failed_llm_warnings: list[str] = []

        for attempt in range(2):
            messages = self._messages(
                normalized_prompt,
                task_spec=normalized_task_spec,
                grounding_catalog=rendered_grounding_catalog,
                previous_output=previous_output,
                validation_error=last_error,
            )
            started_at = time.perf_counter()
            raw = ""
            response = None
            response_elapsed_sec: float | None = None
            response_warnings: list[str] = []
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                    # llama.cpp converts json_schema to a grammar and constrains
                    # every generated token to a schema-valid JSON value.
                    response_format=json_response_format(
                        model=self._model,
                        name="intent_contract",
                        schema=intent_compiler_wire_json_schema(),
                    ),
                    extra_body=openrouter_extra_body(
                        chat_template_kwargs_from_effort(
                            "none", model=self._model
                        )
                    ),
                )
                response_elapsed_sec = round(time.perf_counter() - started_at, 6)
                raw = self._raw_message(response)
                finish_reason = self._finish_reason(response)
                if finish_reason == "length":
                    try:
                        truncated_data = parse_llm_json_object(raw)
                        response_warnings = _merge_warnings(
                            truncated_data.get("warnings")
                        )
                        failed_llm_warnings = _merge_warnings(
                            failed_llm_warnings, response_warnings
                        )
                    except Exception:
                        pass
                    raise ValueError(
                        "finish_reason=length: IntentCompiler output was truncated"
                    )
                data = parse_llm_json_object(raw)
                response_warnings = _merge_warnings(data.get("warnings"))
                failed_llm_warnings = _merge_warnings(
                    failed_llm_warnings, response_warnings
                )
                if "constraints" not in data:
                    raise IncompleteIntentContractError(
                        "response omitted required constraints field"
                    )
                payload = dict(data)
                payload.setdefault("schema_version", self.SCHEMA_VERSION)
                payload.setdefault("prompt", normalized_prompt)
                payload.setdefault("prompt_sha256", prompt_hash)
                payload.setdefault("intent_compiler_spec_version", self.SPEC_VERSION)
                payload["prompt"] = normalized_prompt
                payload["prompt_sha256"] = prompt_hash
                payload["intent_compiler_spec_version"] = self.SPEC_VERSION
                payload["retry_count"] = attempt
                payload.setdefault(
                    "room_type", str(normalized_task_spec.get("room_type") or "")
                )
                payload = _attach_grounding_provenance(payload, grounding_catalog)
                payload = _normalize_side_distribution(payload)
                payload = _normalize_seating_support_relations(payload)
                payload, normalized_centered_above = (
                    _normalize_centered_above_relations(payload, normalized_prompt)
                )
                payload, restored_targets = _restore_missing_target_selectors(
                    payload, normalized_prompt
                )
                payload, restored_fields = _restore_missing_relation_fields(
                    payload, normalized_prompt, normalized_task_spec
                )
                payload = _remove_redundant_edge_faces(payload)
                payload = _normalize_provenance_fields(payload)
                payload = _validate_prompt_grounded_relations(
                    payload, normalized_prompt
                )
                payload, rejected_task_constraints = (
                    _prefer_explicit_prompt_constraints(
                        payload,
                        allowed_inference_reasons=allowed_inference_reasons,
                    )
                )
                result = validate_intent_contract(payload)
                (
                    result,
                    enriched_constraints,
                    enrichment_rejections,
                ) = _enrich_with_deterministic_constraints(
                    result,
                    normalized_prompt,
                    allowed_inference_reasons,
                    normalized_task_spec,
                )
                # LLM-supplied relations and deterministic enrichment must
                # share endpoint/count semantics derived from SceneTaskSpec.
                # In particular, a place setting's cutlery can expand into a
                # fork and knife, so its support constraint is a minimum
                # coverage requirement rather than an exact physical count.
                result["constraints"] = _apply_task_spec_contract_metadata(
                    list(result.get("constraints") or []), normalized_task_spec
                )
                result = ensure_coverage_requirements(
                    result, task_spec=normalized_task_spec
                )
                result = validate_intent_contract(result)
                result["warnings"] = _merge_warnings(
                    failed_llm_warnings, result.get("warnings")
                )
                _validate_contract_completeness(result, normalized_task_spec)
                rejected_task_constraints.extend(enrichment_rejections)
                task_constraint_trace = self._task_compiler_trace_fields(
                    result.get("constraints", []),
                    normalized_interaction,
                    normalized_aesthetic,
                    rejected_task_constraints,
                )
                token_usage = self._token_usage(response)
                completion_tokens = int(token_usage.get("completion_tokens") or 0)
                capacity_warning = ""
                if completion_tokens >= int(self._max_tokens * 0.9):
                    capacity_warning = (
                        "IntentCompiler completion used at least 90% of the "
                        f"{self._max_tokens}-token output limit"
                    )
                    warnings = list(result.get("warnings") or [])
                    if capacity_warning not in warnings:
                        warnings.append(capacity_warning)
                        result["warnings"] = warnings
                status = (
                    "retry_ok"
                    if attempt
                    else ("ok_enriched" if enriched_constraints else "ok")
                )
                self.last_trace = {
                    "status": status,
                    "spec_version": self.SPEC_VERSION,
                    "prompt_sha256": prompt_hash,
                    "normalized_task_spec": normalized_task_spec,
                    "constraints": result.get("constraints", []),
                    "warnings": result.get("warnings", []),
                    "enriched_constraints": enriched_constraints,
                    "normalized_centered_above": normalized_centered_above,
                    "restored_targets": restored_targets,
                    "restored_fields": restored_fields,
                    "retry_count": attempt,
                    "failure_reason": "",
                    "finish_reason": finish_reason,
                    "token_usage": token_usage,
                    "capacity_warning": capacity_warning,
                } | task_constraint_trace
                attempts.append(
                    {
                        "attempt": attempt,
                        "status": status,
                        "finish_reason": finish_reason,
                        "token_usage": token_usage,
                        "warnings": response_warnings,
                        "elapsed_sec": (
                            response_elapsed_sec
                            if response_elapsed_sec is not None
                            else round(time.perf_counter() - started_at, 6)
                        ),
                    }
                )
                self.last_trace["attempts"] = attempts
                _append_llm_debug(
                    build_llm_call_debug_record(
                        stage="intent_compiler",
                        agent_role="intent_compiler",
                        event="compile",
                        prompt=messages,
                        output=raw,
                        raw_response=response,
                    ).model_dump()
                    | {
                        "input": messages,
                        "output": raw,
                        "status": status,
                        "attempt": attempt,
                        "elapsed_sec": response_elapsed_sec,
                        "finish_reason": finish_reason,
                        "enriched_constraints": enriched_constraints,
                        "normalized_centered_above": normalized_centered_above,
                        "restored_targets": restored_targets,
                        "restored_fields": restored_fields,
                        **task_constraint_trace,
                    }
                )
                return result
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                previous_output = (
                    "" if self._finish_reason(response) == "length" else raw
                )
                attempts.append(
                    {
                        "attempt": attempt,
                        "status": "error",
                        "error": last_error,
                        "finish_reason": self._finish_reason(response),
                        "token_usage": self._token_usage(response),
                        "warnings": response_warnings,
                        "elapsed_sec": round(time.perf_counter() - started_at, 6),
                    }
                )
                _append_llm_debug(
                    build_llm_call_debug_record(
                        stage="intent_compiler",
                        agent_role="intent_compiler",
                        event="compile",
                        prompt=messages,
                        output=raw,
                        raw_response=response,
                        error=last_error,
                    ).model_dump()
                    | {
                        "input": messages,
                        "output": raw,
                        "status": "error",
                        "attempt": attempt,
                        "elapsed_sec": round(time.perf_counter() - started_at, 6),
                        "finish_reason": self._finish_reason(response),
                    }
                )
                logger.warning(
                    "IntentCompiler attempt %d failed: %s", attempt + 1, last_error
                )

        fallback = build_intent_contract(
            normalized_prompt,
            room_type=str(normalized_task_spec.get("room_type") or ""),
            task_spec=normalized_task_spec,
        )
        fallback["retry_count"] = 1
        fallback["warnings"] = _merge_warnings(
            failed_llm_warnings,
            fallback.get("warnings"),
            ["LLM contract unavailable or invalid; deterministic prompt parser used"],
        )
        try:
            result = validate_intent_contract(fallback)
            _validate_contract_completeness(result, normalized_task_spec)
        except Exception:
            pass
        else:
            attempts.append(
                {
                    "attempt": 2,
                    "status": "deterministic_fallback",
                    "elapsed_sec": 0.0,
                }
            )
            task_constraint_trace = self._task_compiler_trace_fields(
                result.get("constraints", []),
                normalized_interaction,
                normalized_aesthetic,
            )
            self.last_trace = {
                "status": "deterministic_fallback",
                "spec_version": self.SPEC_VERSION,
                "prompt_sha256": prompt_hash,
                "normalized_task_spec": normalized_task_spec,
                "constraints": result.get("constraints", []),
                "warnings": result.get("warnings", []),
                "retry_count": 1,
                "failure_reason": last_error,
                "attempts": attempts,
            } | task_constraint_trace
            _append_llm_debug(
                build_llm_call_debug_record(
                    stage="intent_compiler",
                    agent_role="intent_compiler",
                    event="deterministic_fallback",
                    prompt=self._messages(
                        normalized_prompt,
                        task_spec=normalized_task_spec,
                        grounding_catalog=rendered_grounding_catalog,
                    ),
                    output=json.dumps(fallback, ensure_ascii=False),
                    error=last_error,
                ).model_dump()
                | {
                    "input": self._messages(
                        normalized_prompt,
                        task_spec=normalized_task_spec,
                        grounding_catalog=rendered_grounding_catalog,
                    ),
                    "output": fallback,
                    "status": "deterministic_fallback",
                    "attempt": 2,
                }
            )
            return result

        self.last_trace = {
            "status": "error",
            "spec_version": self.SPEC_VERSION,
            "prompt_sha256": prompt_hash,
            "normalized_task_spec": normalized_task_spec,
            "constraints": [],
            "retry_count": 1,
            "failure_reason": last_error,
            "attempts": attempts,
        } | self._task_compiler_trace_fields(
            [], normalized_interaction, normalized_aesthetic
        )
        raise IntentCompilationError(
            f"IntentCompiler failed after two attempts: {last_error}",
            trace=self.last_trace,
        )
