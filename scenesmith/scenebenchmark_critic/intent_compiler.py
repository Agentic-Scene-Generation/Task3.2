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

from scenesmith.agent_utils.thinking import chat_template_kwargs_from_effort
from scenesmith.scene_expert.context_bundle import build_llm_call_debug_record
from scenesmith.scenebenchmark_critic.intent_schema import (
    INTENT_COMPILER_SPEC_VERSION,
    INTENT_CONTRACT_SCHEMA_VERSION,
    canonical_selector_category,
    intent_contract_json_schema,
    selector_categories_overlap,
    validate_intent_contract,
)
from scenesmith.scenebenchmark_critic.intent_contract import build_intent_contract
from scenesmith.scenebenchmark_critic.relation_registry import (
    RELATION_REGISTRY,
    relation_spec,
    relations_are_exclusive,
)
from scenesmith.utils.llm_json import parse_llm_json_object

logger = logging.getLogger(__name__)


class IntentCompilationError(RuntimeError):
    """Raised when both independent intent compilation attempts fail."""

    def __init__(self, message: str, *, trace: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.trace = trace or {}


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


_DETERMINISTIC_ENRICHMENT_SOURCES = frozenset({"explicit_prompt", "room_ontology"})
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
        if str(constraint.get("relation") or "") != "flanking":
            continue
        source = str(constraint.get("source") or "")
        if source == "model_inferred":
            evidence = str(constraint.get("inference_reason") or "")
        else:
            evidence = " ".join(
                str(value or "") for value in (constraint.get("evidence_span"), prompt)
            )
        if any(pattern.search(evidence) for pattern in _FLANKING_GROUNDING_PATTERNS):
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
    payload: dict[str, Any], prompt: str
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

    deterministic_constraints = build_intent_contract(prompt).get("constraints") or []
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
            for field in ("edge_frame", "groups", "orientation"):
                if repaired.get(field) or not candidate.get(field):
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
    if source == "room_ontology":
        return True
    return (
        source == "explicit_prompt"
        and str(candidate.get("relation") or "") in _HIGH_CONFIDENCE_EXPLICIT_RELATIONS
    )


def _enrich_with_deterministic_constraints(
    llm_contract: dict[str, Any],
    prompt: str,
    allowed_inference_reasons: tuple[str, ...],
) -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, str]]]:
    """Restore explicit and ontology constraints omitted by a valid LLM response.

    The independent compiler has higher recall for nuanced prompt relations, but
    the deterministic parser owns a small set of high-confidence facts.  A valid
    LLM response must therefore supplement rather than replace those facts.
    """
    deterministic = build_intent_contract(prompt)
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
        if any(
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
        intent_contract_json_schema(), ensure_ascii=False, sort_keys=True
    )
    return f"""\
/no_think
You are the intent_compiler for a 3D indoor scene critic. Extract hard,
geometry-verifiable relations from the original scene prompt and the optional
TaskCompiler constraints supplied below. Do not use inventory object positions,
memory, StageBrief, current scene state, or knowledge not present in these inputs.

The original scene prompt is authoritative. Relations grounded in it use
source="explicit_prompt" and copy a verbatim evidence_span. Supplemental
TaskCompiler interaction_constraints and aesthetic_constraints are expert
inferences with lower priority. A supplemental constraint may become a hard
relation only when it maps precisely to one of the registered relations and
can be checked by that relation's existing geometry evaluator. Such a relation
must use source="model_inferred", leave evidence_span empty, and set
inference_reason to "TaskCompiler <field>: <verbatim constraint>", where field
is interaction_constraints or aesthetic_constraints. Material palette, style,
visual harmony, density, and other advice with no exact registered geometric
relation remain context only and MUST NOT produce a relation. Never invent a
new relation name. If a supplemental constraint duplicates or conflicts with
an explicit prompt relation, emit only the explicit prompt relation.

Map accessibility language according to what the geometry evaluator can
actually measure. "X reachable/accessible from Y" may emit near(X, Y); it does
not imply flanking, a side assignment, or symmetry. "Keep X's front/operation
area clear" may emit clear_access(X, room). General phrases such as "clear
walking paths around the room" do not identify an exact registered endpoint
relation and must remain context only.

Registered relations:
{relation_lines}

Target rule is mandatory.  Relations with target arity 1 ({unary_relations})
MUST include a non-null ``targets`` selector with the target category from the
prompt. Target arity is the number of selector endpoints, not the number of
physical target objects: group relations such as paired_with and
one_per_support may use targets.count greater than one. ``corner_of_room``,
``corner_distribution``, and ``centered_in_room`` use
``{{"category": "room", "count": 1}}`` as that selector.  For example:
``{{"relation": "flanking", "subjects": {{"category": "nightstand", "count": 2}}, "targets": {{"category": "bed", "count": 1}}, "source": "explicit_prompt", "evidence_span": "..."}}``.
Relations with target arity 2 ({binary_relations}) must use a target selector
whose ``secondary_category`` identifies the second target.  Only target arity
0 relations, such as ``required_count``, MUST return ``"targets": null``.

For a rectangular target with edge-specific distribution, emit exactly one
edge_distribution relation. It must contain subjects.count, a target selector
with count 1, edge_frame target_local_rectangle, and groups. Each edge class
has one counts_per_edge pair, sorted descending. The sum of all counts must
equal subjects.count. Use [3, 3] for three objects on each long edge and [1, 0]
for one object on either short edge. Use toward_target only when the prompt
requires all subjects to face inward; do not also emit a duplicate faces row.
Use required_count for every explicitly counted object category used by an
edge_distribution relation. Do not emit one_per_side; that relation was
removed. Evidence spans must be copied from the original prompt verbatim.
For wording such as "two objects on each side of a target", when no long/short
edge is explicitly named, emit the reusable flanking relation instead of
edge_distribution. Reserve edge_distribution for explicit finite rectangular
edge layouts or unambiguous long/short edge wording.

Target shape is strict: a relation with registered target arity 1 must have
exactly one targets selector and must leave
secondary_category, secondary_count, and secondary_role empty. Only a relation
whose registered target arity is 2 may use secondary_category and
secondary_role. Never combine two target nouns into one unary relation; emit
separate relation rows when the prompt states separate relations.

Use centered_on_wall only when the prompt says the subject is centered on the
wall itself. For "X on the wall, centered directly above Y", emit
centered_above(X, Y) plus on_wall(X, wall); the centerline belongs to Y, not to
the full wall.

Use one_per_support for explicit "one X on/at each Y" requirements. Give both
selectors the same explicit count and quantifier="all". Use corner_distribution
only when the prompt explicitly assigns multiple subjects to distinct room
corners or says one subject per corner; ordinary singular corner wording uses
corner_of_room.

For a unary relation that says an object is on/near one, another, or the other
member of a target category that the prompt explicitly repeats, keep one target
selector with count 1 but use targets.quantifier="at_least". This is an
existential relation: any matching target may satisfy it, without selecting an
arbitrary generated object ID. Use quantifier="exactly" only when the prompt
identifies a unique target instance.

For collective subject wording such as a set, collection, assortment, or
several objects, use subjects.quantifier="at_least". The selector count is a
minimum; do not turn an unspecified collection into an exactly-one hard
constraint just because the collection itself is singular.

Treat wall-relative phrases precisely. In "X against the wall behind Y" (and
the equivalent "X against the wall in front of Y"), "behind Y" locates the
wall; it does not state that X is behind Y. Emit against_wall(X, wall) for
that clause and do not emit behind(X, Y) or in_front_of(X, Y), unless a
separate clause explicitly states the object-to-object directional relation.

Return only one JSON object matching this schema:
{schema}

Do not fill runtime fields such as constraint_id or stage; the compiler will
derive those deterministically after validation.
"""


class IntentCompiler:
    """Compile a prompt into a validated v4 contract with one corrective retry."""

    SPEC_VERSION = INTENT_COMPILER_SPEC_VERSION
    SCHEMA_VERSION = INTENT_CONTRACT_SCHEMA_VERSION

    def __init__(
        self,
        model: str,
        api_base_url: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 2048,
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
        referenced = "\n".join(
            [
                *(str(row.get("inference_reason") or "") for row in constraints),
                *(str(row.get("inference_reason") or "") for row in (rejected or [])),
            ]
        ).lower()
        unmapped = {
            field: [text for text in values if text.lower() not in referenced]
            for field, values in (
                ("interaction_constraints", interaction_constraints),
                ("aesthetic_constraints", aesthetic_constraints),
            )
        }
        return {
            "task_compiler_constraints": injected,
            "accepted_task_compiler_constraints": accepted,
            "rejected_task_compiler_constraints": list(rejected or []),
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
        interaction_constraints: tuple[str, ...] = (),
        aesthetic_constraints: tuple[str, ...] = (),
        previous_output: str = "",
        validation_error: str = "",
    ) -> list[dict[str, str]]:
        user = f"Original scene prompt:\n{prompt}"
        if interaction_constraints or aesthetic_constraints:
            supplemental = {
                "interaction_constraints": list(interaction_constraints),
                "aesthetic_constraints": list(aesthetic_constraints),
            }
            user += (
                "\n\nSupplemental TaskCompiler constraints (lower priority than "
                "the original prompt):\n"
                + json.dumps(supplemental, ensure_ascii=False, indent=2)
            )
        if validation_error:
            user += (
                "\n\nThe previous candidate was invalid. Correct it and return a "
                "complete replacement JSON object. Validation error:\n"
                f"{validation_error}\nPrevious candidate:\n{previous_output}"
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
            {"role": "system", "content": _system_prompt()},
            {"role": "user", "content": user},
        ]

    def compile(
        self,
        prompt: str,
        interaction_constraints: list[str] | tuple[str, ...] | None = None,
        aesthetic_constraints: list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        normalized_prompt, prompt_hash = self._prompt_metadata(prompt)
        normalized_interaction = self._normalize_constraints(interaction_constraints)
        normalized_aesthetic = self._normalize_constraints(aesthetic_constraints)
        allowed_inference_reasons = tuple(
            f"TaskCompiler {field}: {text}"
            for field, values in (
                ("interaction_constraints", normalized_interaction),
                ("aesthetic_constraints", normalized_aesthetic),
            )
            for text in values
        )
        previous_output = ""
        last_error = ""
        attempts: list[dict[str, Any]] = []

        for attempt in range(2):
            messages = self._messages(
                normalized_prompt,
                interaction_constraints=normalized_interaction,
                aesthetic_constraints=normalized_aesthetic,
                previous_output=previous_output,
                validation_error=last_error,
            )
            started_at = time.perf_counter()
            raw = ""
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                    # llama.cpp converts json_schema to a grammar and constrains
                    # every generated token to a schema-valid JSON value.
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "intent_contract",
                            "strict": True,
                            "schema": intent_contract_json_schema(),
                        },
                    },
                    extra_body=chat_template_kwargs_from_effort("none"),
                )
                raw = self._raw_message(response)
                data = parse_llm_json_object(raw)
                payload = dict(data)
                payload.setdefault("schema_version", self.SCHEMA_VERSION)
                payload.setdefault("prompt", normalized_prompt)
                payload.setdefault("prompt_sha256", prompt_hash)
                payload.setdefault("intent_compiler_spec_version", self.SPEC_VERSION)
                payload["prompt"] = normalized_prompt
                payload["prompt_sha256"] = prompt_hash
                payload["intent_compiler_spec_version"] = self.SPEC_VERSION
                payload["retry_count"] = attempt
                payload = _normalize_side_distribution(payload)
                payload, normalized_centered_above = (
                    _normalize_centered_above_relations(payload, normalized_prompt)
                )
                payload, restored_targets = _restore_missing_target_selectors(
                    payload, normalized_prompt
                )
                payload, restored_fields = _restore_missing_relation_fields(
                    payload, normalized_prompt
                )
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
                    result, normalized_prompt, allowed_inference_reasons
                )
                rejected_task_constraints.extend(enrichment_rejections)
                task_constraint_trace = self._task_compiler_trace_fields(
                    result.get("constraints", []),
                    normalized_interaction,
                    normalized_aesthetic,
                    rejected_task_constraints,
                )
                self.last_trace = {
                    "status": "ok",
                    "spec_version": self.SPEC_VERSION,
                    "prompt_sha256": prompt_hash,
                    "constraints": result.get("constraints", []),
                    "enriched_constraints": enriched_constraints,
                    "normalized_centered_above": normalized_centered_above,
                    "restored_targets": restored_targets,
                    "restored_fields": restored_fields,
                    "retry_count": attempt,
                    "failure_reason": "",
                } | task_constraint_trace
                attempts.append(
                    {
                        "attempt": attempt,
                        "status": "ok",
                        "elapsed_sec": round(time.perf_counter() - started_at, 6),
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
                        "status": "ok",
                        "attempt": attempt,
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
                previous_output = raw
                attempts.append(
                    {
                        "attempt": attempt,
                        "status": "error",
                        "error": last_error,
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
                        error=last_error,
                    ).model_dump()
                    | {
                        "input": messages,
                        "output": raw,
                        "status": "error",
                        "attempt": attempt,
                    }
                )
                logger.warning(
                    "IntentCompiler attempt %d failed: %s", attempt + 1, last_error
                )

        fallback = build_intent_contract(normalized_prompt)
        fallback["retry_count"] = 1
        fallback["warnings"] = [
            "LLM contract unavailable or invalid; deterministic prompt parser used"
        ]
        try:
            result = validate_intent_contract(fallback)
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
                "status": "fallback",
                "spec_version": self.SPEC_VERSION,
                "prompt_sha256": prompt_hash,
                "constraints": result.get("constraints", []),
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
                        interaction_constraints=normalized_interaction,
                        aesthetic_constraints=normalized_aesthetic,
                    ),
                    output=json.dumps(fallback, ensure_ascii=False),
                    error=last_error,
                ).model_dump()
                | {
                    "input": self._messages(
                        normalized_prompt,
                        interaction_constraints=normalized_interaction,
                        aesthetic_constraints=normalized_aesthetic,
                    ),
                    "output": fallback,
                    "status": "fallback",
                    "attempt": 2,
                }
            )
            return result

        self.last_trace = {
            "status": "error",
            "spec_version": self.SPEC_VERSION,
            "prompt_sha256": prompt_hash,
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
