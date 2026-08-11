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


_DETERMINISTIC_ENRICHMENT_SOURCES = frozenset({"explicit_prompt", "room_ontology"})
_HIGH_CONFIDENCE_EXPLICIT_RELATIONS = frozenset(
    {
        "required_count",
        "on_top_of",
        "mounted_to_wall",
        "mounted_to_ceiling",
    }
)
_RECOVERABLE_MISSING_TARGET_RELATIONS = frozenset(
    {
        "centered_in_room",
        "corner_of_room",
        "flanking",
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
    when the original prompt contains an explicit bilateral cue.
    """
    for constraint in payload.get("constraints") or []:
        if not isinstance(constraint, dict):
            continue
        if str(constraint.get("relation") or "") != "flanking":
            continue
        evidence = " ".join(
            str(value or "") for value in (constraint.get("evidence_span"), prompt)
        )
        if any(pattern.search(evidence) for pattern in _FLANKING_GROUNDING_PATTERNS):
            continue
        subject = str((constraint.get("subjects") or {}).get("category") or "object")
        raise ValueError(
            "flanking hard intent requires explicit bilateral wording in the "
            f"original prompt for subject {subject!r}"
        )
    return payload


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
    Recovery is deliberately limited to room anchors and flanking: for other
    relations, an incomplete LLM contract must still fall back so it cannot
    silently discard unrelated semantic relations from the prompt.
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
        if (
            spec.target_arity == 0
            or spec.name not in _RECOVERABLE_MISSING_TARGET_RELATIONS
            or has_target
        ):
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
    llm_contract: dict[str, Any], prompt: str
) -> tuple[dict[str, Any], list[dict[str, str]]]:
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
        return llm_contract, added
    enriched = dict(llm_contract)
    enriched["constraints"] = constraints
    warnings = list(enriched.get("warnings") or [])
    warnings.append(
        "LLM contract enriched with deterministic explicit/room-ontology constraints"
    )
    enriched["warnings"] = warnings
    return validate_intent_contract(enriched), added


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
You are the intent_compiler for a 3D indoor scene critic. Extract only hard
functional relations explicitly stated in the original scene prompt. Your
input is the original prompt below; do not request or infer any TaskCompiler
output, inventory object positions, memory, StageBrief, or current scene state.

Registered relations:
{relation_lines}

Target rule is mandatory.  Relations with target arity 1 ({unary_relations})
MUST include a non-null ``targets`` selector with the target category from the
prompt.  ``corner_of_room`` and ``centered_in_room`` use
``{{"category": "room", "count": 1}}`` as that selector.  For example:
``{{"relation": "flanking", "subjects": {{"category": "nightstand", "count": 2}}, "targets": {{"category": "bed", "count": 1}}, "source": "explicit_prompt", "evidence_span": "..."}}``.
Relations with target arity 2 ({binary_relations}) must use a target selector
whose ``secondary_category`` identifies the second target.  Only target arity
0 relations, such as ``required_count``, may omit ``targets``.

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

Target cardinality is strict: a relation with registered target arity 1 must
have exactly one targets selector with count 1 and must leave
secondary_category, secondary_count, and secondary_role empty. Only a relation
whose registered target arity is 2 may use secondary_category and
secondary_role. Never combine two target nouns into one unary relation; emit
separate relation rows when the prompt states separate relations.

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
        previous_output: str = "",
        validation_error: str = "",
    ) -> list[dict[str, str]]:
        user = f"Original scene prompt:\n{prompt}"
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
            if "wall-relative directional relation" in validation_error:
                user += (
                    "\nDo not convert 'X against the wall behind/in front of Y' "
                    "into a directional X-to-Y relation. Keep only the wall "
                    "relation unless a separate clause explicitly gives that "
                    "object-to-object direction."
                )
        return [
            {"role": "system", "content": _system_prompt()},
            {"role": "user", "content": user},
        ]

    def compile(self, prompt: str) -> dict[str, Any]:
        normalized_prompt, prompt_hash = self._prompt_metadata(prompt)
        previous_output = ""
        last_error = ""
        attempts: list[dict[str, Any]] = []

        for attempt in range(2):
            messages = self._messages(
                normalized_prompt,
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
                payload, restored_targets = _restore_missing_target_selectors(
                    payload, normalized_prompt
                )
                payload = _validate_prompt_grounded_relations(
                    payload, normalized_prompt
                )
                result = validate_intent_contract(payload)
                result, enriched_constraints = _enrich_with_deterministic_constraints(
                    result, normalized_prompt
                )
                self.last_trace = {
                    "status": "ok",
                    "spec_version": self.SPEC_VERSION,
                    "prompt_sha256": prompt_hash,
                    "constraints": result.get("constraints", []),
                    "enriched_constraints": enriched_constraints,
                    "restored_targets": restored_targets,
                    "retry_count": attempt,
                    "failure_reason": "",
                }
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
                        "restored_targets": restored_targets,
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
            self.last_trace = {
                "status": "fallback",
                "spec_version": self.SPEC_VERSION,
                "prompt_sha256": prompt_hash,
                "constraints": result.get("constraints", []),
                "retry_count": 1,
                "failure_reason": last_error,
                "attempts": attempts,
            }
            _append_llm_debug(
                build_llm_call_debug_record(
                    stage="intent_compiler",
                    agent_role="intent_compiler",
                    event="deterministic_fallback",
                    prompt=self._messages(normalized_prompt),
                    output=json.dumps(fallback, ensure_ascii=False),
                    error=last_error,
                ).model_dump()
                | {
                    "input": self._messages(normalized_prompt),
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
        }
        raise IntentCompilationError(
            f"IntentCompiler failed after two attempts: {last_error}",
            trace=self.last_trace,
        )
