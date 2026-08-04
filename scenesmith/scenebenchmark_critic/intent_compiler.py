"""Independent LLM compiler for hard SceneBenchmark intent contracts."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from scenesmith.agent_utils.thinking import chat_template_kwargs_from_effort
from scenesmith.scene_expert.context_bundle import build_llm_call_debug_record
from scenesmith.scenebenchmark_critic.intent_schema import (
    INTENT_COMPILER_SPEC_VERSION,
    INTENT_CONTRACT_SCHEMA_VERSION,
    intent_contract_json_schema,
    validate_intent_contract,
)
from scenesmith.scenebenchmark_critic.relation_registry import RELATION_REGISTRY
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


def _system_prompt() -> str:
    relation_lines = "\n".join(
        f"- {name}: {spec.prompt_description}"
        for name, spec in sorted(RELATION_REGISTRY.items())
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
                result = validate_intent_contract(payload)
                self.last_trace = {
                    "status": "ok",
                    "spec_version": self.SPEC_VERSION,
                    "prompt_sha256": prompt_hash,
                    "constraints": result.get("constraints", []),
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
