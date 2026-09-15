"""Helpers for parsing LLM-produced JSON with compatibility fallbacks."""

import json
from copy import deepcopy

from typing import Any

import json_repair


_LLAMACPP_MODEL_PREFIXES = ("qwen", "llama.cpp", "llama-cpp", "llama")


def is_llamacpp_model(model: str) -> bool:
    """Return whether ``model`` names a local llama.cpp-compatible model."""
    normalized = str(model or "").strip().lower()
    return normalized.startswith(_LLAMACPP_MODEL_PREFIXES)


def _make_openai_strict_schema(schema: Any) -> Any:
    """Normalize a JSON schema for OpenAI strict structured outputs.

    OpenAI requires every property of every object to appear in ``required``.
    Pydantic omits fields with defaults from that list, and llama.cpp accepts
    that looser form, so the normalization is applied only to OpenAI calls.
    """
    if isinstance(schema, list):
        return [_make_openai_strict_schema(item) for item in schema]
    if not isinstance(schema, dict):
        return schema

    normalized = {
        key: _make_openai_strict_schema(value) for key, value in schema.items()
    }
    # OpenAI strict structured outputs support a deliberately small JSON Schema
    # subset. Conditional ``allOf`` branches are enforced later by Pydantic and
    # the deterministic intent validator, so omit them from the wire schema.
    normalized.pop("allOf", None)
    properties = normalized.get("properties")
    if normalized.get("type") == "object" and isinstance(properties, dict):
        normalized["required"] = list(properties)
        normalized["additionalProperties"] = False
    return normalized


def json_response_format(
    *, model: str, name: str, schema: dict[str, Any]
) -> dict[str, Any]:
    """Build a structured-output request for OpenAI or llama.cpp endpoints."""
    request_schema = (
        schema
        if is_llamacpp_model(model)
        else _make_openai_strict_schema(deepcopy(schema))
    )
    payload: dict[str, Any] = {
        "name": name,
        "schema": request_schema,
    }
    # llama.cpp accepts the JSON Schema grammar but does not consistently
    # implement OpenAI's strict-mode validation keyword.
    if not is_llamacpp_model(model):
        payload["strict"] = True
    return {"type": "json_schema", "json_schema": payload}


def extract_json_text(text: str) -> str:
    """Strip common Markdown fencing and isolate the outer JSON payload."""
    stripped = str(text or "").strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    object_start = stripped.find("{")
    object_end = stripped.rfind("}")
    if object_start >= 0 and object_end >= object_start:
        return stripped[object_start : object_end + 1]

    array_start = stripped.find("[")
    array_end = stripped.rfind("]")
    if array_start >= 0 and array_end >= array_start:
        return stripped[array_start : array_end + 1]

    return stripped


def parse_llm_json(text: str) -> Any:
    """Parse LLM JSON output with repair fallback for formatting drift."""
    cleaned = extract_json_text(text)
    parsed = json_repair.loads(cleaned)

    # Some local models return a JSON-encoded string instead of the requested
    # top-level object. Unwrap a few times before giving up.
    for _ in range(3):
        if not isinstance(parsed, str):
            break
        nested = extract_json_text(parsed)
        reparsed = json_repair.loads(nested)
        if reparsed == parsed:
            break
        parsed = reparsed

    return parsed


def parse_llm_json_object(text: str) -> dict[str, Any]:
    """Parse LLM JSON output and require a top-level object."""
    parsed = parse_llm_json(text)
    if not isinstance(parsed, dict):
        raise ValueError(
            f"Expected top-level JSON object but got {type(parsed).__name__}"
        )
    return parsed


def preview_llm_json(text: str, limit: int = 200) -> str:
    """Return a short preview after applying common JSON cleanup."""
    cleaned = extract_json_text(text)
    return repr(cleaned[:limit]) if cleaned else "empty"


def dumps_llm_json(value: Any) -> str:
    """Serialize a value as JSON."""
    return json.dumps(value)
