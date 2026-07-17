"""Helpers for parsing LLM-produced JSON with compatibility fallbacks."""

import json

from typing import Any

import json_repair


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
