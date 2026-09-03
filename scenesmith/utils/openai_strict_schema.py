"""Compatibility helpers for the OpenAI strict structured-output schema subset."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def make_openai_strict_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a schema compatible with OpenAI ``json_schema.strict`` mode.

    Regular JSON Schema permits object properties that are not required. The
    strict structured-output subset requires every property to appear in that
    object's ``required`` list. Pydantic represents semantically nullable
    fields with an ``anyOf`` containing ``null``; this function preserves that
    type while making the key required. A field with a concrete default is
    likewise required in the wire response, so validation receives an explicit
    value instead of silently applying that default.
    """

    normalized = deepcopy(schema)

    def _normalize(node: Any) -> None:
        if isinstance(node, dict):
            properties = node.get("properties")
            if isinstance(properties, dict):
                node["required"] = list(properties)
            for value in node.values():
                _normalize(value)
        elif isinstance(node, list):
            for value in node:
                _normalize(value)

    _normalize(normalized)
    return normalized
