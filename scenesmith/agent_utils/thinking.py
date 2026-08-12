"""Provider-aware reasoning helpers shared by agent and direct VLM calls."""

from __future__ import annotations

import json
import os
from typing import Any


NO_THINK_VALUES = ("", "none", "minimal", "off", "false", "0", "no_think", "nothink")


def use_responses_api() -> bool:
    """Return whether the active OpenAI-compatible provider requires Responses."""
    return os.environ.get("OPENAI_USE_RESPONSES", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def openai_default_headers() -> dict[str, str] | None:
    """Read optional provider headers without putting credentials in source code."""
    raw = os.environ.get("SCENEEXPERT_OPENAI_DEFAULT_HEADERS_JSON", "").strip()
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("SCENEEXPERT_OPENAI_DEFAULT_HEADERS_JSON must be JSON") from exc
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in value.items()
    ):
        raise ValueError("SCENEEXPERT_OPENAI_DEFAULT_HEADERS_JSON must map strings to strings")
    return value


def thinking_directive_from_effort(effort: Any) -> str:
    """Map config reasoning effort to a Qwen directive when using Chat API."""
    if use_responses_api():
        return ""
    value = str(effort or "").strip().lower()
    if value in NO_THINK_VALUES:
        return "/no_think"
    return "/think"


def responses_api_reasoning_effort(reasoning_effort: Any) -> str:
    """Map no-think style config to the closest OpenAI Responses effort."""
    override = os.environ.get("SCENEEXPERT_FORCE_REASONING_EFFORT", "").strip()
    if override:
        return override
    value = str(reasoning_effort or "").strip().lower()
    if value in ("", "none", "off", "false", "0", "no_think", "nothink"):
        return "minimal"
    return value


def chat_template_kwargs_from_effort(effort: Any) -> dict[str, Any]:
    """Build provider-appropriate reasoning fields for a model request."""
    if use_responses_api():
        return {"reasoning": {"effort": responses_api_reasoning_effort(effort)}}
    return {
        "chat_template_kwargs": {
            "enable_thinking": thinking_directive_from_effort(effort) == "/think"
        }
    }


def prepend_text_thinking_directive(text: str, directive: str) -> str:
    """Prefix text with exactly one Qwen thinking directive when applicable."""
    stripped = text.lstrip()
    for existing in ("/think", "/no_think"):
        if stripped == existing or stripped.startswith(existing + "\n"):
            stripped = stripped[len(existing) :].lstrip()
            break
    if not directive:
        return stripped
    return f"{directive}\n{stripped}" if stripped else directive
