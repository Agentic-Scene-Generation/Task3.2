"""Qwen thinking-mode helpers shared by agent and direct VLM calls."""

from __future__ import annotations

from typing import Any


NO_THINK_VALUES = ("", "none", "minimal", "off", "false", "0", "no_think", "nothink")


def thinking_directive_from_effort(effort: Any) -> str:
    """Map config reasoning effort to a Qwen thinking directive."""
    value = str(effort or "").strip().lower()
    if value in NO_THINK_VALUES:
        return "/no_think"
    return "/think"


def prepend_text_thinking_directive(text: str, directive: str) -> str:
    """Prefix text with exactly one Qwen thinking directive."""
    stripped = text.lstrip()
    for existing in ("/think", "/no_think"):
        if stripped == existing or stripped.startswith(existing + "\n"):
            stripped = stripped[len(existing) :].lstrip()
            break
    return f"{directive}\n{stripped}" if stripped else directive


def responses_api_reasoning_effort(reasoning_effort: Any) -> str:
    """Map no-think style config to the closest OpenAI Responses API effort."""
    value = str(reasoning_effort or "").strip().lower()
    if value in ("", "none", "off", "false", "0", "no_think", "nothink"):
        return "minimal"
    return value
