"""Qwen thinking-mode helpers shared by agent and direct VLM calls."""

from __future__ import annotations

from typing import Any

import re


NO_THINK_VALUES = ("", "none", "minimal", "off", "false", "0", "no_think", "nothink")


def is_qwen38_model(model: Any) -> bool:
    """Return whether a model name identifies the Qwen3.8 family."""
    normalized = re.sub(r"[^a-z0-9]", "", str(model or "").lower())
    return "qwen38" in normalized


def thinking_directive_from_effort(effort: Any, model: Any = None) -> str:
    """Map config reasoning effort to a Qwen thinking directive."""
    if is_qwen38_model(model):
        # Qwen3.8 uses the structured reasoning_effort template argument and
        # does not use Qwen3.6's /think and /no_think prompt switches.
        return ""
    value = str(effort or "").strip().lower()
    if value in NO_THINK_VALUES:
        return "/no_think"
    return "/think"


def chat_template_kwargs_from_effort(
    effort: Any, model: Any = None
) -> dict[str, dict[str, Any]]:
    """Build llama.cpp Qwen template kwargs for the configured effort.

    Qwen3.6 selects thinking with ``enable_thinking``. Qwen3.8 selects its
    reasoning depth with the structured ``reasoning_effort`` template argument.
    The model name determines which contract is sent.
    """
    if is_qwen38_model(model):
        value = str(effort or "").strip().lower()
        if value in NO_THINK_VALUES:
            # Qwen3.8 disables thinking through this structured switch;
            # reasoning_effort="none" is rejected by its template parser.
            return {"chat_template_kwargs": {"enable_thinking": False}}
        if not value:
            return {"chat_template_kwargs": {}}
        return {"chat_template_kwargs": {"reasoning_effort": value}}
    return {
        "chat_template_kwargs": {
            "enable_thinking": thinking_directive_from_effort(effort, model=model)
            == "/think"
        }
    }


def prepend_text_thinking_directive(text: str, directive: str) -> str:
    """Prefix text with exactly one Qwen thinking directive."""
    stripped = text.lstrip()
    for existing in ("/think", "/no_think"):
        if stripped == existing or stripped.startswith(existing + "\n"):
            stripped = stripped[len(existing) :].lstrip()
            break
    if not directive:
        return stripped
    return f"{directive}\n{stripped}" if stripped else directive


def responses_api_reasoning_effort(reasoning_effort: Any) -> str:
    """Map no-think style config to the closest OpenAI Responses API effort."""
    value = str(reasoning_effort or "").strip().lower()
    if value in ("", "none", "off", "false", "0", "no_think", "nothink"):
        return "minimal"
    return value
