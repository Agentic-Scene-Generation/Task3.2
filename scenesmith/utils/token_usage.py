"""Normalize OpenAI-compatible and Agents SDK token-usage payloads."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _value(source: Any, *names: str) -> Any:
    for name in names:
        value = (
            source.get(name)
            if isinstance(source, Mapping)
            else getattr(source, name, None)
        )
        if value is not None:
            return value
    return None


def _integer(value: Any) -> int | None:
    return (
        int(value) if isinstance(value, int) and not isinstance(value, bool) else None
    )


def normalize_token_usage(value: Any) -> dict[str, int]:
    """Return stable audit keys from SDK runs, provider responses, or dicts.

    The input is intentionally duck-typed because the Agents SDK and
    OpenAI-compatible servers use different usage object shapes. Missing values
    are omitted so audit consumers can distinguish unavailable data from zero.
    """
    context_wrapper = _value(value, "context_wrapper")
    usage = _value(context_wrapper, "usage") if context_wrapper is not None else None
    usage = usage if usage is not None else _value(value, "usage")
    if usage is None:
        usage = value

    input_tokens = _integer(_value(usage, "input_tokens", "prompt_tokens"))
    output_tokens = _integer(_value(usage, "output_tokens", "completion_tokens"))
    total_tokens = _integer(_value(usage, "total_tokens"))
    requests = _integer(_value(usage, "requests"))
    final_context_tokens = _integer(_value(usage, "final_input_context_tokens"))
    max_context_tokens = _integer(_value(usage, "max_input_context_tokens"))
    input_details = _value(usage, "input_tokens_details", "prompt_tokens_details")
    output_details = _value(usage, "output_tokens_details", "completion_tokens_details")
    cached_tokens = _integer(_value(input_details, "cached_tokens"))
    if cached_tokens is None:
        cached_tokens = _integer(_value(usage, "input_cached_tokens", "cached_tokens"))
    reasoning_tokens = _integer(_value(output_details, "reasoning_tokens"))
    if reasoning_tokens is None:
        reasoning_tokens = _integer(
            _value(usage, "output_reasoning_tokens", "reasoning_tokens")
        )

    normalized = {
        key: token_count
        for key, token_count in (
            ("input_tokens", input_tokens),
            ("input_cached_tokens", cached_tokens),
            ("output_tokens", output_tokens),
            ("output_reasoning_tokens", reasoning_tokens),
            ("total_tokens", total_tokens),
            ("requests", requests),
            ("final_input_context_tokens", final_context_tokens),
            ("max_input_context_tokens", max_context_tokens),
        )
        if token_count is not None
    }
    if input_tokens is not None and cached_tokens is not None:
        normalized["input_non_cached_tokens"] = max(input_tokens - cached_tokens, 0)
    if output_tokens is not None and reasoning_tokens is not None:
        normalized["output_text_tokens"] = max(output_tokens - reasoning_tokens, 0)

    request_entries = _value(usage, "request_usage_entries")
    if isinstance(request_entries, (list, tuple)):
        context_tokens = [
            token_count
            for entry in request_entries
            if (token_count := _integer(_value(entry, "input_tokens", "prompt_tokens")))
            is not None
        ]
        if context_tokens:
            normalized["final_input_context_tokens"] = context_tokens[-1]
            normalized["max_input_context_tokens"] = max(context_tokens)
    return normalized
