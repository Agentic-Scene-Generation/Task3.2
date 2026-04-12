"""Lightweight LLM utilities for robot_eval module.

Provides simple async structured LLM calls without Agent SDK overhead.
Uses OpenAI's structured output API for guaranteed schema compliance.
"""

import logging

from typing import TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel

console_logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Lazy-initialized client.
_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    """Get or create the async OpenAI client."""
    global _client
    if _client is None:
        _client = AsyncOpenAI()
    return _client


async def structured_llm_call(
    model: str, system_prompt: str, user_input: str, output_type: type[T]
) -> T:
    """Make an async LLM call with structured Pydantic output.

    Uses OpenAI's structured output API which constrains the LLM to only
    produce valid schema-compliant output. Simpler than Agent SDK for
    single-turn LLM calls without tools.

    Args:
        model: Model name (e.g., "gpt-5.2").
        system_prompt: System instructions for the LLM.
        user_input: User message/query.
        output_type: Pydantic model class for structured output.

    Returns:
        Parsed Pydantic model instance.

    Raises:
        OpenAI API errors if the call fails.
    """
    client = _get_client()

    response = await client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
        response_format=output_type,
    )

    parsed = response.choices[0].message.parsed
    if parsed is None:
        raise RuntimeError(
            f"Failed to parse structured output. "
            f"Refusal: {response.choices[0].message.refusal}"
        )

    return parsed
