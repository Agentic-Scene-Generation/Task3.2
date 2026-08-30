#!/usr/bin/env python3
"""Production-path async OpenRouter streaming and persistence smoke test."""

import argparse
import asyncio
import json
import os
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from openai.types.chat import ChatCompletion

from scenesmith.utils.openai import (
    ReasoningPersistenceAsyncOpenAIClient,
    configure_reasoning_persistence,
    reasoning_persistence_context,
)


def load_api_key(path: Path) -> str:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    value = data.get("OPENROUTER_API_KEY") or data.get("OPENAI_API_KEY")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Key file requires OPENROUTER_API_KEY or OPENAI_API_KEY")
    return value.strip()


async def run_test(key_file: Path, db_path: Path) -> None:
    model = "openai/gpt-5.6-luna-pro"
    base_url = "https://openrouter.ai/api/v1"
    os.environ["SCENEEXPERT_REASONING_STREAM"] = "true"
    configure_reasoning_persistence(
        enabled=True,
        provider="openrouter",
        model_id=model,
        base_url=base_url,
    )
    db_path.unlink(missing_ok=True)
    client = ReasoningPersistenceAsyncOpenAIClient(
        api_key=load_api_key(key_file),
        base_url=base_url,
    )
    async with reasoning_persistence_context("async_stream_smoke", db_path):
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "A box has 17 red balls and 29 blue balls. After removing "
                        "8 balls, how many remain? Think carefully, then answer briefly."
                    ),
                }
            ],
            reasoning_effort="high",
            extra_body={"include_reasoning": True},
        )

    if not isinstance(response, ChatCompletion):
        raise AssertionError(f"Expected ChatCompletion, got {type(response)!r}")
    message = response.choices[0].message
    reasoning = getattr(message, "reasoning", None) or getattr(
        message, "reasoning_content", None
    )
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise AssertionError("Streaming response has no reasoning text")
    if not isinstance(message.content, str) or not message.content.strip():
        raise AssertionError("Streaming response has no assembled answer")
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT provider, model, LENGTH(summary) "
            "FROM agent_reasoning_artifacts "
            "WHERE session_id = ? ORDER BY id DESC LIMIT 1",
            ("async_stream_smoke",),
        ).fetchone()
    if row is None or row[0] != "openrouter" or int(row[2] or 0) <= 0:
        raise AssertionError(f"Reasoning artifact was not persisted correctly: {row}")
    print("PASS async OpenRouter streaming")
    print(f"model={response.model}")
    print(f"reasoning_chars={len(reasoning)}")
    print(f"answer_chars={len(message.content)}")
    print(f"persisted_summary_chars={row[2]}")
    print(f"database={db_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--key-file",
        type=Path,
        default=Path("/mnt/afs/visitor33/apikeys/openrouter.json"),
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path(f"/tmp/openrouter_async_stream_{os.getpid()}.db"),
    )
    args = parser.parse_args()
    asyncio.run(run_test(args.key_file, args.database))


if __name__ == "__main__":
    main()
