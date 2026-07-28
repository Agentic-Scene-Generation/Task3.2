#!/usr/bin/env python3
"""Fail-fast smoke test for SceneExpert structured Qwen/vLLM calls."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scenesmith.scene_expert.structured_llm import (  # noqa: E402
    SceneExpertStructuredLLMClient,
    StructuredLLMProfile,
)


class SmokeResponse(BaseModel):
    status: Literal["ok"]
    role: Literal["scene_expert_structured_smoke"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=os.environ.get(
            "SCENEEXPERT_MODEL_ID", "Qwen/Qwen3.5-35B-A3B"
        ),
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"),
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY", "dummy"),
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=float(
            os.environ.get("SCENEEXPERT_STRUCTURED_LLM_SMOKE_TIMEOUT_SECONDS", "45")
        ),
    )
    parser.add_argument(
        "--max-reasoning-chars",
        type=int,
        default=int(os.environ.get("SCENEEXPERT_SMOKE_MAX_REASONING_CHARS", "64")),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile = StructuredLLMProfile(
        thinking_mode="none",
        max_tokens=128,
        retry_max_tokens=256,
        timeout_seconds=args.timeout_seconds,
        temperature=0.0,
        max_attempts=2,
        response_format="json_schema",
    )
    client = SceneExpertStructuredLLMClient(
        model=args.model,
        api_base_url=args.base_url,
        api_key=args.api_key,
        profiles={"structured_smoke": profile},
    )
    result = client.complete(
        role="structured_smoke",
        stage="startup",
        event="structured_output_contract",
        messages=[
            {
                "role": "system",
                "content": (
                    "Return only schema-valid JSON. Do not reason, explain, or use "
                    "markdown."
                ),
            },
            {
                "role": "user",
                "content": (
                    'Return {"status":"ok","role":'
                    '"scene_expert_structured_smoke"}.'
                ),
            },
        ],
        response_model=SmokeResponse,
        profile=profile,
    )
    if not result.success:
        print(
            "SceneExpert structured smoke test failed: "
            f"{result.final_error_kind}: {result.final_error}",
            file=sys.stderr,
        )
        return 2

    max_reasoning = max(
        (attempt.reasoning_chars for attempt in result.attempts), default=0
    )
    if max_reasoning > args.max_reasoning_chars:
        print(
            "SceneExpert structured smoke test failed: no-think response emitted "
            f"{max_reasoning} reasoning characters (limit "
            f"{args.max_reasoning_chars}).",
            file=sys.stderr,
        )
        return 3

    print(
        "SceneExpert structured smoke test passed "
        f"(attempts={len(result.attempts)}, reasoning_chars={max_reasoning})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
