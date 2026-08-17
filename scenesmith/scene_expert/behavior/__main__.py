"""Command-line entry point for prompt-to-behavior-to-grouped-assets planning."""

from __future__ import annotations

import argparse
import os

from pathlib import Path

from scenesmith.scene_expert.behavior import build_behavior_spec


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a resident, weekly behavior, and room/stage-grouped asset "
            "requirements from a scene prompt."
        )
    )
    parser.add_argument("--prompt", required=True, help="Input scene prompt")
    parser.add_argument(
        "--output",
        type=Path,
        help="Write behavior JSON to this path; omit to print to stdout",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("SCENEEXPERT_MODEL_ID"),
        help="OpenAI-compatible model id; omit for deterministic persona fallback",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"),
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY", "dummy"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--room-type", default=None)
    parser.add_argument("--horizon", default="week", choices=["week"])
    args = parser.parse_args(argv)

    spec = build_behavior_spec(
        args.prompt,
        model=args.model,
        api_base_url=args.base_url,
        api_key=args.api_key,
        room_type=args.room_type,
        horizon=args.horizon,
    )
    payload = spec.model_dump_json(indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
        print(args.output)
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
