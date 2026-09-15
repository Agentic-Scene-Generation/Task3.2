#!/usr/bin/env python3
"""Fail-fast OpenRouter runtime check for tool calls and reasoning artifacts."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import requests


def _load_api_key(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    value = data.get("OPENROUTER_API_KEY") or data.get("OPENAI_API_KEY")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("key file requires OPENROUTER_API_KEY or OPENAI_API_KEY")
    return value.strip()


def _reasoning_text(message: dict[str, Any]) -> str:
    value = message.get("reasoning") or message.get("reasoning_content")
    if isinstance(value, str):
        return value
    details = message.get("reasoning_details")
    if isinstance(details, list):
        return "\n".join(
            str(item.get("text") or item.get("summary") or "")
            for item in details
            if isinstance(item, dict)
        )
    return ""


def _provider_slugs(configured: str) -> list[str]:
    """Parse an optional provider allowlist; an empty list means auto-routing."""
    return [value.strip() for value in configured.split(",") if value.strip()]


def _apply_provider_routing(payload: dict[str, Any], configured: str) -> dict[str, Any]:
    """Add OpenRouter's provider constraint only when explicitly requested."""
    providers = _provider_slugs(configured)
    if providers:
        payload["provider"] = {"only": providers, "allow_fallbacks": True}
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--provider-only",
        default="",
        help="Optional comma-separated provider allowlist; empty uses auto-routing",
    )
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--retry-delay-seconds", type=float, default=2.0)
    args = parser.parse_args()

    if args.attempts < 1:
        parser.error("--attempts must be at least 1")

    payload = _apply_provider_routing(
        {
            "model": args.model,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "A room contains 17 chairs and stools with 52 legs total. "
                        "Chairs have four legs and stools have three. Solve and "
                        "verify the two counts, then call submit_counts."
                    ),
                }
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "submit_counts",
                        "description": "Submit the verified furniture counts.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "chairs": {"type": "integer"},
                                "stools": {"type": "integer"},
                            },
                            "required": ["chairs", "stools"],
                            "additionalProperties": False,
                        },
                    },
                }
            ],
            "tool_choice": {
                "type": "function",
                "function": {"name": "submit_counts"},
            },
            "include_reasoning": True,
            "reasoning": {"effort": args.reasoning_effort},
            "max_completion_tokens": 1024,
        },
        args.provider_only,
    )
    requested_routing = (
        f"allowlist:{','.join(_provider_slugs(args.provider_only))}"
        if _provider_slugs(args.provider_only)
        else "automatic"
    )
    url = f"{args.base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {_load_api_key(args.key_file)}",
        "Content-Type": "application/json",
    }
    failures: list[str] = []
    for attempt in range(1, args.attempts + 1):
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=args.timeout_seconds,
            )
            response.raise_for_status()
            body = response.json()
        except requests.RequestException as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            error: Any = str(exc)
            if getattr(exc, "response", None) is not None:
                try:
                    error_body = exc.response.json()
                    error = error_body.get("error", error_body)
                except ValueError:
                    error = exc.response.text[:500]
            failure = (
                f"attempt={attempt}/{args.attempts} "
                f"http_status={status or 'unavailable'} error={str(error)[:500]}"
            )
            failures.append(failure)
            print(f"[WARN] OpenRouter preflight {failure}", file=sys.stderr, flush=True)
        except (ValueError, TypeError) as exc:
            failure = (
                f"attempt={attempt}/{args.attempts} invalid_response={str(exc)[:500]}"
            )
            failures.append(failure)
            print(f"[WARN] OpenRouter preflight {failure}", file=sys.stderr, flush=True)
        else:
            choices = body.get("choices") or []
            choice = choices[0] if choices else {}
            message = choice.get("message", {})
            tool_calls = message.get("tool_calls") or []
            reasoning = _reasoning_text(message)
            usage = body.get("usage") or {}
            metadata = (
                f"attempt={attempt}/{args.attempts} "
                f"request_id={body.get('id', 'unknown')} "
                f"requested_routing={requested_routing} "
                f"provider={body.get('provider', 'unknown')} "
                f"finish_reason={choice.get('finish_reason', 'unknown')} "
                f"tool_calls={len(tool_calls)} reasoning_chars={len(reasoning)} "
                f"reasoning_tokens={((usage.get('completion_tokens_details') or {}).get('reasoning_tokens', 'unknown'))} "
                f"total_tokens={usage.get('total_tokens', 'unknown')}"
            )
            if tool_calls and reasoning.strip():
                print(f"[OK] OpenRouter runtime preflight: {metadata}", flush=True)
                return 0
            response_shape = (
                f" response_keys={sorted(str(key) for key in body.keys())}"
                f" response_error={str(body.get('error'))[:300]}"
            )
            failure = f"{metadata}{response_shape} missing=" + (
                "tool_call_and_reasoning"
                if not tool_calls and not reasoning.strip()
                else ("tool_call" if not tool_calls else "reasoning")
            )
            failures.append(failure)
            print(f"[WARN] OpenRouter preflight {failure}", file=sys.stderr, flush=True)

        if attempt < args.attempts:
            time.sleep(max(0.0, args.retry_delay_seconds))

    diagnostic = "OpenRouter runtime preflight exhausted all attempts; " + " | ".join(
        failures
    )
    print(f"[ERROR] {diagnostic}", file=sys.stderr, flush=True)
    raise RuntimeError(
        diagnostic + "; refusing a synthesis run that requires reasoning artifacts"
    )


if __name__ == "__main__":
    raise SystemExit(main())
