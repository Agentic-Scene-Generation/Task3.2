#!/usr/bin/env python3
"""
OpenRouter API smoke test for GPT-5.6 Luna Pro.

Tests:
1. Models API - check model availability (no cost)
2. Chat completion with tool calling - verify reasoning field (minimal cost)
3. Beta Batch API - verify batch interface (minimal cost)
"""
import json
import os
import sys
import time
from pathlib import Path

import requests


def load_api_key(key_file: Path) -> str:
    """Load OpenRouter API key from JSON file."""
    with open(key_file, "r") as f:
        data = json.load(f)
    key = data.get("OPENROUTER_API_KEY") or data.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("No API key found in JSON file")
    return key


def test_models_api(api_key: str) -> dict:
    """Test 1: Check model availability via models API (free)."""
    print("=" * 70)
    print("TEST 1: Models API - Check GPT-5.6 Luna Pro availability")
    print("=" * 70)

    url = "https://openrouter.ai/api/v1/models"
    headers = {
        "Authorization": f"Bearer {api_key}",
    }

    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    models = response.json()
    target_model = "openai/gpt-5.6-luna-pro"

    found = None
    for model in models.get("data", []):
        if model["id"] == target_model:
            found = model
            break

    if not found:
        print(f"✗ Model '{target_model}' not found")
        return {"success": False, "model": None}

    print(f"✓ Model found: {target_model}")
    print(f"  Context length: {found.get('context_length', 'N/A')}")
    supported_parameters = set(found.get("supported_parameters") or [])
    print(f"  Supports tools: {'tools' in supported_parameters}")
    print(f"  Supports reasoning: {'reasoning' in supported_parameters}")
    print()

    return {"success": True, "model": found}


def test_chat_tool_call(api_key: str) -> dict:
    """Test 2: Chat completion with tool calling and reasoning."""
    print("=" * 70)
    print("TEST 2: Chat Completion - Tool call + Reasoning")
    print("=" * 70)

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Simple tool: get current time
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "Get the current time",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        }
    ]

    payload = {
        "model": "openai/gpt-5.6-luna-pro",
        "messages": [
            {
                "role": "user",
                "content": "What time is it? Use the get_current_time tool.",
            }
        ],
        "tools": tools,
        "reasoning_effort": "high",
        "include_reasoning": True,
    }

    print("Sending request...")
    response = requests.post(url, headers=headers, json=payload, timeout=120)
    response.raise_for_status()

    result = response.json()

    # Check response structure
    choice = result["choices"][0]
    message = choice["message"]

    print(f"✓ Request successful")
    print(f"  Finish reason: {choice.get('finish_reason')}")
    print(f"  Has tool calls: {bool(message.get('tool_calls'))}")

    # Check reasoning field
    reasoning_content = message.get("reasoning") or message.get("reasoning_content")
    reasoning_details = message.get("reasoning_details")

    if reasoning_content:
        print(f"  Reasoning content length: {len(reasoning_content)} chars")
    else:
        print(f"  ⚠ No reasoning_content in response")

    if reasoning_details:
        print("  Reasoning details present: true")
    else:
        print(f"  ⚠ No reasoning_details in response")

    print()

    return {
        "success": True,
        "has_tool_calls": bool(message.get("tool_calls")),
        "has_reasoning_content": bool(reasoning_content),
        "has_reasoning_details": bool(reasoning_details),
        "response": result,
    }


def test_batch_api(
    api_key: str,
    *,
    poll_interval_seconds: float = 10.0,
    poll_timeout_seconds: float = 900.0,
) -> dict:
    """Test 3: Beta Batch API smoke test."""
    print("=" * 70)
    print("TEST 3: Beta Batch API - Text-only smoke test")
    print("=" * 70)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    # Keep endpoint/model before requests: OpenRouter parses this body in order.
    payload = {
        "endpoint": "/v1/chat/completions",
        "model": "openai/gpt-5.6-luna-pro",
        "requests": [
            {
                "custom_id": "promptgen-batch-smoke-1",
                "body": {
                    "model": "openai/gpt-5.6-luna-pro",
                    "messages": [
                        {
                            "role": "user",
                            "content": "Reply with exactly: batch-ok",
                        }
                    ],
                    "max_tokens": 16,
                },
            }
        ],
    }
    created_response = requests.post(
        "https://openrouter.ai/api/beta/batches",
        headers=headers,
        json=payload,
        timeout=60,
    )
    created_response.raise_for_status()
    created = created_response.json()
    batch_id = created.get("id")
    if not batch_id:
        return {"success": False, "error": "Batch response has no id"}

    print(f"Batch submitted: {batch_id}")
    return poll_batch_api(
        api_key,
        batch_id,
        poll_interval_seconds=poll_interval_seconds,
        poll_timeout_seconds=poll_timeout_seconds,
        initial_batch=created,
    )


def poll_batch_api(
    api_key: str,
    batch_id: str,
    *,
    poll_interval_seconds: float = 10.0,
    poll_timeout_seconds: float = 900.0,
    initial_batch: dict | None = None,
) -> dict:
    """Poll an existing Beta Batch without submitting a duplicate request."""
    deadline = time.monotonic() + poll_timeout_seconds
    terminal = {"completed", "failed", "cancelled", "expired"}
    batch = initial_batch or {}
    while batch.get("status") not in terminal:
        if time.monotonic() >= deadline:
            return {
                "success": False,
                "batch_id": batch_id,
                "status": batch.get("status"),
                "error": f"Polling timed out after {poll_timeout_seconds}s",
            }
        time.sleep(poll_interval_seconds)
        poll_response = requests.get(
            f"https://openrouter.ai/api/beta/batches/{batch_id}",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=60,
        )
        poll_response.raise_for_status()
        batch = poll_response.json()
        print(f"  status={batch.get('status')}")

    status = batch.get("status")
    results = batch.get("results") or []
    matching = [
        result
        for result in results
        if result.get("custom_id") == "promptgen-batch-smoke-1"
    ]
    success = status == "completed" and len(matching) == 1
    print(f"Batch terminal status: {status}; results={len(results)}")
    return {
        "success": success,
        "batch_id": batch_id,
        "status": status,
        "result_count": len(results),
        "matched_custom_id": len(matching) == 1,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="OpenRouter API smoke tests for GPT-5.6 Luna Pro"
    )
    parser.add_argument(
        "--key-file",
        type=Path,
        default=Path("/mnt/afs/visitor33/apikeys/openrouter.json"),
        help="Path to OpenRouter API key JSON file",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Confirm paid API test (required for test 2)",
    )
    parser.add_argument(
        "--confirm-batch",
        action="store_true",
        help="Submit and poll one paid text-only Beta Batch request",
    )

    args = parser.parse_args()

    if not args.key_file.exists():
        print(f"ERROR: API key file not found: {args.key_file}", file=sys.stderr)
        sys.exit(1)

    api_key = load_api_key(args.key_file)
    print(f"Loaded API key from: {args.key_file}")
    print()

    results = {}

    # Test 1: Models API (free)
    try:
        results["models_api"] = test_models_api(api_key)
    except Exception as e:
        print(f"✗ Test 1 failed: {e}")
        results["models_api"] = {"success": False, "error": str(e)}

    # Test 2: Chat tool call (paid)
    if args.confirm:
        try:
            results["chat_tool_call"] = test_chat_tool_call(api_key)
        except Exception as e:
            print(f"✗ Test 2 failed: {e}")
            results["chat_tool_call"] = {"success": False, "error": str(e)}
    else:
        print("=" * 70)
        print("TEST 2: SKIPPED - Requires --confirm flag")
        print("=" * 70)
        print("  This test will make a paid API call (~$0.01)")
        print("  Run with --confirm to proceed")
        print()
        results["chat_tool_call"] = {"success": True, "skipped": True}

    # Test 3: Batch API (paid, explicit opt-in)
    if args.confirm_batch:
        try:
            results["batch_api"] = test_batch_api(api_key)
        except Exception as e:
            print(f"✗ Test 3 failed: {e}")
            results["batch_api"] = {"success": False, "error": str(e)}
    else:
        print("TEST 3: SKIPPED - Requires --confirm-batch flag")
        results["batch_api"] = {"success": False, "skipped": True}

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    completed_all = all(
        result.get("success") and not result.get("skipped")
        for result in results.values()
    )
    attempted_success = all(
        result.get("success")
        for result in results.values()
        if not result.get("skipped")
    )

    for test_name, result in results.items():
        status = "✓" if result.get("success") else "✗"
        if result.get("skipped"):
            status = "⊘"
        print(f"{status} {test_name}")

    print()
    if completed_all:
        print("✓ Phase 3 smoke tests complete")
    elif attempted_success:
        print("⊘ Phase 3 smoke tests partially complete; paid checks were skipped")
    else:
        print("✗ Some tests failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
