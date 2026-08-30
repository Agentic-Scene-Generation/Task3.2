#!/usr/bin/env python3
"""
Test OpenRouter reasoning with include_reasoning parameter.
"""
import json
import requests
from pathlib import Path


def load_api_key(key_file: Path) -> str:
    with open(key_file, "r") as f:
        data = json.load(f)
    return data.get("OPENROUTER_API_KEY") or data.get("OPENAI_API_KEY")


def test_with_include_reasoning(api_key: str):
    """Test with include_reasoning parameter."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Tool for testing
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_time",
                "description": "Get current time",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }
    ]

    payload = {
        "model": "openai/gpt-5.6-luna-pro",
        "messages": [
            {
                "role": "user",
                "content": "What time is it? Use the get_time tool.",
            }
        ],
        "tools": tools,
        "reasoning_effort": "xhigh",
        "include_reasoning": True,  # ← Key parameter
    }

    print("=" * 70)
    print("Testing with include_reasoning=True + reasoning_effort=xhigh")
    print("=" * 70)

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    result = response.json()

    choice = result["choices"][0]
    message = choice["message"]

    print(f"✓ Request successful")
    print(f"  Finish reason: {choice.get('finish_reason')}")
    print(f"  Has tool calls: {bool(message.get('tool_calls'))}")

    # Check all reasoning fields
    reasoning_fields = {
        "reasoning_content": message.get("reasoning_content"),
        "reasoning": message.get("reasoning"),
        "thinking": message.get("thinking"),
    }

    found = False
    for field, value in reasoning_fields.items():
        if value:
            found = True
            if isinstance(value, str):
                print(f"  ✓ {field}: {len(value)} chars")
                print(f"    Preview: {value[:200]}")
            else:
                print(f"  ✓ {field}: {json.dumps(value)[:200]}")

    if not found:
        print(f"  ⚠ No reasoning fields in message")
        print(f"\n  Full response keys: {list(result.keys())}")
        print(f"  Choice keys: {list(choice.keys())}")
        print(f"  Message keys: {list(message.keys())}")

    # Check usage
    usage = result.get("usage", {})
    if usage:
        print(f"\n  Usage:")
        print(f"    Prompt tokens: {usage.get('prompt_tokens')}")
        print(f"    Completion tokens: {usage.get('completion_tokens')}")
        reasoning_tokens = usage.get("completion_tokens_details", {}).get(
            "reasoning_tokens"
        )
        if reasoning_tokens:
            print(f"    Reasoning tokens: {reasoning_tokens}")

    return result


if __name__ == "__main__":
    key_file = Path("/mnt/afs/visitor33/apikeys/openrouter.json")
    api_key = load_api_key(key_file)
    result = test_with_include_reasoning(api_key)

    # Save full response for inspection
    with open("/tmp/openrouter_reasoning_test.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n✓ Full response saved to /tmp/openrouter_reasoning_test.json")
