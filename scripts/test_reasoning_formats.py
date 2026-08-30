#!/usr/bin/env python3
"""
Test different reasoning request formats for OpenRouter.
"""
import json
import requests
from pathlib import Path


def load_api_key(key_file: Path) -> str:
    with open(key_file, "r") as f:
        data = json.load(f)
    return data.get("OPENROUTER_API_KEY") or data.get("OPENAI_API_KEY")


def test_reasoning_formats(api_key: str):
    """Try different reasoning request formats."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    formats = [
        {
            "name": "Format 1: reasoning object with effort",
            "payload": {
                "model": "openai/gpt-5.6-luna-pro",
                "messages": [{"role": "user", "content": "Count to 3"}],
                "reasoning": {"effort": "high"},
            },
        },
        {
            "name": "Format 2: top-level reasoning_effort",
            "payload": {
                "model": "openai/gpt-5.6-luna-pro",
                "messages": [{"role": "user", "content": "Count to 3"}],
                "reasoning_effort": "high",
            },
        },
        {
            "name": "Format 3: no reasoning param (baseline)",
            "payload": {
                "model": "openai/gpt-5.6-luna-pro",
                "messages": [{"role": "user", "content": "Count to 3"}],
            },
        },
    ]

    for fmt in formats:
        print("=" * 70)
        print(fmt["name"])
        print("=" * 70)

        try:
            response = requests.post(url, headers=headers, json=fmt["payload"])
            response.raise_for_status()
            result = response.json()

            choice = result["choices"][0]
            message = choice["message"]

            print(f"✓ Request successful")
            print(f"  Finish reason: {choice.get('finish_reason')}")
            print(f"  Content: {message.get('content', '')[:100]}")

            # Check all possible reasoning fields
            reasoning_fields = {
                "reasoning_content": message.get("reasoning_content"),
                "reasoning": message.get("reasoning"),
                "reasoning_details": choice.get("reasoning_details"),
                "thinking": message.get("thinking"),
            }

            found_any = False
            for field, value in reasoning_fields.items():
                if value:
                    found_any = True
                    if isinstance(value, str):
                        print(f"  ✓ {field}: {len(value)} chars")
                        print(f"    Preview: {value[:100]}")
                    else:
                        print(f"  ✓ {field}: {value}")

            if not found_any:
                print(f"  ⚠ No reasoning fields found")

        except Exception as e:
            print(f"✗ Request failed: {e}")

        print()


if __name__ == "__main__":
    key_file = Path("/mnt/afs/visitor33/apikeys/openrouter.json")
    api_key = load_api_key(key_file)
    test_reasoning_formats(api_key)
