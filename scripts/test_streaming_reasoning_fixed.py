#!/usr/bin/env python3
"""
Test OpenRouter streaming API for reasoning (fixed version).
"""
import json
import requests
from pathlib import Path


def load_api_key(key_file: Path) -> str:
    with open(key_file, "r") as f:
        data = json.load(f)
    return data.get("OPENROUTER_API_KEY") or data.get("OPENAI_API_KEY")


def test_streaming_reasoning(api_key: str):
    """Test streaming with reasoning."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "openai/gpt-5.6-luna-pro",
        "messages": [
            {"role": "user", "content": "What is 2+2? Think through it step by step."}
        ],
        "reasoning_effort": "xhigh",
        "include_reasoning": True,
        "stream": True,
    }

    print("=" * 70)
    print("Testing STREAMING with reasoning")
    print("=" * 70)

    response = requests.post(url, headers=headers, json=payload, stream=True)
    response.raise_for_status()

    reasoning_chunks = []
    content_chunks = []
    usage_info = None

    for line in response.iter_lines():
        if not line:
            continue
        line_str = line.decode("utf-8")
        if not line_str.startswith("data: "):
            continue
        data_str = line_str[6:]
        if data_str == "[DONE]":
            break

        try:
            chunk = json.loads(data_str)
            choices = chunk.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})

                # Check for reasoning fields
                reasoning = delta.get("reasoning")
                if reasoning is not None:
                    reasoning_chunks.append(reasoning)

                reasoning_content = delta.get("reasoning_content")
                if reasoning_content is not None:
                    reasoning_chunks.append(reasoning_content)

                content = delta.get("content")
                if content is not None:
                    content_chunks.append(content)

            # Check usage in final chunk
            if "usage" in chunk:
                usage_info = chunk["usage"]

        except json.JSONDecodeError:
            continue

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Reasoning chunks collected: {len(reasoning_chunks)}")
    print(f"Content chunks collected: {len(content_chunks)}")

    if reasoning_chunks:
        full_reasoning = "".join(reasoning_chunks)
        print(f"\n✓ REASONING FOUND ({len(full_reasoning)} chars):")
        print(full_reasoning)
    else:
        print("\n⚠ No reasoning chunks found")

    if content_chunks:
        full_content = "".join(content_chunks)
        print(f"\n✓ Content: {full_content}")

    if usage_info:
        print(f"\n✓ Usage info:")
        print(f"  Prompt tokens: {usage_info.get('prompt_tokens')}")
        print(f"  Completion tokens: {usage_info.get('completion_tokens')}")
        details = usage_info.get("completion_tokens_details", {})
        if details.get("reasoning_tokens"):
            print(f"  Reasoning tokens: {details['reasoning_tokens']}")

    return len(reasoning_chunks) > 0


if __name__ == "__main__":
    key_file = Path("/mnt/afs/visitor33/apikeys/openrouter.json")
    api_key = load_api_key(key_file)
    success = test_streaming_reasoning(api_key)

    print("\n" + "=" * 70)
    if success:
        print("✅ SUCCESS: Reasoning IS available via streaming API!")
        print("Solution: Use stream=True to get reasoning from GPT-5.6 Luna Pro")
    else:
        print("❌ FAILED: No reasoning found")
