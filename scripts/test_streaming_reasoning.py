#!/usr/bin/env python3
"""
Test OpenRouter streaming API to check if reasoning is available in stream chunks.
According to the OpenRouter SDK example, reasoning tokens might appear in streaming.
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
                if "reasoning" in delta:
                    reasoning_chunks.append(delta["reasoning"])
                    print(f"[REASONING CHUNK]: {delta['reasoning'][:100]}")

                if "reasoning_content" in delta:
                    reasoning_chunks.append(delta["reasoning_content"])
                    print(f"[REASONING_CONTENT CHUNK]: {delta['reasoning_content'][:100]}")

                if "content" in delta:
                    content_chunks.append(delta["content"])

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
        print(f"\n✓ Reasoning found ({len(full_reasoning)} chars):")
        print(full_reasoning[:500])
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


if __name__ == "__main__":
    key_file = Path("/mnt/afs/visitor33/apikeys/openrouter.json")
    api_key = load_api_key(key_file)
    test_streaming_reasoning(api_key)
