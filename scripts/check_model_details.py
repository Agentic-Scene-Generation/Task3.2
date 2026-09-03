#!/usr/bin/env python3
"""
Query OpenRouter model details to check reasoning support.
"""
import json
import requests
from pathlib import Path


def load_api_key(key_file: Path) -> str:
    with open(key_file, "r") as f:
        data = json.load(f)
    return data.get("OPENROUTER_API_KEY") or data.get("OPENAI_API_KEY")


def check_model_details(api_key: str, model_id: str):
    """Get detailed model information from OpenRouter."""
    url = "https://openrouter.ai/api/v1/models"
    headers = {"Authorization": f"Bearer {api_key}"}

    response = requests.get(url, headers=headers)
    response.raise_for_status()

    models = response.json().get("data", [])

    target = next((m for m in models if m["id"] == model_id), None)

    if not target:
        print(f"Model {model_id} not found")
        return

    print(f"Model: {target['id']}")
    print(f"Name: {target.get('name', 'N/A')}")
    print(f"Context length: {target.get('context_length', 'N/A')}")
    print(f"\nFull model info:")
    print(json.dumps(target, indent=2))


if __name__ == "__main__":
    key_file = Path("/mnt/afs/visitor33/apikeys/openrouter.json")
    api_key = load_api_key(key_file)

    print("=" * 70)
    print("GPT-5.6 Luna Pro")
    print("=" * 70)
    check_model_details(api_key, "openai/gpt-5.6-luna-pro")

    print("\n" + "=" * 70)
    print("GPT-5.2 (for comparison)")
    print("=" * 70)
    check_model_details(api_key, "openai/gpt-5.2")
