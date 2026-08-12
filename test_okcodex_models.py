#!/usr/bin/env python3
"""Check what models and endpoints okcodex supports."""
import json
import requests


def load_api_key(key_path: str) -> str:
    """Load API key from JSON file."""
    with open(key_path) as f:
        data = json.load(f)
    return data["OPENAI_API_KEY"]


def main():
    KEY_PATH = "/mnt/afs/visitor33/okcodex.json"
    BASE_URL = "https://api.okcodex.cn/v1"

    api_key = load_api_key(KEY_PATH)
    print(f"API key loaded: {api_key[:20]}...")
    print()

    headers = {
        "Authorization": f"Bearer {api_key}",
    }

    # Try to list models
    print("Testing /v1/models endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/models", headers=headers, timeout=30)
        print(f"Status: {response.status_code}")
        if response.ok:
            data = response.json()
            print(json.dumps(data, indent=2))
            print()
            if "data" in data:
                print(f"Found {len(data['data'])} models:")
                for model in data["data"][:10]:  # Show first 10
                    print(f"  - {model.get('id', model)}")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Failed: {e}")

    print()
    print("=" * 60)
    print()

    # Try chat completion with a simple test
    print("Testing /v1/chat/completions endpoint...")
    try:
        payload = {
            "model": "gpt-5.5",
            "messages": [{"role": "user", "content": "Say 'test passed'"}],
            "max_tokens": 10,
        }
        response = requests.post(
            f"{BASE_URL}/chat/completions",
            headers={**headers, "Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )
        print(f"Status: {response.status_code}")
        if response.ok:
            data = response.json()
            print(json.dumps(data, indent=2))
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Failed: {e}")


if __name__ == "__main__":
    main()
