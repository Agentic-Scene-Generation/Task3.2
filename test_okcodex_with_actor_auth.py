#!/usr/bin/env python3
"""Test okcodex with the special x-openai-actor-authorization header."""
import json
import requests
import base64
from pathlib import Path


def load_api_key(key_path: str) -> str:
    with open(key_path) as f:
        data = json.load(f)
    return data["OPENAI_API_KEY"]


def main():
    KEY_PATH = "/mnt/afs/visitor33/okcodex.json"
    BASE_URL = "https://api.okcodex.cn"  # No /v1 suffix

    api_key = load_api_key(KEY_PATH)

    # Special headers from the config
    headers = {
        "Authorization": f"Bearer {api_key}",
        "x-openai-actor-authorization": "local-image-extension",
        "Content-Type": "application/json",
    }

    print("=" * 60)
    print("Test 1: List models with special headers")
    print("=" * 60)

    try:
        response = requests.get(f"{BASE_URL}/v1/models", headers=headers, timeout=30)
        print(f"Status: {response.status_code}")
        if response.ok:
            data = response.json()
            print(f"Found {len(data.get('data', []))} models")
            # Check for image-related models
            image_models = [m for m in data.get('data', []) if 'image' in m.get('id', '').lower()]
            print(f"Image-related models: {[m['id'] for m in image_models]}")
        else:
            print(f"Error: {response.text[:500]}")
    except Exception as e:
        print(f"Failed: {e}")

    print()
    print("=" * 60)
    print("Test 2: Chat completion with gpt-5.5")
    print("=" * 60)

    try:
        payload = {
            "model": "gpt-5.5",
            "messages": [{"role": "user", "content": "Say 'test passed'"}],
            "max_tokens": 10,
        }
        response = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30,
        )
        print(f"Status: {response.status_code}")
        if response.ok:
            data = response.json()
            print("SUCCESS!")
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            print(f"Response: {content}")
        else:
            print(f"Error: {response.text[:500]}")
    except Exception as e:
        print(f"Failed: {e}")

    print()
    print("=" * 60)
    print("Test 3: Image generation with gpt-image-1.5")
    print("=" * 60)

    try:
        payload = {
            "model": "gpt-image-1.5",
            "prompt": "A simple red square on white background",
            "n": 1,
            "size": "512x512",
        }
        response = requests.post(
            f"{BASE_URL}/v1/images/generations",
            headers=headers,
            json=payload,
            timeout=120,
        )
        print(f"Status: {response.status_code}")
        if response.ok:
            print("SUCCESS! Image generation works")
            data = response.json()
            print(f"Response keys: {list(data.keys())}")
            if "data" in data and len(data["data"]) > 0:
                result = data["data"][0]
                print(f"Result keys: {list(result.keys())}")
                if "url" in result:
                    print(f"URL: {result['url'][:100]}...")
        else:
            print(f"Error: {response.text[:500]}")
    except Exception as e:
        print(f"Failed: {e}")

    print()
    print("=" * 60)
    print("Test 4: Image editing with gpt-image-1.5")
    print("=" * 60)

    INPUT_IMAGE = Path(
        "/mnt/afs/visitor33/Task3.2/outputs/critic_probe/"
        "grounded_polygon_acp_20260805_023741/critic_on/batch_001/latest-run/"
        "scene_000/room_living_room/scene_renders/furniture/empty_room_context/0_top.png"
    )

    if not INPUT_IMAGE.exists():
        print(f"Input image not found: {INPUT_IMAGE}")
        return

    with open(INPUT_IMAGE, "rb") as f:
        img_bytes = f.read()
    img_b64 = base64.b64encode(img_bytes).decode('utf-8')
    img_data_url = f"data:image/png;base64,{img_b64}"

    try:
        payload = {
            "model": "gpt-image-1.5",
            "prompt": "Add 2 sofas and a coffee table to this empty room layout",
            "image": img_data_url,
            "n": 1,
        }
        response = requests.post(
            f"{BASE_URL}/v1/images/edits",
            headers=headers,
            json=payload,
            timeout=120,
        )
        print(f"Status: {response.status_code}")
        if response.ok:
            print("SUCCESS! Image editing works")
            data = response.json()
            print(f"Response keys: {list(data.keys())}")

            if "data" in data and len(data["data"]) > 0:
                result = data["data"][0]
                output_dir = Path("/tmp/okcodex_test")
                output_dir.mkdir(exist_ok=True)
                output_path = output_dir / "okcodex_edited_with_auth.png"

                if "url" in result:
                    result_url = result["url"]
                    if result_url.startswith("http"):
                        img_data = requests.get(result_url, timeout=60).content
                    elif result_url.startswith("data:"):
                        _, encoded = result_url.split(",", 1)
                        img_data = base64.b64decode(encoded)
                    else:
                        img_data = base64.b64decode(result_url)

                    output_path.write_bytes(img_data)
                    print(f"Saved to: {output_path}")
        else:
            print(f"Error: {response.text[:500]}")
    except Exception as e:
        print(f"Failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
