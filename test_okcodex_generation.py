#!/usr/bin/env python3
"""Test okcodex image generation (not editing) to understand the API."""
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
    BASE_URL = "https://api.okcodex.cn/v1"
    OUTPUT_DIR = Path("/tmp/okcodex_test")
    OUTPUT_DIR.mkdir(exist_ok=True)

    api_key = load_api_key(KEY_PATH)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Test 1: Try image generation
    print("=" * 60)
    print("Test 1: Image Generation")
    print("=" * 60)
    try:
        payload = {
            "model": "gpt-image-1.5",
            "prompt": "A simple red square on white background",
            "n": 1,
            "size": "512x512",
        }
        response = requests.post(
            f"{BASE_URL}/images/generations",
            headers=headers,
            json=payload,
            timeout=120,
        )
        print(f"Status: {response.status_code}")
        if response.ok:
            data = response.json()
            print("Success! Response structure:")
            print(json.dumps({k: str(v)[:100] for k, v in data.items()}, indent=2))

            if "data" in data and len(data["data"]) > 0:
                # Try to save the image
                result = data["data"][0]
                if "url" in result:
                    img_url = result["url"]
                    if img_url.startswith("http"):
                        img_data = requests.get(img_url, timeout=60).content
                    elif img_url.startswith("data:"):
                        _, encoded = img_url.split(",", 1)
                        img_data = base64.b64decode(encoded)
                    else:
                        img_data = base64.b64decode(img_url)

                    out_path = OUTPUT_DIR / "generation_test.png"
                    out_path.write_bytes(img_data)
                    print(f"Saved to: {out_path}")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Failed: {e}")
        import traceback
        traceback.print_exc()

    print()
    print("=" * 60)
    print("Test 2: Check if vision models can do image editing via chat")
    print("=" * 60)

    # Some services implement image editing as a chat with vision + prompt
    # Let's check if gemini-2.5-flash-image or similar can work
    INPUT_IMAGE = Path(
        "/mnt/afs/visitor33/Task3.2/outputs/critic_probe/"
        "grounded_polygon_acp_20260805_023741/critic_on/batch_001/latest-run/"
        "scene_000/room_living_room/scene_renders/furniture/empty_room_context/0_top.png"
    )

    if INPUT_IMAGE.exists():
        with open(INPUT_IMAGE, "rb") as f:
            img_bytes = f.read()
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')

        try:
            # Try Gemini image model with a vision prompt
            payload = {
                "model": "gemini-2.5-flash-image",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Edit this empty room image by adding 2 sofas and a coffee table. Return the edited image."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_b64}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 1000,
            }

            response = requests.post(
                f"{BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=120,
            )
            print(f"Status: {response.status_code}")
            if response.ok:
                data = response.json()
                print("Success! Gemini image model responded")
                print(json.dumps(data, indent=2)[:500])
            else:
                print(f"Error: {response.text[:500]}")
        except Exception as e:
            print(f"Failed: {e}")


if __name__ == "__main__":
    main()
