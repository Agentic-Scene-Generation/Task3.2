#!/usr/bin/env python3
"""Test if okcodex supports vision + text-to-image workflow."""
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

    api_key = load_api_key(KEY_PATH)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Test gpt-5.5 with vision (analyze the room layout)
    print("=" * 60)
    print("Test: gpt-5.5 with vision (room analysis)")
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

    try:
        payload = {
            "model": "gpt-5.5",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this room layout image in 2 sentences."
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
            "max_tokens": 100,
        }

        response = requests.post(
            f"{BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60,
        )
        print(f"Status: {response.status_code}")
        if response.ok:
            data = response.json()
            print("SUCCESS! GPT-5.5 supports vision.")
            print()
            if "choices" in data and len(data["choices"]) > 0:
                content = data["choices"][0].get("message", {}).get("content", "")
                print(f"Response: {content}")
            print()
            print(json.dumps(data, indent=2)[:1000])
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Failed: {e}")
        import traceback
        traceback.print_exc()

    print()
    print("=" * 60)
    print("Conclusion:")
    print("=" * 60)
    print("If GPT-5.5 vision works, then okcodex does NOT support")
    print("the /v1/images/edits endpoint for image editing.")
    print()
    print("Alternative approach:")
    print("- Use a proper image editing service (Stability AI, Replicate)")
    print("- Use local Qwen-Image-Edit (current setup)")
    print("- Use vision model to generate detailed placement instructions")
    print("  then render programmatically (no image editing)")


if __name__ == "__main__":
    main()
