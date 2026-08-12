#!/usr/bin/env python3
"""Test different parameter formats for okcodex image editing."""
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
    BASE_URL = "https://api.okcodex.cn"

    api_key = load_api_key(KEY_PATH)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "x-openai-actor-authorization": "local-image-extension",
        "Content-Type": "application/json",
    }

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

    PROMPT = "Add 2 sofas and a coffee table to this empty room layout image. Keep the top-down view and preserve all walls and doors."

    # Try different payload formats
    payloads = [
        # Format 1: images array with image_url
        {
            "model": "gpt-image-1.5",
            "prompt": PROMPT,
            "images": [{"image_url": img_data_url}],
            "n": 1,
        },
        # Format 2: messages with image content (like vision chat)
        {
            "model": "gpt-image-1.5",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT},
                        {"type": "image_url", "image_url": {"url": img_data_url}},
                    ]
                }
            ],
        },
        # Format 3: direct image field
        {
            "model": "gpt-image-1.5",
            "prompt": PROMPT,
            "image": img_data_url,
        },
        # Format 4: image_url at top level
        {
            "model": "gpt-image-1.5",
            "prompt": PROMPT,
            "image_url": img_data_url,
        },
    ]

    for i, payload in enumerate(payloads, 1):
        print("=" * 60)
        print(f"Test {i}: Payload format {i}")
        print("=" * 60)
        print(f"Keys: {list(payload.keys())}")
        print()

        try:
            response = requests.post(
                f"{BASE_URL}/v1/images/edits",
                headers=headers,
                json=payload,
                timeout=120,
            )
            print(f"Status: {response.status_code}")

            if response.ok:
                print("✅ SUCCESS! Image editing works with this format")
                data = response.json()
                print(f"Response keys: {list(data.keys())}")
                print(f"Full response: {json.dumps(data, indent=2)[:500]}...")

                if "data" in data and len(data["data"]) > 0:
                    result = data["data"][0]
                    print(f"Result keys: {list(result.keys())}")

                    output_dir = Path("/tmp/okcodex_test")
                    output_dir.mkdir(exist_ok=True)
                    output_path = output_dir / f"okcodex_edited_format{i}.png"

                    if "url" in result:
                        result_url = result["url"]
                        print(f"Image URL type: {result_url[:50]}...")

                        try:
                            if result_url.startswith("http"):
                                img_data = requests.get(result_url, timeout=60).content
                            elif result_url.startswith("data:"):
                                _, encoded = result_url.split(",", 1)
                                img_data = base64.b64decode(encoded)
                            else:
                                img_data = base64.b64decode(result_url)

                            output_path.write_bytes(img_data)
                            print(f"✅ Saved to: {output_path}")
                            print(f"File size: {output_path.stat().st_size} bytes")
                        except Exception as save_error:
                            print(f"❌ Failed to save image: {save_error}")
                            import traceback
                            traceback.print_exc()
                    elif "b64_json" in result:
                        print("Found b64_json field")
                        img_data = base64.b64decode(result["b64_json"])
                        output_path.write_bytes(img_data)
                        print(f"✅ Saved to: {output_path}")
                        print(f"File size: {output_path.stat().st_size} bytes")
                    else:
                        print(f"❌ No 'url' or 'b64_json' in result. Available keys: {list(result.keys())}")

                    print()
                    print("🎉 WORKING PAYLOAD FORMAT:")
                    print(json.dumps({k: "..." if k in ["image", "images", "messages", "image_url"] else v
                                    for k, v in payload.items()}, indent=2))
                    return  # Stop on first success
                else:
                    print("❌ No data in response")
            else:
                error_text = response.text[:300]
                print(f"❌ Error: {error_text}")
                print()

        except Exception as e:
            print(f"❌ Exception: {e}")
            print()

    print("=" * 60)
    print("All formats failed. Image editing may not be supported.")
    print("=" * 60)


if __name__ == "__main__":
    main()
