#!/usr/bin/env python3
"""
Test script for okcodex image editing API.
Validates connectivity and basic functionality before switching SceneSmith config.
"""
import json
import sys
from pathlib import Path

from openai import OpenAI


def load_api_key(key_path: str) -> str:
    """Load API key from JSON file."""
    with open(key_path) as f:
        data = json.load(f)
    return data["OPENAI_API_KEY"]


def test_okcodex_image_edit(
    api_key: str,
    base_url: str,
    model: str,
    input_image_path: Path,
    prompt: str,
    output_path: Path,
):
    """Test okcodex image editing endpoint."""
    print(f"Testing okcodex API:")
    print(f"  base_url: {base_url}")
    print(f"  model: {model}")
    print(f"  input_image: {input_image_path}")
    print(f"  output: {output_path}")
    print()

    # Initialize client
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=120.0,
    )

    # Read input image and convert to base64 data URL
    import base64
    import mimetypes

    with open(input_image_path, "rb") as f:
        image_bytes = f.read()

    # Detect MIME type
    mime_type, _ = mimetypes.guess_type(str(input_image_path))
    if mime_type is None:
        mime_type = "image/png"  # Default to PNG

    # Create base64 data URL
    image_b64 = base64.b64encode(image_bytes).decode('utf-8')
    image_data_url = f"data:{mime_type};base64,{image_b64}"

    print(f"Input image size: {len(image_bytes)} bytes")
    print(f"MIME type: {mime_type}")
    print(f"Data URL length: {len(image_data_url)} chars")
    print(f"Prompt: {prompt[:200]}...")
    print()
    print("Sending request...")

    try:
        # Call image edit endpoint with data URL
        # Note: OpenAI Python SDK might not support this directly,
        # so we'll use the raw API
        import requests

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model,
            "prompt": prompt,
            "image": image_data_url,
            "n": 1,
        }

        api_response = requests.post(
            f"{base_url}/images/edits",
            headers=headers,
            json=payload,
            timeout=120,
        )
        api_response.raise_for_status()
        response_data = api_response.json()

        print(f"Response received: {response_data}")
        print()

        # Download result
        if "data" in response_data and len(response_data["data"]) > 0:
            result_url = response_data["data"][0].get("url") or response_data["data"][0].get("b64_json")
            print(f"Result URL: {result_url[:100] if result_url else 'None'}...")

            # For local testing, check if it's a base64 data URI or http URL
            if result_url.startswith("http"):
                import requests
                img_response = requests.get(result_url, timeout=60)
                img_response.raise_for_status()
                output_path.write_bytes(img_response.content)
                print(f"Saved to: {output_path}")
            elif result_url.startswith("data:image"):
                # Base64 data URI
                import base64
                header, encoded = result_url.split(",", 1)
                img_data = base64.b64decode(encoded)
                output_path.write_bytes(img_data)
                print(f"Saved to: {output_path}")
            else:
                # Might be raw base64 without data: prefix
                import base64
                try:
                    img_data = base64.b64decode(result_url)
                    output_path.write_bytes(img_data)
                    print(f"Saved to: {output_path}")
                except Exception as decode_err:
                    print(f"Unexpected URL format: {result_url[:100]}")
                    print(f"Decode error: {decode_err}")
                    return False

            print(f"Output image size: {output_path.stat().st_size} bytes")
            return True
        else:
            print("No image data in response")
            return False

    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    # Configuration
    KEY_PATH = "/mnt/afs/visitor33/okcodex.json"
    BASE_URL = "https://api.okcodex.cn/v1"
    MODEL = "gpt-image-1.5"  # Adjust if needed

    # Use the existing empty room render from the grounded layout test
    INPUT_IMAGE = Path(
        "/mnt/afs/visitor33/Task3.2/outputs/critic_probe/"
        "grounded_polygon_acp_20260805_023741/critic_on/batch_001/latest-run/"
        "scene_000/room_living_room/scene_renders/furniture/empty_room_context/0_top.png"
    )

    OUTPUT_DIR = Path("/tmp/okcodex_test")
    OUTPUT_DIR.mkdir(exist_ok=True)
    OUTPUT_IMAGE = OUTPUT_DIR / "okcodex_edited.png"

    # Simple test prompt (shorter than the full SceneSmith prompt)
    PROMPT = """Edit this top-down room layout image by adding furniture.

The image shows an empty L-shaped living room from a top-down (overhead) view.

Add the following furniture pieces in appropriate locations:
- 2 sofas facing each other
- 1 coffee table between the sofas
- 2 armchairs
- 1 bookshelf
- 2 side tables
- 1 media console
- 2 floor lamps

Preserve the exact room boundaries, walls, doors, and windows.
Keep the top-down viewpoint and black background.
Arrange furniture to create a functional living room layout with clear walking paths."""

    if not INPUT_IMAGE.exists():
        print(f"ERROR: Input image not found: {INPUT_IMAGE}")
        sys.exit(1)

    # Load API key
    try:
        api_key = load_api_key(KEY_PATH)
        print(f"Loaded API key: {api_key[:20]}...")
        print()
    except Exception as e:
        print(f"ERROR loading API key: {e}")
        sys.exit(1)

    # Run test
    success = test_okcodex_image_edit(
        api_key=api_key,
        base_url=BASE_URL,
        model=MODEL,
        input_image_path=INPUT_IMAGE,
        prompt=PROMPT,
        output_path=OUTPUT_IMAGE,
    )

    if success:
        print()
        print("=" * 60)
        print("SUCCESS: Image editing test passed")
        print(f"Review output at: {OUTPUT_IMAGE}")
        print("=" * 60)
        sys.exit(0)
    else:
        print()
        print("=" * 60)
        print("FAILED: Image editing test failed")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
