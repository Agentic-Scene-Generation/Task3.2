"""
OKCodex-specific context image editor.

OKCodex requires a different API format than standard OpenAI:
- Uses JSON payload with images[].image_url (not multipart/form-data)
- Requires special x-openai-actor-authorization header
- Returns b64_json instead of URL
"""
import base64
import logging
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
from omegaconf import DictConfig
from PIL import Image

from scenesmith.prompts import PROMPTS_DATA_DIR
from scenesmith.prompts.manager import PromptManager
from scenesmith.prompts.registry import ImageGenerationPrompts

console_logger = logging.getLogger(__name__)


class OKCodexContextImageConfig:
    """Configuration for OKCodex image editing service."""

    def __init__(self, cfg: DictConfig):
        """Initialize from OmegaConf config."""
        self.base_url: str = cfg.get("base_url", "https://api.okcodex.cn")
        self.api_key: str = cfg.get("api_key", "")
        self.model: str = cfg.get("model", "gpt-image-1.5")
        self.actor_auth: str = cfg.get("actor_auth", "local-image-extension")

        # Generation parameters (okcodex may ignore some of these)
        self.num_inference_steps: int = cfg.get("num_inference_steps", 50)
        self.true_cfg_scale: float = cfg.get("true_cfg_scale", 4.0)
        self.negative_prompt: str = cfg.get("negative_prompt", " ")
        self.seed: int = cfg.get("seed", 0)

        # Timeouts
        self.timeout_seconds: float = cfg.get("timeout_seconds", 240.0)
        self.connect_timeout_seconds: float = cfg.get("connect_timeout_seconds", 5.0)
        self.max_retries: int = cfg.get("max_retries", 0)

        # Metadata
        self.write_metadata: bool = cfg.get("write_metadata", True)


class OKCodexContextImageEditor:
    """Context image editor using OKCodex API."""

    def __init__(self, config: DictConfig):
        """Initialize OKCodex image editor.

        Args:
            config: OmegaConf DictConfig with okcodex settings
        """
        self.config = OKCodexContextImageConfig(config)
        self.prompt_manager = PromptManager(prompts_dir=PROMPTS_DATA_DIR)
        console_logger.info(
            f"Initialized OKCodex image editor: {self.config.base_url}, "
            f"model={self.config.model}"
        )

    def generate_furniture_context_image(
        self,
        reference_image_path: Path,
        scene_description: str,
        width_m: float,
        length_m: float,
        output_path: Path,
        seed_override: int | None = None,
    ) -> Path:
        """Generate furniture context image using OKCodex API.

        Args:
            reference_image_path: Path to empty room render
            scene_description: Description of the scene for furniture placement
            width_m: Room width in meters
            length_m: Room length in meters
            output_path: Path where to save the generated image
            seed_override: Random seed for generation

        Returns:
            Path to the generated image
        """
        seed = seed_override if seed_override is not None else self.config.seed

        # Build prompt from the authoritative shared template so OKCodex is
        # bound by the same top-down/view-preservation constraints as the
        # qwen_local and gemini backends (do NOT hand-write this prompt).
        prompt = self.prompt_manager.get_prompt(
            ImageGenerationPrompts.FURNITURE_CONTEXT_IMAGE,
            scene_description=scene_description,
            width_m=width_m,
            length_m=length_m,
        )

        console_logger.info(
            f"Generating context image via OKCodex: model={self.config.model}, "
            f"seed={seed}"
        )

        # Read and encode input image
        with open(reference_image_path, "rb") as f:
            img_bytes = f.read()

        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        img_data_url = f"data:image/png;base64,{img_b64}"

        # Prepare headers
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "x-openai-actor-authorization": self.config.actor_auth,
            "Content-Type": "application/json",
        }

        # Prepare payload in okcodex format
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "images": [{"image_url": img_data_url}],
            "n": 1,
        }

        # Add seed if supported (okcodex may ignore this)
        if seed != 0:
            payload["seed"] = seed

        # Call API
        url = f"{self.config.base_url}/v1/images/edits"
        start_time = time.time()

        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=(
                    self.config.connect_timeout_seconds,
                    self.config.timeout_seconds,
                ),
            )
            response.raise_for_status()

            elapsed = time.time() - start_time
            console_logger.info(f"OKCodex API call completed in {elapsed:.1f}s")

            data = response.json()

            # Extract image from response
            if "data" not in data or len(data["data"]) == 0:
                raise ValueError("No image data in OKCodex response")

            result = data["data"][0]

            # OKCodex returns b64_json, not URL
            if "b64_json" in result:
                img_bytes = base64.b64decode(result["b64_json"])
            elif "url" in result:
                result_url = result["url"]
                if result_url.startswith("http"):
                    img_response = requests.get(result_url, timeout=60)
                    img_response.raise_for_status()
                    img_bytes = img_response.content
                elif result_url.startswith("data:"):
                    _, encoded = result_url.split(",", 1)
                    img_bytes = base64.b64decode(encoded)
                else:
                    img_bytes = base64.b64decode(result_url)
            else:
                raise ValueError(
                    f"No 'b64_json' or 'url' in OKCodex response. "
                    f"Keys: {list(result.keys())}"
                )

            # Convert to PIL Image
            image = Image.open(BytesIO(img_bytes))
            console_logger.info(
                f"Received image: {image.size[0]}x{image.size[1]}, mode={image.mode}"
            )

            # Save to output path
            image.save(output_path)
            console_logger.info(f"Saved context image to {output_path}")

            # Write metadata if enabled
            if self.config.write_metadata:
                metadata_path = output_path.with_suffix(".metadata.json")
                import json
                metadata = {
                    "model": self.config.model,
                    "prompt": prompt,
                    "seed": seed,
                    "elapsed_seconds": elapsed,
                    "input_image": str(reference_image_path),
                    "output_size": list(image.size),
                }
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)

            return output_path

        except requests.exceptions.Timeout as e:
            raise TimeoutError(
                f"OKCodex API timeout after {self.config.timeout_seconds}s"
            ) from e
        except requests.exceptions.HTTPError as e:
            error_text = e.response.text if e.response else str(e)
            raise RuntimeError(
                f"OKCodex API error {e.response.status_code if e.response else 'unknown'}: "
                f"{error_text[:500]}"
            ) from e
        except Exception as e:
            raise RuntimeError(f"OKCodex image generation failed: {e}") from e
