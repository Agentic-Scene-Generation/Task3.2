"""
Bailian (Dashscope) context image editor.

Uses Alibaba Cloud's Bailian platform via dashscope SDK.
Supports models: wan2.7-image, wan2.7-image-pro, qwen-image-edit, qwen-image-3.0-pro
"""
import base64
import json
import logging
import time
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Any

from omegaconf import DictConfig
from PIL import Image

from scenesmith.prompts import PROMPTS_DATA_DIR
from scenesmith.prompts.manager import PromptManager
from scenesmith.prompts.registry import ImageGenerationPrompts

console_logger = logging.getLogger(__name__)


class BailianContextImageConfig:
    """Configuration for Bailian image editing service."""

    def __init__(self, cfg: DictConfig):
        """Initialize from OmegaConf config."""
        self.api_key: str = cfg.get("api_key", "")
        self.model: str = cfg.get("model", "wan2.7-image-pro")
        self.size: str = cfg.get("size", "768*768")
        self.normalize_to_reference_size: bool = cfg.get(
            "normalize_to_reference_size", True
        )

        # Timeouts
        self.timeout_seconds: float = cfg.get("timeout_seconds", 240.0)
        self.connect_timeout_seconds: float = cfg.get("connect_timeout_seconds", 5.0)
        self.max_retries: int = cfg.get("max_retries", 0)

        # Metadata
        self.write_metadata: bool = cfg.get("write_metadata", True)


class BailianContextImageEditor:
    """Context image editor using Bailian (Dashscope) API."""

    def __init__(self, config: DictConfig):
        """Initialize Bailian image editor.

        Args:
            config: OmegaConf DictConfig with bailian settings
        """
        self.config = BailianContextImageConfig(config)
        self.prompt_manager = PromptManager(prompts_dir=PROMPTS_DATA_DIR)

        # Import dashscope (lazy import to avoid dependency if not used)
        try:
            import dashscope
            from dashscope import MultiModalConversation
            self.dashscope = dashscope
            self.MultiModalConversation = MultiModalConversation
        except ImportError:
            raise ImportError(
                "dashscope is not installed. Install with: pip install dashscope"
            )

        console_logger.info(
            f"Initialized Bailian image editor: model={self.config.model}"
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
        """Generate furniture context image using Bailian API.

        Args:
            reference_image_path: Path to empty room render
            scene_description: Description of the scene for furniture placement
            width_m: Room width in meters
            length_m: Room length in meters
            output_path: Path where to save the generated image
            seed_override: Random seed for generation (unused for Bailian)

        Returns:
            Path to the generated image
        """
        # Build prompt from the authoritative shared template
        prompt = self.prompt_manager.get_prompt(
            ImageGenerationPrompts.FURNITURE_CONTEXT_IMAGE,
            scene_description=scene_description,
            width_m=width_m,
            length_m=length_m,
        )

        console_logger.info(
            f"Generating context image via Bailian: model={self.config.model}"
        )

        # Read and encode input image
        with open(reference_image_path, "rb") as f:
            img_bytes = f.read()
        with Image.open(reference_image_path) as reference_image:
            reference_size = reference_image.size

        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        img_data_url = f"data:image/png;base64,{img_b64}"

        # Call API
        start_time = time.time()

        try:
            response = self.MultiModalConversation.call(
                api_key=self.config.api_key,
                model=self.config.model,
                size=self.config.size,
                messages=[{
                    "role": "user",
                    "content": [
                        {"image": img_data_url},
                        {"text": prompt}
                    ]
                }]
            )

            elapsed = time.time() - start_time

            if response.status_code != 200:
                raise RuntimeError(
                    f"Bailian API error {response.status_code}: {response.code} - "
                    f"{response.message}"
                )

            console_logger.info(f"Bailian API call completed in {elapsed:.1f}s")

            # Extract image from response
            content = response.output.choices[0].message.content

            if not isinstance(content, list):
                raise ValueError(
                    f"Unexpected response format: content is not a list"
                )

            img_url = None
            for item in content:
                if isinstance(item, dict) and "image" in item:
                    img_url = item["image"]
                    break

            if not img_url:
                raise ValueError("No image found in Bailian response")

            # Download image
            console_logger.info(f"Downloading image from: {img_url[:80]}...")
            urllib.request.urlretrieve(img_url, output_path)

            # Wan may return dimensions that differ slightly from the requested
            # size. Normalize to the source canvas so strict image-grounding
            # geometry checks compare pixel-aligned images.
            with Image.open(output_path) as received_image:
                raw_output_size = received_image.size
                output_mode = received_image.mode
                should_normalize = (
                    self.config.normalize_to_reference_size
                    and raw_output_size != reference_size
                )
                if should_normalize:
                    normalized = received_image.resize(
                        reference_size,
                        Image.Resampling.LANCZOS,
                    )
                    normalized_path = output_path.with_name(
                        f".{output_path.stem}.normalized{output_path.suffix}"
                    )
                    normalized.save(normalized_path)
                    normalized_path.replace(output_path)

            with Image.open(output_path) as final_image:
                output_size = final_image.size
            console_logger.info(
                "Received image: raw=%dx%d, final=%dx%d, mode=%s",
                raw_output_size[0],
                raw_output_size[1],
                output_size[0],
                output_size[1],
                output_mode,
            )

            # Write metadata if enabled
            if self.config.write_metadata:
                metadata_path = output_path.with_suffix(".metadata.json")
                metadata = {
                    "model": self.config.model,
                    "prompt": prompt,
                    "elapsed_seconds": elapsed,
                    "input_image": str(reference_image_path),
                    "input_size": list(reference_size),
                    "requested_size": self.config.size,
                    "raw_output_size": list(raw_output_size),
                    "output_size": list(output_size),
                    "normalized_to_reference_size": should_normalize,
                    "image_url": img_url,
                }
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)

            console_logger.info(f"Saved context image to {output_path}")
            return output_path

        except Exception as e:
            elapsed = time.time() - start_time
            console_logger.error(
                f"Bailian image generation failed after {elapsed:.1f}s: {e}"
            )
            raise RuntimeError(f"Bailian image generation failed: {e}") from e
