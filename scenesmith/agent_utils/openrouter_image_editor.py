"""OpenRouter context image editor using the dedicated Images API."""

from __future__ import annotations

import base64
import json
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


class OpenRouterContextImageConfig:
    """Configuration for OpenRouter image-to-image generation."""

    def __init__(self, cfg: DictConfig):
        self.base_url: str = str(
            cfg.get("base_url", "https://openrouter.ai/api/v1")
        ).rstrip("/")
        self.api_key: str = cfg.get("api_key", "")
        self.model: str = cfg.get("model", "openai/gpt-image-2")
        self.quality: str = cfg.get("quality", "medium")
        self.aspect_ratio: str = cfg.get("aspect_ratio", "1:1")
        self.background: str = cfg.get("background", "opaque")
        self.normalize_to_reference_size: bool = cfg.get(
            "normalize_to_reference_size", True
        )
        self.timeout_seconds: float = cfg.get("timeout_seconds", 300.0)
        self.connect_timeout_seconds: float = cfg.get("connect_timeout_seconds", 5.0)
        self.max_retries: int = cfg.get("max_retries", 0)
        self.write_metadata: bool = cfg.get("write_metadata", True)

        if not self.base_url:
            raise ValueError("openrouter.base_url must not be empty")
        if not self.api_key:
            raise ValueError("openrouter.api_key must not be empty")
        if not self.model:
            raise ValueError("openrouter.model must not be empty")
        if self.quality not in {"auto", "low", "medium", "high"}:
            raise ValueError("openrouter.quality must be auto, low, medium, or high")
        if self.background not in {"auto", "opaque"}:
            raise ValueError("openrouter.background must be auto or opaque")
        if self.timeout_seconds <= 0 or self.connect_timeout_seconds <= 0:
            raise ValueError("openrouter timeouts must be positive")
        if self.max_retries < 0:
            raise ValueError("openrouter.max_retries must not be negative")


class OpenRouterContextImageEditor:
    """Edit context images through OpenRouter's dedicated Images API."""

    def __init__(
        self,
        config: DictConfig,
        *,
        session: requests.Session | None = None,
        prompt_manager: PromptManager | None = None,
    ) -> None:
        self.config = OpenRouterContextImageConfig(config)
        self.session = session or requests.Session()
        self.prompt_manager = prompt_manager or PromptManager(
            prompts_dir=PROMPTS_DATA_DIR
        )
        console_logger.info(
            "Initialized OpenRouter image editor: %s, model=%s",
            self.config.base_url,
            self.config.model,
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
        """Add furniture to an empty-room render while preserving its canvas."""
        prompt = self.prompt_manager.get_prompt(
            ImageGenerationPrompts.FURNITURE_CONTEXT_IMAGE,
            scene_description=scene_description,
            width_m=width_m,
            length_m=length_m,
        )
        with Image.open(reference_image_path) as reference_image:
            reference_size = reference_image.size
        image_b64 = base64.b64encode(reference_image_path.read_bytes()).decode("ascii")
        payload: dict[str, Any] = {
            "model": self.config.model,
            "prompt": prompt,
            "input_references": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_b64}",
                    },
                }
            ],
            "n": 1,
            "quality": self.config.quality,
            "aspect_ratio": self.config.aspect_ratio,
            "background": self.config.background,
        }
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        endpoint = f"{self.config.base_url}/images"
        start_time = time.monotonic()
        try:
            response = self.session.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=(
                    self.config.connect_timeout_seconds,
                    self.config.timeout_seconds,
                ),
            )
            response.raise_for_status()
            result = response.json()
        except requests.exceptions.Timeout as exc:
            raise TimeoutError(
                f"OpenRouter API timeout after {self.config.timeout_seconds}s"
            ) from exc
        except requests.exceptions.HTTPError as exc:
            response_text = exc.response.text if exc.response is not None else str(exc)
            status_code = (
                exc.response.status_code if exc.response is not None else "unknown"
            )
            raise RuntimeError(
                f"OpenRouter API error {status_code}: {response_text[:1000]}"
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"OpenRouter API request failed: {exc}") from exc
        except ValueError as exc:
            raise RuntimeError("OpenRouter API returned invalid JSON") from exc

        elapsed = time.monotonic() - start_time
        data = result.get("data") or []
        if not data or not isinstance(data[0], dict) or not data[0].get("b64_json"):
            raise RuntimeError("OpenRouter response did not contain data[0].b64_json")
        try:
            output_bytes = base64.b64decode(data[0]["b64_json"], validate=True)
            with Image.open(BytesIO(output_bytes)) as received_image:
                received_image.load()
                raw_output_size = received_image.size
                raw_output_format = received_image.format
                final_image = received_image.convert("RGB")
        except Exception as exc:
            raise RuntimeError("OpenRouter returned an invalid raster image") from exc

        normalized = (
            self.config.normalize_to_reference_size
            and raw_output_size != reference_size
        )
        if normalized:
            final_image = final_image.resize(reference_size, Image.Resampling.LANCZOS)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_name(
            f".{output_path.stem}.openrouter.tmp{output_path.suffix}"
        )
        final_image.save(temporary_path, format="PNG")
        temporary_path.replace(output_path)
        output_size = final_image.size

        console_logger.info(
            "OpenRouter image completed in %.1fs: raw=%dx%d, final=%dx%d",
            elapsed,
            raw_output_size[0],
            raw_output_size[1],
            output_size[0],
            output_size[1],
        )
        if self.config.write_metadata:
            metadata = {
                "backend": "openrouter",
                "model": self.config.model,
                "response_model": result.get("model"),
                "prompt": prompt,
                "elapsed_seconds": elapsed,
                "input_image": str(reference_image_path),
                "input_size": list(reference_size),
                "quality": self.config.quality,
                "aspect_ratio": self.config.aspect_ratio,
                "background": self.config.background,
                "raw_output_size": list(raw_output_size),
                "raw_output_format": raw_output_format,
                "media_type": data[0].get("media_type"),
                "output_size": list(output_size),
                "normalized_to_reference_size": normalized,
                "seed_override": seed_override,
                "seed_sent": False,
                "created": result.get("created"),
                "usage": result.get("usage"),
            }
            metadata_path = output_path.with_suffix(".metadata.json")
            metadata_path.write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        return output_path
