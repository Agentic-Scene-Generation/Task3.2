"""Context-image editing backends that are separate from asset generation."""

import base64
import hashlib
import json
import logging
import os
import tempfile
import time

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from PIL import Image

from scenesmith.prompts import PROMPTS_DATA_DIR
from scenesmith.prompts.manager import PromptManager
from scenesmith.prompts.registry import ImageGenerationPrompts

console_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from openai import OpenAI


@dataclass(frozen=True)
class QwenLocalContextImageConfig:
    """Validated client-side settings for the local Qwen image-edit service."""

    base_url: str
    api_key: str
    model: str
    size: str
    num_inference_steps: int
    true_cfg_scale: float
    negative_prompt: str
    seed: int
    timeout_seconds: float
    connect_timeout_seconds: float
    max_retries: int
    write_metadata: bool

    @classmethod
    def from_config(cls, cfg: Any) -> "QwenLocalContextImageConfig":
        """Build and validate settings from the qwen_local config section."""
        settings = cls(
            base_url=str(cfg.get("base_url", "http://127.0.0.1:18020/v1")).rstrip(
                "/"
            ),
            api_key=str(cfg.get("api_key", "not-needed")),
            model=str(cfg.get("model", "Qwen/Qwen-Image-Edit")),
            size=str(cfg.get("size", "auto")),
            num_inference_steps=int(cfg.get("num_inference_steps", 50)),
            true_cfg_scale=float(cfg.get("true_cfg_scale", 4.0)),
            negative_prompt=str(cfg.get("negative_prompt", " ")),
            seed=int(cfg.get("seed", 0)),
            timeout_seconds=float(cfg.get("timeout_seconds", 120)),
            connect_timeout_seconds=float(cfg.get("connect_timeout_seconds", 5)),
            max_retries=int(cfg.get("max_retries", 0)),
            write_metadata=bool(cfg.get("write_metadata", True)),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        """Reject invalid static configuration before issuing a request."""
        if not self.base_url:
            raise ValueError("qwen_local.base_url must not be empty")
        if not self.model:
            raise ValueError("qwen_local.model must not be empty")
        if self.size != "auto":
            raise ValueError(
                "qwen_local.size must be 'auto' so output follows the render size"
            )
        if not 2 <= self.num_inference_steps <= 100:
            raise ValueError("qwen_local.num_inference_steps must be between 2 and 100")
        if not 0.0 <= self.true_cfg_scale <= 20.0:
            raise ValueError("qwen_local.true_cfg_scale must be between 0 and 20")
        if not 0 <= self.seed <= 2**32 - 1:
            raise ValueError("qwen_local.seed must be between 0 and 2^32-1")
        if self.timeout_seconds <= 0:
            raise ValueError("qwen_local.timeout_seconds must be positive")
        if self.connect_timeout_seconds <= 0:
            raise ValueError("qwen_local.connect_timeout_seconds must be positive")
        if self.max_retries < 0:
            raise ValueError("qwen_local.max_retries must not be negative")


class OpenAICompatibleContextImageEditor:
    """Furniture context editor backed by an OpenAI-compatible local service."""

    def __init__(
        self,
        cfg: Any,
        client: "OpenAI | None" = None,
        prompt_manager: PromptManager | None = None,
    ):
        self.config = QwenLocalContextImageConfig.from_config(cfg)
        timeout = httpx.Timeout(
            timeout=self.config.timeout_seconds,
            connect=self.config.connect_timeout_seconds,
        )
        if client is None:
            from openai import OpenAI

            # This endpoint is loopback-only. Ignore cluster-wide HTTP/SOCKS
            # proxy variables, which can otherwise intercept localhost calls.
            self._http_client: httpx.Client | None = httpx.Client(
                timeout=timeout,
                trust_env=False,
            )
            client = OpenAI(
                base_url=self.config.base_url,
                api_key=self.config.api_key,
                http_client=self._http_client,
                max_retries=self.config.max_retries,
            )
        else:
            self._http_client = None
        self.client = client
        self.prompt_manager = prompt_manager or PromptManager(
            prompts_dir=PROMPTS_DATA_DIR
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
        """Edit an empty-room render and preserve its exact output dimensions."""
        prompt = self.prompt_manager.get_prompt(
            ImageGenerationPrompts.FURNITURE_CONTEXT_IMAGE,
            scene_description=scene_description,
            width_m=width_m,
            length_m=length_m,
        )
        return self._edit_image(
            prompt=prompt,
            reference_image_path=reference_image_path,
            output_path=output_path,
            seed_override=seed_override,
        )

    def _edit_image(
        self,
        prompt: str,
        reference_image_path: Path,
        output_path: Path,
        seed_override: int | None = None,
    ) -> Path:
        reference_image_path = Path(reference_image_path)
        output_path = Path(output_path)
        metadata_path = output_path.with_suffix(".metadata.json")
        start_time = time.monotonic()
        effective_seed = (
            self.config.seed if seed_override is None else int(seed_override)
        )
        if not 0 <= effective_seed <= 2**32 - 1:
            raise ValueError("seed_override must be between 0 and 2**32-1")

        with Image.open(reference_image_path) as reference:
            input_size = reference.size
            reference.verify()

        metadata: dict[str, Any] = {
            "backend": "qwen_local",
            "endpoint": self.config.base_url,
            "model": self.config.model,
            "size": self.config.size,
            "input_width": input_size[0],
            "input_height": input_size[1],
            "num_inference_steps": self.config.num_inference_steps,
            "true_cfg_scale": self.config.true_cfg_scale,
            "negative_prompt": self.config.negative_prompt,
            "seed": effective_seed,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "input_image_sha256": _sha256_file(reference_image_path),
            "success": False,
            "error_type": None,
        }

        try:
            console_logger.info(
                "Editing furniture context image with local Qwen service: %s (%dx%d)",
                reference_image_path,
                input_size[0],
                input_size[1],
            )
            with reference_image_path.open("rb") as image_file:
                response = self.client.images.edit(
                    model=self.config.model,
                    image=image_file,
                    prompt=prompt,
                    size=self.config.size,
                    response_format="b64_json",
                    extra_body={
                        "num_inference_steps": self.config.num_inference_steps,
                        "true_cfg_scale": self.config.true_cfg_scale,
                        "negative_prompt": self.config.negative_prompt,
                        "seed": effective_seed,
                    },
                )

            encoded_image = _extract_b64_json(response)
            image_bytes = base64.b64decode(encoded_image, validate=True)
            output_size = _atomic_save_png(
                image_bytes=image_bytes,
                output_path=output_path,
            )
            if output_size != input_size:
                output_path.unlink(missing_ok=True)
                raise ValueError(
                    "Local Qwen output size does not match the reference render: "
                    f"input={input_size}, output={output_size}"
                )

            server_metadata = _extract_server_metadata(response)
            metadata.update(server_metadata)
            metadata.update(
                {
                    "input_width": input_size[0],
                    "input_height": input_size[1],
                    "output_width": output_size[0],
                    "output_height": output_size[1],
                    "client_total_seconds": time.monotonic() - start_time,
                    "server_metadata_available": bool(server_metadata),
                    "success": True,
                    "error_type": None,
                }
            )
            if self.config.write_metadata:
                _atomic_write_json(metadata_path, metadata)

            console_logger.info(
                "Saved local Qwen context image to %s in %.2f seconds",
                output_path,
                metadata["client_total_seconds"],
            )
            return output_path
        except Exception as exc:
            metadata.update(
                {
                    "client_total_seconds": time.monotonic() - start_time,
                    "success": False,
                    "error_type": type(exc).__name__,
                }
            )
            if self.config.write_metadata:
                try:
                    _atomic_write_json(metadata_path, metadata)
                except Exception:
                    console_logger.exception(
                        "Failed to write Qwen context failure metadata: %s",
                        metadata_path,
                    )
            raise


def _extract_b64_json(response: Any) -> str:
    data = getattr(response, "data", None)
    if not data:
        raise ValueError("Local image-edit response contains no data")
    item = data[0]
    encoded = (
        item.get("b64_json")
        if isinstance(item, dict)
        else getattr(item, "b64_json", None)
    )
    if not encoded:
        raise ValueError("Local image-edit response contains no b64_json")
    return str(encoded)


def _extract_server_metadata(response: Any) -> dict[str, Any]:
    metadata = getattr(response, "x_qwen_image_edit", None)
    if isinstance(metadata, dict):
        return metadata

    model_extra = getattr(response, "model_extra", None)
    if isinstance(model_extra, dict):
        metadata = model_extra.get("x_qwen_image_edit")
        if isinstance(metadata, dict):
            return metadata

    model_dump = getattr(response, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, dict):
            metadata = dumped.get("x_qwen_image_edit")
            if isinstance(metadata, dict):
                return metadata
    return {}


def _atomic_save_png(image_bytes: bytes, output_path: Path) -> tuple[int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as temporary_file:
            temporary_file.write(image_bytes)
            temporary_path = Path(temporary_file.name)

        with Image.open(temporary_path) as image:
            image.load()
            output_size = image.size
            if image.format != "PNG":
                converted_path = temporary_path.with_suffix(".png")
                image.convert("RGB").save(converted_path, format="PNG")
                temporary_path.unlink(missing_ok=True)
                temporary_path = converted_path

        os.replace(temporary_path, output_path)
        temporary_path = None
        return output_size
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary_file:
            json.dump(payload, temporary_file, ensure_ascii=False, indent=2)
            temporary_file.write("\n")
            temporary_path = Path(temporary_file.name)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
