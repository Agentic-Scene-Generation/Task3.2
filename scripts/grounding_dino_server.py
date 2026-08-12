"""Persistent, offline GroundingDINO FastAPI sidecar."""

from __future__ import annotations

import asyncio
import base64
import binascii
import io
import os
import time

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch

from fastapi import FastAPI, HTTPException
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field, field_validator
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


DEFAULT_MODEL_PATH = "/mnt/afs-p3/task3_2/visitor33_ljx/checkpoints/grounding-dino-base"
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
MAX_TEXTS_PER_TOKEN_REQUEST = 256


def _positive_env_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name, str(default))
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise RuntimeError(f"{name} must be a positive integer")
    return value


class TokenCountRequest(BaseModel):
    """Tokenizer request used by the orchestration client for exact batching."""

    texts: list[str] = Field(min_length=1, max_length=MAX_TEXTS_PER_TOKEN_REQUEST)

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, texts: list[str]) -> list[str]:
        if any(not text.strip() or len(text) > 20_000 for text in texts):
            raise ValueError("texts must contain non-empty bounded strings")
        return texts


class GroundRequest(BaseModel):
    """One image and one tokenizer-safe category batch."""

    image_base64: str = Field(min_length=1)
    categories: list[str] = Field(min_length=1, max_length=128)
    box_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    text_threshold: float = Field(default=0.20, ge=0.0, le=1.0)

    @field_validator("categories")
    @classmethod
    def validate_categories(cls, categories: list[str]) -> list[str]:
        normalized = [category.strip().lower() for category in categories]
        if any(not category or len(category) > 120 for category in normalized):
            raise ValueError("categories must contain non-empty bounded strings")
        if len(set(normalized)) != len(normalized):
            raise ValueError("categories must not contain duplicates")
        return normalized


class GroundingDinoService:
    """Own the processor/model and serialize bounded GPU inference."""

    def __init__(self) -> None:
        self.model_path = Path(
            os.environ.get("GROUNDING_DINO_MODEL_PATH", DEFAULT_MODEL_PATH)
        )
        self.dtype_name = os.environ.get("GROUNDING_DINO_DTYPE", "float16")
        self.max_concurrency = _positive_env_int("GROUNDING_DINO_MAX_CONCURRENCY", 1)
        self.device = torch.device("cuda:0")
        self.processor: Any | None = None
        self.model: Any | None = None
        self.semaphore = asyncio.Semaphore(self.max_concurrency)
        self.compute_dtype: torch.dtype | None = None
        self.loaded_memory_mib: float | None = None

    def load(self) -> None:
        """Load completely offline and fail instead of silently using CPU."""
        if not torch.cuda.is_available():
            raise RuntimeError("GroundingDINO requires a visible CUDA GPU")
        if not self.model_path.is_dir():
            raise RuntimeError(
                f"GroundingDINO model directory missing: {self.model_path}"
            )
        dtype = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }.get(self.dtype_name.lower())
        if dtype is None:
            raise RuntimeError(f"Unsupported GROUNDING_DINO_DTYPE={self.dtype_name!r}")

        torch.cuda.empty_cache()
        before = torch.cuda.memory_allocated(self.device)
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            local_files_only=True,
        )
        self.compute_dtype = dtype
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.model_path,
            local_files_only=True,
            dtype=torch.float32,
        ).to(self.device)
        self.model.eval()
        after = torch.cuda.memory_allocated(self.device)
        self.loaded_memory_mib = round((after - before) / (1024**2), 2)

    def token_counts(self, texts: list[str]) -> list[int]:
        """Return exact, untruncated tokenizer lengths including special tokens."""
        if self.processor is None:
            raise RuntimeError("GroundingDINO processor is not ready")
        tokenizer = self.processor.tokenizer
        encoded = tokenizer(texts, truncation=False, add_special_tokens=True)
        return [len(input_ids) for input_ids in encoded["input_ids"]]

    async def ground(self, request: GroundRequest) -> dict[str, Any]:
        """Queue inference so multiple workers cannot multiply model memory."""
        image = _decode_image(request.image_base64)
        prompt = ". ".join(request.categories) + "."
        token_count = self.token_counts([prompt])[0]
        max_text_len = int(getattr(self.model.config, "max_text_len", 256))
        if token_count > max_text_len:
            raise ValueError(
                f"category prompt has {token_count} tokens; maximum is {max_text_len}"
            )
        async with self.semaphore:
            return await asyncio.to_thread(
                self._ground_sync,
                image,
                prompt,
                request.box_threshold,
                request.text_threshold,
            )

    def _ground_sync(
        self,
        image: Image.Image,
        prompt: str,
        box_threshold: float,
        text_threshold: float,
    ) -> dict[str, Any]:
        if self.processor is None or self.model is None:
            raise RuntimeError("GroundingDINO model is not ready")
        started_at = time.monotonic()
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        model_dtype = next(self.model.parameters()).dtype
        moved_inputs: dict[str, Any] = {}
        for key, value in inputs.items():
            if not hasattr(value, "to"):
                moved_inputs[key] = value
            elif torch.is_tensor(value) and torch.is_floating_point(value):
                moved_inputs[key] = value.to(device=self.device, dtype=model_dtype)
            else:
                moved_inputs[key] = value.to(self.device)
        if self.compute_dtype is None:
            raise RuntimeError("GroundingDINO compute dtype is not initialized")
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type="cuda",
                dtype=self.compute_dtype,
                enabled=self.compute_dtype != torch.float32,
            ),
        ):
            outputs = self.model(**moved_inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            input_ids=moved_inputs.get("input_ids"),
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[(image.height, image.width)],
        )[0]
        raw_labels = results.get("text_labels", results.get("labels", []))
        detections: list[dict[str, Any]] = []
        for box, score, label in zip(
            results.get("boxes", []), results.get("scores", []), raw_labels
        ):
            phrase = str(label).strip().lower()
            if not phrase:
                continue
            detections.append(
                {
                    "phrase": phrase,
                    "score": round(float(score.detach().cpu().item()), 6),
                    "box_xyxy": [
                        round(float(value), 3) for value in box.detach().cpu().tolist()
                    ],
                }
            )
        return {
            "model": self.model_path.name,
            "image_width": image.width,
            "image_height": image.height,
            "detections": detections,
            "inference_ms": round((time.monotonic() - started_at) * 1000.0, 3),
        }

    def health_payload(self) -> dict[str, Any]:
        ready = self.model is not None and self.processor is not None
        payload: dict[str, Any] = {
            "service": "grounding-dino",
            "ready": ready,
            "model": str(self.model_path),
            "device": str(self.device),
            "dtype": self.dtype_name,
            "max_concurrency": self.max_concurrency,
            "loaded_memory_mib": self.loaded_memory_mib,
        }
        if torch.cuda.is_available():
            payload["cuda_memory_allocated_mib"] = round(
                torch.cuda.memory_allocated(self.device) / (1024**2), 2
            )
            payload["cuda_memory_reserved_mib"] = round(
                torch.cuda.memory_reserved(self.device) / (1024**2), 2
            )
        return payload


service = GroundingDinoService()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load once for the lifetime of the single uvicorn worker."""
    await asyncio.to_thread(service.load)
    yield


app = FastAPI(title="GroundingDINO sidecar", version="1.0", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, Any]:
    return service.health_payload()


@app.post("/v1/token-count")
async def token_count(request: TokenCountRequest) -> dict[str, list[int]]:
    try:
        return {"token_counts": service.token_counts(request.texts)}
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/v1/ground")
async def ground(request: GroundRequest) -> dict[str, Any]:
    try:
        return await service.ground(request)
    except (ValueError, UnidentifiedImageError, binascii.Error) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _decode_image(encoded: str) -> Image.Image:
    try:
        payload = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("image_base64 is not valid base64") from exc
    if not payload or len(payload) > MAX_IMAGE_BYTES:
        raise ValueError(f"image must be between 1 byte and {MAX_IMAGE_BYTES} bytes")
    image = Image.open(io.BytesIO(payload))
    image.load()
    if image.width * image.height > MAX_IMAGE_PIXELS:
        raise ValueError(f"image exceeds {MAX_IMAGE_PIXELS} pixels")
    return image.convert("RGB")
