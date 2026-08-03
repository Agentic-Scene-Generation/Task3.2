#!/usr/bin/env python3
"""Single-GPU, persistent OpenAI-compatible Qwen-Image-Edit service."""

import base64
import io
import logging
import os
import threading
import time
import uuid

from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

# Reuse uvicorn's configured logger so service metrics are emitted at INFO.
LOGGER = logging.getLogger("uvicorn.error")


@dataclass(frozen=True)
class ServerConfig:
    model_dir: Path
    public_model_id: str
    dtype: str
    max_request_bytes: int
    max_dimension: int
    default_num_inference_steps: int
    default_true_cfg_scale: float
    default_negative_prompt: str
    default_seed: int

    @classmethod
    def from_env(cls) -> "ServerConfig":
        project_root = Path(__file__).resolve().parent.parent
        return cls(
            model_dir=Path(
                os.environ.get(
                    "QWEN_IMAGE_EDIT_MODEL_DIR",
                    project_root / "models" / "Qwen-Image-Edit",
                )
            ),
            public_model_id=os.environ.get(
                "QWEN_IMAGE_EDIT_MODEL_ID", "Qwen/Qwen-Image-Edit"
            ),
            dtype=os.environ.get("QWEN_IMAGE_EDIT_DTYPE", "bfloat16"),
            max_request_bytes=int(
                os.environ.get("QWEN_IMAGE_EDIT_MAX_REQUEST_BYTES", 20 * 1024 * 1024)
            ),
            max_dimension=int(
                os.environ.get("QWEN_IMAGE_EDIT_MAX_DIMENSION", "1024")
            ),
            default_num_inference_steps=int(
                os.environ.get("QWEN_IMAGE_EDIT_NUM_INFERENCE_STEPS", "50")
            ),
            default_true_cfg_scale=float(
                os.environ.get("QWEN_IMAGE_EDIT_TRUE_CFG_SCALE", "4.0")
            ),
            default_negative_prompt=os.environ.get(
                "QWEN_IMAGE_EDIT_NEGATIVE_PROMPT", " "
            ),
            default_seed=int(os.environ.get("QWEN_IMAGE_EDIT_SEED", "0")),
        )


class RuntimeState:
    def __init__(self) -> None:
        self.pipeline: Any | None = None
        self.loading = False
        self.load_error: str | None = None
        self.loaded_at: float | None = None
        self.inference_lock = threading.Lock()


def load_qwen_pipeline(config: ServerConfig) -> Any:
    """Load the official diffusers pipeline once and move it to visible cuda:0."""
    import torch

    from diffusers import QwenImageEditPipeline

    if not config.model_dir.is_dir():
        raise FileNotFoundError(f"Qwen model directory not found: {config.model_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the Qwen image service")
    if config.dtype != "bfloat16":
        raise ValueError(
            f"Only bfloat16 is supported in phase 1, got dtype={config.dtype!r}"
        )

    LOGGER.info("Loading Qwen-Image-Edit from %s", config.model_dir)
    pipeline = QwenImageEditPipeline.from_pretrained(
        str(config.model_dir),
        local_files_only=True,
    )
    pipeline.to(torch.bfloat16)
    pipeline.to("cuda")
    pipeline.set_progress_bar_config(disable=None)
    LOGGER.info("Qwen-Image-Edit pipeline loaded on cuda:0")
    return pipeline


def create_app(
    *,
    config: ServerConfig | None = None,
    pipeline_loader: Callable[[ServerConfig], Any] | None = None,
    pipeline_runner: Callable[..., tuple[Image.Image, int]] | None = None,
    load_in_background: bool = True,
) -> FastAPI:
    config = config or ServerConfig.from_env()
    pipeline_loader = pipeline_loader or load_qwen_pipeline
    pipeline_runner = pipeline_runner or _run_pipeline
    runtime = RuntimeState()

    def load_model() -> None:
        runtime.loading = True
        runtime.load_error = None
        try:
            runtime.pipeline = pipeline_loader(config)
            runtime.loaded_at = time.time()
        except Exception as exc:
            runtime.pipeline = None
            runtime.load_error = f"{type(exc).__name__}: {exc}"
            LOGGER.exception("Qwen-Image-Edit model load failed")
        finally:
            runtime.loading = False

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if load_in_background:
            threading.Thread(
                target=load_model,
                name="qwen-image-edit-loader",
                daemon=True,
            ).start()
        else:
            load_model()
        yield

    app = FastAPI(
        title="Qwen-Image-Edit local service",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.qwen_runtime = runtime
    app.state.qwen_config = config

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "service": "qwen-image-edit",
            "status": "healthy",
            "loading": runtime.loading,
            "ready": runtime.pipeline is not None,
            "load_error": runtime.load_error,
        }

    @app.get("/ready")
    def ready() -> JSONResponse:
        if runtime.pipeline is None:
            return JSONResponse(
                status_code=503,
                content={
                    "service": "qwen-image-edit",
                    "ready": False,
                    "loading": runtime.loading,
                    "load_error": runtime.load_error,
                },
            )
        return JSONResponse(
            status_code=200,
            content={
                "service": "qwen-image-edit",
                "ready": True,
                "model": config.public_model_id,
                "loaded_at": runtime.loaded_at,
            },
        )

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": config.public_model_id,
                    "object": "model",
                    "owned_by": "local",
                }
            ],
        }

    @app.post("/v1/images/edits")
    def edit_image(
        image: UploadFile = File(...),
        prompt: str = Form(...),
        model: str = Form(...),
        size: str = Form("auto"),
        n: int = Form(1),
        response_format: str = Form("b64_json"),
        num_inference_steps: int | None = Form(None),
        true_cfg_scale: float | None = Form(None),
        negative_prompt: str | None = Form(None),
        seed: int | None = Form(None),
    ) -> JSONResponse:
        if runtime.pipeline is None:
            raise HTTPException(
                status_code=503,
                detail={
                    "message": "Qwen-Image-Edit is not ready",
                    "loading": runtime.loading,
                    "load_error": runtime.load_error,
                },
            )
        if model != config.public_model_id:
            raise HTTPException(status_code=400, detail=f"Unknown model: {model}")
        if size != "auto":
            raise HTTPException(
                status_code=400,
                detail="size must be 'auto'; output follows the uploaded render",
            )
        if n != 1:
            raise HTTPException(status_code=400, detail="Only n=1 is supported")
        if response_format != "b64_json":
            raise HTTPException(
                status_code=400, detail="Only response_format=b64_json is supported"
            )
        if not prompt or len(prompt) > 20_000:
            raise HTTPException(
                status_code=400, detail="prompt must contain 1-20000 characters"
            )

        steps = (
            config.default_num_inference_steps
            if num_inference_steps is None
            else num_inference_steps
        )
        cfg_scale = (
            config.default_true_cfg_scale
            if true_cfg_scale is None
            else true_cfg_scale
        )
        effective_negative_prompt = (
            config.default_negative_prompt
            if negative_prompt is None
            else negative_prompt
        )
        effective_seed = config.default_seed if seed is None else seed
        _validate_inference_parameters(
            steps=steps,
            cfg_scale=cfg_scale,
            seed=effective_seed,
            negative_prompt=effective_negative_prompt,
        )

        raw_image = image.file.read(config.max_request_bytes + 1)
        if len(raw_image) > config.max_request_bytes:
            raise HTTPException(status_code=413, detail="Uploaded image is too large")
        try:
            with Image.open(io.BytesIO(raw_image)) as opened_image:
                opened_image.load()
                if opened_image.format not in {"PNG", "JPEG"}:
                    raise HTTPException(
                        status_code=400, detail="Only PNG and JPEG images are supported"
                    )
                input_image = opened_image.convert("RGB")
        except HTTPException:
            raise
        except (UnidentifiedImageError, OSError) as exc:
            raise HTTPException(status_code=400, detail="Invalid image file") from exc

        width, height = input_image.size
        if width <= 0 or height <= 0:
            raise HTTPException(status_code=400, detail="Invalid image dimensions")
        if max(width, height) > config.max_dimension:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Image dimensions exceed max {config.max_dimension}: "
                    f"{width}x{height}"
                ),
            )

        dimension_multiple = int(runtime.pipeline.vae_scale_factor) * 2
        if width % dimension_multiple or height % dimension_multiple:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Render dimensions must be divisible by {dimension_multiple}; "
                    f"got {width}x{height}. The service will not resize the render."
                ),
            )

        request_id = f"qie_{uuid.uuid4().hex}"
        queue_started = time.monotonic()
        with runtime.inference_lock:
            queue_wait_seconds = time.monotonic() - queue_started
            inference_started = time.monotonic()
            try:
                output_image, cuda_peak_bytes = pipeline_runner(
                    pipeline=runtime.pipeline,
                    image=input_image,
                    prompt=prompt,
                    width=width,
                    height=height,
                    steps=steps,
                    cfg_scale=cfg_scale,
                    negative_prompt=effective_negative_prompt,
                    seed=effective_seed,
                )
            except Exception as exc:
                try:
                    import torch

                    is_oom = isinstance(exc, torch.cuda.OutOfMemoryError)
                except Exception:
                    is_oom = False
                LOGGER.exception(
                    "Qwen image edit failed request_id=%s input=%dx%d",
                    request_id,
                    width,
                    height,
                )
                raise HTTPException(
                    status_code=503 if is_oom else 500,
                    detail={
                        "message": "Qwen image edit inference failed",
                        "request_id": request_id,
                        "error_type": type(exc).__name__,
                    },
                ) from exc
            inference_seconds = time.monotonic() - inference_started

        if output_image.size != (width, height):
            raise HTTPException(
                status_code=500,
                detail={
                    "message": "Qwen output dimensions do not match the render",
                    "request_id": request_id,
                    "input_size": [width, height],
                    "output_size": list(output_image.size),
                },
            )

        buffer = io.BytesIO()
        output_image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        total_seconds = time.monotonic() - queue_started
        metadata = {
            "request_id": request_id,
            "model": config.public_model_id,
            "size": "auto",
            "input_width": width,
            "input_height": height,
            "output_width": output_image.width,
            "output_height": output_image.height,
            "num_inference_steps": steps,
            "true_cfg_scale": cfg_scale,
            "negative_prompt": effective_negative_prompt,
            "seed": effective_seed,
            "queue_wait_seconds": queue_wait_seconds,
            "inference_seconds": inference_seconds,
            "total_seconds": total_seconds,
            "cuda_max_memory_allocated_bytes": cuda_peak_bytes,
        }
        LOGGER.info(
            "request_id=%s model=%s input=%dx%d output=%dx%d steps=%d "
            "queue_wait_seconds=%.3f inference_seconds=%.3f total_seconds=%.3f "
            "cuda_max_memory_allocated_bytes=%d",
            request_id,
            config.public_model_id,
            width,
            height,
            output_image.width,
            output_image.height,
            steps,
            queue_wait_seconds,
            inference_seconds,
            total_seconds,
            cuda_peak_bytes,
        )
        return JSONResponse(
            content={
                "created": int(time.time()),
                "data": [{"b64_json": encoded}],
                "x_qwen_image_edit": metadata,
            }
        )

    return app


def _validate_inference_parameters(
    *,
    steps: int,
    cfg_scale: float,
    seed: int,
    negative_prompt: str,
) -> None:
    if not 2 <= steps <= 100:
        raise HTTPException(
            status_code=400, detail="num_inference_steps must be between 2 and 100"
        )
    if not 0.0 <= cfg_scale <= 20.0:
        raise HTTPException(
            status_code=400, detail="true_cfg_scale must be between 0 and 20"
        )
    if not 0 <= seed <= 2**32 - 1:
        raise HTTPException(status_code=400, detail="seed must be between 0 and 2^32-1")
    if len(negative_prompt) > 20_000:
        raise HTTPException(
            status_code=400, detail="negative_prompt exceeds 20000 characters"
        )


def _run_pipeline(
    *,
    pipeline: Any,
    image: Image.Image,
    prompt: str,
    width: int,
    height: int,
    steps: int,
    cfg_scale: float,
    negative_prompt: str,
    seed: int,
) -> tuple[Image.Image, int]:
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    generator = torch.manual_seed(seed)
    with torch.inference_mode():
        output = pipeline(
            image=image,
            prompt=prompt,
            generator=generator,
            true_cfg_scale=cfg_scale,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            width=width,
            height=height,
        )
    output_image = output.images[0]
    cuda_peak_bytes = (
        int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    )
    return output_image, cuda_peak_bytes


app = create_app()
