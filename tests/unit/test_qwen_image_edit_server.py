import base64
import io
import json
import sys

from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("multipart")

from fastapi import HTTPException, UploadFile
from PIL import Image

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from qwen_image_edit_server import ServerConfig, create_app  # noqa: E402


class _FakePipeline:
    vae_scale_factor = 8

    def __init__(self, output_size=None):
        self.calls = []
        self.output_size = output_size

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        size = self.output_size or (kwargs["width"], kwargs["height"])
        return SimpleNamespace(images=[Image.new("RGB", size, color="navy")])


def _fake_runner(**kwargs):
    output = kwargs["pipeline"](**kwargs)
    return output.images[0], 0


def _config(tmp_path: Path) -> ServerConfig:
    return ServerConfig(
        model_dir=tmp_path,
        public_model_id="Qwen/Qwen-Image-Edit",
        dtype="bfloat16",
        max_request_bytes=2 * 1024 * 1024,
        max_dimension=1024,
        default_num_inference_steps=50,
        default_true_cfg_scale=4.0,
        default_negative_prompt=" ",
        default_seed=0,
    )


def _image_bytes(size: tuple[int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color="white").save(buffer, format="PNG")
    return buffer.getvalue()


def _edit_endpoint(app):
    for route in app.routes:
        if getattr(route, "path", None) == "/v1/images/edits":
            return route.endpoint
    raise AssertionError("edit endpoint not found")


def _call_edit(app, pipeline, *, size=(768, 512), **overrides):
    app.state.qwen_runtime.pipeline = pipeline
    values = {
        "image": UploadFile(
            file=io.BytesIO(_image_bytes(size)),
            filename="0_top.png",
        ),
        "prompt": "add a sofa",
        "model": "Qwen/Qwen-Image-Edit",
        "size": "auto",
        "n": 1,
        "response_format": "b64_json",
        "num_inference_steps": 2,
        "true_cfg_scale": 4.0,
        "negative_prompt": " ",
        "seed": 0,
    }
    values.update(overrides)
    return _edit_endpoint(app)(**values)


def test_edit_endpoint_preserves_uploaded_render_size(tmp_path):
    pipeline = _FakePipeline()
    app = create_app(
        config=_config(tmp_path),
        pipeline_loader=lambda _cfg: pipeline,
        pipeline_runner=_fake_runner,
        load_in_background=False,
    )
    response = _call_edit(app, pipeline)

    assert response.status_code == 200
    payload = json.loads(response.body)
    metadata = payload["x_qwen_image_edit"]
    assert metadata["input_width"] == 768
    assert metadata["output_width"] == 768
    assert pipeline.calls[0]["width"] == 768
    assert pipeline.calls[0]["height"] == 512
    decoded = base64.b64decode(payload["data"][0]["b64_json"])
    assert Image.open(io.BytesIO(decoded)).size == (768, 512)


def test_rejects_dimensions_that_pipeline_would_silently_round(tmp_path):
    pipeline = _FakePipeline()
    app = create_app(
        config=_config(tmp_path),
        pipeline_loader=lambda _cfg: pipeline,
        pipeline_runner=_fake_runner,
        load_in_background=False,
    )
    with pytest.raises(HTTPException) as exc_info:
        _call_edit(app, pipeline, size=(769, 512))

    assert exc_info.value.status_code == 400
    assert "divisible by 16" in str(exc_info.value.detail)
    assert pipeline.calls == []


def test_rejects_pipeline_output_size_mismatch(tmp_path):
    pipeline = _FakePipeline(output_size=(512, 512))
    app = create_app(
        config=_config(tmp_path),
        pipeline_loader=lambda _cfg: pipeline,
        pipeline_runner=_fake_runner,
        load_in_background=False,
    )
    with pytest.raises(HTTPException) as exc_info:
        _call_edit(app, pipeline, size=(768, 768))

    assert exc_info.value.status_code == 500
    assert "do not match" in str(exc_info.value.detail)
