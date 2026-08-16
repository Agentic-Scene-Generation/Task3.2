import base64
import importlib.util
import io

from pathlib import Path

import pytest

from PIL import Image
from pydantic import ValidationError


SERVER_PATH = Path(__file__).parents[2] / "scripts/grounding_dino_server.py"


def _load_server_module():
    spec = importlib.util.spec_from_file_location(
        "grounding_dino_server_test", SERVER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _encoded_png(size=(20, 10)):
    buffer = io.BytesIO()
    Image.new("RGB", size, "white").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def test_request_validation_and_image_decode():
    server = _load_server_module()
    request = server.GroundRequest(
        image_base64=_encoded_png(),
        categories=["Chair", "coffee table"],
        box_threshold=0.25,
        text_threshold=0.20,
    )

    assert request.categories == ["chair", "coffee table"]
    assert server._decode_image(request.image_base64).size == (20, 10)

    with pytest.raises(ValidationError):
        server.GroundRequest(image_base64="bad", categories=[])
    with pytest.raises(ValidationError):
        server.GroundRequest(image_base64="bad", categories=["chair", "chair"])
    with pytest.raises(ValueError, match="valid base64"):
        server._decode_image("not base64")


def test_token_counts_use_untruncated_processor_tokenizer():
    server = _load_server_module()
    service = server.GroundingDinoService()

    class Tokenizer:
        def __init__(self):
            self.kwargs = None

        def __call__(self, texts, **kwargs):
            self.kwargs = kwargs
            return {"input_ids": [[1] * (index + 3) for index, _ in enumerate(texts)]}

    tokenizer = Tokenizer()
    service.processor = type("Processor", (), {"tokenizer": tokenizer})()

    assert service.token_counts(["chair.", "chair. table."]) == [3, 4]
    assert tokenizer.kwargs == {"truncation": False, "add_special_tokens": True}


def test_health_exposes_model_identity_and_concurrency(monkeypatch):
    server = _load_server_module()
    monkeypatch.setenv("GROUNDING_DINO_MAX_CONCURRENCY", "1")
    service = server.GroundingDinoService()
    payload = service.health_payload()

    assert payload["service"] == "grounding-dino"
    assert payload["ready"] is False
    assert payload["max_concurrency"] == 1
    assert payload["model"].endswith("grounding-dino-base")
