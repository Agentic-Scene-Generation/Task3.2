import base64
import io

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from PIL import Image

from scenesmith.agent_utils.context_image_generation import (
    OpenAICompatibleContextImageEditor,
    QwenLocalContextImageConfig,
)
def _png_b64(size: tuple[int, int]) -> str:
    buffer = io.BytesIO()
    Image.new("RGB", size, color=(120, 130, 140)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _config(**overrides):
    values = {
        "base_url": "http://127.0.0.1:18020/v1",
        "api_key": "not-needed",
        "model": "Qwen/Qwen-Image-Edit",
        "size": "auto",
        "num_inference_steps": 50,
        "true_cfg_scale": 4.0,
        "negative_prompt": " ",
        "seed": 0,
        "timeout_seconds": 120,
        "connect_timeout_seconds": 5,
        "max_retries": 0,
        "write_metadata": True,
    }
    values.update(overrides)
    return values


class _FakePromptManager:
    def get_prompt(self, *_args, **_kwargs):
        return "Add furniture while preserving the room geometry."


def test_local_editor_preserves_render_dimensions_and_writes_metadata(tmp_path):
    input_path = tmp_path / "0_top.png"
    output_path = tmp_path / "context_edited.png"
    Image.new("RGB", (768, 512), color="white").save(input_path)

    server_metadata = {
        "request_id": "qie_test",
        "input_width": 768,
        "input_height": 512,
        "output_width": 768,
        "output_height": 512,
        "inference_seconds": 1.25,
    }
    response = SimpleNamespace(
        data=[SimpleNamespace(b64_json=_png_b64((768, 512)))],
        model_dump=lambda: {"x_qwen_image_edit": server_metadata},
    )
    client = Mock()
    client.images.edit.return_value = response
    editor = OpenAICompatibleContextImageEditor(
        _config(),
        client=client,
        prompt_manager=_FakePromptManager(),
    )

    result = editor.generate_furniture_context_image(
        reference_image_path=input_path,
        scene_description="a living room",
        width_m=6.0,
        length_m=8.0,
        output_path=output_path,
    )

    assert result == output_path
    assert Image.open(output_path).size == (768, 512)
    request = client.images.edit.call_args.kwargs
    assert request["size"] == "auto"
    assert request["extra_body"]["num_inference_steps"] == 50
    metadata = output_path.with_suffix(".metadata.json").read_text()
    assert '"success": true' in metadata
    assert '"output_width": 768' in metadata


def test_local_editor_rejects_output_dimension_mismatch(tmp_path):
    input_path = tmp_path / "0_top.png"
    output_path = tmp_path / "context_edited.png"
    Image.new("RGB", (768, 768), color="white").save(input_path)
    client = Mock()
    client.images.edit.return_value = SimpleNamespace(
        data=[SimpleNamespace(b64_json=_png_b64((512, 512)))],
        model_dump=lambda: {},
    )
    editor = OpenAICompatibleContextImageEditor(
        _config(),
        client=client,
        prompt_manager=_FakePromptManager(),
    )

    with pytest.raises(ValueError, match="does not match"):
        editor.generate_furniture_context_image(
            reference_image_path=input_path,
            scene_description="a living room",
            width_m=6.0,
            length_m=8.0,
            output_path=output_path,
        )

    assert not output_path.exists()
    assert '"success": false' in output_path.with_suffix(
        ".metadata.json"
    ).read_text()


def test_size_must_be_auto():
    with pytest.raises(ValueError, match="must be 'auto'"):
        QwenLocalContextImageConfig.from_config(_config(size="512x512"))


def test_inherit_returns_existing_image_generator_without_local_client():
    module = pytest.importorskip(
        "scenesmith.furniture_agents.stateful_furniture_agent"
    )
    StatefulFurnitureAgent = module.StatefulFurnitureAgent
    agent = StatefulFurnitureAgent.__new__(StatefulFurnitureAgent)
    existing_generator = object()
    agent.cfg = SimpleNamespace(
        context_image_generation={
            "enabled": True,
            "backend": "inherit",
        }
    )
    agent.asset_manager = SimpleNamespace(image_generator=existing_generator)
    agent._qwen_context_image_editor = None

    with patch(
        "scenesmith.furniture_agents.stateful_furniture_agent."
        "OpenAICompatibleContextImageEditor"
    ) as local_editor:
        assert agent._get_context_image_editor() is existing_generator
        local_editor.assert_not_called()


def test_qwen_editor_is_lazy_and_reused():
    module = pytest.importorskip(
        "scenesmith.furniture_agents.stateful_furniture_agent"
    )
    StatefulFurnitureAgent = module.StatefulFurnitureAgent
    agent = StatefulFurnitureAgent.__new__(StatefulFurnitureAgent)
    agent.cfg = SimpleNamespace(
        context_image_generation={
            "enabled": True,
            "backend": "qwen_local",
            "qwen_local": _config(),
        }
    )
    agent.asset_manager = SimpleNamespace(image_generator=object())
    agent._qwen_context_image_editor = None

    with patch(
        "scenesmith.furniture_agents.stateful_furniture_agent."
        "OpenAICompatibleContextImageEditor"
    ) as local_editor:
        created = object()
        local_editor.return_value = created
        assert agent._get_context_image_editor() is created
        assert agent._get_context_image_editor() is created
        local_editor.assert_called_once()


def test_missing_backend_defaults_to_inherit():
    module = pytest.importorskip(
        "scenesmith.furniture_agents.stateful_furniture_agent"
    )
    StatefulFurnitureAgent = module.StatefulFurnitureAgent
    agent = StatefulFurnitureAgent.__new__(StatefulFurnitureAgent)
    existing_generator = object()
    agent.cfg = SimpleNamespace(
        context_image_generation={"enabled": True}
    )
    agent.asset_manager = SimpleNamespace(image_generator=existing_generator)
    agent._qwen_context_image_editor = None
    assert agent._get_context_image_editor() is existing_generator
