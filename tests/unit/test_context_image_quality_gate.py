import base64
import io
import json

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from PIL import Image

from scenesmith.agent_utils.context_image_generation import (
    OpenAICompatibleContextImageEditor,
)
from scenesmith.agent_utils.context_image_quality import (
    ContextImageQualityEvaluator,
    ContextImageQualityGateConfig,
    ContextImageQualityResult,
    evaluate_context_image_deterministic,
    write_context_image_quality_report,
)
from scenesmith.prompts import prompt_manager
from scenesmith.prompts.registry import ImageGenerationPrompts


def _quality_payload(**overrides):
    payload = {
        "passed": True,
        "quality_score": 75,
        "doors_windows_preserved": True,
        "room_geometry_preserved": True,
        "view_preserved": True,
        "rendering_style_preserved": True,
        "furniture_inside_room": True,
        "openings_clear": True,
        "furniture_quality_ok": True,
        "reasons": [],
    }
    payload.update(overrides)
    return payload


def _quality_result(*, score: float, **overrides) -> ContextImageQualityResult:
    payload = _quality_payload(
        passed=score >= 60,
        quality_score=score,
        **overrides,
    )
    return ContextImageQualityResult.from_response_text(
        json.dumps(payload),
        min_score=60,
    )


def _png_b64(size: tuple[int, int]) -> str:
    buffer = io.BytesIO()
    Image.new("RGB", size, color=(80, 100, 120)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _qwen_config():
    return {
        "base_url": "http://127.0.0.1:18020/v1",
        "api_key": "not-needed",
        "model": "Qwen/Qwen-Image-Edit",
        "size": "auto",
        "num_inference_steps": 50,
        "true_cfg_scale": 4.0,
        "negative_prompt": " ",
        "seed": 4,
        "timeout_seconds": 120,
        "connect_timeout_seconds": 5,
        "max_retries": 0,
        "write_metadata": True,
    }


class _FakePromptManager:
    def __init__(self, prompt: str = "quality prompt") -> None:
        self.prompt = prompt
        self.calls = []

    def get_prompt(self, prompt_name, **kwargs):
        self.calls.append((prompt_name, kwargs))
        return self.prompt


class _IndependentEditor:
    def __init__(self, seed: int = 10) -> None:
        self.config = SimpleNamespace(seed=seed)
        self.calls: list[dict] = []

    def generate_furniture_context_image(self, **kwargs):
        self.calls.append(kwargs)
        output_path = Path(kwargs["output_path"])
        seed = int(kwargs["seed_override"])
        image = Image.new("RGB", (12, 12), color=(seed % 255, 30, 40))
        image.putpixel((6, 6), (220, 210, 200))
        image.save(output_path)
        output_path.with_suffix(".metadata.json").write_text(
            json.dumps({"seed": seed}),
            encoding="utf-8",
        )
        return output_path


def _agent_for_quality_workflow(module):
    agent = module.StatefulFurnitureAgent.__new__(module.StatefulFurnitureAgent)
    agent.vlm_service = Mock()
    agent.cfg = SimpleNamespace(
        openai=SimpleNamespace(
            model="test-vlm",
            reasoning_effort=SimpleNamespace(asset_validation="none"),
            verbosity=SimpleNamespace(asset_validation="low"),
        )
    )
    return agent


def _scene():
    return SimpleNamespace(
        text_description="a formal dining room",
        room_geometry=SimpleNamespace(width=6.0, length=8.0),
    )


def test_quality_gate_config_counts_attempts_and_validates_score():
    config = ContextImageQualityGateConfig.from_config(
        {"enabled": True, "max_regenerations": 2, "min_score": 65}
    )
    assert config.enabled is True
    assert config.max_attempts == 3
    assert config.min_score == 65

    with pytest.raises(ValueError, match="between 0 and 10"):
        ContextImageQualityGateConfig.from_config(
            {"enabled": True, "max_regenerations": 11}
        )

    with pytest.raises(ValueError, match="must be an integer"):
        ContextImageQualityGateConfig.from_config(
            {"enabled": True, "max_regenerations": 1.5}
        )

    with pytest.raises(ValueError, match="between 0 and 100"):
        ContextImageQualityGateConfig.from_config({"enabled": True, "min_score": 101})


def test_quality_result_rejects_structural_failure_even_with_high_score():
    result = ContextImageQualityResult.from_response_text(
        json.dumps(
            _quality_payload(
                passed=True,
                quality_score=95,
                doors_windows_preserved=False,
                reasons=["The left window moved."],
            )
        ),
        min_score=60,
    )

    assert result.model_passed is True
    assert result.fallback_eligible is False
    assert result.passed is False
    assert result.reasons == ("The left window moved.",)


def test_quality_result_rejects_style_drift_even_with_high_score():
    result = ContextImageQualityResult.from_response_text(
        json.dumps(
            _quality_payload(
                passed=True,
                quality_score=95,
                rendering_style_preserved=False,
                reasons=["Furniture is cartoon line-art rather than Blender-rendered."],
            )
        ),
        min_score=60,
    )

    assert result.fallback_eligible is False
    assert result.passed is False


def test_quality_result_treats_opening_clearance_and_model_vote_as_soft():
    result = ContextImageQualityResult.from_response_text(
        json.dumps(
            _quality_payload(
                passed=False,
                quality_score=70,
                openings_clear=False,
                reasons=["A chair is close to the entry path."],
            )
        ),
        min_score=60,
    )

    assert result.model_passed is False
    assert result.fallback_eligible is True
    assert result.passed is True

    low_score = ContextImageQualityResult.from_response_text(
        json.dumps(_quality_payload(passed=False, quality_score=30)),
        min_score=60,
    )
    assert low_score.fallback_eligible is False
    assert low_score.passed is False


def test_quality_result_requires_all_structured_fields_and_score():
    payload = _quality_payload()
    del payload["openings_clear"]
    with pytest.raises(ValueError, match="openings_clear"):
        ContextImageQualityResult.from_response_text(json.dumps(payload))


def test_furniture_quality_is_utility_not_structural_eligibility():
    result = ContextImageQualityResult.from_response_text(
        json.dumps(
            _quality_payload(
                quality_score=70,
                furniture_quality_ok=False,
                layout_utility_score=35,
                grounding_utility_score=45,
            )
        ),
        min_score=60,
    )

    assert result.passed is True
    assert result.fallback_eligible is True
    assert result.grounding_quality_mode == "full_reference"

    payload = _quality_payload()
    del payload["rendering_style_preserved"]
    with pytest.raises(ValueError, match="rendering_style_preserved"):
        ContextImageQualityResult.from_response_text(json.dumps(payload))

    payload = _quality_payload()
    del payload["quality_score"]
    with pytest.raises(ValueError, match="quality_score"):
        ContextImageQualityResult.from_response_text(json.dumps(payload))


def test_evaluator_submits_original_then_candidate_with_high_detail(tmp_path):
    original = tmp_path / "0_top.png"
    candidate = tmp_path / "candidate.png"
    Image.new("RGB", (16, 16), color="white").save(original)
    Image.new("RGB", (16, 16), color="gray").save(candidate)

    vlm_service = Mock()
    vlm_service.create_completion.return_value = json.dumps(_quality_payload())
    prompts = _FakePromptManager()
    evaluator = ContextImageQualityEvaluator(vlm_service, prompts=prompts)

    result = evaluator.evaluate(
        original_image_path=original,
        candidate_image_path=candidate,
        scene_description="a compact bedroom",
        model="test-vlm",
        min_score=60,
    )

    assert result.passed is True
    assert prompts.calls[0][1]["min_score"] == 60
    request = vlm_service.create_completion.call_args.kwargs
    assert request["response_format"] == {"type": "json_object"}
    assert request["vision_detail"] == "high"
    content = request["messages"][0]["content"]
    assert content[1]["text"] == "ORIGINAL EMPTY ROOM:"
    assert content[3]["text"] == "EDITED CANDIDATE:"
    images = [item for item in content if item["type"] == "image_url"]
    assert len(images) == 2
    assert all(
        item["image_url"]["url"].startswith("data:image/png;base64,") for item in images
    )


def test_prompts_use_relaxed_quality_rules_without_retry_feedback():
    edit_prompt = prompt_manager.get_prompt(
        ImageGenerationPrompts.FURNITURE_CONTEXT_IMAGE,
        scene_description="a bedroom",
        width_m=4.0,
        length_m=5.0,
    )
    quality_prompt = prompt_manager.get_prompt(
        ImageGenerationPrompts.FURNITURE_CONTEXT_IMAGE_QUALITY,
        scene_description="a bedroom",
        min_score=60,
    )

    assert "same overhead camera" in edit_prompt
    assert "HARD CONSTRAINTS" not in edit_prompt
    assert "FUNCTIONAL LAYOUT CONSTRAINTS" not in edit_prompt
    assert "A previous candidate was rejected" not in edit_prompt
    assert "not rejection reasons by themselves" in quality_prompt
    assert "Do not provide correction instructions" in quality_prompt


def test_context_image_description_excludes_scene_expert_stage_brief():
    module = pytest.importorskip("scenesmith.furniture_agents.stateful_furniture_agent")
    agent = module.StatefulFurnitureAgent.__new__(module.StatefulFurnitureAgent)
    agent._layout_constraint_contract_text = "Layout contract: keep openings clear."
    scene = SimpleNamespace(
        scene_expert_original_description="A bedroom with one bed.",
        text_description=(
            "A bedroom with one bed.\n\n"
            "=== SceneExpert Stage Brief: furniture ===\nInternal workflow details"
        ),
    )

    description = agent._context_scene_description(scene)

    assert description.startswith("A bedroom with one bed.\n\nLayout contract:")
    assert "SceneExpert Stage Brief" not in description
    assert "Internal workflow details" not in description


def test_qwen_independent_sample_uses_seed_override_without_feedback(tmp_path):
    original = tmp_path / "0_top.png"
    output = tmp_path / "context_edited_attempt_02.png"
    Image.new("RGB", (32, 24), color="white").save(original)
    response = SimpleNamespace(
        data=[SimpleNamespace(b64_json=_png_b64((32, 24)))],
        model_dump=lambda: {},
    )
    client = Mock()
    client.images.edit.return_value = response
    prompts = _FakePromptManager("edit prompt")
    editor = OpenAICompatibleContextImageEditor(
        _qwen_config(), client=client, prompt_manager=prompts
    )

    editor.generate_furniture_context_image(
        reference_image_path=original,
        scene_description="a bedroom",
        width_m=4.0,
        length_m=5.0,
        output_path=output,
        seed_override=5,
    )

    assert "retry_feedback" not in prompts.calls[0][1]
    assert client.images.edit.call_args.kwargs["extra_body"]["seed"] == 5
    metadata = json.loads(output.with_suffix(".metadata.json").read_text())
    assert metadata["seed"] == 5


def test_best_candidate_survives_lower_score_and_judge_error(tmp_path):
    module = pytest.importorskip("scenesmith.furniture_agents.stateful_furniture_agent")
    agent = _agent_for_quality_workflow(module)
    editor = _IndependentEditor(seed=10)
    evaluator = Mock()
    evaluator.evaluate.side_effect = [
        _quality_result(score=55),
        _quality_result(score=40, openings_clear=False),
        ValueError("invalid judge JSON"),
    ]
    room_render = tmp_path / "0_top.png"
    output_path = tmp_path / "context_edited.png"
    Image.new("RGB", (12, 12), color="white").save(room_render)
    config = ContextImageQualityGateConfig(
        enabled=True,
        max_regenerations=2,
        min_score=60,
    )

    with patch.object(
        module,
        "ContextImageQualityEvaluator",
        return_value=evaluator,
    ):
        result = agent._generate_quality_gated_context_image(
            scene=_scene(),
            image_editor=editor,
            room_render=room_render,
            output_path=output_path,
            quality_gate=config,
        )

    assert result == output_path
    assert (
        output_path.read_bytes()
        == (tmp_path / "context_edited_attempt_01.png").read_bytes()
    )
    report = json.loads((tmp_path / "context_image_quality.json").read_text())
    assert report["final_status"] == "best_candidate_fallback"
    assert report["accepted_attempt"] == 1
    assert report["best_attempt"] == 1
    assert report["best_score"] == 55
    assert report["selection_mode"] == "best_effort_fallback"
    assert [call["seed_override"] for call in editor.calls] == [10, 11, 12]
    assert all(call["reference_image_path"] == room_render for call in editor.calls)
    assert all("retry_feedback" not in call for call in editor.calls)


def test_normal_pass_scores_all_independent_candidates(tmp_path):
    module = pytest.importorskip("scenesmith.furniture_agents.stateful_furniture_agent")
    agent = _agent_for_quality_workflow(module)
    editor = _IndependentEditor()
    evaluator = Mock()
    evaluator.evaluate.return_value = _quality_result(score=75)
    room_render = tmp_path / "0_top.png"
    output_path = tmp_path / "context_edited.png"
    Image.new("RGB", (12, 12), color="white").save(room_render)

    with patch.object(
        module,
        "ContextImageQualityEvaluator",
        return_value=evaluator,
    ):
        result = agent._generate_quality_gated_context_image(
            scene=_scene(),
            image_editor=editor,
            room_render=room_render,
            output_path=output_path,
            quality_gate=ContextImageQualityGateConfig(
                enabled=True,
                max_regenerations=2,
                min_score=60,
            ),
        )

    assert result == output_path
    assert len(editor.calls) == 3
    report = json.loads((tmp_path / "context_image_quality.json").read_text())
    assert report["final_status"] == "accepted"
    assert report["selection_mode"] == "best_scored_pass"
    assert report["grounding_candidate_path"].endswith("context_edited_attempt_01.png")


def test_all_structurally_invalid_candidates_do_not_publish_reference(tmp_path):
    module = pytest.importorskip("scenesmith.furniture_agents.stateful_furniture_agent")
    agent = _agent_for_quality_workflow(module)
    editor = _IndependentEditor()
    evaluator = Mock()
    evaluator.evaluate.side_effect = [
        _quality_result(score=65, doors_windows_preserved=False),
        _quality_result(score=90, doors_windows_preserved=False),
    ]
    room_render = tmp_path / "0_top.png"
    output_path = tmp_path / "context_edited.png"
    Image.new("RGB", (12, 12), color="white").save(room_render)

    with patch.object(
        module,
        "ContextImageQualityEvaluator",
        return_value=evaluator,
    ):
        result = agent._generate_quality_gated_context_image(
            scene=_scene(),
            image_editor=editor,
            room_render=room_render,
            output_path=output_path,
            quality_gate=ContextImageQualityGateConfig(
                enabled=True,
                max_regenerations=1,
                min_score=60,
            ),
        )

    assert result is None
    assert not output_path.exists()
    report = json.loads((tmp_path / "context_image_quality.json").read_text())
    assert report["final_status"] == "no_eligible_candidate"
    assert report["accepted_attempt"] is None
    assert report["best_attempt"] is None
    assert report["best_score"] is None
    assert report["selection_mode"] == "none"
    assert report["designer_reference_path"] is None
    assert report["designer_visual_reference_path"] is None
    assert report["grounding_candidate_path"].endswith("context_edited_attempt_02.png")
    assert report["grounding_quality_mode"] == "relations_only"


def test_quality_report_is_valid_json(tmp_path):
    report_path = tmp_path / "context_image_quality.json"
    payload = {"final_status": "accepted", "accepted_attempt": 2}
    write_context_image_quality_report(report_path, payload)
    assert json.loads(report_path.read_text()) == payload


def test_deterministic_gate_rejects_repainted_exterior(tmp_path):
    original = tmp_path / "original.png"
    candidate = tmp_path / "candidate.png"
    original_image = Image.new("RGB", (64, 64), "black")
    for x in range(16, 48):
        for y in range(16, 48):
            original_image.putpixel((x, y), (100, 100, 100))
    original_image.save(original)
    Image.new("RGB", (64, 64), "white").save(candidate)

    result = evaluate_context_image_deterministic(original, candidate)

    assert result["passed"] is False
    assert result["background_preserved"] is False
