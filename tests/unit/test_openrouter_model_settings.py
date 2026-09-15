from types import SimpleNamespace

from omegaconf import OmegaConf

from scenesmith.agent_utils.base_stateful_agent import BaseStatefulAgent
from scenesmith.agent_utils.thinking import openrouter_extra_body


def test_openrouter_reasoning_settings_use_sdk_field_and_include_flag() -> None:
    cfg = OmegaConf.create(
        {
            "openai": {
                "model": "openai/gpt-5.6-luna-pro",
                "reasoning_effort": {"designer": "high"},
            }
        }
    )
    agent = SimpleNamespace(
        cfg=cfg,
        _reasoning_request_provider=lambda: "openrouter",
    )

    settings = BaseStatefulAgent._get_model_settings(agent, "designer")

    assert settings is not None
    assert settings.reasoning is not None
    assert settings.reasoning.effort == "high"
    assert settings.extra_body == {"include_reasoning": True}


def test_openrouter_provider_allowlist_is_merged_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("SCENEEXPERT_OPENROUTER_PROVIDER_ONLY", "azure, azure/eu")

    original = {"include_reasoning": True}
    result = openrouter_extra_body(original)

    assert result == {
        "include_reasoning": True,
        "provider": {
            "only": ["azure", "azure/eu"],
            "allow_fallbacks": True,
        },
    }
    assert original == {"include_reasoning": True}


def test_openrouter_provider_allowlist_is_noop_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("SCENEEXPERT_OPENROUTER_PROVIDER_ONLY", raising=False)
    original = {"include_reasoning": True}

    assert openrouter_extra_body(original) is original


def test_openrouter_provider_allowlist_is_noop_when_empty(monkeypatch) -> None:
    monkeypatch.setenv("SCENEEXPERT_OPENROUTER_PROVIDER_ONLY", " , ")
    original = {"include_reasoning": True}

    assert openrouter_extra_body(original) is original
