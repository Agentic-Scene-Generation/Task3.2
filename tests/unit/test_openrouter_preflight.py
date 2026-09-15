from scripts.preflight_openrouter_runtime import _apply_provider_routing


def test_preflight_uses_automatic_routing_when_allowlist_is_empty() -> None:
    payload = {"model": "openai/gpt-5.6-luna-pro"}

    result = _apply_provider_routing(payload, "")

    assert result is payload
    assert "provider" not in result


def test_preflight_adds_explicit_provider_allowlist() -> None:
    payload = {"model": "openai/gpt-5.6-luna-pro"}

    result = _apply_provider_routing(payload, " azure, azure/eu ")

    assert result["provider"] == {
        "only": ["azure", "azure/eu"],
        "allow_fallbacks": True,
    }
