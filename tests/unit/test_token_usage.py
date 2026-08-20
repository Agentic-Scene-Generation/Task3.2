from types import SimpleNamespace

from scenesmith.utils.token_usage import normalize_token_usage


def test_normalizes_agents_sdk_reasoning_cache_and_peak_context() -> None:
    result = SimpleNamespace(
        context_wrapper=SimpleNamespace(
            usage=SimpleNamespace(
                input_tokens=120,
                output_tokens=80,
                total_tokens=200,
                requests=3,
                input_tokens_details=SimpleNamespace(cached_tokens=30),
                output_tokens_details=SimpleNamespace(reasoning_tokens=95),
                request_usage_entries=[
                    SimpleNamespace(input_tokens=60),
                    SimpleNamespace(input_tokens=140),
                    SimpleNamespace(input_tokens=100),
                ],
            )
        )
    )

    assert normalize_token_usage(result) == {
        "input_tokens": 120,
        "input_cached_tokens": 30,
        "input_non_cached_tokens": 90,
        "output_tokens": 80,
        "output_reasoning_tokens": 95,
        "output_text_tokens": 0,
        "total_tokens": 200,
        "requests": 3,
        "final_input_context_tokens": 100,
        "max_input_context_tokens": 140,
    }


def test_normalizes_flat_openai_usage_without_inventing_missing_breakdown() -> None:
    usage = {
        "prompt_tokens": 120,
        "completion_tokens": 40,
        "total_tokens": 160,
        "prompt_tokens_details": {"cached_tokens": 20},
        "completion_tokens_details": {"reasoning_tokens": 15},
    }

    assert normalize_token_usage(usage) == {
        "input_tokens": 120,
        "input_cached_tokens": 20,
        "input_non_cached_tokens": 100,
        "output_tokens": 40,
        "output_reasoning_tokens": 15,
        "output_text_tokens": 25,
        "total_tokens": 160,
    }
