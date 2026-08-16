from pathlib import Path

import pytest

from scenesmith.agent_utils.grounding_dino_client import (
    GroundingDinoClient,
    GroundingDinoClientConfig,
    batch_categories,
    build_grounding_prompt,
)


def _word_counter(texts: list[str]) -> list[int]:
    return [len(text.replace(".", "").split()) + 2 for text in texts]


def test_batch_categories_is_stable_and_never_drops_a_phrase():
    categories = ["chair", "coffee table", "bed", "chair", "nightstand"]
    first = batch_categories(
        categories,
        token_budget=5,
        max_batches=8,
        token_count_many=_word_counter,
    )
    second = batch_categories(
        categories,
        token_budget=5,
        max_batches=8,
        token_count_many=_word_counter,
    )

    assert first == second
    assert [item for batch in first for item in batch["categories"]] == [
        "chair",
        "coffee table",
        "bed",
        "nightstand",
    ]
    assert all(batch["token_count"] <= 5 for batch in first)


def test_batch_categories_fails_instead_of_truncating():
    with pytest.raises(ValueError, match="exceeds"):
        batch_categories(
            ["very long category"],
            token_budget=2,
            max_batches=8,
            token_count_many=_word_counter,
        )

    with pytest.raises(ValueError, match="more than 1 batches"):
        batch_categories(
            ["chair", "table"],
            token_budget=3,
            max_batches=1,
            token_count_many=_word_counter,
        )


def test_prompt_has_stable_grounding_separator():
    assert build_grounding_prompt(["Chair", " coffee table "]) == (
        "chair. coffee table."
    )


def test_client_submits_each_exact_batch(tmp_path, monkeypatch):
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"png bytes")
    config = GroundingDinoClientConfig(token_budget=4)
    client = GroundingDinoClient(config)
    posted_categories: list[list[str]] = []

    def fake_request(method, path, payload=None):
        if path == "/v1/token-count":
            return {"token_counts": _word_counter(payload["texts"])}
        posted_categories.append(payload["categories"])
        return {
            "model": "test-model",
            "image_width": 64,
            "image_height": 32,
            "detections": [],
            "inference_ms": 1.0,
        }

    monkeypatch.setattr(client, "_request_json", fake_request)
    result = client.ground_image(image_path, ["chair", "coffee table", "bed"])

    assert posted_categories == [["chair"], ["coffee table"], ["bed"]]
    assert result["image_width"] == 64
    assert [batch["categories"] for batch in result["batches"]] == posted_categories


def test_client_config_rejects_invalid_threshold():
    with pytest.raises(ValueError, match="box_threshold"):
        GroundingDinoClientConfig.from_config({"box_threshold": 1.5})
