"""Lossless structured DPO loading and OpenAI/Qwen tool argument compatibility."""
from __future__ import annotations

import copy
import json

import pytest

from scripts.train_sceneexpert_dpo import _load_json_dataset, _normalize_tool_arguments


def _message(arguments):
    return {"role": "assistant", "content": "", "tool_calls": [{
        "id": "call_1", "type": "function",
        "function": {"name": "place_asset", "arguments": arguments},
    }]}


def test_argument_normalization_preserves_values_and_original_messages():
    arguments = {"dimensions": [0.95, 0.35], "enabled": True, "optional": None,
                 "nested": {"name": "椅子", "count": 2}}
    messages = [_message(json.dumps(arguments)), _message(arguments)]
    before = copy.deepcopy(messages)
    result = _normalize_tool_arguments(messages)
    assert result[0]["tool_calls"][0]["function"]["arguments"] == arguments
    assert result[1]["tool_calls"][0]["function"]["arguments"] == arguments
    assert messages == before
    result[1]["tool_calls"][0]["function"]["arguments"]["count"] = 3
    assert messages == before


@pytest.mark.parametrize("arguments", ['[]', 'null', '[1]', 'invalid', 4])
def test_non_object_tool_arguments_fail_before_template_rendering(arguments):
    with pytest.raises((ValueError, TypeError)):
        _normalize_tool_arguments([_message(arguments)])


def test_heterogeneous_dataset_round_trip_and_worker_map(tmp_path):
    pytest.importorskip("datasets")
    first = {"prompt": [{"role": "user", "content": "Place"}, _message('{"asset": "chair"}')],
             "chosen": [_message({"dimensions": [0.95, 0.35], "anchored": True})],
             "rejected": [_message({"position": {"x": 0.6}, "optional": None})],
             "tools": [{"type": "function", "function": {"name": "place_asset",
                "parameters": {"type": "object", "properties": {"dimensions": {"type": "array"}}}}}]}
    second = {"prompt": [{"role": "user", "content": [{"type": "text", "text": "Place"}]}],
              "chosen": [_message({"names": ["chair"], "height": 2})],
              "rejected": [_message('{"other": {"value": 0.6}}')]}
    raw = json.dumps(first) + "\n" + json.dumps(second) + "\n"
    (tmp_path / "train.jsonl").write_text(raw)
    (tmp_path / "validation.jsonl").write_text(json.dumps(first) + "\n")
    expected = copy.deepcopy([first, second])
    for row in expected:
        for side in ("prompt", "chosen", "rejected"):
            row[side] = _normalize_tool_arguments(row[side])
        row.setdefault("tools", [])
        row["chat_template_kwargs"] = {"enable_thinking": False}
    train, validation, features = _load_json_dataset(tmp_path, visible_action_only=True)
    assert list(train) == expected
    assert validation[0] == expected[0]
    assert features == {"has_tools": True, "has_images": False}
    mapped = train.map(lambda row: {"prompt_ids": [1, 2]}, num_proc=2)
    for i, row in enumerate(expected):
        assert {k: mapped[i][k] for k in row} == row
        assert mapped[i]["prompt_ids"] == [1, 2]
    assert list(train.select([1])) == [expected[1]]
    assert (tmp_path / "train.jsonl").read_text() == raw
