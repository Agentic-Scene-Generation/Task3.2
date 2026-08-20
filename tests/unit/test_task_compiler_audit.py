"""Audit coverage for TaskCompiler input and output persistence."""

import json

from types import SimpleNamespace

from scenesmith.scene_expert.task_compiler import TaskCompiler


def test_task_compiler_records_full_input_and_output(tmp_path, monkeypatch) -> None:
    raw_output = json.dumps(
        {
            "room_type": "bedroom",
            "style": "standard",
            "required_large_objects": ["bed", "nightstand", "wardrobe"],
        }
    )
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=raw_output))],
        usage=SimpleNamespace(
            prompt_tokens=120,
            completion_tokens=40,
            total_tokens=160,
            prompt_tokens_details=SimpleNamespace(cached_tokens=20),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=15),
        ),
    )
    captured_request = {}

    def create(**kwargs):
        captured_request.update(kwargs)
        return response

    compiler = object.__new__(TaskCompiler)
    compiler._model = "test-model"
    compiler._max_tokens = 512
    compiler._temperature = 0.0
    compiler._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    audit_path = tmp_path / "scene_expert_llm_calls.jsonl"
    monkeypatch.setenv("SCENEEXPERT_LLM_DEBUG_PATH", str(audit_path))

    result = compiler.compile("A bedroom with a bed and wardrobe.")

    record = json.loads(audit_path.read_text(encoding="utf-8"))
    assert result.room_type == "bedroom"
    assert record["agent_role"] == "task_compiler"
    assert record["status"] == "ok"
    assert record["input"] == captured_request["messages"]
    assert record["input"][0]["role"] == "system"
    assert record["input"][1]["content"].endswith("A bedroom with a bed and wardrobe.")
    assert record["output"] == raw_output
    assert record["elapsed_sec"] >= 0
    assert record["token_usage"]["total_tokens"] == 160
    assert record["token_usage"]["input_non_cached_tokens"] == 100
    assert record["token_usage"]["output_reasoning_tokens"] == 15
    assert record["token_usage"]["output_text_tokens"] == 25
    assert record["schema_version"] == "scenesmith.llm_call_debug.v2"
