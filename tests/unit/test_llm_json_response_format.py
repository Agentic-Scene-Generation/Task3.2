from __future__ import annotations

from copy import deepcopy

import pytest

from scenesmith.scene_expert.schemas import SceneTaskSpec
from scenesmith.scene_expert.task_compiler import _task_compiler_wire_schema
from scenesmith.scenebenchmark_critic.intent_schema import (
    intent_compiler_wire_json_schema,
)
from scenesmith.utils.llm_json import is_llamacpp_model, json_response_format


def _assert_objects_are_strict(schema: object) -> None:
    if isinstance(schema, list):
        for value in schema:
            _assert_objects_are_strict(value)
        return
    if not isinstance(schema, dict):
        return
    if schema.get("type") == "object" and isinstance(schema.get("properties"), dict):
        assert schema["required"] == list(schema["properties"])
        assert schema["additionalProperties"] is False
    for value in schema.values():
        _assert_objects_are_strict(value)


def test_openai_schema_requires_every_pydantic_property() -> None:
    response_format = json_response_format(
        model="gpt-5.5",
        name="scene_task_spec",
        schema=SceneTaskSpec.model_json_schema(),
    )

    assert response_format["json_schema"]["strict"] is True
    _assert_objects_are_strict(response_format["json_schema"]["schema"])
    assert (
        "aesthetic_constraints" in response_format["json_schema"]["schema"]["required"]
    )


def test_task_compiler_wire_schema_excludes_internal_metadata() -> None:
    schema = _task_compiler_wire_schema()

    assert "requirement_sources" not in schema["properties"]
    assert schema["required"] == list(schema["properties"])


def test_openai_schema_normalizes_intent_nested_defaults() -> None:
    response_format = json_response_format(
        model="gpt-5.5",
        name="intent_contract",
        schema=intent_compiler_wire_json_schema(),
    )

    _assert_objects_are_strict(response_format["json_schema"]["schema"])
    group = response_format["json_schema"]["schema"]["$defs"]["EdgeDistributionGroup"]
    assert "spacing" in group["required"]


def test_llamacpp_keeps_loose_schema_mode() -> None:
    schema = SceneTaskSpec.model_json_schema()
    response_format = json_response_format(
        model="Qwen/Qwen3.5-35B-A3B",
        name="scene_task_spec",
        schema=schema,
    )

    assert "strict" not in response_format["json_schema"]
    assert (
        "aesthetic_constraints"
        not in response_format["json_schema"]["schema"]["required"]
    )


@pytest.mark.parametrize(
    "model",
    [
        "unsloth/Qwen3.8-27B-GGUF",
        "org/qwen3-27b",
        "meta-llama/Llama-3.1-8B-Instruct",
        "Qwen/Qwen3.5-35B-A3B",
        "llama.cpp",
    ],
)
def test_namespaced_local_models_preserve_optional_intent_fields(model: str) -> None:
    schema = intent_compiler_wire_json_schema()
    original = deepcopy(schema)

    response_format = json_response_format(
        model=model, name="compiler_semantic_ir", schema=schema
    )

    assert is_llamacpp_model(model)
    assert "strict" not in response_format["json_schema"]
    assert response_format["json_schema"]["schema"] == original
    requirement = response_format["json_schema"]["schema"]["properties"][
        "requirements"
    ]["items"]
    assert requirement["required"] == [
        "requirement_id",
        "kind",
        "grounding",
        "target_ref",
    ]
    assert "relation" not in requirement["required"]
    assert "subject_cohort" not in requirement["required"]
    assert schema == original


@pytest.mark.parametrize("model", ["gpt-5.5", "tenant/gpt-5.5", "openai/gpt-4o"])
def test_namespaced_openai_models_keep_strict_schema(model: str) -> None:
    schema = intent_compiler_wire_json_schema()
    original = deepcopy(schema)

    response_format = json_response_format(
        model=model, name="compiler_semantic_ir", schema=schema
    )

    assert not is_llamacpp_model(model)
    assert response_format["json_schema"]["strict"] is True
    _assert_objects_are_strict(response_format["json_schema"]["schema"])
    assert schema == original
