from __future__ import annotations

from scenesmith.scene_expert.schemas import SceneTaskSpec
from scenesmith.scene_expert.task_compiler import _task_compiler_wire_schema
from scenesmith.scenebenchmark_critic.intent_schema import (
    intent_compiler_wire_json_schema,
)
from scenesmith.utils.llm_json import json_response_format


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
