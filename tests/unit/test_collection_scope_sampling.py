"""Scope-conditioned native SDK execution, immutable requests and task ownership."""
from __future__ import annotations

import asyncio
import copy
import json
from types import SimpleNamespace

import pytest

pytest.importorskip("agents")
from agents import Agent, Runner, function_tool
from agents.items import ModelResponse
from agents.models.interface import Model
from agents.usage import Usage
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText

from scenesmith.scene_expert.slow_memory.scope_sampling import (
    ShadowScopeSampler, ScopeSamplingModel, sampling_policy, validate_sampling_proof,
)
from scenesmith.scene_expert.slow_memory.paired import EXPECTED_MODEL, read_json, write_json
from scenesmith.scene_expert.slow_memory.paired_runtime import json_value
from scenesmith.scene_expert.schemas import SceneTaskSpec
from scenesmith.scene_expert.task_compiler import _normalize_stage_ownership
from scenesmith.scenebenchmark_critic.intent_compiler import IntentCompiler
from scenesmith.scenebenchmark_critic.object_taxonomy import canonical_object_category
from scripts.run_sceneexpert_campaign import select_task_filter


def response(name, ordinal):
    if name == "done":
        output = [ResponseOutputMessage(type="message", id="done", role="assistant", status="completed",
            content=[ResponseOutputText(type="output_text", text="done", annotations=[])])]
    else:
        output = [ResponseFunctionToolCall(type="function_call", id=f"id_{ordinal}",
            call_id=f"call_{ordinal}", name=name, arguments="{}")]
    return ModelResponse(output=output, usage=Usage(), response_id=None)


class Scripted(Model):
    def __init__(self, sampler, names):
        self.sampler, self.names, self.inputs = sampler, iter(names), []

    async def get_response(self, *args, **kwargs):
        self.inputs.append(copy.deepcopy(kwargs.get("input", args[1] if len(args) > 1 else [])))
        self.sampler.observe_request(self.sampler.expected)
        return response(next(self.names), len(self.inputs))

    def stream_response(self, *args, **kwargs):
        raise AssertionError("native non-streaming contract")


def request():
    return {"model": EXPECTED_MODEL, "messages": [{"role": "user", "content": "frozen scene"}],
            "tools": [{"type": "function", "function": {"name": "generate_assets"}}]}


def test_native_runner_executes_only_the_selected_first_response(tmp_path):
    executed = []
    @function_tool
    def designer_todo_manager() -> str:
        executed.append("todo")
        return "todo mutated"
    @function_tool
    def generate_assets() -> str:
        executed.append("design")
        return "asset generated"
    sampler = ShadowScopeSampler(tmp_path, request(), sampling_policy(3))
    native = Scripted(sampler, ["designer_todo_manager", "generate_assets", "done"])
    agent = Agent(name="designer", model=ScopeSamplingModel(native, sampler),
                  tools=[designer_todo_manager, generate_assets])
    result = asyncio.run(Runner.run(agent, input="frozen scene", max_turns=3))
    assert executed == ["design"]
    assert native.inputs[0] == native.inputs[1]
    assert len(result.raw_responses) == 2  # selected action and normal continuation only
    proof = read_json(tmp_path / "scope_sampling.json")
    assert proof["selected_ordinal"] == 1
    assert len(proof["attempts"]) == 2
    validate_sampling_proof(tmp_path, sampling_policy(3), request(), json_value(result.raw_responses[0].output))
    with pytest.raises(ValueError, match="executed first response"):
        validate_sampling_proof(tmp_path, sampling_policy(3), request(), [])
    # Rejected first draws stay on disk, never become the executed completion.
    rejected = read_json(tmp_path / "scope_samples/sample_000.json")
    assert rejected["response_items"][0]["name"] == "designer_todo_manager"
    assert not rejected["tools_executed_before_selection"]


@pytest.mark.parametrize("budget", [1, 3])
def test_exhausted_scope_budget_returns_last_draw_without_forcing_actions(tmp_path, budget):
    sampler = ShadowScopeSampler(tmp_path, request(), sampling_policy(budget))
    native = Scripted(sampler, ["designer_todo_manager"] * budget)
    actual = asyncio.run(ScopeSamplingModel(native, sampler).get_response(input=[]))
    proof = read_json(tmp_path / "scope_sampling.json")
    assert len(native.inputs) == budget and len(proof["attempts"]) == budget
    assert not proof["selected_has_direct_design"]
    validate_sampling_proof(tmp_path, sampling_policy(budget), request(), json_value(actual.output))
    assert all(read_json(tmp_path / entry["path"])["requests"] == [request()] for entry in proof["attempts"])
    record = read_json(tmp_path / "scope_samples/sample_000.json")
    record["requests"][0]["messages"] = []
    write_json(tmp_path / "scope_samples/sample_000.json", record)
    with pytest.raises(ValueError, match="draw proof"):
        validate_sampling_proof(tmp_path, sampling_policy(budget), request())


def test_scope_request_drift_fails_before_a_second_response(tmp_path):
    sampler = ShadowScopeSampler(tmp_path, request(), sampling_policy(3))
    class Drift(Scripted):
        async def get_response(self, *args, **kwargs):
            payload = request()
            if self.inputs:
                payload["tools"] = []
            self.sampler.observe_request(payload)
            return await super().get_response(*args, **kwargs)
    native = Drift(sampler, ["designer_todo_manager", "generate_assets"])
    with pytest.raises(ValueError, match="differs before execution"):
        asyncio.run(ScopeSamplingModel(native, sampler).get_response(input=[]))
    assert len(native.inputs) == 1
    assert not (tmp_path / "scope_sampling.json").exists()


@pytest.mark.parametrize("budget", [0, 4])
def test_sampling_budget_is_bounded(budget):
    with pytest.raises(ValueError, match="1..3"):
        sampling_policy(budget)


def test_portable_architectural_misclassification_and_screen_identity():
    raw = SceneTaskSpec(room_type="den", style="zen", required_large_objects=["table"],
        required_small_objects=["bamboo screen", "computer monitor"],
        required_architectural_features=["circular jute rug", "window", "arched alcove"])
    before = raw.model_dump()
    normalized = _normalize_stage_ownership(raw)
    assert "rug" in normalized.required_large_objects
    assert "room_divider" in normalized.required_large_objects
    assert normalized.required_small_objects == ["monitor"]
    assert normalized.required_architectural_features == ["window", "arched alcove"]
    assert raw.model_dump() == before
    again = _normalize_stage_ownership(normalized)
    assert again.model_dump() == normalized.model_dump()
    for name in ("bamboo screen", "folding screen", "privacy screen"):
        assert canonical_object_category(name) == "room_divider"
    assert canonical_object_category("computer screen") == "monitor"
    assert canonical_object_category("projection screen") == "instructional_surface"


def test_inventory_message_requires_its_catalog_endpoint_and_specific_retry():
    compiler = object.__new__(IntentCompiler)
    compiler._model = EXPECTED_MODEL
    messages = compiler._messages("A room with a table.",
        entity_catalog='[{"entity_ref":"inventory:table"}]',
        validation_error="inventory requires an inventory subject_ref")
    text = messages[-1]["content"]
    assert 'kind=inventory requires subject_ref=inventory:<category>' in text
    assert 'Set subject_ref to the exact inventory:<category>' in text
    assert 'Only kind=relation uses relation, subject_ref' not in text


def test_mechanism_filter_keeps_task_splits_and_rejects_reserved_test():
    plan = {"tasks": [{"sceneeval_id": 336, "split": "train"},
                      {"sceneeval_id": 273, "split": "validation"},
                      {"sceneeval_id": 499, "split": "test"}]}
    before = copy.deepcopy(plan)
    assert select_task_filter(plan, "336,273") == {336,273}
    assert select_task_filter(plan, "") is None
    with pytest.raises(ValueError, match="reserved test"):
        select_task_filter(plan, "499")
    with pytest.raises(ValueError, match="outside"):
        select_task_filter(plan, "100")
    assert plan == before


def test_wire_hook_checks_every_sample_after_first_request_exists(tmp_path):
    import httpx
    from scenesmith.scene_expert.slow_memory.paired_wire import capture_wire, client_options
    sampler = ShadowScopeSampler(tmp_path, request(), sampling_policy(3))
    with capture_wire(tmp_path, expected_request=request(), scope_sampler=sampler):
        options = client_options()
    hook = options["http_client"].event_hooks["request"][0]
    sampler.active = True
    asyncio.run(hook(httpx.Request("POST", "http://localhost/v1/chat/completions", json=request())))
    assert (tmp_path / "first_request.json").exists()
    with pytest.raises(ValueError, match="scope sampling request differs"):
        asyncio.run(hook(httpx.Request("POST", "http://localhost/v1/chat/completions",
                                      json={**request(), "tools": []})))
    asyncio.run(options["http_client"].aclose())
