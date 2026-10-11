"""Native SDK boundary, mutable-prefix restoration and direct-action export gates."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

pytest.importorskip("agents")
from agents.models.interface import Model
from omegaconf import OmegaConf

from tests.unit.test_initial_pair_snapshot import _agent
from tests.unit.test_sceneexpert_campaign import record
from scenesmith.agent_utils.workflow_tools import WorkflowTools
from scenesmith.scene_expert.slow_memory.decision_pair import (
    DecisionModel,
    DecisionPair,
    decision_result,
    restore_workflow,
    workflow_owner,
)
from scenesmith.scene_expert.slow_memory.decision_scope import first_turn_has_design
from scenesmith.scene_expert.slow_memory.dpo import export_dpo_dataset
from scenesmith.scene_expert.slow_memory.paired import EXPECTED_MODEL, read_json
from scenesmith.scene_expert.slow_memory.paired_runtime import (
    open_initial_pair,
    json_value,
)
from scenesmith.scene_expert.slow_memory.paired_wire import (
    capture_wire,
    client_options,
    wrap_pair_provider,
)


def tool_item(name, call_id="c"):
    return {
        "type": "function_call",
        "name": name,
        "call_id": call_id,
        "arguments": "{}",
        "id": call_id,
    }


def direct_pair(tmp_path, monkeypatch):
    monkeypatch.setenv("SCENEEXPERT_INITIAL_PAIRS_DIR", str(tmp_path / "pairs"))
    monkeypatch.setenv("SCENEEXPERT_PAIR_BOUNDARY", "first_mutation")
    monkeypatch.delenv("SCENEEXPERT_PAIR_SOURCE_HASH", raising=False)
    monkeypatch.delenv("SCENEEXPERT_INITIAL_PAIR_SHADOW", raising=False)
    agent = _agent(tmp_path)
    agent.cfg.agents = OmegaConf.create({"designer_agent": {"max_turns": 10}})
    agent.designer.tools = list(WorkflowTools().tools.values())
    pair = asyncio.run(open_initial_pair(agent, "initial task"))
    assert isinstance(pair, DecisionPair)
    return agent, pair


def test_shared_observation_and_todo_snapshot_precedes_design_tool_execution(
    tmp_path, monkeypatch
):
    agent, pair = direct_pair(tmp_path, monkeypatch)
    assert not (pair.group / "input_scene").exists()
    assert read_json(pair.group / "status.json")["status"] == "shared_prefix_running"
    owner = workflow_owner(agent.designer.tools)
    owner._designer_todo_manager_impl(action="add", task="Place a bed")
    prefix = [
        {"role": "user", "content": "task"},
        tool_item("observe_scene", "observe"),
        {"type": "function_call_output", "call_id": "observe", "output": "shared view"},
    ]
    pair.observe_request(
        {"model": EXPECTED_MODEL, "messages": [{"role": "user", "content": "first"}]}
    )
    asyncio.run(
        pair.observe_response(
            SimpleNamespace(output=[tool_item("observe_scene")]), (), {}
        )
    )
    pair.observe_request({"model": EXPECTED_MODEL, "messages": prefix})
    asyncio.run(
        pair.observe_response(
            SimpleNamespace(output=[tool_item("generate_assets")]),
            (),
            {"system_instructions": "system", "input": prefix},
        )
    )
    assert pair.frozen and not pair.disabled
    snapshot = read_json(pair.group / "snapshot.json")
    assert snapshot["input"] == prefix and snapshot["remaining_max_turns"] == 9
    assert snapshot["workflow_todos"] == json_value(owner._designer_todos)
    assert read_json(pair.canonical_dir / "first_request.json")["messages"] == prefix
    assert snapshot["state"] == agent.scene.to_state_dict()
    other_tools = list(WorkflowTools().tools.values())
    restore_workflow(other_tools, snapshot["workflow_todos"])
    other = workflow_owner(other_tools)
    other._designer_todo_manager_impl(action="complete", task="Place a bed")
    assert owner._designer_todos[0].status == "pending"
    assert other._designer_todos[0].status == "completed"
    # Subsequent canonical requests cannot overwrite the decision request.
    pair.observe_request({"model": "different", "messages": []})
    assert read_json(pair.canonical_dir / "first_request.json")["messages"] == prefix


@pytest.mark.parametrize("tool", ["check_physics", "unknown_tool"])
def test_unsupported_prefix_is_quarantined_without_changing_canonical_response(
    tmp_path, monkeypatch, tool
):
    agent, pair = direct_pair(tmp_path, monkeypatch)
    response = SimpleNamespace(output=[tool_item(tool)])

    class NativeModel(Model):
        async def get_response(self, *args, **kwargs):
            return response

        def stream_response(self, *args, **kwargs):
            raise AssertionError()

    actual = asyncio.run(DecisionModel(NativeModel(), pair).get_response())
    assert actual is response and pair.disabled and not pair.frozen
    assert read_json(pair.group / "status.json")["phase"] == "decision_boundary"
    with pytest.raises(ValueError, match="no supported"):
        pair.capture_raw(agent, SimpleNamespace())


def test_canonical_result_excludes_shared_items_but_keeps_complete_decision_rollout():
    shared = SimpleNamespace(
        raw_item=tool_item("observe_scene"), to_input_item=lambda: {"prefix": True}
    )
    action = SimpleNamespace(
        raw_item=tool_item("generate_assets", "act"),
        to_input_item=lambda: tool_item("generate_assets", "act"),
    )
    final = SimpleNamespace(
        raw_item={"role": "assistant", "content": "done"},
        to_input_item=lambda: {"role": "assistant", "content": "done"},
    )
    source = SimpleNamespace(
        new_items=[shared, action, final],
        raw_responses=["read", "act", "end"],
        final_output="done",
    )
    actual = decision_result(
        source, {"fork_response_items": [action.raw_item], "prefix_model_turns": 1}
    )
    assert actual.new_items == [action, final] and actual.raw_responses == [
        "act",
        "end",
    ]
    assert actual.input == [] and actual.to_input_list() == [
        action.raw_item,
        final.raw_item,
    ]


@pytest.mark.parametrize(
    "name,valid",
    [
        ("observe_scene", False),
        ("designer_todo_manager", False),
        ("generate_assets", True),
        ("move_furniture_tool", True),
    ],
)
def test_direct_action_classifier(name, valid):
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": name, "arguments": {}}}],
        },
        {"role": "tool", "content": "environment"},
        {
            "role": "assistant",
            "tool_calls": [{"function": {"name": "generate_assets"}}],
        },
    ]
    assert first_turn_has_design(messages) == valid


@pytest.mark.parametrize(
    "rejected_tool,count", [("observe_scene", 0), ("generate_assets", 1)]
)
def test_export_requires_both_direct_targets_and_keeps_observation_in_prompt(
    tmp_path, rejected_tool, count
):
    a, b = record("a", failures=0), record("b", failures=3, score=0.5)
    for row, name in ((a, "generate_assets"), (b, rejected_tool)):
        row.messages = [{"role": "user", "content": "shared observation"}]
        row.completion_messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": name,
                            "arguments": {"asset": row.trajectory_id},
                        }
                    }
                ],
            }
        ]
    source = tmp_path / "source.jsonl"
    source.write_text("\n".join(row.model_dump_json() for row in (a, b)))
    manifest = export_dpo_dataset(
        trajectory_sources=[source],
        output_dir=tmp_path / "out",
        completion_view="first_turn",
        require_direct_action=True,
    )
    assert manifest["stats"]["eligible_pair_count"] == count
    assert manifest["stats"]["require_direct_action"]
    if not count:
        assert (
            manifest["stats"]["diagnostic_reasons"]["first_turn_not_direct_design"] == 1
        )
    else:
        exported = json.loads((tmp_path / "out/all.jsonl").read_text())
        assert exported["prompt"] == a.messages


def test_mismatched_shadow_request_is_rejected_before_transport_or_tools(tmp_path):
    expected = {
        "model": EXPECTED_MODEL,
        "messages": [{"role": "user", "content": "shared"}],
    }
    with capture_wire(tmp_path / "B", expected_request=expected):
        options = client_options()
    hook = options["http_client"].event_hooks["request"][0]
    request = httpx.Request(
        "POST",
        "http://localhost/v1/chat/completions",
        json={**expected, "messages": []},
    )
    with pytest.raises(ValueError, match="differs before execution"):
        asyncio.run(hook(request))
    assert not (tmp_path / "B/first_request.json").exists()
    asyncio.run(options["http_client"].aclose())


def test_default_model_provider_is_unchanged():
    provider = object()
    assert wrap_pair_provider(provider) is provider


def test_native_runner_continues_from_shared_prefix_once(tmp_path, monkeypatch):
    from agents import Agent, Runner
    from agents.items import ModelResponse
    from agents.usage import Usage
    from agents import function_tool
    from openai.types.responses import (
        ResponseFunctionToolCall,
        ResponseOutputMessage,
        ResponseOutputText,
    )

    agent, pair = direct_pair(tmp_path, monkeypatch)
    mutations = []

    @function_tool
    def observe_scene() -> str:
        return "shared room geometry"

    @function_tool
    def generate_assets() -> str:
        mutations.append("asset")
        return "asset generated"

    class ScriptedModel(Model):
        def __init__(self, start=0):
            self.turn = start
            self.inputs = []

        async def get_response(self, *args, **kwargs):
            self.inputs.append(
                json_value(kwargs.get("input", args[1] if len(args) > 1 else []))
            )
            self.turn += 1
            if self.turn <= 3:
                name = ["observe_scene", "designer_todo_manager", "generate_assets"][
                    self.turn - 1
                ]
                arguments = (
                    '{"action":"add","task":"Place a bed"}' if self.turn == 2 else "{}"
                )
                output = [
                    ResponseFunctionToolCall(
                        type="function_call",
                        id=f"id_{self.turn}",
                        call_id=f"call_{self.turn}",
                        name=name,
                        arguments=arguments,
                    )
                ]
            else:
                output = [
                    ResponseOutputMessage(
                        type="message",
                        id="done",
                        role="assistant",
                        status="completed",
                        content=[
                            ResponseOutputText(
                                type="output_text", text="done", annotations=[]
                            )
                        ],
                    )
                ]
            pair.observe_request({"model": EXPECTED_MODEL, "messages": self.inputs[-1]})
            return ModelResponse(output=output, usage=Usage(), response_id=None)

        def stream_response(self, *args, **kwargs):
            raise AssertionError()

    original_tools = [observe_scene, *agent.designer.tools, generate_assets]
    agent.designer.tools = original_tools
    a_model = ScriptedModel()
    a = Agent(
        name="A",
        instructions="system",
        model=DecisionModel(a_model, pair),
        tools=original_tools,
    )
    result_a = asyncio.run(Runner.run(a, input="task", max_turns=10))
    assert pair.frozen and mutations == ["asset"]
    assert pair.snapshot["prefix_model_turns"] == 2
    assert pair.snapshot["workflow_todos"][0]["status"] == "pending"
    from scenesmith.agent_utils.stage_working_memory import _extract_agent_result_trace
    from scenesmith.scene_expert.slow_memory.trajectory import (
        _replay_completion_messages,
    )

    trace = _extract_agent_result_trace(
        decision_result(result_a, pair.snapshot), "done"
    )
    completion = _replay_completion_messages(trace, fallback="done")
    assert completion[0]["tool_calls"][0]["function"]["name"] == "generate_assets"
    assert "observe_scene" not in json.dumps(completion)
    # B uses the frozen native input and independent workflow owner; no prefix replay.
    b_workflow = list(WorkflowTools().tools.values())
    restore_workflow(b_workflow, pair.snapshot["workflow_todos"])
    b_model = ScriptedModel(start=2)
    b = Agent(
        name="B",
        instructions=pair.snapshot["system"],
        model=b_model,
        tools=[observe_scene, *b_workflow, generate_assets],
    )
    asyncio.run(
        Runner.run(
            b,
            input=pair.snapshot["input"],
            max_turns=pair.snapshot["remaining_max_turns"],
        )
    )
    assert b_model.inputs[0] == a_model.inputs[2]
    assert mutations == ["asset", "asset"]
    assert len(workflow_owner(b_workflow)._designer_todos) == 1
