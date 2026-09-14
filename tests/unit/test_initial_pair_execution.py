"""Exercise the native dispatch seam and SDK wire capture without Drake/GPU."""

from __future__ import annotations

import ast
import asyncio
import base64
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace, ModuleType

import pytest

from scenesmith.scene_expert.slow_memory.paired import EXPECTED_MODEL, read_json
from scenesmith.scene_expert.slow_memory.paired import (
    write_json,
    audit_pairs,
)
from scenesmith.scene_expert.slow_memory.paired_wire import capture_wire, client_options
from scenesmith.scene_expert.slow_memory import paired_runtime
from scenesmith.scene_expert.slow_memory.paired_provenance import (
    collect_pair_code_provenance,
)
from scenesmith.scene_expert.slow_memory.dpo import (
    export_dpo_dataset,
    load_trajectories,
)
from scenesmith.scene_expert.slow_memory.schemas import (
    PreferenceEvidence,
    TrajectoryOutcome,
)
from scenesmith.scene_expert.slow_memory.trajectory import TrajectoryCollector
from scenesmith.scene_expert.schemas import SceneTaskSpec

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "mode", ["disabled", "enabled", "failed_shadow", "failed_evidence"]
)
def test_native_initial_dispatch_preserves_canonical_continuation(
    monkeypatch, tmp_path, mode
):
    # Compile the actual changed method; importing the full native class would
    # require Drake. All simulator/model boundaries are explicit test doubles.
    source = ast.parse(
        (ROOT / "scenesmith/agent_utils/base_stateful_agent.py").read_text(
            encoding="utf-8"
        )
    )
    method = next(
        node
        for node in ast.walk(source)
        if isinstance(node, ast.AsyncFunctionDef)
        and node.name == "_request_initial_design_impl"
    )
    calls = []

    class Pair:
        canonical_dir = tmp_path / "A"

        def capture_raw(self, agent, result):
            calls.append("raw")
            assert agent.state == "raw_A"
            if mode == "failed_evidence":
                raise ValueError("evaluator unavailable")

        def fail(self, phase, error):
            calls.append("failed:" + phase)

        async def finish(self, agent, message):
            calls.append("shadow")
            assert agent.state == "returned_A"
            if mode == "failed_shadow":
                raise ValueError("B failed")

    async def open_pair(agent, input_message):
        calls.append("snapshot")
        assert agent.state == "initial"
        assert input_message == "task\n\nmemory\n\nbrief"
        return Pair()

    monkeypatch.setattr(paired_runtime, "open_initial_pair", open_pair)
    if mode == "disabled":
        monkeypatch.delenv("SCENEEXPERT_INITIAL_PAIRS_DIR", raising=False)
    else:
        monkeypatch.setenv("SCENEEXPERT_INITIAL_PAIRS_DIR", str(tmp_path))

    @asynccontextmanager
    async def session_context(session):
        yield

    def begin(**kwargs):
        calls.append("begin")
        return "transaction"

    def end(transaction):
        calls.append("native_safety")
        agent.state = "returned_A"
        return " + safety"

    async def run(**kwargs):
        calls.append("runner_A")
        agent.state = "raw_A"
        return SimpleNamespace(final_output="A result")

    agent = SimpleNamespace(
        state="initial",
        designer=object(),
        designer_session=object(),
        cfg=SimpleNamespace(
            agents=SimpleNamespace(designer_agent=SimpleNamespace(max_turns=5))
        ),
        rendering_manager=SimpleNamespace(last_render_dir=None),
        prompt_registry=SimpleNamespace(get_prompt=lambda **kw: "task"),
        _get_initial_design_prompt_enum=lambda: "initial",
        _get_initial_design_prompt_kwargs=lambda: {},
        _retrieve_working_memory_for_designer=lambda query: "memory",
        _prepare_stage_context_for_llm=lambda **kw: "brief",
        _build_initial_design_input=lambda x: x,
        _begin_furniture_design_transaction=begin,
        _end_furniture_design_transaction=end,
        _reasoning_persistence_context_for_session=session_context,
        _create_run_config=lambda: None,
        _record_module_timing=lambda *args: None,
        _record_llm_call_debug=lambda **kw: None,
        _save_designer_working_memory=lambda **kw: calls.append("save_memory"),
    )
    namespace = {
        "os": os,
        "time": time,
        "Runner": SimpleNamespace(run=run),
        "capture_wire": capture_wire,
        "console_logger": logging.getLogger(__name__),
        "log_agent_usage": lambda **kw: None,
        "log_agent_response": lambda **kw: None,
    }
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
            "native_dispatch",
            "exec",
        ),
        namespace,
    )
    output = asyncio.run(namespace[method.name](agent))
    assert output == "A result + safety"
    assert agent.state == "returned_A"
    assert calls.count("runner_A") == 1
    assert calls[-1] == "save_memory"
    if mode == "disabled":
        assert calls == ["begin", "runner_A", "native_safety", "save_memory"]
    else:
        assert calls.index("raw") < calls.index("native_safety")
        if mode != "failed_evidence":
            assert calls.index("native_safety") < calls.index("shadow")


def test_sdk_wire_capture_sees_serialized_request_and_never_headers(tmp_path):
    openai = pytest.importorskip("openai")
    httpx = pytest.importorskip("httpx")

    async def run():
        with capture_wire(tmp_path):
            http_client = client_options()["http_client"]
        # Use the production client and hooks; replace only network transport.
        http_client._transport = httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "id": "mock",
                    "object": "chat.completion",
                    "created": 0,
                    "model": EXPECTED_MODEL,
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": "done"},
                        }
                    ],
                },
            )
        )
        async with openai.AsyncOpenAI(
            base_url="http://unit-test/v1",
            api_key="never-record-this",
            http_client=http_client,
        ) as client:
            await client.chat.completions.create(
                model=EXPECTED_MODEL,
                messages=[{"role": "user", "content": "frozen"}],
                temperature=0.6,
            )
            await client.chat.completions.create(
                model=EXPECTED_MODEL,
                messages=[{"role": "user", "content": "second turn"}],
                temperature=0.6,
            )

    asyncio.run(run())
    assert (
        read_json(tmp_path / "first_request.json")["messages"][0]["content"] == "frozen"
    )
    assert "never-record-this" not in (tmp_path / "first_request.json").read_text()
    assert client_options() == {}


def test_candidate_capture_preserves_images_and_exports_identical_contexts(tmp_path):
    image = b"\x89PNG\r\n\x1a\nfixture"
    image_url = "data:image/png;base64," + base64.b64encode(image).decode()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Place a bed"},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }
    ]
    for name, verdict, score in [("A", "accepted", 1.0), ("B", "rejected", 0.5)]:
        collector = TrajectoryCollector(
            scene_debug_dir=tmp_path / name,
            prompt="Place a bed",
            scene_id="bedroom",
            run_id=name,
            task_spec=SceneTaskSpec(room_type="bedroom", style="minimal"),
            model_id=EXPECTED_MODEL,
        )
        collector.capture_candidate(
            payload={
                "stage": "furniture",
                "agent_role": "designer",
                "event": "request_initial_design",
                "prompt": messages,
                "conversation_messages": messages,
                "context_snapshot": {"initial_snapshot_sha256": "full-snapshot"},
                "output": "Candidate " + name,
                "capture_policy": "independently_scored_raw_initial_candidate",
            },
            evidence=PreferenceEvidence(
                evidence_id=name,
                kind="deterministic",
                verdict=verdict,
                authoritative=True,
                quality_score=score,
                source="raw",
                report_ref="report.json",
            ),
            outcome=TrajectoryOutcome(
                execution_complete=True,
                tool_call_valid=True,
                hard_passed=verdict == "accepted",
                hard_violation_count=int(verdict != "accepted"),
                causal_link_verified=True,
            ),
            scene_state_path="raw_state.json",
        )
    records, errors = load_trajectories([tmp_path / "A", tmp_path / "B"])
    assert not errors
    assert len(records) == 2
    assert records[0].context_hash == records[1].context_hash
    assert all(record.image_refs and record.prompt_complete for record in records)
    manifest = export_dpo_dataset(
        trajectory_sources=[tmp_path / "A", tmp_path / "B"], output_dir=tmp_path / "dpo"
    )
    assert manifest["stats"]["eligible_pair_count"] == 1
    assert list((tmp_path / "dpo/images").rglob("*.png"))


def test_local_tool_rng_restores_without_consuming_canonical_draws():
    import random
    import numpy as np

    original = paired_runtime.random_state()
    try:
        expected = (random.random(), np.random.random())
        paired_runtime.restore_random_state(original)
        assert (random.random(), np.random.random()) == expected
    finally:
        paired_runtime.restore_random_state(original)


def test_runtime_raw_capture_to_export_binds_each_candidate_before_safety(
    monkeypatch, tmp_path
):
    # Exercise real capture_raw/capture_returned/export with only the simulator
    # evaluator and SDK extraction replaced at their dependency boundaries.
    trace_module = ModuleType("scenesmith.agent_utils.stage_working_memory")
    trace_module._extract_agent_result_trace = lambda result, output: {
        "tool_results": [],
        "assistant_messages": [{"role": "assistant", "content": output}],
    }
    api_module = ModuleType("scenesmith.scenebenchmark_critic.api")
    api_module.evaluate_room_scene = lambda scene, **kw: {
        "summary": {
            "scene_summary": {
                "total_checks": 2,
                "unknown": 0,
                "fail": scene.failures,
                "score": 1.0 - scene.failures / 2,
            }
        }
    }
    cfg_module = ModuleType("scenesmith.scenebenchmark_critic.config")
    cfg_module.critic_config_from_any = lambda cfg: SimpleNamespace(enabled=True)
    for module in (trace_module, api_module, cfg_module):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    source = tmp_path / "canonical"
    source.mkdir()
    (source / "geometry.sdf").write_text("room")
    root = tmp_path / "paired_initial"
    group = root / "group_000"
    files = paired_runtime.copy_scene_tree(source, group / "input_scene")
    snapshot = {
        "source_scene_root": str(source),
        "files": files,
        "cfg": {},
        "code_provenance": collect_pair_code_provenance(),
        "room_id": "bedroom",
        "input": "Place a bed",
        "scene_attributes": {
            "scene_expert_task_spec": {"room_type": "bedroom", "style": "minimal"}
        },
        "state": {"text_description": "Place a bed"},
    }
    write_json(group / "snapshot.json", snapshot)
    pair = paired_runtime.InitialPair(group, snapshot)
    for candidate, failures in (("A", 0), ("B", 1)):
        if candidate == "B":
            paired_runtime.copy_scene_tree(group / "input_scene", group / "B/scene")
        write_json(
            group / candidate / "first_request.json",
            {
                "model": EXPECTED_MODEL,
                "messages": [{"role": "user", "content": "Place a bed"}],
                "tools": [],
            },
        )
        state = {"result": "raw_" + candidate}
        scene = SimpleNamespace(failures=failures, to_state_dict=lambda: state)
        agent = SimpleNamespace(
            scene=scene, cfg={}, asset_manager=SimpleNamespace(_fatal_asset_error=None)
        )
        pair.capture_raw(
            agent,
            SimpleNamespace(final_output="Placed " + candidate),
            candidate=candidate,
        )
        result_path = group / candidate / "result.json"
        assert read_json(result_path)["status"] == "raw_captured"
        state["result"] = "after_native_safety_" + candidate
        pair.capture_returned(agent, "safety changed geometry", candidate)
        result = read_json(result_path)
        assert result["status"] == "completed"
        assert result["raw_state_hash"] != result["returned_state_hash"]
        assert result["verdict"] == ("accepted" if failures == 0 else "rejected")
    write_json(group / "status.json", {"status": "completed"})
    write_json(
        group / "continuation_proof.json",
        {
            "canonical_before": "a",
            "canonical_after": "a",
            "assets_before": "f",
            "assets_after": "f",
            "memory_before": "m",
            "memory_after": "m",
        },
    )
    audit = audit_pairs(root, expected_groups=1)
    assert audit["gate_passed"], audit["errors"]
    assert audit["eligible_pair_count"] == 1
