"""Integration coverage for an exception after a successful compiler response."""

from __future__ import annotations

from copy import deepcopy
import json

import pytest

from scenesmith.scene_expert import hooks
from scenesmith.scene_expert.evaluation_inputs import FrozenCompiledInputStore
from scenesmith.scene_expert.schemas import SceneTaskSpec

PROMPT = (
    "A small grocery store with two refrigerated display cases along one wall. "
    "Two tables near the entrance hold produce bins, each containing at least "
    "three fruits. At least four shopping baskets are stacked by the door."
)
REASON = "ambiguous generation-stage ownership must be reviewed before grounding"


def _hook_args(monkeypatch, tmp_path, compiler, *, enabled=True):
    monkeypatch.setattr(hooks, "TaskCompiler", lambda **_kwargs: compiler)
    monkeypatch.setattr(hooks, "GlobalPlanner", lambda **_kwargs: object())
    monkeypatch.setattr(hooks, "SceneExpertHookRunner", lambda **kwargs: kwargs)
    monkeypatch.setattr(
        hooks, "apply_behavior_template", lambda _prompt, spec, **_kwargs: (spec, None)
    )
    monkeypatch.setattr(
        hooks, "_compile_intent_contract_if_enabled", lambda **_kwargs: ({}, {})
    )
    return hooks.build_hook_runner(
        prompt=PROMPT,
        scene_id=0,
        output_dir=tmp_path,
        cfg_dict={
            "experiment": {
                "scene_expert": {
                    "enabled": True,
                    "mode": "harness_only",
                    "components": {"task_compiler": enabled},
                }
            },
            "furniture_agent": {"openai": {"model": "test-model"}},
            "scenebenchmark_critic": {"enabled": True},
        },
    )


@pytest.mark.parametrize(
    "upstream_kind",
    ["structured_success", "empty_trace", "missing_trace", "prior_fallback"],
)
def test_hook_fallback_records_typed_error_without_erasing_upstream(
    monkeypatch, tmp_path, upstream_kind
):
    class Compiler:
        def compile(self, prompt):
            assert prompt == PROMPT
            raise ValueError(REASON)

    compiler = Compiler()
    if upstream_kind == "structured_success":
        compiler.last_trace = {
            "status": "ok",
            "failure_reason": "",
            "attempts": [{"status": "ok"}],
            "raw_task_spec": {
                "room_type": "grocery store",
                "functional_zones": ["produce"],
            },
            "normalized_task_spec": {
                "required_large_objects": ["produce_bin", "produce_bin"]
            },
        }
    elif upstream_kind == "empty_trace":
        compiler.last_trace = {}
    elif upstream_kind == "prior_fallback":
        compiler.last_trace = {
            "status": "fallback",
            "failure_reason": "prior parsing failure",
        }
    prior = deepcopy(getattr(compiler, "last_trace", {}))
    args = _hook_args(monkeypatch, tmp_path, compiler)
    spec, trace = args["task_spec"], args["task_compiler_trace"]
    assert spec.compiler_status == "degraded"
    assert spec.compiler_failure_reason == "ValueError: " + REASON
    assert trace["status"] == "fallback"
    assert trace["failure_reason"] == spec.compiler_failure_reason
    assert trace["exception_type"] == "ValueError"
    assert trace["fallback_source"] == "hook_compile_exception"
    assert trace["upstream_trace"] == prior
    assert getattr(compiler, "last_trace", {}) == prior
    if upstream_kind == "structured_success":
        assert trace["upstream_trace"]["status"] == "ok"
        assert trace["raw_task_spec"] == prior["raw_task_spec"]
        assert trace["normalized_task_spec"] == prior["normalized_task_spec"]
    path = args["trace_logger"].record_task_compiler(spec, trace)
    saved = json.loads(path.read_text(encoding="utf8"))
    assert saved["compiler_status"] == "degraded"
    assert saved["failure_reason"] == "ValueError: " + REASON
    assert saved["task_spec"]["compiler_failure_reason"] == saved["failure_reason"]
    assert saved["structured_output"]["status"] == "fallback"
    assert saved["structured_output"]["upstream_trace"] == prior
    # The same deterministic fallback is retained; this is not a semantic repair.
    assert spec.room_type == "room"
    assert spec.interaction_constraints == []


def test_successful_compiler_result_and_trace_are_unchanged(monkeypatch, tmp_path):
    spec = SceneTaskSpec(
        room_type="grocery store", style="standard", functional_zones=["produce"]
    )

    class Compiler:
        last_trace = {
            "status": "ok",
            "failure_reason": "",
            "attempts": [{"status": "ok"}],
        }

        def compile(self, prompt):
            assert prompt == PROMPT
            return spec

    compiler = Compiler()
    args = _hook_args(monkeypatch, tmp_path, compiler)
    assert args["task_spec"] == spec
    assert args["task_compiler_trace"]["status"] == "ok"
    assert args["task_compiler_trace"]["failure_reason"] == ""
    assert "upstream_trace" not in args["task_compiler_trace"]
    assert compiler.last_trace == {
        "status": "ok",
        "failure_reason": "",
        "attempts": [{"status": "ok"}],
    }


def test_disabled_compiler_does_not_gain_exception_fallback_trace(
    monkeypatch, tmp_path
):
    class Compiler:
        def compile(self, _prompt):
            raise AssertionError("Disabled compiler must not execute")

    args = _hook_args(monkeypatch, tmp_path, Compiler(), enabled=False)
    assert args["task_compiler_trace"]["status"] == "disabled"
    assert "upstream_trace" not in args["task_compiler_trace"]
    assert args["task_spec"].compiler_status == "degraded"
    assert args["task_spec"].compiler_failure_reason == ""


def test_frozen_compiled_input_replay_retains_the_actual_hook_error(
    monkeypatch, tmp_path
):
    class Compiler:
        last_trace = {"status": "ok", "failure_reason": ""}

        def compile(self, _prompt):
            raise ValueError(REASON)

    args = _hook_args(monkeypatch, tmp_path / "hook", Compiler())
    payload = {
        "task_spec": args["task_spec"].model_dump(mode="json"),
        "intent_contract": {},
        "compiler_metadata": {"task_compiler_trace": args["task_compiler_trace"]},
    }
    store = FrozenCompiledInputStore(tmp_path / "frozen")
    first, first_identity = store.load_or_create(
        scene_id=0, prompt=PROMPT, producer=lambda: payload
    )

    def should_not_recompile():
        raise AssertionError("Frozen replay must not compile a second time")

    second, second_identity = store.load_or_create(
        scene_id=0, prompt=PROMPT, producer=should_not_recompile
    )
    assert first_identity["source"] == "materialized"
    assert second_identity["source"] == "frozen_replay"
    assert first == second
    assert second["task_spec"]["compiler_failure_reason"] == "ValueError: " + REASON
    assert (
        second["compiler_metadata"]["task_compiler_trace"]["upstream_trace"]["status"]
        == "ok"
    )
