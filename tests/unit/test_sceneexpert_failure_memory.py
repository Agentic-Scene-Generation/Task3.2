"""Failure-path tests for additive harness/memory observability."""

from __future__ import annotations

import time

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("pydrake", reason="hook runner imports the production Drake scene")

from scenesmith.scene_expert.hooks import SceneExpertHookRunner
from scenesmith.scene_expert.schemas import (
    MemoryPack,
    SceneTaskSpec,
    StageExecutionEvidence,
)


def _runner(tmp_path) -> SceneExpertHookRunner:
    runner = object.__new__(SceneExpertHookRunner)
    runner._component_flags = {
        "trace": True,
        "memory_writer": True,
        "verifier": True,
        "stage_working_memory": True,
    }
    runner._trace_logger = Mock()
    runner._trace_logger.build_trace_summary.return_value = "failed trace"
    runner._trace_logger.build_memory_writer_evidence.return_value = {
        "trace_id": "trace_000000",
        "stages": [],
    }
    runner._trace_logger.finalize.return_value = {"status": "completed"}
    runner._memory_writer = Mock()
    runner._memory_writer.write.return_value = []
    runner._memory_writer.last_trace = {
        "write_status": "no_valid_candidates",
        "fallback_written": False,
    }
    runner._memory_store = Mock()
    runner._memory_store.revision = 2
    runner._memory_store.apply_updates.return_value = {
        "added": 0,
        "merged": 0,
        "revision": 3,
    }
    runner._retriever = None
    runner._prompt = "A classroom with six desks and paired chairs."
    runner._scene_id = 0
    runner._output_dir = tmp_path
    runner._scene_debug_dir = tmp_path / "scene_000" / "scene_expert"
    runner._mode = "harness_memory"
    runner._qwen_model = "qwen-test"
    runner._experiment_name = "ablation_4c"
    runner._config_hash = "config-hash"
    runner._task_spec = SceneTaskSpec(room_type="classroom")
    runner._current_stage = "furniture"
    runner._current_memory_pack = MemoryPack(memory_bank_id="bank-1")
    runner._current_relation_context = None
    runner._current_planner_trace = {"status": "ok"}
    runner._current_stage_brief = None
    runner._current_execution_evidence = StageExecutionEvidence()
    runner._latest_scene = SimpleNamespace(metadata={})
    runner._completed_stages = ["floor_plan"]
    runner._configured_stop_stage = "manipuland"
    runner._allow_long_term_memory_updates = True
    runner._stage_reports = []
    runner._qwen_calls = 1
    runner._stage_start_time = time.time()
    return runner


def test_main_hard_gate_runs_failure_writer_without_changing_failure(tmp_path) -> None:
    runner = _runner(tmp_path)
    error = (
        "Furniture stage failed with unresolved core relations: "
        "paired_with:student_desk"
    )

    runner.finalize_failure(error)

    assert len(runner._stage_reports) == 1
    assert runner._stage_reports[0].hard_check_report["hard_valid"] is False
    runner._memory_writer.write.assert_called_once()
    writer_report = runner._memory_writer.write.call_args.kwargs["full_report"]
    assert writer_report.pass_scene is False
    assert writer_report.outcome_status == "FAILED"
    runner._memory_store.append_event.assert_called_once()
    saved_trace = runner._trace_logger.save.call_args.args[0]
    assert saved_trace["status"] == "failed"
    assert saved_trace["error"] == error


def test_arbitrary_runtime_failure_remains_trace_only(tmp_path) -> None:
    runner = _runner(tmp_path)
    runner.save_partial_trace = Mock()
    runner._current_execution_evidence.stage_agent_invoked = True
    runner._latest_scene.metadata["scenesmith_planner_failure"] = {
        "schema_version": "scenesmith.planner_failure.v1",
        "finish_reason": "no_scene_mutation_after_recovery",
        "successful_designer_mutations": 0,
        "checkpoint_state": {
            "has_current_checkpoint": False,
            "has_previous_checkpoint": False,
        },
    }

    runner.finalize_failure("CUDA connection reset", error_type="ConnectionError")

    runner.save_partial_trace.assert_called_once_with(error="CUDA connection reset")
    runner._memory_writer.write.assert_not_called()
    runner._memory_store.append_event.assert_not_called()
    assert runner._current_execution_evidence.degraded is True
    failure = runner._current_execution_evidence.runtime_failure
    assert failure["error_type"] == "ConnectionError"
    assert failure["stage_agent_invoked"] is True
    assert (
        failure["planner_diagnostics"]["finish_reason"]
        == "no_scene_mutation_after_recovery"
    )
    assert failure["planner_diagnostics"]["checkpoint_state"] == {
        "has_current_checkpoint": False,
        "has_previous_checkpoint": False,
    }
    runner._trace_logger.log_stage.assert_called_once()
    runner._trace_logger.save_stage_context.assert_called_once()


def test_shared_base_hard_failure_does_not_write_long_term_memory(tmp_path) -> None:
    runner = _runner(tmp_path)
    runner._configured_stop_stage = "floor_plan"
    runner._allow_long_term_memory_updates = False
    runner._flush_skill_outcomes = Mock()

    runner.finalize_failure(
        "Floor plan stage failed with unresolved core relations: room_boundary"
    )

    runner._memory_writer.write.assert_not_called()
    runner._flush_skill_outcomes.assert_not_called()
    status = runner._trace_logger.record_component_status.call_args_list[-1]
    assert status.args[0] == "memory_writer"
    assert status.args[1]["write_status"] == "skipped_non_terminal_pipeline"
    assert status.args[1]["configured_stop_stage"] == "floor_plan"
