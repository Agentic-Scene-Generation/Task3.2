"""Tests for production SceneExpert stage-commit retries."""

import time

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from scenesmith.experiments import indoor_scene_generation as scene_generation
from scenesmith.scene_expert.harness import RepairDecision
from scenesmith.scene_expert.hooks import SceneExpertHookRunner, StageCommitResult
from scenesmith.scene_expert.schemas import RepairResult, StageVerifyReport, VerifyIssue


def _failed_report(stage: str) -> StageVerifyReport:
    return StageVerifyReport(
        stage=stage,
        pass_stage=False,
        issues=[
            VerifyIssue(
                issue_type="missing_object",
                object_name="wastebasket",
                description="Required wastebasket is missing",
            )
        ],
    )


def test_failed_hook_stage_is_uncommitted_until_retry_verifies(tmp_path) -> None:
    stage = "manipuland"
    failed_report = _failed_report(stage)
    passed_report = StageVerifyReport(stage=stage, pass_stage=True)
    repair_result = RepairResult(
        repair_type="local_repair",
        failure_type="missing_object",
        repair_action="Place the missing wastebasket on the floor.",
    )
    runner = object.__new__(SceneExpertHookRunner)
    runner._mode = "harness_only"
    runner._original_text_descriptions = {stage: "original prompt"}
    runner._stage_verifier = Mock()
    runner._stage_verifier.verify.side_effect = [failed_report, passed_report]
    runner._task_spec = SimpleNamespace(room_type="bedroom")
    runner._current_stage_brief = None
    runner._harness = Mock(
        decide_repair=Mock(
            return_value=RepairDecision(
                should_repair=True,
                strategy="local_repair",
                reason="Attempt 1/1",
            )
        )
    )
    runner._repair_controller = Mock()
    runner._repair_controller.repair.return_value = repair_result
    runner._pending_stage_repairs = {}
    runner._stage_reports = []
    runner._completed_stages = []
    runner._stage_start_time = time.time()
    runner._current_memory_pack = SimpleNamespace()
    runner._current_relation_context = None
    runner._current_planner_trace = {}
    runner._qwen_calls = 0
    runner._commit_stage_memory = Mock()
    runner._trace_logger = Mock()
    scene = SimpleNamespace(text_description="brief", objects={})

    failed = runner.post_stage(stage, scene, tmp_path)

    assert failed == StageCommitResult(
        stage=stage,
        passed=False,
        retryable=True,
        reason="Attempt 1/1",
    )
    assert runner._stage_reports == []
    assert runner._completed_stages == []
    assert runner._pending_stage_repairs[stage] == (repair_result, failed_report)
    scene.text_description = "retry prompt"
    assert runner._inject_pending_stage_repair(stage, scene)
    assert scene.text_description.endswith(
        "[REPAIR INSTRUCTION]\nPlace the missing wastebasket on the floor."
    )

    passed = runner.post_stage(stage, scene, tmp_path)

    assert passed == StageCommitResult(stage=stage, passed=True)
    assert runner._stage_reports == [passed_report]
    assert runner._completed_stages == [stage]
    assert runner._pending_stage_repairs == {}
    assert repair_result.repair_verified
    assert repair_result.new_scene_state == str(tmp_path)
    assert runner._repair_controller.record_failure_to_memory.call_count == 2


def test_noop_stage_signal_is_scoped_to_current_stage() -> None:
    runner = object.__new__(SceneExpertHookRunner)
    runner._current_stage = "wall_mounted"
    runner._current_planner_trace = {"status": "no_op"}

    assert runner.should_skip_stage_agent("wall_mounted")
    assert not runner.should_skip_stage_agent("ceiling_mounted")

    runner._current_planner_trace = {"status": "completed"}
    assert not runner.should_skip_stage_agent("wall_mounted")


def test_pipeline_noop_gate_uses_hook_authority() -> None:
    hooks = Mock()
    hooks.should_skip_stage_agent.return_value = True

    assert scene_generation._should_skip_noop_scene_expert_stage(
        hooks, "ceiling_mounted"
    )
    hooks.should_skip_stage_agent.assert_called_once_with("ceiling_mounted")

    assert not scene_generation._should_skip_noop_scene_expert_stage(
        None, "wall_mounted"
    )


def test_rejected_stage_restarts_from_the_failed_stage(tmp_path) -> None:
    room_spec = SimpleNamespace(prompt="A practical office.")
    house_layout = SimpleNamespace(
        room_ids=["office"],
        get_room_spec=lambda room_id: room_spec,
        get_room_geometry=lambda room_id: object(),
    )
    logger = SimpleNamespace(room_context=lambda room_id: nullcontext(tmp_path))
    attempts: list[str] = []
    final_scene = object()

    def generate_room(**kwargs):
        attempts.append(kwargs["start_stage"])
        if len(attempts) == 1:
            raise scene_generation.SceneExpertStageCommitError(
                "wall_mounted",
                retryable=True,
                reason="missing wall object",
            )
        return final_scene

    with (
        patch.object(scene_generation, "_generate_room", side_effect=generate_room),
        patch.object(scene_generation, "custom_span", return_value=nullcontext()),
    ):
        rooms = scene_generation._run_sequential_room_generation(
            house_layout=house_layout,
            logger=logger,
            cfg_dict={},
            start_stage="furniture",
            stop_stage="manipuland",
            scene_expert_hooks=Mock(),
        )

    assert attempts == ["furniture", "wall_mounted"]
    assert rooms == {"office": final_scene}


def test_checkpoint_retry_restores_current_prompt_and_contract() -> None:
    scene = SimpleNamespace(text_description="", metadata={})

    def restore(state):
        scene.text_description = state["text_description"]
        scene.metadata = dict(state["metadata"])

    scene.restore_from_state_dict = Mock(side_effect=restore)
    contract = {"contract_id": "current-contract"}

    scene_generation._restore_room_stage_checkpoint(
        scene=scene,
        state={
            "text_description": "stale prompt\n\n=== SceneExpert Stage Brief ===",
            "metadata": {"scenebenchmark_intent_contract": {"contract_id": "old"}},
        },
        room_prompt="A practical office with four desks.",
        intent_contract=contract,
    )

    assert scene.text_description == "A practical office with four desks."
    assert scene.metadata["scenebenchmark_intent_contract"] == contract
    assert scene.scenebenchmark_intent_contract == contract


def test_exhausted_stage_repair_remains_an_explicit_failure(tmp_path) -> None:
    room_spec = SimpleNamespace(prompt="A practical office.")
    house_layout = SimpleNamespace(
        room_ids=["office"],
        get_room_spec=lambda room_id: room_spec,
        get_room_geometry=lambda room_id: object(),
    )
    logger = SimpleNamespace(room_context=lambda room_id: nullcontext(tmp_path))
    failure = scene_generation.SceneExpertStageCommitError(
        "manipuland",
        retryable=False,
        reason="Repair budget exhausted",
    )

    with (
        patch.object(
            scene_generation, "_generate_room", side_effect=failure
        ) as generate,
        patch.object(scene_generation, "custom_span", return_value=nullcontext()),
        pytest.raises(scene_generation.SceneExpertStageCommitError, match="exhausted"),
    ):
        scene_generation._run_sequential_room_generation(
            house_layout=house_layout,
            logger=logger,
            cfg_dict={},
            start_stage="furniture",
            stop_stage="manipuland",
            scene_expert_hooks=Mock(),
        )

    generate.assert_called_once()
