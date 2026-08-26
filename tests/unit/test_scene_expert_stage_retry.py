"""Tests for production SceneExpert stage-commit retries."""

import asyncio
import json
import time

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from scenesmith.experiments import indoor_scene_generation as scene_generation
from scenesmith.agent_utils.base_stateful_agent import BaseStatefulAgent
from scenesmith.agent_utils.room import AgentType
from scenesmith.scene_expert.harness import Harness, RepairDecision
from scenesmith.scene_expert.hooks import SceneExpertHookRunner, StageCommitResult
from scenesmith.scene_expert.schemas import (
    FullVerifyReport,
    RepairResult,
    SceneTaskSpec,
    StageVerifyReport,
    VerifyIssue,
)
from scenesmith.scene_expert.verifier import FullVerifier, StageVerifier
from scenesmith.scenebenchmark_critic.config import CriticConfig


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


def test_wall_stage_verifier_hard_fails_fresh_core_visual_issue(tmp_path) -> None:
    payload = {
        "results": [
            {
                "check_id": "wall_visibility__mirror_0",
                "metric": "visual_clearance",
                "scoring_tier": "core",
                "label": "degraded",
                "primary_object": "mirror_0",
                "diagnostics": {"occluded_fraction": 0.08},
            }
        ]
    }

    report = StageVerifier().verify(
        stage="wall_mounted",
        stage_output_dir=str(tmp_path),
        task_spec=SceneTaskSpec(room_type="bedroom", style="standard"),
        scene_state_info={"object_names": ["mirror_0"]},
        deterministic_critic_payload=payload,
    )

    assert not report.pass_stage
    assert {issue.issue_type for issue in report.issues} == {"visual_clearance_failure"}


def test_wall_stage_verifier_ignores_auxiliary_visual_issue(tmp_path) -> None:
    report = StageVerifier().verify(
        stage="wall_mounted",
        stage_output_dir=str(tmp_path),
        task_spec=SceneTaskSpec(room_type="bedroom", style="standard"),
        scene_state_info={"object_names": ["mirror_0"]},
        deterministic_critic_payload={
            "results": [
                {
                    "check_id": "auxiliary-window",
                    "metric": "visual_clearance",
                    "scoring_tier": "auxiliary",
                    "label": "fail",
                    "primary_object": "mirror_0",
                }
            ]
        },
    )

    assert report.pass_stage


def _binding_failure_payload(*, earliest_stage: str) -> dict:
    return {
        "results": [
            {
                "check_id": "intent_on_top_of__binding",
                "metric": "functional_dependency",
                "scoring_tier": "core",
                "label": "fail",
                "primary_object": "plush_toy",
                "reason": "Required endpoint is missing.",
                "diagnostics": {
                    "binding_issue": "missing",
                    "earliest_stage": earliest_stage,
                },
                "evidence": {
                    "intent_constraint": {
                        "constraint_id": "plush_on_desk",
                        "relation": "on_top_of",
                        "stage": earliest_stage,
                        "strength": "hard",
                    }
                },
            }
        ]
    }


def test_stage_verifier_defers_future_endpoint_binding_failure(tmp_path) -> None:
    report = StageVerifier().verify(
        stage="furniture",
        stage_output_dir=str(tmp_path),
        task_spec=SceneTaskSpec(room_type="bedroom", style="standard"),
        scene_state_info={"object_names": []},
        deterministic_critic_payload=_binding_failure_payload(
            earliest_stage="manipuland"
        ),
    )

    assert report.pass_stage


def test_stage_verifier_fails_due_endpoint_binding_failure(tmp_path) -> None:
    report = StageVerifier().verify(
        stage="manipuland",
        stage_output_dir=str(tmp_path),
        task_spec=SceneTaskSpec(room_type="bedroom", style="standard"),
        scene_state_info={"object_names": []},
        deterministic_critic_payload=_binding_failure_payload(
            earliest_stage="manipuland"
        ),
    )

    assert not report.pass_stage
    assert [issue.issue_type for issue in report.issues] == ["contract_binding_failure"]
    assert report.issues[0].constraint_id == "plush_on_desk"


def test_full_verifier_fails_fresh_final_deterministic_payload() -> None:
    report = FullVerifier().verify(
        stage_reports=[StageVerifyReport(stage="manipuland", pass_stage=True)],
        deterministic_critic_payload=_binding_failure_payload(
            earliest_stage="manipuland"
        ),
    )

    assert not report.deterministic_pass
    assert not report.pass_scene


def test_hook_finalize_refreshes_final_deterministic_payload(tmp_path) -> None:
    fresh_payload = _binding_failure_payload(earliest_stage="manipuland")
    scene = SimpleNamespace()
    runner = object.__new__(SceneExpertHookRunner)
    runner._mode = "harness_only"
    runner._scene_id = 1
    runner._stage_reports = [StageVerifyReport(stage="manipuland", pass_stage=True)]
    runner._latest_scene = scene
    runner._latest_deterministic_payload = {"results": []}
    runner._critic_config = CriticConfig(enabled=True)
    runner._full_verifier = Mock(
        verify=Mock(return_value=FullVerifyReport(deterministic_pass=False))
    )
    runner._trace_logger = Mock()
    runner._trace_logger.finalize.return_value = {}
    runner._trace_logger.save.return_value = tmp_path / "trace.json"
    runner._qwen_model = "qwen"
    runner._memory_writer = None
    runner._memory_store = None

    with patch(
        "scenesmith.scenebenchmark_critic.api.evaluate_room_scene",
        return_value=fresh_payload,
    ) as evaluate:
        runner.finalize(str(tmp_path))

    evaluate.assert_called_once_with(
        scene,
        config=runner._critic_config,
        stage="final_scene_verification",
        annotate_assets=False,
    )
    assert (
        runner._full_verifier.verify.call_args.kwargs["deterministic_critic_payload"]
        is fresh_payload
    )


def test_hook_finalize_does_not_promote_partial_shared_base(tmp_path) -> None:
    runner = object.__new__(SceneExpertHookRunner)
    runner._mode = "harness_memory"
    runner._scene_id = 1
    runner._stage_reports = [StageVerifyReport(stage="floor_plan", pass_stage=True)]
    runner._latest_scene = None
    runner._latest_deterministic_payload = None
    runner._critic_config = CriticConfig(enabled=False)
    runner._component_flags = {"trace": False, "verifier": False}
    runner._full_verifier = Mock()
    runner._trace_logger = None
    runner._qwen_model = "qwen"
    runner._memory_writer = Mock()
    runner._memory_store = Mock()
    runner._allow_long_term_memory_updates = False
    runner._pending_skill_observations = [Mock()]
    runner._write_long_term_memory = Mock()
    runner._flush_skill_outcomes = Mock()
    runner._capture_main_repair_activity = Mock()

    runner.finalize(str(tmp_path))

    runner._write_long_term_memory.assert_not_called()
    runner._flush_skill_outcomes.assert_not_called()
    assert runner._pending_skill_observations == []


def test_wall_post_stage_passes_fresh_deterministic_payload_to_verifier(
    tmp_path,
) -> None:
    stage = "wall_mounted"
    payload = {
        "results": [
            {
                "check_id": "wall_visibility__mirror_0",
                "metric": "visual_clearance",
                "scoring_tier": "core",
                "label": "pass",
                "primary_object": "mirror_0",
            }
        ]
    }
    runner = object.__new__(SceneExpertHookRunner)
    runner._mode = "harness_only"
    runner._original_text_descriptions = {stage: "original prompt"}
    runner._stage_verifier = Mock(
        verify=Mock(return_value=StageVerifyReport(stage=stage, pass_stage=True))
    )
    runner._task_spec = SceneTaskSpec(room_type="bedroom", style="standard")
    runner._current_stage_brief = None
    runner._critic_config = CriticConfig(
        enabled=True,
        metrics=("visual_clearance",),
    )
    runner._harness = Mock()
    runner._repair_controller = Mock()
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
    scene = SimpleNamespace(
        text_description="wall brief",
        objects={},
        room_geometry=None,
    )

    with patch(
        "scenesmith.scenebenchmark_critic.api.evaluate_room_scene",
        return_value=payload,
    ) as evaluate:
        result = runner.post_stage(stage, scene, tmp_path)

    assert result == StageCommitResult(stage=stage, passed=True)
    evaluate.assert_called_once_with(
        scene,
        config=runner._critic_config,
        stage="wall_mounted_post_stage",
        annotate_assets=False,
    )
    assert (
        runner._stage_verifier.verify.call_args.kwargs["deterministic_critic_payload"]
        is payload
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
        quality_failure=True,
    )
    assert runner._stage_reports == []
    assert runner._completed_stages == []
    assert runner._pending_stage_repairs[stage] == (repair_result, failed_report)
    scene.text_description = "retry prompt"
    assert runner._inject_pending_stage_repair(stage, scene)
    assert scene.text_description.endswith(
        "[REPAIR INSTRUCTION]\nPlace the missing wastebasket on the floor."
    )
    assert repair_result.execution_status == "executed"

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


def test_degraded_policy_advances_after_quality_repair_budget_is_exhausted(
    tmp_path,
) -> None:
    hooks = Mock()
    hooks.post_stage.return_value = StageCommitResult(
        stage="furniture",
        passed=False,
        retryable=False,
        reason="Repair budget exhausted",
        quality_failure=True,
    )

    scene_generation._commit_scene_expert_stage(
        hooks=hooks,
        stage="furniture",
        scene=Mock(),
        room_dir=tmp_path,
        allow_degraded_quality=True,
    )
    hooks.accept_degraded_stage.assert_called_once_with("furniture")


def test_accept_degraded_stage_advances_harness_for_next_stage() -> None:
    runner = object.__new__(SceneExpertHookRunner)
    runner._current_stage = "wall_mounted"
    runner._completed_stages = ["floor_plan", "furniture"]
    runner._pending_stage_repairs = {"wall_mounted": (Mock(), Mock())}
    runner._harness = Harness(SimpleNamespace())

    runner.accept_degraded_stage("wall_mounted")

    assert runner._completed_stages == [
        "floor_plan",
        "furniture",
        "wall_mounted",
    ]
    assert runner._pending_stage_repairs == {}
    assert runner._harness.validate_stage_order(
        runner._completed_stages, "ceiling_mounted"
    )

    with pytest.raises(ValueError, match="current stage"):
        runner.accept_degraded_stage("ceiling_mounted")


def test_exhausted_quality_failure_is_retained_for_final_verification(
    tmp_path,
) -> None:
    failed_report = _failed_report("furniture")
    runner = object.__new__(SceneExpertHookRunner)
    runner._mode = "harness_only"
    runner._original_text_descriptions = {"furniture": "original prompt"}
    runner._stage_verifier = Mock(verify=Mock(return_value=failed_report))
    runner._task_spec = SimpleNamespace(room_type="office")
    runner._current_stage_brief = None
    runner._harness = Mock(
        decide_repair=Mock(
            return_value=RepairDecision(
                should_repair=False,
                strategy="none",
                reason="Repair budget exhausted",
            )
        )
    )
    runner._repair_controller = Mock()
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

    result = runner.post_stage("furniture", scene, tmp_path)

    assert result == StageCommitResult(
        stage="furniture",
        passed=False,
        retryable=False,
        reason="Repair budget exhausted",
        quality_failure=True,
    )
    assert runner._stage_reports == [failed_report]
    assert runner._completed_stages == []


@pytest.mark.parametrize(
    "result",
    [
        StageCommitResult(
            stage="furniture",
            passed=False,
            retryable=True,
            reason="Retry requested",
            quality_failure=True,
        ),
        StageCommitResult(
            stage="furniture",
            passed=False,
            retryable=False,
            reason="verification error: corrupt scores",
            quality_failure=False,
        ),
    ],
)
def test_degraded_policy_does_not_swallow_retry_or_verifier_error(
    tmp_path, result
) -> None:
    hooks = Mock()
    hooks.post_stage.return_value = result

    with pytest.raises(scene_generation.SceneExpertStageCommitError):
        scene_generation._commit_scene_expert_stage(
            hooks=hooks,
            stage="furniture",
            scene=Mock(),
            room_dir=tmp_path,
            allow_degraded_quality=True,
        )


def test_rejected_stage_restarts_from_the_failed_stage(tmp_path) -> None:
    room_spec = SimpleNamespace(prompt="A practical office.")
    house_layout = SimpleNamespace(
        room_ids=["office"],
        get_room_spec=lambda room_id: room_spec,
        get_room_geometry=lambda room_id: object(),
    )
    logger = SimpleNamespace(room_context=lambda room_id: nullcontext(tmp_path))
    attempts: list[tuple[str, int]] = []
    final_scene = object()

    def generate_room(**kwargs):
        attempts.append((kwargs["start_stage"], kwargs["scene_expert_retry_attempt"]))
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

    assert attempts == [("furniture", 0), ("wall_mounted", 1)]
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


def test_rejected_furniture_checkpoint_rebinds_current_prompt_and_contract(
    tmp_path,
) -> None:
    checkpoint_path = tmp_path / "scene_state.json"
    checkpoint_path.write_text(
        json.dumps(
            {
                "text_description": "stale prompt",
                "metadata": {"scenebenchmark_intent_contract": {"source": "old"}},
                "objects": {"table_0": {"object_type": "furniture"}},
            }
        ),
        encoding="utf-8",
    )
    scene = SimpleNamespace(text_description="", metadata={}, objects={})

    def restore(state):
        scene.text_description = state["text_description"]
        scene.metadata = dict(state["metadata"])
        scene.objects = {
            object_id: SimpleNamespace(
                object_type=SimpleNamespace(value=object_data["object_type"])
            )
            for object_id, object_data in state["objects"].items()
        }

    scene.restore_from_state_dict = Mock(side_effect=restore)
    contract = {"source": "current"}

    scene_generation._restore_rejected_furniture_checkpoint(
        scene=scene,
        checkpoint_state_path=checkpoint_path,
        room_prompt="current prompt",
        intent_contract=contract,
        attempt=1,
    )

    assert scene.text_description == "current prompt"
    assert scene.scenebenchmark_intent_contract is contract
    assert scene.metadata["scenebenchmark_intent_contract"] is contract
    assert set(scene.objects) == {"table_0"}


@pytest.mark.parametrize(
    ("contents", "expected"),
    [
        (None, "checkpoint is missing"),
        ("not json", "checkpoint is unreadable"),
        (
            json.dumps({"objects": {"wall_0": {"object_type": "wall"}}}),
            "no furniture objects",
        ),
    ],
)
def test_rejected_furniture_checkpoint_requires_a_valid_candidate(
    tmp_path, contents, expected
) -> None:
    checkpoint_path = tmp_path / "scene_state.json"
    if contents is not None:
        checkpoint_path.write_text(contents, encoding="utf-8")
    scene = SimpleNamespace(text_description="", metadata={}, objects={})
    scene.restore_from_state_dict = Mock(
        side_effect=lambda state: setattr(
            scene,
            "objects",
            {
                object_id: SimpleNamespace(
                    object_type=SimpleNamespace(value=object_data["object_type"])
                )
                for object_id, object_data in state.get("objects", {}).items()
            },
        )
    )

    with pytest.raises((FileNotFoundError, RuntimeError), match=expected) as error:
        scene_generation._restore_rejected_furniture_checkpoint(
            scene=scene,
            checkpoint_state_path=checkpoint_path,
            room_prompt="current prompt",
            intent_contract={"source": "current"},
            attempt=2,
        )

    assert "stage=furniture" in str(error.value)
    assert "attempt=2" in str(error.value)
    assert str(checkpoint_path) in str(error.value)


class _ReviewPlannerAgent(BaseStatefulAgent):
    def __init__(self) -> None:
        self.cfg = SimpleNamespace(
            max_critique_rounds=1,
            planner_context_limits={},
            auto_score_after_design_attempts=False,
        )
        self.scene = Mock()

    @property
    def agent_type(self) -> AgentType:
        return AgentType.FURNITURE

    def _get_final_scores_directory(self) -> Path:
        return Path("/tmp")

    def _get_critique_prompt_enum(self):
        return None

    def _get_design_change_prompt_enum(self):
        return None

    def _get_initial_design_prompt_enum(self):
        return None

    def _get_initial_design_prompt_kwargs(self) -> dict:
        return {}

    def _set_placement_noise_profile(self, _mode) -> None:
        return None


def test_review_existing_hides_initial_design_and_blocks_early_finish() -> None:
    agent = _ReviewPlannerAgent()
    agent._planner_skip_initial_design = True
    agent._planner_review_existing = True
    tools = {tool.name: tool for tool in agent._create_planner_tools()}

    assert "request_initial_design" not in tools
    result = asyncio.run(tools["finish_stage"].on_invoke_tool(Mock(), "{}"))
    assert "FINISH_STAGE_BLOCKED" in result
    assert not agent._planner_terminal_stop


def test_review_existing_can_finish_after_a_workflow_call() -> None:
    agent = _ReviewPlannerAgent()
    agent._planner_skip_initial_design = True
    agent._planner_review_existing = True
    tools = {tool.name: tool for tool in agent._create_planner_tools()}
    agent._planner_review_existing_workflow_calls = 1

    result = asyncio.run(tools["finish_stage"].on_invoke_tool(Mock(), "{}"))

    assert "FINISH_STAGE_ACCEPTED" in result
    assert agent._planner_terminal_stop
