"""Tests for production SceneExpert stage-commit retries."""

import asyncio
import json
import time

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
from omegaconf import OmegaConf
from openai import APITimeoutError

from scenesmith.experiments import indoor_scene_generation as scene_generation
from scenesmith.agent_utils import base_stateful_agent
from scenesmith.agent_utils.base_stateful_agent import BaseStatefulAgent
from scenesmith.agent_utils.room import AgentType
from scenesmith.scene_expert.harness import Harness, RepairDecision
from scenesmith.scene_expert.context_bundle import build_llm_call_debug_record
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
from scenesmith.scenebenchmark_critic.stage_ownership import (
    normalize_result_stage_ownership,
)
from scenesmith.scenebenchmark_critic.intent_contract import (
    apply_contract_execution_states,
)
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.extensions.intent_contract import (
    evaluate_intent_contract_extensions,
)


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


def test_stage_verifier_inventory_accepts_specialized_category_parent(tmp_path) -> None:
    report = StageVerifier().verify(
        stage="furniture",
        stage_output_dir=str(tmp_path),
        task_spec=SceneTaskSpec(
            room_type="living_room",
            style="standard",
            required_large_objects=["chair"],
        ),
        scene_state_info={
            "object_records": [
                {
                    "name": "armchair_0",
                    "aliases": ["armchair"],
                    "description": "Upholstered lounge chair",
                }
            ]
        },
    )

    assert report.pass_stage
    assert not report.issues


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


def test_stage_verifier_does_not_repair_upstream_failure_in_later_stage(
    tmp_path,
) -> None:
    report = StageVerifier().verify(
        stage="ceiling_mounted",
        stage_output_dir=str(tmp_path),
        task_spec=SceneTaskSpec(room_type="living_room", style="standard"),
        scene_state_info={"object_names": []},
        deterministic_critic_payload=_binding_failure_payload(
            earliest_stage="furniture"
        ),
    )

    assert report.pass_stage


def test_stage_verifier_fails_real_support_readiness_at_target_stage(tmp_path) -> None:
    constraint = {
        "constraint_id": "utensil_on_cooler",
        "relation": "on_top_of",
        "subjects": {
            "category": "utensil",
            "count": 1,
            "stage": "manipuland",
        },
        "targets": {"category": "cooler", "count": 1},
        "source": "explicit_prompt",
        "strength": "hard",
        "stage": "manipuland",
        "evidence_span": "utensils on the cooler",
    }
    case_pack = {
        "stage": "furniture",
        "scene_geometry": {
            "objects": [
                {
                    "id": "cooler_0",
                    "category": "cooler",
                    "object_type": "furniture",
                    "bbox_world": {
                        "min": [-0.4, -0.3, 0.0],
                        "max": [0.4, 0.3, 0.4],
                    },
                    "support_surfaces": [],
                }
            ]
        },
        "intent_contract": {"constraints": [constraint]},
    }
    results = evaluate_intent_contract_extensions(case_pack)
    apply_contract_execution_states(case_pack, results)

    assert results[0]["diagnostics"]["earliest_stage"] == "furniture"
    report = StageVerifier().verify(
        stage="furniture",
        stage_output_dir=str(tmp_path),
        task_spec=SceneTaskSpec(room_type="dining_room", style="standard"),
        scene_state_info={"object_names": ["cooler_0"]},
        deterministic_critic_payload={"results": results},
    )

    assert not report.pass_stage
    assert report.issues[0].constraint_id == "utensil_on_cooler"


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


@pytest.mark.parametrize(
    ("result", "expected_blocker"),
    [
        (
            {
                "check_id": "room_containment__cabinet_0",
                "metric": "functional_dependency",
                "relation_type": "room_containment",
                "label": "fail",
                "scoring_tier": "core",
                "primary_object": "cabinet_0",
                "diagnostics": {"earliest_stage": "furniture"},
            },
            "room_containment",
        ),
        (
            {
                "check_id": "intent__support_readiness__basket_0",
                "metric": "functional_dependency",
                "relation_type": "support_readiness",
                "label": "fail",
                "scoring_tier": "core",
                "primary_object": "basket_0",
                "intent_constraint": {"relation": "on_top_of"},
                "diagnostics": {
                    "support_readiness": True,
                    "earliest_stage": "furniture",
                },
            },
            "support_readiness",
        ),
    ],
)
def test_final_blocker_survives_resumed_stage_filtering(
    result, expected_blocker
) -> None:
    payload = {"results": [result]}

    resumed_stage = StageVerifier().verify(
        stage="wall_mounted",
        stage_output_dir=".",
        task_spec=SceneTaskSpec(room_type="living_room", style="standard"),
        scene_state_info={"object_names": []},
        deterministic_critic_payload=payload,
    )
    assert resumed_stage.pass_stage

    final_report = FullVerifier().verify(
        stage_reports=[resumed_stage], deterministic_critic_payload=payload
    )

    assert final_report.non_degradable_blockers == [expected_blocker]
    with pytest.raises(RuntimeError, match="non-degradable blocker"):
        scene_generation._raise_for_non_degradable_final_blockers(final_report)


def test_finalization_keeps_ordinary_quality_failure_degradable() -> None:
    scene_generation._raise_for_non_degradable_final_blockers(
        FullVerifyReport(deterministic_pass=False)
    )


def test_hook_finalize_refreshes_final_deterministic_payload(tmp_path) -> None:
    fresh_payload = _binding_failure_payload(earliest_stage="manipuland")
    scene = SimpleNamespace()
    runner = object.__new__(SceneExpertHookRunner)
    runner._mode = "harness_only"
    runner._component_flags = {"verifier": True, "trace": True}
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
    runner._component_flags = {"verifier": True}
    runner._current_stage = stage
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
    runner._component_flags = {"verifier": True, "repair": True}
    runner._current_stage = stage
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


def test_legacy_noop_signal_never_skips_native_stage() -> None:
    runner = object.__new__(SceneExpertHookRunner)
    runner._current_stage = "wall_mounted"
    runner._current_planner_trace = {"status": "no_op"}
    runner._stage_policies = {"wall_mounted": "auto"}

    assert not runner.should_skip_stage_agent("wall_mounted")
    assert not runner.should_skip_stage_agent("ceiling_mounted")

    runner._current_planner_trace = {"status": "completed"}
    assert not runner.should_skip_stage_agent("wall_mounted")


def test_pipeline_noop_gate_is_non_skipping_compatibility_shim() -> None:
    hooks = Mock()
    hooks.should_skip_stage_agent.return_value = True

    assert not scene_generation._should_skip_noop_scene_expert_stage(
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


@pytest.mark.parametrize("blocker", ["room_containment", "support_readiness"])
def test_degraded_policy_does_not_advance_typed_blocker(tmp_path, blocker) -> None:
    hooks = Mock()
    hooks.post_stage.return_value = StageCommitResult(
        stage="furniture",
        passed=False,
        retryable=False,
        reason="Repair budget exhausted",
        quality_failure=True,
        non_degradable_blockers=(blocker,),
    )

    with pytest.raises(
        scene_generation.SceneExpertStageCommitError,
        match=rf"non-degradable blocker\(s\): {blocker}",
    ):
        scene_generation._commit_scene_expert_stage(
            hooks=hooks,
            stage="furniture",
            scene=Mock(),
            room_dir=tmp_path,
            allow_degraded_quality=True,
        )

    hooks.accept_degraded_stage.assert_not_called()


def test_accept_degraded_stage_advances_harness_for_next_stage() -> None:
    runner = object.__new__(SceneExpertHookRunner)
    runner._current_stage = "wall_mounted"
    runner._completed_stages = ["floor_plan", "furniture"]
    runner._pending_stage_repairs = {"wall_mounted": (Mock(), Mock())}
    runner._harness = Harness(SimpleNamespace())
    runner._component_flags = {"harness": True}

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
    runner._component_flags = {"verifier": True, "repair": True}
    runner._current_stage = "furniture"
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


def test_non_deterministic_containment_named_issue_is_not_a_stage_blocker(
    tmp_path,
) -> None:
    failed_report = StageVerifyReport(
        stage="furniture",
        pass_stage=False,
        issues=[
            VerifyIssue(
                issue_type="low_functionality",
                relation="room_containment",
                scoring_tier="core",
                description="Visual review requires another repair attempt",
            )
        ],
    )
    runner = object.__new__(SceneExpertHookRunner)
    runner._mode = "harness_only"
    runner._component_flags = {"verifier": True, "repair": True}
    runner._current_stage = "furniture"
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

    result = runner.post_stage(
        "furniture", SimpleNamespace(text_description="brief", objects={}), tmp_path
    )

    assert result.non_degradable_blockers == ()


def test_support_readiness_is_a_non_degradable_stage_blocker(tmp_path) -> None:
    failed_report = StageVerifyReport(
        stage="furniture",
        pass_stage=False,
        issues=[
            VerifyIssue(
                issue_type="deterministic_relation_failure",
                relation="on_top_of",
                scoring_tier="core",
                description="Basket has no verified support surface",
                diagnostics={"support_readiness": True},
            )
        ],
    )
    runner = object.__new__(SceneExpertHookRunner)
    runner._mode = "harness_only"
    runner._component_flags = {"verifier": True, "repair": True}
    runner._current_stage = "furniture"
    runner._original_text_descriptions = {"furniture": "original prompt"}
    runner._stage_verifier = Mock(verify=Mock(return_value=failed_report))
    runner._task_spec = SimpleNamespace(room_type="bathroom")
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

    result = runner.post_stage(
        "furniture", SimpleNamespace(text_description="brief", objects={}), tmp_path
    )

    assert result.non_degradable_blockers == ("support_readiness",)


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


class _TransactionalScene:
    def __init__(self) -> None:
        self.state = {"objects": {"existing_0": {"category": "existing"}}}
        self.restore_calls = 0

    def to_state_dict(self) -> dict:
        return json.loads(json.dumps(self.state))

    def restore_from_state_dict(self, state: dict) -> None:
        self.restore_calls += 1
        self.state = json.loads(json.dumps(state))

    def content_hash(self) -> str:
        return json.dumps(self.state, sort_keys=True)


def _designer_transaction_agent() -> _ReviewPlannerAgent:
    agent = _ReviewPlannerAgent()
    agent.cfg = SimpleNamespace(
        agents=SimpleNamespace(designer_agent=SimpleNamespace(max_turns=2))
    )
    agent.scene = _TransactionalScene()
    agent.furniture_safety_controller = SimpleNamespace(
        enabled=True,
        begin_designer_call=Mock(),
        end_designer_call=Mock(side_effect=RuntimeError("cleanup failed")),
    )
    invalid_state = SimpleNamespace(
        hard_valid=False,
        hard_reasons=["missing required sofa: expected 1, found 0"],
    )
    agent._evaluate_current_furniture_hard_state = Mock(return_value=invalid_state)
    agent._checkpoint_eligible_furniture_hard_state = Mock(
        side_effect=lambda state: state
    )
    agent._try_deterministic_repair_for_hard_state = Mock(
        side_effect=AssertionError("abort must not run deterministic repair")
    )
    agent._remember_furniture_hard_valid_scene_state = Mock()
    agent._persist_furniture_hard_valid_checkpoint = Mock()
    agent._end_furniture_design_transaction = Mock(
        side_effect=AssertionError("abort must not commit the transaction")
    )
    agent.prompt_registry = Mock()
    agent.prompt_registry.get_prompt.return_value = "Design the room."
    agent._retrieve_working_memory_for_designer = Mock(return_value="")
    agent._prepare_stage_context_for_llm = Mock(return_value="")
    agent._reasoning_persistence_context_for_session = lambda _: nullcontext()
    agent.designer_session = SimpleNamespace(session_id="designer-test")
    agent.designer = SimpleNamespace()
    agent.rendering_manager = SimpleNamespace(
        last_render_dir=None,
        clear_cache=Mock(),
    )
    agent._create_run_config = Mock(return_value=None)
    agent._record_llm_call_debug = Mock()
    return agent


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


@pytest.mark.parametrize(
    ("hashes", "expected_mutations"),
    [(["before", "after"], 1), (["before", "before"], 0)],
)
def test_design_change_counts_only_committed_scene_mutation(
    hashes: list[str], expected_mutations: int
) -> None:
    agent = _ReviewPlannerAgent()
    agent.scene.content_hash.side_effect = hashes
    agent.stage_working_memory = Mock()
    agent._planner_orchestration_calls = 0
    agent._request_design_change_impl = AsyncMock(return_value="designer completed")
    tools = {tool.name: tool for tool in agent._create_planner_tools()}

    result = asyncio.run(
        tools["request_design_change"].on_invoke_tool(
            Mock(), '{"instruction": "repair the current candidate"}'
        )
    )

    assert "designer completed" in result
    assert agent._planner_successful_designer_mutations == expected_mutations


@pytest.mark.parametrize(
    ("operation", "expected_call_kind"),
    [("initial", "initial"), ("change", "change")],
)
def test_failed_designer_transaction_aborts_without_repair_or_masking_timeout(
    monkeypatch, caplog, operation: str, expected_call_kind: str
) -> None:
    agent = _designer_transaction_agent()
    before_state = agent.scene.to_state_dict()
    before_hash = agent.scene.content_hash()
    timeout = APITimeoutError(
        request=httpx.Request("POST", "http://127.0.0.1:8002/v1/chat/completions")
    )

    async def mutate_then_timeout(**_kwargs):
        agent.scene.state["objects"]["partial_0"] = {"category": "partial"}
        raise timeout

    monkeypatch.setattr(base_stateful_agent.Runner, "run", mutate_then_timeout)
    caplog.set_level("ERROR", logger=base_stateful_agent.__name__)

    with pytest.raises(APITimeoutError) as error:
        if operation == "initial":
            asyncio.run(agent._request_initial_design_impl())
        else:
            asyncio.run(agent._request_design_change_impl("repair the layout"))

    assert error.value is timeout
    assert agent.scene.to_state_dict() == before_state
    assert agent.scene.content_hash() == before_hash
    assert agent.scene.restore_calls == 1
    agent.furniture_safety_controller.begin_designer_call.assert_called_once_with(
        call_kind=expected_call_kind
    )
    agent.furniture_safety_controller.end_designer_call.assert_called_once_with()
    agent._evaluate_current_furniture_hard_state.assert_called_once_with()
    agent._try_deterministic_repair_for_hard_state.assert_not_called()
    agent._remember_furniture_hard_valid_scene_state.assert_not_called()
    agent._persist_furniture_hard_valid_checkpoint.assert_not_called()
    agent._end_furniture_design_transaction.assert_not_called()
    agent.rendering_manager.clear_cache.assert_called_once_with()
    assert agent._record_llm_call_debug.call_args.kwargs["exception"] is timeout
    assert "Failed to end the furniture safety controller call" in caplog.text


def test_planner_terminal_child_failure_skips_no_mutation_recovery(monkeypatch) -> None:
    agent = _ReviewPlannerAgent()
    agent._planner_successful_designer_mutations = 0
    agent._planner_terminal_failure = {
        "operation": "request_initial_design",
        "child_agent": "designer",
        "error_type": "APITimeoutError",
        "error": "request timed out",
        "recovered": False,
    }
    agent.planner = SimpleNamespace(instructions="planner")
    agent.planner_session = SimpleNamespace(session_id="planner-test")
    agent._reasoning_persistence_context_for_session = lambda _: nullcontext()
    agent._create_run_config = lambda: None
    agent._record_module_timing = lambda *args, **kwargs: None
    agent._record_llm_call_debug = lambda **kwargs: None
    run = AsyncMock(return_value=SimpleNamespace(final_output="stopped"))
    monkeypatch.setattr(base_stateful_agent.Runner, "run", run)

    with pytest.raises(base_stateful_agent.PlannerStageFailure) as error:
        asyncio.run(
            agent._run_planner_workflow(
                runner_input="start", max_turns=2, require_initial_design=True
            )
        )

    assert error.value.reason == "child_failure"
    assert error.value.stage == "furniture"
    assert error.value.operation == "request_initial_design"
    assert error.value.root_error_type == "APITimeoutError"
    assert error.value.retryable is True
    assert run.await_count == 1


def test_planner_no_mutation_recovery_reports_called_workflow(monkeypatch) -> None:
    agent = _ReviewPlannerAgent()
    agent._planner_successful_designer_mutations = 0
    agent._planner_designer_workflow_calls = 1
    agent._planner_last_designer_workflow_operation = "request_initial_design"
    agent.planner = SimpleNamespace(instructions="planner")
    agent.planner_session = SimpleNamespace(session_id="planner-test")
    agent._reasoning_persistence_context_for_session = lambda _: nullcontext()
    agent._create_run_config = lambda: None
    agent._record_module_timing = lambda *args, **kwargs: None
    agent._record_llm_call_debug = lambda **kwargs: None
    run = AsyncMock(return_value=SimpleNamespace(final_output="stopped"))
    monkeypatch.setattr(base_stateful_agent.Runner, "run", run)

    with pytest.raises(
        base_stateful_agent.PlannerWorkflowNoMutationError,
    ) as error:
        asyncio.run(
            agent._run_planner_workflow(
                runner_input="start", max_turns=2, require_initial_design=True
            )
        )

    assert error.value.reason == "no_mutation"
    assert error.value.stage_execution_attempt == 1
    assert run.await_count == 2
    recovery_input = run.await_args_list[1].kwargs["input"]
    assert "produced no committed scene mutation" in recovery_input
    assert "returned without calling a workflow tool" not in recovery_input


def test_stage_execution_attempt_isolates_session_ids_but_keeps_db_paths(
    monkeypatch, tmp_path
) -> None:
    created: list[tuple[str, Path]] = []

    class FakeSession:
        def __init__(self, *, session_id: str, db_path: Path) -> None:
            self.session_id = session_id
            self.db_path = db_path
            created.append((session_id, db_path))

    agent = _ReviewPlannerAgent()
    agent.cfg = SimpleNamespace(session_memory=None)
    agent.logger = SimpleNamespace(output_dir=tmp_path)
    agent._stage_execution_attempt = 1
    monkeypatch.setattr(base_stateful_agent, "SQLiteSession", FakeSession)

    agent._create_sessions(session_prefix="ceiling_", stage_execution_attempt=1)
    first_attempt = list(created)
    created.clear()
    agent._create_sessions(session_prefix="ceiling_", stage_execution_attempt=2)
    second_attempt = list(created)

    assert {session_id for session_id, _ in first_attempt} == {
        "ceiling_designer_attempt_01",
        "ceiling_critic_attempt_01",
        "ceiling_planner_attempt_01",
    }
    assert {session_id for session_id, _ in second_attempt} == {
        "ceiling_designer_attempt_02",
        "ceiling_critic_attempt_02",
        "ceiling_planner_attempt_02",
    }
    assert {path.name for _, path in first_attempt} == {
        "ceiling_designer.db",
        "ceiling_critic.db",
        "ceiling_planner.db",
    }
    assert {path.name for _, path in second_attempt} == {
        "ceiling_designer.db",
        "ceiling_critic.db",
        "ceiling_planner.db",
    }


@pytest.mark.parametrize("attempt", [0, -1, False, 1.5])
def test_create_sessions_rejects_invalid_explicit_stage_execution_attempt(
    monkeypatch, tmp_path, attempt: object
) -> None:
    agent = _ReviewPlannerAgent()
    agent.cfg = SimpleNamespace(session_memory=None)
    agent.logger = SimpleNamespace(output_dir=tmp_path)
    agent._stage_execution_attempt = 1
    monkeypatch.setattr(base_stateful_agent, "SQLiteSession", object)

    with pytest.raises(ValueError, match="stage_execution_attempt"):
        agent._create_sessions(
            session_prefix="ceiling_", stage_execution_attempt=attempt
        )


@pytest.mark.parametrize("attempt", [0, -1, False, 1.5])
def test_planner_stage_failure_requires_integral_positive_execution_attempt(
    attempt: object,
) -> None:
    with pytest.raises(ValueError, match="stage_execution_attempt"):
        base_stateful_agent.PlannerStageFailure(
            reason="no_tool_call",
            stage="furniture",
            workflow_calls=0,
            successful_mutations=0,
            operation="request_initial_design",
            stage_execution_attempt=attempt,
        )


@pytest.mark.parametrize("value", [False, True, 1.5, -1, 0])
def test_stage_execution_attempt_requires_positive_integer(value: object) -> None:
    agent = _ReviewPlannerAgent()
    agent.cfg = SimpleNamespace(stage_execution_attempt=value)

    with pytest.raises(ValueError, match="stage_execution_attempt"):
        agent._configured_stage_execution_attempt()


@pytest.mark.parametrize(
    ("role", "budget"),
    [("planner", 1536), ("designer", 24576), ("critic", 12288)],
)
def test_role_completion_budget_reaches_model_settings(role: str, budget: int) -> None:
    agent = _ReviewPlannerAgent()
    agent.cfg = SimpleNamespace(
        openai=SimpleNamespace(
            model="Qwen/test",
            service_tier=None,
            reasoning_effort=SimpleNamespace(
                planner="none", designer="none", critic="none"
            ),
            max_output_tokens=SimpleNamespace(
                planner=1536, designer=24576, critic=12288
            ),
        )
    )

    settings = agent._get_model_settings(settings_key=role)

    assert settings.max_tokens == budget


def test_critic_tool_choice_is_a_string_request_setting() -> None:
    agent = _ReviewPlannerAgent()
    agent.cfg = SimpleNamespace(
        openai=SimpleNamespace(
            model="Qwen/test",
            service_tier=None,
            reasoning_effort=SimpleNamespace(critic="none"),
            max_output_tokens=SimpleNamespace(critic=12288),
        )
    )

    settings = agent._get_model_settings(
        settings_key="critic", tool_choice="observe_scene"
    )

    assert settings.tool_choice == "observe_scene"
    assert settings.to_json_dict()["tool_choice"] == "observe_scene"


def test_stage_ownership_preserves_explicit_owner_and_wall_check_semantics() -> None:
    explicit = normalize_result_stage_ownership(
        {"metric": "future_metric", "diagnostics": {"earliest_stage": "ceiling"}}
    )
    wall = normalize_result_stage_ownership(
        {"metric": "visual_clearance", "check_id": "wall_visibility__mirror_0"}
    )

    assert explicit["diagnostics"]["earliest_stage"] == "ceiling_mounted"
    assert explicit["diagnostics"]["owner_resolution"] == "producer"
    assert wall["diagnostics"]["earliest_stage"] == "wall_mounted"
    assert wall["diagnostics"]["owner_resolution"] == "check_semantics"


def test_stage_ownership_fails_closed_for_malformed_and_ambiguous_physics() -> None:
    malformed = normalize_result_stage_ownership(
        {"metric": "future_metric", "diagnostics": "not-an-object"}
    )
    collision = normalize_result_stage_ownership(
        {
            "metric": "physics_collision",
            "evidence": {
                "physics_evidence": {
                    "object_a_id": "legacy_a",
                    "object_b_id": "legacy_b",
                }
            },
        }
    )

    assert malformed["diagnostics"] == {
        "owner_resolution": "final_only",
        "owner_resolution_error": "malformed_diagnostics",
    }
    assert collision["diagnostics"]["owner_resolution"] == "final_only"
    assert "earliest_stage" not in collision["diagnostics"]


def test_llm_audit_leaves_unknown_duration_unavailable_and_marks_length() -> None:
    result = SimpleNamespace(
        raw_responses=[
            SimpleNamespace(choices=[SimpleNamespace(finish_reason="length")])
        ]
    )

    record = build_llm_call_debug_record(
        stage="furniture",
        agent_role="designer",
        event="request_initial_design",
        prompt="place furniture",
        result=result,
    )

    assert record.elapsed_sec is None
    assert record.length_exhausted is True
    assert record.client_cancelled is None


def _agent_with_role_completion_budgets(
    **budgets: object,
) -> _ReviewPlannerAgent:
    agent = _ReviewPlannerAgent()
    agent.cfg = SimpleNamespace(
        openai=SimpleNamespace(max_output_tokens=SimpleNamespace(**budgets))
    )
    return agent


@pytest.mark.parametrize("value", [1, 1536, 4096.0])
def test_role_completion_budget_accepts_positive_integer_values(value: object) -> None:
    agent = _agent_with_role_completion_budgets(
        planner=1536, designer=24576, critic=value
    )

    assert agent._role_max_output_tokens("critic") == int(value)


def test_role_completion_budget_requires_each_configured_role() -> None:
    agent = _agent_with_role_completion_budgets(planner=1536, designer=24576)

    with pytest.raises(
        ValueError, match=r"openai\.max_output_tokens\.critic is required"
    ):
        agent._role_max_output_tokens("critic")


@pytest.mark.parametrize("value", [True, 1.5, "not-a-number", 0, -1])
def test_role_completion_budget_rejects_invalid_values(value: object) -> None:
    agent = _agent_with_role_completion_budgets(
        planner=1536, designer=24576, critic=value
    )

    with pytest.raises(ValueError, match="must be a positive integer"):
        agent._role_max_output_tokens("critic")


@pytest.mark.parametrize("value", [0, 1, 2, 2.0])
def test_provider_retry_budget_accepts_non_negative_integers(value: object) -> None:
    agent = _ReviewPlannerAgent()
    agent.cfg = SimpleNamespace(api_timeout=SimpleNamespace(max_retries=value))

    assert agent._provider_max_retries() == int(value)


@pytest.mark.parametrize("value", [True, 1.5, "invalid", -1])
def test_provider_retry_budget_rejects_invalid_values(value: object) -> None:
    agent = _ReviewPlannerAgent()
    agent.cfg = SimpleNamespace(api_timeout=SimpleNamespace(max_retries=value))

    with pytest.raises(ValueError, match="must be a non-negative integer"):
        agent._provider_max_retries()


def test_run_config_disables_hidden_provider_retries(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:8002/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    agent = _ReviewPlannerAgent()
    agent.cfg = SimpleNamespace(
        api_timeout=SimpleNamespace(max_retries=0),
        session_memory=SimpleNamespace(
            intra_turn_observation_stripping=SimpleNamespace(enabled=False)
        ),
    )

    run_config = agent._create_run_config()
    wrapped_client = run_config.model_provider._client

    assert wrapped_client._client.max_retries == 0


def test_model_settings_propagates_stage_agent_api_timeouts() -> None:
    agent = _ReviewPlannerAgent()
    agent.cfg = SimpleNamespace(
        api_timeout=SimpleNamespace(connect=10.0, read=600, write=600, pool=600),
        openai=SimpleNamespace(
            model="Qwen/test",
            service_tier=None,
            reasoning_effort=SimpleNamespace(planner="none"),
            max_output_tokens=SimpleNamespace(planner=1536),
        ),
    )

    settings = agent._get_model_settings(settings_key="planner")
    timeout = settings.extra_args["timeout"]

    assert timeout.connect == 10.0
    assert timeout.read == 600
    assert timeout.write == 600
    assert timeout.pool == 600


@pytest.mark.parametrize(
    "config_path",
    [
        "configurations/furniture_agent/base_furniture_agent.yaml",
        "configurations/wall_agent/base_wall_agent.yaml",
        "configurations/ceiling_agent/base_ceiling_agent.yaml",
        "configurations/manipuland_agent/base_manipuland_agent.yaml",
    ],
)
def test_stage_agent_llm_capacity_settings_are_consistent(config_path: str) -> None:
    cfg = OmegaConf.load(Path(__file__).resolve().parents[2] / config_path)

    assert OmegaConf.to_container(cfg.openai.max_output_tokens, resolve=True) == {
        "planner": 1536,
        "designer": 24576,
        "critic": 12288,
    }
    assert OmegaConf.to_container(cfg.api_timeout, resolve=True) == {
        "connect": 10.0,
        "read": 600,
        "write": 600,
        "pool": 600,
        "max_retries": 0,
    }


@pytest.mark.parametrize("stage", ["furniture", "wall_mounted", "ceiling_mounted"])
def test_unknown_owner_is_final_only_for_every_nonfinal_stage(
    stage: str, tmp_path
) -> None:
    payload = {
        "results": [
            {
                "check_id": "new_metric__unowned",
                "metric": "new_metric",
                "scoring_tier": "core",
                "label": "fail",
                "reason": "unowned deterministic failure",
                "diagnostics": {},
            }
        ]
    }

    report = StageVerifier().verify(
        stage=stage,
        stage_output_dir=str(tmp_path),
        task_spec=SceneTaskSpec(room_type="bedroom", style="standard"),
        scene_state_info={"object_names": []},
        deterministic_critic_payload=payload,
    )

    assert report.pass_stage
    assert not report.issues
    assert (
        not StageVerifier()
        .verify(
            stage="final",
            stage_output_dir=str(tmp_path),
            task_spec=SceneTaskSpec(room_type="bedroom", style="standard"),
            scene_state_info={"object_names": []},
            deterministic_critic_payload=payload,
        )
        .pass_stage
    )
