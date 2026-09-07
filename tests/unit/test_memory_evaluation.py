"""Evidence gates: never turn preparation, a failed tool, or missing costs into gain."""

from __future__ import annotations

import json

from copy import deepcopy
from pathlib import Path

import pytest

from scenesmith.scene_expert.evaluation_costs import collect_attempt_costs
from scenesmith.scene_expert.memory.evidence import constraint_observation
from scenesmith.scene_expert.memory.state import state_hash
from scenesmith.scene_expert.memory.usage import collect_memory_usage
from scenesmith.scene_expert.paired_metrics import compare_run_metrics
from scenesmith.scene_expert.run_metrics import collect_run_metrics, write_run_metrics
from tests.unit.test_sceneexpert_paired_metrics import _run


def test_assignment_inventory_retains_never_started_cases(tmp_path: Path) -> None:
    inventory = tmp_path / "assigned_cases.csv"
    inventory.write_text(
        "batch_id,scene_index,prompt,case_id,critic_goal\n"
        "batch_001,0,room A,case-a,quality\n"
        "batch_002,1,room B,case-b,quality\n",
        encoding="utf-8",
    )
    metrics = collect_run_metrics(tmp_path, process_exit_code=1)
    assert metrics["assignment_inventory_complete"] is True
    assert metrics["summary"]["expected_scenes"] == 2
    assert metrics["summary"]["missing_or_nonterminal_scenes"] == 2
    write_run_metrics(metrics)
    assert (tmp_path / "metrics" / "memory_usage.json").is_file()
    assert (tmp_path / "metrics" / "attempt_costs.json").is_file()
    inventory.write_text(
        "batch_id,scene_index,prompt,case_id\n../escape,0,bad,case-a\n",
        encoding="utf-8",
    )
    assert collect_run_metrics(tmp_path)["assignment_inventory_complete"] is False


def test_unknown_assignment_cannot_claim_all_assigned_pair_gain() -> None:
    before = _run("cold", ready=True, time_sec=100, critic=0.8)
    after = _run("warm", ready=True, time_sec=80, critic=0.9)
    before.pop("assignment_inventory_complete")
    result = compare_run_metrics(before, after)
    assert result["comparison_ready"] is False
    assert (
        "assigned_case_inventory_missing_or_invalid" in result["data_quality_warnings"]
    )


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def fixture(root: Path) -> tuple[dict, dict, Path]:
    constraint = {
        "constraint_id": "current-facing",
        "relation": "facing",
        "subjects": {"category": "chair"},
        "targets": {"category": "table"},
    }
    check = constraint_observation(
        {
            "check_id": "check-current",
            "label": "pass",
            "contract_state": "satisfied",
            "scoring_tier": "core",
        },
        constraint,
        "furniture",
    )
    source = {
        "memory_id": "same-id",
        "memory_type": "success",
        "content_hash": "source-hash",
        "source_task_ids": ["other-task"],
        "spatial_relations": [
            {"subject_role": "chair", "target_role": "table", "relation_type": "facing"}
        ],
    }
    item = {
        "source": source,
        "text": "[Memory success:same-id / adapted]\nCheck the chair axis; rotate chair_1 toward table_1.",
        "adaptation": {
            "bindings": [
                {
                    "source_role": "chair",
                    "current_role": "chair",
                    "object_ids": ["chair_1"],
                }
            ]
        },
    }
    activity = {
        "stages": {
            "furniture": {
                "retrieval": {"selections": [source]},
                "injection": {
                    "schema_version": "memory-context.v1",
                    "accepted_items": [item],
                },
                "relation_context": {"hard_constraints": [constraint]},
                "verify_report": {
                    "stage": "furniture",
                    "pass_stage": True,
                    "hard_check_report": {"constraint_evidence": [check]},
                },
            }
        }
    }
    state = {"schema_version": "memory-scene-state.v1", "room": {}, "objects": []}
    state["fingerprint"] = state_hash(state)
    payload = {
        "stage": "furniture",
        "agent_role": "designer",
        "event": "request_design_change",
        "prompt": "Native feedback\n" + item["text"],
        "output": "I used memory.",
        "context_snapshot": {"decision_state": state},
        "agent_trace": {
            "tool_calls": [
                {
                    "id": "call-1",
                    "function": {
                        "name": "rotate_object",
                        "arguments": {"object_id": "chair_1", "yaw": 90},
                    },
                }
            ],
            "tool_results": [
                {
                    "tool_call_id": "call-1",
                    "output": {"success": True, "object_id": "chair_1"},
                }
            ],
        },
    }
    ref = Path("scene_expert/audit/llm_payloads/designer.json")
    write(root / ref, payload)
    write(
        root / "scene_expert/timing/llm_calls.jsonl",
        {
            "stage": "furniture",
            "agent_role": "designer",
            "event": "request_design_change",
            "payload_ref": ref.as_posix(),
            "created_at": "2026-09-07T00:01:00Z",
            "elapsed_sec": 10,
            "token_usage": {"total_tokens": 100},
        },
    )
    return activity, payload, root / ref


def test_exact_payload_action_and_target_are_separate_and_auditable(tmp_path):
    activity, payload, _ = fixture(tmp_path)
    result = collect_memory_usage(tmp_path, activity)
    row = result["items"][0]
    assert row["delivered"] and row["action_observed"] and row["target_verified"]
    assert row["causal_benefit"] is None
    assert row["requests"][0]["related_mutations"][0]["tool_call_id"] == "call-1"
    assert (
        result["initial_checkpoints"]["furniture"]["fingerprint"]
        == payload["context_snapshot"]["decision_state"]["fingerprint"]
    )


@pytest.mark.parametrize(
    "fault",
    [
        "metadata_only",
        "assistant_claim",
        "wrong_id",
        "failed_tool",
        "wrong_call_id",
        "duplicate_result",
        "provider_error",
        "source_hash",
    ],
)
def test_unproven_or_failed_actions_never_count_as_memory_use(tmp_path, fault):
    activity, payload, path = fixture(tmp_path)
    if fault == "metadata_only":
        payload["context_snapshot"]["memory_text"] = payload.pop("prompt")
    elif fault == "assistant_claim":
        payload["agent_trace"] = {}
    elif fault == "wrong_id":
        payload["agent_trace"]["tool_calls"][0]["function"]["arguments"][
            "object_id"
        ] = "chair_10"
        payload["agent_trace"]["tool_results"][0]["output"]["object_id"] = "chair_10"
    elif fault == "failed_tool":
        payload["agent_trace"]["tool_results"][0]["output"]["success"] = False
    elif fault == "wrong_call_id":
        payload["agent_trace"]["tool_results"][0]["tool_call_id"] = "other"
    elif fault == "duplicate_result":
        payload["agent_trace"]["tool_results"] *= 2
    elif fault == "provider_error":
        payload["error"] = "timeout"
    else:
        activity["stages"]["furniture"]["injection"]["accepted_items"][0]["source"] = {
            **activity["stages"]["furniture"]["retrieval"]["selections"][0],
            "content_hash": "wrong-hash",
        }
    write(path, payload)
    row = collect_memory_usage(tmp_path, activity)["items"][0]
    assert row["action_observed"] is None


def test_stage_pass_cannot_verify_specific_target(tmp_path):
    activity, _, _ = fixture(tmp_path)
    activity["stages"]["furniture"]["verify_report"]["hard_check_report"] = {
        "hard_valid": True
    }
    assert (
        collect_memory_usage(tmp_path, activity)["items"][0]["target_verified"] is None
    )


def test_typed_identity_and_payload_reference_traversal(tmp_path):
    activity, _, _ = fixture(tmp_path)
    source = deepcopy(activity["stages"]["furniture"]["retrieval"]["selections"][0])
    source["memory_type"] = "failure"
    activity["stages"]["furniture"]["retrieval"]["selections"].append(source)
    assert not collect_memory_usage(tmp_path, activity)["items"][1]["delivered"]
    write(
        tmp_path / "scene_expert/timing/llm_calls.jsonl",
        {
            "stage": "furniture",
            "agent_role": "designer",
            "event": "request_design_change",
            "payload_ref": "../secret.json",
        },
    )
    result = collect_memory_usage(tmp_path, activity)
    assert not result["items"][0]["payload_observed"]
    assert result["warnings"]


def cost_attempt(root: Path, attempt: int, start: str, end: str, duration: int) -> None:
    write(
        root / "scene_status.json",
        {
            "attempt": attempt,
            "status": "completed" if attempt == 2 else "failed",
            "updated_at": end,
        },
    )
    write(
        root / "scene_expert/trace/trace_000000_partial.json",
        {"total_time_sec": duration, "runtime_identity": {"scene_started_at": start}},
    )


def test_costs_include_failed_attempt_and_retry_gap_without_double_counting(tmp_path):
    current = tmp_path / "scene_000"
    old = tmp_path / "failed_attempts/scene_000_attempt_01_stamp"
    cost_attempt(old, 1, "2026-09-07T00:00:00Z", "2026-09-07T00:01:00Z", 55)
    cost_attempt(current, 2, "2026-09-07T00:01:10Z", "2026-09-07T00:02:00Z", 45)
    write(
        current / "scene_expert/timing/llm_calls.jsonl",
        {
            "agent_role": "designer",
            "elapsed_sec": 40,
            "token_usage": {"total_tokens": 0},
        },
    )
    write(
        current / "scene_expert/timing/stage_working_timing.jsonl",
        {"module": "render", "elapsed_sec": 8},
    )
    cost = collect_attempt_costs(current, 0)
    assert cost["all_attempt_cost_complete"]
    assert cost["all_attempt_time_sec"] == 120
    assert cost["attempt_service_time_sec"] == 110
    assert cost["all_attempt_trace_time_sec"] == 100
    assert cost["final_attempt_trace_time_sec"] == 45
    assert (
        cost["attempts"][1]["role_costs"]["designer"]["token_usage_missing_calls"] == 0
    )


def test_missing_attempt_or_usage_is_unknown_not_zero(tmp_path):
    current = tmp_path / "scene_000"
    cost_attempt(current, 2, "2026-09-07T00:01:10Z", "2026-09-07T00:02:00Z", 45)
    write(
        current / "scene_expert/timing/llm_calls.jsonl",
        {"agent_role": "designer", "error": "timeout"},
    )
    cost = collect_attempt_costs(current, 0)
    assert cost["all_attempt_time_sec"] is None
    assert cost["observed_attempt_time_lower_bound_sec"] == 50
    assert (
        cost["attempts"][0]["role_costs"]["designer"]["token_usage_missing_calls"] == 1
    )


def test_pair_never_uses_last_success_trace_as_full_cost():
    baseline = _run("cold", ready=True, time_sec=100, critic=0.8)
    treatment = _run("warm", ready=True, time_sec=50, critic=0.8)
    treatment["scenes"][0]["all_attempt_time_sec"] = 150
    result = compare_run_metrics(baseline, treatment)
    assert result["pairs"][0]["time_delta_sec"] == 50
    assert result["pairs"][0]["treatment_final_trace_time_sec"] == 50
    treatment["scenes"][0]["all_attempt_cost_complete"] = False
    result = compare_run_metrics(baseline, treatment)
    assert result["pairs"][0]["time_delta_sec"] is None
    assert not result["speed_comparison_ready"]


@pytest.mark.parametrize("fault", ["duplicate", "checkpoint", "legacy"])
def test_pair_critical_identity_gates(tmp_path, fault):
    baseline = _run("cold", ready=True, time_sec=100, critic=0.8)
    treatment = _run("warm", ready=True, time_sec=80, critic=0.9)
    if fault == "duplicate":
        treatment["scenes"].append(deepcopy(treatment["scenes"][0]))
    elif fault == "checkpoint":
        treatment["scenes"][0]["initial_checkpoints"]["furniture"][0][
            "fingerprint"
        ] = "other-start"
    else:
        treatment["scenes"][0].pop("memory_usage")
    assert not compare_run_metrics(baseline, treatment)["comparison_ready"]


@pytest.mark.parametrize("missing_arm", ["baseline", "treatment", "both"])
def test_manual_runtime_labels_are_not_required_for_speed_comparison(missing_arm):
    baseline = _run("cold", ready=True, time_sec=100, critic=0.8)
    treatment = _run("warm", ready=True, time_sec=80, critic=0.9)
    for arm, metrics in (("baseline", baseline), ("treatment", treatment)):
        if missing_arm in {arm, "both"}:
            metrics["scenes"][0]["runtime_identity"] = {
                "hostname": "same-host",
                "software": {"python": "test-runtime"},
            }
    result = compare_run_metrics(baseline, treatment)
    assert result["comparison_ready"]
    assert result["speed_comparison_ready"]
    assert result["summary"]["all_assigned_total_time_delta_sec"] == -20
    assert result["pairs"][0]["runtime_resource_match"] is None
    assert result["hardware_equivalence_verified"] is False


@pytest.mark.parametrize("field", ["resource_class", "service_deployment"])
def test_recorded_resource_contradictions_still_block_speed_comparison(field):
    baseline = _run("cold", ready=True, time_sec=100, critic=0.8)
    treatment = _run("warm", ready=True, time_sec=80, critic=0.9)
    treatment["scenes"][0]["runtime_identity"][field] = "actually-different"
    result = compare_run_metrics(baseline, treatment)
    assert result["comparison_ready"]
    assert result["speed_comparison_ready"] is False
    assert (
        "conflicting_legacy_runtime_resource_labels" in result["data_quality_warnings"]
    )


@pytest.mark.parametrize("fault", ["model", "software", "cost", "config", "source"])
def test_removing_manual_labels_preserves_objective_pair_gates(fault):
    baseline = _run("cold", ready=True, time_sec=100, critic=0.8)
    treatment = _run("warm", ready=True, time_sec=80, critic=0.9)
    for metrics in (baseline, treatment):
        metrics["scenes"][0]["runtime_identity"] = {
            "software": {"python": "test-runtime"}
        }
    row = treatment["scenes"][0]
    if fault == "model":
        treatment["experiment_identity"]["models"] = ["different-model"]
    elif fault == "software":
        row["runtime_identity"]["software"] = {"python": "different-runtime"}
    elif fault == "cost":
        row["all_attempt_cost_complete"] = False
    elif fault == "config":
        row["control_signature"] = "different-config"
    else:
        row["source_bundle_hash"] = "different-source"
    result = compare_run_metrics(baseline, treatment)
    assert result["speed_comparison_ready"] is False


def test_creation_requires_actual_new_object_state_not_free_text_role(tmp_path):
    activity, payload, path = fixture(tmp_path)
    entry = activity["stages"]["furniture"]
    entry["injection"]["accepted_items"][0]["adaptation"]["bindings"][0][
        "object_ids"
    ] = []
    payload["agent_trace"]["tool_calls"][0]["function"] = {
        "name": "add_furniture_to_scene_tool",
        "arguments": {"asset_id": "asset-chair"},
    }
    write(path, payload)
    assert (
        collect_memory_usage(tmp_path, activity)["items"][0]["action_observed"] is None
    )
    entry["post_scene_state"] = {
        "objects": [{"object_id": "chair_1", "name": "office_chair_1"}]
    }
    assert (
        collect_memory_usage(tmp_path, activity)["items"][0]["action_observed"] is True
    )


def test_repeated_stage_history_preserves_prior_decision_and_target(tmp_path):
    activity, payload, path = fixture(tmp_path)
    old = deepcopy(activity["stages"]["furniture"])
    old["prepared_at_epoch"] = 0
    old["finished_at_epoch"] = 100
    old["verify_report"]["hard_check_report"]["constraint_evidence"] = []
    activity["stage_attempt_history"] = [{"stage": "furniture", "entry": old}]
    rows = collect_memory_usage(tmp_path, activity)["items"]
    # This payload is later than the archived stage; cannot credit it twice.
    assert not rows[0]["delivered"]
    assert rows[0]["target_verified"] is None
    assert rows[1]["delivered"] and rows[1]["target_verified"]


def test_invalid_optional_evidence_does_not_raise(tmp_path):
    result = collect_memory_usage(
        tmp_path, {"stages": {"furniture": {"injection": "invalid"}}}
    )
    assert not result["evidence_complete"]
    assert result["warnings"]


def test_runtime_captures_initial_status_without_git_requirement(tmp_path, monkeypatch):
    from scenesmith.scene_expert.trace_logger import TraceLogger

    write(
        tmp_path / "scene_000/scene_status.json",
        {"status": "running", "updated_at": "2026-09-07T00:00:00Z"},
    )
    monkeypatch.setenv("SCENEEXPERT_EVAL_RESOURCE_CLASS", "test-gpu")
    monkeypatch.setenv("SCENEEXPERT_EVAL_SERVICE_DEPLOYMENT", "old-placeholder")
    logger = TraceLogger(str(tmp_path), scene_index=0, prompt="Office")
    trace = json.loads(logger.save_partial().read_text(encoding="utf-8"))
    assert trace["runtime_identity"]["scene_started_at"] == "2026-09-07T00:00:00Z"
    assert "resource_class" not in trace["runtime_identity"]
    assert "service_deployment" not in trace["runtime_identity"]


def test_checkpoint_registration_and_file_tamper_detection(tmp_path):
    from scenesmith.scene_expert.checkpoint_evaluation import register, verify

    write(tmp_path / "fixed.json", {"fixed": True})
    identities = {
        key: key + "-hash"
        for key in (
            "shared_base_fingerprint",
            "task_spec_fingerprint",
            "intent_contract_fingerprint",
            "decision_state_fingerprint",
        )
    }
    spec = {
        "problems": [
            {
                "case_id": f"case-{i}",
                "task_id": f"task-{i}",
                "stage": "furniture",
                "inputs": {
                    key: "fixed.json"
                    for key in (
                        "checkpoint",
                        "task_spec",
                        "intent_contract",
                        "model_config",
                        "tool_config",
                    )
                },
                "expected_identity": identities,
            }
            for i in range(6)
        ],
        "acceptance": {
            "max_mean_quality_loss": 0.01,
            "min_cost_reduction_fraction": 0.1,
        },
    }
    manifest = register(spec, tmp_path)
    assert verify(manifest) == []
    write(tmp_path / "fixed.json", {"fixed": False})
    assert any("checkpoint_input_changed" in reason for reason in verify(manifest))
    spec["problems"] *= 2
    with pytest.raises(ValueError):
        register(spec, tmp_path)


def test_missing_cases_and_duplicate_comparisons_do_not_pass_acceptance(tmp_path):
    from scenesmith.scene_expert.checkpoint_evaluation import assess
    from scenesmith.scene_expert.memory.evidence import evidence_hash

    manifest = {
        "registered_at": "2026-01-01T00:00:00Z",
        "problems": [
            {
                "case_id": "bedroom_a",
                "task_id": "task-a",
                "expected_identity": {},
                "inputs": {},
            }
        ],
        "acceptance": {
            "max_mean_quality_loss": 0.01,
            "min_cost_reduction_fraction": 0.1,
        },
    }
    manifest["manifest_hash"] = evidence_hash(manifest)
    comparison = compare_run_metrics(
        _run("cold", ready=True, time_sec=100, critic=0.8),
        _run("warm", ready=True, time_sec=80, critic=0.9),
    )
    result = assess(manifest, [comparison, comparison])
    assert result["decision"] == "inconclusive"
    assert "need_two_independent_balanced_pairs" in result["reasons"]


def test_registered_balanced_acceptance_is_result_neutral(tmp_path):
    from scenesmith.scene_expert.checkpoint_evaluation import assess, register

    write(tmp_path / "fixed.json", {"fixed": True})
    identity = {
        key: key + "-hash"
        for key in (
            "shared_base_fingerprint",
            "task_spec_fingerprint",
            "intent_contract_fingerprint",
            "decision_state_fingerprint",
        )
    }
    problems = [
        {
            "case_id": f"c{i}",
            "task_id": f"t{i}",
            "stage": "furniture",
            "inputs": {
                key: "fixed.json"
                for key in (
                    "checkpoint",
                    "task_spec",
                    "intent_contract",
                    "model_config",
                    "tool_config",
                )
            },
            "expected_identity": identity,
        }
        for i in range(6)
    ]
    manifest = register(
        {
            "problems": problems,
            "acceptance": {
                "max_mean_quality_loss": 0.01,
                "min_cost_reduction_fraction": 0.1,
            },
        },
        tmp_path,
    )
    comparisons = [
        {
            "baseline_run_id": f"off-{i}",
            "treatment_run_id": f"on-{i}",
            "comparison_ready": True,
            "speed_comparison_ready": True,
            "quality_delta_ready": True,
            "checkpoint_plan_hash": manifest["manifest_hash"],
            "earliest_scene_started_at": "2030-01-01T00:00:00Z",
            "arm_orders": [order],
            "pairs": [
                {
                    "case_id": row["case_id"],
                    "task_id": row["task_id"],
                    "fixed_input_identity": identity,
                    "critic_score_delta": 0.01,
                    "baseline_time_sec": 100,
                    "treatment_time_sec": 80,
                    "outcome_transition": "both_completed",
                }
                for row in problems
            ],
        }
        for i, order in enumerate(("off_on", "on_off"))
    ]
    assert (
        assess(manifest, comparisons)["decision"]
        == "meets_preregistered_engineering_threshold"
    )
    for result in comparisons:
        for row in result["pairs"]:
            row["critic_score_delta"] = -0.2
    assert (
        assess(manifest, comparisons)["decision"]
        == "does_not_meet_preregistered_engineering_threshold"
    )
    comparisons[0]["pairs"].pop()
    assert assess(manifest, comparisons)["decision"] == "inconclusive"


def test_bad_usage_shape_blocks_cost_claim_without_crashing(tmp_path):
    current = tmp_path / "scene_000"
    cost_attempt(current, 1, "2026-09-07T00:00:00Z", "2026-09-07T00:01:00Z", 55)
    write(current / "scene_expert/timing/llm_calls.jsonl", {"token_usage": "broken"})
    result = collect_attempt_costs(current, 0)
    assert not result["all_attempt_cost_complete"]
    assert result["all_attempt_time_sec"] is None
    assert result["warnings"]
