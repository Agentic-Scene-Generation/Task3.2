from __future__ import annotations

import csv
import hashlib
import json

from pathlib import Path

from scenesmith.scene_expert.run_metrics import collect_run_metrics, write_run_metrics


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_manifest(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("scene_index", "prompt", "case_id", "critic_goal"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "scene_index": 0,
                "prompt": "bedroom",
                "case_id": "bedroom",
                "critic_goal": "quality",
            }
        )
        writer.writerow(
            {
                "scene_index": 1,
                "prompt": "living room",
                "case_id": "long_living_room",
                "critic_goal": "quality",
            }
        )


def test_metrics_survive_partial_failure_and_attribute_repairs(tmp_path) -> None:
    output_root = tmp_path / "run_001"
    batch = output_root / "critic_on" / "batch_001"
    hydra = batch / "hydra"
    _write_manifest(batch / "batch_cases.csv")
    (output_root / "critic_on" / "batch_001.log").write_text(
        "not a batch directory", encoding="utf-8"
    )
    bedroom_task_id = "task_" + hashlib.sha256(b"bedroom").hexdigest()[:16]

    scene0 = hydra / "scene_000"
    _write_json(
        scene0 / "scene_status.json",
        {"status": "completed", "attempt": 1, "prompt": "bedroom"},
    )
    _write_json(
        hydra / "traces" / "trace_000000.json",
        {
            "status": "completed",
            "total_time_sec": 120.0,
            "experiment_name": "ablation_4c",
            "config_hash": "config-1",
            "experiment_signature": "stable-4c-signature",
            "model": "qwen-test",
            "code_provenance": {
                "git_revision": "abc123",
                "dirty": False,
                "source_hashes": {"hooks.py": "hash-1"},
            },
            "component_flags": {
                "fast_memory_retrieval": True,
                "memory_writer": True,
            },
            "component_status": {
                "task_compiler": {"source": "llm", "degraded": False},
                "memory_writer": {
                    "write_status": "promoted",
                    "candidate_count": 2,
                    "promoted_count": 1,
                    "fallback_written": False,
                    "llm_skill_candidate_count": 1,
                    "bootstrap_skill_eligible_stage_count": 1,
                    "bootstrap_skill_candidate_count": 2,
                    "bootstrap_skill_persisted_candidate_count": 2,
                    "bootstrap_skill_rejected_count": 0,
                    "skill_persisted_candidate_count": 1,
                    "skill_promoted_active_count": 0,
                    "skill_rejected_count": 1,
                    "skill_rejection_reasons": {"stage_gate_failed": 1},
                    "store_apply": {
                        "added": 1,
                        "merged": 0,
                        "skill_candidate_added": 1,
                        "skill_candidate_merged": 0,
                        "skill_promoted_active": 0,
                    },
                },
            },
            "stages": [
                {
                    "stage": "furniture",
                    "memory_pack": {
                        "success_case_ids": ["success-1"],
                        "failure_case_ids": [],
                        "skill_names": [],
                        "retrieved_source_task_ids": {
                            "success-1": [
                                "task_from_another_prompt",
                                "task_from_first_stage",
                            ]
                        },
                        "retrieved_source_run_ids": {
                            "success-1": ["run-cold", "run-earlier"]
                        },
                        "memory_bank_id": "bank-1",
                        "memory_bank_revision": 2,
                        "selection_decisions": [
                            {
                                "memory_id": "failure-unverified",
                                "memory_type": "failure",
                                "decision": "rejected",
                                "reasons": ["unverified_failure"],
                            }
                        ],
                    },
                    "planner_trace": {"status": "ok"},
                    "execution_evidence": {
                        "stage_policy": "auto",
                        "optional_assets_allowed": True,
                        "required_objects": ["bed"],
                        "required_first_instruction_applicable": True,
                        "required_first_instruction_delivered": True,
                        "optional_autonomy_preserved": True,
                        "required_satisfied_objects": ["bed"],
                        "required_missing_objects": [],
                        "required_coverage": 1.0,
                        "requirement_status": "satisfied",
                        "designer_prompt_contains_brief": True,
                        "injected_memory_hash": "abc",
                        "designer_prompt_contains_memory": True,
                    },
                    "verify_report": {
                        "pass_stage": True,
                        "issues": [],
                        "hard_check_report": {
                            "hard_valid": True,
                            "relation_satisfaction": 0.9,
                        },
                    },
                    "repair_actions": [
                        {
                            "repair_owner": "scene_expert_repair_controller",
                            "repair_type": "local_repair",
                            "failure_type": "missing_object",
                            "repair_action": "Add the missing object.",
                            "execution_status": "planned",
                        }
                    ],
                },
                {
                    "stage": "furniture",
                    "memory_pack": {
                        "success_case_ids": ["success-1"],
                        "failure_case_ids": [],
                        "skill_names": [],
                        "retrieved_source_task_ids": {
                            "success-1": ["task_from_another_prompt"]
                        },
                        "retrieved_source_run_ids": {"success-1": ["run-cold"]},
                        "memory_bank_id": "bank-1",
                        "memory_bank_revision": 2,
                    },
                    "planner_trace": {"status": "ok"},
                    "execution_evidence": {
                        "designer_prompt_contains_brief": True,
                        "injected_memory_hash": "abc",
                        "designer_prompt_contains_memory": True,
                    },
                    "repair_actions": [
                        {
                            "repair_owner": "scene_expert_repair_controller",
                            "repair_type": "local_repair",
                            "failure_type": "missing_object",
                            "repair_action": "Add the missing object.",
                            "execution_status": "executed",
                        }
                    ],
                },
            ],
            "final_report": {
                "overall_score": 0.8,
                "pass_scene": True,
                "generation_status": "complete",
                "requirement_status": "satisfied",
                "quality_status": "passed",
            },
        },
    )
    _write_json(
        scene0
        / "room_main"
        / "scenebenchmark_critic"
        / "final_scene"
        / "scenebenchmark_critic.json",
        {
            "summary": {
                "scene_summary": {
                    "effective_checks": 4,
                    "pass": 4,
                    "degraded": 0,
                    "fail": 0,
                    "unknown": 0,
                    "score": 1.0,
                }
            }
        },
    )

    scene1 = hydra / "scene_001"
    _write_json(
        scene1 / "scene_status.json",
        {
            "schema_version": "scenesmith.scene_status.v2",
            "status": "failed",
            "attempt": 1,
            "prompt": "living room",
            "error": "physics hard violation: collisions",
        },
    )
    _write_json(
        scene1 / "scene_expert" / "trace" / "trace_000001_partial.json",
        {
            "status": "failed",
            "error": "physics hard violation: collisions",
            "total_time_sec": 45.0,
            "component_flags": {"fast_memory_retrieval": True},
            "stages": [],
        },
    )
    _write_json(
        scene1 / "scene_expert" / "stages" / "000_furniture_pre.json",
        {
            "stage": "furniture",
            "memory_pack": {
                "success_case_ids": [],
                "failure_case_ids": [],
                "skill_names": [],
            },
            "execution_evidence": {},
        },
    )
    timing_dir = scene1 / "scene_expert" / "timing"
    timing_dir.mkdir(parents=True, exist_ok=True)
    (timing_dir / "memory_retrieval.jsonl").write_text(
        json.dumps(
            {
                "retriever_type": "hybrid",
                "stage": "furniture",
                "total_sec": 0.5,
                "zero_result_reason": "no_structurally_compatible_memory",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (timing_dir / "repair_events.jsonl").write_text(
        json.dumps(
            {
                "schema_version": "scenesmith.repair_event.v1",
                "repair_owner": "scenesmith_core",
                "strategy": "deterministic_hard_state",
                "status": "accepted",
                "affected_objects": [{"object_id": "chair_1"}],
                "detail": {"resolved": True, "elapsed_sec": 0.25},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (timing_dir / "stage_working_timing.jsonl").write_text(
        json.dumps(
            {
                "module": "deterministic_repair",
                "event": "critic_hard_state",
                "elapsed_sec": 0.3,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    metrics = collect_run_metrics(output_root, process_exit_code=1)

    assert metrics["quality_comparison_ready"] is False
    assert metrics["summary"]["expected_scenes"] == 2
    assert metrics["summary"]["completed_scenes"] == 1
    assert metrics["summary"]["failed_scenes"] == 1
    assert metrics["summary"]["recorded_failed_scenes"] == 0
    assert metrics["summary"]["failure_class_counts"] == {"unclassified_legacy": 1}
    assert metrics["summary"]["critic_mean_score"] == 1.0
    assert metrics["summary"]["hard_constraint_pass_rate"] == 1.0
    assert metrics["summary"]["mean_relation_satisfaction"] == 0.9
    assert metrics["summary"]["memory_injection_delivery_rate"] == 1.0
    assert metrics["summary"]["generation_complete_rate"] == 0.5
    assert metrics["summary"]["required_satisfaction_rate"] == 1.0
    assert metrics["summary"]["mean_required_coverage"] == 1.0
    assert metrics["summary"]["quality_pass_rate"] == 1.0
    assert metrics["summary"]["required_first_instruction_delivery_rate"] == 1.0
    assert metrics["summary"]["optional_autonomy_preservation_rate"] == 1.0
    assert metrics["summary"]["memory_cross_task_verified_scene_coverage"] == 0.5
    assert metrics["memory_closed_loop_observed"] is True
    assert metrics["summary"]["task_compiler_llm_scenes"] == 1
    assert metrics["summary"]["global_planner_llm_stage_count"] == 1
    assert metrics["summary"]["brief_injection_verified_stage_count"] == 1
    assert metrics["memory_identity"]["memory_bank_ids"] == ["bank-1"]
    assert metrics["code_provenance"]["git_revision"] == "abc123"
    assert metrics["code_provenance"]["source_bundle_hash"]
    assert metrics["experiment_identity"]["source_bundle_hashes"] == [
        metrics["code_provenance"]["source_bundle_hash"]
    ]
    assert metrics["scenes"][0]["generation_status"] == "complete"
    assert metrics["scenes"][0]["requirement_status"] == "satisfied"
    assert metrics["scenes"][0]["quality_status"] == "passed"
    assert metrics["scenes"][0]["required_objects_by_stage"] == {"furniture": ["bed"]}
    assert metrics["scenes"][0]["memory_retrieved_source_task_ids"]["success-1"] == [
        "task_from_another_prompt",
        "task_from_first_stage",
    ]
    assert metrics["scenes"][0]["memory_retrieved_source_run_ids"]["success-1"] == [
        "run-cold",
        "run-earlier",
    ]
    assert metrics["summary"]["memory_selection_rejection_count"] == 1
    assert metrics["summary"]["memory_selection_rejection_reasons"] == {
        "unverified_failure": 1
    }
    assert not any(
        "batch_001.log" in warning for warning in metrics["data_quality_warnings"]
    )
    assert metrics["summary"]["memory_writer_fallback_writes"] == 0
    assert metrics["summary"]["memory_writer_persisted_records"] == 1
    assert metrics["summary"]["llm_skill_candidate_count"] == 1
    assert metrics["summary"]["bootstrap_skill_eligible_stage_count"] == 1
    assert metrics["summary"]["bootstrap_skill_candidate_count"] == 2
    assert metrics["summary"]["bootstrap_skill_persisted_candidate_count"] == 2
    assert metrics["summary"]["bootstrap_skill_rejected_count"] == 0
    assert metrics["summary"]["skill_persisted_candidate_count"] == 1
    assert metrics["summary"]["skill_promoted_active_count"] == 0
    assert metrics["summary"]["skill_rejected_count"] == 1
    assert metrics["summary"]["skill_rejection_reasons"] == {"stage_gate_failed": 1}
    assert metrics["summary"]["skill_store_candidate_added"] == 1
    assert metrics["summary"]["scenesmith_repair_events"] == 1
    assert metrics["summary"]["scenesmith_repairs_accepted"] == 1
    assert metrics["summary"]["scenesmith_repairs_resolved"] == 1
    assert metrics["summary"]["scenesmith_repair_affected_objects"] == 1
    assert metrics["summary"]["scenesmith_repair_timing_events"] == 1
    assert metrics["summary"]["scenesmith_repair_overhead_sec"] == 0.3
    assert metrics["summary"]["sceneexpert_repair_plans"] == 1
    assert metrics["summary"]["sceneexpert_repairs_executed"] == 1
    assert metrics["summary"]["memory_zero_result_reasons"] == [
        "no_structurally_compatible_memory"
    ]
    assert (
        bedroom_task_id
        not in metrics["scenes"][0]["memory_retrieved_source_task_ids"]["success-1"]
    )

    paths = write_run_metrics(metrics)
    assert all(Path(path).is_file() for path in paths.values())
    rows = list(csv.DictReader(Path(paths["scene_csv"]).open("r", encoding="utf-8")))
    assert [row["status"] for row in rows] == ["completed", "failed"]
    assert rows[1]["failure_class"] == "unclassified_legacy"


def test_recorded_failure_stays_out_of_quality_denominators(tmp_path) -> None:
    output_root = tmp_path / "run_recorded"
    batch = output_root / "critic_on" / "batch_001"
    hydra = batch / "hydra"
    _write_manifest(batch / "batch_cases.csv")

    scene0 = hydra / "scene_000"
    _write_json(
        scene0 / "scene_status.json",
        {"status": "completed", "attempt": 1, "prompt": "bedroom"},
    )
    scene1 = hydra / "scene_001"
    _write_json(
        scene1 / "scene_status.json",
        {
            "schema_version": "scenesmith.scene_status.v3",
            "scene_id": 1,
            "status": "failed",
            "attempt": 1,
            "prompt": "living room",
            "error": "bounded ceiling workflow produced no mutation",
            "failure": {
                "failure_class": "stage_unavailable",
                "stage": "ceiling_mounted",
                "error_type": "PlannerWorkflowNoMutationError",
                "message": "bounded ceiling workflow produced no mutation",
                "reason": "no_mutation",
                "retryable": False,
                "root_error_type": "",
                "stage_execution_attempt": 1,
                "attempt": 1,
                "recordable": True,
                "provenance": {"workflow_calls": 2},
            },
        },
    )
    for scene_dir, score in ((scene0, 0.75), (scene1, 1.0)):
        _write_json(
            scene_dir
            / "room_main"
            / "scenebenchmark_critic"
            / "final_scene"
            / "scenebenchmark_critic.json",
            {
                "summary": {
                    "scene_summary": {
                        "effective_checks": 4,
                        "pass": 4,
                        "degraded": 0,
                        "fail": 0,
                        "unknown": 0,
                        "score": score,
                    }
                }
            },
        )

    metrics = collect_run_metrics(output_root, process_exit_code=0)

    assert metrics["schema_version"] == "sceneexpert.run_metrics.v5"
    assert metrics["quality_comparison_ready"] is False
    assert metrics["summary"]["completed_scenes"] == 1
    assert metrics["summary"]["failed_scenes"] == 1
    assert metrics["summary"]["recorded_failed_scenes"] == 1
    assert metrics["summary"]["failure_class_counts"] == {"stage_unavailable": 1}
    assert metrics["summary"]["critic_mean_score"] == 0.75
    failed = next(row for row in metrics["scenes"] if row["status"] == "failed")
    assert failed["failure_class"] == "stage_unavailable"
    assert failed["failure_stage"] == "ceiling_mounted"
    assert failed["failure_error_type"] == "PlannerWorkflowNoMutationError"
    assert failed["failure_recordable"] is True


def test_malformed_jsonl_is_a_warning_not_a_metrics_failure(tmp_path) -> None:
    output_root = tmp_path / "run_002"
    batch = output_root / "critic_on" / "batch_001"
    _write_manifest(batch / "batch_cases.csv")
    timing = (
        batch
        / "hydra"
        / "scene_000"
        / "scene_expert"
        / "timing"
        / "memory_retrieval.jsonl"
    )
    timing.parent.mkdir(parents=True, exist_ok=True)
    timing.write_text("{not-json}\n", encoding="utf-8")

    metrics = collect_run_metrics(output_root)

    assert any(
        warning.startswith("malformed_jsonl:")
        for warning in metrics["data_quality_warnings"]
    )


def test_frozen_evaluation_contract_is_materialized_in_run_metrics(tmp_path) -> None:
    output_root = tmp_path / "memory_on"
    batch = output_root / "critic_on" / "batch_001"
    hydra = batch / "hydra"
    batch.mkdir(parents=True)
    with (batch / "batch_cases.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("scene_index", "prompt", "case_id"))
        writer.writeheader()
        writer.writerow(
            {"scene_index": 0, "prompt": "classroom", "case_id": "classroom-a"}
        )
    _write_json(
        hydra / "scene_000" / "scene_status.json",
        {"status": "completed", "prompt": "classroom"},
    )
    _write_json(
        hydra / "traces" / "trace_000000.json",
        {
            "status": "completed",
            "experiment_name": "ablation_5_qwen3_full",
            "config_hash": "exact-memory-on",
            "experiment_signature": "semantic-memory-on",
            "control_signature": "same-non-treatment-semantics",
            "model": "qwen-test",
            "memory_identity": {
                "bank_id": "bank-1",
                "revision": 9,
                "content_fingerprint": "sceneexpert.memory_snapshot.v1:abc",
                "memory_dir": "/memory/frozen",
                "read_only": True,
            },
            "evaluation_contract": {
                "pair_id": "pair-1",
                "controlled_dimension": "fast_memory_retrieval",
                "arm": "memory_on",
                "require_frozen_memory": True,
                "shared_base_identity": {
                    "fingerprint": "sceneexpert.shared_base_snapshot.v1:def"
                },
                "compiled_inputs_identity": {
                    "compiled_input_fingerprint": "sceneexpert.compiled_input.v1:aaa",
                    "task_spec_fingerprint": "sceneexpert.task_spec.v1:bbb",
                    "intent_contract_fingerprint": "sceneexpert.intent_contract.v1:ccc",
                },
            },
            "component_flags": {
                "fast_memory_retrieval": True,
                "memory_writer": False,
                "slow_memory_capture": True,
            },
            "component_status": {
                "memory_snapshot": {"unchanged": True, "success": True}
            },
            "final_report": {
                "generation_status": "complete",
                "requirement_status": "satisfied",
                "quality_status": "passed",
            },
            "stages": [],
        },
    )

    metrics = collect_run_metrics(output_root)

    assert metrics["evaluation_contract"]["contract_ready"] is True
    assert metrics["evaluation_contract"]["arms"] == ["memory_on"]
    assert metrics["memory_identity"]["frozen_all_unchanged"] is True
    assert metrics["memory_identity"]["snapshot_identity_stable"] is True
    assert metrics["experiment_identity"]["control_signatures"] == [
        "same-non-treatment-semantics"
    ]
