"""Regression coverage for evidence-safe slow memory and DPO curation."""

from __future__ import annotations

import json
import hashlib

from pathlib import Path

from omegaconf import OmegaConf

from scenesmith.agent_utils.stage_working_memory import _extract_agent_result_trace
from scenesmith.scene_expert.schemas import SceneTaskSpec, StageVerifyReport
from scenesmith.scene_expert.slow_memory.dpo import (
    build_preference_pairs,
    export_dpo_dataset,
    validate_dataset_dir,
)
from scenesmith.scene_expert.slow_memory.evaluation import (
    evaluate_scene_level_promotion,
)
from scenesmith.scene_expert.slow_memory.importer import import_teacher_trajectories
from scenesmith.scene_expert.slow_memory.schemas import (
    PreferenceEvidence,
    TrajectoryOutcome,
    TrajectoryRecord,
)
from scenesmith.scene_expert.slow_memory.training import (
    evaluate_training_promotion,
    validate_training_request,
)
from scenesmith.scene_expert.slow_memory.trajectory import TrajectoryCollector


def _evidence(verdict: str, score: float) -> PreferenceEvidence:
    return PreferenceEvidence(
        evidence_id=f"evidence_{verdict}_{score}",
        kind="critic_and_deterministic",
        verdict=verdict,
        source="main_scenebenchmark_critic_plus_stage_rules",
        authoritative=True,
        quality_score=score,
        report_ref="evidence/report.json",
    )


def _trajectory(
    *,
    suffix: str,
    task_id: str,
    prompt: str,
    response: str,
    verdict: str,
    score: float,
    context_hash: str | None = None,
) -> TrajectoryRecord:
    return TrajectoryRecord(
        trajectory_id=f"trajectory_{suffix}",
        created_at="2026-08-21T00:00:00Z",
        run_id=f"run_{suffix}",
        scene_id=f"scene_{suffix}",
        task_id=task_id,
        stage="furniture",
        agent_role="designer",
        event="request_initial_design",
        task_type="designer_initial",
        context_hash=context_hash or f"context_{task_id}",
        prompt=prompt,
        response=response,
        response_hash=f"response_{suffix}",
        evidence=_evidence(verdict, score),
        source_refs=["audit/payload.json", "evidence/report.json"],
        messages=[{"role": "user", "content": prompt}],
        completion_messages=[{"role": "assistant", "content": response}],
        outcome=TrajectoryOutcome(
            execution_complete=True,
            tool_call_valid=True,
            hard_passed=verdict == "accepted",
            hard_violation_count=0 if verdict == "accepted" else 1,
            causal_link_verified=True,
        ),
    )


def test_collector_labels_only_final_designer_call_and_redacts_secrets(
    tmp_path: Path,
) -> None:
    scene_debug = tmp_path / "scene_000" / "scene_expert"
    payload_dir = scene_debug / "audit" / "llm_payloads"
    payload_dir.mkdir(parents=True)
    common = {
        "schema_version": "1.0",
        "stage": "furniture",
        "agent_role": "designer",
        "created_at": "2026-08-21T00:00:00Z",
        "error": "",
    }
    (payload_dir / "1000_designer_initial.json").write_text(
        json.dumps(
            {
                **common,
                "event": "request_initial_design",
                "prompt": "Place desks; api_key=sk-secret123456",
                "output": "Initial attempt",
            }
        ),
        encoding="utf-8",
    )
    (payload_dir / "2000_designer_change.json").write_text(
        json.dumps(
            {
                **common,
                "event": "request_design_change",
                "prompt": "Correct every chair orientation",
                "output": "Corrected attempt",
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "rotate_object",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
                "context_snapshot": {
                    "relations": [{"subject": "chair", "target": "desk"}]
                },
                "agent_trace": {
                    "assistant_messages": [],
                    "tool_calls": [
                        {
                            "type": "function",
                            "id": "call_rotate",
                            "function": {
                                "name": "rotate_object",
                                "arguments": {"yaw": 3.14},
                            },
                        }
                    ],
                    "tool_results": [
                        {"tool_call_id": "call_rotate", "output": {"ok": True}}
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    collector = TrajectoryCollector(
        scene_debug_dir=scene_debug,
        prompt="A classroom with desks and chairs",
        scene_id="scene_000",
        run_id="run_001",
        task_spec=SceneTaskSpec(room_type="classroom", style="modern"),
    )

    summary = collector.capture_stage(
        stage="furniture",
        verify_report=StageVerifyReport(
            stage="furniture",
            pass_stage=True,
            visual_scores={"semantic": 0.9},
            rule_scores={"deterministic_issue_free": 1.0},
            score_source="scenebenchmark_critic",
            vlm_scoring_performed=True,
        ),
    )

    rows = [
        TrajectoryRecord.model_validate_json(line)
        for line in collector.trajectory_path.read_text(encoding="utf-8").splitlines()
    ]
    assert summary == {"designer": 2, "critic": 0, "repair": 0, "unlabeled": 1}
    assert rows[0].evidence.verdict == "unlabeled"
    assert rows[1].evidence.verdict == "accepted"
    assert "sk-secret123456" not in rows[0].prompt
    assert "[REDACTED]" in rows[0].prompt
    assert Path(scene_debug / rows[1].evidence.report_ref).exists()
    assert rows[1].schema_version == "sceneexpert.trajectory.v2"
    assert rows[1].action_trace[0]["tool_call"]["function"]["name"] == "rotate_object"
    assert rows[1].completion_messages[1]["role"] == "tool"
    assert rows[1].spatial_context["relations"][0]["subject"] == "chair"


def test_agent_result_trace_preserves_replayable_tool_events() -> None:
    class FakeResult:
        input = [{"role": "user", "content": "Move the chair"}]
        new_items = [
            {
                "type": "tool_call_item",
                "raw_item": {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "move_object",
                    "arguments": '{"x": 1.25}',
                },
            },
            {
                "type": "tool_call_output_item",
                "raw_item": {"call_id": "call_1"},
                "output": {"ok": True},
            },
        ]
        raw_responses: list[dict] = []

        def to_input_list(self) -> list[dict]:
            return [*self.input, *self.new_items]

    trace = _extract_agent_result_trace(FakeResult(), "Chair moved")

    assert trace["run_input"] == FakeResult.input
    assert trace["tool_calls"][0]["function"] == {
        "name": "move_object",
        "arguments": {"x": 1.25},
    }
    assert trace["tool_results"][0]["output"] == {"ok": True}
    assert len(trace["replay_items"]) == 3


def test_pair_builder_requires_exact_context_and_authoritative_evidence() -> None:
    accepted = _trajectory(
        suffix="accepted",
        task_id="task_a",
        prompt="same prompt",
        response="good response",
        verdict="accepted",
        score=1.8,
    )
    rejected = _trajectory(
        suffix="rejected",
        task_id="task_a",
        prompt="same prompt",
        response="bad response",
        verdict="rejected",
        score=0.2,
    )
    accepted = accepted.model_copy(
        update={"spatial_context": {"run_id": "run_a", "objects": ["desk"]}},
        deep=True,
    )
    rejected = rejected.model_copy(
        update={"spatial_context": {"run_id": "run_b", "objects": ["desk"]}},
        deep=True,
    )
    near_match = _trajectory(
        suffix="near",
        task_id="task_a",
        prompt="different prompt",
        response="other response",
        verdict="rejected",
        score=0.1,
        context_hash="different_context",
    )

    pairs, diagnostics = build_preference_pairs([accepted, rejected, near_match])

    assert len(pairs) == 1
    assert pairs[0].provenance["exact_prompt_match"] is True
    assert pairs[0].chosen[0]["content"] == "good response"
    assert any(
        item["reason"] == "missing_exact_context_counterpart"
        and "trajectory_near" in item["trajectory_ids"]
        for item in diagnostics
    )

    corrupt_chosen = _trajectory(
        suffix="corrupt_chosen",
        task_id="task_b",
        prompt="prompt one",
        response="good",
        verdict="accepted",
        score=1.8,
        context_hash="forged_context",
    )
    corrupt_rejected = _trajectory(
        suffix="corrupt_rejected",
        task_id="task_b",
        prompt="prompt two",
        response="bad",
        verdict="rejected",
        score=0.1,
        context_hash="forged_context",
    )
    corrupt_pairs, corrupt_diagnostics = build_preference_pairs(
        [corrupt_chosen, corrupt_rejected]
    )
    assert corrupt_pairs == []
    assert corrupt_diagnostics[0]["reason"] == "context_hash_prompt_mismatch"


def test_pair_builder_can_rank_two_real_accepted_outputs_with_main_critic() -> None:
    high = _trajectory(
        suffix="high",
        task_id="task_ranked",
        prompt="same prompt",
        response="higher quality response",
        verdict="accepted",
        score=1.9,
    )
    low = _trajectory(
        suffix="low",
        task_id="task_ranked",
        prompt="same prompt",
        response="lower quality response",
        verdict="accepted",
        score=1.2,
    )

    pairs, diagnostics = build_preference_pairs([high, low])

    assert diagnostics == []
    assert len(pairs) == 1
    assert pairs[0].chosen_trajectory_id == high.trajectory_id
    assert pairs[0].rejected_trajectory_id == low.trajectory_id
    assert pairs[0].rejected_evidence.verdict == "rejected"
    assert pairs[0].rejected_evidence.details["observed_runtime_verdict"] == "accepted"
    assert pairs[0].provenance["preference_basis"] == "relative_main_critic_ranking"


def test_tool_pair_preserves_observed_calls_and_spatial_context() -> None:
    tool = {
        "type": "function",
        "function": {
            "name": "move_object",
            "description": "Move an object in world coordinates.",
            "parameters": {
                "type": "object",
                "properties": {"x": {"type": "number"}},
            },
        },
    }
    accepted = _trajectory(
        suffix="tool_accepted",
        task_id="tool_task",
        prompt="Repair the desk-chair relation",
        response="Moved the chair to the desk front.",
        verdict="accepted",
        score=1.8,
    ).model_copy(
        update={
            "task_type": "designer_repair",
            "event": "request_design_change",
            "tools": [tool],
            "spatial_context": {"relations": [{"subject": "chair", "target": "desk"}]},
            "completion_messages": [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "id": "call_1",
                            "function": {
                                "name": "move_object",
                                "arguments": {"x": 1.25},
                            },
                        }
                    ],
                },
                {"role": "assistant", "content": "Moved the chair."},
            ],
        },
        deep=True,
    )
    rejected = _trajectory(
        suffix="tool_rejected",
        task_id="tool_task",
        prompt="Repair the desk-chair relation",
        response="Moved the chair away from the desk.",
        verdict="rejected",
        score=0.2,
    ).model_copy(
        update={
            "task_type": "designer_repair",
            "event": "request_design_change",
            "tools": [tool],
            "spatial_context": accepted.spatial_context,
            "completion_messages": [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "id": "call_2",
                            "function": {
                                "name": "move_object",
                                "arguments": {"x": 4.5},
                            },
                        }
                    ],
                },
                {"role": "assistant", "content": "Moved the chair."},
            ],
        },
        deep=True,
    )

    pairs, diagnostics = build_preference_pairs([accepted, rejected])

    assert diagnostics == []
    assert len(pairs) == 1
    assert pairs[0].tools == [tool]
    assert pairs[0].task_type == "designer_repair"
    assert pairs[0].chosen[0]["tool_calls"][0]["function"]["arguments"]["x"] == 1.25
    assert pairs[0].spatial_context == accepted.spatial_context


def test_critic_pair_requires_independent_causal_evidence() -> None:
    accepted = _trajectory(
        suffix="critic_accepted",
        task_id="critic_task",
        prompt="Critique this exact candidate",
        response="Rotate the chair toward the desk.",
        verdict="accepted",
        score=1.8,
    ).model_copy(
        update={
            "agent_role": "critic",
            "event": "score_scene",
            "task_type": "critic_advice",
            "outcome": TrajectoryOutcome(causal_link_verified=False),
        },
        deep=True,
    )
    rejected = _trajectory(
        suffix="critic_rejected",
        task_id="critic_task",
        prompt="Critique this exact candidate",
        response="No change is needed.",
        verdict="rejected",
        score=0.2,
    ).model_copy(
        update={
            "agent_role": "critic",
            "event": "score_scene",
            "task_type": "critic_advice",
            "outcome": TrajectoryOutcome(causal_link_verified=False),
        },
        deep=True,
    )

    blocked, diagnostics = build_preference_pairs([accepted, rejected])
    verified, verified_diagnostics = build_preference_pairs(
        [
            accepted.model_copy(
                update={
                    "outcome": accepted.outcome.model_copy(
                        update={"causal_link_verified": True, "hard_passed": True}
                    )
                },
                deep=True,
            ),
            rejected.model_copy(
                update={
                    "outcome": rejected.outcome.model_copy(
                        update={"causal_link_verified": True, "hard_passed": False}
                    )
                },
                deep=True,
            ),
        ]
    )

    assert blocked == []
    assert all(
        item["reason"] == "critic_causal_link_unverified" for item in diagnostics
    )
    assert verified_diagnostics == []
    assert len(verified) == 1


def test_exporter_materializes_images_and_vlm_message_placeholders(
    tmp_path: Path,
) -> None:
    image = tmp_path / "render.png"
    image.write_bytes(b"not-a-real-png-but-content-addressed")
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    rows = [
        _trajectory(
            suffix="vlm_accepted",
            task_id="vlm_task",
            prompt="Inspect the rendered classroom",
            response="Valid spatial arrangement",
            verdict="accepted",
            score=1.8,
        ),
        _trajectory(
            suffix="vlm_rejected",
            task_id="vlm_task",
            prompt="Inspect the rendered classroom",
            response="Invalid spatial arrangement",
            verdict="rejected",
            score=0.2,
        ),
    ]
    rows = [
        row.model_copy(
            update={
                "image_refs": [
                    {
                        "path": str(image),
                        "sha256": digest,
                        "size_bytes": image.stat().st_size,
                    }
                ]
            },
            deep=True,
        )
        for row in rows
    ]
    source = tmp_path / "trajectories.jsonl"
    source.write_text(
        "\n".join(row.model_dump_json() for row in rows) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "dataset"

    manifest = export_dpo_dataset(
        trajectory_sources=[source],
        output_dir=output,
        validation_ratio=0,
        test_ratio=0,
    )
    pair = json.loads((output / "train.jsonl").read_text(encoding="utf-8"))

    assert manifest["validation"]["valid"]
    assert pair["images"] == [f"images/{digest}.png"]
    assert (output / pair["images"][0]).is_file()
    assert pair["prompt"][0]["content"][0] == {"type": "image"}
    assert pair["chosen"][0]["content"][0]["type"] == "text"


def test_external_teacher_import_is_lossless_and_does_not_infer_labels(
    tmp_path: Path,
) -> None:
    source = tmp_path / "teacher.jsonl"
    common = {
        "task_id": "teacher_task",
        "scenario_family_id": "classroom_family",
        "stage": "furniture",
        "agent_role": "designer",
        "event": "request_initial_design",
        "task_type": "designer_initial",
        "messages": [
            {"role": "user", "content": "Place desks; api_key=sk-secret123456"}
        ],
        "spatial_context": {"required_relation": "chair faces desk"},
    }
    source.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {
                    **common,
                    "sample_id": "chosen",
                    "completion_messages": [
                        {"role": "assistant", "content": "Correct placement"}
                    ],
                    "outcome": {
                        "execution_complete": True,
                        "hard_passed": True,
                        "hard_violation_count": 0,
                        "causal_link_verified": True,
                    },
                    "evidence": {
                        "kind": "critic_and_deterministic",
                        "verdict": "accepted",
                        "authoritative": True,
                        "quality_score": 1.8,
                    },
                },
                {
                    **common,
                    "sample_id": "rejected",
                    "completion_messages": [
                        {"role": "assistant", "content": "Incorrect placement"}
                    ],
                    "outcome": {
                        "execution_complete": True,
                        "hard_passed": False,
                        "hard_violation_count": 1,
                        "causal_link_verified": True,
                    },
                    "evidence": {
                        "kind": "critic_and_deterministic",
                        "verdict": "rejected",
                        "authoritative": True,
                        "quality_score": 0.2,
                    },
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )
    imported = tmp_path / "trajectories.jsonl"

    summary = import_teacher_trajectories(
        source_paths=[source],
        output_path=imported,
    )
    rows = [
        TrajectoryRecord.model_validate_json(line)
        for line in imported.read_text(encoding="utf-8").splitlines()
    ]
    pairs, diagnostics = build_preference_pairs(rows)

    assert summary["record_count"] == 2
    assert summary["rejected_count"] == 0
    assert all("sk-secret123456" not in row.prompt for row in rows)
    assert diagnostics == []
    assert len(pairs) == 1


def test_exporter_keeps_task_groups_out_of_other_splits(tmp_path: Path) -> None:
    source = tmp_path / "trajectories.jsonl"
    rows: list[TrajectoryRecord] = []
    for index in range(6):
        task_id = f"task_{index}"
        prompt = f"prompt {index}"
        rows.extend(
            [
                _trajectory(
                    suffix=f"{index}_accepted",
                    task_id=task_id,
                    prompt=prompt,
                    response=f"good {index}",
                    verdict="accepted",
                    score=1.9,
                ),
                _trajectory(
                    suffix=f"{index}_rejected",
                    task_id=task_id,
                    prompt=prompt,
                    response=f"bad {index}",
                    verdict="rejected",
                    score=0.1,
                ),
            ]
        )
    source.write_text(
        "\n".join(row.model_dump_json() for row in rows) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "dataset"

    manifest = export_dpo_dataset(
        trajectory_sources=[source],
        output_dir=output,
        validation_ratio=0.2,
        test_ratio=0.2,
    )

    assert manifest["validation"]["valid"]
    assert manifest["stats"]["eligible_pair_count"] == 6
    validation = validate_dataset_dir(output)
    split_groups = {
        split: set(groups)
        for split, groups in validation["split_leakage_groups"].items()
    }
    assert not split_groups["train"] & split_groups["validation"]
    assert not split_groups["train"] & split_groups["test"]
    assert not split_groups["validation"] & split_groups["test"]


def test_training_preflight_rejects_gguf_and_accepts_transformers_checkpoint(
    tmp_path: Path,
) -> None:
    source = tmp_path / "trajectories.jsonl"
    rows = [
        _trajectory(
            suffix="accepted",
            task_id="task_a",
            prompt="same prompt",
            response="good",
            verdict="accepted",
            score=1.8,
        ),
        _trajectory(
            suffix="rejected",
            task_id="task_a",
            prompt="same prompt",
            response="bad",
            verdict="rejected",
            score=0.1,
        ),
    ]
    source.write_text(
        "\n".join(row.model_dump_json() for row in rows) + "\n",
        encoding="utf-8",
    )
    dataset = tmp_path / "dataset"
    export_dpo_dataset(trajectory_sources=[source], output_dir=dataset)
    model_dir = tmp_path / "qwen_transformers"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    config = {
        "model": {
            "name_or_path": str(model_dir),
            "backend": "unsloth",
            "tuning_mode": "qlora",
            "max_length": 4096,
            "multimodal": False,
            "tool_calling": True,
        },
        "lora": {"target_modules": ["q_proj", "v_proj"]},
        "data": {
            "dataset_dir": str(dataset),
            "allow_unsafe_small_dataset": True,
            "minimum_train_pairs": 1,
            "minimum_unique_train_groups": 1,
        },
        "training": {"output_dir": str(tmp_path / "training")},
        "quality_gate": {"require_validation": False, "require_test": False},
    }

    invalid = validate_training_request(
        config, dataset_dir=dataset, model_name_or_path="model.gguf"
    )
    valid = validate_training_request(config, dataset_dir=dataset)

    assert not invalid["valid"]
    assert any("GGUF" in error for error in invalid["errors"])
    assert valid["valid"]


def test_training_promotion_rejects_weak_offline_preference_accuracy() -> None:
    config = {
        "quality_gate": {
            "require_validation": True,
            "minimum_preference_accuracy": 0.55,
        }
    }

    rejected = evaluate_training_promotion(
        config,
        evaluation_metrics={"eval_rewards/accuracies": 0.51, "eval_loss": 0.8},
    )
    accepted = evaluate_training_promotion(
        config,
        evaluation_metrics={"eval_rewards/accuracies": 0.68, "eval_loss": 0.6},
    )

    assert not rejected["promotable"]
    assert accepted["promotable"]
    assert accepted["scene_level_gate_required"] is True


def test_scene_level_promotion_requires_controlled_paired_gain() -> None:
    def metrics(model: str, score: float) -> dict:
        return {
            "run_id": model,
            "quality_comparison_ready": True,
            "experiment_identity": {
                "models": [model],
                "code_revisions": ["revision"],
            },
            "code_provenance": {
                "git_status_hash": "clean",
                "source_hashes": {"slow_memory": "same"},
            },
            "memory_identity": {"memory_bank_ids": ["frozen_bank"]},
            "scenes": [
                {
                    "case_id": f"case_{index}",
                    "prompt": f"prompt {index}",
                    "status": "completed",
                    "critic_zero_fail": True,
                    "sceneexpert_pass_scene": True,
                    "critic_score": score,
                    "relation_satisfaction": score,
                }
                for index in range(2)
            ],
        }

    thresholds = {
        "minimum_paired_cases": 2,
        "minimum_mean_critic_score_delta": 0.01,
        "minimum_win_minus_loss_rate": 0.05,
        "require_memory_snapshot": True,
    }
    passed = evaluate_scene_level_promotion(
        metrics("base", 0.7),
        metrics("adapter", 0.8),
        thresholds=thresholds,
    )
    failed = evaluate_scene_level_promotion(
        metrics("base", 0.7),
        metrics("adapter", 0.6),
        thresholds=thresholds,
    )

    assert passed["promotable"]
    assert passed["summary"]["mean_critic_score_delta"] > 0
    assert not failed["promotable"]


def test_full_ablation_selects_only_the_explicit_slow_model(monkeypatch) -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configurations"
    monkeypatch.setenv("SCENEEXPERT_MODEL_ID", "base-model")
    monkeypatch.setenv("SCENEEXPERT_FULL_MODEL_ID", "sceneexpert-dpo")
    full = OmegaConf.load(config_dir / "config.yaml")
    full.experiment = OmegaConf.load(
        config_dir / "experiment" / "ablation_5_qwen3_full.yaml"
    )
    full.furniture_agent = OmegaConf.load(
        config_dir / "furniture_agent" / "base_furniture_agent.yaml"
    )
    full.floor_plan_agent = OmegaConf.load(
        config_dir / "floor_plan_agent" / "base_floor_plan_agent.yaml"
    )
    memory = OmegaConf.load(config_dir / "config.yaml")
    memory.experiment = OmegaConf.load(
        config_dir / "experiment" / "ablation_4_qwen3_harness_memory.yaml"
    )
    memory.furniture_agent = OmegaConf.load(
        config_dir / "furniture_agent" / "base_furniture_agent.yaml"
    )

    assert full.llm.model_id == "sceneexpert-dpo"
    assert full.furniture_agent.openai.model == "sceneexpert-dpo"
    assert full.floor_plan_agent.openai.model == "sceneexpert-dpo"
    assert memory.llm.model_id == "base-model"
    assert memory.furniture_agent.openai.model == "base-model"
