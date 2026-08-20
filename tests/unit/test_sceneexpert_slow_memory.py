"""Regression coverage for evidence-safe slow memory and DPO curation."""

from __future__ import annotations

import json

from pathlib import Path

from omegaconf import OmegaConf

from scenesmith.scene_expert.schemas import SceneTaskSpec, StageVerifyReport
from scenesmith.scene_expert.slow_memory.dpo import (
    build_preference_pairs,
    export_dpo_dataset,
    validate_dataset_dir,
)
from scenesmith.scene_expert.slow_memory.schemas import (
    PreferenceEvidence,
    TrajectoryRecord,
)
from scenesmith.scene_expert.slow_memory.training import validate_training_request
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
        context_hash=context_hash or f"context_{task_id}",
        prompt=prompt,
        response=response,
        response_hash=f"response_{suffix}",
        evidence=_evidence(verdict, score),
        source_refs=["audit/payload.json", "evidence/report.json"],
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
    assert summary == {"designer": 2, "repair": 0, "unlabeled": 1}
    assert rows[0].evidence.verdict == "unlabeled"
    assert rows[1].evidence.verdict == "accepted"
    assert "sk-secret123456" not in rows[0].prompt
    assert "[REDACTED]" in rows[0].prompt
    assert Path(scene_debug / rows[1].evidence.report_ref).exists()


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
        },
        "lora": {"target_modules": ["q_proj", "v_proj"]},
        "data": {"dataset_dir": str(dataset)},
        "training": {"output_dir": str(tmp_path / "training")},
    }

    invalid = validate_training_request(
        config, dataset_dir=dataset, model_name_or_path="model.gguf"
    )
    valid = validate_training_request(config, dataset_dir=dataset)

    assert not invalid["valid"]
    assert any("GGUF" in error for error in invalid["errors"])
    assert valid["valid"]


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
