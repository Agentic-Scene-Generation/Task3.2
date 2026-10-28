"""Synthetic JSON evidence must not gain passing flags through truthiness."""

from __future__ import annotations

import json

from pathlib import Path

import pytest

from scenesmith.scene_expert.run_metrics import collect_run_metrics
from scenesmith.scene_expert.slow_memory.evaluation import evaluate_scene_level_promotion


def _metrics(model: str, score: float) -> dict:
    return {
        "run_id": "synthetic-" + model,
        "quality_comparison_ready": True,
        "experiment_identity": {"models": [model]},
        "code_provenance": {"source_bundle_hash": "synthetic-identical-source"},
        "memory_identity": {"memory_bank_ids": ["synthetic-frozen-bank"]},
        "scenes": [
            {
                "case_id": f"synthetic-{index}",
                "prompt": f"synthetic prompt {index}",
                "status": "completed",
                "hard_constraint_pass": False,
                "sceneexpert_pass_scene": False,
                "critic_score": score,
                "relation_satisfaction": score,
            }
            for index in range(20)
        ],
    }


def _gate(before: dict, after: dict) -> dict:
    return evaluate_scene_level_promotion(
        before,
        after,
        thresholds={
            "minimum_paired_cases": 20,
            "minimum_mean_critic_score_delta": 0.01,
            "minimum_win_minus_loss_rate": 0.05,
        },
    )


def test_valid_boolean_evidence_keeps_existing_relative_gate_policy() -> None:
    """Type validation adds no new absolute pass-rate requirement."""
    result = _gate(_metrics("base", 0.7), _metrics("adapter", 0.8))
    assert result["promotable"] is True
    assert result["summary"]["candidate_scene_pass_rate"] == 0.0
    assert result["summary"]["candidate_hard_pass_rate"] == 0.0


@pytest.mark.parametrize("side", ["baseline", "candidate"])
@pytest.mark.parametrize("value", ["false", 1])
def test_comparison_readiness_requires_a_json_boolean(side: str, value: object) -> None:
    before, after = _metrics("base", 0.7), _metrics("adapter", 0.8)
    target = before if side == "baseline" else after
    target["quality_comparison_ready"] = value
    result = _gate(before, after)
    assert result["promotable"] is False
    assert any("not quality-comparison ready" in item for item in result["failures"])


@pytest.mark.parametrize(
    "field", ["sceneexpert_pass_scene", "hard_constraint_pass", "critic_zero_fail"]
)
@pytest.mark.parametrize("value", ["false", 1])
def test_malformed_pass_flags_cannot_create_promotion_evidence(
    field: str, value: object
) -> None:
    before, after = _metrics("base", 0.7), _metrics("adapter", 0.8)
    if field == "critic_zero_fail":
        for metrics in (before, after):
            for row in metrics["scenes"]:
                del row["hard_constraint_pass"]
                row[field] = False
    after["scenes"][0][field] = value
    result = _gate(before, after)
    assert result["promotable"] is False
    assert any(field in item and "boolean" in item for item in result["failures"])
    assert result["pairs"][0]["candidate_scene_pass"] is False
    assert result["pairs"][0]["candidate_hard_pass"] is False


def test_null_and_missing_pass_flags_keep_legacy_unknown_behavior() -> None:
    before, after = _metrics("base", 0.7), _metrics("adapter", 0.8)
    for metrics in (before, after):
        metrics["scenes"][0]["sceneexpert_pass_scene"] = None
        del metrics["scenes"][0]["hard_constraint_pass"]
    assert _gate(before, after)["promotable"] is True


def _write_run(root: Path, final_flag: object, stage_flag: object) -> None:
    batch = root / "critic_on" / "batch_synthetic"
    hydra = batch / "hydra"
    scene = hydra / "scene_000"
    traces = hydra / "traces"
    scene.mkdir(parents=True)
    traces.mkdir()
    (batch / "batch_cases.csv").write_text(
        "scene_index,prompt,case_id,critic_goal\n0,synthetic,synthetic,quality\n",
        encoding="utf-8",
    )
    (scene / "scene_status.json").write_text(
        json.dumps({"status": "completed", "prompt": "synthetic"}), encoding="utf-8"
    )
    trace = {
        "status": "completed",
        "final_report": {"pass_scene": final_flag},
        "stages": [
            {
                "stage": "furniture",
                "execution_evidence": {"stage_agent_invoked": True},
                "verify_report": {"pass_stage": stage_flag, "issues": []},
            }
        ],
    }
    (traces / "trace_000000.json").write_text(json.dumps(trace), encoding="utf-8")
    critic = scene / "room_synthetic/scenebenchmark_critic/final_scene"
    critic.mkdir(parents=True)
    (critic / "scenebenchmark_critic.json").write_text(
        json.dumps({"summary": {"scene_summary": {"effective_checks": 1, "fail": 0}}}),
        encoding="utf-8",
    )


@pytest.mark.parametrize("value", ["false", 1])
def test_raw_verifier_strings_and_numbers_stay_unknown_and_not_ready(
    tmp_path: Path, value: object
) -> None:
    root = tmp_path / "synthetic-run"
    _write_run(root, value, value)
    metrics = collect_run_metrics(root)
    row = metrics["scenes"][0]
    assert row["sceneexpert_pass_scene"] is None
    assert row["hard_constraint_pass"] is False
    assert metrics["quality_comparison_ready"] is False
    assert any(
        item.startswith("invalid_verifier_boolean:")
        for item in metrics["data_quality_warnings"]
    )


@pytest.mark.parametrize("value", [True, False])
def test_raw_booleans_remain_unchanged_without_a_type_warning(
    tmp_path: Path, value: bool
) -> None:
    root = tmp_path / "synthetic-run"
    _write_run(root, value, value)
    metrics = collect_run_metrics(root)
    assert metrics["scenes"][0]["sceneexpert_pass_scene"] is value
    assert metrics["scenes"][0]["hard_constraint_pass"] is value
    assert metrics["quality_comparison_ready"] is True
    assert not any(
        item.startswith("invalid_verifier_boolean:")
        for item in metrics["data_quality_warnings"]
    )


@pytest.mark.parametrize("value", ["false", 1])
def test_invalid_explicit_hard_flag_cannot_fall_back_to_stage_pass(
    tmp_path: Path, value: object
) -> None:
    root = tmp_path / "synthetic-run"
    _write_run(root, False, True)
    path = root / "critic_on/batch_synthetic/hydra/traces/trace_000000.json"
    trace = json.loads(path.read_text(encoding="utf-8"))
    trace["stages"][0]["verify_report"]["hard_check_report"] = {"hard_valid": value}
    path.write_text(json.dumps(trace), encoding="utf-8")
    metrics = collect_run_metrics(root)
    assert metrics["scenes"][0]["hard_constraint_pass"] is False
    assert metrics["quality_comparison_ready"] is False
    assert any("hard_valid" in item for item in metrics["data_quality_warnings"])
