"""Synthetic regressions for finite numeric evidence and unique paired cases."""

from __future__ import annotations

import copy
import json

from pathlib import Path

import pytest

from scenesmith.scene_expert.slow_memory.evaluation import (
    evaluate_scene_level_paths,
    evaluate_scene_level_promotion,
)


def _metrics(model: str, score: float) -> dict:
    return {
        "run_id": "synthetic-" + model,
        "quality_comparison_ready": True,
        "experiment_identity": {"models": ["synthetic-" + model]},
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


def _thresholds() -> dict:
    return {
        "minimum_paired_cases": 20,
        "minimum_mean_critic_score_delta": 0.01,
        "minimum_win_minus_loss_rate": 0.05,
    }


def _gate(before: dict, after: dict) -> dict:
    return evaluate_scene_level_promotion(before, after, thresholds=_thresholds())


def test_finite_unique_control_keeps_the_existing_relative_policy() -> None:
    result = _gate(_metrics("base", 0.7), _metrics("adapter", 0.72))
    assert result["promotable"] is True
    assert result["summary"]["candidate_scene_pass_rate"] == 0.0


@pytest.mark.parametrize("field", ["critic_score", "relation_satisfaction"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), True])
def test_invalid_numeric_evidence_is_rejected(field: str, value: object) -> None:
    before, after = _metrics("base", 0.7), _metrics("adapter", 0.72)
    after["scenes"][0][field] = value
    result = _gate(before, after)
    assert result["promotable"] is False
    assert any(field in item and "finite" in item for item in result["failures"])
    json.dumps(result, allow_nan=False)


def test_standard_json_number_overflow_is_rejected_at_the_file_entry(
    tmp_path: Path,
) -> None:
    before, after = _metrics("base", 0.7), _metrics("adapter", 0.72)
    after["scenes"][0]["critic_score"] = "__OVERFLOW_NUMBER__"
    left, right = tmp_path / "base.json", tmp_path / "adapter.json"
    left.write_text(json.dumps(before, allow_nan=False), encoding="utf-8")
    right.write_text(
        json.dumps(after, allow_nan=False).replace('"__OVERFLOW_NUMBER__"', "1e309"),
        encoding="utf-8",
    )
    result = evaluate_scene_level_paths(
        baseline=left, candidate=right, thresholds=_thresholds()
    )
    assert result["promotable"] is False
    assert any("critic_score" in item and "finite" in item for item in result["failures"])


@pytest.mark.parametrize("side", ["baseline", "candidate"])
@pytest.mark.parametrize("last", [True, False])
def test_duplicate_case_is_rejected_regardless_of_row_order(side: str, last: bool) -> None:
    before, after = _metrics("base", 0.7), _metrics("adapter", 0.72)
    target = before if side == "baseline" else after
    row = copy.deepcopy(target["scenes"][0])
    row["critic_score"] = 0.0
    target["scenes"].insert(len(target["scenes"]) if last else 0, row)
    result = _gate(before, after)
    assert result["promotable"] is False
    assert result["summary"]["paired_cases"] == 19
    assert any("duplicate case_id synthetic-0" in item for item in result["failures"])


def test_null_metric_retains_existing_missing_evidence_policy() -> None:
    before, after = _metrics("base", 0.7), _metrics("adapter", 0.72)
    after["scenes"][0]["critic_score"] = None
    assert _gate(before, after)["promotable"] is True


def test_finite_operands_with_overflowed_delta_cannot_pass() -> None:
    before, after = _metrics("base", 0.7), _metrics("adapter", 0.72)
    before["scenes"][0]["critic_score"] = -1e308
    after["scenes"][0]["critic_score"] = 1e308
    result = _gate(before, after)
    assert result["promotable"] is False
    assert any("non-finite or overflowed" in item for item in result["failures"])


@pytest.mark.parametrize("case_id", [None, True, 7, " "])
def test_case_identity_is_not_silently_coerced(case_id: object) -> None:
    before, after = _metrics("base", 0.7), _metrics("adapter", 0.72)
    after["scenes"][0]["case_id"] = case_id
    result = _gate(before, after)
    assert result["promotable"] is False
    assert any("nonempty string case_id" in item for item in result["failures"])


@pytest.mark.parametrize("rows", [False, {}, "malformed"])
def test_invalid_scene_collection_is_not_treated_as_missing(rows: object) -> None:
    before, after = _metrics("base", 0.7), _metrics("adapter", 0.72)
    after["scenes"] = rows
    result = _gate(before, after)
    assert result["promotable"] is False
    assert "adapter: scenes must be a list" in result["failures"]


def test_integer_overflow_is_rejected_without_raising() -> None:
    before, after = _metrics("base", 0.7), _metrics("adapter", 0.72)
    after["scenes"][0]["critic_score"] = 10 ** 400
    result = _gate(before, after)
    assert result["promotable"] is False
    assert any("critic_score" in item and "finite" in item for item in result["failures"])
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "key", ["minimum_mean_critic_score_delta", "minimum_win_minus_loss_rate"]
)
def test_nonfinite_threshold_input_cannot_bypass_a_gate(key: str) -> None:
    thresholds = _thresholds()
    thresholds[key] = float("nan")
    result = evaluate_scene_level_promotion(
        _metrics("base", 0.7), _metrics("adapter", 0.72), thresholds=thresholds
    )
    assert result["promotable"] is False
