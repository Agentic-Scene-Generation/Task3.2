"""DPO pilot admission counts independent validation scenarios, not only rows."""

from pathlib import Path

import pytest

from scenesmith.scene_expert.slow_memory.dpo import export_dpo_dataset
from scenesmith.scene_expert.slow_memory.schemas import (
    PreferenceEvidence,
    TrajectoryOutcome,
    TrajectoryRecord,
)
from scenesmith.scene_expert.slow_memory.training import (
    apply_training_profile,
    validate_training_request,
)


def _validation_package(tmp_path: Path, validation_groups: int) -> tuple[Path, dict]:
    """Export distinct observed-pair fixtures sharing scenario families."""
    rows = []
    assignments = {}
    for split, count in [("train", 8), ("validation", 4)]:
        for index in range(count):
            task_id = f"task_{split}_{index}"
            family = (
                task_id
                if split == "train"
                else f"validation_family_{index % validation_groups}"
            )
            assignments[family] = split
            prompt = f"Synthetic scene {task_id}"
            for verdict, score in [("accepted", 1.8), ("rejected", 0.1)]:
                suffix = f"{split}_{index}_{verdict}"
                rows.append(
                    TrajectoryRecord(
                        trajectory_id=f"trajectory_{suffix}",
                        created_at="2026-10-17T00:00:00Z",
                        run_id=f"synthetic_{suffix}",
                        scene_id=f"scene_{suffix}",
                        task_id=task_id,
                        scenario_family_id=family,
                        stage="furniture",
                        agent_role="designer",
                        event="request_initial_design",
                        task_type="designer_initial",
                        context_hash=f"context_{task_id}",
                        prompt=prompt,
                        response=verdict,
                        response_hash=suffix,
                        evidence=PreferenceEvidence(
                            evidence_id=f"evidence_{suffix}",
                            kind="critic_and_deterministic",
                            verdict=verdict,
                            source="synthetic_test_fixture",
                            authoritative=True,
                            quality_score=score,
                            report_ref="synthetic/report.json",
                        ),
                        source_refs=["synthetic/payload.json", "synthetic/report.json"],
                        messages=[{"role": "user", "content": prompt}],
                        completion_messages=[{"role": "assistant", "content": verdict}],
                        outcome=TrajectoryOutcome(
                            execution_complete=True,
                            tool_call_valid=True,
                            hard_passed=verdict == "accepted",
                            hard_violation_count=0 if verdict == "accepted" else 1,
                            causal_link_verified=True,
                        ),
                    )
                )
    source = tmp_path / "synthetic_trajectories.jsonl"
    source.write_text(
        "\n".join(row.model_dump_json() for row in rows) + "\n", encoding="utf-8"
    )
    dataset = tmp_path / "dataset"
    export_dpo_dataset(
        trajectory_sources=[source],
        output_dir=dataset,
        split_assignments=assignments,
        completion_view="first_turn",
    )
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    config = apply_training_profile(
        {
            "model": {
                "name_or_path": str(model),
                "backend": "unsloth",
                "tuning_mode": "qlora",
                "max_length": 4096,
                "multimodal": False,
                "tool_calling": True,
            },
            "lora": {"target_modules": ["q_proj"]},
            "training": {"output_dir": str(tmp_path / "training")},
        },
        "furniture_initial_pilot",
    )
    return dataset, config


@pytest.mark.parametrize(
    "validation_groups,expected_valid", [(1, False), (3, False), (4, True)]
)
def test_pilot_rejects_dependent_validation_pairs(
    tmp_path: Path,
    validation_groups: int,
    expected_valid: bool,
) -> None:
    dataset, config = _validation_package(tmp_path, validation_groups)
    result = validate_training_request(config, dataset_dir=dataset)

    assert result["dataset_validation"]["valid"]
    assert result["split_stats"]["train"]["unique_leakage_groups"] == 8
    assert result["split_stats"]["validation"]["pair_count"] == 4
    assert result["valid"] is expected_valid
    if not expected_valid:
        assert result["errors"] == [
            f"validation split has {validation_groups} independent scenario groups; "
            "at least 4 are required"
        ]


def test_small_dataset_override_cannot_bypass_validation_independence(
    tmp_path: Path,
) -> None:
    dataset, config = _validation_package(tmp_path, 1)
    config["data"]["allow_unsafe_small_dataset"] = True

    result = validate_training_request(config, dataset_dir=dataset)

    assert not result["valid"]
    assert any("independent scenario groups" in error for error in result["errors"])


def test_validation_group_minimum_is_opt_in_for_existing_configs(
    tmp_path: Path,
) -> None:
    dataset, config = _validation_package(tmp_path, 1)
    config["data"].pop("minimum_unique_validation_groups")

    assert validate_training_request(config, dataset_dir=dataset)["valid"]
