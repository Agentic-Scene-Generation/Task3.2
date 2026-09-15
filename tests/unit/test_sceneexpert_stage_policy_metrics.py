from __future__ import annotations

import pytest

from scenesmith.scene_expert.schemas import (
    REQUIRED_FIRST_DIRECTIVE,
    SceneTaskSpec,
    StageBrief,
    StageExecutionEvidence,
    StageVerifyReport,
)
from scenesmith.scene_expert.verifier import (
    FullVerifier,
    StageVerifier,
    evaluate_required_asset_coverage,
)


def _task_spec() -> SceneTaskSpec:
    return SceneTaskSpec(
        room_type="classroom",
        style="modern",
        required_large_objects=["student desk", "chair", "bookcase"],
    )


def test_auto_brief_delivers_required_first_without_closing_optional_design() -> None:
    brief = StageBrief(
        stage="furniture",
        stage_policy="auto",
        optional_assets_allowed=True,
        required_objects=["student desk", "chair"],
        stage_objective="Create a functional classroom.",
    )

    injection = brief.to_injection_text()

    assert REQUIRED_FIRST_DIRECTIVE in injection
    assert "OPTIONAL ASSET RECOMMENDATIONS" in injection
    assert "native designer judgment" in injection
    assert "skip" not in injection.casefold()


def test_required_coverage_is_independent_from_optional_scene_content() -> None:
    evidence = evaluate_required_asset_coverage(
        _task_spec(),
        "furniture",
        {
            "object_names": [
                "student_desk_0",
                "chair_0",
                "optional_floor_lamp_0",
                "optional_rug_0",
            ]
        },
    )

    assert evidence["requirement_status"] == "partial"
    assert evidence["required_satisfied_objects"] == ["student desk", "chair"]
    assert evidence["required_missing_objects"] == ["bookcase"]
    assert evidence["required_coverage"] == pytest.approx(2 / 3)


def test_stage_verifier_persists_required_coverage_without_scoring_optional_assets(
    tmp_path,
) -> None:
    report = StageVerifier(critic_bridge_enabled=False).verify(
        stage="furniture",
        stage_output_dir=str(tmp_path),
        task_spec=_task_spec(),
        scene_state_info={"object_names": ["student_desk_0", "chair_0", "rug_0"]},
    )

    assert report.pass_stage is False
    assert report.requirement_status == "partial"
    assert report.required_missing_objects == ["bookcase"]
    assert report.required_coverage == pytest.approx(2 / 3)


def test_full_result_keeps_quality_separate_from_required_fulfillment() -> None:
    report = FullVerifier().verify(
        [
            StageVerifyReport(
                stage="furniture",
                pass_stage=False,
                visual_scores={"semantic": 0.9, "aesthetic": 0.8},
                required_objects=["student desk", "chair"],
                required_satisfied_objects=["student desk"],
                required_missing_objects=["chair"],
                required_coverage=0.5,
                requirement_status="partial",
            )
        ]
    )

    assert report.pass_scene is False
    assert report.requirement_status == "partial"
    assert report.quality_status == "passed"
    assert report.generation_status == "unknown"


def test_new_execution_evidence_fields_are_backward_compatible() -> None:
    evidence = StageExecutionEvidence.model_validate(
        {"stage_policy": "auto", "required_objects": ["desk"]}
    )

    assert evidence.required_first_instruction_delivered is False
    assert evidence.optional_autonomy_preserved is False
    assert evidence.requirement_status == "unknown"
    assert evidence.stage_outcome == "running"
