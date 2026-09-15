"""Strict adapters from authoritative main failures to wrapper evidence."""

from __future__ import annotations

from scenesmith.scene_expert.schemas import StageVerifyReport, VerifyIssue

_SUPPORTED_STAGES = {
    "floor_plan",
    "furniture",
    "wall_mounted",
    "ceiling_mounted",
    "manipuland",
}


def main_hard_failure_report(
    error: str, current_stage: str
) -> StageVerifyReport | None:
    """Convert only a recognized main hard-gate error into writer evidence."""
    compact_error = " ".join(str(error or "").split())
    lowered = compact_error.casefold()
    markers = (
        "stage failed with unresolved core relations:",
        "stage failed with unresolved hard constraints:",
    )
    marker = next((value for value in markers if value in lowered), "")
    if not marker or current_stage not in _SUPPORTED_STAGES:
        return None
    details = compact_error[lowered.index(marker) + len(marker) :].strip()
    issue_description = details or compact_error
    return StageVerifyReport(
        stage=current_stage,
        pass_stage=False,
        issues=[
            VerifyIssue(
                issue_type="main_pipeline_hard_failure",
                object_name=details,
                description=issue_description,
            )
        ],
        critique_summary=compact_error,
        score_source="scenesmith_main_hard_gate",
        vlm_scoring_performed=False,
        hard_check_report={
            "hard_valid": False,
            "failed_checks": [issue_description],
            "source": "scenesmith_main_hard_gate",
        },
    )
