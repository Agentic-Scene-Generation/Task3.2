"""RepairController: handles stage repair after failed verification.

Three strategies (lightest to heaviest):
1. local_repair  — generates a text instruction for the designer to fix specific objects
2. stage_regeneration — re-runs the stage from its checkpoint with an updated brief
3. rollback — reverts to the previous stage checkpoint (uses SceneSmith resume_from_path)

Hook-runner MVP records local_repair / stage_regeneration / rollback decisions
in trace and memory context, but does not re-enter SceneSmith agents in the
same hook call. Actual re-execution requires explicit pipeline-level support.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from scenesmith.scene_expert.memory.schemas import FailureCase
from scenesmith.scene_expert.memory.store import FastMemoryStore
from scenesmith.scene_expert.schemas import (
    RepairResult,
    SceneTaskSpec,
    StageBrief,
    StageVerifyReport,
    VerifyIssue,
)

console_logger = logging.getLogger(__name__)


def _issue_to_repair_instruction(
    issue: VerifyIssue, stage_brief: StageBrief | None
) -> str:
    """Convert a verifier issue into a concrete designer instruction."""
    base = f"Fix issue: {issue.issue_type}"
    if issue.object_name:
        base += f" for object '{issue.object_name}'"
    if issue.description:
        base += f". Details: {issue.description}"

    # Augment with repair hints from the stage brief's failure patterns
    if stage_brief:
        relevant = [
            p
            for p in stage_brief.failure_patterns_to_avoid
            if issue.object_name.lower() in p.lower() or issue.issue_type in p.lower()
        ]
        if relevant:
            base += f". Memory hint: {relevant[0]}"

    return base


def _build_repair_instruction(
    verify_report: StageVerifyReport,
    stage_brief: StageBrief | None,
    memory_store: FastMemoryStore | None,
) -> str:
    """Build a repair instruction string for the local_repair strategy."""
    instructions: list[str] = []

    for issue in verify_report.issues:
        instructions.append(_issue_to_repair_instruction(issue, stage_brief))

    # Add suggestions from verifier
    if verify_report.repair_suggestions:
        instructions.extend(verify_report.repair_suggestions)

    # Look up matching failure cases in memory for additional context
    if memory_store and verify_report.issues:
        for issue in verify_report.issues[:2]:  # limit to first 2 issues
            for case in memory_store.failure_cases:
                if (
                    case.stage == verify_report.stage
                    and case.failure_type == issue.issue_type
                    and case.repair_action
                    and (
                        not issue.object_name
                        or case.object.lower() in issue.object_name.lower()
                    )
                ):
                    instructions.append(f"Similar past fix: {case.repair_action}")
                    break

    if not instructions:
        instructions = [
            f"Review the {verify_report.stage} stage output and fix any quality issues"
        ]

    return "\n".join(f"{i+1}. {inst}" for i, inst in enumerate(instructions))


class RepairController:
    """Executes repair strategies based on the Harness's repair decision."""

    def __init__(self, memory_store: FastMemoryStore | None = None) -> None:
        self._memory_store = memory_store

    def repair(
        self,
        repair_type: str,
        stage: str,
        verify_report: StageVerifyReport,
        scene_path: str,
        stage_brief: StageBrief | None = None,
        task_spec: SceneTaskSpec | None = None,
    ) -> RepairResult:
        """Execute the selected repair strategy.

        Args:
            repair_type: "local_repair", "stage_regeneration", or "rollback".
            stage: Stage name being repaired.
            verify_report: The verifier report that triggered this repair.
            scene_path: Path to the current stage's scene output directory.
            stage_brief: Current stage brief (for context-aware repair instructions).
            task_spec: Task specification (for stage_regeneration brief update).

        Returns:
            RepairResult documenting the repair action taken.
        """
        console_logger.info(
            f"RepairController: executing {repair_type} for stage {stage}"
        )

        if repair_type == "local_repair":
            return self._local_repair(stage, verify_report, scene_path, stage_brief)
        elif repair_type == "stage_regeneration":
            return self._stage_regeneration(stage, verify_report, scene_path, task_spec)
        elif repair_type == "rollback":
            return self._rollback(stage, scene_path)
        else:
            console_logger.warning(f"Unknown repair type: {repair_type}, skipping")
            return RepairResult(
                repair_type="skipped", failure_type="unknown_repair_type"
            )

    def _local_repair(
        self,
        stage: str,
        verify_report: StageVerifyReport,
        scene_path: str,
        stage_brief: StageBrief | None,
    ) -> RepairResult:
        """Generate a text repair instruction for the SceneSmith designer.

        The instruction is stored in the RepairResult.repair_action field.
        The pipeline is responsible for injecting this instruction into the designer.
        """
        failure_types = [issue.issue_type for issue in verify_report.issues]
        instruction = _build_repair_instruction(
            verify_report, stage_brief, self._memory_store
        )

        console_logger.info(
            f"LocalRepair [{stage}]: {len(verify_report.issues)} issues, "
            f"types={failure_types}"
        )

        return RepairResult(
            repair_type="local_repair",
            failure_type="; ".join(failure_types),
            repair_action=instruction,
            repair_verified=False,  # Will be updated after re-verification
            new_scene_state=scene_path,
        )

    def _stage_regeneration(
        self,
        stage: str,
        verify_report: StageVerifyReport,
        scene_path: str,
        task_spec: SceneTaskSpec | None,
    ) -> RepairResult:
        """Request full stage regeneration from checkpoint.

        Signals to the pipeline to re-run the entire stage.
        The pipeline handles the actual re-execution.
        """
        failure_types = [issue.issue_type for issue in verify_report.issues]
        action = (
            f"Regenerate entire {stage} stage from checkpoint. "
            f"Previous issues: {'; '.join(issue.description for issue in verify_report.issues)}. "
            f"Ensure stage brief constraints are strictly followed."
        )

        console_logger.info(f"StageRegeneration [{stage}]: requesting full stage redo")

        return RepairResult(
            repair_type="stage_regeneration",
            failure_type="; ".join(failure_types),
            repair_action=action,
            repair_verified=False,
            new_scene_state="",  # Will be updated by pipeline after re-execution
        )

    def _rollback(self, stage: str, scene_path: str) -> RepairResult:
        """Roll back to the previous stage checkpoint.

        Note: Full rollback requires SceneSmith's resume_from_path mechanism.
        This method returns the decision; the pipeline executes the actual rollback.
        """
        console_logger.info(
            f"Rollback [{stage}]: requesting rollback to previous checkpoint"
        )
        return RepairResult(
            repair_type="rollback",
            failure_type="quality_too_low",
            repair_action=f"Roll back stage {stage} to previous checkpoint and re-execute",
            repair_verified=False,
            new_scene_state="",
        )

    def record_failure_to_memory(
        self,
        stage: str,
        room_type: str,
        repair_result: RepairResult,
        verify_report: StageVerifyReport,
        repair_verified: bool,
    ) -> FailureCase | None:
        """Build a FailureCase from a completed repair for the memory store.

        Returns None if there's nothing worth recording (e.g., no clear failure pattern).
        """
        if not verify_report.issues:
            return None

        primary_issue = verify_report.issues[0]
        case_id = f"fail_{room_type}_{stage}_{id(verify_report):08x}"

        case = FailureCase(
            failure_id=case_id,
            room_type=room_type,
            stage=stage,
            object=primary_issue.object_name,
            failure_type=primary_issue.issue_type,
            bad_pattern=primary_issue.description,
            failure_reason=primary_issue.description,
            repair_action=repair_result.repair_action,
            repair_verified=repair_verified,
        )

        if self._memory_store and repair_verified:
            self._memory_store.add_failure_case(case)
            console_logger.debug(f"RepairController: recorded failure case {case_id}")

        return case
