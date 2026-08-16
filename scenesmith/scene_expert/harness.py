"""Deterministic Harness: FSM, budget control, and repair strategy selection.

Pure Python — no LLM calls. Controls stage execution flow and ensures
Qwen3 cannot skip stages, randomly call tools, or exceed budgets.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from omegaconf import DictConfig

from scenesmith.scene_expert.schemas import (
    HarnessContext,
    MemoryPack,
    SceneTaskSpec,
    StageBrief,
    StageBudget,
    StageVerifyReport,
)

console_logger = logging.getLogger(__name__)

# Fixed stage order matching SceneSmith's pipeline
STAGE_ORDER = [
    "floor_plan",
    "furniture",
    "wall_mounted",
    "ceiling_mounted",
    "manipuland",
]

# Maps SceneSmith internal stage names (for checkpoint loading)
STAGE_TO_CHECKPOINT = {
    "floor_plan": None,  # First stage — no prior checkpoint
    "furniture": "house_layout.json",
    "wall_mounted": "scene_after_furniture",
    "ceiling_mounted": "scene_after_wall_objects",
    "manipuland": "scene_after_ceiling_objects",
}


@dataclass
class RepairDecision:
    """Harness decision on how to handle a failed stage."""

    should_repair: bool
    strategy: str  # "local_repair", "stage_regeneration", "rollback", "skip"
    reason: str = ""


class Harness:
    """Deterministic controller for the SceneExpert online loop.

    Responsibilities:
    - Validates stage order (FSM — cannot skip stages)
    - Assigns per-stage budgets
    - Decides repair strategy based on verify reports
    - Tracks repair attempt counts per stage
    """

    def __init__(self, cfg: DictConfig) -> None:
        self._cfg = cfg
        self._repair_counts: dict[str, int] = {}  # stage → repair attempts so far

    def reset(self) -> None:
        """Reset per-run state (call before each new scene generation)."""
        self._repair_counts = {}

    def build_context(
        self,
        stage: str,
        task_spec: SceneTaskSpec,
        memory_pack: MemoryPack,
        stage_brief: StageBrief | None = None,
    ) -> HarnessContext:
        """Assemble the HarnessContext for a stage execution.

        Args:
            stage: Current stage name.
            task_spec: Compiled task specification.
            memory_pack: Retrieved memory for this stage.
            stage_brief: Generated StageBrief (may be None before planner runs).

        Returns:
            HarnessContext with budget filled in.
        """
        if stage not in STAGE_ORDER:
            raise ValueError(f"Unknown stage: {stage}. Valid stages: {STAGE_ORDER}")

        budget = self._get_stage_budget(stage)
        return HarnessContext(
            stage=stage,
            task_spec=task_spec,
            memory_pack=memory_pack,
            stage_brief=stage_brief,
            stage_budget=budget,
            allowed_scene_smith_stage=stage,
        )

    def decide_repair(
        self, stage: str, verify_report: StageVerifyReport
    ) -> RepairDecision:
        """Decide whether and how to repair a failed stage.

        Uses a light-to-heavy escalation:
          attempt 0: local_repair
          attempt 1: stage_regeneration
          attempt 2+: skip (budget exhausted)

        Args:
            stage: Stage that failed verification.
            verify_report: The verifier's assessment.

        Returns:
            RepairDecision with strategy and reason.
        """
        if verify_report.pass_stage:
            return RepairDecision(
                should_repair=False, strategy="skip", reason="Stage passed"
            )

        budget = self._get_stage_budget(stage)
        attempt = self._repair_counts.get(stage, 0)

        if attempt >= budget.max_repair_steps:
            return RepairDecision(
                should_repair=False,
                strategy="skip",
                reason=f"Repair budget exhausted (max_repair_steps={budget.max_repair_steps})",
            )

        # Escalate strategy based on attempt number
        if attempt == 0:
            strategy = "local_repair"
        elif attempt == 1:
            strategy = "stage_regeneration"
        else:
            strategy = "rollback"

        self._repair_counts[stage] = attempt + 1

        reason = (
            f"Attempt {attempt + 1}/{budget.max_repair_steps}: "
            f"issues={[i.issue_type for i in verify_report.issues]}"
        )
        console_logger.info(
            f"Harness repair decision for {stage}: {strategy} ({reason})"
        )

        return RepairDecision(should_repair=True, strategy=strategy, reason=reason)

    def _get_stage_budget(self, stage: str) -> StageBudget:
        """Get per-stage budget from config."""
        stage_cfg = getattr(self._cfg, "stage_budget", None)
        if stage_cfg is None:
            return StageBudget()

        # Check stage-specific override first
        stage_override = getattr(stage_cfg, stage, None)
        if stage_override is not None:
            return StageBudget(
                max_designer_iterations=getattr(
                    stage_override, "max_designer_iterations", 2
                ),
                max_repair_steps=getattr(stage_override, "max_repair_steps", 1),
            )

        # Fall back to default
        default = getattr(stage_cfg, "default", None)
        if default is not None:
            return StageBudget(
                max_designer_iterations=getattr(default, "max_designer_iterations", 2),
                max_repair_steps=getattr(default, "max_repair_steps", 1),
            )

        return StageBudget()

    def validate_stage_order(
        self, completed_stages: list[str], next_stage: str
    ) -> bool:
        """Assert stage execution follows the fixed FSM order.

        Args:
            completed_stages: Stages already completed in this run.
            next_stage: The stage about to be executed.

        Returns:
            True if the transition is valid.

        Raises:
            ValueError: If stage order is violated.
        """
        if next_stage not in STAGE_ORDER:
            raise ValueError(f"Unknown stage: {next_stage}")

        expected_idx = len(completed_stages)
        actual_idx = STAGE_ORDER.index(next_stage)

        if actual_idx != expected_idx:
            expected_stage = (
                STAGE_ORDER[expected_idx] if expected_idx < len(STAGE_ORDER) else "done"
            )
            raise ValueError(
                f"Stage order violation: expected '{expected_stage}' "
                f"(index {expected_idx}) but got '{next_stage}' (index {actual_idx}). "
                f"Completed: {completed_stages}"
            )
        return True
