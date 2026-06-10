"""Pydantic schemas for the SceneExpert fast memory system."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SuccessCase(BaseModel):
    """A recorded successful scene generation pattern."""

    case_id: str
    room_type: str
    style: str = ""
    stage: str
    task_signature: list[str] = Field(
        default_factory=list,
        description="Key object names / zone names — used for retrieval matching",
    )
    successful_pattern: list[str] = Field(
        default_factory=list,
        description="Description of what worked well in this stage",
    )
    placement_reference: list[str] = Field(
        default_factory=list,
        description=(
            "Exact object placements that achieved these scores. "
            "One entry per object: 'object_id (name): x=..., y=..., yaw=...'. "
            "Injected directly into the designer prompt as a spatial reference."
        ),
    )
    scores: dict[str, float] = Field(default_factory=dict)
    trace_ref: str = ""

    def to_hint_text(self) -> str:
        """Compress into a single retrieval hint string (for GlobalPlanner context)."""
        patterns = "; ".join(self.successful_pattern)
        score_str = ", ".join(f"{k}={v:.2f}" for k, v in self.scores.items())
        return (
            f"[Success/{self.stage}] {self.room_type} ({self.style}): {patterns}"
            + (f" [scores: {score_str}]" if score_str else "")
        )

    def to_placement_text(self) -> str:
        """Format placement_reference as a designer-readable reference block."""
        if not self.placement_reference:
            return ""
        score_str = ", ".join(f"{k}={v:.2f}" for k, v in self.scores.items())
        lines = [
            f"=== Reference Layout ({self.stage} / {self.room_type} / {self.style}) ===",
            f"Scores achieved: {score_str}",
            "Object placements that produced these scores:",
        ]
        for entry in self.placement_reference:
            lines.append(f"  {entry}")
        lines.append(
            "Use this as a spatial reference. "
            "Adapt positions to the current room size and prompt if needed."
        )
        lines.append("=== End Reference Layout ===")
        return "\n".join(lines)


class FailureCase(BaseModel):
    """A recorded failure pattern with its verified repair action."""

    failure_id: str
    room_type: str
    stage: str
    object: str = ""
    failure_type: str = ""  # e.g., "unreachable", "collision", "missing_object"
    bad_pattern: str = ""
    failure_reason: str = ""
    repair_action: str = ""
    repair_verified: bool = False

    def to_hint_text(self) -> str:
        """Format as an avoid-rule hint."""
        return (
            f"[Avoid/{self.stage}] In {self.room_type}: {self.bad_pattern}"
            + (f" — reason: {self.failure_reason}" if self.failure_reason else "")
            + (f" — fix: {self.repair_action}" if self.repair_action else "")
        )


class Skill(BaseModel):
    """A reusable procedural skill template."""

    skill_name: str
    stage: str
    room_types: list[str] = Field(default_factory=list)
    preconditions: list[str] = Field(default_factory=list)
    procedure: list[str] = Field(default_factory=list)
    failure_avoidance: list[str] = Field(default_factory=list)
    postconditions: list[str] = Field(default_factory=list)

    def to_procedure_text(self) -> str:
        """Format skill as an ordered procedure for prompt injection."""
        lines = [f"[Skill: {self.skill_name}]"]
        if self.preconditions:
            lines.append("Preconditions: " + ", ".join(self.preconditions))
        if self.procedure:
            lines.append("Steps:")
            lines.extend(f"  {i+1}. {step}" for i, step in enumerate(self.procedure))
        if self.failure_avoidance:
            lines.append("Avoid:")
            lines.extend(f"  - {rule}" for rule in self.failure_avoidance)
        return "\n".join(lines)


class MemoryUpdateOp(BaseModel):
    """A single memory update operation from the memory writer."""

    op: str  # "ADD", "UPDATE", "NOOP"
    memory_type: str  # "success_case", "failure_case", "skill"
    content: dict = Field(default_factory=dict)
    target_id: str = ""  # for UPDATE: the case_id / skill_name to update
