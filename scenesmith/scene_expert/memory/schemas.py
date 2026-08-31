"""Stable data contracts for the SceneExpert fast-memory bank.

The persisted record schemas are intentionally backward compatible: old JSONL
records load with conservative defaults, while new records carry provenance and
bank-version metadata.  The compact ``MemoryWriterResponse`` schemas are kept
separate from persisted records so an LLM never owns record identity, quality
scores, provenance, or promotion status.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

MEMORY_SCHEMA_VERSION = "sceneexpert.memory.v3"
MemoryStatus = Literal["candidate", "active", "quarantined"]
MemorySource = Literal["llm", "deterministic", "legacy", "imported"]


class MemorySourceProvenance(BaseModel):
    """Stable locator for the evidence from which a memory was derived."""

    task_id: str = ""
    run_id: str = ""
    trace_id: str = ""
    scene_state_path: str = ""
    stage: str = ""
    prompt_fingerprint: str = ""
    evidence_refs: list[str] = Field(default_factory=list)
    critic_source: str = ""


class SpatialRelationMemory(BaseModel):
    """Transferable spatial relation grounded in critic or geometry evidence.

    Exact coordinates are optional because they are usually scene-specific.  A
    relation is safe to promote without coordinates only when its semantic
    endpoints and evidence reference are retained.
    """

    relation_type: str = Field(min_length=1)
    subject_role: str = ""
    target_role: str = ""
    normalized_offset: list[float] = Field(default_factory=list, max_length=3)
    yaw_delta_deg: float | None = None
    clearance_m: dict[str, float] = Field(default_factory=dict)
    cardinality: dict[str, Any] = Field(default_factory=dict)
    evidence_source: Literal[
        "critic", "deterministic", "scene_geometry", "task_contract", "legacy"
    ] = "legacy"
    evidence_ref: str = ""
    geometry_verified: bool = False
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    def to_guidance_text(self) -> str:
        """Render a compact designer-facing relation without inventing geometry."""
        endpoints = " ".join(
            value
            for value in (self.subject_role, self.relation_type, self.target_role)
            if value
        )
        parts = [endpoints or self.relation_type]
        if self.normalized_offset:
            parts.append(
                "normalized offset="
                + ",".join(f"{value:.3f}" for value in self.normalized_offset)
            )
        if self.yaw_delta_deg is not None:
            parts.append(f"yaw delta={self.yaw_delta_deg:.1f}deg")
        if self.clearance_m:
            parts.append(
                "clearance="
                + ",".join(
                    f"{key}:{value:.3f}m"
                    for key, value in sorted(self.clearance_m.items())
                )
            )
        return "; ".join(parts)


class SkillApplicability(BaseModel):
    """Explicit guardrails preventing a skill from leaking to unrelated tasks."""

    room_types: list[str] = Field(default_factory=list)
    excluded_room_types: list[str] = Field(default_factory=list)
    required_object_roles: list[str] = Field(default_factory=list)
    required_relation_types: list[str] = Field(default_factory=list)
    forbidden_conditions: list[str] = Field(default_factory=list)


class MemoryUtilityObservation(BaseModel):
    """One immutable observation of how selected memory affected a later run."""

    memory_id: str
    memory_type: Literal["success", "failure", "skill"]
    task_id: str = ""
    run_id: str = ""
    stage: str = ""
    selected_rank: int = Field(default=0, ge=0)
    retrieval_score: float | None = None
    injected: bool = False
    retrieved: bool = True
    planner_selected: bool = False
    prompt_delivered: bool = False
    stage_passed: bool | None = None
    quality_delta: float | None = None
    latency_delta_sec: float | None = None
    outcome: Literal["positive", "negative", "neutral", "unknown"] = "unknown"
    outcome_basis: str = ""
    evidence_ref: str = ""


class MemoryRecordBase(BaseModel):
    """Common lifecycle and provenance fields for every persisted record."""

    schema_version: str = MEMORY_SCHEMA_VERSION
    status: MemoryStatus = "active"
    source: MemorySource = "legacy"
    source_task_id: str = ""
    source_run_id: str = ""
    source_task_ids: list[str] = Field(default_factory=list)
    source_run_ids: list[str] = Field(default_factory=list)
    prompt_fingerprint: str = ""
    evidence_refs: list[str] = Field(default_factory=list)
    critic_evidence: list[str] = Field(default_factory=list)
    bank_version: int = Field(default=0, ge=0)
    observation_count: int = Field(default=1, ge=1)
    positive_utility_count: int = Field(default=0, ge=0)
    negative_utility_count: int = Field(default=0, ge=0)
    updated_at: str = ""
    provenance: MemorySourceProvenance = Field(default_factory=MemorySourceProvenance)
    spatial_relations: list[SpatialRelationMemory] = Field(default_factory=list)


class SuccessCase(MemoryRecordBase):
    """A recorded successful scene generation pattern."""

    case_id: str
    promotion_scope: Literal["scene", "stage"] = "scene"
    source_scene_passed: bool = True
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
    required_objects: list[str] = Field(default_factory=list)
    functional_zones: list[str] = Field(default_factory=list)
    scene_summary: str = ""
    positive_guidance: list[str] = Field(default_factory=list)
    embedding_text: str = ""
    confidence: float = 0.5
    quality_score: float = 0.5
    created_at: str = ""
    last_used_at: str = ""
    usage_count: int = 0

    def to_hint_text(self) -> str:
        """Compress into a single retrieval hint string (for GlobalPlanner context)."""
        patterns = "; ".join(self.positive_guidance or self.successful_pattern)
        score_str = ", ".join(f"{k}={v:.2f}" for k, v in self.scores.items())
        return f"[Success/{self.stage}] {self.room_type} ({self.style}): {patterns}" + (
            f" [scores: {score_str}]" if score_str else ""
        )

    def to_positive_guidance(self) -> str:
        """Format as positive guidance for future hybrid memory injection."""
        guidance = self.positive_guidance or self.successful_pattern
        lines = [f"[Positive/{self.stage}] {self.room_type} ({self.style})"]
        lines.extend(f"- {item}" for item in guidance if item)
        return "\n".join(lines)

    def to_placement_text(self) -> str:
        """Format placement_reference as a designer-readable reference block."""
        if not self.placement_reference and not self.spatial_relations:
            return ""
        score_str = ", ".join(f"{k}={v:.2f}" for k, v in self.scores.items())
        lines = [
            f"=== Reference Layout ({self.stage} / {self.room_type} / {self.style}) ===",
            f"Scores achieved: {score_str}",
            "Grounded spatial references that produced these scores:",
        ]
        for entry in self.placement_reference:
            lines.append(f"  {entry}")
        for relation in self.spatial_relations:
            lines.append(f"  relation: {relation.to_guidance_text()}")
        lines.append(
            "Use this as a spatial reference. "
            "Adapt positions to the current room size and prompt if needed."
        )
        lines.append("=== End Reference Layout ===")
        return "\n".join(lines)


class FailureCase(MemoryRecordBase):
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
    required_objects: list[str] = Field(default_factory=list)
    functional_zones: list[str] = Field(default_factory=list)
    scene_summary: str = ""
    embedding_text: str = ""
    confidence: float = 0.5
    quality_score: float = 0.5
    created_at: str = ""
    last_used_at: str = ""
    usage_count: int = 0
    scope: str = "object"  # "global" | "stage" | "room" | "object"
    is_deterministic: bool = False
    repeat_count: int = 1
    negative_constraint: str = ""
    critic_check: str = ""

    def to_hint_text(self) -> str:
        """Format as an avoid-rule hint."""
        avoid_text = self.negative_constraint or self.bad_pattern
        return (
            f"[Avoid/{self.stage}] In {self.room_type}: {avoid_text}"
            + (f" — reason: {self.failure_reason}" if self.failure_reason else "")
            + (f" — fix: {self.repair_action}" if self.repair_action else "")
            + (f" — check: {self.critic_check}" if self.critic_check else "")
        )

    def to_negative_constraint(self) -> str:
        """Format as a compact negative constraint for future hybrid injection."""
        avoid_text = self.negative_constraint or self.bad_pattern
        parts = [f"[Avoid/{self.stage}] {avoid_text}"]
        if self.repair_action:
            parts.append(f"Fix: {self.repair_action}")
        if self.critic_check:
            parts.append(f"Check: {self.critic_check}")
        if self.spatial_relations:
            parts.append(
                "Relations: "
                + " | ".join(
                    relation.to_guidance_text() for relation in self.spatial_relations
                )
            )
        return " ".join(part for part in parts if part)


class Skill(MemoryRecordBase):
    """A reusable procedural skill template."""

    skill_name: str
    stage: str
    room_type: str = ""
    style: str = ""
    room_types: list[str] = Field(default_factory=list)
    required_objects: list[str] = Field(default_factory=list)
    functional_zones: list[str] = Field(default_factory=list)
    scene_summary: str = ""
    preconditions: list[str] = Field(default_factory=list)
    procedure: list[str] = Field(default_factory=list)
    failure_avoidance: list[str] = Field(default_factory=list)
    postconditions: list[str] = Field(default_factory=list)
    embedding_text: str = ""
    confidence: float = 0.5
    quality_score: float = 0.5
    success_rate: float = 0.0
    trace_ref: str = ""
    created_at: str = ""
    last_used_at: str = ""
    usage_count: int = 0
    applicability: SkillApplicability = Field(default_factory=SkillApplicability)
    utility_observations: list[MemoryUtilityObservation] = Field(default_factory=list)

    def to_procedure_text(self) -> str:
        """Format skill as an ordered procedure for prompt injection."""
        lines = [f"[Skill: {self.skill_name}]"]
        applicable_rooms = self.applicability.room_types or self.room_types
        if applicable_rooms:
            lines.append("Applicable rooms: " + ", ".join(applicable_rooms))
        if self.applicability.required_object_roles:
            lines.append(
                "Required object roles: "
                + ", ".join(self.applicability.required_object_roles)
            )
        if self.applicability.required_relation_types:
            lines.append(
                "Required task relations: "
                + ", ".join(self.applicability.required_relation_types)
            )
        if self.applicability.forbidden_conditions:
            lines.append(
                "Do not apply when: "
                + ", ".join(self.applicability.forbidden_conditions)
            )
        if self.preconditions:
            lines.append("Preconditions: " + ", ".join(self.preconditions))
        if self.procedure:
            lines.append("Steps:")
            lines.extend(f"  {i+1}. {step}" for i, step in enumerate(self.procedure))
        if self.failure_avoidance:
            lines.append("Avoid:")
            lines.extend(f"  - {rule}" for rule in self.failure_avoidance)
        if self.postconditions:
            lines.append("Postconditions:")
            lines.extend(f"  - {condition}" for condition in self.postconditions)
        if self.spatial_relations:
            lines.append("Spatial relations:")
            lines.extend(
                f"  - {relation.to_guidance_text()}"
                for relation in self.spatial_relations
            )
        return "\n".join(lines)


class MemoryUpdateOp(BaseModel):
    """A single memory update operation from the memory writer."""

    op: Literal["ADD", "UPDATE", "NOOP"]
    memory_type: Literal["success_case", "failure_case", "skill"]
    content: dict = Field(default_factory=dict)
    target_id: str = ""  # for UPDATE: the case_id / skill_name to update


class SuccessMemoryCandidate(BaseModel):
    """Small, strict LLM output for one reusable successful pattern."""

    model_config = ConfigDict(extra="forbid")

    stage: str = Field(min_length=1)
    successful_pattern: list[str] = Field(min_length=1)
    positive_guidance: list[str] = Field(default_factory=list)


class FailureMemoryCandidate(BaseModel):
    """Small, strict LLM output for one transferable failure lesson."""

    model_config = ConfigDict(extra="forbid")

    stage: str = Field(min_length=1)
    object: str = ""
    failure_type: str = Field(min_length=1)
    bad_pattern: str = Field(min_length=1)
    failure_reason: str = Field(min_length=1)
    repair_action: str = Field(min_length=1)
    repair_verified: bool = False
    scope: Literal["global", "stage", "room", "object"] = "stage"
    is_deterministic: bool = False
    negative_constraint: str = ""
    critic_check: str = ""


class SkillMemoryCandidate(BaseModel):
    """Small, strict LLM output for a genuinely reusable procedure."""

    model_config = ConfigDict(extra="forbid")

    skill_name: str = Field(min_length=1)
    stage: str = Field(min_length=1)
    preconditions: list[str] = Field(default_factory=list)
    procedure: list[str] = Field(min_length=2)
    failure_avoidance: list[str] = Field(default_factory=list)
    postconditions: list[str] = Field(default_factory=list)


class MemoryWriterResponse(BaseModel):
    """Validated MemoryWriter response before deterministic promotion."""

    model_config = ConfigDict(extra="forbid")

    success_cases: list[SuccessMemoryCandidate] = Field(
        default_factory=list, max_length=5
    )
    failure_cases: list[FailureMemoryCandidate] = Field(
        default_factory=list, max_length=5
    )
    skills: list[SkillMemoryCandidate] = Field(default_factory=list, max_length=2)
    noop_reason: str = ""
