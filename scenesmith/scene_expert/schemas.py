"""Pydantic data schemas for SceneExpert MVP.

All inter-module data contracts are defined here to ensure
type safety and easy JSON serialization across the pipeline.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ---------------------------------------------------------------------------
# TaskCompiler output
# ---------------------------------------------------------------------------


class ObjectSelectorSpec(BaseModel):
    """Semantic endpoint selector; generated object ids are bound later."""

    model_config = ConfigDict(extra="forbid")

    category: str = Field(min_length=1)
    count: int | None = Field(default=None, ge=1)
    quantifier: Literal["all", "exactly", "at_least", "minimum"] = "all"
    role: str = ""
    secondary_category: str = ""
    secondary_count: int | None = Field(default=None, ge=1)
    secondary_role: str = ""

    @field_validator("category", "secondary_category", mode="before")
    @classmethod
    def _normalize_category(cls, value: Any) -> str:
        return "_".join(str(value or "").strip().lower().split())

    @field_validator("role", "secondary_role", mode="before")
    @classmethod
    def _normalize_role(cls, value: Any) -> str:
        return str(value or "").strip().lower()


class SceneTaskSpec(BaseModel):
    """Structured scene requirements extracted from a raw text prompt."""

    model_config = ConfigDict(extra="forbid")

    room_type: str = Field(
        ..., description="Primary room type, e.g. 'bedroom', 'kitchen'"
    )
    style: str = Field(
        ..., description="Aesthetic style, e.g. 'cozy modern', 'industrial'"
    )
    required_large_objects: list[str] = Field(
        default_factory=list,
        description="Furniture-scale objects that must be present (floor plan / furniture stage)",
    )
    required_wall_objects: list[str] = Field(
        default_factory=list,
        description="Wall-mounted objects required",
    )
    required_ceiling_objects: list[str] = Field(
        default_factory=list,
        description="Ceiling-mounted objects required",
    )
    required_small_objects: list[str] = Field(
        default_factory=list,
        description="Manipuland-scale objects required (books, cups, etc.)",
    )
    required_architectural_features: list[str] = Field(
        default_factory=list,
        description="Explicit structural features such as windows and exposed beams",
    )
    suggested_large_objects: list[str] = Field(default_factory=list)
    suggested_wall_objects: list[str] = Field(default_factory=list)
    suggested_ceiling_objects: list[str] = Field(default_factory=list)
    suggested_small_objects: list[str] = Field(default_factory=list)
    requirement_sources: dict[str, list[str]] = Field(default_factory=dict)
    functional_zones: list[str] = Field(
        default_factory=list,
        description="Spatial zones within the room, e.g. ['sleeping_zone', 'working_zone']",
    )
    interaction_constraints: list[str] = Field(
        default_factory=list,
        description="Robot-interaction constraints: reachability, clearance, support surface rules",
    )
    aesthetic_constraints: list[str] = Field(
        default_factory=list,
        description="Visual / style constraints: material palette, density, symmetry, etc.",
    )
    compiler_status: Literal["ok", "degraded"] = "ok"
    compiler_failure_reason: str = ""
    compiler_spec_version: str = "scenesmith.task_compiler.v6"


# ---------------------------------------------------------------------------
# Memory pack returned by retriever
# ---------------------------------------------------------------------------


class MemoryPack(BaseModel):
    """Retrieved memory snippets to inject into a stage's StageBrief."""

    success_hints: list[str] = Field(
        default_factory=list,
        description="Compressed text hints from successful similar cases (for GlobalPlanner)",
    )
    failure_hints: list[str] = Field(
        default_factory=list,
        description="Avoid-rule hints derived from failure cases",
    )
    skill_texts: list[str] = Field(
        default_factory=list,
        description="Skill procedure texts formatted for prompt injection",
    )
    placement_reference: str = Field(
        default="",
        description=(
            "Formatted placement reference block from the top success case. "
            "Injected directly into the designer prompt, bypassing GlobalPlanner."
        ),
    )
    success_case_ids: list[str] = Field(default_factory=list)
    failure_case_ids: list[str] = Field(default_factory=list)
    skill_names: list[str] = Field(default_factory=list)
    retrieved_source_task_ids: dict[str, list[str]] = Field(
        default_factory=dict,
        description=(
            "Selected memory ID to source task IDs. This is audit metadata used "
            "to prove that guidance came from another task rather than a replay."
        ),
    )
    retrieved_source_run_ids: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Selected memory ID to source run IDs for experiment provenance.",
    )
    memory_bank_id: str = ""
    memory_bank_revision: int = 0

    def deduplicated(self) -> "MemoryPack":
        """Return an order-preserving copy without repeated prompt content."""

        def unique_text(values: list[str]) -> list[str]:
            seen: set[str] = set()
            result: list[str] = []
            for value in values:
                text = " ".join(str(value or "").split())
                key = text.casefold()
                if text and key not in seen:
                    result.append(text)
                    seen.add(key)
            return result

        return self.model_copy(
            update={
                "success_hints": unique_text(self.success_hints),
                "failure_hints": unique_text(self.failure_hints),
                "skill_texts": unique_text(self.skill_texts),
                "success_case_ids": unique_text(self.success_case_ids),
                "failure_case_ids": unique_text(self.failure_case_ids),
                "skill_names": unique_text(self.skill_names),
            }
        )


# ---------------------------------------------------------------------------
# Harness internals
# ---------------------------------------------------------------------------


class StageBudget(BaseModel):
    """Per-stage execution budget."""

    max_designer_iterations: int = 2
    max_repair_steps: int = 1


class FloorPlanReservation(BaseModel):
    """One deterministic capacity requirement owned by the floor-plan stage."""

    model_config = ConfigDict(extra="forbid")

    reservation_id: str
    kind: Literal[
        "wall_anchor",
        "opposed_anchor_pair",
        "functional_zone",
        "opening_adjacency",
    ]
    source_constraint_ids: list[str] = Field(default_factory=list)
    room_type: str = ""
    subject_categories: list[str] = Field(default_factory=list)
    target_categories: list[str] = Field(default_factory=list)
    wall_role: str = ""
    min_wall_width_m: float = Field(default=0.0, ge=0.0)
    min_zone_area_m2: float = Field(default=0.0, ge=0.0)
    count: int = Field(default=1, ge=1)
    hard: bool = True


class FloorPlanReservationManifest(BaseModel):
    """Serializable future-capacity contract shared by floor-plan validators."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "scenesmith.floor_plan_reservations.v2"
    enabled: bool = False
    reservations: list[FloorPlanReservation] = Field(default_factory=list)
    explicit_window_count: int = Field(default=0, ge=0)
    explicit_window_required: bool = False
    preserve_entrance_route: bool = True
    adaptive_window_budget: bool = True
    max_implicit_windows_per_wall: int = Field(default=1, ge=0)


class StageRelationContext(BaseModel):
    """Exact hard-intent projection for one stage."""

    model_config = ConfigDict(extra="forbid")

    stage: str
    hard_constraints: list[dict[str, Any]] = Field(default_factory=list)
    floor_plan_reservations: list[dict[str, Any]] = Field(default_factory=list)
    floor_plan_manifest: FloorPlanReservationManifest | None = None
    resolved_opening_reservations: dict[str, Any] = Field(default_factory=dict)
    contract_constraint_count: int = 0
    projected_constraint_count: int = 0
    projection_coverage: float = Field(default=1.0, ge=0.0, le=1.0)

    @property
    def hard_constraint_ids(self) -> list[str]:
        return [
            str(item.get("constraint_id") or "")
            for item in self.hard_constraints
            if str(item.get("constraint_id") or "")
        ]


class HarnessContext(BaseModel):
    """All inputs the Harness assembles before executing a SceneSmith stage."""

    stage: str
    task_spec: SceneTaskSpec
    memory_pack: MemoryPack
    relation_context: StageRelationContext | None = None
    stage_brief: "StageBrief | None" = None
    stage_budget: StageBudget = Field(default_factory=StageBudget)
    allowed_scene_smith_stage: str = ""

    model_config = {"arbitrary_types_allowed": True}


# ---------------------------------------------------------------------------
# Global Planner output
# ---------------------------------------------------------------------------


class StageBrief(BaseModel):
    """Expert planning hint generated by the Global Planner for one stage."""

    model_config = ConfigDict(extra="forbid")

    stage: str
    stage_objective: str = Field(
        ...,
        description="One-sentence goal for this stage",
    )
    recommended_skills: list[str] = Field(
        default_factory=list,
        description="Names of skills from memory to apply",
    )
    constraints_for_designer: list[str] = Field(
        default_factory=list,
        description="Concrete placement/arrangement constraints for the designer agent",
    )
    checks_for_critic: list[str] = Field(
        default_factory=list,
        description="Verification items the critic should evaluate",
    )
    failure_patterns_to_avoid: list[str] = Field(
        default_factory=list,
        description="Known failure patterns retrieved from memory to explicitly avoid",
    )

    def to_injection_text(self) -> str:
        """Format StageBrief as a compact text block for prompt injection."""
        lines = [
            f"=== SceneExpert Stage Brief: {self.stage} ===",
            f"Objective: {self.stage_objective}",
            (
                "Priority: The original user task is authoritative. This brief is "
                "advisory and must not contradict explicit object, position, or facing "
                "relations from that task."
            ),
        ]
        if self.constraints_for_designer:
            lines.append("Designer constraints:")
            lines.extend(f"  - {c}" for c in self.constraints_for_designer)
        if self.failure_patterns_to_avoid:
            lines.append("Known failure patterns to avoid:")
            lines.extend(f"  - {p}" for p in self.failure_patterns_to_avoid)
        if self.checks_for_critic:
            lines.append("Critic should verify:")
            lines.extend(f"  - {c}" for c in self.checks_for_critic)
        if self.recommended_skills:
            lines.append(f"Recommended skills: {', '.join(self.recommended_skills)}")
        lines.append("=== End Stage Brief ===")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Verifier outputs
# ---------------------------------------------------------------------------


class VerifyIssue(BaseModel):
    issue_type: str  # e.g., "unreachable", "missing_object", "overcrowded"
    object_name: str = ""
    description: str = ""
    constraint_id: str = ""
    relation: str = ""
    subject_ids: list[str] = Field(default_factory=list)
    target_ids: list[str] = Field(default_factory=list)
    metric: str = ""
    scoring_tier: str = ""
    repair_strategy: str = ""
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class StageVerifyReport(BaseModel):
    """Verification result after a single SceneSmith stage."""

    stage: str
    pass_stage: bool
    scores: dict[str, float] = Field(
        default_factory=dict,
        description="Backward-compatible alias of visual_scores",
    )
    visual_scores: dict[str, float] = Field(default_factory=dict)
    rule_scores: dict[str, float] = Field(default_factory=dict)
    issues: list[VerifyIssue] = Field(default_factory=list)
    informational_issues: list[VerifyIssue] = Field(default_factory=list)
    repair_suggestions: list[str] = Field(default_factory=list)
    critique_summary: str = Field(
        default="",
        description="Full critic summary text from SceneSmith scores.yaml — richest signal for memory",
    )
    score_source: str = "unknown"
    vlm_scoring_performed: bool = False
    hard_check_report: dict = Field(default_factory=dict)
    runtime_repair_events: list[str] = Field(default_factory=list)


class FullVerifyReport(BaseModel):
    """Final whole-scene verification result."""

    semantic_score: float = 0.0
    aesthetic_score: float = 0.0
    plausibility_score: float = 0.0
    style_consistency: float = 0.0
    collision_free_rate: float = 0.0
    stability_score: float = 0.0
    walkable_area_ratio: float = 0.0
    reachability_score: float = 0.0
    support_relation_accuracy: float = 0.0
    overall_score: float = 0.0
    deterministic_pass: bool = False
    pass_scene: bool = False
    expected_stages: list[str] = Field(default_factory=list)
    completed_stages: list[str] = Field(default_factory=list)
    missing_stages: list[str] = Field(default_factory=list)
    outcome_status: str = "COMPLETE"
    degraded_reasons: list[str] = Field(default_factory=list)
    measured_metrics: dict[str, bool] = Field(default_factory=dict)
    metric_sources: dict[str, str] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Repair Controller
# ---------------------------------------------------------------------------


class RepairResult(BaseModel):
    """Outcome of a repair attempt."""

    repair_type: str  # "local_repair", "stage_regeneration", "rollback", "skipped"
    repair_owner: str = "scene_expert_repair_controller"
    execution_status: str = "planned"
    failure_type: str = ""
    repair_action: str = ""
    repair_verified: bool = False
    new_scene_state: str = ""


# ---------------------------------------------------------------------------
# Trace entry (per stage)
# ---------------------------------------------------------------------------


class StageCost(BaseModel):
    qwen_calls: int = 0
    stage_time_sec: float = 0.0


class StageExecutionEvidence(BaseModel):
    """Auditable proof that optional guidance reached a stage agent."""

    task_spec_source: str = "unknown"
    stage_brief_source: str = "unknown"
    retrieved_memory_ids: list[str] = Field(default_factory=list)
    context_bundle_hash: str = ""
    injected_brief_hash: str = ""
    injected_memory_hash: str = ""
    designer_prompt_hash: str = ""
    designer_prompt_contains_brief: bool = False
    designer_prompt_contains_memory: bool = False
    placement_reference_injected: bool = False
    degraded: bool = False


class StageTraceEntry(BaseModel):
    stage: str
    memory_pack: MemoryPack
    relation_context: StageRelationContext | None = None
    planner_trace: dict[str, Any] = Field(default_factory=dict)
    stage_brief: StageBrief | None = None
    scene_state_path: str = ""
    verify_report: StageVerifyReport | None = None
    repair_actions: list[RepairResult] = Field(default_factory=list)
    cost: StageCost = Field(default_factory=StageCost)
    execution_evidence: StageExecutionEvidence = Field(
        default_factory=StageExecutionEvidence
    )
