"""Pydantic data schemas for SceneExpert MVP.

All inter-module data contracts are defined here to ensure
type safety and easy JSON serialization across the pipeline.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# TaskCompiler output
# ---------------------------------------------------------------------------


class SceneTaskSpec(BaseModel):
    """Structured scene requirements extracted from a raw text prompt."""

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
    max_planner_turns: int = 4
    max_designer_turns: int = 12
    max_critic_turns: int = 6
    max_wall_clock_seconds: float = 0.0
    critic_reserve_fraction: float = 0.25
    final_critic_reserve_fraction: float = 0.10
    fallback_reserve_fraction: float = 0.10
    finalization_reserve_fraction: float = 0.05
    max_asset_requests: int = 0
    max_optional_object_families: int = 0
    max_assets_per_request: int = 0
    max_semantic_retries_per_family: int = 2


class HarnessContext(BaseModel):
    """All inputs the Harness assembles before executing a SceneSmith stage."""

    stage: str
    task_spec: SceneTaskSpec
    memory_pack: MemoryPack
    stage_brief: "StageBrief | None" = None
    stage_budget: StageBudget = Field(default_factory=StageBudget)
    allowed_scene_smith_stage: str = ""

    model_config = {"arbitrary_types_allowed": True}


# ---------------------------------------------------------------------------
# Global Planner output
# ---------------------------------------------------------------------------


class StageBrief(BaseModel):
    """Expert planning hint generated by the Global Planner for one stage."""

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


class StageVerifyReport(BaseModel):
    """Verification result after a single SceneSmith stage."""

    stage: str
    pass_stage: bool
    scores: dict[str, float] = Field(
        default_factory=dict,
        description="Scores 0-1 for semantic, aesthetic, physics, interaction",
    )
    issues: list[VerifyIssue] = Field(default_factory=list)
    repair_suggestions: list[str] = Field(default_factory=list)
    critique_summary: str = Field(
        default="",
        description="Full critic summary text from SceneSmith scores.yaml — richest signal for memory",
    )
    score_source: str = Field(
        default="unknown",
        description=(
            "Origin of numeric scores: vlm_critic, deterministic_hard_check, "
            "critic_fallback, unavailable, or unknown"
        ),
    )
    vlm_scoring_performed: bool = False
    hard_check_report: dict = Field(
        default_factory=dict,
        description="Deterministic validation provenance kept separate from VLM scores",
    )


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
    pass_scene: bool = False


# ---------------------------------------------------------------------------
# Repair Controller
# ---------------------------------------------------------------------------


class RepairResult(BaseModel):
    """Outcome of a repair attempt."""

    repair_type: str  # "local_repair", "stage_regeneration", "rollback", "skipped"
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
    """Auditable proof that SceneExpert inputs reached a stage agent."""

    task_spec_source: str = "unknown"
    stage_brief_source: str = "unknown"
    retrieved_memory_ids: list[str] = Field(default_factory=list)
    context_bundle_hash: str = ""
    injected_brief_hash: str = ""
    designer_prompt_hash: str = ""
    designer_prompt_contains_brief: bool = False
    degraded: bool = False


class StageTraceEntry(BaseModel):
    stage: str
    memory_pack: MemoryPack
    stage_brief: StageBrief | None = None
    scene_state_path: str = ""
    verify_report: StageVerifyReport | None = None
    repair_actions: list[RepairResult] = Field(default_factory=list)
    cost: StageCost = Field(default_factory=StageCost)
    execution_evidence: StageExecutionEvidence = Field(
        default_factory=StageExecutionEvidence
    )
