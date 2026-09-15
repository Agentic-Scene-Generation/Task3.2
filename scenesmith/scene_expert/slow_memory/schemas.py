"""Versioned schemas for provenance-safe SceneExpert slow memory."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

PreferenceVerdict = Literal["accepted", "rejected", "unlabeled"]
AgentRole = Literal[
    "designer",
    "critic",
    "repair",
    "planner",
    "task_compiler",
    "global_planner",
]
PreferenceTaskType = Literal[
    "designer_initial",
    "designer_repair",
    "critic_advice",
    "deterministic_repair",
    "legacy",
]
EvidenceKind = Literal[
    "critic",
    "deterministic",
    "critic_and_deterministic",
    "none",
]


class PreferenceEvidence(BaseModel):
    """Authoritative evidence attached to one observed response."""

    evidence_id: str
    kind: EvidenceKind = "none"
    verdict: PreferenceVerdict = "unlabeled"
    source: str = ""
    authoritative: bool = False
    quality_score: float | None = Field(default=None, ge=0.0, le=2.0)
    report_ref: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


class TrajectoryOutcome(BaseModel):
    """Label-only execution evidence for one observed policy decision.

    These fields are intentionally kept outside the model-visible prompt.  They
    support lexicographic curation where transport validity and hard geometry
    dominate visual or stylistic scores.
    """

    execution_complete: bool | None = None
    tool_call_valid: bool | None = None
    stage_passed: bool | None = None
    hard_passed: bool | None = None
    hard_violation_count: int | None = Field(default=None, ge=0)
    new_hard_violation_count: int | None = Field(default=None, ge=0)
    resolved_constraint_ids: list[str] = Field(default_factory=list)
    introduced_constraint_ids: list[str] = Field(default_factory=list)
    relation_satisfaction: float | None = Field(default=None, ge=0.0, le=1.0)
    deterministic_score: float | None = Field(default=None, ge=0.0)
    visual_score: float | None = Field(default=None, ge=0.0)
    tool_call_count: int | None = Field(default=None, ge=0)
    latency_sec: float | None = Field(default=None, ge=0.0)
    score_vector: dict[str, float] = Field(default_factory=dict)
    issue_ids: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    causal_link_verified: bool = False

    def preference_key(self) -> tuple[float, ...]:
        """Return a stable hard-first ranking key for offline curation."""

        def truth(value: bool | None) -> float:
            return 1.0 if value is True else (0.0 if value is False else -1.0)

        def value_or(value: float | int | None, default: float) -> float:
            return float(value) if value is not None else default

        return (
            truth(self.execution_complete),
            truth(self.tool_call_valid),
            truth(self.hard_passed),
            -value_or(self.new_hard_violation_count, 1_000_000.0),
            -value_or(self.hard_violation_count, 1_000_000.0),
            truth(self.stage_passed),
            value_or(self.relation_satisfaction, -1.0),
            value_or(self.deterministic_score, -1.0),
            value_or(self.visual_score, -1.0),
            -value_or(self.tool_call_count, 1_000_000.0),
            -value_or(self.latency_sec, 1_000_000.0),
        )


class TrajectoryRecord(BaseModel):
    """One immutable, replay-oriented SceneSmith policy observation.

    ``prompt`` and ``response`` retain the v1 text view.  V2 additionally keeps
    exact conversational messages, tool calls, spatial state, media references,
    and label-only outcomes so a converter never has to reconstruct behavior
    from a final natural-language summary.
    """

    schema_version: Literal[
        "sceneexpert.trajectory.v1", "sceneexpert.trajectory.v2"
    ] = "sceneexpert.trajectory.v2"
    trajectory_id: str
    created_at: str
    run_id: str
    scene_id: str
    task_id: str
    experiment_signature: str = ""
    config_hash: str = ""
    model_id: str = ""
    stage: str
    agent_role: AgentRole
    event: str
    task_type: PreferenceTaskType = "legacy"
    scenario_family_id: str = ""
    context_hash: str
    prompt: str
    response: str
    response_hash: str
    prompt_complete: bool = True
    response_complete: bool = True
    evidence: PreferenceEvidence
    source_refs: list[str] = Field(default_factory=list)
    messages: list[dict[str, Any]] = Field(default_factory=list)
    completion_messages: list[dict[str, Any]] = Field(default_factory=list)
    tools: list[dict[str, Any]] = Field(default_factory=list)
    image_refs: list[dict[str, Any]] = Field(default_factory=list)
    spatial_context: dict[str, Any] = Field(default_factory=dict)
    action_trace: list[dict[str, Any]] = Field(default_factory=list)
    outcome: TrajectoryOutcome = Field(default_factory=TrajectoryOutcome)
    provenance: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _require_trainable_payload(self) -> "TrajectoryRecord":
        if not self.prompt.strip():
            raise ValueError("trajectory prompt must be non-empty")
        if not self.response.strip():
            raise ValueError("trajectory response must be non-empty")
        if self.evidence.authoritative and not self.evidence.report_ref:
            raise ValueError("authoritative evidence must reference its report")
        if self.evidence.authoritative and not self.source_refs:
            raise ValueError("authoritative trajectory must retain source references")
        if self.completion_messages and not any(
            message.get("role") == "assistant"
            for message in self.completion_messages
            if isinstance(message, dict)
        ):
            raise ValueError("trajectory completion must contain an assistant decision")
        return self


class DPOPreferencePair(BaseModel):
    """TRL-compatible tool/VLM preference pair plus audit metadata."""

    schema_version: Literal["sceneexpert.dpo_pair.v1", "sceneexpert.dpo_pair.v2"] = (
        "sceneexpert.dpo_pair.v2"
    )
    pair_id: str
    prompt: list[dict[str, Any]]
    chosen: list[dict[str, Any]]
    rejected: list[dict[str, Any]]
    tools: list[dict[str, Any]] = Field(default_factory=list)
    images: list[str] = Field(default_factory=list)
    context_hash: str
    leakage_group: str
    task_id: str
    stage: str
    agent_role: AgentRole
    task_type: PreferenceTaskType = "legacy"
    chosen_trajectory_id: str
    rejected_trajectory_id: str
    chosen_evidence: PreferenceEvidence
    rejected_evidence: PreferenceEvidence
    chosen_outcome: TrajectoryOutcome = Field(default_factory=TrajectoryOutcome)
    rejected_outcome: TrajectoryOutcome = Field(default_factory=TrajectoryOutcome)
    quality_margin: float = Field(gt=0.0)
    spatial_context: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_preference(self) -> "DPOPreferencePair":
        if self.chosen == self.rejected:
            raise ValueError("chosen and rejected responses must differ")
        if self.chosen_evidence.verdict != "accepted":
            raise ValueError("chosen trajectory is not accepted")
        if self.rejected_evidence.verdict != "rejected":
            raise ValueError("rejected trajectory is not rejected")
        if not self.chosen_evidence.authoritative:
            raise ValueError("chosen trajectory lacks authoritative evidence")
        if not self.rejected_evidence.authoritative:
            raise ValueError("rejected trajectory lacks authoritative evidence")
        if not self.prompt or not any(
            message.get("role") == "user"
            for message in self.prompt
            if isinstance(message, dict)
        ):
            raise ValueError("preference prompt must contain a user message")
        for name, completion in (("chosen", self.chosen), ("rejected", self.rejected)):
            if not any(
                message.get("role") == "assistant"
                for message in completion
                if isinstance(message, dict)
            ):
                raise ValueError(f"{name} completion must contain an assistant message")
        for tool in self.tools:
            function = tool.get("function") if isinstance(tool, dict) else None
            if (
                not isinstance(function, dict)
                or not str(function.get("name") or "").strip()
            ):
                raise ValueError("every tool must use a named function JSON schema")
        return self
