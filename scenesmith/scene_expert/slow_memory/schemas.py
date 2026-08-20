"""Versioned schemas for provenance-safe SceneExpert slow memory."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

PreferenceVerdict = Literal["accepted", "rejected", "unlabeled"]
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


class TrajectoryRecord(BaseModel):
    """One immutable designer or deterministic-repair observation."""

    schema_version: Literal["sceneexpert.trajectory.v1"] = "sceneexpert.trajectory.v1"
    trajectory_id: str
    created_at: str
    run_id: str
    scene_id: str
    task_id: str
    experiment_signature: str = ""
    config_hash: str = ""
    model_id: str = ""
    stage: str
    agent_role: Literal["designer", "repair"]
    event: str
    context_hash: str
    prompt: str
    response: str
    response_hash: str
    prompt_complete: bool = True
    response_complete: bool = True
    evidence: PreferenceEvidence
    source_refs: list[str] = Field(default_factory=list)
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
        return self


class DPOPreferencePair(BaseModel):
    """TRL-compatible conversational preference pair plus audit metadata."""

    schema_version: Literal["sceneexpert.dpo_pair.v1"] = "sceneexpert.dpo_pair.v1"
    pair_id: str
    prompt: list[dict[str, str]]
    chosen: list[dict[str, str]]
    rejected: list[dict[str, str]]
    context_hash: str
    leakage_group: str
    task_id: str
    stage: str
    agent_role: Literal["designer", "repair"]
    chosen_trajectory_id: str
    rejected_trajectory_id: str
    chosen_evidence: PreferenceEvidence
    rejected_evidence: PreferenceEvidence
    quality_margin: float
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
        return self
