from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

MetricName = Literal[
    "functional_dependency",
    "spatial_accessibility",
    "interaction_clearance",
    "visual_clearance",
]
MetricLabel = Literal["pass", "degraded", "fail", "unknown"]
FunctionalCategoryName = Literal[
    "graspable",
    "supportable",
    "sittable",
    "openable",
    "containable",
    "sleepable",
    "toggleable",
]


class VLMCheckResult(BaseModel):
    check_id: str
    metric: MetricName
    label: MetricLabel
    asset_fact_used: bool | None = None
    asset_fact_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence_conflict: bool | None = None
    reason: str
    blocking_objects: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class FunctionalCategoryPrediction(BaseModel):
    object_id: str
    functional_categories: list[FunctionalCategoryName] = Field(default_factory=list)
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)


class FunctionalDependencyProposal(BaseModel):
    subject_id: str
    target_ids: list[str] = Field(default_factory=list)
    relation_type: str
    expected_use: str
    scoring_tier: str = "core"
    priority: float = Field(default=0.5, ge=0.0, le=1.0)
    reason: str


class FunctionalDependencyProposalSet(BaseModel):
    proposals: list[FunctionalDependencyProposal] = Field(default_factory=list)
