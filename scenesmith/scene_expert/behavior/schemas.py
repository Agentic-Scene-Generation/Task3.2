"""Typed contracts for deterministic resident behavior planning."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

BehaviorStage = Literal["furniture", "wall_mounted", "ceiling_mounted", "manipuland"]


class PersonaSpec(BaseModel):
    """Resident persona used by the template planner."""

    model_config = ConfigDict(extra="forbid")

    name: str
    age: int = Field(ge=0)
    job: str
    health: str
    traits: list[str] = Field(default_factory=list)
    description: str


class ActionStep(BaseModel):
    """One deterministic action within a scheduled activity."""

    model_config = ConfigDict(extra="forbid")

    step: int = Field(ge=1)
    action: str
    object: str
    start: str
    end: str
    note: str


class ScheduleActivity(BaseModel):
    """Source-compatible weekly schedule entry."""

    model_config = ConfigDict(extra="forbid")

    activity: str
    start: str
    end: str
    location: str = "at_home"
    room: str | None = None
    detail_level: Literal["fine", "coarse"] = "fine"


class DetailedActivity(BaseModel):
    """Source-compatible room routine entry with action steps."""

    model_config = ConfigDict(extra="forbid")

    activity: str
    start: str
    end: str
    steps: list[ActionStep] = Field(default_factory=list)


class ObjectNeed(BaseModel):
    """Source-compatible object need without Task3.2-only metadata."""

    model_config = ConfigDict(extra="forbid")

    name: str
    role: Literal["furniture", "manipuland"]
    required: bool
    quantity: int = Field(default=1, ge=1)
    support_target: str | None = None
    reason: str
    source_activities: list[str] = Field(default_factory=list)


class AssetNeed(BaseModel):
    """Behavior-derived asset requirement with provenance."""

    model_config = ConfigDict(extra="forbid")

    name: str
    stage: BehaviorStage
    role: Literal["furniture", "manipuland"]
    required: bool
    quantity: int = Field(default=1, ge=1)
    support_target: str | None = None
    reason: str
    source: Literal["explicit_prompt", "behavior_template"]
    source_activities: list[str] = Field(default_factory=list)


class BehaviorRelation(BaseModel):
    """Semantic placement relation implied by the behavior template."""

    model_config = ConfigDict(extra="forbid")

    subject: str
    relation: Literal["on", "facing", "near"]
    object: str
    required: bool
    reason: str


class RoomBehaviorSpec(BaseModel):
    """Behavior, assets, and relations grouped for one room."""

    model_config = ConfigDict(extra="forbid")

    room_type: str
    weekly_schedule: dict[str, list[DetailedActivity]] = Field(default_factory=dict)
    assets_by_stage: dict[BehaviorStage, list[AssetNeed]] = Field(default_factory=dict)
    relations: list[BehaviorRelation] = Field(default_factory=list)


class BehaviorSpec(BaseModel):
    """Complete output of the deterministic behavior template planner."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    scene_prompt: str
    target_rooms: list[str]
    horizon: Literal["week"] = "week"
    persona: PersonaSpec
    weekly_schedule: dict[str, list[ScheduleActivity]] = Field(default_factory=dict)
    detailed_routines: dict[str, dict[str, list[DetailedActivity]]] = Field(
        default_factory=dict
    )
    object_needs: dict[str, list[ObjectNeed]] = Field(default_factory=dict)
    assets_by_room_and_stage: dict[str, dict[BehaviorStage, list[AssetNeed]]] = Field(
        default_factory=dict
    )
    placement_relations: dict[str, list[BehaviorRelation]] = Field(default_factory=dict)
    room_behavior_blocks: dict[str, str] = Field(default_factory=dict)
    enriched_prompt: str = ""
    rooms: list[RoomBehaviorSpec]
    generation_mode: Literal["deterministic_template"] = "deterministic_template"
    persona_generation: Literal["model", "fallback"] = "fallback"
