"""Structured context bundles for SceneExpert LLM calls.

The bundle is intentionally compact and serializable.  It gives every LLM-based
SceneExpert/SceneSmith agent a shared view of the current task, stage, scene
state, retrieved memory, and unresolved hard issues, while also producing debug
records that make empty model responses inspectable after an ACP run.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from scenesmith.scene_expert.schemas import MemoryPack, SceneTaskSpec, StageBrief


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def compact_text(value: Any, max_chars: int = 1000) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def stable_hash(payload: Any) -> str:
    try:
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    except TypeError:
        encoded = repr(payload)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


class ObjectContext(BaseModel):
    object_id: str
    name: str = ""
    object_type: str = ""
    category: str = ""
    translation: list[float] = Field(default_factory=list)
    yaw_deg: float | None = None
    bbox_min: list[float] | None = None
    bbox_max: list[float] | None = None
    size: list[float] | None = None
    immutable: bool = False


class ForbiddenZone(BaseModel):
    zone_id: str
    zone_type: str
    severity: str = "hard"
    wall: str = ""
    bounds_xy: list[float] = Field(default_factory=list)
    source: str = ""
    clearance_m: float = 0.0

    def overlaps_xy(self, bounds: tuple[Any, Any]) -> float:
        if len(self.bounds_xy) != 4:
            return 0.0
        min_x, min_y, max_x, max_y = [float(v) for v in self.bounds_xy]
        obj_min, obj_max = bounds
        overlap_x = min(max_x, float(obj_max[0])) - max(min_x, float(obj_min[0]))
        overlap_y = min(max_y, float(obj_max[1])) - max(min_y, float(obj_min[1]))
        return max(0.0, overlap_x) * max(0.0, overlap_y)


class LLMCallDebugRecord(BaseModel):
    schema_version: str = "1.0"
    created_at: str = Field(default_factory=utc_now)
    stage: str
    agent_role: str
    event: str
    prompt_chars: int = 0
    prompt_hash: str = ""
    prompt_excerpt: str = ""
    # The full prompt is intentionally truncated in prompt_excerpt. Keep an
    # explicit marker so long critic prompts remain auditable after a replay.
    prompt_contains_scenebenchmark_context: bool = False
    output_chars: int = 0
    output_excerpt: str = ""
    finish_reasons: list[str] = Field(default_factory=list)
    token_usage: dict[str, int] = Field(default_factory=dict)
    raw_response_excerpt: str = ""
    error: str = ""


class StageContextBundle(BaseModel):
    schema_version: str = "1.0"
    created_at: str = Field(default_factory=utc_now)
    stage: str
    agent_role: str = ""
    event: str = ""
    trace_id: str = ""
    scene_id: str = ""
    task_spec: dict[str, Any] = Field(default_factory=dict)
    stage_brief: dict[str, Any] | None = None
    scene_summary: str = ""
    object_table: list[ObjectContext] = Field(default_factory=list)
    forbidden_zones: list[ForbiddenZone] = Field(default_factory=list)
    retrieved_memory: dict[str, Any] = Field(default_factory=dict)
    history_summary: str = ""
    last_hard_issues: list[str] = Field(default_factory=list)
    prompt_profile: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_llm_text(self, max_chars: int = 3200) -> str:
        """Return a concise human-readable context block for agent prompts."""
        lines = [
            f"=== StageContextBundle: {self.stage} / {self.agent_role or 'agent'} ===",
        ]
        if self.task_spec:
            lines.append("Task spec: " + compact_text(self.task_spec, 420))
        if self.scene_summary:
            lines.append("Scene state: " + compact_text(self.scene_summary, 520))
        if self.stage_brief:
            lines.append("Stage brief: " + compact_text(self.stage_brief, 520))
        if self.last_hard_issues:
            lines.append("Unresolved hard issues:")
            lines.extend(
                f"  - {compact_text(issue, 180)}" for issue in self.last_hard_issues[:5]
            )
        if self.forbidden_zones:
            lines.append("Forbidden / clearance zones:")
            for zone in self.forbidden_zones[:8]:
                lines.append(
                    "  - "
                    f"{zone.zone_id}: type={zone.zone_type}, severity={zone.severity}, "
                    f"wall={zone.wall or 'unknown'}, bounds_xy={zone.bounds_xy}"
                )
        if self.object_table:
            lines.append("Current objects:")
            for obj in self.object_table[:24]:
                loc = (
                    f" at ({obj.translation[0]:.2f},{obj.translation[1]:.2f})"
                    if len(obj.translation) >= 2
                    else ""
                )
                size = f" size={obj.size}" if obj.size else ""
                lines.append(
                    f"  - {obj.object_id}: {obj.name or obj.category} "
                    f"type={obj.object_type}{loc}{size}"
                )
        if self.retrieved_memory:
            lines.append(
                "Retrieved memory: " + compact_text(self.retrieved_memory, 520)
            )
        if self.history_summary:
            lines.append(
                "Recent stage history: " + compact_text(self.history_summary, 520)
            )
        lines.append("=== End StageContextBundle ===")
        text = "\n".join(lines)
        return text if len(text) <= max_chars else text[: max_chars - 3] + "..."

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.model_dump(), indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
            newline="\n",
        )
        return path


def _object_category(object_id: str, obj: Any) -> str:
    text = f"{object_id} {getattr(obj, 'name', '')} {getattr(obj, 'description', '')}".lower()
    if "nightstand" in text or "bedside" in text:
        return "nightstand"
    if any(term in text for term in ("wardrobe", "closet", "armoire")):
        return "wardrobe"
    if "bed" in text:
        return "bed"
    if any(term in text for term in ("door", "window", "opening")):
        return "opening"
    return ""


def _yaw_deg(obj: Any) -> float | None:
    try:
        from pydrake.all import RollPitchYaw

        return float(math.degrees(RollPitchYaw(obj.transform.rotation()).yaw_angle()))
    except Exception:
        return None


def _object_context(object_id: str, obj: Any) -> ObjectContext:
    translation: list[float] = []
    try:
        translation = [float(v) for v in obj.transform.translation()]
    except Exception:
        pass
    bbox_min = None
    bbox_max = None
    size = None
    try:
        bounds = obj.compute_world_bounds()
        if bounds is not None:
            world_min, world_max = bounds
            bbox_min = [float(v) for v in world_min]
            bbox_max = [float(v) for v in world_max]
            size = [round(float(world_max[i] - world_min[i]), 4) for i in range(3)]
    except Exception:
        pass
    object_type = getattr(
        getattr(obj, "object_type", ""), "value", getattr(obj, "object_type", "")
    )
    return ObjectContext(
        object_id=str(object_id),
        name=str(getattr(obj, "name", "") or ""),
        object_type=str(object_type or ""),
        category=_object_category(str(object_id), obj),
        translation=translation,
        yaw_deg=_yaw_deg(obj),
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        size=size,
        immutable=bool(getattr(obj, "immutable", False)),
    )


def build_scene_summary(scene: Any | None) -> str:
    if scene is None:
        return ""
    parts: list[str] = []
    room_geometry = getattr(scene, "room_geometry", None)
    if room_geometry is not None:
        width = getattr(room_geometry, "width", None)
        length = getattr(room_geometry, "length", None)
        if width is not None and length is not None:
            parts.append(f"room_size={float(length):.2f}m x {float(width):.2f}m")
        openings = getattr(room_geometry, "openings", []) or []
        if openings:
            opening_bits = []
            for idx, opening in enumerate(openings[:12]):
                wall = getattr(opening, "wall_direction", None)
                wall = getattr(wall, "value", wall)
                typ = getattr(opening, "opening_type", None)
                typ = getattr(typ, "value", typ)
                opening_bits.append(f"{idx}:{typ or 'opening'}@{wall or 'wall'}")
            parts.append("openings=" + ", ".join(opening_bits))
    text = getattr(scene, "text_description", "")
    if text:
        parts.append("description=" + compact_text(text, 700))
    object_count = len(getattr(scene, "objects", {}) or {})
    parts.append(f"object_count={object_count}")
    return "; ".join(parts)


def build_stage_context_bundle(
    *,
    stage: str,
    agent_role: str = "",
    event: str = "",
    task_spec: SceneTaskSpec | None = None,
    stage_brief: StageBrief | None = None,
    scene: Any | None = None,
    memory_pack: MemoryPack | None = None,
    forbidden_zones: list[ForbiddenZone] | None = None,
    history_summary: str = "",
    last_hard_issues: list[str] | None = None,
    prompt: Any = "",
    trace_id: str = "",
    scene_id: str = "",
    metadata: dict[str, Any] | None = None,
) -> StageContextBundle:
    objects: list[ObjectContext] = []
    if scene is not None:
        for object_id, obj in (getattr(scene, "objects", {}) or {}).items():
            try:
                objects.append(_object_context(str(object_id), obj))
            except Exception:
                continue

    retrieved_memory = {}
    if memory_pack is not None:
        retrieved_memory = {
            "success_hints": len(memory_pack.success_hints),
            "failure_hints": len(memory_pack.failure_hints),
            "skills": len(memory_pack.skill_texts),
            "has_placement_reference": bool(memory_pack.placement_reference),
            "success_excerpt": [
                compact_text(x, 180) for x in memory_pack.success_hints[:3]
            ],
            "failure_excerpt": [
                compact_text(x, 180) for x in memory_pack.failure_hints[:3]
            ],
        }
    prompt_text = _stringify_prompt(prompt)
    return StageContextBundle(
        stage=stage,
        agent_role=agent_role,
        event=event,
        trace_id=trace_id,
        scene_id=scene_id,
        task_spec=task_spec.model_dump() if task_spec is not None else {},
        stage_brief=stage_brief.model_dump() if stage_brief is not None else None,
        scene_summary=build_scene_summary(scene),
        object_table=objects,
        forbidden_zones=forbidden_zones or [],
        retrieved_memory=retrieved_memory,
        history_summary=history_summary,
        last_hard_issues=list(last_hard_issues or []),
        prompt_profile={
            "prompt_chars": len(prompt_text),
            "prompt_hash": stable_hash(prompt_text),
            "prompt_excerpt": compact_text(prompt_text, 1200),
        },
        metadata=metadata or {},
    )


def _stringify_prompt(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    try:
        return json.dumps(prompt, ensure_ascii=False, default=str)
    except TypeError:
        return repr(prompt)


def build_llm_call_debug_record(
    *,
    stage: str,
    agent_role: str,
    event: str,
    prompt: Any,
    output: Any = "",
    result: Any = None,
    raw_response: Any = None,
    error: str = "",
) -> LLMCallDebugRecord:
    prompt_text = _stringify_prompt(prompt)
    output_text = _stringify_prompt(output)
    return LLMCallDebugRecord(
        stage=stage,
        agent_role=agent_role,
        event=event,
        prompt_chars=len(prompt_text),
        prompt_hash=stable_hash(prompt_text),
        prompt_excerpt=compact_text(prompt_text, 1800),
        prompt_contains_scenebenchmark_context=(
            "Additional SceneBenchmark geometry critic context" in prompt_text
        ),
        output_chars=len(output_text),
        output_excerpt=compact_text(output_text, 1800),
        finish_reasons=_extract_finish_reasons(result or raw_response),
        token_usage=_extract_token_usage(result),
        raw_response_excerpt=(
            compact_text(_stringify_prompt(raw_response), 2400)
            if raw_response is not None
            else ""
        ),
        error=error,
    )


def _extract_token_usage(result: Any) -> dict[str, int]:
    usage = getattr(getattr(result, "context_wrapper", None), "usage", None)
    if usage is None:
        return {}
    fields = {
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
        "requests": getattr(usage, "requests", None),
    }
    return {k: int(v) for k, v in fields.items() if isinstance(v, int)}


def _extract_finish_reasons(value: Any) -> list[str]:
    reasons: list[str] = []
    if value is None:
        return reasons
    raw_responses = getattr(value, "raw_responses", None) or []
    for response in raw_responses:
        reasons.extend(_extract_finish_reasons(response))
    choices = getattr(value, "choices", None) or []
    for choice in choices:
        reason = getattr(choice, "finish_reason", None)
        if reason:
            reasons.append(str(reason))
    return reasons
