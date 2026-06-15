"""Hybrid scoring helpers for SceneExpert memory retrieval."""

from __future__ import annotations

import time

from dataclasses import dataclass

from scenesmith.scene_expert.memory.retriever import _tokenize
from scenesmith.scene_expert.memory.schemas import FailureCase, Skill, SuccessCase
from scenesmith.scene_expert.schemas import SceneTaskSpec

MemoryRecord = SuccessCase | FailureCase | Skill


@dataclass(frozen=True)
class HybridScoreWeights:
    embedding_similarity: float = 0.45
    object_overlap: float = 0.20
    room_stage_match: float = 0.15
    memory_quality_score: float = 0.10
    recency_or_verified: float = 0.10


def task_required_objects(task_spec: SceneTaskSpec, stage: str | None = None) -> list[str]:
    """Return required objects, optionally focused on the current stage."""
    by_stage = {
        "floor_plan": task_spec.required_large_objects,
        "furniture": task_spec.required_large_objects,
        "wall_mounted": task_spec.required_wall_objects,
        "ceiling_mounted": task_spec.required_ceiling_objects,
        "manipuland": task_spec.required_small_objects,
    }
    if stage in by_stage:
        focused = by_stage[stage]
        if focused:
            return focused
    return (
        task_spec.required_large_objects
        + task_spec.required_wall_objects
        + task_spec.required_ceiling_objects
        + task_spec.required_small_objects
    )


def normalized_token_set(items: list[str] | tuple[str, ...]) -> set[str]:
    tokens: set[str] = set()
    for item in items:
        tokens.update(_tokenize(item))
    return {token for token in tokens if token}


def object_overlap(record_objects: list[str], task_objects: list[str]) -> float:
    """Token-level overlap between memory objects and current task objects."""
    record_tokens = normalized_token_set(record_objects)
    task_tokens = normalized_token_set(task_objects)
    if not record_tokens or not task_tokens:
        return 0.0
    return len(record_tokens & task_tokens) / len(record_tokens | task_tokens)


def room_compatible(record_room: str, task_room: str) -> bool:
    """Return true when room labels are empty or token-compatible."""
    if not record_room or not task_room:
        return True
    record_norm = record_room.lower().replace("_", " ").strip()
    task_norm = task_room.lower().replace("_", " ").strip()
    if record_norm == task_norm:
        return True
    return bool(normalized_token_set([record_room]) & normalized_token_set([task_room]))


def record_required_objects(record: MemoryRecord) -> list[str]:
    if isinstance(record, SuccessCase):
        return record.required_objects or record.task_signature
    if isinstance(record, FailureCase):
        return record.required_objects or ([record.object] if record.object else [])
    return record.required_objects


def record_room_compatible(record: MemoryRecord, task_spec: SceneTaskSpec) -> bool:
    if isinstance(record, Skill):
        rooms = list(record.room_types)
        if record.room_type:
            rooms.append(record.room_type)
        if not rooms:
            return True
        return any(room_compatible(room, task_spec.room_type) for room in rooms)
    return room_compatible(getattr(record, "room_type", ""), task_spec.room_type)


def room_stage_match(record: MemoryRecord, task_spec: SceneTaskSpec, stage: str) -> float:
    score = 0.0
    if record.stage == stage:
        score += 0.6
    if record_room_compatible(record, task_spec):
        score += 0.4
    return min(1.0, score)


def compute_memory_quality(record: MemoryRecord, memory_type: str) -> float:
    """Quality signal independent of query similarity."""
    if memory_type == "success" and isinstance(record, SuccessCase):
        if record.scores:
            return min(
                1.0,
                0.35 * record.scores.get("semantic", 0.5)
                + 0.25 * record.scores.get("aesthetic", 0.5)
                + 0.20 * record.scores.get("interaction", 0.5)
                + 0.20 * record.scores.get("physics", 0.5),
            )
        return record.quality_score

    if memory_type == "failure" and isinstance(record, FailureCase):
        verified = 1.0 if record.repair_verified else 0.4
        deterministic = 1.0 if record.is_deterministic else 0.5
        repeat = min(1.0, max(1, record.repeat_count) / 5.0)
        return 0.45 * verified + 0.35 * deterministic + 0.20 * repeat

    if memory_type == "skill" and isinstance(record, Skill):
        return max(record.success_rate, record.confidence, record.quality_score)

    return getattr(record, "quality_score", 0.5)


def compute_recency_or_verified(record: MemoryRecord, memory_type: str) -> float:
    """Small bonus for verified/reused memory without making recency mandatory."""
    if memory_type == "failure" and isinstance(record, FailureCase):
        if record.repair_verified:
            return 1.0
        if record.is_deterministic:
            return 0.8
        return 0.4

    usage_count = getattr(record, "usage_count", 0)
    usage_bonus = min(1.0, usage_count / 10.0) if usage_count else 0.0
    timestamp = getattr(record, "last_used_at", "") or getattr(record, "created_at", "")
    if not timestamp:
        return max(0.5, usage_bonus)
    try:
        # ISO timestamps sort roughly by recency; parsing all variants is not worth
        # making retrieval fragile, so this bonus is intentionally conservative.
        parsed = time.strptime(timestamp[:19], "%Y-%m-%dT%H:%M:%S")
        age_sec = max(0.0, time.time() - time.mktime(parsed))
        recency = max(0.0, 1.0 - age_sec / (30.0 * 24.0 * 3600.0))
    except Exception:
        recency = 0.5
    return max(usage_bonus, recency)


def hybrid_score(
    embedding_similarity: float,
    record: MemoryRecord,
    task_spec: SceneTaskSpec,
    stage: str,
    memory_type: str,
    weights: HybridScoreWeights = HybridScoreWeights(),
) -> float:
    task_objects = task_required_objects(task_spec, stage)
    obj_score = object_overlap(record_required_objects(record), task_objects)
    stage_room_score = room_stage_match(record, task_spec, stage)
    quality = compute_memory_quality(record, memory_type)
    recency = compute_recency_or_verified(record, memory_type)
    return (
        weights.embedding_similarity * embedding_similarity
        + weights.object_overlap * obj_score
        + weights.room_stage_match * stage_room_score
        + weights.memory_quality_score * quality
        + weights.recency_or_verified * recency
    )
