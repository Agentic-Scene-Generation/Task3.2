"""Deterministic retrieval text builders for SceneExpert memory records.

The MemoryWriter is asked to emit ``embedding_text``, but index construction must
not depend on the model doing that perfectly. These helpers provide stable,
low-noise fallback text for success, failure, and skill records.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from scenesmith.scene_expert.memory.schemas import FailureCase, Skill, SuccessCase

MemoryRecord = SuccessCase | FailureCase | Skill


def _clean(value: object) -> str:
    """Return a compact one-line string for retrieval text fields."""
    if value is None:
        return ""
    text = str(value).strip()
    return " ".join(text.split())


def _join_items(items: Sequence[object]) -> str:
    return ", ".join(text for item in items if (text := _clean(item)))


def _join_scores(scores: Mapping[str, float]) -> str:
    parts: list[str] = []
    for key, value in scores.items():
        try:
            parts.append(f"{key} {float(value):.2f}")
        except (TypeError, ValueError):
            parts.append(f"{key} {_clean(value)}")
    return ", ".join(parts)


def _append_line(lines: list[str], key: str, value: object) -> None:
    text = _clean(value)
    if text:
        lines.append(f"{key}={text}")


def _append_list(lines: list[str], key: str, items: Sequence[object]) -> None:
    text = _join_items(items)
    if text:
        lines.append(f"{key}={text}")


def _build_success_text(record: SuccessCase) -> str:
    lines = [
        "memory_type=success",
        f"stage={record.stage}",
        f"room_type={record.room_type}",
    ]
    _append_line(lines, "style", record.style)
    _append_list(lines, "required_objects", record.required_objects or record.task_signature)
    _append_list(lines, "functional_zones", record.functional_zones)
    _append_line(lines, "scene_summary", record.scene_summary)
    _append_list(lines, "task_signature", record.task_signature)
    _append_list(lines, "success_pattern", record.successful_pattern)
    _append_list(lines, "positive_guidance", record.positive_guidance)
    if record.scores:
        lines.append(f"scores={_join_scores(record.scores)}")
    _append_line(lines, "quality_score", f"{record.quality_score:.2f}")
    _append_line(lines, "confidence", f"{record.confidence:.2f}")
    return "\n".join(lines)


def _build_failure_text(record: FailureCase) -> str:
    lines = [
        "memory_type=failure",
        f"stage={record.stage}",
        f"room_type={record.room_type}",
        f"scope={record.scope}",
    ]
    _append_line(lines, "object", record.object)
    _append_list(lines, "required_objects", record.required_objects)
    _append_list(lines, "functional_zones", record.functional_zones)
    _append_line(lines, "scene_summary", record.scene_summary)
    _append_line(lines, "failure_type", record.failure_type)
    _append_line(lines, "bad_pattern", record.bad_pattern)
    _append_line(lines, "failure_reason", record.failure_reason)
    _append_line(lines, "negative_constraint", record.negative_constraint)
    _append_line(lines, "critic_check", record.critic_check)
    _append_line(lines, "repair_action", record.repair_action)
    lines.append(f"repair_verified={str(record.repair_verified).lower()}")
    lines.append(f"is_deterministic={str(record.is_deterministic).lower()}")
    lines.append(f"repeat_count={record.repeat_count}")
    _append_line(lines, "quality_score", f"{record.quality_score:.2f}")
    _append_line(lines, "confidence", f"{record.confidence:.2f}")
    return "\n".join(lines)


def _build_skill_text(record: Skill) -> str:
    lines = [
        "memory_type=skill",
        f"stage={record.stage}",
        f"skill={record.skill_name}",
    ]
    _append_line(lines, "room_type", record.room_type)
    _append_line(lines, "style", record.style)
    _append_list(lines, "room_types", record.room_types)
    _append_list(lines, "required_objects", record.required_objects)
    _append_list(lines, "functional_zones", record.functional_zones)
    _append_line(lines, "scene_summary", record.scene_summary)
    _append_list(lines, "preconditions", record.preconditions)
    _append_list(lines, "procedure", record.procedure)
    _append_list(lines, "failure_avoidance", record.failure_avoidance)
    _append_list(lines, "postconditions", record.postconditions)
    _append_line(lines, "success_rate", f"{record.success_rate:.2f}")
    _append_line(lines, "quality_score", f"{record.quality_score:.2f}")
    _append_line(lines, "confidence", f"{record.confidence:.2f}")
    return "\n".join(lines)


def build_embedding_text(record: MemoryRecord) -> str:
    """Build structured retrieval text for a memory record."""
    if isinstance(record, SuccessCase):
        return _build_success_text(record)
    if isinstance(record, FailureCase):
        return _build_failure_text(record)
    if isinstance(record, Skill):
        return _build_skill_text(record)
    raise TypeError(f"Unsupported memory record type: {type(record)!r}")
