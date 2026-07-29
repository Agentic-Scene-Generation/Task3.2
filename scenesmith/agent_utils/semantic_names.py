"""Stable semantic labels shared by asset selection and critic binding."""

from __future__ import annotations

import re

from typing import Any

from scenesmith.agent_utils.room import ObjectType


def normalize_semantic_name(value: object) -> str:
    """Return a conservative snake_case label, or an empty string."""
    text = str(value or "").strip().lower()
    # Possessive role labels describe the same object category (for example,
    # "teacher's desk" and "teacher desk").  Keeping the possessive ``s``
    # creates a separate vocabulary term that no selector or repair template
    # owns, so discard it before building the stable snake_case label.
    text = re.sub(r"(?<=[a-z])['\u2019]s\b", "", text)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def semantic_name_candidates_for_request(
    task_spec: Any,
    short_names: list[str],
    object_type: ObjectType,
) -> list[list[str]]:
    """Build per-asset VLM labels from TaskCompiler-owned vocabulary.

    The original short name is always retained as the backwards-compatible
    fallback.  The task spec can be either a Pydantic object or its serialized
    dictionary because checkpoints retain the latter.
    """
    stage_key = {
        ObjectType.FURNITURE: "required_large_objects",
        ObjectType.WALL_MOUNTED: "required_wall_objects",
        ObjectType.CEILING_MOUNTED: "required_ceiling_objects",
        ObjectType.MANIPULAND: "required_small_objects",
    }.get(object_type)

    def get(key: str, default: Any) -> Any:
        if isinstance(task_spec, dict):
            return task_spec.get(key, default)
        return getattr(task_spec, key, default)

    allowed: list[str] = []
    if stage_key:
        allowed.extend(get(stage_key, []) or [])
    for constraint in get("intent_constraints", []) or []:
        if not isinstance(constraint, dict):
            continue
        for selector_key in ("subjects", "subject", "targets", "target"):
            selector = constraint.get(selector_key)
            if isinstance(selector, dict):
                allowed.append(selector.get("category", ""))

    normalized_allowed = _unique_normalized(allowed)
    return [
        _unique_normalized([*normalized_allowed, short_name])
        or [normalize_semantic_name(short_name)]
        for short_name in short_names
    ]


def _unique_normalized(values: list[object]) -> list[str]:
    result: list[str] = []
    for value in values:
        normalized = normalize_semantic_name(value)
        if normalized and normalized not in result:
            result.append(normalized)
    return result
