"""Stable semantic labels shared by asset selection and critic binding."""

from __future__ import annotations

import re

from typing import Any

from scenesmith.agent_utils.room import ObjectType
from scenesmith.scenebenchmark_critic.intent_contract import (
    intent_contract_constraints_for_scene,
)


_MEDIA_SUPPORT_ROLES = frozenset(
    {
        "tv_stand",
        "tv_console",
        "media_console",
        "media_center",
        "entertainment_center",
        "entertainment_unit",
    }
)
_INDEPENDENT_DISPLAY_ROLES = frozenset(
    {
        "television",
        "tv",
        "display",
        "screen",
        "computer_display",
        "computer_monitor",
        "monitor",
    }
)


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
    *,
    scene: Any | None = None,
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
    for constraint in intent_contract_constraints_for_scene(scene):
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


def forbidden_semantic_components_for_request(
    task_spec: Any,
    short_names: list[str],
    object_type: ObjectType,
    *,
    scene: Any | None = None,
) -> list[list[str]]:
    """Return per-asset components forbidden by independently owned inventory.

    A support and the object it supports are separate physical requests when both
    appear as independent TaskCompiler/intent roles.  Asset retrieval must not
    silently collapse those roles into one composite mesh.
    """
    if object_type != ObjectType.FURNITURE:
        return [[] for _ in short_names]

    def get(key: str, default: Any) -> Any:
        if isinstance(task_spec, dict):
            return task_spec.get(key, default)
        return getattr(task_spec, key, default)

    roles: list[object] = []
    for field in (
        "required_large_objects",
        "required_wall_objects",
        "required_ceiling_objects",
        "required_small_objects",
    ):
        roles.extend(get(field, []) or [])
    for constraint in intent_contract_constraints_for_scene(scene):
        if not isinstance(constraint, dict):
            continue
        for selector_key in ("subjects", "subject", "targets", "target"):
            selector = constraint.get(selector_key)
            if isinstance(selector, dict):
                roles.append(selector.get("category", ""))

    normalized_roles = set(_unique_normalized(roles))
    has_separate_media_roles = bool(
        normalized_roles & _MEDIA_SUPPORT_ROLES
        and normalized_roles & _INDEPENDENT_DISPLAY_ROLES
    )
    result: list[list[str]] = []
    for short_name in short_names:
        normalized = normalize_semantic_name(short_name)
        is_support = normalized in _MEDIA_SUPPORT_ROLES or any(
            normalized.startswith(f"{role}_") for role in _MEDIA_SUPPORT_ROLES
        )
        result.append(["television"] if has_separate_media_roles and is_support else [])
    return result


def _unique_normalized(values: list[object]) -> list[str]:
    result: list[str] = []
    for value in values:
        normalized = normalize_semantic_name(value)
        if normalized and normalized not in result:
            result.append(normalized)
    return result
