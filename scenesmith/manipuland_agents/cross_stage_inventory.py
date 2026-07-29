"""Cross-stage inventory guards for manipuland placement."""

import re

from typing import Any

from scenesmith.agent_utils.room import ObjectType


_FLOOR_COVERING_PATTERN = re.compile(
    r"\b(?:area\s+)?(?:rug|carpet|floor\s+mat|doormat|door\s+mat)\b",
    re.IGNORECASE,
)
_SINGLE_FLOOR_COVERING_REQUEST_PATTERN = re.compile(
    r"^\s*(?:(?:required|optional)\s*:\s*)?"
    r"(?:(?:a|an|the)\s+)?"
    r"(?:(?:small|large|round|square|rectangular|area|floor)\s+)*"
    r"(?:rug|carpet|mat|doormat)\s*[.!]?\s*$",
    re.IGNORECASE,
)


def is_floor_covering_text(value: Any) -> bool:
    """Return whether free-form asset identity denotes a floor covering."""
    normalized = re.sub(r"[^a-z0-9]+", " ", str(value or "").lower())
    return _FLOOR_COVERING_PATTERN.search(normalized) is not None


def existing_floor_covering_ids(scene: Any) -> list[str]:
    """Find already-realized rug-like objects across scene stages."""
    found: list[str] = []
    for obj in getattr(scene, "objects", {}).values():
        metadata = getattr(obj, "metadata", None) or {}
        identity = " ".join(
            str(value or "")
            for value in (
                metadata.get("semantic_name") if isinstance(metadata, dict) else None,
                getattr(obj, "name", None),
                getattr(obj, "description", None),
                getattr(obj, "object_id", None),
            )
        )
        if is_floor_covering_text(identity):
            found.append(str(getattr(obj, "object_id", "")))
    return sorted(object_id for object_id in found if object_id)


def is_floor_target(scene: Any, furniture_id: Any) -> bool:
    """Return whether a manipuland assignment targets the room floor."""
    obj = scene.get_object(furniture_id)
    if obj is None:
        return False
    object_type = getattr(obj, "object_type", "")
    return (
        object_type == ObjectType.FLOOR
        or str(getattr(object_type, "value", object_type)).lower() == "floor"
    )


def is_single_floor_covering_request(value: Any) -> bool:
    """Identify assignments whose only requested item is one floor covering."""
    return (
        _SINGLE_FLOOR_COVERING_REQUEST_PATTERN.fullmatch(str(value or "")) is not None
    )


def redundant_floor_covering_request_indices(
    scene: Any,
    furniture_id: Any,
    object_descriptions: list[str],
    short_names: list[str],
) -> list[int]:
    """Return duplicate covering request indices for an existing floor inventory."""
    if not existing_floor_covering_ids(scene) or not is_floor_target(
        scene, furniture_id
    ):
        return []
    return [
        index
        for index, (description, short_name) in enumerate(
            zip(object_descriptions, short_names, strict=False)
        )
        if is_floor_covering_text(description) or is_floor_covering_text(short_name)
    ]
