"""Cross-stage inventory guards for manipuland placement."""

import re

from typing import Any

from scenesmith.agent_utils.room import ObjectType
from scenesmith.scenebenchmark_critic.intent_contract import (
    intent_contract_constraints_for_scene,
    selected_ids,
)


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


def contract_bound_support_object_ids(scene: Any, target_id: Any) -> list[str]:
    """Return hard-contract objects supported by the current furniture.

    Furniture-stage objects can be physically placed on a support without a
    manipuland ``placement_info`` record. Preserve those objects in the
    manipuland agent's focused context so the critic does not mistake them for
    missing surface items and generate duplicates.
    """
    objects = getattr(scene, "objects", {}) or {}
    object_rows = []
    for object_id, obj in objects.items():
        metadata = getattr(obj, "metadata", None) or {}
        object_type = getattr(obj, "object_type", "")
        semantic_name = (
            metadata.get("semantic_name", "") if isinstance(metadata, dict) else ""
        )
        object_rows.append(
            {
                "id": str(object_id),
                "name": getattr(obj, "name", ""),
                "description": getattr(obj, "description", ""),
                "object_type": getattr(object_type, "value", object_type),
                "category": semantic_name,
                "category_norm": semantic_name,
                "metadata": metadata if isinstance(metadata, dict) else {},
            }
        )

    target_id_text = str(target_id)
    supported_ids: set[str] = set()
    for constraint in intent_contract_constraints_for_scene(scene):
        if str(constraint.get("relation") or "") != "on_top_of":
            continue
        if str(constraint.get("strength") or "hard").lower() != "hard":
            continue
        target_ids = selected_ids(constraint.get("targets"), object_rows)
        if target_id_text not in target_ids:
            continue
        supported_ids.update(selected_ids(constraint.get("subjects"), object_rows))

    supported_ids.discard(target_id_text)
    return sorted(supported_ids)
