"""Merge behavior-template assets into SceneExpert task inventory."""

from __future__ import annotations

from collections import Counter

from scenesmith.scene_expert.behavior.schemas import BehaviorSpec
from scenesmith.scene_expert.schemas import SceneTaskSpec

_STAGE_FIELDS = {
    "furniture": "required_large_objects",
    "wall_mounted": "required_wall_objects",
    "ceiling_mounted": "required_ceiling_objects",
    "manipuland": "required_small_objects",
}

_ALIASES = {
    "couch": "sofa",
    "tv": "television",
    "fridge": "refrigerator",
    "counter": "kitchen_counter",
}


def _key(value: str) -> str:
    normalized = "_".join(str(value).strip().lower().split())
    return _ALIASES.get(normalized, normalized)


def _room_key(value: str) -> str:
    return "".join(
        character
        for character in str(value).strip().lower()
        if character not in {" ", "_", "-"}
    )


def merge_behavior_assets(
    task_spec: SceneTaskSpec,
    behavior_spec: BehaviorSpec,
    *,
    inferred_assets_are_required: bool,
) -> SceneTaskSpec:
    """Merge required assets with explicit inventory taking precedence.

    Quantity is represented by repeated category labels in SceneTaskSpec, so the
    merged quantity is the maximum explicit or behavior-derived count.
    """
    updates: dict[str, list[str]] = {}
    primary_room = _room_key(task_spec.room_type)
    for stage, field in _STAGE_FIELDS.items():
        existing = list(getattr(task_spec, field))
        existing_counts = Counter(_key(item) for item in existing)
        desired: dict[str, tuple[str, int]] = {}
        for room in behavior_spec.rooms:
            if _room_key(room.room_type) != primary_room:
                continue
            for item in room.assets_by_stage.get(stage, []):
                should_merge = item.source == "explicit_prompt" or (
                    inferred_assets_are_required and item.required
                )
                if not should_merge:
                    continue
                category = _key(item.name)
                current = desired.get(category, (item.name, 0))
                desired[category] = (current[0], max(current[1], item.quantity))
        for category, (label, count) in desired.items():
            existing.extend([label] * max(0, count - existing_counts[category]))
        updates[field] = existing

    return task_spec.model_copy(update=updates)
