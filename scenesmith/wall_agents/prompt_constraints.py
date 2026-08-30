"""Helpers for extracting deterministic wall-agent prompt constraints."""

import re

from typing import Any

_EXPLICIT_WALL_DISPLAY_PATTERN = re.compile(
    r"(?:\b(?:wall[- ]mounted|mounted|hung|hanging)\s+"
    r"(?:flat[- ]screen\s+)?(?:tv|television|monitor|screen|display)\b)"
    r"|(?:\b(?:tv|television|monitor|screen|display)\s+"
    r"(?:is\s+)?(?:on|above|against)\s+(?:the\s+)?"
    r"(?:opposite\s+)?wall\b)"
    r"|(?:\b(?:tv|television|monitor|screen|display)\s+"
    r"(?:is\s+)?(?:mounted|hung|hanging)\s+(?:on|against)\s+"
    r"(?:the\s+)?(?:opposite\s+)?wall\b)",
    re.IGNORECASE,
)
_EXPLICIT_MOUNT_VERB_PATTERN = re.compile(
    r"\b(wall[- ]mounted|mounted|hung|hanging)\b", re.IGNORECASE
)
_MEDIA_FURNITURE_PATTERN = re.compile(
    r"\b(tv stand|television stand|media console|media cabinet|entertainment center)\b",
    re.IGNORECASE,
)
_MEDIA_GROUP_ON_WALL_PATTERN = re.compile(
    r"\b(?:tv stand|television stand|media console|media cabinet|entertainment center)"
    r"\s+and\s+(?:a\s+|an\s+|the\s+)?(?:tv|television|display)\s+"
    r"(?:is\s+)?(?:on|against)\s+(?:the\s+)?(?:opposite\s+)?wall\b",
    re.IGNORECASE,
)
_NO_EXPLICIT_WALL_REQUIREMENTS = (
    "- No explicit wall-object obligations were extracted from the prompt. "
    "This is not an empty-stage signal: use the room type, style, available wall "
    "area, openings, and existing furniture to design appropriate optional wall "
    "objects while preserving objects allocated to other stages."
)


def _task_spec_values(task_spec: Any, field: str) -> list[str]:
    if task_spec is None:
        return []
    values = (
        task_spec.get(field, [])
        if isinstance(task_spec, dict)
        else getattr(task_spec, field, [])
    )
    return [str(value) for value in values or []]


def _media_display_category(obj: Any) -> str | None:
    metadata = getattr(obj, "metadata", None) or {}
    values = (
        metadata.get("semantic_name") if isinstance(metadata, dict) else None,
        getattr(obj, "name", None),
        getattr(obj, "description", None),
        getattr(obj, "object_id", None),
    )
    identity = " ".join(str(value or "") for value in values).lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", identity)
    if any(token in normalized for token in ("tv_stand", "media_console", "cabinet")):
        return None
    if re.search(r"(?:^|_)(?:tv|television)(?:_|$)", normalized):
        return "television"
    if re.search(r"(?:^|_)(?:display|screen)(?:_|$)", normalized):
        return "display"
    return None


def _object_type_value(obj: Any) -> str:
    value = getattr(obj, "object_type", "")
    return str(getattr(value, "value", value)).lower()


def _task_spec_media_count(scene: Any) -> int | None:
    task_spec = getattr(scene, "scene_expert_task_spec", None)
    if task_spec is None:
        return None
    names: list[str] = []
    for field in ("required_large_objects", "required_wall_objects"):
        values = (
            task_spec.get(field, [])
            if isinstance(task_spec, dict)
            else getattr(task_spec, field, [])
        )
        names.extend(str(value) for value in values or [])
    return sum(
        1
        for name in names
        if re.search(r"\b(?:tv|television|display|screen)\b", name, re.IGNORECASE)
        and not _MEDIA_FURNITURE_PATTERN.search(name)
    )


def converge_cross_stage_media_inventory(
    scene: Any, required_wall_constraints: str
) -> list[str]:
    """Remove a cross-stage duplicate display while preserving prompt ownership.

    Furniture and wall agents can independently realize the same television.
    When the TaskCompiler requests only one display, retain the wall instance
    only for explicit mounting language; otherwise retain the furniture-stage
    instance. Multiple explicitly requested displays are left untouched.
    """
    objects = list(getattr(scene, "objects", {}).values())
    furniture = [
        obj
        for obj in objects
        if _object_type_value(obj) == "furniture"
        and _media_display_category(obj) is not None
    ]
    wall = [
        obj
        for obj in objects
        if _object_type_value(obj) == "wall_mounted"
        and _media_display_category(obj) is not None
    ]
    if not furniture or not wall:
        return []
    requested_count = _task_spec_media_count(scene)
    if requested_count is not None and requested_count > 1:
        return []

    explicit_wall = "REQUIRED media display" in required_wall_constraints
    duplicates = furniture if explicit_wall else wall
    removed: list[str] = []
    for obj in duplicates:
        object_id = getattr(obj, "object_id", None)
        if object_id is None:
            continue
        scene.remove_object(object_id)
        removed.append(str(object_id))
    return removed


def build_required_wall_object_constraints(
    room_description: str, *, task_spec: Any | None = None
) -> str:
    """Extract explicit wall-object obligations from the room prompt."""
    normalized = " ".join(room_description.split())
    lower_text = normalized.lower()
    requirements: list[str] = []

    # When SceneExpert has compiled the prompt, its inventory is the sole source
    # of hard wall obligations. It is a minimum rather than an exhaustive design
    # list; optional same-stage decoration remains under the designer policy.
    required_wall_objects = _task_spec_values(task_spec, "required_wall_objects")
    if task_spec is not None and not required_wall_objects:
        return _NO_EXPLICIT_WALL_REQUIREMENTS

    has_media_furniture = bool(_MEDIA_FURNITURE_PATTERN.search(normalized))
    has_explicit_mount_verb = bool(_EXPLICIT_MOUNT_VERB_PATTERN.search(normalized))
    has_explicit_wall_display = bool(_EXPLICIT_WALL_DISPLAY_PATTERN.search(normalized))
    # "TV stand and television on the opposite wall" locates an entertainment
    # group at a wall; it does not say the television itself is wall-mounted.
    # A direct mounting verb remains authoritative even when a stand is named.
    if _MEDIA_GROUP_ON_WALL_PATTERN.search(normalized) and not has_explicit_mount_verb:
        has_explicit_wall_display = False

    # Do not promote a desktop monitor to a wall requirement merely because the
    # prompt also mentions a back wall. TVs are explicit; other displays require
    # direct wall-placement language.
    if has_explicit_wall_display:
        relation_bits: list[str] = []
        if "opposite wall" in lower_text:
            relation_bits.append("use the opposite wall called out in the prompt")
        if has_media_furniture:
            relation_bits.append(
                "place it on the wall containing the TV stand/media console, centered above that support"
            )
        relation_bits.append("face the sofa/seating area when one is specified")
        relation_suffix = f" ({'; '.join(relation_bits)})" if relation_bits else ""
        requirements.append(
            "- REQUIRED media display: place a wall-mounted television/display"
            f"{relation_suffix}. Do not defer it to manipulands or move it to an "
            "arbitrary side wall to avoid a window. If an existing window overlaps "
            "the required support centerline, call list_windows and repair that exact "
            "window (resize first, then move, then remove only if necessary) before "
            "placing or aligning the display. Never leave the display offset from its "
            "support to preserve the window."
        )

    if not requirements:
        if required_wall_objects:
            return (
                "- REQUIRED wall objects from the TaskCompiler inventory: "
                f"{', '.join(required_wall_objects)}. Complete these first, then "
                "add capacity-appropriate optional wall objects when allowed by "
                "the SceneExpert stage policy. Preserve objects allocated to other "
                "stages."
            )
        return _NO_EXPLICIT_WALL_REQUIREMENTS

    return "\n".join(requirements)
