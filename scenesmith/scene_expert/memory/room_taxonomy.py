"""Canonical room taxonomy used by SceneExpert memory transfer.

Room compatibility is intentionally conservative.  A shared lexical token such
as ``room`` is not evidence that a meeting room and a dining room may reuse the
same placement procedure.
"""

from __future__ import annotations

import re

_GENERIC_ROOM_TYPES = frozenset({"", "*", "all", "any", "generic", "room"})

_ROOM_ALIASES: dict[str, frozenset[str]] = {
    "bathroom": frozenset(
        {
            "bathroom",
            "bath_room",
            "restroom",
            "toilet_room",
            "washroom",
            "卫生间",
            "浴室",
        }
    ),
    "bedroom": frozenset(
        {
            "bedroom",
            "bed_room",
            "guest_bedroom",
            "master_bedroom",
            "sleeping_room",
            "卧室",
        }
    ),
    "classroom": frozenset({"classroom", "class_room", "teaching_room", "教室"}),
    "dining_room": frozenset(
        {
            "dining_room",
            "dining_area",
            "dining_space",
            "restaurant_dining_room",
            "餐厅",
        }
    ),
    "kitchen": frozenset({"kitchen", "kitchenette", "cooking_area", "厨房"}),
    "living_room": frozenset(
        {"family_room", "living_room", "lounge", "sitting_room", "客厅"}
    ),
    "meeting_room": frozenset(
        {"boardroom", "conference_room", "meeting_room", "meeting_space", "会议室"}
    ),
    "office": frozenset({"home_office", "office", "work_room", "workspace", "办公室"}),
    "study": frozenset({"library_study", "study", "study_room", "书房"}),
}

_ALIAS_TO_CANONICAL = {
    alias: canonical
    for canonical, aliases in _ROOM_ALIASES.items()
    for alias in aliases
}

# These are the only transfers allowed after canonicalization. The disjoint
# entries are deliberate: meeting-room and dining-room procedures have
# different seating/topology contracts even though both labels contain "room".
_EXPLICIT_COMPATIBILITY: dict[str, frozenset[str]] = {
    canonical: frozenset({canonical}) for canonical in _ROOM_ALIASES
}


def normalize_room_label(value: str) -> str:
    """Normalize spelling without inferring semantic compatibility."""
    return re.sub(r"[^\w]+", "_", str(value or "").strip().casefold()).strip("_")


def canonical_room_type(value: str) -> str:
    """Return a stable room type while preserving unknown specific labels."""
    normalized = normalize_room_label(value)
    if normalized in _GENERIC_ROOM_TYPES:
        return normalized
    return _ALIAS_TO_CANONICAL.get(normalized, normalized)


def room_types_compatible(record_room: str, task_room: str) -> bool:
    """Return whether a room-scoped record may transfer to the current task."""
    record_canonical = canonical_room_type(record_room)
    task_canonical = canonical_room_type(task_room)
    if record_canonical in _GENERIC_ROOM_TYPES or task_canonical in _GENERIC_ROOM_TYPES:
        return True
    allowed = _EXPLICIT_COMPATIBILITY.get(
        record_canonical, frozenset({record_canonical})
    )
    return task_canonical in allowed
