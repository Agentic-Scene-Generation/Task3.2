"""Helpers for extracting deterministic wall-agent prompt constraints."""

import re

_NAMED_TV_PATTERN = re.compile(r"\b(tv|television)\b", re.IGNORECASE)
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
_WALL_PLACEMENT_PATTERN = re.compile(
    r"\b(wall|wall-mounted|mounted|hung|hanging)\b", re.IGNORECASE
)
_MEDIA_FURNITURE_PATTERN = re.compile(
    r"\b(tv stand|television stand|media console|media cabinet|entertainment center)\b",
    re.IGNORECASE,
)


def build_required_wall_object_constraints(room_description: str) -> str:
    """Extract explicit wall-object obligations from the room prompt."""
    normalized = " ".join(room_description.split())
    lower_text = normalized.lower()
    requirements: list[str] = []

    has_named_tv = bool(_NAMED_TV_PATTERN.search(normalized))
    has_explicit_wall_display = bool(_EXPLICIT_WALL_DISPLAY_PATTERN.search(normalized))
    has_wall_hint = bool(_WALL_PLACEMENT_PATTERN.search(normalized))
    has_media_furniture = bool(_MEDIA_FURNITURE_PATTERN.search(normalized))

    # Do not promote a desktop monitor to a wall requirement merely because the
    # prompt also mentions a back wall. TVs are explicit; other displays require
    # direct wall-placement language.
    if has_explicit_wall_display or (
        has_named_tv and (has_wall_hint or has_media_furniture)
    ):
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
        return (
            "- No explicit wall-object obligations were extracted from the prompt. "
            "Decorate walls contextually."
        )

    return "\n".join(requirements)
