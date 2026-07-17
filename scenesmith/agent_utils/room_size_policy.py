"""Deterministic room-size safeguards for floor-plan generation.

The model still chooses the initial dimensions.  This module only constrains
unqualified single-room prompts to a generous, room-type-specific envelope so
that a global tool limit cannot accidentally become the selected room size.
"""

from __future__ import annotations

import math
import re

from dataclasses import dataclass
from typing import Any


DEFAULT_ROOM_SIZE_POLICY: dict[str, Any] = {
    "enabled": True,
    "apply_in_house_mode": False,
    "preserve_explicit_dimensions": True,
    "default": {
        "min_side_m": 2.0,
        "min_area_m2": 9.0,
        "max_area_m2": 48.0,
        "max_side_m": 8.0,
    },
    "rooms": {
        "bedroom": {
            "min_side_m": 2.8,
            "min_area_m2": 9.0,
            "max_area_m2": 24.0,
            "max_side_m": 5.5,
        },
        "living_room": {
            "min_side_m": 3.0,
            "min_area_m2": 16.0,
            "max_area_m2": 40.0,
            "max_side_m": 7.0,
        },
        "kitchen": {
            "min_side_m": 2.4,
            "min_area_m2": 7.0,
            "max_area_m2": 25.0,
            "max_side_m": 6.0,
        },
        "bathroom": {
            "min_side_m": 1.8,
            "min_area_m2": 4.0,
            "max_area_m2": 12.0,
            "max_side_m": 4.0,
        },
        "dining_room": {
            "min_side_m": 2.5,
            "min_area_m2": 10.0,
            "max_area_m2": 32.0,
            "max_side_m": 7.0,
        },
        "office": {
            "min_side_m": 2.4,
            "min_area_m2": 8.0,
            "max_area_m2": 30.0,
            "max_side_m": 7.0,
        },
        "classroom": {
            "min_side_m": 4.0,
            "min_area_m2": 30.0,
            "max_area_m2": 100.0,
            "max_side_m": 12.0,
        },
    },
}

_ROOM_TERM = (
    r"(?:bedroom|living[ -]?room|dining[ -]?room|kitchen|bath(?:room)?|"
    r"office|classroom|studio|apartment|house|room|space)"
)
_DIMENSION_PAIR = (
    r"(?:\d+(?:\.\d+)?\s*(?:m|meter(?:s)?|metre(?:s)?|米)?\s*"
    r"(?:x|×|by)\s*\d+(?:\.\d+)?\s*(?:m|meter(?:s)?|metre(?:s)?|米))"
)
_AREA_VALUE = r"(?:\d+(?:\.\d+)?\s*(?:m\s*(?:2|²)|square\s+met(?:er|re)s?|平方米))"
_CHINESE_ROOM_TERM = (
    r"(?:卧室|客厅|餐厅|厨房|浴室|卫生间|办公室|教室|房间|空间|公寓|住宅)"
)
_EXPLICIT_ROOM_DIMENSION_PATTERNS = (
    re.compile(
        rf"\b{_ROOM_TERM}\b[^.!?]{{0,30}}\b"
        rf"(?:measur(?:e|es|ing)|siz(?:e|ed)|dimensions?(?:\s+of)?|is|at|of)\b"
        rf"[^.!?]{{0,12}}{_DIMENSION_PAIR}",
        re.IGNORECASE,
    ),
    re.compile(rf"{_DIMENSION_PAIR}[^.!?]{{0,30}}\b{_ROOM_TERM}\b", re.IGNORECASE),
    re.compile(
        rf"\b{_ROOM_TERM}\b[^.!?]{{0,24}}\b"
        rf"(?:area(?:\s+of)?|measur(?:e|es|ing)|siz(?:e|ed)|is|of)\b"
        rf"[^.!?]{{0,12}}{_AREA_VALUE}",
        re.IGNORECASE,
    ),
    re.compile(rf"{_AREA_VALUE}[^.!?]{{0,20}}\b{_ROOM_TERM}\b", re.IGNORECASE),
    re.compile(
        rf"{_CHINESE_ROOM_TERM}[^。！？]{{0,16}}(?:尺寸|大小|长宽|面积|为|是|约)"
        rf"[^。！？]{{0,8}}(?:{_DIMENSION_PAIR}|{_AREA_VALUE})",
        re.IGNORECASE,
    ),
    re.compile(
        rf"(?:{_DIMENSION_PAIR}|{_AREA_VALUE})[^。！？]{{0,16}}{_CHINESE_ROOM_TERM}",
        re.IGNORECASE,
    ),
)


@dataclass(frozen=True)
class RoomSizeAdjustment:
    """Result of applying a room-size envelope."""

    width: float
    depth: float
    changed: bool = False
    reason: str = ""


def prompt_has_explicit_room_dimensions(prompt: str) -> bool:
    """Return whether the prompt explicitly sizes the room itself.

    Object dimensions such as "a 2m x 3m rug" intentionally do not match.
    """

    text = " ".join(str(prompt or "").split())
    return any(pattern.search(text) for pattern in _EXPLICIT_ROOM_DIMENSION_PATTERNS)


def normalize_room_type(room_type: str) -> str:
    """Map free-form room labels to a configured size-policy category."""

    normalized = re.sub(r"[^a-z0-9]+", "_", str(room_type or "").lower()).strip("_")
    original = str(room_type or "")
    if "卧室" in original:
        return "bedroom"
    if "客厅" in original:
        return "living_room"
    if "餐厅" in original:
        return "dining_room"
    if "厨房" in original:
        return "kitchen"
    if "浴室" in original or "卫生间" in original:
        return "bathroom"
    if "教室" in original:
        return "classroom"
    if "办公室" in original or "书房" in original:
        return "office"
    if "bed" in normalized:
        return "bedroom"
    if "living" in normalized or normalized in {"lounge", "family_room"}:
        return "living_room"
    if "dining" in normalized:
        return "dining_room"
    if "kitchen" in normalized:
        return "kitchen"
    if "bath" in normalized or normalized in {"toilet", "washroom"}:
        return "bathroom"
    if "class" in normalized:
        return "classroom"
    if "office" in normalized or "study" in normalized:
        return "office"
    return normalized or "default"


def normalize_room_dimensions(
    *,
    room_type: str,
    width: float,
    depth: float,
    prompt: str = "",
    mode: str = "room",
    policy: Any | None = None,
) -> RoomSizeAdjustment:
    """Constrain room dimensions while preserving aspect ratio when practical."""

    source_width = float(width)
    source_depth = float(depth)
    if source_width <= 0 or source_depth <= 0:
        return RoomSizeAdjustment(source_width, source_depth)

    policy_obj = policy if policy is not None else DEFAULT_ROOM_SIZE_POLICY
    if not bool(_get(policy_obj, "enabled", True)):
        return RoomSizeAdjustment(source_width, source_depth)
    if mode == "house" and not bool(_get(policy_obj, "apply_in_house_mode", False)):
        return RoomSizeAdjustment(source_width, source_depth)
    if bool(_get(policy_obj, "preserve_explicit_dimensions", True)) and (
        prompt_has_explicit_room_dimensions(prompt)
    ):
        return RoomSizeAdjustment(source_width, source_depth)

    category = normalize_room_type(room_type)
    default_bounds = _get(policy_obj, "default", DEFAULT_ROOM_SIZE_POLICY["default"])
    rooms = _get(policy_obj, "rooms", DEFAULT_ROOM_SIZE_POLICY["rooms"])
    bounds = _get(rooms, category, default_bounds)
    min_side = float(
        _get(bounds, "min_side_m", _get(default_bounds, "min_side_m", 0.0))
    )
    min_area = float(
        _get(bounds, "min_area_m2", _get(default_bounds, "min_area_m2", 0.0))
    )
    max_area = float(
        _get(bounds, "max_area_m2", _get(default_bounds, "max_area_m2", math.inf))
    )
    max_side = float(
        _get(bounds, "max_side_m", _get(default_bounds, "max_side_m", math.inf))
    )

    adjusted_width = source_width
    adjusted_depth = source_depth
    reasons: list[str] = []

    if min_side > 0 and min(adjusted_width, adjusted_depth) < min_side:
        adjusted_width = max(adjusted_width, min_side)
        adjusted_depth = max(adjusted_depth, min_side)
        reasons.append(f"side<{min_side:g}m")
    if max_side > 0 and max(adjusted_width, adjusted_depth) > max_side:
        adjusted_width = min(adjusted_width, max_side)
        adjusted_depth = min(adjusted_depth, max_side)
        reasons.append(f"side>{max_side:g}m")

    area = adjusted_width * adjusted_depth
    if max_area > 0 and area > max_area:
        scale = math.sqrt(max_area / area)
        adjusted_width = max(min_side, adjusted_width * scale)
        adjusted_depth = max(min_side, adjusted_depth * scale)
        reasons.append(f"area>{max_area:g}m2")

    area = adjusted_width * adjusted_depth
    if min_area > 0 and area < min_area:
        scale = math.sqrt(min_area / area)
        adjusted_width = min(max_side, adjusted_width * scale)
        adjusted_depth = min(max_side, adjusted_depth * scale)
        reasons.append(f"area<{min_area:g}m2")

    adjusted_width = round(adjusted_width, 3)
    adjusted_depth = round(adjusted_depth, 3)
    changed = not (
        math.isclose(adjusted_width, source_width, abs_tol=1e-3)
        and math.isclose(adjusted_depth, source_depth, abs_tol=1e-3)
    )
    return RoomSizeAdjustment(
        width=adjusted_width,
        depth=adjusted_depth,
        changed=changed,
        reason=(
            f"unqualified {category} constrained by " + ", ".join(reasons)
            if changed
            else ""
        ),
    )


def _get(value: Any, key: str, default: Any) -> Any:
    """Read either a mapping/DictConfig key or an object attribute."""

    if value is None:
        return default
    getter = getattr(value, "get", None)
    if callable(getter):
        result = getter(key, default)
        return default if result is None else result
    result = getattr(value, key, default)
    return default if result is None else result
