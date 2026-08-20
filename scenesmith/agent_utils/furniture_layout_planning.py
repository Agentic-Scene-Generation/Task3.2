"""Deterministic furniture layout planning and plausibility checks.

The generic opening-aware reservation pass keeps furniture out of door and open
connection clearances and exposes continuous wall segments left by doors and
windows.  The bedroom-specific pass catches missing bed anchors, oversized bed
retrieval, and critic scores that miss obvious human-layout issues.
"""

from __future__ import annotations

import math

from dataclasses import dataclass, field
from typing import Any

import numpy as np


WALLS = ("north", "south", "east", "west")
BED_TERMS = ("bed", "queen bed", "king bed", "twin bed", "single bed")
NIGHTSTAND_TERMS = ("nightstand", "bedside table")
WARDROBE_TERMS = ("wardrobe", "closet", "armoire")


@dataclass(frozen=True)
class BedroomAnchorPlan:
    """Small deterministic plan for bedroom furniture anchoring."""

    bed_head_wall: str | None
    bed_head_wall_reason: str
    avoid_head_walls: list[str] = field(default_factory=list)
    wall_openings: dict[str, list[str]] = field(default_factory=dict)

    def to_guidance_text(self, room_length: float, room_width: float) -> str:
        if not self.bed_head_wall:
            return ""

        avoid = "; ".join(self.avoid_head_walls) or "none"
        wall_status = ", ".join(
            f"{wall}: {','.join(types) if types else 'solid'}"
            for wall, types in self.wall_openings.items()
        )
        return (
            "Bedroom anchor plan:\n"
            f"- Room footprint is {room_length:.2f}m x {room_width:.2f}m.\n"
            f"- Anchor the bed headboard on {self.bed_head_wall}_wall "
            f"({self.bed_head_wall_reason}).\n"
            f"- Keep the bed headboard within 0.15-0.25m of "
            f"{self.bed_head_wall}_wall; do not leave the bed floating in the "
            "middle when a solid wall anchor exists.\n"
            "- Rotate the bed so its headboard faces the anchor wall; the bed "
            "foot should point back into the open room.\n"
            "- Put the two nightstands symmetrically on the left and right sides "
            "of the bed, near the headboard side, with similar clearance.\n"
            "- Place the wardrobe against a wall or in a corner without blocking "
            "the door, windows, or bed/nightstand access.\n"
            f"- Avoid bed-head walls: {avoid}.\n"
            f"- Wall opening summary: {wall_status}."
        )


@dataclass(frozen=True)
class OpeningReservationZone:
    """World-space opening clearance projected into a furniture reservation."""

    zone_id: str
    opening_type: str
    wall: str
    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]
    severity: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "zone_id": self.zone_id,
            "opening_type": self.opening_type,
            "wall": self.wall,
            "bounds_xyz": [*self.bounds_min, *self.bounds_max],
            "severity": self.severity,
        }


@dataclass(frozen=True)
class OpeningFreeWallSegment:
    """Continuous wall interval not occupied by a door or window span."""

    wall: str
    axis: str
    start: float
    end: float

    @property
    def length_m(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict[str, Any]:
        return {
            "wall": self.wall,
            "axis": self.axis,
            "start": self.start,
            "end": self.end,
            "length_m": self.length_m,
        }


@dataclass(frozen=True)
class OpeningAwareReservationPlan:
    """Resolved floor-plan reservations consumed by furniture placement."""

    zones: list[OpeningReservationZone] = field(default_factory=list)
    usable_wall_segments: list[OpeningFreeWallSegment] = field(default_factory=list)
    fully_opening_free_walls: list[str] = field(default_factory=list)
    opening_margin_m: float = 0.15

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "scenesmith.opening_aware_reservations.v1",
            "opening_margin_m": self.opening_margin_m,
            "zones": [zone.to_dict() for zone in self.zones],
            "usable_wall_segments": [
                segment.to_dict() for segment in self.usable_wall_segments
            ],
            "fully_opening_free_walls": list(self.fully_opening_free_walls),
        }

    def to_guidance_text(self) -> str:
        if not self.zones:
            return ""
        hard_zones = [zone for zone in self.zones if zone.severity == "hard"]
        window_zones = [zone for zone in self.zones if zone.opening_type == "window"]
        lines = [
            "Opening-aware floor reservations (deterministic geometry; plan around "
            "these before placing furniture):"
        ]
        if hard_zones:
            lines.append(
                "- Hard door/open-connection clearances: "
                + "; ".join(
                    f"{zone.zone_id} on {zone.wall}_wall bounds_xyz="
                    f"{[round(value, 3) for value in (*zone.bounds_min, *zone.bounds_max)]}"
                    for zone in hard_zones
                )
                + ". No furniture footprint may overlap these volumes."
            )
        if window_zones:
            lines.append(
                "- Window access/visibility clearances: "
                + "; ".join(
                    f"{zone.zone_id} on {zone.wall}_wall bounds_xyz="
                    f"{[round(value, 3) for value in (*zone.bounds_min, *zone.bounds_max)]}"
                    for zone in window_zones
                )
                + ". Do not put tall or wall-backed focal furniture through the "
                "window volume; furniture fully below the sill may remain valid."
            )
        if self.fully_opening_free_walls:
            lines.append(
                "- Fully opening-free anchor walls: "
                + ", ".join(f"{wall}_wall" for wall in self.fully_opening_free_walls)
                + ". Prefer these for primary wall-backed or focal groups."
            )
        if self.usable_wall_segments:
            lines.append(
                f"- Continuous usable wall segments after a {self.opening_margin_m:.2f}m "
                "lateral opening margin: "
                + "; ".join(
                    f"{segment.wall}_wall {segment.axis}="
                    f"[{segment.start:.2f},{segment.end:.2f}] "
                    f"({segment.length_m:.2f}m)"
                    for segment in self.usable_wall_segments
                )
                + "."
            )
        lines.append(
            "- Reserve each connected wall-backed furniture group on one compatible "
            "usable segment. Do not move it onto a door/window wall when a solid "
            "opening-free wall satisfies the same task relation."
        )
        return "\n".join(lines)


@dataclass(frozen=True)
class AssetSizePolicyResult:
    """Normalized furniture asset requests."""

    object_descriptions: list[str]
    short_names: list[str]
    desired_dimensions: list[list[float]]
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class BedroomPlausibilityReport:
    """Deterministic soft plausibility report for bedroom layouts."""

    score: float
    penalty: float
    issues: list[str] = field(default_factory=list)
    anchor_plan: BedroomAnchorPlan | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": float(self.score),
            "penalty": float(self.penalty),
            "issues": list(self.issues),
            "anchor_plan": {
                "bed_head_wall": (
                    self.anchor_plan.bed_head_wall if self.anchor_plan else None
                ),
                "bed_head_wall_reason": (
                    self.anchor_plan.bed_head_wall_reason if self.anchor_plan else None
                ),
                "avoid_head_walls": (
                    list(self.anchor_plan.avoid_head_walls) if self.anchor_plan else []
                ),
                "wall_openings": (
                    dict(self.anchor_plan.wall_openings) if self.anchor_plan else {}
                ),
            },
        }


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    try:
        if hasattr(cfg, "get"):
            return cfg.get(key, default)
    except Exception:
        pass
    return getattr(cfg, key, default)


def _cfg_bool(cfg: Any, key: str, default: bool) -> bool:
    return bool(_cfg_get(cfg, key, default))


def _cfg_float(cfg: Any, key: str, default: float) -> float:
    try:
        return float(_cfg_get(cfg, key, default))
    except (TypeError, ValueError):
        return default


def _cfg_float_list(cfg: Any, key: str, default: list[float]) -> list[float]:
    value = _cfg_get(cfg, key, default)
    try:
        result = [float(x) for x in list(value)]
    except (TypeError, ValueError):
        return list(default)
    return result if len(result) == len(default) else list(default)


def _text(value: Any) -> str:
    return str(value or "").lower().replace("_", " ")


def _opening_wall(opening: Any) -> str | None:
    raw = None
    if isinstance(opening, dict):
        raw = opening.get("wall_direction")
    else:
        raw = getattr(opening, "wall_direction", None)
    raw = getattr(raw, "value", raw)
    return str(raw).lower() if raw else None


def _opening_type(opening: Any) -> str:
    raw = None
    if isinstance(opening, dict):
        raw = opening.get("opening_type")
    else:
        raw = getattr(opening, "opening_type", None)
    raw = getattr(raw, "value", raw)
    return str(raw).lower() if raw else "opening"


def _opening_value(opening: Any, name: str, default: Any = None) -> Any:
    if isinstance(opening, dict):
        return opening.get(name, default)
    return getattr(opening, name, default)


def _opening_id(opening: Any, index: int) -> str:
    return str(_opening_value(opening, "opening_id", "") or f"opening_{index}")


def _opening_clearance_bounds(
    opening: Any,
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    opening_type = _opening_type(opening)
    if opening_type == "door" and not isinstance(opening, dict):
        try:
            from scenesmith.agent_utils.clearance_zones import (
                door_swing_clearance_bounds,
            )

            bounds = door_swing_clearance_bounds(opening)
            if bounds is not None:
                return tuple(float(v) for v in bounds[0]), tuple(
                    float(v) for v in bounds[1]
                )
        except (AttributeError, TypeError, ValueError):
            pass
    if opening_type == "open":
        center = _opening_value(opening, "center_world")
        wall = _opening_wall(opening)
        try:
            center_x = float(center[0])
            center_y = float(center[1])
            half_width = max(0.4, float(_opening_value(opening, "width", 0.0)) / 2.0)
        except (TypeError, ValueError, IndexError):
            return None
        depth = 1.0
        if wall == "north":
            return (
                (center_x - half_width, center_y - depth, 0.0),
                (center_x + half_width, center_y, 2.5),
            )
        if wall == "south":
            return (
                (center_x - half_width, center_y, 0.0),
                (center_x + half_width, center_y + depth, 2.5),
            )
        if wall == "east":
            return (
                (center_x - depth, center_y - half_width, 0.0),
                (center_x, center_y + half_width, 2.5),
            )
        if wall == "west":
            return (
                (center_x, center_y - half_width, 0.0),
                (center_x + depth, center_y + half_width, 2.5),
            )
        return None
    lower = _opening_value(opening, "clearance_bbox_min")
    upper = _opening_value(opening, "clearance_bbox_max")
    try:
        lower_values = tuple(float(v) for v in lower)
        upper_values = tuple(float(v) for v in upper)
    except (TypeError, ValueError):
        return None
    if len(lower_values) < 3 or len(upper_values) < 3:
        return None
    return lower_values[:3], upper_values[:3]


def _opening_tangent_center(opening: Any, wall: str) -> float | None:
    center = _opening_value(opening, "center_world")
    try:
        return float(center[0] if wall in {"north", "south"} else center[1])
    except (TypeError, ValueError, IndexError):
        bounds = _opening_clearance_bounds(opening)
        if bounds is None:
            return None
        axis = 0 if wall in {"north", "south"} else 1
        return (bounds[0][axis] + bounds[1][axis]) / 2.0


def _subtract_intervals(
    start: float,
    end: float,
    blocked: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for lower, upper in sorted(blocked):
        lower = max(start, lower)
        upper = min(end, upper)
        if upper <= lower:
            continue
        if not merged or lower > merged[-1][1]:
            merged.append([lower, upper])
        else:
            merged[-1][1] = max(merged[-1][1], upper)
    free: list[tuple[float, float]] = []
    cursor = start
    for lower, upper in merged:
        if lower > cursor:
            free.append((cursor, lower))
        cursor = max(cursor, upper)
    if cursor < end:
        free.append((cursor, end))
    return free


def build_opening_aware_reservation_plan(
    scene: Any,
    *,
    opening_margin_m: float = 0.15,
) -> OpeningAwareReservationPlan:
    """Resolve door/window geometry into usable furniture reservation regions."""
    size = _room_size(scene)
    room_geometry = getattr(scene, "room_geometry", None)
    raw_openings = getattr(room_geometry, "openings", None) if room_geometry else None
    try:
        openings = list(raw_openings or [])
    except TypeError:
        openings = []
    if size is None:
        return OpeningAwareReservationPlan(opening_margin_m=opening_margin_m)

    zones: list[OpeningReservationZone] = []
    openings_by_wall: dict[str, list[Any]] = {wall: [] for wall in WALLS}
    for index, opening in enumerate(openings):
        wall = _opening_wall(opening)
        if wall not in openings_by_wall:
            continue
        openings_by_wall[wall].append(opening)
        bounds = _opening_clearance_bounds(opening)
        if bounds is None:
            continue
        opening_type = _opening_type(opening)
        zones.append(
            OpeningReservationZone(
                zone_id=_opening_id(opening, index),
                opening_type=opening_type,
                wall=wall,
                bounds_min=bounds[0],
                bounds_max=bounds[1],
                severity="hard" if opening_type in {"door", "open"} else "advisory",
            )
        )

    room_length, room_width = size
    segments: list[OpeningFreeWallSegment] = []
    free_walls: list[str] = []
    for wall in WALLS:
        axis = "x" if wall in {"north", "south"} else "y"
        half_length = _wall_length(room_length, room_width, wall) / 2.0
        wall_openings = openings_by_wall[wall]
        if not wall_openings:
            free_walls.append(wall)
        blocked: list[tuple[float, float]] = []
        for opening in wall_openings:
            center = _opening_tangent_center(opening, wall)
            try:
                width = max(0.0, float(_opening_value(opening, "width", 0.0)))
            except (TypeError, ValueError):
                width = 0.0
            if center is None or width <= 0.0:
                continue
            blocked.append(
                (
                    center - width / 2.0 - opening_margin_m,
                    center + width / 2.0 + opening_margin_m,
                )
            )
        segments.extend(
            OpeningFreeWallSegment(wall=wall, axis=axis, start=start, end=end)
            for start, end in _subtract_intervals(-half_length, half_length, blocked)
            if end - start > 1e-6
        )
    return OpeningAwareReservationPlan(
        zones=zones,
        usable_wall_segments=segments,
        fully_opening_free_walls=free_walls,
        opening_margin_m=opening_margin_m,
    )


def format_opening_aware_reservation_guidance(scene: Any) -> str:
    """Return concrete opening-aware reservation guidance for initial design."""
    return build_opening_aware_reservation_plan(scene).to_guidance_text()


def opening_clearance_conflict_for_transform(
    *,
    scene: Any,
    scene_obj: Any,
    transform: Any,
) -> OpeningReservationZone | None:
    """Return a hard opening zone intersected by a candidate furniture pose."""
    metadata = getattr(scene_obj, "metadata", {}) or {}
    if metadata.get("asset_source") == "thin_covering":
        return None
    bbox_min = getattr(scene_obj, "bbox_min", None)
    bbox_max = getattr(scene_obj, "bbox_max", None)
    if bbox_min is None or bbox_max is None:
        return None
    try:
        corners = np.array(
            [
                transform @ np.array([x, y, z], dtype=float)
                for x in (float(bbox_min[0]), float(bbox_max[0]))
                for y in (float(bbox_min[1]), float(bbox_max[1]))
                for z in (float(bbox_min[2]), float(bbox_max[2]))
            ]
        )
    except (TypeError, ValueError, IndexError):
        return None
    world_min = np.min(corners, axis=0)
    world_max = np.max(corners, axis=0)
    for zone in build_opening_aware_reservation_plan(scene).zones:
        if zone.severity != "hard":
            continue
        overlap = [
            min(float(world_max[axis]), zone.bounds_max[axis])
            - max(float(world_min[axis]), zone.bounds_min[axis])
            for axis in range(3)
        ]
        if all(value > 1e-5 for value in overlap):
            return zone
    return None


def _room_size(scene: Any) -> tuple[float, float] | None:
    room_geometry = getattr(scene, "room_geometry", None)
    if room_geometry is None:
        return None
    length = float(getattr(room_geometry, "length", 0.0) or 0.0)
    width = float(getattr(room_geometry, "width", 0.0) or 0.0)
    if length <= 0 or width <= 0:
        return None
    return length, width


def _room_bounds(scene: Any) -> tuple[float, float, float, float] | None:
    size = _room_size(scene)
    if size is None:
        return None
    length, width = size
    return (-length / 2.0, -width / 2.0, length / 2.0, width / 2.0)


def _wall_length(room_length: float, room_width: float, wall: str) -> float:
    return room_length if wall in ("north", "south") else room_width


def _wall_opening_types(scene: Any) -> dict[str, list[str]]:
    wall_openings = {wall: [] for wall in WALLS}
    room_geometry = getattr(scene, "room_geometry", None)
    openings = list(getattr(room_geometry, "openings", []) or [])
    for opening in openings:
        wall = _opening_wall(opening)
        if wall in wall_openings:
            wall_openings[wall].append(_opening_type(opening))
    return wall_openings


def is_bedroom_scene(scene: Any) -> bool:
    """Return whether this room should receive bedroom-specific checks."""
    # SceneExpert temporarily appends StageBrief and retrieved-memory text to
    # ``text_description``.  Those hints can mention beds or wardrobes while
    # describing a failure from another room type, so they must never change
    # which room-specific verifier is active.
    original_description = getattr(
        scene,
        "scene_expert_original_description",
        getattr(scene, "text_description", ""),
    )
    text = f"{_text(getattr(scene, 'room_type', ''))} " f"{_text(original_description)}"
    return (
        "bedroom" in text
        or "nightstand" in text
        or any(term in text.split() for term in ("bed", "beds"))
    )


def scene_text_explicitly_requests_large_bed(scene: Any) -> bool:
    text = _text(getattr(scene, "text_description", ""))
    return any(
        term in text for term in ("queen bed", "king bed", "queen-size", "king-size")
    )


def scene_text_explicitly_requests_small_bed(scene: Any) -> bool:
    text = _text(getattr(scene, "text_description", ""))
    return any(term in text for term in ("twin bed", "single bed", "twin-size"))


def build_bedroom_anchor_plan(
    scene: Any, cfg: Any | None = None
) -> BedroomAnchorPlan | None:
    """Choose a preferred bed-head wall from room openings."""
    if not is_bedroom_scene(scene):
        return None
    if not _cfg_bool(cfg, "enabled", True) or not _cfg_bool(
        cfg, "anchor_planning", True
    ):
        return None

    size = _room_size(scene)
    if size is None:
        return None
    room_length, room_width = size
    wall_openings = _wall_opening_types(scene)

    best_wall: str | None = None
    best_score = -1e9
    for wall in WALLS:
        opening_types = wall_openings[wall]
        score = _wall_length(room_length, room_width, wall) * 0.2
        if not opening_types:
            score += 4.0
        if "door" in opening_types or "open" in opening_types:
            score -= 6.0
        if "window" in opening_types:
            score -= 3.0
        if score > best_score:
            best_score = score
            best_wall = wall

    if best_wall is None:
        return None

    best_types = wall_openings[best_wall]
    if not best_types:
        reason = "solid wall without doors/windows"
    else:
        reason = "least obstructed available wall"

    avoid_walls = [
        f"{wall}_wall has {','.join(types)}"
        for wall, types in wall_openings.items()
        if wall != best_wall and types
    ]
    return BedroomAnchorPlan(
        bed_head_wall=best_wall,
        bed_head_wall_reason=reason,
        avoid_head_walls=avoid_walls,
        wall_openings=wall_openings,
    )


def format_bedroom_anchor_guidance(scene: Any, cfg: Any | None = None) -> str:
    """Return prompt guidance for the initial furniture design call."""
    if not _cfg_bool(cfg, "enabled", True) or not _cfg_bool(
        cfg, "anchor_planning", True
    ):
        return ""
    plan = build_bedroom_anchor_plan(scene, cfg=cfg)
    size = _room_size(scene)
    if plan is None or size is None:
        return ""
    return plan.to_guidance_text(room_length=size[0], room_width=size[1])


def _infer_category(text: str) -> str | None:
    normalized = _text(text)
    if any(term in normalized for term in NIGHTSTAND_TERMS):
        return "nightstand"
    if any(term in normalized for term in WARDROBE_TERMS):
        return "wardrobe"
    if any(term in normalized for term in BED_TERMS):
        return "bed"
    return None


def _normalize_dimensions(
    dimensions: list[float], minimum: list[float], maximum: list[float]
) -> list[float]:
    normalized: list[float] = []
    for idx, value in enumerate(dimensions[:3]):
        value_f = float(value)
        normalized.append(max(float(minimum[idx]), min(float(maximum[idx]), value_f)))
    while len(normalized) < 3:
        normalized.append(float(minimum[len(normalized)]))
    return normalized


def apply_bedroom_asset_size_policy(
    *,
    scene: Any,
    object_descriptions: list[str],
    short_names: list[str],
    desired_dimensions: list[list[float]],
    cfg: Any | None = None,
) -> AssetSizePolicyResult:
    """Normalize bedroom furniture asset requests before retrieval/generation."""
    descriptions = list(object_descriptions)
    names = list(short_names)
    dimensions = [list(dims) for dims in desired_dimensions]
    notes: list[str] = []

    if not is_bedroom_scene(scene):
        return AssetSizePolicyResult(descriptions, names, dimensions, notes)
    if not _cfg_bool(cfg, "enabled", True) or not _cfg_bool(
        cfg, "asset_size_gating", True
    ):
        return AssetSizePolicyResult(descriptions, names, dimensions, notes)

    room_size = _room_size(scene)
    room_area = room_size[0] * room_size[1] if room_size else 0.0
    small_room_area = _cfg_float(cfg, "small_room_area_m2", 20.0)
    bed_default = _cfg_float_list(
        cfg, "unqualified_bed_default_dimensions", [1.60, 2.05, 0.80]
    )
    bed_min = _cfg_float_list(cfg, "bed_min_dimensions", [1.20, 1.80, 0.35])
    bed_max = _cfg_float_list(cfg, "unqualified_bed_max_dimensions", [1.75, 2.20, 1.20])
    large_bed_max = _cfg_float_list(
        cfg, "explicit_large_bed_max_dimensions", [2.10, 2.30, 1.20]
    )
    nightstand_min = _cfg_float_list(
        cfg, "nightstand_min_dimensions", [0.30, 0.30, 0.35]
    )
    nightstand_max = _cfg_float_list(
        cfg, "nightstand_max_dimensions", [0.65, 0.55, 0.75]
    )
    wardrobe_min = _cfg_float_list(cfg, "wardrobe_min_dimensions", [0.60, 0.35, 1.60])
    wardrobe_max = _cfg_float_list(cfg, "wardrobe_max_dimensions", [1.20, 0.70, 2.35])

    prompt_requests_large_bed = scene_text_explicitly_requests_large_bed(scene)
    prompt_requests_small_bed = scene_text_explicitly_requests_small_bed(scene)
    count = min(len(descriptions), len(names), len(dimensions))
    for idx in range(count):
        category = _infer_category(f"{names[idx]} {descriptions[idx]}")
        dims = dimensions[idx]
        if len(dims) < 3:
            dims = [*dims, *([0.0] * (3 - len(dims)))]

        if category == "bed":
            desc_text = _text(f"{names[idx]} {descriptions[idx]}")
            requested_large = "queen" in desc_text or "king" in desc_text
            oversize = any(float(dims[i]) > bed_max[i] for i in range(3))
            if (
                not prompt_requests_large_bed
                and not prompt_requests_small_bed
                and (requested_large or oversize or room_area <= small_room_area)
            ):
                descriptions[idx] = (
                    "Compact standard double bed with headboard, mattress, "
                    "pillows, and bedding"
                )
                if "queen" in _text(names[idx]) or "king" in _text(names[idx]):
                    names[idx] = "bed"
                dimensions[idx] = list(bed_default)
                notes.append(
                    "rewrote unqualified bedroom bed request to compact standard "
                    f"dimensions {dimensions[idx]}"
                )
            else:
                max_dims = large_bed_max if prompt_requests_large_bed else bed_max
                dimensions[idx] = _normalize_dimensions(dims, bed_min, max_dims)

        elif category == "nightstand":
            normalized = _normalize_dimensions(dims, nightstand_min, nightstand_max)
            if normalized != dims[:3]:
                notes.append(
                    f"clamped nightstand dimensions from {dims[:3]} to {normalized}"
                )
            dimensions[idx] = normalized

        elif category == "wardrobe":
            normalized = _normalize_dimensions(dims, wardrobe_min, wardrobe_max)
            if normalized != dims[:3]:
                notes.append(
                    f"clamped wardrobe dimensions from {dims[:3]} to {normalized}"
                )
            dimensions[idx] = normalized

    return AssetSizePolicyResult(descriptions, names, dimensions, notes)


def _is_furniture_object(obj: Any) -> bool:
    object_type = getattr(obj, "object_type", "")
    value = getattr(object_type, "value", object_type)
    return str(value).lower() == "furniture"


def _object_category(object_id: Any, obj: Any) -> str | None:
    return _infer_category(
        f"{object_id} {getattr(obj, 'name', '')} {getattr(obj, 'description', '')}"
    )


def _world_bounds(obj: Any) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        bounds = obj.compute_world_bounds()
    except Exception:
        return None
    if bounds is None:
        return None
    return np.asarray(bounds[0], dtype=float), np.asarray(bounds[1], dtype=float)


def _object_dimensions(obj: Any) -> tuple[float, float, float] | None:
    bbox_min = getattr(obj, "bbox_min", None)
    bbox_max = getattr(obj, "bbox_max", None)
    if bbox_min is None or bbox_max is None:
        return None
    dims = np.asarray(bbox_max, dtype=float) - np.asarray(bbox_min, dtype=float)
    return float(abs(dims[0])), float(abs(dims[1])), float(abs(dims[2]))


def _translation_xy(obj: Any) -> np.ndarray:
    translation = obj.transform.translation()
    return np.asarray(translation[:2], dtype=float)


def _rotation_matrix(obj: Any) -> np.ndarray:
    return np.asarray(obj.transform.rotation().matrix(), dtype=float)


def _bed_head_vector_xy(obj: Any) -> np.ndarray:
    rotation = _rotation_matrix(obj)
    # HSSD bed assets use their local +Y facing/arrow direction toward the
    # foot of the bed.  The headboard is therefore on the opposite end.
    head = -(rotation @ np.array([0.0, 1.0, 0.0]))
    xy = head[:2]
    norm = float(np.linalg.norm(xy))
    if norm <= 1e-8:
        return np.array([0.0, 1.0])
    return xy / norm


def _bed_lateral_vector_xy(obj: Any) -> np.ndarray:
    rotation = _rotation_matrix(obj)
    lateral = rotation @ np.array([1.0, 0.0, 0.0])
    xy = lateral[:2]
    norm = float(np.linalg.norm(xy))
    if norm <= 1e-8:
        return np.array([1.0, 0.0])
    return xy / norm


def _wall_for_vector(vector: np.ndarray) -> str:
    x, y = float(vector[0]), float(vector[1])
    if abs(x) >= abs(y):
        return "east" if x >= 0 else "west"
    return "north" if y >= 0 else "south"


def _distance_to_wall(
    bounds: tuple[np.ndarray, np.ndarray],
    room_bounds: tuple[float, float, float, float],
    wall: str,
) -> float:
    world_min, world_max = bounds
    min_x, min_y, max_x, max_y = room_bounds
    if wall == "north":
        return max_y - float(world_max[1])
    if wall == "south":
        return float(world_min[1]) - min_y
    if wall == "east":
        return max_x - float(world_max[0])
    return float(world_min[0]) - min_x


def _opening_axis_interval(opening: Any, wall: str) -> tuple[float, float] | None:
    center = None
    width = None
    if isinstance(opening, dict):
        center = opening.get("center_world")
        width = opening.get("width")
    else:
        center = getattr(opening, "center_world", None)
        width = getattr(opening, "width", None)
    if center is None or width is None:
        return None
    axis_index = 0 if wall in ("north", "south") else 1
    center_value = float(center[axis_index])
    half_width = float(width) / 2.0
    return center_value - half_width, center_value + half_width


def _object_axis_interval(
    bounds: tuple[np.ndarray, np.ndarray], wall: str
) -> tuple[float, float]:
    world_min, world_max = bounds
    axis_index = 0 if wall in ("north", "south") else 1
    return float(world_min[axis_index]), float(world_max[axis_index])


def _intervals_overlap(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return max(a[0], b[0]) <= min(a[1], b[1])


def _opening_clearance_bounds_xy(
    opening: Any,
) -> tuple[np.ndarray, np.ndarray] | None:
    if isinstance(opening, dict):
        lower = opening.get("clearance_bbox_min")
        upper = opening.get("clearance_bbox_max")
        center = opening.get("center_world")
        width = opening.get("width")
    else:
        lower = getattr(opening, "clearance_bbox_min", None)
        upper = getattr(opening, "clearance_bbox_max", None)
        center = getattr(opening, "center_world", None)
        width = getattr(opening, "width", None)
    if lower is not None and upper is not None:
        try:
            return (
                np.asarray(lower, dtype=float)[:2],
                np.asarray(upper, dtype=float)[:2],
            )
        except (TypeError, ValueError):
            return None

    wall = _opening_wall(opening)
    opening_type = _opening_type(opening)
    if wall not in WALLS or center is None or width is None:
        return None
    try:
        center_xy = np.asarray(center, dtype=float)[:2]
        half_width = float(width) / 2.0
    except (TypeError, ValueError):
        return None
    depth = 0.8 if opening_type in {"door", "open"} else 0.5
    lower_xy = center_xy.copy()
    upper_xy = center_xy.copy()
    if wall in {"north", "south"}:
        lower_xy[0] -= half_width
        upper_xy[0] += half_width
        if wall == "north":
            lower_xy[1] -= depth
        else:
            upper_xy[1] += depth
    else:
        lower_xy[1] -= half_width
        upper_xy[1] += half_width
        if wall == "east":
            lower_xy[0] -= depth
        else:
            upper_xy[0] += depth
    return lower_xy, upper_xy


def _rectangles_overlap(
    lower_a: np.ndarray,
    upper_a: np.ndarray,
    lower_b: np.ndarray,
    upper_b: np.ndarray,
) -> bool:
    return bool(
        min(float(upper_a[0]), float(upper_b[0]))
        > max(float(lower_a[0]), float(lower_b[0]))
        and min(float(upper_a[1]), float(upper_b[1]))
        > max(float(lower_a[1]), float(lower_b[1]))
    )


def _bed_has_opening_safe_wall_anchor(scene: Any, bed: Any) -> bool:
    """Return whether a compact bed can touch some wall without blocking openings."""
    room_bounds = _room_bounds(scene)
    dims = _object_dimensions(bed)
    if room_bounds is None or dims is None:
        return True
    bed_width, bed_depth = sorted(dims[:2])
    if bed_width <= 0.0 or bed_depth <= 0.0:
        return True

    room_geometry = getattr(scene, "room_geometry", None)
    margin = max(0.03, float(getattr(room_geometry, "wall_thickness", 0.08) or 0.08))
    opening_bounds = [
        bounds
        for opening in list(getattr(room_geometry, "openings", []) or [])
        if (bounds := _opening_clearance_bounds_xy(opening)) is not None
    ]
    if not opening_bounds:
        return True

    min_x, min_y, max_x, max_y = room_bounds
    for wall in WALLS:
        tangent_axis = 0 if wall in {"north", "south"} else 1
        tangent_half = bed_width / 2.0
        normal_half = bed_depth / 2.0
        tangent_min = (min_x if tangent_axis == 0 else min_y) + tangent_half + margin
        tangent_max = (max_x if tangent_axis == 0 else max_y) - tangent_half - margin
        if tangent_min > tangent_max:
            continue
        tangent_values = {0.0, tangent_min, tangent_max}
        for lower, upper in opening_bounds:
            tangent_values.update(
                (
                    float(lower[tangent_axis]) - tangent_half - margin,
                    float(upper[tangent_axis]) + tangent_half + margin,
                )
            )
        for tangent in tangent_values:
            if tangent < tangent_min - 1e-6 or tangent > tangent_max + 1e-6:
                continue
            if wall == "north":
                center = np.array([tangent, max_y - normal_half - margin])
                half = np.array([tangent_half, normal_half])
            elif wall == "south":
                center = np.array([tangent, min_y + normal_half + margin])
                half = np.array([tangent_half, normal_half])
            elif wall == "east":
                center = np.array([max_x - normal_half - margin, tangent])
                half = np.array([normal_half, tangent_half])
            else:
                center = np.array([min_x + normal_half + margin, tangent])
                half = np.array([normal_half, tangent_half])
            lower = center - half
            upper = center + half
            if not any(
                _rectangles_overlap(lower, upper, opening_lower, opening_upper)
                for opening_lower, opening_upper in opening_bounds
            ):
                return True
    return False


def _bed_overlaps_opening_on_wall(
    scene: Any, bounds: tuple[np.ndarray, np.ndarray], wall: str
) -> str | None:
    # The tangent intervals alone are not enough: a deliberately interior bed
    # can share an X/Y projection with a window while its headboard is well
    # inside the room. Only call this a headboard-opening conflict when the
    # headboard is actually close enough to that wall to be an anchor.
    room_bounds = _room_bounds(scene)
    if room_bounds is not None:
        room_geometry = getattr(scene, "room_geometry", None)
        wall_thickness = float(getattr(room_geometry, "wall_thickness", 0.08) or 0.08)
        anchor_distance = max(0.25, wall_thickness + 0.03)
        if _distance_to_wall(bounds, room_bounds, wall) > anchor_distance:
            return None

    room_geometry = getattr(scene, "room_geometry", None)
    openings = list(getattr(room_geometry, "openings", []) or [])
    obj_interval = _object_axis_interval(bounds, wall)
    for opening in openings:
        if _opening_wall(opening) != wall:
            continue
        interval = _opening_axis_interval(opening, wall)
        if interval and _intervals_overlap(obj_interval, interval):
            return _opening_type(opening)
    return None


def evaluate_bedroom_layout_plausibility(
    scene: Any, cfg: Any | None = None
) -> BedroomPlausibilityReport:
    """Evaluate bedroom-specific human-layout plausibility."""
    if not is_bedroom_scene(scene):
        return BedroomPlausibilityReport(score=1.0, penalty=0.0)
    if not _cfg_bool(cfg, "enabled", True) or not _cfg_bool(
        cfg, "plausibility_verifier", True
    ):
        return BedroomPlausibilityReport(score=1.0, penalty=0.0)

    plan = build_bedroom_anchor_plan(scene, cfg=cfg)
    room_bounds = _room_bounds(scene)
    issues: list[str] = []
    penalty = 0.0

    objects = [
        (object_id, obj, _object_category(object_id, obj))
        for object_id, obj in getattr(scene, "objects", {}).items()
        if _is_furniture_object(obj)
    ]
    beds = [
        (object_id, obj) for object_id, obj, category in objects if category == "bed"
    ]
    nightstands = [
        (object_id, obj)
        for object_id, obj, category in objects
        if category == "nightstand"
    ]
    wardrobes = [
        (object_id, obj)
        for object_id, obj, category in objects
        if category == "wardrobe"
    ]

    if not beds:
        return BedroomPlausibilityReport(
            score=0.65,
            penalty=0.35,
            issues=["bedroom plausibility: no bed object found"],
            anchor_plan=plan,
        )

    bed_id, bed = beds[0]
    bed_bounds = _world_bounds(bed)
    if bed_bounds is not None and room_bounds is not None and plan:
        actual_head_wall = _wall_for_vector(_bed_head_vector_xy(bed))
        expected_head_wall = plan.bed_head_wall
        if expected_head_wall and actual_head_wall != expected_head_wall:
            issues.append(
                "bedroom plausibility: bed headboard faces "
                f"{actual_head_wall}_wall, expected {expected_head_wall}_wall"
            )
            penalty += 0.12

        anchor_wall = expected_head_wall or actual_head_wall
        if anchor_wall:
            wall_distance = _distance_to_wall(bed_bounds, room_bounds, anchor_wall)
            max_distance = _cfg_float(cfg, "bed_wall_anchor_max_distance_m", 0.25)
            if wall_distance > max_distance:
                if _bed_has_opening_safe_wall_anchor(scene, bed):
                    issues.append(
                        "bedroom plausibility: bed headboard is not anchored to "
                        f"{anchor_wall}_wall (distance {wall_distance:.2f}m)"
                    )
                    penalty += 0.12
                else:
                    issues.append(
                        "bedroom plausibility: bed uses an interior opening-safe "
                        "layout because every wall anchor blocks an opening"
                    )
                    penalty += 0.04

        opening_type = _bed_overlaps_opening_on_wall(
            scene, bed_bounds, actual_head_wall
        )
        if opening_type in ("window", "door", "open"):
            issues.append(
                "bedroom plausibility: bed headboard overlaps/targets "
                f"{opening_type} on {actual_head_wall}_wall"
            )
            penalty += 0.08

    bed_dims = _object_dimensions(bed)
    if bed_dims is not None and not scene_text_explicitly_requests_large_bed(scene):
        short_side = min(bed_dims[0], bed_dims[1])
        long_side = max(bed_dims[0], bed_dims[1])
        max_footprint = _cfg_float_list(
            cfg, "unqualified_bed_max_footprint", [1.75, 2.20]
        )
        if short_side > max_footprint[0] or long_side > max_footprint[1]:
            issues.append(
                "bedroom plausibility: unqualified bed asset is oversized "
                f"(footprint {short_side:.2f}m x {long_side:.2f}m)"
            )
            penalty += 0.08

    if len(nightstands) >= 2:
        try:
            bed_center = _translation_xy(bed)
            lateral = _bed_lateral_vector_xy(bed)
            lateral_positions = [
                float(np.dot(_translation_xy(obj) - bed_center, lateral))
                for _, obj in nightstands[:2]
            ]
            if math.prod(lateral_positions) >= 0:
                issues.append(
                    "bedroom plausibility: nightstands are not on opposite bed sides"
                )
                penalty += 0.08
        except Exception:
            pass

    if wardrobes and room_bounds is not None:
        wardrobe_id, wardrobe = wardrobes[0]
        wardrobe_bounds = _world_bounds(wardrobe)
        if wardrobe_bounds is not None:
            min_x, min_y, max_x, max_y = room_bounds
            world_min, world_max = wardrobe_bounds
            nearest_wall_distance = min(
                float(world_min[0]) - min_x,
                max_x - float(world_max[0]),
                float(world_min[1]) - min_y,
                max_y - float(world_max[1]),
            )
            if nearest_wall_distance > _cfg_float(
                cfg, "wardrobe_wall_max_distance_m", 0.35
            ):
                issues.append(
                    "bedroom plausibility: wardrobe is floating away from walls "
                    f"(nearest wall distance {nearest_wall_distance:.2f}m)"
                )
                penalty += 0.06

    penalty = min(penalty, _cfg_float(cfg, "max_plausibility_penalty", 0.35))
    score = max(0.0, 1.0 - penalty)
    return BedroomPlausibilityReport(
        score=score,
        penalty=penalty,
        issues=issues,
        anchor_plan=plan,
    )
