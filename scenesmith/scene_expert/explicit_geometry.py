"""Deterministic parsing and comparison for exact user-specified room geometry."""

from __future__ import annotations

import json
import math
import re

from collections.abc import Mapping, Sequence
from typing import Any

from scenesmith.scene_expert.schemas import ExplicitFloorGeometryPolicy


_RECTANGULAR_DIMENSIONS = re.compile(
    r"\bwidth\s*=\s*(?P<width>\d+(?:\.\d+)?)\s*m?\s*"
    r"(?:,|and)?\s*length\s*=\s*(?P<length>\d+(?:\.\d+)?)\s*m?",
    re.IGNORECASE,
)
_RECTANGULAR_MEASURES = re.compile(
    r"\brectangular\b[^.\n]{0,100}?\b(?:measures?|dimensions?)\s*"
    r"(?P<width>\d+(?:\.\d+)?)\s*m?\s*(?:by|x|×)\s*"
    r"(?P<length>\d+(?:\.\d+)?)\s*m?",
    re.IGNORECASE,
)
_DECLARED_AREA = re.compile(
    r"\b(?:expected\s+area|area)\s*(?:is|of|=|:)?\s*"
    r"(?:approximately|about|around)?\s*(?P<area>\d+(?:\.\d+)?)\s*"
    r"(?:m(?:²|2)|square\s+meters?)",
    re.IGNORECASE,
)


def polygon_area_m2(vertices: Sequence[Sequence[float]]) -> float:
    """Return the absolute shoelace area for a non-empty polygon boundary."""
    points = [(float(vertex[0]), float(vertex[1])) for vertex in vertices]
    if len(points) < 3:
        return 0.0
    return abs(
        0.5
        * sum(
            x0 * y1 - x1 * y0
            for (x0, y0), (x1, y1) in zip(points, points[1:] + points[:1])
        )
    )


def normalize_polygon_vertices(
    vertices: Sequence[Sequence[float]], *, precision: int = 4
) -> list[tuple[float, float]]:
    """Normalize a simple vertex loop for translation/orientation-safe matching.

    This deliberately does not apply floor-plan validity thresholds: it is used to
    compare a prompt contract with the generated result, not to silently repair a
    user polygon.
    """
    points: list[tuple[float, float]] = []
    for raw in vertices:
        if (
            not isinstance(raw, Sequence)
            or isinstance(raw, (str, bytes))
            or len(raw) != 2
        ):
            raise ValueError("Each polygon vertex must contain two numeric values.")
        x, y = float(raw[0]), float(raw[1])
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("Polygon vertices must be finite.")
        point = (round(x, precision), round(y, precision))
        if not points or point != points[-1]:
            points.append(point)
    if len(points) > 1 and points[0] == points[-1]:
        points.pop()
    if len(points) < 3 or polygon_area_m2(points) <= 0.0:
        raise ValueError("Polygon needs at least three non-collinear vertices.")

    min_x = min(point[0] for point in points)
    min_y = min(point[1] for point in points)
    points = [
        (round(x - min_x, precision), round(y - min_y, precision)) for x, y in points
    ]
    signed_area = 0.5 * sum(
        x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(points, points[1:] + points[:1])
    )
    if signed_area < 0.0:
        points.reverse()
    start = min(range(len(points)), key=lambda index: points[index])
    return points[start:] + points[:start]


def _json_array_after(text: str, start: int) -> list[Any] | None:
    """Parse the first JSON array beginning at or after ``start``."""
    array_start = text.find("[[", start)
    if array_start < 0:
        return None
    depth = 0
    for index in range(array_start, len(text)):
        char = text[index]
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                try:
                    value = json.loads(text[array_start : index + 1])
                except json.JSONDecodeError:
                    return None
                return value if isinstance(value, list) else None
    return None


def _vertices_from_prompt(prompt: str) -> list[tuple[float, float]] | None:
    for match in re.finditer(r"\bvertices?\b", prompt, re.IGNORECASE):
        raw = _json_array_after(prompt, match.end())
        if raw is None:
            continue
        try:
            return normalize_polygon_vertices(raw)
        except (TypeError, ValueError):
            continue
    return None


def _area_tolerance(expected: float, policy: Mapping[str, Any]) -> float:
    absolute = float(policy.get("absolute_area_tolerance_m2", 0.10))
    relative = float(policy.get("relative_area_tolerance", 0.01))
    return max(absolute, abs(expected) * relative)


def explicit_geometry_from_prompt(
    prompt: str,
    policy: Mapping[str, Any] | None = None,
) -> ExplicitFloorGeometryPolicy:
    """Return an auditable geometry policy only for conservative exact markers."""
    raw_policy = dict(policy or {})
    if not bool(raw_policy.get("enabled", True)):
        return ExplicitFloorGeometryPolicy()

    text = str(prompt or "")
    lowered = text.lower()
    area_policy = str(raw_policy.get("functional_zone_area", "advisory")).lower()
    if area_policy not in {"hard", "advisory"}:
        area_policy = "advisory"
    common = {
        "detected": True,
        "source": "explicit_prompt",
        "functional_zone_area_policy": area_policy,
        "verify_geometry_match": bool(raw_policy.get("verify_geometry_match", True)),
        "absolute_area_tolerance_m2": float(
            raw_policy.get("absolute_area_tolerance_m2", 0.10)
        ),
        "relative_area_tolerance": float(
            raw_policy.get("relative_area_tolerance", 0.01)
        ),
        "vertex_tolerance_m": float(raw_policy.get("vertex_tolerance_m", 0.01)),
    }

    dimensions = _RECTANGULAR_DIMENSIONS.search(text)
    if dimensions is None:
        dimensions = _RECTANGULAR_MEASURES.search(text)
    if dimensions is not None and "rectangular" in lowered:
        width = float(dimensions.group("width"))
        length = float(dimensions.group("length"))
        if width > 0.0 and length > 0.0:
            return ExplicitFloorGeometryPolicy(
                **common,
                mode="room",
                expected_width_m=width,
                expected_length_m=length,
                expected_area_m2=round(width * length, 4),
                evidence=[dimensions.group(0)],
            )

    vertices = _vertices_from_prompt(text)
    exact_boundary = any(
        marker in lowered
        for marker in (
            "exact boundary",
            "complete floor boundary",
            "vertices define the",
            "vertices must be used as-is",
        )
    )
    if vertices is None or not exact_boundary:
        return ExplicitFloorGeometryPolicy()
    computed_area = polygon_area_m2(vertices)
    if computed_area <= 0.0:
        return ExplicitFloorGeometryPolicy()
    declared = _DECLARED_AREA.search(text)
    evidence = ["ordered vertices define the exact boundary"]
    if declared is not None:
        declared_area = float(declared.group("area"))
        if abs(declared_area - computed_area) > _area_tolerance(
            declared_area, raw_policy
        ):
            return ExplicitFloorGeometryPolicy()
        evidence.append(declared.group(0))
    return ExplicitFloorGeometryPolicy(
        **common,
        mode="polygon",
        expected_area_m2=computed_area,
        expected_vertices=vertices,
        evidence=evidence,
    )
