"""Deterministic geometry helpers for the single-room polygon mode.

This module is the source of truth for polygon validation and derived geometry.
Callers must not repair invalid input implicitly: invalid polygons are returned to
the designer with an actionable error instead.
"""

import math

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from shapely.geometry import MultiPoint, Polygon


Vertex2D = tuple[float, float]


@dataclass(frozen=True)
class PolygonValidationConfig:
    """Validation limits for polygon room footprints."""

    max_vertices: int = 32
    min_area_m2: float = 4.0
    min_edge_length_m: float = 0.8
    min_interior_angle_deg: float = 20.0
    coordinate_precision: int = 4
    min_dimension_m: float = 1.5
    max_dimension_m: float = 20.0


@dataclass(frozen=True)
class PolygonEdge:
    """Geometry derived from one canonical counter-clockwise polygon edge."""

    index: int
    start: Vertex2D
    end: Vertex2D
    length: float
    tangent: Vertex2D
    inward_normal: Vertex2D
    yaw: float


class PolygonValidationError(ValueError):
    """Raised when a polygon footprint violates the public input contract."""


def _signed_area(vertices: Sequence[Vertex2D]) -> float:
    return 0.5 * sum(
        x0 * y1 - x1 * y0
        for (x0, y0), (x1, y1) in zip(vertices, vertices[1:] + vertices[:1])
    )


def polygon_aabb(vertices: Sequence[Vertex2D]) -> tuple[float, float, float, float]:
    """Return ``(min_x, min_y, max_x, max_y)`` for a non-empty footprint."""
    if not vertices:
        raise ValueError("Cannot compute an AABB for an empty polygon.")
    xs = [vertex[0] for vertex in vertices]
    ys = [vertex[1] for vertex in vertices]
    return min(xs), min(ys), max(xs), max(ys)


def canonicalize_polygon(
    vertices: Sequence[Sequence[float]],
    config: PolygonValidationConfig | None = None,
    *,
    normalize_aabb_min: bool = True,
) -> list[Vertex2D]:
    """Validate and canonicalize a simple polygon room footprint.

    The returned vertices are quantized, counter-clockwise, start at the
    lexicographically smallest vertex, and (for design inputs) have their AABB
    minimum translated to the origin.
    """
    cfg = config or PolygonValidationConfig()
    if not isinstance(vertices, Sequence) or isinstance(vertices, (str, bytes)):
        raise PolygonValidationError("vertices must be a JSON array of [x, y] pairs.")

    points: list[Vertex2D] = []
    for index, raw in enumerate(vertices):
        if (
            not isinstance(raw, Sequence)
            or isinstance(raw, (str, bytes))
            or len(raw) != 2
        ):
            raise PolygonValidationError(
                f"Vertex {index} must contain exactly two coordinates [x, y]."
            )
        try:
            x, y = float(raw[0]), float(raw[1])
        except (TypeError, ValueError) as exc:
            raise PolygonValidationError(
                f"Vertex {index} contains a non-numeric coordinate."
            ) from exc
        if not math.isfinite(x) or not math.isfinite(y):
            raise PolygonValidationError(
                f"Vertex {index} coordinates must be finite numbers."
            )
        point = (round(x, cfg.coordinate_precision), round(y, cfg.coordinate_precision))
        if not points or point != points[-1]:
            points.append(point)

    if len(points) > 1 and points[0] == points[-1]:
        points.pop()
    if len(points) < 3:
        raise PolygonValidationError("A polygon needs at least 3 distinct vertices.")

    # Strictly collinear intermediate vertices do not define separate walls.
    changed = True
    epsilon = 10 ** (-(cfg.coordinate_precision + 1))
    while changed and len(points) >= 3:
        changed = False
        filtered: list[Vertex2D] = []
        for index, current in enumerate(points):
            previous = points[index - 1]
            following = points[(index + 1) % len(points)]
            cross = (current[0] - previous[0]) * (following[1] - current[1]) - (
                current[1] - previous[1]
            ) * (following[0] - current[0])
            if abs(cross) <= epsilon:
                changed = True
                continue
            filtered.append(current)
        points = filtered

    if len(points) < 3:
        raise PolygonValidationError("Polygon vertices are collinear or degenerate.")
    if len(points) > cfg.max_vertices:
        raise PolygonValidationError(
            f"Polygon has {len(points)} vertices; maximum is {cfg.max_vertices}."
        )

    polygon = Polygon(points)
    if not polygon.is_valid or not polygon.exterior.is_simple:
        raise PolygonValidationError(
            "Polygon is self-intersecting or otherwise invalid; provide a simple boundary."
        )
    if polygon.is_empty or polygon.area <= 0:
        raise PolygonValidationError("Polygon area must be greater than zero.")
    if polygon.interiors:
        raise PolygonValidationError("Polygon holes are not supported.")
    if polygon.area < cfg.min_area_m2:
        raise PolygonValidationError(
            f"Polygon area must be at least {cfg.min_area_m2:g} m^2; got "
            f"{polygon.area:.3f} m^2."
        )

    min_x, min_y, max_x, max_y = polygon_aabb(points)
    width, depth = max_x - min_x, max_y - min_y
    for name, value in (("width", width), ("depth", depth)):
        if not cfg.min_dimension_m <= value <= cfg.max_dimension_m:
            raise PolygonValidationError(
                f"Polygon AABB {name} must be {cfg.min_dimension_m:g}-"
                f"{cfg.max_dimension_m:g} m; got {value:.3f} m."
            )

    for edge in polygon_edges(points):
        if edge.length < cfg.min_edge_length_m:
            raise PolygonValidationError(
                f"Edge {edge.index} is {edge.length:.3f} m; minimum is "
                f"{cfg.min_edge_length_m:g} m."
            )

    for index in range(len(points)):
        previous = np.asarray(points[index - 1], dtype=float) - np.asarray(
            points[index], dtype=float
        )
        following = np.asarray(
            points[(index + 1) % len(points)], dtype=float
        ) - np.asarray(points[index], dtype=float)
        cosine = float(
            np.clip(
                np.dot(previous, following)
                / (np.linalg.norm(previous) * np.linalg.norm(following)),
                -1.0,
                1.0,
            )
        )
        smaller_angle = math.degrees(math.acos(cosine))
        if smaller_angle < cfg.min_interior_angle_deg:
            raise PolygonValidationError(
                f"Vertex {index} angle is {smaller_angle:.2f} degrees; minimum is "
                f"{cfg.min_interior_angle_deg:g} degrees."
            )

    if normalize_aabb_min:
        points = [
            (
                round(x - min_x, cfg.coordinate_precision),
                round(y - min_y, cfg.coordinate_precision),
            )
            for x, y in points
        ]
    if _signed_area(points) < 0:
        points.reverse()

    start_index = min(range(len(points)), key=lambda index: points[index])
    return points[start_index:] + points[:start_index]


def polygon_edges(vertices: Sequence[Vertex2D]) -> list[PolygonEdge]:
    """Derive stable edge frames from counter-clockwise vertices."""
    edges: list[PolygonEdge] = []
    for index, (start, end) in enumerate(zip(vertices, vertices[1:] + vertices[:1])):
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = math.hypot(dx, dy)
        if length <= 0:
            raise PolygonValidationError(f"Edge {index} has zero length.")
        tangent = (dx / length, dy / length)
        edges.append(
            PolygonEdge(
                index=index,
                start=start,
                end=end,
                length=length,
                tangent=tangent,
                inward_normal=(-tangent[1], tangent[0]),
                yaw=math.atan2(tangent[1], tangent[0]),
            )
        )
    return edges


def to_room_local_vertices(vertices: Sequence[Vertex2D]) -> list[Vertex2D]:
    """Translate AABB-min design coordinates into AABB-centered room coordinates."""
    min_x, min_y, max_x, max_y = polygon_aabb(vertices)
    center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2
    return [(x - center_x, y - center_y) for x, y in vertices]


def triangulate_polygon(vertices: Sequence[Vertex2D]) -> list[tuple[int, int, int]]:
    """Triangulate a canonical simple polygon with deterministic ear clipping."""
    if len(vertices) < 3:
        raise PolygonValidationError("A polygon needs at least 3 vertices.")
    if _signed_area(vertices) < 0:
        raise PolygonValidationError(
            "Triangulation requires counter-clockwise vertices."
        )

    remaining = list(range(len(vertices)))
    triangles: list[tuple[int, int, int]] = []
    epsilon = 1e-10

    def point_in_triangle(
        point: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray
    ) -> bool:
        crosses = []
        for p0, p1 in ((a, b), (b, c), (c, a)):
            edge = p1 - p0
            rel = point - p0
            crosses.append(edge[0] * rel[1] - edge[1] * rel[0])
        return min(crosses) >= -epsilon

    while len(remaining) > 3:
        ear_found = False
        for offset, current in enumerate(remaining):
            previous = remaining[offset - 1]
            following = remaining[(offset + 1) % len(remaining)]
            a = np.asarray(vertices[previous], dtype=float)
            b = np.asarray(vertices[current], dtype=float)
            c = np.asarray(vertices[following], dtype=float)
            ab, bc = b - a, c - b
            if ab[0] * bc[1] - ab[1] * bc[0] <= epsilon:
                continue
            if any(
                point_in_triangle(np.asarray(vertices[index], dtype=float), a, b, c)
                for index in remaining
                if index not in (previous, current, following)
            ):
                continue
            triangles.append((previous, current, following))
            del remaining[offset]
            ear_found = True
            break
        if not ear_found:
            raise PolygonValidationError(
                "Polygon triangulation failed for this boundary."
            )

    triangles.append(tuple(remaining))
    return triangles


def polygon_covers_vertices(
    boundary_vertices: Sequence[Vertex2D], candidate_vertices: Sequence[Vertex2D]
) -> bool:
    """Return whether a boundary covers an entire candidate polygon."""
    return Polygon(boundary_vertices).covers(Polygon(candidate_vertices))


def transformed_bbox_footprint(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    transform,
    scale: float = 1.0,
) -> list[Vertex2D]:
    """Return the complete transformed XY footprint of an object-frame AABB."""
    corners = [
        np.array([x * scale, y * scale, z * scale])
        for x in (bbox_min[0], bbox_max[0])
        for y in (bbox_min[1], bbox_max[1])
        for z in (bbox_min[2], bbox_max[2])
    ]
    projected = [tuple((transform @ corner)[:2]) for corner in corners]
    hull = MultiPoint(projected).convex_hull
    if not isinstance(hull, Polygon):
        return projected[:4]
    return [tuple(point) for point in list(hull.exterior.coords)[:-1]]


def room_geometry_covers_object(
    room_geometry,
    scene_object,
    *,
    transform=None,
    bbox_scale: float = 1.0,
    margin: float = 0.0,
) -> bool:
    """Check a full object footprint against an exact polygon room boundary.

    Rectangle modes intentionally return True so their existing checks remain the
    only active implementation path.
    """
    if room_geometry is None:
        return True
    footprint_vertices = getattr(room_geometry, "footprint_vertices", None)
    # Only the concrete optional-list discriminator enables polygon behavior.
    # This also keeps legacy callers and lightweight test doubles on the exact
    # rectangle path instead of interpreting arbitrary attributes as geometry.
    if not isinstance(footprint_vertices, (list, tuple)):
        return True
    if scene_object.bbox_min is None or scene_object.bbox_max is None:
        return False
    boundary = Polygon(footprint_vertices)
    if margin:
        boundary = boundary.buffer(-margin, join_style="mitre")
    if boundary.is_empty:
        return False
    footprint = transformed_bbox_footprint(
        scene_object.bbox_min,
        scene_object.bbox_max,
        transform or scene_object.transform,
        scale=bbox_scale,
    )
    return bool(boundary.covers(Polygon(footprint)))


def exact_surface_covers_object(surface, scene_object) -> bool:
    """Check a complete transformed AABB footprint on an exact support surface."""
    if surface.exact_boundary_vertices is None:
        return True
    if scene_object.bbox_min is None or scene_object.bbox_max is None:
        return False
    world_footprint = transformed_bbox_footprint(
        scene_object.bbox_min,
        scene_object.bbox_max,
        scene_object.transform,
    )
    inverse = surface.transform.inverse()
    surface_footprint = [
        tuple((inverse @ np.array([x, y, surface.transform.translation()[2]]))[:2])
        for x, y in world_footprint
    ]
    return bool(
        Polygon(surface.exact_boundary_vertices).covers(Polygon(surface_footprint))
    )
