from __future__ import annotations

import math

from dataclasses import dataclass
from typing import Any

DECOR_CATEGORIES = {
    "art",
    "artwork",
    "clock",
    "detector",
    "frame",
    "mirror",
    "painting",
    "picture",
    "poster",
    "print",
    "smoke_detector",
    "vase",
    "wall_art",
    "wall_clock",
    "wall_light",
    "wall_sconce",
}

SMALL_OBJECT_CATEGORIES = {
    "apple",
    "bottle",
    "book",
    "bowl",
    "box",
    "brick",
    "candle",
    "car",
    "card",
    "coaster",
    "colander",
    "cup",
    "detector",
    "dish",
    "fork",
    "knife",
    "keyboard",
    "magazine",
    "mouse",
    "mug",
    "newspaper",
    "pan",
    "paper",
    "phone",
    "plate",
    "plaque",
    "pot",
    "remote",
    "s0",
    "s1",
    "s2",
    "s3",
    "s4",
    "s5",
    "smartphone",
    "spoon",
    "statue",
    "tablespoon",
    "toy",
    "tray",
    "tumbler",
    "tv_remote_control",
}

FLOOR_COVERING_CATEGORIES = {
    "bath_mat",
    "carpet",
    "doormat",
    "floor_mat",
    "mat",
    "rug",
    "runner_rug",
}


@dataclass(frozen=True)
class GeometryStore:
    raw: dict[str, Any]
    objects: dict[str, dict[str, Any]]
    rooms: list[dict[str, Any]]
    relations: list[dict[str, Any]]
    task_relation_graph: dict[str, Any]


def load_geometry(case_pack: dict[str, Any]) -> GeometryStore | None:
    raw = case_pack.get("scene_geometry")
    if not isinstance(raw, dict):
        return None
    objects = {
        str(obj.get("id")): obj
        for obj in raw.get("objects") or []
        if isinstance(obj, dict) and obj.get("id")
    }
    rooms = [room for room in raw.get("rooms") or [] if isinstance(room, dict)]
    return GeometryStore(
        raw=raw,
        objects=objects,
        rooms=rooms,
        relations=[rel for rel in raw.get("relations") or [] if isinstance(rel, dict)],
        task_relation_graph=raw.get("task_relation_graph") or {},
    )


def object_category(obj: dict[str, Any] | None) -> str:
    if not obj:
        return ""
    return str(obj.get("category_norm") or obj.get("category") or "").strip()


def object_affordances(obj: dict[str, Any] | None) -> set[str]:
    hints = (obj or {}).get("functional_hints") or {}
    return {
        str(item).strip()
        for item in (
            hints.get("functional_categories")
            or hints.get("candidate_affordances")
            or []
        )
        if str(item).strip()
    }


def object_bbox(obj: dict[str, Any] | None) -> dict[str, Any] | None:
    bbox = (obj or {}).get("bbox_world")
    return bbox if isinstance(bbox, dict) else None


def bbox_center_xy(obj: dict[str, Any] | None) -> tuple[float, float] | None:
    bbox = object_bbox(obj)
    center = (bbox or {}).get("center")
    if isinstance(center, list) and len(center) >= 2:
        return float(center[0]), float(center[1])
    return None


def bbox_min_max_xy(
    obj: dict[str, Any] | None,
) -> tuple[float, float, float, float] | None:
    bbox = object_bbox(obj)
    bmin = (bbox or {}).get("min")
    bmax = (bbox or {}).get("max")
    if (
        isinstance(bmin, list)
        and isinstance(bmax, list)
        and len(bmin) >= 2
        and len(bmax) >= 2
    ):
        return float(bmin[0]), float(bmin[1]), float(bmax[0]), float(bmax[1])
    return None


def object_footprint_polygon(
    obj: dict[str, Any] | None,
) -> list[tuple[float, float]] | None:
    raw = (obj or {}).get("footprint_world")
    if isinstance(raw, list) and len(raw) >= 3:
        points: list[tuple[float, float]] = []
        for point in raw:
            if isinstance(point, list) and len(point) >= 2:
                points.append((float(point[0]), float(point[1])))
            elif isinstance(point, tuple) and len(point) >= 2:
                points.append((float(point[0]), float(point[1])))
        if len(points) >= 3:
            return points

    bbox = object_bbox(obj)
    center = (bbox or {}).get("center")
    size = (bbox or {}).get("size")
    if (
        isinstance(center, list)
        and isinstance(size, list)
        and len(center) >= 2
        and len(size) >= 2
    ):
        cx, cy = float(center[0]), float(center[1])
        sx, sy = max(float(size[0]), 0.0), max(float(size[1]), 0.0)
        yaw = yaw_rad(obj)
        fx, fy = math.cos(yaw), math.sin(yaw)
        px, py = -fy, fx
        corners: list[tuple[float, float]] = []
        for du, dv in ((-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)):
            corners.append(
                (cx + du * sx * fx + dv * sy * px, cy + du * sx * fy + dv * sy * py)
            )
        if sx > 0.0 and sy > 0.0:
            return corners

    bounds = bbox_min_max_xy(obj)
    if bounds is None:
        return None
    x0, y0, x1, y1 = bounds
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def polygon_bounds_xy(
    polygon: list[tuple[float, float]],
) -> tuple[float, float, float, float]:
    return (
        min(x for x, _ in polygon),
        min(y for _, y in polygon),
        max(x for x, _ in polygon),
        max(y for _, y in polygon),
    )


def point_in_polygon_xy(x: float, y: float, polygon: list[tuple[float, float]]) -> bool:
    inside = False
    j = len(polygon) - 1
    for i, point in enumerate(polygon):
        xi, yi = point
        xj, yj = polygon[j]
        intersects = (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (
            (yj - yi) or 1e-9
        ) + xi
        if intersects:
            inside = not inside
        j = i
    return inside


def nav_obstacle_class(obj: dict[str, Any] | None, *, height_threshold_m: float) -> str:
    raw = str((obj or {}).get("nav_obstacle_class") or "").strip()
    if not obj:
        return "small_ignore"
    category = object_category(obj)
    hints = obj.get("functional_hints") or {}
    group = str(hints.get("category_group") or "")
    if is_floor_covering(obj):
        return "low_ignore"
    if raw in {
        "blocking",
        "low_ignore",
        "mounted_ignore",
        "small_ignore",
        "decor_ignore",
    }:
        return raw
    z_span = bbox_height_span(obj)
    if group == "decor" or category in DECOR_CATEGORIES:
        return "decor_ignore"
    if z_span is None:
        return "small_ignore"
    z_min, z_max = z_span
    if z_min > max(0.25, height_threshold_m * 0.2):
        return "mounted_ignore"
    if is_small_object(obj):
        return "small_ignore"
    if z_max < 0.45:
        return "low_ignore"
    if z_min > 0.25:
        return "mounted_ignore"
    return "blocking"


def bbox_height_span(obj: dict[str, Any] | None) -> tuple[float, float] | None:
    bbox = object_bbox(obj)
    bmin = (bbox or {}).get("min")
    bmax = (bbox or {}).get("max")
    if (
        isinstance(bmin, list)
        and isinstance(bmax, list)
        and len(bmin) >= 3
        and len(bmax) >= 3
    ):
        return float(bmin[2]), float(bmax[2])
    return None


def yaw_rad(obj: dict[str, Any] | None) -> float:
    return math.radians(float((obj or {}).get("yaw_deg") or 0.0))


def front_vector(obj: dict[str, Any] | None) -> tuple[float, float]:
    yaw = yaw_rad(obj)
    # SceneSmith canonical convention: yaw=0 means front points along +Y.
    base = (-math.sin(yaw), math.cos(yaw))
    face = _horizontal_front_face(obj)
    return _face_dir_from_base(base, face)


def side_vector(obj: dict[str, Any] | None) -> tuple[float, float]:
    fx, fy = front_vector(obj)
    return -fy, fx


def distance_xy(a: dict[str, Any] | None, b: dict[str, Any] | None) -> float | None:
    ca = bbox_center_xy(a)
    cb = bbox_center_xy(b)
    if ca is None or cb is None:
        return None
    return math.hypot(ca[0] - cb[0], ca[1] - cb[1])


def bbox_gap_xy(a: dict[str, Any] | None, b: dict[str, Any] | None) -> float | None:
    ab = bbox_min_max_xy(a)
    bb = bbox_min_max_xy(b)
    if ab is None or bb is None:
        return None
    ax0, ay0, ax1, ay1 = ab
    bx0, by0, bx1, by1 = bb
    gap_x = max(0.0, max(bx0 - ax1, ax0 - bx1))
    gap_y = max(0.0, max(by0 - ay1, ay0 - by1))
    return math.hypot(gap_x, gap_y)


def angle_to_target_deg(
    subject: dict[str, Any], target: dict[str, Any]
) -> float | None:
    sc = bbox_center_xy(subject)
    tc = bbox_center_xy(target)
    return _angle_to_point_deg(subject, sc, tc)


def _angle_to_point_deg(
    subject: dict[str, Any],
    subject_center: tuple[float, float] | None,
    target_point: tuple[float, float] | None,
) -> float | None:
    sc = subject_center
    tc = target_point
    if sc is None or tc is None:
        return None
    vx, vy = front_vector(subject)
    tx, ty = tc[0] - sc[0], tc[1] - sc[1]
    norm = math.hypot(tx, ty)
    if norm <= 1e-6:
        return 0.0
    dot = max(-1.0, min(1.0, (vx * tx + vy * ty) / norm))
    return abs(math.degrees(math.acos(dot)))


def seating_angle_to_target_deg(
    subject: dict[str, Any], target: dict[str, Any]
) -> tuple[float | None, str]:
    subject_center = bbox_center_xy(subject)
    target_point, target_mode = _seating_target_point_xy(
        subject, target, subject_center=subject_center
    )
    primary_angle = _angle_to_point_deg(subject, subject_center, target_point)
    if primary_angle is None:
        return None, "missing"

    if _should_use_seating_depth_axis_fallback(subject):
        front_extent = _footprint_extent_along(subject, front_vector(subject))
        side_extent = _footprint_extent_along(subject, side_vector(subject))
        if (
            front_extent is not None
            and side_extent is not None
            and side_extent > front_extent * 1.1
        ):
            sc = subject_center
            tc = target_point
            if sc is not None and tc is not None:
                tx, ty = tc[0] - sc[0], tc[1] - sc[1]
                norm = math.hypot(tx, ty)
                if norm > 1e-6:
                    sx, sy = side_vector(subject)
                    candidates = [(sx, sy), (-sx, -sy)]
                    fallback_angle = min(
                        _angle_between_axis_and_target(axis, (tx, ty), norm)
                        for axis in candidates
                    )
                    if fallback_angle + 1e-6 < primary_angle:
                        return fallback_angle, "depth_axis_fallback"

    reversed_angle = _reversed_front_axis_angle_deg(
        subject, subject_center, target_point
    )
    if _should_use_reversed_front_fallback(
        subject,
        target,
        target_mode=target_mode,
        primary_angle=primary_angle,
        reversed_angle=reversed_angle,
    ):
        return reversed_angle, "reversed_front_fallback"
    return primary_angle, target_mode


def _seating_target_point_xy(
    subject: dict[str, Any],
    target: dict[str, Any],
    *,
    subject_center: tuple[float, float] | None,
) -> tuple[tuple[float, float] | None, str]:
    target_center = bbox_center_xy(target)
    if not _should_use_seating_target_edge_fallback(subject, target):
        return target_center, "front"
    target_polygon = object_footprint_polygon(target)
    if not target_polygon or subject_center is None:
        return target_center, "front"
    front_hit = _front_ray_target_surface_point_xy(
        subject, target, subject_center, target_polygon
    )
    if front_hit is not None:
        return front_hit, "front_ray_surface"
    nearest_point, nearest_mode = _nearest_seating_surface_point_xy(
        subject_center, target_polygon
    )
    if nearest_point is None:
        return target_center, "front"
    return nearest_point, nearest_mode


def _should_use_seating_target_edge_fallback(
    subject: dict[str, Any], target: dict[str, Any]
) -> bool:
    if bbox_gap_xy(subject, target) is None or bbox_gap_xy(subject, target) > 0.25:
        return False
    if object_category(target) not in {
        "bar_table",
        "coffee_table",
        "counter",
        "desk",
        "dining_table",
        "table",
    }:
        return False
    target_polygon = object_footprint_polygon(target)
    if not target_polygon:
        return False
    bounds = polygon_bounds_xy(target_polygon)
    width = max(bounds[2] - bounds[0], 0.0)
    depth = max(bounds[3] - bounds[1], 0.0)
    long_side = max(width, depth)
    return long_side >= 0.45


def _front_ray_target_surface_point_xy(
    subject: dict[str, Any],
    target: dict[str, Any],
    subject_center: tuple[float, float],
    target_polygon: list[tuple[float, float]],
) -> tuple[float, float] | None:
    category = object_category(target)
    if category not in {"bar_table", "coffee_table", "desk", "dining_table", "table"}:
        return None
    target_center = bbox_center_xy(target)
    if target_center is None:
        return None
    fx, fy = front_vector(subject)
    tx, ty = target_center[0] - subject_center[0], target_center[1] - subject_center[1]
    target_dist = math.hypot(tx, ty)
    if target_dist > 1e-6:
        alignment = (fx * tx + fy * ty) / target_dist
        if alignment < -0.15:
            return None
    return _ray_polygon_boundary_intersection_xy(
        subject_center, (fx, fy), target_polygon
    )


def _ray_polygon_boundary_intersection_xy(
    origin: tuple[float, float],
    direction: tuple[float, float],
    polygon: list[tuple[float, float]],
) -> tuple[float, float] | None:
    ox, oy = origin
    dx, dy = direction
    direction_norm = math.hypot(dx, dy)
    if direction_norm <= 1e-9:
        return None
    dx, dy = dx / direction_norm, dy / direction_norm
    best_t: float | None = None
    best_point: tuple[float, float] | None = None
    for index, start in enumerate(polygon):
        end = polygon[(index + 1) % len(polygon)]
        sx, sy = start
        ex, ey = end
        vx, vy = ex - sx, ey - sy
        denom = _cross_xy((dx, dy), (vx, vy))
        if abs(denom) <= 1e-9:
            continue
        rel = (sx - ox, sy - oy)
        t = _cross_xy(rel, (vx, vy)) / denom
        u = _cross_xy(rel, (dx, dy)) / denom
        if t <= 1e-6 or u < -1e-6 or u > 1.0 + 1e-6:
            continue
        if best_t is None or t < best_t:
            best_t = t
            best_point = (ox + t * dx, oy + t * dy)
    return best_point


def _cross_xy(a: tuple[float, float], b: tuple[float, float]) -> float:
    return a[0] * b[1] - a[1] * b[0]


def _nearest_point_on_polygon_boundary_xy(
    point: tuple[float, float],
    polygon: list[tuple[float, float]],
) -> tuple[float, float] | None:
    nearest, _dist_sq = _nearest_point_on_polygon_boundary_with_distance_xy(
        point, polygon
    )
    return nearest


def _nearest_point_on_polygon_boundary_with_distance_xy(
    point: tuple[float, float],
    polygon: list[tuple[float, float]],
) -> tuple[tuple[float, float] | None, float | None]:
    if len(polygon) < 2:
        return None, None
    px, py = point
    best_point: tuple[float, float] | None = None
    best_dist_sq: float | None = None
    for index, start in enumerate(polygon):
        end = polygon[(index + 1) % len(polygon)]
        nearest = _nearest_point_on_segment_xy((px, py), start, end)
        dist_sq = (nearest[0] - px) ** 2 + (nearest[1] - py) ** 2
        if best_dist_sq is None or dist_sq < best_dist_sq:
            best_dist_sq = dist_sq
            best_point = nearest
    return best_point, best_dist_sq


def _nearest_seating_surface_point_xy(
    point: tuple[float, float],
    polygon: list[tuple[float, float]],
) -> tuple[tuple[float, float] | None, str]:
    nearest_point, nearest_dist_sq = (
        _nearest_point_on_polygon_boundary_with_distance_xy(point, polygon)
    )
    if len(polygon) < 2 or nearest_point is None:
        return nearest_point, "nearest_surface"

    segment_lengths: list[tuple[float, tuple[float, float], tuple[float, float]]] = []
    for index, start in enumerate(polygon):
        end = polygon[(index + 1) % len(polygon)]
        length = math.hypot(end[0] - start[0], end[1] - start[1])
        if length > 1e-6:
            segment_lengths.append((length, start, end))
    if not segment_lengths:
        return nearest_point, "nearest_surface"

    longest = max(length for length, _start, _end in segment_lengths)
    shortest = min(length for length, _start, _end in segment_lengths)
    if longest / max(shortest, 1e-6) < 1.35:
        return nearest_point, "nearest_surface"

    px, py = point
    best_long_point: tuple[float, float] | None = None
    best_long_dist_sq: float | None = None
    for length, start, end in segment_lengths:
        if length < longest * 0.65:
            continue
        candidate = _nearest_point_on_segment_xy(point, start, end)
        dist_sq = (candidate[0] - px) ** 2 + (candidate[1] - py) ** 2
        if best_long_dist_sq is None or dist_sq < best_long_dist_sq:
            best_long_dist_sq = dist_sq
            best_long_point = candidate

    if best_long_point is None:
        return nearest_point, "nearest_surface"
    if nearest_dist_sq is None or best_long_dist_sq is None:
        return best_long_point, "nearest_long_surface"
    if math.sqrt(best_long_dist_sq) <= math.sqrt(nearest_dist_sq) + 0.35:
        return best_long_point, "nearest_long_surface"
    return nearest_point, "nearest_surface"


def _nearest_point_on_segment_xy(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> tuple[float, float]:
    px, py = point
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    denom = dx * dx + dy * dy
    if denom <= 1e-9:
        return start
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / denom))
    return x1 + t * dx, y1 + t * dy


def _should_use_seating_depth_axis_fallback(obj: dict[str, Any] | None) -> bool:
    if not obj:
        return False
    category = object_category(obj)
    if category not in {
        "chair",
        "dining_chair",
        "office_chair",
        "armchair",
        "bench",
        "stool",
    }:
        return False
    hints = obj.get("functional_hints") or {}
    source = str(hints.get("classification_source") or "").strip().lower()
    return source in {"", "heuristic"}


def _should_use_reversed_front_fallback(
    subject: dict[str, Any],
    target: dict[str, Any],
    *,
    target_mode: str,
    primary_angle: float,
    reversed_angle: float | None,
) -> bool:
    if target_mode not in {"nearest_surface", "nearest_long_surface"}:
        return False
    if primary_angle < 150.0 or reversed_angle is None or reversed_angle > 30.0:
        return False
    gap = bbox_gap_xy(subject, target)
    if gap is None or gap > 0.25:
        return False
    hints = subject.get("functional_hints") or {}
    source = str(hints.get("classification_source") or "").strip().lower()
    if source not in {"asset_annotation", "heuristic"}:
        return False
    if object_category(subject) not in {
        "chair",
        "dining_chair",
        "office_chair",
        "armchair",
        "bench",
        "stool",
    }:
        return False
    front_hint = str(hints.get("front_hint") or "").strip().lower()
    surface_map = hints.get("interaction_surface_map") or {}
    front_terms = " ".join(
        str(value or "").strip().lower() for value in (surface_map.get("front") or [])
    )
    if front_hint == "back" or "backrest" in front_terms:
        return True
    if gap <= 0.02 and any(
        term in front_terms for term in ("seat", "front edge", "seat edge")
    ):
        return True
    return False


def _reversed_front_axis_angle_deg(
    subject: dict[str, Any],
    subject_center: tuple[float, float] | None,
    target_point: tuple[float, float] | None,
) -> float | None:
    sc = subject_center
    tc = target_point
    if sc is None or tc is None:
        return None
    vx, vy = front_vector(subject)
    tx, ty = tc[0] - sc[0], tc[1] - sc[1]
    norm = math.hypot(tx, ty)
    if norm <= 1e-6:
        return 0.0
    dot = max(-1.0, min(1.0, ((-vx) * tx + (-vy) * ty) / norm))
    return abs(math.degrees(math.acos(dot)))


def _footprint_extent_along(
    obj: dict[str, Any] | None, axis: tuple[float, float]
) -> float | None:
    polygon = object_footprint_polygon(obj)
    if not polygon:
        return None
    length = math.hypot(axis[0], axis[1])
    if length <= 1e-6:
        return None
    ux, uy = axis[0] / length, axis[1] / length
    projections = [x * ux + y * uy for x, y in polygon]
    if not projections:
        return None
    return max(projections) - min(projections)


def _angle_between_axis_and_target(
    axis: tuple[float, float],
    target_vec: tuple[float, float],
    target_norm: float,
) -> float:
    dot = max(
        -1.0,
        min(1.0, (axis[0] * target_vec[0] + axis[1] * target_vec[1]) / target_norm),
    )
    return abs(math.degrees(math.acos(dot)))


def _horizontal_front_face(obj: dict[str, Any] | None) -> str | None:
    hints = (obj or {}).get("functional_hints") or {}
    face = str(hints.get("front_hint") or "").strip().lower()
    if face in {"front", "back", "left", "right"}:
        return face
    return None


def _face_dir_from_base(
    base: tuple[float, float], face: str | None
) -> tuple[float, float]:
    fx, fy = base
    if face == "back":
        return -fx, -fy
    if face == "left":
        return -fy, fx
    if face == "right":
        return fy, -fx
    return fx, fy


def floor_polygon_for_object(
    store: GeometryStore, obj: dict[str, Any] | None = None
) -> list[tuple[float, float]] | None:
    room_id = str((obj or {}).get("room") or (obj or {}).get("room_id") or "")
    rooms = store.rooms
    if room_id:
        rooms = [
            room for room in store.rooms if str(room.get("id") or "") == room_id
        ] or store.rooms
    for room in rooms:
        polygon = room.get("floor_polygon")
        if isinstance(polygon, list) and len(polygon) >= 3:
            return [
                (float(point[0]), float(point[1]))
                for point in polygon
                if isinstance(point, list) and len(point) >= 2
            ]
        bbox = room.get("bbox") or {}
        bmin = bbox.get("min")
        bmax = bbox.get("max")
        if (
            isinstance(bmin, list)
            and isinstance(bmax, list)
            and len(bmin) >= 2
            and len(bmax) >= 2
        ):
            x0, y0 = float(bmin[0]), float(bmin[1])
            x1, y1 = float(bmax[0]), float(bmax[1])
            return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return None


def is_decor_or_high_mounted(obj: dict[str, Any], *, height_threshold_m: float) -> bool:
    category = object_category(obj)
    hints = obj.get("functional_hints") or {}
    group = str(hints.get("category_group") or "")
    z_span = bbox_height_span(obj)
    if group == "decor" or category in DECOR_CATEGORIES:
        if z_span is None:
            return True
        return z_span[0] > 0.5
    if z_span and z_span[0] > height_threshold_m:
        return True
    return False


def footprint_size_xy(obj: dict[str, Any] | None) -> tuple[float, float] | None:
    bbox = object_bbox(obj)
    size = (bbox or {}).get("size")
    if isinstance(size, list) and len(size) >= 2:
        return float(size[0]), float(size[1])
    return None


def footprint_area_xy(obj: dict[str, Any] | None) -> float | None:
    size = footprint_size_xy(obj)
    if size is None:
        return None
    return max(size[0], 0.0) * max(size[1], 0.0)


def is_small_object(obj: dict[str, Any] | None) -> bool:
    category = object_category(obj)
    if category in SMALL_OBJECT_CATEGORIES:
        return True
    affordances = object_affordances(obj)
    if affordances == {"graspable"}:
        return True
    area = footprint_area_xy(obj)
    z_span = bbox_height_span(obj)
    if area is not None and z_span is not None:
        height = z_span[1] - z_span[0]
        if area <= 0.12 and height <= 0.5:
            return True
    return False


def is_floor_covering(obj: dict[str, Any] | None) -> bool:
    if not obj:
        return False
    category = object_category(obj).lower()
    hints = obj.get("functional_hints") or {}
    group = str(hints.get("category_group") or "").strip().lower()
    if group == "soft_furnishing" and any(
        token in category for token in FLOOR_COVERING_CATEGORIES
    ):
        return True
    if category in FLOOR_COVERING_CATEGORIES:
        return True
    for keyword in hints.get("category_keywords") or []:
        text = str(keyword).strip().lower()
        if text in FLOOR_COVERING_CATEGORIES:
            return True
    return False


def is_walkway_obstacle(
    obj: dict[str, Any] | None, *, height_threshold_m: float
) -> bool:
    return nav_obstacle_class(obj, height_threshold_m=height_threshold_m) == "blocking"
