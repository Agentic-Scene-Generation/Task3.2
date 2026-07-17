from __future__ import annotations

import math

from typing import Any

import numpy as np

from scenesmith.scenebenchmark_critic.core.geometry import (
    GeometryStore,
    bbox_min_max_xy,
    front_vector,
    is_walkway_obstacle,
    object_affordances,
    object_bbox,
)


def _access_zones(
    subject: dict[str, Any],
    affordance: str,
    depth: float,
) -> list[tuple[str, dict[str, Any]]]:
    bbox = object_bbox(subject) or {}
    center = bbox.get("center") or []
    size = bbox.get("size") or []
    if len(center) < 2 or len(size) < 2:
        return []
    cx, cy = float(center[0]), float(center[1])
    sx, sy = max(float(size[0]), 0.2), max(float(size[1]), 0.2)
    long_span = max(sx, sy)
    short_span = min(sx, sy)
    fx, fy = front_vector(subject)
    px, py = -fy, fx

    def zone(
        name: str,
        dx: float,
        dy: float,
        ux: tuple[float, float],
        vx: tuple[float, float],
        half_u: float,
        half_v: float,
    ):
        return (
            name,
            {
                "center": (cx + dx, cy + dy),
                "u": ux,
                "v": vx,
                "half_u": half_u,
                "half_v": half_v,
            },
        )

    front_offset = short_span * 0.5 + depth * 0.5
    side_offset = long_span * 0.5 + depth * 0.5
    front = zone(
        "front",
        fx * front_offset,
        fy * front_offset,
        (px, py),
        (fx, fy),
        long_span * 0.55,
        depth * 0.5,
    )
    if affordance == "openable":
        return [front]
    if affordance in {"sittable", "supportable", "graspable", ""} or object_affordances(
        subject
    ):
        left = zone(
            "left",
            px * side_offset,
            py * side_offset,
            (fx, fy),
            (px, py),
            short_span * 0.65,
            depth * 0.5,
        )
        right = zone(
            "right",
            -px * side_offset,
            -py * side_offset,
            (fx, fy),
            (px, py),
            short_span * 0.65,
            depth * 0.5,
        )
        return [front, left, right]
    return [front]


def _oriented_rect_mask(
    xs: np.ndarray, ys: np.ndarray, zone: dict[str, Any]
) -> np.ndarray:
    cx, cy = zone["center"]
    ux, uy = zone["u"]
    vx, vy = zone["v"]
    dx = xs - cx
    dy = ys - cy
    u_proj = dx * ux + dy * uy
    v_proj = dx * vx + dy * vy
    return (np.abs(u_proj) <= zone["half_u"]) & (np.abs(v_proj) <= zone["half_v"])


def _zone_aabb(zone: dict[str, Any]) -> tuple[float, float, float, float]:
    cx, cy = zone["center"]
    ux, uy = zone["u"]
    vx, vy = zone["v"]
    corners = []
    for su in (-1.0, 1.0):
        for sv in (-1.0, 1.0):
            corners.append(
                (
                    cx + su * zone["half_u"] * ux + sv * zone["half_v"] * vx,
                    cy + su * zone["half_u"] * uy + sv * zone["half_v"] * vy,
                )
            )
    return (
        min(x for x, _ in corners),
        min(y for _, y in corners),
        max(x for x, _ in corners),
        max(y for _, y in corners),
    )


def _blocking_objects_for_zone(
    store: GeometryStore,
    subject_id: str,
    zone_aabb: tuple[float, float, float, float],
    *,
    limit: int,
    height_threshold_m: float,
    ignored_object_ids: set[str] | None = None,
) -> list[str]:
    zx0, zy0, zx1, zy1 = zone_aabb
    scored: list[tuple[float, str]] = []
    ignored = ignored_object_ids or set()
    for obj_id, obj in store.objects.items():
        if obj_id == subject_id or obj_id in ignored:
            continue
        if not is_walkway_obstacle(obj, height_threshold_m=height_threshold_m):
            continue
        bounds = bbox_min_max_xy(obj)
        if bounds is None:
            continue
        ox0, oy0, ox1, oy1 = bounds
        gap_x = max(0.0, max(ox0 - zx1, zx0 - ox1))
        gap_y = max(0.0, max(oy0 - zy1, zy0 - oy1))
        gap = math.hypot(gap_x, gap_y)
        if gap <= 0.35:
            scored.append((gap, obj_id))
    scored.sort(key=lambda item: (item[0], item[1]))
    return [obj_id for _, obj_id in scored[:limit]]
