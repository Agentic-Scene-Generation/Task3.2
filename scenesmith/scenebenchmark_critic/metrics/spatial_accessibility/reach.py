from __future__ import annotations

from typing import Any

import numpy as np

from scenesmith.scenebenchmark_critic.core.geometry import (
    bbox_height_span,
    front_vector,
    object_bbox,
)


def _min_reach_distance(
    subject: dict[str, Any],
    affordance: str,
    xs: np.ndarray,
    ys: np.ndarray,
    stance_mask: np.ndarray,
    profile: dict[str, Any],
) -> dict[str, Any]:
    rows, cols = np.where(stance_mask)
    if len(rows) == 0:
        return _reach_solution(float("inf"), None, None)
    points = _target_reach_points(subject, affordance)
    if not points:
        return _reach_solution(float("inf"), None, None)
    best = float("inf")
    best_posture: str | None = None
    best_origin_height: float | None = None
    planar_only = affordance == "sittable"
    postures = _posture_origin_heights(profile)
    for tx, ty, tz in points:
        dx = xs[rows, cols] - tx
        dy = ys[rows, cols] - ty
        if planar_only:
            distances = np.sqrt(dx * dx + dy * dy)
            if len(distances):
                candidate = float(np.min(distances))
                if candidate < best:
                    best = candidate
                    best_posture = "seated"
                    best_origin_height = None
            continue
        for posture, arm_z in postures:
            dz = arm_z - tz
            distances = np.sqrt(dx * dx + dy * dy + dz * dz)
            if len(distances):
                candidate = float(np.min(distances))
                if candidate < best:
                    best = candidate
                    best_posture = posture
                    best_origin_height = arm_z
    return _reach_solution(best, best_posture, best_origin_height)


def _reach_solution(
    distance: float, posture: str | None, origin_height: float | None
) -> dict[str, Any]:
    return {
        "min_reach_distance_m": distance,
        "reach_posture": posture,
        "reach_origin_height_m": origin_height,
    }


def _posture_origin_heights(profile: dict[str, Any]) -> list[tuple[str, float]]:
    standing_height = float(profile["arm_origin_height_m"])
    crouch_factor = float(profile.get("crouch_factor") or 0.0)
    if crouch_factor <= 0.0:
        return [("standing", standing_height)]
    crouch_height = standing_height * (1.0 - crouch_factor)
    if abs(crouch_height - standing_height) < 1e-6:
        return [("standing", standing_height)]
    return [("standing", standing_height), ("crouch", crouch_height)]


def _target_reach_points(
    subject: dict[str, Any], affordance: str
) -> list[tuple[float, float, float]]:
    faces = [
        face
        for face in (subject.get("interaction_faces") or [])
        if isinstance(face, dict)
        and (
            not affordance
            or affordance in set(face.get("affordances") or [])
            or face.get("name") == "front"
        )
    ]
    points: list[tuple[float, float, float]] = []
    for face in faces:
        center = face.get("center") or []
        if isinstance(center, list) and len(center) >= 3:
            points.append(
                (
                    float(center[0]),
                    float(center[1]),
                    _interaction_z(subject, affordance, float(center[2])),
                )
            )
    if points:
        return points

    bbox = object_bbox(subject) or {}
    center = bbox.get("center") or []
    size = bbox.get("size") or []
    if len(center) < 3 or len(size) < 3:
        return []
    cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
    sx, sy = max(float(size[0]), 0.2), max(float(size[1]), 0.2)
    fx, fy = front_vector(subject)
    px, py = -fy, fx
    z = _interaction_z(subject, affordance, cz)
    front = (cx + fx * sy / 2.0, cy + fy * sy / 2.0, z)
    if affordance == "openable":
        return [front]
    return [
        front,
        (cx + px * sx / 2.0, cy + py * sx / 2.0, z),
        (cx - px * sx / 2.0, cy - py * sx / 2.0, z),
    ]


def _interaction_z(subject: dict[str, Any], affordance: str, fallback: float) -> float:
    heights = (
        subject.get("interaction_height_m")
        or (subject.get("functional_hints") or {}).get("interaction_height_m")
        or {}
    )
    value = heights.get(affordance) if isinstance(heights, dict) else None
    if isinstance(value, (int, float)):
        return float(value)
    z_span = bbox_height_span(subject)
    if z_span is None:
        return fallback
    z_min, z_max = z_span
    if affordance == "supportable":
        return z_max
    if affordance == "sittable":
        return min(max(z_min + (z_max - z_min) * 0.45, 0.35), 0.65)
    if affordance == "openable":
        return min(max((z_min + z_max) * 0.5, 0.7), 1.4)
    return fallback
