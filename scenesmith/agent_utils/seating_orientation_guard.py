"""Deterministic seating orientation guard for furniture-stage scenes."""

from __future__ import annotations

import logging
import math

from dataclasses import dataclass

import numpy as np

from pydrake.math import RigidTransform, RollPitchYaw

from scenesmith.agent_utils.room import ObjectType, RoomScene, SceneObject
from scenesmith.utils.geometry_utils import compute_optimal_facing_yaw

console_logger = logging.getLogger(__name__)

SEATING_TOKENS = {
    "armchair",
    "bar_stool",
    "bench",
    "chair",
    "dining_chair",
    "loveseat",
    "office_chair",
    "sofa",
    "stool",
}
SURFACE_TOKENS = {
    "bar_table",
    "coffee_table",
    "counter",
    "desk",
    "dining_table",
    "island",
    "table",
    "work_surface",
}


@dataclass(frozen=True)
class SeatingOrientationFix:
    subject_id: str
    target_id: str
    old_yaw_deg: float
    new_yaw_deg: float
    angle_to_target_deg: float


def align_seating_to_nearest_surface(
    scene: RoomScene,
    *,
    max_target_distance_m: float = 2.0,
    repair_angle_threshold_deg: float = 120.0,
    wall_anchor_gap_ratio: float = 0.45,
    standalone_surface_gap_ratio: float = 0.5,
    wall_preference_margin_ratio: float = 0.2,
) -> list[SeatingOrientationFix]:
    """Rotate clearly backward seating toward its nearest functional surface."""
    furniture = [
        obj for obj in scene.objects.values() if obj.object_type == ObjectType.FURNITURE
    ]
    seating = [obj for obj in furniture if _is_seating(obj)]
    surfaces = [obj for obj in furniture if _is_functional_surface(obj)]
    fixes: list[SeatingOrientationFix] = []
    if not seating:
        return fixes

    for seat in seating:
        wall_target = _nearest_wall_anchor(
            seat,
            scene.get_objects_by_type(ObjectType.WALL),
            max_gap_ratio=wall_anchor_gap_ratio,
        )
        target = _nearest_surface(seat, surfaces, max_distance_m=max_target_distance_m)
        if target is None or (
            wall_target is not None
            and _is_standalone_wall_seating(
                seat,
                wall_target,
                target,
                surface_gap_ratio=standalone_surface_gap_ratio,
                wall_margin_ratio=wall_preference_margin_ratio,
            )
            and _is_wall_anchor_candidate(seat)
        ):
            if wall_target is None:
                continue
            old_rpy = RollPitchYaw(seat.transform.rotation())
            seat_center = seat.transform.translation()
            target_point = _wall_away_target_point(seat, wall_target)
            if target_point is None:
                continue
            new_yaw_deg = compute_optimal_facing_yaw(
                origin_a=seat_center,
                target_point=target_point,
            )
            # 2026-07-12 修改原因：独立墙边座椅不应依赖 guest/visitor 名称；
            # 当它本来就是靠墙摆放时，兜底为背靠最近墙面、前向室内，保证平行稳定。
            seat.transform = RigidTransform(
                rpy=RollPitchYaw(
                    old_rpy.roll_angle(),
                    old_rpy.pitch_angle(),
                    math.radians(new_yaw_deg),
                ),
                p=seat.transform.translation(),
            )
            fixes.append(
                SeatingOrientationFix(
                    subject_id=str(seat.object_id),
                    target_id=str(wall_target.object_id),
                    old_yaw_deg=math.degrees(old_rpy.yaw_angle()),
                    new_yaw_deg=new_yaw_deg,
                    angle_to_target_deg=180.0,
                )
            )
            continue
        angle = _front_angle_to_target_deg(seat, target)
        if angle is None or angle < repair_angle_threshold_deg:
            continue
        old_rpy = RollPitchYaw(seat.transform.rotation())
        new_yaw_deg = compute_optimal_facing_yaw(
            origin_a=seat.transform.translation(),
            target_point=target.transform.translation(),
        )
        # 2026-07-09 修改原因：LLM 家具阶段可能因物理误报回滚到更高总分但
        # seating 背对 table/desk 的 checkpoint；这里只修正明确背向的座椅 yaw。
        seat.transform = RigidTransform(
            rpy=RollPitchYaw(
                old_rpy.roll_angle(),
                old_rpy.pitch_angle(),
                math.radians(new_yaw_deg),
            ),
            p=seat.transform.translation(),
        )
        fixes.append(
            SeatingOrientationFix(
                subject_id=str(seat.object_id),
                target_id=str(target.object_id),
                old_yaw_deg=math.degrees(old_rpy.yaw_angle()),
                new_yaw_deg=new_yaw_deg,
                angle_to_target_deg=angle,
            )
        )

    if fixes:
        console_logger.info(
            "Seating orientation guard aligned %d object(s): %s",
            len(fixes),
            ", ".join(
                f"{fix.subject_id}->{fix.target_id} "
                f"{fix.old_yaw_deg:.1f}°→{fix.new_yaw_deg:.1f}°"
                for fix in fixes
            ),
        )
    return fixes


def _nearest_wall_anchor(
    seat: SceneObject,
    walls: list[SceneObject],
    *,
    max_gap_ratio: float,
) -> SceneObject | None:
    if not _is_wall_anchor_candidate(seat):
        return None
    ranked: list[tuple[float, str, SceneObject]] = []
    seat_bounds = seat.compute_world_bounds()
    if seat_bounds is None:
        return None
    seat_min, seat_max = seat_bounds
    footprint_scale = _seat_footprint_scale(seat)
    if footprint_scale is None:
        return None
    for wall in walls:
        wall_bounds = wall.compute_world_bounds()
        if wall_bounds is None:
            continue
        wall_min, wall_max = wall_bounds
        gap = _aabb_gap_xy(seat_min, seat_max, wall_min, wall_max)
        if gap <= footprint_scale * max_gap_ratio:
            ranked.append((gap, str(wall.object_id), wall))
    if not ranked:
        return None
    ranked.sort(key=lambda item: (item[0], item[1]))
    return ranked[0][2]


def _surface_gap_xy(seat: SceneObject, surface: SceneObject | None) -> float | None:
    if surface is None:
        return None
    seat_bounds = seat.compute_world_bounds()
    surface_bounds = surface.compute_world_bounds()
    if seat_bounds is None or surface_bounds is None:
        return None
    return _aabb_gap_xy(
        seat_bounds[0], seat_bounds[1], surface_bounds[0], surface_bounds[1]
    )


def _is_standalone_wall_seating(
    seat: SceneObject,
    wall: SceneObject,
    surface: SceneObject,
    *,
    surface_gap_ratio: float,
    wall_margin_ratio: float,
) -> bool:
    wall_gap = _surface_gap_xy(seat, wall)
    surface_gap = _surface_gap_xy(seat, surface)
    footprint_scale = _seat_footprint_scale(seat)
    if wall_gap is None or surface_gap is None or footprint_scale is None:
        return False
    # 2026-07-12 修改原因：用相对几何关系区分墙边候客座椅与桌边工作座椅，
    # 避免固定 0.45m/0.8m 阈值只适配单个书房尺寸。
    return (
        surface_gap >= footprint_scale * surface_gap_ratio
        and wall_gap + footprint_scale * wall_margin_ratio < surface_gap
    )


def _seat_footprint_scale(seat: SceneObject) -> float | None:
    bounds = seat.compute_world_bounds()
    if bounds is None:
        return None
    span = np.asarray(bounds[1] - bounds[0], dtype=float)[:2]
    positive = span[span > 1e-6]
    return float(np.min(positive)) if positive.size else None


def _wall_away_target_point(seat: SceneObject, wall: SceneObject) -> np.ndarray | None:
    seat_bounds = seat.compute_world_bounds()
    wall_bounds = wall.compute_world_bounds()
    if seat_bounds is None or wall_bounds is None:
        return None
    wall_min, wall_max = wall_bounds
    wall_span = wall_max - wall_min
    seat_center = seat.transform.translation()
    wall_center = wall.transform.translation()
    target = np.array(seat_center, dtype=float)

    # 2026-07-11 修改原因：沿长墙摆放的多把空闲椅必须沿墙法线朝室内；
    # 若按“远离墙中心点”计算，会让不同位置的椅子呈扇形而无法保持平行。
    normal_axis = 0 if float(wall_span[0]) < float(wall_span[1]) else 1
    direction = float(seat_center[normal_axis] - wall_center[normal_axis])
    if abs(direction) < 1e-6:
        return None
    target[normal_axis] += 1.0 if direction > 0.0 else -1.0
    return target


def _aabb_gap_xy(
    a_min: np.ndarray,
    a_max: np.ndarray,
    b_min: np.ndarray,
    b_max: np.ndarray,
) -> float:
    dx = max(float(b_min[0] - a_max[0]), float(a_min[0] - b_max[0]), 0.0)
    dy = max(float(b_min[1] - a_max[1]), float(a_min[1] - b_max[1]), 0.0)
    return math.hypot(dx, dy)


def _is_wall_anchor_candidate(obj: SceneObject) -> bool:
    tokens = _object_tokens(obj)
    return bool(tokens & {"armchair", "chair", "dining_chair", "office_chair"})


def _nearest_surface(
    seat: SceneObject,
    surfaces: list[SceneObject],
    *,
    max_distance_m: float,
) -> SceneObject | None:
    ranked: list[tuple[float, str, SceneObject]] = []
    seat_xy = seat.transform.translation()[:2]
    for surface in surfaces:
        if surface.object_id == seat.object_id:
            continue
        surface_xy = surface.transform.translation()[:2]
        distance = float(np.linalg.norm(surface_xy - seat_xy))
        if distance <= max_distance_m:
            ranked.append((distance, str(surface.object_id), surface))
    if not ranked:
        return None
    ranked.sort(key=lambda item: (item[0], item[1]))
    return ranked[0][2]


def _front_angle_to_target_deg(
    subject: SceneObject, target: SceneObject
) -> float | None:
    origin = subject.transform.translation()
    target_vec = target.transform.translation() - origin
    target_xy = target_vec[:2]
    norm = float(np.linalg.norm(target_xy))
    if norm < 1e-6:
        return None
    front = subject.transform.rotation().matrix() @ np.array([0.0, 1.0, 0.0])
    front_xy = front[:2]
    front_norm = float(np.linalg.norm(front_xy))
    if front_norm < 1e-6:
        return None
    cos_angle = float(np.dot(front_xy / front_norm, target_xy / norm))
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.degrees(math.acos(cos_angle))


def _is_seating(obj: SceneObject) -> bool:
    tokens = _object_tokens(obj)
    return bool(tokens & SEATING_TOKENS)


def _is_functional_surface(obj: SceneObject) -> bool:
    tokens = _object_tokens(obj)
    return bool(tokens & SURFACE_TOKENS)


def _object_tokens(obj: SceneObject) -> set[str]:
    text = " ".join(
        str(value or "")
        for value in (
            obj.object_id,
            obj.name,
            obj.description,
            obj.metadata.get("category"),
            obj.metadata.get("category_norm"),
            obj.metadata.get("scale_profile"),
        )
    )
    return {
        token
        for token in text.lower().replace("-", "_").replace(" ", "_").split("_")
        if token
    } | _compound_tokens(text)


def _compound_tokens(text: str) -> set[str]:
    normalized = text.lower().replace("-", "_").replace(" ", "_")
    out: set[str] = set()
    for token in SEATING_TOKENS | SURFACE_TOKENS:
        if token in normalized:
            out.add(token)
    return out
