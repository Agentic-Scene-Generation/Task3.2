"""Deterministic seating orientation guard for furniture-stage scenes."""

from __future__ import annotations

import logging
import math
import re

from dataclasses import dataclass
from typing import Any

import numpy as np

from pydrake.math import RigidTransform, RollPitchYaw
from shapely.geometry import Polygon, box

from scenesmith.agent_utils.clearance_zones import (
    compute_door_clearance_violations,
    compute_window_clearance_violations,
)
from scenesmith.agent_utils.room import ObjectType, RoomScene, SceneObject
from scenesmith.scenebenchmark_critic.auto_repair_stats import record_auto_repair_call
from scenesmith.scenebenchmark_critic.config import CriticConfig, critic_config_from_any
from scenesmith.utils.geometry_utils import compute_optimal_facing_yaw

console_logger = logging.getLogger(__name__)
_LEGACY_CONFIG_WARNING_EMITTED = False

SEATING_TOKENS = {
    "armchair",
    "bar_stool",
    "bench",
    "chair",
    "dining_chair",
    "loveseat",
    "l_sofa",
    "office_chair",
    "sectional",
    "sofa",
    "stool",
}
SOFA_TOKENS = {
    "l_sofa",
    "loveseat",
    "sectional",
    "sofa",
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
    allowed_targets_by_seat: dict[str, set[str]] | None = None,
    config: CriticConfig | Any | None = None,
    max_repairs: int | None = None,
    record_stats: bool = True,
    max_target_distance_m: float = 2.0,
    repair_angle_threshold_deg: float = 45.0,
    # Generated furniture can start roughly one seat-depth away from its
    # intended wall.  Keep this relative to the asset rather than using a
    # case-specific absolute gap, then snap the final pose to the wall.
    wall_anchor_gap_ratio: float = 1.0,
    standalone_surface_gap_ratio: float = 0.5,
    wall_preference_margin_ratio: float = 0.2,
    wall_inward_angle_threshold_deg: float = 5.0,
) -> list[SeatingOrientationFix]:
    """Rotate seating only toward an allowed or legacy-nearest target.

    ``allowed_targets_by_seat`` is supplied by intent-contract mode. An empty
    mapping entry leaves ordinary seating untouched and applies only the
    reversible nearest-wall parallel preference to sofa-like seating.
    """
    global _LEGACY_CONFIG_WARNING_EMITTED
    if config is None:
        if not _LEGACY_CONFIG_WARNING_EMITTED:
            console_logger.warning(
                "align_seating_to_nearest_surface called without critic config; "
                "using legacy uncontrolled behavior"
            )
            _LEGACY_CONFIG_WARNING_EMITTED = True
        if record_stats:
            record_auto_repair_call(scene, module="seating_orientation", stage="seating_orientation_guard", status="legacy_uncontrolled")
        critic_config: CriticConfig | None = None
    else:
        critic_config = config if isinstance(config, CriticConfig) else critic_config_from_any(config)
        if not critic_config.enabled:
            if record_stats:
                record_auto_repair_call(scene, module="seating_orientation", stage="seating_orientation_guard", status="skipped", skip_reason="critic_disabled")
            return []
        if not critic_config.auto_repair.should_repair("seating_orientation"):
            if record_stats:
                record_auto_repair_call(scene, module="seating_orientation", stage="seating_orientation_guard", status="skipped", skip_reason="module_disabled")
            return []
        if max_repairs is None:
            max_repairs = critic_config.auto_repair.max_repairs_per_call
        elif critic_config.auto_repair.max_repairs_per_call is not None:
            max_repairs = min(max_repairs, critic_config.auto_repair.max_repairs_per_call)

    furniture = [
        obj for obj in scene.objects.values() if obj.object_type == ObjectType.FURNITURE
    ]
    seating = [obj for obj in furniture if _is_seating(obj)]
    surfaces = [obj for obj in furniture if _is_functional_surface(obj)]
    fixes: list[SeatingOrientationFix] = []
    if not seating:
        if critic_config is not None and record_stats:
            record_auto_repair_call(scene, module="seating_orientation", stage="seating_orientation_guard", status="no_targets", candidate_budget=None)
        return fixes

    for seat in seating:
        if max_repairs is not None and len(fixes) >= max_repairs:
            break
        allowed_targets = (
            allowed_targets_by_seat.get(str(seat.object_id), set())
            if allowed_targets_by_seat is not None
            else None
        )
        walls = scene.get_objects_by_type(ObjectType.WALL)
        if allowed_targets_by_seat is not None and not allowed_targets:
            default_fix = _align_default_sofa_parallel_to_wall(scene, seat, walls)
            if default_fix is not None:
                fixes.append(default_fix)
            continue
        if allowed_targets is not None:
            walls = [wall for wall in walls if str(wall.object_id) in allowed_targets]
        candidate_surfaces = surfaces
        if allowed_targets is not None:
            # Contract targets can be wall-mounted media.  Restricting this
            # lookup to floor-standing work surfaces made an explicit
            # sofa->television relation disappear and let the legacy nearest
            # table fallback rotate the sofa later in the pipeline.
            candidate_surfaces = [
                target
                for target in scene.objects.values()
                if str(target.object_id) in allowed_targets
                and target.object_id != seat.object_id
                and target.object_type not in (ObjectType.WALL, ObjectType.FLOOR)
            ]
        wall_target = _nearest_wall_anchor(
            seat,
            walls,
            max_gap_ratio=wall_anchor_gap_ratio,
            peer_seating=seating,
        )
        target = _nearest_surface(
            seat, candidate_surfaces, max_distance_m=max_target_distance_m
        )
        angle = _front_angle_to_target_deg(seat, target) if target is not None else None
        wall_inward_angle = None
        if wall_target is not None:
            wall_inward_point = _wall_away_target_point(seat, wall_target)
            if wall_inward_point is not None:
                wall_inward_angle = _front_angle_to_point_deg(seat, wall_inward_point)
        already_faces_wall_interior = (
            wall_inward_angle is not None
            and wall_inward_angle <= wall_inward_angle_threshold_deg
        )
        # Dining-table seats remain table-relative even when an outer slot is
        # close to a room wall.  A wall-backed pose would destroy the complete
        # one-seat-per-edge arrangement repaired by the critic.
        target_is_dining_table = target is not None and _is_dining_table(target)
        # A current yaw that happens to point diagonally toward a desk is not
        # evidence that a wall-side seat belongs to that desk. Prefer the work
        # surface only when the prompt explicitly authorizes that relation.
        prompt_requires_surface_facing = _has_explicit_surface_facing_contract(
            scene, seat
        ) or bool(target is not None and allowed_targets is not None)
        if (
            wall_target is not None
            and _is_wall_anchor_candidate(seat)
            and not prompt_requires_surface_facing
            and not already_faces_wall_interior
            and (
                target is None
                or (
                    not target_is_dining_table
                    and _is_standalone_wall_seating(
                        seat,
                        wall_target,
                        target,
                        surface_gap_ratio=standalone_surface_gap_ratio,
                        wall_margin_ratio=wall_preference_margin_ratio,
                    )
                )
            )
        ):
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
            new_transform = _wall_backed_transform(
                seat,
                wall_target,
                new_yaw_deg=new_yaw_deg,
            )
            if new_transform is None:
                continue
            if not _apply_transform_without_geometry_regression(
                scene,
                seat,
                new_transform,
            ):
                continue
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
        if angle is None or angle <= repair_angle_threshold_deg:
            continue
        old_rpy = RollPitchYaw(seat.transform.rotation())
        new_yaw_deg = compute_optimal_facing_yaw(
            origin_a=seat.transform.translation(),
            target_point=target.transform.translation(),
        )
        # 2026-07-22 修改原因：functional dependency 要求座椅朝向 table/desk
        # 的夹角不超过 45°；独立墙边座椅已在上方分支优先按墙法线朝向室内。
        candidate = RigidTransform(
            rpy=RollPitchYaw(
                old_rpy.roll_angle(),
                old_rpy.pitch_angle(),
                math.radians(new_yaw_deg),
            ),
            p=seat.transform.translation(),
        )
        if not _apply_transform_without_geometry_regression(
            scene,
            seat,
            candidate,
        ):
            continue
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
    if critic_config is not None and record_stats:
        record_auto_repair_call(
            scene, module="seating_orientation", stage="seating_orientation_guard",
            status="committed" if fixes else "no_targets", accepted_rounds=len(fixes),
            fix_records=len(fixes), object_ids=(fix.subject_id for fix in fixes),
            relation_types=("seating_orientation" for _ in fixes),
        )
    return fixes


def _nearest_wall_anchor(
    seat: SceneObject,
    walls: list[SceneObject],
    *,
    max_gap_ratio: float,
    peer_seating: list[SceneObject] | None = None,
) -> SceneObject | None:
    if not _is_wall_anchor_candidate(seat):
        return None
    ranked: list[tuple[int, float, str, SceneObject]] = []
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
            support = _wall_row_support(
                seat,
                wall,
                peer_seating or [],
                max_gap_ratio=max_gap_ratio,
            )
            ranked.append((-support, gap, str(wall.object_id), wall))
    if not ranked:
        return None
    ranked.sort(key=lambda item: (item[0], item[1], item[2]))
    return ranked[0][3]


def _wall_row_support(
    seat: SceneObject,
    wall: SceneObject,
    peer_seating: list[SceneObject],
    *,
    max_gap_ratio: float,
) -> int:
    """Count nearby seats aligned in a row parallel to the candidate wall."""
    wall_bounds = wall.compute_world_bounds()
    seat_scale = _seat_footprint_scale(seat)
    if wall_bounds is None or seat_scale is None:
        return 0
    wall_span = wall_bounds[1] - wall_bounds[0]
    normal_axis = 0 if float(wall_span[0]) < float(wall_span[1]) else 1
    seat_normal = float(seat.transform.translation()[normal_axis])
    support = 0
    for peer in peer_seating:
        if peer.object_id == seat.object_id:
            continue
        peer_bounds = peer.compute_world_bounds()
        peer_scale = _seat_footprint_scale(peer)
        if peer_bounds is None or peer_scale is None:
            continue
        peer_gap = _aabb_gap_xy(
            peer_bounds[0], peer_bounds[1], wall_bounds[0], wall_bounds[1]
        )
        if peer_gap > peer_scale * max_gap_ratio:
            continue
        peer_normal = float(peer.transform.translation()[normal_axis])
        if abs(peer_normal - seat_normal) <= max(seat_scale, peer_scale) * 0.5:
            support += 1
    return support


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


def _has_explicit_surface_facing_contract(scene: RoomScene, seat: SceneObject) -> bool:
    """Return whether the active critic contract explicitly binds this seat."""
    contracts = getattr(scene, "_scenebenchmark_orientation_contracts", None)
    if not isinstance(contracts, dict):
        return False
    contract = contracts.get(str(seat.object_id))
    if not isinstance(contract, dict) or not contract.get("prompt_explicit_facing"):
        return False
    return str(contract.get("relation_type") or "") in {
        "furniture_faces_furniture",
        "seat_faces_surface",
        "seating_to_work_surface",
    }


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


def _wall_backed_transform(
    seat: SceneObject,
    wall: SceneObject,
    *,
    new_yaw_deg: float,
    back_gap_m: float = 0.03,
) -> RigidTransform | None:
    """Rotate a standalone seat inward and place its back near the wall face."""
    wall_bounds = wall.compute_world_bounds()
    if wall_bounds is None:
        return None
    wall_min, wall_max = wall_bounds
    wall_span = wall_max - wall_min
    normal_axis = 0 if float(wall_span[0]) < float(wall_span[1]) else 1
    seat_center = np.asarray(seat.transform.translation(), dtype=float)
    wall_center = np.asarray(wall.transform.translation(), dtype=float)
    direction = float(seat_center[normal_axis] - wall_center[normal_axis])
    if abs(direction) < 1e-6:
        return None

    old_rpy = RollPitchYaw(seat.transform.rotation())
    rotated = RigidTransform(
        rpy=RollPitchYaw(
            old_rpy.roll_angle(),
            old_rpy.pitch_angle(),
            math.radians(new_yaw_deg),
        ),
        p=seat_center,
    )
    old_transform = seat.transform
    try:
        seat.transform = rotated
        seat_bounds = seat.compute_world_bounds()
    finally:
        seat.transform = old_transform
    if seat_bounds is None:
        return None

    seat_min, seat_max = seat_bounds
    translation = seat_center.copy()
    if direction > 0.0:
        translation[normal_axis] += (
            float(wall_max[normal_axis]) + back_gap_m - float(seat_min[normal_axis])
        )
    else:
        translation[normal_axis] += (
            float(wall_min[normal_axis]) - back_gap_m - float(seat_max[normal_axis])
        )
    return RigidTransform(R=rotated.rotation(), p=translation)


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
    return bool(
        tokens
        & {
            "armchair",
            "bar_stool",
            "bench",
            "chair",
            "dining_chair",
            "office_chair",
            "sectional",
            "sofa",
            "stool",
        }
    )


def _is_sofa_like(obj: SceneObject) -> bool:
    tokens = _object_tokens(obj)
    return bool(tokens & SOFA_TOKENS) or ("gaming" in tokens and "sofa" in tokens)


def _align_default_sofa_parallel_to_wall(
    scene: RoomScene,
    seat: SceneObject,
    walls: list[SceneObject],
    *,
    tolerance_deg: float = 3.0,
) -> SeatingOrientationFix | None:
    """Cardinalize an unconstrained sofa without worsening its geometry.

    This is a presentation preference, not a hard semantic relation.  It is
    therefore used only when the intent contract supplied no facing/wall
    target, and it changes yaw only.  Explicit media, table, and wall-facing
    contracts are handled by the branches above.
    """
    if not _is_sofa_like(seat) or not walls:
        return None
    seat_bounds = seat.compute_world_bounds()
    if seat_bounds is None:
        return None
    ranked: list[tuple[float, str, SceneObject]] = []
    for wall in walls:
        wall_bounds = wall.compute_world_bounds()
        if wall_bounds is None:
            continue
        ranked.append(
            (
                _aabb_gap_xy(
                    seat_bounds[0], seat_bounds[1], wall_bounds[0], wall_bounds[1]
                ),
                str(wall.object_id),
                wall,
            )
        )
    if not ranked:
        return None
    ranked.sort(key=lambda item: (item[0], item[1]))
    wall = ranked[0][2]
    target_point = _wall_away_target_point(seat, wall)
    if target_point is None:
        return None
    angle = _front_angle_to_point_deg(seat, target_point)
    if angle is None or angle <= tolerance_deg:
        return None

    old_rpy = RollPitchYaw(seat.transform.rotation())
    new_yaw_deg = compute_optimal_facing_yaw(
        origin_a=seat.transform.translation(),
        target_point=target_point,
    )
    candidate = RigidTransform(
        rpy=RollPitchYaw(
            old_rpy.roll_angle(),
            old_rpy.pitch_angle(),
            math.radians(new_yaw_deg),
        ),
        p=seat.transform.translation(),
    )
    if not _apply_transform_without_geometry_regression(scene, seat, candidate):
        return None
    return SeatingOrientationFix(
        subject_id=str(seat.object_id),
        target_id=str(wall.object_id),
        old_yaw_deg=math.degrees(old_rpy.yaw_angle()),
        new_yaw_deg=new_yaw_deg,
        angle_to_target_deg=angle,
    )


def _room_overflow_area(scene: RoomScene, obj: SceneObject) -> float:
    geometry = scene.room_geometry
    footprint = _object_footprint_polygon(obj)
    if geometry is None or footprint is None:
        return 0.0
    length = float(getattr(geometry, "length", 0.0) or 0.0)
    width = float(getattr(geometry, "width", 0.0) or 0.0)
    if length <= 0.0 or width <= 0.0:
        return 0.0
    room = box(-length / 2.0, -width / 2.0, length / 2.0, width / 2.0)
    return max(0.0, float(footprint.area - footprint.intersection(room).area))


def _apply_transform_without_geometry_regression(
    scene: RoomScene,
    subject: SceneObject,
    candidate: RigidTransform,
) -> bool:
    """Apply a seating candidate only when physical geometry does not worsen."""
    old_transform = subject.transform
    old_overflow = _room_overflow_area(scene, subject)
    old_overlap = _furniture_overlap_volume(scene, subject)
    old_wall_overlap = _wall_overlap_volume(scene, subject)
    old_openings = _opening_violation_profile(scene, subject)
    subject.transform = candidate
    new_overflow = _room_overflow_area(scene, subject)
    new_overlap = _furniture_overlap_volume(scene, subject)
    new_wall_overlap = _wall_overlap_volume(scene, subject)
    new_openings = _opening_violation_profile(scene, subject)
    if (
        new_overflow > old_overflow + 1e-6
        or new_overlap > old_overlap + 1e-6
        or new_wall_overlap > old_wall_overlap + 1e-6
        or _severity_profile_regresses(old_openings, new_openings)
    ):
        subject.transform = old_transform
        return False
    return True


def _object_footprint_polygon(obj: SceneObject) -> Polygon | None:
    if obj.bbox_min is None or obj.bbox_max is None:
        return None
    lower = np.asarray(obj.bbox_min, dtype=float)
    upper = np.asarray(obj.bbox_max, dtype=float)
    local = (
        (lower[0], lower[1]),
        (upper[0], lower[1]),
        (upper[0], upper[1]),
        (lower[0], upper[1]),
    )
    points = []
    for x, y in local:
        world = obj.transform @ np.asarray([x, y, 0.0], dtype=float)
        points.append((float(world[0]), float(world[1])))
    footprint = Polygon(points)
    return footprint if footprint.is_valid and not footprint.is_empty else None


def _overlap_volume(first: SceneObject, second: SceneObject) -> float:
    first_footprint = _object_footprint_polygon(first)
    second_footprint = _object_footprint_polygon(second)
    first_bounds = first.compute_world_bounds()
    second_bounds = second.compute_world_bounds()
    if (
        first_footprint is None
        or second_footprint is None
        or first_bounds is None
        or second_bounds is None
    ):
        return 0.0
    overlap_z = max(
        0.0,
        float(
            min(first_bounds[1][2], second_bounds[1][2])
            - max(first_bounds[0][2], second_bounds[0][2])
        ),
    )
    return float(first_footprint.intersection(second_footprint).area) * overlap_z


def _furniture_overlap_volume(
    scene: RoomScene,
    subject: SceneObject,
) -> float:
    total = 0.0
    for other in scene.objects.values():
        if (
            other.object_id == subject.object_id
            or other.object_type != ObjectType.FURNITURE
        ):
            continue
        total += _overlap_volume(subject, other)
    return total


def _wall_overlap_volume(scene: RoomScene, subject: SceneObject) -> float:
    return sum(
        _overlap_volume(subject, wall)
        for wall in scene.get_objects_by_type(ObjectType.WALL)
    )


def _opening_violation_profile(
    scene: RoomScene,
    subject: SceneObject,
) -> dict[str, float]:
    subject_id = str(subject.object_id)
    profile: dict[str, float] = {}
    try:
        for violation in compute_door_clearance_violations(scene):
            if violation.furniture_id == subject_id:
                profile[f"door:{violation.door_label}"] = float(
                    violation.penetration_depth
                )
    except Exception:
        console_logger.debug("Could not evaluate seating door clearance", exc_info=True)
    try:
        for violation in compute_window_clearance_violations(scene):
            if violation.furniture_id == subject_id:
                profile[f"window:{violation.window_label}"] = max(
                    0.0,
                    float(violation.furniture_top_height - violation.sill_height),
                )
    except Exception:
        console_logger.debug(
            "Could not evaluate seating window clearance",
            exc_info=True,
        )
    return profile


def _severity_profile_regresses(
    before: dict[str, float],
    after: dict[str, float],
) -> bool:
    return bool(after.keys() - before.keys()) or any(
        after[key] > before[key] + 1e-6 for key in before.keys() & after.keys()
    )


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
    return _front_angle_to_point_deg(subject, target.transform.translation())


def _front_angle_to_point_deg(
    subject: SceneObject, target_point: np.ndarray
) -> float | None:
    origin = subject.transform.translation()
    target_vec = target_point - origin
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


def _is_dining_table(obj: SceneObject) -> bool:
    tokens = _object_tokens(obj)
    return "dining_table" in tokens or ("dining" in tokens and "table" in tokens)


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
    for token in SEATING_TOKENS | SOFA_TOKENS | SURFACE_TOKENS:
        if re.search(rf"(?:^|_){re.escape(token)}(?:_|$)", normalized):
            out.add(token)
    return out
