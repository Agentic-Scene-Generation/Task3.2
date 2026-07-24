"""Critic-driven deterministic repair for furniture relations."""

from __future__ import annotations

import logging
import math

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np

from pydrake.math import RigidTransform, RollPitchYaw

from scenesmith.agent_utils.room import ObjectType, RoomScene, SceneObject, UniqueID
from scenesmith.scenebenchmark_critic.api import evaluate_room_scene
from scenesmith.scenebenchmark_critic.config import CriticConfig, critic_config_from_any

console_logger = logging.getLogger(__name__)

_ISSUE_LABELS = {"fail", "degraded"}
_REPAIRABLE_RELATIONS = {
    "dining_seat_distribution",
    "seating_to_work_surface",
    "wall_backed_storage_alignment",
    "workstation_focal_alignment",
}


@dataclass(frozen=True)
class FurnitureRelationFix:
    object_id: str
    relation_type: str
    check_id: str
    old_xy: tuple[float, float]
    new_xy: tuple[float, float]
    old_yaw_deg: float
    new_yaw_deg: float


@dataclass(frozen=True)
class _RepairTarget:
    object_id: str
    relation_type: str
    check_id: str
    target_center_xy: tuple[float, float]
    target_yaw_deg: float | None


@dataclass(frozen=True)
class _PayloadScore:
    fail_ids: frozenset[str]
    degraded_ids: frozenset[str]
    labels: dict[str, str]

    @property
    def global_key(self) -> tuple[int, int]:
        return len(self.fail_ids), len(self.degraded_ids)

    @property
    def issue_ids(self) -> frozenset[str]:
        return self.fail_ids | self.degraded_ids


def improve_furniture_relations(
    scene: RoomScene,
    *,
    config: CriticConfig | Any | None = None,
    max_repairs: int = 8,
    max_translation_m: float = 1.5,
) -> list[FurnitureRelationFix]:
    """Apply critic targets only when the whole-scene evaluation improves."""
    critic_config = (
        config if isinstance(config, CriticConfig) else critic_config_from_any(config)
    )
    if not critic_config.enabled or not critic_config.metric_enabled(
        "functional_dependency"
    ):
        return []

    fixes: list[FurnitureRelationFix] = []
    for _ in range(max_repairs):
        baseline_payload = _evaluate(scene, critic_config)
        baseline_score = _score_payload(baseline_payload)
        accepted = False
        for target in _repair_targets(scene, baseline_payload):
            obj = scene.objects.get(UniqueID(target.object_id))
            if obj is None or obj.object_type != ObjectType.FURNITURE:
                continue
            current_center = _world_center_xy(obj)
            if current_center is None:
                continue
            if (
                math.hypot(
                    target.target_center_xy[0] - current_center[0],
                    target.target_center_xy[1] - current_center[1],
                )
                > max_translation_m
            ):
                continue
            candidate_scene = deepcopy(scene)
            candidate_obj = candidate_scene.objects[UniqueID(target.object_id)]
            new_transform = _transform_for_target(candidate_obj, target)
            if new_transform is None or not _within_floor_bounds(
                candidate_scene, candidate_obj, new_transform
            ):
                continue
            _move_object_with_surfaces(
                candidate_scene, candidate_obj.object_id, new_transform
            )
            candidate_payload = _evaluate(candidate_scene, critic_config)
            if not _candidate_improves(
                baseline_payload,
                candidate_payload,
                baseline_score=baseline_score,
                check_id=target.check_id,
            ):
                continue

            old_rpy = RollPitchYaw(obj.transform.rotation())
            old_xy = tuple(float(value) for value in obj.transform.translation()[:2])
            moved_obj = candidate_scene.objects[obj.object_id]
            new_rpy = RollPitchYaw(moved_obj.transform.rotation())
            new_xy = tuple(
                float(value) for value in moved_obj.transform.translation()[:2]
            )
            _copy_scene_object_poses_and_surfaces(
                source_scene=candidate_scene, target_scene=scene
            )
            fixes.append(
                FurnitureRelationFix(
                    object_id=target.object_id,
                    relation_type=target.relation_type,
                    check_id=target.check_id,
                    old_xy=old_xy,
                    new_xy=new_xy,
                    old_yaw_deg=math.degrees(old_rpy.yaw_angle()),
                    new_yaw_deg=math.degrees(new_rpy.yaw_angle()),
                )
            )
            accepted = True
            break
        if not accepted:
            break

    if fixes:
        console_logger.info(
            "Furniture relation repair accepted %d move(s): %s",
            len(fixes),
            ", ".join(
                f"{fix.object_id}:{fix.relation_type} "
                f"({fix.old_xy[0]:.2f},{fix.old_xy[1]:.2f})->"
                f"({fix.new_xy[0]:.2f},{fix.new_xy[1]:.2f})"
                for fix in fixes
            ),
        )
    return fixes


def _evaluate(scene: RoomScene, config: CriticConfig) -> dict[str, Any]:
    return evaluate_room_scene(
        scene,
        config=config,
        stage="furniture_relation_repair",
        annotate_assets=False,
    )


def _score_payload(payload: dict[str, Any]) -> _PayloadScore:
    labels: dict[str, str] = {}
    fail_ids: set[str] = set()
    degraded_ids: set[str] = set()
    for index, result in enumerate(payload.get("results") or []):
        if str(result.get("scoring_tier") or "").lower() == "ignored":
            continue
        check_id = str(result.get("check_id") or f"result_{index}")
        label = str(result.get("label") or "unknown")
        labels[check_id] = label
        if label == "fail":
            fail_ids.add(check_id)
        elif label == "degraded":
            degraded_ids.add(check_id)
    return _PayloadScore(frozenset(fail_ids), frozenset(degraded_ids), labels)


def _candidate_improves(
    baseline_payload: dict[str, Any],
    candidate_payload: dict[str, Any],
    *,
    baseline_score: _PayloadScore,
    check_id: str,
) -> bool:
    candidate_score = _score_payload(candidate_payload)
    if candidate_score.issue_ids - baseline_score.issue_ids:
        return False
    if candidate_score.global_key > baseline_score.global_key:
        return False
    for existing_id, old_label in baseline_score.labels.items():
        if existing_id == check_id:
            continue
        new_label = candidate_score.labels.get(existing_id)
        if new_label is None:
            continue
        if _label_severity(new_label) > _label_severity(old_label):
            return False
    old_result = _result_by_id(baseline_payload, check_id)
    new_result = _result_by_id(candidate_payload, check_id)
    if old_result is None or new_result is None:
        return False
    return _result_severity(new_result) < _result_severity(old_result)


def _result_by_id(payload: dict[str, Any], check_id: str) -> dict[str, Any] | None:
    return next(
        (
            result
            for result in payload.get("results") or []
            if str(result.get("check_id") or "") == check_id
        ),
        None,
    )


def _result_severity(result: dict[str, Any]) -> tuple[int, float]:
    label = str(result.get("label") or "unknown")
    diagnostics = result.get("diagnostics") or {}
    relation = str(result.get("relation_type") or "")
    magnitude = 0.0
    if relation == "dining_seat_distribution":
        for slot in diagnostics.get("seat_slots") or []:
            magnitude += max(
                0.0,
                float(slot.get("deviation_m") or 0.0)
                - float(slot.get("allowed_deviation_m") or 0.0),
            )
            facing = slot.get("facing_error_deg")
            if facing is not None:
                magnitude += max(
                    0.0,
                    (float(facing) - float(slot.get("facing_allowed_error_deg") or 0.0))
                    / 90.0,
                )
    elif relation == "workstation_focal_alignment":
        magnitude = max(
            0.0,
            float(diagnostics.get("lateral_offset_m") or 0.0)
            - float(diagnostics.get("lateral_tolerance_m") or 0.0),
        ) + max(
            0.0, (float(diagnostics.get("angle_to_focus_deg") or 0.0) - 25.0) / 90.0
        )
    elif relation == "wall_backed_storage_alignment":
        magnitude = max(
            0.0,
            float(diagnostics.get("nearest_wall_gap_m") or 0.0)
            - float(diagnostics.get("allowed_wall_gap_m") or 0.0),
        ) + max(
            0.0,
            (
                float(diagnostics.get("front_error_deg") or 0.0)
                - float(diagnostics.get("allowed_front_error_deg") or 0.0)
            )
            / 90.0,
        )
    return _label_severity(label), round(magnitude, 9)


def _label_severity(label: str) -> int:
    return {"pass": 0, "degraded": 1, "fail": 2, "unknown": 3}.get(label, 3)


def _repair_targets(scene: RoomScene, payload: dict[str, Any]) -> list[_RepairTarget]:
    targets: list[_RepairTarget] = []
    for result in payload.get("results") or []:
        relation = str(result.get("relation_type") or "")
        if (
            result.get("label") not in _ISSUE_LABELS
            or relation not in _REPAIRABLE_RELATIONS
        ):
            continue
        check_id = str(result.get("check_id") or "")
        diagnostics = result.get("diagnostics") or {}
        if relation == "dining_seat_distribution":
            for slot in diagnostics.get("seat_slots") or []:
                if slot.get("aligned") and slot.get("facing_aligned") is not False:
                    continue
                target = _target_from_facing_diagnostics(
                    scene,
                    object_id=str(slot.get("seat_id") or ""),
                    relation_type=relation,
                    check_id=check_id,
                    diagnostics=slot,
                )
                if target is not None:
                    targets.append(target)
        elif relation == "workstation_focal_alignment":
            target = _target_from_facing_diagnostics(
                scene,
                object_id=str(diagnostics.get("seat_id") or ""),
                relation_type=relation,
                check_id=check_id,
                diagnostics=diagnostics,
            )
            if target is not None:
                targets.append(target)
        elif relation == "seating_to_work_surface":
            assignment = diagnostics.get("seat_surface_assignment") or {}
            slot = assignment.get("target_slot") or {}
            center = _xy(slot.get("center_xy"))
            yaw = _float_or_none(slot.get("yaw_deg"))
            object_id = str(result.get("primary_object") or "")
            if center is not None and yaw is not None and object_id:
                targets.append(
                    _RepairTarget(object_id, relation, check_id, center, yaw)
                )
        elif relation == "wall_backed_storage_alignment":
            object_id = str(
                diagnostics.get("object_id") or result.get("primary_object") or ""
            )
            for pose in diagnostics.get("candidate_poses") or []:
                center = _xy(pose.get("target_center_xy_m"))
                yaw = _float_or_none(pose.get("target_yaw_deg"))
                if object_id and center is not None and yaw is not None:
                    targets.append(
                        _RepairTarget(object_id, relation, check_id, center, yaw)
                    )
    targets.sort(
        key=lambda target: (
            0 if _result_by_id(payload, target.check_id).get("label") == "fail" else 1,
            target.check_id,
            target.object_id,
        )
    )
    return targets


def _target_from_facing_diagnostics(
    scene: RoomScene,
    *,
    object_id: str,
    relation_type: str,
    check_id: str,
    diagnostics: dict[str, Any],
) -> _RepairTarget | None:
    center = _xy(diagnostics.get("target_center_xy_m"))
    if not object_id or center is None:
        return None
    obj = scene.objects.get(UniqueID(object_id))
    current_front = _xy(diagnostics.get("current_front_xy"))
    facing_target = _xy(diagnostics.get("facing_target_xy_m"))
    yaw: float | None = None
    if obj is not None and current_front is not None and facing_target is not None:
        desired = (facing_target[0] - center[0], facing_target[1] - center[1])
        if math.hypot(*desired) > 1e-6 and math.hypot(*current_front) > 1e-6:
            delta = math.degrees(
                math.atan2(
                    current_front[0] * desired[1] - current_front[1] * desired[0],
                    current_front[0] * desired[0] + current_front[1] * desired[1],
                )
            )
            current_yaw = math.degrees(
                RollPitchYaw(obj.transform.rotation()).yaw_angle()
            )
            yaw = current_yaw + delta
    return _RepairTarget(object_id, relation_type, check_id, center, yaw)


def _xy(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        return float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _world_center_xy(obj: SceneObject) -> tuple[float, float] | None:
    bounds = obj.compute_world_bounds()
    if bounds is None:
        return None
    center = (
        np.asarray(bounds[0], dtype=float) + np.asarray(bounds[1], dtype=float)
    ) / 2.0
    return float(center[0]), float(center[1])


def _transform_for_target(
    obj: SceneObject, target: _RepairTarget
) -> RigidTransform | None:
    bounds = obj.compute_world_bounds()
    if bounds is None or obj.bbox_min is None or obj.bbox_max is None:
        return None
    world_center = (
        np.asarray(bounds[0], dtype=float) + np.asarray(bounds[1], dtype=float)
    ) / 2.0
    old_rpy = RollPitchYaw(obj.transform.rotation())
    yaw = (
        math.radians(target.target_yaw_deg)
        if target.target_yaw_deg is not None
        else old_rpy.yaw_angle()
    )
    new_rpy = RollPitchYaw(old_rpy.roll_angle(), old_rpy.pitch_angle(), yaw)
    local_center = (
        np.asarray(obj.bbox_min, dtype=float) + np.asarray(obj.bbox_max, dtype=float)
    ) / 2.0
    target_world_center = np.array(
        [target.target_center_xy[0], target.target_center_xy[1], world_center[2]],
        dtype=float,
    )
    translation = target_world_center - new_rpy.ToRotationMatrix().multiply(
        local_center
    )
    return RigidTransform(rpy=new_rpy, p=translation)


def _within_floor_bounds(
    scene: RoomScene, obj: SceneObject, transform: RigidTransform
) -> bool:
    geometry = scene.room_geometry
    if geometry is None or geometry.length <= 0 or geometry.width <= 0:
        return True
    old_transform = obj.transform
    try:
        obj.transform = transform
        bounds = obj.compute_world_bounds()
    finally:
        obj.transform = old_transform
    if bounds is None:
        return False
    lower, upper = bounds
    margin = 0.03
    return (
        float(lower[0]) >= -float(geometry.length) / 2.0 + margin
        and float(upper[0]) <= float(geometry.length) / 2.0 - margin
        and float(lower[1]) >= -float(geometry.width) / 2.0 + margin
        and float(upper[1]) <= float(geometry.width) / 2.0 - margin
    )


def _move_object_with_surfaces(
    scene: RoomScene, object_id: UniqueID, new_transform: RigidTransform
) -> None:
    obj = scene.objects[object_id]
    old_transform = obj.transform
    delta = new_transform @ old_transform.inverse()
    moved_surface_ids = {surface.surface_id for surface in obj.support_surfaces}
    scene.move_object(object_id=object_id, new_transform=new_transform)
    for surface in obj.support_surfaces:
        surface.transform = delta @ surface.transform
    for child in scene.objects.values():
        placement = child.placement_info
        if placement is None or placement.parent_surface_id not in moved_surface_ids:
            continue
        child.transform = delta @ child.transform
        for surface in child.support_surfaces:
            surface.transform = delta @ surface.transform


def _copy_scene_object_poses_and_surfaces(
    *, source_scene: RoomScene, target_scene: RoomScene
) -> None:
    for object_id, source_obj in source_scene.objects.items():
        target_obj = target_scene.objects.get(object_id)
        if target_obj is None:
            continue
        target_obj.transform = source_obj.transform
        target_obj.support_surfaces = deepcopy(source_obj.support_surfaces)
