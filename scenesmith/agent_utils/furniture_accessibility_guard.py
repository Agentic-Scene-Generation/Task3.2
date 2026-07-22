"""Deterministic furniture accessibility guard for storage-like furniture."""

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

STORAGE_TOKENS = {
    "bookcase",
    "bookshelf",
    "buffet",
    "cabinet",
    "console",
    "credenza",
    "dresser",
    "drawer",
    "fridge",
    "nightstand",
    "refrigerator",
    "shelf",
    "sideboard",
    "storage",
    "wardrobe",
}
SEATING_TOKENS = {"bench", "chair", "loveseat", "sofa", "stool"}


@dataclass(frozen=True)
class FurnitureAccessibilityFix:
    subject_id: str
    old_xy: tuple[float, float]
    new_xy: tuple[float, float]
    old_fail_count: int
    new_fail_count: int
    old_score: float
    new_score: float


@dataclass(frozen=True)
class _CandidateScore:
    scene: RoomScene
    xy: tuple[float, float]
    fail_count: int
    degraded_count: int
    score: float
    subject_label: str | None
    subject_ratio: float


def improve_storage_front_access(
    scene: RoomScene,
    *,
    config: CriticConfig | Any | None = None,
    max_translation_m: float = 1.0,
    step_m: float = 0.2,
) -> list[FurnitureAccessibilityFix]:
    """Move storage-like furniture laterally when SceneBenchmark front access fails."""
    critic_config = _spatial_config(config)
    baseline_payload = _evaluate(scene, critic_config)
    baseline_score = _score_scene(baseline_payload)
    failing_subjects = _failing_storage_subjects(scene, baseline_payload)
    if not failing_subjects:
        return []

    fixes: list[FurnitureAccessibilityFix] = []
    working_scene = scene
    working_payload = baseline_payload
    working_score = baseline_score

    for subject_id in failing_subjects:
        subject = working_scene.objects.get(UniqueID(subject_id))
        if subject is None:
            continue
        best = _best_candidate(
            working_scene,
            subject,
            current_score=working_score,
            config=critic_config,
            max_translation_m=max_translation_m,
            step_m=step_m,
        )
        if best is None:
            continue
        old_xy = tuple(float(v) for v in subject.transform.translation()[:2])
        moved_subject = best.scene.objects[UniqueID(subject_id)]
        new_xy = tuple(float(v) for v in moved_subject.transform.translation()[:2])
        _copy_scene_object_poses_and_surfaces(
            source_scene=best.scene,
            target_scene=working_scene,
        )
        fixes.append(
            FurnitureAccessibilityFix(
                subject_id=subject_id,
                old_xy=old_xy,
                new_xy=new_xy,
                old_fail_count=working_score[0],
                new_fail_count=best.fail_count,
                old_score=working_score[2],
                new_score=best.score,
            )
        )
        working_payload = _evaluate(working_scene, critic_config)
        working_score = _score_scene(working_payload)

    if fixes:
        console_logger.info(
            "Furniture accessibility guard moved %d storage object(s): %s",
            len(fixes),
            ", ".join(
                f"{fix.subject_id} ({fix.old_xy[0]:.2f},{fix.old_xy[1]:.2f})→"
                f"({fix.new_xy[0]:.2f},{fix.new_xy[1]:.2f}) "
                f"fail {fix.old_fail_count}->{fix.new_fail_count}"
                for fix in fixes
            ),
        )
    return fixes


def _spatial_config(config: CriticConfig | Any | None) -> CriticConfig:
    base = critic_config_from_any(config) if config is not None else CriticConfig()
    extra = dict(base.extra)
    return CriticConfig(
        enabled=True,
        metrics=("spatial_accessibility", "functional_dependency"),
        room_stage_hooks=("scene_after_furniture",),
        house_stage_hooks=(),
        inject_into_llm_critic=base.inject_into_llm_critic,
        agent_prompt_context_filter_enabled=base.agent_prompt_context_filter_enabled,
        agent_prompt_context_debug_write=base.agent_prompt_context_debug_write,
        hard_gate=False,
        max_issues_for_prompt=base.max_issues_for_prompt,
        fail_gate_threshold=base.fail_gate_threshold,
        degraded_gate_threshold=base.degraded_gate_threshold,
        asset_annotation=dict(base.asset_annotation),
        extra=extra,
    )


def _evaluate(scene: RoomScene, config: CriticConfig) -> dict[str, Any]:
    return evaluate_room_scene(
        scene,
        config=config,
        stage="scene_after_furniture",
        annotate_assets=False,
    )


def _score_scene(payload: dict[str, Any]) -> tuple[int, int, float]:
    summary = payload.get("summary", {}).get("scene_summary", {})
    return (
        int(summary.get("fail", 0)),
        int(summary.get("degraded", 0)),
        float(summary.get("score", 0.0)),
    )


def _failing_storage_subjects(
    scene: RoomScene, payload: dict[str, Any]
) -> list[str]:
    subjects: list[str] = []
    seen: set[str] = set()
    for result in payload.get("results", []):
        if result.get("metric") != "spatial_accessibility":
            continue
        if result.get("label") != "fail":
            continue
        subject_id = str(result.get("primary_object") or result.get("subject_id") or "")
        if not subject_id or subject_id in seen:
            continue
        obj = scene.objects.get(UniqueID(subject_id))
        if obj is None or obj.object_type != ObjectType.FURNITURE:
            continue
        if _is_storage_like(obj):
            subjects.append(subject_id)
            seen.add(subject_id)
    return subjects


def _best_candidate(
    scene: RoomScene,
    subject: SceneObject,
    *,
    current_score: tuple[int, int, float],
    config: CriticConfig,
    max_translation_m: float,
    step_m: float,
) -> _CandidateScore | None:
    directions = _candidate_directions(scene, subject)
    if not directions:
        return None

    best: _CandidateScore | None = None
    for direction in directions:
        for distance in _candidate_distances(max_translation_m, step_m):
            candidate_scene = deepcopy(scene)
            candidate = candidate_scene.objects[subject.object_id]
            old_pos = candidate.transform.translation()
            new_pos = old_pos.copy()
            new_pos[:2] = old_pos[:2] + direction * distance
            if not _within_floor_bounds(candidate_scene, candidate, new_pos):
                continue
            _move_object_with_surfaces(candidate_scene, candidate.object_id, new_pos)
            payload = _evaluate(candidate_scene, config)
            fail_count, degraded_count, score = _score_scene(payload)
            subject_label, subject_ratio = _subject_access_result(
                payload, str(subject.object_id)
            )
            candidate_score = _CandidateScore(
                scene=candidate_scene,
                xy=(float(new_pos[0]), float(new_pos[1])),
                fail_count=fail_count,
                degraded_count=degraded_count,
                score=score,
                subject_label=subject_label,
                subject_ratio=subject_ratio,
            )
            if _candidate_is_better(candidate_score, best, current_score):
                best = candidate_score

    if best is None:
        return None
    if not _beats_current(best, current_score):
        return None
    return best


def _candidate_is_better(
    candidate: _CandidateScore,
    best: _CandidateScore | None,
    current_score: tuple[int, int, float],
) -> bool:
    if candidate.subject_label == "fail":
        return False
    if not _beats_current(candidate, current_score):
        return False
    if best is None:
        return True
    return (
        candidate.fail_count,
        candidate.degraded_count,
        -_label_rank(candidate.subject_label),
        -candidate.score,
        -candidate.subject_ratio,
    ) < (
        best.fail_count,
        best.degraded_count,
        -_label_rank(best.subject_label),
        -best.score,
        -best.subject_ratio,
    )


def _beats_current(
    candidate: _CandidateScore,
    current_score: tuple[int, int, float],
) -> bool:
    current_fail, current_degraded, current_score_value = current_score
    if candidate.fail_count < current_fail:
        return True
    if candidate.fail_count > current_fail:
        return False
    if candidate.degraded_count < current_degraded:
        return True
    if candidate.degraded_count > current_degraded:
        return False
    return candidate.score > current_score_value + 1e-6


def _subject_access_result(
    payload: dict[str, Any], subject_id: str
) -> tuple[str | None, float]:
    for result in payload.get("results", []):
        if result.get("metric") != "spatial_accessibility":
            continue
        if str(result.get("primary_object") or result.get("subject_id")) != subject_id:
            continue
        diagnostics = result.get("diagnostics") or {}
        return result.get("label"), float(diagnostics.get("access_ratio", 0.0) or 0.0)
    return None, 0.0


def _label_rank(label: str | None) -> int:
    return {"pass": 3, "degraded": 2, "unknown": 1, "fail": 0}.get(label or "", 0)


def _candidate_directions(scene: RoomScene, obj: SceneObject) -> list[np.ndarray]:
    pos = obj.transform.translation()
    bounds = _room_xy_bounds(scene)
    if bounds is None:
        return [
            np.array([1.0, 0.0]),
            np.array([-1.0, 0.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, -1.0]),
        ]

    min_x, max_x, min_y, max_y = bounds
    distances = {
        "west": abs(pos[0] - min_x),
        "east": abs(max_x - pos[0]),
        "south": abs(pos[1] - min_y),
        "north": abs(max_y - pos[1]),
    }
    nearest = min(distances, key=distances.get)
    if nearest in {"north", "south"}:
        return [np.array([1.0, 0.0]), np.array([-1.0, 0.0])]
    return [np.array([0.0, 1.0]), np.array([0.0, -1.0])]


def _candidate_distances(max_translation_m: float, step_m: float) -> list[float]:
    steps = max(1, int(math.floor(max_translation_m / step_m)))
    distances: list[float] = []
    for i in range(1, steps + 1):
        value = round(i * step_m, 6)
        distances.extend([value, -value])
    return distances


def _room_xy_bounds(scene: RoomScene) -> tuple[float, float, float, float] | None:
    geometry = scene.room_geometry
    if geometry is None or geometry.length <= 0 or geometry.width <= 0:
        floor_bounds = (
            geometry.floor.compute_world_bounds()
            if geometry is not None and geometry.floor is not None
            else None
        )
        if floor_bounds is None:
            return None
        bbox_min, bbox_max = floor_bounds
        return (
            float(bbox_min[0]),
            float(bbox_max[0]),
            float(bbox_min[1]),
            float(bbox_max[1]),
        )
    return (
        -float(geometry.length) / 2.0,
        float(geometry.length) / 2.0,
        -float(geometry.width) / 2.0,
        float(geometry.width) / 2.0,
    )


def _within_floor_bounds(
    scene: RoomScene, obj: SceneObject, new_position: np.ndarray
) -> bool:
    bounds = _room_xy_bounds(scene)
    if bounds is None:
        return True
    min_x, max_x, min_y, max_y = bounds
    half_extent = _world_half_extent_xy(obj)
    margin = 0.05
    return (
        min_x + half_extent[0] + margin <= new_position[0] <= max_x - half_extent[0] - margin
        and min_y + half_extent[1] + margin <= new_position[1] <= max_y - half_extent[1] - margin
    )


def _world_half_extent_xy(obj: SceneObject) -> np.ndarray:
    bounds = obj.compute_world_bounds()
    if bounds is None:
        return np.array([0.0, 0.0])
    bbox_min, bbox_max = bounds
    return (bbox_max[:2] - bbox_min[:2]) / 2.0


def _move_object_with_surfaces(
    scene: RoomScene, object_id: UniqueID, new_position: np.ndarray
) -> None:
    obj = scene.objects[object_id]
    old_transform = obj.transform
    moved_surface_ids = {surface.surface_id for surface in obj.support_surfaces}
    old_rpy = RollPitchYaw(old_transform.rotation())
    new_transform = RigidTransform(
        rpy=RollPitchYaw(
            old_rpy.roll_angle(),
            old_rpy.pitch_angle(),
            old_rpy.yaw_angle(),
        ),
        p=new_position,
    )
    delta = new_transform @ old_transform.inverse()
    # 2026-07-09 修改原因：家具横移 guard 需要同步 world-frame support
    # surfaces，否则后续 manipuland stage 会看到旧的支撑面位置。
    scene.move_object(object_id=object_id, new_transform=new_transform)
    for surface in obj.support_surfaces:
        surface.transform = delta @ surface.transform
    _move_children_on_surfaces(scene, moved_surface_ids=moved_surface_ids, delta=delta)


def _move_children_on_surfaces(
    scene: RoomScene, *, moved_surface_ids: set[UniqueID], delta: RigidTransform
) -> None:
    for child in scene.objects.values():
        if child.placement_info is None:
            continue
        if child.placement_info.parent_surface_id not in moved_surface_ids:
            continue
        # 2026-07-09 修改原因：final 回放或后续 guard 若移动已有摆件的
        # storage/table，子物体也必须保持相对 support surface 的世界位置。
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


def _is_storage_like(obj: SceneObject) -> bool:
    tokens = _object_tokens(obj)
    return bool(tokens & STORAGE_TOKENS) and not bool(tokens & SEATING_TOKENS)


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
    normalized = text.lower().replace("-", "_").replace(" ", "_")
    tokens = {token for token in normalized.split("_") if token}
    for token in STORAGE_TOKENS | SEATING_TOKENS:
        if token in normalized:
            tokens.add(token)
    return tokens
