"""Deterministic last-resort repair for occluded wall-mounted objects."""

from __future__ import annotations

import logging
import math

from dataclasses import dataclass
from typing import Any, Iterable, Protocol

import numpy as np

from scenesmith.agent_utils.room import (
    ObjectType,
    PlacementInfo,
    RoomScene,
    SceneObject,
    UniqueID,
)
from scenesmith.scenebenchmark_critic.api import evaluate_room_scene
from scenesmith.scenebenchmark_critic.config import CriticConfig, critic_config_from_any

console_logger = logging.getLogger(__name__)

_ISSUE_LABELS = {"fail", "degraded"}
_REPAIRABLE_RELATIONS = {"wall_mounted_visibility", "wall_mounted_overlap"}


class WallSurfaceLike(Protocol):
    """Wall-surface operations needed by the critic repair guard."""

    surface_id: UniqueID

    def check_object_bounds(
        self,
        position_x: float,
        position_z: float,
        object_width: float,
        object_height: float,
    ) -> tuple[bool, str | None]: ...

    def to_world_pose(
        self, position_x: float, position_z: float, rotation_deg: float = 0.0
    ) -> Any: ...


@dataclass(frozen=True)
class VisualClearanceFix:
    """One accepted same-wall move."""

    object_id: str
    wall_surface_id: str
    old_position: tuple[float, float]
    new_position: tuple[float, float]
    old_issue_count: int
    new_issue_count: int


def improve_wall_visual_clearance(
    scene: RoomScene,
    *,
    wall_surfaces: Iterable[WallSurfaceLike],
    config: Any,
    step_m: float = 0.2,
    max_shift_m: float = 2.0,
    max_repairs: int = 8,
) -> list[VisualClearanceFix]:
    """Move still-occluded wall objects after the LLM repair budget is exhausted."""
    critic_config = (
        config if isinstance(config, CriticConfig) else critic_config_from_any(config)
    )
    if not critic_config.enabled or "visual_clearance" not in critic_config.metrics:
        return []

    surfaces = {str(surface.surface_id): surface for surface in wall_surfaces}
    fixes: list[VisualClearanceFix] = []
    for _ in range(max_repairs):
        baseline = _evaluate(scene, config)
        baseline_score = _score_payload(baseline)
        issue_ids = _repairable_object_ids(baseline)
        accepted = False
        for object_id in issue_ids:
            obj = scene.objects.get(UniqueID(object_id))
            if obj is None or obj.object_type != ObjectType.WALL_MOUNTED:
                continue
            placement = obj.placement_info
            if placement is None:
                continue
            surface = surfaces.get(str(placement.parent_surface_id))
            if surface is None:
                continue
            best = _best_candidate(
                scene,
                obj=obj,
                surface=surface,
                config=config,
                baseline_score=baseline_score,
                step_m=step_m,
                max_shift_m=max_shift_m,
            )
            if best is None:
                continue
            old_position = tuple(float(value) for value in placement.position_2d)
            new_x, new_z, new_transform, new_placement, new_score = best
            obj.transform = new_transform
            obj.placement_info = new_placement
            fixes.append(
                VisualClearanceFix(
                    object_id=object_id,
                    wall_surface_id=str(surface.surface_id),
                    old_position=old_position,
                    new_position=(new_x, new_z),
                    old_issue_count=baseline_score.issue_count,
                    new_issue_count=new_score.issue_count,
                )
            )
            accepted = True
            break
        if not accepted:
            break

    if fixes:
        console_logger.info(
            "Visual-clearance guard moved %d wall object(s): %s",
            len(fixes),
            ", ".join(
                f"{fix.object_id} ({fix.old_position[0]:.2f},"
                f"{fix.old_position[1]:.2f})->({fix.new_position[0]:.2f},"
                f"{fix.new_position[1]:.2f})"
                for fix in fixes
            ),
        )
    return fixes


@dataclass(frozen=True, order=True)
class _PayloadScore:
    all_fail: int
    all_degraded: int
    visual_fail: int
    visual_degraded: int
    visual_severity: float

    @property
    def issue_count(self) -> int:
        return self.visual_fail + self.visual_degraded


def _evaluate(scene: RoomScene, config: Any) -> dict[str, Any]:
    return evaluate_room_scene(
        scene,
        config=config,
        stage="wall_visual_clearance_repair",
        annotate_assets=False,
    )


def _score_payload(payload: dict[str, Any]) -> _PayloadScore:
    all_fail = all_degraded = visual_fail = visual_degraded = 0
    severity = 0.0
    for result in payload.get("results") or []:
        if str(result.get("scoring_tier") or "").lower() == "ignored":
            continue
        label = str(result.get("label") or "")
        if label == "fail":
            all_fail += 1
        elif label == "degraded":
            all_degraded += 1
        if result.get("metric") != "visual_clearance":
            continue
        if label == "fail":
            visual_fail += 1
        elif label == "degraded":
            visual_degraded += 1
        diagnostics = result.get("diagnostics") or {}
        severity += float(
            diagnostics.get("occluded_fraction")
            or diagnostics.get("overlap_ratio")
            or 0.0
        )
    return _PayloadScore(
        all_fail=all_fail,
        all_degraded=all_degraded,
        visual_fail=visual_fail,
        visual_degraded=visual_degraded,
        visual_severity=round(severity, 9),
    )


def _repairable_object_ids(payload: dict[str, Any]) -> list[str]:
    ranked: list[tuple[int, str]] = []
    seen: set[str] = set()
    for result in payload.get("results") or []:
        if result.get("metric") != "visual_clearance":
            continue
        if result.get("label") not in _ISSUE_LABELS:
            continue
        if result.get("relation_type") not in _REPAIRABLE_RELATIONS:
            continue
        object_id = str(result.get("primary_object") or "")
        if not object_id or object_id in seen:
            continue
        seen.add(object_id)
        ranked.append((0 if result.get("label") == "fail" else 1, object_id))
    ranked.sort()
    return [object_id for _, object_id in ranked]


def _best_candidate(
    scene: RoomScene,
    *,
    obj: SceneObject,
    surface: WallSurfaceLike,
    config: Any,
    baseline_score: _PayloadScore,
    step_m: float,
    max_shift_m: float,
) -> tuple[float, float, Any, PlacementInfo, _PayloadScore] | None:
    placement = obj.placement_info
    if placement is None:
        return None
    old_transform = obj.transform
    old_placement = placement
    old_x, old_z = (float(value) for value in placement.position_2d)
    rotation_degrees = math.degrees(float(placement.rotation_2d))
    object_width = float(obj.bbox_max[0] - obj.bbox_min[0])
    object_height = float(obj.bbox_max[2] - obj.bbox_min[2])
    best: tuple[float, float, Any, PlacementInfo, _PayloadScore] | None = None
    best_key: tuple[_PayloadScore, float, float, float] | None = None

    try:
        for new_x, new_z in _candidate_positions(
            old_x,
            old_z,
            step_m=step_m,
            max_shift_m=max_shift_m,
        ):
            valid, _ = surface.check_object_bounds(
                position_x=new_x,
                position_z=new_z,
                object_width=object_width,
                object_height=object_height,
            )
            if not valid:
                continue
            new_transform = surface.to_world_pose(
                position_x=new_x,
                position_z=new_z,
                rotation_deg=rotation_degrees,
            )
            new_placement = PlacementInfo(
                parent_surface_id=surface.surface_id,
                position_2d=np.array([new_x, new_z]),
                rotation_2d=float(placement.rotation_2d),
                placement_method=placement.placement_method,
            )
            obj.transform = new_transform
            obj.placement_info = new_placement
            payload = _evaluate(scene, config)
            score = _score_payload(payload)
            if score >= baseline_score or _object_still_has_visual_issue(
                payload, str(obj.object_id)
            ):
                continue
            move_distance = math.hypot(new_x - old_x, new_z - old_z)
            key = (score, move_distance, new_z - old_z, new_x)
            if best_key is None or key < best_key:
                best_key = key
                best = (new_x, new_z, new_transform, new_placement, score)
    finally:
        obj.transform = old_transform
        obj.placement_info = old_placement
    return best


def _object_still_has_visual_issue(payload: dict[str, Any], object_id: str) -> bool:
    for result in payload.get("results") or []:
        if result.get("metric") != "visual_clearance":
            continue
        if result.get("label") not in _ISSUE_LABELS:
            continue
        involved = {
            str(result.get("primary_object") or ""),
            *(str(value) for value in result.get("related_objects") or []),
        }
        if object_id in involved:
            return True
    return False


def _candidate_positions(
    old_x: float,
    old_z: float,
    *,
    step_m: float,
    max_shift_m: float,
) -> list[tuple[float, float]]:
    steps = max(1, int(math.floor(max_shift_m / step_m)))
    positions: list[tuple[float, float]] = []
    for index in range(1, steps + 1):
        delta = round(index * step_m, 6)
        positions.extend(((old_x + delta, old_z), (old_x - delta, old_z)))
    for index in range(1, steps + 1):
        delta = round(index * step_m, 6)
        positions.extend(((old_x, old_z + delta), (old_x, old_z - delta)))
    for index in range(1, steps + 1):
        delta = round(index * step_m, 6)
        for horizontal_sign in (1.0, -1.0):
            positions.append((old_x + horizontal_sign * delta, old_z + delta))
    return positions
