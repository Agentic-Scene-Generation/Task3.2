"""Deterministic last-resort repair for occluded wall-mounted objects."""

from __future__ import annotations

import logging
import math

from dataclasses import dataclass
from collections.abc import Callable, Iterator, Sequence
from typing import Any, Iterable, Protocol, overload

import numpy as np

from scenesmith.agent_utils.room import (
    ObjectType,
    PlacementInfo,
    RoomScene,
    SceneObject,
    UniqueID,
)
from scenesmith.scenebenchmark_critic.api import evaluate_room_scene
from scenesmith.scenebenchmark_critic.auto_repair_stats import record_auto_repair_call
from scenesmith.scenebenchmark_critic.config import CriticConfig, critic_config_from_any

console_logger = logging.getLogger(__name__)

_ISSUE_LABELS = {"fail", "degraded"}
_VISUAL_REPAIRABLE_RELATIONS = {
    "wall_mounted_visibility",
    "wall_mounted_overlap",
}
_WALL_RELATION_REPAIRABLE_RELATIONS = {"seating_to_media"}
_REPAIRABLE_RELATIONS = (
    _VISUAL_REPAIRABLE_RELATIONS | _WALL_RELATION_REPAIRABLE_RELATIONS
)


class WallSurfaceLike(Protocol):
    """Wall-surface operations needed by the critic repair guard."""

    surface_id: UniqueID
    bounding_box_min: list[float]
    bounding_box_max: list[float]

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
    """One accepted transactional wall-object move."""

    object_id: str
    old_wall_surface_id: str
    new_wall_surface_id: str
    old_position: tuple[float, float]
    new_position: tuple[float, float]
    old_issue_count: int
    new_issue_count: int
    old_occlusion_fraction: float
    new_occlusion_fraction: float
    resolved_check_ids: tuple[str, ...] = ()
    moved_object_ids: tuple[str, ...] = ()

    @property
    def wall_surface_id(self) -> str:
        """Compatibility alias for the accepted destination surface."""
        return self.new_wall_surface_id


@dataclass(frozen=True)
class VisualClearanceRejection:
    object_id: str
    reason: str


@dataclass(frozen=True)
class VisualClearanceRepairReport(Sequence[VisualClearanceFix]):
    accepted_fixes: tuple[VisualClearanceFix, ...] = ()
    rejections: tuple[VisualClearanceRejection, ...] = ()

    @overload
    def __getitem__(self, index: int) -> VisualClearanceFix: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[VisualClearanceFix]: ...

    def __getitem__(self, index: int | slice) -> Any:
        return self.accepted_fixes[index]

    def __len__(self) -> int:
        return len(self.accepted_fixes)

    def __iter__(self) -> Iterator[VisualClearanceFix]:
        return iter(self.accepted_fixes)


def improve_wall_visual_clearance(
    scene: RoomScene,
    *,
    wall_surfaces: Iterable[WallSurfaceLike],
    config: Any,
    step_m: float = 0.2,
    max_shift_m: float = 2.0,
    max_repairs: int = 8,
    max_candidate_evaluations: int | None = None,
    on_accept: Callable[[], None] | None = None,
) -> VisualClearanceRepairReport:
    """Repair deterministic wall-object failures after the LLM budget is exhausted.

    Besides visual overlap, the same transactional candidate search repairs hard
    cross-stage relations whose movable endpoint is wall mounted. Related wall
    objects connected by an already-passing explicit relation move as one group,
    which lets a television and its media cabinet change walls without sacrificing
    ``centered_above``.
    """
    critic_config = (
        config if isinstance(config, CriticConfig) else critic_config_from_any(config)
    )
    if not critic_config.enabled:
        record_auto_repair_call(scene, module="visual_clearance", stage="wall_visual_clearance_repair", status="skipped", skip_reason="critic_disabled")
        return VisualClearanceRepairReport()
    if not {"visual_clearance", "functional_dependency"}.intersection(critic_config.metrics):
        record_auto_repair_call(scene, module="visual_clearance", stage="wall_visual_clearance_repair", status="skipped", skip_reason="metric_disabled")
        return VisualClearanceRepairReport()
    if not critic_config.auto_repair.should_repair("visual_clearance"):
        record_auto_repair_call(scene, module="visual_clearance", stage="wall_visual_clearance_repair", status="skipped", skip_reason="module_disabled")
        return VisualClearanceRepairReport()

    configured_budget = critic_config.auto_repair.visual_clearance_budget
    if configured_budget is None:
        configured_budget = critic_config.auto_repair.max_candidate_evaluations
    candidate_budget = 2048 if configured_budget is None else configured_budget
    if max_candidate_evaluations is not None:
        candidate_budget = min(candidate_budget, max_candidate_evaluations)
    if candidate_budget == 0:
        record_auto_repair_call(scene, module="visual_clearance", stage="wall_visual_clearance_repair", status="skipped", skip_reason="budget_zero", candidate_budget=0)
        return VisualClearanceRepairReport()
    repair_limit = max_repairs
    if critic_config.auto_repair.max_repairs_per_call is not None:
        repair_limit = min(repair_limit, critic_config.auto_repair.max_repairs_per_call)

    surfaces = {str(surface.surface_id): surface for surface in wall_surfaces}
    fixes: list[VisualClearanceFix] = []
    rejections: dict[str, VisualClearanceRejection] = {}
    candidate_evaluations = 0
    for _ in range(repair_limit):
        baseline = _evaluate(scene, config)
        baseline_score = _score_payload(baseline)
        issue_ids = _repairable_object_ids(scene, baseline)
        accepted = False
        for object_id in issue_ids:
            obj = scene.objects.get(UniqueID(object_id))
            if obj is None or obj.object_type != ObjectType.WALL_MOUNTED:
                rejections[object_id] = VisualClearanceRejection(
                    object_id, "object_missing_or_not_wall_mounted"
                )
                continue
            placement = obj.placement_info
            if placement is None:
                rejections[object_id] = VisualClearanceRejection(
                    object_id, "missing_wall_placement"
                )
                continue
            surface = surfaces.get(str(placement.parent_surface_id))
            if surface is None:
                rejections[object_id] = VisualClearanceRejection(
                    object_id, "current_wall_surface_missing"
                )
                continue
            remaining_budget = candidate_budget - candidate_evaluations
            if remaining_budget <= 0:
                rejections[object_id] = VisualClearanceRejection(
                    object_id, "candidate_budget_exhausted"
                )
                break
            target_check_ids = _repairable_failure_ids(baseline, object_id)
            if not target_check_ids:
                continue
            best, evaluated, reason = _best_candidate(
                scene,
                obj=obj,
                current_surface=surface,
                surfaces=surfaces.values(),
                config=config,
                baseline_payload=baseline,
                baseline_score=baseline_score,
                step_m=step_m,
                max_shift_m=max_shift_m,
                candidate_budget=remaining_budget,
                target_check_ids=target_check_ids,
            )
            candidate_evaluations += evaluated
            if best is None:
                rejections[object_id] = VisualClearanceRejection(object_id, reason)
                continue
            old_position = tuple(float(value) for value in placement.position_2d)
            (
                destination,
                new_x,
                new_z,
                new_transform,
                new_placement,
                new_score,
                new_payload,
                moved_object_ids,
            ) = best
            _apply_group_candidate(
                scene,
                anchor_id=object_id,
                destination=destination,
                new_x=new_x,
                new_z=new_z,
                group_object_ids=moved_object_ids,
            )
            fix = VisualClearanceFix(
                object_id=object_id,
                old_wall_surface_id=str(surface.surface_id),
                new_wall_surface_id=str(destination.surface_id),
                old_position=old_position,
                new_position=(new_x, new_z),
                old_issue_count=len(target_check_ids),
                new_issue_count=len(target_check_ids & _failing_check_ids(new_payload)),
                old_occlusion_fraction=_object_visual_severity(baseline, object_id),
                new_occlusion_fraction=_object_visual_severity(new_payload, object_id),
                resolved_check_ids=tuple(sorted(target_check_ids)),
                moved_object_ids=moved_object_ids,
            )
            fixes.append(fix)
            if on_accept is not None:
                on_accept()
            rejections.pop(object_id, None)
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
    report = VisualClearanceRepairReport(
        accepted_fixes=tuple(fixes),
        rejections=tuple(rejections[key] for key in sorted(rejections)),
    )
    record_auto_repair_call(
        scene, module="visual_clearance", stage="wall_visual_clearance_repair",
        status="committed" if fixes else "no_targets", candidate_budget=candidate_budget,
        candidates_evaluated=candidate_evaluations, accepted_rounds=len(fixes),
        fix_records=len(fixes),
        object_ids=(object_id for fix in fixes for object_id in (fix.moved_object_ids or (fix.object_id,))),
        relation_types=("visual_clearance" for _ in fixes),
    )
    return report


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
        if str(result.get("scoring_tier") or "").lower() in {
            "ignored",
            "auxiliary",
        }:
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


def _repairable_object_ids(scene: RoomScene, payload: dict[str, Any]) -> list[str]:
    ranked: list[tuple[int, str]] = []
    seen: set[str] = set()
    for result in payload.get("results") or []:
        if str(result.get("scoring_tier") or "").lower() in {
            "ignored",
            "auxiliary",
        }:
            continue
        if result.get("label") not in _ISSUE_LABELS:
            continue
        if result.get("relation_type") not in _REPAIRABLE_RELATIONS:
            continue
        constraint = (result.get("evidence") or {}).get("intent_constraint") or {}
        if constraint and str(constraint.get("strength") or "hard").lower() != "hard":
            continue
        for object_id in sorted(_result_object_ids(result)):
            if object_id in seen:
                continue
            obj = scene.objects.get(UniqueID(object_id))
            if obj is None or obj.object_type != ObjectType.WALL_MOUNTED:
                continue
            seen.add(object_id)
            ranked.append((0 if result.get("label") == "fail" else 1, object_id))
    ranked.sort()
    return [object_id for _, object_id in ranked]


def _repairable_failure_ids(payload: dict[str, Any], object_id: str) -> frozenset[str]:
    return frozenset(
        str(result.get("check_id") or f"result_{index}")
        for index, result in enumerate(payload.get("results") or [])
        if result.get("relation_type") in _REPAIRABLE_RELATIONS
        and result.get("label") in _ISSUE_LABELS
        and str(result.get("scoring_tier") or "core").lower() == "core"
        and object_id in _result_object_ids(result)
        and (
            not (result.get("evidence") or {}).get("intent_constraint")
            or str(
                ((result.get("evidence") or {}).get("intent_constraint") or {}).get(
                    "strength"
                )
                or "hard"
            ).lower()
            == "hard"
        )
    )


def _best_candidate(
    scene: RoomScene,
    *,
    obj: SceneObject,
    current_surface: WallSurfaceLike,
    surfaces: Iterable[WallSurfaceLike],
    config: Any,
    baseline_payload: dict[str, Any],
    baseline_score: _PayloadScore,
    step_m: float,
    max_shift_m: float,
    candidate_budget: int,
    target_check_ids: frozenset[str],
) -> tuple[
    tuple[
        WallSurfaceLike,
        float,
        float,
        Any,
        PlacementInfo,
        _PayloadScore,
        dict[str, Any],
        tuple[str, ...],
    ]
    | None,
    int,
    str,
]:
    placement = obj.placement_info
    if placement is None:
        return None, 0, "missing_wall_placement"
    group = _wall_move_group(scene, baseline_payload, obj)
    snapshots = {
        str(member.object_id): (member.transform, member.placement_info)
        for member in group
    }
    old_x, old_z = (float(value) for value in placement.position_2d)
    object_width = float(obj.bbox_max[0] - obj.bbox_min[0])
    object_height = float(obj.bbox_max[2] - obj.bbox_min[2])
    old_transform = obj.transform
    old_world_center = np.asarray(old_transform.translation(), dtype=float)
    best: (
        tuple[
            WallSurfaceLike,
            float,
            float,
            Any,
            PlacementInfo,
            _PayloadScore,
            dict[str, Any],
            tuple[str, ...],
        ]
        | None
    ) = None
    evaluated = 0
    legal_candidates = 0
    baseline_fail_ids = _core_hard_fail_ids(baseline_payload)
    protected_relations = frozenset().union(
        *(
            _passing_explicit_relation_ids(baseline_payload, str(member.object_id))
            for member in group
        )
    )
    group_object_ids = tuple(sorted(str(member.object_id) for member in group))

    try:
        ordered_surfaces = sorted(
            surfaces,
            key=lambda surface: (
                str(surface.surface_id) != str(current_surface.surface_id),
                str(surface.surface_id),
            ),
        )
        for same_wall in (True, False):
            group_best = None
            group_key = None
            for surface in ordered_surfaces:
                if (
                    str(surface.surface_id) == str(current_surface.surface_id)
                ) != same_wall:
                    continue
                positions = (
                    _candidate_positions(
                        old_x,
                        old_z,
                        step_m=step_m,
                        max_shift_m=max_shift_m,
                    )
                    if same_wall
                    else _surface_grid_positions(
                        surface,
                        object_width=object_width,
                        object_height=object_height,
                        step_m=step_m,
                    )
                )
                for new_x, new_z in positions:
                    if evaluated >= candidate_budget:
                        break
                    valid, _ = surface.check_object_bounds(
                        position_x=new_x,
                        position_z=new_z,
                        object_width=object_width,
                        object_height=object_height,
                    )
                    if not valid:
                        continue
                    member_candidates = _group_candidate_poses(
                        group,
                        anchor=obj,
                        destination=surface,
                        anchor_x=new_x,
                        anchor_z=new_z,
                    )
                    if member_candidates is None:
                        continue
                    legal_candidates += 1
                    for member, transform, candidate_placement in member_candidates:
                        member.transform = transform
                        member.placement_info = candidate_placement
                    new_transform, new_placement = next(
                        (transform, candidate_placement)
                        for member, transform, candidate_placement in member_candidates
                        if member is obj
                    )
                    payload = _evaluate(scene, config)
                    evaluated += 1
                    score = _score_payload(payload)
                    if target_check_ids & _failing_check_ids(payload):
                        continue
                    if _core_hard_fail_ids(payload) - baseline_fail_ids:
                        continue
                    candidate_passing = frozenset().union(
                        *(
                            _passing_explicit_relation_ids(payload, object_id)
                            for object_id in group_object_ids
                        )
                    )
                    if not protected_relations.issubset(candidate_passing):
                        continue
                    if (
                        score.all_fail,
                        score.all_degraded,
                        *_visual_score_key(score),
                    ) >= (
                        baseline_score.all_fail,
                        baseline_score.all_degraded,
                        *_visual_score_key(baseline_score),
                    ):
                        continue
                    world_distance = float(
                        np.linalg.norm(
                            np.asarray(new_transform.translation(), dtype=float)
                            - old_world_center
                        )
                    )
                    key = (
                        score.visual_fail,
                        score.visual_degraded,
                        score.visual_severity,
                        world_distance,
                        str(surface.surface_id),
                        new_x,
                        new_z,
                    )
                    if group_key is None or key < group_key:
                        group_key = key
                        group_best = (
                            surface,
                            new_x,
                            new_z,
                            new_transform,
                            new_placement,
                            score,
                            payload,
                            group_object_ids,
                        )
                if evaluated >= candidate_budget:
                    break
            if group_best is not None:
                best = group_best
                break
            if evaluated >= candidate_budget:
                break
    finally:
        for object_id, (old_transform, old_placement) in snapshots.items():
            member = scene.objects.get(UniqueID(object_id))
            if member is not None:
                member.transform = old_transform
                member.placement_info = old_placement
    if best is not None:
        return best, evaluated, ""
    if evaluated >= candidate_budget:
        return None, evaluated, "candidate_budget_exhausted"
    if legal_candidates == 0:
        return None, evaluated, "no_legal_wall_candidate"
    relation_failure = any(
        result.get("relation_type") in _WALL_RELATION_REPAIRABLE_RELATIONS
        and str(result.get("check_id") or "") in target_check_ids
        for result in baseline_payload.get("results") or []
    )
    return (
        None,
        evaluated,
        (
            "no_candidate_resolved_hard_wall_relation"
            if relation_failure
            else "no_candidate_resolved_visual_issue"
        ),
    )


def _wall_move_group(
    scene: RoomScene, payload: dict[str, Any], anchor: SceneObject
) -> tuple[SceneObject, ...]:
    """Return same-wall objects coupled to the anchor by passing hard relations."""
    anchor_placement = anchor.placement_info
    if anchor_placement is None:
        return (anchor,)
    anchor_surface_id = str(anchor_placement.parent_surface_id)
    adjacency: dict[str, set[str]] = {}
    for result in payload.get("results") or []:
        if str(result.get("label") or "") != "pass":
            continue
        constraint = (result.get("evidence") or {}).get("intent_constraint") or {}
        if not constraint or str(constraint.get("relation") or "") == "required_count":
            continue
        if str(constraint.get("strength") or "hard").lower() != "hard":
            continue
        mounted_ids = []
        for object_id in sorted(_result_object_ids(result)):
            member = scene.objects.get(UniqueID(object_id))
            member_placement = getattr(member, "placement_info", None)
            if (
                member is not None
                and member.object_type == ObjectType.WALL_MOUNTED
                and member_placement is not None
                and str(member_placement.parent_surface_id) == anchor_surface_id
            ):
                mounted_ids.append(object_id)
        for first in mounted_ids:
            adjacency.setdefault(first, set()).update(
                second for second in mounted_ids if second != first
            )

    selected = {str(anchor.object_id)}
    pending = [str(anchor.object_id)]
    while pending:
        current = pending.pop()
        for related in sorted(adjacency.get(current, ())):
            if related not in selected:
                selected.add(related)
                pending.append(related)
    members = [scene.objects.get(UniqueID(object_id)) for object_id in sorted(selected)]
    return tuple(member for member in members if member is not None)


def _group_candidate_poses(
    group: Sequence[SceneObject],
    *,
    anchor: SceneObject,
    destination: WallSurfaceLike,
    anchor_x: float,
    anchor_z: float,
) -> list[tuple[SceneObject, Any, PlacementInfo]] | None:
    """Build one rigid local-layout transfer to a destination wall surface."""
    anchor_placement = anchor.placement_info
    if anchor_placement is None:
        return None
    old_anchor_x, old_anchor_z = (
        float(value) for value in anchor_placement.position_2d
    )
    candidates: list[tuple[SceneObject, Any, PlacementInfo]] = []
    for member in group:
        placement = member.placement_info
        if placement is None:
            return None
        old_x, old_z = (float(value) for value in placement.position_2d)
        new_x = anchor_x + old_x - old_anchor_x
        new_z = anchor_z + old_z - old_anchor_z
        width = float(member.bbox_max[0] - member.bbox_min[0])
        height = float(member.bbox_max[2] - member.bbox_min[2])
        valid, _ = destination.check_object_bounds(
            position_x=new_x,
            position_z=new_z,
            object_width=width,
            object_height=height,
        )
        if not valid:
            return None
        transform = destination.to_world_pose(
            position_x=new_x,
            position_z=new_z,
            rotation_deg=math.degrees(float(placement.rotation_2d)),
        )
        candidates.append(
            (
                member,
                transform,
                PlacementInfo(
                    parent_surface_id=destination.surface_id,
                    position_2d=np.array([new_x, new_z]),
                    rotation_2d=float(placement.rotation_2d),
                    placement_method=placement.placement_method,
                ),
            )
        )
    return candidates


def _apply_group_candidate(
    scene: RoomScene,
    *,
    anchor_id: str,
    destination: WallSurfaceLike,
    new_x: float,
    new_z: float,
    group_object_ids: Sequence[str],
) -> None:
    members = [scene.objects.get(UniqueID(object_id)) for object_id in group_object_ids]
    group = tuple(member for member in members if member is not None)
    anchor = scene.objects.get(UniqueID(anchor_id))
    if anchor is None or len(group) != len(group_object_ids):
        raise ValueError("Accepted wall repair group is no longer present in the scene")
    candidates = _group_candidate_poses(
        group,
        anchor=anchor,
        destination=destination,
        anchor_x=new_x,
        anchor_z=new_z,
    )
    if candidates is None:
        raise ValueError(
            "Accepted wall repair candidate is no longer geometrically valid"
        )
    for member, transform, placement in candidates:
        member.transform = transform
        member.placement_info = placement


def _failing_check_ids(payload: dict[str, Any]) -> frozenset[str]:
    return frozenset(
        str(result.get("check_id") or f"result_{index}")
        for index, result in enumerate(payload.get("results") or [])
        if str(result.get("scoring_tier") or "core").lower() == "core"
        and str(result.get("label") or "").lower() in _ISSUE_LABELS
    )


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


def _visual_score_key(score: _PayloadScore) -> tuple[int, int, float]:
    return score.visual_fail, score.visual_degraded, score.visual_severity


def _core_hard_fail_ids(payload: dict[str, Any]) -> frozenset[str]:
    return frozenset(
        str(result.get("check_id") or f"result_{index}")
        for index, result in enumerate(payload.get("results") or [])
        if str(result.get("scoring_tier") or "core").lower() == "core"
        and str(result.get("label") or "").lower() == "fail"
    )


def _result_object_ids(result: dict[str, Any]) -> set[str]:
    diagnostics = result.get("diagnostics") or {}
    return {
        str(value)
        for value in (
            result.get("primary_object"),
            *(result.get("related_objects") or []),
            *(result.get("selected_related_objects") or []),
            *(diagnostics.get("selected_target_ids") or []),
        )
        if str(value or "")
    }


def _passing_explicit_relation_ids(
    payload: dict[str, Any], object_id: str
) -> frozenset[str]:
    protected: set[str] = set()
    for index, result in enumerate(payload.get("results") or []):
        if str(result.get("label") or "") != "pass":
            continue
        constraint = (result.get("evidence") or {}).get("intent_constraint") or {}
        if not constraint or str(constraint.get("relation") or "") == "required_count":
            continue
        if object_id not in _result_object_ids(result):
            continue
        protected.add(str(result.get("check_id") or f"result_{index}"))
    return frozenset(protected)


def _object_visual_severity(payload: dict[str, Any], object_id: str) -> float:
    severity = 0.0
    for result in payload.get("results") or []:
        if result.get("metric") != "visual_clearance":
            continue
        if object_id not in _result_object_ids(result):
            continue
        diagnostics = result.get("diagnostics") or {}
        severity = max(
            severity,
            float(
                diagnostics.get("occluded_fraction")
                or diagnostics.get("overlap_ratio")
                or 0.0
            ),
        )
    return round(severity, 9)


def _surface_grid_positions(
    surface: WallSurfaceLike,
    *,
    object_width: float,
    object_height: float,
    step_m: float,
) -> list[tuple[float, float]]:
    lower = np.asarray(surface.bounding_box_min, dtype=float)
    upper = np.asarray(surface.bounding_box_max, dtype=float)
    x_min = float(lower[0]) + object_width / 2.0
    x_max = float(upper[0]) - object_width / 2.0
    z_min = float(lower[2]) + object_height / 2.0
    z_max = float(upper[2]) - object_height / 2.0
    if x_min > x_max or z_min > z_max:
        return []

    def values(start: float, end: float) -> list[float]:
        count = max(0, int(math.floor((end - start) / step_m)))
        result = [round(start + index * step_m, 6) for index in range(count + 1)]
        if not result or end - result[-1] > 1e-6:
            result.append(round(end, 6))
        return result

    return [
        (position_x, position_z)
        for position_z in values(z_min, z_max)
        for position_x in values(x_min, x_max)
    ]


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
