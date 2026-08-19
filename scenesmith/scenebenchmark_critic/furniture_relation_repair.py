"""Critic-driven deterministic repair for furniture relations."""

from __future__ import annotations

import logging
import math
import re

from itertools import permutations
from collections.abc import Callable, Collection
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from pydrake.math import RigidTransform, RollPitchYaw, RotationMatrix
from shapely.affinity import translate as translate_polygon
from shapely.geometry import Polygon

from scenesmith.agent_utils.clearance_zones import (
    compute_door_clearance_violations,
    door_swing_clearance_bounds,
)
from scenesmith.agent_utils.furniture_accessibility_guard import (
    improve_storage_front_access,
)
from scenesmith.agent_utils.physics_validation import compute_scene_collisions
from scenesmith.agent_utils.room import (
    ObjectType,
    PlacementInfo,
    RoomScene,
    SceneObject,
    UniqueID,
)
from scenesmith.scenebenchmark_critic.api import evaluate_room_scene
from scenesmith.scenebenchmark_critic.config import CriticConfig, critic_config_from_any
from scenesmith.scenebenchmark_critic.opening_geometry import opening_physical_bounds
from scenesmith.scenebenchmark_critic.relation_registry import (
    relation_spec,
    repair_relation_types,
)
from scenesmith.utils.geometry_utils import compute_optimal_facing_yaw

if TYPE_CHECKING:
    from scenesmith.wall_agents.tools.wall_surface import WallSurface

console_logger = logging.getLogger(__name__)

_ISSUE_LABELS = {"fail", "degraded"}
_FURNITURE_REPAIR_STRATEGIES = (
    "furniture_relation",
    "edge_distribution",
    "support_relation",
)
_FURNITURE_GATE_STRATEGIES = (
    "furniture_relation",
    "edge_distribution",
)
_SYSTEM_REPAIRABLE_RELATIONS = frozenset(
    {"activity_zone_separation", "room_containment"}
)
_PAIRED_SURFACE_RELATION = "furniture_faces_furniture"
_OPENING_CLEARANCE_MARGIN_M = 0.03
_WALL_BACKED_CONTACT_GAP_M = 0.03
# ``back_against_wall`` accepts a 0.25 m gap by default.  Retain a small
# numerical margin when a rotation-only repair needs the less disruptive
# in-tolerance pose instead of forcing a newly conflicting flush placement.
_WALL_BACKED_DEFAULT_MAX_GAP_M = 0.25
_WALL_BACKED_GAP_MARGIN_M = 0.02
_ROOM_CONTAINMENT_MARGIN_M = 0.02
# A room-center repair only needs to enter the evaluator's allowed region.  A
# small interior margin avoids reintroducing nearby clearance conflicts solely
# to reach an arbitrary exact coordinate.
_ROOM_CENTER_REPAIR_MARGIN_M = 0.01
_WORK_SEAT_SURFACE_GAP_M = 0.12
_WORKSTATION_AISLE_REPAIR_MARGIN_M = 0.01
_LOCAL_ACCESS_REPAIR_MARGIN_M = 0.02
# The functional contract measures corner proximity from the object center,
# while collision meshes can extend beyond a rendered asset's display bounds.
# Keep a compact set of inward alternatives so a semantically valid corner
# target does not have to be flush with both walls.
_CORNER_REPAIR_MARGINS_M = (0.04, 0.12, 0.20)
_ROOM_BOUNDED_TOPOLOGY_RELATIONS = frozenset(
    {
        "corner_of_room",
        "corner_distribution",
        "edge_distribution",
        "activity_zone_separation",
        "room_containment",
    }
)


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
class _RepairPose:
    object_id: str
    target_center_xy: tuple[float, float]
    target_yaw_deg: float | None
    target_center_z: float | None = None


@dataclass(frozen=True)
class _RepairTarget:
    object_id: str
    relation_type: str
    check_id: str
    target_center_xy: tuple[float, float]
    target_yaw_deg: float | None
    group_object_ids: tuple[str, ...] = ()
    member_poses: tuple[_RepairPose, ...] = ()
    target_center_z: float | None = None
    target_wall_id: str | None = None


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


@dataclass
class _ScenePoseSnapshot:
    """Reversible pose-only snapshot used while scoring one repair candidate.

    A ``RoomScene`` can contain large trimesh objects inside support surfaces.
    Copying the scene therefore copies geometry, not just the pose being tested.
    Relation repair only mutates transforms, so retain pose-sized transform
    snapshots and surface references instead of cloning the complete scene.
    """

    object_transforms: dict[UniqueID, RigidTransform]
    object_placements: dict[UniqueID, PlacementInfo | None]
    surface_states: dict[
        UniqueID,
        tuple[list[Any], tuple[tuple[Any, RigidTransform], ...]],
    ]

    @classmethod
    def capture(cls, scene: RoomScene) -> "_ScenePoseSnapshot":
        surface_states: dict[
            UniqueID,
            tuple[list[Any], tuple[tuple[Any, RigidTransform], ...]],
        ] = {}
        object_transforms: dict[UniqueID, RigidTransform] = {}
        object_placements: dict[UniqueID, PlacementInfo | None] = {}
        for object_id, obj in scene.objects.items():
            object_transforms[object_id] = RigidTransform(
                R=obj.transform.rotation(), p=obj.transform.translation()
            )
            object_placements[object_id] = _copy_placement_info(obj.placement_info)
            surfaces = obj.support_surfaces
            surface_states[object_id] = (
                surfaces,
                tuple(
                    (
                        surface,
                        RigidTransform(
                            R=surface.transform.rotation(),
                            p=surface.transform.translation(),
                        ),
                    )
                    for surface in surfaces
                ),
            )
        return cls(object_transforms, object_placements, surface_states)

    def restore(self, scene: RoomScene) -> None:
        for object_id, transform in self.object_transforms.items():
            obj = scene.objects.get(object_id)
            if obj is None:
                continue
            obj.transform = transform
            obj.placement_info = _copy_placement_info(
                self.object_placements.get(object_id)
            )
            surfaces, states = self.surface_states[object_id]
            # Preserve the original list identity, but also restore its
            # contents in case an evaluator mutated the container in place.
            surfaces[:] = [surface for surface, _ in states]
            obj.support_surfaces = surfaces
            for surface, surface_transform in states:
                surface.transform = surface_transform


def _copy_placement_info(placement: PlacementInfo | None) -> PlacementInfo | None:
    if placement is None:
        return None
    return PlacementInfo(
        parent_surface_id=placement.parent_surface_id,
        position_2d=np.asarray(placement.position_2d, dtype=float).copy(),
        rotation_2d=float(placement.rotation_2d),
        placement_method=str(placement.placement_method),
    )


def improve_furniture_relations(
    scene: RoomScene,
    *,
    config: CriticConfig | Any | None = None,
    max_repairs: int = 8,
    max_translation_m: float = 3.0,
    max_candidate_evaluations: int = 64,
    allowed_relation_types: Collection[str] | None = None,
    candidate_validator: Callable[[RoomScene], bool] | None = None,
) -> list[FurnitureRelationFix]:
    """Apply selected critic targets when the whole-scene evaluation improves.

    ``allowed_relation_types=None`` preserves the historical behavior of repairing
    every supported relation. Callers may provide a narrow allowlist for a delayed
    stage that owns only one relation family.  ``candidate_validator`` can add a
    stage-owned hard-constraint gate, such as physics validation, before a pose is
    accepted.  Relation metrics alone cannot observe every hard scene invariant.
    """
    critic_config = (
        config if isinstance(config, CriticConfig) else critic_config_from_any(config)
    )
    if not critic_config.enabled or not critic_config.metric_enabled(
        "functional_dependency"
    ):
        return []

    try:
        configured_budget = int(
            critic_config.extra.get(
                "relation_repair_max_candidate_evaluations",
                max_candidate_evaluations,
            )
        )
    except (TypeError, ValueError):
        configured_budget = max_candidate_evaluations
    candidate_budget = max(1, configured_budget)
    relation_allowlist = (
        None
        if allowed_relation_types is None
        else frozenset(str(value) for value in allowed_relation_types)
    )
    evaluation_stage = (
        "wall_visual_clearance_repair"
        if relation_allowlist
        and "instructional_surface_alignment" in relation_allowlist
        else "furniture_relation_repair"
    )

    def evaluate_for_repair() -> dict[str, Any]:
        if evaluation_stage == "furniture_relation_repair":
            return _evaluate(scene, critic_config)
        return _evaluate(scene, critic_config, evaluation_stage)

    candidate_evaluations = 0
    fixes: list[FurnitureRelationFix] = []
    for _ in range(max_repairs):
        baseline_payload = evaluate_for_repair()
        baseline_score = _score_payload(baseline_payload)
        accepted = False
        for target in _repair_targets(scene, baseline_payload):
            if (
                relation_allowlist is not None
                and target.relation_type not in relation_allowlist
            ):
                continue
            if candidate_evaluations >= candidate_budget:
                break
            obj = scene.objects.get(UniqueID(target.object_id))
            if obj is None or not _repair_target_type_allowed(
                obj, target.relation_type
            ):
                continue
            current_center = _world_center_xy(obj)
            if current_center is None:
                continue
            translation_limit = max_translation_m
            if target.relation_type in {
                "seating_to_media",
                "seating_to_work_surface",
            }:
                translation_limit *= 3.0
            elif target.relation_type == "front_axis_alignment" and target.member_poses:
                # A wall-side anchor can need to traverse the room to align with
                # a centered subject. Its explicitly near dependents must travel
                # with it, and their valid final slots may be farther than the
                # single-object repair budget.
                translation_limit *= 1.5
            elif target.relation_type in _ROOM_BOUNDED_TOPOLOGY_RELATIONS:
                # These targets are canonical positions computed from either
                # the room boundary or a target-local table frame. Inventory
                # recovery can add an object at a distant valid fallback pose,
                # so the local default must not discard its only legal target
                # before the normal whole-scene safety gates evaluate it.
                translation_limit = max(translation_limit, _room_diagonal_m(scene))
            elif target.relation_type == "object_on_support":
                # A hard media/support contract can be discovered only after a
                # display was placed on a distant wall or outside the room. The
                # candidate is still subject to the whole-scene no-regression
                # gate, so it may traverse the room but never farther than its
                # diagonal. This budget must not depend on window-clearance moves.
                translation_limit = max(translation_limit, _room_diagonal_m(scene))
            if _target_max_translation(scene, target) > translation_limit:
                continue
            snapshot = _ScenePoseSnapshot.capture(scene)
            keep_candidate = False
            baseline_collision_pairs: set[tuple[str, str]] | None = None
            if target.relation_type == "room_containment":
                try:
                    baseline_collision_pairs = {
                        tuple(sorted((pair.object_a_id, pair.object_b_id)))
                        for pair in compute_scene_collisions(scene)
                    }
                except Exception:
                    console_logger.warning(
                        "Could not validate baseline collisions for containment repair",
                        exc_info=True,
                    )
                    continue
            try:
                candidate_obj = scene.objects[UniqueID(target.object_id)]
                new_transform = _transform_for_target(candidate_obj, target)
                if new_transform is None:
                    continue
                if not _apply_repair_target(scene, target, new_transform):
                    continue
                if target.relation_type == "room_center_alignment":
                    # Centering a dining group can narrow the wall-side service
                    # aisle.  Resolve that secondary effect inside this same
                    # rollback scope so the strict whole-scene gate evaluates
                    # the coordinated layout, not an intentionally temporary
                    # accessibility regression.
                    improve_storage_front_access(
                        scene,
                        config=critic_config,
                        max_candidate_evaluations=16,
                        repair_degraded=True,
                    )
                if baseline_collision_pairs is not None:
                    try:
                        candidate_collision_pairs = {
                            tuple(sorted((pair.object_a_id, pair.object_b_id)))
                            for pair in compute_scene_collisions(scene)
                        }
                    except Exception:
                        console_logger.warning(
                            "Containment repair collision validation failed",
                            exc_info=True,
                        )
                        continue
                    if candidate_collision_pairs - baseline_collision_pairs:
                        continue
                candidate_evaluations += 1
                if candidate_validator is not None:
                    try:
                        candidate_valid = bool(candidate_validator(scene))
                    except Exception:
                        console_logger.warning(
                            "Relation repair candidate validator failed; "
                            "rejecting candidate %s",
                            target.object_id,
                            exc_info=True,
                        )
                        candidate_valid = False
                    if not candidate_valid:
                        continue
                candidate_payload = evaluate_for_repair()
                if not _candidate_improves(
                    baseline_payload,
                    candidate_payload,
                    baseline_score=baseline_score,
                    check_id=target.check_id,
                ):
                    continue

                moved_ids = tuple(pose.object_id for pose in target.member_poses) or (
                    target.object_id,
                )
                for moved_id in moved_ids:
                    moved_key = UniqueID(moved_id)
                    old_transform = snapshot.object_transforms.get(moved_key)
                    moved_obj = scene.objects.get(moved_key)
                    if old_transform is None or moved_obj is None:
                        continue
                    old_rpy = RollPitchYaw(old_transform.rotation())
                    old_xy = tuple(
                        float(value) for value in old_transform.translation()[:2]
                    )
                    new_rpy = RollPitchYaw(moved_obj.transform.rotation())
                    new_xy = tuple(
                        float(value) for value in moved_obj.transform.translation()[:2]
                    )
                    old_yaw_deg = math.degrees(old_rpy.yaw_angle())
                    new_yaw_deg = math.degrees(new_rpy.yaw_angle())
                    yaw_delta = abs((new_yaw_deg - old_yaw_deg + 180.0) % 360.0 - 180.0)
                    if (
                        math.hypot(new_xy[0] - old_xy[0], new_xy[1] - old_xy[1]) < 1e-7
                        and yaw_delta < 1e-7
                    ):
                        continue
                    fixes.append(
                        FurnitureRelationFix(
                            object_id=moved_id,
                            relation_type=target.relation_type,
                            check_id=target.check_id,
                            old_xy=old_xy,
                            new_xy=new_xy,
                            old_yaw_deg=old_yaw_deg,
                            new_yaw_deg=new_yaw_deg,
                        )
                    )
                keep_candidate = True
                accepted = True
                break
            finally:
                if not keep_candidate:
                    snapshot.restore(scene)
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
    console_logger.info(
        "Furniture relation repair evaluated %d candidate pose(s) (budget=%d)",
        candidate_evaluations,
        candidate_budget,
    )
    return fixes


def unresolved_furniture_relation_failures(
    scene: RoomScene,
    *,
    config: CriticConfig | Any | None = None,
) -> list[dict[str, Any]]:
    """Return unresolved core furniture-relation failures after final repair."""
    critic_config = (
        config if isinstance(config, CriticConfig) else critic_config_from_any(config)
    )
    if not critic_config.enabled or not critic_config.metric_enabled(
        "functional_dependency"
    ):
        return []

    payload = _evaluate(scene, critic_config)
    return [
        result
        for result in payload.get("results") or []
        if str(result.get("label") or "").lower() == "fail"
        and str(result.get("scoring_tier") or "").lower()
        not in {"ignored", "auxiliary"}
        and _is_due_at_furniture_stage(result)
        and (
            str(result.get("relation_type") or "")
            in repair_relation_types(strategies=_FURNITURE_GATE_STRATEGIES)
            or str(result.get("relation_type") or "") in _SYSTEM_REPAIRABLE_RELATIONS
            or _is_required_media_on_support_result(payload, scene, result)
            or _is_paired_surface_facing_result(payload, result)
            or _is_hard_furniture_contract_failure(result)
        )
    ]


def _is_due_at_furniture_stage(result: dict[str, Any]) -> bool:
    """Keep later-stage contract failures out of the furniture hard gate.

    The relation registry describes geometry semantics, while the compiled
    contract assigns ownership to a generation stage.  A monitor-on-desk
    relation, for example, is validly unresolved until the manipuland stage
    creates the monitor.  Legacy or inferred results without contract stage
    metadata retain the existing furniture-stage behavior.
    """
    constraint = (result.get("evidence") or {}).get("intent_constraint") or {}
    stage = str(constraint.get("stage") or "").strip().lower()
    return not stage or stage == "furniture"


def _is_hard_furniture_contract_failure(result: dict[str, Any]) -> bool:
    constraint = (result.get("evidence") or {}).get("intent_constraint") or {}
    return bool(
        constraint
        and str(constraint.get("stage") or "") == "furniture"
        and str(constraint.get("strength") or "").lower() == "hard"
        and str(result.get("contract_state") or "") == "failed"
    )


def _evaluate(
    scene: RoomScene,
    config: CriticConfig,
    stage: str = "furniture_relation_repair",
) -> dict[str, Any]:
    return evaluate_room_scene(
        scene,
        config=config,
        stage=stage,
        annotate_assets=False,
    )


def _score_payload(payload: dict[str, Any]) -> _PayloadScore:
    labels: dict[str, str] = {}
    fail_ids: set[str] = set()
    degraded_ids: set[str] = set()
    for index, result in enumerate(payload.get("results") or []):
        if str(result.get("scoring_tier") or "").lower() in {"ignored", "auxiliary"}:
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
    introduced_failures = candidate_score.fail_ids - baseline_score.fail_ids
    target_result = _result_by_id(baseline_payload, check_id)
    target_constraint = ((target_result or {}).get("evidence") or {}).get(
        "intent_constraint"
    ) or {}
    defers_window_repair = bool(
        introduced_failures
        and all(
            failure_id.startswith("window_clearance__")
            for failure_id in introduced_failures
        )
        and str(target_constraint.get("strength") or "").lower() == "hard"
        and str(target_constraint.get("stage") or "furniture") == "furniture"
    )
    # Hard failures are the primary repair objective.  An exact functional
    # slot can legitimately turn a heuristic accessibility check from pass to
    # degraded (for example, moving four remote dining chairs back to the four
    # requested table edges).  Rejecting every new soft degradation before
    # comparing hard failures leaves the original, unusable layout untouched.
    #
    # Never introduce a new hard failure except a window-clearance failure
    # created while satisfying an explicit hard furniture relation. The
    # stateful furniture transaction immediately hands that case to atomic
    # window migration and rolls the relation move back if migration fails.
    # When at least one hard failure is removed, compare scores
    # lexicographically and allow a soft degradation; later repair rounds can
    # improve it. Without a hard-failure reduction, retain the strict
    # no-regression behavior.
    if introduced_failures and not defers_window_repair:
        return False
    if candidate_score.global_key > baseline_score.global_key:
        return False
    hard_fail_reduced = len(candidate_score.fail_ids) < len(baseline_score.fail_ids)
    if (
        not hard_fail_reduced
        and candidate_score.degraded_ids - baseline_score.degraded_ids
    ):
        return False
    for existing_id, old_label in baseline_score.labels.items():
        if existing_id == check_id:
            continue
        new_label = candidate_score.labels.get(existing_id)
        if new_label is None:
            continue
        if not hard_fail_reduced and _label_severity(new_label) > _label_severity(
            old_label
        ):
            return False
    old_result = _result_by_id(baseline_payload, check_id)
    new_result = _result_by_id(candidate_payload, check_id)
    if old_result is None or new_result is None:
        return False
    if (
        str(old_result.get("relation_type") or "") == "activity_zone_separation"
        and str(new_result.get("label") or "").lower() != "pass"
    ):
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
    if relation == "edge_distribution":
        for slot in (
            diagnostics.get("seat_slots") or diagnostics.get("edge_slots") or []
        ):
            magnitude += max(
                0.0,
                float(slot.get("deviation_m") or 0.0)
                - float(slot.get("allowed_deviation_m") or 0.0),
            )
            normal_deviation = slot.get("normal_deviation_m")
            if normal_deviation is not None:
                magnitude += max(
                    0.0,
                    float(normal_deviation)
                    - float(slot.get("allowed_normal_deviation_m") or 0.0),
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
    elif relation == "room_center_alignment":
        magnitude = max(
            0.0,
            float(diagnostics.get("offset_m") or 0.0)
            - float(diagnostics.get("allowed_offset_m") or 0.0),
        )
    elif relation == "seating_to_media":
        magnitude = max(
            0.0,
            float(diagnostics.get("lateral_offset_m") or 0.0)
            - float(diagnostics.get("media_axis_pass_offset_m") or 0.0),
        )
    elif relation == "activity_zone_separation":
        magnitude = max(
            0.0,
            float(diagnostics.get("intersection_area_m2") or 0.0),
        )
    elif relation in {"between_alignment", "centered_between_alignment"}:
        if relation == "centered_between_alignment":
            magnitude = max(
                0.0,
                float(diagnostics.get("midpoint_error_m") or 0.0)
                - float(diagnostics.get("midpoint_tolerance_m") or 0.0),
            )
        else:
            fraction = float(diagnostics.get("segment_fraction") or 0.0)
            magnitude = max(0.0, 0.10 - fraction, fraction - 0.90) + max(
                0.0,
                float(diagnostics.get("lateral_offset_m") or 0.0)
                - float(diagnostics.get("lateral_tolerance_m") or 0.0),
            )
    elif relation == "flanking":
        signs = set(int(value) for value in diagnostics.get("side_signs") or [])
        magnitude = float(int(1 not in signs) + int(-1 not in signs))
    elif relation == "centered_on_wall":
        magnitude = max(
            0.0,
            float(diagnostics.get("tangent_error_m") or 0.0)
            - float(diagnostics.get("allowed_tangent_error_m") or 0.0),
        ) + max(0.0, float(diagnostics.get("normal_error_m") or 0.0) - 0.10)
    elif relation in {"front_axis_alignment", "rear_axis_alignment"}:
        magnitude = max(
            0.0,
            float(diagnostics.get("minimum_forward_distance_m") or 0.0)
            - float(diagnostics.get("forward_distance_m") or 0.0),
        ) + max(
            0.0,
            float(diagnostics.get("lateral_offset_m") or 0.0)
            - float(diagnostics.get("lateral_tolerance_m") or 0.0),
        )
    return _label_severity(label), round(magnitude, 9)


def _label_severity(label: str) -> int:
    return {"pass": 0, "degraded": 1, "fail": 2, "unknown": 3}.get(label, 3)


def _is_registry_repairable_relation(relation_type: str) -> bool:
    return relation_type in _SYSTEM_REPAIRABLE_RELATIONS or relation_type in (
        repair_relation_types(strategies=_FURNITURE_REPAIR_STRATEGIES)
    )


@dataclass(frozen=True)
class _RepairHandlerContext:
    scene: RoomScene
    payload: dict[str, Any]
    result: dict[str, Any]
    check_id: str
    relation: str
    diagnostics: dict[str, Any]
    coordinated_front_checks: set[str]
    claimed_near_checks: set[str]


def _paired_surface_targets(context: _RepairHandlerContext) -> list[_RepairTarget]:
    facing_dependency = _authorized_surface_facing_dependency(
        context.payload, context.result
    )
    if facing_dependency is None:
        return []
    object_id = str(context.result.get("primary_object") or "")
    target_id = next(
        (
            str(item)
            for item in (
                context.result.get("selected_related_objects")
                or context.result.get("related_objects")
                or []
            )
            if str(item)
        ),
        "",
    )
    subject = context.scene.objects.get(UniqueID(object_id))
    target = context.scene.objects.get(UniqueID(target_id))
    subject_center = _world_center_xy(subject) if subject is not None else None
    target_center = _world_center_xy(target) if target is not None else None
    if subject_center is None or target_center is None:
        return []
    dx = target_center[0] - subject_center[0]
    dy = target_center[1] - subject_center[1]
    if math.hypot(dx, dy) <= 1e-6:
        return []
    target_yaw = math.degrees(math.atan2(-dx, dy))
    if str(facing_dependency.get("subject_face") or "front") == "back":
        target_yaw = (target_yaw + 180.0) % 360.0
    return [
        _RepairTarget(
            object_id,
            context.relation,
            context.check_id,
            subject_center,
            target_yaw,
        )
    ]


def _generic_near_repair_targets(
    context: _RepairHandlerContext,
) -> list[_RepairTarget]:
    if context.check_id in context.claimed_near_checks:
        return []
    return _generic_near_targets(context.scene, context.result, context.check_id)


def _corner_repair_targets(context: _RepairHandlerContext) -> list[_RepairTarget]:
    return _corner_of_room_targets(context.scene, context.result, context.check_id)


def _corner_distribution_repair_targets(
    context: _RepairHandlerContext,
) -> list[_RepairTarget]:
    target = (context.diagnostics.get("assignment") or {}).get("target_corner_xy_m")
    assigned_corner = (
        (float(target[0]), float(target[1]))
        if isinstance(target, list) and len(target) >= 2
        else None
    )
    candidates = _corner_of_room_targets(
        context.scene,
        context.result,
        context.check_id,
        assigned_corner=assigned_corner,
    )
    return [
        _RepairTarget(
            candidate.object_id,
            "corner_distribution",
            candidate.check_id,
            candidate.target_center_xy,
            candidate.target_yaw_deg,
        )
        for candidate in candidates
    ]


def _clear_access_repair_targets(
    context: _RepairHandlerContext,
) -> list[_RepairTarget]:
    target = _workstation_aisle_target(
        context.scene, context.diagnostics, context.check_id
    )
    if target is not None:
        return [target]
    return _local_access_repair_targets(context)


def _local_access_repair_targets(
    context: _RepairHandlerContext,
) -> list[_RepairTarget]:
    """Move reported local-access blockers sideways out of a usable front zone."""
    subject_id = str(context.result.get("primary_object") or "")
    subject = context.scene.objects.get(UniqueID(subject_id))
    subject_center = _world_center_xy(subject) if subject is not None else None
    if subject is None or subject_center is None:
        return []
    rotation = subject.transform.rotation().matrix()
    front = (float(rotation[0, 1]), float(rotation[1, 1]))
    magnitude = math.hypot(*front)
    if magnitude <= 1e-6:
        return []
    side = (-front[1] / magnitude, front[0] / magnitude)
    half_width = _float_or_none(context.diagnostics.get("half_width_m"))
    if half_width is None or half_width <= 0.0:
        return []

    targets: list[_RepairTarget] = []
    for blocker_id in context.diagnostics.get("blocking_ids") or []:
        blocker = context.scene.objects.get(UniqueID(str(blocker_id)))
        blocker_center = _world_center_xy(blocker) if blocker is not None else None
        if blocker is None or blocker_center is None:
            continue
        lateral = (blocker_center[0] - subject_center[0]) * side[0] + (
            blocker_center[1] - subject_center[1]
        ) * side[1]
        direction = 1.0 if lateral >= 0.0 else -1.0
        required_lateral = half_width + _LOCAL_ACCESS_REPAIR_MARGIN_M
        shift = max(0.0, required_lateral - abs(lateral))
        if shift <= 1e-6:
            continue
        target_center = (
            blocker_center[0] + direction * side[0] * shift,
            blocker_center[1] + direction * side[1] * shift,
        )
        candidate = _RepairTarget(
            str(blocker_id),
            "clear_access",
            context.check_id,
            target_center,
            None,
        )
        if _member_poses_fit_floor(
            context.scene,
            (_RepairPose(candidate.object_id, target_center, None),),
            context.check_id,
        ):
            targets.append(candidate)
    return targets


def _table_seat_repair_targets(context: _RepairHandlerContext) -> list[_RepairTarget]:
    table_id = str(context.result.get("primary_object") or "")
    topology_slots = context.diagnostics.get("topology_repair_slots") or []
    if topology_slots:
        topology_targets = [
            target
            for slot in topology_slots
            if (
                target := _target_from_facing_diagnostics(
                    context.scene,
                    object_id=str(slot.get("seat_id") or ""),
                    relation_type=context.relation,
                    check_id=context.check_id,
                    diagnostics=slot,
                )
            )
            is not None
        ]
        if len(topology_targets) == len(topology_slots):
            return _coordinated_dining_targets(
                topology_targets,
                topology_slots,
                scene=context.scene,
                table_id=table_id,
            )
    dining_targets: list[_RepairTarget] = []
    dining_diagnostics: list[dict[str, Any]] = []
    for slot in context.diagnostics.get("seat_slots") or []:
        target = _target_from_facing_diagnostics(
            context.scene,
            object_id=str(slot.get("seat_id") or ""),
            relation_type=context.relation,
            check_id=context.check_id,
            diagnostics=slot,
        )
        if target is not None:
            dining_targets.append(target)
            dining_diagnostics.append(slot)
    if len(dining_targets) > 1 and context.diagnostics.get("coordinated_one_per_edge"):
        return _coordinated_dining_targets(
            dining_targets,
            dining_diagnostics,
            scene=context.scene,
            table_id=table_id,
        )

    targets: list[_RepairTarget] = []
    for target, slot in zip(dining_targets, dining_diagnostics):
        if slot.get("aligned") and slot.get("facing_aligned") is not False:
            continue
        targets.extend(_dining_clearance_targets(target, slot))
        targets.append(target)
    return targets


def _workstation_focal_repair_targets(
    context: _RepairHandlerContext,
) -> list[_RepairTarget]:
    target = _target_from_facing_diagnostics(
        context.scene,
        object_id=str(context.diagnostics.get("seat_id") or ""),
        relation_type=context.relation,
        check_id=context.check_id,
        diagnostics=context.diagnostics,
    )
    return [target] if target is not None else []


def _faces_room_repair_targets(context: _RepairHandlerContext) -> list[_RepairTarget]:
    target = _target_from_facing_diagnostics(
        context.scene,
        object_id=str(context.result.get("primary_object") or ""),
        relation_type=context.relation,
        check_id=context.check_id,
        diagnostics=context.diagnostics,
    )
    return [target] if target is not None else []


def _seating_surface_repair_targets(
    context: _RepairHandlerContext,
) -> list[_RepairTarget]:
    assignment = context.diagnostics.get("seat_surface_assignment") or {}
    slot = assignment.get("target_slot") or {}
    center = _xy(slot.get("center_xy"))
    yaw = _float_or_none(slot.get("yaw_deg"))
    object_id = str(context.result.get("primary_object") or "")
    if center is None or yaw is None or not object_id:
        return []
    return [_RepairTarget(object_id, context.relation, context.check_id, center, yaw)]


def _seating_media_repair_targets(
    context: _RepairHandlerContext,
) -> list[_RepairTarget]:
    subject_id = str(context.result.get("primary_object") or "")
    media_id = next(
        (
            str(item)
            for item in (
                context.result.get("selected_related_objects")
                or context.diagnostics.get("selected_target_ids")
                or context.result.get("related_objects")
                or []
            )
            if str(item)
        ),
        "",
    )
    subject = context.scene.objects.get(UniqueID(subject_id))
    media = context.scene.objects.get(UniqueID(media_id))
    subject_center = _world_center_xy(subject) if subject is not None else None
    media_center = _world_center_xy(media) if media is not None else None
    lateral_axis = _xy(context.diagnostics.get("lateral_axis_xy"))
    signed_offset = _float_or_none(context.diagnostics.get("signed_lateral_offset_m"))
    if (
        subject is None
        or media is None
        or subject.object_type != ObjectType.FURNITURE
        or media.object_type != ObjectType.FURNITURE
        or subject_center is None
        or media_center is None
        or lateral_axis is None
        or signed_offset is None
        or abs(signed_offset) <= 1e-6
    ):
        return []

    norm = math.hypot(*lateral_axis)
    if norm <= 1e-6:
        return []
    side = (lateral_axis[0] / norm, lateral_axis[1] / norm)
    correction = (side[0] * signed_offset, side[1] * signed_offset)
    # Retain the viewing distance while permitting a modest shared motion along
    # the viewing axis.  A lateral-only correction can remain blocked by an
    # unrelated object that sits in the middle of the existing media group.
    # The zero-offset variants remain first so ordinary layouts take the least
    # disruptive repair.
    along_axis = (-side[1], side[0])
    along_offset_m = min(0.6, max(0.25, 0.12 * math.dist(subject_center, media_center)))
    lateral_candidates = (
        (
            (correction[0] * 0.5, correction[1] * 0.5),
            (-correction[0] * 0.5, -correction[1] * 0.5),
        ),
        (correction, (0.0, 0.0)),
        ((0.0, 0.0), (-correction[0], -correction[1])),
    )

    targets: list[_RepairTarget] = []
    for shared_along_delta in (
        (0.0, 0.0),
        (along_axis[0] * along_offset_m, along_axis[1] * along_offset_m),
        (-along_axis[0] * along_offset_m, -along_axis[1] * along_offset_m),
    ):
        for seating_delta, media_delta in lateral_candidates:
            seating_target = (
                subject_center[0] + seating_delta[0] + shared_along_delta[0],
                subject_center[1] + seating_delta[1] + shared_along_delta[1],
            )
            media_target = (
                media_center[0] + media_delta[0] + shared_along_delta[0],
                media_center[1] + media_delta[1] + shared_along_delta[1],
            )
            poses = _living_media_member_poses(
                context,
                subject_id=subject_id,
                media_id=media_id,
                seating_target=seating_target,
                media_target=media_target,
            )
            if len(poses) < 2:
                continue
            targets.append(
                _RepairTarget(
                    subject_id,
                    context.relation,
                    context.check_id,
                    seating_target,
                    None,
                    member_poses=poses,
                )
            )
    return targets


def _living_media_member_poses(
    context: _RepairHandlerContext,
    *,
    subject_id: str,
    media_id: str,
    seating_target: tuple[float, float],
    media_target: tuple[float, float],
) -> tuple[_RepairPose, ...]:
    subject = context.scene.objects.get(UniqueID(subject_id))
    media = context.scene.objects.get(UniqueID(media_id))
    subject_center = _world_center_xy(subject) if subject is not None else None
    media_center = _world_center_xy(media) if media is not None else None
    if subject_center is None or media_center is None:
        return ()

    # A failed hard support contract is exactly the dependency that must move
    # with the media anchor.  Restricting this graph to passing checks leaves a
    # television embedded in its old console and makes the atomic candidate
    # fail the furniture stage's physical validator before it can be scored.
    media_group = _hard_media_support_component(context, seed_id=media_id)
    media_delta = (
        media_target[0] - media_center[0],
        media_target[1] - media_center[1],
    )
    poses: dict[str, _RepairPose] = {
        subject_id: _RepairPose(
            subject_id,
            seating_target,
            compute_optimal_facing_yaw(
                np.array([*seating_target, 0.0], dtype=float),
                np.array([*media_target, 0.0], dtype=float),
            ),
        )
    }
    planned_centers = {subject_id: seating_target}
    for object_id in sorted(media_group):
        obj = context.scene.objects.get(UniqueID(object_id))
        center = _world_center_xy(obj) if obj is not None else None
        if obj is None or obj.object_type != ObjectType.FURNITURE or center is None:
            continue
        target_center = (
            center[0] + media_delta[0],
            center[1] + media_delta[1],
        )
        poses[object_id] = _RepairPose(object_id, target_center, None)
        planned_centers[object_id] = target_center

    # Restore each explicitly required support pose after translating the
    # component.  XY is centered on the moved support; Z is placed on its top
    # surface so a previously intersecting display and console are repaired in
    # the same reversible candidate.
    for subject_id_on_support, support_id in _hard_support_edges(context):
        if (
            subject_id_on_support not in media_group
            or support_id not in media_group
            or support_id not in planned_centers
        ):
            continue
        subject_on_support = context.scene.objects.get(UniqueID(subject_id_on_support))
        support = context.scene.objects.get(UniqueID(support_id))
        subject_bounds = (
            subject_on_support.compute_world_bounds()
            if subject_on_support is not None
            else None
        )
        support_bounds = support.compute_world_bounds() if support is not None else None
        if (
            subject_on_support is None
            or subject_on_support.object_type != ObjectType.FURNITURE
            or support is None
            or support.object_type != ObjectType.FURNITURE
            or subject_bounds is None
            or support_bounds is None
        ):
            continue
        subject_height = float(subject_bounds[1][2] - subject_bounds[0][2])
        target_center_z = float(support_bounds[1][2]) + 0.01 + subject_height / 2.0
        poses[subject_id_on_support] = _RepairPose(
            subject_id_on_support,
            planned_centers[support_id],
            _media_display_yaw_for_support(support),
            target_center_z=target_center_z,
        )

    for result in context.payload.get("results") or []:
        relation = str(result.get("relation_type") or "")
        if relation not in {
            "between_alignment",
            "centered_between_alignment",
        } or not _is_hard_furniture_contract_result(result):
            continue
        anchor_ids = [
            str(item)
            for item in (
                result.get("selected_related_objects")
                or result.get("related_objects")
                or (result.get("diagnostics") or {}).get("anchor_ids")
                or []
            )
            if str(item)
        ]
        if len(anchor_ids) != 2 or subject_id not in anchor_ids:
            continue
        other_id = next((item for item in anchor_ids if item != subject_id), "")
        if other_id not in media_group:
            continue
        dependent_id = str(result.get("primary_object") or "")
        dependent = context.scene.objects.get(UniqueID(dependent_id))
        other = context.scene.objects.get(UniqueID(other_id))
        if dependent is None or dependent.object_type != ObjectType.FURNITURE:
            continue
        # The midpoint is always a valid, deterministic member of an explicit
        # between relation.  It repairs a failing coffee-table relationship
        # instead of merely preserving an already-correct relative offset.
        target_center = (
            (seating_target[0] + planned_centers[other_id][0]) / 2.0,
            (seating_target[1] + planned_centers[other_id][1]) / 2.0,
        )
        poses[dependent_id] = _RepairPose(dependent_id, target_center, None)
    return tuple(poses.values())


def _hard_support_edges(context: _RepairHandlerContext) -> list[tuple[str, str]]:
    """Return hard furniture-to-furniture support edges regardless of status."""
    edges: list[tuple[str, str]] = []
    for result in context.payload.get("results") or []:
        if str(
            result.get("relation_type") or ""
        ) != "object_on_support" or not _is_hard_furniture_contract_result(result):
            continue
        subject_id = str(result.get("primary_object") or "")
        subject = context.scene.objects.get(UniqueID(subject_id))
        if subject is None or subject.object_type != ObjectType.FURNITURE:
            continue
        support_ids = [
            str(item)
            for item in (
                result.get("selected_related_objects")
                or (result.get("diagnostics") or {}).get("selected_target_ids")
                or result.get("related_objects")
                or []
            )
            if str(item)
        ]
        for support_id in support_ids:
            support = context.scene.objects.get(UniqueID(support_id))
            if support is not None and support.object_type == ObjectType.FURNITURE:
                edges.append((subject_id, support_id))
    return edges


def _hard_media_support_component(
    context: _RepairHandlerContext, *, seed_id: str
) -> set[str]:
    component = {seed_id}
    edges = _hard_support_edges(context)
    changed = True
    while changed:
        changed = False
        for first, second in edges:
            if first in component or second in component:
                before = len(component)
                component.update((first, second))
                changed = changed or len(component) > before
    return component


def _room_center_repair_targets(
    context: _RepairHandlerContext,
) -> list[_RepairTarget]:
    object_id = str(context.result.get("primary_object") or "")
    center = _xy(context.diagnostics.get("room_center_xy"))
    if not object_id or center is None:
        return []
    group_ids = _room_center_group_ids(context.scene, context.result, object_id, center)
    return _room_center_targets(
        context.scene,
        object_id=object_id,
        relation_type=context.relation,
        check_id=context.check_id,
        room_center=center,
        allowed_offset_m=_float_or_none(context.diagnostics.get("allowed_offset_m")),
        group_ids=group_ids,
    )


def _centered_edge_distribution_targets(
    scene: RoomScene, payload: dict[str, Any]
) -> list[_RepairTarget]:
    """Center a target and place its edge group in one reversible candidate."""
    targets: list[_RepairTarget] = []
    results = payload.get("results") or []
    for center_result in results:
        if (
            center_result.get("label") not in _ISSUE_LABELS
            or str(center_result.get("relation_type") or "") != "room_center_alignment"
        ):
            continue
        object_id = str(center_result.get("primary_object") or "")
        room_center = _xy(
            (center_result.get("diagnostics") or {}).get("room_center_xy")
        )
        anchor = scene.objects.get(UniqueID(object_id))
        anchor_center = _world_center_xy(anchor) if anchor is not None else None
        if not object_id or room_center is None or anchor_center is None:
            continue
        delta = (room_center[0] - anchor_center[0], room_center[1] - anchor_center[1])
        for edge_result in results:
            if (
                edge_result.get("label") not in _ISSUE_LABELS
                or str(edge_result.get("relation_type") or "") != "edge_distribution"
                or str(edge_result.get("primary_object") or "") != object_id
            ):
                continue
            edge_context = _RepairHandlerContext(
                scene=scene,
                payload=payload,
                result=edge_result,
                check_id=str(edge_result.get("check_id") or ""),
                relation="edge_distribution",
                diagnostics=edge_result.get("diagnostics") or {},
                coordinated_front_checks=set(),
                claimed_near_checks=set(),
            )
            for edge_target in _table_seat_repair_targets(edge_context):
                if not edge_target.member_poses:
                    continue
                poses = [
                    _RepairPose(object_id, room_center, None),
                    *[
                        _RepairPose(
                            pose.object_id,
                            (
                                pose.target_center_xy[0] + delta[0],
                                pose.target_center_xy[1] + delta[1],
                            ),
                            pose.target_yaw_deg,
                        )
                        for pose in edge_target.member_poses
                    ],
                ]
                targets.append(
                    _RepairTarget(
                        object_id,
                        "room_center_alignment",
                        str(center_result.get("check_id") or ""),
                        room_center,
                        None,
                        member_poses=tuple(poses),
                    )
                )
    return targets


def _between_repair_targets(context: _RepairHandlerContext) -> list[_RepairTarget]:
    object_id = str(context.result.get("primary_object") or "")
    center = _xy(context.diagnostics.get("target_center_xy_m"))
    if center is None:
        # Contract extensions can report the two selected anchors but omit the
        # optional diagnostic repair coordinate.  The midpoint is the exact
        # geometry target for centered-between and a valid in-segment target
        # for between, so retain deterministic repair coverage without relying
        # on a particular evaluator payload shape.
        anchor_ids = [
            str(anchor_id)
            for anchor_id in (
                context.result.get("selected_related_objects")
                or context.result.get("related_objects")
                or context.diagnostics.get("anchor_ids")
                or []
            )
            if str(anchor_id)
        ]
        if len(anchor_ids) == 2:
            first = context.scene.objects.get(UniqueID(anchor_ids[0]))
            second = context.scene.objects.get(UniqueID(anchor_ids[1]))
            first_center = _world_center_xy(first) if first is not None else None
            second_center = _world_center_xy(second) if second is not None else None
            if first_center is not None and second_center is not None:
                center = (
                    (first_center[0] + second_center[0]) / 2.0,
                    (first_center[1] + second_center[1]) / 2.0,
                )
    if not object_id or center is None:
        return []
    target = _RepairTarget(object_id, context.relation, context.check_id, center, None)
    return [_preserve_passing_flanking_group(context.scene, context.payload, target)]


def _flanking_repair_targets(context: _RepairHandlerContext) -> list[_RepairTarget]:
    poses: list[_RepairPose] = []
    for slot in context.diagnostics.get("target_slots") or []:
        object_id = str(slot.get("object_id") or "")
        center = _xy(slot.get("target_center_xy_m"))
        yaw = _float_or_none(slot.get("target_yaw_deg"))
        if object_id and center is not None and yaw is not None:
            poses.append(_RepairPose(object_id, center, yaw))
    if len(poses) < 2:
        return []
    anchor = poses[0]
    return [
        _RepairTarget(
            anchor.object_id,
            context.relation,
            context.check_id,
            anchor.target_center_xy,
            anchor.target_yaw_deg,
            member_poses=tuple(poses),
        )
    ]


def _centered_wall_repair_targets(
    context: _RepairHandlerContext,
) -> list[_RepairTarget]:
    object_id = str(context.result.get("primary_object") or "")
    center = _xy(context.diagnostics.get("target_center_xy_m"))
    yaw = _float_or_none(context.diagnostics.get("target_yaw_deg"))
    if not object_id or center is None or yaw is None:
        return []
    return [_RepairTarget(object_id, context.relation, context.check_id, center, yaw)]


def _front_axis_repair_targets(
    context: _RepairHandlerContext,
) -> list[_RepairTarget]:
    if context.check_id in context.coordinated_front_checks:
        return []
    object_id = str(context.diagnostics.get("repair_object_id") or "")
    center = _xy(context.diagnostics.get("repair_target_center_xy_m"))
    if not object_id or center is None:
        return []
    return [_RepairTarget(object_id, context.relation, context.check_id, center, None)]


def _wall_operation_repair_targets(
    context: _RepairHandlerContext,
) -> list[_RepairTarget]:
    object_id = str(context.result.get("primary_object") or "")
    center = _xy(context.diagnostics.get("target_center_xy_m"))
    yaw = _float_or_none(context.diagnostics.get("target_yaw_deg"))
    if not object_id or center is None or yaw is None:
        return []
    return [
        _RepairTarget(
            object_id,
            context.relation,
            context.check_id,
            center,
            yaw,
            target_wall_id=(
                str(context.diagnostics.get("presenter_wall_id") or "") or None
            ),
        )
    ]


def _support_repair_targets(context: _RepairHandlerContext) -> list[_RepairTarget]:
    one_per_support = context.relation == "one_per_support"
    if not one_per_support and not _is_required_media_on_support_result(
        context.payload, context.scene, context.result
    ):
        return []
    object_id = str(
        context.diagnostics.get("repair_subject_id")
        or context.result.get("primary_object")
        or ""
    )
    support_candidates = (
        context.diagnostics.get("missing_target_ids")
        if one_per_support
        else (
            context.result.get("selected_related_objects")
            or context.diagnostics.get("selected_target_ids")
            or context.result.get("related_objects")
            or []
        )
    )
    support_id = next((str(item) for item in support_candidates or [] if str(item)), "")
    subject = context.scene.objects.get(UniqueID(object_id))
    support = context.scene.objects.get(UniqueID(support_id))
    subject_bounds = subject.compute_world_bounds() if subject is not None else None
    support_bounds = support.compute_world_bounds() if support is not None else None
    support_center = _world_center_xy(support) if support is not None else None
    if (
        not object_id
        or support_center is None
        or subject_bounds is None
        or support_bounds is None
    ):
        return []
    subject_height = float(subject_bounds[1][2] - subject_bounds[0][2])
    target_center_z = float(support_bounds[1][2]) + 0.01 + subject_height / 2.0
    target_yaw_deg = (
        _media_display_yaw_for_support(support)
        if _is_media_on_support_result(context.scene, context.result)
        else None
    )
    direct_target = _RepairTarget(
        object_id,
        context.relation,
        context.check_id,
        support_center,
        target_yaw_deg,
        target_center_z=target_center_z,
    )
    return [direct_target]


def _preserved_between_center(
    *,
    subject_center: tuple[float, float],
    other_center: tuple[float, float],
    target_other_center: tuple[float, float],
    support_center: tuple[float, float],
    target_support_center: tuple[float, float],
    centered: bool,
) -> tuple[float, float] | None:
    """Map a between-dependent into the coordinate system of moved anchors."""
    old_axis = np.asarray(support_center, dtype=float) - np.asarray(
        other_center, dtype=float
    )
    new_axis = np.asarray(target_support_center, dtype=float) - np.asarray(
        target_other_center, dtype=float
    )
    old_length = float(np.linalg.norm(old_axis))
    new_length = float(np.linalg.norm(new_axis))
    if old_length < 1e-6 or new_length < 1e-6:
        return None
    if centered:
        target = np.asarray(target_other_center, dtype=float) + new_axis / 2.0
    else:
        relative = np.asarray(subject_center, dtype=float) - np.asarray(
            other_center, dtype=float
        )
        fraction = float(np.dot(relative, old_axis) / (old_length * old_length))
        lateral_m = float(
            (old_axis[0] * relative[1] - old_axis[1] * relative[0]) / old_length
        )
        new_perpendicular = np.array([-new_axis[1], new_axis[0]]) / new_length
        target = (
            np.asarray(target_other_center, dtype=float)
            + fraction * new_axis
            + lateral_m * new_perpendicular
        )
    return float(target[0]), float(target[1])


def _wall_backed_storage_repair_targets(
    context: _RepairHandlerContext,
) -> list[_RepairTarget]:
    object_id = str(
        context.diagnostics.get("object_id")
        or context.result.get("primary_object")
        or ""
    )
    targets: list[_RepairTarget] = []
    for pose in context.diagnostics.get("candidate_poses") or []:
        center = _xy(pose.get("target_center_xy_m"))
        yaw = _float_or_none(pose.get("target_yaw_deg"))
        if object_id and center is not None and yaw is not None:
            targets.append(
                _RepairTarget(
                    object_id, context.relation, context.check_id, center, yaw
                )
            )
    return targets


def _back_against_wall_repair_targets(
    context: _RepairHandlerContext,
) -> list[_RepairTarget]:
    object_id = str(context.result.get("primary_object") or "")
    wall_ids = [
        str(item)
        for item in (
            context.result.get("selected_related_objects")
            or context.result.get("related_objects")
            or context.diagnostics.get("selected_target_ids")
            or []
        )
        if str(item)
    ]
    wall_id = next(
        (
            candidate_id
            for candidate_id in wall_ids
            if _scene_wall(context.scene, candidate_id) is not None
        ),
        None,
    )
    max_gap_m = _wall_backed_max_gap_m(context.payload, context.check_id)
    return [
        _RepairTarget(
            object_id,
            context.relation,
            context.check_id,
            target[0],
            target[1],
        )
        for target in _wall_backed_targets(
            context.scene,
            object_id,
            wall_id,
            max_gap_m=max_gap_m,
        )
    ]


def _room_containment_repair_targets(
    context: _RepairHandlerContext,
) -> list[_RepairTarget]:
    """Move an escaped floor object minimally back into the usable room bounds."""
    object_id = str(context.result.get("primary_object") or "")
    obj = context.scene.objects.get(UniqueID(object_id))
    geometry = context.scene.room_geometry
    if obj is None or geometry is None:
        return []

    for result in context.payload.get("results") or []:
        if (
            str(result.get("primary_object") or "") != object_id
            or str(result.get("relation_type") or "") != "back_against_wall"
        ):
            continue
        wall_id = next(
            (
                str(value)
                for value in (
                    result.get("selected_related_objects")
                    or result.get("related_objects")
                    or []
                )
                if _scene_wall(context.scene, str(value)) is not None
            ),
            None,
        )
        if wall_id is None:
            continue
        wall_targets = _wall_backed_targets(context.scene, object_id, wall_id)
        if wall_targets:
            return [
                _RepairTarget(
                    object_id,
                    "room_containment",
                    context.check_id,
                    center,
                    yaw,
                )
                for center, yaw in wall_targets
            ]

    center = _world_center_xy(obj)
    if center is None:
        return []

    diagnostics = context.result.get("diagnostics") or {}
    footprint_values = diagnostics.get("footprint_world")
    room_values = diagnostics.get("room_floor_polygon")
    try:
        footprint = Polygon(footprint_values)
        floor_polygon = Polygon(room_values)
    except (TypeError, ValueError):
        footprint = Polygon()
        floor_polygon = Polygon()
    has_floor_polygons = not (
        footprint.is_empty
        or floor_polygon.is_empty
        or not footprint.is_valid
        or not floor_polygon.is_valid
    )
    usable_floor: Polygon | None = None
    if not has_floor_polygons:
        bounds = obj.compute_world_bounds()
        if bounds is None:
            return []
        lower = np.asarray(bounds[0], dtype=float)
        upper = np.asarray(bounds[1], dtype=float)
        room_min = np.array(
            [
                -float(geometry.length) / 2.0 + _ROOM_CONTAINMENT_MARGIN_M,
                -float(geometry.width) / 2.0 + _ROOM_CONTAINMENT_MARGIN_M,
            ]
        )
        room_max = np.array(
            [
                float(geometry.length) / 2.0 - _ROOM_CONTAINMENT_MARGIN_M,
                float(geometry.width) / 2.0 - _ROOM_CONTAINMENT_MARGIN_M,
            ]
        )
    else:
        usable_floor = floor_polygon.buffer(-_ROOM_CONTAINMENT_MARGIN_M, join_style=2)
        if usable_floor.is_empty:
            return []
        lower = np.asarray(footprint.bounds[:2], dtype=float)
        upper = np.asarray(footprint.bounds[2:], dtype=float)
        room_min = np.asarray(usable_floor.bounds[:2], dtype=float)
        room_max = np.asarray(usable_floor.bounds[2:], dtype=float)

    object_span = upper - lower
    if np.any(object_span > room_max - room_min):
        return []
    shift = np.zeros(2, dtype=float)
    for axis in range(2):
        if lower[axis] < room_min[axis]:
            shift[axis] = room_min[axis] - lower[axis]
        elif upper[axis] > room_max[axis]:
            shift[axis] = room_max[axis] - upper[axis]
    if np.linalg.norm(shift) <= 1e-9:
        return []
    if usable_floor is not None:
        candidate_footprint = translate_polygon(
            footprint, xoff=float(shift[0]), yoff=float(shift[1])
        )
        if not usable_floor.covers(candidate_footprint):
            return []
    return [
        _RepairTarget(
            object_id,
            "room_containment",
            context.check_id,
            (center[0] + float(shift[0]), center[1] + float(shift[1])),
            None,
        )
    ]


def _activity_zone_repair_targets(
    context: _RepairHandlerContext,
) -> list[_RepairTarget]:
    """Translate a relation-derived dining group beside the viewing axis."""
    group_ids = tuple(
        sorted(
            {
                str(object_id)
                for object_id in context.diagnostics.get("group_object_ids") or []
                if object_id
            }
        )
    )
    anchor_id = str(context.result.get("primary_object") or "")
    normal_raw = context.diagnostics.get("view_normal_unit") or []
    if anchor_id not in group_ids or len(group_ids) < 2 or len(normal_raw) < 2:
        return []
    normal = np.asarray(normal_raw[:2], dtype=float)
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-9:
        return []
    normal /= norm
    centers: dict[str, tuple[float, float]] = {}
    for object_id in group_ids:
        obj = context.scene.objects.get(UniqueID(object_id))
        center = _world_center_xy(obj) if obj is not None else None
        if obj is None or obj.object_type != ObjectType.FURNITURE or center is None:
            return []
        centers[object_id] = center

    room_diagonal = _room_diagonal_m(context.scene)
    step_m = 0.25
    distances = [
        round(step_m * step, 3)
        for step in range(1, int(math.floor(room_diagonal / step_m)) + 1)
    ]
    if room_diagonal > 0.0 and (
        not distances or abs(distances[-1] - room_diagonal) > 1e-6
    ):
        distances.append(room_diagonal)
    targets: list[_RepairTarget] = []
    for distance in distances:
        for sign in (-1.0, 1.0):
            delta = normal * distance * sign
            poses = tuple(
                _RepairPose(
                    object_id=object_id,
                    target_center_xy=(
                        centers[object_id][0] + float(delta[0]),
                        centers[object_id][1] + float(delta[1]),
                    ),
                    target_yaw_deg=None,
                )
                for object_id in group_ids
            )
            anchor_center = centers[anchor_id]
            targets.append(
                _RepairTarget(
                    object_id=anchor_id,
                    relation_type="activity_zone_separation",
                    check_id=context.check_id,
                    target_center_xy=(
                        anchor_center[0] + float(delta[0]),
                        anchor_center[1] + float(delta[1]),
                    ),
                    target_yaw_deg=None,
                    member_poses=poses,
                )
            )
    return targets


_REPAIR_TARGET_HANDLERS = {
    "activity_zone_separation": _activity_zone_repair_targets,
    "faces": _faces_room_repair_targets,
    _PAIRED_SURFACE_RELATION: _paired_surface_targets,
    "generic_near_relation": _generic_near_repair_targets,
    "corner_of_room": _corner_repair_targets,
    "corner_distribution": _corner_distribution_repair_targets,
    "clear_access": _clear_access_repair_targets,
    "edge_distribution": _table_seat_repair_targets,
    "workstation_focal_alignment": _workstation_focal_repair_targets,
    "seating_to_media": _seating_media_repair_targets,
    "seating_to_work_surface": _seating_surface_repair_targets,
    "room_center_alignment": _room_center_repair_targets,
    "between_alignment": _between_repair_targets,
    "centered_between_alignment": _between_repair_targets,
    "flanking": _flanking_repair_targets,
    "centered_on_wall": _centered_wall_repair_targets,
    "front_axis_alignment": _front_axis_repair_targets,
    "rear_axis_alignment": _front_axis_repair_targets,
    "operation_zone_at_wall": _wall_operation_repair_targets,
    "instructional_surface_alignment": _wall_operation_repair_targets,
    "object_on_support": _support_repair_targets,
    "one_per_support": _support_repair_targets,
    "wall_backed_storage_alignment": _wall_backed_storage_repair_targets,
    "back_against_wall": _back_against_wall_repair_targets,
    "room_containment": _room_containment_repair_targets,
}


def _repair_targets(scene: RoomScene, payload: dict[str, Any]) -> list[_RepairTarget]:
    targets, coordinated_front_checks, claimed_near_checks = (
        _front_alignment_near_group_targets(scene, payload)
    )
    centered_edge_targets = _centered_edge_distribution_targets(scene, payload)
    for result in payload.get("results") or []:
        if result.get("label") not in _ISSUE_LABELS:
            continue
        if str(result.get("scoring_tier") or "").lower() in {"ignored", "auxiliary"}:
            continue
        check_id = str(result.get("check_id") or "")
        relation = str(result.get("relation_type") or "")
        intent_constraint = (result.get("evidence") or {}).get(
            "intent_constraint"
        ) or {}
        compiled_relation = str(intent_constraint.get("relation") or "")
        if (
            compiled_relation
            and relation_spec(compiled_relation).repair_strategy is None
        ):
            continue
        if not _is_registry_repairable_relation(relation):
            continue
        handler = _REPAIR_TARGET_HANDLERS.get(relation)
        if handler is None:
            continue
        diagnostics = result.get("diagnostics") or {}
        context = _RepairHandlerContext(
            scene=scene,
            payload=payload,
            result=result,
            check_id=check_id,
            relation=relation,
            diagnostics=diagnostics,
            coordinated_front_checks=coordinated_front_checks,
            claimed_near_checks=claimed_near_checks,
        )
        targets.extend(handler(context))
    targets.sort(key=lambda target: _repair_target_priority(scene, payload, target))
    ordered_targets = _order_targets_by_position_dependency(
        scene,
        payload,
        _prioritize_coordinated_seating_targets(targets),
    )
    return [*centered_edge_targets, *ordered_targets]


def _is_coordinated_media_target(target: _RepairTarget) -> bool:
    return bool(
        target.relation_type == "seating_to_media"
        and len({pose.object_id for pose in target.member_poses}) >= 2
    )


def _is_coordinated_table_seating_target(
    scene: RoomScene, target: _RepairTarget
) -> bool:
    """Identify an atomic table-anchor plus seating topology candidate."""
    if target.relation_type != "edge_distribution" or len(target.member_poses) < 2:
        return False
    anchor = scene.objects.get(UniqueID(target.object_id))
    if anchor is None or _is_seating_object(anchor):
        return False
    member_ids = {pose.object_id for pose in target.member_poses}
    if str(anchor.object_id) not in member_ids:
        return False
    return any(
        _is_seating_object(obj)
        for pose in target.member_poses
        if (obj := scene.objects.get(UniqueID(pose.object_id))) is not None
    )


def _repair_target_priority(
    scene: RoomScene, payload: dict[str, Any], target: _RepairTarget
) -> tuple[int, int, int, str, str]:
    """Give atomic repairs precedence in every candidate ordering path."""
    return (
        # This target restores the viewing axis plus explicit support and
        # between dependents in one physically valid state. Try it before
        # independent wall adjustments that might move a shared anchor.
        0 if _is_coordinated_media_target(target) else 1,
        # An exact table-local chair topology can be physically impossible at
        # the current table anchor because an entrance sweep crosses one of
        # its seats. Its table-and-seat candidate must precede a cheaper
        # single-seat pose that would satisfy only part of that contract.
        0 if _is_coordinated_table_seating_target(scene, target) else 1,
        # A hard prompt-authorized support pair that currently intersects is
        # both a contract and physical failure. It must precede single-object
        # candidates whose strict physics gate would reject the current state.
        0 if _is_hard_support_collision_target(scene, payload, target) else 1,
        (
            0
            if (_result_by_id(payload, target.check_id) or {}).get("label") == "fail"
            else 1
        ),
        target.check_id,
        target.object_id,
    )


def _is_hard_support_collision_target(
    scene: RoomScene,
    payload: dict[str, Any],
    target: _RepairTarget,
) -> bool:
    """Return whether a target directly clears a hard support-pair collision."""
    if target.relation_type not in {"object_on_support", "one_per_support"}:
        return False
    result = _result_by_id(payload, target.check_id)
    constraint = ((result or {}).get("evidence") or {}).get("intent_constraint") or {}
    if str(constraint.get("strength") or "").lower() != "hard":
        return False
    subject = scene.objects.get(UniqueID(target.object_id))
    subject_bounds = subject.compute_world_bounds() if subject is not None else None
    if subject_bounds is None:
        return False
    diagnostics = (result or {}).get("diagnostics") or {}
    support_ids = (
        (result or {}).get("selected_related_objects")
        or diagnostics.get("selected_target_ids")
        or diagnostics.get("missing_target_ids")
        or (result or {}).get("related_objects")
        or []
    )
    for support_id in support_ids:
        support = scene.objects.get(UniqueID(str(support_id)))
        support_bounds = support.compute_world_bounds() if support is not None else None
        if support_bounds is None:
            continue
        if bool(
            np.all(subject_bounds[0] < support_bounds[1])
            and np.all(subject_bounds[1] > support_bounds[0])
        ):
            return True
    return False


def _order_targets_by_position_dependency(
    scene: RoomScene,
    payload: dict[str, Any],
    targets: list[_RepairTarget],
) -> list[_RepairTarget]:
    """Move positional anchors before repairs whose target pose uses them.

    Most repair targets are recomputed after every accepted move.  A target
    that supplies an anchor for another target must therefore run first:
    otherwise the dependent move can be accepted against an obsolete anchor
    position and then become invalid when that anchor moves.  The dependency
    is derived from evaluator diagnostics rather than room-specific relation
    names, so the ordering also applies to future multi-object relations.
    """
    if len(targets) < 2:
        return targets

    base_keys = [
        (*_repair_target_priority(scene, payload, target), index)
        for index, target in enumerate(targets)
    ]
    moved_ids = [_target_moved_ids(target) for target in targets]
    position_inputs = [
        _target_position_input_ids(payload, target) for target in targets
    ]
    successors: list[set[int]] = [set() for _ in targets]
    indegrees = [0 for _ in targets]
    for anchor_index, anchor_ids in enumerate(moved_ids):
        if not anchor_ids:
            continue
        for dependent_index, inputs in enumerate(position_inputs):
            if (
                anchor_index == dependent_index
                or not anchor_ids.intersection(inputs)
                or anchor_ids.intersection(moved_ids[dependent_index])
                or dependent_index in successors[anchor_index]
            ):
                continue
            successors[anchor_index].add(dependent_index)
            indegrees[dependent_index] += 1

    remaining = set(range(len(targets)))
    ordered_indices: list[int] = []
    while remaining:
        ready = [index for index in remaining if indegrees[index] == 0]
        # Cycles represent mutually constrained geometry. Retain the existing
        # deterministic severity/check ordering in that case and let the
        # whole-scene candidate gate decide whether either move is valid.
        selected = min(ready or remaining, key=lambda index: base_keys[index])
        remaining.remove(selected)
        ordered_indices.append(selected)
        for dependent_index in successors[selected]:
            indegrees[dependent_index] -= 1
    return [targets[index] for index in ordered_indices]


def _target_moved_ids(target: _RepairTarget) -> frozenset[str]:
    member_ids = {pose.object_id for pose in target.member_poses if pose.object_id}
    return (
        frozenset(member_ids or {target.object_id}) if target.object_id else frozenset()
    )


def _target_position_input_ids(
    payload: dict[str, Any], target: _RepairTarget
) -> frozenset[str]:
    result = _result_by_id(payload, target.check_id) or {}
    diagnostics = result.get("diagnostics") or {}
    inputs = {
        str(object_id)
        for object_id in (
            result.get("selected_related_objects")
            or result.get("related_objects")
            or []
        )
        if object_id
    }
    inputs.update(
        str(object_id) for object_id in diagnostics.get("anchor_ids") or [] if object_id
    )
    if str(result.get("relation_type") or "") == "flanking":
        primary_object = str(result.get("primary_object") or "")
        if primary_object:
            inputs.add(primary_object)
    return frozenset(inputs)


def _front_alignment_near_group_targets(
    scene: RoomScene,
    payload: dict[str, Any],
) -> tuple[list[_RepairTarget], set[str], set[str]]:
    """Keep explicit near dependents with a movable front-alignment anchor.

    A centered rug that must be in front of a wall-side sofa is repaired by
    sliding the sofa laterally.  Moving that sofa alone can invalidate plants,
    side tables, or other furniture explicitly required to stay near it.  Build
    the whole arrangement as one candidate so the normal whole-scene gate sees
    the intended end state instead of rejecting a necessary intermediate one.
    """
    targets: list[_RepairTarget] = []
    coordinated_front_checks: set[str] = set()
    claimed_near_checks: set[str] = set()
    results = payload.get("results") or []

    for result in results:
        if (
            str(result.get("relation_type") or "") != "front_axis_alignment"
            or str(result.get("label") or "") not in _ISSUE_LABELS
            or str(result.get("scoring_tier") or "").lower() in {"ignored", "auxiliary"}
        ):
            continue
        check_id = str(result.get("check_id") or "")
        subject_id = str(result.get("primary_object") or "")
        diagnostics = result.get("diagnostics") or {}
        anchor_id = str(diagnostics.get("repair_object_id") or "")
        anchor_center = _xy(diagnostics.get("repair_target_center_xy_m"))
        if (
            not check_id
            or not anchor_id
            or anchor_center is None
            or anchor_id == subject_id
            or not _is_room_center_anchor(payload, subject_id)
        ):
            continue
        anchor = scene.objects.get(UniqueID(anchor_id))
        if anchor is None or anchor.object_type != ObjectType.FURNITURE:
            continue

        dependent_rows: dict[str, dict[str, Any]] = {}
        near_check_ids: set[str] = set()
        for near_result in results:
            if not _is_explicit_near_result_for(near_result, anchor_id):
                continue
            dependent_id = str(near_result.get("primary_object") or "")
            dependent = scene.objects.get(UniqueID(dependent_id))
            if (
                not dependent_id
                or dependent_id == anchor_id
                or dependent is None
                or dependent.object_type != ObjectType.FURNITURE
            ):
                continue
            dependent_rows.setdefault(dependent_id, near_result)
            near_check_id = str(near_result.get("check_id") or "")
            if near_check_id:
                near_check_ids.add(near_check_id)

        dependents = [
            (object_id, dependent_rows[object_id])
            for object_id in sorted(dependent_rows)
        ]
        if not dependents:
            continue

        # Two sides are the least disruptive valid slots for paired dependent
        # furniture.  Larger groups can use the remaining sides as a fallback.
        slot_count = 2 if len(dependents) <= 2 else 4
        if len(dependents) > slot_count:
            continue
        candidate_centers = [
            _near_candidate_centers(
                scene.objects[UniqueID(object_id)],
                anchor,
                near_result,
                target_center=anchor_center,
            )
            for object_id, near_result in dependents
        ]
        if any(len(centers) < slot_count for centers in candidate_centers):
            continue

        candidate_targets: list[_RepairTarget] = []
        for slots in permutations(range(slot_count), len(dependents)):
            poses = [_RepairPose(anchor_id, anchor_center, None)]
            valid = True
            for (object_id, _), centers, slot in zip(
                dependents, candidate_centers, slots
            ):
                center = centers[slot]
                if center is None:
                    valid = False
                    break
                poses.append(_RepairPose(object_id, center, None))
            if not valid:
                continue
            candidate_targets.append(
                _RepairTarget(
                    anchor_id,
                    "front_axis_alignment",
                    check_id,
                    anchor_center,
                    None,
                    member_poses=tuple(poses),
                )
            )

        if not candidate_targets:
            continue
        candidate_targets.sort(
            key=lambda target: _repair_target_translation_key(scene, target)
        )
        targets.extend(candidate_targets)
        coordinated_front_checks.add(check_id)
        claimed_near_checks.update(near_check_ids)

    return targets, coordinated_front_checks, claimed_near_checks


def _is_room_center_anchor(payload: dict[str, Any], object_id: str) -> bool:
    return any(
        str(result.get("primary_object") or "") == object_id
        and str(
            (result.get("evidence") or {}).get("intent_constraint", {}).get("relation")
            or ""
        )
        == "centered_in_room"
        for result in payload.get("results") or []
    )


def _is_explicit_near_result_for(result: dict[str, Any], target_id: str) -> bool:
    if str(result.get("relation_type") or "") != "generic_near_relation" or str(
        result.get("scoring_tier") or ""
    ).lower() in {"ignored", "auxiliary"}:
        return False
    constraint = (result.get("evidence") or {}).get("intent_constraint") or {}
    if str(constraint.get("relation") or "") not in {"near", "next_to"}:
        return False
    return target_id in _result_related_object_ids(result)


def _result_related_object_ids(result: dict[str, Any]) -> set[str]:
    diagnostics = result.get("diagnostics") or {}
    return {
        str(object_id)
        for object_id in (
            result.get("selected_related_objects")
            or diagnostics.get("selected_target_ids")
            or result.get("related_objects")
            or []
        )
        if str(object_id)
    }


def _generic_near_targets(
    scene: RoomScene,
    result: dict[str, Any],
    check_id: str,
) -> list[_RepairTarget]:
    """Return floor-bounded adjacency candidates for a prompt-bound pair.

    Generic ``near`` and ``next_to`` checks deliberately have no category
    specific geometry.  Keep their repair equally general: preserve the
    subject's orientation and offer positions just outside each side of the
    selected target's footprint.  The usual whole-scene scoring gate decides
    whether a candidate is legal with the rest of the layout.
    """
    # A cardinality aggregate can be failed while this subject-target endpoint
    # already passes.  Do not move it away from a valid named relation.
    if any(
        str(item.get("label") or "") == "pass"
        for item in (result.get("diagnostics") or {}).get("target_evaluations") or []
        if isinstance(item, dict)
    ):
        return []
    object_id = str(result.get("primary_object") or "")
    target_ids = _result_related_object_ids(result)
    subject = scene.objects.get(UniqueID(object_id))
    target: SceneObject | None = None
    target_bounds: tuple[np.ndarray, np.ndarray] | None = None
    if subject is None or subject.object_type != ObjectType.FURNITURE:
        target_bounds = _structural_opening_bounds(scene, object_id)
        if target_bounds is None:
            return []
        for target_id in sorted(target_ids):
            candidate = scene.objects.get(UniqueID(str(target_id)))
            if candidate is not None and candidate.object_type == ObjectType.FURNITURE:
                subject = candidate
                object_id = str(candidate.object_id)
                break
        else:
            return []
    subject_bounds = subject.compute_world_bounds()
    if subject_bounds is None:
        return []

    if target_bounds is None:
        for target_id in sorted(target_ids):
            candidate = scene.objects.get(UniqueID(str(target_id)))
            if candidate is not None and candidate.object_id != subject.object_id:
                target = candidate
                break
            target_bounds = _structural_opening_bounds(scene, str(target_id))
            if target_bounds is not None:
                break
    if target is None and target_bounds is None:
        return []
    target_center = (
        _world_center_xy(target)
        if target is not None
        else _bounds_center_xy(target_bounds)
    )
    subject_center = _world_center_xy(subject)
    if target_center is None or subject_center is None:
        return []
    candidates = _near_candidate_centers(
        subject,
        target if target is not None else target_bounds,
        result,
    )
    candidates.sort(
        key=lambda center: math.hypot(
            center[0] - subject_center[0], center[1] - subject_center[1]
        )
    )
    return [
        _RepairTarget(object_id, "generic_near_relation", check_id, center, None)
        for center in candidates
    ]


def _near_candidate_centers(
    subject: SceneObject,
    target: SceneObject | tuple[np.ndarray, np.ndarray],
    result: dict[str, Any],
    *,
    target_center: tuple[float, float] | None = None,
) -> list[tuple[float, float]]:
    """Return adjacent slots around the target's current or requested bounds."""
    subject_bounds = subject.compute_world_bounds()
    target_bounds = (
        target.compute_world_bounds() if isinstance(target, SceneObject) else target
    )
    current_target_center = (
        _world_center_xy(target)
        if isinstance(target, SceneObject)
        else _bounds_center_xy(target_bounds)
    )
    if subject_bounds is None or target_bounds is None or current_target_center is None:
        return []
    target_center = target_center or current_target_center
    constraint = (result.get("evidence") or {}).get("intent_constraint") or {}
    relation = str(constraint.get("relation") or "near")
    try:
        max_gap_m = float(relation_spec(relation).thresholds.get("max_gap_m", 0.6))
    except (TypeError, ValueError):
        max_gap_m = 0.6
    desired_gap_m = max(0.05, min(0.35, max_gap_m * 0.4))
    subject_lower, subject_upper = subject_bounds
    target_lower, target_upper = target_bounds
    shift_x = target_center[0] - current_target_center[0]
    shift_y = target_center[1] - current_target_center[1]
    subject_half_x = (float(subject_upper[0]) - float(subject_lower[0])) / 2.0
    subject_half_y = (float(subject_upper[1]) - float(subject_lower[1])) / 2.0
    return [
        (
            float(target_upper[0]) + shift_x + desired_gap_m + subject_half_x,
            target_center[1],
        ),
        (
            float(target_lower[0]) + shift_x - desired_gap_m - subject_half_x,
            target_center[1],
        ),
        (
            target_center[0],
            float(target_upper[1]) + shift_y + desired_gap_m + subject_half_y,
        ),
        (
            target_center[0],
            float(target_lower[1]) + shift_y - desired_gap_m - subject_half_y,
        ),
    ]


def _bounds_center_xy(
    bounds: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[float, float] | None:
    if bounds is None:
        return None
    center = (
        np.asarray(bounds[0], dtype=float) + np.asarray(bounds[1], dtype=float)
    ) / 2.0
    return float(center[0]), float(center[1])


def _structural_opening_bounds(
    scene: RoomScene, opening_id: str
) -> tuple[np.ndarray, np.ndarray] | None:
    geometry = scene.room_geometry
    if geometry is None:
        return None
    for opening in getattr(geometry, "openings", ()) or ():
        if str(getattr(opening, "opening_id", "") or "") != opening_id:
            continue
        return opening_physical_bounds(
            opening,
            wall_thickness_m=float(getattr(geometry, "wall_thickness", 0.05) or 0.05),
        )
    return None


def _corner_of_room_targets(
    scene: RoomScene,
    result: dict[str, Any],
    check_id: str,
    *,
    assigned_corner: tuple[float, float] | None = None,
) -> list[_RepairTarget]:
    """Return bounded, floor-safe positions near one or more room corners."""
    object_id = str(result.get("primary_object") or "")
    obj = scene.objects.get(UniqueID(object_id))
    geometry = scene.room_geometry
    if (
        obj is None
        or obj.object_type != ObjectType.FURNITURE
        or geometry is None
        or geometry.length <= 0
        or geometry.width <= 0
    ):
        return []
    bounds = obj.compute_world_bounds()
    center = _world_center_xy(obj)
    if bounds is None or center is None:
        return []
    lower, upper = bounds
    half_x = (float(upper[0]) - float(lower[0])) / 2.0
    half_y = (float(upper[1]) - float(lower[1])) / 2.0
    sign_xs = (-1.0, 1.0)
    sign_ys = (-1.0, 1.0)
    if assigned_corner is not None:
        sign_xs = (math.copysign(1.0, assigned_corner[0]),)
        sign_ys = (math.copysign(1.0, assigned_corner[1]),)

    candidates: list[tuple[float, float]] = []
    for margin_m in _CORNER_REPAIR_MARGINS_M:
        # Keep the candidate inside _within_floor_bounds while remaining close
        # to both walls. The evaluator's corner criterion is center-based.
        x_extent = float(geometry.length) / 2.0 - margin_m - half_x
        y_extent = float(geometry.width) / 2.0 - margin_m - half_y
        if x_extent <= 0.0 or y_extent <= 0.0:
            continue
        candidates.extend(
            (sign_x * x_extent, sign_y * y_extent)
            for sign_x in sign_xs
            for sign_y in sign_ys
        )
    if assigned_corner is not None:
        candidates.sort(key=lambda candidate: math.dist(candidate, assigned_corner))
    else:
        candidates.sort(key=lambda candidate: math.dist(candidate, center))
    return [
        _RepairTarget(object_id, "corner_of_room", check_id, candidate, None)
        for candidate in candidates
    ]


def _preserve_passing_flanking_group(
    scene: RoomScene,
    payload: dict[str, Any],
    target: _RepairTarget,
    *,
    require_hard_furniture_contract: bool = False,
) -> _RepairTarget:
    """Move a flanked anchor and its valid side group as one candidate.

    A collision repair can move a table away from its semantic midpoint while
    leaving the two side seats in a valid flanking arrangement. Moving only the
    table back can create shallow seat collisions, after which physics moves the
    table away again. Preserve actual member poses only when the group also has
    a passing hard proximity relation, such as a seat being near a sofa.
    Otherwise the evaluator's canonical side slots remain the most stable repair
    target.
    """
    anchor = scene.objects.get(UniqueID(target.object_id))
    current_center = _world_center_xy(anchor) if anchor is not None else None
    if current_center is None:
        return target
    flanking = next(
        (
            result
            for result in payload.get("results") or []
            if str(result.get("relation_type") or "") == "flanking"
            and result.get("label") == "pass"
            and str(result.get("primary_object") or "") == target.object_id
            and str(result.get("scoring_tier") or "").lower()
            not in {"ignored", "auxiliary"}
            and (
                not require_hard_furniture_contract
                or _is_hard_furniture_contract_result(result)
            )
        ),
        None,
    )
    if flanking is None:
        return target

    delta = (
        target.target_center_xy[0] - current_center[0],
        target.target_center_xy[1] - current_center[1],
    )
    poses = [
        _RepairPose(
            target.object_id,
            target.target_center_xy,
            target.target_yaw_deg,
        )
    ]
    member_ids = tuple(
        flanking.get("selected_related_objects")
        or flanking.get("related_objects")
        or []
    )
    if _flanking_members_have_external_hard_relation(
        payload,
        anchor_id=str(target.object_id),
        member_ids=member_ids,
    ):
        for object_id in member_ids:
            object_id = str(object_id or "")
            member = scene.objects.get(UniqueID(object_id))
            center = _world_center_xy(member) if member is not None else None
            if not object_id or member is None or center is None:
                continue
            yaw = math.degrees(RollPitchYaw(member.transform.rotation()).yaw_angle())
            poses.append(
                _RepairPose(
                    object_id,
                    (center[0] + delta[0], center[1] + delta[1]),
                    yaw,
                )
            )
    else:
        for slot in (flanking.get("diagnostics") or {}).get("target_slots") or []:
            object_id = str(slot.get("object_id") or "")
            center = _xy(slot.get("target_center_xy_m"))
            yaw = _float_or_none(slot.get("target_yaw_deg"))
            if not object_id or center is None or yaw is None:
                continue
            poses.append(
                _RepairPose(
                    object_id,
                    (center[0] + delta[0], center[1] + delta[1]),
                    yaw,
                )
            )
    unique_poses = tuple({pose.object_id: pose for pose in poses}.values())
    if len(unique_poses) < 3:
        return target
    return replace(target, member_poses=unique_poses)


def _flanking_members_have_external_hard_relation(
    payload: dict[str, Any],
    *,
    anchor_id: str,
    member_ids: Collection[str],
) -> bool:
    """Whether a hard proximity relation requires preserving member offsets."""
    group_ids = {str(anchor_id), *(str(object_id) for object_id in member_ids)}
    member_ids_set = group_ids - {str(anchor_id)}
    for result in payload.get("results") or []:
        if (
            result.get("label") != "pass"
            or str(result.get("relation_type") or "") != "generic_near_relation"
            or not _is_hard_furniture_contract_result(result)
        ):
            continue
        involved_ids = {
            str(result.get("primary_object") or ""),
            *(
                str(object_id)
                for object_id in (
                    result.get("selected_related_objects")
                    or result.get("related_objects")
                    or []
                )
            ),
        }
        if involved_ids & member_ids_set and involved_ids - group_ids:
            return True
    return False


def _is_hard_furniture_contract_result(result: dict[str, Any]) -> bool:
    constraint = (result.get("evidence") or {}).get("intent_constraint") or {}
    return bool(
        str(constraint.get("stage") or "").lower() == "furniture"
        and str(constraint.get("strength") or "").lower() == "hard"
    )


def _authorized_surface_facing_dependency(
    payload: dict[str, Any], result: dict[str, Any]
) -> dict[str, Any] | None:
    """Require exact prompt-contract provenance before rotating a work surface."""
    if str(result.get("relation_type") or "") != _PAIRED_SURFACE_RELATION:
        return None
    check_id = str(result.get("check_id") or "")
    checks = (payload.get("case_pack") or {}).get("checks") or []
    check = next(
        (
            item
            for item in checks
            if isinstance(item, dict) and str(item.get("check_id") or "") == check_id
        ),
        None,
    )
    if not isinstance(check, dict) or check.get("check_source") != "intent_contract":
        return None
    evidence = check.get("evidence") or {}
    constraint = evidence.get("intent_constraint") or {}
    paired_surface = bool(
        evidence.get("paired_surface_facing")
        and str(constraint.get("relation") or "") == "paired_with"
        and str(constraint.get("strength") or "").lower() == "hard"
        and str(constraint.get("source") or "").lower()
        in {"explicit_prompt", "room_ontology"}
    )
    prompt_facing = bool(
        str(constraint.get("relation") or "") == "faces"
        and str(constraint.get("strength") or "hard").lower() == "hard"
        and str(constraint.get("source") or "").lower()
        in {"explicit_prompt", "model_inferred", "room_ontology"}
    )
    if not (paired_surface or prompt_facing):
        return None
    dependency = evidence.get("dependency") or {}
    return dict(dependency) if isinstance(dependency, dict) else {}


def _is_paired_surface_facing_result(
    payload: dict[str, Any], result: dict[str, Any]
) -> bool:
    """Return whether a hard prompt-authorized surface facing check is gated."""
    return _authorized_surface_facing_dependency(payload, result) is not None


_MEDIA_SUBJECT_NAMES = {
    "display",
    "flat_screen_tv",
    "monitor",
    "screen",
    "television",
    "tv",
    "wall_mounted_television",
    "wall_mounted_tv",
}
_MEDIA_SUPPORT_NAMES = {
    "entertainment_center",
    "media_center",
    "media_console",
    "tv_console",
    "tv_stand",
}


def _is_media_on_support_result(scene: RoomScene, result: dict[str, Any]) -> bool:
    """Restrict support repair and its hard gate to media/support pairs.

    ``object_on_support`` is also emitted for ordinary floor objects and small
    manipulands. Those checks are useful critic evidence, but making every one
    furniture-repairable turns incidental physics failures into stage aborts.
    """
    if str(result.get("relation_type") or "") != "object_on_support":
        return False
    subject_id = str(result.get("primary_object") or "")
    support_id = next(
        (
            str(item)
            for item in (
                result.get("selected_related_objects")
                or (result.get("diagnostics") or {}).get("selected_target_ids")
                or result.get("related_objects")
                or []
            )
            if str(item)
        ),
        "",
    )
    subject = scene.objects.get(UniqueID(subject_id))
    support = scene.objects.get(UniqueID(support_id))
    return bool(
        subject is not None
        and support is not None
        and _object_semantic_names(subject) & _MEDIA_SUBJECT_NAMES
        and not (_object_semantic_names(subject) & _MEDIA_SUPPORT_NAMES)
        and _object_semantic_names(support) & _MEDIA_SUPPORT_NAMES
    )


def _is_required_media_on_support_result(
    payload: dict[str, Any], scene: RoomScene, result: dict[str, Any]
) -> bool:
    """Allow media support repair only for a hard prompt contract.

    A television and a media console commonly appear together, but the prompt
    may intentionally describe a floor-standing TV, a wall-mounted display, or
    two independent objects.  In contract mode, generic dependency proposals
    must therefore remain critic evidence unless an explicit prompt relation
    authorized this physical support requirement.
    """
    if not _is_media_on_support_result(scene, result):
        return False
    case_pack = payload.get("case_pack") or {}
    check_id = str(result.get("check_id") or "")
    for check in case_pack.get("checks") or []:
        if not isinstance(check, dict) or str(check.get("check_id") or "") != check_id:
            continue
        constraint = (check.get("evidence") or {}).get("intent_constraint") or {}
        return bool(
            check.get("check_source") == "intent_contract"
            and str(constraint.get("relation") or "") == "on_top_of"
            and str(constraint.get("strength") or "") == "hard"
        )
    return False


def _media_display_yaw_for_support(support: SceneObject) -> float:
    """Orient a freestanding display with its media console's viewing face."""
    return math.degrees(RollPitchYaw(support.transform.rotation()).yaw_angle())


def _object_semantic_names(obj: SceneObject) -> set[str]:
    """Return normalized, instance-suffix-free semantic identity labels."""
    values = (
        obj.metadata.get("semantic_name"),
        obj.metadata.get("category_norm"),
        obj.metadata.get("category"),
        obj.name,
        obj.object_id,
    )
    names: set[str] = set()
    for value in values:
        normalized = re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")
        if normalized:
            names.add(re.sub(r"_\d+$", "", normalized))
    return names


def _prioritize_coordinated_seating_targets(
    targets: list[_RepairTarget],
) -> list[_RepairTarget]:
    """Try mutually dependent seat repairs as one reversible candidate first.

    A partially generated classroom can leave several chairs at room edges.
    Moving one chair at a time may keep the aggregate seating rule failed and
    therefore fail the whole-scene improvement gate, even though the complete
    set of assigned seat slots is valid.  Group only the already-failing,
    deterministic seat-to-surface targets; the normal individual candidates
    remain as a fallback and the existing whole-scene gate still decides
    whether any candidate is accepted.
    """
    seating_targets = [
        target
        for target in targets
        if target.relation_type == "seating_to_work_surface"
        and not target.member_poses
        and target.object_id
    ]
    if len(seating_targets) < 2:
        return targets
    unique_poses = tuple(
        {
            target.object_id: _RepairPose(
                target.object_id,
                target.target_center_xy,
                target.target_yaw_deg,
            )
            for target in seating_targets
        }.values()
    )
    if len(unique_poses) < 2:
        return targets
    anchor = seating_targets[0]
    coordinated = _RepairTarget(
        anchor.object_id,
        anchor.relation_type,
        anchor.check_id,
        anchor.target_center_xy,
        anchor.target_yaw_deg,
        member_poses=unique_poses,
    )
    return [coordinated, *targets]


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


def _dining_clearance_targets(
    target: _RepairTarget,
    diagnostics: dict[str, Any],
) -> list[_RepairTarget]:
    """Add table-relative poses that use the slot's allowed outward tolerance."""
    facing_target = _xy(diagnostics.get("facing_target_xy_m"))
    allowed_deviation = _float_or_none(diagnostics.get("allowed_normal_deviation_m"))
    if facing_target is None or allowed_deviation is None or allowed_deviation <= 0.0:
        return []

    outward = (
        target.target_center_xy[0] - facing_target[0],
        target.target_center_xy[1] - facing_target[1],
    )
    outward_length = math.hypot(*outward)
    if outward_length <= 1e-6:
        return []
    outward_unit = (outward[0] / outward_length, outward[1] / outward_length)

    # The exact functional-dependency slot can leave only a narrow table gap.
    # Try progressively more aisle space while remaining within the evaluator's
    # own normal-deviation allowance. The whole-scene gate retains only a pose
    # that also preserves accessibility and the other critic checks.
    return [
        _RepairTarget(
            target.object_id,
            target.relation_type,
            target.check_id,
            (
                target.target_center_xy[0]
                + outward_unit[0] * allowed_deviation * ratio,
                target.target_center_xy[1]
                + outward_unit[1] * allowed_deviation * ratio,
            ),
            target.target_yaw_deg,
            target.group_object_ids,
        )
        for ratio in (0.5, 0.9)
    ]


def _workstation_aisle_target(
    scene: RoomScene,
    diagnostics: dict[str, Any],
    check_id: str,
) -> _RepairTarget | None:
    """Clear a prompt-required aisle between two desks as one atomic move."""
    if diagnostics.get("evaluation_mode") != "between_workstations":
        return None
    endpoint_ids = [
        str(object_id)
        for object_id in diagnostics.get("aisle_endpoint_ids") or []
        if str(object_id)
    ]
    if len(endpoint_ids) != 2 or endpoint_ids[0] == endpoint_ids[1]:
        return None
    free_depth = _float_or_none(diagnostics.get("free_depth_m"))
    required_clearance = _float_or_none(diagnostics.get("required_clearance_m"))
    if free_depth is None or required_clearance is None:
        return None

    first_id, second_id = endpoint_ids
    first = scene.objects.get(UniqueID(first_id))
    second = scene.objects.get(UniqueID(second_id))
    first_center = _world_center_xy(first) if first is not None else None
    second_center = _world_center_xy(second) if second is not None else None
    if (
        first is None
        or second is None
        or first.object_type != ObjectType.FURNITURE
        or second.object_type != ObjectType.FURNITURE
        or first_center is None
        or second_center is None
    ):
        return None
    direction = (
        second_center[0] - first_center[0],
        second_center[1] - first_center[1],
    )
    distance = math.hypot(*direction)
    if distance <= 1e-6:
        return None
    axis = (direction[0] / distance, direction[1] / distance)
    side = (-axis[1], axis[0])
    # Leave a small margin so numerical projection cannot immediately bring a
    # just-sufficient aisle back below the contracted clearance threshold.
    widening = max(0.0, required_clearance - free_depth)
    if widening > 0.0:
        widening += max(0.01, required_clearance * 0.02)
    offset = widening / 2.0
    first_target = (
        first_center[0] - axis[0] * offset,
        first_center[1] - axis[1] * offset,
    )
    second_target = (
        second_center[0] + axis[0] * offset,
        second_center[1] + axis[1] * offset,
    )
    blocking_seats = []
    for object_id in diagnostics.get("blocking_ids") or []:
        seat = scene.objects.get(UniqueID(str(object_id)))
        if seat is None or seat.object_type != ObjectType.FURNITURE:
            continue
        if not _is_seating_object(seat):
            return None
        blocking_seats.append((str(object_id), seat))

    if not blocking_seats:
        return _RepairTarget(
            first_id,
            "clear_access",
            check_id,
            first_target,
            None,
            member_poses=(
                _RepairPose(first_id, first_target, None),
                _RepairPose(second_id, second_target, None),
            ),
        )

    seat_specs: list[tuple[str, SceneObject, str, tuple[float, float]]] = []
    for seat_id, seat in blocking_seats:
        seat_center = _world_center_xy(seat)
        if seat_center is None:
            return None
        desk_id = min(
            (first_id, second_id),
            key=lambda object_id: math.hypot(
                seat_center[0]
                - (first_center if object_id == first_id else second_center)[0],
                seat_center[1]
                - (first_center if object_id == first_id else second_center)[1],
            ),
        )
        outward = (-axis[0], -axis[1]) if desk_id == first_id else axis
        desk = first if desk_id == first_id else second
        face_direction = _workstation_seat_face_direction(desk, outward)
        if face_direction is None:
            # A seat can only be moved out of this aisle when an actual usable
            # desk face points away from it. Otherwise do not turn a clear-path
            # repair into an unassigned, side-on office chair.
            return None
        seat_specs.append((seat_id, seat, desk_id, face_direction))

    achieved_clearance = free_depth + 2.0 * offset
    max_inward_shift = max(
        0.0,
        (achieved_clearance - required_clearance - _WORKSTATION_AISLE_REPAIR_MARGIN_M)
        / 2.0,
    )

    def member_poses(inward_shift: float) -> tuple[_RepairPose, ...]:
        desk_targets = {
            first_id: (
                first_target[0] + axis[0] * inward_shift,
                first_target[1] + axis[1] * inward_shift,
            ),
            second_id: (
                second_target[0] - axis[0] * inward_shift,
                second_target[1] - axis[1] * inward_shift,
            ),
        }
        poses = [
            _RepairPose(first_id, desk_targets[first_id], None),
            _RepairPose(second_id, desk_targets[second_id], None),
        ]
        for seat_id, seat, desk_id, face_direction in seat_specs:
            desk = first if desk_id == first_id else second
            distance = (
                _projected_half_extent_xy(desk, face_direction)
                + _projected_half_extent_xy(seat, face_direction)
                + _WORK_SEAT_SURFACE_GAP_M
            )
            target_desk_center = desk_targets[desk_id]
            seat_target = (
                target_desk_center[0] + face_direction[0] * distance,
                target_desk_center[1] + face_direction[1] * distance,
            )
            poses.append(
                _RepairPose(
                    seat_id,
                    seat_target,
                    compute_optimal_facing_yaw(
                        origin_a=np.array([seat_target[0], seat_target[1], 0.0]),
                        target_point=np.array(
                            [target_desk_center[0], target_desk_center[1], 0.0]
                        ),
                    ),
                )
            )
        return tuple(poses)

    chosen_poses = member_poses(0.0)
    if not _member_poses_fit_floor(scene, chosen_poses, check_id):
        if max_inward_shift <= 0.0:
            return None
        furthest_poses = member_poses(max_inward_shift)
        if not _member_poses_fit_floor(scene, furthest_poses, check_id):
            return None
        lower, upper = 0.0, max_inward_shift
        for _ in range(16):
            midpoint = (lower + upper) / 2.0
            midpoint_poses = member_poses(midpoint)
            if _member_poses_fit_floor(scene, midpoint_poses, check_id):
                upper = midpoint
                furthest_poses = midpoint_poses
            else:
                lower = midpoint
        chosen_poses = furthest_poses
    return _RepairTarget(
        first_id,
        "clear_access",
        check_id,
        chosen_poses[0].target_center_xy,
        None,
        member_poses=chosen_poses,
    )


def _workstation_seat_face_direction(
    desk: SceneObject, outward: tuple[float, float]
) -> tuple[float, float] | None:
    """Return the desk face closest to the aisle-exterior direction."""
    rotation = desk.transform.rotation().matrix()
    front = (float(rotation[0, 1]), float(rotation[1, 1]))
    magnitude = math.hypot(*front)
    if magnitude <= 1e-6:
        return None
    front = (front[0] / magnitude, front[1] / magnitude)
    alignment = front[0] * outward[0] + front[1] * outward[1]
    direction = front if alignment >= 0.0 else (-front[0], -front[1])
    return direction if abs(alignment) >= 0.85 else None


def _projected_half_extent_xy(obj: SceneObject, axis: tuple[float, float]) -> float:
    if obj.bbox_min is None or obj.bbox_max is None:
        return 0.0
    half_extent = (
        np.asarray(obj.bbox_max, dtype=float) - np.asarray(obj.bbox_min, dtype=float)
    ) / 2.0
    rotation = obj.transform.rotation().matrix()
    return float(
        sum(
            abs(
                axis[0] * float(rotation[0, index])
                + axis[1] * float(rotation[1, index])
            )
            * float(half_extent[index])
            for index in range(2)
        )
    )


def _member_poses_fit_floor(
    scene: RoomScene, poses: tuple[_RepairPose, ...], check_id: str
) -> bool:
    for pose in poses:
        obj = scene.objects.get(UniqueID(pose.object_id))
        if obj is None:
            return False
        target = _RepairTarget(
            pose.object_id,
            "clear_access",
            check_id,
            pose.target_center_xy,
            pose.target_yaw_deg,
        )
        transform = _transform_for_target(obj, target)
        if transform is None or not _within_floor_bounds(scene, obj, transform):
            return False
    return True


def _coordinated_dining_targets(
    exact_targets: list[_RepairTarget],
    diagnostics: list[dict[str, Any]],
    *,
    scene: RoomScene | None = None,
    table_id: str = "",
) -> list[_RepairTarget]:
    """Build atomic all-seat candidates within evaluator-authorized slots.

    The edge evaluator owns the normal gap band for seating around a table.
    Try the exact functional slot first, then only the bounded pull-back range
    reported by that evaluator.  This keeps an unrelated clearance conflict
    from stretching an entire dining group into a visually unusable layout.
    """
    if len(exact_targets) != len(diagnostics) or not exact_targets:
        return []
    ordered = sorted(
        zip(exact_targets, diagnostics),
        key=lambda item: item[0].object_id,
    )
    candidates: list[_RepairTarget] = []
    # Prefer the exact table-local assignment.  Larger gaps remain a bounded
    # fallback for real collision/clearance conflicts.
    for ratio in (0.0, 0.5, 0.9):
        poses: list[_RepairPose] = []
        for target, slot in ordered:
            center = target.target_center_xy
            if ratio > 0.0:
                facing_target = _xy(slot.get("facing_target_xy_m"))
                allowed = _float_or_none(slot.get("allowed_normal_deviation_m"))
                if facing_target is not None and allowed is not None and allowed > 0.0:
                    outward = (
                        center[0] - facing_target[0],
                        center[1] - facing_target[1],
                    )
                    length = math.hypot(*outward)
                    if length > 1e-6:
                        center = (
                            center[0] + outward[0] / length * allowed * ratio,
                            center[1] + outward[1] / length * allowed * ratio,
                        )
            poses.append(_RepairPose(target.object_id, center, target.target_yaw_deg))
        anchor = poses[0]
        candidates.append(
            _RepairTarget(
                anchor.object_id,
                exact_targets[0].relation_type,
                exact_targets[0].check_id,
                anchor.target_center_xy,
                anchor.target_yaw_deg,
                member_poses=tuple(poses),
            )
        )

    relocation_candidates = _door_safe_dining_group_targets(
        scene,
        table_id=table_id,
        check_id=exact_targets[0].check_id,
        relation_type=exact_targets[0].relation_type,
        chair_poses=tuple(
            _RepairPose(
                target.object_id, target.target_center_xy, target.target_yaw_deg
            )
            for target, _ in ordered
        ),
    )
    # A local chair layout that remains in a doorway merely converts the
    # topology failure into a physical deadlock. Try a verified door-safe
    # table-and-chair translation before the otherwise least-disruptive local
    # variants. Every pose still goes through the normal floor, physics, and
    # whole-scene scoring gates.
    return [*relocation_candidates, *candidates]


def _door_safe_dining_group_targets(
    scene: RoomScene | None,
    *,
    table_id: str,
    check_id: str,
    relation_type: str,
    chair_poses: tuple[_RepairPose, ...],
) -> list[_RepairTarget]:
    """Translate a table and its exact chair topology away from a blocking door.

    Edge-distribution slots are table-local. When those slots cross an entrance
    sweep, clearing one chair breaks the hard topology contract and moving it
    back restores the door violation. This proposes a single rigid XY
    translation of the complete dining group only when the exact topology is
    blocked by a door. The regular candidate validator remains responsible for
    all other physical and semantic constraints.
    """
    if scene is None or not table_id or not chair_poses:
        return []
    table = scene.objects.get(UniqueID(table_id))
    table_center = _world_center_xy(table) if table is not None else None
    if (
        table is None
        or table.object_type != ObjectType.FURNITURE
        or table_center is None
    ):
        return []

    table_yaw_deg = math.degrees(RollPitchYaw(table.transform.rotation()).yaw_angle())
    group_poses = (
        _RepairPose(table_id, table_center, table_yaw_deg),
        *chair_poses,
    )
    blocked_door_ids = _door_ids_blocked_by_member_poses(scene, group_poses)
    if not blocked_door_ids:
        return []

    openings = {
        str(getattr(opening, "opening_id", "")): opening
        for opening in getattr(scene.room_geometry, "openings", ()) or ()
    }
    candidates: list[_RepairTarget] = []
    for door_id in sorted(blocked_door_ids):
        clearance = door_swing_clearance_bounds(openings.get(door_id))
        if clearance is None:
            continue
        lower, upper = clearance
        door_center = (
            (float(lower[0]) + float(upper[0])) / 2.0,
            (float(lower[1]) + float(upper[1])) / 2.0,
        )
        direction = _away_from_door_direction(table_center, door_center)
        if direction is None:
            continue
        for distance_m in _dining_group_translation_distances(scene):
            translated_poses = tuple(
                _RepairPose(
                    pose.object_id,
                    (
                        pose.target_center_xy[0] + direction[0] * distance_m,
                        pose.target_center_xy[1] + direction[1] * distance_m,
                    ),
                    pose.target_yaw_deg,
                    pose.target_center_z,
                )
                for pose in group_poses
            )
            if not _member_poses_fit_floor(scene, translated_poses, check_id):
                continue
            if _door_ids_blocked_by_member_poses(scene, translated_poses):
                continue
            anchor = translated_poses[0]
            candidates.append(
                _RepairTarget(
                    anchor.object_id,
                    relation_type,
                    check_id,
                    anchor.target_center_xy,
                    anchor.target_yaw_deg,
                    member_poses=translated_poses,
                )
            )
            break
    return candidates


def _door_ids_blocked_by_member_poses(
    scene: RoomScene, poses: tuple[_RepairPose, ...]
) -> set[str]:
    """Return doors blocked by a proposed furniture group without mutating it."""
    originals: dict[UniqueID, RigidTransform] = {}
    member_ids: set[str] = set()
    try:
        for pose in poses:
            object_id = UniqueID(pose.object_id)
            obj = scene.objects.get(object_id)
            if obj is None:
                return set()
            transform = _transform_for_target(
                obj,
                _RepairTarget(
                    pose.object_id,
                    "edge_distribution",
                    "door_safe_dining_group",
                    pose.target_center_xy,
                    pose.target_yaw_deg,
                    target_center_z=pose.target_center_z,
                ),
            )
            if transform is None:
                return set()
            originals[object_id] = obj.transform
            obj.transform = transform
            member_ids.add(str(object_id))
        return {
            str(violation.door_label)
            for violation in compute_door_clearance_violations(scene)
            if str(violation.furniture_id) in member_ids
        }
    finally:
        for object_id, transform in originals.items():
            obj = scene.objects.get(object_id)
            if obj is not None:
                obj.transform = transform


def _away_from_door_direction(
    group_center: tuple[float, float], door_center: tuple[float, float]
) -> tuple[float, float] | None:
    """Use the dominant clearance axis to move a group away from a doorway."""
    delta_x = group_center[0] - door_center[0]
    delta_y = group_center[1] - door_center[1]
    if abs(delta_x) >= abs(delta_y) and abs(delta_x) > 1e-6:
        return (math.copysign(1.0, delta_x), 0.0)
    if abs(delta_y) > 1e-6:
        return (0.0, math.copysign(1.0, delta_y))
    return None


def _dining_group_translation_distances(scene: RoomScene) -> tuple[float, ...]:
    """Offer compact, deterministic offsets bounded by the room diagonal."""
    maximum = min(1.5, _room_diagonal_m(scene))
    return tuple(
        round(0.25 * step, 3) for step in range(1, int(math.floor(maximum / 0.25)) + 1)
    )


def _target_max_translation(scene: RoomScene, target: _RepairTarget) -> float:
    return _repair_target_translation_key(scene, target)[0]


def _room_diagonal_m(scene: RoomScene) -> float:
    geometry = scene.room_geometry
    if geometry is None:
        return 0.0
    return math.hypot(
        float(getattr(geometry, "length", 0.0) or 0.0),
        float(getattr(geometry, "width", 0.0) or 0.0),
    )


def _repair_target_translation_key(
    scene: RoomScene, target: _RepairTarget
) -> tuple[float, float]:
    poses = target.member_poses or (
        _RepairPose(target.object_id, target.target_center_xy, target.target_yaw_deg),
    )
    distances: list[float] = []
    for pose in poses:
        obj = scene.objects.get(UniqueID(pose.object_id))
        center = _world_center_xy(obj) if obj is not None else None
        if center is None:
            return float("inf"), float("inf")
        distances.append(
            math.hypot(
                pose.target_center_xy[0] - center[0],
                pose.target_center_xy[1] - center[1],
            )
        )
    return max(distances, default=0.0), sum(distances)


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


def _is_seating_object(obj: SceneObject) -> bool:
    return bool(_object_role_tokens(obj) & {"chair", "seat", "stool", "bench"})


def _room_center_group_ids(
    scene: RoomScene,
    result: dict[str, Any],
    object_id: str,
    room_center: tuple[float, float],
) -> tuple[str, ...]:
    """Return furniture that should follow a room-center anchor.

    Explicitly related objects always travel with the anchor.  Nearby seating is
    inferred only for shared work surfaces, where moving the surface alone
    breaks its functional seating arrangement.  A generic substring check used
    to classify ``two_seater_sofa`` as a table seat and moved it with an
    unrelated centered rug, which made the otherwise valid rug repair leave the
    room bounds.
    """
    group_ids = [object_id]
    for related_id in (
        result.get("related_objects") or result.get("selected_related_objects") or []
    ):
        related_id = str(related_id or "")
        related = scene.objects.get(UniqueID(related_id))
        if (
            related_id
            and related is not None
            and related.object_type == ObjectType.FURNITURE
            and related_id not in group_ids
        ):
            group_ids.append(related_id)

    anchor = scene.objects.get(UniqueID(object_id))
    if anchor is None or not _is_shared_seating_anchor(anchor):
        return tuple(group_ids)

    for candidate in scene.objects.values():
        if candidate.object_type != ObjectType.FURNITURE:
            continue
        if not _is_seating_object(candidate):
            continue
        candidate_center = _world_center_xy(candidate)
        if candidate_center is None:
            continue
        if (
            math.hypot(
                candidate_center[0] - room_center[0],
                candidate_center[1] - room_center[1],
            )
            <= 2.5
            and str(candidate.object_id) not in group_ids
        ):
            group_ids.append(str(candidate.object_id))
    return tuple(group_ids)


def _room_center_targets(
    scene: RoomScene,
    *,
    object_id: str,
    relation_type: str,
    check_id: str,
    room_center: tuple[float, float],
    allowed_offset_m: float | None,
    group_ids: tuple[str, ...],
) -> list[_RepairTarget]:
    """Return the least-disruptive centered pose before the exact center.

    Centered-in-room contracts define an allowed radius, not a requirement to
    overlap the mathematical room origin.  Moving only as far as needed keeps
    the repair compatible with independent clearance and front-placement
    constraints.  The exact center remains a fallback for layouts where it is
    also feasible.
    """
    targets: list[_RepairTarget] = []
    obj = scene.objects.get(UniqueID(object_id))
    current = _world_center_xy(obj) if obj is not None else None
    if current is not None and allowed_offset_m is not None and allowed_offset_m > 0:
        delta_x = current[0] - room_center[0]
        delta_y = current[1] - room_center[1]
        current_offset = math.hypot(delta_x, delta_y)
        target_offset = max(0.0, allowed_offset_m - _ROOM_CENTER_REPAIR_MARGIN_M)
        if current_offset > target_offset + 1e-6:
            targets.append(
                _RepairTarget(
                    object_id,
                    relation_type,
                    check_id,
                    (
                        room_center[0] + delta_x * target_offset / current_offset,
                        room_center[1] + delta_y * target_offset / current_offset,
                    ),
                    None,
                    group_ids,
                )
            )
    targets.append(
        _RepairTarget(
            object_id,
            relation_type,
            check_id,
            room_center,
            None,
            group_ids,
        )
    )
    return targets


def _is_shared_seating_anchor(obj: SceneObject) -> bool:
    return bool(_object_role_tokens(obj) & {"table", "desk", "workbench", "island"})


def _object_role_tokens(obj: SceneObject) -> set[str]:
    values = (
        obj.object_id,
        obj.name,
        obj.description,
        obj.metadata.get("semantic_name"),
        obj.metadata.get("asset_short_name"),
        obj.metadata.get("category"),
        obj.metadata.get("category_norm"),
    )
    return {
        token
        for value in values
        for token in re.split(r"[^a-z0-9]+", str(value or "").lower())
        if token
    }


def _is_wall(obj: SceneObject | None) -> bool:
    return obj is not None and obj.object_type == ObjectType.WALL


def _scene_wall(scene: RoomScene, wall_id: str) -> SceneObject | None:
    """Resolve an architectural wall whether or not it is a scene object.

    Furniture-only stages normally keep walls in ``room_geometry.walls``.  The
    critic serializes those walls into scene geometry, so a relation result can
    correctly name a wall that is absent from ``scene.objects``.  Repair must
    use the same source or it cannot materialize a valid wall-backed pose.
    """
    if not wall_id:
        return None
    wall = scene.objects.get(UniqueID(wall_id))
    if _is_wall(wall):
        return wall
    for candidate in getattr(scene.room_geometry, "walls", ()) or ():
        if str(candidate.object_id) == wall_id and _is_wall(candidate):
            return candidate
    return None


def _wall_backed_max_gap_m(payload: dict[str, Any], check_id: str) -> float:
    """Read the evaluator's wall-distance tolerance when the check exposes it."""
    check = next(
        (
            candidate
            for candidate in (payload.get("case_pack") or {}).get("checks") or []
            if isinstance(candidate, dict)
            and str(candidate.get("check_id") or "") == check_id
        ),
        {},
    )
    evidence = check.get("evidence") if isinstance(check, dict) else None
    dependency = evidence.get("dependency") if isinstance(evidence, dict) else None
    value = dependency.get("max_distance_m") if isinstance(dependency, dict) else None
    try:
        return max(_WALL_BACKED_CONTACT_GAP_M, float(value))
    except (TypeError, ValueError):
        return _WALL_BACKED_DEFAULT_MAX_GAP_M


def _wall_backed_targets(
    scene: RoomScene,
    object_id: str,
    wall_id: str | None,
    *,
    max_gap_m: float = _WALL_BACKED_DEFAULT_MAX_GAP_M,
) -> list[tuple[tuple[float, float], float]]:
    """Return wall-backed poses that preserve openings and nearby relations.

    A pose flush to a wall can intersect a door's swept/clearance region.  The
    later physics repair then moves the furniture off the wall, which leaves a
    prompt-core relation invalid.  Enumerate lateral positions on the selected
    wall before scoring so the normal relation and the doorway both survive.

    Also offer the largest in-tolerance normal gap.  An object just outside the
    evaluator threshold should move only far enough to satisfy the relation;
    forcing it flush can create a wall collision or break nearby relations.
    """
    if not object_id or not wall_id:
        return []
    seat = scene.objects.get(UniqueID(object_id))
    wall = _scene_wall(scene, wall_id)
    if (
        seat is None
        or wall is None
        or seat.object_type != ObjectType.FURNITURE
        or not _is_wall(wall)
    ):
        return []
    seat_bounds = seat.compute_world_bounds()
    wall_bounds = wall.compute_world_bounds()
    if seat_bounds is None or wall_bounds is None:
        return []

    wall_min, wall_max = wall_bounds
    wall_span = np.asarray(wall_max - wall_min, dtype=float)
    normal_axis = 0 if float(wall_span[0]) < float(wall_span[1]) else 1
    tangent_axis = 1 - normal_axis
    seat_center = np.asarray(_world_center_xy(seat), dtype=float)
    wall_center = (np.asarray(wall_min, dtype=float) + wall_max) / 2.0
    # Choose the wall side containing the room centre.  An escaped object can
    # already be on the exterior side, so its current centre is not authoritative.
    direction = float(-wall_center[normal_axis])
    if abs(direction) < 1e-6:
        return []
    inward = 1.0 if direction > 0.0 else -1.0
    desired_front = np.zeros(2, dtype=float)
    desired_front[normal_axis] = inward
    # RigidTransform yaw maps local +Y to (-sin(yaw), cos(yaw)).
    yaw_deg = math.degrees(math.atan2(-desired_front[0], desired_front[1]))
    yaw = math.radians(yaw_deg)
    axis_x = np.array([math.cos(yaw), math.sin(yaw)])
    axis_y = np.array([-math.sin(yaw), math.cos(yaw)])
    local_min = np.asarray(seat.bbox_min, dtype=float)
    local_max = np.asarray(seat.bbox_max, dtype=float)
    corners = np.array(
        [
            [local_min[0], local_min[1]],
            [local_min[0], local_max[1]],
            [local_max[0], local_min[1]],
            [local_max[0], local_max[1]],
        ]
    )
    # Project the local footprint directly into world axes after the yaw change.
    world_corners = np.outer(corners[:, 0], axis_x) + np.outer(corners[:, 1], axis_y)
    normal_projection = world_corners[:, normal_axis]
    tangent_projection = world_corners[:, tangent_axis]
    normal_half = (
        float(normal_projection.max()) - float(normal_projection.min())
    ) / 2.0
    tangent_half = (
        float(tangent_projection.max()) - float(tangent_projection.min())
    ) / 2.0
    wall_tangent_center = float((wall_min + wall_max)[tangent_axis]) / 2.0
    tangent_limit = max(
        0.0,
        float(wall_span[tangent_axis]) / 2.0 - tangent_half - 0.05,
    )
    tangent_min = wall_tangent_center - tangent_limit
    tangent_max = wall_tangent_center + tangent_limit
    initial_tangent = min(
        max(float(seat_center[tangent_axis]), tangent_min), tangent_max
    )

    # A door on this wall is a hard physical exclusion, independent of whether
    # the critic happened to emit a separate opening-clearance check.  Values
    # immediately to either side of every door are sufficient candidates for a
    # one-dimensional wall interval; final whole-scene scoring chooses among
    # the feasible alternatives.
    tangent_values = [initial_tangent]
    for opening in _same_wall_door_openings(scene, wall_id):
        clearance_min = getattr(opening, "clearance_bbox_min", None)
        clearance_max = getattr(opening, "clearance_bbox_max", None)
        if clearance_min is None or clearance_max is None:
            continue
        try:
            tangent_values.extend(
                (
                    float(clearance_min[tangent_axis])
                    - tangent_half
                    - _OPENING_CLEARANCE_MARGIN_M,
                    float(clearance_max[tangent_axis])
                    + tangent_half
                    + _OPENING_CLEARANCE_MARGIN_M,
                )
            )
        except (TypeError, ValueError, IndexError):
            continue

    gap_values = [_WALL_BACKED_CONTACT_GAP_M]
    relaxed_gap = max(
        _WALL_BACKED_CONTACT_GAP_M,
        float(max_gap_m) - _WALL_BACKED_GAP_MARGIN_M,
    )
    if relaxed_gap > gap_values[0]:
        gap_values.append(relaxed_gap)

    candidates: list[tuple[tuple[float, float], float]] = []
    for gap_m in gap_values:
        for tangent_value in tangent_values:
            if tangent_value < tangent_min - 1e-6 or tangent_value > tangent_max + 1e-6:
                continue
            target = seat_center.copy()
            target[normal_axis] = float(
                (wall_min + wall_max)[normal_axis]
            ) / 2.0 + inward * (
                float(wall_span[normal_axis]) / 2.0 + normal_half + gap_m
            )
            target[tangent_axis] = tangent_value
            center = (float(target[0]), float(target[1]))
            if _wall_backed_pose_blocks_door(
                scene=scene,
                obj=seat,
                center=center,
                yaw_deg=yaw_deg,
                wall_id=wall_id,
            ):
                continue
            candidates.append((center, yaw_deg))

    # Prefer the least disruptive legal pose.  This preserves nearby prompt
    # relations while still allowing a later candidate to take a flush pose if
    # it scores better for the whole scene.
    unique_candidates = {
        (round(center[0], 9), round(center[1], 9), round(yaw, 9)): (center, yaw)
        for center, yaw in candidates
    }
    return sorted(
        unique_candidates.values(),
        key=lambda candidate: (
            math.hypot(
                candidate[0][0] - float(seat_center[0]),
                candidate[0][1] - float(seat_center[1]),
            ),
            candidate[0],
        ),
    )


def _wall_backed_target(
    scene: RoomScene, object_id: str, wall_id: str | None
) -> tuple[tuple[float, float], float] | None:
    """Return the first legal wall-backed pose for private callers."""
    return next(iter(_wall_backed_targets(scene, object_id, wall_id)), None)


def _same_wall_door_openings(scene: RoomScene, wall_id: str) -> list[Any]:
    """Return door openings attached to ``wall_id`` using stable geometry IDs."""
    geometry = scene.room_geometry
    if geometry is None:
        return []
    normalized_wall_id = str(wall_id).lower()
    openings: list[Any] = []
    for opening in getattr(geometry, "openings", ()) or ():
        opening_type = getattr(opening, "opening_type", "")
        if hasattr(opening_type, "value"):
            opening_type = opening_type.value
        if str(opening_type).lower() != "door":
            continue
        direction = str(getattr(opening, "wall_direction", "") or "").lower()
        if direction and direction not in normalized_wall_id:
            continue
        openings.append(opening)
    return openings


def _wall_backed_pose_blocks_door(
    *,
    scene: RoomScene,
    obj: SceneObject,
    center: tuple[float, float],
    yaw_deg: float,
    wall_id: str,
) -> bool:
    """Check a proposed wall pose against same-wall door clearance boxes."""
    target = _RepairTarget(
        str(obj.object_id),
        "back_against_wall",
        "",
        center,
        yaw_deg,
    )
    transform = _transform_for_target(obj, target)
    if transform is None:
        return True
    old_transform = obj.transform
    try:
        obj.transform = transform
        bounds = obj.compute_world_bounds()
    finally:
        obj.transform = old_transform
    if bounds is None:
        return True
    lower, upper = (
        np.asarray(bounds[0], dtype=float),
        np.asarray(bounds[1], dtype=float),
    )
    for opening in _same_wall_door_openings(scene, wall_id):
        zone_min = getattr(opening, "clearance_bbox_min", None)
        zone_max = getattr(opening, "clearance_bbox_max", None)
        if zone_min is None or zone_max is None:
            continue
        try:
            clearance_min = np.asarray(zone_min, dtype=float)
            clearance_max = np.asarray(zone_max, dtype=float)
        except (TypeError, ValueError):
            continue
        if (
            float(lower[0]) < float(clearance_max[0])
            and float(upper[0]) > float(clearance_min[0])
            and float(lower[1]) < float(clearance_max[1])
            and float(upper[1]) > float(clearance_min[1])
            and float(lower[2]) < float(clearance_max[2])
            and float(upper[2]) > float(clearance_min[2])
        ):
            return True
    return False


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
        [
            target.target_center_xy[0],
            target.target_center_xy[1],
            (
                target.target_center_z
                if target.target_center_z is not None
                else world_center[2]
            ),
        ],
        dtype=float,
    )
    translation = target_world_center - new_rpy.ToRotationMatrix().multiply(
        local_center
    )
    return RigidTransform(rpy=new_rpy, p=translation)


def _apply_repair_target(
    scene: RoomScene, target: _RepairTarget, anchor_transform: RigidTransform
) -> bool:
    """Apply an anchor pose, optionally translating its associated furniture group."""
    if target.member_poses:
        transforms: dict[UniqueID, RigidTransform] = {}
        for pose in target.member_poses:
            object_id = UniqueID(pose.object_id)
            obj = scene.objects.get(object_id)
            if obj is None or obj.object_type != ObjectType.FURNITURE:
                return False
            pose_target = _RepairTarget(
                pose.object_id,
                target.relation_type,
                target.check_id,
                pose.target_center_xy,
                pose.target_yaw_deg,
                target_center_z=pose.target_center_z,
            )
            transform = _transform_for_target(obj, pose_target)
            if transform is None or not _within_floor_bounds(scene, obj, transform):
                return False
            transforms[object_id] = transform
        for object_id, transform in transforms.items():
            _move_object_with_surfaces(scene, object_id, transform)
        return bool(transforms)

    object_ids = target.group_object_ids or (target.object_id,)
    anchor = scene.objects.get(UniqueID(target.object_id))
    if anchor is None or not _repair_target_type_allowed(anchor, target.relation_type):
        return False
    if target.relation_type == "instructional_surface_alignment":
        if anchor.object_type == ObjectType.WALL_MOUNTED:
            return _move_wall_mounted_instructional_surface(
                scene,
                anchor,
                target_wall_id=target.target_wall_id,
                target_transform=anchor_transform,
            )
        _move_object_with_surfaces(scene, anchor.object_id, anchor_transform)
        return True
    delta = anchor_transform @ anchor.transform.inverse()
    transforms: dict[UniqueID, RigidTransform] = {}
    for object_id in object_ids:
        obj = scene.objects.get(UniqueID(object_id))
        if obj is None or obj.object_type != ObjectType.FURNITURE:
            continue
        transform = (
            anchor_transform if object_id == target.object_id else delta @ obj.transform
        )
        if not _within_floor_bounds(scene, obj, transform):
            return False
        transforms[obj.object_id] = transform
    if target.object_id not in {str(object_id) for object_id in transforms}:
        return False
    for object_id, transform in transforms.items():
        _move_object_with_surfaces(scene, object_id, transform)
    return True


def _repair_target_type_allowed(obj: SceneObject, relation_type: str) -> bool:
    if obj.object_type == ObjectType.FURNITURE:
        return True
    return (
        relation_type == "instructional_surface_alignment"
        and obj.object_type == ObjectType.WALL_MOUNTED
    )


def _move_wall_mounted_instructional_surface(
    scene: RoomScene,
    obj: SceneObject,
    *,
    target_wall_id: str | None,
    target_transform: RigidTransform,
) -> bool:
    """Move a focal surface through the canonical wall-local placement model."""
    surface = _target_wall_surface(scene, target_wall_id)
    if surface is None:
        return False

    local_translation = surface.transform.inverse().multiply(
        target_transform.translation()
    )
    position_x = float(local_translation[0])
    position_z = float(local_translation[2])
    rotation_2d = _wall_rotation_2d(surface, target_transform)
    world_transform = surface.to_world_pose(
        position_x=position_x,
        position_z=position_z,
        rotation_deg=math.degrees(rotation_2d),
    )
    _move_object_with_surfaces(scene, obj.object_id, world_transform)
    obj.placement_info = PlacementInfo(
        parent_surface_id=surface.surface_id,
        position_2d=np.array([position_x, position_z], dtype=float),
        rotation_2d=rotation_2d,
        placement_method="wall_placement",
    )
    return True


def _target_wall_surface(
    scene: RoomScene, target_wall_id: str | None
) -> WallSurface | None:
    if not target_wall_id or scene.room_geometry is None:
        return None
    # Import lazily: wall_agents package initialization loads the stateful base
    # agent, which itself imports this repair module.
    from scenesmith.wall_agents.tools.wall_surface import (
        extract_wall_surfaces_from_room_geometry,
    )

    surfaces = extract_wall_surfaces_from_room_geometry(
        scene.room_geometry,
        room_id=scene.room_id,
    )
    normalized_target = str(target_wall_id).strip().lower()
    exact = next(
        (
            surface
            for surface in surfaces
            if normalized_target
            in {
                str(surface.surface_id).lower(),
                str(surface.wall_id).lower(),
            }
        ),
        None,
    )
    if exact is not None:
        return exact
    return next(
        (
            surface
            for surface in surfaces
            if re.search(
                rf"(^|_){re.escape(surface.wall_direction.value)}(?:_wall)?$",
                normalized_target,
            )
        ),
        None,
    )


def _wall_rotation_2d(surface: WallSurface, world_transform: RigidTransform) -> float:
    local_rotation = (
        surface.transform.rotation().inverse().multiply(world_transform.rotation())
    )
    placement_rotation = local_rotation.multiply(RotationMatrix.MakeZRotation(-math.pi))
    return -float(RollPitchYaw(placement_rotation).pitch_angle())


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
    margin = _ROOM_CONTAINMENT_MARGIN_M
    epsilon = 1e-6
    return (
        float(lower[0]) >= -float(geometry.length) / 2.0 + margin - epsilon
        and float(upper[0]) <= float(geometry.length) / 2.0 - margin + epsilon
        and float(lower[1]) >= -float(geometry.width) / 2.0 + margin - epsilon
        and float(upper[1]) <= float(geometry.width) / 2.0 - margin + epsilon
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
