from __future__ import annotations

from typing import Any

import numpy as np

from scenesmith.scenebenchmark_critic.core.geometry import (
    GeometryStore,
    floor_polygon_for_object,
    is_small_object,
    load_geometry,
    object_affordances,
    object_bbox,
)
from scenesmith.scenebenchmark_critic.metrics.spatial_accessibility.config import (
    _agent_profiles,
    _params,
)
from scenesmith.scenebenchmark_critic.metrics.spatial_accessibility.grid import (
    _build_grid,
    _entry_component,
    _largest_component,
)
from scenesmith.scenebenchmark_critic.metrics.spatial_accessibility.obstacles import (
    _obstacle_mask,
)
from scenesmith.scenebenchmark_critic.metrics.spatial_accessibility.reach import (
    _min_reach_distance,
)
from scenesmith.scenebenchmark_critic.metrics.spatial_accessibility.results import (
    _label_rank,
    _profile_diagnostics,
    _result,
    _unknown,
)
from scenesmith.scenebenchmark_critic.metrics.spatial_accessibility.zones import (
    _access_zones,
    _blocking_objects_for_zone,
    _oriented_rect_mask,
    _zone_aabb,
)


def evaluate_spatial_accessibility(
    store: GeometryStore | dict[str, Any],
    check: dict[str, Any],
    config: Any,
) -> dict[str, Any]:
    if isinstance(store, dict):
        loaded = load_geometry(store)
        if loaded is None:
            return _unknown(check, "Rule accessibility could not load scene geometry.")
        store = loaded
    subject_id = str(check.get("subject_id") or "")
    subject = store.objects.get(subject_id)
    if subject is None:
        return _unknown(
            check, f"Rule accessibility could not find subject object `{subject_id}`."
        )
    if object_bbox(subject) is None:
        return _unknown(
            check,
            f"Rule accessibility could not find bbox geometry for `{subject_id}`.",
        )

    polygon = floor_polygon_for_object(store, subject)
    if not polygon:
        return _unknown(check, "Rule accessibility could not find room floor geometry.")

    params = _params(config)
    profiles = _agent_profiles(config, params)
    grid = _build_grid(polygon, params["grid_resolution_m"])
    if grid is None:
        return _unknown(check, "Rule accessibility could not build a valid floor grid.")
    xs, ys, floor_mask = grid
    zones = _access_zones(
        subject, str(check.get("affordance") or ""), params["access_zone_depth_m"]
    )
    if not zones:
        return _unknown(
            check,
            f"Rule accessibility could not infer an access zone for `{subject_id}`.",
        )

    # 2026-07-11 修改原因：功能依赖确认的配套座椅属于桌面的预期使用拓扑，
    # 仅从该桌面的通行障碍与 blocker 诊断中排除，仍保留座椅自身的可达性检查。
    expected_companion_ids = {
        str(item) for item in (check.get("expected_companion_ids") or []) if str(item)
    }
    profile_results = [
        _evaluate_profile(
            store,
            subject,
            subject_id,
            str(check.get("affordance") or ""),
            xs,
            ys,
            floor_mask,
            polygon,
            zones,
            params,
            profile,
            expected_companion_ids,
        )
        for profile in profiles
    ]
    aggregate = min(profile_results, key=lambda item: _label_rank(item["label"]))
    label = aggregate["label"]
    confidence = aggregate["confidence"]
    blockers = aggregate["blocking_objects"] if label != "pass" else []
    best_ratio = float(aggregate["access_ratio"])
    best_side = aggregate["access_side"]
    reason = (
        f"Rule accessibility: limiting profile `{aggregate['profile_id']}` has best `{best_side}` access zone "
        f"for `{subject_id}` with {best_ratio:.2f} connected stance overlap and "
        f"{aggregate['min_reach_distance_m']:.2f}m minimum reach distance."
    )
    if aggregate.get("reach_posture") == "crouch":
        reason += " Minimum reach uses crouch/lean posture."
    if blockers and label != "pass":
        reason += f" Nearby/intersecting obstacles: {', '.join(blockers)}."
    return _result(
        check,
        label=label,
        reason=reason,
        confidence=confidence,
        blocking_objects=blockers if label != "pass" else [],
        diagnostics={
            "access_ratio": best_ratio,
            "access_side": best_side,
            "min_reach_distance_m": aggregate["min_reach_distance_m"],
            "reach_posture": aggregate["reach_posture"],
            "reach_origin_height_m": aggregate["reach_origin_height_m"],
            "reachable_stance_count": aggregate["reachable_stance_count"],
            "per_profile": {
                item["profile_id"]: _profile_diagnostics(item)
                for item in profile_results
            },
            "zone_scores": aggregate["zone_scores"],
            "expected_companion_ids": sorted(expected_companion_ids),
        },
    )


def _evaluate_profile(
    store: GeometryStore,
    subject: dict[str, Any],
    subject_id: str,
    affordance: str,
    xs: np.ndarray,
    ys: np.ndarray,
    floor_mask: np.ndarray,
    floor_polygon: list[tuple[float, float]],
    zones: list[tuple[str, dict[str, Any]]],
    params: dict[str, float],
    profile: dict[str, Any],
    expected_companion_ids: set[str],
) -> dict[str, Any]:
    obstacle_mask = _obstacle_mask(
        store,
        subject_id,
        xs,
        ys,
        params,
        profile,
        ignored_object_ids=expected_companion_ids,
    )
    walkable = floor_mask & ~obstacle_mask
    component = _entry_component(store, walkable, xs, ys, floor_polygon)
    if component is None:
        component = _largest_component(walkable)
    if component is None:
        return {
            "profile_id": profile["id"],
            "label": "fail",
            "confidence": 0.9,
            "access_ratio": 0.0,
            "access_side": None,
            "zone_scores": {},
            "reachable_stance_count": 0,
            "min_reach_distance_m": float("inf"),
            "reach_posture": None,
            "reach_origin_height_m": None,
            "blocking_objects": [],
        }

    reach_only = _uses_reach_only_access(subject, affordance)
    scored: list[tuple[float, str, np.ndarray, tuple[float, float, float, float]]] = []
    for zone_name, zone in zones:
        zone_mask = _oriented_rect_mask(xs, ys, zone)
        zone_area = int(zone_mask.sum())
        if zone_area <= 0:
            scored.append((0.0, zone_name, zone_mask, _zone_aabb(zone)))
            continue
        ratio = float((zone_mask & component).sum() / zone_area)
        scored.append((ratio, zone_name, zone_mask, _zone_aabb(zone)))
    scored.sort(key=lambda item: item[0], reverse=True)
    best_ratio, best_side, best_zone_mask, best_aabb = scored[0]
    if reach_only:
        stance_mask = component
        best_ratio = 1.0
        best_side = "connected_floor"
    else:
        stance_mask = best_zone_mask & component
    reach = _min_reach_distance(subject, affordance, xs, ys, stance_mask, profile)
    min_reach = float(reach["min_reach_distance_m"])
    reachable_stance_count = int(stance_mask.sum())
    reach_radius = float(profile["reach_radius_m"])

    if best_ratio >= params["pass_ratio"] and min_reach <= reach_radius:
        label = "pass"
        confidence = 0.92
    elif best_ratio >= params["degraded_ratio"] and min_reach <= reach_radius * 1.15:
        label = "degraded"
        confidence = 0.82
    else:
        label = "fail"
        confidence = 0.88

    blockers = _blocking_objects_for_zone(
        store,
        subject_id,
        best_aabb,
        limit=5,
        height_threshold_m=params["height_threshold_m"],
        ignored_object_ids=expected_companion_ids,
    )
    return {
        "profile_id": profile["id"],
        "label": label,
        "confidence": confidence,
        "access_ratio": best_ratio,
        "access_side": best_side,
        "zone_scores": {side: ratio for ratio, side, _, _ in scored},
        "reachable_stance_count": reachable_stance_count,
        "min_reach_distance_m": min_reach,
        "reach_posture": reach["reach_posture"],
        "reach_origin_height_m": reach["reach_origin_height_m"],
        "blocking_objects": blockers if label != "pass" else [],
    }


def _uses_reach_only_access(subject: dict[str, Any], affordance: str) -> bool:
    hints = subject.get("functional_hints") or {}
    scene_type = str(hints.get("scene_object_type") or subject.get("object_type") or "")
    scene_type = scene_type.strip().lower().replace("-", "_")
    if scene_type == "manipuland":
        return True
    return affordance == "graspable" and (
        is_small_object(subject) or object_affordances(subject) == {"graspable"}
    )
