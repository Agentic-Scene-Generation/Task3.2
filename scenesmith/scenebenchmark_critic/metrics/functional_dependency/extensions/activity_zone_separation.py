"""Relation-derived separation between dining and media-viewing activity zones."""

from __future__ import annotations

import math

from typing import Any

from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from scenesmith.scenebenchmark_critic.core.geometry import (
    bbox_center_xy,
    load_geometry,
    object_category,
    object_footprint_polygon,
)
from scenesmith.scenebenchmark_critic.intent_contract import (
    _normalize_selector_category,
    bound_ids,
    contract_constraints,
)
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.semantics import (
    _is_media_target,
    _is_seating_subject,
)


_DINING_RELATIONS = {"dining_set", "edge_distribution"}
_MEDIA_RELATION = "seating_to_media"
_MIN_INTERSECTION_AREA_M2 = 0.01
_MIN_VIEW_HALF_WIDTH_M = 0.45
_VIEW_SIDE_MARGIN_M = 0.15


def evaluate_activity_zone_separation(
    case_pack: dict[str, Any],
) -> list[dict[str, Any]]:
    """Keep an independently declared dining group out of a media view corridor."""
    store = load_geometry(case_pack)
    if store is None:
        return []
    checks = [
        check
        for check in [
            *(case_pack.get("checks") or []),
            *(case_pack.get("_prior_extension_results") or []),
        ]
        if isinstance(check, dict)
    ]
    dining_ids = _dining_group_ids(checks, store.objects)
    if len(dining_ids) < 2:
        return []
    dining_anchor = _dining_anchor_id(dining_ids, store.objects)
    dining_shape = _group_shape(dining_ids, store.objects)
    if not dining_anchor or dining_shape is None:
        return []

    results: list[dict[str, Any]] = []
    for seat_id, media_id, proxy_for in _media_axes(case_pack, checks, store.objects):
        seat = store.objects.get(seat_id)
        media = store.objects.get(media_id)
        if (
            seat is None
            or media is None
            or seat_id in dining_ids
            or media_id in dining_ids
        ):
            continue
        seat_center = bbox_center_xy(seat)
        media_center = bbox_center_xy(media)
        if seat_center is None or media_center is None:
            continue
        axis = (
            float(media_center[0] - seat_center[0]),
            float(media_center[1] - seat_center[1]),
        )
        axis_length = math.hypot(*axis)
        if axis_length <= 1e-6:
            continue
        unit_axis = (axis[0] / axis_length, axis[1] / axis_length)
        unit_normal = (-unit_axis[1], unit_axis[0])
        half_width = _view_half_width(seat, unit_normal)
        corridor = LineString([seat_center, media_center]).buffer(
            half_width,
            cap_style=2,
            join_style=2,
        )
        seat_shape = _object_shape(seat)
        media_shape = _object_shape(media)
        if seat_shape is not None:
            corridor = corridor.difference(seat_shape)
        if media_shape is not None:
            corridor = corridor.difference(media_shape)
        intersection_area = float(dining_shape.intersection(corridor).area)
        blocked = intersection_area > _MIN_INTERSECTION_AREA_M2
        check_id = f"activity_zone_separation__{seat_id}__{media_id}__{dining_anchor}"
        results.append(
            {
                "check_id": check_id,
                "metric": "functional_dependency",
                "label": "fail" if blocked else "pass",
                "confidence": 1.0,
                "primary_object": dining_anchor,
                "related_objects": sorted(dining_ids | {seat_id, media_id}),
                "selected_related_objects": sorted(dining_ids),
                "blocking_objects": sorted(dining_ids) if blocked else [],
                "relation_type": "activity_zone_separation",
                "reason": (
                    "The dining activity group intersects the seating-to-media "
                    f"view corridor by {intersection_area:.3f} square metres."
                    if blocked
                    else "The dining activity group stays beside the media view corridor."
                ),
                "diagnostics": {
                    "group_object_ids": sorted(dining_ids),
                    "seating_object_id": seat_id,
                    "media_object_id": media_id,
                    "media_proxy_for_category": proxy_for,
                    "view_axis_unit": [unit_axis[0], unit_axis[1]],
                    "view_normal_unit": [unit_normal[0], unit_normal[1]],
                    "view_half_width_m": half_width,
                    "intersection_area_m2": intersection_area,
                },
                "evidence": {
                    "constraint": (
                        "independent relation-derived dining and media activity groups"
                    )
                },
                "evaluation_source": "scenesmith_activity_zone_separation",
                "scoring_tier": "core",
            }
        )
    return results


def _media_axes(
    case_pack: dict[str, Any],
    checks: list[dict[str, Any]],
    objects: dict[str, dict[str, Any]],
) -> list[tuple[str, str, str]]:
    """Resolve media corridors from checks or the immutable hard intent graph."""
    axes: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    for media_check in checks:
        if str(media_check.get("relation_type") or "") != _MEDIA_RELATION:
            continue
        seat_id = str(
            media_check.get("subject_id") or media_check.get("primary_object") or ""
        )
        media_id = next(
            (
                str(value)
                for value in (
                    media_check.get("target_ids")
                    or media_check.get("related_objects")
                    or []
                )
                if str(value) in objects
            ),
            "",
        )
        if seat_id in objects and media_id:
            axes.append((seat_id, media_id, ""))
            seen.add((seat_id, media_id))

    object_rows = list(objects.values())
    constraints = contract_constraints(case_pack)
    support_constraints = [
        row
        for row in constraints
        if str(row.get("relation") or "") == "on_top_of"
        and str(row.get("strength") or "hard").lower() == "hard"
    ]
    for relation in constraints:
        if str(relation.get("relation") or "") not in {"faces", "across_from"}:
            continue
        if str(relation.get("strength") or "hard").lower() != "hard":
            continue
        seat_ids = [
            object_id
            for object_id in bound_ids(relation.get("subjects"), object_rows)
            if _is_seating_subject(objects.get(object_id) or {})
        ]
        media_ids = [
            object_id
            for object_id in bound_ids(relation.get("targets"), object_rows)
            if _is_media_target(objects.get(object_id) or {})
        ]
        proxy_for = ""
        if not media_ids:
            requested_category = _normalize_selector_category(
                (relation.get("targets") or {}).get("category")
            )
            for support_relation in support_constraints:
                supported_category = _normalize_selector_category(
                    (support_relation.get("subjects") or {}).get("category")
                )
                if not requested_category or supported_category != requested_category:
                    continue
                support_ids = bound_ids(support_relation.get("targets"), object_rows)
                if support_ids:
                    media_ids = [support_ids[0]]
                    proxy_for = requested_category
                    break
        for seat_id in seat_ids:
            for media_id in media_ids:
                if (seat_id, media_id) in seen:
                    continue
                axes.append((seat_id, media_id, proxy_for))
                seen.add((seat_id, media_id))
    return axes


def _check_ids(check: dict[str, Any]) -> set[str]:
    subject_id = check.get("subject_id") or check.get("primary_object")
    target_ids = check.get("target_ids") or check.get("related_objects") or []
    return {str(value) for value in [subject_id, *target_ids] if value}


def _dining_group_ids(
    checks: list[dict[str, Any]],
    objects: dict[str, dict[str, Any]],
) -> set[str]:
    groups: dict[str, set[str]] = {relation: set() for relation in _DINING_RELATIONS}
    for check in checks:
        relation = str(check.get("relation_type") or "")
        if relation not in _DINING_RELATIONS:
            continue
        ids = {object_id for object_id in _check_ids(check) if object_id in objects}
        categories = {object_category(objects[object_id]) for object_id in ids}
        if any("table" in category for category in categories) and any(
            "chair" in category or category in {"bench", "stool"}
            for category in categories
        ):
            groups[relation].update(ids)
    return groups["edge_distribution"] or groups["dining_set"]


def _dining_anchor_id(
    object_ids: set[str],
    objects: dict[str, dict[str, Any]],
) -> str:
    tables = sorted(
        object_id
        for object_id in object_ids
        if "table" in object_category(objects.get(object_id))
    )
    return tables[0] if tables else ""


def _object_shape(obj: dict[str, Any]) -> Polygon | None:
    points = object_footprint_polygon(obj)
    if not points:
        return None
    shape = Polygon(points)
    if shape.is_empty or not shape.is_valid:
        return None
    return shape


def _group_shape(
    object_ids: set[str],
    objects: dict[str, dict[str, Any]],
) -> Polygon | None:
    shapes = [
        shape
        for object_id in sorted(object_ids)
        if (shape := _object_shape(objects[object_id])) is not None
    ]
    if not shapes:
        return None
    merged = unary_union(shapes).convex_hull
    return merged if isinstance(merged, Polygon) and not merged.is_empty else None


def _view_half_width(
    seat: dict[str, Any],
    normal: tuple[float, float],
) -> float:
    points = object_footprint_polygon(seat) or []
    if not points:
        return _MIN_VIEW_HALF_WIDTH_M
    projections = [point[0] * normal[0] + point[1] * normal[1] for point in points]
    seat_half_width = (max(projections) - min(projections)) / 2.0
    return max(_MIN_VIEW_HALF_WIDTH_M, seat_half_width + _VIEW_SIDE_MARGIN_M)
