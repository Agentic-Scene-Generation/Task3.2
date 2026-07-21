from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from scenesmith.scenebenchmark_critic.core.geometry import (
    bbox_gap_xy,
    object_footprint_polygon,
)
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.profiles import (
    ObjectFunctionProfile,
    object_function_profile,
)


@dataclass(frozen=True)
class SupportThresholds:
    pass_overlap: float = 0.55
    edge_overlap: float = 0.30
    degraded_overlap: float = 0.15
    pass_height_tolerance_m: float = 0.14
    edge_height_tolerance_m: float = 0.18
    degraded_height_tolerance_m: float = 0.28
    clearance_slack_m: float = 0.10
    coarse_top_height_tolerance_m: float = 0.42


@dataclass(frozen=True)
class SupportSurfaceCandidate:
    surface_id: str
    kind: str
    polygon_xy: tuple[tuple[float, float], ...]
    height_z: float
    clearance_m: float
    source: str


@dataclass(frozen=True)
class SupportEvidence:
    surface_id: str
    surface_kind: str
    source: str
    overlap_ratio: float
    height_delta_m: float
    clearance_m: float
    subject_height_m: float
    score: float


@dataclass(frozen=True)
class SupportAssessment:
    label: str
    confidence: float
    reason: str
    evidence: SupportEvidence | None


def assess_direct_support(
    subject: dict[str, Any],
    target: dict[str, Any],
    *,
    thresholds: SupportThresholds = SupportThresholds(),
) -> SupportAssessment | None:
    subject_poly = object_footprint_polygon(subject)
    subject_bbox = subject.get("bbox_world") or {}
    smin = subject_bbox.get("min") or []
    smax = subject_bbox.get("max") or []
    if not subject_poly or len(smin) < 3 or len(smax) < 3:
        return None

    target_profile = object_function_profile(target)
    candidates = build_support_surface_candidates(target, target_profile)
    if not candidates:
        return None

    subject_area = _polygon_area_xy(subject_poly)
    if subject_area <= 1e-6:
        return None
    subject_bottom = float(smin[2])
    subject_height = max(float(smax[2]) - subject_bottom, 0.0)

    evidence = [
        _surface_evidence(
            subject_poly,
            subject_area,
            subject_bottom,
            subject_height,
            candidate,
            thresholds=thresholds,
        )
        for candidate in candidates
    ]
    best = max(evidence, key=lambda item: item.score)
    return _classify_support_evidence(subject, target, best, thresholds=thresholds)


def build_support_surface_candidates(
    target: dict[str, Any],
    profile: ObjectFunctionProfile | None = None,
) -> list[SupportSurfaceCandidate]:
    profile = profile or object_function_profile(target)
    candidates: list[SupportSurfaceCandidate] = []
    for index, region in enumerate(target.get("support_regions") or []):
        if not isinstance(region, dict):
            continue
        access_type = str(region.get("access_type") or "").strip().lower()
        if access_type in {"sealed", "blocked", "decorative"}:
            continue
        polygon = _region_polygon(region)
        height = _number(region.get("height_world_z"))
        if height is None:
            height = _number(region.get("height_z"))
        if len(polygon) < 3 or height is None:
            continue
        candidates.append(
            SupportSurfaceCandidate(
                surface_id=str(region.get("region_id") or f"support_region_{index}"),
                kind=str(region.get("support_kind") or "support_region")
                .strip()
                .lower(),
                polygon_xy=tuple(polygon),
                height_z=height,
                clearance_m=max(_number(region.get("clearance_above_m")) or 0.0, 0.0),
                source="support_region",
            )
        )

    target_poly = object_footprint_polygon(target)
    target_bbox = target.get("bbox_world") or {}
    target_top = _bbox_top_z(target_bbox)
    if profile.can_support_top and target_poly and target_top is not None:
        candidate_heights = [
            candidate.height_z
            for candidate in candidates
            if candidate.kind == "top_surface" or candidate.source == "bbox_profile"
        ]
        if not any(abs(height - target_top) <= 0.03 for height in candidate_heights):
            candidates.append(
                SupportSurfaceCandidate(
                    surface_id="bbox_top",
                    kind="top_surface",
                    polygon_xy=tuple(target_poly),
                    height_z=target_top,
                    clearance_m=10.0,
                    source="bbox_profile",
                )
            )
        inferred_height = _inferred_interaction_top_z(target)
        known_heights = [candidate.height_z for candidate in candidates]
        if inferred_height is not None and not any(
            abs(height - inferred_height) <= 0.03 for height in known_heights
        ):
            candidates.append(
                SupportSurfaceCandidate(
                    surface_id="interaction_top",
                    kind="top_surface",
                    polygon_xy=tuple(target_poly),
                    height_z=inferred_height,
                    clearance_m=10.0,
                    source="functional_hints",
                )
            )

    return candidates


def support_assessment_diagnostics(
    assessment: SupportAssessment | None,
) -> dict[str, Any]:
    if assessment is None or assessment.evidence is None:
        return {}
    evidence = assessment.evidence
    return {
        "support_surface_id": evidence.surface_id,
        "support_surface_kind": evidence.surface_kind,
        "support_surface_source": evidence.source,
        "support_overlap_ratio": round(evidence.overlap_ratio, 4),
        "support_height_delta_m": round(evidence.height_delta_m, 4),
        "support_clearance_m": round(evidence.clearance_m, 4),
        "support_evidence_score": round(evidence.score, 4),
    }


def _inferred_interaction_top_z(target: dict[str, Any]) -> float | None:
    hints = target.get("functional_hints") or {}
    interaction_height = hints.get("interaction_height_m") or {}
    if not isinstance(interaction_height, dict):
        return None
    bbox_top = _bbox_top_z(target.get("bbox_world") or {})
    if bbox_top is None:
        return None
    values = [
        _number(interaction_height.get("min")),
        _number(interaction_height.get("max")),
    ]
    for value in values:
        if value is not None and value >= bbox_top - 0.03:
            return value
    return None


def _surface_level_tolerance(
    evidence: SupportEvidence, thresholds: SupportThresholds
) -> float:
    if evidence.source == "functional_hints":
        return thresholds.coarse_top_height_tolerance_m
    return thresholds.degraded_height_tolerance_m


def _coarse_top_support_match(
    subject: dict[str, Any],
    target: dict[str, Any],
    evidence: SupportEvidence,
    *,
    thresholds: SupportThresholds,
) -> bool:
    if evidence.source != "functional_hints":
        return False
    if evidence.surface_kind != "top_surface":
        return False
    if evidence.overlap_ratio < 0.65:
        return False
    if evidence.height_delta_m > thresholds.coarse_top_height_tolerance_m:
        return False
    profile = object_function_profile(subject)
    if profile.is_small_placeable or profile.source == "explicit":
        return True
    bbox = subject.get("bbox_world") or {}
    size = bbox.get("size") or []
    if len(size) < 3:
        return False
    width = max(float(size[0]), 0.0)
    depth = max(float(size[1]), 0.0)
    height = max(float(size[2]), 0.0)
    return width <= 0.7 and depth <= 0.7 and width * depth <= 0.35 and height <= 1.1


def _classify_support_evidence(
    subject: dict[str, Any],
    target: dict[str, Any],
    evidence: SupportEvidence,
    *,
    thresholds: SupportThresholds,
) -> SupportAssessment:
    overlap = evidence.overlap_ratio
    dz = evidence.height_delta_m
    fits_clearance = (
        evidence.subject_height_m <= evidence.clearance_m + thresholds.clearance_slack_m
    )
    subject_profile = object_function_profile(subject)

    if (
        overlap >= thresholds.pass_overlap
        and dz <= thresholds.pass_height_tolerance_m
        and fits_clearance
    ):
        label, confidence = "pass", 0.91
    elif (
        _small_placeable_interaction_top_match(subject_profile, evidence)
        and fits_clearance
    ):
        label, confidence = "pass", 0.87
    elif (
        overlap >= thresholds.edge_overlap
        and dz <= thresholds.edge_height_tolerance_m
        and fits_clearance
    ):
        label, confidence = "pass", 0.86
    elif (
        _rigid_top_strong_overlap_match(subject_profile, target, evidence)
        and fits_clearance
    ):
        label, confidence = "pass", 0.84
    elif (
        _coarse_top_support_match(subject, target, evidence, thresholds=thresholds)
        and fits_clearance
    ):
        label, confidence = "pass", 0.84
    elif (
        subject_profile.source == "explicit"
        and subject_profile.is_small_placeable
        and overlap >= 0.80
        and dz <= thresholds.degraded_height_tolerance_m
        and fits_clearance
    ):
        label, confidence = "pass", 0.83
    elif overlap >= thresholds.degraded_overlap and dz <= _surface_level_tolerance(
        evidence, thresholds
    ):
        label, confidence = "degraded", 0.71
    else:
        gap = bbox_gap_xy(subject, target)
        if gap is not None and gap <= 0.08 and dz <= thresholds.edge_height_tolerance_m:
            label, confidence = "degraded", 0.66
        else:
            label, confidence = "fail", 0.86

    clearance_note = "fits clearance" if fits_clearance else "exceeds clearance"
    reason = (
        f"unified support score selected {evidence.surface_kind} `{evidence.surface_id}` from {evidence.source}: "
        f"overlap {overlap:.2f}, height delta {dz:.2f}m, {clearance_note}, score {evidence.score:.2f}."
    )
    return SupportAssessment(
        label=label, confidence=confidence, reason=reason, evidence=evidence
    )


def _small_placeable_interaction_top_match(
    subject_profile: ObjectFunctionProfile,
    evidence: SupportEvidence,
) -> bool:
    if evidence.source != "functional_hints":
        return False
    if evidence.surface_kind != "top_surface":
        return False
    if not subject_profile.is_small_placeable:
        return False
    return evidence.overlap_ratio >= 0.30 and evidence.height_delta_m <= 0.08


def _rigid_top_strong_overlap_match(
    subject_profile: ObjectFunctionProfile,
    target: dict[str, Any],
    evidence: SupportEvidence,
) -> bool:
    if evidence.surface_kind != "top_surface":
        return False
    if evidence.source not in {"bbox_profile", "functional_hints"}:
        return False
    if not subject_profile.is_small_placeable:
        return False
    target_profile = object_function_profile(target)
    if not (
        target_profile.can_support_top
        or target_profile.is_work_surface
        or target_profile.is_bedside_surface
    ):
        return False
    return evidence.overlap_ratio >= 0.85 and evidence.height_delta_m <= 0.30


def _surface_evidence(
    subject_poly: list[tuple[float, float]],
    subject_area: float,
    subject_bottom: float,
    subject_height: float,
    candidate: SupportSurfaceCandidate,
    *,
    thresholds: SupportThresholds,
) -> SupportEvidence:
    overlap = (
        _convex_overlap_area(subject_poly, list(candidate.polygon_xy)) / subject_area
    )
    dz = abs(subject_bottom - candidate.height_z)
    clearance_penalty = max(
        subject_height - candidate.clearance_m - thresholds.clearance_slack_m, 0.0
    )
    height_penalty = min(dz, _surface_level_tolerance_candidate(candidate, thresholds))
    score = overlap - height_penalty * 1.5 - clearance_penalty * 0.75
    return SupportEvidence(
        surface_id=candidate.surface_id,
        surface_kind=candidate.kind,
        source=candidate.source,
        overlap_ratio=overlap,
        height_delta_m=dz,
        clearance_m=candidate.clearance_m,
        subject_height_m=subject_height,
        score=score,
    )


def _surface_level_tolerance_candidate(
    candidate: SupportSurfaceCandidate, thresholds: SupportThresholds
) -> float:
    if candidate.source == "functional_hints":
        return thresholds.coarse_top_height_tolerance_m
    return thresholds.degraded_height_tolerance_m


def _region_polygon(region: dict[str, Any]) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for point in region.get("polygon_world_xy") or []:
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            points.append((float(point[0]), float(point[1])))
    return points


def _bbox_top_z(bbox: dict[str, Any]) -> float | None:
    tmax = bbox.get("max") or []
    if len(tmax) >= 3:
        return float(tmax[2])
    center = bbox.get("center") or []
    size = bbox.get("size") or []
    if len(center) >= 3 and len(size) >= 3:
        return float(center[2]) + float(size[2]) / 2.0
    return None


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _polygon_area_xy(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    return (
        abs(
            sum(
                x0 * points[(index + 1) % len(points)][1]
                - points[(index + 1) % len(points)][0] * y0
                for index, (x0, y0) in enumerate(points)
            )
        )
        * 0.5
    )


def _convex_overlap_area(
    subject_poly: list[tuple[float, float]], region_poly: list[tuple[float, float]]
) -> float:
    if len(subject_poly) < 3 or len(region_poly) < 3:
        return 0.0
    clipper = (
        region_poly if _signed_area(region_poly) >= 0 else list(reversed(region_poly))
    )
    output = (
        subject_poly
        if _signed_area(subject_poly) >= 0
        else list(reversed(subject_poly))
    )
    for index, edge_start in enumerate(clipper):
        edge_end = clipper[(index + 1) % len(clipper)]
        input_points = output
        output = []
        if not input_points:
            break
        previous = input_points[-1]
        for current in input_points:
            current_inside = _left(current, edge_start, edge_end) >= -1e-9
            previous_inside = _left(previous, edge_start, edge_end) >= -1e-9
            if current_inside:
                if not previous_inside:
                    output.append(
                        _intersection(previous, current, edge_start, edge_end)
                    )
                output.append(current)
            elif previous_inside:
                output.append(_intersection(previous, current, edge_start, edge_end))
            previous = current
    return _polygon_area_xy(output)


def _signed_area(points: list[tuple[float, float]]) -> float:
    return (
        sum(
            x0 * points[(index + 1) % len(points)][1]
            - points[(index + 1) % len(points)][0] * y0
            for index, (x0, y0) in enumerate(points)
        )
        * 0.5
    )


def _left(
    point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]
) -> float:
    return (end[0] - start[0]) * (point[1] - start[1]) - (end[1] - start[1]) * (
        point[0] - start[0]
    )


def _intersection(
    p0: tuple[float, float],
    p1: tuple[float, float],
    e0: tuple[float, float],
    e1: tuple[float, float],
) -> tuple[float, float]:
    x1, y1 = p0
    x2, y2 = p1
    x3, y3 = e0
    x4, y4 = e1
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-9:
        return p1
    return (
        ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom,
        ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom,
    )
