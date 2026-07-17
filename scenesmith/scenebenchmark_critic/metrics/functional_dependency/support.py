from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from scenesmith.scenebenchmark_critic.core.geometry import (
    GeometryStore,
    bbox_gap_xy,
    is_small_object,
    object_category,
    object_footprint_polygon,
)
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.constants import *
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.profiles import (
    object_function_profile,
)
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.semantics import (
    _category_group,
    _category_surface_family_match,
    _category_token_has_any,
    _has_support_storage_semantics,
    _is_any_lamp_object,
    _is_supported_small_subject,
    _is_upright_reading_material,
    _scene_object_type,
    _support_modes,
    _token_text_has_any,
)
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.support_scoring import (
    SupportAssessment,
    assess_direct_support,
    support_assessment_diagnostics,
)


@dataclass(frozen=True)
class SupportRelationResult:
    label: str
    confidence: float
    reason: str
    evaluation_path: str
    evidence: dict[str, Any] = field(default_factory=dict)


def _eval_object_on_support(
    subject: dict[str, Any],
    target: dict[str, Any],
    _relation_type: str,
    *,
    store: GeometryStore | None = None,
) -> tuple[str, float, str]:
    result = evaluate_support_relation(subject, target, _relation_type, store=store)
    return result.label, result.confidence, result.reason


def evaluate_support_relation(
    subject: dict[str, Any],
    target: dict[str, Any],
    _relation_type: str,
    *,
    store: GeometryStore | None = None,
) -> SupportRelationResult:
    sb = subject.get("bbox_world") or {}
    tb = target.get("bbox_world") or {}
    smin = sb.get("min") or []
    smax = sb.get("max") or []
    tcenter = tb.get("center") or []
    tmin = tb.get("min") or []
    tmax = tb.get("max") or []
    tsize = tb.get("size") or []
    if (
        len(smin) < 3
        or len(smax) < 3
        or len(tmin) < 3
        or len(tmax) < 3
        or len(tcenter) < 3
        or len(tsize) < 3
    ):
        gap = bbox_gap_xy(subject, target)
        if gap is not None and gap <= 0.35:
            return _support_result(
                "degraded",
                0.65,
                f"objects are close ({gap:.2f}m), but support height could not be verified.",
                "bbox_fallback",
            )
        return _support_result(
            "unknown", 0.0, "missing support geometry.", "bbox_fallback"
        )
    overlap_ratio = _footprint_overlap_ratio_xy(subject, target, inflate=0.05)
    support_top = _support_top_z(tb)
    dz = abs(float(smin[2]) - support_top)
    gap = bbox_gap_xy(subject, target)
    support_modes = _support_modes(target)
    stack_result = _eval_manipuland_stack_support_result(subject, target, store=store)
    if stack_result is not None:
        return stack_result
    indirect_result = _eval_indirect_support_via_intermediary_result(
        subject, target, store=store
    )
    if indirect_result is not None and indirect_result.label == "pass":
        return indirect_result
    pending_indirect_result = (
        indirect_result
        if indirect_result is not None and indirect_result.label == "degraded"
        else None
    )

    direct_result = _direct_support_relation_result(subject, target)

    region_result = _tuple_support_result(
        _eval_object_on_support_regions(subject, target, store=store),
        subject=subject,
        target=target,
        default_path="support_region",
    )
    if region_result is not None:
        if _prefer_direct_support_result(direct_result, region_result):
            return direct_result
        if (
            region_result.label in {"fail", "unknown"}
            and pending_indirect_result is not None
        ):
            return pending_indirect_result
        return region_result
    thin_support_result = _tuple_support_result(
        _thin_support_top_fallback(subject, target, subject_bottom=float(smin[2])),
        subject=subject,
        target=target,
        default_path="thin_edge",
    )
    if thin_support_result is not None:
        return thin_support_result
    if (
        not _valid_support_regions(target)
        and direct_result is not None
        and direct_result.label in {"pass", "degraded"}
    ):
        return direct_result
    if overlap_ratio >= 0.55 and dz <= 0.12:
        return _support_result(
            "pass",
            0.9,
            f"subject overlaps the support top well (ratio {overlap_ratio:.2f}) with height delta {dz:.2f}m.",
            "bbox_fallback",
            _bbox_support_evidence(
                subject, target, overlap_ratio=overlap_ratio, height_delta=dz
            ),
        )
    if overlap_ratio >= 0.30 and dz <= 0.18:
        return _support_result(
            "pass",
            0.86,
            f"subject is plausibly supported near the surface edge (ratio {overlap_ratio:.2f}, dz {dz:.2f}m).",
            "bbox_fallback",
            _bbox_support_evidence(
                subject, target, overlap_ratio=overlap_ratio, height_delta=dz
            ),
        )
    if overlap_ratio >= 0.85 and dz <= 0.24:
        return _support_result(
            "pass",
            0.86,
            f"subject strongly overlaps the support and the height delta is still plausible: ratio {overlap_ratio:.2f}, dz {dz:.2f}m.",
            "bbox_fallback",
            _bbox_support_evidence(
                subject, target, overlap_ratio=overlap_ratio, height_delta=dz
            ),
        )
    multilevel_shelf_result = _tuple_support_result(
        _multilevel_shelf_support_fallback(
            subject, target, overlap_ratio=overlap_ratio
        ),
        subject=subject,
        target=target,
        default_path="bbox_fallback",
    )
    if multilevel_shelf_result is not None:
        return multilevel_shelf_result
    coarse_surface_result = _tuple_support_result(
        _coarse_work_surface_top_fallback(
            subject, target, overlap_ratio=overlap_ratio, dz=dz
        ),
        subject=subject,
        target=target,
        default_path="bbox_fallback",
    )
    if coarse_surface_result is not None:
        return coarse_surface_result
    edge_fallback = _tuple_support_result(
        _semantic_support_edge_fallback(
            subject, target, float(smin[2]), overlap_ratio=overlap_ratio
        ),
        subject=subject,
        target=target,
        default_path="bbox_fallback",
    )
    if edge_fallback is not None:
        return edge_fallback
    low_table_result = _tuple_support_result(
        _low_table_lower_shelf_support_fallback(
            subject, target, overlap_ratio=overlap_ratio
        ),
        subject=subject,
        target=target,
        default_path="bbox_fallback",
    )
    if low_table_result is not None:
        return low_table_result
    if pending_indirect_result is not None:
        return pending_indirect_result
    if not _valid_support_regions(target) and _plausible_internal_support(
        subject,
        target,
        overlap_ratio=overlap_ratio,
        support_top=support_top,
        support_modes=support_modes,
    ):
        mode_text = (
            "internal storage"
            if "internal_storage" in support_modes
            else "shelf support"
        )
        return _support_result(
            "unknown",
            0.0,
            f"subject is strongly contained within the target footprint, and `{target.get('id')}` exposes plausible {mode_text}, "
            "but coarse geometry cannot verify the internal shelf height.",
            "bbox_fallback",
        )
    if overlap_ratio >= 0.85 and dz <= 0.35:
        return _support_result(
            "degraded",
            0.7,
            f"support relation is vertically approximate despite strong overlap: ratio {overlap_ratio:.2f}, height delta {dz:.2f}m.",
            "bbox_fallback",
            _bbox_support_evidence(
                subject, target, overlap_ratio=overlap_ratio, height_delta=dz
            ),
        )
    if overlap_ratio >= 0.15 and dz <= 0.28:
        return _support_result(
            "degraded",
            0.7,
            f"support relation is approximate: overlap ratio {overlap_ratio:.2f}, height delta {dz:.2f}m.",
            "bbox_fallback",
            _bbox_support_evidence(
                subject, target, overlap_ratio=overlap_ratio, height_delta=dz
            ),
        )
    if gap is not None and gap <= 0.08 and dz <= 0.2:
        return _support_result(
            "degraded",
            0.66,
            f"subject is very near the support but overlap is weak: gap {gap:.2f}m, dz {dz:.2f}m.",
            "bbox_fallback",
            _bbox_support_evidence(
                subject, target, overlap_ratio=overlap_ratio, height_delta=dz
            ),
        )
    return _support_result(
        "fail",
        0.85,
        f"subject is not on or near the support surface: overlap ratio {overlap_ratio:.2f}, height delta {dz:.2f}m.",
        "bbox_fallback",
        _bbox_support_evidence(
            subject, target, overlap_ratio=overlap_ratio, height_delta=dz
        ),
    )


def _support_result(
    label: str,
    confidence: float,
    reason: str,
    evaluation_path: str,
    evidence: dict[str, Any] | None = None,
) -> SupportRelationResult:
    payload = dict(evidence or {})
    payload["support_evaluation_path"] = evaluation_path
    return SupportRelationResult(label, confidence, reason, evaluation_path, payload)


def _tuple_support_result(
    result: tuple[str, float, str] | None,
    *,
    subject: dict[str, Any],
    target: dict[str, Any],
    default_path: str,
) -> SupportRelationResult | None:
    if result is None:
        return None
    label, confidence, reason = result
    path = _infer_support_path_from_reason(reason, default_path)
    evidence = _support_fallback_diagnostics(subject, target, path)
    return _support_result(label, confidence, reason, path, evidence)


def _direct_support_relation_result(
    subject: dict[str, Any], target: dict[str, Any]
) -> SupportRelationResult | None:
    assessment = assess_direct_support(subject, target)
    if assessment is None:
        return None
    path = _support_path_from_assessment(assessment)
    reason = _direct_support_reason(assessment, path)
    return _support_result(
        assessment.label,
        assessment.confidence,
        reason,
        path,
        support_assessment_diagnostics(assessment),
    )


def _support_path_from_assessment(assessment: SupportAssessment) -> str:
    evidence = assessment.evidence
    if evidence is None:
        return "direct"
    if evidence.source == "support_region":
        return "support_region"
    if evidence.source == "bbox_profile":
        return "bbox_fallback"
    return "direct"


def _direct_support_reason(assessment: SupportAssessment, path: str) -> str:
    evidence = assessment.evidence
    if evidence is None:
        return assessment.reason
    if path == "bbox_fallback":
        if evidence.overlap_ratio >= 0.85 and evidence.height_delta_m <= 0.30:
            return (
                "unified support score selected strong bbox fallback "
                f"`{evidence.surface_id}`: overlap {evidence.overlap_ratio:.2f}, "
                f"height delta {evidence.height_delta_m:.2f}m; coarse bbox underestimates tabletop height."
            )
        return (
            "unified support score selected bbox top fallback "
            f"`{evidence.surface_id}`: overlap {evidence.overlap_ratio:.2f}, "
            f"height delta {evidence.height_delta_m:.2f}m."
        )
    if evidence.source == "functional_hints":
        return (
            "unified support score selected interaction_top from functional_hints "
            f"`{evidence.surface_id}`: overlap {evidence.overlap_ratio:.2f}, "
            f"height delta {evidence.height_delta_m:.2f}m."
        )
    return assessment.reason


def _direct_support_can_override_regions(result: SupportRelationResult | None) -> bool:
    if result is None or result.label != "pass":
        return False
    if result.evaluation_path not in {"direct", "bbox_fallback"}:
        return False
    overlap = _float_evidence(result.evidence, "support_overlap_ratio", 0.0)
    dz = _float_evidence(result.evidence, "support_height_delta_m", 999.0)
    source = str(result.evidence.get("support_surface_source") or "")
    if source == "functional_hints":
        return (overlap >= 0.30 and dz <= 0.08) or (overlap >= 0.85 and dz <= 0.30)
    if source == "bbox_profile":
        return overlap >= 0.85 and dz <= 0.30
    return False


def _float_evidence(evidence: dict[str, Any], key: str, default: float) -> float:
    value = evidence.get(key)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _prefer_direct_support_result(
    direct: SupportRelationResult | None,
    current: SupportRelationResult,
) -> bool:
    if not _direct_support_can_override_regions(direct):
        return False
    if current.label != "pass":
        return True
    return direct.confidence > current.confidence + 0.03


def _infer_support_path_from_reason(reason: str, default_path: str) -> str:
    text = reason.lower()
    if "target rescue" in text:
        return "target_rescue"
    if "manipuland stack chain" in text:
        return "stack"
    if "indirect" in text or "intermediary" in text:
        return "indirect"
    if "thin shelf" in text or "thin edge" in text or "front-edge" in text:
        return "thin_edge"
    if (
        "bbox" in text
        or "coarse" in text
        or "multilevel shelf support" in text
        or "lower-shelf support" in text
    ):
        return "bbox_fallback"
    if "support region" in text or "matched " in text:
        return "support_region"
    return default_path


def _support_fallback_diagnostics(
    subject: dict[str, Any], target: dict[str, Any], path: str
) -> dict[str, Any]:
    if path == "thin_edge":
        thin_overlap = _thin_support_overlap_ratio_xy(subject, target)
        wide_overlap = _footprint_overlap_ratio_xy(subject, target, inflate=0.24)
        dz = _subject_to_bbox_top_delta(subject, target)
        evidence = _bbox_support_evidence(
            subject,
            target,
            overlap_ratio=thin_overlap if thin_overlap is not None else wide_overlap,
            height_delta=dz,
            source="thin_edge_bbox",
        )
        evidence["support_wide_overlap_ratio"] = round(wide_overlap, 4)
        return evidence
    if path == "bbox_fallback":
        overlap = _footprint_overlap_ratio_xy(subject, target, inflate=0.05)
        thin_overlap = _thin_support_overlap_ratio_xy(subject, target)
        if thin_overlap is not None:
            overlap = max(overlap, thin_overlap)
        return _bbox_support_evidence(
            subject,
            target,
            overlap_ratio=overlap,
            height_delta=_subject_to_bbox_top_delta(subject, target),
        )
    assessment = assess_direct_support(subject, target)
    return support_assessment_diagnostics(assessment)


def _bbox_support_evidence(
    subject: dict[str, Any],
    target: dict[str, Any],
    *,
    overlap_ratio: float,
    height_delta: float,
    source: str = "bbox_profile",
) -> dict[str, Any]:
    return {
        "support_surface_id": "bbox_top",
        "support_surface_kind": "top_surface",
        "support_surface_source": source,
        "support_overlap_ratio": round(overlap_ratio, 4),
        "support_height_delta_m": round(height_delta, 4),
        "support_clearance_m": 10.0,
        "support_evidence_score": round(overlap_ratio - height_delta * 1.5, 4),
    }


def _subject_to_bbox_top_delta(
    subject: dict[str, Any], target: dict[str, Any]
) -> float:
    sb = subject.get("bbox_world") or {}
    smin = sb.get("min") or []
    if len(smin) < 3:
        return 999.0
    return abs(float(smin[2]) - _support_top_z(target.get("bbox_world") or {}))


def _eval_object_on_support_regions(
    subject: dict[str, Any],
    target: dict[str, Any],
    *,
    store: GeometryStore | None = None,
) -> tuple[str, float, str] | None:
    regions = _valid_support_regions(target)
    if not regions:
        return None
    sb = subject.get("bbox_world") or {}
    smin = sb.get("min") or []
    smax = sb.get("max") or []
    if len(smin) < 3 or len(smax) < 3:
        return (
            "unknown",
            0.0,
            "support regions are available, but subject height geometry is missing.",
        )

    subject_poly = object_footprint_polygon(subject)
    subject_area = _polygon_area_xy(subject_poly or [])
    if not subject_poly or subject_area <= 1e-6:
        return (
            "unknown",
            0.0,
            "support regions are available, but subject footprint geometry is missing.",
        )

    subject_bottom = float(smin[2])
    subject_height = max(float(smax[2]) - float(smin[2]), 0.0)
    best: tuple[float, float, dict[str, Any], float] | None = None
    for region in regions:
        region_poly = _region_world_polygon(region)
        region_area = _polygon_area_xy(region_poly)
        if region_area <= 1e-6:
            continue
        overlap_area = _convex_overlap_area(subject_poly, region_poly)
        overlap_ratio = overlap_area / subject_area
        height = _region_height_world(region)
        if height is None:
            continue
        dz = abs(subject_bottom - height)
        rank = overlap_ratio - dz * 1.5
        if best is None or rank > best[0]:
            best = (rank, overlap_ratio, region, dz)

    if best is None:
        bbox_fallback = _bbox_top_support_fallback(
            subject, target, subject_bottom, None
        )
        if bbox_fallback is not None:
            return bbox_fallback
        if not _is_horizontally_plausible_support_candidate(subject, target):
            gap = bbox_gap_xy(subject, target)
            gap_text = "unknown" if gap is None else f"{gap:.2f}m"
            return (
                "fail",
                0.85,
                "support regions are degenerate, and the subject is not horizontally near the target "
                f"(bbox gap {gap_text}).",
            )
        return (
            "unknown",
            0.0,
            "support regions are available, but no valid region geometry could be evaluated.",
        )

    _rank, overlap_ratio, region, dz = best
    clearance = _region_clearance(region)
    region_id = str(region.get("region_id") or "support_region")
    kind = str(region.get("support_kind") or "support_region")
    if (
        _is_upright_reading_material(subject)
        and _is_internal_support_region(region)
        and overlap_ratio >= 0.55
        and dz <= 0.14
    ):
        return (
            "pass",
            0.88,
            f"matched {kind} `{region_id}` for upright reading material with overlap {overlap_ratio:.2f} "
            f"and height delta {dz:.2f}m; ignoring noisy internal shelf clearance {clearance:.2f}m.",
        )
    if overlap_ratio >= 0.55 and dz <= 0.14 and subject_height <= clearance + 0.08:
        return (
            "pass",
            0.9,
            f"matched {kind} `{region_id}` with overlap {overlap_ratio:.2f}, height delta {dz:.2f}m, and clearance {clearance:.2f}m.",
        )
    reading_edge_result = _reading_material_shelf_region_fallback(
        subject,
        target,
        region=region,
        kind=kind,
        region_id=region_id,
        overlap_ratio=overlap_ratio,
        dz=dz,
        clearance=clearance,
        subject_height=subject_height,
    )
    if reading_edge_result is not None:
        return reading_edge_result
    small_upright_result = _small_upright_top_surface_region_fallback(
        subject,
        target,
        kind=kind,
        region_id=region_id,
        overlap_ratio=overlap_ratio,
        dz=dz,
        clearance=clearance,
        subject_height=subject_height,
    )
    if small_upright_result is not None:
        return small_upright_result
    weight_rack_result = _weight_plate_rack_region_fallback(
        subject,
        target,
        kind=kind,
        region_id=region_id,
        overlap_ratio=overlap_ratio,
        dz=dz,
        clearance=clearance,
        subject_height=subject_height,
    )
    if weight_rack_result is not None:
        return weight_rack_result
    if overlap_ratio >= 0.35 and dz <= 0.18 and subject_height <= clearance + 0.14:
        return (
            "degraded",
            0.72,
            f"partially matched {kind} `{region_id}` near the support edge: overlap {overlap_ratio:.2f}, height delta {dz:.2f}m, clearance {clearance:.2f}m.",
        )
    if overlap_ratio >= 0.55 and dz <= 0.14:
        return (
            "degraded",
            0.7,
            f"matched {kind} `{region_id}`, but the subject nearly exceeds available clearance: overlap {overlap_ratio:.2f}, clearance {clearance:.2f}m.",
        )
    truncated_shelf_result = _truncated_multilevel_region_support_fallback(
        subject,
        target,
        overlap_ratio=overlap_ratio,
        subject_bottom=subject_bottom,
    )
    if truncated_shelf_result is not None:
        return truncated_shelf_result
    if kind == "top_surface":
        bbox_fallback = _bbox_top_support_fallback(
            subject, target, subject_bottom, overlap_ratio, region_id=region_id
        )
        if bbox_fallback is not None:
            return bbox_fallback
    stack_result = _eval_manipuland_stack_support(subject, target, store=store)
    if stack_result is not None:
        return stack_result
    indirect_result = _eval_indirect_support_via_intermediary(
        subject, target, store=store
    )
    if indirect_result is not None:
        return indirect_result
    edge_fallback = _semantic_support_edge_fallback(
        subject, target, subject_bottom, overlap_ratio=overlap_ratio
    )
    if edge_fallback is not None:
        return edge_fallback
    soft_surface_result = _soft_sleeping_surface_support_fallback(
        subject,
        target,
        overlap_ratio=overlap_ratio,
        subject_bottom=subject_bottom,
    )
    if soft_surface_result is not None:
        return soft_surface_result
    return (
        "fail",
        0.85,
        f"subject does not match any support region on `{target.get('id')}`: best overlap {overlap_ratio:.2f}, height delta {dz:.2f}m.",
    )


def _reading_material_shelf_region_fallback(
    subject: dict[str, Any],
    target: dict[str, Any],
    *,
    region: dict[str, Any],
    kind: str,
    region_id: str,
    overlap_ratio: float,
    dz: float,
    clearance: float,
    subject_height: float,
) -> tuple[str, float, str] | None:
    if not _is_multilevel_shelf_like_target(target):
        return None
    if not _token_text_has_any(subject, STACKABLE_SUPPORT_TEXT_HINTS):
        return None
    if subject_height > clearance + 0.16:
        return None
    if kind == "top_surface" and overlap_ratio >= 0.45 and dz <= 0.18:
        return (
            "pass",
            0.82,
            f"reading material is plausibly supported on the bookshelf top edge via `{region_id}`: "
            f"overlap {overlap_ratio:.2f}, height delta {dz:.2f}m, clearance {clearance:.2f}m.",
        )
    if _is_internal_support_region(region) and overlap_ratio >= 0.50 and dz <= 0.14:
        return (
            "pass",
            0.82,
            f"reading material partially matches internal shelf `{region_id}` near the shelf edge: "
            f"overlap {overlap_ratio:.2f}, height delta {dz:.2f}m, clearance {clearance:.2f}m.",
        )
    return None


def _small_upright_top_surface_region_fallback(
    subject: dict[str, Any],
    target: dict[str, Any],
    *,
    kind: str,
    region_id: str,
    overlap_ratio: float,
    dz: float,
    clearance: float,
    subject_height: float,
) -> tuple[str, float, str] | None:
    if kind != "top_surface":
        return None
    if not _is_bbox_edge_fallback_target(target):
        return None
    if not _is_small_upright_surface_object(subject):
        return None
    if overlap_ratio < 0.85 or dz > 0.18:
        return None
    if subject_height > clearance + 0.10:
        return None
    return (
        "pass",
        0.84,
        f"matched {kind} `{region_id}` for a small upright object with strong top overlap despite conservative height extraction: "
        f"overlap {overlap_ratio:.2f}, height delta {dz:.2f}m, clearance {clearance:.2f}m.",
    )


def _weight_plate_rack_region_fallback(
    subject: dict[str, Any],
    target: dict[str, Any],
    *,
    kind: str,
    region_id: str,
    overlap_ratio: float,
    dz: float,
    clearance: float,
    subject_height: float,
) -> tuple[str, float, str] | None:
    if not _is_weight_plate_subject(subject):
        return None
    if not _is_weight_storage_rack_target(target):
        return None
    if kind not in {"top_surface", "internal_shelf"}:
        return None
    if subject_height > clearance + 0.14:
        return None
    if overlap_ratio < 0.35 or dz > 0.18:
        return None
    if _footprint_overlap_ratio_xy(subject, target, inflate=0.08) < 0.35:
        return None
    return (
        "pass",
        0.84,
        f"matched weight-storage rack {kind} `{region_id}` for a weight plate near the support bar: "
        f"overlap {overlap_ratio:.2f}, height delta {dz:.2f}m, clearance {clearance:.2f}m.",
    )


def _truncated_multilevel_region_support_fallback(
    subject: dict[str, Any],
    target: dict[str, Any],
    *,
    overlap_ratio: float,
    subject_bottom: float,
) -> tuple[str, float, str] | None:
    if not _is_multilevel_shelf_like_target(target):
        return None
    if not _is_supported_small_subject(subject):
        return None

    sb = subject.get("bbox_world") or {}
    tb = target.get("bbox_world") or {}
    smax = sb.get("max") or []
    tmin = tb.get("min") or []
    tmax = tb.get("max") or []
    if len(smax) < 3 or len(tmin) < 3 or len(tmax) < 3:
        return None

    target_floor = float(tmin[2])
    target_top = float(tmax[2])
    subject_top = float(smax[2])
    if subject_bottom <= target_top + 0.08:
        return None
    if subject_top > target_floor + 2.4:
        return None

    inflated_overlap = _footprint_overlap_ratio_xy(subject, target, inflate=0.12)
    effective_overlap = max(overlap_ratio, inflated_overlap)
    if not _shelf_footprint_fallback_subject_allowed(subject, target):
        return None
    if effective_overlap >= 0.70:
        return (
            "pass",
            0.82,
            "subject sits within a multilevel shelf/bookcase footprint above the extracted low support regions; "
            f"accepting truncated shelf-region support with overlap {effective_overlap:.2f}.",
        )
    if effective_overlap >= 0.55:
        return (
            "degraded",
            0.68,
            "subject is near a multilevel shelf/bookcase footprint above low extracted support regions, "
            f"but support level is approximate: overlap {effective_overlap:.2f}.",
        )
    return None


def _bbox_top_support_fallback(
    subject: dict[str, Any],
    target: dict[str, Any],
    subject_bottom: float,
    region_overlap: float | None,
    *,
    region_id: str | None = None,
) -> tuple[str, float, str] | None:
    bbox_top = _support_top_z(target.get("bbox_world") or {})
    bbox_dz = abs(subject_bottom - bbox_top)
    bbox_overlap = _footprint_overlap_ratio_xy(subject, target, inflate=0.05)
    region_text = (
        "degenerate region geometry"
        if region_overlap is None
        else f"region overlap {region_overlap:.2f}"
    )
    thin_overlap = _thin_support_overlap_ratio_xy(subject, target)
    wide_overlap = 0.0
    gap: float | None = None
    if thin_overlap is not None:
        bbox_overlap = max(bbox_overlap, thin_overlap)
        wide_overlap = _footprint_overlap_ratio_xy(subject, target, inflate=0.24)
        gap = bbox_gap_xy(subject, target)
    id_text = f" `{region_id}`" if region_id else ""
    if bbox_overlap >= 0.30 and bbox_dz <= 0.18:
        return (
            "pass",
            0.86,
            f"matched conservative top region{id_text} via bbox top fallback: {region_text}, "
            f"bbox overlap {bbox_overlap:.2f}, height delta {bbox_dz:.2f}m.",
        )
    if _is_small_bedside_edge_surface_match(
        subject, target, bbox_overlap=bbox_overlap, height_delta=bbox_dz
    ):
        return (
            "pass",
            0.82,
            f"support region is conservative for a small bedside object, but strong bbox fallback top/edge support is plausible{id_text}: "
            f"{region_text}, bbox overlap {bbox_overlap:.2f}, height delta {bbox_dz:.2f}m.",
        )
    if _is_strong_bbox_top_surface_match(
        subject, target, bbox_overlap=bbox_overlap, height_delta=bbox_dz
    ):
        return (
            "pass",
            0.84,
            f"matched top support via strong bbox fallback{id_text}: {region_text}, "
            f"bbox overlap {bbox_overlap:.2f}, height delta {bbox_dz:.2f}m.",
        )
    if (
        thin_overlap is not None
        and gap is not None
        and gap <= 0.25
        and wide_overlap >= 0.45
        and bbox_dz <= 0.08
    ):
        return (
            "pass",
            0.84,
            f"thin shelf support is plausible at the edge via bbox fallback{id_text}: {region_text}, "
            f"gap {gap:.2f}m, wide bbox overlap {wide_overlap:.2f}, height delta {bbox_dz:.2f}m.",
        )
    if _is_tiny_bedside_clock_edge_match(
        subject, target, bbox_overlap=bbox_overlap, height_delta=bbox_dz
    ):
        return (
            "pass",
            0.8,
            f"matched conservative top region{id_text} for a tiny bedside clock near the furniture edge: {region_text}, "
            f"bbox overlap {bbox_overlap:.2f}, height delta {bbox_dz:.2f}m.",
        )
    if bbox_overlap >= 0.15 and bbox_dz <= 0.28:
        return (
            "degraded",
            0.7,
            f"top support is approximate via bbox fallback{id_text}: {region_text}, "
            f"bbox overlap {bbox_overlap:.2f}, height delta {bbox_dz:.2f}m.",
        )
    if (
        thin_overlap is not None
        and gap is not None
        and gap <= 0.08
        and wide_overlap >= 0.35
        and bbox_dz <= 0.18
    ):
        return (
            "degraded",
            0.68,
            f"thin shelf support is approximate near the front corner: {region_text}, "
            f"gap {gap:.2f}m, wide bbox overlap {wide_overlap:.2f}, height delta {bbox_dz:.2f}m.",
        )
    return None


def _is_strong_bbox_top_surface_match(
    subject: dict[str, Any],
    target: dict[str, Any],
    *,
    bbox_overlap: float,
    height_delta: float,
) -> bool:
    if bbox_overlap < 0.85 or height_delta > 0.30:
        return False
    if not _is_bbox_edge_fallback_target(target):
        return False
    category = object_category(target)
    if category in {"counter", "island"} and bbox_overlap < 0.95:
        return False
    return _is_supported_small_subject(subject)


def _is_small_bedside_edge_surface_match(
    subject: dict[str, Any],
    target: dict[str, Any],
    *,
    bbox_overlap: float,
    height_delta: float,
) -> bool:
    if bbox_overlap < 0.95 or height_delta > 0.32:
        return False
    if object_category(target) not in NIGHTSTANDS | {"end_table", "side_table"}:
        return False
    bbox = subject.get("bbox_world") or {}
    size = bbox.get("size") or []
    if len(size) < 3:
        return False
    width = max(float(size[0]), 0.0)
    depth = max(float(size[1]), 0.0)
    height = max(float(size[2]), 0.0)
    return width <= 0.18 and depth <= 0.18 and width * depth <= 0.025 and height <= 0.20


def _is_tiny_bedside_clock_edge_match(
    subject: dict[str, Any],
    target: dict[str, Any],
    *,
    bbox_overlap: float,
    height_delta: float,
) -> bool:
    if bbox_overlap < 0.20 or height_delta > 0.32:
        return False
    if object_category(target) not in NIGHTSTANDS | {"end_table", "side_table"}:
        return False
    if not _token_text_has_any(subject, ("clock", "alarm", "alarm_clock")):
        return False
    bbox = subject.get("bbox_world") or {}
    size = bbox.get("size") or []
    center = bbox.get("center") or []
    target_bbox = target.get("bbox_world") or {}
    tmin = target_bbox.get("min") or []
    tmax = target_bbox.get("max") or []
    if len(size) < 3 or len(center) < 2 or len(tmin) < 2 or len(tmax) < 2:
        return False
    width = max(float(size[0]), 0.0)
    depth = max(float(size[1]), 0.0)
    height = max(float(size[2]), 0.0)
    if width > 0.16 or depth > 0.16 or width * depth > 0.02 or height > 0.10:
        return False
    cx, cy = float(center[0]), float(center[1])
    return (
        float(tmin[0]) - 0.12 <= cx <= float(tmax[0]) + 0.12
        and float(tmin[1]) - 0.12 <= cy <= float(tmax[1]) + 0.12
    )


def _is_small_upright_surface_object(subject: dict[str, Any]) -> bool:
    category = object_category(subject)
    allowed_tokens = {"cup", "mug", "glass", "tumbler", "vase", "bottle", "carafe"}
    if category not in allowed_tokens and not _category_token_has_any(
        subject, tuple(sorted(allowed_tokens))
    ):
        return False
    bbox = subject.get("bbox_world") or {}
    size = bbox.get("size") or []
    if len(size) < 3:
        return False
    width = max(float(size[0]), 0.0)
    depth = max(float(size[1]), 0.0)
    height = max(float(size[2]), 0.0)
    footprint = width * depth
    return width <= 0.22 and depth <= 0.22 and footprint <= 0.04 and height <= 0.45


def _coarse_work_surface_top_fallback(
    subject: dict[str, Any],
    target: dict[str, Any],
    *,
    overlap_ratio: float,
    dz: float,
) -> tuple[str, float, str] | None:
    if _valid_support_regions(target):
        return None
    category = object_category(target)
    if category not in WORK_SURFACES | NIGHTSTANDS | {
        "end_table",
        "console",
        "counter",
        "island",
        "side_table",
    }:
        return None
    if overlap_ratio >= 0.85 and dz <= 0.32:
        return (
            "pass",
            0.82,
            f"coarse work-surface bbox likely underestimates tabletop height: overlap ratio {overlap_ratio:.2f}, height delta {dz:.2f}m.",
        )
    if overlap_ratio >= 0.55 and dz <= 0.30:
        return (
            "degraded",
            0.68,
            f"coarse work-surface bbox makes tabletop height approximate: overlap ratio {overlap_ratio:.2f}, height delta {dz:.2f}m.",
        )
    return None


def _multilevel_shelf_support_fallback(
    subject: dict[str, Any],
    target: dict[str, Any],
    *,
    overlap_ratio: float,
) -> tuple[str, float, str] | None:
    if _valid_support_regions(target):
        return None
    if not _is_multilevel_shelf_like_target(target):
        return None
    if not _is_supported_small_subject(subject):
        return None

    sb = subject.get("bbox_world") or {}
    tb = target.get("bbox_world") or {}
    smin = sb.get("min") or []
    smax = sb.get("max") or []
    tmin = tb.get("min") or []
    tmax = tb.get("max") or []
    if len(smin) < 3 or len(smax) < 3 or len(tmin) < 3 or len(tmax) < 3:
        return None

    subject_bottom = float(smin[2])
    subject_top = float(smax[2])
    target_floor = float(tmin[2])
    target_top = float(tmax[2])
    if subject_top < target_floor - 0.05:
        return None

    bbox_height = max(target_top - target_floor, 0.0)
    target_bbox_is_shallow = bbox_height <= 0.65
    above_truncated_bbox = subject_bottom > target_top + 0.08
    inside_or_near_bbox_height = subject_bottom <= target_top + 0.30
    plausible_vertical_span = subject_top <= target_floor + 2.4
    if not (
        target_bbox_is_shallow or above_truncated_bbox or inside_or_near_bbox_height
    ):
        return None
    if not plausible_vertical_span:
        return None

    inflated_overlap = _footprint_overlap_ratio_xy(subject, target, inflate=0.12)
    effective_overlap = max(overlap_ratio, inflated_overlap)
    if not _shelf_footprint_fallback_subject_allowed(subject, target):
        return None
    if effective_overlap >= 0.70:
        return (
            "pass",
            0.82,
            "subject is contained by a shelf/bookcase footprint; "
            f"accepting multilevel shelf support despite coarse target bbox height: overlap {effective_overlap:.2f}.",
        )
    if effective_overlap >= 0.55:
        return (
            "degraded",
            0.68,
            "subject is near a shelf/bookcase footprint, but support level is approximate with coarse target bbox height: "
            f"overlap {effective_overlap:.2f}.",
        )
    return None


def _low_table_lower_shelf_support_fallback(
    subject: dict[str, Any],
    target: dict[str, Any],
    *,
    overlap_ratio: float,
) -> tuple[str, float, str] | None:
    if _valid_support_regions(target):
        return None
    if not _is_low_open_table_target(target):
        return None
    if not _token_text_has_any(subject, STACKABLE_SUPPORT_TEXT_HINTS):
        return None

    sb = subject.get("bbox_world") or {}
    tb = target.get("bbox_world") or {}
    smin = sb.get("min") or []
    smax = sb.get("max") or []
    tmin = tb.get("min") or []
    tmax = tb.get("max") or []
    if len(smin) < 3 or len(smax) < 3 or len(tmin) < 3 or len(tmax) < 3:
        return None

    target_height = max(float(tmax[2]) - float(tmin[2]), 0.0)
    subject_bottom = float(smin[2])
    subject_top = float(smax[2])
    if overlap_ratio < 0.85 or target_height < 0.28 or target_height > 0.75:
        return None
    if subject_bottom < float(tmin[2]) + 0.05:
        return None
    if subject_top > float(tmax[2]) - 0.10:
        return None
    if subject_top > float(tmin[2]) + target_height * 0.62:
        return None

    within_x = (
        float(smin[0]) >= float(tmin[0]) - 0.04
        and float(smax[0]) <= float(tmax[0]) + 0.04
    )
    within_y = (
        float(smin[1]) >= float(tmin[1]) - 0.04
        and float(smax[1]) <= float(tmax[1]) + 0.04
    )
    if not (within_x and within_y):
        return None
    return (
        "pass",
        0.81,
        "subject is strongly contained within a low open-table footprint and sits well below the tabletop; "
        "accepting plausible lower-shelf support despite missing support-region geometry.",
    )


def _eval_manipuland_stack_support(
    subject: dict[str, Any],
    target: dict[str, Any],
    *,
    store: GeometryStore | None,
) -> tuple[str, float, str] | None:
    result = _eval_manipuland_stack_support_result(subject, target, store=store)
    if result is None:
        return None
    return result.label, result.confidence, result.reason


def _eval_manipuland_stack_support_result(
    subject: dict[str, Any],
    target: dict[str, Any],
    *,
    store: GeometryStore | None,
) -> SupportRelationResult | None:
    if store is None:
        return None
    if _scene_object_type(subject) != "manipuland":
        return None
    if not _is_valid_manipuland_stack_target(target):
        return None

    chain = _find_manipuland_stack_chain(
        subject,
        target,
        store=store,
        visited={str(subject.get("id") or "")},
        depth=0,
    )
    if chain is None or len(chain) < 2:
        return None
    bottom = chain[-1]
    bottom_result = _eval_stack_bottom_support_result(bottom, target)
    if bottom_result.label not in {"pass", "degraded"}:
        return None
    chain_ids = [str(obj.get("id") or "unknown") for obj in chain]
    target_id = str(target.get("id") or "target")
    evidence = dict(bottom_result.evidence)
    evidence["support_chain_adjacent_height_deltas_m"] = (
        _stack_chain_adjacent_height_deltas(chain)
    )
    evidence.update(
        {
            "support_chain_ids": chain_ids + [target_id],
            "support_bottom_object_id": chain_ids[-1],
            "support_bottom_evaluation_path": bottom_result.evaluation_path,
        }
    )
    return _support_result(
        "pass" if bottom_result.label == "pass" else "degraded",
        min(max(bottom_result.confidence, 0.74), 0.86),
        "manipuland stack chain "
        f"{' -> '.join(f'`{item}`' for item in chain_ids)} -> `{target_id}` is supported; "
        f"bottom `{chain_ids[-1]}` relation: {bottom_result.reason}",
        "stack",
        evidence,
    )


def _find_manipuland_stack_chain(
    subject: dict[str, Any],
    target: dict[str, Any],
    *,
    store: GeometryStore,
    visited: set[str],
    depth: int,
) -> list[dict[str, Any]] | None:
    if depth >= 6:
        return None
    candidates = [
        obj
        for obj in store.objects.values()
        if _is_stack_chain_candidate(subject, obj, target=target, visited=visited)
    ]
    candidates.sort(key=lambda obj: _stack_candidate_rank(subject, obj))
    for candidate in candidates:
        candidate_id = str(candidate.get("id") or "")
        bottom_result = _eval_stack_bottom_support_result(candidate, target)
        if bottom_result.label in {"pass", "degraded"}:
            return [subject, candidate]
        child_chain = _find_manipuland_stack_chain(
            candidate,
            target,
            store=store,
            visited=visited | {candidate_id},
            depth=depth + 1,
        )
        if child_chain is not None:
            return [subject] + child_chain
    return None


def _is_stack_chain_candidate(
    upper: dict[str, Any],
    lower: dict[str, Any],
    *,
    target: dict[str, Any],
    visited: set[str],
) -> bool:
    lower_id = str(lower.get("id") or "")
    if not lower_id or lower_id in visited or lower_id == str(target.get("id") or ""):
        return False
    if _scene_object_type(lower) != "manipuland":
        return False
    if not _manipuland_directly_on(upper, lower):
        return False
    return True


def _manipuland_directly_on(upper: dict[str, Any], lower: dict[str, Any]) -> bool:
    upper_bbox = upper.get("bbox_world") or {}
    lower_bbox = lower.get("bbox_world") or {}
    upper_min = upper_bbox.get("min") or []
    lower_min = lower_bbox.get("min") or []
    lower_max = lower_bbox.get("max") or []
    if len(upper_min) < 3 or len(lower_min) < 3 or len(lower_max) < 3:
        return False
    if float(lower_min[2]) > float(upper_min[2]) + 0.03:
        return False
    vertical_delta = float(upper_min[2]) - float(lower_max[2])
    min_vertical_delta = -0.16 if _is_surface_intermediary(lower) else -0.08
    if vertical_delta < min_vertical_delta or vertical_delta > 0.22:
        return False
    overlap = _footprint_overlap_ratio_xy(upper, lower, inflate=0.04)
    gap = bbox_gap_xy(upper, lower)
    return overlap >= 0.25 or (gap is not None and gap <= 0.06)


def _stack_candidate_rank(
    upper: dict[str, Any], lower: dict[str, Any]
) -> tuple[float, float]:
    upper_min = (upper.get("bbox_world") or {}).get("min") or []
    lower_max = (lower.get("bbox_world") or {}).get("max") or []
    vertical_delta = (
        abs(float(upper_min[2]) - float(lower_max[2]))
        if len(upper_min) >= 3 and len(lower_max) >= 3
        else 999.0
    )
    overlap = _footprint_overlap_ratio_xy(upper, lower, inflate=0.04)
    return vertical_delta, -overlap


def _stack_chain_adjacent_height_deltas(chain: list[dict[str, Any]]) -> list[float]:
    deltas: list[float] = []
    for upper, lower in zip(chain, chain[1:]):
        upper_min = (upper.get("bbox_world") or {}).get("min") or []
        lower_max = (lower.get("bbox_world") or {}).get("max") or []
        if len(upper_min) < 3 or len(lower_max) < 3:
            continue
        deltas.append(round(float(upper_min[2]) - float(lower_max[2]), 4))
    return deltas


def _eval_stack_bottom_support_result(
    subject: dict[str, Any],
    target: dict[str, Any],
) -> SupportRelationResult:
    direct_result = _direct_support_relation_result(subject, target)
    if direct_result is not None and direct_result.label in {"pass", "degraded"}:
        return direct_result

    assessment = assess_direct_support(subject, target)
    if assessment is None or assessment.evidence is None:
        return evaluate_support_relation(
            subject, target, "object_on_support", store=None
        )

    evidence = support_assessment_diagnostics(assessment)
    gap = bbox_gap_xy(subject, target)
    if gap is not None:
        evidence["support_gap_xy_m"] = round(gap, 4)
    support_evidence = assessment.evidence
    if (
        _is_surface_intermediary(subject)
        and support_evidence.surface_kind == "top_surface"
        and gap is not None
        and gap <= 0.16
        and support_evidence.height_delta_m <= 0.12
    ):
        return _support_result(
            "pass",
            0.82,
            "stack bottom support accepts a nearby overhanging secondary surface on the furniture top: "
            f"gap {gap:.2f}m, height delta {support_evidence.height_delta_m:.2f}m.",
            "stack_bottom",
            evidence,
        )
    if (
        _is_surface_intermediary(subject)
        and support_evidence.surface_kind == "top_surface"
        and gap is not None
        and gap <= 0.20
        and support_evidence.height_delta_m <= 0.18
    ):
        return _support_result(
            "degraded",
            0.72,
            "stack bottom support finds a plausible nearby secondary surface aligned to the furniture top, "
            f"but the overhang is approximate: gap {gap:.2f}m, height delta {support_evidence.height_delta_m:.2f}m.",
            "stack_bottom",
            evidence,
        )
    return evaluate_support_relation(subject, target, "object_on_support", store=None)


def _is_valid_manipuland_stack_target(target: dict[str, Any]) -> bool:
    if _scene_object_type(target) not in {"furniture", "wall_mounted"}:
        return False
    return _is_primary_support_target(target)


def _eval_indirect_support_via_intermediary(
    subject: dict[str, Any],
    target: dict[str, Any],
    *,
    store: GeometryStore | None,
) -> tuple[str, float, str] | None:
    result = _eval_indirect_support_via_intermediary_result(
        subject, target, store=store
    )
    if result is None:
        return None
    return result.label, result.confidence, result.reason


def _eval_indirect_support_via_intermediary_result(
    subject: dict[str, Any],
    target: dict[str, Any],
    *,
    store: GeometryStore | None,
) -> SupportRelationResult | None:
    if store is None:
        return None
    subject_bbox = subject.get("bbox_world") or {}
    smin = subject_bbox.get("min") or []
    if len(smin) < 3:
        return None
    subject_bottom = float(smin[2])
    for intermediary in store.objects.values():
        intermediary_id = str(intermediary.get("id") or "")
        if not intermediary_id or intermediary_id in {
            str(subject.get("id") or ""),
            str(target.get("id") or ""),
        }:
            continue
        if not _is_secondary_support_intermediary(subject, intermediary):
            continue
        mid_result = evaluate_support_relation(
            intermediary, target, "object_on_support", store=None
        )
        if mid_result.label not in {"pass", "degraded"}:
            continue
        if not _indirect_intermediary_target_match(intermediary, target, mid_result):
            continue
        mid_bbox = intermediary.get("bbox_world") or {}
        mid_min = mid_bbox.get("min") or []
        mid_max = mid_bbox.get("max") or []
        mid_center = mid_bbox.get("center") or []
        subject_center = subject_bbox.get("center") or []
        if len(mid_min) < 3 or len(mid_max) < 3:
            continue
        if (
            len(mid_center) >= 3
            and len(subject_center) >= 3
            and float(mid_center[2]) > float(subject_center[2]) + 0.02
        ):
            continue
        if float(mid_min[2]) > subject_bottom + 0.05:
            continue
        overlap = _footprint_overlap_ratio_xy(subject, intermediary, inflate=0.08)
        gap = bbox_gap_xy(subject, intermediary)
        dz = abs(subject_bottom - _support_top_z(mid_bbox))
        evidence = dict(mid_result.evidence)
        evidence.update(
            {
                "support_intermediary_id": intermediary_id,
                "support_chain_ids": [
                    str(subject.get("id") or "subject"),
                    intermediary_id,
                    str(target.get("id") or "target"),
                ],
                "support_intermediary_evaluation_path": mid_result.evaluation_path,
                "support_overlap_ratio": round(overlap, 4),
                "support_height_delta_m": round(dz, 4),
            }
        )
        if overlap >= 0.55 and dz <= 0.22:
            return _support_result(
                "pass",
                0.83,
                f"subject is plausibly supported indirectly via `{intermediary_id}` on `{target.get('id')}`: overlap {overlap:.2f}, height delta {dz:.2f}m.",
                "indirect",
                evidence,
            )
        if (
            _is_surface_intermediary(intermediary)
            and gap is not None
            and gap <= 0.08
            and dz <= 0.22
        ):
            return _support_result(
                "degraded",
                0.66,
                f"subject appears indirectly supported via nearby `{intermediary_id}` on `{target.get('id')}`: gap {gap:.2f}m, height delta {dz:.2f}m.",
                "indirect",
                evidence,
            )
    return None


def _indirect_intermediary_target_match(
    intermediary: dict[str, Any],
    target: dict[str, Any],
    mid_result: SupportRelationResult,
) -> bool:
    overlap = _footprint_overlap_ratio_xy(intermediary, target, inflate=0.05)
    gap = bbox_gap_xy(intermediary, target)
    dz = _subject_to_bbox_top_delta(intermediary, target)
    path = mid_result.evaluation_path
    if path == "support_region":
        evidence_overlap = _float_evidence(
            mid_result.evidence, "support_overlap_ratio", 0.0
        )
        evidence_dz = _float_evidence(
            mid_result.evidence, "support_height_delta_m", 999.0
        )
        return evidence_overlap >= 0.30 and evidence_dz <= 0.30
    if path == "bbox_fallback":
        return overlap >= 0.30 and dz <= 0.30
    if path in {"thin_edge", "direct"}:
        return (overlap >= 0.25 and dz <= 0.24) or (
            gap is not None and gap <= 0.08 and dz <= 0.20
        )
    return False


def _soft_sleeping_surface_support_fallback(
    subject: dict[str, Any],
    target: dict[str, Any],
    *,
    overlap_ratio: float,
    subject_bottom: float,
) -> tuple[str, float, str] | None:
    if object_category(target) not in BEDS and _category_group(target) != "sleeping":
        return None
    if not _is_supported_small_subject(subject):
        return None

    sb = subject.get("bbox_world") or {}
    tb = target.get("bbox_world") or {}
    smax = sb.get("max") or []
    tmin = tb.get("min") or []
    tmax = tb.get("max") or []
    tcenter = tb.get("center") or []
    if len(smax) < 3 or len(tmin) < 3 or len(tmax) < 3 or len(tcenter) < 3:
        return None

    subject_top = float(smax[2])
    target_center_z = float(tcenter[2])
    target_top = float(tmax[2])
    if overlap_ratio < 0.75:
        return None
    if subject_bottom < target_center_z + 0.05:
        return None
    if subject_bottom > target_top + 0.12:
        return None
    if subject_top > target_top + 0.15:
        return None
    if subject_top < float(tmin[2]) + 0.35:
        return None
    return (
        "pass",
        0.8,
        "subject strongly overlaps a bed sleeping surface; accepting soft-surface support because bedding likely sits below the extracted rigid top region.",
    )


def _is_secondary_support_intermediary(
    subject: dict[str, Any], obj: dict[str, Any]
) -> bool:
    if _is_surface_intermediary(obj):
        return True
    if _is_stackable_support_subject(subject) and _is_stackable_support_subject(obj):
        return True
    return False


def _shelf_footprint_fallback_subject_allowed(
    subject: dict[str, Any], target: dict[str, Any]
) -> bool:
    if _token_text_has_any(subject, STACKABLE_SUPPORT_TEXT_HINTS):
        return True
    if object_category(subject) in {"plant", "vase"} or _token_text_has_any(
        subject, ("plant", "potted_plant", "vase")
    ):
        return True
    return False


def _is_surface_intermediary(obj: dict[str, Any]) -> bool:
    category = object_category(obj)
    if category not in SECONDARY_SUPPORT_CATEGORIES and not _token_text_has_any(
        obj, tuple(SECONDARY_SUPPORT_CATEGORIES)
    ):
        return False
    if _category_group(obj) == "seating":
        return False
    if _category_token_has_any(obj, SOFT_SUPPORT_TARGET_REJECT_HINTS):
        return False
    return not _is_any_lamp_object(obj)


def _is_stackable_support_subject(obj: dict[str, Any]) -> bool:
    if _category_group(obj) == "seating":
        return False
    if _category_token_has_any(obj, SOFT_SUPPORT_TARGET_REJECT_HINTS):
        return False
    if _is_any_lamp_object(obj):
        return False
    return _token_text_has_any(obj, STACKABLE_SUPPORT_TEXT_HINTS)


def _thin_support_top_fallback(
    subject: dict[str, Any],
    target: dict[str, Any],
    *,
    subject_bottom: float,
) -> tuple[str, float, str] | None:
    thin_overlap = _thin_support_overlap_ratio_xy(subject, target)
    if thin_overlap is None:
        return None
    bbox_top = _support_top_z(target.get("bbox_world") or {})
    dz = abs(subject_bottom - bbox_top)
    wide_overlap = _footprint_overlap_ratio_xy(subject, target, inflate=0.24)
    gap = bbox_gap_xy(subject, target)
    if gap is not None and gap <= 0.25 and wide_overlap >= 0.45 and dz <= 0.08:
        return (
            "pass",
            0.84,
            f"thin shelf support is plausible at the edge with wide edge tolerance: gap {gap:.2f}m, "
            f"wide overlap {wide_overlap:.2f}, height delta {dz:.2f}m.",
        )
    if thin_overlap >= 0.50 and dz <= 0.18:
        return (
            "pass",
            0.84,
            f"thin shelf support is plausible with front-edge tolerance: overlap {thin_overlap:.2f}, height delta {dz:.2f}m.",
        )
    if thin_overlap >= 0.35 and dz <= 0.28:
        return (
            "degraded",
            0.68,
            f"thin shelf support is approximate with front-edge tolerance: overlap {thin_overlap:.2f}, height delta {dz:.2f}m.",
        )
    return None


def _semantic_support_edge_fallback(
    subject: dict[str, Any],
    target: dict[str, Any],
    subject_bottom: float,
    *,
    overlap_ratio: float,
) -> tuple[str, float, str] | None:
    if not _is_bbox_edge_fallback_target(target):
        return None
    bbox_overlap = _footprint_overlap_ratio_xy(subject, target, inflate=0.18)
    gap = bbox_gap_xy(subject, target)
    bbox_top = _support_top_z(target.get("bbox_world") or {})
    dz = abs(subject_bottom - bbox_top)
    if bbox_overlap >= 0.85 and dz <= 0.24:
        return (
            "pass",
            0.84,
            f"support region is conservative, but bbox top/edge support is plausible: region overlap {overlap_ratio:.2f}, "
            f"bbox overlap {bbox_overlap:.2f}, height delta {dz:.2f}m.",
        )
    if _is_small_bedside_edge_surface_match(
        subject, target, bbox_overlap=bbox_overlap, height_delta=dz
    ):
        return (
            "pass",
            0.82,
            f"support region is conservative for a small bedside object, but bbox top/edge support is plausible: "
            f"region overlap {overlap_ratio:.2f}, bbox overlap {bbox_overlap:.2f}, height delta {dz:.2f}m.",
        )
    if _is_tiny_bedside_clock_edge_match(
        subject, target, bbox_overlap=bbox_overlap, height_delta=dz
    ):
        return (
            "pass",
            0.8,
            f"support region is conservative for a tiny bedside clock near the furniture edge, but top support is still plausible: "
            f"region overlap {overlap_ratio:.2f}, bbox overlap {bbox_overlap:.2f}, height delta {dz:.2f}m.",
        )
    if bbox_overlap >= 0.65 and dz <= 0.30:
        return (
            "degraded",
            0.68,
            f"support region is conservative near the furniture edge: region overlap {overlap_ratio:.2f}, "
            f"bbox overlap {bbox_overlap:.2f}, height delta {dz:.2f}m.",
        )
    if gap is not None and gap <= 0.08 and bbox_overlap >= 0.45 and dz <= 0.24:
        return (
            "degraded",
            0.66,
            f"support region misses a near-edge object, but bbox geometry is plausible: gap {gap:.2f}m, "
            f"bbox overlap {bbox_overlap:.2f}, height delta {dz:.2f}m.",
        )
    return None


def _is_weight_plate_subject(subject: dict[str, Any]) -> bool:
    return _token_text_has_any(subject, ("weight plate", "weight_plate"))


def _is_weight_storage_rack_target(target: dict[str, Any]) -> bool:
    category = object_category(target)
    if category != "rack" and not _category_token_has_any(target, ("rack",)):
        return False
    return _token_text_has_any(
        target,
        (
            "store weights",
            "weight storage",
            "horizontal bars",
            "horizontal bars for weight storage",
        ),
    )


def _is_bbox_edge_fallback_target(target: dict[str, Any]) -> bool:
    category = object_category(target)
    category_group = _category_group(target)
    if category in {
        "cabinet",
        "counter",
        "desk",
        "drawer",
        "dresser",
        "table",
        "dining_table",
        "bar_table",
        "coffee_table",
        "island",
        "nightstand",
        "shelf",
        "wall_shelf",
        "bookshelf",
        "credenza",
        "sideboard",
        "console",
        "buffet",
        "media_console",
        "storage_furniture",
        "tv_stand",
        "wardrobe",
    }:
        return True
    if category_group in SUPPORT_CATEGORY_GROUPS and _has_support_storage_semantics(
        target
    ):
        return True
    return _category_surface_family_match(target)


def _thin_support_overlap_ratio_xy(
    subject: dict[str, Any], target: dict[str, Any]
) -> float | None:
    if not _is_thin_linear_support(target):
        return None
    sb = subject.get("bbox_world") or {}
    tb = target.get("bbox_world") or {}
    smin = sb.get("min") or []
    smax = sb.get("max") or []
    tmin = tb.get("min") or []
    tmax = tb.get("max") or []
    tsize = tb.get("size") or []
    if (
        len(smin) < 2
        or len(smax) < 2
        or len(tmin) < 2
        or len(tmax) < 2
        or len(tsize) < 2
    ):
        return None
    sx0, sy0, sx1, sy1 = float(smin[0]), float(smin[1]), float(smax[0]), float(smax[1])
    tx0, ty0, tx1, ty1 = float(tmin[0]), float(tmin[1]), float(tmax[0]), float(tmax[1])
    x_size = abs(float(tsize[0]))
    y_size = abs(float(tsize[1]))
    long_inflate = 0.05
    front_edge_inflate = 0.18
    if x_size <= y_size:
        tx0 -= front_edge_inflate
        tx1 += front_edge_inflate
        ty0 -= long_inflate
        ty1 += long_inflate
    else:
        tx0 -= long_inflate
        tx1 += long_inflate
        ty0 -= front_edge_inflate
        ty1 += front_edge_inflate
    overlap_x = max(0.0, min(sx1, tx1) - max(sx0, tx0))
    overlap_y = max(0.0, min(sy1, ty1) - max(sy0, ty0))
    subject_area = max(sx1 - sx0, 0.0) * max(sy1 - sy0, 0.0)
    if subject_area <= 1e-6:
        return None
    return overlap_x * overlap_y / subject_area


def _valid_support_regions(target: dict[str, Any]) -> list[dict[str, Any]]:
    raw_regions = target.get("support_regions")
    if not isinstance(raw_regions, list):
        return []
    regions: list[dict[str, Any]] = []
    for region in raw_regions:
        if not isinstance(region, dict):
            continue
        access_type = str(region.get("access_type") or "").strip().lower()
        if access_type in {"sealed", "blocked", "decorative"}:
            continue
        if _region_height_world(region) is None:
            continue
        if len(_region_world_polygon(region)) < 3:
            continue
        regions.append(region)
    return regions


def _region_world_polygon(region: dict[str, Any]) -> list[tuple[float, float]]:
    raw = region.get("polygon_world_xy") or []
    points: list[tuple[float, float]] = []
    if not isinstance(raw, list):
        return points
    for point in raw:
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            try:
                points.append((float(point[0]), float(point[1])))
            except Exception:
                continue
    return points


def _region_height_world(region: dict[str, Any]) -> float | None:
    value = region.get("height_world_z")
    if value is None:
        value = region.get("height_z")
    try:
        return float(value)
    except Exception:
        return None


def _region_clearance(region: dict[str, Any]) -> float:
    try:
        return max(float(region.get("clearance_above_m")), 0.0)
    except Exception:
        return 1.0


def _is_internal_support_region(region: dict[str, Any]) -> bool:
    kind = str(region.get("support_kind") or "").strip().lower()
    access_type = str(region.get("access_type") or "").strip().lower()
    if access_type in {
        "front_open",
        "front-open",
        "open_shelf",
        "openable_storage",
        "internal_storage",
    }:
        return True
    return any(term in kind for term in ("cabinet", "drawer", "shelf", "storage"))


def _polygon_area_xy(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for index, (x0, y0) in enumerate(points):
        x1, y1 = points[(index + 1) % len(points)]
        area += x0 * y1 - x1 * y0
    return abs(area) * 0.5


def _convex_overlap_area(
    subject_poly: list[tuple[float, float]], region_poly: list[tuple[float, float]]
) -> float:
    if len(subject_poly) < 3 or len(region_poly) < 3:
        return 0.0
    clipper = region_poly
    if _signed_polygon_area_xy(clipper) < 0:
        clipper = list(reversed(clipper))
    output = (
        subject_poly
        if _signed_polygon_area_xy(subject_poly) >= 0
        else list(reversed(subject_poly))
    )
    for edge_index, edge_start in enumerate(clipper):
        edge_end = clipper[(edge_index + 1) % len(clipper)]
        input_points = output
        output = []
        if not input_points:
            break
        previous = input_points[-1]
        for current in input_points:
            current_inside = _point_left_of_edge(current, edge_start, edge_end) >= -1e-9
            previous_inside = (
                _point_left_of_edge(previous, edge_start, edge_end) >= -1e-9
            )
            if current_inside:
                if not previous_inside:
                    output.append(
                        _line_intersection(previous, current, edge_start, edge_end)
                    )
                output.append(current)
            elif previous_inside:
                output.append(
                    _line_intersection(previous, current, edge_start, edge_end)
                )
            previous = current
    return _polygon_area_xy(output)


def _signed_polygon_area_xy(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for index, (x0, y0) in enumerate(points):
        x1, y1 = points[(index + 1) % len(points)]
        area += x0 * y1 - x1 * y0
    return area * 0.5


def _point_left_of_edge(
    point: tuple[float, float],
    edge_start: tuple[float, float],
    edge_end: tuple[float, float],
) -> float:
    return (edge_end[0] - edge_start[0]) * (point[1] - edge_start[1]) - (
        edge_end[1] - edge_start[1]
    ) * (point[0] - edge_start[0])


def _line_intersection(
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
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return float(px), float(py)


def _object_on_support_rank(
    subject: dict[str, Any], target: dict[str, Any]
) -> tuple[float, float] | None:
    assessment = assess_direct_support(subject, target)
    if assessment is not None and assessment.evidence is not None:
        if (
            assessment.label == "fail"
            and not _is_horizontally_plausible_support_candidate(subject, target)
        ):
            return None
        label_rank = {"pass": 0.0, "degraded": 1.0, "unknown": 2.0, "fail": 3.0}.get(
            assessment.label, 3.0
        )
        evidence = assessment.evidence
        return label_rank, evidence.height_delta_m + max(
            1.0 - evidence.overlap_ratio, 0.0
        )

    label, _confidence, _reason = _eval_object_on_support(
        subject, target, "object_on_support"
    )
    if label not in {"pass", "unknown", "degraded"}:
        return None
    sb = subject.get("bbox_world") or {}
    tb = target.get("bbox_world") or {}
    smin = sb.get("min") or []
    if len(smin) >= 3:
        dz = abs(float(smin[2]) - _support_top_z(tb))
    else:
        dz = 999.0
    overlap = _footprint_overlap_ratio_xy(subject, target, inflate=0.05)
    if label in {
        "unknown",
        "degraded",
    } and not _is_horizontally_plausible_support_candidate(subject, target):
        return None
    label_rank = {"pass": 0.0, "unknown": 1.0, "degraded": 2.0}.get(label, 3.0)
    return label_rank, dz + max(1.0 - overlap, 0.0)


def _is_horizontally_plausible_support_candidate(
    subject: dict[str, Any],
    target: dict[str, Any],
    *,
    max_gap: float = 0.35,
) -> bool:
    if _footprint_overlap_ratio_xy(subject, target, inflate=0.18) > 0.0:
        return True
    gap = bbox_gap_xy(subject, target)
    return gap is not None and gap <= max_gap


def _is_support_target(target: dict[str, Any]) -> bool:
    profile = object_function_profile(target)
    # 2026-07-17 修改原因：灯具和小型 manipuland 即使带有噪声
    # `can_support_top` 标注，也不应成为其他物体的主支撑目标。
    if _is_any_lamp_object(target):
        return False
    if profile.source == "explicit" and profile.is_small_placeable:
        scene_type = _scene_object_type(target)
        if scene_type == "manipuland":
            return False
    if profile.source == "explicit" and (
        profile.can_support_top or profile.has_internal_shelf
    ):
        return True
    if _category_token_has_any(target, SMALL_OBJECT_TEXT_HINTS):
        return False
    if _category_token_has_any(target, SOFT_SUPPORT_TARGET_REJECT_HINTS):
        return False
    if _has_strong_support_target_semantics(target):
        return True
    category = object_category(target)
    if is_small_object(target):
        return False
    if _valid_support_regions(target):
        return True
    if category in SUPPORTS:
        return True
    if _has_support_storage_semantics(target):
        return True
    return _category_token_has_any(
        target, SUPPORT_SURFACE_HINTS + ("wall_shelf", "storage_furniture")
    )


def _is_primary_support_target(target: dict[str, Any]) -> bool:
    if _is_rigid_bench_platform_support_target(target):
        return True
    if not _is_support_target(target):
        return False
    return (
        object_category(target) not in SEATING and _category_group(target) != "seating"
    )


def _is_rigid_bench_platform_support_target(target: dict[str, Any]) -> bool:
    if object_category(target) != "bench":
        return False
    if _category_token_has_any(
        target,
        SOFT_SUPPORT_TARGET_REJECT_HINTS + ("beanbag", "pillow", "cushion", "seat_pad"),
    ):
        return False
    bbox = target.get("bbox_world") or {}
    size = bbox.get("size") or []
    tmax = bbox.get("max") or []
    if len(size) < 3 or len(tmax) < 3:
        return False
    x_size = abs(float(size[0]))
    y_size = abs(float(size[1]))
    z_size = abs(float(size[2]))
    long_span = max(x_size, y_size)
    short_span = min(x_size, y_size)
    top_height = float(tmax[2])
    return (
        long_span >= 0.75
        and short_span >= 0.22
        and z_size <= 0.65
        and top_height <= 0.85
    )


def _is_lamp_surface_target(target: dict[str, Any]) -> bool:
    if not _is_primary_support_target(target):
        return False
    if _category_group(target) == "lighting":
        return False
    return not _is_any_lamp_object(target)


def _has_strong_support_target_semantics(target: dict[str, Any]) -> bool:
    """Allow thin shelves/cabinets that look small by bbox but are semantic supports."""
    category = object_category(target)
    category_group = _category_group(target)
    if category in SUPPORTS:
        return True
    if category_group in SUPPORT_CATEGORY_GROUPS and _has_support_storage_semantics(
        target
    ):
        return True
    if category_group in SUPPORT_CATEGORY_GROUPS and _valid_support_regions(target):
        return True
    if _category_surface_family_match(target):
        return True
    return _category_token_has_any(
        target, SUPPORT_SURFACE_HINTS + ("wall_shelf", "storage_furniture")
    )


def _is_thin_linear_support(target: dict[str, Any]) -> bool:
    category = object_category(target)
    if category not in {"shelf", "wall_shelf"} and "open_shelf" not in _support_modes(
        target
    ):
        return False
    bbox = target.get("bbox_world") or {}
    size = bbox.get("size") or []
    if len(size) < 3:
        return False
    x_size = abs(float(size[0]))
    y_size = abs(float(size[1]))
    z_size = abs(float(size[2]))
    long_span = max(x_size, y_size)
    short_span = min(x_size, y_size)
    return (
        long_span >= 0.45
        and short_span <= 0.12
        and short_span <= long_span * 0.25
        and z_size <= 0.45
    )


def _is_multilevel_shelf_like_target(target: dict[str, Any]) -> bool:
    category = object_category(target)
    if category in {"bookshelf", "bookcase"}:
        return True
    if category in {"shelf", "wall_shelf"} and "open_shelf" in _support_modes(target):
        return True
    if _category_token_has_any(
        target, ("bookshelf", "bookcase", "shelving", "shelving_unit")
    ):
        return True
    return _token_text_has_any(
        target, ("bookshelf", "bookcase", "shelving", "shelving_unit")
    )


def _support_top_z(bbox: dict[str, Any]) -> float:
    tmax = bbox.get("max") or []
    tcenter = bbox.get("center") or []
    tsize = bbox.get("size") or []
    if len(tmax) >= 3:
        return float(tmax[2])
    if len(tcenter) >= 3 and len(tsize) >= 3:
        return float(tcenter[2]) + float(tsize[2]) / 2.0
    return 0.0


def _is_low_open_table_target(target: dict[str, Any]) -> bool:
    category = object_category(target)
    if category == "coffee_table":
        return True
    hints = target.get("functional_hints") or {}
    keywords = " ".join(
        str(value or "").strip().lower()
        for value in (hints.get("category_keywords") or [])
    )
    text = " ".join(
        [
            str(target.get("id") or "").strip().lower(),
            str(target.get("category") or "").strip().lower(),
            str(target.get("category_norm") or "").strip().lower(),
            keywords,
        ]
    )
    return any(
        phrase in text
        for phrase in (
            "coffee table",
            "center table",
            "low table",
            "endtable",
            "sidetable",
            "end_table",
            "side_table",
        )
    )


def _plausible_internal_support(
    subject: dict[str, Any],
    target: dict[str, Any],
    *,
    overlap_ratio: float,
    support_top: float,
    support_modes: set[str],
) -> bool:
    if not (support_modes & {"open_shelf", "internal_storage"}):
        return False
    sb = subject.get("bbox_world") or {}
    tb = target.get("bbox_world") or {}
    smin = sb.get("min") or []
    smax = sb.get("max") or []
    tmin = tb.get("min") or []
    tmax = tb.get("max") or []
    if len(smin) < 3 or len(smax) < 3 or len(tmin) < 3 or len(tmax) < 3:
        return False
    if overlap_ratio < 0.55:
        return False
    if float(smax[2]) > support_top - 0.08:
        return False
    within_x = (
        float(smin[0]) >= float(tmin[0]) - 0.06
        and float(smax[0]) <= float(tmax[0]) + 0.06
    )
    within_y = (
        float(smin[1]) >= float(tmin[1]) - 0.06
        and float(smax[1]) <= float(tmax[1]) + 0.06
    )
    within_z = (
        float(smin[2]) >= float(tmin[2]) - 0.03
        and float(smax[2]) <= float(tmax[2]) + 0.06
    )
    return within_x and within_y and within_z


def _footprint_overlap_ratio_xy(
    subject: dict[str, Any], target: dict[str, Any], *, inflate: float
) -> float:
    sb = subject.get("bbox_world") or {}
    tb = target.get("bbox_world") or {}
    smin = sb.get("min") or []
    smax = sb.get("max") or []
    tmin = tb.get("min") or []
    tmax = tb.get("max") or []
    if len(smin) < 2 or len(smax) < 2 or len(tmin) < 2 or len(tmax) < 2:
        return 0.0
    sx0, sy0, sx1, sy1 = float(smin[0]), float(smin[1]), float(smax[0]), float(smax[1])
    tx0 = float(tmin[0]) - inflate
    ty0 = float(tmin[1]) - inflate
    tx1 = float(tmax[0]) + inflate
    ty1 = float(tmax[1]) + inflate
    overlap_x = max(0.0, min(sx1, tx1) - max(sx0, tx0))
    overlap_y = max(0.0, min(sy1, ty1) - max(sy0, ty0))
    overlap = overlap_x * overlap_y
    subject_area = max(sx1 - sx0, 0.0) * max(sy1 - sy0, 0.0)
    if subject_area <= 1e-6:
        return 0.0
    return overlap / subject_area
