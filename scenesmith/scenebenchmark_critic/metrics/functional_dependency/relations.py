from __future__ import annotations

import math

from typing import Any

from scenesmith.scenebenchmark_critic.core.geometry import (
    GeometryStore,
    angle_to_target_deg,
    bbox_center_xy,
    bbox_gap_xy,
    bbox_height_span,
    distance_xy,
    front_vector,
    object_category,
    seating_angle_to_target_deg,
    side_vector,
)
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.constants import *
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.profiles import (
    object_function_profile,
)
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.results import (
    _empty_fd_diagnostics,
    _fd_diagnostics_from_targets,
    _fd_label_rank,
    _relation_label_rank,
    _result_scoring_tier_payload,
    _target_eval_payload,
    _unknown,
)
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.seat_surface_assignment import (
    ASSIGNMENT_SOURCE,
)
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.semantics import (
    _category_group,
    _category_token_has_any,
    _is_actionable_seating_surface_pair,
    _is_any_lamp_object,
    _is_lamp_subject,
    _is_media_target,
    _is_nightstand_target,
    _is_seating_subject,
    _is_side_surface_target,
    _is_computer_peripheral_subject,
    _is_computer_screen_target,
    _is_directional_facing_subject,
    _is_facing_relation_target,
    _is_supported_small_subject,
    _is_work_surface_target,
    _scene_object_type,
)
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.support import (
    _eval_object_on_support,
    _is_lamp_surface_target,
    _is_primary_support_target,
    evaluate_support_relation,
)


def evaluate_functional_dependency(
    store: GeometryStore,
    check: dict[str, Any],
) -> dict[str, Any]:
    subject_id = str(check.get("subject_id") or "")
    target_ids = [str(item) for item in (check.get("target_ids") or []) if str(item)]
    subject = store.objects.get(subject_id)
    targets = [
        store.objects[target_id]
        for target_id in target_ids
        if target_id in store.objects
    ]
    if subject is None:
        return _unknown(
            check, f"Rule dependency could not find subject object `{subject_id}`."
        )
    if not targets:
        return _unknown(
            check, f"Rule dependency found no valid target objects for `{subject_id}`."
        )

    relation_type = _normalize_relation_type(
        str(
            check.get("relation_type")
            or _infer_relation_type(subject, targets[0])
            or "generic_relation"
        )
    )
    label, confidence, reason, diagnostics = _eval_relation_over_targets(
        store, subject, targets, relation_type, check=check
    )
    selected_related_objects = [
        str(item)
        for item in (diagnostics.get("selected_target_ids") or [])
        if str(item)
    ]
    repair_advice = ""
    assignment_evidence = check.get("evidence") or {}
    assignment_check = _is_required_seat_surface_assignment(check)
    living_room_coffee_pair = (
        relation_type == "seating_to_work_surface"
        and object_category(subject) in LIVING_ROOM_SEATING
        and any(object_category(target) == "coffee_table" for target in targets)
    )
    if living_room_coffee_pair and label in {"degraded", "fail"}:
        repair_targets = selected_related_objects or target_ids
        target_text = ", ".join(f"`{item}`" for item in repair_targets)
        # 2026-07-16 修改原因：客厅椅允许面向最近茶几，但必须指向茶几所在的
        # 室内活动区；宽松射线相交检查会把约 90 度的外翻姿态误当作可用。
        repair_advice = (
            f"Nearest-focus repair: rotate `{subject_id}` toward {target_text} until "
            "the SceneBenchmark seat-to-focus angle is at most 45 degrees. Use the "
            "exact optimal rotation for this seat/target pair even if the facing "
            "tool's broad `is_facing` boolean is true; do not leave the chair facing "
            "outward from the local activity area."
        )
    elif assignment_check and label in {"unknown", "degraded", "fail"}:
        slot = assignment_evidence.get("target_slot") or {}
        center = slot.get("center_xy") or []
        yaw = slot.get("yaw_deg")
        if len(center) >= 2 and yaw is not None:
            repair_advice = (
                f"One-to-one work-seat repair: move `{subject_id}` beside "
                f"`{target_ids[0]}` near XY [{float(center[0]):.3f}, "
                f"{float(center[1]):.3f}] and set yaw near {float(yaw):.1f} degrees. "
                "Keep this chair assigned to this surface; do not reuse the same "
                "desk for another work chair."
            )
        else:
            repair_advice = (
                f"One-to-one work-seat repair: move `{subject_id}` to a usable edge "
                f"of `{target_ids[0]}` and rotate the chair toward the work surface."
            )
        diagnostics["seat_surface_assignment"] = assignment_evidence
    return {
        "check_id": check.get("check_id"),
        "metric": "functional_dependency",
        "label": label,
        "reason": f"Rule dependency `{relation_type}`: subject `{subject_id}`; {reason}",
        "blocking_objects": [],
        "confidence": confidence,
        "evaluation_source": "rule_functional_dependency",
        "primary_object": subject_id,
        "related_objects": target_ids,
        "selected_related_objects": selected_related_objects,
        "relation_type": relation_type,
        "diagnostics": diagnostics,
        "repair_advice": repair_advice,
        **_result_scoring_tier_payload(check.get("scoring_tier")),
    }


def _infer_relation_type(subject: dict[str, Any], target: dict[str, Any]) -> str | None:
    sc = object_category(subject)
    subject_profile = object_function_profile(subject)
    if (
        _is_seating_subject(subject)
        and _is_work_surface_target(target)
        and _is_actionable_seating_surface_pair(subject, target)
    ):
        return "seating_to_work_surface"
    if _is_seating_subject(subject) and _is_media_target(target):
        return "seating_to_media"
    if (
        sc in BEDS
        or (
            subject_profile.source == "explicit" and subject_profile.is_sleeping_surface
        )
    ) and _is_nightstand_target(target):
        return "bed_to_nightstand"
    if (
        _is_supported_small_subject(subject) and _is_primary_support_target(target)
    ) or _is_soft_furnishing_seating_support_pair(subject, target):
        return "object_on_support"
    if _is_lamp_subject(subject) and _is_lamp_surface_target(target):
        return "lamp_to_surface"
    return None


def _normalize_relation_type(relation_type: str) -> str:
    return {
        "back_to_wall": "back_against_wall",
        "face_to": "furniture_faces_furniture",
        "faces": "furniture_faces_furniture",
        "facing": "furniture_faces_furniture",
        "front_faces": "furniture_faces_furniture",
        "media_viewing": "seating_to_media",
        "seat_faces_table": "seat_faces_surface",
        "bedside": "bedside_pair",
        "bed_to_nightstands": "bedside_pair",
        "near": "generic_near_relation",
        "generic_relation": "generic_near_relation",
    }.get(relation_type, relation_type)


def _eval_relation_over_targets(
    store: GeometryStore,
    subject: dict[str, Any],
    targets: list[dict[str, Any]],
    relation_type: str,
    *,
    check: dict[str, Any] | None = None,
) -> tuple[str, float, str, dict[str, Any]]:
    if relation_type == "dining_set":
        return _eval_dining_set(subject, targets)
    if relation_type == "workstation":
        return _eval_workstation(subject, targets)
    if relation_type == "bedside_pair":
        return _eval_bedside_pair(subject, targets)

    scored: list[dict[str, Any]] = []
    for target in targets:
        target_relation = relation_type
        eval_subject = subject
        eval_target = target
        direction_note = ""
        if relation_type == "generic_near_relation":
            inferred = _infer_relation_type(subject, target)
            if inferred and _relation_target_is_valid(subject, target, inferred):
                target_relation = inferred
            elif _relation_target_is_valid(target, subject, "object_on_support"):
                target_relation = "object_on_support"
                eval_subject = target
                eval_target = subject
                direction_note = f"interpreted reversed support direction: `{target.get('id')}` is supported by `{subject.get('id')}`; "
        elif not _relation_target_is_valid(subject, target, relation_type):
            inferred = _infer_relation_type(subject, target)
            if inferred and _relation_target_is_valid(subject, target, inferred):
                target_relation = inferred
            elif _relation_target_is_valid(target, subject, "object_on_support"):
                target_relation = "object_on_support"
                eval_subject = target
                eval_target = subject
                direction_note = f"interpreted reversed support direction: `{target.get('id')}` is supported by `{subject.get('id')}`; "
            else:
                scored.append(
                    _target_eval_payload(
                        target,
                        "fail",
                        0.3,
                        "target category is not compatible with relation.",
                        relation_type,
                    )
                )
                continue
        evaluator = {
            "seating_to_work_surface": _eval_seating_to_surface,
            "seat_faces_surface": _eval_seating_to_surface,
            "seating_to_media": _eval_facing_relation,
            "bed_to_nightstand": _eval_bed_to_nightstand,
            "object_on_support": _eval_object_on_support,
            "lamp_to_surface": _eval_object_on_support,
        }.get(target_relation, _eval_generic_near_relation)
        if target_relation in {
            "back_against_wall",
            "side_or_back_against_wall",
            "mounted_to_wall",
            "mounted_to_ceiling",
            "object_on_floor",
            "computer_peripheral_faces_screen",
            "furniture_faces_furniture",
            "display_faces_user",
        }:
            label, confidence, reason = _eval_annotated_dependency_relation(
                subject, target, target_relation, check
            )
            scored.append(
                _target_eval_payload(target, label, confidence, reason, target_relation)
            )
        elif evaluator is _eval_object_on_support:
            support_result = evaluate_support_relation(
                eval_subject, eval_target, target_relation, store=store
            )
            scored.append(
                _target_eval_payload(
                    target,
                    support_result.label,
                    support_result.confidence,
                    direction_note + support_result.reason,
                    target_relation,
                    evidence=support_result.evidence,
                )
            )
        else:
            if evaluator is _eval_seating_to_surface:
                if _is_required_seat_surface_assignment(check):
                    label, confidence, reason = _eval_assigned_seating_slot(
                        subject, target, check or {}
                    )
                else:
                    label, confidence, reason = _eval_seating_to_surface(
                        subject, target, target_relation
                    )
            else:
                label, confidence, reason = evaluator(subject, target, target_relation)
            scored.append(
                _target_eval_payload(target, label, confidence, reason, target_relation)
            )
    if not scored:
        return "unknown", 0.0, "no target could be evaluated.", _empty_fd_diagnostics()

    best = max(
        scored,
        key=lambda item: (
            _relation_label_rank(relation_type, item["label"]),
            item["confidence"],
        ),
    )
    rescue = _maybe_rescue_support_target(
        store, subject, targets, relation_type, scored, best
    )
    if rescue is not None:
        return rescue
    diagnostics = _fd_diagnostics_from_targets(scored, selected=[best["target_id"]])
    return (
        best["label"],
        best["confidence"],
        f"selected `{best['target_id']}`; {best['reason']}",
        diagnostics,
    )


def _maybe_rescue_support_target(
    store: GeometryStore,
    subject: dict[str, Any],
    declared_targets: list[dict[str, Any]],
    relation_type: str,
    scored: list[dict[str, Any]],
    best: dict[str, Any],
) -> tuple[str, float, str, dict[str, Any]] | None:
    if relation_type not in {"object_on_support", "lamp_to_surface"}:
        return None
    if any(item.get("label") == "pass" for item in scored):
        return None
    if best.get("label") == "degraded" and float(best.get("confidence") or 0.0) >= 0.78:
        return None

    subject_id = str(subject.get("id") or "")
    declared_ids = {
        str(target.get("id") or "") for target in declared_targets if target.get("id")
    }
    candidates: list[dict[str, Any]] = []
    for candidate in store.objects.values():
        candidate_id = str(candidate.get("id") or "")
        if (
            not candidate_id
            or candidate_id == subject_id
            or candidate_id in declared_ids
        ):
            continue
        if not _is_support_rescue_candidate(subject, candidate, relation_type):
            continue
        support_result = evaluate_support_relation(
            subject, candidate, relation_type, store=store
        )
        if support_result.label != "pass" or support_result.confidence < 0.80:
            continue
        if not _rescue_is_clear_improvement(best, support_result, relation_type):
            continue
        candidates.append(
            {
                "candidate": candidate,
                "support_result": support_result,
                "rank": _support_rescue_rank(subject, candidate, support_result),
            }
        )

    if not candidates:
        return None
    selected = min(candidates, key=lambda item: item["rank"])
    candidate = selected["candidate"]
    support_result = selected["support_result"]
    candidate_id = str(candidate.get("id") or "")
    original_ids = [
        str(target.get("id") or "") for target in declared_targets if target.get("id")
    ]
    evidence = dict(support_result.evidence)
    evidence.update(
        {
            "support_evaluation_path": "target_rescue",
            "rescue_support_evaluation_path": support_result.evaluation_path,
            "rescue_from_target_ids": original_ids,
            "rescue_selected_target_id": candidate_id,
            "rescue_original_best_label": best.get("label"),
            "rescue_original_best_reason": best.get("reason"),
        }
    )
    rescue_confidence = min(float(support_result.confidence), 0.84)
    rescue_reason = (
        f"target rescue selected `{candidate_id}` after declared target `{best.get('target_id')}` "
        f"scored {best.get('label')}; candidate support: {support_result.reason}"
    )
    rescue_payload = _target_eval_payload(
        candidate,
        "pass",
        rescue_confidence,
        rescue_reason,
        relation_type,
        evidence=evidence,
    )
    rescue_scored = scored + [rescue_payload]
    diagnostics = _fd_diagnostics_from_targets(rescue_scored, selected=[candidate_id])
    diagnostics.update(
        {
            "support_evaluation_path": "target_rescue",
            "rescue_from_target_ids": original_ids,
            "rescue_selected_target_id": candidate_id,
            "rescue_original_best_label": best.get("label"),
            "rescue_original_best_reason": best.get("reason"),
        }
    )
    return (
        "pass",
        rescue_confidence,
        f"selected `{candidate_id}` via target rescue; original target `{best.get('target_id')}` was {best.get('label')}; {support_result.reason}",
        diagnostics,
    )


def _is_support_rescue_candidate(
    subject: dict[str, Any],
    candidate: dict[str, Any],
    relation_type: str,
) -> bool:
    if relation_type == "object_on_support":
        return _is_primary_support_target(
            candidate
        ) or _is_soft_furnishing_seating_support_pair(subject, candidate)
    if relation_type != "lamp_to_surface":
        return False
    if _lamp_rescue_rejects_target(candidate):
        return False
    if _is_lamp_surface_target(candidate):
        return True
    return _is_rigid_lamp_rescue_platform(candidate)


def _lamp_rescue_rejects_target(candidate: dict[str, Any]) -> bool:
    category = object_category(candidate).lower()
    group = _category_group(candidate)
    if category in BEDS or group == "sleeping":
        return True
    if category in {
        "chair",
        "office_chair",
        "dining_chair",
        "armchair",
        "sofa",
        "loveseat",
    }:
        return True
    if _category_token_has_any(candidate, SOFT_SUPPORT_TARGET_REJECT_HINTS):
        return True
    if _category_token_has_any(
        candidate,
        (
            "art",
            "artwork",
            "painting",
            "picture",
            "poster",
            "mirror",
            "wall_art",
            "wall_mirror",
        ),
    ):
        return True
    if group in {"decor", "lighting"}:
        return True
    return _is_any_lamp_object(candidate)


def _is_rigid_lamp_rescue_platform(candidate: dict[str, Any]) -> bool:
    category = object_category(candidate).lower()
    if category not in {"bench", "stool", "ottoman"} and not _candidate_text_has_any(
        candidate, ("bench", "stool", "ottoman")
    ):
        return False
    bbox = candidate.get("bbox_world") or {}
    size = bbox.get("size") or []
    tmax = bbox.get("max") or []
    if len(size) < 3 or len(tmax) < 3:
        return False
    x_size = abs(float(size[0]))
    y_size = abs(float(size[1]))
    z_size = abs(float(size[2]))
    if max(x_size, y_size) < 0.28 or min(x_size, y_size) < 0.20:
        return False
    return z_size <= 0.75 and float(tmax[2]) <= 0.95


def _candidate_text_has_any(candidate: dict[str, Any], terms: tuple[str, ...]) -> bool:
    hints = candidate.get("functional_hints") or {}
    parts = [
        str(candidate.get("id") or ""),
        str(candidate.get("category") or ""),
        str(candidate.get("category_norm") or ""),
        str(hints.get("category_group") or ""),
    ]
    parts.extend(str(item or "") for item in (hints.get("category_keywords") or []))
    text = " ".join(parts).lower()
    return any(term in text for term in terms)


def _rescue_is_clear_improvement(
    best: dict[str, Any],
    support_result: Any,
    relation_type: str,
) -> bool:
    best_label = str(best.get("label") or "unknown")
    if _relation_label_rank(
        relation_type, support_result.label
    ) <= _relation_label_rank(relation_type, best_label):
        return False
    if best_label == "degraded" and float(best.get("confidence") or 0.0) >= 0.78:
        return support_result.confidence >= float(best.get("confidence") or 0.0) + 0.08
    return True


def _support_rescue_rank(
    subject: dict[str, Any],
    candidate: dict[str, Any],
    support_result: Any,
) -> tuple[float, float, float]:
    gap = bbox_gap_xy(subject, candidate)
    gap_rank = gap if gap is not None else 999.0
    overlap_rank = -_float_evidence(
        support_result.evidence, "support_overlap_ratio", 0.0
    )
    height_rank = _float_evidence(
        support_result.evidence, "support_height_delta_m", 999.0
    )
    return gap_rank, height_rank, overlap_rank


def _float_evidence(evidence: dict[str, Any], key: str, default: float) -> float:
    value = evidence.get(key)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _eval_dining_set(
    subject: dict[str, Any], targets: list[dict[str, Any]]
) -> tuple[str, float, str, dict[str, Any]]:
    table = (
        subject
        if object_category(subject) in DINING_TABLES
        else next(
            (target for target in targets if object_category(target) in DINING_TABLES),
            None,
        )
    )
    chairs = [obj for obj in ([subject] + targets) if _is_seating_subject(obj)]
    if table is None or not chairs:
        return (
            "fail",
            0.82,
            "missing dining table or seating targets.",
            _empty_fd_diagnostics(),
        )
    valid: list[dict[str, Any]] = []
    for chair in chairs:
        label, confidence, reason = _eval_seating_to_surface(
            chair, table, "seating_to_work_surface"
        )
        valid.append(
            _target_eval_payload(chair, label, confidence, reason, "dining_set")
        )
    pass_count = sum(1 for item in valid if item["label"] == "pass")
    degraded_count = sum(1 for item in valid if item["label"] in {"pass", "degraded"})
    if pass_count >= 2:
        label, confidence = "pass", 0.9
    elif degraded_count >= 1:
        label, confidence = "degraded", 0.74
    else:
        label, confidence = "fail", 0.84
    diagnostics = _fd_diagnostics_from_targets(
        valid,
        selected=[
            item["target_id"] for item in valid if item["label"] in {"pass", "degraded"}
        ],
    )
    diagnostics["cardinality_score"] = min(len(chairs) / 2.0, 1.0)
    return (
        label,
        confidence,
        f"{pass_count} seats strongly pair with dining table `{table.get('id')}`.",
        diagnostics,
    )


def _eval_workstation(
    subject: dict[str, Any], targets: list[dict[str, Any]]
) -> tuple[str, float, str, dict[str, Any]]:
    surface_candidates = [
        target for target in targets if _is_work_surface_target(target)
    ]
    if _is_work_surface_target(subject):
        surface_candidates.insert(0, subject)
    seat_candidates = [obj for obj in ([subject] + targets) if _is_seating_subject(obj)]
    if not surface_candidates or not seat_candidates:
        return "fail", 0.82, "missing work surface or seat.", _empty_fd_diagnostics()
    scored: list[dict[str, Any]] = []
    for seat in seat_candidates:
        for surface in surface_candidates:
            label, confidence, reason = _eval_seating_to_surface(
                seat, surface, "seating_to_work_surface"
            )
            payload = _target_eval_payload(
                surface, label, confidence, reason, "workstation"
            )
            payload["seat_id"] = seat.get("id")
            scored.append(payload)
    best = max(
        scored, key=lambda item: (_fd_label_rank(item["label"]), item["confidence"])
    )
    diagnostics = _fd_diagnostics_from_targets(scored, selected=[best["target_id"]])
    return (
        best["label"],
        best["confidence"],
        f"selected seat `{best.get('seat_id')}` and work surface `{best['target_id']}`; {best['reason']}",
        diagnostics,
    )


def _eval_bedside_pair(
    subject: dict[str, Any], targets: list[dict[str, Any]]
) -> tuple[str, float, str, dict[str, Any]]:
    bed = subject if object_category(subject) in BEDS else None
    if bed is None:
        return "fail", 0.82, "subject is not a bed.", _empty_fd_diagnostics()
    nightstands = [target for target in targets if _is_nightstand_target(target)]
    if not nightstands:
        return "fail", 0.82, "no nightstand target found.", _empty_fd_diagnostics()
    scored = []
    for target in nightstands:
        label, confidence, reason = _eval_bed_to_nightstand(
            bed, target, "bed_to_nightstand"
        )
        axis_label, axis_confidence, axis_reason = _eval_bedside_axis_alignment(
            bed, target
        )
        label, confidence, reason = _combine_bedside_results(
            label,
            confidence,
            reason,
            axis_label,
            axis_confidence,
            axis_reason,
        )
        scored.append(
            _target_eval_payload(target, label, confidence, reason, "bedside_pair")
        )
    labels = [str(item.get("label") or "unknown") for item in scored]
    if "fail" in labels:
        label, confidence = "fail", 0.88
    elif "degraded" in labels:
        label, confidence = "degraded", 0.78
    elif "unknown" in labels:
        label, confidence = "unknown", 0.0
    else:
        label, confidence = "pass", 0.90
    selected_ids = [item["target_id"] for item in scored]
    diagnostics = _fd_diagnostics_from_targets(scored, selected=selected_ids)
    diagnostics["cardinality_score"] = min(len(nightstands) / 2.0, 1.0)
    failed_ids = [
        item["target_id"] for item in scored if item.get("label") == "fail"
    ]
    if failed_ids:
        reason = f"bedside target(s) failed adjacency or front-axis alignment: {', '.join(failed_ids)}."
    else:
        reason = f"all {len(nightstands)} bedside target(s) satisfy adjacency and front-axis alignment."
    return (
        label,
        confidence,
        reason,
        diagnostics,
    )


def _eval_seating_to_surface(
    subject: dict[str, Any],
    target: dict[str, Any],
    _relation_type: str,
    *,
    topology_required: bool = False,
) -> tuple[str, float, str]:
    gap = bbox_gap_xy(subject, target)
    angle, angle_mode = seating_angle_to_target_deg(subject, target)
    if gap is None or angle is None:
        return "unknown", 0.0, "missing distance or orientation geometry."
    angle_note = ""
    if angle_mode == "depth_axis_fallback":
        angle_note = " using seating depth-axis fallback"
    elif angle_mode == "nearest_surface":
        angle_note = " using target-edge fallback"
    elif angle_mode == "nearest_long_surface":
        angle_note = " using nearest long table-edge fallback"
    elif angle_mode == "front_ray_surface":
        angle_note = " using front-facing table-edge fallback"
    elif angle_mode == "reversed_front_fallback":
        angle_note = " using flipped-front fallback"
    if not topology_required and not _is_actionable_seating_surface_pair(subject, target):
        return (
            "unknown",
            0.0,
            "chair can stand alone here; no nearby usable table or counter relation is required "
            f"(gap {gap:.2f}m, facing angle {angle:.0f}deg{angle_note}).",
        )
    living_room_pair = (
        object_category(subject) in {"armchair", "chair"}
        and object_category(target) == "coffee_table"
    )
    if _is_side_surface_target(target):
        if gap <= 0.55:
            return (
                "pass",
                0.88,
                f"seat is adjacent to a side surface with gap {gap:.2f}m.",
            )
        if gap <= 0.95:
            return (
                "degraded",
                0.72,
                f"side surface is usable but loose: gap {gap:.2f}m.",
            )
        return "fail", 0.8, f"side surface is too far from the seat: gap {gap:.2f}m."
    # 2026-07-16 修改原因：原规则先执行通用 close-pair <=110 度分支，导致
    # renders_003 中朝房间外侧、与茶几夹角约 90 度的扶手椅仍然 pass。
    # 客厅独立椅若按就近原则绑定茶几，应严格朝向该局部活动区。
    # 2026-07-16 修改原因：紧贴茶几侧边的座椅可以是侧向就座位，不能因
    # 目标边缘角度为 90 度就覆盖通用 close-pair 通过规则；只有存在活动间距时
    # 才要求客厅椅明确朝向茶几焦点。
    if living_room_pair and gap > 0.35:
        if gap <= 1.35 and angle <= 45.0:
            return (
                "pass",
                0.93,
                f"living-room chair faces its nearby coffee-table focus: gap {gap:.2f}m, "
                f"facing angle {angle:.0f}deg{angle_note}.",
            )
        if gap <= 1.35 and angle <= 110.0:
            return (
                "degraded",
                0.82,
                f"living-room chair points obliquely or outward from its nearby coffee-table focus: "
                f"gap {gap:.2f}m, facing angle {angle:.0f}deg{angle_note}.",
            )
        return (
            "fail",
            0.88,
            f"living-room chair does not face its nearby coffee-table focus: gap {gap:.2f}m, "
            f"facing angle {angle:.0f}deg{angle_note}.",
        )
    if gap <= 0.35 and angle <= 110.0:
        return (
            "pass",
            0.9,
            f"close seating pair with gap {gap:.2f}m and facing angle {angle:.0f}deg{angle_note}.",
        )
    if gap <= 0.35 and angle <= 150.0:
        return (
            "pass",
            0.86,
            f"seat is tight to the surface despite noisy yaw: gap {gap:.2f}m, facing angle {angle:.0f}deg{angle_note}.",
        )
    if gap <= 0.45 and angle <= 140.0:
        return (
            "pass",
            0.84,
            f"seat remains close enough for paired use: gap {gap:.2f}m, facing angle {angle:.0f}deg{angle_note}.",
        )
    if gap <= 0.8 and angle <= 100.0:
        return (
            "pass",
            0.88,
            f"paired seating relation is close and plausibly oriented: gap {gap:.2f}m, facing angle {angle:.0f}deg{angle_note}.",
        )
    if gap <= 1.3 and angle <= 75.0:
        return (
            "pass",
            0.86,
            f"seat is moderately spaced but well oriented: gap {gap:.2f}m, facing angle {angle:.0f}deg{angle_note}.",
        )
    if gap <= 1.2 and angle <= 75.0:
        return (
            "pass",
            0.9,
            f"gap {gap:.2f}m and facing angle {angle:.0f}deg{angle_note} support paired use.",
        )
    if angle > 150.0:
        return (
            "fail",
            0.85,
            f"subject is back-facing relative to the target: gap {gap:.2f}m, facing angle {angle:.0f}deg{angle_note}.",
        )
    if gap <= 1.0 and angle <= 125.0:
        return (
            "pass",
            0.82,
            f"relation is usable despite moderate spacing or yaw: gap {gap:.2f}m, facing angle {angle:.0f}deg{angle_note}.",
        )
    if gap <= 0.45 and angle <= 150.0:
        return (
            "degraded",
            0.78,
            f"close pair is usable but rotated: gap {gap:.2f}m, facing angle {angle:.0f}deg{angle_note}.",
        )
    if gap <= 1.8 and angle <= 110.0:
        return (
            "degraded",
            0.75,
            f"relation is weak: gap {gap:.2f}m, facing angle {angle:.0f}deg{angle_note}.",
        )
    return (
        "fail",
        0.85,
        f"target is too far or poorly oriented: gap {gap:.2f}m, facing angle {angle:.0f}deg{angle_note}.",
    )


def _is_required_seat_surface_assignment(check: dict[str, Any] | None) -> bool:
    evidence = (check or {}).get("evidence") or {}
    return bool(
        evidence.get("topology_required")
        and evidence.get("assignment_source") == ASSIGNMENT_SOURCE
    )


def _eval_assigned_seating_slot(
    subject: dict[str, Any], target: dict[str, Any], check: dict[str, Any]
) -> tuple[str, float, str]:
    evidence = check.get("evidence") or {}
    slot = evidence.get("target_slot") or {}
    raw_center = slot.get("center_xy") or []
    target_yaw = slot.get("yaw_deg")
    subject_center = bbox_center_xy(subject)
    if subject_center is None or len(raw_center) < 2 or target_yaw is None:
        return _eval_seating_to_surface(
            subject, target, "seating_to_work_surface", topology_required=True
        )
    slot_center = (float(raw_center[0]), float(raw_center[1]))
    center_error = math.hypot(
        subject_center[0] - slot_center[0], subject_center[1] - slot_center[1]
    )
    current_yaw = float(subject.get("yaw_deg") or 0.0)
    yaw_error = abs((current_yaw - float(target_yaw) + 180.0) % 360.0 - 180.0)
    size = (subject.get("bbox_world") or {}).get("size") or []
    footprint_scale = (
        min(float(size[0]), float(size[1])) if len(size) >= 2 else 0.5
    )
    position_tolerance = max(0.25, 0.65 * footprint_scale)
    if center_error <= position_tolerance and yaw_error <= 30.0:
        label, confidence = "pass", 0.95
    elif center_error <= 1.75 * position_tolerance and yaw_error <= 60.0:
        label, confidence = "degraded", 0.82
    else:
        label, confidence = "fail", 0.93
    return (
        label,
        confidence,
        f"assigned one-to-one slot at [{slot_center[0]:.2f}, {slot_center[1]:.2f}] "
        f"with yaw {float(target_yaw):.1f}deg; center error {center_error:.2f}m and "
        f"yaw error {yaw_error:.0f}deg.",
    )


def _eval_facing_relation(
    subject: dict[str, Any], target: dict[str, Any], _relation_type: str
) -> tuple[str, float, str]:
    dist = distance_xy(subject, target)
    angle = angle_to_target_deg(subject, target)
    if dist is None or angle is None:
        return "unknown", 0.0, "missing distance or orientation geometry."
    living_room_media = object_category(
        subject
    ) in LIVING_ROOM_SEATING and _is_media_target(target)
    wall_mounted_media = _is_wall_mounted_media_target(target)
    media_front_angle = (
        angle_to_target_deg(target, subject) if wall_mounted_media else None
    )
    if wall_mounted_media and media_front_angle is None:
        return "unknown", 0.0, "missing wall-mounted media front orientation geometry."

    # 2026-07-14 修改原因：seating_to_media 原先只检查“座椅 front -> TV”，
    # 会把屏幕背向座椅或明显偏离观看区域的壁挂电视判为 pass。壁挂媒体还
    # 必须满足“TV front -> 座椅”方向；普通电视柜/非壁挂媒体继续使用原有
    # 允许斜向观看的规则，避免把正常的家具布局误判为失败。
    if wall_mounted_media and media_front_angle is not None:
        if media_front_angle > 45.0:
            media_alignment_reason = (
                "wall-mounted media front faces away from the seat"
                if media_front_angle > 90.0
                else "wall-mounted media is not directly aligned with the seat"
            )
            return (
                "fail",
                0.88,
                f"distance {dist:.2f}m and seating facing angle {angle:.0f}deg, "
                f"but {media_alignment_reason} "
                f"({media_front_angle:.0f}deg).",
            )

    if 0.8 <= dist <= 5.5 and angle <= 65.0:
        label, confidence, reason = (
            "pass",
            0.9,
            f"distance {dist:.2f}m and facing angle {angle:.0f}deg support viewing/use.",
        )
    elif living_room_media and 0.8 <= dist <= 5.5 and angle <= 100.0:
        label, confidence, reason = (
            "pass",
            0.84,
            f"living-room seating has usable media view: distance {dist:.2f}m, "
            f"angle {angle:.0f}deg.",
        )
    elif living_room_media and dist <= 6.5 and angle <= 125.0:
        label, confidence, reason = (
            "degraded",
            0.72,
            f"media relation is usable but oblique: distance {dist:.2f}m, "
            f"angle {angle:.0f}deg.",
        )
    elif dist <= 6.5 and angle <= 100.0:
        label, confidence, reason = (
            "degraded",
            0.75,
            f"relation is usable but weak: distance {dist:.2f}m, angle {angle:.0f}deg.",
        )
    else:
        label, confidence, reason = (
            "fail",
            0.85,
            f"relation does not support use: distance {dist:.2f}m, angle {angle:.0f}deg.",
        )

    if wall_mounted_media and media_front_angle is not None:
        if media_front_angle > 35.0:
            return (
                "degraded" if label != "fail" else label,
                min(confidence, 0.78),
                f"{reason.rstrip('.')} but media is not directly aligned with the "
                f"seat ({media_front_angle:.0f}deg).",
            )
        return (
            label,
            confidence,
            f"{reason.rstrip('.')}. wall-mounted media front angle "
            f"{media_front_angle:.0f}deg.",
        )
    return label, confidence, reason


def _is_wall_mounted_media_target(target: dict[str, Any]) -> bool:
    if not _is_media_target(target):
        return False
    hints = target.get("functional_hints") or {}
    object_types = {
        str(value).strip().lower().replace("-", "_").replace(" ", "_")
        for value in (hints.get("scene_object_type"), target.get("object_type"))
        if value
    }
    return bool(object_types & {"wall_mounted", "mounted"})


def _eval_bed_to_nightstand(
    subject: dict[str, Any], target: dict[str, Any], _relation_type: str
) -> tuple[str, float, str]:
    gap = bbox_gap_xy(subject, target)
    if gap is None:
        return "unknown", 0.0, "missing distance geometry."
    if gap <= 0.45:
        return "pass", 0.9, f"nightstand is adjacent to bed with {gap:.2f}m bbox gap."
    if gap <= 0.9:
        return (
            "degraded",
            0.75,
            f"nightstand is nearby but not tight to the bed: {gap:.2f}m gap.",
        )
    return "fail", 0.85, f"nightstand is too far from the bed: {gap:.2f}m gap."


BEDSIDE_PARALLEL_MAX_ANGLE_DEG = 20.0


def _eval_bedside_axis_alignment(
    bed: dict[str, Any], nightstand: dict[str, Any]
) -> tuple[str, float, str]:
    """Check bedside front axes as unoriented axes, allowing 0 or 180 degrees."""
    # 2026-07-10 修改原因：床头柜应与床保持同一 front 轴，而不是朝向床中心；
    # FD 规则需要稳定地识别 0/180 度同轴布局，避免 LLM 触发错误旋转。
    bed_front = front_vector(bed)
    nightstand_front = front_vector(nightstand)
    dot = max(
        -1.0,
        min(1.0, abs(bed_front[0] * nightstand_front[0] + bed_front[1] * nightstand_front[1])),
    )
    angle = math.degrees(math.acos(dot))
    if angle <= BEDSIDE_PARALLEL_MAX_ANGLE_DEG:
        return "pass", 0.90, f"front axes are parallel within {angle:.0f}deg."
    if angle <= 45.0:
        return "degraded", 0.70, f"front axes are mildly misaligned by {angle:.0f}deg."
    return "fail", 0.88, f"front axes are not parallel: {angle:.0f}deg apart."


def _combine_bedside_results(
    distance_label: str,
    distance_confidence: float,
    distance_reason: str,
    axis_label: str,
    axis_confidence: float,
    axis_reason: str,
) -> tuple[str, float, str]:
    """Combine required bedside distance and axis constraints conservatively."""
    if "fail" in {distance_label, axis_label}:
        label = "fail"
    elif "degraded" in {distance_label, axis_label}:
        label = "degraded"
    else:
        label = "pass"
    confidence = min(distance_confidence, axis_confidence)
    return label, confidence, f"{distance_reason} {axis_reason}"


def _eval_annotated_dependency_relation(
    subject: dict[str, Any],
    target: dict[str, Any],
    relation_type: str,
    check: dict[str, Any] | None,
) -> tuple[str, float, str]:
    dependency = _dependency_payload(check)
    if relation_type == "back_against_wall":
        return _eval_face_against_wall(
            subject,
            target,
            dependency,
            candidate_faces=("back",),
            allow_distance_degraded=False,
            relation_type=relation_type,
        )
    if relation_type == "side_or_back_against_wall":
        return _eval_face_against_wall(
            subject,
            target,
            dependency,
            candidate_faces=("back", "left", "right"),
            allow_distance_degraded=False,
            relation_type=relation_type,
        )
    if relation_type == "mounted_to_wall":
        return _eval_face_against_wall(
            subject,
            target,
            dependency,
            candidate_faces=("back", "front"),
            relation_type=relation_type,
        )
    if relation_type == "mounted_to_ceiling":
        return _eval_vertical_attachment(
            subject, target, dependency, relation_type=relation_type
        )
    if relation_type == "object_on_floor":
        return _eval_vertical_attachment(
            subject, target, dependency, relation_type=relation_type
        )
    if relation_type == "furniture_faces_furniture":
        return _eval_face_to_target(subject, target, dependency, relation_type)
    if relation_type == "computer_peripheral_faces_screen":
        return _eval_face_to_target(subject, target, dependency, relation_type)
    if relation_type == "display_faces_user":
        return _eval_face_to_target(subject, target, dependency, relation_type)
    return _eval_generic_near_relation(subject, target, relation_type)


def _dependency_payload(check: dict[str, Any] | None) -> dict[str, Any]:
    evidence = (check or {}).get("evidence") or {}
    dependency = evidence.get("dependency") if isinstance(evidence, dict) else None
    return dict(dependency) if isinstance(dependency, dict) else {}


def _eval_face_against_wall(
    subject: dict[str, Any],
    target: dict[str, Any],
    dependency: dict[str, Any],
    *,
    candidate_faces: tuple[str, ...],
    allow_distance_degraded: bool = True,
    relation_type: str = "back_against_wall",
) -> tuple[str, float, str]:
    if object_category(target) != "wall":
        return "fail", 0.3, "target is not a wall architecture object."
    gap = bbox_gap_xy(subject, target)
    if gap is None:
        return "unknown", 0.0, "missing wall distance geometry."
    max_distance = _dependency_float(dependency, "max_distance_m", 0.25)
    max_angle = _dependency_float(dependency, "max_angle_deg", 45.0)
    requested_face = str(dependency.get("subject_face") or "").strip().lower()
    faces = (requested_face,) if requested_face in candidate_faces else candidate_faces

    scored: list[tuple[float, str]] = []
    for face in faces:
        # 2026-07-11 修改原因：长墙不能用“家具中心 -> 整面墙中心”的斜向量
        # 评估 back_against_wall。沿同一墙不同位置、yaw 完全相同的两把椅子会
        # 因此得到不同角度；墙关系必须只比较该墙的局部法线。
        angle = _face_angle_to_wall_deg(subject, target, face)
        if angle is not None:
            scored.append((angle, face))
    if not scored:
        return "unknown", 0.0, "missing wall orientation geometry."
    best_angle, best_face = min(scored, key=lambda item: item[0])
    if gap <= max_distance and best_angle <= max_angle:
        return (
            "pass",
            0.9,
            f"{best_face} face is wall-aligned: gap {gap:.2f}m, angle {best_angle:.0f}deg.",
        )
    if gap <= max_distance and best_angle <= max_angle + 25.0:
        return (
            "degraded",
            0.72,
            f"{best_face} face is near wall but loose: gap {gap:.2f}m, angle {best_angle:.0f}deg.",
        )
    if (
        allow_distance_degraded
        and gap <= max_distance * 1.5
        and best_angle <= max_angle + 25.0
    ):
        return (
            "degraded",
            0.72,
            f"{best_face} face is near wall but loose: gap {gap:.2f}m, angle {best_angle:.0f}deg.",
        )
    thin_mount_result = _eval_thin_wall_mounted_contact(
        subject, target, gap=gap, max_distance=max_distance
    )
    if thin_mount_result is not None:
        return thin_mount_result
    footprint_result = _eval_wall_contact_footprint_fallback(
        subject, target, gap=gap, max_distance=max_distance, relation_type=relation_type
    )
    if footprint_result is not None:
        return footprint_result
    return (
        "fail",
        0.84,
        f"no allowed face is backed by the wall: gap {gap:.2f}m, best {best_face} angle {best_angle:.0f}deg.",
    )


def _eval_wall_contact_footprint_fallback(
    subject: dict[str, Any],
    target: dict[str, Any],
    *,
    gap: float,
    max_distance: float,
    relation_type: str,
) -> tuple[str, float, str] | None:
    # 2026-07-08 修改原因：bookshelf/floating shelf 等资产的 front/back
    # 有时与实际贴墙短轴不一致；用 AABB 短轴只兜底存储/墙挂类贴墙物。
    if object_category(target) != "wall" or gap > max_distance:
        return None
    wall_axis = _wall_normal_axis(target)
    subject_size = _bbox_size_xyz(subject)
    if wall_axis is None or subject_size is None:
        return None
    normal_index = 0 if wall_axis == "x" else 1
    normal_span = subject_size[normal_index]
    parallel_span = subject_size[1 - normal_index]
    if normal_span <= 0.0 or parallel_span <= 0.0:
        return None
    scene_type = _scene_object_type(subject)
    if relation_type == "mounted_to_wall" and _is_projecting_wall_mount(subject):
        if _footprint_has_wall_contact_shape(normal_span, parallel_span, limit_m=0.35):
            return (
                "pass",
                0.86,
                (
                    "wall-mounted footprint is flush with the wall: "
                    f"gap {gap:.2f}m, {wall_axis}-axis projection {normal_span:.2f}m."
                ),
            )
    if (
        relation_type in {"back_against_wall", "side_or_back_against_wall"}
        and scene_type == "furniture"
        and _is_storage_or_work_wall_backed(subject)
    ):
        if _footprint_has_wall_contact_shape(normal_span, parallel_span, limit_m=0.45):
            return (
                "pass",
                0.84,
                (
                    "storage/work furniture footprint is flush with the wall: "
                    f"gap {gap:.2f}m, {wall_axis}-axis depth {normal_span:.2f}m."
                ),
            )
    return None


def _footprint_has_wall_contact_shape(
    normal_span: float, parallel_span: float, *, limit_m: float
) -> bool:
    return (
        normal_span <= max(limit_m, parallel_span * 0.45)
        and parallel_span >= normal_span * 1.5
    )


def _is_projecting_wall_mount(subject: dict[str, Any]) -> bool:
    if _scene_object_type(subject) != "wall_mounted":
        return False
    category = object_category(subject)
    group = _category_group(subject)
    profile = object_function_profile(subject)
    return (
        category in MEDIA
        or category in {"mirror", "wall_mirror", "shelf", "wall_shelf", "wall_art"}
        or group in {"media", "storage", "storage_surface", "work_surface"}
        or profile.can_support_top
    )


def _is_storage_or_work_wall_backed(subject: dict[str, Any]) -> bool:
    category = object_category(subject)
    group = _category_group(subject)
    if category in SEATING or category in BEDS:
        return False
    profile = object_function_profile(subject)
    return (
        category in SUPPORTS
        or category in WORK_SURFACES
        or group in {"storage", "storage_surface", "work_surface"}
        or profile.has_internal_shelf
        or profile.is_work_surface
    )


def _eval_thin_wall_mounted_contact(
    subject: dict[str, Any],
    target: dict[str, Any],
    *,
    gap: float,
    max_distance: float,
) -> tuple[str, float, str] | None:
    # 2026-07-08 修改原因：墙画/钟/电视等薄墙挂物的 canonical front/back
    # 有时与实际贴墙薄轴不一致；仅对 wall_mounted 薄物体启用 AABB 薄轴兜底。
    if (
        _scene_object_type(subject) != "wall_mounted"
        or object_category(target) != "wall"
    ):
        return None
    if gap > max_distance:
        return None
    wall_axis = _wall_normal_axis(target)
    subject_size = _bbox_size_xyz(subject)
    if wall_axis is None or subject_size is None:
        return None
    normal_index = 0 if wall_axis == "x" else 1
    normal_span = subject_size[normal_index]
    in_plane_span = max(subject_size[1 - normal_index], subject_size[2])
    if normal_span <= 0.0 or in_plane_span <= 0.0:
        return None
    if normal_span > max(0.12, in_plane_span * 0.12):
        return None
    return (
        "pass",
        0.86,
        (
            "thin wall-mounted footprint is flush with the wall: "
            f"gap {gap:.2f}m, {wall_axis}-axis thickness {normal_span:.2f}m."
        ),
    )


def _wall_normal_axis(target: dict[str, Any]) -> str | None:
    size = _bbox_size_xyz(target)
    if size is None:
        return None
    sx, sy, _sz = size
    if sx <= 0.0 or sy <= 0.0:
        return None
    if sx <= sy * 0.25:
        return "x"
    if sy <= sx * 0.25:
        return "y"
    return None


def _bbox_size_xyz(obj: dict[str, Any]) -> tuple[float, float, float] | None:
    size = ((obj.get("bbox_world") or {}).get("size")) or []
    if not isinstance(size, list) or len(size) < 3:
        return None
    try:
        return (
            abs(float(size[0])),
            abs(float(size[1])),
            abs(float(size[2])),
        )
    except (TypeError, ValueError):
        return None


def _eval_face_to_target(
    subject: dict[str, Any],
    target: dict[str, Any],
    dependency: dict[str, Any],
    relation_type: str,
) -> tuple[str, float, str]:
    gap = bbox_gap_xy(subject, target)
    if gap is None:
        return "unknown", 0.0, "missing distance geometry."
    max_distance = _dependency_float(dependency, "max_distance_m", 1.8)
    max_angle = _dependency_float(dependency, "max_angle_deg", 60.0)
    subject_face = str(dependency.get("subject_face") or "front").strip().lower()
    target_face = str(dependency.get("target_face") or "any").strip().lower()
    subject_angle = _face_angle_to_target_deg(subject, target, subject_face)
    if subject_angle is None:
        return "unknown", 0.0, "missing subject orientation geometry."

    target_angle: float | None = None
    if target_face not in {"", "any", "none", "null"}:
        target_angle = _face_angle_to_target_deg(target, subject, target_face)

    target_clause = ""
    target_ok = True
    if target_angle is not None:
        target_ok = target_angle <= max_angle
        target_clause = f", target {target_face} angle {target_angle:.0f}deg"

    if gap <= max_distance and subject_angle <= max_angle and target_ok:
        return (
            "pass",
            0.88,
            f"`{relation_type}` holds: gap {gap:.2f}m, subject {subject_face} angle {subject_angle:.0f}deg{target_clause}.",
        )
    if gap <= max_distance * 1.25 and subject_angle <= max_angle + 30.0:
        return (
            "degraded",
            0.72,
            f"`{relation_type}` is weak: gap {gap:.2f}m, subject {subject_face} angle {subject_angle:.0f}deg{target_clause}.",
        )
    return (
        "fail",
        0.84,
        f"`{relation_type}` fails: gap {gap:.2f}m, subject {subject_face} angle {subject_angle:.0f}deg{target_clause}.",
    )


def _eval_vertical_attachment(
    subject: dict[str, Any],
    target: dict[str, Any],
    dependency: dict[str, Any],
    *,
    relation_type: str,
) -> tuple[str, float, str]:
    subject_span = bbox_height_span(subject)
    target_span = bbox_height_span(target)
    if subject_span is None or target_span is None:
        return "unknown", 0.0, "missing vertical attachment geometry."
    max_distance = _dependency_float(dependency, "max_distance_m", 0.12)
    if relation_type == "object_on_floor":
        if object_category(target) != "floor":
            return "fail", 0.3, "target is not a floor architecture object."
        vertical_gap = abs(subject_span[0] - target_span[1])
        label_target = "floor"
    else:
        if object_category(target) != "ceiling":
            return "fail", 0.3, "target is not a ceiling architecture object."
        vertical_gap = abs(target_span[0] - subject_span[1])
        label_target = "ceiling"
    if vertical_gap <= max_distance:
        return (
            "pass",
            0.88,
            f"object is attached to {label_target}: vertical gap {vertical_gap:.2f}m.",
        )
    if vertical_gap <= max_distance * 2.0:
        return (
            "degraded",
            0.7,
            f"object is close to {label_target} but not tight: vertical gap {vertical_gap:.2f}m.",
        )
    return (
        "fail",
        0.84,
        f"object is not attached to {label_target}: vertical gap {vertical_gap:.2f}m.",
    )


def _face_angle_to_target_deg(
    subject: dict[str, Any], target: dict[str, Any], face: str
) -> float | None:
    sc = bbox_center_xy(subject)
    tc = bbox_center_xy(target)
    if sc is None or tc is None:
        return None
    axis = _face_axis(subject, face)
    if axis is None:
        return None
    tx, ty = tc[0] - sc[0], tc[1] - sc[1]
    norm = (tx * tx + ty * ty) ** 0.5
    if norm <= 1e-6:
        return 0.0
    dot = max(-1.0, min(1.0, (axis[0] * tx + axis[1] * ty) / norm))
    return abs(math.degrees(math.acos(dot)))


def _face_angle_to_wall_deg(
    subject: dict[str, Any], target: dict[str, Any], face: str
) -> float | None:
    axis = _face_axis(subject, face)
    subject_center = bbox_center_xy(subject)
    wall_center = bbox_center_xy(target)
    wall_axis = _wall_normal_axis(target)
    if axis is None or subject_center is None or wall_center is None:
        return None
    if wall_axis == "x":
        wall_direction = (1.0 if wall_center[0] >= subject_center[0] else -1.0, 0.0)
    elif wall_axis == "y":
        wall_direction = (0.0, 1.0 if wall_center[1] >= subject_center[1] else -1.0)
    else:
        return _face_angle_to_target_deg(subject, target, face)
    dot = max(
        -1.0,
        min(1.0, axis[0] * wall_direction[0] + axis[1] * wall_direction[1]),
    )
    return abs(math.degrees(math.acos(dot)))


def _face_axis(obj: dict[str, Any], face: str) -> tuple[float, float] | None:
    face = str(face or "").strip().lower()
    fx, fy = front_vector(obj)
    sx, sy = side_vector(obj)
    if face == "front":
        return fx, fy
    if face == "back":
        return -fx, -fy
    if face == "left":
        return sx, sy
    if face == "right":
        return -sx, -sy
    return None


def _dependency_float(dependency: dict[str, Any], key: str, default: float) -> float:
    try:
        return float(dependency.get(key))
    except (TypeError, ValueError):
        return default


def _eval_generic_near_relation(
    subject: dict[str, Any], target: dict[str, Any], relation_type: str
) -> tuple[str, float, str]:
    gap = bbox_gap_xy(subject, target)
    if gap is None:
        return "unknown", 0.0, "missing distance geometry."
    if gap <= 0.6:
        return (
            "pass",
            0.75,
            f"generic `{relation_type}` target is nearby with {gap:.2f}m gap.",
        )
    if gap <= 1.2:
        return (
            "degraded",
            0.65,
            f"generic `{relation_type}` target is somewhat far with {gap:.2f}m gap.",
        )
    return (
        "fail",
        0.75,
        f"generic `{relation_type}` target is too far with {gap:.2f}m gap.",
    )


def _preferred_relations_for_subject(subject: dict[str, Any]) -> list[str]:
    profile = object_function_profile(subject)
    if profile.source == "explicit" and profile.is_seating:
        subject_category = object_category(subject)
        if subject_category in LIVING_ROOM_SEATING:
            return ["seating_to_media", "seating_to_work_surface"]
        return ["seating_to_work_surface", "seating_to_media"]
    if profile.source == "explicit" and profile.is_sleeping_surface:
        return ["bed_to_nightstand"]
    if (
        profile.source == "explicit"
        and profile.is_small_placeable
        and not (
            profile.can_support_top
            or profile.has_internal_shelf
            or profile.is_work_surface
            or profile.is_media_target
        )
    ):
        return ["object_on_support"]
    subject_category = object_category(subject)
    if subject_category in SEATING and not _is_seating_subject(subject):
        return []
    if subject_category in LIVING_ROOM_SEATING:
        return ["seating_to_media", "seating_to_work_surface"]
    if subject_category in SEATING:
        return ["seating_to_work_surface", "seating_to_media"]
    if subject_category in BEDS:
        return ["bed_to_nightstand"]
    if _is_lamp_subject(subject):
        return ["lamp_to_surface"]
    if _is_supported_small_subject(subject):
        return ["object_on_support"]
    return []


def _best_template_target(
    subject: dict[str, Any],
    relation_type: str,
    objects: list[dict[str, Any]],
) -> dict[str, Any] | None:
    from scenesmith.scenebenchmark_critic.metrics.functional_dependency.proposer import (
        _rank_targets_for_relation,
    )

    target_by_id = {
        str(target.get("id") or ""): target for target in objects if target.get("id")
    }
    ranked_ids = _rank_targets_for_relation(subject, relation_type, objects)
    if not ranked_ids:
        return None
    return target_by_id[ranked_ids[0]]


def _angle_penalty(
    subject: dict[str, Any], target: dict[str, Any], relation_type: str
) -> float:
    if relation_type not in {"seating_to_work_surface", "seating_to_media"}:
        return 0.0
    angle = angle_to_target_deg(subject, target)
    if angle is None:
        return 0.0
    return max(angle - 60.0, 0.0) / 60.0


def _relation_target_is_valid(
    subject: dict[str, Any], target: dict[str, Any], relation_type: str
) -> bool:
    relation_type = _normalize_relation_type(relation_type)
    if relation_type == "seat_faces_surface":
        return _is_seating_subject(subject) and _is_work_surface_target(target)
    if relation_type == "furniture_faces_furniture":
        return _is_directional_facing_subject(subject) and _is_facing_relation_target(
            target
        )
    if relation_type in {
        "back_against_wall",
        "side_or_back_against_wall",
        "mounted_to_wall",
    }:
        return object_category(target) == "wall"
    if relation_type == "mounted_to_ceiling":
        return object_category(target) == "ceiling"
    if relation_type == "object_on_floor":
        return object_category(target) == "floor"
    if relation_type == "display_faces_user":
        return object_category(target) not in {"wall", "floor", "ceiling"}
    if relation_type == "computer_peripheral_faces_screen":
        return _is_computer_peripheral_subject(subject) and _is_computer_screen_target(
            target
        )
    if relation_type == "seating_to_work_surface":
        return _is_seating_subject(subject) and _is_work_surface_target(target)
    if relation_type == "seating_to_media":
        return _is_seating_subject(subject) and _is_media_target(target)
    if relation_type == "bed_to_nightstand":
        profile = object_function_profile(subject)
        return (
            object_category(subject) in BEDS
            or (profile.source == "explicit" and profile.is_sleeping_surface)
        ) and _is_nightstand_target(target)
    if relation_type == "object_on_support":
        return (
            _is_supported_small_subject(subject) and _is_primary_support_target(target)
        ) or _is_soft_furnishing_seating_support_pair(
            subject, target
        )
    if relation_type == "lamp_to_surface":
        return _is_lamp_subject(subject) and _is_lamp_surface_target(target)
    if relation_type == "dining_set":
        return (
            object_category(subject) in DINING_TABLES and _is_seating_subject(target)
        ) or (_is_seating_subject(subject) and object_category(target) in DINING_TABLES)
    if relation_type == "workstation":
        return (
            _is_work_surface_target(subject)
            or _is_work_surface_target(target)
            or _is_seating_subject(target)
        )
    if relation_type == "bedside_pair":
        profile = object_function_profile(subject)
        return (
            object_category(subject) in BEDS
            or (profile.source == "explicit" and profile.is_sleeping_surface)
        ) and _is_nightstand_target(target)
    if relation_type == "generic_near_relation":
        return True
    return False


def _is_soft_furnishing_seating_support_pair(
    subject: dict[str, Any], target: dict[str, Any]
) -> bool:
    # 2026-07-13 修改原因：primary support 默认排除 seating，适合杯盘等硬质
    # 小物，但抱枕、靠垫、毯子本来就应由 sofa/chair 的座面支撑。
    if _scene_object_type(subject) != "manipuland":
        return False
    if not _category_token_has_any(
        subject, ("pillow", "cushion", "bolster", "blanket", "throw")
    ):
        return False
    return object_category(target) in SEATING or _category_group(target) == "seating"
