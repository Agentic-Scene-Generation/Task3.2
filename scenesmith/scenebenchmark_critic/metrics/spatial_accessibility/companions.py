"""Cross-metric companion annotations for spatial accessibility."""

from __future__ import annotations

from typing import Any

from scenesmith.scenebenchmark_critic.metrics.functional_dependency.semantics import (
    _is_actionable_seating_surface_pair,
    _is_seating_subject,
)
from scenesmith.scenebenchmark_critic.intent_contract import (
    bound_ids,
    contract_constraints,
)
from scenesmith.scenebenchmark_critic.core.geometry import is_small_object


_THIN_COVERING_CATEGORIES = {
    "area_rug",
    "carpet",
    "floor_covering",
    "floor_mat",
    "mat",
    "rug",
}


def attach_expected_access_companions(
    case_pack: dict[str, Any], objects: dict[str, dict[str, Any]]
) -> None:
    """Exclude explicitly paired seats from their surface's access obstacles.

    A seat is an expected occupant of the access zone only when a generated
    dependency or the original intent contract binds that same seat to the same
    target. This covers explicit seating cohorts without authorizing an
    unrelated nearby chair.
    """
    companions_by_surface: dict[str, set[str]] = {}
    for check in case_pack.get("checks") or []:
        if not isinstance(check, dict):
            continue
        if (
            check.get("metric") != "functional_dependency"
            or check.get("relation_type") != "seating_to_work_surface"
        ):
            continue
        seat_id = str(check.get("subject_id") or "")
        seat = objects.get(seat_id)
        if seat is None:
            continue
        for target_id in check.get("target_ids") or []:
            surface_id = str(target_id or "")
            surface = objects.get(surface_id)
            if surface is not None and _is_actionable_seating_surface_pair(
                seat, surface
            ):
                companions_by_surface.setdefault(surface_id, set()).add(seat_id)

    for seat_id, surface_id in _contract_companion_pairs(case_pack, objects):
        companions_by_surface.setdefault(surface_id, set()).add(seat_id)

    for check in case_pack.get("checks") or []:
        if not isinstance(check, dict):
            continue
        if check.get("metric") != "spatial_accessibility":
            continue
        subject_id = str(check.get("subject_id") or "")
        companion_ids = companions_by_surface.get(subject_id)
        if companion_ids:
            existing_ids = {
                str(item)
                for item in (check.get("expected_companion_ids") or [])
                if str(item)
            }
            check["expected_companion_ids"] = sorted(existing_ids | companion_ids)


def attach_expected_clearance_companions(
    case_pack: dict[str, Any], objects: dict[str, dict[str, Any]]
) -> None:
    """Apply contract-bound topology to interaction-clearance checks.

    A companion is ignored only in its anchor's zone.  Its own clearance check
    and unrelated blockers remain authoritative.
    """
    companion_by_anchor: dict[str, set[str]] = {}
    for subject_id, target_id in _contract_companion_pairs(case_pack, objects):
        companion_by_anchor.setdefault(target_id, set()).add(subject_id)

    object_rows = list(objects.values())
    direct_access_ids = {
        object_id
        for constraint in contract_constraints(
            case_pack, relations=("clear_access",), include_auxiliary=False
        )
        for object_id in bound_ids(constraint.get("subjects"), object_rows)
    }
    for check in case_pack.get("checks") or []:
        if (
            not isinstance(check, dict)
            or check.get("metric") != "interaction_clearance"
        ):
            continue
        subject_id = str(check.get("subject_id") or "")
        subject = objects.get(subject_id)
        clearance = check.get("clearance_result")
        if subject is None or not isinstance(clearance, dict):
            continue
        companions = companion_by_anchor.get(subject_id, set())
        if companions:
            _remove_clearance_companion_intrusions(clearance, companions)
            check["expected_companion_ids"] = sorted(companions)
        if subject_id in direct_access_ids or not _default_no_independent_access(
            subject
        ):
            continue
        check["scoring_tier"] = "ignored"
        clearance["label"] = "pass"
        clearance["blocking_objects"] = []
        clearance["intrusions"] = []
        clearance["clearance_policy"] = "no_independent_access_zone"


def _contract_companion_pairs(
    case_pack: dict[str, Any], objects: dict[str, dict[str, Any]]
) -> set[tuple[str, str]]:
    rows = list(objects.values())
    pairs = set(_hard_bound_seating_pairs(case_pack, objects))
    for constraint in contract_constraints(
        case_pack,
        relations=("paired_with", "flanking", "edge_distribution", "surround"),
        include_auxiliary=False,
    ):
        if str(constraint.get("strength") or "hard").lower() != "hard":
            continue
        subject_ids = bound_ids(constraint.get("subjects"), rows)
        target_ids = bound_ids(constraint.get("targets"), rows)
        if len(target_ids) != 1:
            continue
        for subject_id in subject_ids:
            subject = objects.get(subject_id)
            if subject is None or not _is_seating_subject(subject):
                continue
            target_id = target_ids[0]
            target = objects.get(target_id)
            if target is not None and _is_actionable_seating_surface_pair(
                subject, target
            ):
                pairs.add((subject_id, target_id))
    return pairs


def _remove_clearance_companion_intrusions(
    clearance: dict[str, Any], companions: set[str]
) -> None:
    blockers = [
        str(item)
        for item in clearance.get("blocking_objects") or []
        if str(item) not in companions
    ]
    clearance["blocking_objects"] = sorted(set(blockers))
    clearance["intrusions"] = [
        intrusion
        for intrusion in clearance.get("intrusions") or []
        if str(intrusion.get("object_id") or "") not in companions
    ]
    if not blockers:
        clearance["label"] = "pass"


def _default_no_independent_access(subject: dict[str, Any]) -> bool:
    hints = subject.get("functional_hints")
    if not isinstance(hints, dict):
        hints = {}
    metadata = subject.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    # Explicit access metadata overrides the default inheritance from a
    # support surface.  A supported small object can still require direct
    # human access when its affordance says so.
    for key in (
        "independent_access_required",
        "requires_independent_access",
        "direct_access_required",
        "requires_direct_access",
    ):
        if bool(hints.get(key)) or bool(metadata.get(key)) or bool(subject.get(key)):
            return False
    category = str(
        subject.get("category_norm") or subject.get("category") or ""
    ).lower()
    if category in _THIN_COVERING_CATEGORIES:
        return True
    placement = subject.get("placement_info")
    return bool(
        str(subject.get("object_type") or "").lower() == "manipuland"
        and isinstance(placement, dict)
        and placement.get("parent_surface_id")
        and is_small_object(subject)
    )


def _hard_bound_seating_pairs(
    case_pack: dict[str, Any], objects: dict[str, dict[str, Any]]
) -> set[tuple[str, str]]:
    """Return unambiguous prompt-bound seating-to-surface pairs.

    The target must resolve to exactly one physical object for each relation.
    This keeps a broad group statement (for example, chairs facing a table
    group) from silently excluding arbitrary obstacles from spatial checks.
    """
    object_rows = list(objects.values())
    pairs_by_relation: dict[str, set[tuple[str, str]]] = {
        "in_front_of": set(),
        "faces": set(),
    }
    for relation, pairs in pairs_by_relation.items():
        for constraint in contract_constraints(
            case_pack,
            relations=(relation,),
            include_auxiliary=False,
        ):
            if str(constraint.get("strength") or "hard").lower() != "hard":
                continue
            target_ids = bound_ids(constraint.get("targets"), object_rows)
            if len(target_ids) != 1:
                continue
            target_id = target_ids[0]
            for seat_id in bound_ids(constraint.get("subjects"), object_rows):
                seat = objects.get(seat_id)
                if seat is not None and _is_seating_subject(seat):
                    pairs.add((seat_id, target_id))
    return pairs_by_relation["in_front_of"] & pairs_by_relation["faces"]
