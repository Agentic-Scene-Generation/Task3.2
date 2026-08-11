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


def attach_expected_access_companions(
    case_pack: dict[str, Any], objects: dict[str, dict[str, Any]]
) -> None:
    """Exclude explicitly paired seats from their surface's access obstacles.

    A seat is an expected occupant of the access zone only when the original
    intent contract binds that same seat to the same target through both hard
    ``in_front_of`` and ``faces`` relations.  This extends the existing
    inferred desk/table pairing to purpose-specific surfaces such as a
    dressing table, without authorizing an unrelated nearby chair.
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

    for seat_id, surface_id in _hard_bound_seating_pairs(case_pack, objects):
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
