"""Cross-metric companion annotations for spatial accessibility."""

from __future__ import annotations

from typing import Any

from scenesmith.scenebenchmark_critic.metrics.functional_dependency.semantics import (
    _is_actionable_seating_surface_pair,
)


def attach_expected_access_companions(
    case_pack: dict[str, Any], objects: dict[str, dict[str, Any]]
) -> None:
    """Exclude a seat's paired work surface from its accessibility obstacles."""
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

    for check in case_pack.get("checks") or []:
        if not isinstance(check, dict):
            continue
        if check.get("metric") != "spatial_accessibility":
            continue
        subject_id = str(check.get("subject_id") or "")
        companion_ids = companions_by_surface.get(subject_id)
        if companion_ids:
            check["expected_companion_ids"] = sorted(companion_ids)
