"""Normalize deterministic critic repair ownership in one place."""

from __future__ import annotations

from typing import Any

from scenesmith.scenebenchmark_critic.relation_registry import STAGE_ORDER


_VALID_STAGE_OWNERS = frozenset(stage for stage in STAGE_ORDER if stage != "final")
_METRIC_STAGE_OWNERS = {
    "spatial_accessibility": "furniture",
    "interaction_clearance": "furniture",
}
_CHECK_PREFIX_STAGE_OWNERS = {
    "wall_visibility__": "wall_mounted",
    "wall_visual_clearance__": "wall_mounted",
    "ceiling_visual_clearance__": "ceiling_mounted",
    "manipuland_support__": "manipuland",
}


def _stage_owner(value: Any) -> str:
    aliases = {"wall": "wall_mounted", "ceiling": "ceiling_mounted"}
    owner = aliases.get(str(value or "").strip().lower(), str(value or "").strip())
    return owner if owner in _VALID_STAGE_OWNERS else ""


def _explicit_object_stage_owners(
    diagnostics: dict[str, Any], evidence: dict[str, Any]
) -> set[str]:
    """Use declared object-stage evidence; never infer collision ownership by name."""
    owners: set[str] = set()
    for mapping in (diagnostics, evidence):
        for key in (
            "object_stage",
            "primary_object_stage",
            "target_stage",
            "subject_stage",
        ):
            owner = _stage_owner(mapping.get(key))
            if owner:
                owners.add(owner)
        stages = mapping.get("object_stages")
        if isinstance(stages, list):
            owners.update(owner for value in stages if (owner := _stage_owner(value)))
    objects = evidence.get("objects")
    for item in objects if isinstance(objects, list) else []:
        if isinstance(item, dict):
            for key in ("stage", "object_stage", "owner_stage"):
                owner = _stage_owner(item.get(key))
                if owner:
                    owners.add(owner)
    return owners


def normalize_result_stage_ownership(result: dict[str, Any]) -> dict[str, Any]:
    """Return a result with a valid repair owner or explicit final-only evidence."""
    normalized = dict(result)
    raw_diagnostics = normalized.get("diagnostics")
    if raw_diagnostics is not None and not isinstance(raw_diagnostics, dict):
        normalized["diagnostics"] = {
            "owner_resolution": "final_only",
            "owner_resolution_error": "malformed_diagnostics",
        }
        return normalized
    diagnostics = dict(raw_diagnostics or {})
    evidence = normalized.get("evidence")
    constraint = evidence.get("intent_constraint") if isinstance(evidence, dict) else {}
    constraint = constraint if isinstance(constraint, dict) else {}
    evidence = evidence if isinstance(evidence, dict) else {}
    candidates = (diagnostics.get("earliest_stage"), constraint.get("stage"))
    owner = next(
        (
            _stage_owner(candidate)
            for candidate in candidates
            if _stage_owner(candidate)
        ),
        "",
    )
    resolution = "producer" if owner else ""
    if not owner:
        object_owners = _explicit_object_stage_owners(diagnostics, evidence)
        if len(object_owners) == 1:
            owner = next(iter(object_owners))
            resolution = "object_stage"
    if not owner:
        owner = _METRIC_STAGE_OWNERS.get(str(normalized.get("metric") or ""), "")
        resolution = "metric_semantics" if owner else ""
    if not owner:
        check_id = str(normalized.get("check_id") or "")
        owner = next(
            (
                stage
                for prefix, stage in _CHECK_PREFIX_STAGE_OWNERS.items()
                if check_id.startswith(prefix)
            ),
            "",
        )
        resolution = "check_semantics" if owner else ""
    if owner:
        diagnostics["earliest_stage"] = owner
        diagnostics["owner_resolution"] = resolution
    else:
        diagnostics.pop("earliest_stage", None)
        diagnostics["owner_resolution"] = "final_only"
    normalized["diagnostics"] = diagnostics
    return normalized


def normalize_result_stage_ownerships(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Apply the canonical owner policy before aggregation and reporting."""
    return [normalize_result_stage_ownership(result) for result in results]
