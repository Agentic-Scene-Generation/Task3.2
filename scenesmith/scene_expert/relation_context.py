"""Stage projection for hard intent and floor-plan reservations."""

from __future__ import annotations

from typing import Any

from scenesmith.scene_expert.schemas import (
    FloorPlanReservation,
    FloorPlanReservationManifest,
    SceneTaskSpec,
    StageRelationContext,
)
from scenesmith.scenebenchmark_critic.intent_schema import canonical_selector_category
from scenesmith.scenebenchmark_critic.relation_registry import (
    ROOM_RELATIVE_WALL_CATEGORIES,
)

_FLOOR_PLAN_RESERVATION_RELATIONS = frozenset(
    {"against_wall", "centered_on_wall", "on_wall"}
)
_WALL_TARGET_CATEGORIES = frozenset({"wall", *ROOM_RELATIVE_WALL_CATEGORIES})
_SEATING_CATEGORIES = frozenset(
    {
        "armchair",
        "bench",
        "chair",
        "loveseat",
        "sectional_sofa",
        "sofa",
    }
)
_MEDIA_CATEGORIES = frozenset(
    {
        "display",
        "media_console",
        "media_unit",
        "projector_screen",
        "television",
        "tv",
        "tv_stand",
    }
)
_WALL_WIDTH_PROFILES_M = {
    "bed": 2.6,
    "bookshelf": 1.2,
    "cabinet": 1.2,
    "desk": 1.4,
    "dressing_table": 1.2,
    "media_console": 1.8,
    "media_unit": 1.8,
    "mirror": 0.8,
    "projector_screen": 1.8,
    "sectional_sofa": 3.0,
    "shelf": 1.0,
    "sofa": 2.6,
    "television": 1.5,
    "tv": 1.5,
    "tv_stand": 1.8,
    "wardrobe": 2.0,
    "water_dispenser": 0.8,
}
_ZONE_AREA_PROFILES_M2 = {
    "dining": 5.0,
    "living": 7.0,
    "sleeping": 6.0,
    "storage": 3.0,
    "teaching": 8.0,
    "work": 4.0,
    "working": 4.0,
}


def _normalize_relation(value: Any) -> str:
    return "_".join(str(value or "").strip().lower().replace("-", " ").split())


def _floor_plan_reservations(constraints: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Project future wall anchors into opening-reservation requirements.

    Furniture and wall-mounted stages need physical wall area which is decided
    earlier by the floor plan. This projection deliberately contains only
    explicit wall relations: proximity and seating constraints must not affect
    doors or windows.
    """
    reservations: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for constraint in constraints:
        if str(constraint.get("stage") or "") == "floor_plan":
            continue
        relation = str(constraint.get("relation") or "")
        if relation not in _FLOOR_PLAN_RESERVATION_RELATIONS:
            continue
        target = constraint.get("targets") or {}
        target_category = canonical_selector_category(target.get("category"))
        if target_category not in _WALL_TARGET_CATEGORIES:
            continue
        subject = constraint.get("subjects") or {}
        subject_category = canonical_selector_category(subject.get("category"))
        if not subject_category:
            continue
        wall_role = str(target.get("role") or "").strip().lower()
        key = (subject_category, relation, wall_role)
        if key in seen:
            continue
        seen.add(key)
        reservations.append(
            {
                "constraint_id": str(constraint.get("constraint_id") or ""),
                "source_stage": str(constraint.get("stage") or ""),
                "relation": relation,
                "subjects": dict(subject),
                "targets": dict(target),
            }
        )
    return reservations


def _selector_count(selector: dict[str, Any]) -> int:
    value = selector.get("count")
    return max(1, int(value)) if isinstance(value, (int, float)) else 1


def _wall_width_for(category: str) -> float:
    canonical = canonical_selector_category(category)
    if canonical in _WALL_WIDTH_PROFILES_M:
        return _WALL_WIDTH_PROFILES_M[canonical]
    for suffix, width in _WALL_WIDTH_PROFILES_M.items():
        if canonical.endswith(f"_{suffix}"):
            return width
    return 1.0


def _zone_area_for(zone: str) -> float:
    canonical = canonical_selector_category(zone).removesuffix("_zone")
    for token, area in _ZONE_AREA_PROFILES_M2.items():
        if token in canonical.split("_"):
            return area
    return 4.0


def _reservation_room_scope(room_type: str) -> str:
    """Return a room-type scope that can be matched against generated rooms.

    Task compilation uses a comma-separated label when a prompt names several
    rooms. That label describes the whole scene, not an individual RoomSpec, so
    using it as an exact reservation selector makes the capacity gate
    unsatisfiable. An empty selector scopes the reservation to all placed rooms;
    single-room tasks retain their precise room-type check.
    """
    value = str(room_type or "").strip()
    if any(separator in value for separator in (",", ";", "/", "&")):
        return ""
    return value


def _explicit_window_count(constraints: list[dict[str, Any]]) -> int:
    counts: list[int] = []
    for constraint in constraints:
        if _normalize_relation(constraint.get("relation")) != "required_count":
            continue
        subject = constraint.get("subjects") or {}
        if canonical_selector_category(subject.get("category")) != "window":
            continue
        counts.append(_selector_count(subject))
    return max(counts, default=0)


def _media_pair_reservations(
    constraints: list[dict[str, Any]], room_type: str
) -> list[FloorPlanReservation]:
    reservations: list[FloorPlanReservation] = []
    seen: set[tuple[str, str]] = set()
    for constraint in constraints:
        relation = _normalize_relation(constraint.get("relation"))
        if relation not in {"across_from", "faces", "seating_to_media"}:
            continue
        subject = constraint.get("subjects") or {}
        target = constraint.get("targets") or {}
        subject_category = canonical_selector_category(subject.get("category"))
        target_category = canonical_selector_category(target.get("category"))
        pair = (subject_category, target_category)
        reverse_pair = (target_category, subject_category)
        if not (
            (
                subject_category in _SEATING_CATEGORIES
                and target_category in _MEDIA_CATEGORIES
            )
            or (
                target_category in _SEATING_CATEGORIES
                and subject_category in _MEDIA_CATEGORIES
            )
        ):
            continue
        canonical_pair = pair if pair[0] in _SEATING_CATEGORIES else reverse_pair
        if canonical_pair in seen:
            continue
        seen.add(canonical_pair)
        source_id = str(constraint.get("constraint_id") or "")
        reservations.append(
            FloorPlanReservation(
                reservation_id=f"media_pair__{canonical_pair[0]}__{canonical_pair[1]}",
                kind="opposed_anchor_pair",
                source_constraint_ids=[source_id] if source_id else [],
                room_type=room_type,
                subject_categories=[canonical_pair[0]],
                target_categories=[canonical_pair[1]],
                min_wall_width_m=max(
                    _wall_width_for(canonical_pair[0]),
                    _wall_width_for(canonical_pair[1]),
                ),
            )
        )
    return reservations


def _floor_plan_manifest(
    *,
    constraints: list[dict[str, Any]],
    task_spec: SceneTaskSpec,
    enabled: bool,
) -> FloorPlanReservationManifest:
    reservations: list[FloorPlanReservation] = []
    room_scope = _reservation_room_scope(task_spec.room_type)
    for index, raw in enumerate(_floor_plan_reservations(constraints)):
        subject = raw.get("subjects") or {}
        target = raw.get("targets") or {}
        category = canonical_selector_category(subject.get("category"))
        source_id = str(raw.get("constraint_id") or "")
        reservations.append(
            FloorPlanReservation(
                reservation_id=source_id or f"wall_anchor__{category}__{index}",
                kind="wall_anchor",
                source_constraint_ids=[source_id] if source_id else [],
                room_type=room_scope,
                subject_categories=[category],
                target_categories=[canonical_selector_category(target.get("category"))],
                wall_role=str(target.get("role") or "").strip().lower(),
                min_wall_width_m=_wall_width_for(category),
                count=_selector_count(subject),
            )
        )
    reservations.extend(_media_pair_reservations(constraints, room_scope))
    for index, zone in enumerate(task_spec.functional_zones):
        canonical = canonical_selector_category(zone)
        if not canonical:
            continue
        reservations.append(
            FloorPlanReservation(
                reservation_id=f"functional_zone__{canonical}__{index}",
                kind="functional_zone",
                room_type=room_scope,
                subject_categories=[canonical],
                min_zone_area_m2=_zone_area_for(canonical),
            )
        )
    explicit_window_count = _explicit_window_count(constraints)
    return FloorPlanReservationManifest(
        enabled=enabled,
        reservations=reservations,
        explicit_window_count=explicit_window_count,
        explicit_window_required=explicit_window_count > 0,
    )


class StageRelationProjector:
    """Build deterministic hard-intent context without adding LLM calls."""

    def __init__(
        self,
        *,
        floor_plan_reservation_gate_enabled: bool = False,
    ) -> None:
        self._floor_plan_reservation_gate_enabled = bool(
            floor_plan_reservation_gate_enabled
        )

    def project(
        self,
        *,
        stage: str,
        task_spec: SceneTaskSpec,
        intent_contract: dict[str, Any] | None,
        scene: Any | None = None,
    ) -> StageRelationContext:
        constraints = [
            dict(item)
            for item in ((intent_contract or {}).get("constraints") or [])
            if isinstance(item, dict)
        ]
        hard_constraints = [
            item for item in constraints if str(item.get("stage") or "") == stage
        ]
        manifest = (
            _floor_plan_manifest(
                constraints=constraints,
                task_spec=task_spec,
                enabled=self._floor_plan_reservation_gate_enabled,
            )
            if stage == "floor_plan"
            else None
        )
        return StageRelationContext(
            stage=stage,
            hard_constraints=hard_constraints,
            floor_plan_reservations=(
                _floor_plan_reservations(constraints) if stage == "floor_plan" else []
            ),
            floor_plan_manifest=manifest,
            contract_constraint_count=len(constraints),
            projected_constraint_count=len(hard_constraints),
            projection_coverage=1.0,
        )
