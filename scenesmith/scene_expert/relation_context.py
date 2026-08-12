"""Stage projection for hard intent and advisory HSSD relation priors."""

from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from scenesmith.scene_expert.schemas import (
    AdvisoryRelationPrior,
    FloorPlanReservation,
    FloorPlanReservationManifest,
    ObjectSelectorSpec,
    SceneTaskSpec,
    StageRelationContext,
    SuppressedRelationPrior,
)
from scenesmith.scenebenchmark_critic.intent_schema import (
    canonical_selector_category,
    selector_categories_overlap,
)
from scenesmith.scenebenchmark_critic.relation_registry import (
    ROOM_RELATIVE_WALL_CATEGORIES,
    relations_are_exclusive,
)


DEFAULT_HSSD_LOOKUP = (
    Path(__file__).resolve().parents[1]
    / "scenebenchmark_critic"
    / "asset_annotation_data"
    / "hssd_annotation_lookup.json.gz"
)
_STAGE_ORDER = (
    "floor_plan",
    "furniture",
    "wall_mounted",
    "ceiling_mounted",
    "manipuland",
)
_TASK_STAGE_FIELDS = {
    "furniture": "required_large_objects",
    "wall_mounted": "required_wall_objects",
    "ceiling_mounted": "required_ceiling_objects",
    "manipuland": "required_small_objects",
}
_RELATION_FAMILIES = {
    "against": "position",
    "against_wall": "position",
    "aligned_with": "position",
    "around": "position",
    "beside": "position",
    "between": "position",
    "centered_above": "position",
    "centered_between": "position",
    "centered_in_room": "position",
    "centered_on_wall": "position",
    "corner_of_room": "position",
    "corner_distribution": "position",
    "edge_distribution": "position",
    "flanking": "position",
    "in_front_of": "position",
    "near": "position",
    "next_to": "position",
    "surround": "position",
    "surrounded_by": "position",
    "faces": "orientation",
    "operation_zone_at_wall": "orientation",
    "instructional_surface_alignment": "orientation",
    "supports": "support",
    "placed_on": "support",
    "on_top_of": "support",
    "one_per_support": "support",
    "attached_to": "attachment",
    "mounted_on": "attachment",
    "on_wall": "attachment",
    "hang_from_ceiling": "attachment",
}
_HARD_ORIENTATION_OWNERS = {
    "against_wall",
    "back_against_wall",
    "centered_on_wall",
    "operation_zone_at_wall",
}
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
_GENERIC_CATEGORY_FALLBACKS = {
    "armchair": "chair",
    "dining_chair": "chair",
    "guest_chair": "chair",
    "office_chair": "chair",
    "rocking_chair": "chair",
    "student_chair": "chair",
    "coffee_table": "table",
    "conference_table": "table",
    "dining_table": "table",
    "side_table": "table",
    "student_desk": "desk",
    "teacher_desk": "desk",
}


def _normalize_relation(value: Any) -> str:
    return "_".join(str(value or "").strip().lower().replace("-", " ").split())


def relation_family(relation: Any) -> str:
    """Return the conflict family shared by hard and advisory relations."""
    return _RELATION_FAMILIES.get(
        _normalize_relation(relation),
        "position",
    )


def _prior_signature(prior: dict[str, Any]) -> tuple[str, ...]:
    target_kind = str(prior.get("target_kind") or "")
    target = (
        prior.get("target_category")
        if target_kind == "asset_category"
        else prior.get("environment_anchor")
    )
    return (
        _normalize_relation(prior.get("relation_type")),
        target_kind,
        canonical_selector_category(target),
        canonical_selector_category(prior.get("relative_facing")),
        canonical_selector_category(prior.get("relative_position")),
        canonical_selector_category(prior.get("height_relation")),
    )


def hssd_prior_category_support(
    record: dict[str, Any],
    prior: dict[str, Any],
    *,
    lookup_path: str | Path = DEFAULT_HSSD_LOOKUP,
) -> float:
    """Return the exact-category asset support ratio for one relation prior."""
    category = canonical_selector_category(
        record.get("category_key") or record.get("category")
    )
    if not category:
        return 0.0
    category_sizes, index = _load_category_prior_index(str(Path(lookup_path).resolve()))
    category_size = category_sizes.get(category, 0)
    entry = index.get(category, {}).get(_prior_signature(prior))
    return (entry[0] / category_size) if entry and category_size else 0.0


@lru_cache(maxsize=4)
def _load_category_prior_index(
    path_text: str,
) -> tuple[
    dict[str, int], dict[str, dict[tuple[str, ...], tuple[int, dict[str, Any]]]]
]:
    """Load one exact category index and category-level prior support counts."""
    with gzip.open(path_text, "rt", encoding="utf-8") as stream:
        records = json.load(stream)

    category_sizes: Counter[str] = Counter()
    support_counts: dict[str, Counter[tuple[str, ...]]] = defaultdict(Counter)
    representatives: dict[str, dict[tuple[str, ...], dict[str, Any]]] = defaultdict(
        dict
    )
    for record in records.values():
        if not isinstance(record, dict):
            continue
        category = canonical_selector_category(
            record.get("category_key") or record.get("category")
        )
        if not category:
            continue
        category_sizes[category] += 1
        seen: set[tuple[str, ...]] = set()
        for prior in record.get("relation_priors") or []:
            if not isinstance(prior, dict):
                continue
            signature = _prior_signature(prior)
            if not signature[0] or signature in seen:
                continue
            seen.add(signature)
            support_counts[category][signature] += 1
            representatives[category].setdefault(signature, dict(prior))

    index: dict[str, dict[tuple[str, ...], tuple[int, dict[str, Any]]]] = {}
    for category, counts in support_counts.items():
        index[category] = {
            signature: (count, representatives[category][signature])
            for signature, count in counts.items()
        }
    return dict(category_sizes), index


def _task_categories(task_spec: SceneTaskSpec) -> tuple[Counter[str], dict[str, str]]:
    counts: Counter[str] = Counter()
    owners: dict[str, str] = {}
    for stage, field in _TASK_STAGE_FIELDS.items():
        for value in getattr(task_spec, field):
            category = canonical_selector_category(value)
            if not category:
                continue
            counts[category] += 1
            owners.setdefault(category, stage)
    return counts, owners


def _scene_categories(scene: Any | None) -> set[str]:
    categories: set[str] = set()
    if scene is None:
        return categories
    for object_id, obj in (getattr(scene, "objects", {}) or {}).items():
        metadata = getattr(obj, "metadata", None)
        values = [
            object_id,
            getattr(obj, "name", ""),
            getattr(obj, "description", ""),
            metadata.get("category", "") if isinstance(metadata, dict) else "",
            metadata.get("category_norm", "") if isinstance(metadata, dict) else "",
        ]
        for value in values:
            category = canonical_selector_category(value)
            if category:
                categories.add(category)
    return categories


def _annotation_categories_for(
    subject_category: str, indexed: Iterable[str]
) -> list[str]:
    if subject_category in indexed:
        return [subject_category]
    fallback = _GENERIC_CATEGORY_FALLBACKS.get(subject_category)
    return [fallback] if fallback and fallback in indexed else []


def _matching_categories(selector: str, available: Iterable[str]) -> list[str]:
    selector = canonical_selector_category(selector)
    return sorted(
        {
            candidate
            for candidate in available
            if selector_categories_overlap(selector, candidate)
            or _GENERIC_CATEGORY_FALLBACKS.get(candidate) == selector
            or _GENERIC_CATEGORY_FALLBACKS.get(selector) == candidate
        }
    )


def _selector_overlaps(first: dict[str, Any], category: str) -> bool:
    first_category = canonical_selector_category(first.get("category"))
    return bool(first_category) and bool(
        _matching_categories(first_category, [category])
    )


def _hard_conflicts(
    prior: AdvisoryRelationPrior,
    hard_constraints: list[dict[str, Any]],
) -> tuple[str | None, list[str]]:
    conflicts: list[str] = []
    duplicates: list[str] = []
    for constraint in hard_constraints:
        hard_relation = _normalize_relation(constraint.get("relation"))
        hard_family = _RELATION_FAMILIES.get(hard_relation)
        owns_orientation = (
            prior.relation_family == "orientation"
            and hard_relation in _HARD_ORIENTATION_OWNERS
        )
        if hard_family != prior.relation_family and not owns_orientation:
            continue
        subject = constraint.get("subjects") or {}
        target = constraint.get("targets") or {}
        if not isinstance(subject, dict) or not _selector_overlaps(
            subject, prior.subject_selector.category
        ):
            continue
        constraint_id = str(constraint.get("constraint_id") or "")
        same_relation_name = hard_relation in {
            _normalize_relation(prior.relation),
            {"beside": "next_to", "against": "against_wall", "around": "surround"}.get(
                _normalize_relation(prior.relation), ""
            ),
        }
        target_matches = bool(
            prior.target_selector is not None
            and isinstance(target, dict)
            and _selector_overlaps(target, prior.target_selector.category)
        )
        if prior.environment_anchor and isinstance(target, dict):
            target_matches = target_matches or _selector_overlaps(
                target, prior.environment_anchor
            )
        same_relation = same_relation_name and target_matches
        if same_relation:
            duplicates.append(constraint_id)
        elif owns_orientation or hard_family in {
            "orientation",
            "support",
            "attachment",
        }:
            conflicts.append(constraint_id)
        elif hard_family == "position" and _position_relations_are_exclusive(
            _normalize_relation(prior.relation), hard_relation
        ):
            conflicts.append(constraint_id)
    if conflicts:
        return "hard_conflict", [value for value in conflicts if value]
    if duplicates:
        return "duplicate", [value for value in duplicates if value]
    return None, []


def _position_relations_are_exclusive(first: str, second: str) -> bool:
    """Recognize only position pairs that cannot both hold for one subject."""
    return relations_are_exclusive(first, second)


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
                room_type=task_spec.room_type,
                subject_categories=[category],
                target_categories=[canonical_selector_category(target.get("category"))],
                wall_role=str(target.get("role") or "").strip().lower(),
                min_wall_width_m=_wall_width_for(category),
                count=_selector_count(subject),
            )
        )
    reservations.extend(_media_pair_reservations(constraints, task_spec.room_type))
    for index, zone in enumerate(task_spec.functional_zones):
        canonical = canonical_selector_category(zone)
        if not canonical:
            continue
        reservations.append(
            FloorPlanReservation(
                reservation_id=f"functional_zone__{canonical}__{index}",
                kind="functional_zone",
                room_type=task_spec.room_type,
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
    """Build deterministic per-stage relation context without adding LLM calls."""

    def __init__(
        self,
        *,
        lookup_path: str | Path = DEFAULT_HSSD_LOOKUP,
        confidence_threshold: float = 0.80,
        category_support_threshold: float = 0.80,
        max_priors_per_stage: int = 4,
        floor_plan_reservation_gate_enabled: bool = False,
    ) -> None:
        self._lookup_path = Path(lookup_path)
        self._confidence_threshold = float(confidence_threshold)
        self._category_support_threshold = float(category_support_threshold)
        self._max_priors_per_stage = max(0, int(max_priors_per_stage))
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
        if stage == "floor_plan" or not self._lookup_path.exists():
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
                    _floor_plan_reservations(constraints)
                    if stage == "floor_plan"
                    else []
                ),
                floor_plan_manifest=manifest,
                contract_constraint_count=len(constraints),
                projected_constraint_count=len(hard_constraints),
                projection_coverage=1.0,
            )
        stage_field = _TASK_STAGE_FIELDS.get(stage)
        stage_inventory = list(getattr(task_spec, stage_field)) if stage_field else []
        if not stage_inventory and not hard_constraints:
            # A soft prior may refine an active stage, but it may not create
            # stage work or inventory on its own.
            return StageRelationContext(
                stage=stage,
                hard_constraints=hard_constraints,
                contract_constraint_count=len(constraints),
                projected_constraint_count=0,
                projection_coverage=1.0,
            )

        category_sizes, prior_index = _load_category_prior_index(
            str(self._lookup_path.resolve())
        )
        task_counts, owners = _task_categories(task_spec)
        scene_categories = _scene_categories(scene)
        for category in scene_categories:
            owners.setdefault(category, "furniture")
        available_categories = set(task_counts) | scene_categories
        active: list[AdvisoryRelationPrior] = []
        suppressed: list[SuppressedRelationPrior] = []
        seen: set[tuple[str, str, str]] = set()

        for subject_category in sorted(available_categories):
            for annotation_category in _annotation_categories_for(
                subject_category, prior_index
            ):
                category_size = category_sizes.get(annotation_category, 0)
                for signature, (support_count, raw) in prior_index[
                    annotation_category
                ].items():
                    target_kind = str(raw.get("target_kind") or "")
                    target_category = canonical_selector_category(
                        raw.get("target_category")
                    )
                    matches = (
                        _matching_categories(target_category, available_categories)
                        if target_kind == "asset_category"
                        else []
                    )
                    owner_candidates = [
                        owners.get(match) for match in matches if owners.get(match)
                    ]
                    subject_stage = owners.get(subject_category, "furniture")
                    prior_stage = max(
                        [subject_stage, *owner_candidates],
                        key=_STAGE_ORDER.index,
                    )
                    if prior_stage != stage:
                        continue
                    support = support_count / category_size if category_size else 0.0
                    confidence = float(raw.get("confidence") or 0.0)
                    target_selector = None
                    if len(matches) == 1:
                        target_selector = ObjectSelectorSpec(
                            category=matches[0],
                            count=task_counts.get(matches[0]) or None,
                        )
                    payload = {
                        "relation": signature[0],
                        "subject": subject_category,
                        "target": matches[0] if len(matches) == 1 else target_category,
                        "anchor": canonical_selector_category(
                            raw.get("environment_anchor")
                        ),
                        "provenance": str(raw.get("provenance") or ""),
                    }
                    prior_id = (
                        "hssd_prior_"
                        + hashlib.sha1(
                            json.dumps(payload, sort_keys=True).encode("utf-8")
                        ).hexdigest()[:12]
                    )
                    prior = AdvisoryRelationPrior(
                        prior_id=prior_id,
                        relation=signature[0],
                        relation_family=relation_family(signature[0]),
                        subject_selector=ObjectSelectorSpec(
                            category=subject_category,
                            count=task_counts.get(subject_category) or None,
                        ),
                        target_selector=target_selector,
                        environment_anchor=canonical_selector_category(
                            raw.get("environment_anchor")
                        ),
                        distance_range_m=[
                            float(value)
                            for value in (raw.get("distance_range_m") or [])
                            if isinstance(value, (int, float))
                        ][:2],
                        relative_facing=str(raw.get("relative_facing") or ""),
                        relative_position=str(raw.get("relative_position") or ""),
                        height_relation=str(raw.get("height_relation") or ""),
                        confidence=confidence,
                        category_support=support,
                        provenance=str(raw.get("provenance") or ""),
                    )

                    reason: str | None = None
                    conflict_ids: list[str] = []
                    if not prior.provenance.startswith("seed_rule:"):
                        reason = "low_confidence"
                    elif (
                        confidence < self._confidence_threshold
                        or support < self._category_support_threshold
                    ):
                        reason = "low_confidence"
                    elif target_kind == "asset_category" and not matches:
                        reason = "missing_target"
                    elif target_kind == "asset_category" and len(matches) > 1:
                        reason = "ambiguous_target"
                    else:
                        reason, conflict_ids = _hard_conflicts(prior, hard_constraints)

                    dedupe_key = (
                        prior.relation,
                        prior.subject_selector.category,
                        (
                            prior.target_selector.category
                            if prior.target_selector
                            else prior.environment_anchor
                        ),
                    )
                    if reason is None and dedupe_key in seen:
                        reason = "duplicate"
                    if reason is not None:
                        suppressed.append(
                            SuppressedRelationPrior(
                                prior=prior,
                                reason=reason,
                                conflicting_constraint_ids=conflict_ids,
                            )
                        )
                        continue
                    seen.add(dedupe_key)
                    active.append(prior)

        active.sort(
            key=lambda item: (item.confidence, item.category_support, item.prior_id),
            reverse=True,
        )
        for prior in active[self._max_priors_per_stage :]:
            suppressed.append(SuppressedRelationPrior(prior=prior, reason="duplicate"))
        active = active[: self._max_priors_per_stage]
        return StageRelationContext(
            stage=stage,
            hard_constraints=hard_constraints,
            advisory_hssd_priors=active,
            suppressed_priors=suppressed,
            contract_constraint_count=len(constraints),
            projected_constraint_count=len(hard_constraints),
            projection_coverage=1.0,
        )
