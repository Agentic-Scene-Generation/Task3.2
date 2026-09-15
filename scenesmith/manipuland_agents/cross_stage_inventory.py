"""Cross-stage inventory guards for manipuland placement."""

import re

from dataclasses import dataclass
from typing import Any

from scenesmith.agent_utils.room import ObjectType
from scenesmith.scenebenchmark_critic.intent_contract import (
    intent_contract_constraints_for_scene,
    selected_ids,
)
from scenesmith.scenebenchmark_critic.object_taxonomy import (
    canonical_object_category,
    categories_are_equivalent,
    execution_owner,
    generation_owner,
)


_FLOOR_COVERING_PATTERN = re.compile(
    r"\b(?:area\s+)?(?:rug|carpet|floor\s+mat|doormat|door\s+mat)\b",
    re.IGNORECASE,
)
_SINGLE_FLOOR_COVERING_REQUEST_PATTERN = re.compile(
    r"^\s*(?:(?:required|optional)\s*:\s*)?"
    r"(?:(?:a|an|the)\s+)?"
    r"(?:(?:small|large|round|square|rectangular|area|floor)\s+)*"
    r"(?:rug|carpet|mat|doormat)\s*[.!]?\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ManipulandSupportCohort:
    """One target-local share of a hard manipuland support constraint."""

    constraint_id: str
    relation: str
    category: str
    quantifier: str
    target_id: str
    target_ids: tuple[str, ...]
    required_count: int
    cohort_total: int
    subject_selector: dict[str, Any]


def contract_manipuland_support_cohorts(scene: Any) -> list[ManipulandSupportCohort]:
    """Project hard support relations into distinct per-target inventory shares.

    Counts are split deterministically across every bound support target. Different
    support constraints remain separate cohorts, so one physical instance cannot
    satisfy both a bed and a desk requirement.
    """
    constraints = intent_contract_constraints_for_scene(scene)
    if not constraints:
        return []
    objects = getattr(scene, "objects", {}) or {}
    if not isinstance(objects, dict):
        return []
    object_rows = _contract_object_rows(objects)
    declared_small = _declared_manipuland_categories(scene)
    cohorts: list[ManipulandSupportCohort] = []
    for index, constraint in enumerate(constraints):
        relation = str(constraint.get("relation") or "")
        if (
            relation not in {"on_top_of", "one_per_support"}
            or str(constraint.get("stage") or "") != "manipuland"
            or str(constraint.get("strength") or "hard").lower() != "hard"
        ):
            continue
        subjects = constraint.get("subjects") or {}
        targets = constraint.get("targets") or {}
        if not isinstance(subjects, dict) or not isinstance(targets, dict):
            continue
        category = canonical_object_category(subjects.get("category"))
        if not category:
            continue
        declared_owner = (
            "manipuland"
            if any(
                categories_are_equivalent(category, declared)
                for declared in declared_small
            )
            else ""
        )
        if (
            generation_owner(
                category,
                relation=relation,
                endpoint="subject",
                declared_owner=declared_owner,
            )
            != "manipuland"
        ):
            continue
        target_ids = tuple(sorted(set(selected_ids(targets, object_rows))))
        if not target_ids:
            continue
        try:
            cohort_total = max(1, int(subjects.get("count") or 1))
        except (TypeError, ValueError):
            cohort_total = 1
        if relation == "one_per_support":
            cohort_total = max(cohort_total, len(target_ids))
        quotient, remainder = divmod(cohort_total, len(target_ids))
        constraint_id = str(
            constraint.get("constraint_id") or f"support_cohort_{index}"
        )
        quantifier = str(subjects.get("quantifier") or "minimum")
        for target_index, target_id in enumerate(target_ids):
            required_count = quotient + (1 if target_index < remainder else 0)
            if relation == "one_per_support":
                required_count = max(1, required_count)
            if required_count <= 0:
                continue
            cohorts.append(
                ManipulandSupportCohort(
                    constraint_id=constraint_id,
                    relation=relation,
                    category=category,
                    quantifier=quantifier,
                    target_id=target_id,
                    target_ids=target_ids,
                    required_count=required_count,
                    cohort_total=cohort_total,
                    subject_selector=dict(subjects),
                )
            )
    return sorted(
        cohorts,
        key=lambda cohort: (
            cohort.target_id,
            cohort.category,
            cohort.constraint_id,
        ),
    )


def _declared_manipuland_categories(scene: Any) -> list[str]:
    task_spec = getattr(scene, "scene_expert_task_spec", None)
    if isinstance(task_spec, dict):
        values = task_spec.get("required_small_objects", []) or []
    else:
        values = getattr(task_spec, "required_small_objects", []) or []
    return [
        category for value in values if (category := canonical_object_category(value))
    ]


def is_floor_covering_text(value: Any) -> bool:
    """Return whether free-form asset identity denotes a floor covering."""
    normalized = re.sub(r"[^a-z0-9]+", " ", str(value or "").lower())
    return _FLOOR_COVERING_PATTERN.search(normalized) is not None


def existing_floor_covering_ids(scene: Any) -> list[str]:
    """Find already-realized rug-like objects across scene stages."""
    found: list[str] = []
    for obj in getattr(scene, "objects", {}).values():
        metadata = getattr(obj, "metadata", None) or {}
        identity = " ".join(
            str(value or "")
            for value in (
                metadata.get("semantic_name") if isinstance(metadata, dict) else None,
                getattr(obj, "name", None),
                getattr(obj, "description", None),
                getattr(obj, "object_id", None),
            )
        )
        if is_floor_covering_text(identity):
            found.append(str(getattr(obj, "object_id", "")))
    return sorted(object_id for object_id in found if object_id)


def is_floor_target(scene: Any, furniture_id: Any) -> bool:
    """Return whether a manipuland assignment targets the room floor."""
    obj = scene.get_object(furniture_id)
    if obj is None:
        return False
    object_type = getattr(obj, "object_type", "")
    return (
        object_type == ObjectType.FLOOR
        or str(getattr(object_type, "value", object_type)).lower() == "floor"
    )


def is_single_floor_covering_request(value: Any) -> bool:
    """Identify assignments whose only requested item is one floor covering."""
    return (
        _SINGLE_FLOOR_COVERING_REQUEST_PATTERN.fullmatch(str(value or "")) is not None
    )


def redundant_floor_covering_request_indices(
    scene: Any,
    furniture_id: Any,
    object_descriptions: list[str],
    short_names: list[str],
) -> list[int]:
    """Return duplicate covering request indices for an existing floor inventory."""
    if not existing_floor_covering_ids(scene) or not is_floor_target(
        scene, furniture_id
    ):
        return []
    return [
        index
        for index, (description, short_name) in enumerate(
            zip(object_descriptions, short_names, strict=False)
        )
        if is_floor_covering_text(description) or is_floor_covering_text(short_name)
    ]


def satisfied_furniture_owned_floor_requirements(
    scene: Any,
    furniture_id: Any,
    request_text: Any,
) -> dict[str, list[str]]:
    """Return fulfilled furniture-owned requirements mentioned by a floor target.

    A floor workflow may be selected by the VLM even when an earlier furniture
    stage already created the prompt-required floor-standing object.  The task
    specification owns both the stage and cardinality, so only suppress a
    request when that authoritative count is present in the global inventory.
    This deliberately does not suppress optional decoration or objects that
    belong on a furniture support surface.
    """
    if not is_floor_target(scene, furniture_id):
        return {}

    fulfilled: dict[str, list[str]] = {}
    for category, required_count in _furniture_owned_required_categories(scene):
        if not _text_mentions_category(request_text, category):
            continue
        object_ids = _scene_furniture_ids_matching_category(scene, category)
        if len(object_ids) >= required_count:
            fulfilled[category] = object_ids
    return fulfilled


def is_single_explicit_required_category_request(
    request_text: Any,
    categories: list[str] | set[str] | tuple[str, ...],
) -> bool:
    """Whether text is one explicit REQUIRED request for a fulfilled category.

    Whole-target skipping is intentionally narrow.  A compound request keeps
    its workflow so unsatisfied required or optional items can still be placed;
    the generation-tool guard handles the satisfied item individually.
    """
    text = str(request_text or "").strip()
    if not text or "\n" in text:
        return False
    match = re.fullmatch(r"required\s*:\s*(.+?)[.!]?", text, re.IGNORECASE)
    if match is None:
        return False
    requested = match.group(1).strip()
    if not requested or re.search(r"\b(?:and|or|with)\b|[,;/]", requested, re.I):
        return False
    return any(_text_mentions_category(requested, category) for category in categories)


def redundant_furniture_owned_floor_request_indices(
    *,
    fulfilled_requirements: dict[str, list[str]],
    object_descriptions: list[str],
    short_names: list[str],
) -> list[int]:
    """Return generated assets that would duplicate fulfilled floor requirements."""
    if not fulfilled_requirements:
        return []
    return [
        index
        for index, (description, short_name) in enumerate(
            zip(object_descriptions, short_names, strict=False)
        )
        if any(
            _text_mentions_category(description, category)
            or _text_mentions_category(short_name, category)
            for category in fulfilled_requirements
        )
    ]


def _furniture_owned_required_categories(scene: Any) -> list[tuple[str, int]]:
    """Read authoritative furniture inventory counts from the task specification."""
    task_spec = getattr(scene, "scene_expert_task_spec", None)
    if isinstance(task_spec, dict):
        required_values = task_spec.get("required_large_objects", []) or []
    else:
        required_values = getattr(task_spec, "required_large_objects", []) or []

    grouped: list[list[Any]] = []
    for value in required_values:
        category = canonical_object_category(value)
        if (
            not category
            or execution_owner(category, existing_owner="furniture") != "furniture"
        ):
            continue
        existing = next(
            (
                group
                for group in grouped
                if categories_are_equivalent(category, group[0])
            ),
            None,
        )
        if existing is None:
            grouped.append([category, 1])
        else:
            existing[1] += 1
    return [(str(category), int(count)) for category, count in grouped]


def _scene_furniture_ids_matching_category(scene: Any, category: str) -> list[str]:
    """Find furniture-stage objects matching a canonical inventory category."""
    matched: list[str] = []
    for object_id, obj in (getattr(scene, "objects", {}) or {}).items():
        object_type = getattr(obj, "object_type", "")
        object_type_value = str(getattr(object_type, "value", object_type)).lower()
        if object_type_value != ObjectType.FURNITURE.value:
            continue
        identity = " ".join(
            str(value or "") for value in _object_identity_values(obj, object_id)
        )
        if _text_mentions_category(identity, category):
            matched.append(str(getattr(obj, "object_id", object_id)))
    return sorted(object_id for object_id in matched if object_id)


def _object_identity_values(obj: Any, object_id: Any) -> tuple[Any, ...]:
    metadata = getattr(obj, "metadata", None) or {}
    semantic_name = metadata.get("semantic_name") if isinstance(metadata, dict) else ""
    return (
        semantic_name,
        getattr(obj, "name", ""),
        getattr(obj, "description", ""),
        getattr(obj, "object_id", object_id),
    )


def _text_mentions_category(value: Any, category: str) -> bool:
    """Match a category against free text without treating descriptive text as a label."""
    normalized = re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()
    if not normalized:
        return False
    words = normalized.split()
    for start in range(len(words)):
        for end in range(start + 1, min(len(words), start + 4) + 1):
            candidate = canonical_object_category("_".join(words[start:end]))
            if categories_are_equivalent(candidate, category):
                return True
    return False


def contract_bound_support_object_ids(scene: Any, target_id: Any) -> list[str]:
    """Return hard-contract objects supported by the current furniture.

    Selector membership alone cannot identify a realized support relation.  A
    contract may select several equivalent desks, for example, while each desk
    still needs its own monitor.  Only expose objects whose saved placement is
    actually on one of the current furniture's support surfaces.
    """
    objects = getattr(scene, "objects", {}) or {}
    object_rows = _contract_object_rows(objects)

    target_id_text = str(target_id)
    target_object = _scene_object_by_id(objects, target_id_text)
    target_surface_ids = {
        str(getattr(surface, "surface_id", ""))
        for surface in getattr(target_object, "support_surfaces", []) or []
        if str(getattr(surface, "surface_id", ""))
    }
    if not target_surface_ids:
        return []

    supported_ids: set[str] = set()
    for constraint in intent_contract_constraints_for_scene(scene):
        if str(constraint.get("relation") or "") != "on_top_of":
            continue
        if str(constraint.get("strength") or "hard").lower() != "hard":
            continue
        target_ids = selected_ids(constraint.get("targets"), object_rows)
        if target_id_text not in target_ids:
            continue
        for subject_id in selected_ids(constraint.get("subjects"), object_rows):
            subject = _scene_object_by_id(objects, subject_id)
            placement = getattr(subject, "placement_info", None)
            parent_surface_id = str(getattr(placement, "parent_surface_id", "") or "")
            if parent_surface_id in target_surface_ids:
                supported_ids.add(subject_id)

    supported_ids.discard(target_id_text)
    return sorted(supported_ids)


def violates_hard_one_per_support_reparenting(
    scene: Any,
    object_id: Any,
    *,
    source_surface_id: Any,
    target_surface_id: Any,
) -> bool:
    """Return whether moving an object would merge hard one-per-support slots."""
    source_surface_id = str(source_surface_id or "")
    target_surface_id = str(target_surface_id or "")
    if not source_surface_id or source_surface_id == target_surface_id:
        return False

    objects = getattr(scene, "objects", {}) or {}
    object_rows = _contract_object_rows(objects)
    object_id_text = str(object_id)
    for constraint in intent_contract_constraints_for_scene(scene):
        if str(constraint.get("relation") or "") != "one_per_support":
            continue
        if str(constraint.get("strength") or "hard").lower() != "hard":
            continue
        subject_ids = set(selected_ids(constraint.get("subjects"), object_rows))
        if object_id_text not in subject_ids:
            continue
        target_ids = selected_ids(constraint.get("targets"), object_rows)
        target_surface_ids = {
            target_id: {
                str(getattr(surface, "surface_id", ""))
                for surface in getattr(
                    _scene_object_by_id(objects, target_id), "support_surfaces", []
                )
                or []
                if str(getattr(surface, "surface_id", ""))
            }
            for target_id in target_ids
        }
        source_owner = next(
            (
                target_id
                for target_id, surface_ids in target_surface_ids.items()
                if source_surface_id in surface_ids
            ),
            None,
        )
        target_owner = next(
            (
                target_id
                for target_id, surface_ids in target_surface_ids.items()
                if target_surface_id in surface_ids
            ),
            None,
        )
        if source_owner is not None and target_owner is not None:
            return source_owner != target_owner
    return False


def _contract_object_rows(objects: dict[Any, Any]) -> list[dict[str, Any]]:
    """Build the selector rows once for cross-stage contract checks."""
    rows = []
    for object_id, obj in objects.items():
        metadata = getattr(obj, "metadata", None) or {}
        object_type = getattr(obj, "object_type", "")
        semantic_name = (
            metadata.get("semantic_name", "") if isinstance(metadata, dict) else ""
        )
        rows.append(
            {
                "id": str(object_id),
                "name": getattr(obj, "name", ""),
                "description": getattr(obj, "description", ""),
                "object_type": getattr(object_type, "value", object_type),
                "category": semantic_name,
                "category_norm": semantic_name,
                "metadata": metadata if isinstance(metadata, dict) else {},
            }
        )
    return rows


def _scene_object_by_id(objects: dict[Any, Any], object_id: Any) -> Any | None:
    """Look up an object without assuming the scene's ID key subtype."""
    object_id_text = str(object_id)
    for candidate_id, obj in objects.items():
        if str(candidate_id) == object_id_text:
            return obj
    return None
