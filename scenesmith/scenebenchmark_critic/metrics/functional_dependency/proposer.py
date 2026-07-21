from __future__ import annotations

import re

from typing import Any, Callable
from scenesmith.scenebenchmark_critic.core.geometry import (
    GeometryStore,
    bbox_gap_xy,
    distance_xy,
    load_geometry,
    object_affordances,
    object_category,
)
from scenesmith.scenebenchmark_critic.core.models import (
    FunctionalDependencyProposal,
)
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.constants import *
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.orientation_contracts import (
    orientation_contract_subjects,
)
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.profiles import (
    object_function_profile,
)
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.relations import (
    _angle_penalty,
    _infer_relation_type,
    _preferred_relations_for_subject,
    _relation_target_is_valid,
)
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.results import (
    _check_key,
    _normalize_scoring_tier,
    _proposal_to_check,
    _scoring_tier_rank,
)
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.semantics import (
    _is_actionable_seating_surface_pair,
    _is_core_media_target,
    _is_core_work_surface_target,
    _object_text,
)
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.support import (
    _eval_object_on_support,
    _object_on_support_rank,
)

VlmProposer = Callable[..., list[FunctionalDependencyProposal]]


def augment_functional_dependency_checks(
    case_pack: dict[str, Any],
    config: Any,
    *,
    metric_filter: list[str] | None,
    progress=lambda _message: None,
    vlm_proposer: VlmProposer | None = None,
) -> bool:
    if metric_filter is not None and "functional_dependency" not in set(metric_filter):
        return False
    store = load_geometry(case_pack)
    if store is None:
        return False

    proposals = propose_dependency_relations(
        case_pack, store, config, progress=progress, vlm_proposer=vlm_proposer
    )
    proposal_keys = {_proposal_key(proposal) for proposal in proposals}
    existing = {
        _check_key(check)
        for check in case_pack.get("checks", []) or []
        if str(check.get("metric") or "") == "functional_dependency"
    }
    added = False
    checks = []
    for check in case_pack.get("checks", []) or []:
        if str(check.get("metric") or "") != "functional_dependency":
            checks.append(check)
            continue
        if str(check.get("check_source") or "") != "fd_relation_proposer":
            checks.append(check)
            continue
        full_key = _check_key(check)
        if full_key not in proposal_keys:
            existing.discard(full_key)
            added = True
            continue
        checks.append(check)
    for proposal in proposals:
        check = _proposal_to_check(case_pack, proposal)
        key = _check_key(check)
        if key in existing:
            continue
        checks.append(check)
        existing.add(key)
        added = True
    if added:
        case_pack["checks"] = checks
    return added


def propose_dependency_relations(
    case_pack: dict[str, Any],
    store: GeometryStore,
    config: Any,
    *,
    progress=lambda _message: None,
    vlm_proposer: VlmProposer | None = None,
) -> list[FunctionalDependencyProposal]:
    max_proposals = int(
        getattr(getattr(config, "run", config), "max_fd_relation_proposals", 8) or 8
    )
    contracted_orientation_subjects = orientation_contract_subjects(case_pack)
    proposer_mode = (
        str(
            getattr(
                getattr(config, "run", config), "fd_relation_proposer_mode", "template"
            )
            or "template"
        )
        .strip()
        .lower()
    )
    vlm_proposals: list[FunctionalDependencyProposal] = []
    if proposer_mode in {"vlm", "hybrid", "auto"}:
        progress(
            "VLM relation proposal is excluded from this critic-only migration; "
            "using deterministic templates"
        )
    else:
        progress(
            f"Using template FD relation proposer for up to {max_proposals} proposals"
        )
    normalized = _normalize_proposals(
        case_pack, vlm_proposals, store, max_proposals=max_proposals
    )
    normalized = _without_contracted_orientation_subjects(
        normalized, contracted_orientation_subjects
    )
    if len(normalized) >= max_proposals:
        return normalized

    template_proposals = _normalize_proposals(
        case_pack,
        _template_proposals(store, max_proposals=max_proposals * 2),
        store,
        max_proposals=max_proposals * 2,
    )
    template_proposals = _without_contracted_orientation_subjects(
        template_proposals, contracted_orientation_subjects
    )
    seen = {_proposal_key(proposal) for proposal in normalized}
    for proposal in template_proposals:
        key = _proposal_key(proposal)
        if key in seen:
            continue
        normalized.append(proposal)
        seen.add(key)
        if len(normalized) >= max_proposals:
            break
    normalized.sort(
        key=lambda item: (-item.priority, item.subject_id, ",".join(item.target_ids))
    )
    return normalized[:max_proposals]


def _without_contracted_orientation_subjects(
    proposals: list[FunctionalDependencyProposal],
    contracted_subjects: set[str],
) -> list[FunctionalDependencyProposal]:
    if not contracted_subjects:
        return proposals
    return [
        proposal
        for proposal in proposals
        if not (
            proposal.subject_id in contracted_subjects
            and proposal.relation_type
            in {"seating_to_media", "seating_to_work_surface"}
        )
    ]


def _propose_via_vlm(
    case_pack: dict[str, Any],
    store: GeometryStore,
    config: Any,
    *,
    max_proposals: int,
    progress,
) -> list[FunctionalDependencyProposal]:
    del case_pack, store, config, max_proposals
    progress("VLM relation proposal is not available; using deterministic templates")
    return []


def _fd_proposer_system_prompt() -> str:
    return (
        "You propose functional dependency checks for a 3D scene. "
        "Use only provided object ids. Return relation proposals only; do not judge pass/fail. "
        "Subjects list the objects that need proposals; each subject also includes preferred relations and candidate target ids. "
        "Targets list only additional candidate target objects. "
        "Prefer meaningful use relations such as chair-desk, sofa-TV, bed-nightstand, object-on-support, lamp-surface. "
        "Avoid clearance, visibility, and accessibility concerns."
    )


def _object_summary(obj: dict[str, Any]) -> dict[str, Any]:
    profile = object_function_profile(obj)
    return {
        "id": obj.get("id"),
        "category": object_category(obj),
        "affordances": list(object_affordances(obj)),
        "room": obj.get("room") or obj.get("room_id"),
        "function_profile": {
            "can_support_top": profile.can_support_top,
            "has_internal_shelf": profile.has_internal_shelf,
            "is_small_placeable": profile.is_small_placeable,
            "is_seating": profile.is_seating,
            "is_work_surface": profile.is_work_surface,
            "is_media_target": profile.is_media_target,
            "is_bedside_surface": profile.is_bedside_surface,
            "is_sleeping_surface": profile.is_sleeping_surface,
        },
    }


def _build_fd_proposer_payload(
    case_pack: dict[str, Any],
    store: GeometryStore,
    *,
    max_proposals: int,
) -> dict[str, Any]:
    subject_limit = max(max_proposals * PROPOSER_SUBJECT_MULTIPLIER, max_proposals)
    target_limit = max(max_proposals * PROPOSER_TARGET_MULTIPLIER, max_proposals)
    subjects = _proposer_subject_candidates(store, limit=subject_limit)
    subject_ids = {str(subject.get("id")) for subject in subjects}

    targets = _proposer_target_candidates(
        store,
        subjects=subjects,
        limit=target_limit,
    )
    target_map = {str(target.get("id")): target for target in targets}
    subject_payload = []
    for subject in subjects:
        relation_targets: dict[str, list[str]] = {}
        for relation in _preferred_relations_for_subject(subject):
            ranked = _rank_targets_for_relation(
                subject, relation, list(target_map.values())
            )
            if ranked:
                relation_targets[relation] = ranked[:PROPOSER_MAX_TARGETS_PER_RELATION]
        if not relation_targets:
            continue
        summary = _object_summary(subject)
        summary["preferred_relations"] = list(relation_targets)
        summary["candidate_target_ids"] = relation_targets
        subject_payload.append(summary)

    extra_targets = [
        _object_summary(target)
        for target_id, target in target_map.items()
        if target_id not in subject_ids
    ]
    return {
        "task_instruction": _compact_task_instruction(
            case_pack.get("task_instruction")
        ),
        "room_type": case_pack.get("room_type"),
        "max_proposals": max_proposals,
        "subjects": subject_payload,
        "targets": extra_targets,
    }


def _compact_task_instruction(task_instruction: Any) -> str:
    text = str(task_instruction or "").strip()
    if not text:
        return ""
    text = " ".join(text.split())
    if len(text) <= PROPOSER_MAX_TASK_CHARS:
        return text
    sentence_break = text.find(". ")
    if 0 < sentence_break < PROPOSER_MAX_TASK_CHARS:
        return text[: sentence_break + 1]
    return text[: PROPOSER_MAX_TASK_CHARS - 3].rstrip() + "..."


def _proposer_subject_candidates(
    store: GeometryStore, *, limit: int
) -> list[dict[str, Any]]:
    ranked: list[tuple[float, str, dict[str, Any]]] = []
    for obj in store.objects.values():
        relations = _preferred_relations_for_subject(obj)
        if not relations:
            continue
        best_priority = max(
            _proposal_priority(object_category(obj), relation) for relation in relations
        )
        ranked.append((best_priority, str(obj.get("id") or ""), obj))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    selected: list[dict[str, Any]] = []
    category_counts: dict[tuple[tuple[str, ...], str], int] = {}
    for _priority, _obj_id, obj in ranked:
        relations = tuple(_preferred_relations_for_subject(obj))
        category = object_category(obj)
        key = (relations, category)
        current = category_counts.get(key, 0)
        if current >= _subject_category_cap(relations):
            continue
        selected.append(obj)
        category_counts[key] = current + 1
        if len(selected) >= limit:
            break
    return selected


def _proposer_target_candidates(
    store: GeometryStore,
    *,
    subjects: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    ranked: dict[str, tuple[float, dict[str, Any]]] = {}
    for subject in subjects:
        for relation in _preferred_relations_for_subject(subject):
            for rank, target_id in enumerate(
                _rank_targets_for_relation(
                    subject, relation, list(store.objects.values())
                )
            ):
                target = store.objects.get(target_id)
                if target is None:
                    continue
                score = (
                    _proposal_priority(object_category(subject), relation) - rank * 0.01
                )
                current = ranked.get(target_id)
                if current is None or score > current[0]:
                    ranked[target_id] = (score, target)
    sorted_targets = sorted(
        ranked.items(),
        key=lambda item: (-item[1][0], item[0]),
    )
    return [target for _, (_, target) in sorted_targets[:limit]]


def _rank_targets_for_relation(
    subject: dict[str, Any],
    relation_type: str,
    objects: list[dict[str, Any]],
) -> list[str]:
    ranked: list[tuple[float, float, str]] = []
    for target in objects:
        target_id = str(target.get("id") or "")
        if not target_id or target_id == str(subject.get("id") or ""):
            continue
        if not _relation_target_is_valid(subject, target, relation_type):
            continue
        if (
            relation_type == "seating_to_work_surface"
            and not _is_actionable_seating_surface_pair(subject, target)
        ):
            continue
        if relation_type == "object_on_support":
            support_rank = _object_on_support_rank(subject, target)
            if support_rank is None:
                continue
            ranked.append((*support_rank, target_id))
            continue
        gap = bbox_gap_xy(subject, target)
        dist = distance_xy(subject, target)
        angle_penalty = _angle_penalty(subject, target, relation_type)
        ranked.append(
            (
                gap if gap is not None else 999.0,
                (dist if dist is not None else 999.0) + angle_penalty,
                target_id,
            )
        )
    ranked.sort(key=lambda item: (item[0], item[1], item[2]))
    return [target_id for _, _, target_id in ranked]


def _template_proposals(
    store: GeometryStore, *, max_proposals: int
) -> list[FunctionalDependencyProposal]:
    proposals: list[FunctionalDependencyProposal] = []
    objects = list(store.objects.values())
    for subject in objects:
        subject_category = object_category(subject)
        for relation in _preferred_relations_for_subject(subject):
            target = _best_template_target(subject, relation, objects)
            if target is None:
                continue
            proposals.append(
                FunctionalDependencyProposal(
                    subject_id=str(subject.get("id")),
                    target_ids=[str(target.get("id"))],
                    relation_type=relation,
                    expected_use=_expected_use(relation),
                    scoring_tier="core",
                    priority=_proposal_priority(subject_category, relation),
                    reason="Template fallback from category, semantics, and geometry.",
                )
            )
            break
        if len(proposals) >= max_proposals:
            break
    proposals.sort(
        key=lambda item: (-item.priority, item.subject_id, ",".join(item.target_ids))
    )
    return proposals[:max_proposals]


def _best_template_target(
    subject: dict[str, Any],
    relation_type: str,
    objects: list[dict[str, Any]],
) -> dict[str, Any] | None:
    target_by_id = {
        str(target.get("id") or ""): target for target in objects if target.get("id")
    }
    ranked_ids = _rank_targets_for_relation(subject, relation_type, objects)
    if not ranked_ids:
        return None
    return target_by_id[ranked_ids[0]]


def _normalize_proposals(
    case_pack: dict[str, Any],
    proposals: list[FunctionalDependencyProposal],
    store: GeometryStore,
    *,
    max_proposals: int,
) -> list[FunctionalDependencyProposal]:
    normalized: list[FunctionalDependencyProposal] = []
    seen: set[tuple[str, tuple[str, ...], str]] = set()
    object_ids = set(store.objects)
    for proposal in proposals:
        subject_id = str(proposal.subject_id or "")
        if subject_id not in object_ids:
            continue
        subject = store.objects[subject_id]
        target_ids = []
        for target_id in proposal.target_ids:
            target = str(target_id or "")
            if (
                target
                and target in object_ids
                and target != subject_id
                and target not in target_ids
            ):
                target_ids.append(target)
        if not target_ids:
            continue
        relation_type = proposal.relation_type or _infer_relation_type(
            subject, store.objects[target_ids[0]]
        )
        if not relation_type:
            continue
        compatible_targets: list[str] = []
        for target_id in target_ids:
            target = store.objects[target_id]
            resolved_relation = relation_type
            if not _relation_target_is_valid(subject, target, relation_type):
                inferred_relation = _infer_relation_type(subject, target)
                if not inferred_relation or not _relation_target_is_valid(
                    subject, target, inferred_relation
                ):
                    continue
                resolved_relation = inferred_relation
            if compatible_targets and resolved_relation != relation_type:
                continue
            relation_type = resolved_relation
            compatible_targets.append(target_id)
        if not compatible_targets:
            continue
        target_ids = compatible_targets
        key = (subject_id, tuple(target_ids), relation_type)
        if key in seen:
            continue
        if _should_skip_proposal(
            case_pack, subject, store.objects[target_ids[0]], relation_type
        ):
            continue
        seen.add(key)
        normalized.append(
            FunctionalDependencyProposal(
                subject_id=subject_id,
                target_ids=target_ids[:2],
                relation_type=relation_type,
                expected_use=proposal.expected_use or _expected_use(relation_type),
                scoring_tier=_proposal_scoring_tier(
                    case_pack,
                    subject,
                    store.objects[target_ids[0]],
                    relation_type,
                    proposal.scoring_tier,
                ),
                priority=proposal.priority,
                reason=proposal.reason,
            )
        )
    normalized = _prioritize_proposals(normalized, store)
    normalized.sort(
        key=lambda item: (
            _scoring_tier_rank(item.scoring_tier),
            -item.priority,
            item.subject_id,
            ",".join(item.target_ids),
        )
    )
    return normalized[:max_proposals]


def _should_skip_proposal(
    case_pack: dict[str, Any],
    subject: dict[str, Any],
    target: dict[str, Any],
    relation_type: str,
) -> bool:
    if relation_type == "seating_to_work_surface":
        return not _is_actionable_seating_surface_pair(subject, target)
    if relation_type != "object_on_support":
        return False
    subject_category = object_category(subject)
    scoring_tier = _proposal_scoring_tier(
        case_pack, subject, target, relation_type, "core"
    )
    if scoring_tier == "core":
        return False
    label, _confidence, _reason = _eval_object_on_support(
        subject, target, relation_type
    )
    return label == "fail"


def _expected_use(relation_type: str) -> str:
    return {
        "seating_to_work_surface": "sit at and use the work/table surface",
        "seating_to_media": "sit and view the media object",
        "bed_to_nightstand": "reach bedside storage or surface from the bed",
        "object_on_support": "small object is supported by an appropriate surface",
        "lamp_to_surface": "lamp serves the nearby support surface",
    }.get(relation_type, "nearby objects form a valid use relation")


def _proposal_priority(subject_category: str, relation_type: str) -> float:
    if relation_type in {
        "seating_to_work_surface",
        "seating_to_media",
    } and subject_category in {"sofa", "loveseat"}:
        return 0.95
    return {
        "seating_to_work_surface": 0.9,
        "seating_to_media": 0.85,
        "bed_to_nightstand": 0.8,
        "object_on_support": 0.7,
        "lamp_to_surface": 0.65,
    }.get(relation_type, 0.5)


def _proposal_key(
    proposal: FunctionalDependencyProposal,
) -> tuple[str, tuple[str, ...], str]:
    return proposal.subject_id, tuple(proposal.target_ids), proposal.relation_type


def _subject_category_cap(relations: tuple[str, ...]) -> int:
    relation_set = set(relations)
    if relation_set == {"object_on_support"}:
        return 2
    if relation_set == {"lamp_to_surface"}:
        return 2
    if relation_set == {"bed_to_nightstand"}:
        return 1
    return 4


def _proposal_scoring_tier(
    case_pack: dict[str, Any],
    subject: dict[str, Any],
    target: dict[str, Any],
    relation_type: str,
    raw_tier: Any,
) -> str:
    explicit = _normalize_scoring_tier(raw_tier)
    if explicit != "core":
        return explicit
    if relation_type == "seating_to_work_surface":
        if not _is_core_work_surface_target(target):
            return "auxiliary"
        gap = bbox_gap_xy(subject, target)
        if gap is not None and gap > 1.8:
            return "auxiliary"
        return "core"
    if relation_type == "seating_to_media":
        if object_category(subject) not in LIVING_ROOM_SEATING:
            return "auxiliary"
        return "core" if _is_core_media_target(target) else "auxiliary"
    if relation_type == "bed_to_nightstand":
        return "core"
    if relation_type == "lamp_to_surface":
        return "auxiliary"
    if relation_type != "object_on_support":
        return explicit

    subject_category = object_category(subject)
    if subject_category in CORE_SUPPORTED_SMALL and _task_mentions_object(
        case_pack, subject
    ):
        return "core"
    if not object_affordances(subject) and subject_category in DECORATIVE_SUPPORT_SMALL:
        return "ignored"
    if _task_mentions_object(case_pack, subject) or _task_mentions_object(
        case_pack, target
    ):
        return "auxiliary"
    return "auxiliary"


def _task_mentions_object(case_pack: dict[str, Any], obj: dict[str, Any]) -> bool:
    haystack = str(case_pack.get("task_instruction") or "").strip().lower()
    if not haystack:
        return False
    tokens = {
        token
        for token in re.split(r"[^a-z0-9]+", _object_text(obj))
        if token
        and token
        not in {
            "room",
            "generated",
            "assets",
            "scene",
            "s0",
            "s1",
            "s2",
            "s3",
            "s4",
            "s5",
        }
    }
    category = object_category(obj)
    if category:
        tokens.update(part for part in category.split("_") if part)
        tokens.add(category.replace("_", " "))
    return any(token and token in haystack for token in tokens)


def _prioritize_proposals(
    proposals: list[FunctionalDependencyProposal],
    store: GeometryStore,
) -> list[FunctionalDependencyProposal]:
    groups: dict[tuple[str, str], list[FunctionalDependencyProposal]] = {}
    for proposal in proposals:
        if proposal.scoring_tier != "core":
            continue
        if proposal.relation_type not in {
            "seating_to_work_surface",
            "seating_to_media",
        }:
            continue
        groups.setdefault((proposal.subject_id, proposal.relation_type), []).append(
            proposal
        )

    for (_subject_id, _relation_type), items in groups.items():
        if len(items) <= 1:
            continue
        ranked = sorted(items, key=lambda item: _proposal_rank_key(item, store))
        for item in ranked[1:]:
            item.scoring_tier = "auxiliary"
    return proposals


def _proposal_rank_key(
    proposal: FunctionalDependencyProposal,
    store: GeometryStore,
) -> tuple[float, float, float, str]:
    subject = store.objects.get(proposal.subject_id)
    target = store.objects.get(proposal.target_ids[0]) if proposal.target_ids else None
    if subject is None or target is None:
        return (999.0, 999.0, -proposal.priority, proposal.subject_id)
    gap = bbox_gap_xy(subject, target)
    dist = distance_xy(subject, target)
    angle_penalty = _angle_penalty(subject, target, proposal.relation_type)
    return (
        gap if gap is not None else 999.0,
        (dist if dist is not None else 999.0) + angle_penalty,
        -proposal.priority,
        proposal.subject_id,
    )
