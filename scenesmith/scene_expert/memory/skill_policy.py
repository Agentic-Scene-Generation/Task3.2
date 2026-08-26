"""Deterministic applicability and hard-contract policy for reusable skills."""

from __future__ import annotations

import json
import re

from dataclasses import dataclass
from typing import Any

from scenesmith.scene_expert.memory.room_taxonomy import (
    canonical_room_type,
    room_types_compatible,
)
from scenesmith.scene_expert.memory.schemas import Skill
from scenesmith.scene_expert.schemas import (
    SceneTaskSpec,
    SkillSelectionDecision,
    StageRelationContext,
)

_RELATION_ALIASES = {
    "facing": "faces",
    "face": "faces",
    "facing_toward": "faces",
    "away_from": "faces_away_from",
    "facing_away": "faces_away_from",
    "close_to": "near",
    "beside": "near",
    "far": "far_from",
    "centered_in_room": "room_center_alignment",
}
_CONFLICTING_RELATIONS = frozenset(
    {
        frozenset({"faces", "faces_away_from"}),
        frozenset({"left_of", "right_of"}),
        frozenset({"in_front_of", "behind"}),
        frozenset({"near", "far_from"}),
        frozenset({"against_wall", "room_center_alignment"}),
    }
)
_GENERIC_OBJECT_FAMILIES = {
    "chair": frozenset(
        {
            "armchair",
            "chair",
            "dining_chair",
            "guest_chair",
            "office_chair",
            "student_chair",
            "stool",
        }
    ),
    "desk": frozenset({"desk", "reception_desk", "student_desk", "teacher_desk"}),
    "table": frozenset(
        {
            "coffee_table",
            "conference_table",
            "desk",
            "dining_table",
            "dressing_table",
            "side_table",
            "table",
        }
    ),
}


def _normalize(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().casefold()).strip("_")


def _relation(value: Any) -> str:
    normalized = _normalize(value)
    return _RELATION_ALIASES.get(normalized, normalized)


def _object_role(value: Any) -> str:
    normalized = _normalize(value)
    # TaskCompiler inventory is free text and often plural, whereas intent
    # selectors are singular.  Keep the normalization local so the lightweight
    # memory retriever does not import Drake-backed critic packages.
    if normalized.endswith("ies"):
        normalized = normalized[:-3] + "y"
    elif normalized.endswith("ses"):
        normalized = normalized[:-2]
    elif normalized.endswith("s") and not normalized.endswith("ss"):
        normalized = normalized[:-1]
    return normalized


def _selector_category(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("category") or value.get("role")
    return _object_role(value)


def _category_compatible(first: str, second: str) -> bool:
    left = _object_role(first)
    right = _object_role(second)
    if not left or not right:
        return False
    if left == right:
        return True
    if any(
        (left == generic and right in members) or (right == generic and left in members)
        for generic, members in _GENERIC_OBJECT_FAMILIES.items()
    ):
        return True
    # Preserve meaningful modifiers while allowing explicit role inheritance,
    # e.g. office_chair satisfies a declared chair precondition.
    return left.endswith("_" + right) or right.endswith("_" + left)


def _task_objects(task_spec: SceneTaskSpec, stage: str) -> list[str]:
    by_stage = {
        "floor_plan": (
            task_spec.required_large_objects + task_spec.required_architectural_features
        ),
        "furniture": task_spec.required_large_objects,
        "wall_mounted": task_spec.required_wall_objects,
        "ceiling_mounted": task_spec.required_ceiling_objects,
        "manipuland": task_spec.required_small_objects,
    }
    return [_object_role(item) for item in by_stage.get(stage, [])]


def _contract_facts(
    relation_context: StageRelationContext | None,
) -> tuple[list[dict[str, Any]], set[str], set[str]]:
    constraints = list(relation_context.hard_constraints) if relation_context else []
    relations = {_relation(item.get("relation")) for item in constraints}
    objects = {
        category
        for item in constraints
        for category in (
            _selector_category(item.get("subjects")),
            _selector_category(item.get("targets")),
        )
        if category
    }
    return constraints, relations, objects


def _endpoints_match(
    skill_subject: str,
    skill_target: str,
    constraint: dict[str, Any],
) -> bool:
    constraint_subject = _selector_category(constraint.get("subjects"))
    constraint_target = _selector_category(constraint.get("targets"))
    if skill_subject and not _category_compatible(skill_subject, constraint_subject):
        return False
    if skill_target and not _category_compatible(skill_target, constraint_target):
        return False
    return bool(skill_subject or skill_target)


def _cardinality_conflicts(
    skill_cardinality: dict[str, Any],
    constraint: dict[str, Any],
) -> bool:
    if not skill_cardinality:
        return False
    subjects = constraint.get("subjects") or {}
    targets = constraint.get("targets") or {}
    comparisons = {
        "subject_count": subjects.get("count"),
        "target_count": targets.get("count"),
        "orientation": constraint.get("orientation"),
        "edge_frame": constraint.get("edge_frame"),
        "groups": constraint.get("groups"),
    }
    for key, contract_value in comparisons.items():
        skill_value = skill_cardinality.get(key)
        if skill_value is None or contract_value is None:
            continue
        if json.dumps(skill_value, sort_keys=True, default=str) != json.dumps(
            contract_value, sort_keys=True, default=str
        ):
            return True
    return False


def _hard_contract_conflicts(
    skill: Skill,
    constraints: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    reasons: list[str] = []
    constraint_ids: list[str] = []
    for spatial in skill.spatial_relations:
        skill_relation = _relation(spatial.relation_type)
        for constraint in constraints:
            if not _endpoints_match(
                spatial.subject_role, spatial.target_role, constraint
            ):
                continue
            contract_relation = _relation(constraint.get("relation"))
            relation_pair = frozenset({skill_relation, contract_relation})
            conflicts = relation_pair in _CONFLICTING_RELATIONS
            cardinality_conflict = (
                skill_relation == contract_relation
                and _cardinality_conflicts(spatial.cardinality, constraint)
            )
            if not conflicts and not cardinality_conflict:
                continue
            constraint_id = str(constraint.get("constraint_id") or "")
            if constraint_id:
                constraint_ids.append(constraint_id)
            if conflicts:
                reasons.append(
                    "hard_relation_conflict:"
                    f"{skill_relation}!={contract_relation}:{constraint_id or 'unnamed'}"
                )
            if cardinality_conflict:
                reasons.append(
                    "hard_cardinality_conflict:"
                    f"{skill_relation}:{constraint_id or 'unnamed'}"
                )
    return reasons, constraint_ids


@dataclass(frozen=True)
class SkillPolicyResult:
    """Internal result carrying both the decision row and eligibility flag."""

    eligible: bool
    decision: SkillSelectionDecision


def evaluate_skill_for_task(
    skill: Skill,
    task_spec: SceneTaskSpec,
    stage: str,
    relation_context: StageRelationContext | None = None,
) -> SkillPolicyResult:
    """Reject a skill before ranking when its explicit contract is unsafe."""
    reasons: list[str] = []
    conflicting_ids: list[str] = []
    constraints, contract_relations, contract_objects = _contract_facts(
        relation_context
    )
    task_objects = {*_task_objects(task_spec, stage), *contract_objects}
    declared_rooms = [
        *skill.room_types,
        *skill.applicability.room_types,
        *([skill.room_type] if skill.room_type else []),
    ]

    if skill.stage != stage:
        reasons.append("stage_mismatch")
    if any(
        room_types_compatible(room, task_spec.room_type)
        for room in skill.applicability.excluded_room_types
    ):
        reasons.append("excluded_room_type")
    if declared_rooms and not any(
        room_types_compatible(room, task_spec.room_type) for room in declared_rooms
    ):
        reasons.append("incompatible_room_type")

    required_roles = [
        _object_role(role)
        for role in (
            skill.applicability.required_object_roles or skill.required_objects
        )
        if _object_role(role)
    ]
    missing_roles = [
        role
        for role in required_roles
        if not any(
            _category_compatible(role, task_object) for task_object in task_objects
        )
    ]
    if missing_roles:
        reasons.append("missing_required_roles:" + ",".join(sorted(missing_roles)))

    required_relations = {
        _relation(value) for value in skill.applicability.required_relation_types
    }
    missing_relations = sorted(required_relations - contract_relations)
    if missing_relations:
        reasons.append("missing_required_relations:" + ",".join(missing_relations))

    hard_conflicts, hard_conflict_ids = _hard_contract_conflicts(skill, constraints)
    reasons.extend(hard_conflicts)
    conflicting_ids.extend(hard_conflict_ids)

    eligible = not reasons and skill.status == "active"
    matched_constraints = [
        str(item.get("constraint_id") or "")
        for item in constraints
        if _relation(item.get("relation")) in required_relations
        and str(item.get("constraint_id") or "")
    ]
    decision = SkillSelectionDecision(
        skill_name=skill.skill_name,
        decision="eligible" if eligible else "rejected",
        reasons=reasons,
        canonical_task_room=canonical_room_type(task_spec.room_type),
        canonical_skill_rooms=sorted(
            {canonical_room_type(room) for room in declared_rooms if room}
        ),
        required_object_roles=required_roles,
        required_relation_types=sorted(required_relations),
        matched_constraint_ids=matched_constraints,
        conflicting_constraint_ids=sorted(set(conflicting_ids)),
    )
    return SkillPolicyResult(eligible=eligible, decision=decision)
