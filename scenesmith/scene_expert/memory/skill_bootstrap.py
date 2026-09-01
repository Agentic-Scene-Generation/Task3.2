"""Deterministically bootstrap Skill candidates from verified stage evidence.

The MemoryWriter model remains responsible for semantic lesson extraction, but
it is not allowed to be the sole gateway to the Skill lifecycle.  This module
builds conservative procedural candidates directly from three authoritative
signals that already exist in the SceneExpert trace:

* the prompt-derived task contract,
* proof that the native SceneSmith stage agent actually ran, and
* a passing report from the main critic bridge.

Planner prose, retrieved memory, and model-proposed ``recommended_skills`` are
deliberately excluded.  That prevents an injected Skill from citing itself as
new independent support.  Every returned draft is still only a candidate; the
store requires compatible evidence from another task before retrieval can see
it.
"""

from __future__ import annotations

import re

from dataclasses import dataclass
from typing import Any

from scenesmith.scene_expert.memory.schemas import SkillMemoryCandidate

STAGE_ORDER = (
    "floor_plan",
    "furniture",
    "wall_mounted",
    "ceiling_mounted",
    "manipuland",
)

_REQUIRED_OBJECT_KEYS = {
    # Floor-plan does not place the large assets.  It may still produce a Skill
    # when its own hard relation contract contains grounded geometry evidence.
    "floor_plan": "",
    "furniture": "required_large_objects",
    "wall_mounted": "required_wall_objects",
    "ceiling_mounted": "required_ceiling_objects",
    "manipuland": "required_small_objects",
}


@dataclass(frozen=True)
class GroundedSkillDraft:
    """One deterministic draft plus the evidence used to construct it."""

    candidate: SkillMemoryCandidate
    stage: str
    required_objects: tuple[str, ...]
    relation_types: tuple[str, ...]
    constraint_ids: tuple[str, ...]


@dataclass(frozen=True)
class SkillBootstrapResult:
    """Skill drafts and a complete per-stage eligibility audit."""

    drafts: tuple[GroundedSkillDraft, ...]
    decisions: tuple[dict[str, Any], ...]

    @property
    def eligible_passing_stages(self) -> list[str]:
        return _unique(
            str(item.get("stage") or "")
            for item in self.decisions
            if item.get("decision") == "generated"
        )


def bootstrap_grounded_skills(
    evidence_payload: dict[str, Any],
    *,
    max_candidates: int = 5,
    min_procedure_steps: int = 2,
) -> SkillBootstrapResult:
    """Return conservative Skill drafts from exact-stage passing evidence.

    One relation-scoped draft is emitted per distinct hard-contract relation;
    an additional coverage draft represents required-first execution.  This is
    intentionally narrower than one task-sized mega-Skill, so compatible
    evidence can transfer even when two prompts contain different optional or
    unrelated assets.
    """

    stage_map = {
        str(item.get("stage") or ""): item
        for item in evidence_payload.get("stages", []) or []
        if isinstance(item, dict)
    }
    task_spec = dict(evidence_payload.get("task_spec") or {})
    room_type = _clean_text(task_spec.get("room_type")) or "room"
    drafts: list[GroundedSkillDraft] = []
    decisions: list[dict[str, Any]] = []

    for stage in STAGE_ORDER:
        stage_evidence = stage_map.get(stage, {})
        reasons: list[str] = []
        if not stage_evidence:
            reasons.append("missing_stage_evidence")
        report = _mapping(stage_evidence.get("verify_report"))
        execution = _mapping(stage_evidence.get("execution_evidence"))
        if not report or not bool(report.get("pass_stage")):
            reasons.append("stage_not_verified_passing")
        if not bool(execution.get("stage_agent_invoked")):
            reasons.append("stage_agent_not_proven_invoked")
        if _has_deterministic_failure(report):
            reasons.append("deterministic_stage_failure")
        relation_context = _mapping(stage_evidence.get("relation_context"))
        constraints = [
            item
            for item in relation_context.get("hard_constraints", []) or []
            if isinstance(item, dict)
        ]
        required_objects = _required_objects(
            task_spec=task_spec,
            execution_evidence=execution,
            stage=stage,
        )
        relations = _relation_descriptors(constraints)
        if not required_objects and not relations:
            reasons.append("no_grounded_task_contract")

        if reasons:
            decisions.append(
                {
                    "stage": stage,
                    "decision": "skipped",
                    "reasons": _unique(reasons),
                    "required_object_count": len(required_objects),
                    "hard_relation_count": len(relations),
                }
            )
            continue
        scoped_drafts = _stage_drafts(
            stage=stage,
            room_type=room_type,
            required_objects=required_objects,
            relations=relations,
            min_procedure_steps=max(2, int(min_procedure_steps)),
        )
        if not scoped_drafts:
            decisions.append(
                {
                    "stage": stage,
                    "decision": "skipped",
                    "reasons": ["insufficient_grounded_procedure"],
                    "required_object_count": len(required_objects),
                    "hard_relation_count": len(relations),
                }
            )
            continue
        for draft in scoped_drafts:
            if len(drafts) >= max(0, int(max_candidates)):
                decisions.append(
                    {
                        "stage": stage,
                        "decision": "skipped",
                        "reasons": ["scene_candidate_limit_reached"],
                        "skill_name": draft.candidate.skill_name,
                        "required_object_count": len(draft.required_objects),
                        "hard_relation_count": len(draft.relation_types),
                    }
                )
                continue
            drafts.append(draft)
            decisions.append(
                {
                    "stage": stage,
                    "decision": "generated",
                    "reasons": [],
                    "skill_name": draft.candidate.skill_name,
                    "required_object_count": len(draft.required_objects),
                    "hard_relation_count": len(draft.relation_types),
                    "relation_types": list(draft.relation_types),
                    "constraint_ids": list(draft.constraint_ids),
                }
            )

    return SkillBootstrapResult(drafts=tuple(drafts), decisions=tuple(decisions))


def _required_objects(
    *,
    task_spec: dict[str, Any],
    execution_evidence: dict[str, Any],
    stage: str,
) -> list[str]:
    key = _REQUIRED_OBJECT_KEYS.get(stage, "")
    from_task = task_spec.get(key, []) if key else []
    from_execution = execution_evidence.get("required_objects", [])
    return _clean_list([*(from_task or []), *(from_execution or [])])


def _relation_descriptors(
    constraints: list[dict[str, Any]],
) -> list[tuple[str, str, str, dict[str, Any]]]:
    output: list[tuple[str, str, str, dict[str, Any]]] = []
    seen: set[tuple[str, str, str]] = set()
    for constraint in constraints:
        relation = _clean_text(
            constraint.get("relation_type")
            or constraint.get("relation")
            or constraint.get("predicate")
            or constraint.get("type")
        )
        if not relation:
            continue
        subject = _selector_label(
            constraint.get("subject")
            or constraint.get("subjects")
            or constraint.get("subject_selector")
            or constraint.get("source")
        )
        target = _selector_label(
            constraint.get("target")
            or constraint.get("targets")
            or constraint.get("target_selector")
            or constraint.get("reference")
        )
        identity = (relation.casefold(), subject.casefold(), target.casefold())
        if identity in seen:
            continue
        seen.add(identity)
        output.append((relation, subject, target, constraint))
    return output


def _stage_drafts(
    *,
    stage: str,
    room_type: str,
    required_objects: list[str],
    relations: list[tuple[str, str, str, dict[str, Any]]],
    min_procedure_steps: int,
) -> list[GroundedSkillDraft]:
    drafts: list[GroundedSkillDraft] = []
    for relation, subject, target, constraint in relations:
        if relation == "required_count":
            continue
        scoped_objects = _unique([subject, target])
        relation_types = [relation]
        procedure = _relation_procedure(
            relation=relation,
            subject=subject,
            target=target,
            constraint=constraint,
        )
        if len(procedure) < min_procedure_steps:
            continue
        constraint_ids = tuple(
            _unique([str(constraint.get("constraint_id") or "")])
        )
        name = _skill_name(stage, relation_types, scoped_objects)
        drafts.append(
            GroundedSkillDraft(
                candidate=SkillMemoryCandidate(
                    skill_name=name,
                    stage=stage,
                    preconditions=_preconditions(
                        room_type, scoped_objects, relation_types
                    ),
                    procedure=procedure,
                    failure_avoidance=_failure_avoidance(relation_types),
                    postconditions=_postconditions(scoped_objects, relation_types),
                ),
                stage=stage,
                required_objects=tuple(scoped_objects),
                relation_types=tuple(relation_types),
                constraint_ids=constraint_ids,
            )
        )

    if required_objects:
        relation_types = ["required_coverage"]
        procedure = [
            "Place and stabilize the prompt-required assets before adding optional "
            "assets: " + ", ".join(required_objects) + ".",
            "Before completing the stage, verify every prompt-required asset is "
            "present, usable, and accepted by the authoritative hard checks.",
        ]
        if len(procedure) >= min_procedure_steps:
            required_constraint_ids = tuple(
                _unique(
                    str(constraint.get("constraint_id") or "")
                    for relation, _, _, constraint in relations
                    if relation == "required_count"
                )
            )
            name = _skill_name(stage, relation_types, required_objects)
            drafts.append(
                GroundedSkillDraft(
                    candidate=SkillMemoryCandidate(
                        skill_name=name,
                        stage=stage,
                        preconditions=_preconditions(
                            room_type, required_objects, relation_types
                        ),
                        procedure=procedure,
                        failure_avoidance=[],
                        postconditions=_postconditions(
                            required_objects, relation_types
                        ),
                    ),
                    stage=stage,
                    required_objects=tuple(required_objects),
                    relation_types=tuple(relation_types),
                    constraint_ids=required_constraint_ids,
                )
            )
    return drafts


def _relation_procedure(
    *,
    relation: str,
    subject: str,
    target: str,
    constraint: dict[str, Any],
) -> list[str]:
    endpoints = " and ".join(value for value in (subject, target) if value)
    cardinality = _cardinality_text(constraint)
    detail = f" between {endpoints}" if endpoints else ""
    if cardinality:
        # The exact counts remain in the source trace.  A transferable Skill
        # must follow the *current* task's cardinality instead of replaying the
        # source task's numbers into another scene.
        detail += " using the current task's explicit cardinality"
    return [
        f"Bind the applicable assets to the hard {relation} contract{detail} "
        "before placing unrelated optional assets.",
        f"Preserve {relation} while refining the layout, then verify that exact "
        "relation with the authoritative critic before completing the stage.",
    ]


def _preconditions(
    room_type: str,
    required_objects: list[str],
    relation_types: list[str],
) -> list[str]:
    values = [f"The current room is compatible with room type {room_type}."]
    if required_objects:
        values.append(
            "The task explicitly requires these stage assets: "
            + ", ".join(required_objects)
            + "."
        )
    if relation_types:
        values.append(
            "The task contract includes these relation families: "
            + ", ".join(relation_types)
            + "."
        )
    return values


def _failure_avoidance(relation_types: list[str]) -> list[str]:
    return [
        "Do not infer hard-contract satisfaction from visual proximity alone; "
        "verify " + relation + " explicitly."
        for relation in relation_types
        if relation != "required_coverage"
    ]


def _postconditions(
    required_objects: list[str], relation_types: list[str]
) -> list[str]:
    output: list[str] = []
    if required_objects:
        output.append("All prompt-required stage assets are present and usable.")
    if relation_types:
        output.append(
            "The authoritative critic reports all applicable hard relations passing."
        )
    return output


def _skill_name(
    stage: str, relation_types: list[str], object_roles: list[str]
) -> str:
    families = [_slug(value) for value in relation_types[:1] if _slug(value)]
    roles = [_slug(value) for value in object_roles[:2] if _slug(value)]
    suffix = "_".join([*families, *roles]) or "required_first"
    return f"verified_{_slug(stage)}_{suffix}_procedure"


def _cardinality_text(constraint: dict[str, Any]) -> str:
    values: list[str] = []
    for key in ("count", "min_count", "max_count", "quantifier"):
        if constraint.get(key) is not None:
            values.append(f"{key}={constraint[key]}")
    for prefix in ("subject", "target"):
        selector = constraint.get(prefix) or constraint.get(prefix + "s")
        if not isinstance(selector, dict):
            continue
        for key in ("count", "min_count", "max_count", "quantifier"):
            if selector.get(key) is not None:
                values.append(f"{prefix}_{key}={selector[key]}")
    return ", ".join(values)


def _has_deterministic_failure(report: dict[str, Any]) -> bool:
    hard_report = _mapping(report.get("hard_check_report"))
    return bool(
        hard_report.get("hard_valid") is False
        or hard_report.get("pass") is False
        or hard_report.get("failed_checks")
        or hard_report.get("hard_failures")
    )


def _selector_label(value: Any) -> str:
    if isinstance(value, str):
        return _clean_text(value)
    if not isinstance(value, dict):
        return ""
    for key in ("role", "category", "object", "name", "id"):
        if label := _clean_text(value.get(key)):
            return label
    return ""


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _clean_list(values: Any) -> list[str]:
    if not isinstance(values, (list, tuple, set)):
        return []
    return _unique(_clean_text(value) for value in values)


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _slug(value: Any) -> str:
    return re.sub(
        r"[^a-z0-9一-鿿]+", "_", str(value or "").casefold()
    ).strip("_")


def _unique(values: Any) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _clean_text(value)
        key = text.casefold()
        if text and key not in seen:
            output.append(text)
            seen.add(key)
    return output
