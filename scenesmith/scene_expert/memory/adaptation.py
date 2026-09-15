"""Deterministic validation of Planner memory advice and decision-time scope.

This is not a second planner or a geometry repair engine. Source evidence stays
immutable; accepted actions remain advisory, subordinate to the current task.
"""

from __future__ import annotations

import json
import re

from scenesmith.scene_expert.memory.advisory import advice_check_reasons
from scenesmith.scene_expert.memory.placement_methods import (
    instruction_reasons,
    methods_valid,
)
from scenesmith.scene_expert.memory.schemas import (
    PlacementExperience,
    Skill,
    SpatialRelationMemory,
)
from scenesmith.scene_expert.memory.skill_policy import (
    _category_compatible,
    _endpoints_match,
    _hard_contract_conflicts,
    _relation,
)
from scenesmith.scene_expert.memory.state import observed_roles, state_hash
from scenesmith.scene_expert.schemas import (
    AcceptedMemoryItem,
    MemoryAdaptation,
    RetrievedMemorySelection,
    SceneTaskSpec,
    StageBrief,
    StageRelationContext,
)


def required_roles(task_spec: SceneTaskSpec | None, stage: str) -> list[str]:
    """Return the immutable stage inventory, not previously observed objects."""
    if task_spec is None:
        return []
    return list(
        {
            "floor_plan": task_spec.required_architectural_features
            + task_spec.required_large_objects,
            "furniture": task_spec.required_large_objects,
            "wall_mounted": task_spec.required_wall_objects,
            "ceiling_mounted": task_spec.required_ceiling_objects,
            "manipuland": task_spec.required_small_objects,
        }.get(stage, [])
    )


def source_conflicts(
    source: RetrievedMemorySelection, context: StageRelationContext | None
) -> list[str]:
    """Apply the same structured hard-intent guard to all memory record types."""
    if context is None:
        return []
    proxy = Skill(
        skill_name=source.memory_id,
        stage=context.stage,
        spatial_relations=[
            SpatialRelationMemory.model_validate(row)
            for row in source.spatial_relations
        ],
    )
    reasons, _ = _hard_contract_conflicts(
        proxy,
        [
            row
            for row in context.hard_constraints
            if str(row.get("strength") or "hard") == "hard"
        ],
    )
    return reasons


def scoped_relations(
    source: RetrievedMemorySelection, choice: MemoryAdaptation
) -> list[dict]:
    """Select original evidence rows, never rewrite the source or its hash.

    Legacy choices retain their original all-relations scope. Invalid scopes
    fail closed in validation; an empty scope cannot bypass spatial guards.
    """
    indices = choice.source_relation_indices
    if indices is None:
        return source.spatial_relations
    if len(set(indices)) != len(indices) or any(
        type(i) is not int or not 0 <= i < len(source.spatial_relations)
        for i in indices
    ):
        return []
    return [source.spatial_relations[i] for i in indices]


def relation_scope_reasons(
    source: RetrievedMemorySelection, choice: MemoryAdaptation
) -> list[str]:
    """Require explicit nonempty, in-range scope when spatial evidence exists."""
    indices = choice.source_relation_indices
    if indices is None:
        return []
    reasons = []
    steps = source.placement_experience.get("method_steps") or []
    selected = choice.source_method_step_indices
    advice_only = bool(selected) and all(
        type(i) is int
        and 0 <= i < len(steps)
        and not steps[i].get("episode_ids")
        and steps[i].get("critic_refs")
        for i in selected
    )
    if source.spatial_relations and not indices and not advice_only:
        reasons.append("empty_spatial_scope")
    if len(set(indices)) != len(indices):
        reasons.append("duplicate_source_relation_index")
    if any(
        type(i) is not int or not 0 <= i < len(source.spatial_relations)
        for i in indices
    ):
        reasons.append("unknown_source_relation_index")
    return reasons


def conflicts_with_accepted(
    source: RetrievedMemorySelection, accepted: list[AcceptedMemoryItem], stage: str
) -> bool:
    """Do not send incompatible spatial precedents in the same advice bundle."""
    rows = []
    for item in accepted:
        for relation in scoped_relations(item.source, item.adaptation):
            cardinality = relation.get("cardinality") or {}
            row = {
                key: value
                for key, value in cardinality.items()
                if not key.startswith(("subject_", "target_"))
            }
            row.update(
                {
                    "constraint_id": item.source.memory_id,
                    "relation": relation.get("relation_type"),
                    "subjects": {"category": relation.get("subject_role")},
                    "targets": {"category": relation.get("target_role")},
                }
            )
            if "subject_count" in cardinality:
                row["subjects"]["count"] = cardinality["subject_count"]
            if "target_count" in cardinality:
                row["targets"]["count"] = cardinality["target_count"]
            rows.append(row)
    return bool(
        source_conflicts(
            source, StageRelationContext(stage=stage, hard_constraints=rows)
        )
    )


def effective_relation_grade(row: dict) -> str:
    """Never promote an old boolean or an edited claim in the prompt view."""
    relation = SpatialRelationMemory.model_validate(row)
    if relation.has_verified_geometry:
        return "verified_pass"
    if relation.has_verified_failure:
        return "verified_fail"
    return (
        "requirement_only"
        if relation.verification_status == "requirement_only"
        else "unverified"
    )


def _template_rows(relation: dict, context: StageRelationContext | None) -> list[dict]:
    return [
        row
        for row in (context.hard_constraints if context else [])
        if _relation(row.get("relation") or row.get("relation_type"))
        == _relation(relation.get("relation_type"))
        and _endpoints_match(
            str(relation.get("subject_role") or ""),
            str(relation.get("target_role") or ""),
            row,
        )
    ]


def validate_adaptation(
    source: RetrievedMemorySelection,
    choice: MemoryAdaptation,
    *,
    stage: str,
    task_spec: SceneTaskSpec | None,
    context: StageRelationContext | None,
    state: dict,
    brief: StageBrief | None,
) -> list[str]:
    """Reject unsupported identities/bindings and explicit contract conflicts."""
    # Keep the existing full-source hard-conflict guard. Scoping limits what
    # must be bound/rendered; it is not permission to evade an intent conflict.
    reasons = source_conflicts(source, context) + relation_scope_reasons(source, choice)
    reasons += advice_check_reasons(
        source.model_dump(mode="json"),
        choice.model_dump(mode="json"),
        context.model_dump(mode="json") if context else {},
    )
    relations = scoped_relations(source, choice)
    quoted_roles: set[str] = set()
    if (
        source.placement_experience.get("method_steps")
        or source.placement_experience.get("schema_version")
        == "placement-experience.v2"
    ):
        try:
            experience = PlacementExperience.model_validate(source.placement_experience)
        except (ValueError, TypeError):
            return list(dict.fromkeys([*reasons, "invalid_method_step_contract"]))
        if not methods_valid(experience):
            reasons.append("invalid_method_step_contract")
        indices = choice.source_method_step_indices
        indices = (
            list(range(len(experience.method_steps))) if indices is None else indices
        )
        if (
            not indices
            or len(set(indices)) != len(indices)
            or any(
                type(i) is not int or not 0 <= i < len(experience.method_steps)
                for i in indices
            )
        ):
            reasons.append("invalid_source_method_scope")
        else:
            scoped_ids = {r.get("evidence_ref") for r in relations}
            if any(
                set(experience.method_steps[i].episode_ids) - scoped_ids
                for i in indices
            ):
                reasons.append("method_step_source_outside_relation_scope")
            quote_ids = {
                ref for i in indices for ref in experience.method_steps[i].critic_refs
            }
            quoted_roles = {
                role
                for quote in experience.critic_advice
                if quote.evidence_id in quote_ids
                for role in quote.object_roles
            }
        # IDs such as chair_1 are legitimate current bindings, not copied source
        # distances. Ignore only exact IDs actually present in the current state.
        current_ids = {
            str(o.get("object_id"))
            for o in state.get("objects", [])
            if o.get("object_id")
        }

        def without_current_ids(text: str) -> str:
            for oid in sorted(current_ids, key=len, reverse=True):
                text = re.sub(
                    r"(?<!\w)" + re.escape(oid) + r"(?!\w)", "current asset", text
                )
            return text

        if any(
            instruction_reasons(without_current_ids(s))
            for s in [*choice.actions, *choice.checks, *choice.preconditions]
        ):
            reasons.append("unverified_memory_target_or_guarantee")
    if state.get("observation_error"):
        reasons.append("scene_observation_unavailable")
    if not source.content_hash or choice.source_content_hash != source.content_hash:
        reasons.append("source_hash_mismatch")
    if not any(x.strip() for x in choice.actions) or not any(
        x.strip() for x in choice.checks
    ):
        reasons.append("missing_action_or_check")
    allowed_roles = required_roles(task_spec, stage) + observed_roles(state)
    for constraint in (context.hard_constraints if context else []):
        for plural, singular in (("subjects", "subject"), ("targets", "target")):
            selector = constraint.get(plural) or constraint.get(singular)
            if isinstance(selector, dict):
                allowed_roles.append(
                    str(selector.get("category") or selector.get("role") or "")
                )
    if brief is not None and brief.optional_assets_allowed:
        allowed_roles += [item.name for item in brief.optional_asset_recommendations]
    objects = {row["object_id"]: row for row in state.get("objects", [])}
    source_roles = {
        str(row.get(key) or "")
        for row in relations
        for key in ("subject_role", "target_role")
    } - {""}
    for binding in choice.bindings:
        if not _category_compatible(binding.source_role, binding.current_role):
            reasons.append("incompatible_role_transfer")
        if not binding.current_role.strip() or not any(
            _category_compatible(binding.current_role, role) for role in allowed_roles
        ):
            reasons.append("unbound_current_role")
        if source_roles and not any(
            _category_compatible(binding.source_role, role)
            for role in source_roles | quoted_roles
        ):
            reasons.append("unknown_source_role")
        for object_id in binding.object_ids:
            row = objects.get(object_id)
            if row is None:
                reasons.append("unknown_object_id")
            elif not any(
                _category_compatible(binding.current_role, str(row.get(key) or ""))
                for key in ("name", "category")
            ):
                reasons.append("object_role_mismatch")
    if source_roles and any(
        not any(
            _category_compatible(role, binding.source_role)
            for binding in choice.bindings
        )
        for role in source_roles
    ):
        reasons.append("missing_role_binding")
    # Parameterized templates must be bound to a current explicit contract; do
    # not substitute source counts or let the LLM invent new hard quantities.
    for relation in relations:
        if relation.get("template_parameters"):
            candidates = _template_rows(relation, context)
            if not candidates:
                reasons.append("unbound_template_parameters")
    return list(dict.fromkeys(reasons))


def render_accepted_item(
    source: RetrievedMemorySelection,
    choice: MemoryAdaptation,
    context: StageRelationContext | None,
) -> str:
    """Render adapted actions only; raw source advice/layout cannot reappear."""
    relations = scoped_relations(source, choice)
    grades = [effective_relation_grade(row) for row in relations]
    lines = [
        f"[Memory {source.memory_type}:{source.memory_id} / {choice.decision}]",
        "Source evidence grades: " + ", ".join(grades or ["legacy/unverified"]),
    ]
    if source.memory_type == "skill":
        lines.append(f"[Skill: {source.memory_id}]")
    if source.placement_experience:
        lines.append(
            "Transfer hypotheses only: source observations and critic opinions do not prove this method works here. Recompute using current assets; do not impose source constants as requirements."
        )
        if source.placement_experience.get("method_steps"):
            indices = choice.source_method_step_indices
            indices = (
                list(range(len(source.placement_experience["method_steps"])))
                if indices is None
                else indices
            )
            lines.append(
                "Selected source method steps (zero-based): "
                + ", ".join(map(str, indices))
            )
    if choice.advice_checks:
        lines.append(
            "Read-only spatial observations (not additional scoring criteria):"
        )
        lines.extend(
            "- " + json.dumps(check.model_dump(mode="json"), ensure_ascii=False)
            for check in choice.advice_checks
        )
    if relations:
        lines.append(
            "Source relation semantics (advisory observations, not new requirements):"
        )
        lines.extend(
            "- " + SpatialRelationMemory.model_validate(row).to_guidance_text()
            for row in relations
        )
    for binding in choice.bindings:
        lines.append(
            f"Bind {binding.source_role} -> {binding.current_role}; current IDs={binding.object_ids or 'to be selected in this task'}"
        )
    for title, values in (
        ("Preconditions", choice.preconditions),
        ("Actions", choice.actions),
        ("Checks", choice.checks),
    ):
        if values:
            lines.append(title + ":")
            lines.extend("- " + value.strip() for value in values if value.strip())
    if any(row.get("template_parameters") for row in relations) and context:
        lines.append(
            "Bind template quantities from current intent, never source counts: "
            + json.dumps(
                [
                    row
                    for relation in relations
                    if relation.get("template_parameters")
                    for row in _template_rows(relation, context)
                ],
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    return "\n".join(lines)


def decision_applicability(
    item: AcceptedMemoryItem,
    *,
    stage: str,
    source_state: dict,
    current_state: dict,
    query: str,
    repair: bool,
    context: StageRelationContext | None,
) -> list[str]:
    """Recheck cached advice against a changed scene; never call an LLM here."""
    reasons = source_conflicts(item.source, context) + relation_scope_reasons(
        item.source, item.adaptation
    )
    if source_state.get("observation_error") or current_state.get("observation_error"):
        reasons.append("scene_observation_unavailable")
    if source_state.get("room") != current_state.get("room"):
        reasons.append("room_geometry_changed")
    previous = {row["object_id"]: row for row in source_state.get("objects", [])}
    current = {row["object_id"]: row for row in current_state.get("objects", [])}
    for binding in item.adaptation.bindings:
        for object_id in binding.object_ids:
            if object_id not in current:
                reasons.append("bound_object_missing")
            elif object_id in previous and state_hash(
                previous[object_id]
            ) != state_hash(current[object_id]):
                reasons.append("bound_object_state_changed")
    if repair:
        # Local relevance is additional to main's unmodified critique/request.
        # A generic repair instruction is not a license to resend the whole bank.
        terms = [b.current_role for b in item.adaptation.bindings]
        terms += [oid for b in item.adaptation.bindings for oid in b.object_ids]
        terms += [
            str(row.get("relation_type") or "")
            for row in scoped_relations(item.source, item.adaptation)
        ]
        if not terms:
            terms = item.adaptation.checks
        tokens = {
            token
            for term in terms
            for token in re.findall(
                r"[a-z0-9\u4e00-\u9fff]+", term.casefold().replace("_", " ")
            )
            if len(token) > 3
        }
        query_tokens = set(
            re.findall(r"[a-z0-9\u4e00-\u9fff]+", query.casefold().replace("_", " "))
        )
        if not tokens.intersection(query_tokens):
            reasons.append("repair_issue_mismatch")
    return list(dict.fromkeys(reasons))
