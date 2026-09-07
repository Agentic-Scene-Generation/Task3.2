"""Build the unique accepted memory bundle; never reappend raw recall text."""

from __future__ import annotations

from scenesmith.scene_expert.memory.adaptation import (
    conflicts_with_accepted,
    render_accepted_item,
    validate_adaptation,
)
from scenesmith.scene_expert.schemas import (
    AcceptedMemoryItem,
    MemoryInjectionBundle,
    MemoryPack,
    SceneTaskSpec,
    StageBrief,
    StageRelationContext,
)

MEMORY_START = "=== Accepted Cross-task Memory (advisory) ==="
MEMORY_END = "=== End Accepted Cross-task Memory ==="


def format_accepted_memory(items: list[AcceptedMemoryItem]) -> str:
    """Render whole accepted items with an explicit task/critic priority rule."""
    if not items:
        return ""
    return "\n\n".join(
        [
            MEMORY_START,
            "Current task, actual scene and current critic feedback take priority. These are optional design aids, not required assets or scoring criteria. Recheck anchors and tool conventions; never replay source asset IDs or world coordinates.",
            *[item.text for item in items],
            MEMORY_END,
        ]
    )


def enrich_stage_brief_with_memory(
    stage_brief: StageBrief, memory_pack: MemoryPack
) -> StageBrief:
    """Compatibility entrypoint: raw memory must not override Planner rejection."""
    return stage_brief.model_copy(deep=True)


def _task_only_brief(brief: StageBrief, pack: MemoryPack) -> StageBrief:
    """Prevent known verbatim memory from leaking through generic brief fields.

    This guards exact text, not an oracle for arbitrary LLM paraphrases. The
    Planner schema/prompt separates advisory memory from immutable task fields.
    """
    snippets = []
    for row in pack.selections:
        snippets.extend(row.injected_text.splitlines())
    for choice in brief.memory_adaptations:
        snippets.extend(choice.actions + choice.checks + choice.preconditions)
    normalized = [
        " ".join(text.strip(" -0123456789.").casefold().split()) for text in snippets
    ]
    normalized = [text for text in normalized if len(text) >= 24]

    def from_memory(text: str) -> bool:
        value = " ".join(text.casefold().split())
        return any(
            fragment in value or (len(value) >= 24 and value in fragment)
            for fragment in normalized
        )

    updates = {"recommended_skills": []}
    for field in (
        "constraints_for_designer",
        "checks_for_critic",
        "failure_patterns_to_avoid",
    ):
        updates[field] = [
            text for text in getattr(brief, field) if not from_memory(text)
        ]
    if from_memory(brief.stage_objective):
        updates["stage_objective"] = (
            f"Complete the {brief.stage} stage according to the current task."
        )
    updates["optional_asset_recommendations"] = [
        item
        for item in brief.optional_asset_recommendations
        if not any(
            from_memory(value)
            for value in (item.name, item.rationale, item.placement_guidance)
        )
    ]
    return brief.model_copy(update=updates)


def build_memory_injection_bundle(
    *,
    stage: str,
    stage_brief: StageBrief | None,
    memory_pack: MemoryPack,
    task_spec: SceneTaskSpec | None = None,
    relation_context: StageRelationContext | None = None,
    max_chars: int | None = None,
) -> MemoryInjectionBundle:
    """Validate explicit Planner choices and build exactly one advisory block.

    Old/fallback planners without choices abstain. A recommended skill name is
    not sufficient acceptance. Rejected/missing/changed sources are never used.
    """
    pack = memory_pack.deduplicated()
    choices = stage_brief.memory_adaptations if stage_brief else []
    choice_map = {}
    for choice in choices:
        key = (choice.memory_type, choice.memory_id)
        choice_map[key] = None if key in choice_map else choice
    limit = max(
        0,
        int(
            max_chars
            if max_chars is not None
            else pack.selection_policy.get("max_total_chars", 8000)
        ),
    )
    accepted: list[AcceptedMemoryItem] = []
    decisions = []
    source_keys = {(row.memory_type, row.memory_id) for row in pack.selections}
    for source in pack.selections:
        key = (source.memory_type, source.memory_id)
        choice = choice_map.get(key)
        reasons = []
        if choice is None:
            reasons = [
                (
                    "duplicate_planner_choice"
                    if key in choice_map
                    else "planner_not_accepted"
                )
            ]
        elif choice.decision == "rejected":
            reasons = ["planner_rejected"]
        else:
            reasons = validate_adaptation(
                source,
                choice,
                stage=stage,
                task_spec=task_spec,
                context=relation_context,
                state=pack.current_scene_state,
                brief=stage_brief,
            )
            if conflicts_with_accepted(source, accepted, stage):
                reasons.append("accepted_memory_conflict")
        if not reasons and choice is not None:
            item = AcceptedMemoryItem(
                source=source,
                adaptation=choice,
                text=render_accepted_item(source, choice, relation_context),
            )
            if any(
                old.text.split("\n", 1)[-1] == item.text.split("\n", 1)[-1]
                for old in accepted
            ):
                reasons.append("duplicate_adapted_content")
            elif len(format_accepted_memory([*accepted, item])) > limit:
                reasons.append("adapted_bundle_budget_pruned")
            else:
                accepted.append(item)
        decisions.append(
            {
                "memory_type": source.memory_type,
                "memory_id": source.memory_id,
                "decision": "rejected" if reasons else choice.decision,
                "reasons": reasons,
                "planner_reason": choice.reason if choice else "",
            }
        )
    for choice in choices:
        if (choice.memory_type, choice.memory_id) not in source_keys:
            decisions.append(
                {
                    "memory_type": choice.memory_type,
                    "memory_id": choice.memory_id,
                    "decision": "rejected",
                    "reasons": ["unknown_memory_id"],
                }
            )
    # Generic task guidance stays independent. Memory-owned text is rendered
    # only from the accepted list, never recommended_skills or raw hints/layout.
    clean_brief = _task_only_brief(stage_brief, pack) if stage_brief else None
    brief_text = clean_brief.to_injection_text() if clean_brief else ""
    memory_text = format_accepted_memory(accepted)
    skills = [
        item.source.memory_id for item in accepted if item.source.memory_type == "skill"
    ]
    return MemoryInjectionBundle(
        stage=stage,
        planner_stage_brief=stage_brief,
        enriched_stage_brief=clean_brief,
        brief_text=brief_text,
        memory_text=memory_text,
        final_text="\n\n".join(x for x in (brief_text, memory_text) if x),
        selected_memory_ids=[item.source.memory_id for item in accepted],
        retrieved_skill_names=pack.skill_names,
        planner_selected_skill_names=skills,
        prompt_delivered_skill_names=skills,
        accepted_items=accepted,
        adaptation_decisions=decisions,
        current_scene_state=pack.current_scene_state,
        task_spec=task_spec.model_dump(mode="json") if task_spec else {},
        relation_context=(
            relation_context.model_dump(mode="json") if relation_context else {}
        ),
    )
