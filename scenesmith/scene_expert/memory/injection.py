"""Canonical transformation from retrieved memory to one prompt injection block."""

from __future__ import annotations

from scenesmith.scene_expert.schemas import (
    MemoryInjectionBundle,
    MemoryPack,
    StageBrief,
)


def _normalized_skill_name(value: str) -> str:
    return "_".join(str(value or "").strip().casefold().replace("-", " ").split())


def _compact(text: str, max_chars: int = 300) -> str:
    compact = " ".join(str(text or "").strip().split())
    return compact if len(compact) <= max_chars else compact[: max_chars - 3] + "..."


def _extend_unique(target: list[str], values: list[str]) -> list[str]:
    seen = {value.strip().casefold() for value in target if value.strip()}
    for value in values:
        text = value.strip()
        key = text.casefold()
        if text and key not in seen:
            target.append(text)
            seen.add(key)
    return target


def enrich_stage_brief_with_memory(
    stage_brief: StageBrief,
    memory_pack: MemoryPack,
) -> StageBrief:
    """Project success/failure lessons into the brief exactly once."""
    success_rules = [
        "Retrieved success memory: " + _compact(hint)
        for hint in memory_pack.success_hints[:3]
    ]
    failure_rules = [_compact(hint) for hint in memory_pack.failure_hints[:3]]
    return stage_brief.model_copy(
        update={
            "constraints_for_designer": _extend_unique(
                list(stage_brief.constraints_for_designer), success_rules
            ),
            "failure_patterns_to_avoid": _extend_unique(
                list(stage_brief.failure_patterns_to_avoid), failure_rules
            ),
        }
    )


def _format_skill_procedures(memory_pack: MemoryPack) -> str:
    """Keep full procedures direct; success/failure text already lives in the brief."""
    if not memory_pack.skill_texts:
        return ""
    parts = ["=== SceneExpert Retrieved Skill Procedures ==="]
    parts.extend(text.strip() for text in memory_pack.skill_texts[:2] if text.strip())
    parts.append("=== End Retrieved Skill Procedures ===")
    return "\n".join(parts)


def build_memory_injection_bundle(
    *,
    stage: str,
    stage_brief: StageBrief | None,
    memory_pack: MemoryPack,
) -> MemoryInjectionBundle:
    """Build the only memory-derived text block allowed into a stage prompt."""
    pack = memory_pack.deduplicated()
    enriched = (
        enrich_stage_brief_with_memory(stage_brief, pack)
        if stage_brief is not None
        else None
    )
    brief_text = enriched.to_injection_text() if enriched is not None else ""
    memory_text = _format_skill_procedures(pack)
    placement_text = pack.placement_reference.strip()
    final_text = "\n\n".join(
        part for part in (brief_text, memory_text, placement_text) if part
    )
    selected_ids = list(
        dict.fromkeys(
            [
                *pack.success_case_ids,
                *pack.failure_case_ids,
                *pack.skill_names,
            ]
        )
    )
    retrieved_skill_names = list(pack.skill_names)
    recommended = {
        _normalized_skill_name(value)
        for value in (stage_brief.recommended_skills if stage_brief is not None else [])
        if _normalized_skill_name(value)
    }
    planner_selected_skill_names = [
        name
        for name in retrieved_skill_names
        if _normalized_skill_name(name) in recommended
    ]
    prompt_delivered_skill_names = [
        name
        for name in retrieved_skill_names
        if f"[skill: {name.casefold()}]" in memory_text.casefold()
    ]
    return MemoryInjectionBundle(
        stage=stage,
        planner_stage_brief=stage_brief,
        enriched_stage_brief=enriched,
        brief_text=brief_text,
        memory_text=memory_text,
        placement_text=placement_text,
        final_text=final_text,
        selected_memory_ids=selected_ids,
        retrieved_skill_names=retrieved_skill_names,
        planner_selected_skill_names=planner_selected_skill_names,
        prompt_delivered_skill_names=prompt_delivered_skill_names,
    )
