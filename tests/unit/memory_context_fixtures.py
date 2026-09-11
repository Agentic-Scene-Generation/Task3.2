"""Explicit synthetic Planner acceptance for existing injection regression tests."""

from scenesmith.scene_expert.schemas import (
    MemoryAdaptation,
    MemoryRoleBinding,
    StageBrief,
)


def accepting_brief(pack, **kwargs):
    """Bind synthetic legacy fixture identities; never a production fallback."""
    canonical = pack.deduplicated()
    pack.selections = [
        row.model_copy(update={"content_hash": row.content_hash or "synthetic-fixture"})
        for row in canonical.selections
    ]
    choices = []
    for row in pack.selections:
        roles = sorted(
            {
                str(relation.get(key) or "")
                for relation in row.spatial_relations
                for key in ("subject_role", "target_role")
            }
            - {""}
        )
        text = row.injected_text
        if row.memory_type == "skill" and text.startswith("[Skill:"):
            text = text.split("\n", 1)[-1]
        choices.append(
            MemoryAdaptation(
                memory_type=row.memory_type,
                memory_id=row.memory_id,
                source_content_hash=row.content_hash,
                decision="accepted",
                reason="Explicit test fixture decision",
                bindings=[
                    MemoryRoleBinding(source_role=role, current_role=role)
                    for role in roles
                ],
                actions=[text or "Apply the fixture's source lesson."],
                checks=["Check current task constraints after the action."],
            )
        )
    return StageBrief(memory_adaptations=choices, **kwargs)
