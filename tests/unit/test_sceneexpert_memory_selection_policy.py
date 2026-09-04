from __future__ import annotations

from pathlib import Path

from scenesmith.scene_expert.memory.schemas import (
    FailureCase,
    Skill,
    SpatialRelationMemory,
    SuccessCase,
)
from scenesmith.scene_expert.memory.selection_policy import (
    BudgetedMemoryRetriever,
    MemoryInjectionPolicy,
)
from scenesmith.scene_expert.memory.store import FastMemoryStore
from scenesmith.scene_expert.schemas import (
    MemoryPack,
    RetrievedMemorySelection,
    SceneTaskSpec,
    StageRelationContext,
)


class _Delegate:
    def __init__(self, pack: MemoryPack) -> None:
        self.pack = pack

    def retrieve(self, *_args, **_kwargs) -> MemoryPack:
        return self.pack


def _selection(memory_id: str, memory_type: str, rank: int, text: str):
    return RetrievedMemorySelection(
        memory_id=memory_id,
        memory_type=memory_type,
        rank=rank,
        score=1.0 / rank,
        injected_text=text,
    )


def test_policy_bounds_each_bank_and_rejects_unverified_failure(
    tmp_path: Path,
) -> None:
    store = FastMemoryStore(str(tmp_path / "memory"))
    store.add_success_case(
        SuccessCase(
            case_id="success_1",
            room_type="classroom",
            stage="furniture",
            required_objects=["student desk"],
            positive_guidance=["Anchor desks toward the teaching wall."],
        )
    )
    store.add_success_case(
        SuccessCase(
            case_id="success_2",
            room_type="classroom",
            stage="furniture",
            required_objects=["student desk"],
            positive_guidance=["Keep aisles clear."],
        )
    )
    store.add_failure_case(
        FailureCase(
            failure_id="failure_unverified",
            room_type="classroom",
            stage="furniture",
            object="student desk",
            bad_pattern="Desks block the aisle.",
            repair_verified=False,
            is_deterministic=False,
        )
    )
    store.add_failure_case(
        FailureCase(
            failure_id="failure_verified",
            room_type="classroom",
            stage="furniture",
            object="student desk",
            bad_pattern="Desks face away from the teaching wall.",
            repair_verified=True,
            spatial_relations=[
                SpatialRelationMemory(
                    relation_type="faces",
                    subject_role="student desk",
                    target_role="teaching wall",
                    evidence_source="critic",
                )
            ],
        )
    )
    store.add_skill(
        Skill(
            skill_name="align_classroom_desks",
            room_type="classroom",
            stage="furniture",
            required_objects=["student desk"],
            procedure=["Place the teaching anchor first.", "Align desks toward it."],
        )
    )
    pack = MemoryPack(
        success_hints=["success one", "success two"],
        failure_hints=["unverified failure", "verified failure"],
        skill_texts=["skill procedure"],
        placement_reference="grounded layout",
        success_case_ids=["success_1", "success_2"],
        failure_case_ids=["failure_unverified", "failure_verified"],
        skill_names=["align_classroom_desks"],
        selections=[
            _selection("success_1", "success", 1, "success one"),
            _selection("success_2", "success", 2, "success two"),
            _selection("failure_unverified", "failure", 1, "unverified failure"),
            _selection("failure_verified", "failure", 2, "verified failure"),
            _selection("align_classroom_desks", "skill", 1, "skill procedure"),
        ],
    )
    retriever = BudgetedMemoryRetriever(
        _Delegate(pack),
        store=store,
        policy=MemoryInjectionPolicy(),
    )

    result = retriever.retrieve(
        SceneTaskSpec(
            room_type="classroom",
            style="modern",
            required_large_objects=["student desk"],
        ),
        "furniture",
        relation_context=StageRelationContext(
            stage="furniture",
            hard_constraints=[{"constraint_id": "c1", "relation_type": "faces"}],
        ),
    )

    assert result.success_case_ids == ["success_1"]
    assert result.failure_case_ids == ["failure_verified"]
    assert result.skill_names == ["align_classroom_desks"]
    assert len(result.selections) == 3
    rejected = {
        decision.memory_id: decision.reasons
        for decision in result.selection_decisions
        if decision.decision == "rejected"
    }
    assert "type_budget_pruned" in rejected["success_2"]
    assert "unverified_failure" in rejected["failure_unverified"]
    assert result.selection_policy["selected_records"] == 3


def test_policy_requires_failure_object_or_relation_grounding(tmp_path: Path) -> None:
    store = FastMemoryStore(str(tmp_path / "memory"))
    store.add_failure_case(
        FailureCase(
            failure_id="bed_failure",
            room_type="bedroom",
            stage="furniture",
            object="bed",
            bad_pattern="Bed blocks the door.",
            repair_verified=True,
        )
    )
    pack = MemoryPack(
        failure_hints=["avoid bed blocking"],
        failure_case_ids=["bed_failure"],
        selections=[_selection("bed_failure", "failure", 1, "avoid bed blocking")],
    )
    result = BudgetedMemoryRetriever(
        _Delegate(pack), store=store, policy=MemoryInjectionPolicy()
    ).retrieve(
        SceneTaskSpec(
            room_type="classroom",
            style="modern",
            required_large_objects=["student desk"],
        ),
        "furniture",
    )

    assert result.failure_case_ids == []
    assert result.selection_decisions[0].reasons == ["object_mismatch"]
