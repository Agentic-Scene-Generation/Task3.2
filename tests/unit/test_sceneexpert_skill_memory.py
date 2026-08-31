"""Regression coverage for safe skill transfer and verified self-updates."""

from __future__ import annotations

from tempfile import TemporaryDirectory

from scenesmith.scene_expert.memory.activity import MemoryActivityLogger
from scenesmith.scene_expert.memory.hybrid_retriever import HybridMemoryRetriever
from scenesmith.scene_expert.memory.injection import build_memory_injection_bundle
from scenesmith.scene_expert.memory.retriever import MemoryRetriever
from scenesmith.scene_expert.memory.room_taxonomy import room_types_compatible
from scenesmith.scene_expert.memory.schemas import (
    MemoryUtilityObservation,
    Skill,
    SkillApplicability,
    SpatialRelationMemory,
)
from scenesmith.scene_expert.memory.skill_policy import evaluate_skill_for_task
from scenesmith.scene_expert.memory.store import FastMemoryStore
from scenesmith.scene_expert.schemas import (
    MemoryPack,
    RetrievedMemorySelection,
    SceneTaskSpec,
    SkillSelectionDecision,
    StageBrief,
    StageExecutionEvidence,
    StageRelationContext,
    StageVerifyReport,
    VerifyIssue,
)


def _meeting_task() -> SceneTaskSpec:
    return SceneTaskSpec(
        room_type="meeting room",
        style="modern",
        required_large_objects=["conference table", "office chairs"],
    )


def _edge_context(*, chair_count: int = 7) -> StageRelationContext:
    return StageRelationContext(
        stage="furniture",
        hard_constraints=[
            {
                "constraint_id": "meeting-seat-topology",
                "stage": "furniture",
                "relation": "edge_distribution",
                "subjects": {"category": "office_chair", "count": chair_count},
                "targets": {"category": "conference_table", "count": 1},
                "edge_frame": "target_local_rectangle",
                "orientation": "toward_target",
                "groups": [
                    {"edge_class": "long", "counts_per_edge": [3, 3]},
                    {"edge_class": "short", "counts_per_edge": [1, 0]},
                ],
            }
        ],
    )


def _edge_skill(
    *,
    name: str = "distribute_table_seating",
    room_type: str = "meeting_room",
    chair_count: int = 7,
    source_task_id: str = "source_task",
    source_run_id: str = "source_run",
) -> Skill:
    return Skill(
        skill_name=name,
        stage="furniture",
        room_type=room_type,
        room_types=[room_type],
        required_objects=["chair", "table"],
        preconditions=["Bind the seating group to one rectangular table."],
        procedure=[
            "Reserve the required edge slots before placement.",
            "Place every chair in its assigned slot facing the table.",
        ],
        failure_avoidance=["Do not replace the explicit edge counts with symmetry."],
        postconditions=["Every required edge slot is occupied exactly once."],
        applicability=SkillApplicability(
            room_types=[room_type],
            required_object_roles=["chair", "table"],
            required_relation_types=["edge_distribution"],
        ),
        spatial_relations=[
            SpatialRelationMemory(
                relation_type="edge_distribution",
                subject_role="chair",
                target_role="table",
                cardinality={
                    "subject_count": chair_count,
                    "target_count": 1,
                },
                evidence_source="task_contract",
                evidence_ref="source-edge-contract",
            )
        ],
        source_task_id=source_task_id,
        source_run_id=source_run_id,
        source_task_ids=[source_task_id],
        source_run_ids=[source_run_id],
        quality_score=0.8,
        confidence=0.8,
    )


def test_meeting_and_dining_rooms_are_never_token_compatible() -> None:
    assert room_types_compatible("conference room", "meeting_room")
    assert room_types_compatible("dining area", "dining_room")
    assert room_types_compatible("会议室", "meeting_room")
    assert not room_types_compatible("餐厅", "会议室")
    assert not room_types_compatible("dining_room", "meeting_room")
    assert not room_types_compatible("meeting_room", "dining_room")


def test_dining_edge_skill_is_rejected_for_meeting_room_before_ranking() -> None:
    with TemporaryDirectory() as temp_dir:
        store = FastMemoryStore(temp_dir)
        assert store.add_skill(
            _edge_skill(
                name="dining_four_side_distribution",
                room_type="dining_room",
                chair_count=4,
            )
        )
        retriever = MemoryRetriever(store, max_skills=2)

        pack = retriever.retrieve(
            _meeting_task(),
            "furniture",
            relation_context=_edge_context(chair_count=7),
        )

    assert pack.skill_names == []
    decision = next(
        item
        for item in pack.skill_filter_decisions
        if item.skill_name == "dining_four_side_distribution"
    )
    assert decision.decision == "rejected"
    assert "incompatible_room_type" in decision.reasons


def test_hybrid_filter_uses_the_same_room_and_contract_skill_policy() -> None:
    retriever = object.__new__(HybridMemoryRetriever)
    retriever._object_overlap_threshold = 0.15
    decisions: dict[str, SkillSelectionDecision] = {}
    skill = _edge_skill(
        name="dining_four_side_distribution",
        room_type="dining_room",
        chair_count=4,
    )

    accepted = retriever._structured_filter(
        skill,
        _meeting_task(),
        "furniture",
        "skill",
        relation_context=_edge_context(chair_count=7),
        skill_decisions=decisions,
    )

    assert not accepted
    assert decisions[skill.skill_name].decision == "rejected"
    assert "incompatible_room_type" in decisions[skill.skill_name].reasons


def test_generic_edge_skill_conflicting_with_hard_cardinality_is_rejected() -> None:
    skill = _edge_skill(room_type="generic", chair_count=4)

    policy = evaluate_skill_for_task(
        skill,
        _meeting_task(),
        "furniture",
        relation_context=_edge_context(chair_count=7),
    )

    assert not policy.eligible
    assert any(
        reason.startswith("hard_cardinality_conflict:edge_distribution")
        for reason in policy.decision.reasons
    )
    assert policy.decision.conflicting_constraint_ids == ["meeting-seat-topology"]


def test_complete_skill_is_injected_once_and_funnel_is_explicit() -> None:
    skill = _edge_skill()
    skill_text = skill.to_procedure_text()
    pack = MemoryPack(
        skill_names=[skill.skill_name],
        skill_texts=[skill_text],
        selections=[
            RetrievedMemorySelection(
                memory_id=skill.skill_name,
                memory_type="skill",
                rank=1,
                injected_text=skill_text,
            )
        ],
        skill_filter_decisions=[
            SkillSelectionDecision(
                skill_name=skill.skill_name,
                decision="selected",
                required_object_roles=["chair", "table"],
                required_relation_types=["edge_distribution"],
                matched_constraint_ids=["meeting-seat-topology"],
            )
        ],
    )
    bundle = build_memory_injection_bundle(
        stage="furniture",
        stage_brief=StageBrief(
            stage="furniture",
            stage_objective="Build the meeting layout.",
            recommended_skills=[skill.skill_name],
        ),
        memory_pack=pack,
    )

    assert bundle.final_text.count("[Skill: distribute_table_seating]") == 1
    assert bundle.final_text.count(skill.preconditions[0]) == 1
    assert bundle.final_text.count(skill.procedure[0]) == 1
    assert bundle.final_text.count(skill.failure_avoidance[0]) == 1
    assert bundle.final_text.count(skill.postconditions[0]) == 1
    assert bundle.retrieved_skill_names == [skill.skill_name]
    assert bundle.planner_selected_skill_names == [skill.skill_name]
    assert bundle.prompt_delivered_skill_names == [skill.skill_name]


def test_skill_funnel_labels_relevant_hard_failure_without_causal_claim() -> None:
    skill = _edge_skill()
    skill_text = skill.to_procedure_text()
    pack = MemoryPack(
        skill_names=[skill.skill_name],
        skill_texts=[skill_text],
        selections=[
            RetrievedMemorySelection(
                memory_id=skill.skill_name,
                memory_type="skill",
                rank=1,
                injected_text=skill_text,
            )
        ],
        skill_filter_decisions=[
            SkillSelectionDecision(
                skill_name=skill.skill_name,
                decision="selected",
                required_object_roles=["chair", "table"],
                required_relation_types=["edge_distribution"],
                matched_constraint_ids=["meeting-seat-topology"],
            )
        ],
    )
    bundle = build_memory_injection_bundle(
        stage="furniture",
        stage_brief=StageBrief(
            stage="furniture",
            stage_objective="Build the meeting layout.",
            recommended_skills=[skill.skill_name],
        ),
        memory_pack=pack,
    )
    with TemporaryDirectory() as temp_dir:
        logger = MemoryActivityLogger(
            temp_dir,
            scene_id="scene_003",
            task_spec=_meeting_task(),
            task_id="task_current",
            run_id="run_current",
        )
        logger.record_pre_stage(
            stage="furniture",
            memory_pack=pack,
            relation_context=_edge_context(),
            planner_trace={"status": "ok"},
            injection_bundle=bundle,
            execution_evidence=StageExecutionEvidence(
                designer_prompt_contains_memory=True,
                retrieved_skill_names=bundle.retrieved_skill_names,
                planner_selected_skill_names=bundle.planner_selected_skill_names,
                prompt_delivered_skill_names=bundle.prompt_delivered_skill_names,
            ),
        )
        observations = logger.record_post_stage(
            stage="furniture",
            verify_report=StageVerifyReport(
                stage="furniture",
                pass_stage=False,
                issues=[
                    VerifyIssue(
                        issue_type="edge_distribution",
                        object_name="office_chair",
                        description="office chairs violate the required table edges",
                    )
                ],
                hard_check_report={
                    "hard_valid": False,
                    "failed_checks": ["edge_distribution:office_chair"],
                },
            ),
            repair_actions=[],
            scene_state_path="/run/scene_003/final_furniture",
        )

    assert len(observations) == 1
    observation = observations[0]
    assert observation.retrieved
    assert observation.planner_selected
    assert observation.prompt_delivered
    assert observation.outcome == "negative"
    assert observation.outcome_basis == "relevant_hard_failure_after_skill_delivery"


def test_only_independent_skill_evidence_refines_and_repeated_harm_quarantines() -> (
    None
):
    with TemporaryDirectory() as temp_dir:
        store = FastMemoryStore(temp_dir)
        original = _edge_skill()
        assert store.add_skill(original)

        same_task_candidate = _edge_skill(
            source_task_id="source_task",
            source_run_id="second_run_same_task",
        ).model_copy(
            update={
                "procedure": ["Unsafe same-task rewrite", "Do not accept this."],
                "quality_score": 0.95,
            }
        )
        assert not store.add_skill(same_task_candidate)
        assert store.skills[0].procedure == original.procedure

        independent_candidate = _edge_skill(
            source_task_id="independent_source_task",
            # Different scene/task evidence in one ACP run is independent even
            # though the run identifier is intentionally shared.
            source_run_id="source_run",
        ).model_copy(
            update={
                "procedure": [
                    "Bind slots from the hard edge contract.",
                    "Place and orient each chair against its bound slot.",
                ],
                "quality_score": 0.95,
            }
        )
        assert store.add_skill(independent_candidate)
        assert store.skills[0].procedure == independent_candidate.procedure
        assert store.skills[0].observation_count == 2

        first = MemoryUtilityObservation(
            memory_id=original.skill_name,
            memory_type="skill",
            task_id="utility_task_a",
            run_id="utility_run_a",
            stage="furniture",
            injected=True,
            prompt_delivered=True,
            outcome="negative",
            outcome_basis="relevant_hard_failure_after_skill_delivery",
        )
        duplicate_task = first.model_copy(update={"run_id": "utility_run_a_retry"})
        second = first.model_copy(
            update={"task_id": "utility_task_b", "run_id": "utility_run_a"}
        )
        first_summary = store.record_skill_outcomes([first, duplicate_task])
        second_summary = store.record_skill_outcomes([second])

        assert first_summary["updated"] == 1
        assert first_summary["skipped_non_independent"] == 1
        assert second_summary["quarantined"] == [original.skill_name]
        assert store.skills[0].negative_utility_count == 2
        assert store.skills[0].status == "quarantined"
        assert store.active_skills == []


def test_stage_candidates_require_two_independent_semantic_supports() -> None:
    with TemporaryDirectory() as temp_dir:
        store = FastMemoryStore(temp_dir)
        first = _edge_skill(
            name="distribute_table_seating",
            source_task_id="candidate_task_a",
            source_run_id="candidate_run_a",
        ).model_copy(
            update={
                "status": "candidate",
                "source_scene_passed": False,
                "promotion_scope": "stage",
                "activation_reason": "stage_pass_awaiting_independent_support",
            }
        )
        assert store.add_skill(first)
        assert len(store.skills) == 1
        assert store.skills[0].status == "candidate"
        assert store.active_skills == []
        assert store.manifest["counts"]["candidate_skill"] == 1
        assert store.last_apply_summary["skill_candidate_added"] == 1

        same_task_alias = _edge_skill(
            name="bind_contract_seats_to_edges",
            source_task_id="candidate_task_a",
            source_run_id="candidate_run_retry",
        ).model_copy(update={"status": "candidate"})
        assert not store.add_skill(same_task_alias)
        assert store.skills[0].independent_support_count == 1

        independent_alias = _edge_skill(
            name="bind_contract_seats_to_edges",
            source_task_id="candidate_task_b",
            source_run_id="candidate_run_b",
        ).model_copy(update={"status": "candidate"})
        assert store.add_skill(independent_alias)

        promoted = store.skills[0]
        assert len(store.skills) == 1
        assert promoted.status == "active"
        assert promoted.independent_support_count == 2
        assert promoted.activation_reason == "independent_stage_support_threshold_met"
        assert promoted.skill_aliases == [
            "distribute_table_seating",
            "bind_contract_seats_to_edges",
        ]
        assert store.active_skills == [promoted]
        assert store.last_apply_summary["skill_promoted_active"] == 1
        assert store.manifest["counts"]["candidate_skill"] == 0
        assert store.manifest["counts"]["active_skill"] == 1


def test_same_skill_name_with_different_semantics_never_merges_support() -> None:
    with TemporaryDirectory() as temp_dir:
        store = FastMemoryStore(temp_dir)
        meeting_skill = _edge_skill(
            name="arrange_required_seating",
            room_type="meeting_room",
            chair_count=7,
            source_task_id="meeting_task",
            source_run_id="meeting_run",
        ).model_copy(update={"status": "candidate"})
        dining_skill = _edge_skill(
            name="arrange_required_seating",
            room_type="dining_room",
            chair_count=4,
            source_task_id="dining_task",
            source_run_id="dining_run",
        ).model_copy(update={"status": "candidate"})

        assert store.add_skill(meeting_skill)
        assert store.add_skill(dining_skill)

        assert len(store.skills) == 2
        assert all(skill.status == "candidate" for skill in store.skills)
        assert all(skill.independent_support_count == 1 for skill in store.skills)
        assert store.active_skills == []


def test_verified_success_updates_skill_utility_only_once_per_task() -> None:
    with TemporaryDirectory() as temp_dir:
        store = FastMemoryStore(temp_dir)
        skill = _edge_skill()
        assert store.add_skill(skill)
        observation = MemoryUtilityObservation(
            memory_id=skill.skill_name,
            memory_type="skill",
            task_id="successful_transfer_task",
            run_id="shared_acp_run",
            stage="furniture",
            injected=True,
            prompt_delivered=True,
            stage_passed=True,
            outcome="positive",
            outcome_basis="verified_stage_pass_after_skill_delivery",
        )

        first = store.record_skill_outcomes([observation])
        duplicate = store.record_skill_outcomes(
            [observation.model_copy(update={"run_id": "retry_run"})]
        )

        assert first["updated"] == 1
        assert duplicate["updated"] == 0
        assert duplicate["skipped_non_independent"] == 1
        assert store.skills[0].positive_utility_count == 1
        assert store.skills[0].negative_utility_count == 0
        assert store.skills[0].status == "active"
