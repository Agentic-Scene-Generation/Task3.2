"""CPU integration tests for explicit acceptance and native request delivery."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from scenesmith.scene_expert.context_bundle import build_stage_context_bundle
from scenesmith.scene_expert.global_planner import (
    GlobalPlanner,
    _format_memory_for_prompt,
)
from scenesmith.scene_expert.memory.contracts import selection_from_record
from scenesmith.scene_expert.memory.delivery import prepare_memory_delivery
from scenesmith.scene_expert.memory.hybrid_retriever import HybridMemoryRetriever
from scenesmith.scene_expert.memory.injection import (
    MEMORY_START,
    build_memory_injection_bundle,
)
from scenesmith.scene_expert.memory.retriever import MemoryRetriever
from scenesmith.scene_expert.memory.schemas import SpatialRelationMemory, SuccessCase
from scenesmith.scene_expert.memory.selection_policy import (
    BudgetedMemoryRetriever,
    MemoryInjectionPolicy,
)
from scenesmith.scene_expert.memory.state import build_memory_scene_state
from scenesmith.scene_expert.memory.store import FastMemoryStore
from scenesmith.scene_expert.schemas import (
    HarnessContext,
    MemoryAdaptation,
    MemoryPack,
    MemoryRoleBinding,
    SceneTaskSpec,
    StageBrief,
    StageRelationContext,
)
from scenesmith.scene_expert.structured_llm import StructuredLLMResult


def scene():
    rotation = SimpleNamespace(matrix=lambda: np.eye(3))
    obj = SimpleNamespace(
        name="office chair",
        category="chair",
        object_type="furniture",
        immutable=False,
        transform=SimpleNamespace(
            translation=lambda: [1.0, 2.0, 0.0], rotation=lambda: rotation
        ),
        support_surfaces=[],
        compute_world_bounds=lambda: ([0.7, 1.7, 0.0], [1.3, 2.3, 1.0]),
    )
    return SimpleNamespace(
        objects={"chair_1": obj},
        text_description="Design an office",
        room_geometry=SimpleNamespace(width=4.0, length=5.0, height=3.0, openings=[]),
    )


def task():
    return SceneTaskSpec(
        room_type="office", style="modern", required_large_objects=["chair", "table"]
    )


def candidate(tmp_path, *, relation=None, text="RAW_SOURCE_SENTINEL"):
    record = SuccessCase(
        case_id="s1",
        room_type="office",
        stage="furniture",
        successful_pattern=[text],
        spatial_relations=[relation] if relation else [],
    )
    return selection_from_record(record, rank=1, memory_dir=tmp_path)


def choice(source, **overrides):
    values = dict(
        memory_type=source.memory_type,
        memory_id=source.memory_id,
        source_content_hash=source.content_hash,
        decision="adapted",
        reason="Apply to the current chair",
        bindings=[
            MemoryRoleBinding(
                source_role="chair", current_role="chair", object_ids=["chair_1"]
            )
        ],
        preconditions=["Check the current chair forward axis."],
        actions=["ACTION_SENTINEL: orient the chair to its current task anchor."],
        checks=["CHECK_SENTINEL: verify chair orientation using the current intent."],
    )
    return MemoryAdaptation(**(values | overrides))


def bundle_for(tmp_path, obj_scene=None, **choice_overrides):
    obj_scene = obj_scene or scene()
    source = candidate(tmp_path)
    pack = MemoryPack(
        selections=[source], current_scene_state=build_memory_scene_state(obj_scene)
    )
    brief = StageBrief(
        stage="furniture",
        stage_objective="Design the office",
        memory_adaptations=[choice(source, **choice_overrides)],
    )
    return build_memory_injection_bundle(
        stage="furniture", stage_brief=brief, memory_pack=pack, task_spec=task()
    )


def attach(obj_scene, bundle):
    obj_scene.scene_expert_accepted_memory_bundle = bundle.model_dump(mode="json")
    obj_scene.scene_expert_memory_delivery_enabled = True


def deliver(
    obj_scene,
    event="request_initial_design",
    prompt="Initial design",
    role="designer",
    stage="furniture",
):
    return prepare_memory_delivery(
        scene=obj_scene, stage=stage, agent_role=role, event=event, prompt=prompt
    )


def test_rejected_or_unaccepted_raw_memory_never_returns(tmp_path):
    source = candidate(tmp_path)
    for choices in ([], [choice(source, decision="rejected")]):
        pack = MemoryPack(selections=[source], placement_reference="RAW_LAYOUT")
        brief = StageBrief(
            stage="furniture",
            stage_objective="Office",
            recommended_skills=["unknown"],
            memory_adaptations=choices,
        )
        bundle = build_memory_injection_bundle(
            stage="furniture", stage_brief=brief, memory_pack=pack
        )
        assert not bundle.accepted_items and not bundle.memory_text
        assert (
            "RAW_SOURCE_SENTINEL" not in bundle.final_text
            and "RAW_LAYOUT" not in bundle.final_text
        )
        assert "unknown" not in bundle.brief_text


@pytest.mark.parametrize(
    "override,reason",
    [
        ({"source_content_hash": "wrong"}, "source_hash_mismatch"),
        (
            {"bindings": [MemoryRoleBinding(source_role="chair", current_role="sofa")]},
            "incompatible_role_transfer",
        ),
        (
            {
                "bindings": [
                    MemoryRoleBinding(
                        source_role="chair",
                        current_role="chair",
                        object_ids=["made_up"],
                    )
                ]
            },
            "unknown_object_id",
        ),
        ({"checks": []}, "missing_action_or_check"),
    ],
)
def test_invalid_adaptation_abstains(tmp_path, override, reason):
    bundle = bundle_for(tmp_path, **override)
    assert not bundle.accepted_items
    assert reason in bundle.adaptation_decisions[0]["reasons"]


def test_unknown_and_duplicate_planner_ids_are_not_guessed(tmp_path):
    source = candidate(tmp_path)
    brief = StageBrief(
        stage="furniture",
        stage_objective="Office",
        memory_adaptations=[
            choice(source),
            choice(source),
            choice(source, memory_id="invented"),
        ],
    )
    bundle = build_memory_injection_bundle(
        stage="furniture",
        stage_brief=brief,
        memory_pack=MemoryPack(selections=[source]),
    )
    assert not bundle.accepted_items
    assert bundle.adaptation_decisions[0]["reasons"] == ["duplicate_planner_choice"]
    assert bundle.adaptation_decisions[-1]["reasons"] == ["unknown_memory_id"]


def test_full_actions_preserved_and_whole_bundle_budget_applied(tmp_path):
    long_action = "Check the anchor and supporting surface. " * 20 + "TAIL_ACTION"
    bundle = bundle_for(tmp_path, actions=[long_action])
    assert (
        "TAIL_ACTION" in bundle.memory_text and "CHECK_SENTINEL" in bundle.memory_text
    )
    source = bundle.accepted_items[0].source
    limited = build_memory_injection_bundle(
        stage="furniture",
        stage_brief=bundle.planner_stage_brief,
        memory_pack=MemoryPack(
            selections=[source], current_scene_state=bundle.current_scene_state
        ),
        task_spec=task(),
        max_chars=200,
    )
    assert not limited.memory_text
    assert "adapted_bundle_budget_pruned" in limited.adaptation_decisions[0]["reasons"]


@pytest.mark.parametrize(
    "field,value", [("orientation", "outward"), ("count", 2), ("edge_frame", "world")]
)
def test_success_spatial_conflict_rejected_before_and_after_planner(
    tmp_path, field, value
):
    relation = SpatialRelationMemory(
        relation_type="edge_distribution",
        subject_role="chair",
        target_role="table",
        cardinality={"orientation": "inward", "count": 4, "edge_frame": "table_local"},
    )
    source = candidate(tmp_path, relation=relation)
    context = StageRelationContext(
        stage="furniture",
        hard_constraints=[
            {
                "constraint_id": "layout",
                "relation": "edge_distribution",
                "subjects": {"category": "chair"},
                "targets": {"category": "table"},
                "orientation": "inward",
                "count": 4,
                "edge_frame": "table_local",
            }
            | {field: value}
        ],
    )
    pack = MemoryPack(selections=[source])
    brief = StageBrief(
        stage="furniture", stage_objective="Office", memory_adaptations=[choice(source)]
    )
    bundle = build_memory_injection_bundle(
        stage="furniture",
        stage_brief=brief,
        memory_pack=pack,
        task_spec=task(),
        relation_context=context,
    )
    assert not bundle.accepted_items
    assert any(
        "hard_cardinality_conflict" in reason
        for reason in bundle.adaptation_decisions[0]["reasons"]
    )
    store = FastMemoryStore(str(tmp_path / "bank"))
    store.add_success_case(
        SuccessCase(
            case_id="s1",
            room_type="office",
            stage="furniture",
            successful_pattern=["Use the reference"],
            spatial_relations=[relation],
        )
    )
    result = BudgetedMemoryRetriever(
        SimpleNamespace(retrieve=lambda *a, **kw: pack),
        store=store,
        policy=MemoryInjectionPolicy(),
    ).retrieve(task(), "furniture", relation_context=context)
    assert not result.success_case_ids


def test_observed_optional_objects_enable_retrieval_without_changing_required(tmp_path):
    empty_task = SceneTaskSpec(room_type="office", style="modern")
    original = empty_task.model_dump()
    store = FastMemoryStore(str(tmp_path / "bank"))
    store.add_success_case(
        SuccessCase(
            case_id="optional-chair",
            room_type="office",
            stage="furniture",
            task_signature=["chair"],
            successful_pattern=["Use actual chair orientation"],
        )
    )
    retriever = MemoryRetriever(store)
    assert not retriever.retrieve(empty_task, "furniture").success_case_ids
    state = build_memory_scene_state(scene())
    pack = retriever.retrieve(empty_task, "furniture", scene_state=state)
    assert pack.success_case_ids == ["optional-chair"]
    assert pack.current_scene_state == state
    hybrid = HybridMemoryRetriever(
        store,
        str(store.memory_dir),
        SimpleNamespace(
            encode=lambda texts: np.asarray(
                [[1.0, 0.0] for _ in texts], dtype=np.float32
            )
        ),
        auto_build_indexes=True,
    )
    assert hybrid.retrieve(
        empty_task, "furniture", scene_state=state
    ).success_case_ids == ["optional-chair"]
    assert empty_task.model_dump() == original


def test_scene_snapshot_includes_actual_geometry_and_explicit_supports():
    obj_scene = scene()
    surface = SimpleNamespace(
        surface_id="seat",
        bounding_box_min=[0.0, 0.0, 0.0],
        bounding_box_max=[0.4, 0.4, 0.0],
        transform=SimpleNamespace(translation=lambda: [1.0, 2.0, 0.5]),
    )
    obj_scene.objects["chair_1"].support_surfaces = [surface]
    state = build_memory_scene_state(obj_scene)
    assert state["room"]["width_m"] == 4
    assert state["objects"][0]["yaw_deg"] == 0
    assert state["objects"][0]["support_surfaces"][0]["world_translation"] == [
        1.0,
        2.0,
        0.5,
    ]
    assert build_memory_scene_state(obj_scene) == state
    assert build_memory_scene_state(obj_scene, max_objects=0)["omitted_objects"] == 1


def test_initial_and_repair_delivery_is_relevant_nonduplicated_and_read_only(tmp_path):
    obj_scene = scene()
    bundle = bundle_for(tmp_path, obj_scene)
    attach(obj_scene, bundle)
    before = deepcopy(obj_scene.scene_expert_accepted_memory_bundle)
    initial = deliver(obj_scene)
    assert initial["text"].count("ACTION_SENTINEL") == 1
    assert initial["application_observed"] is None
    duplicate = deliver(obj_scene, prompt="Initial design\n" + initial["text"])
    assert (
        not duplicate["text"]
        and duplicate["decisions"][0]["decision"] == "already_in_request"
    )
    native_duplicate = deliver(
        obj_scene,
        prompt="\n".join(
            bundle.accepted_items[0].adaptation.preconditions
            + bundle.accepted_items[0].adaptation.actions
            + bundle.accepted_items[0].adaptation.checks
        ),
    )
    assert not native_duplicate["text"]
    repair = deliver(
        obj_scene, "request_design_change", "Critic: chair_1 faces the wrong direction"
    )
    assert repair["text"].count("ACTION_SENTINEL") == 1
    unrelated = deliver(
        obj_scene, "request_design_change", "Critic: change the wall lamp color"
    )
    assert not unrelated["text"]
    assert "repair_issue_mismatch" in unrelated["decisions"][0]["reasons"]
    assert obj_scene.scene_expert_accepted_memory_bundle == before
    assert obj_scene.text_description == "Design an office"


@pytest.mark.parametrize(
    "mutation,reason",
    [
        ("remove", "bound_object_missing"),
        ("move", "bound_object_state_changed"),
        ("resize", "room_geometry_changed"),
    ],
)
def test_removed_moved_or_rollback_geometry_invalidates_bound_advice(
    tmp_path, mutation, reason
):
    obj_scene = scene()
    attach(obj_scene, bundle_for(tmp_path, obj_scene))
    if mutation == "remove":
        obj_scene.objects.clear()
    elif mutation == "move":
        obj_scene.objects["chair_1"].transform.translation = lambda: [3.0, 2.0, 0.0]
    else:
        obj_scene.room_geometry.width = 6.0
    delivery = deliver(obj_scene, "request_design_change", "Fix chair_1 orientation")
    assert not delivery["text"]
    assert reason in delivery["decisions"][0]["reasons"]


def test_critic_off_flag_and_wrong_stage_cannot_receive_memory(tmp_path, monkeypatch):
    obj_scene = scene()
    assert deliver(obj_scene) == {}
    attach(obj_scene, bundle_for(tmp_path, obj_scene))
    assert deliver(obj_scene, role="critic", event="request_critique") == {}
    assert not deliver(obj_scene, stage="wall_mounted")["text"]
    monkeypatch.setenv("SCENEEXPERT_INJECT_STAGE_CONTEXT_BUNDLE", "0")
    assert deliver(obj_scene)["status"] == "context_injection_disabled"
    monkeypatch.delenv("SCENEEXPERT_INJECT_STAGE_CONTEXT_BUNDLE")
    obj_scene.scene_expert_memory_delivery_enabled = False
    assert deliver(obj_scene) == {}


def test_existing_context_builder_delivers_full_advice_but_never_to_critic(tmp_path):
    obj_scene = scene()
    bundle = bundle_for(tmp_path, obj_scene, actions=["X" * 800 + "ACTION_TAIL"])
    attach(obj_scene, bundle)
    designer = build_stage_context_bundle(
        stage="furniture",
        agent_role="designer",
        event="request_initial_design",
        scene=obj_scene,
        prompt="Design",
        forbidden_zones=[],
    )
    assert "ACTION_TAIL" in designer.to_llm_text(max_chars=300)
    assert "CHECK_SENTINEL" in designer.to_llm_text(max_chars=300)
    critic = build_stage_context_bundle(
        stage="furniture",
        agent_role="critic",
        event="request_critique",
        scene=obj_scene,
        prompt="Review",
        forbidden_zones=[],
    )
    assert not critic.memory_delivery
    assert (
        "ACTION_TAIL" not in critic.to_llm_text()
        and "RAW_SOURCE_SENTINEL" not in critic.to_llm_text()
    )
    designer.save(tmp_path / "context.json")
    assert "ACTION_TAIL" in (tmp_path / "context.json").read_text(encoding="utf-8")


def test_existing_planner_call_receives_identity_and_preserves_explicit_decisions(
    tmp_path,
):
    source = candidate(tmp_path)
    source.placement_text = "Verified spatial reference with frame table_local"
    pack = MemoryPack(selections=[source])
    text = _format_memory_for_prompt(pack)
    assert source.content_hash in text and "table_local" in text and "s1" in text
    desired = StageBrief(
        stage="furniture",
        stage_objective="Office",
        memory_adaptations=[choice(source, decision="rejected")],
    )
    calls = []

    class Client:
        def complete(self, **kwargs):
            calls.append(kwargs)
            return StructuredLLMResult(value=desired)

    planner = GlobalPlanner(model="unused", llm_client=Client())
    actual = planner.generate_stage_brief(
        HarnessContext(stage="furniture", task_spec=task(), memory_pack=pack),
        scene_state_summary="chair_1 is already present",
        original_task="Design an office",
    )
    assert len(calls) == 1
    assert actual.memory_adaptations == desired.memory_adaptations
    assert "chair_1 is already present" in calls[0]["messages"][1]["content"]
    assert not build_memory_injection_bundle(
        stage="furniture", stage_brief=actual, memory_pack=pack
    ).memory_text


def test_planner_failure_does_not_auto_accept_memory(tmp_path):
    source = candidate(tmp_path)

    class Client:
        def complete(self, **kwargs):
            return StructuredLLMResult(final_error="offline")

    planner = GlobalPlanner(model="unused", llm_client=Client())
    pack = MemoryPack(selections=[source])
    brief = planner.generate_stage_brief(
        HarnessContext(stage="furniture", task_spec=task(), memory_pack=pack)
    )
    assert not brief.memory_adaptations
    assert not build_memory_injection_bundle(
        stage="furniture", stage_brief=brief, memory_pack=pack
    ).memory_text


def test_known_rejected_text_cannot_leak_via_generic_brief_fields(tmp_path):
    raw = "Do not put a chair anywhere near the table in this old example."
    source = candidate(tmp_path, text=raw)
    brief = StageBrief(
        stage="furniture",
        stage_objective="Design an office",
        constraints_for_designer=[raw, "Respect the current task."],
        failure_patterns_to_avoid=[raw],
        memory_adaptations=[choice(source, decision="rejected", actions=[raw])],
    )
    bundle = build_memory_injection_bundle(
        stage="furniture",
        stage_brief=brief,
        memory_pack=MemoryPack(selections=[source]),
    )
    assert raw not in bundle.final_text
    assert "Respect the current task." in bundle.final_text


def test_conflicting_accepted_sources_cannot_share_a_bundle(tmp_path):
    a = candidate(
        tmp_path,
        relation=SpatialRelationMemory(
            relation_type="faces", subject_role="chair", target_role="table"
        ),
    )
    b = candidate(
        tmp_path,
        relation=SpatialRelationMemory(
            relation_type="faces_away_from", subject_role="chair", target_role="table"
        ),
    )
    b.memory_id = "s2"
    bindings = [
        MemoryRoleBinding(source_role=role, current_role=role)
        for role in ("chair", "table")
    ]
    choices = [choice(a, bindings=bindings), choice(b, bindings=bindings)]
    brief = StageBrief(
        stage="furniture", stage_objective="Office", memory_adaptations=choices
    )
    bundle = build_memory_injection_bundle(
        stage="furniture",
        stage_brief=brief,
        memory_pack=MemoryPack(selections=[a, b]),
        task_spec=task(),
    )
    assert len(bundle.accepted_items) == 1
    assert "accepted_memory_conflict" in bundle.adaptation_decisions[1]["reasons"]


def test_memory_observation_and_delivery_errors_do_not_remove_native_context(tmp_path):
    broken = SimpleNamespace(objects=object())
    assert build_memory_scene_state(broken)["observation_error"]
    obj_scene = scene()
    obj_scene.scene_expert_memory_delivery_enabled = True
    obj_scene.scene_expert_accepted_memory_bundle = {"invalid": "data"}
    bundle = build_stage_context_bundle(
        stage="furniture",
        agent_role="designer",
        event="request_design_change",
        scene=obj_scene,
        forbidden_zones=[],
        prompt="Fix the current chair",
    )
    assert bundle.memory_delivery["status"] == "memory_delivery_error"
    assert "Current objects:" in bundle.to_llm_text()
    assert MEMORY_START not in bundle.to_llm_text()


def test_semantic_embeddings_do_not_encode_source_task_or_run_ids():
    from scenesmith.scene_expert.memory.text_builder import build_embedding_text

    record = SuccessCase(
        case_id="s",
        room_type="office",
        stage="furniture",
        successful_pattern=["Check the anchor direction"],
        source_task_id="secret-task-a",
        source_run_id="run-a",
    )
    other = record.model_copy(
        update={"source_task_id": "other-task-b", "source_run_id": "run-b"}
    )
    assert build_embedding_text(record) == build_embedding_text(other)
    assert "secret-task-a" not in build_embedding_text(record)


def test_memory_off_preserves_native_context_text():
    from scenesmith.scene_expert.context_bundle import StageContextBundle

    native = StageContextBundle(
        stage="furniture",
        agent_role="designer",
        scene_summary="Current room",
        history_summary="Native working memory",
    )
    assert (
        native.to_llm_text()
        == native.model_copy(update={"memory_delivery": {}}).to_llm_text()
    )
    assert MEMORY_START not in native.to_llm_text()


def test_numbered_runtime_object_name_binds_without_inventing_category(tmp_path):
    obj_scene = scene()
    obj_scene.objects["chair_1"].name = "office_chair_1"
    obj_scene.objects["chair_1"].category = ""
    bundle = bundle_for(tmp_path, obj_scene)
    assert len(bundle.accepted_items) == 1
    assert bundle.current_scene_state["objects"][0]["category"] == ""


def test_conflicting_legacy_text_for_same_identity_is_ambiguous():
    pack = MemoryPack(
        success_case_ids=["s", "s"], success_hints=["Face inward.", "Face outward."]
    ).deduplicated()
    assert pack.success_case_ids == ["s"]
    assert pack.success_hints == [""]


def test_unrelated_memory_marker_does_not_suppress_unseen_advice(tmp_path):
    obj_scene = scene()
    bundle = bundle_for(tmp_path, obj_scene)
    attach(obj_scene, bundle)
    delivery = deliver(
        obj_scene,
        prompt=MEMORY_START + "\nUnrelated cached advice from native working memory.",
    )
    assert delivery["text"].count("ACTION_SENTINEL") == 1
    assert delivery["decisions"][0]["decision"] == "delivered"
