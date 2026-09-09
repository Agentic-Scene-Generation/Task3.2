"""CPU-only, end-to-end regressions for evidence-bound spatial memory."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from scenesmith.scene_expert.memory.adaptation import validate_adaptation
from scenesmith.scene_expert.memory.advisory import observe_advice
from scenesmith.scene_expert.memory.contracts import selection_from_record
from scenesmith.scene_expert.memory.evidence import (
    constraint_observation,
    writer_prompt_evidence,
)
from scenesmith.scene_expert.memory.placement import (
    collect_placement_episodes,
    pair_measurements,
    valid_episode,
    valid_state,
)
from scenesmith.scene_expert.memory.retriever import MemoryRetriever
from scenesmith.scene_expert.memory.schemas import (
    FailureMemoryCandidate,
    PlacementEpisode,
    SuccessCase,
    SuccessMemoryCandidate,
)
from scenesmith.scene_expert.memory.state import state_hash
from scenesmith.scene_expert.memory.store import FastMemoryStore
from scenesmith.scene_expert.memory.writer import MemoryWriter
from scenesmith.scene_expert.schemas import (
    FullVerifyReport,
    MemoryAdaptation,
    SceneTaskSpec,
    StageRelationContext,
)
from tests.unit.test_memory_evaluation import fixture, write
from tests.unit.test_memory_writer_resilience import _evidence


def state(*objects: dict) -> dict:
    result = {"objects": list(objects), "room": {}}
    return {**result, "fingerprint": state_hash(result)}


def object_row(oid: str, name: str, x: float, yaw: float = 0) -> dict:
    return {
        "object_id": oid,
        "name": name,
        "translation": [x, 0, 0],
        "yaw_deg": yaw,
        "bbox_min": [x - 0.25, -0.25, 0],
        "bbox_max": [x + 0.25, 0.25, 1],
    }


def captured(tmp_path: Path, *, stage: str = "furniture") -> tuple[dict, dict]:
    activity, payload, path = fixture(tmp_path)
    entry = activity["stages"]["furniture"]
    entry["prepared_at_epoch"] = 1788739200  # 2026-09-07T00:00:00Z
    entry["finished_at_epoch"] = 1788739320
    entry["post_scene_state"] = state(
        object_row("chair_1", "chair", 1, 90), object_row("table_1", "table", 0)
    )
    entry["scene_state_path"] = str(tmp_path / "final_furniture")
    payload["context_snapshot"]["decision_state"] = state(
        object_row("chair_1", "chair", 2), object_row("table_1", "table", 0)
    )
    if stage != "furniture":
        payload["stage"] = stage
        entry["verify_report"]["stage"] = stage
    write(path, payload)
    write(
        tmp_path / "scene_expert/timing/llm_calls.jsonl",
        {
            "stage": stage,
            "agent_role": "designer",
            "event": "request_design_change",
            "created_at": "2026-09-07T00:01:00Z",
            "payload_ref": str(path.relative_to(tmp_path)),
        },
    )
    catalog = collect_placement_episodes(tmp_path, stage, entry)
    return catalog, entry


def bound_record(tmp_path: Path) -> tuple[SuccessCase, dict]:
    catalog, entry = captured(tmp_path)
    episode = catalog["episodes"][0]
    writer = MemoryWriter(model="unused", llm_client=SimpleNamespace())
    candidate = SuccessMemoryCandidate(
        stage="furniture",
        successful_pattern=["Adjust the chair relative to table edges."],
        episode_ids=[episode["episode_id"]],
        procedure=[
            "Inspect the table anchor frame and chair footprint.",
            "Align the chair after checking the table edge and available spacing.",
        ],
        applicability=["A chair is being placed beside a table."],
    )
    evidence = _evidence()
    evidence["placement_experience_catalog"] = catalog
    context = writer._canonical_context(evidence, "")
    content = writer._success_content(
        candidate, context, FullVerifyReport(pass_scene=True, overall_score=0.9)
    )
    assert writer._bind_placement(content, candidate, evidence, failure=False)
    return SuccessCase.model_validate(content), entry


def adapted(tmp_path: Path) -> tuple[dict, dict]:
    record, entry = bound_record(tmp_path)
    source = selection_from_record(record, rank=1, memory_dir=tmp_path)
    episode = record.placement_experience.episodes[0]
    choice = MemoryAdaptation(
        memory_type="success",
        memory_id=record.case_id,
        source_content_hash=source.content_hash,
        decision="adapted",
        source_relation_indices=[0],
        bindings=[
            {
                "source_role": "chair",
                "current_role": "chair",
                "object_ids": ["chair_1"],
            },
            {
                "source_role": "table",
                "current_role": "table",
                "object_ids": ["table_1"],
            },
        ],
        actions=["Check the table frame before aligning the chair."],
        checks=["Measure relative transform yaw."],
        advice_checks=[
            {
                "source_episode_id": episode.episode_id,
                "metric": "relative_yaw_deg",
                "subject_role": "chair",
                "anchor_role": "table",
            }
        ],
    )
    return {
        "source": source.model_dump(mode="json"),
        "adaptation": choice.model_dump(mode="json"),
    }, entry


def test_observation_frame_is_not_world_or_semantic_front() -> None:
    subject, anchor = object_row("a", "chair", 2, 180), object_row("b", "table", 0, 90)
    metrics = pair_measurements(subject, anchor)
    assert metrics["anchor_local_offset_m"] == [0, -2, 0]
    assert metrics["relative_yaw_deg"] == 90
    assert metrics["aabb_separation_m"] == 1.5
    assert "clearance_m" not in metrics
    subject["translation"][0] = float("nan")
    assert "anchor_local_offset_m" not in pair_measurements(subject, anchor)


@pytest.mark.parametrize(
    "stage", ["furniture", "wall_mounted", "ceiling_mounted", "manipuland"]
)
def test_collects_all_stages_with_action_and_state_provenance(tmp_path, stage) -> None:
    catalog, _ = captured(tmp_path, stage=stage)
    assert len(catalog["episodes"]) == 1
    episode = PlacementEpisode.model_validate(catalog["episodes"][0])
    assert valid_episode(episode)
    assert episode.actions[0]["tool_call_id"] == "call-1"
    assert episode.before_measurements["relative_yaw_deg"] == 0
    assert episode.measurements["relative_yaw_deg"] == 90
    assert episode.stage == stage
    assert episode.repair_verified is False
    episode.measurements["relative_yaw_deg"] = 180
    assert not valid_episode(episode)


@pytest.mark.parametrize(
    "fault", ["wrong_window", "wrong_stage", "failed_call", "changed_state"]
)
def test_capture_does_not_join_unrelated_or_failed_evidence(tmp_path, fault) -> None:
    _, entry = captured(tmp_path)
    path = tmp_path / "scene_expert/audit/llm_payloads/designer.json"
    if fault == "wrong_window":
        entry["prepared_at_epoch"] = entry["finished_at_epoch"] + 1000
    elif fault == "changed_state":
        entry["post_scene_state"]["objects"][0]["yaw_deg"] = 123
    else:
        import json

        payload = json.loads(path.read_text())
        payload["stage" if fault == "wrong_stage" else "error"] = "wrong"
        write(path, payload)
    assert collect_placement_episodes(tmp_path, "furniture", entry)["episodes"] == []


def test_writer_retains_exact_episode_not_whole_stage_contract(tmp_path) -> None:
    record, _ = bound_record(tmp_path)
    assert record.required_objects == ["bed", "nightstand"]  # Source metadata retained.
    assert record.spatial_relations[0].subject_role == "chair"
    assert record.spatial_relations[0].relation_type == "observed_relative_pose"
    assert not record.spatial_relations[0].has_verified_geometry
    selection = selection_from_record(record, rank=1, memory_dir=tmp_path)
    assert "Inspect the table anchor frame" in selection.injected_text
    assert "chair_1" not in selection.injected_text
    assert "bed" not in selection.injected_text


@pytest.mark.parametrize(
    "fault", ["missing_id", "wrong_stage", "no_procedure", "inventory_only", "fake_id"]
)
def test_writer_rejects_ungrounded_or_redundant_new_candidates(tmp_path, fault) -> None:
    record, _ = bound_record(tmp_path)
    episode = record.placement_experience.episodes[0]
    writer = MemoryWriter(model="unused", llm_client=SimpleNamespace())
    candidate = SuccessMemoryCandidate(
        stage="furniture",
        successful_pattern=["A rule"],
        episode_ids=[episode.episode_id],
        procedure=["Inspect anchor frame.", "Align relative to table edge."],
        applicability=["Chair/table."],
    )
    if fault == "missing_id":
        candidate.episode_ids = []
    if fault == "fake_id":
        candidate.episode_ids = ["invented"]
    if fault == "wrong_stage":
        candidate.stage = "wall_mounted"
    if fault == "no_procedure":
        candidate.procedure = []
    if fault == "inventory_only":
        candidate.procedure = [
            "Place all required objects.",
            "Check the required count.",
        ]
    assert not writer._bind_placement(
        {},
        candidate,
        {"placement_experience_catalog": {"episodes": [episode.model_dump()]}},
        failure=False,
    )


def test_failure_does_not_inherit_stagewide_repair_claim(tmp_path) -> None:
    record, _ = bound_record(tmp_path)
    episode = record.placement_experience.episodes[0]
    candidate = FailureMemoryCandidate(
        stage="furniture",
        failure_type="collision",
        bad_pattern="Blocked chair",
        failure_reason="collision",
        repair_action="Move chair",
        repair_verified=True,
        episode_ids=[episode.episode_id],
        procedure=["Inspect anchor frame.", "Move away from table edge."],
        applicability=["Chair/table"],
    )
    writer = MemoryWriter(model="unused", llm_client=SimpleNamespace())
    assert not writer._bind_placement(
        {},
        candidate,
        {"placement_experience_catalog": {"episodes": [episode.model_dump()]}},
        failure=True,
    )


def test_advice_measures_yaw_without_borrowing_inventory_pass(tmp_path) -> None:
    item, entry = adapted(tmp_path)
    result = observe_advice(item, "furniture", entry)
    assert result[0]["status"] == "measured"
    assert result[0]["after_value"] == 90
    assert result[0]["quality_gain"] is None
    assert result[0]["causal_gain"] is None
    item["adaptation"]["advice_checks"][0]["constraint_id"] = "current-facing"
    assert observe_advice(item, "furniture", entry)[0]["status"] == "unknown"


def test_ambiguous_created_role_cannot_select_a_convenient_object(tmp_path) -> None:
    item, entry = adapted(tmp_path)
    item["adaptation"]["bindings"][0]["object_ids"] = []
    entry["post_scene_state"] = state(
        *entry["post_scene_state"]["objects"], object_row("chair_2", "chair", 4)
    )
    assert (
        observe_advice(item, "furniture", entry)[0]["reason"]
        == "missing_or_ambiguous_object_binding"
    )


def test_new_adaptation_requires_observation_and_legacy_remains_readable(
    tmp_path,
) -> None:
    item, entry = adapted(tmp_path)
    from scenesmith.scene_expert.schemas import RetrievedMemorySelection

    source = RetrievedMemorySelection.model_validate(item["source"])
    choice = MemoryAdaptation.model_validate(item["adaptation"])
    context = StageRelationContext(stage="furniture")
    args = dict(
        stage="furniture",
        task_spec=SceneTaskSpec(
            room_type="classroom", style="", required_large_objects=["chair", "table"]
        ),
        context=context,
        state=entry["post_scene_state"],
        brief=None,
    )
    assert validate_adaptation(source, choice, **args) == []
    choice.advice_checks = []
    assert "missing_advice_observation" in validate_adaptation(source, choice, **args)
    legacy = SuccessCase(
        case_id="old",
        stage="furniture",
        room_type="classroom",
        successful_pattern=["Align chair."],
    )
    assert (
        selection_from_record(legacy, rank=1, memory_dir=tmp_path).placement_experience
        == {}
    )


def test_readonly_roundtrip_and_optional_objects_are_retrievable(tmp_path) -> None:
    record, _ = bound_record(tmp_path)
    bank = FastMemoryStore(str(tmp_path / "bank"))
    bank.add_success_case(record)
    bank.refresh_if_changed()
    assert (
        bank.active_success_cases[0].placement_experience == record.placement_experience
    )
    # The learned chair/table pair is not in the source's bed/nightstand inventory.
    pack = MemoryRetriever(bank).retrieve(
        SceneTaskSpec(
            room_type="bedroom", style="", required_large_objects=["chair", "table"]
        ),
        "furniture",
    )
    assert record.case_id in pack.success_case_ids
    before = (bank.memory_dir / "success_cases.jsonl").read_bytes()
    selection_from_record(
        bank.active_success_cases[0], rank=1, memory_dir=bank.memory_dir
    )
    assert (bank.memory_dir / "success_cases.jsonl").read_bytes() == before


def test_prompt_projection_is_bounded_and_preserves_runtime_catalog(tmp_path) -> None:
    catalog, _ = captured(tmp_path)
    catalog["episodes"] *= 100
    evidence = {"placement_experience_catalog": catalog}
    projected = writer_prompt_evidence(evidence)
    assert len(projected["placement_experience_catalog"]["episodes"]) == 4
    assert len(evidence["placement_experience_catalog"]["episodes"]) == 100


def test_duplicate_object_ids_are_not_valid_state() -> None:
    assert not valid_state(
        state(object_row("same", "chair", 0), object_row("same", "table", 1))
    )


def test_writer_projection_does_not_hide_successful_retry(tmp_path) -> None:
    catalog, _ = captured(tmp_path)
    failed = deepcopy(catalog["episodes"][0])
    failed["stage_passed"] = False
    catalog["episodes"] = [failed] * 32 + catalog["episodes"]
    compact = writer_prompt_evidence({"placement_experience_catalog": catalog})
    assert any(
        e["stage_passed"] for e in compact["placement_experience_catalog"]["episodes"]
    )


def with_native_checks(entry: dict, label: str, other_label: str = "pass") -> dict:
    constraint = entry["relation_context"]["hard_constraints"][0]
    rows = []
    for oid, result in (("chair_1", label), ("unrelated_chair", other_label)):
        rows.append(
            constraint_observation(
                {
                    "check_id": "face-" + oid,
                    "label": result,
                    "scoring_tier": "core",
                    "contract_state": "evaluated",
                    "metric": "facing_error_deg",
                    "primary_object": oid,
                    "related_objects": ["table_1"],
                },
                constraint,
                "furniture",
            )
        )
    entry["verify_report"]["hard_check_report"]["constraint_evidence"] = rows
    return entry


def test_native_pair_does_not_inherit_another_chairs_failure(tmp_path) -> None:
    _, entry = captured(tmp_path)
    with_native_checks(entry, "pass", "fail")
    catalog = collect_placement_episodes(tmp_path, "furniture", entry)
    assert len(catalog["episodes"]) == 1  # No reversed facing inference.
    episode = catalog["episodes"][0]
    assert episode["native_checks"][0]["status"] == "verified_pass"
    assert episode["subject"]["object_id"] == "chair_1"


def test_new_failure_has_exact_pair_evidence_but_not_verified_repair(tmp_path) -> None:
    _, entry = captured(tmp_path)
    with_native_checks(entry, "fail")
    entry["verify_report"]["pass_stage"] = False
    catalog = collect_placement_episodes(tmp_path, "furniture", entry)
    episode = catalog["episodes"][0]
    writer = MemoryWriter(model="unused", llm_client=SimpleNamespace())
    evidence = _evidence(stage_passed=False, repair_verified=True)
    evidence["placement_experience_catalog"] = catalog
    candidate = FailureMemoryCandidate(
        stage="furniture",
        failure_type="facing",
        bad_pattern="Chair faces away.",
        failure_reason="Native facing check failed.",
        repair_action="Recheck front.",
        repair_verified=True,
        episode_ids=[episode["episode_id"]],
        procedure=[
            "Inspect the asset front before rotation.",
            "Recheck facing after rotating.",
        ],
        applicability=["Chair beside a table."],
    )
    content = writer._failure_content(
        candidate, writer._canonical_context(evidence, "")
    )
    assert writer._bind_placement(content, candidate, evidence, failure=True)
    from scenesmith.scene_expert.memory.schemas import MemoryUpdateOp

    ops = writer._gate_and_enrich_ops(
        [MemoryUpdateOp(op="ADD", memory_type="failure_case", content=content)],
        FullVerifyReport(pass_scene=False),
        evidence,
    )
    assert len(ops) == 1
    assert ops[0].content["repair_verified"] is False
    assert ops[0].content["scope"] == "object"


def test_new_success_uses_selected_retry_not_first_stage_entry(tmp_path) -> None:
    record, _ = bound_record(tmp_path)
    evidence = _evidence(stage_passed=False)
    writer = MemoryWriter(model="unused", llm_client=SimpleNamespace())
    from scenesmith.scene_expert.memory.schemas import MemoryUpdateOp

    ops = writer._gate_and_enrich_ops(
        [
            MemoryUpdateOp(
                op="ADD", memory_type="success_case", content=record.model_dump()
            )
        ],
        FullVerifyReport(pass_scene=False),
        evidence,
    )
    assert len(ops) == 1
    assert ops[0].content["promotion_scope"] == "stage"


def test_exact_native_advice_check_cannot_certify_different_metric(tmp_path) -> None:
    item, entry = adapted(tmp_path)
    with_native_checks(entry, "pass")
    catalog = collect_placement_episodes(tmp_path, "furniture", entry)
    episode = catalog["episodes"][0]
    item["source"]["placement_experience"]["episodes"] = [episode]
    item["source"]["spatial_relations"][0]["evidence_ref"] = episode["episode_id"]
    check = item["adaptation"]["advice_checks"][0]
    check.update(
        source_episode_id=episode["episode_id"],
        metric="native_constraint",
        constraint_id="current-facing",
    )
    assert observe_advice(item, "furniture", entry)[0]["status"] == "verified_pass"
    entry["verify_report"]["hard_check_report"]["constraint_evidence"][0][
        "metric"
    ] = "required_count"
    assert observe_advice(item, "furniture", entry)[0]["status"] == "unknown"


def test_catalog_keeps_all_attempts_and_capture_gate_stays_off(tmp_path) -> None:
    from scenesmith.scene_expert.memory.activity import MemoryActivityLogger

    logger = MemoryActivityLogger(
        tmp_path / "scene_expert",
        scene_id="s",
        task_spec=SceneTaskSpec(room_type="bedroom", style=""),
    )
    report = None
    logger.record_post_stage(
        stage="furniture",
        verify_report=report,
        repair_actions=[],
        scene_state_path="",
        capture_placement=False,
    )
    assert logger.placement_experience_catalog()["episodes"] == []
    assert (
        logger._payload["stages"]["furniture"]["placement_episodes"]["disabled"] is True
    )


def test_writer_call_roundtrip_uses_real_catalog_and_compact_model_output(
    tmp_path,
) -> None:
    from scenesmith.scene_expert.memory.schemas import MemoryWriterResponse
    from scenesmith.scene_expert.structured_llm import StructuredLLMResult
    from tests.unit.test_memory_writer_resilience import _FakeStructuredClient

    catalog, _ = captured(tmp_path)
    candidate = SuccessMemoryCandidate(
        stage="furniture",
        successful_pattern=["Align with table edge."],
        episode_ids=[catalog["episodes"][0]["episode_id"]],
        procedure=[
            "Inspect the table frame.",
            "Rotate the chair relative to its anchor.",
        ],
        applicability=["Chair/table."],
    )
    client = _FakeStructuredClient(
        StructuredLLMResult(value=MemoryWriterResponse(success_cases=[candidate]))
    )
    writer = MemoryWriter(
        model="unused", llm_client=client, skill_bootstrap_enabled=False
    )
    evidence = _evidence()
    evidence["placement_experience_catalog"] = catalog
    ops = writer.write(
        "",
        FullVerifyReport(pass_scene=True, overall_score=0.9),
        evidence_payload=evidence,
    )
    assert len(ops) == 1
    assert ops[0].content["placement_experience"]["procedure"] == candidate.procedure
    assert writer.last_trace["placement_candidate_decisions"][0]["decision"] == "bound"


def test_native_check_validation_is_not_repeated_per_object_pair(
    tmp_path, monkeypatch
) -> None:
    from scenesmith.scene_expert.memory import placement

    _, entry = captured(tmp_path)
    with_native_checks(entry, "pass")
    entry["post_scene_state"] = state(
        *entry["post_scene_state"]["objects"],
        *[object_row(f"optional_{i}", "optional shelf", i + 3) for i in range(60)],
    )
    original = placement.resolve_constraint_evidence
    calls = []

    def counted(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(placement, "resolve_constraint_evidence", counted)
    assert collect_placement_episodes(tmp_path, "furniture", entry)["episodes"]
    assert len(calls) == 1
