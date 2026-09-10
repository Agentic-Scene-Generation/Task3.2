"""Writer-only CPU regressions for request budgets and real persistence semantics."""

import json

from copy import deepcopy
from types import SimpleNamespace

import pytest

from scenesmith.scene_expert.memory.schemas import (
    MemoryUpdateOp,
    MemoryWriterResponse,
    SuccessMemoryCandidate,
)
from scenesmith.scene_expert.memory.store import FastMemoryStore
from scenesmith.scene_expert.memory.writer import MemoryWriter
from scenesmith.scene_expert.memory.writer_prompt import (
    INSTRUCTION,
    WriterPromptBudgetError,
    build_writer_prompt,
)
from scenesmith.scene_expert.run_metrics import _writer_metrics
from scenesmith.scene_expert.schemas import FullVerifyReport
from scenesmith.scene_expert.structured_llm import (
    SceneExpertStructuredLLMClient,
    StructuredLLMProfile,
    StructuredLLMResult,
)
from tests.unit.test_memory_writer_resilience import _evidence, _FakeStructuredClient
from tests.unit.test_placement_experience import captured
from tests.unit.test_scene_expert_structured_llm import _FakeOpenAI, _Payload, _response


def evidence_with_catalog(tmp_path):
    catalog, _ = captured(tmp_path)
    evidence = _evidence()
    evidence["placement_experience_catalog"] = catalog
    return evidence


def prompt(evidence, budget=30000):
    return build_writer_prompt(
        evidence=evidence,
        final_report={"pass_scene": False},
        trace_summary="Long narrative. " * 40000,
        related_old_memory="Old advice " * 50000,
        max_user_bytes=budget,
    )


def test_unbounded_audit_fields_do_not_enter_model_context(tmp_path):
    evidence = evidence_with_catalog(tmp_path)
    evidence["code_provenance"] = {"source_hashes": {"huge": "x" * 500000}}
    evidence["stages"][0]["memory_pack"] = {"old": "x" * 500000}
    evidence["stages"][0]["verify_report"]["critique_summary"] = "位置错误。" * 40000
    original = deepcopy(evidence)
    text, meta = prompt(evidence, 16000)
    assert len(text.encode("utf-8")) <= 16000
    data = json.loads(text[len(INSTRUCTION) :])
    assert "code_provenance" not in data["evidence"]
    assert "memory_pack" not in data["evidence"]["stages"][0]
    assert meta["selected_episode_ids"]
    assert evidence == original
    assert meta["source_evidence_bytes"] > 1000000


def test_unknown_episodes_cannot_crowd_out_passing_evidence(tmp_path):
    from scenesmith.scene_expert.memory.schemas import PlacementEpisode

    evidence = evidence_with_catalog(tmp_path)
    good = evidence["placement_experience_catalog"]["episodes"][0]
    unknown = PlacementEpisode.model_validate(good).model_copy(
        update={"stage_passed": False}
    )
    unknown.episode_id = unknown.content_hash()
    evidence["placement_experience_catalog"]["episodes"] = [
        unknown.model_dump()
    ] * 100 + [good]
    _, meta = prompt(evidence)
    assert meta["selected_episode_ids"] == [good["episode_id"]]
    assert any(r["reason"] == "no_verified_outcome" for r in meta["omissions"])


def test_byte_budget_drops_whole_episodes_never_mutates_measurements(tmp_path):
    evidence = evidence_with_catalog(tmp_path)
    text, meta = prompt(evidence, 1400)
    assert len(text.encode("utf-8")) <= 1400
    assert not meta["selected_episode_ids"]
    assert (
        json.loads(text[len(INSTRUCTION) :])["evidence"][
            "placement_experience_catalog"
        ]["episodes"]
        == []
    )
    with pytest.raises(WriterPromptBudgetError):
        prompt(evidence, 20)


def context_error():
    return type("BadRequestError", (Exception,), {})(
        "request (98575 tokens) exceeds the available context size (65536 tokens), "
        "'type': 'exceed_context_size_error', 'n_ctx': 65536"
    )


@pytest.mark.parametrize("reducer", [None, lambda messages, error: messages])
def test_context_overflow_never_retries_unchanged_input(reducer):
    fake = _FakeOpenAI([context_error()])
    client = SceneExpertStructuredLLMClient(model="unused", client=fake)
    result = client.complete(
        role="test",
        stage="stage",
        event="test",
        messages=[{"role": "user", "content": "x" * 1000}],
        response_model=_Payload,
        context_reducer=reducer,
        profile=StructuredLLMProfile(max_attempts=2),
    )
    assert result.final_error_kind == "context_length"
    assert len(fake.calls) == 1


def test_context_retry_reduces_bytes_without_increasing_output_or_downgrading():
    fake = _FakeOpenAI([context_error(), _response(content='{"value":"ok"}')])
    client = SceneExpertStructuredLLMClient(model="unused", client=fake)
    result = client.complete(
        role="test",
        stage="stage",
        event="test",
        messages=[{"role": "user", "content": "x" * 1000}],
        response_model=_Payload,
        context_reducer=lambda m, e: [
            {"role": "user", "content": "small complete input"}
        ],
        profile=StructuredLLMProfile(
            max_tokens=100, retry_max_tokens=200, max_attempts=2
        ),
    )
    assert result.success
    assert fake.calls[0]["max_tokens"] == fake.calls[1]["max_tokens"] == 100
    assert (
        fake.calls[0]["response_format"]["type"]
        == fake.calls[1]["response_format"]["type"]
    )
    assert result.attempts[1].retry_strategy == "reduce_input_context"


def test_writer_context_recovery_and_full_input_archive(tmp_path):
    evidence = evidence_with_catalog(tmp_path)
    evidence["stages"][0]["verify_report"]["critique_summary"] = "\n\n".join(
        f"Observation {i}: " + "The chair is placed beside the table. " * 20
        for i in range(8)
    )
    fake = _FakeOpenAI(
        [
            context_error(),
            _response(
                content=MemoryWriterResponse(
                    noop_reason="No useful method"
                ).model_dump_json()
            ),
        ]
    )
    client = SceneExpertStructuredLLMClient(model="unused", client=fake)
    writer = MemoryWriter(
        model="unused",
        llm_client=client,
        debug_dir=tmp_path / "audit",
        skill_bootstrap_enabled=False,
    )
    assert writer.write("Summary", FullVerifyReport(), evidence_payload=evidence) == []
    assert len(fake.calls) == 2
    assert writer.last_trace["success"]
    sizes = [p["user_bytes"] for p in writer.last_trace["prompt_projection_attempts"]]
    assert sizes[1] < sizes[0]
    audit = json.loads(
        (tmp_path / "audit/memory_writer_input.json").read_text(encoding="utf-8")
    )
    assert audit["evidence"] == evidence
    assert (tmp_path / "audit/memory_writer_prompt_02.json").is_file()


def test_low_overall_score_does_not_discard_verified_stage_experience(tmp_path):
    evidence = evidence_with_catalog(tmp_path)
    episode = evidence["placement_experience_catalog"]["episodes"][0]
    response = MemoryWriterResponse(
        success_cases=[
            SuccessMemoryCandidate(
                stage="furniture",
                successful_pattern=["Align a chair relative to the table."],
                episode_ids=[episode["episode_id"]],
                procedure=[
                    "Inspect the table frame and chair footprint.",
                    "Adjust the chair offset and relative yaw relative to the table in the available space.",
                ],
                applicability=["A chair placed next to a table."],
            )
        ]
    )
    writer = MemoryWriter(
        model="unused",
        llm_client=_FakeStructuredClient(StructuredLLMResult(value=response)),
        debug_dir=tmp_path / "audit",
        skill_bootstrap_enabled=False,
    )
    ops = writer.write(
        "",
        FullVerifyReport(pass_scene=True, overall_score=0.65),
        evidence_payload=evidence,
    )
    assert len(ops) == 1
    assert ops[0].content["promotion_scope"] == "stage"
    assert ops[0].content["placement_experience"]["episodes"][0] == episode
    store = FastMemoryStore(tmp_path / "bank")
    applied = store.apply_updates(ops)
    writer.record_store_result(applied)
    assert writer.last_trace["persisted_count"] == 1
    assert writer.last_trace["persistence_confirmed"]
    again = store.apply_updates(ops)
    writer.record_store_result(again)
    assert writer.last_trace["persisted_count"] == 0
    assert writer.last_trace["proposed_mutation_count"] == 1
    reopened = FastMemoryStore(tmp_path / "bank")
    assert len(reopened.success_cases) == 1
    assert (
        reopened.success_cases[0].placement_experience.episodes[0].episode_id
        == episode["episode_id"]
    )
    summary = store.apply_updates(
        [
            MemoryUpdateOp(
                op="UPDATE",
                memory_type="success_case",
                target_id=reopened.success_cases[0].case_id,
                content={"positive_guidance": ["Use the anchor's frame."]},
            )
        ]
    )
    assert summary["active_records_changed"] == 1


def test_proposed_counts_are_not_reported_as_store_writes(tmp_path):
    status = {
        "candidate_count": 5,
        "persisted_count": 5,
        "promoted_count": 5,
        "store_apply": {
            "added": 1,
            "merged": 1,
            "updated": 0,
            "active_records_changed": 1,
        },
    }
    metrics = _writer_metrics(
        tmp_path, {"component_status": {"memory_writer": status}}, []
    )
    assert metrics["memory_writer_persisted"] == 2
    assert metrics["memory_writer_promoted"] == 1
    del status["store_apply"]
    metrics = _writer_metrics(
        tmp_path, {"component_status": {"memory_writer": status}}, []
    )
    assert metrics["memory_writer_persisted"] == 0
    assert not metrics["memory_writer_persistence_observed"]


def test_store_preserves_distinct_procedures_with_identical_summary(tmp_path):
    from tests.unit.test_placement_experience import bound_record

    record, _ = bound_record(tmp_path)
    other = record.model_copy(
        deep=True, update={"case_id": "other", "source_run_id": "independent-run"}
    )
    other.placement_experience.procedure = [
        "Inspect the table frame.",
        "Rotate the chair after verifying its offset.",
    ]
    store = FastMemoryStore(tmp_path / "bank")
    summary = store.apply_updates(
        [
            MemoryUpdateOp(op="ADD", memory_type="success_case", content=r.model_dump())
            for r in (record, other)
        ]
    )
    assert summary["added"] == 2
    assert summary["merged"] == 0
    reopened = FastMemoryStore(tmp_path / "bank")
    assert len(reopened.success_cases) == 2
    assert (
        reopened.success_cases[0].placement_experience.procedure
        != reopened.success_cases[1].placement_experience.procedure
    )


def test_catalog_ids_not_delivered_to_model_cannot_be_promoted(tmp_path):
    from tests.unit.test_placement_experience import bound_record

    record, _ = bound_record(tmp_path)
    episode = record.placement_experience.episodes[0]
    writer = MemoryWriter(model="unused", llm_client=SimpleNamespace())
    writer._visible_episode_ids = set()
    candidate = SuccessMemoryCandidate(
        stage=episode.stage,
        successful_pattern=["A spatial method"],
        episode_ids=[episode.episode_id],
        procedure=record.placement_experience.procedure,
        applicability=record.placement_experience.applicability,
    )
    assert not writer._bind_placement(
        record.model_dump(),
        candidate,
        {"placement_experience_catalog": {"episodes": [episode.model_dump()]}},
        failure=False,
    )
    assert (
        "episode_not_in_model_input"
        in writer.last_trace["placement_candidate_decisions"][0]["reasons"]
    )


def test_replay_uses_exact_archive_before_reconstruction(tmp_path):
    from scripts.replay_sceneexpert_memory_writer import load_input

    evidence = evidence_with_catalog(tmp_path)
    client = _FakeStructuredClient(StructuredLLMResult(value=MemoryWriterResponse()))
    source = tmp_path / "scene_expert"
    writer = MemoryWriter(
        model="unused",
        llm_client=client,
        debug_dir=source / "memory",
        skill_bootstrap_enabled=False,
    )
    writer.write("complete summary", FullVerifyReport(), evidence_payload=evidence)
    loaded = load_input(source)
    assert loaded["evidence"] == evidence
    assert loaded["replay_source"] == "exact_writer_input"
