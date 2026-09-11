"""Regression coverage for the real 091 replay's missing links and overclaims."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from scenesmith.scene_expert.memory.adaptation import validate_adaptation
from scenesmith.scene_expert.memory.contracts import selection_from_record
from scenesmith.scene_expert.memory.placement import experience_text, pair_measurements
from scenesmith.scene_expert.memory.placement_methods import (
    bind_method_steps,
    critic_catalog,
    instruction_reasons,
    methods_valid,
)
from scenesmith.scene_expert.memory.schemas import (
    MemoryUpdateOp,
    PlacementEpisode,
    PlacementMethodCandidate,
    SuccessCase,
    SuccessMemoryCandidate,
)
from scenesmith.scene_expert.memory.store import FastMemoryStore
from scenesmith.scene_expert.memory.writer import MemoryWriter
from scenesmith.scene_expert.memory.writer_prompt import build_writer_prompt
from scenesmith.scene_expert.schemas import FullVerifyReport
from tests.unit.test_memory_writer_resilience import _evidence
from tests.unit.test_placement_experience import adapted, object_row


def source():
    """Small geometric analogue of the two 091 furniture source pairs."""
    evidence = _evidence()
    bed = object_row("bed_0", "bed", 0)
    wall = object_row("south_wall", "south_wall", 0.5094)
    nightstand = object_row("nightstand_1", "nightstand", -0.5492)
    episodes = []
    for anchor in [wall, nightstand]:
        e = PlacementEpisode(
            episode_id="",
            stage="furniture",
            state_fingerprint="same-final-state",
            subject=bed,
            anchor=anchor,
            measurements=pair_measurements(bed, anchor),
            actions=[
                {
                    "tool_call_id": "place-bed",
                    "tool_name": "add_furniture_to_scene_tool",
                }
            ],
            evidence_refs=["original/scene_expert/audit/designer.json"],
            stage_passed=True,
            stage_scores={"semantic": 0.8},
        )
        e.episode_id = e.content_hash()
        episodes.append(e)
    evidence["placement_experience_catalog"] = {
        "episodes": [e.model_dump(mode="json") for e in episodes]
    }
    evidence["stages"][0]["verify_report"]["critique_summary"] = (
        "The bed is backed against the wall with a small observed gap. "
        "The nightstands flank the bed.\n\n"
        "The reading chair remains incorrectly facing the bookshelf. Do not treat this as a repaired success."
    )
    return evidence, episodes


def candidate(episodes):
    wall, nightstand = episodes
    return SuccessMemoryCandidate(
        stage="furniture",
        successful_pattern=[wall.episode_id],
        positive_guidance=["Always copy the source gap of 0.01m."],
        episode_ids=[nightstand.episode_id],
        procedure=["Unsafe stale text must not be injected."],
        applicability=["A bedroom with a bed and flanking nightstands."],
        method_steps=[
            PlacementMethodCandidate(
                instruction="Inspect the bed footprint and recompute its gap to the wall.",
                episode_ids=[wall.episode_id],
            ),
            PlacementMethodCandidate(
                instruction="Align the nightstands relative to the bed and inspect spacing.",
                episode_ids=[nightstand.episode_id],
            ),
        ],
    )


def bind(evidence, proposal):
    writer = MemoryWriter(model="unused", llm_client=SimpleNamespace())
    ctx = writer._canonical_context(evidence, "")
    content = writer._success_content(
        proposal, ctx, FullVerifyReport(pass_scene=True, overall_score=0.9)
    )
    ok = writer._bind_placement(content, proposal, evidence, failure=False)
    return ok, content, writer


def test_per_step_union_restores_omitted_bed_wall_and_normalizes_all_text(tmp_path):
    evidence, episodes = source()
    ok, content, writer = bind(evidence, candidate(episodes))
    assert ok
    record = SuccessCase.model_validate(content)
    exp = record.placement_experience
    assert exp.schema_version == "placement-experience.v2"
    assert methods_valid(exp)
    assert {e.episode_id for e in exp.episodes} == {e.episode_id for e in episodes}
    assert record.successful_pattern == record.positive_guidance == exp.procedure
    assert all(s.verification_status == "transfer_unverified" for s in exp.method_steps)
    assert exp.episodes[0].measurements["aabb_separation_m"] == 0.0094
    store = FastMemoryStore(tmp_path / "bank")
    ops = writer._gate_and_enrich_ops(
        [MemoryUpdateOp(op="ADD", memory_type="success_case", content=content)],
        FullVerifyReport(pass_scene=True, overall_score=0.9),
        evidence,
    )
    assert store.apply_updates(ops)["added"] == 1
    reopened = FastMemoryStore(tmp_path / "bank").success_cases[0]
    text = selection_from_record(
        reopened, rank=1, memory_dir=tmp_path / "bank"
    ).injected_text
    assert "recompute its gap" in text and "0.0094" in text
    assert "Always copy" not in text and "Unsafe stale" not in text
    assert "Transfer hypotheses" in text and "Source observation only" in text
    assert "recompute its gap" in reopened.embedding_text


def test_mentioning_wall_without_citing_wall_does_not_borrow_stage_success():
    evidence, episodes = source()
    c = candidate(episodes)
    c.method_steps[0].episode_ids = [episodes[1].episode_id]
    ok, content, writer = bind(evidence, c)
    # Reject the unsupported wall step, not the independent bedside method.
    assert ok
    assert content["placement_experience"]["procedure"] == [
        c.method_steps[1].instruction
    ]
    decisions = writer.last_trace["placement_method_decisions"][0]["steps"]
    assert any("uncited_object_roles:wall" in r for r in decisions[0]["reasons"])


@pytest.mark.parametrize(
    "instruction",
    [
        "Anchor the bed with a 0.01m gap and 0° yaw.",
        "Keep fixtures at least one metre apart.",
        "Verify distance >1m to ensure distinct light pools.",
        "This method guarantees collision-free placement.",
        "Always use this arrangement.",
        "a" * 64,
    ],
)
def test_constants_and_guarantees_do_not_become_transfer_instructions(instruction):
    assert instruction_reasons(instruction)


def test_methods_retain_flexible_geometry_reasoning():
    assert not instruction_reasons(
        "Inspect the 3D footprint, align to the anchor and remeasure the available space."
    )


def test_critic_quotes_are_exact_hashed_opinions_not_invented_geometry(tmp_path):
    evidence, episodes = source()
    quote = critic_catalog(evidence)[0]
    assert quote.quote in evidence["stages"][0]["verify_report"]["critique_summary"]
    assert quote.evidence_id == quote.content_hash()
    c = candidate(episodes)
    c.method_steps[0].critic_refs = [quote.evidence_id]
    ok, content, _ = bind(evidence, c)
    assert ok
    record = SuccessCase.model_validate(content)
    assert record.placement_experience.critic_advice[0] == quote
    assert "source-scene opinion" in experience_text(record.placement_experience)
    assert not any(r.has_verified_geometry for r in record.spatial_relations)
    record.placement_experience.critic_advice[0].quote = "Changed assertion"
    assert not methods_valid(record.placement_experience)
    rendered = selection_from_record(record, rank=1, memory_dir=tmp_path)
    assert rendered.injected_text == ""
    assert "invalid_method_step_contract" in rendered.evidence_warnings


@pytest.mark.parametrize(
    "fault", ["wrong_stage", "invisible", "missing", "other_snapshot"]
)
def test_step_sources_must_be_visible_and_from_matching_snapshot(fault):
    evidence, episodes = source()
    c = candidate(episodes)
    quotes = {q.evidence_id: q for q in critic_catalog(evidence)}
    quote = next(iter(quotes.values()))
    c.method_steps[0].critic_refs = [quote.evidence_id]
    visible = set(quotes)
    if fault == "wrong_stage":
        quote.stage = "ceiling_mounted"
    elif fault == "other_snapshot":
        quote.state_fingerprint = "different-attempt"
    elif fault == "missing":
        quotes.clear()
    else:
        visible.clear()
    steps, decisions = bind_method_steps(
        c,
        {e.episode_id: e for e in episodes},
        quotes,
        visible_episodes={e.episode_id for e in episodes},
        visible_quotes=visible,
    )
    assert len(steps) == 1
    assert decisions[0]["decision"] == "rejected"


def test_ambiguous_round_does_not_mix_critic_quotes_with_final_geometry():
    evidence, episodes = source()
    evidence["stages"].append(deepcopy(evidence["stages"][0]))
    assert critic_catalog(evidence) == []
    # Pure spatial methods still work: absence of prose does not block collection.
    assert bind(evidence, candidate(episodes))[0]


def test_different_stage_attempts_cannot_be_combined_into_one_method():
    evidence, episodes = source()
    episodes[1].state_fingerprint = "old-attempt"
    episodes[1].episode_id = episodes[1].content_hash()
    evidence["placement_experience_catalog"]["episodes"] = [
        e.model_dump() for e in episodes
    ]
    ok, _, writer = bind(evidence, candidate(episodes))
    assert not ok
    assert (
        "method_snapshot_mismatch"
        in writer.last_trace["placement_candidate_decisions"][0]["reasons"]
    )


def test_prompt_packs_complete_quote_units_within_budget():
    evidence, _ = source()
    text, meta = build_writer_prompt(
        evidence=evidence,
        final_report={},
        trace_summary="",
        related_old_memory="",
        max_user_bytes=16000,
    )
    assert len(text.encode("utf-8")) <= 16000
    assert meta["selected_critic_ids"]
    assert all(q.quote in text.replace("\\n", "\n") for q in critic_catalog(evidence))
    assert evidence["stages"][0]["verify_report"]["critique_summary"]


def test_old_bank_is_readable_without_fabricating_per_step_verification(tmp_path):
    evidence, episodes = source()
    ok, content, _ = bind(evidence, candidate(episodes))
    assert ok
    exp = content["placement_experience"]
    exp.pop("method_steps")
    exp.pop("critic_advice")
    exp["schema_version"] = "placement-experience.v1"
    record = SuccessCase.model_validate(content)
    rendered = selection_from_record(record, rank=1, memory_dir=tmp_path)
    assert "Legacy unverified hypotheses" in rendered.injected_text
    assert "legacy_method_step_binding_unavailable" in rendered.evidence_warnings


def test_planner_cannot_reintroduce_source_numeric_targets(tmp_path):
    from scenesmith.scene_expert.schemas import (
        MemoryAdaptation,
        RetrievedMemorySelection,
    )

    item, entry = adapted(tmp_path)
    source_record = RetrievedMemorySelection.model_validate(item["source"])
    choice = MemoryAdaptation.model_validate(item["adaptation"])
    choice.actions = ["Keep a 0.5m gap to the table."]
    reasons = validate_adaptation(
        source_record,
        choice,
        stage="furniture",
        task_spec=None,
        context=None,
        state=entry["post_scene_state"],
        brief=None,
    )
    assert "unverified_memory_target_or_guarantee" in reasons


def test_planner_step_scope_keeps_required_source_pairs(tmp_path):
    from scenesmith.scene_expert.schemas import (
        MemoryAdaptation,
        RetrievedMemorySelection,
    )

    item, entry = adapted(tmp_path)
    selected = RetrievedMemorySelection.model_validate(item["source"])
    choice = MemoryAdaptation.model_validate(item["adaptation"])
    choice.source_relation_indices = []
    reasons = validate_adaptation(
        selected,
        choice,
        stage="furniture",
        task_spec=None,
        context=None,
        state=entry["post_scene_state"],
        brief=None,
    )
    assert "method_step_source_outside_relation_scope" in reasons
    choice.source_method_step_indices = []
    reasons = validate_adaptation(
        selected,
        choice,
        stage="furniture",
        task_spec=None,
        context=None,
        state=entry["post_scene_state"],
        brief=None,
    )
    assert "invalid_source_method_scope" in reasons


def test_planner_can_reference_real_current_object_ids(tmp_path):
    from scenesmith.scene_expert.schemas import (
        MemoryAdaptation,
        RetrievedMemorySelection,
    )

    item, entry = adapted(tmp_path)
    selected = RetrievedMemorySelection.model_validate(item["source"])
    choice = MemoryAdaptation.model_validate(item["adaptation"])
    choice.actions = ["Inspect table_1 and recompute space for chair_1."]
    assert (
        validate_adaptation(
            selected,
            choice,
            stage="furniture",
            task_spec=None,
            context=None,
            state=entry["post_scene_state"],
            brief=None,
        )
        == []
    )


def test_planner_can_select_supported_method_subset_without_requiring_all_pairs(
    tmp_path,
):
    from scenesmith.scene_expert.schemas import MemoryAdaptation

    evidence, episodes = source()
    _, content, _ = bind(evidence, candidate(episodes))
    selected = selection_from_record(
        SuccessCase.model_validate(content), rank=1, memory_dir=tmp_path
    )
    c = MemoryAdaptation(
        memory_type="success",
        memory_id=selected.memory_id,
        source_content_hash=selected.content_hash,
        decision="adapted",
        source_relation_indices=[1],
        source_method_step_indices=[1],
        actions=[
            "Align nightstands relative to the bed after checking available space."
        ],
        checks=["Measure bedside separation."],
        bindings=[
            {"source_role": "bed", "current_role": "bed", "object_ids": ["bed_0"]},
            {
                "source_role": "nightstand",
                "current_role": "nightstand",
                "object_ids": ["nightstand_1"],
            },
        ],
        advice_checks=[
            {
                "source_episode_id": episodes[1].episode_id,
                "metric": "aabb_separation_m",
                "subject_role": "bed",
                "anchor_role": "nightstand",
            }
        ],
    )
    state = {"objects": [episodes[1].subject, episodes[1].anchor]}
    assert (
        validate_adaptation(
            selected,
            c,
            stage="furniture",
            task_spec=None,
            context=None,
            state=state,
            brief=None,
        )
        == []
    )
    c.source_method_step_indices = [0, 1]
    assert "method_step_source_outside_relation_scope" in validate_adaptation(
        selected,
        c,
        stage="furniture",
        task_spec=None,
        context=None,
        state=state,
        brief=None,
    )


def test_optional_empty_quote_envelope_never_overruns_byte_budget():
    from scenesmith.scene_expert.memory.writer_prompt import WriterPromptBudgetError

    evidence, _ = source()
    for budget in range(1000, 7000, 17):
        try:
            text, meta = build_writer_prompt(
                evidence=evidence,
                final_report={},
                trace_summary="",
                related_old_memory="",
                max_user_bytes=budget,
            )
        except WriterPromptBudgetError:
            continue
        assert meta["user_bytes"] == len(text.encode("utf-8")) <= budget


def test_writer_call_persists_explicit_steps_and_quotes_without_fallback(tmp_path):
    from scenesmith.scene_expert.memory.schemas import MemoryWriterResponse
    from scenesmith.scene_expert.structured_llm import StructuredLLMResult
    from tests.unit.test_memory_writer_resilience import _FakeStructuredClient

    evidence, episodes = source()
    c = candidate(episodes)
    c.method_steps[0].critic_refs = [critic_catalog(evidence)[0].evidence_id]
    client = _FakeStructuredClient(
        StructuredLLMResult(value=MemoryWriterResponse(success_cases=[c]))
    )
    writer = MemoryWriter(
        model="unused", llm_client=client, skill_bootstrap_enabled=False
    )
    ops = writer.write(
        "", FullVerifyReport(pass_scene=False), evidence_payload=evidence
    )
    assert len(ops) == 1
    assert ops[0].content["promotion_scope"] == "stage"
    assert ops[0].content["source_scene_passed"] is False
    assert ops[0].content["placement_experience"]["critic_advice"]
    assert (
        writer.last_trace["placement_method_decisions"][0]["steps"][0]["decision"]
        == "bound_hypothesis"
    )
    bank = FastMemoryStore(tmp_path / "bank")
    assert bank.apply_updates(ops)["added"] == 1
    assert "recompute its gap" in bank.success_cases[0].embedding_text


def test_missing_v2_steps_cannot_downgrade_into_legacy_injection(tmp_path):
    from scenesmith.scene_expert.memory.placement import experience_priority

    evidence, episodes = source()
    _, content, _ = bind(evidence, candidate(episodes))
    record = SuccessCase.model_validate(content)
    record.placement_experience.method_steps = []
    assert experience_priority(record) == 0
    assert (
        selection_from_record(record, rank=1, memory_dir=tmp_path).injected_text == ""
    )


def test_shared_legacy_references_do_not_get_explicit_binding_priority(tmp_path):
    from scenesmith.scene_expert.memory.placement import experience_priority

    item, _ = adapted(tmp_path)
    from tests.unit.test_placement_experience import bound_record

    record, _ = bound_record(tmp_path)
    assert methods_valid(record.placement_experience)
    assert experience_priority(record) == 1
    assert "legacy_shared_step_references" in item["source"]["evidence_warnings"]


def test_critic_mentioned_current_role_is_usable_but_not_geometry_verified(tmp_path):
    from scenesmith.scene_expert.schemas import (
        MemoryAdaptation,
        RetrievedMemorySelection,
    )

    evidence, episodes = source()
    chair = object_row("reading_chair_0", "reading chair", 2)
    source_chair = episodes[1].model_copy(deep=True)
    source_chair.subject = chair
    source_chair.stage = "floor_plan"
    source_chair.measurements = pair_measurements(chair, source_chair.anchor)
    source_chair.episode_id = source_chair.content_hash()
    evidence["placement_experience_catalog"]["episodes"].append(
        source_chair.model_dump()
    )
    quote = next(q for q in critic_catalog(evidence) if "reading chair" in q.quote)
    assert "reading chair" in quote.object_roles
    # Mentioning a wall is not evidence for every named wall in the source.
    assert "south wall" not in critic_catalog(evidence)[0].object_roles
    c = candidate(episodes)
    c.method_steps[0].critic_refs = [quote.evidence_id]
    ok, content, _ = bind(evidence, c)
    assert ok
    selected = selection_from_record(
        SuccessCase.model_validate(content), rank=1, memory_dir=tmp_path
    )
    item, entry = adapted(tmp_path)
    choice = MemoryAdaptation.model_validate(item["adaptation"])
    # Reuse a valid adaptation fixture but bind the bed/nightstand/wall pairs.
    choice.memory_id = selected.memory_id
    choice.source_content_hash = selected.content_hash
    choice.source_relation_indices = [0, 1]
    choice.actions = [
        "Inspect reading_chair_0 and the bedside placement in the current room."
    ]
    choice.checks = ["Inspect object spacing."]
    choice.advice_checks = []
    choice.bindings = [
        type(choice.bindings[0])(source_role=role, current_role=role, object_ids=[oid])
        for role, oid in [
            ("bed", "bed_0"),
            ("south_wall", "south_wall"),
            ("nightstand", "nightstand_1"),
            ("reading chair", "reading_chair_0"),
        ]
    ]
    reasons = validate_adaptation(
        selected,
        choice,
        stage="furniture",
        task_spec=None,
        context=None,
        state={
            "objects": [
                episodes[0].subject,
                episodes[0].anchor,
                episodes[1].anchor,
                chair,
            ]
        },
        brief=None,
    )
    assert "unknown_source_role" not in reasons
    assert "object_role_mismatch" not in reasons
    assert "unbound_current_role" not in reasons
    assert (
        "missing_advice_observation" in reasons
    )  # A quote cannot replace measured checks.
