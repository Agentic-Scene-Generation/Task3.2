"""Local evidence may survive stage failure, without laundering the failure."""

from copy import deepcopy

import pytest

from scenesmith.scene_expert.memory.placement_methods import (
    critic_catalog,
    methods_valid,
)
from scenesmith.scene_expert.memory.placement_outcomes import success_scope
from scenesmith.scene_expert.memory.schemas import (
    MemoryUpdateOp,
    SuccessCase,
    PlacementMethodCandidate,
    PlacementRelationCandidate,
)
from scenesmith.scene_expert.memory.writer_prompt import build_writer_prompt
from scenesmith.scene_expert.schemas import FullVerifyReport
from tests.unit.test_placement_method_contract import source, candidate, bind


def local_source():
    evidence, episodes = source()
    for e in episodes:
        e.stage_passed = False
        relation = (
            "against_wall" if e.anchor["object_id"] == "south_wall" else "next_to"
        )
        e.native_checks = [
            {
                "status": "verified_pass",
                "constraint": {"relation": relation},
                "check": {
                    "label": "pass",
                    "scoring_tier": "core",
                    "stage": "furniture",
                    "observations": {
                        "primary_object": e.subject["object_id"],
                        "related_objects": [e.anchor["object_id"]],
                    },
                },
            }
        ]
        e.episode_id = e.content_hash()
    evidence["stages"][0]["verify_report"]["pass_stage"] = False
    evidence["placement_experience_catalog"]["episodes"] = [
        e.model_dump() for e in episodes
    ]
    c = candidate(episodes)
    c.method_steps[1].instruction = (
        "Place the nightstand next to the bed, checking the footprint before adjustment."
    )
    return evidence, episodes, c


def test_projection_binding_and_store_gate_agree_on_local_pass(tmp_path):
    evidence, episodes, c = local_source()
    _, meta = build_writer_prompt(
        evidence=evidence,
        final_report={},
        trace_summary="",
        related_old_memory="",
        max_user_bytes=30000,
    )
    assert set(e.episode_id for e in episodes) <= set(meta["selected_episode_ids"])
    ok, content, writer = bind(evidence, c)
    assert ok
    ops = writer._gate_and_enrich_ops(
        [MemoryUpdateOp(op="ADD", memory_type="success_case", content=content)],
        FullVerifyReport(pass_scene=False, overall_score=0.3),
        evidence_payload=evidence,
    )
    assert len(ops) == 1
    r = SuccessCase.model_validate(ops[0].content)
    assert r.promotion_scope == "relation" and r.source_scene_passed is False
    assert all(e.stage_passed is False for e in r.placement_experience.episodes)
    assert r.placement_experience.limitations and methods_valid(r.placement_experience)
    assert (
        writer.last_trace["placement_promotion_decisions"][-1]["reason"]
        == "relation_success"
    )
    from scenesmith.scene_expert.memory.store import FastMemoryStore
    from scenesmith.scene_expert.memory.contracts import selection_from_record

    bank = FastMemoryStore(tmp_path / "bank")
    assert bank.apply_updates(ops)["added"] == 1
    stored = FastMemoryStore(tmp_path / "bank").success_cases[0]
    assert stored.promotion_scope == "relation"
    selection = selection_from_record(stored, rank=1, memory_dir=tmp_path / "bank")
    assert "Source stage did not pass" in selection.injected_text
    assert "next_to:verified_pass" in selection.injected_text


@pytest.mark.parametrize(
    "fault", ["unknown", "wrong_pair", "wrong_stage", "aggregate_label"]
)
def test_unknown_or_unbound_check_does_not_create_local_success(fault):
    evidence, episodes, c = local_source()
    e = episodes[1]
    row = e.native_checks[0]
    if fault == "unknown":
        row["status"] = "unknown"
    elif fault == "wrong_pair":
        row["check"]["observations"]["related_objects"] = ["chair_0"]
    elif fault == "wrong_stage":
        row["check"]["stage"] = "wall_mounted"
    else:
        row["check"]["label"] = "unknown"
    e.episode_id = e.content_hash()
    assert success_scope(e, [e]) is None


def test_conflicting_reversed_pair_blocks_projection_and_final_promotion():
    evidence, episodes, c = local_source()
    ok, content, writer = bind(evidence, c)
    assert ok
    conflict = deepcopy(episodes[1])
    conflict.subject, conflict.anchor = conflict.anchor, conflict.subject
    from scenesmith.scene_expert.memory.placement import pair_measurements

    conflict.measurements = pair_measurements(conflict.subject, conflict.anchor)
    conflict.native_checks[0]["status"] = "verified_fail"
    conflict.episode_id = conflict.content_hash()
    evidence["placement_experience_catalog"]["episodes"].append(conflict.model_dump())
    assert success_scope(episodes[1], [*episodes, conflict]) is None
    assert not bind(evidence, c)[0]
    assert not writer._gate_and_enrich_ops(
        [MemoryUpdateOp(op="ADD", memory_type="success_case", content=content)],
        FullVerifyReport(pass_scene=True, overall_score=0.9),
        evidence_payload=evidence,
    )


def test_near_pass_does_not_certify_facing_method():
    evidence, episodes, c = local_source()
    c.method_steps[1].instruction = "Orient the nightstand to face the bed."
    ok, _, writer = bind(evidence, c)
    assert not ok
    assert (
        "method_not_supported_by_passing_relation"
        in writer.last_trace["placement_candidate_decisions"][-1]["reasons"]
    )


def test_critic_only_method_cannot_borrow_unrelated_local_pass():
    evidence, episodes, c = local_source()
    c.method_steps = [
        PlacementMethodCandidate(
            instruction="Align the nightstands relative to the bed.",
            critic_refs=[critic_catalog(evidence)[0].evidence_id],
        )
    ]
    assert not bind(evidence, c)[0]


def test_incomplete_checklist_is_not_a_transferable_method():
    evidence, episodes = source()
    c = candidate(episodes)
    c.method_steps[1].instruction = "Always align the nightstands relative to the bed."
    ok, _, writer = bind(evidence, c)
    assert not ok
    assert (
        "no_supported_placement_action"
        in writer.last_trace["placement_candidate_decisions"][-1]["reasons"]
    )


def test_wrong_numeric_alignment_metric_becomes_exact_critic_advice():
    evidence, episodes = source()
    c = candidate(episodes)
    step = c.method_steps[1]
    e = episodes[1]
    step.critic_refs = [critic_catalog(evidence)[0].evidence_id]
    step.relations = [
        PlacementRelationCandidate(
            episode_id=e.episode_id,
            subject_id=e.subject["object_id"],
            anchor_id=e.anchor["object_id"],
            metric="aabb_separation_m",
        )
    ]
    ok, content, _ = bind(evidence, c)
    assert ok
    exp = SuccessCase.model_validate(content).placement_experience
    assert exp.method_steps[1].episode_ids == []
    assert exp.method_steps[1].evidence_kinds == ["critic_advice"]
    assert exp.critic_advice[0].quote == critic_catalog(evidence)[0].quote
    assert methods_valid(exp)


def test_generic_endpoints_and_fixture_prefix_are_not_pair_evidence():
    from scenesmith.scene_expert.memory.placement_scope import (
        aliases,
        pair_scope_matches,
    )

    evidence, episodes = source()
    assert not pair_scope_matches(
        "Place it in the center of the functional zone.", episodes[1]
    )
    assert "desk" not in aliases(
        {"name": "desk_pendant_0", "object_type": "ceiling_mounted"}
    )
    from tests.unit.test_placement_relation_binding import ceiling_source

    _, lights = ceiling_source()
    assert not pair_scope_matches("Place the fixture above the desk.", lights[2])


def test_stage_pass_is_not_changed_by_local_pass_support():
    _, episodes = source()
    assert success_scope(episodes[0], episodes) == "stage"
    assert episodes[0].stage_passed is True


def test_projection_preserves_native_relation_identity():
    evidence, _, _ = local_source()
    import json

    text, _ = build_writer_prompt(
        evidence=evidence,
        final_report={},
        trace_summary="",
        related_old_memory="",
        max_user_bytes=30000,
    )
    assert '"relation":"against_wall"' in text
    assert '"relation":"next_to"' in text
