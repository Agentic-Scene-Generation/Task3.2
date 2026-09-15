"""091 live-response regressions, with small independently reproducible geometry."""

from copy import deepcopy

import pytest

from scenesmith.scene_expert.memory.adaptation import validate_adaptation
from scenesmith.scene_expert.memory.advisory import advice_check_reasons, observe_advice
from scenesmith.scene_expert.memory.contracts import selection_from_record
from scenesmith.scene_expert.memory.placement import experience_text, pair_measurements
from scenesmith.scene_expert.memory.placement_methods import (
    critic_catalog,
    instruction_reasons,
    methods_valid,
)
from scenesmith.scene_expert.memory.schemas import (
    PlacementEpisode,
    PlacementMethodCandidate,
    PlacementRelationCandidate,
    SuccessCase,
    SuccessMemoryCandidate,
)
from scenesmith.scene_expert.schemas import MemoryAdaptation
from tests.unit.test_placement_experience import object_row
from tests.unit.test_placement_method_contract import bind, candidate, source


@pytest.mark.parametrize("connector", ["to ensure", "so that", "to make sure"])
def test_live_bedside_purpose_is_not_a_universal_guarantee(connector):
    evidence, episodes = source()
    c = candidate(episodes)
    c.method_steps[1].instruction = (
        f"Place nightstands on the lateral sides of the bed {connector} they flank the bed rather than being placed against the wall independently."
    )
    c.method_steps[1].critic_refs = [critic_catalog(evidence)[0].evidence_id]
    ok, content, _ = bind(evidence, c)
    assert ok
    assert len(content["placement_experience"]["method_steps"]) == 2


@pytest.mark.parametrize(
    "text",
    [
        "Always guarantee a perfect layout.",
        "This will prevent overlap in every room.",
        "Maintain a 0.5m gap.",
        "Place lamps at the optimal distance.",
    ],
)
def test_actual_guarantees_and_fixed_targets_still_rejected(text):
    assert instruction_reasons(text)


def ceiling_source():
    evidence, _ = source()
    a, b = object_row("ambient_flush_0", "ambient_flush_0", 0), object_row(
        "ambient_flush_1", "ambient_flush_1", 1.7
    )
    a["object_type"] = b["object_type"] = "ceiling_mounted"
    east, north = object_row("east_wall", "east_wall", 3), object_row(
        "north_wall", "north_wall", 4
    )
    episodes = []
    for subject, anchor in ((b, east), (b, north), (a, b)):
        e = PlacementEpisode(
            episode_id="",
            stage="ceiling_mounted",
            state_fingerprint="same-final-state",
            subject=subject,
            anchor=anchor,
            measurements=pair_measurements(subject, anchor),
            actions=[{"tool_call_id": "place", "tool_name": "place_fixture"}],
            evidence_refs=["scene.json"],
            stage_passed=True,
            stage_scores={"semantic": 0.8},
        )
        e.episode_id = e.content_hash()
        episodes.append(e)
    evidence["placement_experience_catalog"]["episodes"] = [
        e.model_dump() for e in episodes
    ]
    entry = evidence["stages"][0]
    entry["stage"] = entry["verify_report"]["stage"] = "ceiling_mounted"
    entry["verify_report"][
        "critique_summary"
    ] = "The ceiling fixtures define separate reading and sleeping zones.\n\nInter-fixture spacing: center-to-center distance is adequate in this source scene; inspect light distribution when moving fixtures."
    return evidence, episodes


def lighting_candidate(evidence, episodes, *, declared=False):
    step = PlacementMethodCandidate(
        instruction="Ensure sufficient center-to-center distance between fixtures to prevent light pool overlap and maintain distinct functional zones.",
        episode_ids=[
            e.episode_id for e in (episodes[2:] if declared else episodes[:2])
        ],
        critic_refs=[critic_catalog(evidence)[1].evidence_id],
    )
    if declared:
        e = episodes[2]
        step.relations = [
            PlacementRelationCandidate(
                episode_id=e.episode_id,
                subject_id=e.subject["object_id"],
                anchor_id=e.anchor["object_id"],
                metric="bbox_center_distance_m",
            )
        ]
    return SuccessMemoryCandidate(
        stage="ceiling_mounted",
        successful_pattern=["Lighting layout"],
        applicability=["Ceiling fixtures serve separate functional zones."],
        method_steps=[step],
    )


def test_live_wrong_wall_pairs_are_retained_only_as_stage_context(tmp_path):
    evidence, episodes = ceiling_source()
    ok, content, writer = bind(evidence, lighting_candidate(evidence, episodes))
    assert ok
    record = SuccessCase.model_validate(content)
    exp = record.placement_experience
    assert methods_valid(exp)
    assert exp.method_steps[0].episode_ids == []
    assert exp.method_steps[0].evidence_kinds == ["critic_advice"]
    assert set(exp.source_context_episode_ids) == {e.episode_id for e in episodes[:2]}
    assert record.spatial_relations == []
    assert "Stage context only" in experience_text(exp)
    assert "Source observation only" not in experience_text(exp)
    decision = writer.last_trace["placement_method_decisions"][0]["steps"][0]
    assert decision["evidence_scope"] == "critic_advice_only"
    assert len(decision["warnings"]) == 2
    selected = selection_from_record(record, rank=1, memory_dir=tmp_path)
    choice = MemoryAdaptation(
        memory_type="success",
        memory_id=selected.memory_id,
        source_content_hash=selected.content_hash,
        decision="adapted",
        source_method_step_indices=[0],
        source_relation_indices=[],
        actions=["Review light distribution for the current functional zones."],
        checks=["Ask the designer to inspect the resulting lighting."],
        advice_checks=[],
    )
    assert (
        validate_adaptation(
            selected,
            choice,
            stage="ceiling_mounted",
            task_spec=None,
            context=None,
            state={"objects": []},
            brief=None,
        )
        == []
    )
    item = {"source": selected.model_dump(), "adaptation": choice.model_dump()}
    observation = observe_advice(item, "ceiling_mounted", {})
    assert observation[0]["status"] == "unknown"
    assert observation[0]["quality_gain"] is None


def test_critic_only_response_needs_no_fabricated_pair_declaration():
    evidence, episodes = ceiling_source()
    c = lighting_candidate(evidence, episodes)
    c.method_steps[0].episode_ids = []
    ok, content, writer = bind(evidence, c)
    assert ok
    assert content["spatial_relations"] == []
    assert len(content["placement_experience"]["source_context_episode_ids"]) == 1
    assert (
        writer.last_trace["placement_candidate_decisions"][0]["method_episode_ids"]
        == []
    )


def test_plural_cross_category_endpoints_are_not_mistaken_for_peer_pairs():
    from scenesmith.scene_expert.memory.placement_scope import pair_scope_matches

    _, episodes = source()
    assert pair_scope_matches(
        "Check spacing between beds and nightstands.", episodes[1]
    )
    assert not pair_scope_matches("Check spacing between nightstands.", episodes[1])


def test_unrelated_geometry_without_a_quote_cannot_create_a_method():
    evidence, episodes = ceiling_source()
    c = lighting_candidate(evidence, episodes)
    c.method_steps[0].critic_refs = []
    ok, _, writer = bind(evidence, c)
    assert not ok
    assert (
        "no_supported_method_steps"
        in writer.last_trace["placement_candidate_decisions"][0]["reasons"]
    )


def test_exact_fixture_pair_and_metric_are_code_bound():
    evidence, episodes = ceiling_source()
    ok, content, _ = bind(
        evidence, lighting_candidate(evidence, episodes, declared=True)
    )
    assert ok
    record = SuccessCase.model_validate(content)
    exp = record.placement_experience
    binding = exp.method_steps[0].relation_bindings[0]
    assert (binding.subject_id, binding.anchor_id) == (
        "ambient_flush_0",
        "ambient_flush_1",
    )
    assert binding.metric == "bbox_center_distance_m" and binding.value == 1.7
    assert exp.episodes[0].measurements["aabb_separation_m"] == 1.2
    assert methods_valid(exp)
    assert record.spatial_relations[0].evidence_ref == episodes[2].episode_id
    binding.value = 999
    assert not methods_valid(exp)


@pytest.mark.parametrize(
    "fault", ["subject", "anchor", "metric", "episode", "duplicate"]
)
def test_declared_pair_or_metric_mismatch_is_not_relabelled_as_measured(fault):
    evidence, episodes = ceiling_source()
    c = lighting_candidate(evidence, episodes, declared=True)
    r = c.method_steps[0].relations[0]
    if fault == "subject":
        r.subject_id = "ambient_flush_1"
    elif fault == "anchor":
        r.anchor_id = "east_wall"
    elif fault == "metric":
        r.metric = "aabb_separation_m"
    elif fault == "episode":
        r.episode_id = episodes[0].episode_id
    else:
        c.method_steps[0].relations.append(r.model_copy())
    assert not bind(evidence, c)[0]


def test_old_wrong_pair_record_is_readable_but_not_injectable(tmp_path):
    evidence, episodes = ceiling_source()
    _, content, _ = bind(
        evidence, lighting_candidate(evidence, episodes, declared=True)
    )
    exp = content["placement_experience"]
    exp["episodes"] = [e.model_dump() for e in episodes[:2]]
    step = exp["method_steps"][0]
    step["episode_ids"] = [e.episode_id for e in episodes[:2]]
    step.pop("relation_bindings")
    step.pop("relation_binding_version")
    record = SuccessCase.model_validate(content)
    assert not methods_valid(record.placement_experience)
    assert (
        selection_from_record(record, rank=1, memory_dir=tmp_path).injected_text == ""
    )


def test_center_distance_cannot_be_evaluated_by_wall_gap_or_wrong_metric(tmp_path):
    evidence, episodes = ceiling_source()
    _, content, _ = bind(
        evidence, lighting_candidate(evidence, episodes, declared=True)
    )
    selected = selection_from_record(
        SuccessCase.model_validate(content), rank=1, memory_dir=tmp_path
    )
    choice = {
        "source_relation_indices": [0],
        "source_method_step_indices": [0],
        "bindings": [
            {"source_role": n} for n in ("ambient_flush_0", "ambient_flush_1")
        ],
        "advice_checks": [
            {
                "source_episode_id": episodes[2].episode_id,
                "metric": "aabb_separation_m",
                "subject_role": "ambient_flush_0",
                "anchor_role": "ambient_flush_1",
            }
        ],
    }
    assert "missing_method_metric_observation" in advice_check_reasons(
        selected.model_dump(), choice, {}
    )
    choice["advice_checks"][0]["metric"] = "bbox_center_distance_m"
    assert advice_check_reasons(selected.model_dump(), choice, {}) == []


def test_missing_applicability_has_precise_diagnostic():
    evidence, episodes = source()
    c = candidate(episodes)
    c.applicability = []
    ok, _, writer = bind(evidence, c)
    assert not ok
    assert writer.last_trace["placement_candidate_decisions"][0]["reasons"] == [
        "missing_applicability"
    ]


def test_center_measurement_observes_real_pair_without_claiming_gain(tmp_path):
    from scenesmith.scene_expert.memory.state import state_hash

    evidence, episodes = ceiling_source()
    _, content, _ = bind(
        evidence, lighting_candidate(evidence, episodes, declared=True)
    )
    selected = selection_from_record(
        SuccessCase.model_validate(content), rank=1, memory_dir=tmp_path
    )
    e = episodes[2]
    choice = {
        "source_relation_indices": [0],
        "source_method_step_indices": [0],
        "bindings": [
            {
                "source_role": o["name"],
                "current_role": o["name"],
                "object_ids": [o["object_id"]],
            }
            for o in (e.subject, e.anchor)
        ],
        "advice_checks": [
            {
                "source_episode_id": e.episode_id,
                "metric": "bbox_center_distance_m",
                "subject_role": e.subject["name"],
                "anchor_role": e.anchor["name"],
            }
        ],
    }
    before = {"objects": deepcopy([e.subject, e.anchor])}
    before["fingerprint"] = state_hash(before)
    after = {"objects": deepcopy([e.subject, e.anchor])}
    for key in ("translation", "bbox_min", "bbox_max"):
        after["objects"][1][key][0] += 0.5
    after["fingerprint"] = state_hash(after)
    result = observe_advice(
        {"source": selected.model_dump(), "adaptation": choice},
        "ceiling_mounted",
        {"post_scene_state": after, "injection": {"current_scene_state": before}},
    )[0]
    assert result["status"] == "measured"
    assert result["source_value"] == result["before_value"] == 1.7
    assert result["after_value"] == 2.2 and result["delta"] == 0.5
    assert result["quality_gain"] is None and result["causal_gain"] is None


def test_non_lighting_named_endpoint_substitution_is_not_measured():
    from scenesmith.scene_expert.memory.placement_scope import pair_scope_matches

    _, episodes = source()
    assert not pair_scope_matches("Align the bed relative to the chair.", episodes[1])
    assert pair_scope_matches("Align the nightstand relative to the bed.", episodes[1])


def test_missing_bbox_center_is_advice_not_invented_measurement():
    evidence, episodes = ceiling_source()
    e = episodes[2]
    e.anchor.pop("bbox_min")
    e.measurements = pair_measurements(e.subject, e.anchor)
    e.episode_id = e.content_hash()
    evidence["placement_experience_catalog"]["episodes"] = [
        x.model_dump() for x in episodes
    ]
    ok, content, writer = bind(
        evidence, lighting_candidate(evidence, episodes, declared=True)
    )
    assert ok
    assert content["spatial_relations"] == []
    assert content["placement_experience"]["method_steps"][0]["evidence_kinds"] == [
        "critic_advice"
    ]
    assert writer.last_trace["placement_method_decisions"][0]["steps"][0][
        "warnings"
    ] == ["unavailable_step_metric:" + e.episode_id]
