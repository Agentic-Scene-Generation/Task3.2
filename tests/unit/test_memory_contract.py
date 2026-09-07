"""CPU-only regressions for memory identity, spatial evidence and legacy reads."""

from __future__ import annotations

import json

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from scenesmith.scene_expert.memory.audit_contract import audit_memory_contract
from scenesmith.scene_expert.memory.contracts import selection_from_record
from scenesmith.scene_expert.memory.evidence import (
    constraint_observation,
    resolve_constraint_evidence,
    writer_prompt_evidence,
)
from scenesmith.scene_expert.memory.injection import build_memory_injection_bundle
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
from scenesmith.scene_expert.memory.text_builder import build_embedding_text
from scenesmith.scene_expert.memory.writer import MemoryWriter
from scenesmith.scene_expert.schemas import (
    MemoryPack,
    RetrievedMemorySelection,
    SceneTaskSpec,
    StageBrief,
)


def _constraint() -> dict:
    return {
        "constraint_id": "c1",
        "relation": "edge_distribution",
        "stage": "furniture",
        "subjects": {"role": "chair", "count": 4},
        "targets": {"role": "table"},
        "orientation": "inward",
        "edge_frame": "table_local",
        "count": 4,
        "groups": [{"edge": "long", "count": 2}],
    }


def _evidence(*labels: str, stage_pass: bool = True) -> dict:
    constraint = _constraint()
    rows = [
        constraint_observation(
            {
                "check_id": f"check_{i}",
                "label": label,
                "scoring_tier": "core",
                "contract_state": "evaluated",
                "evidence": {"intent_constraint": constraint},
                "diagnostics": {"yaw_error_deg": 0},
                "primary_object": f"chair_{i}",
            },
            constraint,
            "furniture",
        )
        for i, label in enumerate(labels)
    ]
    return {
        "stage": "furniture",
        "scene_state_path": "/run/final_furniture",
        "relation_context": {"hard_constraints": [constraint]},
        "verify_report": {
            "stage": "furniture",
            "pass_stage": stage_pass,
            "hard_check_report": {"constraint_evidence": rows},
        },
    }


def _relation(*labels: str, stage_pass: bool = True) -> SpatialRelationMemory:
    writer = MemoryWriter(model="unused", llm_client=SimpleNamespace())
    return writer._spatial_relations(_evidence(*labels, stage_pass=stage_pass))[0]


def _success(
    memory_id: str, text: str = "Anchor the table first.", *, layout: bool = False
) -> SuccessCase:
    return SuccessCase(
        case_id=memory_id,
        room_type="classroom",
        stage="furniture",
        positive_guidance=[text],
        source_task_id=f"source_{memory_id}",
        spatial_relations=[_relation("pass")] if layout else [],
    )


def _retrieve(store: FastMemoryStore, pack: MemoryPack, **policy) -> MemoryPack:
    return BudgetedMemoryRetriever(
        SimpleNamespace(retrieve=lambda *args, **kwargs: pack),
        store=store,
        policy=MemoryInjectionPolicy(**policy),
    ).retrieve(
        SceneTaskSpec(
            room_type="classroom",
            style="modern",
            required_large_objects=["desk", "chair", "table"],
        ),
        "furniture",
    )


def test_equal_text_from_distinct_sources_survives_until_admission(
    tmp_path: Path,
) -> None:
    store = FastMemoryStore(str(tmp_path / "bank"))
    failures = [
        FailureCase(
            failure_id=name,
            room_type="classroom",
            stage="furniture",
            object="desk",
            bad_pattern="SHARED" if name != "c" else "WRONG_CONTENT",
            failure_type=f"failure_{name}",
            repair_verified=name != "a",
        )
        for name in ("a", "b", "c")
    ]
    for record in failures:
        store.add_failure_case(record)
    rows = [
        selection_from_record(record, rank=i + 1, memory_dir=store.memory_dir)
        for i, record in enumerate(store.active_failure_cases)
    ]
    pack = MemoryPack(
        failure_case_ids=["a", "b", "c"],
        failure_hints=[r.injected_text for r in rows],
        selections=rows,
    )
    assert len(pack.deduplicated().failure_hints) == 3
    result = _retrieve(store, pack)
    assert result.failure_case_ids == ["b"]
    assert "SHARED" in result.failure_hints[0]
    assert "WRONG_CONTENT" not in result.failure_hints[0]
    assert result.failure_hints[0] == result.selections[0].injected_text


def test_pruned_record_cannot_donate_layout_or_budget(tmp_path: Path) -> None:
    store = FastMemoryStore(str(tmp_path / "bank"))
    a, b = _success("a"), _success("b", layout=True)
    store.add_success_case(a)
    store.add_success_case(b)
    pack = MemoryPack(
        success_case_ids=["a", "b"],
        success_hints=["wrong_a", "wrong_b"],
        placement_reference=b.to_placement_text(),
    )
    result = _retrieve(store, pack, max_total_chars=len(a.to_positive_guidance()))
    assert result.success_case_ids == ["a"]
    assert result.success_hints == [a.to_positive_guidance()]
    assert result.placement_reference == ""
    assert result.placement_memory_ids == []
    assert result.selection_policy["selected_chars"] == len(a.to_positive_guidance())


def test_dedup_keeps_signed_spatial_values_and_original_retrieval_order(
    tmp_path: Path,
) -> None:
    store = FastMemoryStore(str(tmp_path / "bank"))
    store.add_success_case(_success("a", "Local offset = -1"))
    store.add_success_case(_success("b", "Local offset = +1"))
    rows = [
        selection_from_record(r, rank=i + 1, memory_dir=store.memory_dir)
        for i, r in enumerate(store.active_success_cases)
    ]
    pack = MemoryPack(selections=rows).deduplicated()
    assert pack.success_case_ids == ["a", "b"]
    assert _retrieve(store, pack, max_success_cases=2).success_case_ids == ["a", "b"]


def test_selected_record_owns_layout_and_source_hash(tmp_path: Path) -> None:
    store = FastMemoryStore(str(tmp_path / "bank"))
    record = _success("a", layout=True)
    store.add_success_case(record)
    result = _retrieve(
        store,
        MemoryPack(
            success_case_ids=["a"],
            success_hints=["wrong"],
            placement_reference="UNBOUND",
        ),
    )
    assert result.placement_reference == record.to_placement_text()
    assert result.placement_memory_ids == ["a"]
    assert result.selections[0].content_hash
    assert result.retrieved_source_task_ids == {"a": ["source_a"]}
    assert result.deduplicated() == result


def test_changed_content_is_rejected_instead_of_rebinding_old_rank(
    tmp_path: Path,
) -> None:
    store = FastMemoryStore(str(tmp_path / "bank"))
    record = _success("a")
    row = selection_from_record(record, rank=1, memory_dir=store.memory_dir)
    store.add_success_case(
        record.model_copy(update={"positive_guidance": ["Changed action"]})
    )
    result = _retrieve(store, MemoryPack(success_case_ids=["a"], selections=[row]))
    assert result.success_case_ids == []
    assert "source_content_changed" in result.selection_decisions[0].reasons


def test_cross_bank_same_id_keeps_typed_content_and_provenance(tmp_path: Path) -> None:
    store = FastMemoryStore(str(tmp_path / "bank"))
    store.add_success_case(_success("same"))
    store.add_failure_case(
        FailureCase(
            failure_id="same",
            room_type="classroom",
            stage="furniture",
            object="desk",
            bad_pattern="Keep exit open",
            repair_verified=True,
            source_task_id="failed_task",
        )
    )
    result = _retrieve(
        store, MemoryPack(success_case_ids=["same"], failure_case_ids=["same"])
    )
    assert len(result.selections) == 2
    by_type = {r.memory_type: r for r in result.selections}
    assert by_type["success"].source_task_ids == ["source_same"]
    assert by_type["failure"].source_task_ids == ["failed_task"]
    assert "Anchor" in result.success_hints[0]
    assert "exit" in result.failure_hints[0]
    assert "same" not in result.retrieved_source_task_ids


def test_ambiguous_legacy_parallel_payload_never_guesses_alignment() -> None:
    pack = MemoryPack(
        success_case_ids=["a", "b", "c"],
        success_hints=["ab_shared", "c"],
        placement_reference="unbound",
    )
    repaired = pack.deduplicated()
    assert repaired.success_case_ids == ["a", "b", "c"]
    assert repaired.success_hints == ["", "", ""]
    assert repaired.placement_reference == ""


def test_long_failure_keeps_action_check_and_preconditions(tmp_path: Path) -> None:
    store = FastMemoryStore(str(tmp_path / "bank"))
    record = FailureCase(
        failure_id="f",
        room_type="classroom",
        stage="furniture",
        object="desk",
        bad_pattern="Avoid historical misplaced desk. " * 20,
        repair_action="ACTION_MARKER: orient the desk toward the teaching wall.",
        critic_check="CHECK_MARKER: verify actual forward direction.",
        repair_verified=True,
    )
    store.add_failure_case(record)
    pack = _retrieve(store, MemoryPack(failure_case_ids=["f"]))
    bundle = build_memory_injection_bundle(
        stage="furniture",
        stage_brief=StageBrief(stage="furniture", stage_objective="Design"),
        memory_pack=pack,
    )
    assert "ACTION_MARKER" in bundle.final_text and "CHECK_MARKER" in bundle.final_text
    assert "..." not in bundle.final_text
    assert (
        _retrieve(
            store, MemoryPack(failure_case_ids=["f"]), max_total_chars=300
        ).failure_case_ids
        == []
    )


def test_relation_parameters_are_lossless_in_prompt_and_embedding() -> None:
    a = _relation("pass")
    b = a.model_copy(
        update={
            "cardinality": {"count": 2, "orientation": "outward", "edge_frame": "world"}
        }
    )
    text = a.to_guidance_text()
    for marker in (
        '"count":4',
        '"orientation":"inward"',
        '"edge_frame":"table_local"',
        '"subject_count":4',
        '"groups"',
    ):
        assert marker in text
    assert text != b.to_guidance_text()
    assert not b.has_verified_geometry
    assert build_embedding_text(
        _success("s").model_copy(update={"spatial_relations": [a]})
    ) != build_embedding_text(
        _success("s").model_copy(update={"spatial_relations": [b]})
    )


@pytest.mark.parametrize("stage_pass", [True, False])
def test_stage_pass_alone_never_verifies_spatial_requirements(stage_pass: bool) -> None:
    relation = _relation(stage_pass=stage_pass)
    assert relation.verification_status == "requirement_only"
    assert not relation.geometry_verified and not relation.has_verified_geometry
    assert relation.evidence_source == "task_contract"


def test_exact_constraint_pass_survives_unrelated_stage_failure() -> None:
    relation = _relation("pass", "pass", stage_pass=False)
    assert relation.has_verified_geometry and relation.geometry_verified
    assert relation.verification_status == "verified_pass"
    assert len(relation.verification_evidence) == 2
    assert relation.verification_evidence[0].observations["primary_object"] == "chair_0"
    assert relation.verification_evidence[0].scene_state_path == "/run/final_furniture"


@pytest.mark.parametrize(
    "labels,status",
    [
        (("pass", "fail"), "verified_fail"),
        (("pass", "unknown"), "inconclusive"),
        (("pass", "degraded"), "inconclusive"),
    ],
)
def test_partial_or_failed_relation_is_not_a_verified_solution(labels, status) -> None:
    relation = _relation(*labels)
    assert relation.verification_status == status
    assert not relation.has_verified_geometry and not relation.geometry_verified


@pytest.mark.parametrize(
    "field,value",
    [
        ("stage", "manipuland"),
        ("constraint_hash", "other"),
        ("result_hash", ""),
        ("check_id", ""),
        ("label", "unknown"),
        ("evaluation_state", "deferred"),
        ("scoring_tier", "auxiliary"),
    ],
)
def test_wrong_or_uncertain_evidence_cannot_certify_geometry(field, value) -> None:
    evidence = _evidence("pass")
    evidence["verify_report"]["hard_check_report"]["constraint_evidence"][0][
        field
    ] = value
    assert resolve_constraint_evidence(_constraint(), evidence)[0] != "verified_pass"


def test_changed_constraint_with_reused_id_does_not_reuse_old_pass() -> None:
    evidence = _evidence("pass")
    changed = deepcopy(_constraint())
    changed["orientation"] = "outward"
    status, _ = resolve_constraint_evidence(changed, evidence)
    assert status == "inconclusive"


def test_changed_failed_claim_cannot_reuse_old_failure_label() -> None:
    relation = _relation("fail")
    assert relation.has_verified_failure
    assert "evidence=verified_fail" in relation.to_guidance_text()
    changed = relation.model_copy(update={"cardinality": {"orientation": "outward"}})
    assert not changed.has_verified_failure
    assert "evidence=verified_fail" not in changed.to_guidance_text()


def test_legacy_geometry_flag_is_audited_not_trusted_or_rewritten(
    tmp_path: Path,
) -> None:
    bank = tmp_path / "legacy"
    bank.mkdir()
    payload = {
        "case_id": "old",
        "room_type": "classroom",
        "stage": "furniture",
        "placement_reference": ["old_world_coordinate"],
        "positive_guidance": ["Keep an approach aisle"],
        "spatial_relations": [
            {
                "relation_type": "facing",
                "geometry_verified": True,
                "evidence_source": "critic",
            }
        ],
    }
    path = bank / "success_cases.jsonl"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    before = path.read_bytes()
    first = audit_memory_contract(bank)
    assert audit_memory_contract(bank) == first
    assert list(bank.iterdir()) == [path]
    assert path.read_bytes() == before
    row = first["records"][0]
    assert row["spatial_status"][0]["stored_geometry_verified"] is True
    assert row["spatial_status"][0]["verified_geometry_eligible"] is False
    assert row["payload"]["placement_text"] == ""
    assert "Keep an approach aisle" in row["payload"]["injected_text"]
    assert "legacy_world_layout_omitted" in row["warnings"]


def test_unverified_repair_is_not_presented_as_verified_fix() -> None:
    record = FailureCase(
        failure_id="f",
        stage="furniture",
        room_type="classroom",
        repair_action="Try rotating it",
    )
    assert "Unverified suggestion" in record.to_negative_constraint()
    assert "verified fix" not in record.to_hint_text()


def test_legacy_v2_round_trip_and_typed_evidence_survive_store(tmp_path: Path) -> None:
    store = FastMemoryStore(str(tmp_path / "bank"))
    old = _success("old").model_copy(update={"schema_version": "sceneexpert.memory.v2"})
    store.add_success_case(old)
    new = _success("new", layout=True)
    store.add_success_case(new)
    loaded = FastMemoryStore(str(store.memory_dir), read_only=True)
    assert len(loaded.active_success_cases) == 2
    assert loaded.active_success_cases[1].spatial_relations[0].has_verified_geometry
    assert loaded.active_success_cases[1].spatial_relations[0].verification_evidence


def test_native_outcomes_are_compact_for_writer_but_full_evidence_is_preserved() -> (
    None
):
    original = {"stages": [_evidence("pass", "unknown")]}
    before = deepcopy(original)
    projected = writer_prompt_evidence(original)
    rows = projected["stages"][0]["verify_report"]["hard_check_report"][
        "constraint_evidence"
    ]
    assert [r["label"] for r in rows] == ["pass", "unknown"]
    assert rows[0]["primary_object"] == "chair_0"
    assert "observations" not in rows[0]
    assert original == before


def test_semantically_different_successes_do_not_merge_spatial_claims(
    tmp_path: Path,
) -> None:
    store = FastMemoryStore(str(tmp_path / "bank"))
    a = _success("a", layout=True)
    b = _success("b", layout=True)
    b.spatial_relations[0].cardinality["orientation"] = "outward"
    store.add_success_case(a)
    store.add_success_case(b)
    assert len(store.active_success_cases) == 2
    assert len(store.active_success_cases[0].spatial_relations) == 1


def test_ambiguous_identity_is_rejected_not_last_record_wins(tmp_path: Path) -> None:
    store = FastMemoryStore(str(tmp_path / "bank"))
    # Older JSONL files can have two distinct Skills sharing a descriptive name.
    store.skills = [
        Skill(skill_name="same", stage="furniture", procedure=["first"]),
        Skill(skill_name="same", stage="furniture", procedure=["second"]),
    ]
    result = _retrieve(store, MemoryPack(skill_names=["same"]))
    assert result.skill_names == []
    assert "ambiguous_record_identity" in result.selection_decisions[0].reasons


def test_contract_audit_exposes_malformed_and_duplicate_records(tmp_path: Path) -> None:
    bank = tmp_path / "bank"
    bank.mkdir()
    record = _success("same").model_dump_json()
    (bank / "success_cases.jsonl").write_text(
        record + "\n" + record + "\nnot json\n", encoding="utf-8"
    )
    audit = audit_memory_contract(bank)
    assert audit["record_count"] == 3
    assert not any(r["reader_eligible"] for r in audit["records"])
    assert "malformed_record" in audit["records"][-1]["warnings"]


def test_index_format_upgrade_preserves_frozen_bank_and_rebuilds_semantic_text(
    tmp_path: Path,
) -> None:
    from unittest.mock import patch

    import numpy as np

    from scenesmith.scene_expert.memory.hybrid_retriever import HybridMemoryRetriever
    from scenesmith.scene_expert.memory.text_builder import EMBEDDING_TEXT_VERSION

    store = FastMemoryStore(str(tmp_path / "bank"))
    store.add_success_case(
        _success("s", layout=True).model_copy(
            update={"embedding_text": "STALE_EMBEDDING_TEXT"}
        )
    )
    frozen = FastMemoryStore(str(store.memory_dir), read_only=True)
    before = {p.name: p.read_bytes() for p in store.memory_dir.iterdir() if p.is_file()}
    seen_texts = []

    def encode(texts):
        seen_texts.extend(texts)
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)

    with patch(
        "scenesmith.scene_expert.memory.hybrid_retriever.tempfile.gettempdir",
        return_value=str(tmp_path),
    ):
        retriever = HybridMemoryRetriever(
            frozen,
            str(store.memory_dir),
            SimpleNamespace(encode=encode),
            auto_build_indexes=True,
        )
        pack = retriever.retrieve(
            SceneTaskSpec(
                room_type="classroom",
                style="modern",
                required_large_objects=["chair", "table"],
            ),
            "furniture",
        )
    assert pack.success_case_ids == ["s"]
    assert store.memory_dir not in retriever._index_dir.parents
    assert not (store.memory_dir / "indexes").exists()
    assert {
        p.name: p.read_bytes() for p in store.memory_dir.iterdir() if p.is_file()
    } == before
    assert not any("STALE_EMBEDDING_TEXT" in text for text in seen_texts)
    assert any('"edge_frame":"table_local"' in text for text in seen_texts)
    index = retriever._load_index("success", "furniture")
    assert index.manifest["embedding_text_version"] == EMBEDDING_TEXT_VERSION
    index.manifest.pop("embedding_text_version")
    assert not retriever._index_matches_records(index, frozen.active_success_cases)
