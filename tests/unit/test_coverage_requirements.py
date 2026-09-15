"""Focused coverage-audit tests for explicit critic requirements."""

from scenesmith.scenebenchmark_critic.intent_contract import (
    build_intent_contract,
)
from scenesmith.scenebenchmark_critic.intent_schema import (
    INTENT_CONTRACT_SCHEMA_VERSION,
    validate_intent_contract,
)
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.extensions.intent_contract import (
    evaluate_intent_contract_extensions,
)
from scenesmith.scenebenchmark_critic.reports import format_markdown_report


def _coverage_case(requirement, *, stage="final", geometry=None):
    return {
        "stage": stage,
        "scene_geometry": geometry if geometry is not None else {"rooms": []},
        "intent_contract": {
            "schema_version": INTENT_CONTRACT_SCHEMA_VERSION,
            "constraints": [],
            "coverage_requirements": [requirement],
        },
    }


def test_v5_payload_migrates_with_empty_coverage_list() -> None:
    result = validate_intent_contract(
        {
            "schema_version": "scenesmith.intent_contract.v5",
            "prompt": "A bedroom with a bed.",
            "constraints": [],
        }
    )

    assert result["schema_version"] == INTENT_CONTRACT_SCHEMA_VERSION
    assert result["coverage_requirements"] == []
    assert result["coverage_ledger"] == []


def test_unresolved_and_soft_scope_coverage_are_visible_without_passing() -> None:
    unresolved = {
        "requirement_id": "coverage_unresolved",
        "kind": "unresolved",
        "disposition": "unresolved",
        "normalized": "ambiguous_chair_reference",
        "earliest_stage": "furniture",
        "final_stage": "final",
        "source": "explicit_prompt",
        "evidence_span": "place it by the chair",
    }
    soft_scope = {
        "requirement_id": "coverage_style",
        "kind": "soft_scope",
        "disposition": "soft_scope",
        "normalized": "cozy_atmosphere",
        "earliest_stage": "floor_plan",
        "final_stage": "final",
        "source": "explicit_prompt",
        "evidence_span": "make it cozy",
    }
    case = {
        **_coverage_case(unresolved, stage="final"),
        "intent_contract": {
            "schema_version": INTENT_CONTRACT_SCHEMA_VERSION,
            "constraints": [],
            "coverage_requirements": [unresolved, soft_scope],
        },
    }

    results = evaluate_intent_contract_extensions(case)

    assert [(row["label"], row["scoring_tier"]) for row in results] == [
        ("unknown", "core"),
        ("unknown", "auxiliary"),
    ]
    assert results[0]["diagnostics"]["coverage_status"] == "unresolved"


def test_walk_in_closet_is_compiled_as_functional_zone_coverage() -> None:
    contract = build_intent_contract(
        "A bedroom with a real walk-in closet and three wardrobes."
    )

    rows = contract["coverage_requirements"]
    assert len(rows) == 1
    assert rows[0]["kind"] == "functional_zone"
    assert rows[0]["normalized"] == "walk_in_closet"
    assert rows[0]["evidence_span"] == "walk-in closet"


def test_missing_walk_in_closet_is_core_failure() -> None:
    requirement = {
        "requirement_id": "coverage_closet",
        "kind": "functional_zone",
        "normalized": "walk_in_closet",
        "earliest_stage": "floor_plan",
        "final_stage": "final",
        "source": "explicit_prompt",
        "evidence_span": "walk-in closet",
    }
    results = evaluate_intent_contract_extensions(
        _coverage_case(
            requirement,
            geometry={"rooms": [{"id": "bedroom_0", "room_type": "bedroom"}]},
        )
    )

    assert [(row["relation_type"], row["label"]) for row in results] == [
        ("coverage", "fail")
    ]
    assert results[0]["scoring_tier"] == "core"
    assert results[0]["diagnostics"]["evidence_span"] == "walk-in closet"


def test_walk_in_closet_room_is_structural_pass() -> None:
    requirement = {
        "requirement_id": "coverage_closet",
        "kind": "functional_zone",
        "normalized": "walk_in_closet",
        "earliest_stage": "floor_plan",
        "final_stage": "final",
        "source": "explicit_prompt",
        "evidence_span": "walk-in closet",
    }
    case = _coverage_case(
        requirement,
        geometry={"rooms": [{"id": "closet_0", "room_type": "walk_in_closet"}]},
    )

    results = evaluate_intent_contract_extensions(case)

    assert results[0]["label"] == "pass"
    assert results[0]["diagnostics"]["matched_room_ids"] == ["closet_0"]


def test_unsupported_containment_is_visible_as_degraded_core_coverage() -> None:
    contract = build_intent_contract(
        "A playroom with a toy chest overflowing with colorful building blocks."
    )
    rows = [
        row
        for row in contract["coverage_requirements"]
        if row["kind"] == "unsupported_relation"
    ]
    assert rows
    assert rows[0]["evidence_span"] == (
        "a toy chest overflowing with colorful building blocks"
    )

    results = evaluate_intent_contract_extensions(
        _coverage_case(rows[0], stage="manipuland", geometry={"rooms": []})
    )

    assert results[0]["label"] == "degraded"
    assert results[0]["scoring_tier"] == "core"
    assert results[0]["diagnostics"]["coverage_status"] == "degraded"


def test_coverage_without_structural_evidence_is_unknown() -> None:
    requirement = {
        "requirement_id": "coverage_closet",
        "kind": "functional_zone",
        "normalized": "walk_in_closet",
        "earliest_stage": "floor_plan",
        "final_stage": "final",
        "source": "explicit_prompt",
        "evidence_span": "walk-in closet",
    }
    results = evaluate_intent_contract_extensions(
        _coverage_case(
            requirement,
            geometry={"objects": [{"id": "wardrobe_0", "category": "wardrobe"}]},
        )
    )

    assert results[0]["label"] == "unknown"
    assert results[0]["scoring_tier"] == "auxiliary"


def test_future_coverage_requirement_is_pending_not_failed() -> None:
    requirement = {
        "requirement_id": "coverage_relation",
        "kind": "functional_zone",
        "normalized": "walk_in_closet",
        "earliest_stage": "manipuland",
        "final_stage": "final",
        "source": "explicit_prompt",
        "evidence_span": "walk-in closet",
    }
    results = evaluate_intent_contract_extensions(
        _coverage_case(requirement, stage="furniture", geometry={"rooms": []})
    )

    assert results[0]["label"] == "unknown"
    assert results[0]["contract_state"] == "pending"
    assert results[0]["scoring_tier"] == "auxiliary"


def test_report_keeps_coverage_status_and_prompt_evidence_visible() -> None:
    requirement = {
        "requirement_id": "coverage_closet",
        "kind": "functional_zone",
        "normalized": "walk_in_closet",
        "earliest_stage": "floor_plan",
        "final_stage": "final",
        "source": "explicit_prompt",
        "evidence_span": "walk-in closet",
    }
    case = _coverage_case(
        requirement,
        geometry={"rooms": [{"id": "bedroom_0", "room_type": "bedroom"}]},
    )
    results = evaluate_intent_contract_extensions(case)
    payload = {
        "case_pack": case,
        "results": results,
        "summary": {},
        "gate": {"label": "report_only"},
        "scope": "unit",
        "stage": "final",
    }

    report = format_markdown_report(payload)

    assert "## Coverage Requirements" in report
    assert "coverage_closet" in report
    assert "walk-in closet" in report
    assert "status=`fail`" in report
