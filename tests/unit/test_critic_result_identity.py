from copy import deepcopy

import pytest

from scenesmith.scenebenchmark_critic.aggregation import aggregate_results
from scenesmith.scenebenchmark_critic.result_identity import (
    deduplicate_checks,
    deduplicate_results,
    semantic_signature,
)


def test_semantic_duplicate_checks_keep_producer_provenance() -> None:
    checks = deduplicate_checks(
        [
            {
                "check_id": "intent_a",
                "metric": "functional_dependency",
                "subject_id": "chair_0",
                "target_ids": ["table_0"],
                "relation_type": "seating_to_work_surface",
                "scoring_tier": "core",
            },
            {
                "check_id": "metadata_b",
                "metric": "functional_dependency",
                "subject_id": "chair_0",
                "target_ids": ["table_0"],
                "relation_type": "seating_to_work_surface",
                "scoring_tier": "core",
            },
        ]
    )

    assert len(checks) == 1
    assert checks[0]["producer_check_ids"] == ["intent_a", "metadata_b"]
    assert checks[0]["canonical_check_id"].startswith("semantic_")


def test_semantic_duplicate_results_count_once_and_keep_worst_label() -> None:
    results = deduplicate_results(
        [
            {
                "check_id": "intent_a",
                "metric": "functional_dependency",
                "relation_type": "object_on_support",
                "primary_object": "book_0",
                "related_objects": ["shelf_0"],
                "label": "degraded",
                "scoring_tier": "core",
                "evidence": {"intent_constraint": {"constraint_id": "constraint_a"}},
            },
            {
                "check_id": "extension_b",
                "metric": "functional_dependency",
                "relation_type": "object_on_support",
                "primary_object": "book_0",
                "related_objects": ["shelf_0"],
                "label": "fail",
                "scoring_tier": "core",
                "evidence": {"intent_constraint": {"constraint_id": "constraint_b"}},
            },
        ]
    )

    assert len(results) == 1
    assert results[0]["label"] == "fail"
    assert results[0]["diagnostics"]["label_disagreements"] == [
        "degraded",
        "fail",
    ]
    assert [
        row["constraint_id"] for row in results[0]["evidence"]["intent_constraints"]
    ] == ["constraint_a", "constraint_b"]
    assert aggregate_results(results)["scene_summary"]["total_checks"] == 1


def test_semantic_identity_keeps_different_thresholds_separate() -> None:
    results = deduplicate_results(
        [
            {
                "check_id": "near_a",
                "metric": "functional_dependency",
                "relation_type": "generic_near_relation",
                "primary_object": "plant_0",
                "related_objects": ["sofa_0"],
                "label": "pass",
                "diagnostics": {"max_gap_m": 1.0},
                "scoring_tier": "core",
            },
            {
                "check_id": "near_b",
                "metric": "functional_dependency",
                "relation_type": "generic_near_relation",
                "primary_object": "plant_0",
                "related_objects": ["sofa_0"],
                "label": "pass",
                "diagnostics": {"max_gap_m": 1.5},
                "scoring_tier": "core",
            },
        ]
    )

    assert len(results) == 2


def test_transitive_merge_keeps_all_producer_and_constraint_provenance() -> None:
    results = deduplicate_results(
        [
            {
                "check_id": "producer_a",
                "metric": "functional_dependency",
                "relation_type": "object_on_support",
                "primary_object": "book_0",
                "related_objects": ["shelf_0"],
                "label": "fail",
                "evidence": {"intent_constraint": {"constraint_id": "constraint_a"}},
            },
            {
                "check_id": "producer_b",
                "producer_check_ids": ["producer_b", "producer_c"],
                "metric": "functional_dependency",
                "relation_type": "object_on_support",
                "primary_object": "book_0",
                "related_objects": ["shelf_0"],
                "label": "fail",
                "evidence": {"intent_constraint": {"constraint_id": "constraint_b"}},
            },
        ]
    )

    assert len(results) == 1
    assert results[0]["producer_check_ids"] == [
        "producer_a",
        "producer_b",
        "producer_c",
    ]
    assert results[0]["source_constraint_ids"] == [
        "constraint_a",
        "constraint_b",
    ]
    assert [
        row["constraint_id"] for row in results[0]["evidence"]["intent_constraints"]
    ] == ["constraint_a", "constraint_b"]


def test_asymmetric_relation_signature_preserves_endpoint_ownership() -> None:
    book_on_shelf = {
        "metric": "functional_dependency",
        "relation_type": "object_on_support",
        "primary_object": "book_0",
        "related_objects": ["bookshelf_0"],
    }
    shelf_on_book = {
        "metric": "functional_dependency",
        "relation_type": "object_on_support",
        "primary_object": "bookshelf_0",
        "related_objects": ["book_0"],
    }

    assert semantic_signature(book_on_shelf) != semantic_signature(shelf_on_book)
    assert len(deduplicate_results([book_on_shelf, shelf_on_book])) == 2


def _annotation_facing(alias: str = "faces") -> dict:
    return {
        "check_id": alias,
        "metric": "functional_dependency",
        "primary_object": "desk_0",
        "related_objects": ["office_chair_0"],
        "label": "fail",
        "scoring_tier": "core",
        "evidence": {
            "annotation_source": "asset_orientation_dependency",
            "dependency_key": "orientation_dependencies",
            "dependency": {
                "relation_type": alias,
                "max_distance_m": 1.1,
                "source": "hssd_annotations:week27_asset_policy",
            },
        },
    }


@pytest.mark.parametrize("deduplicate", [deduplicate_checks, deduplicate_results])
def test_annotation_alias_duplicate_counts_once_without_mutating_evidence(
    deduplicate,
) -> None:
    first = _annotation_facing()
    second = _annotation_facing("front_faces")
    second["evidence"]["dependency"].update(
        source="hssd_annotations:relation_priors",
        distance_required=True,
        target_kind="object_category",
        confidence=0.88,
    )
    rows = [first, second]
    before = deepcopy(rows)
    merged = deduplicate(rows)

    assert rows == before
    assert len(merged) == 1
    assert merged[0]["producer_check_ids"] == ["faces", "front_faces"]
    assert len(merged[0]["evidence"]["annotation_dependencies"]) == 2
    assert aggregate_results(merged)["scene_summary"]["total_checks"] == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"max_distance_m": 1.5},
        {"distance_required": False},
        {"max_angle_deg": 45},
        {"subject_face": "back"},
        {"target_face": "front"},
    ],
)
def test_annotation_identity_preserves_distinct_facing_criteria(changes: dict) -> None:
    first = _annotation_facing()
    second = _annotation_facing("front_faces")
    second["evidence"]["dependency"].update(changes)
    assert len(deduplicate_results([first, second])) == 2
    assert len(deduplicate_checks([first, second])) == 2


@pytest.mark.parametrize(
    "changes",
    [
        {"primary_object": "chair_0"},
        {"related_objects": ["other_chair_0"]},
        {"stage": "manipuland"},
        {"cohort": "future"},
        {"authority": "intent"},
        {"scoring_tier": "auxiliary"},
    ],
)
def test_annotation_identity_preserves_scope_and_endpoints(changes: dict) -> None:
    first = _annotation_facing()
    second = _annotation_facing("front_faces")
    second.update(changes)
    assert len(deduplicate_results([first, second])) == 2


def test_annotation_identity_uses_effective_defaults_and_legacy_orientation_only() -> (
    None
):
    first = _annotation_facing()
    second = _annotation_facing("front_faces")
    second["evidence"]["dependency"].update(
        max_angle_deg=60, subject_face=" FRONT ", target_face="none"
    )
    assert len(deduplicate_results([first, second])) == 1

    first["evidence"]["dependency"].pop("max_distance_m")
    second["evidence"]["dependency"].update(distance_required=False)
    assert len(deduplicate_results([first, second])) == 1

    second["evidence"]["dependency"].update(source="explicit_rule")
    second["evidence"]["dependency"].pop("distance_required")
    assert len(deduplicate_results([first, second])) == 2


def test_alias_normalization_does_not_merge_other_annotation_or_intent_sources() -> (
    None
):
    first = _annotation_facing()
    second = _annotation_facing("front_faces")
    for row in (first, second):
        row["evidence"]["annotation_source"] = "intent_compiler"
    assert len(deduplicate_results([first, second])) == 2


def test_annotation_merge_keeps_transitive_provenance_and_worst_unknown() -> None:
    rows = [_annotation_facing(alias) for alias in ("faces", "facing", "face_to")]
    rows[0]["label"] = "pass"
    rows[1]["label"] = "unknown"
    rows[2]["label"] = "pass"
    before = deepcopy(rows)
    first_merge = deduplicate_results(rows[:2])
    merged = deduplicate_results(first_merge + rows[2:])
    assert len(merged) == 1
    assert merged[0]["label"] == "unknown"
    assert merged[0]["producer_check_ids"] == ["face_to", "faces", "facing"]
    assert len(merged[0]["evidence"]["annotation_dependencies"]) == 3
    assert rows == before


def test_facing_defaults_preserve_nonfinite_legacy_and_explicit_distance_rules() -> (
    None
):
    from scenesmith.scenebenchmark_critic.core.facing_criteria import (
        facing_dependency_criteria,
    )

    assert facing_dependency_criteria({}) == {
        "distance_required": True,
        "max_distance_m": 1.8,
        "max_angle_deg": 60.0,
        "subject_face": "front",
        "target_face": "any",
    }
    for invalid in (None, "invalid", float("nan"), float("inf"), -1):
        assert not facing_dependency_criteria(
            {"source": "hssd_annotations:legacy", "max_distance_m": invalid}
        )["distance_required"]
