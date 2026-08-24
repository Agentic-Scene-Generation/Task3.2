from scenesmith.scenebenchmark_critic.aggregation import aggregate_results
from scenesmith.scenebenchmark_critic.result_identity import (
    deduplicate_checks,
    deduplicate_results,
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
