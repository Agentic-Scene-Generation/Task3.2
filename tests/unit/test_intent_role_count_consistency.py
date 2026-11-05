"""Keep native role counts separate without weakening same-selector checks."""

from copy import deepcopy

import pytest

from pydantic import ValidationError

from scenesmith.scenebenchmark_critic.intent_schema import validate_intent_contract
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.extensions.intent_contract import (
    _evaluate_required_count,
)


def required(count, role="", quantifier="at_least"):
    return {
        "constraint_id": f"count_{role or 'all'}_{count}",
        "relation": "required_count",
        "subjects": {
            "category": "cutlery",
            "role": role,
            "count": count,
            "quantifier": quantifier,
        },
        "strength": "hard",
        "source": "explicit_prompt",
        "evidence_span": "Eight settings, each with a fork and a knife.",
        "stage": "manipuland",
    }


def contract(rows):
    return {
        "schema_version": "scenesmith.intent_contract.v8",
        "prompt": "Eight settings, each with a fork and a knife.",
        "constraints": rows,
    }


def test_aggregate_and_distinct_native_role_counts_are_preserved():
    rows = [required(16), required(8, "fork"), required(8, "knife")]
    before = deepcopy(rows)
    result = validate_intent_contract(contract(rows), validate_prompt_semantics=False)
    assert [
        (r["subjects"]["role"], r["subjects"]["count"]) for r in result["constraints"]
    ] == [
        ("", 16),
        ("fork", 8),
        ("knife", 8),
    ]
    assert rows == before


@pytest.mark.parametrize("roles", [("", ""), ("fork", "fork"), ("Fork", "fork")])
def test_conflicting_counts_for_the_same_normalized_selector_still_fail(roles):
    with pytest.raises(ValidationError, match="conflicting required_count"):
        validate_intent_contract(
            contract([required(8, roles[0]), required(7, roles[1])]),
            validate_prompt_semantics=False,
        )


def object_rows(forks, knives, generic=0):
    return [
        {
            "id": f"{kind}_{i}",
            "name": kind,
            "category": "cutlery",
            "category_norm": "cutlery",
            "object_type": "manipuland",
            "metadata": {"semantic_name": kind},
        }
        for kind, count in [("fork", forks), ("knife", knives), ("cutlery", generic)]
        for i in range(count)
    ]


@pytest.mark.parametrize(
    "forks,knives,generic,expected",
    [
        (8, 8, 0, ("pass", "pass")),
        (16, 0, 0, ("pass", "fail")),
        (0, 16, 0, ("fail", "pass")),
        (0, 0, 16, ("fail", "fail")),
    ],
)
def test_native_count_evaluation_keeps_forks_and_knives_distinct(
    forks, knives, generic, expected
):
    rows = [required(16), required(8, "fork"), required(8, "knife")]
    validated = validate_intent_contract(
        contract(rows), validate_prompt_semantics=False
    )
    objects = object_rows(forks, knives, generic)
    leaf_results = [
        _evaluate_required_count(row, objects, "core")
        for row in validated["constraints"]
        if row["subjects"].get("role")
    ]
    assert tuple(r["label"] for r in leaf_results) == expected


@pytest.mark.parametrize("same_role", [False, True])
def test_edge_distribution_compares_only_the_matching_role_inventory(same_role):
    inventory = required(4, "guest" if same_role else "teacher", "exactly")
    inventory["subjects"]["category"] = "chair"
    inventory["stage"] = "furniture"
    edge = {
        "relation": "edge_distribution",
        "subjects": {"category": "chair", "role": "guest", "count": 5},
        "targets": {"category": "table", "count": 1},
        "edge_frame": "target_local_rectangle",
        "groups": [
            {"edge_class": "long", "counts_per_edge": [2, 2]},
            {"edge_class": "short", "counts_per_edge": [1, 0]},
        ],
        "orientation": "toward_target",
        "source": "explicit_prompt",
        "evidence_span": "five guest chairs around a table",
    }
    value = contract([inventory, edge])
    if same_role:
        with pytest.raises(ValidationError, match="conflicts with exact"):
            validate_intent_contract(value, validate_prompt_semantics=False)
    else:
        validate_intent_contract(value, validate_prompt_semantics=False)
