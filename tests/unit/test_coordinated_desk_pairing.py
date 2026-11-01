"""Literal desk-cohort pairing coverage without model or scene execution."""

import pytest

from scenesmith.scenebenchmark_critic.intent_contract import build_intent_contract
from scenesmith.scenebenchmark_critic.intent_schema import validate_intent_contract


@pytest.mark.parametrize(
    ("prompt", "count", "category"),
    [
        (
            (
                "A computer lab with six desks, each with a monitor, keyboard, "
                "and chair. A printer sits in the corner."
            ),
            6,
            "chair",
        ),
        (
            "An office with 4 desks, each with a monitor and an office chair.",
            4,
            "office_chair",
        ),
        (
            (
                "A lab with three desks, each with one monitor, a keyboard, "
                "and one chair."
            ),
            3,
            "chair",
        ),
        (
            "Six desks, each with a monitor and one chair. Six chairs in total.",
            6,
            "chair",
        ),
    ],
)
def test_explicit_coordinated_desk_chair_pairing(
    prompt: str, count: int, category: str
) -> None:
    contract = build_intent_contract(prompt)
    validate_intent_contract(contract)
    pairs = [row for row in contract["constraints"] if row["relation"] == "paired_with"]

    assert len(pairs) == 1
    pair = pairs[0]
    assert pair["subjects"] == {
        "category": category,
        "count": count,
        "quantifier": "all",
    }
    assert pair["targets"] == {
        "category": "desk",
        "count": count,
        "quantifier": "all",
    }
    assert pair["source"] == "explicit_prompt"
    assert pair["strength"] == "hard"
    assert pair["evidence_span"] in prompt.lower()
    assert any(
        row["relation"] == "required_count"
        and row["subjects"]["category"] == category
        and row["subjects"].get("count") == count
        and row["subjects"].get("quantifier") == "at_least"
        for row in contract["constraints"]
    )


@pytest.mark.parametrize(
    "prompt",
    [
        "A lab with desks, each with a monitor, keyboard, and chair.",
        "A lab with six desks, a monitor, keyboard, and chair.",
        "A lab with six desks, each with a monitor and two chairs.",
        "Avoid six desks, each with a monitor, keyboard, and chair.",
        "A lab with six desks. Each with a monitor, keyboard, and chair.",
        (
            "Four desks, each with a monitor and chair; "
            "six desks, each with a monitor and chair."
        ),
        "Six cabinets, each with a monitor and chair.",
        "One desk, each with a monitor and chair.",
        "Six desks, each with a monitor and chair. Eight chairs in total.",
        (
            "Six desks, each with a monitor and one chair. "
            "One chair sits in the corner."
        ),
        "Six desks, each with a monitor and no chair.",
        "Six desks, each with a monitor, keyboard, or chair.",
        "Six desks, each with a monitor and optional chair.",
        "Six desks, each with a monitor and at most one chair.",
        "Two desks sit at the front. Six desks, each with a monitor and chair.",
    ],
)
def test_ambiguous_coordinated_lists_do_not_authorize_pairing(prompt: str) -> None:
    contract = build_intent_contract(prompt)
    validate_intent_contract(contract)

    assert not any(row["relation"] == "paired_with" for row in contract["constraints"])


def test_existing_student_desk_pairing_preserves_role() -> None:
    contract = build_intent_contract(
        "A classroom with six student desks, each with a chair."
    )
    pairs = [row for row in contract["constraints"] if row["relation"] == "paired_with"]

    assert len(pairs) == 1
    assert pairs[0]["subjects"]["category"] == "student_chair"
    assert pairs[0]["targets"]["category"] == "student_desk"
    assert pairs[0]["subjects"]["count"] == 6


def test_existing_explicit_pair_each_desk_wording_still_works() -> None:
    contract = build_intent_contract(
        "An office with four desks. Pair each desk with exactly one office chair."
    )
    pairs = [row for row in contract["constraints"] if row["relation"] == "paired_with"]

    assert len(pairs) == 1
    assert pairs[0]["subjects"]["category"] == "office_chair"
    assert pairs[0]["subjects"]["count"] == pairs[0]["targets"]["count"] == 4
