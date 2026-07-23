from types import SimpleNamespace

from scenesmith.agent_utils.room import UniqueID
from scenesmith.agent_utils.scene_analyzer import FurnitureSelection
from scenesmith.manipuland_agents.stateful_manipuland_agent import (
    StatefulManipulandAgent,
)
from scenesmith.scenebenchmark_critic.manipuland_targets import (
    infer_prompt_manipuland_obligations,
)


def _object(object_id: str, name: str):
    return SimpleNamespace(
        object_id=UniqueID(object_id),
        name=name,
        description=name,
        immutable=False,
    )


def test_dining_prompt_requires_table_and_sideboard_targets() -> None:
    prompt = (
        "A dining room with a dining table and table settings for four including "
        "plates, cutlery, and glasses. A centerpiece vase sits in the middle of "
        "the table, and a set of coasters sits on the sideboard."
    )

    obligations = infer_prompt_manipuland_obligations(prompt)

    assert [(item.category, item.target_count) for item in obligations] == [
        ("dining_table", 1),
        ("sideboard", 1),
    ]


def test_recovery_adds_missing_dining_table_without_duplicate_sideboard() -> None:
    table = _object("dining_table_0", "dining table")
    sideboard = _object("sideboard_0", "sideboard")
    scene = SimpleNamespace(
        scene_expert_original_description=(
            "A dining room with a dining table and table settings for four including "
            "plates, cutlery, and glasses. A centerpiece vase sits in the middle of "
            "the table, and a set of coasters sits on the sideboard."
        ),
        text_description="",
        objects={table.object_id: table, sideboard.object_id: sideboard},
        get_object=lambda object_id: {
            table.object_id: table,
            sideboard.object_id: sideboard,
        }.get(object_id),
    )
    agent = object.__new__(StatefulManipulandAgent)
    agent.cfg = SimpleNamespace(scenebenchmark_critic={"enabled": True})
    selections = [
        FurnitureSelection(
            furniture_id=sideboard.object_id,
            suggested_items="REQUIRED: coasters",
            prompt_constraints="prompt",
            style_notes="",
        )
    ]

    recovered = agent._recover_prompt_required_manipuland_targets(
        scene=scene, furniture_data=selections
    )

    assert [selection.furniture_id for selection in recovered] == [
        sideboard.object_id,
        table.object_id,
    ]


def test_bilateral_bedside_prompt_recovers_both_nightstands() -> None:
    prompt = (
        "A nightstand with a table lamp on each side of the bed. An alarm clock "
        "sits on one nightstand and a book on the other."
    )

    obligations = infer_prompt_manipuland_obligations(prompt)

    assert [(item.category, item.target_count) for item in obligations] == [
        ("nightstand", 2)
    ]
