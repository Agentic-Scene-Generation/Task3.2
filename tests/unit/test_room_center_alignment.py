from scenesmith.scenebenchmark_critic.intent_contract import build_intent_contract
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.extensions.intent_contract import (
    evaluate_intent_contract_extensions,
)


def _case(prompt: str) -> dict:
    return {
        "task_instruction": prompt,
        "scene_geometry": {
            "rooms": [
                {
                    "id": "bedroom",
                    "bbox": {"min": [-2.5, -2.0, 0.0], "max": [2.5, 2.0, 3.0]},
                }
            ],
            "objects": [
                {
                    "id": "bed_0",
                    "category": "bed",
                    "bbox_world": {
                        "center": [0.0, 1.01, 0.3],
                        "min": [-0.8, 0.1, 0.0],
                        "max": [0.8, 1.92, 0.6],
                    },
                }
            ],
        },
    }


def _evaluate(prompt: str, *, original_prompt: str | None = None) -> list[dict]:
    task_prompt = original_prompt or prompt
    case_pack = _case(prompt)
    case_pack["stage"] = "furniture"
    case_pack["original_task_instruction"] = task_prompt
    case_pack["intent_contract"] = build_intent_contract(task_prompt)
    return [
        result
        for result in evaluate_intent_contract_extensions(case_pack)
        if result.get("relation_type") == "room_center_alignment"
    ]


def test_room_center_ignores_wall_centering_and_visual_axis_context() -> None:
    prompt = (
        "A bedroom with a bed centered on the main wall. "
        "Maintain clear floor space in the center of the room to allow for the "
        "clear visual axis from bed to dresser."
    )

    assert _evaluate(prompt) == []


def test_room_center_keeps_explicit_anchor_contract() -> None:
    results = _evaluate("Place the bed in the center of the room.")

    assert len(results) == 1
    assert results[0]["primary_object"] == "bed_0"
    assert results[0]["label"] == "fail"


def test_room_center_supports_contains_wording() -> None:
    results = _evaluate("The center of the room contains a bed.")

    assert len(results) == 1
    assert results[0]["primary_object"] == "bed_0"


def test_room_center_ignores_stage_brief_failure_examples() -> None:
    prompt = (
        "A bedroom with a bed against the wall. "
        "Known failure patterns to avoid: Placing the bed in the middle of the "
        "room without wall support."
    )

    assert _evaluate(prompt) == []


def test_room_center_uses_original_task_not_injected_stage_brief() -> None:
    case_pack = _case(
        "A bedroom with a bed against the wall.\n\n"
        "=== SceneExpert Stage Brief: furniture ===\n"
        "Use the bed as the central anchor for this arrangement."
    )
    case_pack["original_task_instruction"] = "A bedroom with a bed against the wall."

    assert (
        _evaluate(
            case_pack["task_instruction"],
            original_prompt=case_pack["original_task_instruction"],
        )
        == []
    )


def test_room_center_does_not_treat_central_anchor_as_room_center() -> None:
    assert _evaluate("Use the bed as the central anchor of the sleeping zone.") == []


def test_room_center_ignores_off_center_counterexample_but_keeps_request() -> None:
    results = _evaluate(
        "Place the bed in the center of the room. " "Avoid leaving the bed off-center."
    )

    assert len(results) == 1
    assert results[0]["primary_object"] == "bed_0"
