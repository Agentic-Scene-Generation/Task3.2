from scenesmith.scenebenchmark_critic.metrics.functional_dependency.extensions.room_center import (
    evaluate_room_center_alignment,
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


def test_room_center_ignores_wall_centering_and_visual_axis_context() -> None:
    prompt = (
        "A bedroom with a bed centered on the main wall. "
        "Maintain clear floor space in the center of the room to allow for the "
        "clear visual axis from bed to dresser."
    )

    assert evaluate_room_center_alignment(_case(prompt)) == []


def test_room_center_keeps_explicit_anchor_contract() -> None:
    results = evaluate_room_center_alignment(
        _case("Place the bed in the center of the room.")
    )

    assert len(results) == 1
    assert results[0]["primary_object"] == "bed_0"
    assert results[0]["label"] == "fail"


def test_room_center_supports_contains_wording() -> None:
    results = evaluate_room_center_alignment(
        _case("The center of the room contains a bed.")
    )

    assert len(results) == 1
    assert results[0]["primary_object"] == "bed_0"
