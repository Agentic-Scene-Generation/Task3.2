from scenesmith.agent_utils.house import HouseLayout, RoomSpec
from scenesmith.experiments.indoor_scene_generation import _apply_runtime_task_prompt


def test_single_room_runtime_prompt_replaces_persisted_stage_brief() -> None:
    raw_prompt = "A study with guest chairs facing into the room."
    layout = HouseLayout(
        house_prompt="A study.\n\n=== SceneExpert Stage Brief: floor_plan ===\n...",
        room_specs=[
            RoomSpec(
                room_id="study",
                room_type="study",
                prompt=(
                    "A study with guest chairs facing the desk.\n\n"
                    "=== SceneExpert Stage Brief: floor_plan ===\n..."
                ),
            )
        ],
    )

    assert _apply_runtime_task_prompt(layout, raw_prompt)
    assert layout.house_prompt == raw_prompt
    assert layout.get_room_spec("study").prompt == raw_prompt


def test_multi_room_runtime_prompt_does_not_guess_room_mapping() -> None:
    layout = HouseLayout(
        house_prompt="Old house prompt",
        room_specs=[
            RoomSpec(room_id="bedroom", room_type="bedroom", prompt="Bedroom task."),
            RoomSpec(
                room_id="living_room",
                room_type="living_room",
                prompt="Living room task.",
            ),
        ],
    )

    assert _apply_runtime_task_prompt(layout, "A house with two rooms.")
    assert layout.house_prompt == "A house with two rooms."
    assert layout.get_room_spec("bedroom").prompt == "Bedroom task."
    assert layout.get_room_spec("living_room").prompt == "Living room task."
