"""SceneBenchmark-style rule critic integration for SceneSmith."""

from scenesmith.scenebenchmark_critic.adapter import (
    house_scene_to_case_pack,
    room_scene_to_case_pack,
)
from scenesmith.scenebenchmark_critic.api import (
    evaluate_house_scene,
    evaluate_room_scene,
    format_prompt_context,
    write_house_stage_report,
    write_room_stage_report,
)
from scenesmith.scenebenchmark_critic.config import CriticConfig

__all__ = [
    "CriticConfig",
    "evaluate_house_scene",
    "evaluate_room_scene",
    "format_prompt_context",
    "house_scene_to_case_pack",
    "room_scene_to_case_pack",
    "write_house_stage_report",
    "write_room_stage_report",
]
