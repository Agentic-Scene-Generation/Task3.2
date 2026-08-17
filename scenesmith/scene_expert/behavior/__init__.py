"""Opt-in deterministic resident behavior planning for SceneExpert."""

from scenesmith.scene_expert.behavior.assets import merge_behavior_assets
from scenesmith.scene_expert.behavior.integration import apply_behavior_template
from scenesmith.scene_expert.behavior.planner import (
    TemplateBehaviorPlanner,
    build_behavior_spec,
)
from scenesmith.scene_expert.behavior.schemas import BehaviorSpec

__all__ = [
    "BehaviorSpec",
    "TemplateBehaviorPlanner",
    "apply_behavior_template",
    "build_behavior_spec",
    "merge_behavior_assets",
]
