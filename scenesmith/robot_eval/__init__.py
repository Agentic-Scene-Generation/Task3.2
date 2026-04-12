"""Robot evaluation module for task/scene generation and success validation.

This module provides:
- Task generation: Convert human natural language tasks into diverse scene prompts
- Success validation: Validate task completion using LLM agents with state + vision tools
- Policy interface: Predicate extraction and resolution for policies needing structured goals
- Scene loading: Load scenes from dmd.yaml + scene_state.json for evaluation
"""

import os

from omegaconf import DictConfig, OmegaConf

from scenesmith.robot_eval.dmd_scene import DMDScene, load_scene_for_validation

__all__ = [
    "create_robot_eval_config",
    "DMDScene",
    "load_scene_for_validation",
]


def create_robot_eval_config(
    model: str | None = None, vision_detail: str = "high"
) -> DictConfig:
    """Create minimal config for standalone robot_eval usage.

    This helper creates a self-contained config for robot_eval scripts without
    requiring the full Hydra configuration. Useful for CLI scripts and testing.

    Args:
        model: OpenAI model name. Defaults to OPENAI_MODEL env var or "gpt-5.2".
        vision_detail: Vision detail level ("low", "high", "auto").

    Returns:
        DictConfig compatible with robot_eval classes.

    Example:
        >>> from scenesmith.robot_eval import create_robot_eval_config
        >>> from scenesmith.robot_eval.task_generation.scene_prompt_generator import (
        ...     ScenePromptGenerator,
        ... )
        >>> cfg = create_robot_eval_config()
        >>> generator = ScenePromptGenerator(cfg=cfg)
    """
    return OmegaConf.create(
        {
            "openai": {
                "model": model or os.environ.get("OPENAI_MODEL", "gpt-5.2"),
                "vision_detail": vision_detail,
            }
        }
    )
