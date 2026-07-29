"""Configuration precedence helpers shared by SceneExpert entry points."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def deep_merge_dicts(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict:
    """Merge nested mappings without mutating either input."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_scene_expert_config(cfg_dict: Mapping[str, Any]) -> dict:
    """Merge root defaults with the active experiment's ablation overrides."""
    root_cfg = cfg_dict.get("scene_expert", {}) or {}
    experiment = cfg_dict.get("experiment", {}) or {}
    experiment_cfg = experiment.get("scene_expert")
    if experiment_cfg is None:
        return dict(root_cfg)
    return deep_merge_dicts(root_cfg, experiment_cfg)


def scene_expert_execution_control_enabled(
    cfg_dict: Mapping[str, Any],
) -> bool:
    """Return whether SceneExpert may override native SceneSmith execution limits.

    Requirement grounding, memory retrieval, verification, and tracing are not
    execution-budget features and remain active when this switch is disabled.
    Only turn, token, deadline, retry, and acquisition-count overrides are
    gated here.
    """
    scene_expert_cfg = resolve_scene_expert_config(cfg_dict)
    if not bool(scene_expert_cfg.get("enabled", False)):
        return False
    control = scene_expert_cfg.get("execution_control", {}) or {}
    return bool(control.get("enabled", True))


def resolve_scene_expert_stage_budget(
    cfg_dict: Mapping[str, Any], stage: str
) -> dict[str, object]:
    """Resolve the effective default-plus-stage execution budget."""
    scene_expert_cfg = resolve_scene_expert_config(cfg_dict)
    if not scene_expert_execution_control_enabled(cfg_dict):
        return {}
    stage_budgets = scene_expert_cfg.get("stage_budget", {}) or {}
    default_budget = stage_budgets.get("default", {}) or {}
    stage_budget = stage_budgets.get(stage, {}) or {}
    control = scene_expert_cfg.get("execution_control", {}) or {}
    return {
        **dict(default_budget),
        **dict(stage_budget),
        "execution_control_enabled": True,
        "execution_control_profile": str(control.get("profile", "quality")),
    }
