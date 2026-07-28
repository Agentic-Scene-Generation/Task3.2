"""Configuration precedence helpers shared by SceneExpert entry points."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _as_bool(value: Any, *, default: bool) -> bool:
    """Parse Hydra/env-style boolean values without treating "false" as true."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


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


def scene_expert_runtime_budget_enabled(cfg_dict: Mapping[str, Any]) -> bool:
    """Return whether SceneExpert may inject runtime limits into SceneSmith agents."""
    scene_expert_cfg = resolve_scene_expert_config(cfg_dict)
    if not _as_bool(scene_expert_cfg.get("enabled"), default=False):
        return False
    control_cfg = scene_expert_cfg.get("runtime_budget_control", {}) or {}
    return _as_bool(control_cfg.get("enabled"), default=True)


def resolve_scene_expert_stage_budget(
    cfg_dict: Mapping[str, Any], stage: str
) -> dict[str, object]:
    """Resolve the effective default-plus-stage execution budget."""
    scene_expert_cfg = resolve_scene_expert_config(cfg_dict)
    if not scene_expert_runtime_budget_enabled(cfg_dict):
        return {}
    stage_budgets = scene_expert_cfg.get("stage_budget", {}) or {}
    default_budget = stage_budgets.get("default", {}) or {}
    stage_budget = stage_budgets.get(stage, {}) or {}
    return {**dict(default_budget), **dict(stage_budget)}
