"""Configuration precedence and feature-gate helpers for optional extensions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_COMPONENT_NAMES = (
    "task_compiler",
    "harness",
    "harness_budget",
    "global_planner",
    "prompt_injection",
    "fast_memory_retrieval",
    "memory_writer",
    "stage_working_memory",
    "verifier",
    "repair",
    "critic_bridge",
    "trace",
    "structured_llm",
    "slow_memory_capture",
)

_MODE_COMPONENTS = {
    "disabled": frozenset(),
    "harness_only": frozenset(
        {
            "task_compiler",
            "harness",
            "harness_budget",
            "global_planner",
            "prompt_injection",
            "critic_bridge",
            "verifier",
            "repair",
            "trace",
        }
    ),
    "harness_memory": frozenset(
        {
            "task_compiler",
            "harness",
            "harness_budget",
            "global_planner",
            "prompt_injection",
            "fast_memory_retrieval",
            "memory_writer",
            "stage_working_memory",
            "critic_bridge",
            "verifier",
            "repair",
            "trace",
        }
    ),
    "full": frozenset(
        {
            "task_compiler",
            "harness",
            "harness_budget",
            "global_planner",
            "prompt_injection",
            "fast_memory_retrieval",
            "memory_writer",
            "stage_working_memory",
            "critic_bridge",
            "verifier",
            "repair",
            "trace",
            "slow_memory_capture",
        }
    ),
}


def _as_bool(value: Any, default: bool = False) -> bool:
    """Coerce resolved Hydra values without treating ``"false"`` as true."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
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


def resolve_component_flags(cfg_dict: Mapping[str, Any]) -> dict[str, bool]:
    """Resolve legacy mode presets plus explicit per-component overrides.

    ``mode`` remains a backwards-compatible preset. An explicit
    ``components.<name>.enabled`` (or boolean shorthand) always wins, which
    makes ablation runs independent instead of coupling several mechanisms to
    one coarse mode string.
    """
    scene_expert_cfg = resolve_scene_expert_config(cfg_dict)
    master_enabled = _as_bool(scene_expert_cfg.get("enabled"), False)
    mode = str(scene_expert_cfg.get("mode", "disabled") or "disabled")
    preset = _MODE_COMPONENTS.get(mode, frozenset()) if master_enabled else frozenset()
    component_cfg = scene_expert_cfg.get("components", {}) or {}

    resolved: dict[str, bool] = {}
    for name in _COMPONENT_NAMES:
        value = component_cfg.get(name)
        if isinstance(value, Mapping):
            value = value.get("enabled")
        resolved[name] = master_enabled and _as_bool(value, name in preset)
    return resolved


def scene_expert_component_enabled(cfg_dict: Mapping[str, Any], component: str) -> bool:
    """Return the effective state for one independently gated component."""
    if component not in _COMPONENT_NAMES:
        raise KeyError(
            f"Unknown optional component {component!r}; expected one of "
            f"{', '.join(_COMPONENT_NAMES)}"
        )
    return resolve_component_flags(cfg_dict)[component]
