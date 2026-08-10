"""Configuration precedence and feature-gate helpers for optional extensions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_COMPONENT_NAMES = (
    "task_compiler",
    "harness",
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
)

_MODE_COMPONENTS = {
    "disabled": frozenset(),
    "harness_only": frozenset(
        {
            "task_compiler",
            "harness",
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
    if not _as_bool(scene_expert_cfg.get("enabled"), False):
        return False
    control = scene_expert_cfg.get("execution_control", {}) or {}
    return _as_bool(control.get("enabled"), False)


def _budget_group_enabled(control: Mapping[str, Any], group: str) -> bool:
    """Resolve a fine-grained execution-control switch under the master gate."""
    return _as_bool(control.get(f"{group}_enabled"), True)


def _budget_key_group(key: str) -> str | None:
    """Map one legacy flat budget key to its independently gated policy."""
    if "token" in key:
        return "token_budget"
    if "asset" in key or "semantic_retries" in key or "optional_object" in key:
        return "asset_budget"
    if key.startswith(("min_output_objects", "max_output_objects")):
        return "output_budget"
    if (
        "attempt" in key
        or "retries" in key
        or "regeneration" in key
        or "repair_steps" in key
        or "designer_iterations" in key
        or "retry_budget_multiplier" in key
    ):
        return "retry_budget"
    if key.endswith("_seconds") or "reserve_fraction" in key:
        if "active_max_seconds" in key:
            return "role_lease"
        return "wall_clock_budget"
    if key.endswith("_turns"):
        return "role_lease"
    return None


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
    merged_budget = {
        **dict(default_budget),
        **dict(stage_budget),
    }
    filtered_budget = {
        key: value
        for key, value in merged_budget.items()
        if (group := _budget_key_group(str(key))) is None
        or _budget_group_enabled(control, group)
    }
    return {
        **filtered_budget,
        "execution_control_enabled": True,
        "execution_control_profile": str(control.get("profile", "quality")),
    }
