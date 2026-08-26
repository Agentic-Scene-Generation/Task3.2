"""Configuration helpers for the embedded SceneBenchmark critic."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Any, Literal

RepairModule = Literal[
    "furniture_relations",
    "visual_clearance",
    "storage_accessibility",
    "seating_orientation",
    "window_clearance",
]

_REPAIR_MODULES = frozenset(
    {
        "furniture_relations",
        "visual_clearance",
        "storage_accessibility",
        "seating_orientation",
        "window_clearance",
    }
)


@dataclass(frozen=True)
class AutoRepairConfig:
    """Controls deterministic repairs initiated by the SceneBenchmark critic.

    This deliberately does not control the independent furniture safety
    controller.  Keeping that boundary here makes evaluate-only experiments
    explicit instead of silently changing their safety semantics.
    """

    enabled: bool = True
    furniture_relations: bool = True
    visual_clearance: bool = True
    storage_accessibility: bool = True
    seating_orientation: bool = True
    window_clearance: bool = True
    max_repairs_per_call: int | None = None
    max_candidate_evaluations: int | None = None
    furniture_relation_budget: int | None = None
    visual_clearance_budget: int | None = None
    storage_accessibility_budget: int | None = None

    def should_repair(self, module: RepairModule) -> bool:
        if module not in _REPAIR_MODULES:
            raise ValueError(
                f"Unknown auto-repair module {module!r}; expected one of "
                f"{', '.join(sorted(_REPAIR_MODULES))}"
            )
        return self.enabled and bool(getattr(self, module))

# 2026-07-16 修改原因：critic 迁移到统一 registry 后，四个一级指标必须
# 使用同一默认集合，避免视觉规则在 API、配置和回放之间被静默漏掉。
DEFAULT_METRICS = (
    "functional_dependency",
    "spatial_accessibility",
    "interaction_clearance",
    "visual_clearance",
)


@dataclass(frozen=True)
class CriticConfig:
    enabled: bool = False
    metrics: tuple[str, ...] = DEFAULT_METRICS
    room_stage_hooks: tuple[str, ...] = ("scene_after_furniture", "final_scene")
    house_stage_hooks: tuple[str, ...] = ()
    inject_into_llm_critic: bool = True
    agent_prompt_context_filter_enabled: bool = True
    agent_prompt_context_debug_write: bool = False
    hard_gate: bool = False
    max_issues_for_prompt: int = 8
    fail_gate_threshold: int = 1
    degraded_gate_threshold: int = 999999
    asset_annotation: dict[str, Any] = field(default_factory=dict)
    intent_compiler: dict[str, Any] = field(default_factory=dict)
    visual_grounding: dict[str, Any] = field(default_factory=dict)
    vlm_hard_gate: bool = False
    auto_repair: AutoRepairConfig = field(default_factory=AutoRepairConfig)
    extra: dict[str, Any] = field(default_factory=dict)

    def metric_enabled(self, metric: str) -> bool:
        return metric in set(self.metrics)

    def room_stage_enabled(self, stage: str) -> bool:
        return stage in set(self.room_stage_hooks)

    def house_stage_enabled(self, stage: str) -> bool:
        return stage in set(self.house_stage_hooks)


def critic_config_from_any(cfg: Any) -> CriticConfig:
    """Extract critic config from a full experiment config or an agent config."""
    if os.environ.get("CRITIC_CONSTRAINT_MODE") is not None:
        raise ValueError(
            "SceneBenchmark critic CRITIC_CONSTRAINT_MODE was removed; delete the "
            "old legacy/shadow/contract environment override. Intent contracts are "
            "always hard."
        )
    raw = _get(cfg, "scenebenchmark_critic", None)
    if raw is None:
        experiment = _get(cfg, "experiment", None)
        raw = _get(experiment, "scenebenchmark_critic", None)
    if raw is None:
        return CriticConfig()

    data = _to_plain_dict(raw)
    known = {
        "enabled",
        "metrics",
        "room_stage_hooks",
        "house_stage_hooks",
        "inject_into_llm_critic",
        "agent_prompt_context_filter_enabled",
        "agent_prompt_context_debug_write",
        "hard_gate",
        "max_issues_for_prompt",
        "fail_gate_threshold",
        "degraded_gate_threshold",
        "asset_annotation",
        "intent_compiler",
        "visual_grounding",
        "vlm_hard_gate",
        "auto_repair",
    }
    removed_mode_fields = {
        "constraint_mode",
        "intent_contract_mode",
        "CRITIC_CONSTRAINT_MODE",
    }
    supplied_removed_modes = sorted(removed_mode_fields & set(data))
    if supplied_removed_modes:
        raise ValueError(
            "SceneBenchmark critic "
            f"{', '.join(supplied_removed_modes)} was removed; delete the old "
            "legacy/shadow/contract override. Intent contracts are always hard."
        )
    extra = {key: value for key, value in data.items() if key not in known}
    metrics = _as_tuple(data.get("metrics", DEFAULT_METRICS), DEFAULT_METRICS)
    # 2026-07-16 修改原因：旧调度器会静默跳过未知 metric，迁移后应在配置
    # 入口直接失败，避免回放得到缺指标但看似成功的报告。
    from scenesmith.scenebenchmark_critic.metrics.registry import get_metric_plugins

    get_metric_plugins(metrics)
    return CriticConfig(
        enabled=_as_bool(data.get("enabled", False)),
        metrics=metrics,
        room_stage_hooks=_as_tuple(
            data.get("room_stage_hooks", ("scene_after_furniture", "final_scene")),
            ("scene_after_furniture", "final_scene"),
        ),
        house_stage_hooks=_as_tuple(data.get("house_stage_hooks", ()), ()),
        inject_into_llm_critic=_as_bool(data.get("inject_into_llm_critic", True)),
        agent_prompt_context_filter_enabled=_as_bool(
            data.get("agent_prompt_context_filter_enabled", True)
        ),
        agent_prompt_context_debug_write=_as_bool(
            data.get("agent_prompt_context_debug_write", False)
        ),
        hard_gate=_as_bool(data.get("hard_gate", False)),
        max_issues_for_prompt=int(data.get("max_issues_for_prompt", 8)),
        fail_gate_threshold=int(data.get("fail_gate_threshold", 1)),
        degraded_gate_threshold=int(data.get("degraded_gate_threshold", 999999)),
        asset_annotation=_as_dict(data.get("asset_annotation")),
        intent_compiler=_as_dict(data.get("intent_compiler")),
        visual_grounding=_as_dict(data.get("visual_grounding")),
        # VLM output is evidence-only in v1.  Retaining this parsed field makes
        # an accidental future config opt-in observable without granting it
        # authority in the evaluator.
        vlm_hard_gate=_as_bool(data.get("vlm_hard_gate", False)),
        auto_repair=_auto_repair_config_from_any(data.get("auto_repair", None)),
        extra=extra,
    )


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _to_plain_dict(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        return dict(obj)
    if hasattr(obj, "items"):
        return {key: value for key, value in obj.items()}
    return {
        key: getattr(obj, key)
        for key in dir(obj)
        if not key.startswith("_") and not callable(getattr(obj, key))
    }


def _auto_repair_config_from_any(value: Any) -> AutoRepairConfig:
    """Parse only documented auto-repair shapes, with useful config errors."""
    if value is None:
        return AutoRepairConfig()
    if type(value) is bool:
        return AutoRepairConfig(enabled=value)
    if not isinstance(value, dict) and not hasattr(value, "items"):
        raise ValueError("scenebenchmark_critic.auto_repair must be a boolean or mapping")
    data = dict(value.items())
    known = {
        "enabled",
        "furniture_relations",
        "visual_clearance",
        "storage_accessibility",
        "seating_orientation",
        "window_clearance",
        "max_repairs_per_call",
        "max_candidate_evaluations",
        "furniture_relation_budget",
        "visual_clearance_budget",
        "storage_accessibility_budget",
    }
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(
            "Unknown scenebenchmark_critic.auto_repair field(s): "
            f"{', '.join(unknown)}. Valid fields: {', '.join(sorted(known))}"
        )
    bool_fields = known - {
        "max_repairs_per_call",
        "max_candidate_evaluations",
        "furniture_relation_budget",
        "visual_clearance_budget",
        "storage_accessibility_budget",
    }
    parsed_bools = {
        name: _as_strict_bool(data.get(name, True), f"auto_repair.{name}")
        for name in bool_fields
    }
    return AutoRepairConfig(
        **parsed_bools,
        max_repairs_per_call=_as_non_negative_int_or_none(
            data.get("max_repairs_per_call"), "auto_repair.max_repairs_per_call"
        ),
        max_candidate_evaluations=_as_non_negative_int_or_none(
            data.get("max_candidate_evaluations"),
            "auto_repair.max_candidate_evaluations",
        ),
        furniture_relation_budget=_as_non_negative_int_or_none(
            data.get("furniture_relation_budget"), "auto_repair.furniture_relation_budget"
        ),
        visual_clearance_budget=_as_non_negative_int_or_none(
            data.get("visual_clearance_budget"), "auto_repair.visual_clearance_budget"
        ),
        storage_accessibility_budget=_as_non_negative_int_or_none(
            data.get("storage_accessibility_budget"), "auto_repair.storage_accessibility_budget"
        ),
    )


def _as_strict_bool(value: Any, name: str) -> bool:
    if type(value) is bool:
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "on", "1"}:
            return True
        if normalized in {"false", "no", "off", "0"}:
            return False
    raise ValueError(f"{name} must be a boolean (or true/false-style string)")


def _as_non_negative_int_or_none(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if type(value) is int:
        if value >= 0:
            return value
    elif isinstance(value, str) and value.isdecimal():
        return int(value)
    raise ValueError(f"{name} must be None or a non-negative integer")


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off", ""}:
            return False
    return bool(value)


def _as_tuple(value: Any, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return default
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    return tuple(value or ())


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "items"):
        return {key: item for key, item in value.items()}
    return {}
