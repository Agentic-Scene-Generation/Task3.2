"""Configuration helpers for the embedded SceneBenchmark critic."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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
    extra: dict[str, Any] = field(default_factory=dict)

    def metric_enabled(self, metric: str) -> bool:
        return metric in set(self.metrics)

    def room_stage_enabled(self, stage: str) -> bool:
        return stage in set(self.room_stage_hooks)

    def house_stage_enabled(self, stage: str) -> bool:
        return stage in set(self.house_stage_hooks)


def critic_config_from_any(cfg: Any) -> CriticConfig:
    """Extract critic config from a full experiment config or an agent config."""
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
    }
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
