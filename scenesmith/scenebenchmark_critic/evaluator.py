"""Registry-driven critic check construction and execution."""

from __future__ import annotations

from typing import Any

from scenesmith.scenebenchmark_critic.aggregation import (
    _normalize_result,
    _to_rule_config,
)
from scenesmith.scenebenchmark_critic.config import (
    DEFAULT_METRICS,
    CriticConfig,
    critic_config_from_any,
)
from scenesmith.scenebenchmark_critic.core.geometry import load_geometry
from scenesmith.scenebenchmark_critic.metrics.registry import get_metric_plugins
from scenesmith.scenebenchmark_critic.metrics.spatial_accessibility.companions import (
    attach_expected_access_companions,
)


def build_all_checks(
    case_pack: dict[str, Any],
    metrics: tuple[str, ...] | list[str] | None = None,
) -> list[dict[str, Any]]:
    """Build checks through every selected plugin exactly once."""
    selected = tuple(metrics or DEFAULT_METRICS)
    plugins = get_metric_plugins(selected)
    checks: list[dict[str, Any]] = []
    seen: set[str] = set()
    for plugin in plugins:
        if plugin.check_builder is None:
            continue
        for check in plugin.check_builder(case_pack, selected):
            check_id = str(check.get("check_id") or "")
            if check_id and check_id not in seen:
                checks.append(check)
                seen.add(check_id)
    return checks


def prepare_case_pack(
    case_pack: dict[str, Any],
    config: CriticConfig | Any | None = None,
) -> tuple[CriticConfig, tuple[Any, ...]]:
    critic_config = _coerce_config(config)
    plugins = get_metric_plugins(critic_config.metrics)
    rule_config = _to_rule_config(critic_config)
    for plugin in plugins:
        if plugin.check_augmenter is not None:
            plugin.check_augmenter(
                case_pack,
                rule_config,
                metric_filter=list(critic_config.metrics),
                progress=lambda _message: None,
            )
    store = load_geometry(case_pack)
    if store is not None and "spatial_accessibility" in critic_config.metrics:
        attach_expected_access_companions(case_pack, store.objects)
    return critic_config, plugins


def run_case_pack_checks(
    case_pack: dict[str, Any],
    config: CriticConfig | Any | None = None,
) -> list[dict[str, Any]]:
    """Evaluate per-check rules and scene extensions via the registry."""
    critic_config, plugins = prepare_case_pack(case_pack, config)
    enabled = {plugin.name: plugin for plugin in plugins}
    rule_config = _to_rule_config(critic_config)
    results: list[dict[str, Any]] = []
    for check in case_pack.get("checks") or []:
        metric = str(check.get("metric") or "")
        plugin = enabled.get(metric)
        if plugin is None or plugin.rule_evaluator is None:
            continue
        result = plugin.rule_evaluator(case_pack, check, rule_config)
        if result is not None:
            results.append(_normalize_result(result, check))
    for plugin in plugins:
        for extension in plugin.extension_evaluators:
            for result in extension(case_pack):
                normalized = _normalize_result(
                    result,
                    {
                        "check_id": result.get("check_id"),
                        "metric": plugin.name,
                        "subject_id": result.get("primary_object"),
                        "target_ids": result.get("related_objects") or [],
                    },
                )
                # 2026-07-16 修改原因：扩展结果必须归属注册插件，防止旧
                # 单文件规则把新指标报告到 interaction_clearance 等错误分组。
                if normalized.get("metric") != plugin.name:
                    raise ValueError(
                        f"Metric extension {plugin.name!r} emitted "
                        f"{normalized.get('metric')!r}"
                    )
                results.append(normalized)
    return results


def evaluate_case_pack(
    case_pack: dict[str, Any],
    config: CriticConfig | Any | None = None,
) -> dict[str, Any]:
    """Return raw results and the canonical aggregate for a case pack."""
    from scenesmith.scenebenchmark_critic.aggregation import aggregate_results

    results = run_case_pack_checks(case_pack, config=config)
    return {"results": results, "summary": aggregate_results(results, case_pack=case_pack)}


def _coerce_config(config: CriticConfig | Any | None) -> CriticConfig:
    if isinstance(config, CriticConfig):
        return config
    if config is None:
        return CriticConfig(enabled=True)
    return critic_config_from_any(config)
