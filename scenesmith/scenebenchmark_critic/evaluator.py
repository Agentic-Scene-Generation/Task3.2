"""Registry-driven critic check construction and execution."""

from __future__ import annotations

import time

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
from scenesmith.scenebenchmark_critic.intent_contract import (
    augment_contract_checks,
    constraint_mode,
    set_contract_mode,
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
    set_contract_mode(case_pack, constraint_mode(critic_config))
    # Contract checks are inserted before legacy plugin augmenters.  In shadow
    # mode they are deliberately ignored for scoring, which gives replay logs a
    # side-by-side comparison without changing the current repair loop.
    augment_contract_checks(case_pack)
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
    timing: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate per-check rules and scene extensions via the registry."""
    timing_start = time.perf_counter()
    prepare_start = time.perf_counter()
    critic_config, plugins = prepare_case_pack(case_pack, config)
    if timing is not None:
        timing["prepare_case_pack_sec"] = round(time.perf_counter() - prepare_start, 6)
    enabled = {plugin.name: plugin for plugin in plugins}
    rule_config = _to_rule_config(critic_config)
    results: list[dict[str, Any]] = []
    rule_times: dict[str, float] = {}
    for check in case_pack.get("checks") or []:
        metric = str(check.get("metric") or "")
        plugin = enabled.get(metric)
        if plugin is None or plugin.rule_evaluator is None:
            continue
        check_start = time.perf_counter()
        result = plugin.rule_evaluator(case_pack, check, rule_config)
        rule_times[metric] = rule_times.get(metric, 0.0) + (
            time.perf_counter() - check_start
        )
        if result is not None:
            results.append(_normalize_result(result, check))
    extension_times: dict[str, float] = {}
    for plugin in plugins:
        for extension in plugin.extension_evaluators:
            extension_start = time.perf_counter()
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
            extension_times[plugin.name] = extension_times.get(plugin.name, 0.0) + (
                time.perf_counter() - extension_start
            )
    if timing is not None:
        timing["rule_evaluator_sec_by_metric"] = {
            key: round(value, 6) for key, value in rule_times.items()
        }
        timing["extension_evaluator_sec_by_metric"] = {
            key: round(value, 6) for key, value in extension_times.items()
        }
        timing["run_case_pack_checks_sec"] = round(
            time.perf_counter() - timing_start, 6
        )
    return results


def evaluate_case_pack(
    case_pack: dict[str, Any],
    config: CriticConfig | Any | None = None,
) -> dict[str, Any]:
    """Return raw results and the canonical aggregate for a case pack."""
    from scenesmith.scenebenchmark_critic.aggregation import aggregate_results

    results = run_case_pack_checks(case_pack, config=config)
    return {
        "results": results,
        "summary": aggregate_results(results, case_pack=case_pack),
    }


def _coerce_config(config: CriticConfig | Any | None) -> CriticConfig:
    if isinstance(config, CriticConfig):
        return config
    if config is None:
        return CriticConfig(enabled=True)
    return critic_config_from_any(config)
