"""Result normalization and aggregation for critic plugins."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from scenesmith.scenebenchmark_critic.config import CriticConfig, critic_config_from_any

LABEL_TO_SCORE = {"pass": 1.0, "degraded": 0.5, "fail": 0.0}


@dataclass(frozen=True)
class _RuleConfig:
    run: "_RuleRunConfig"
    provider: dict[str, Any] | None = None


@dataclass(frozen=True)
class _RuleRunConfig:
    metrics: list[str] | None = None
    accessibility_grid_resolution_m: float = 0.05
    accessibility_agent_width_m: float = 0.50
    accessibility_obstacle_height_threshold_m: float = 1.8
    accessibility_access_zone_depth_m: float = 0.55
    accessibility_pass_ratio: float = 0.60
    accessibility_degraded_ratio: float = 0.25
    accessibility_agent_profiles: list[dict[str, Any]] | None = None
    fd_relation_proposer_mode: str = "template"
    max_fd_relation_proposals: int = 8


def aggregate_results(
    results_or_case_pack: list[dict[str, Any]] | dict[str, Any],
    results: list[dict[str, Any]] | None = None,
    *,
    case_pack: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if results is None:
        results = (
            list(results_or_case_pack) if isinstance(results_or_case_pack, list) else []
        )
    else:
        case_pack = (
            results_or_case_pack
            if isinstance(results_or_case_pack, dict)
            else case_pack
        )
    checks_by_id = _checks_by_id(case_pack)
    object_acc: dict[str, dict[str, Any]] = {}
    by_metric: dict[str, dict[str, Any]] = {}
    by_metric_diag: dict[str, dict[str, Any]] = {}
    scene = _new_bucket()
    scene_diag = _new_bucket()
    for result in results:
        check_id = str(result.get("check_id") or "")
        check = checks_by_id.get(check_id, {})
        metric = str(result.get("metric") or check.get("metric") or "unknown")
        subject_id = str(
            result.get("primary_object") or check.get("subject_id") or "unknown"
        )
        bucket = by_metric.setdefault(metric, _new_bucket())
        diag_bucket = by_metric_diag.setdefault(metric, _new_bucket())
        object_bucket = object_acc.setdefault(
            subject_id,
            {
                "subject_id": subject_id,
                "checks": [],
                "metric_summary": {},
                "metric_diagnostic_summary": {},
            },
        )
        object_metric_bucket = object_bucket["metric_summary"].setdefault(
            metric, _new_bucket()
        )
        object_metric_diag_bucket = object_bucket[
            "metric_diagnostic_summary"
        ].setdefault(metric, _new_bucket())
        label = str(result.get("label") or "unknown")
        scoring_tier = _normalize_scoring_tier(
            result.get("scoring_tier") or check.get("scoring_tier")
        )
        counted = scoring_tier != "ignored"
        row = _object_result_row(
            result, check, subject_id, metric, label, scoring_tier, counted
        )
        object_bucket["checks"].append(row)

        _accumulate(bucket, label, scoring_tier=scoring_tier, counted=counted)
        _accumulate(
            object_metric_bucket, label, scoring_tier=scoring_tier, counted=counted
        )
        _accumulate(scene, label, scoring_tier=scoring_tier, counted=counted)
        _accumulate(diag_bucket, label, scoring_tier=scoring_tier, counted=True)
        _accumulate(
            object_metric_diag_bucket,
            label,
            scoring_tier=scoring_tier,
            counted=True,
        )
        _accumulate(scene_diag, label, scoring_tier=scoring_tier, counted=True)
    object_results: list[dict[str, Any]] = []
    for subject_id, payload in sorted(object_acc.items()):
        object_results.append(
            {
                "subject_id": subject_id,
                "checks": payload["checks"],
                "metric_summary": {
                    metric: _finish_bucket(bucket)
                    for metric, bucket in sorted(payload["metric_summary"].items())
                },
                "metric_diagnostic_summary": {
                    metric: _finish_bucket(bucket)
                    for metric, bucket in sorted(
                        payload["metric_diagnostic_summary"].items()
                    )
                },
            }
        )
    return {
        "object_results": object_results,
        "scene_summary": _finish_bucket(scene),
        "scene_diagnostic_summary": _finish_bucket(scene_diag),
        "metric_summary": {
            metric: _finish_bucket(bucket)
            for metric, bucket in sorted(by_metric.items())
        },
        "metric_diagnostic_summary": {
            metric: _finish_bucket(bucket)
            for metric, bucket in sorted(by_metric_diag.items())
        },
    }


def _object_result_row(
    result: dict[str, Any],
    check: dict[str, Any],
    subject_id: str,
    metric: str,
    label: str,
    scoring_tier: str,
    counted: bool,
) -> dict[str, Any]:
    score = LABEL_TO_SCORE.get(label)
    return {
        "check_id": result.get("check_id") or check.get("check_id"),
        "metric": metric,
        "label": label,
        "confidence": result.get("confidence"),
        "reason": result.get("reason"),
        "blocking_objects": result.get("blocking_objects") or [],
        "primary_object": result.get("primary_object") or subject_id,
        "related_objects": result.get("related_objects")
        or check.get("target_ids")
        or [],
        "selected_related_objects": result.get("selected_related_objects") or [],
        "diagnostics": result.get("diagnostics") or {},
        "evidence": result.get("evidence") or {},
        "score": score,
        "known": score is not None,
        "scoring_tier": scoring_tier,
        "counted_in_summary": counted,
    }


def _checks_by_id(case_pack: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not isinstance(case_pack, dict):
        return {}
    return {
        str(check.get("check_id")): check
        for check in case_pack.get("checks") or []
        if isinstance(check, dict) and check.get("check_id")
    }


def _normalize_result(result: dict[str, Any], check: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(result)
    normalized.setdefault("primary_object", str(check.get("subject_id") or ""))
    normalized.setdefault(
        "related_objects",
        [str(item) for item in (check.get("target_ids") or []) if str(item)],
    )
    normalized.setdefault("selected_related_objects", [])
    normalized.setdefault("blocking_objects", [])
    normalized.setdefault("confidence", 0.0)
    normalized.setdefault("reason", "")
    normalized.setdefault("check_id", check.get("check_id"))
    normalized.setdefault("metric", check.get("metric"))
    if "evidence" not in normalized and isinstance(normalized.get("diagnostics"), dict):
        normalized["evidence"] = normalized["diagnostics"]
    return normalized


def _to_rule_config(config: CriticConfig) -> _RuleConfig:
    extra = config.extra or {}
    asset_annotation = config.asset_annotation or {}

    def _get(name: str, default: Any) -> Any:
        return extra.get(name, default)

    model = asset_annotation.get("model")
    return _RuleConfig(
        provider={"model": model} if model else None,
        run=_RuleRunConfig(
            metrics=list(config.metrics),
            accessibility_grid_resolution_m=float(
                _get("accessibility_grid_resolution_m", 0.05)
            ),
            accessibility_agent_width_m=float(
                _get("accessibility_agent_width_m", 0.50)
            ),
            accessibility_obstacle_height_threshold_m=float(
                _get("accessibility_obstacle_height_threshold_m", 1.8)
            ),
            accessibility_access_zone_depth_m=float(
                _get("accessibility_access_zone_depth_m", 0.55)
            ),
            accessibility_pass_ratio=float(_get("accessibility_pass_ratio", 0.60)),
            accessibility_degraded_ratio=float(
                _get("accessibility_degraded_ratio", 0.25)
            ),
            accessibility_agent_profiles=_get("accessibility_agent_profiles", None),
            fd_relation_proposer_mode=str(
                _get("fd_relation_proposer_mode", "template")
            ),
            max_fd_relation_proposals=int(_get("max_fd_relation_proposals", 8)),
        ),
    )


def _coerce_config(config: CriticConfig | Any | None) -> CriticConfig:
    if isinstance(config, CriticConfig):
        return config
    if config is None:
        return CriticConfig(enabled=True)
    return critic_config_from_any(config)


def _new_bucket() -> dict[str, float]:
    return {
        "all_checks": 0,
        "total_checks": 0,
        "pass": 0,
        "degraded": 0,
        "fail": 0,
        "unknown": 0,
        "score_sum": 0.0,
        "effective_checks": 0,
        "excluded_ignored": 0,
        "excluded_auxiliary": 0,
    }


def _accumulate(
    bucket: dict[str, float], label: str, *, scoring_tier: str, counted: bool
) -> None:
    bucket["all_checks"] += 1
    if not counted:
        if scoring_tier == "ignored":
            bucket["excluded_ignored"] += 1
        elif scoring_tier == "auxiliary":
            bucket["excluded_auxiliary"] += 1
        return
    bucket["total_checks"] += 1
    if label in {"pass", "degraded", "fail", "unknown"}:
        bucket[label] += 1
    else:
        bucket["unknown"] += 1
        label = "unknown"
    if label in LABEL_TO_SCORE:
        bucket["score_sum"] += LABEL_TO_SCORE[label]
        bucket["effective_checks"] += 1


def _normalize_scoring_tier(value: Any) -> str:
    tier = str(value or "core").strip().lower()
    if tier in {"core", "auxiliary", "ignored"}:
        return tier
    return "core"


def _finish_bucket(bucket: dict[str, float]) -> dict[str, Any]:
    effective = int(bucket["effective_checks"])
    total_pass = int(bucket["pass"])
    total = int(bucket["total_checks"])
    all_checks = int(bucket["all_checks"])
    excluded_ignored = int(bucket["excluded_ignored"])
    excluded_auxiliary = int(bucket["excluded_auxiliary"])
    return {
        "all_checks": all_checks,
        "total_checks": total,
        "pass": total_pass,
        "degraded": int(bucket["degraded"]),
        "fail": int(bucket["fail"]),
        "unknown": int(bucket["unknown"]),
        "score_sum": bucket["score_sum"],
        "effective_checks": effective,
        "excluded_auxiliary": excluded_auxiliary,
        "excluded_ignored": excluded_ignored,
        "excluded_checks": excluded_auxiliary + excluded_ignored,
        "coverage": effective / total if total else 0.0,
        "effective_pass_rate": total_pass / effective if effective else None,
        "score": bucket["score_sum"] / effective if effective else None,
    }
