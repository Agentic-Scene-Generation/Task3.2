"""Stable bridge from SceneBenchmark critic reports into SceneExpert.

This module is the only SceneExpert runtime component that knows the
SceneBenchmark report schema.  It deliberately depends on the critic's public
``evaluate_room_scene`` API and keeps the full provider payload on disk while
passing a compact, version-independent evidence object to verifier and memory
consumers.
"""

from __future__ import annotations

import json
import logging

from pathlib import Path
from typing import Any, Callable

from scenesmith.scene_expert.schemas import CriticEvidence, CriticEvidenceResult

console_logger = logging.getLogger(__name__)

SCENEBENCHMARK_REPORT_SCHEMA = "scenesmith.scenebenchmark_critic.report.v1"


class SceneBenchmarkCriticBridge:
    """Collect and normalize post-stage deterministic critic evidence."""

    def __init__(
        self,
        *,
        enabled: bool,
        critic_config: Any,
        output_dir: str | Path,
        stages: list[str] | tuple[str, ...] | None = None,
        persist_raw_reports: bool = True,
        require_schema_version: bool = True,
        evaluator: Callable[..., dict[str, Any]] | None = None,
    ) -> None:
        self._enabled = bool(enabled)
        self._critic_config = critic_config
        self._output_dir = Path(output_dir)
        self._stages = set(
            stages or ("furniture", "wall_mounted", "ceiling_mounted", "manipuland")
        )
        self._persist_raw_reports = bool(persist_raw_reports)
        self._require_schema_version = bool(require_schema_version)
        self._evaluator = evaluator

    @property
    def enabled(self) -> bool:
        return self._enabled

    def collect_room_stage(self, scene: Any, stage: str) -> CriticEvidence:
        """Evaluate one completed room stage and return normalized evidence."""
        if not self._enabled:
            return CriticEvidence(status="disabled", stage=stage)
        if stage not in self._stages:
            return CriticEvidence(status="stage_disabled", stage=stage)

        report_path: Path | None = None
        try:
            evaluator = self._evaluator
            if evaluator is None:
                # Lazy import keeps SceneExpert's lightweight unit tests usable
                # without Drake and the full SceneBenchmark dependency chain.
                from scenesmith.scenebenchmark_critic import evaluate_room_scene

                evaluator = evaluate_room_scene
            payload = evaluator(
                scene,
                config=self._critic_config,
                stage=f"scene_expert_post_{stage}",
            )
            report_path = self._persist_report(stage, payload)
            return self._normalize(
                payload,
                stage=stage,
                report_path=report_path,
            )
        except Exception as exc:
            console_logger.warning(
                "[SceneExpert] SceneBenchmark critic collection failed for %s: %s",
                stage,
                exc,
            )
            return CriticEvidence(
                status="error",
                stage=stage,
                report_path=str(report_path or ""),
                error=f"{type(exc).__name__}: {exc}",
            )

    def _persist_report(self, stage: str, payload: dict[str, Any]) -> Path | None:
        if not self._persist_raw_reports:
            return None
        path = self._output_dir / stage / "scenebenchmark_critic.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        return path

    def _normalize(
        self,
        payload: dict[str, Any],
        *,
        stage: str,
        report_path: Path | None,
    ) -> CriticEvidence:
        if not isinstance(payload, dict):
            raise TypeError("SceneBenchmark critic payload must be a mapping")
        schema_version = str(payload.get("schema_version") or "")
        if (
            self._require_schema_version
            and schema_version != SCENEBENCHMARK_REPORT_SCHEMA
        ):
            raise ValueError(
                "Unsupported SceneBenchmark critic report schema: "
                f"{schema_version or '<missing>'}"
            )

        summary = payload.get("summary") or {}
        scene_summary = summary.get("scene_summary") or {}
        metric_summary = summary.get("metric_summary") or {}
        gate = payload.get("gate") or {}
        metric_scores = {
            str(metric): float(metric_values["score"])
            for metric, metric_values in metric_summary.items()
            if isinstance(metric_values, dict)
            and isinstance(metric_values.get("score"), (int, float))
        }
        results = [
            self._normalize_result(result)
            for result in payload.get("results") or []
            if isinstance(result, dict)
        ]
        scene_score = scene_summary.get("score")
        return CriticEvidence(
            provider_schema_version=schema_version,
            status="ok",
            available=True,
            stage=stage,
            scope=str(payload.get("scope") or ""),
            report_path=str(report_path or ""),
            gate_enabled=bool(gate.get("enabled", False)),
            gate_blocked=bool(gate.get("blocked", False)),
            gate_label=str(gate.get("label") or "report_only"),
            scene_score=(
                float(scene_score) if isinstance(scene_score, (int, float)) else None
            ),
            core_total_checks=int(scene_summary.get("total_checks") or 0),
            core_pass_count=int(scene_summary.get("pass") or 0),
            core_degraded_count=int(scene_summary.get("degraded") or 0),
            core_fail_count=int(scene_summary.get("fail") or 0),
            core_unknown_count=int(scene_summary.get("unknown") or 0),
            metric_scores=metric_scores,
            results=results,
        )

    @staticmethod
    def _normalize_result(result: dict[str, Any]) -> CriticEvidenceResult:
        confidence = result.get("confidence")
        return CriticEvidenceResult(
            check_id=str(result.get("check_id") or ""),
            metric=str(result.get("metric") or "unknown"),
            label=str(result.get("label") or "unknown").strip().lower(),
            scoring_tier=str(result.get("scoring_tier") or "core").strip().lower(),
            primary_object=str(result.get("primary_object") or ""),
            related_objects=[
                str(item) for item in (result.get("related_objects") or [])
            ],
            reason=str(result.get("reason") or ""),
            repair_advice=str(result.get("repair_advice") or ""),
            confidence=(
                float(confidence) if isinstance(confidence, (int, float)) else None
            ),
        )
