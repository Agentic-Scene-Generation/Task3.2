"""Public API for embedding the SceneBenchmark critic in SceneSmith."""

from __future__ import annotations

import logging
import json
import os
import threading
import time
from datetime import datetime, timezone

from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

from scenesmith.agent_utils.house import HouseScene
from scenesmith.agent_utils.room import ObjectType, RoomScene
from scenesmith.scenebenchmark_critic.adapter import (
    house_scene_to_case_pack,
    room_scene_to_case_pack,
)
from scenesmith.scenebenchmark_critic.config import CriticConfig, critic_config_from_any
from scenesmith.scenebenchmark_critic.intent_contract import (
    contract_seating_targets,
)
from scenesmith.scenebenchmark_critic.evaluator import run_case_pack_checks
from scenesmith.scenebenchmark_critic.reports import (
    build_evaluation_payload,
    format_prompt_context as _format_prompt_context,
    write_report,
)

if TYPE_CHECKING:
    from scenesmith.agent_utils.blender.server_manager import BlenderServer

console_logger = logging.getLogger(__name__)


def _critic_timing_path(root: Path | None) -> Path | None:
    """Resolve the durable JSONL destination for critic timing events."""
    override = os.environ.get("SCENEBENCHMARK_CRITIC_TIMING_PATH", "").strip()
    if override:
        return Path(override)
    if root is None:
        return None
    return root / "scene_expert" / "timing" / "scenebenchmark_critic_timing.jsonl"


def _scene_output_root(scene_or_house: Any) -> Path | None:
    """Return a scene/house output root without requiring a configured path."""
    for attribute in ("scene_dir", "house_dir"):
        try:
            value = getattr(scene_or_house, attribute, None)
        except Exception:
            value = None
        if value:
            try:
                return Path(value)
            except (TypeError, ValueError):
                continue
    return None


def _record_critic_timing(
    *,
    scene_or_house: Any,
    stage: str,
    scope: str,
    status: str,
    started_at: float,
    steps: dict[str, Any],
    details: dict[str, Any] | None = None,
    evaluation: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    """Append one independent timing record; never affect critic execution."""
    path = _critic_timing_path(_scene_output_root(scene_or_house))
    if path is None:
        return
    record: dict[str, Any] = {
        "schema_version": "scenesmith.scenebenchmark_critic.timing.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "thread_id": threading.get_ident(),
        "stage": stage,
        "scope": scope,
        "status": status,
        "elapsed_sec": round(time.perf_counter() - started_at, 6),
        "steps": steps,
        "details": details or {},
    }
    if error:
        record["error"] = error
    if evaluation is not None:
        record["evaluation"] = evaluation
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            stream.flush()
    except Exception as timing_error:  # pragma: no cover - audit must not break critic
        console_logger.warning(
            "Could not write SceneBenchmark critic timing %s: %s",
            path,
            timing_error,
        )


def evaluate_room_scene(
    scene: RoomScene,
    *,
    config: CriticConfig | Any | None = None,
    stage: str = "adhoc",
    raw_config: Any | None = None,
    annotate_assets: bool = False,
    blender_server: "BlenderServer | None" = None,
) -> dict[str, Any]:
    critic_config = _coerce_config(config)
    timing_start = time.perf_counter()
    timing_steps: dict[str, Any] = {}
    scope = f"room:{scene.room_id}"
    try:
        step_start = time.perf_counter()
        # This migration intentionally keeps the rule critic self-contained.
        # Asset annotation/VLM filtering stays outside this branch, so no model
        # request is made while building prompt feedback.
        case_pack = room_scene_to_case_pack(
            scene,
            stage=stage,
            metrics=list(critic_config.metrics),
        )
        timing_steps["case_pack_build_sec"] = round(time.perf_counter() - step_start, 6)
        step_start = time.perf_counter()
        # Prompt contracts own orientation. The stabilizer only fills asset
        # metadata and cannot create an additional layout requirement.
        timing_steps["orientation_contract_sec"] = round(
            time.perf_counter() - step_start, 6
        )
        check_timing: dict[str, Any] = {}
        step_start = time.perf_counter()
        results = run_case_pack_checks(
            case_pack, config=critic_config, timing=check_timing
        )
        timing_steps["check_execution_wall_sec"] = round(
            time.perf_counter() - step_start, 6
        )
        timing_steps.update(check_timing)
        step_start = time.perf_counter()
        payload = build_evaluation_payload(
            case_pack=case_pack,
            results=results,
            stage=stage,
            scope=scope,
            config=critic_config,
        )
        timing_steps["payload_build_sec"] = round(time.perf_counter() - step_start, 6)
        _record_critic_timing(
            scene_or_house=scene,
            stage=stage,
            scope=scope,
            status="ok",
            started_at=timing_start,
            steps=timing_steps,
            details={
                "metrics": list(critic_config.metrics),
                "case_pack_check_count": len(case_pack.get("checks") or []),
                "result_count": len(results),
            },
            evaluation={
                "results": payload.get("results", []),
                "summary": payload.get("summary", {}),
                "gate": payload.get("gate", {}),
            },
        )
        return payload
    except Exception as exc:
        _record_critic_timing(
            scene_or_house=scene,
            stage=stage,
            scope=scope,
            status="error",
            started_at=timing_start,
            steps=timing_steps,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise


def evaluate_house_scene(
    house: HouseScene,
    *,
    config: CriticConfig | Any | None = None,
    stage: str = "adhoc",
    include_object_types: list[ObjectType] | tuple[ObjectType, ...] | None = None,
) -> dict[str, Any]:
    critic_config = _coerce_config(config)
    timing_start = time.perf_counter()
    timing_steps: dict[str, Any] = {}
    scope = "house"
    try:
        step_start = time.perf_counter()
        case_pack = house_scene_to_case_pack(
            house,
            stage=stage,
            metrics=list(critic_config.metrics),
            include_object_types=include_object_types,
        )
        timing_steps["case_pack_build_sec"] = round(time.perf_counter() - step_start, 6)
        check_timing: dict[str, Any] = {}
        step_start = time.perf_counter()
        results = run_case_pack_checks(
            case_pack, config=critic_config, timing=check_timing
        )
        timing_steps["check_execution_wall_sec"] = round(
            time.perf_counter() - step_start, 6
        )
        timing_steps.update(check_timing)
        step_start = time.perf_counter()
        payload = build_evaluation_payload(
            case_pack=case_pack,
            results=results,
            stage=stage,
            scope=scope,
            config=critic_config,
        )
        timing_steps["payload_build_sec"] = round(time.perf_counter() - step_start, 6)
        _record_critic_timing(
            scene_or_house=house,
            stage=stage,
            scope=scope,
            status="ok",
            started_at=timing_start,
            steps=timing_steps,
            details={
                "metrics": list(critic_config.metrics),
                "case_pack_check_count": len(case_pack.get("checks") or []),
                "result_count": len(results),
                "include_object_types": [
                    str(item) for item in (include_object_types or [])
                ],
            },
            evaluation={
                "results": payload.get("results", []),
                "summary": payload.get("summary", {}),
                "gate": payload.get("gate", {}),
            },
        )
        return payload
    except Exception as exc:
        _record_critic_timing(
            scene_or_house=house,
            stage=stage,
            scope=scope,
            status="error",
            started_at=timing_start,
            steps=timing_steps,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise


def write_room_stage_report(
    scene: RoomScene,
    output_dir: Path,
    *,
    config: CriticConfig | Any | None = None,
    stage: str,
    raw_config: Any | None = None,
    blender_server: "BlenderServer | None" = None,
) -> dict[str, Any] | None:
    critic_config = _coerce_config(config)
    if not critic_config.enabled or not critic_config.room_stage_enabled(stage):
        return None
    payload = evaluate_room_scene(
        scene,
        config=critic_config,
        raw_config=raw_config or config,
        stage=stage,
        annotate_assets=False,
    )
    write_report(output_dir, payload)
    console_logger.info("SceneBenchmark critic report saved to %s", output_dir)
    return payload


def write_house_stage_report(
    house: HouseScene,
    output_dir: Path,
    *,
    config: CriticConfig | Any | None = None,
    stage: str,
    include_object_types: list[ObjectType] | tuple[ObjectType, ...] | None = None,
) -> dict[str, Any] | None:
    critic_config = _coerce_config(config)
    if not critic_config.enabled or not critic_config.house_stage_enabled(stage):
        return None
    payload = evaluate_house_scene(
        house,
        config=critic_config,
        stage=stage,
        include_object_types=include_object_types,
    )
    write_report(output_dir, payload)
    console_logger.info("SceneBenchmark critic report saved to %s", output_dir)
    return payload


def format_prompt_context(
    payload: dict[str, Any], *, max_issues: int | None = None
) -> str:
    if max_issues is None:
        max_issues = 8
    return _format_prompt_context(payload, max_issues=max_issues)


def seating_orientation_targets(
    scene: RoomScene,
    *,
    config: CriticConfig | Any | None = None,
) -> dict[str, set[str]] | None:
    """Return prompt-authorized direct seating targets in contract mode.

    The final furniture guard executes before a normal critic pass, so it
    cannot inspect an already-built payload.  Build only the local semantic
    case pack here; this makes a direct yaw change subject to the same original
    prompt contract as later geometry evaluation.
    """
    critic_config = _coerce_config(config)
    case_pack = room_scene_to_case_pack(
        scene,
        metrics=("functional_dependency",),
    )
    return contract_seating_targets(case_pack)


def _coerce_config(config: CriticConfig | Any | None) -> CriticConfig:
    if isinstance(config, CriticConfig):
        return config
    if config is None:
        return CriticConfig()
    return critic_config_from_any(config)
