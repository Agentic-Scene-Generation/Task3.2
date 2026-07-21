"""Public API for embedding the SceneBenchmark critic in SceneSmith."""

from __future__ import annotations

import logging

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
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.orientation_contracts import (
    stabilize_orientation_contracts,
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
    # This migration intentionally keeps the rule critic self-contained.  Asset
    # annotation/VLM filtering stays outside this branch, so no model request is
    # made while building prompt feedback.
    case_pack = room_scene_to_case_pack(
        scene, stage=stage, metrics=list(critic_config.metrics)
    )
    stabilize_orientation_contracts(
        case_pack,
        scene,
        critic_config,
        stage=stage,
    )
    results = run_case_pack_checks(case_pack, config=critic_config)
    return build_evaluation_payload(
        case_pack=case_pack,
        results=results,
        stage=stage,
        scope=f"room:{scene.room_id}",
        config=critic_config,
    )


def evaluate_house_scene(
    house: HouseScene,
    *,
    config: CriticConfig | Any | None = None,
    stage: str = "adhoc",
    include_object_types: list[ObjectType] | tuple[ObjectType, ...] | None = None,
) -> dict[str, Any]:
    critic_config = _coerce_config(config)
    case_pack = house_scene_to_case_pack(
        house,
        stage=stage,
        metrics=list(critic_config.metrics),
        include_object_types=include_object_types,
    )
    results = run_case_pack_checks(case_pack, config=critic_config)
    return build_evaluation_payload(
        case_pack=case_pack,
        results=results,
        stage=stage,
        scope="house",
        config=critic_config,
    )


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


def _coerce_config(config: CriticConfig | Any | None) -> CriticConfig:
    if isinstance(config, CriticConfig):
        return config
    if config is None:
        return CriticConfig()
    return critic_config_from_any(config)
