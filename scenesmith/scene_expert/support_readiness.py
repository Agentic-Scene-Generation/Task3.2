"""Produce native geometric support evidence before the stage commit gate."""

from __future__ import annotations

import logging
import time
from typing import Any

from omegaconf import OmegaConf

from scenesmith.agent_utils.room import (
    RoomScene,
    extract_and_propagate_support_surfaces,
)
from scenesmith.agent_utils.support_surface_extraction import (
    SupportSurfaceExtractionConfig,
)

console_logger = logging.getLogger(__name__)


def prepare_stage_support_evidence(
    scene: RoomScene,
    payload: dict[str, Any],
    config: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Extract existing native surfaces for failed, present support targets.

    The verifier's requirements and labels are unchanged. Empty annotations,
    extraction errors and unsuitable geometry remain failures; inferred bbox
    surfaces are never created here. Ordinary unknown results and the special
    upholstered-seat policy remain owned by their existing consumers.
    """
    if not config:
        return []
    target_ids = {
        str(row.get("primary_object") or "")
        for row in payload.get("results", [])
        if row.get("label") == "fail"
        and row.get("diagnostics", {}).get("support_readiness") is True
    }
    if not target_ids:
        return []
    hsm_config = SupportSurfaceExtractionConfig.from_config(OmegaConf.create(config))
    evidence: list[dict[str, Any]] = []
    for object_id in sorted(target_ids):
        obj = next(
            (obj for key, obj in scene.objects.items() if str(key) == object_id), None
        )
        if obj is None or obj.support_surfaces:
            continue
        started = time.perf_counter()
        row: dict[str, Any] = {
            "object_id": object_id,
            "geometry_path": str(obj.geometry_path),
            "surface_policy": "general",
            "source": "native_support_extraction",
        }
        try:
            surfaces = extract_and_propagate_support_surfaces(
                scene=scene, furniture_object=obj, config=hsm_config
            )
            row.update(
                status="extracted" if surfaces else "empty", surfaces=len(surfaces)
            )
        except (OSError, ValueError) as error:
            row.update(status="error", surfaces=0, error=str(error))
            console_logger.warning(
                "Support evidence unavailable for %s: %s", object_id, error
            )
        row["elapsed_sec"] = time.perf_counter() - started
        evidence.append(row)
    return evidence
