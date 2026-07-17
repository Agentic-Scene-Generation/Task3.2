"""Visual-clearance metric evaluator and scene extensions."""

from __future__ import annotations

from typing import Any

from scenesmith.scenebenchmark_critic.metrics.visual_clearance.furniture_occlusion import (
    evaluate_wall_mounted_visibility,
)
from scenesmith.scenebenchmark_critic.metrics.visual_clearance.wall_overlap import (
    evaluate_wall_mounted_overlap,
)


def evaluate_visual_clearance(case_pack: dict[str, Any]) -> list[dict[str, Any]]:
    """Run furniture occlusion and same-wall overlap exactly once."""
    return [
        *evaluate_wall_mounted_visibility(case_pack),
        *evaluate_wall_mounted_overlap(case_pack),
    ]
