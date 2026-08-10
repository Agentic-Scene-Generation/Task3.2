"""Interaction-clearance metric implementation."""

from scenesmith.scenebenchmark_critic.metrics.interaction_clearance.evaluator import (
    build_door_clearance_checks,
    evaluate_clearance,
    get_clearance,
    get_clearance_for_metadata,
)

__all__ = [
    "build_door_clearance_checks",
    "evaluate_clearance",
    "get_clearance",
    "get_clearance_for_metadata",
]
