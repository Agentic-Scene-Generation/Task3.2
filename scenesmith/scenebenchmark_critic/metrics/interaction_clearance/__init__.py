"""Interaction-clearance metric implementation."""

from scenesmith.scenebenchmark_critic.metrics.interaction_clearance.evaluator import (
    evaluate_clearance,
    get_clearance,
    get_clearance_for_metadata,
)

__all__ = ["evaluate_clearance", "get_clearance", "get_clearance_for_metadata"]
