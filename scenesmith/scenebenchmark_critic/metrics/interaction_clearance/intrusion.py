"""Intrusion geometry for interaction-clearance checks."""

from scenesmith.scenebenchmark_critic.metrics.interaction_clearance.evaluator import (
    aabb_overlap_volume,
    aabb_volume,
    intrusions,
)

__all__ = ["aabb_overlap_volume", "aabb_volume", "intrusions"]
