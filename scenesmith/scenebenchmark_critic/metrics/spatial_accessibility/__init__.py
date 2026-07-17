"""Spatial-accessibility metric implementation."""

from scenesmith.scenebenchmark_critic.metrics.spatial_accessibility.builder import (
    build_spatial_accessibility_checks,
)
from scenesmith.scenebenchmark_critic.metrics.spatial_accessibility.evaluator import (
    evaluate_spatial_accessibility,
)

__all__ = [
    "build_spatial_accessibility_checks",
    "evaluate_spatial_accessibility",
]
