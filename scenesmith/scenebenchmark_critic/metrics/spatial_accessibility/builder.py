"""Check construction for spatial accessibility."""

from __future__ import annotations

from typing import Any

from scenesmith.scenebenchmark_critic.metrics.functional_dependency.builder import (
    build_checks,
)


def build_spatial_accessibility_checks(
    case_pack: dict[str, Any],
    metrics: tuple[str, ...] | list[str] | None = None,
) -> list[dict[str, Any]]:
    """Build accessibility checks from the shared scene-object policy."""
    selected = metrics or ("spatial_accessibility",)
    return [
        check
        for check in build_checks(case_pack, metrics=selected)
        if check.get("metric") == "spatial_accessibility"
    ]
