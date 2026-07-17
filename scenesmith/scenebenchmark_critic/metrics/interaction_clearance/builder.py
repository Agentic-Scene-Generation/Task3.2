"""Check construction for interaction clearance."""

from __future__ import annotations

from typing import Any

from scenesmith.scenebenchmark_critic.metrics.functional_dependency.builder import (
    build_checks,
)


def build_interaction_clearance_checks(
    case_pack: dict[str, Any],
    metrics: tuple[str, ...] | list[str] | None = None,
) -> list[dict[str, Any]]:
    """Build clearance and window-opening checks."""
    selected = metrics or ("interaction_clearance",)
    return [
        check
        for check in build_checks(case_pack, metrics=selected)
        if check.get("metric") == "interaction_clearance"
    ]
