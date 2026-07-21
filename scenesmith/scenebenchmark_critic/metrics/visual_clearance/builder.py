"""Builder entry point for the scene-level visual-clearance metric."""

from __future__ import annotations

from typing import Any


def build_visual_clearance_checks(
    _case_pack: dict[str, Any],
    _metrics: tuple[str, ...] | list[str] | None = None,
) -> list[dict[str, Any]]:
    """Visual clearance is evaluated as scene extensions, not per-object checks."""
    return []
