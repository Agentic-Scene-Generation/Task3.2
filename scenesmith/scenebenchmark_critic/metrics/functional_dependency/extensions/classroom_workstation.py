"""Compatibility entry point for the removed classroom aggregate rule."""

from __future__ import annotations

from typing import Any


def evaluate_classroom_workstation_distribution(
    case_pack: dict[str, Any],
) -> list[dict[str, Any]]:
    del case_pack
    return []
