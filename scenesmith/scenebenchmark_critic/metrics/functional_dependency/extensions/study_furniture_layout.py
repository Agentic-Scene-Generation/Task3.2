"""Compatibility entry point for the removed implicit study layout rule."""

from __future__ import annotations

from typing import Any


def evaluate_study_furniture_layout(
    case_pack: dict[str, Any],
) -> list[dict[str, Any]]:
    del case_pack
    return []
