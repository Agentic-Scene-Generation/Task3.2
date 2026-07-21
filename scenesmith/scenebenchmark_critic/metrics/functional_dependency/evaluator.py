"""Functional-dependency check evaluator."""

from __future__ import annotations

from typing import Any

from scenesmith.scenebenchmark_critic.core.geometry import load_geometry
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.relations import (
    evaluate_functional_dependency as _evaluate_functional_dependency,
)


def evaluate_functional_dependency(
    case_pack: dict[str, Any],
    check: dict[str, Any],
    _config: Any | None = None,
) -> dict[str, Any] | None:
    store = load_geometry(case_pack)
    if store is None:
        return None
    return _evaluate_functional_dependency(store, check)
