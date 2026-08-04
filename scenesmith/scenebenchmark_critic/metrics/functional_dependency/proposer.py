"""Compatibility surface for the removed geometry relation proposer.

The compiled v4 intent contract is the sole authority for functional
relations. These entry points remain so registry/plugin callers do not need a
parallel mode.
"""

from __future__ import annotations

from typing import Any, Callable

from scenesmith.scenebenchmark_critic.core.geometry import GeometryStore
from scenesmith.scenebenchmark_critic.core.models import FunctionalDependencyProposal

VlmProposer = Callable[..., list[FunctionalDependencyProposal]]


def augment_functional_dependency_checks(
    case_pack: dict[str, Any],
    config: Any,
    *,
    metric_filter: list[str] | None,
    progress=lambda _message: None,
    vlm_proposer: VlmProposer | None = None,
) -> bool:
    del case_pack, config, metric_filter, vlm_proposer
    progress("Using compiled intent-contract relations")
    return False


def propose_dependency_relations(
    case_pack: dict[str, Any],
    store: GeometryStore,
    config: Any,
    *,
    progress=lambda _message: None,
    vlm_proposer: VlmProposer | None = None,
) -> list[FunctionalDependencyProposal]:
    del case_pack, store, config, vlm_proposer
    progress("Using compiled intent-contract relations")
    return []


def _rank_targets_for_relation(
    subject: dict[str, Any],
    relation_type: str,
    targets: list[dict[str, Any]],
) -> list[str]:
    """Retained for an unused private helper in ``relations.py``."""
    del subject, relation_type
    return sorted(str(target.get("id") or "") for target in targets if target.get("id"))
