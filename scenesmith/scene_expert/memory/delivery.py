"""Decision-boundary delivery through the existing SceneExpert context hook."""

from __future__ import annotations

import logging
import os

from typing import Any

from scenesmith.scene_expert.memory.adaptation import decision_applicability
from scenesmith.scene_expert.memory.injection import format_accepted_memory
from scenesmith.scene_expert.memory.state import build_memory_scene_state
from scenesmith.scene_expert.schemas import MemoryInjectionBundle, StageRelationContext

console_logger = logging.getLogger(__name__)


def prepare_memory_delivery(
    *,
    scene: Any | None,
    stage: str,
    agent_role: str,
    event: str,
    prompt: str,
    last_hard_issues: list[str] | None = None,
) -> dict:
    """Isolate optional memory errors without suppressing native context."""
    try:
        return _prepare_memory_delivery(
            scene=scene,
            stage=stage,
            agent_role=agent_role,
            event=event,
            prompt=prompt,
            last_hard_issues=last_hard_issues,
        )
    except Exception as exc:
        console_logger.warning("Memory decision delivery withheld: %s", exc)
        return {
            "status": "memory_delivery_error",
            "text": "",
            "decisions": [],
            "error_type": type(exc).__name__,
        }


def _prepare_memory_delivery(
    *,
    scene: Any | None,
    stage: str,
    agent_role: str,
    event: str,
    prompt: str,
    last_hard_issues: list[str] | None,
) -> dict:
    """Return advice for this request only; never touch scene, sessions or bank.

    The native request/critique is unchanged. No retrieval or Planner LLM is
    invoked for repairs. Delivery is not proof of application or causal gain.
    """
    if agent_role != "designer" or event not in {
        "request_initial_design",
        "request_design_change",
    }:
        return {}
    raw = getattr(scene, "scene_expert_accepted_memory_bundle", None)
    if not raw or not getattr(scene, "scene_expert_memory_delivery_enabled", False):
        return {}
    if os.environ.get(
        "SCENEEXPERT_INJECT_STAGE_CONTEXT_BUNDLE", "1"
    ).strip().lower() not in {"1", "true", "yes", "y", "on"}:
        return {"status": "context_injection_disabled", "text": "", "decisions": []}
    bundle = MemoryInjectionBundle.model_validate(raw)
    if bundle.stage != stage:
        return {"status": "stage_mismatch", "text": "", "decisions": []}
    current = build_memory_scene_state(scene)
    context_raw = (
        getattr(scene, "scene_expert_relation_context", None) or bundle.relation_context
    )
    context = StageRelationContext.model_validate(context_raw) if context_raw else None
    selected, decisions = [], []
    for item in bundle.accepted_items:
        reasons = decision_applicability(
            item,
            stage=stage,
            source_state=bundle.current_scene_state,
            current_state=current,
            query=prompt + "\n" + "\n".join(last_hard_issues or []),
            repair=event == "request_design_change",
            context=context,
        )
        normalized_prompt = " ".join(prompt.split())
        fragments = (
            item.adaptation.preconditions
            + item.adaptation.actions
            + item.adaptation.checks
        )
        already_present = not reasons and (
            " ".join(item.text.split()) in normalized_prompt
            or (
                bool(fragments)
                and all(
                    " ".join(value.split()) in normalized_prompt
                    for value in fragments
                    if value.strip()
                )
            )
        )
        if not reasons and not already_present:
            selected.append(item)
        decisions.append(
            {
                "memory_type": item.source.memory_type,
                "memory_id": item.source.memory_id,
                "content_hash": item.source.content_hash,
                "decision": (
                    "withheld"
                    if reasons
                    else ("already_in_request" if already_present else "delivered")
                ),
                "reasons": reasons,
            }
        )
    return {
        "schema_version": "memory-delivery.v1",
        "status": "prepared",
        "stage": stage,
        "event": event,
        "state_fingerprint": current["fingerprint"],
        "text": format_accepted_memory(selected),
        "decisions": decisions,
        "application_observed": None,
    }
