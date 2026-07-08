"""Base class for stateful agents using planner/designer/critic workflow.

This module provides the shared framework for all design agents (floor plan,
furniture, wall, manipuland), extracting the common multi-agent architecture
while allowing domain-specific customization through abstract methods and
subclass-defined tools.
"""

import copy
import logging
import os
import shutil
import time

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import yaml

import os

from agents import (
    Agent,
    FunctionTool,
    ModelSettings,
    RunConfig,
    Runner,
    RunResult,
    SQLiteSession,
    function_tool,
)
from agents.memory.session import Session
from agents.models.openai_provider import OpenAIProvider
from omegaconf import DictConfig
from openai import Timeout

from scenesmith.agent_utils.action_logger import log_scene_action
from scenesmith.agent_utils.checkpoint_state import initialize_checkpoint_attributes
from scenesmith.agent_utils.furniture_safety import (
    FurnitureSafetyController,
    HardStateEvaluation,
)
from scenesmith.agent_utils.intra_turn_image_filter import IntraTurnImageFilter
from scenesmith.agent_utils.physics_tools import check_physics_violations
from scenesmith.agent_utils.placement_noise import PlacementNoiseMode
from scenesmith.agent_utils.room import AgentType
from scenesmith.agent_utils.scoring import (
    CategoryScore,
    CeilingCritiqueWithScores,
    CritiqueWithScores,
    FurnitureCritiqueWithScores,
    ManipulandCritiqueWithScores,
    WallCritiqueWithScores,
    compute_total_score,
    format_score_deltas_for_planner,
    log_agent_response,
    log_critique_scores,
    scores_to_dict,
)
from scenesmith.agent_utils.stage_working_memory import StageWorkingMemory
from scenesmith.scene_expert.context_bundle import build_stage_context_bundle
from scenesmith.agent_utils.thinking import (
    prepend_text_thinking_directive,
    thinking_directive_from_effort,
)
from scenesmith.agent_utils.turn_trimming_session import TurnTrimmingSession
from scenesmith.prompts import prompt_registry
from scenesmith.utils.logging import BaseLogger
from scenesmith.utils.openai import encode_image_to_base64

console_logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    try:
        if hasattr(cfg, "get"):
            return cfg.get(key, default)
    except Exception:
        pass
    return getattr(cfg, key, default)


def log_agent_usage(result: RunResult, agent_name: str) -> None:
    """Log token usage from an agent run.

    Args:
        result: The RunResult from Runner.run().
        agent_name: Human-readable name for the agent (e.g., "DESIGNER", "CRITIC").
    """
    usage = result.context_wrapper.usage
    cached = (
        usage.input_tokens_details.cached_tokens if usage.input_tokens_details else 0
    )
    reasoning = (
        usage.output_tokens_details.reasoning_tokens
        if usage.output_tokens_details
        else 0
    )
    # Get final context size from last request (context only grows during a run).
    final_context = (
        usage.request_usage_entries[-1].input_tokens
        if usage.request_usage_entries
        else usage.input_tokens
    )
    console_logger.info(
        f"[{agent_name}] Token usage: "
        f"input={usage.input_tokens:,}, "
        f"output={usage.output_tokens:,}, "
        f"reasoning={reasoning:,}, "
        f"cached={cached:,}, "
        f"total={usage.total_tokens:,}, "
        f"requests={usage.requests}, "
        f"final_context_length={final_context:,}"
    )


class BaseStatefulAgent(ABC):
    """Base class for stateful agents with planner/designer/critic workflow.

    This class provides the shared framework for multi-agent design workflows,
    including:
    - Session management (SQLiteSession for persistent conversation history)
    - Checkpoint state initialization and rollback functionality
    - Agent creation patterns (planner, designer, critic)
    - Shared configuration and logging infrastructure

    Domain-specific behavior is implemented through abstract methods and
    subclass-defined tools/prompts, keeping the framework general while
    allowing specialization.

    Required attributes (initialized by subclasses):
    - self.scene: Scene object with restore_from_state_dict() method
    - self.rendering_manager: RenderingManager with clear_cache() method
    - self.previous_scene_checkpoint: Previous scene state dict
    - self.scene_checkpoint: Current scene state dict
    - self.previous_checkpoint_scores: Previous scores
    - self.checkpoint_scores: Current scores
    - self.previous_scores: Scores from last iteration
    - self.previous_checkpoint_render_dir: Previous render directory
    - self.checkpoint_render_dir: Current render directory
    - self.cfg: Config with reset thresholds
    """

    # Whether this agent places objects (includes placement style tool).
    # Override to False in floor plan agent which doesn't place objects.
    _is_placement_agent: bool = True

    @property
    @abstractmethod
    def agent_type(self) -> AgentType:
        """Return the type of this agent for collision filtering.

        Each agent type can only modify certain object types:
        - FURNITURE: Floor-standing furniture
        - MANIPULAND: Objects placed on furniture surfaces
        - WALL_MOUNTED: Objects mounted on walls
        - CEILING_MOUNTED: Objects mounted on ceilings

        Returns:
            AgentType for this agent.
        """

    def __init__(
        self,
        cfg: DictConfig,
        logger: BaseLogger,
        geometry_server_host: str = "127.0.0.1",
        geometry_server_port: int = 7000,
        hssd_server_host: str = "127.0.0.1",
        hssd_server_port: int = 7001,
    ):
        """Initialize base placement agent with shared infrastructure.

        Args:
            cfg: Hydra configuration object.
            logger: Logger for experiment tracking.
            geometry_server_host: Host for geometry generation server.
            geometry_server_port: Port for geometry generation server.
            hssd_server_host: Host for HSSD retrieval server.
            hssd_server_port: Port for HSSD retrieval server.
        """
        self.cfg = cfg
        self.logger = logger
        self.geometry_server_host = geometry_server_host
        self.geometry_server_port = geometry_server_port
        self.hssd_server_host = hssd_server_host
        self.hssd_server_port = hssd_server_port

        # Use global prompt registry (same pattern as domain base classes).
        self.prompt_registry = prompt_registry

        # Initialize checkpoint state (N-1 and N pattern for rollback).
        initialize_checkpoint_attributes(target=self)

        safety_cfg = getattr(cfg, "furniture_safety_controller", None)
        self.furniture_safety_controller = FurnitureSafetyController(safety_cfg)
        self._planner_initial_design_tool_calls = 0
        self._planner_critique_tool_calls = 0
        self._planner_design_change_tool_calls = 0
        self._planner_budget_exhausted = False
        self._critic_failed = False
        working_memory_cfg = _cfg_get(cfg, "stage_working_memory", {})
        working_memory_enabled = bool(
            _cfg_get(working_memory_cfg, "enabled", True)
        )
        self.stage_working_memory = StageWorkingMemory(
            root_dir=logger.output_dir,
            stage=self.agent_type.value,
            enabled=working_memory_enabled,
        )
        self._critic_candidate_cache: dict[str, Any] = {}
        self._critic_output_type: type[CritiqueWithScores] | None = None
        self._last_scored_scene_hash: str | None = None
        self._last_critique_render_profile = "final"
        self._pending_hard_repair_hint = ""
        self._hard_repair_design_change_calls = 0

    def _record_module_timing(
        self,
        module: str,
        event: str,
        start_time: float,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Record elapsed time for per-stage optimization analysis."""
        elapsed = time.time() - start_time
        try:
            self.stage_working_memory.record_timing(
                module=module,
                event=event,
                elapsed_sec=elapsed,
                extra=extra,
            )
        except Exception as e:
            console_logger.warning("Failed to record timing %s/%s: %s", module, event, e)

    def _retrieve_working_memory_for_designer(self, query: str) -> str:
        """Fetch compact online memory to inject into the next designer call."""
        try:
            memory_text = self.stage_working_memory.retrieve_for_designer(
                query=query,
                max_items=int(
                    _cfg_get(
                        _cfg_get(self.cfg, "stage_working_memory", {}),
                        "max_retrieved_items",
                        3,
                    )
                ),
            )
        except Exception as e:
            console_logger.warning("Stage working memory retrieval failed: %s", e)
            return ""
        if memory_text:
            console_logger.info(
                "[StageWorkingMemory] injecting %d chars into designer prompt",
                len(memory_text),
            )
        return memory_text

    def _stage_context_max_chars(self) -> int:
        try:
            return int(os.environ.get("SCENEEXPERT_STAGE_CONTEXT_MAX_CHARS", "2400"))
        except ValueError:
            return 2400

    def _stage_context_injection_enabled(self) -> bool:
        value = os.environ.get("SCENEEXPERT_INJECT_STAGE_CONTEXT_BUNDLE", "1")
        return value.strip().lower() in ("1", "true", "yes", "y", "on")

    def _prepare_stage_context_for_llm(
        self,
        *,
        agent_role: str,
        event: str,
        prompt: Any,
        last_hard_issues: list[str] | None = None,
    ) -> str:
        """Build, save, and optionally return a compact StageContextBundle."""
        try:
            bundle = build_stage_context_bundle(
                stage=self.agent_type.value,
                agent_role=agent_role,
                event=event,
                scene=getattr(self, "scene", None),
                history_summary=getattr(self, "_pending_hard_repair_hint", ""),
                last_hard_issues=last_hard_issues
                or ([self._pending_hard_repair_hint] if self._pending_hard_repair_hint else []),
                prompt=prompt,
                metadata={
                    "scene_expert_stage": getattr(
                        getattr(self, "scene", None),
                        "scene_expert_stage",
                        self.agent_type.value,
                    ),
                    "scene_expert_task_spec": getattr(
                        getattr(self, "scene", None),
                        "scene_expert_task_spec",
                        {},
                    ),
                    "scene_expert_brief": getattr(
                        getattr(self, "scene", None),
                        "scene_expert_brief",
                        "",
                    ),
                },
            )
            self.stage_working_memory.save_context_bundle(bundle)
            if not self._stage_context_injection_enabled():
                return ""
            return bundle.to_llm_text(max_chars=self._stage_context_max_chars())
        except Exception as e:
            console_logger.warning("Failed to prepare StageContextBundle: %s", e)
            return ""

    def _record_llm_call_debug(
        self,
        *,
        agent_role: str,
        event: str,
        prompt: Any,
        output: Any = "",
        result: Any = None,
        error: str = "",
    ) -> None:
        try:
            self.stage_working_memory.record_llm_call(
                agent_role=agent_role,
                event=event,
                prompt=prompt,
                output=output,
                result=result,
                error=error,
            )
        except Exception as e:
            console_logger.warning("Failed to record LLM call debug: %s", e)

    def _save_designer_working_memory(
        self,
        *,
        render_dir_before: Path | None,
        event: str,
        text: str,
    ) -> None:
        """Attach designer output to a newly created render, if one exists."""
        render_dir = self.rendering_manager.last_render_dir
        if render_dir is None or render_dir == render_dir_before:
            return
        try:
            self.stage_working_memory.save_render_record(
                render_dir=render_dir,
                role="designer",
                event=event,
                scene=self.scene,
                text=text,
            )
        except Exception as e:
            console_logger.warning("Failed to save designer working memory: %s", e)

    def _save_critic_working_memory(
        self,
        *,
        render_dir: Path | None,
        event: str,
        scores: CritiqueWithScores,
        critique: str,
        physics_context: str,
    ) -> None:
        """Attach critic scores and critique text to the current render memory."""
        if render_dir is None:
            return
        try:
            self.stage_working_memory.save_render_record(
                render_dir=render_dir,
                role="critic",
                event=event,
                scene=self.scene,
                scores=scores,
                critique=critique,
                extra={"physics_context": physics_context[:1500]},
            )
        except Exception as e:
            console_logger.warning("Failed to save critic working memory: %s", e)

    def _write_scores_and_memory(
        self,
        *,
        response: CritiqueWithScores,
        images_dir: Path | None,
        physics_context: str,
        event: str = "critique",
    ) -> None:
        """Persist scores.yaml and stage working memory for a critiqued render."""
        if images_dir:
            scores_dict = scores_to_dict(response)
            scores_path = images_dir / "scores.yaml"
            with open(scores_path, "w") as f:
                yaml.dump(
                    data=scores_dict,
                    stream=f,
                    default_flow_style=False,
                    sort_keys=False,
                )
            console_logger.info(f"Scores saved to: {scores_path}")
            self._save_critic_working_memory(
                render_dir=images_dir,
                event=event,
                scores=response,
                critique=response.critique,
                physics_context=physics_context,
            )
        else:
            console_logger.error(
                "No render directory available - scores not saved to file"
            )

    def _configure_furniture_safety_for_scene(self, scene_description: str) -> None:
        """Reset furniture safety counters and required-object inference."""
        controller = getattr(self, "furniture_safety_controller", None)
        if controller and controller.enabled:
            controller.reset_for_scene(scene_description=scene_description)
            try:
                self.stage_working_memory.set_required_counts(
                    controller.required_counts
                )
            except Exception as e:
                console_logger.warning(
                    "Failed to configure stage working memory requirements: %s",
                    e,
                )

    def _record_furniture_design_change_budget(self) -> str | None:
        """Gate planner-requested furniture design changes."""
        controller = getattr(self, "furniture_safety_controller", None)
        if not controller or not controller.enabled:
            return None

        allowed, message = controller.record_design_change(
            has_prior_critique=self.previous_scores is not None
        )
        if allowed:
            return None

        console_logger.info(message)
        return message

    def _evaluate_current_furniture_hard_state(
        self, physics_context: str | None = None
    ) -> HardStateEvaluation | None:
        """Run deterministic furniture hard checks for the current scene."""
        controller = getattr(self, "furniture_safety_controller", None)
        if not controller or not controller.enabled:
            return None
        if physics_context is None:
            current_furniture_id = getattr(self, "current_furniture_id", None)
            physics_context = check_physics_violations(
                scene=self.scene,
                cfg=self.cfg,
                current_furniture_id=current_furniture_id,
                agent_type=self.agent_type,
            )
        return controller.evaluate_scene_state(
            scene=self.scene,
            physics_context=physics_context,
        )

    def _critic_fast_path_cfg(self) -> Any:
        return _cfg_get(self.cfg, "critic_fast_path", {})

    def _critic_fast_path_enabled(self, key: str, default: bool = True) -> bool:
        return bool(_cfg_get(self._critic_fast_path_cfg(), key, default))

    def _hard_repair_design_change_limit(self) -> int:
        return int(
            _cfg_get(
                self._critic_fast_path_cfg(),
                "max_hard_repair_design_changes",
                1,
            )
        )

    def _hard_repair_allowance_available(self) -> bool:
        return (
            bool(self._pending_hard_repair_hint)
            and self._hard_repair_design_change_calls
            < self._hard_repair_design_change_limit()
        )

    def _deterministic_repair_enabled(self) -> bool:
        controller_cfg = _cfg_get(self.cfg, "furniture_safety_controller", {})
        repair_cfg = _cfg_get(controller_cfg, "deterministic_repair", {})
        return bool(_cfg_get(repair_cfg, "enabled", False))

    def _deterministic_repair_max_attempts(self) -> int:
        controller_cfg = _cfg_get(self.cfg, "furniture_safety_controller", {})
        repair_cfg = _cfg_get(controller_cfg, "deterministic_repair", {})
        return max(1, int(_cfg_get(repair_cfg, "max_attempts", 2)))

    def _attempt_deterministic_repair(
        self, hard_state: HardStateEvaluation
    ) -> tuple[bool, list[str]]:
        """Stage-specific hook for code-level hard-fail repair.

        Subclasses can override this to repair deterministic geometry failures
        before involving the LLM again. The base implementation is deliberately
        inert so non-furniture stages keep their existing behavior.
        """
        return False, []

    def _try_deterministic_repair_for_hard_state(
        self,
        hard_state: HardStateEvaluation | None,
        *,
        source: str,
    ) -> tuple[HardStateEvaluation | None, str | None, list[str]]:
        if hard_state is None or hard_state.hard_valid:
            return hard_state, None, []
        if not self._deterministic_repair_enabled():
            return hard_state, None, []

        current_state = hard_state
        physics_context: str | None = None
        all_actions: list[str] = []
        max_attempts = self._deterministic_repair_max_attempts()
        for attempt in range(1, max_attempts + 1):
            before_hash = self.scene.content_hash() if self.scene is not None else ""
            repair_start = time.time()
            repaired, actions = self._attempt_deterministic_repair(current_state)
            all_actions.extend(actions)
            self._record_module_timing(
                "deterministic_repair",
                source,
                repair_start,
                extra={
                    "attempt": attempt,
                    "max_attempts": max_attempts,
                    "attempted": True,
                    "repaired": bool(repaired),
                    "actions": actions,
                    "hard_reasons": current_state.hard_reasons,
                },
            )
            if not repaired:
                break

            console_logger.info(
                "Deterministic repair attempt %d/%d from %s: %s",
                attempt,
                max_attempts,
                source,
                "; ".join(actions) if actions else "(no action details)",
            )
            self.rendering_manager.clear_cache()
            self._reset_critic_candidate_cache()
            physics_context = self._get_cached_physics_context()
            repaired_hard_state = self._evaluate_current_hard_state(
                physics_context=physics_context
            )
            if repaired_hard_state is not None and repaired_hard_state.hard_valid:
                console_logger.info(
                    "Deterministic repair resolved hard-check failure from %s "
                    "after %d attempt(s)",
                    source,
                    attempt,
                )
                return repaired_hard_state, physics_context, all_actions

            after_hash = self.scene.content_hash() if self.scene is not None else ""
            current_state = repaired_hard_state or current_state
            if after_hash == before_hash:
                console_logger.info(
                    "Deterministic repair made no scene-state change on attempt "
                    "%d/%d; stopping repair loop.",
                    attempt,
                    max_attempts,
                )
                break

        remaining = (
            "; ".join(current_state.hard_reasons)
            if current_state and current_state.hard_reasons
            else "unknown remaining hard failure"
        )
        console_logger.info(
            "Deterministic repair did not fully resolve hard-check failure "
            "from %s: %s",
            source,
            remaining,
        )
        return current_state, physics_context, all_actions

    def _reset_critic_candidate_cache(self) -> None:
        self._critic_candidate_cache = {
            "scene_hash": self.scene.content_hash(),
        }

    def _cache_valid_for_current_scene(self) -> bool:
        return self._critic_candidate_cache.get("scene_hash") == self.scene.content_hash()

    def _get_cached_physics_context(self) -> str:
        if self._cache_valid_for_current_scene() and "physics_context" in self._critic_candidate_cache:
            return self._critic_candidate_cache["physics_context"]

        current_furniture_id = getattr(self, "current_furniture_id", None)
        physics_start = time.time()
        physics_context = check_physics_violations(
            scene=self.scene,
            cfg=self.cfg,
            current_furniture_id=current_furniture_id,
            agent_type=self.agent_type,
        )
        self._record_module_timing("critic", "physics_context", physics_start)
        self._critic_candidate_cache["physics_context"] = physics_context
        return physics_context

    def _parse_generic_physics_hard_reasons(self, physics_context: str) -> list[str]:
        text = str(physics_context or "").lower()
        if "no physics violations detected" in text:
            return []
        hard_reasons: list[str] = []
        hard_sections = (
            "collisions (",
            "thin covering overlaps",
            "thin covering boundary violations",
            "door clearance violations",
            "open connection blocked",
            "wall height exceeded",
            "geometry construction failed",
            "drake/qhull geometry construction failed",
        )
        for section in hard_sections:
            if section in text:
                hard_reasons.append(f"physics hard violation: {section.rstrip(' (')}")
        fallen_terms = ("fell off", "fallen", "below floor", "floor penetration")
        if any(term in text for term in fallen_terms):
            hard_reasons.append("fallen or below-floor object indicated by physics")
        return hard_reasons

    def _evaluate_current_hard_state(
        self, physics_context: str | None = None
    ) -> HardStateEvaluation | None:
        """Run deterministic hard checks for the current candidate.

        Furniture uses the richer FurnitureSafetyController. Other placement
        stages use physics hard sections only, so soft window warnings still go
        to the VLM critic.
        """
        if physics_context is None:
            physics_context = self._get_cached_physics_context()

        controller = getattr(self, "furniture_safety_controller", None)
        if controller and controller.enabled:
            return self._evaluate_current_furniture_hard_state(
                physics_context=physics_context
            )

        hard_reasons = self._parse_generic_physics_hard_reasons(physics_context)
        if not hard_reasons:
            return HardStateEvaluation(hard_valid=True)
        return HardStateEvaluation(hard_valid=False, hard_reasons=hard_reasons)

    def _repair_hint_from_hard_state(
        self, hard_state: HardStateEvaluation | None
    ) -> str:
        if hard_state is None or hard_state.hard_valid:
            return ""
        reasons = hard_state.hard_reasons or ["unknown deterministic hard failure"]
        missing = [
            reason
            for reason in reasons
            if reason.lower().startswith("missing required")
        ]
        hints: list[str] = []
        if missing:
            missing_text = "; ".join(missing)
            hints.append(
                "Missing required objects detected. The next designer action must "
                f"generate/place these objects before any decorative changes: {missing_text}."
            )
        if any("collision" in reason.lower() for reason in reasons):
            hints.append(
                "Resolve collisions by moving, snapping, or reducing only the involved "
                "modifiable objects; do not delete required prompt objects."
            )
        if any("door" in reason.lower() or "open connection" in reason.lower() for reason in reasons):
            hints.append(
                "Clear door/open-connection clearance first; keep a walkable path from "
                "the doorway into the room."
            )
        if any("fallen" in reason.lower() or "below-floor" in reason.lower() for reason in reasons):
            hints.append(
                "Restore fallen or below-floor objects onto a valid support surface, "
                "or remove only optional unstable small objects."
            )
        if any(
            "geometry construction" in reason.lower()
            or "drake/qhull" in reason.lower()
            for reason in reasons
        ):
            hints.append(
                "A deterministic geometry construction error occurred. Replace or "
                "repair the problematic asset if it is required; otherwise roll back "
                "to the last hard-valid checkpoint before continuing."
            )
        if not hints:
            hints.append(
                "Repair the listed hard violations first, then request another critique."
            )
        return " ".join(hints)

    def _make_category_score(self, name: str, grade: int, comment: str) -> CategoryScore:
        return CategoryScore(name=name, grade=max(0, min(10, int(grade))), comment=comment)

    def _make_deterministic_critique_scores(
        self,
        *,
        hard_state: HardStateEvaluation,
        physics_context: str,
    ) -> CritiqueWithScores:
        """Create a low synthetic score when hard checks fail before VLM scoring."""
        output_type = self._critic_output_type or type(self.previous_scores)
        reasons = hard_state.hard_reasons or ["deterministic hard-check failure"]
        reason_text = "; ".join(reasons)
        repair_hint = self._repair_hint_from_hard_state(hard_state)
        critique = (
            "DETERMINISTIC HARD-CHECK FAILED BEFORE VLM SCORING. "
            f"Hard issues: {reason_text}. {repair_hint} "
            "Do not finish the stage until these hard issues are repaired."
        )
        hard_comment = f"Hard fail: {reason_text}"
        repair_comment = repair_hint or hard_comment

        if output_type is FurnitureCritiqueWithScores:
            return FurnitureCritiqueWithScores(
                critique=critique,
                realism=self._make_category_score("realism", 3, hard_comment),
                functionality=self._make_category_score("functionality", 2, repair_comment),
                layout=self._make_category_score("layout", 3, hard_comment),
                layout_plausibility=self._make_category_score(
                    "layout_plausibility", 2, hard_comment
                ),
                holistic_completeness=self._make_category_score(
                    "holistic_completeness", 2, repair_comment
                ),
                prompt_following=self._make_category_score(
                    "prompt_following", 2, repair_comment
                ),
                reachability=self._make_category_score("reachability", 2, hard_comment),
            )
        if output_type is ManipulandCritiqueWithScores:
            return ManipulandCritiqueWithScores(
                critique=critique,
                realism=self._make_category_score("realism", 3, hard_comment),
                functionality=self._make_category_score("functionality", 2, repair_comment),
                layout=self._make_category_score("layout", 3, hard_comment),
                holistic_completeness=self._make_category_score(
                    "holistic_completeness", 2, repair_comment
                ),
                prompt_following=self._make_category_score(
                    "prompt_following", 3, repair_comment
                ),
            )
        if output_type is WallCritiqueWithScores:
            return WallCritiqueWithScores(
                critique=critique,
                realism=self._make_category_score("realism", 3, hard_comment),
                functionality=self._make_category_score("functionality", 2, repair_comment),
                layout=self._make_category_score("layout", 3, hard_comment),
                holistic_completeness=self._make_category_score(
                    "holistic_completeness", 3, repair_comment
                ),
                prompt_following=self._make_category_score(
                    "prompt_following", 3, repair_comment
                ),
            )
        if output_type is CeilingCritiqueWithScores:
            return CeilingCritiqueWithScores(
                critique=critique,
                realism=self._make_category_score("realism", 3, hard_comment),
                functionality=self._make_category_score("functionality", 2, repair_comment),
                layout=self._make_category_score("layout", 3, hard_comment),
                prompt_following=self._make_category_score(
                    "prompt_following", 3, repair_comment
                ),
            )
        raise TypeError(
            "Cannot create deterministic critique for output type "
            f"{getattr(output_type, '__name__', output_type)}"
        )

    def _critic_render_profile_name(self, update_checkpoint: bool) -> str:
        if (
            update_checkpoint
            and self._critic_fast_path_enabled("use_intermediate_render_profile", True)
        ):
            rendering_cfg = getattr(self.cfg, "rendering", None)
            profile_cfg = getattr(rendering_cfg, "intermediate_profile", None)
            if profile_cfg is not None and bool(getattr(profile_cfg, "enabled", False)):
                return "intermediate"
        return "final"

    def _can_skip_final_critique(self, current_scene_hash: str) -> bool:
        return (
            self.checkpoint_scene_hash is not None
            and current_scene_hash == self.checkpoint_scene_hash
            and self._last_scored_scene_hash == current_scene_hash
            and self._last_critique_render_profile == "final"
        )

    def _get_critic_scene_state_direct(self) -> str | None:
        if not self._critic_fast_path_enabled("direct_scene_state_cache", True):
            return None
        if self._cache_valid_for_current_scene() and "scene_state" in self._critic_candidate_cache:
            return self._critic_candidate_cache["scene_state"]

        owner = getattr(self, "_critic_scene_tools", None)
        if owner is None:
            return None
        method = getattr(owner, "_get_current_scene_state_impl", None)
        if method is None:
            method = getattr(owner, "_get_current_scene_impl", None)
        if method is None:
            return None
        scene_state_start = time.time()
        scene_state = method()
        self._record_module_timing(
            "critic", "get_current_scene_state_direct", scene_state_start
        )
        self._critic_candidate_cache["scene_state"] = scene_state
        return scene_state

    def _observe_scene_for_synthetic_score(self, render_profile: str) -> Path | None:
        owner = getattr(self, "_critic_vision_tools", None)
        method = getattr(owner, "_observe_scene_impl", None) if owner is not None else None
        if method is None:
            return self.rendering_manager.last_render_dir
        observe_start = time.time()
        try:
            with self.rendering_manager.use_render_profile(render_profile):
                method()
        except Exception as exc:
            console_logger.warning(
                "Synthetic hard-fail render failed; preserving previous render dir: %s",
                exc,
            )
        self._record_module_timing(
            "critic",
            "observe_scene_synthetic",
            observe_start,
            extra={"render_profile": render_profile},
        )
        return self.rendering_manager.last_render_dir

    def _begin_furniture_design_transaction(
        self, call_kind: str
    ) -> dict[str, Any] | None:
        """Start a guarded furniture designer call with a rollback snapshot."""
        controller = getattr(self, "furniture_safety_controller", None)
        if not controller or not controller.enabled:
            return None

        controller.begin_designer_call(call_kind=call_kind)
        pre_state = copy.deepcopy(self.scene.to_state_dict())
        pre_hard = self._evaluate_current_furniture_hard_state()
        if pre_hard and pre_hard.hard_valid:
            controller.remember_hard_valid_scene_state(
                scene_state=pre_state,
                source=f"pre-{call_kind}",
            )

        return {
            "call_kind": call_kind,
            "pre_state": pre_state,
            "pre_hard_valid": bool(pre_hard and pre_hard.hard_valid),
        }

    def _restore_furniture_scene_state(self, scene_state: dict[str, Any]) -> None:
        self.scene.restore_from_state_dict(scene_state)
        self.rendering_manager.clear_cache()

    def _end_furniture_design_transaction(
        self, transaction: dict[str, Any] | None
    ) -> str:
        """Validate a designer call and rollback if hard constraints fail."""
        if transaction is None:
            return ""

        controller = getattr(self, "furniture_safety_controller", None)
        if not controller or not controller.enabled:
            return ""

        try:
            hard_eval = self._evaluate_current_furniture_hard_state()
            call_kind = transaction["call_kind"]
            if hard_eval and hard_eval.hard_valid:
                controller.remember_hard_valid_scene_state(
                    scene_state=copy.deepcopy(self.scene.to_state_dict()),
                    source=f"post-{call_kind}",
                )
                return (
                    "\n\n**Safety Controller:** deterministic hard checks passed "
                    f"after {call_kind} designer call."
                )

            reasons = (
                "; ".join(hard_eval.hard_reasons)
                if hard_eval and hard_eval.hard_reasons
                else "unknown deterministic hard-check failure"
            )
            rollback_state = controller.best_scene_state
            rollback_source = "best hard-valid checkpoint"
            if rollback_state is None and transaction.get("pre_hard_valid"):
                rollback_state = transaction["pre_state"]
                rollback_source = "pre-call hard-valid snapshot"

            if rollback_state is not None:
                self._restore_furniture_scene_state(rollback_state)
                controller.should_finish = True
                console_logger.info(
                    "Safety controller rolled back %s designer call to %s: %s",
                    transaction["call_kind"],
                    rollback_source,
                    reasons,
                )
                return (
                    "\n\n**Safety Controller:** rolled back this designer call to "
                    f"the {rollback_source}; deterministic hard checks failed "
                    f"({reasons}). Finish with the restored checkpoint."
                )

            console_logger.info(
                "Safety controller found hard-invalid state but no rollback "
                "snapshot is available: %s",
                reasons,
            )
            return (
                "\n\n**Safety Controller:** deterministic hard checks failed "
                f"({reasons}), but no hard-valid rollback snapshot exists yet."
            )
        finally:
            controller.end_designer_call()

    def _apply_furniture_safety_after_critique(
        self,
        scores: CritiqueWithScores,
        images_dir: Path | None,
        physics_context: str | None = None,
    ) -> tuple[str, CritiqueWithScores | None, Path | None, bool]:
        """Evaluate a critiqued scene and rollback to best if needed."""
        controller = getattr(self, "furniture_safety_controller", None)
        if not controller or not controller.enabled:
            return "", scores, images_dir, True

        hard_state_evaluation = self._evaluate_current_furniture_hard_state(
            physics_context=physics_context
        )
        candidate_state = copy.deepcopy(self.scene.to_state_dict())
        decision = controller.consider_candidate(
            scores=scores,
            scene_state=candidate_state,
            render_dir=images_dir,
            hard_state_evaluation=hard_state_evaluation,
        )

        checkpoint_scores: CritiqueWithScores | None = None
        checkpoint_render_dir: Path | None = None
        checkpoint_accepted = decision.accepted
        if decision.accepted:
            checkpoint_scores = scores
            checkpoint_render_dir = images_dir
        if decision.rollback_to_best and controller.best_scene_state is not None:
            self._restore_furniture_scene_state(controller.best_scene_state)
            if controller.best_scores is not None:
                checkpoint_scores = controller.best_scores
            checkpoint_render_dir = controller.best_render_dir
            checkpoint_accepted = True
            console_logger.info(
                "Safety controller restored best hard-valid checkpoint "
                f"(weighted_score={controller.best_weighted_score:.3f})."
            )
        elif not decision.accepted:
            console_logger.info(
                "Safety controller did not accept this critique as a checkpoint; "
                "hard-invalid candidates will not be used for final rollback."
            )

        return (
            f"\n\n**Safety Controller:** {decision.message}",
            checkpoint_scores,
            checkpoint_render_dir,
            checkpoint_accepted,
        )

    def _get_model_settings(
        self,
        settings_key: str | None = None,
        tool_choice: str | None = None,
        parallel_tool_calls: bool | None = None,
    ) -> ModelSettings | None:
        """Create ModelSettings with timeout, reasoning effort, verbosity, and tool.

        Args:
            settings_key: Key in cfg.openai.reasoning_effort and cfg.openai.verbosity
                for this agent (e.g., "designer", "critic", "planner"). If None,
                no reasoning effort or verbosity is set.
            tool_choice: Tool name to force as first call (e.g., "observe_scene").
                Resets after first tool call by default to prevent infinite loops.
            parallel_tool_calls: Whether to allow parallel tool calls. Set to False
                for planner agents to prevent race conditions on shared sessions.

        Returns:
            ModelSettings with timeout, reasoning, verbosity, and tool_choice if
            configured, None otherwise.
        """
        kwargs: dict = {}
        extra_args: dict = {}

        # Add timeout if configured (api_timeout is optional).
        if hasattr(self.cfg, "api_timeout"):
            timeout_cfg = self.cfg.api_timeout
            timeout = Timeout(
                connect=timeout_cfg.connect,
                read=timeout_cfg.read,
                write=timeout_cfg.write,
                pool=timeout_cfg.pool,
            )
            extra_args["timeout"] = timeout

        # Add service_tier if configured (non-null/non-empty).
        service_tier = getattr(self.cfg.openai, "service_tier", None)
        if service_tier:
            extra_args["service_tier"] = service_tier

        if extra_args:
            kwargs["extra_args"] = extra_args

        # Note: reasoning_effort and verbosity are OpenAI Responses API specific
        # parameters and are not supported by open-source model APIs (e.g., vLLM).

        # Add tool_choice to force specific tool call first.
        if tool_choice:
            kwargs["tool_choice"] = tool_choice

        # Add parallel_tool_calls setting if specified.
        if parallel_tool_calls is not None:
            kwargs["parallel_tool_calls"] = parallel_tool_calls

        return ModelSettings(**kwargs) if kwargs else None

    def _get_agent_instructions(self, prompt_enum: Any, settings_key: str, **kwargs: Any) -> str:
        """Render prompt instructions and attach the configured thinking mode."""
        instructions = self.prompt_registry.get_prompt(prompt_enum=prompt_enum, **kwargs)
        effort = None
        if hasattr(self.cfg, "openai") and hasattr(self.cfg.openai, "reasoning_effort"):
            effort = getattr(self.cfg.openai.reasoning_effort, settings_key, None)
        directive = thinking_directive_from_effort(effort)
        return prepend_text_thinking_directive(instructions, directive)

    def _create_designer_agent(
        self, tools: list[FunctionTool], prompt_enum: Any, **prompt_kwargs: Any
    ) -> Agent:
        """Create designer agent with tools and domain-specific prompt.

        This method provides the shared pattern for designer agent creation,
        allowing subclasses to specify the prompt enum and context.

        Args:
            tools: Tools to provide to the designer.
            prompt_enum: Prompt enum from domain-specific registry.
            **prompt_kwargs: Additional kwargs for prompt template rendering.

        Returns:
            Configured designer agent.
        """
        designer_config = self.cfg.agents.designer_agent
        return Agent(
            name=designer_config.name,
            model=self.cfg.openai.model,
            tools=tools,
            instructions=self._get_agent_instructions(
                prompt_enum=prompt_enum, settings_key="designer", **prompt_kwargs
            ),
            model_settings=self._get_model_settings(settings_key="designer"),
        )

    def _create_critic_agent(
        self,
        tools: list[FunctionTool],
        prompt_enum: Any,
        output_type: type[CritiqueWithScores],
        **prompt_kwargs: Any,
    ) -> Agent:
        """Create critic agent with structured output.

        This method provides the shared pattern for critic agent creation,
        allowing subclasses to specify the prompt enum and context.

        Args:
            tools: Tools to provide to the critic.
            prompt_enum: Prompt enum from domain-specific registry.
            output_type: CritiqueWithScores subclass for structured output.
            **prompt_kwargs: Additional kwargs for prompt template rendering.

        Returns:
            Configured critic agent with domain-specific CritiqueWithScores type.
        """
        critic_config = self.cfg.agents.critic_agent
        self._critic_output_type = output_type
        return Agent(
            name=critic_config.name,
            model=self.cfg.openai.model,
            tools=tools,
            instructions=self._get_agent_instructions(
                prompt_enum=prompt_enum, settings_key="critic", **prompt_kwargs
            ),
            output_type=output_type,
            # Force observe_scene tool call first to ensure visual context.
            model_settings=self._get_model_settings(
                settings_key="critic", tool_choice="observe_scene"
            ),
        )

    def _create_planner_agent(
        self, tools: list[FunctionTool], prompt_enum: Any, **prompt_kwargs: Any
    ) -> Agent:
        """Create planner agent for workflow coordination.

        This method provides the shared pattern for planner agent creation,
        allowing subclasses to specify the prompt enum and context.

        Args:
            tools: Tools to provide to the planner.
            prompt_enum: Prompt enum from domain-specific registry.
            **prompt_kwargs: Additional kwargs for prompt template rendering.

        Returns:
            Configured planner agent.
        """
        planner_config = self.cfg.agents.planner_agent
        return Agent(
            name=planner_config.name,
            model=self.cfg.openai.model,
            tools=tools,
            instructions=self._get_agent_instructions(
                prompt_enum=prompt_enum, settings_key="planner", **prompt_kwargs
            ),
            # Disable parallel tool calls to prevent race conditions on shared
            # sessions (designer_session, critic_session). When the model returns
            # multiple tool calls in one response, they would otherwise run
            # concurrently and cause SQLite locking issues.
            model_settings=self._get_model_settings(
                settings_key="planner", parallel_tool_calls=False
            ),
        )

    def _create_sessions(self, session_prefix: str = "") -> tuple[Session, Session]:
        """Create designer and critic sessions for persistent conversation history.

        Sessions are optionally wrapped with TurnTrimmingSession for memory
        management if session_memory is enabled in config.

        Args:
            session_prefix: Optional prefix for session IDs (e.g., furniture ID).

        Returns:
            Tuple of (designer_session, critic_session).
        """
        designer_id = f"{session_prefix}designer" if session_prefix else "designer"
        critic_id = f"{session_prefix}critic" if session_prefix else "critic"

        designer_sqlite = SQLiteSession(
            session_id=designer_id,
            db_path=self.logger.output_dir / f"{designer_id}.db",
        )
        critic_sqlite = SQLiteSession(
            session_id=critic_id,
            db_path=self.logger.output_dir / f"{critic_id}.db",
        )

        # Wrap with memory management if configured.
        memory_cfg = self.cfg.session_memory
        if memory_cfg and memory_cfg.enabled:
            console_logger.info(
                f"Enabling turn-trimming session (keep_last_n_turns="
                f"{memory_cfg.keep_last_n_turns}, summarization="
                f"{memory_cfg.enable_summarization})"
            )
            designer_session: Session = TurnTrimmingSession(
                wrapped_session=designer_sqlite, cfg=self.cfg
            )
            critic_session: Session = TurnTrimmingSession(
                wrapped_session=critic_sqlite, cfg=self.cfg
            )
        else:
            designer_session = designer_sqlite
            critic_session = critic_sqlite

        return designer_session, critic_session

    def _create_run_config(self) -> RunConfig:
        """Create RunConfig with intra-turn image filter if enabled.

        The filter strips images from older observe_scene outputs within a turn,
        keeping only the last N observations with images intact. This reduces
        token usage when agents call observe_scene multiple times within a turn.

        Returns:
            RunConfig with call_model_input_filter set if enabled, empty otherwise.
        """
        # Use an OpenAIProvider pointed at the vLLM endpoint so that model names
        # with arbitrary org prefixes (e.g. "Qwen/...") are passed through as-is
        # instead of being routed through MultiProvider's prefix registry.
        # use_responses=False forces /chat/completions instead of /responses,
        # because vLLM's --tool-call-parser hermes only works on /chat/completions.
        provider = OpenAIProvider(
            base_url=os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"),
            api_key=os.environ.get("OPENAI_API_KEY", "dummy"),
            use_responses=False,
        )
        intra_cfg = self.cfg.session_memory.intra_turn_observation_stripping
        if intra_cfg.enabled:
            return RunConfig(
                model_provider=provider,
                call_model_input_filter=IntraTurnImageFilter(cfg=self.cfg),
            )

        return RunConfig(model_provider=provider)

    def _should_reset_to_checkpoint(
        self,
        current_scores: CritiqueWithScores,
        previous_scores: CritiqueWithScores | None,
    ) -> tuple[bool, str]:
        """Check if current scores warrant resetting to previous checkpoint.

        Uses same threshold logic as planner agent instructions.

        Args:
            current_scores: Scores for the current scene state.
            previous_scores: Scores from the previous checkpoint (N-1).

        Returns:
            (should_reset, reason) tuple where reason explains which threshold
            was exceeded.
        """
        if previous_scores is None:
            return False, ""

        # Check single category drops.
        current_scores_list = current_scores.get_scores()
        previous_scores_list = previous_scores.get_scores()
        for current_score, previous_score in zip(
            current_scores_list, previous_scores_list
        ):
            drop = previous_score.grade - current_score.grade

            if drop >= self.cfg.reset_single_category_threshold:
                return True, f"{current_score.name} dropped {drop} points"

        # Check total sum drop.
        current_sum = compute_total_score(current_scores)
        previous_sum = compute_total_score(previous_scores)
        total_drop = previous_sum - current_sum

        if total_drop >= self.cfg.reset_total_sum_threshold:
            return True, f"total score dropped {total_drop} points"

        return False, ""

    @log_scene_action
    def _perform_checkpoint_reset(self, checkpoint_state_dict: dict) -> None:
        """Restore scene and scores to previous checkpoint (N-1).

        This is the core reset operation shared by both the planner tool
        and the final scene validation logic.

        Args:
            checkpoint_state_dict: Checkpoint state dictionary to restore from.
                During normal operation, this is self.previous_scene_checkpoint.
                During replay, this is the logged checkpoint state.
        """
        # Restore scene from checkpoint (N-1 iteration).
        self.scene.restore_from_state_dict(checkpoint_state_dict)

        # Clear render cache to force new renders after reset.
        self.rendering_manager.clear_cache()

        # Reset score tracking to previous checkpoint state.
        # Note: During replay, these may be None which is okay.
        if self.previous_checkpoint_scores is not None:
            self.checkpoint_scores = copy.deepcopy(self.previous_checkpoint_scores)
            self.previous_scores = copy.deepcopy(self.previous_checkpoint_scores)

        # Invalidate current checkpoint since we went back.
        # Note: During replay, these may be None which is okay.
        if self.previous_scene_checkpoint is not None:
            self.scene_checkpoint = self.previous_scene_checkpoint
            self.checkpoint_render_dir = self.previous_checkpoint_render_dir

    @abstractmethod
    def _get_final_scores_directory(self) -> Path:
        """Get the directory path for saving final scene scores.

        Returns:
            Path to the directory where final scores should be saved.
        """

    async def _finalize_scene_and_scores(self) -> None:
        """Validate final scene against thresholds and save scores.

        This method checks if the final scene's scores are degraded compared
        to the previous checkpoint. If so, it resets to the better checkpoint.
        Finally, it copies the scores to the final_scene directory for easy access.

        The final directory path is determined by the subclass implementation
        of _get_final_scores_directory().
        """
        controller = getattr(self, "furniture_safety_controller", None)
        if (
            controller
            and controller.enabled
            and controller.best_scene_state is not None
        ):
            self._restore_furniture_scene_state(controller.best_scene_state)
            self.scene_checkpoint = copy.deepcopy(controller.best_scene_state)
            if controller.best_scores is not None:
                self.checkpoint_scores = controller.best_scores
            if controller.best_render_dir is not None:
                self.checkpoint_render_dir = controller.best_render_dir
                self.final_render_dir = controller.best_render_dir
            console_logger.info(
                "Final furniture scene restored to best hard-valid checkpoint "
                f"(weighted_score={controller.best_weighted_score:.3f})."
            )

        # Check if final scores warrant resetting to previous checkpoint.
        # Use previous_scores (actual final critique) vs checkpoint_scores (last checkpoint).
        # Note: Final critique uses update_checkpoint=False, so previous_scores holds the
        # actual final scores while checkpoint_scores holds the last iteration's scores.
        if self.previous_scores is not None and self.checkpoint_scores is not None:
            should_reset, reason = self._should_reset_to_checkpoint(
                current_scores=self.previous_scores,
                previous_scores=self.checkpoint_scores,
            )

            console_logger.debug(
                f"Reset check result: should_reset={should_reset}, reason={reason}"
            )

            if should_reset:
                console_logger.info(
                    f"Final scene scores are degraded ({reason}). "
                    f"Resetting to checkpoint (N-1)."
                )

                # Restore scene to checkpoint (N-1) directly. Don't use
                # _perform_checkpoint_reset() here since that's designed for mid-loop
                # resets and modifies checkpoint tracking variables.
                self.scene.restore_from_state_dict(self.scene_checkpoint)
                self.rendering_manager.clear_cache()

                scores_parts = [
                    f"{score.name}={score.grade}"
                    for score in self.checkpoint_scores.get_scores()
                ]
                console_logger.info(
                    f"Final scene restored to checkpoint state. "
                    f"Checkpoint scores: {', '.join(scores_parts)}"
                )

                # Update final_render_dir to point to restored checkpoint's render.
                self.final_render_dir = self.checkpoint_render_dir

        fail_on_hard_constraints = bool(
            _cfg_get(self.cfg, "fail_stage_on_unresolved_hard_constraints", True)
        )
        if (
            controller
            and getattr(controller, "enabled", False)
            and fail_on_hard_constraints
        ):
            final_hard_state = self._evaluate_current_hard_state()
            final_hard_state, _, final_repair_actions = (
                self._try_deterministic_repair_for_hard_state(
                    final_hard_state,
                    source="finalize",
                )
            )
            if final_repair_actions:
                console_logger.info(
                    "Deterministic repair attempted during finalization: %s",
                    "; ".join(final_repair_actions),
                )
            if final_hard_state is not None and not final_hard_state.hard_valid:
                if getattr(controller, "best_scene_state", None) is not None:
                    self._restore_furniture_scene_state(controller.best_scene_state)
                    self.scene_checkpoint = copy.deepcopy(controller.best_scene_state)
                    if controller.best_scores is not None:
                        self.checkpoint_scores = controller.best_scores
                    if controller.best_render_dir is not None:
                        self.checkpoint_render_dir = controller.best_render_dir
                        self.final_render_dir = controller.best_render_dir
                    console_logger.info(
                        "Final hard-check failed after repair; restored best "
                        "hard-valid checkpoint instead of failing the stage."
                    )
                    final_hard_state = self._evaluate_current_hard_state()
                    if final_hard_state is None or final_hard_state.hard_valid:
                        reasons = ""
                    else:
                        reasons = "; ".join(final_hard_state.hard_reasons)
                else:
                    reasons = "; ".join(final_hard_state.hard_reasons)
            if (
                final_hard_state is not None
                and not final_hard_state.hard_valid
            ):
                reasons = "; ".join(final_hard_state.hard_reasons)
                console_logger.error(
                    "Furniture stage failed with unresolved deterministic hard "
                    "constraints: %s",
                    reasons,
                )
                raise RuntimeError(
                    "Furniture stage failed with unresolved hard constraints: "
                    f"{reasons}"
                )

        # Copy final scores and renders to per-stage directory.
        # Use final_render_dir (tracks actual last render) instead of checkpoint_render_dir
        # (which may be stale when final critique uses update_checkpoint=False).
        render_dir_to_copy = self.final_render_dir or self.checkpoint_render_dir
        if render_dir_to_copy is not None:
            final_scene_dir = self._get_final_scores_directory()
            final_scene_dir.mkdir(parents=True, exist_ok=True)

            # Copy scores.
            scores_source = render_dir_to_copy / "scores.yaml"
            if scores_source.exists():
                scores_dest = final_scene_dir / "scores.yaml"
                shutil.copy(scores_source, scores_dest)
                console_logger.info(f"Saved final scores to {scores_dest}")
            else:
                console_logger.warning(
                    f"Scores file not found at {scores_source}, cannot copy"
                )

            controller = getattr(self, "furniture_safety_controller", None)
            plausibility_report = (
                getattr(controller, "best_plausibility_report", None)
                if controller and controller.enabled
                else None
            )
            if plausibility_report:
                plausibility_dest = final_scene_dir / "plausibility.yaml"
                with open(plausibility_dest, "w") as f:
                    yaml.dump(
                        data=plausibility_report,
                        stream=f,
                        default_flow_style=False,
                        sort_keys=False,
                    )
                console_logger.info(
                    f"Saved deterministic plausibility report to {plausibility_dest}"
                )

            # Copy render images.
            render_images = list(render_dir_to_copy.glob("*.png"))
            if render_images:
                for img_path in render_images:
                    img_dest = final_scene_dir / img_path.name
                    shutil.copy(img_path, img_dest)
                console_logger.info(
                    f"Copied {len(render_images)} render images to {final_scene_dir}"
                )
            else:
                console_logger.warning(
                    f"No render images found in {render_dir_to_copy}"
                )

    def _create_reset_checkpoint_tool(self) -> FunctionTool:
        """Create tool for resetting scene to previous checkpoint.

        Returns:
            FunctionTool that allows agents to reset to previous checkpoint.
        """

        @function_tool
        async def reset_scene_to_checkpoint(reason: str) -> str:
            """Reset scene to previous iteration state when changes made it worse.

            Use this when the designer's changes resulted in significant score
            degradation.

            Args:
                reason: Explanation of why you're resetting.

            Returns:
                Confirmation with checkpoint details and scores.
            """
            console_logger.info("Tool called: reset_scene_to_checkpoint")
            if self._planner_budget_exhausted:
                return self._planner_budget_stop_message("reset_scene_to_checkpoint")

            if (
                self.previous_scene_checkpoint is None
                or self.previous_checkpoint_scores is None
            ):
                console_logger.warning("No previous checkpoint available to reset to.")
                return (
                    "ERROR: No previous checkpoint available to reset to. "
                    "You must call request_critique() at least twice to create "
                    "enough checkpoints for reset functionality."
                )

            self._perform_checkpoint_reset(
                checkpoint_state_dict=self.previous_scene_checkpoint
            )

            # Log reset event.
            console_logger.info(f"Scene reset to checkpoint. Reason: {reason}")

            # Return confirmation with checkpoint scores.
            # Build scores string dynamically using get_scores() for agent-agnostic output.
            scores_parts = [
                f"{score.name}={score.grade}"
                for score in self.checkpoint_scores.get_scores()
            ]
            scores_str = ", ".join(scores_parts)

            return (
                f"Scene reset to state from 2 iterations ago.\n"
                f"Checkpoint scores: {scores_str}\n"
                f"Reset reason: {reason}\n"
                "Continue with design improvements from this restored state."
            )

        return reset_scene_to_checkpoint

    def _create_placement_style_tool(self) -> FunctionTool:
        """Create tool for selecting placement style (natural vs perfect).

        Returns:
            FunctionTool that allows agents to select placement style.
        """

        @function_tool
        def select_placement_style(style: str) -> str:
            """Select placement style based on scene prompt analysis.

            MUST be called FIRST before any placement operations.

            Analyzes the scene description to determine whether to use:
            - "natural": Realistic, lived-in scenes with slight imperfections
            - "perfect": Precise, exhibition-quality placement with no variation

            Args:
                style: Either "natural" or "perfect"

            Returns:
                Confirmation of selected style and readiness for placement.
            """
            style_lower = style.lower()
            if style_lower == "natural":
                mode = PlacementNoiseMode.NATURAL
            elif style_lower == "perfect":
                mode = PlacementNoiseMode.PERFECT
            else:
                console_logger.warning(
                    f"Invalid placement style '{style}', defaulting to 'natural'"
                )
                mode = PlacementNoiseMode.NATURAL
                style_lower = "natural"

            # Set noise profile on domain-specific tools.
            self._set_placement_noise_profile(mode)
            self.placement_style = style_lower

            return (
                f"Placement style set to '{style_lower}'. "
                f"Ready for placement with {style_lower} variation."
            )

        return select_placement_style

    def _reset_planner_budget_tracking(self) -> None:
        self._planner_initial_design_tool_calls = 0
        self._planner_critique_tool_calls = 0
        self._planner_design_change_tool_calls = 0
        self._planner_budget_exhausted = False
        self._critic_failed = False
        self._pending_hard_repair_hint = ""
        self._hard_repair_design_change_calls = 0

    def _stop_planner_after_failure(self, reason: str) -> str:
        """Convert a nested agent failure into a deterministic planner stop."""
        self._planner_budget_exhausted = True
        controller = getattr(self, "furniture_safety_controller", None)
        if controller and getattr(controller, "enabled", False):
            controller.should_finish = True
        return (
            f"STOP: {reason} Do not restart the initial design or call more "
            "planner tools. Return the final concise workflow summary now."
        )

    def _planner_context_limit(self, key: str, default: int) -> int:
        limits_cfg = getattr(self.cfg, "planner_context_limits", None)
        try:
            return int(_cfg_get(limits_cfg, key, default))
        except (TypeError, ValueError):
            return default

    def _truncate_planner_tool_output(
        self,
        text: str,
        *,
        label: str,
        max_chars: int,
    ) -> str:
        """Keep planner tool outputs bounded inside a single Runner.run."""
        if max_chars <= 0 or len(text) <= max_chars:
            return text
        head_chars = max_chars // 2
        tail_chars = max_chars - head_chars
        console_logger.info(
            "Truncated %s result for planner context from %d to %d chars",
            label,
            len(text),
            max_chars,
        )
        return (
            text[:head_chars]
            + "\n\n[... planner context truncated; middle omitted ...]\n\n"
            + text[-tail_chars:]
        )

    def _planner_budget_stop_message(self, tool_name: str) -> str:
        self._planner_budget_exhausted = True
        controller = getattr(self, "furniture_safety_controller", None)
        if controller and getattr(controller, "enabled", False):
            controller.should_finish = True
        return (
            f"STOP: {tool_name} is blocked because the configured "
            f"max_critique_rounds={self.cfg.max_critique_rounds} budget has "
            "been reached. Do not call request_critique(), "
            "request_design_change(), or reset_scene_to_checkpoint() again. "
            "Return your final concise workflow summary now. The framework will "
            "run the final critique automatically after the planner exits."
        )

    def _planner_budget_hint_after_critique(self) -> str:
        if self._planner_critique_tool_calls < int(self.cfg.max_critique_rounds):
            return ""
        return (
            "\n\n[Planner budget] This is the last allowed planner critique. "
            "If changes are still needed, call request_design_change() once to "
            "address the critique, then return the final summary. Do not call "
            "request_critique() again."
        )

    def _planner_budget_hint_after_design_change(self) -> str:
        if self._planner_design_change_tool_calls < int(self.cfg.max_critique_rounds):
            return ""
        self._planner_budget_exhausted = True
        return (
            "\n\n[Planner budget] The configured critique-improvement cycle "
            "budget is complete. Do not call more planner tools. Return the "
            "final concise workflow summary now; the final critique will be "
            "computed automatically."
        )

    def _auto_score_after_design_attempts_enabled(self) -> bool:
        """Whether planner-level design attempts should be critiqued immediately."""
        return bool(_cfg_get(self.cfg, "auto_score_after_design_attempts", False))

    async def _score_design_attempt_if_configured(self, attempt_label: str) -> str:
        """Run a critique after a planner-level design attempt when configured.

        This closes the candidate-evaluation loop without scoring every transient
        observe_scene render inside a designer call.  The critic's forced
        observe_scene step saves scores.yaml next to the render for the current
        final candidate state.
        """
        if not self._auto_score_after_design_attempts_enabled():
            return ""
        if self.cfg.max_critique_rounds <= 0:
            return ""
        if self._planner_critique_tool_calls >= int(self.cfg.max_critique_rounds):
            self._planner_budget_exhausted = True
            return "\n\n" + self._planner_budget_stop_message(
                f"auto_score_after_{attempt_label.replace(' ', '_')}"
            )

        current_hash = self.scene.content_hash()
        if (
            self.checkpoint_scene_hash is not None
            and current_hash == self.checkpoint_scene_hash
        ):
            return (
                "\n\n[Auto scoring] Scene is unchanged since the previous scored "
                f"candidate; skipped duplicate critique after {attempt_label}."
            )

        self._planner_critique_tool_calls += 1
        score_start = time.time()
        console_logger.info(
            "Auto-scoring planner-level design attempt after %s", attempt_label
        )
        try:
            critique = await self._request_critique_impl(update_checkpoint=True)
        except Exception as exc:
            self._critic_failed = True
            console_logger.exception(
                "Automatic critic scoring failed after %s; stopping planner",
                attempt_label,
            )
            return "\n\n" + self._stop_planner_after_failure(
                "Critic scoring failed with "
                f"{type(exc).__name__}: {exc}."
            )
        self._record_module_timing(
            "planner",
            f"auto_score_after_{attempt_label.replace(' ', '_')}",
            score_start,
        )
        budget_hint = ""
        if self._planner_critique_tool_calls >= int(self.cfg.max_critique_rounds):
            self._planner_budget_exhausted = True
            budget_hint = (
                "\n\n[Planner budget] The configured scored-candidate budget "
                "is complete. Do not call more planner tools. Return the final "
                "concise workflow summary now."
            )
        return (
            f"\n\n## Auto Critique After {attempt_label.title()}\n"
            f"{critique}"
            f"{budget_hint}"
        )

    def _create_planner_tools(self) -> list[FunctionTool]:
        """Create planner tools for the design workflow.

        Returns tools that the planner uses to coordinate designer and critic:
        - select_placement_style: Set natural vs perfect placement (placement agents only)
        - request_initial_design: Request initial design from designer
        - request_critique: Request evaluation from critic
        - request_design_change: Request design modifications based on feedback
        - reset_scene_to_checkpoint: Reset to last checkpoint state

        Returns:
            List of function tools for planner agent.
        """
        self._reset_planner_budget_tracking()

        @function_tool
        async def request_initial_design() -> str:
            """Request the designer to create the initial design.

            The designer will analyze the context and create an appropriate
            initial layout or arrangement.

            Returns:
                Designer's report of what was created and why.
            """
            if self._planner_budget_exhausted:
                return self._stop_planner_after_failure(
                    "The current design stage has already been marked complete or failed."
                )
            if self._planner_initial_design_tool_calls >= 1:
                return self._stop_planner_after_failure(
                    "request_initial_design is a one-shot operation and has already "
                    "completed."
                )
            self._planner_initial_design_tool_calls += 1
            result = await self._request_initial_design_impl()
            result += await self._score_design_attempt_if_configured(
                "initial design"
            )
            return self._truncate_planner_tool_output(
                result,
                label="initial design",
                max_chars=self._planner_context_limit(
                    "initial_design_max_chars", 5000
                ),
            )

        @function_tool
        async def request_critique() -> str:
            """Request the critic to evaluate the current design.

            The critic will examine the current state and provide feedback
            on what works well and what needs improvement.

            Returns:
                Critic's detailed evaluation with specific improvement suggestions.
            """
            if self._planner_budget_exhausted:
                return self._stop_planner_after_failure(
                    "The current design stage has already been marked complete or failed."
                )
            if (
                self._auto_score_after_design_attempts_enabled()
                and self.checkpoint_scene_hash is not None
                and self.scene.content_hash() == self.checkpoint_scene_hash
            ):
                return (
                    "Current scene already has a critique score from the latest "
                    "auto-scored candidate. Read the previous Auto Critique instead "
                    "of re-scoring an unchanged layout."
                )

            if self._planner_critique_tool_calls >= int(self.cfg.max_critique_rounds):
                return self._planner_budget_stop_message("request_critique")

            self._planner_critique_tool_calls += 1
            try:
                result = await self._request_critique_impl()
            except Exception as exc:
                self._critic_failed = True
                console_logger.exception("Planner-requested critic scoring failed")
                return self._stop_planner_after_failure(
                    "Critic scoring failed with "
                    f"{type(exc).__name__}: {exc}."
                )
            result = self._truncate_planner_tool_output(
                result,
                label="critique",
                max_chars=self._planner_context_limit("critique_max_chars", 7000),
            )
            return result + self._planner_budget_hint_after_critique()

        @function_tool
        async def request_design_change(instruction: str) -> str:
            """Request the designer to address specific issues.

            Based on the critic's feedback, provide clear instructions about
            what to change. The designer will modify the design to address
            the issues while maintaining what works well.

            Args:
                instruction: Specific changes to make based on critique feedback.

            Returns:
                Designer's report of what was changed.
            """
            if self._planner_budget_exhausted:
                return self._stop_planner_after_failure(
                    "The current design stage has already been marked complete or failed."
                )
            counts_as_critique_cycle = (
                self._planner_critique_tool_calls
                > self._planner_design_change_tool_calls
            )
            if (
                self._auto_score_after_design_attempts_enabled()
                and self._planner_critique_tool_calls
                >= int(self.cfg.max_critique_rounds)
                and not self._hard_repair_allowance_available()
            ):
                return self._planner_budget_stop_message("request_design_change")

            if counts_as_critique_cycle and self._planner_design_change_tool_calls >= int(
                self.cfg.max_critique_rounds
            ) and not self._hard_repair_allowance_available():
                return self._planner_budget_stop_message("request_design_change")

            safety_block = self._record_furniture_design_change_budget()
            if safety_block:
                return safety_block

            hard_repair_allowance = self._hard_repair_allowance_available()
            if hard_repair_allowance:
                instruction = (
                    f"{instruction}\n\nMANDATORY HARD-CHECK REPAIR: "
                    f"{self._pending_hard_repair_hint}"
                )

            result = await self._request_design_change_impl(instruction)
            if hard_repair_allowance:
                self._hard_repair_design_change_calls += 1
            result += await self._score_design_attempt_if_configured(
                "design change"
            )
            result = self._truncate_planner_tool_output(
                result,
                label="design change",
                max_chars=self._planner_context_limit(
                    "design_change_max_chars", 5000
                ),
            )
            if counts_as_critique_cycle:
                self._planner_design_change_tool_calls += 1
                if not self._auto_score_after_design_attempts_enabled():
                    result += self._planner_budget_hint_after_design_change()
            return result

        @function_tool
        async def finish_stage(summary: str = "") -> str:
            """Finish the current stage and stop planner tool use.

            Use this when the design is accepted, the Safety Controller says to
            finish, or the critique/design budget is exhausted.

            Args:
                summary: Concise final workflow summary for this stage.

            Returns:
                Confirmation that the planner should return its final answer.
            """
            console_logger.info("Tool called: finish_stage")
            if self._hard_repair_allowance_available():
                return (
                    "FINISH_STAGE_BLOCKED: a deterministic hard-check failure is "
                    "still pending repair. You must call request_design_change() "
                    f"first with this repair requirement: {self._pending_hard_repair_hint}"
                )
            self._planner_budget_exhausted = True
            controller = getattr(self, "furniture_safety_controller", None)
            if controller and getattr(controller, "enabled", False):
                controller.should_finish = True
            compact_summary = " ".join(str(summary or "").split())
            if compact_summary:
                console_logger.info("Planner finish_stage summary: %s", compact_summary)
            return (
                "FINISH_STAGE_ACCEPTED: do not call any more planner tools. "
                "Return your final concise workflow summary now. The framework "
                "will run the final critique automatically after the planner exits."
                + (f"\nSummary: {compact_summary}" if compact_summary else "")
            )

        tools: list[FunctionTool] = [request_initial_design]

        # Only add critique-related tools if critique rounds are enabled.
        # This prevents the planner from accidentally calling critique tools
        # when max_critique_rounds is 0.
        if self.cfg.max_critique_rounds > 0:
            reset_scene_to_checkpoint = self._create_reset_checkpoint_tool()
            tools.extend(
                [request_critique, request_design_change, reset_scene_to_checkpoint]
            )

        # Add placement style tool for placement agents (not floor plan).
        if self._is_placement_agent:
            placement_style_tool = self._create_placement_style_tool()
            tools.insert(0, placement_style_tool)

        tools.append(finish_stage)
        return tools

    @abstractmethod
    def _get_critique_prompt_enum(self) -> Any:
        """Get the prompt enum for critic runner instruction.

        Returns:
            Prompt enum for domain-specific critic instruction.
        """

    @abstractmethod
    def _set_placement_noise_profile(self, mode: PlacementNoiseMode) -> None:
        """Set placement noise profile for domain-specific tools.

        Args:
            mode: Placement noise mode (NATURAL or PERFECT).
        """

    def _get_extra_critique_kwargs(self) -> dict[str, Any]:
        """Get extra keyword arguments for critic prompt template.

        Override in subclasses to inject domain-specific context into critic prompts.
        For example, furniture agent overrides this to add reachability context.

        Returns:
            Dictionary of extra kwargs to pass to prompt rendering.
        """
        return {}

    async def _request_critique_impl(self, update_checkpoint: bool = True) -> str:
        """Implementation for critique request.

        Runs critic agent in three deterministic steps to guarantee the tool
        call sequence observe_scene → get_current_scene_state → free evaluation:

          Step 1 (tool_choice=observe_scene):   forces visual render first.
          Step 2 (tool_choice=get_current_scene_state): forces scene data fetch.
          Step 3 (tool_choice=none, output_type set): critic scores freely.

        Each step feeds its output into the next via to_input_list(), so the
        session accumulates the full context before the scoring turn.

        Args:
            update_checkpoint: Whether to shift checkpoints. Set to False for
                final critique calls to preserve N-1 checkpoint for reset check.

        Returns:
            Critique text with optional score deltas for planner.
        """
        console_logger.info("Tool called: request_critique")
        critique_start = time.time()
        self._reset_critic_candidate_cache()

        # Get physics violations using the same logic as the check_physics tool.
        # The result is cached per candidate and reused by deterministic checks
        # and critic prompt construction.
        physics_context = self._get_cached_physics_context()
        hard_state = (
            self._evaluate_current_hard_state(physics_context=physics_context)
            if self._critic_fast_path_enabled("hard_check_first", True)
            else None
        )
        render_profile = self._critic_render_profile_name(update_checkpoint)
        hard_state, repaired_physics_context, repair_actions = (
            self._try_deterministic_repair_for_hard_state(
                hard_state,
                source="pre_critique",
            )
        )
        if repaired_physics_context is not None:
            physics_context = repaired_physics_context
        if repair_actions:
            console_logger.info(
                "[CRITIC harness] Deterministic repair actions before scoring: %s",
                "; ".join(repair_actions),
            )

        if (
            hard_state is not None
            and not hard_state.hard_valid
            and self._critic_fast_path_enabled("skip_vlm_on_hard_fail", True)
        ):
            self._pending_hard_repair_hint = self._repair_hint_from_hard_state(
                hard_state
            )
            console_logger.info(
                "[CRITIC harness] Deterministic hard-check failed; skipping VLM "
                "score_scene and returning repair-first critique: %s",
                "; ".join(hard_state.hard_reasons),
            )
            images_dir = None
            if self._critic_fast_path_enabled("render_hard_fail_candidate", True):
                images_dir = self._observe_scene_for_synthetic_score(render_profile)

            response = self._make_deterministic_critique_scores(
                hard_state=hard_state,
                physics_context=physics_context,
            )
            self._record_llm_call_debug(
                agent_role="critic",
                event="deterministic_hard_fail_short_circuit",
                prompt={
                    "physics_context": physics_context,
                    "hard_reasons": hard_state.hard_reasons,
                    "soft_reasons": hard_state.soft_reasons,
                },
                output=response.critique,
            )
            log_agent_response(response=response.critique, agent_name="CRITIC")
            log_critique_scores(response, title="DETERMINISTIC CRITIQUE SCORES")
            self._write_scores_and_memory(
                response=response,
                images_dir=images_dir,
                physics_context=physics_context,
                event="deterministic_hard_fail",
            )

            score_change_msg = ""
            if self.previous_scores is not None:
                score_change_msg = format_score_deltas_for_planner(
                    current_scores=response,
                    previous_scores=self.previous_scores,
                    format_style="detailed",
                )

            controller = getattr(self, "furniture_safety_controller", None)
            if controller and controller.enabled:
                (
                    safety_msg,
                    checkpoint_scores,
                    checkpoint_render_dir,
                    checkpoint_accepted,
                ) = self._apply_furniture_safety_after_critique(
                    scores=response,
                    images_dir=images_dir,
                    physics_context=physics_context,
                )
            else:
                safety_msg = (
                    "\n\n**Hard Check:** "
                    + self._repair_hint_from_hard_state(hard_state)
                )
                checkpoint_scores = None
                checkpoint_render_dir = None
                checkpoint_accepted = False

            if update_checkpoint and checkpoint_accepted:
                self.previous_scene_checkpoint = self.scene_checkpoint
                self.previous_checkpoint_scores = self.checkpoint_scores
                self.previous_checkpoint_render_dir = self.checkpoint_render_dir
                self.scene_checkpoint = copy.deepcopy(self.scene.to_state_dict())
                self.checkpoint_scores = checkpoint_scores
                self.checkpoint_render_dir = checkpoint_render_dir
                self.checkpoint_scene_hash = self.scene.content_hash()
            elif update_checkpoint:
                console_logger.info(
                    "Skipping checkpoint update because deterministic hard-check "
                    "failed."
                )

            self.previous_scores = response
            self.final_render_dir = checkpoint_render_dir or images_dir
            self._last_scored_scene_hash = self.scene.content_hash()
            self._last_critique_render_profile = render_profile
            self._record_module_timing(
                "critic",
                "request_critique_total",
                critique_start,
                extra={
                    "update_checkpoint": update_checkpoint,
                    "hard_check_short_circuit": True,
                    "render_profile": render_profile,
                },
            )
            return response.critique + score_change_msg + safety_msg

        if hard_state is not None and not hard_state.hard_valid:
            self._pending_hard_repair_hint = self._repair_hint_from_hard_state(
                hard_state
            )
        else:
            self._pending_hard_repair_hint = ""
            self._hard_repair_design_change_calls = 0

        prompt_enum = self._get_critique_prompt_enum()
        extra_kwargs = self._get_extra_critique_kwargs()

        critique_instruction = self.prompt_registry.get_prompt(
            prompt_enum=prompt_enum,
            physics_context=physics_context,
            placement_style=self.placement_style,
            **extra_kwargs,
        )
        context_block = self._prepare_stage_context_for_llm(
            agent_role="critic",
            event="request_critique",
            prompt=critique_instruction,
            last_hard_issues=hard_state.hard_reasons if hard_state else [],
        )
        if context_block:
            critique_instruction += "\n\n" + context_block
        run_config = self._create_run_config()

        # Build three critic variants that differ only in model_settings.
        # Steps 1 and 2 use tool_use_behavior="stop_on_first_tool" so the runner
        # exits cleanly after the forced tool call (no extra LLM turn needed),
        # which avoids MaxTurnsExceeded. parallel_tool_calls=False keeps the
        # forced step to a single tool invocation. output_type=None on steps 1/2
        # prevents premature attempts to emit structured JSON before scoring.
        base_settings = self.critic.model_settings

        critic_observe = self.critic.clone(
            output_type=None,
            tool_use_behavior="stop_on_first_tool",
            model_settings=base_settings.resolve(
                ModelSettings(
                    tool_choice="observe_scene", parallel_tool_calls=False
                )
            ),
        )
        critic_scene_state = self.critic.clone(
            output_type=None,
            tool_use_behavior="stop_on_first_tool",
            model_settings=base_settings.resolve(
                ModelSettings(
                    tool_choice="get_current_scene_state",
                    parallel_tool_calls=False,
                )
            ),
        )
        # Step 3 must be a pure structured-output request. Keeping the original
        # tools or resolving tool_choice=None can preserve the forced
        # observe_scene choice and creates an invalid tools + response_format
        # request for vLLM's Qwen tool parser.
        critic_score = self.critic.clone(
            tools=[],
            model_settings=base_settings.resolve(
                ModelSettings(tool_choice="none", parallel_tool_calls=False)
            ),
        )

        # All three steps share self.critic_session so history accumulates
        # naturally; inputs are strings (lists are illegal when a session is
        # used without a custom session_input_callback).

        # Step 1: force observe_scene; stop_on_first_tool returns immediately.
        console_logger.info("[CRITIC harness] Step 1: observe_scene")
        observe_start = time.time()
        with self.rendering_manager.use_render_profile(render_profile):
            result_observe = await Runner.run(
                starting_agent=critic_observe,
                input=critique_instruction,
                session=self.critic_session,
                run_config=run_config,
            )
        self._record_module_timing("critic", "observe_scene", observe_start)
        log_agent_usage(result=result_observe, agent_name="CRITIC (observe)")
        self._record_llm_call_debug(
            agent_role="critic",
            event="observe_scene",
            prompt=critique_instruction,
            output=result_observe.final_output or "",
            result=result_observe,
        )

        # Step 2: force get_current_scene_state; session carries Step 1 history.
        direct_scene_state = self._get_critic_scene_state_direct()
        scene_state_block = ""
        if direct_scene_state:
            max_scene_state_chars = int(
                _cfg_get(self._critic_fast_path_cfg(), "scene_state_max_chars", 16000)
            )
            if len(direct_scene_state) > max_scene_state_chars:
                direct_scene_state = (
                    direct_scene_state[:max_scene_state_chars]
                    + "\n...[scene state truncated for critic fast path]..."
                )
            scene_state_block = (
                "\n\nExact get_current_scene_state output for this candidate:\n"
                f"{direct_scene_state}"
            )
            console_logger.info(
                "[CRITIC harness] Step 2: get_current_scene_state direct cache"
            )
        else:
            console_logger.info("[CRITIC harness] Step 2: get_current_scene_state")
            scene_state_start = time.time()
            result_scene = await Runner.run(
                starting_agent=critic_scene_state,
                input="Now retrieve exact object data with get_current_scene_state.",
                session=self.critic_session,
                run_config=run_config,
            )
            self._record_module_timing(
                "critic", "get_current_scene_state", scene_state_start
            )
            log_agent_usage(result=result_scene, agent_name="CRITIC (scene_state)")
            self._record_llm_call_debug(
                agent_role="critic",
                event="get_current_scene_state",
                prompt="Now retrieve exact object data with get_current_scene_state.",
                output=result_scene.final_output or "",
                result=result_scene,
            )
        if direct_scene_state:
            self._record_llm_call_debug(
                agent_role="critic",
                event="get_current_scene_state_direct",
                prompt="direct scene state cache",
                output=direct_scene_state,
            )

        # Step 3: free evaluation with structured output. The critic now has
        # the observation images and the scene-state JSON in its session
        # history and can run STEPS 3-6 of the YAML workflow.
        console_logger.info("[CRITIC harness] Step 3: evaluate and score")
        score_start = time.time()
        score_prompt = (
            "Steps 1 and 2 of the MANDATORY EVALUATION WORKFLOW are "
            "complete (scene observed, object data retrieved). Now perform "
            "STEPS 3-6 (physics review, placement evaluation, "
            "lighting/coverage analysis, synthesis) and return your final "
            "critique with scores."
            f"{scene_state_block}"
        )
        result = await Runner.run(
            starting_agent=critic_score,
            input=score_prompt,
            session=self.critic_session,
            max_turns=self.cfg.agents.critic_agent.max_turns,
            run_config=run_config,
        )
        self._record_module_timing("critic", "score_scene", score_start)
        log_agent_usage(result=result, agent_name="CRITIC (score)")
        self._record_llm_call_debug(
            agent_role="critic",
            event="score_scene",
            prompt=score_prompt,
            output=result.final_output or "",
            result=result,
        )

        # Parse structured output.
        response = result.final_output
        if not isinstance(response, CritiqueWithScores):
            raise TypeError(
                "Critic returned an unexpected final output type: "
                f"{type(response).__name__}"
            )

        # Log critique text and scores to console.
        log_agent_response(response=response.critique, agent_name="CRITIC")
        log_critique_scores(response, title="CRITIQUE SCORES")

        # Save scores to YAML next to scene renders (from observe_scene call).
        images_dir = self.rendering_manager.last_render_dir
        self._write_scores_and_memory(
            response=response,
            images_dir=images_dir,
            physics_context=physics_context,
        )

        # Compute score deltas and format for planner if we have previous scores.
        score_change_msg = ""
        if self.previous_scores is not None:
            score_change_msg = format_score_deltas_for_planner(
                current_scores=response,
                previous_scores=self.previous_scores,
                format_style="detailed",
            )

        safety_msg, checkpoint_scores, checkpoint_render_dir, checkpoint_accepted = (
            self._apply_furniture_safety_after_critique(
                scores=response,
                images_dir=images_dir,
                physics_context=physics_context,
            )
        )

        # Shift checkpoints only during iteration critiques, not final critique.
        # This preserves N-1 checkpoint for reset check in _finalize_scene_and_scores.
        if update_checkpoint and checkpoint_accepted:
            # Shift current checkpoint to previous before saving new one.
            # This maintains N-1 and N checkpoints for rollback functionality.
            self.previous_scene_checkpoint = self.scene_checkpoint
            self.previous_checkpoint_scores = self.checkpoint_scores
            self.previous_checkpoint_render_dir = self.checkpoint_render_dir

            # Save new checkpoint (current scene state).
            self.scene_checkpoint = copy.deepcopy(self.scene.to_state_dict())
            self.checkpoint_scores = checkpoint_scores
            self.checkpoint_render_dir = checkpoint_render_dir

            # Reuse render cache hash for checkpoint change detection.
            self.checkpoint_scene_hash = self.scene.content_hash()
        elif update_checkpoint:
            console_logger.info(
                "Skipping checkpoint update because furniture safety rejected "
                "the current candidate."
            )

        # Always update previous_scores for delta formatting in planner.
        self.previous_scores = response

        # Always track the final render directory (separate from checkpoint logic).
        # This is needed because final critique uses update_checkpoint=False, but we
        # still need to know the actual last render dir for copying to final output.
        self.final_render_dir = checkpoint_render_dir or images_dir
        self._last_scored_scene_hash = self.scene.content_hash()
        self._last_critique_render_profile = render_profile

        # Return natural language critique with score deltas for planner.
        self._record_module_timing(
            "critic",
            "request_critique_total",
            critique_start,
            extra={
                "update_checkpoint": update_checkpoint,
                "hard_check_short_circuit": False,
                "render_profile": render_profile,
            },
        )
        return response.critique + score_change_msg + safety_msg

    @abstractmethod
    def _get_design_change_prompt_enum(self) -> Any:
        """Get the prompt enum for design change instruction.

        Returns:
            Prompt enum for domain-specific design change instruction.
        """

    async def _request_design_change_impl(self, instruction: str) -> str:
        """Implementation for design change request.

        Args:
            instruction: Specific changes to make based on critique feedback.

        Returns:
            Designer's report of what was changed.
        """
        console_logger.info("Tool called: request_design_change")
        transaction = self._begin_furniture_design_transaction(call_kind="change")

        # Get instruction from prompt registry with domain-specific enum.
        prompt_enum = self._get_design_change_prompt_enum()
        full_instruction = self.prompt_registry.get_prompt(
            prompt_enum=prompt_enum,
            instruction=instruction,
        )
        memory_context = self._retrieve_working_memory_for_designer(instruction)
        if memory_context:
            full_instruction += "\n\n" + memory_context
        context_block = self._prepare_stage_context_for_llm(
            agent_role="designer",
            event="request_design_change",
            prompt=full_instruction,
        )
        if context_block:
            full_instruction += "\n\n" + context_block

        # Designer run with critique-based instruction.
        designer_start = time.time()
        render_dir_before = self.rendering_manager.last_render_dir
        try:
            result = await Runner.run(
                starting_agent=self.designer,
                input=full_instruction,
                session=self.designer_session,
                max_turns=self.cfg.agents.designer_agent.max_turns,
                run_config=self._create_run_config(),
            )
        except Exception as exc:
            self._record_llm_call_debug(
                agent_role="designer",
                event="request_design_change",
                prompt=full_instruction,
                error=f"{type(exc).__name__}: {exc}",
            )
            self._end_furniture_design_transaction(transaction)
            raise
        self._record_module_timing("designer", "request_design_change", designer_start)
        log_agent_usage(result=result, agent_name="DESIGNER (CHANGE)")
        self._record_llm_call_debug(
            agent_role="designer",
            event="request_design_change",
            prompt=full_instruction,
            output=result.final_output or "",
            result=result,
        )

        if result.final_output:
            log_agent_response(
                response=result.final_output, agent_name="DESIGNER (CHANGE)"
            )

        safety_msg = self._end_furniture_design_transaction(transaction)
        self._save_designer_working_memory(
            render_dir_before=render_dir_before,
            event="design_change",
            text=(result.final_output or "") + safety_msg,
        )
        return (result.final_output or "") + safety_msg

    @abstractmethod
    def _get_initial_design_prompt_enum(self) -> Any:
        """Get the prompt enum for initial design instruction.

        Returns:
            Prompt enum for domain-specific initial design instruction.
        """

    @abstractmethod
    def _get_initial_design_prompt_kwargs(self) -> dict:
        """Get prompt kwargs for initial design instruction.

        Returns:
            Dictionary of kwargs to pass to get_prompt() for initial design.
        """

    def _get_context_image_path(self) -> Path | None:
        """Get optional context image path for initial design.

        Subclasses can override to provide an AI-generated reference image
        that will be included in the initial design user message.

        Returns:
            Path to context image, or None if not available.
        """
        return None

    def _build_initial_design_input(self, instruction: str) -> str | list[dict]:
        """Build the input for initial design request.

        If a context image is available, constructs a multimodal message
        with both text instruction and the reference image.

        Args:
            instruction: Text instruction for the designer.

        Returns:
            Either plain text or a list with a multimodal user message.
        """
        context_image_path = self._get_context_image_path()
        if context_image_path and context_image_path.exists():
            # Build multimodal input with text + image.
            console_logger.info(
                f"Including context image in initial design: {context_image_path}"
            )
            image_base64 = encode_image_to_base64(context_image_path)
            return [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": instruction},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{image_base64}",
                        },
                    ],
                }
            ]
        # No context image - use plain text.
        return instruction

    async def _request_initial_design_impl(self) -> str:
        """Implementation for initial design request.

        Returns:
            Designer's report of initial design.
        """
        console_logger.info("Tool called: request_initial_design")
        transaction = self._begin_furniture_design_transaction(call_kind="initial")

        # Get instruction from prompt registry with domain-specific enum and kwargs.
        prompt_enum = self._get_initial_design_prompt_enum()
        prompt_kwargs = self._get_initial_design_prompt_kwargs()
        instruction = self.prompt_registry.get_prompt(
            prompt_enum=prompt_enum, **prompt_kwargs
        )
        memory_context = self._retrieve_working_memory_for_designer("initial design")
        if memory_context:
            instruction += "\n\n" + memory_context
        context_block = self._prepare_stage_context_for_llm(
            agent_role="designer",
            event="request_initial_design",
            prompt=instruction,
        )
        if context_block:
            instruction += "\n\n" + context_block

        # Build input (may include context image if enabled).
        input_message = self._build_initial_design_input(instruction)

        # Designer runs with initial design instruction.
        designer_start = time.time()
        render_dir_before = self.rendering_manager.last_render_dir
        try:
            result = await Runner.run(
                starting_agent=self.designer,
                input=input_message,
                session=self.designer_session,
                max_turns=self.cfg.agents.designer_agent.max_turns,
                run_config=self._create_run_config(),
            )
        except Exception as exc:
            self._record_llm_call_debug(
                agent_role="designer",
                event="request_initial_design",
                prompt=input_message,
                error=f"{type(exc).__name__}: {exc}",
            )
            self._end_furniture_design_transaction(transaction)
            raise
        self._record_module_timing("designer", "request_initial_design", designer_start)
        log_agent_usage(result=result, agent_name="DESIGNER (INITIAL)")
        self._record_llm_call_debug(
            agent_role="designer",
            event="request_initial_design",
            prompt=input_message,
            output=result.final_output or "",
            result=result,
        )

        if result.final_output:
            log_agent_response(
                response=result.final_output, agent_name="DESIGNER (INITIAL)"
            )

        safety_msg = self._end_furniture_design_transaction(transaction)
        self._save_designer_working_memory(
            render_dir_before=render_dir_before,
            event="initial_design",
            text=(result.final_output or "") + safety_msg,
        )
        return (result.final_output or "") + safety_msg
