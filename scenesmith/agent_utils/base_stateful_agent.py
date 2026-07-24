"""Base class for stateful agents using planner/designer/critic workflow.

This module provides the shared framework for all design agents (floor plan,
furniture, wall, manipuland), extracting the common multi-agent architecture
while allowing domain-specific customization through abstract methods and
subclass-defined tools.
"""

import asyncio
import copy
import logging
import os
import shutil
import time

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

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
    hard_state_repair_objective,
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
from scenesmith.agent_utils.thinking import (
    prepend_text_thinking_directive,
    thinking_directive_from_effort,
)
from scenesmith.agent_utils.turn_trimming_session import TurnTrimmingSession
from scenesmith.prompts import prompt_registry
from scenesmith.scene_expert.context_bundle import build_stage_context_bundle
from scenesmith.scene_expert.critic_feedback import (
    CriticFeedback,
    critic_feedback_contract,
    direct_critic_scoring_instructions,
    parse_critic_feedback,
)
from scenesmith.scene_expert.exceptions import StageValidationError
from scenesmith.utils.logging import BaseLogger
from scenesmith.utils.openai import encode_image_to_base64

console_logger = logging.getLogger(__name__)


@dataclass
class _AgentExecutionLease:
    """A pausable active-time lease for one nested Agents SDK invocation."""

    role: str
    timeout: asyncio.Timeout
    remaining_seconds: float
    resumed_at: float
    active_elapsed_seconds: float = 0.0

    def pause(self) -> None:
        if self.resumed_at <= 0:
            return
        now = asyncio.get_running_loop().time()
        elapsed = max(0.0, now - self.resumed_at)
        self.active_elapsed_seconds += elapsed
        self.remaining_seconds = max(0.0, self.remaining_seconds - elapsed)
        self.resumed_at = 0.0
        try:
            self.timeout.reschedule(None)
        except RuntimeError:
            # An already-expiring timeout cannot be rescheduled. Accounting is
            # still correct and the surrounding context will raise TimeoutError.
            pass

    def resume(self, maximum_seconds: float | None = None) -> None:
        if self.resumed_at > 0:
            return
        if maximum_seconds is not None:
            self.remaining_seconds = min(
                self.remaining_seconds,
                max(0.0, maximum_seconds),
            )
        loop = asyncio.get_running_loop()
        self.resumed_at = loop.time()
        try:
            self.timeout.reschedule(loop.time() + max(0.001, self.remaining_seconds))
        except RuntimeError:
            # The parent has already expired while its child was unwinding.
            pass


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
        working_memory_enabled = bool(_cfg_get(working_memory_cfg, "enabled", True))
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
        self._stage_runtime_budget: dict[str, Any] = {}
        self._stage_runtime_started_at: float | None = None
        self._critic_evaluation_started_at: float | None = None
        self._stage_runtime_exhausted = False
        self._stage_runtime_phase = "agent"
        self._allow_degraded_stage_completion = False
        self._degraded_stage_reasons: list[str] = []
        self._last_score_provenance: dict[str, Any] = {}
        self._last_trusted_critic_candidate: dict[str, Any] | None = None
        self._critical_retry_compact_context = False
        self._critical_retry_budget_expanded = False
        self._stage_role_active_consumed: dict[str, float] = {}
        self._agent_execution_leases: list[_AgentExecutionLease] = []
        self._last_critic_feedback = CriticFeedback()

    def _configure_stage_runtime(self, scene: Any) -> None:
        """Bind SceneExpert's advisory budget to this stage's real execution."""
        raw_budget = getattr(scene, "scene_expert_stage_budget", {}) or {}
        self.configure_stage_runtime_budget(raw_budget)

        asset_manager = getattr(self, "asset_manager", None)
        configure_asset_budget = getattr(
            asset_manager, "configure_runtime_budget", None
        )
        if callable(configure_asset_budget):
            configure_asset_budget(
                stage=str(
                    getattr(scene, "scene_expert_stage", self.agent_type.value)
                ),
                budget=self._stage_runtime_budget,
                required_objects=list(
                    getattr(scene, "scene_expert_required_objects", []) or []
                ),
            )

    def _refresh_asset_runtime_budget(self) -> None:
        """Reset the per-attempt asset gate for a critical completion retry."""
        # Floor-plan agents deliberately operate on ``self.layout`` and never
        # define ``self.scene``.  ``configure_stage_runtime_budget()`` is also
        # their public budget entry point, so asset-budget refresh must remain a
        # placement-only no-op instead of assuming every stateful agent owns a
        # RoomScene.
        scene = getattr(self, "scene", None)
        if scene is None:
            return
        asset_manager = getattr(self, "asset_manager", None)
        configure_asset_budget = getattr(
            asset_manager, "configure_runtime_budget", None
        )
        if callable(configure_asset_budget):
            configure_asset_budget(
                stage=str(
                    getattr(scene, "scene_expert_stage", self.agent_type.value)
                ),
                budget=self._stage_runtime_budget,
                required_objects=list(
                    getattr(scene, "scene_expert_required_objects", []) or []
                ),
            )

    def _expand_critical_retry_budget(self) -> None:
        """Grant one quality-critical retry a larger, fresh role budget."""
        if not self._stage_runtime_budget or self._critical_retry_budget_expanded:
            return
        multiplier = max(
            1.0,
            float(
                self._stage_budget_value(
                    "critical_retry_budget_multiplier",
                    1.5,
                )
                or 1.5
            ),
        )
        for key in (
            "max_wall_clock_seconds",
            "planner_active_max_seconds",
            "designer_active_max_seconds",
            "critic_active_max_seconds",
            "critic_evaluation_max_seconds",
            "max_designer_turns",
            "max_critic_turns",
            "max_asset_requests",
            "max_semantic_retries_per_family",
        ):
            value = float(self._stage_runtime_budget.get(key, 0) or 0)
            if value <= 0:
                continue
            expanded = value * multiplier
            self._stage_runtime_budget[key] = (
                int(round(expanded))
                if key.startswith("max_") and not key.endswith("_seconds")
                else expanded
            )
        self._critical_retry_budget_expanded = True

    def configure_stage_runtime_budget(self, raw_budget: Any) -> None:
        """Bind an execution budget when no ``RoomScene`` object is available.

        Floor-plan generation runs in an isolated house-level subprocess, so it
        cannot receive the ``RoomScene`` attributes used by placement stages.
        This public entry point gives that worker the same turn and wall-clock
        enforcement without reaching into private runtime state.
        """
        try:
            self._stage_runtime_budget = dict(raw_budget)
        except (TypeError, ValueError):
            self._stage_runtime_budget = {}
        self._stage_runtime_started_at = time.monotonic()
        self._critic_evaluation_started_at = None
        self._stage_runtime_exhausted = False
        self._critical_retry_budget_expanded = False
        self._refresh_asset_runtime_budget()
        self._stage_runtime_phase = "agent"
        self._last_score_provenance = {}
        self._last_trusted_critic_candidate = None
        self._stage_role_active_consumed = {}
        self._agent_execution_leases = []
        self._last_critic_feedback = CriticFeedback()

    async def prepare_stage_regeneration(self, reasons: list[str]) -> None:
        """Reset conversational/checkpoint state before a full stage redesign.

        Generated assets and server processes are intentionally retained, but the
        designer and critic histories are cleared so the retry is a new layout
        proposal rather than another incremental edit of the rejected candidate.
        """
        for session_name in ("designer_session", "critic_session"):
            session = getattr(self, session_name, None)
            clear_session = getattr(session, "clear_session", None)
            if callable(clear_session):
                await clear_session()
        initialize_checkpoint_attributes(self)
        self._reset_planner_budget_tracking()
        self._reset_critic_candidate_cache()
        # A full stage regeneration is a new agent attempt.  Reusing the
        # exhausted timestamp made the replacement designer and critic no-ops,
        # even though the expensive assets are intentionally cached.
        self._stage_runtime_started_at = time.monotonic()
        self._critic_evaluation_started_at = None
        self._stage_runtime_exhausted = False
        self._stage_role_active_consumed = {}
        self._agent_execution_leases = []
        rendering_manager = getattr(self, "rendering_manager", None)
        if rendering_manager is not None:
            rendering_manager.clear_cache()
        self._allow_degraded_stage_completion = False
        self._degraded_stage_reasons = []
        console_logger.warning(
            "Reset %s designer/critic state for stage regeneration: %s",
            self.agent_type.value,
            "; ".join(reasons),
        )

    async def retry_final_critic_evaluation(self) -> None:
        """Retry only the final visual decision for an otherwise valid scene."""
        self._expand_critical_retry_budget()
        self._stage_runtime_started_at = time.monotonic()
        self._critic_evaluation_started_at = None
        self._stage_runtime_exhausted = False
        self._stage_role_active_consumed.pop("critic", None)
        self._stage_runtime_phase = "fallback"
        self._critical_retry_compact_context = True
        try:
            await self._request_critique_impl(update_checkpoint=False)
            await self._finalize_scene_and_scores()
        finally:
            self._critical_retry_compact_context = False
            self._stage_runtime_phase = "agent"

    async def complete_repair_exhausted_stage(self, reasons: list[str]) -> None:
        """Persist a diagnosed degraded stage instead of aborting the pipeline."""
        self._allow_degraded_stage_completion = True
        self._degraded_stage_reasons = list(reasons)
        if self.scene is not None:
            setattr(
                self.scene,
                "scene_expert_degraded_stage_reasons",
                list(reasons),
            )
        await self._finalize_scene_and_scores()

    def _stage_budget_value(self, key: str, default: Any) -> Any:
        return self._stage_runtime_budget.get(key, default)

    def _planner_completion_contract(self) -> str:
        """Return a runtime planner directive for mandatory stage output."""
        if not self._stage_runtime_budget:
            return ""
        minimum = max(
            0,
            int(
                getattr(
                    getattr(self, "scene", None),
                    "scene_expert_min_output_objects",
                    0,
                )
                or self._stage_budget_value("min_output_objects", 0)
                or 0
            ),
        )
        configured_maximum = int(
            getattr(
                getattr(self, "scene", None),
                "scene_expert_max_output_objects",
                0,
            )
            or self._stage_budget_value("max_output_objects", 0)
            or 0
        )
        if minimum <= 0:
            return ""
        maximum_clause = (
            f" and no more than {max(configured_maximum, minimum)}"
            if configured_maximum > 0
            else ""
        )
        return (
            "\n\nRUNTIME STAGE COMPLETION CONTRACT: You must call "
            "request_initial_design() and ensure the designer places at least "
            f"{minimum}{maximum_clause} stage-native objects before finish_stage. "
            "A zero-object result is not valid for this run, even if the generic "
            "room guidance says decoration is optional. Rules only validate the "
            "count; let the designer choose suitable object types and placements."
        )

    def _effective_critique_round_limit(self) -> int:
        configured = max(0, int(_cfg_get(self.cfg, "max_critique_rounds", 0)))
        if not self._stage_runtime_budget:
            return configured
        stage_limit = max(
            0,
            int(
                self._stage_budget_value(
                    "max_designer_iterations",
                    configured,
                )
            ),
        )
        return min(configured, stage_limit)

    def _remaining_role_active_seconds(self, role: str) -> float | None:
        """Return a role's exclusive active-time allowance.

        Planner time is charged only while the planner model is active. Nested
        designer/critic calls pause the planner lease, so the parent coordinator
        can no longer cancel an otherwise valid child call merely because both
        were charged for the same wall-clock interval.
        """

        key = {
            "planner": "planner_active_max_seconds",
            "designer": "designer_active_max_seconds",
            "critic": "critic_active_max_seconds",
        }.get(role)
        if not key:
            return None
        limit = float(self._stage_budget_value(key, 0.0) or 0.0)
        if limit <= 0:
            return None
        consumed = float(self._stage_role_active_consumed.get(role, 0.0))
        return limit - consumed

    def _begin_critic_evaluation(self) -> None:
        """Start one isolated visual-scoring transaction for this candidate.

        Critic evidence rendering happens before this boundary.  Resetting the
        critic's active lease here prevents earlier candidates in SceneSmith's
        native planner loop from starving the final candidate of its configured
        evaluation window.
        """

        self._critic_evaluation_started_at = time.monotonic()
        self._stage_role_active_consumed.pop("critic", None)

    def _critic_score_call_timeout(
        self,
        provider_default_seconds: float,
    ) -> float | None:
        """Return the provider fallback timeout for a visual score request.

        SceneExpert has one authoritative transaction deadline:
        ``critic_evaluation_max_seconds``.  Applying a second, shorter per-call
        timeout caused structured-output recovery turns to be cancelled even
        after the first backend request completed successfully.  Disabled
        SceneExpert paths retain their provider-specific legacy timeout.
        """

        evaluation_limit = float(
            self._stage_budget_value("critic_evaluation_max_seconds", 0.0) or 0.0
        )
        if self._stage_runtime_budget and evaluation_limit > 0:
            return None
        return (
            float(provider_default_seconds)
            if float(provider_default_seconds) > 0
            else None
        )

    @staticmethod
    def _minimum_positive_seconds(*values: float | None) -> float | None:
        positive = [float(value) for value in values if value is not None]
        return min(positive) if positive else None

    def _pause_parent_execution_lease(self, role: str) -> _AgentExecutionLease | None:
        """Pause a planner's active-time lease while its child agent runs."""

        if not self._agent_execution_leases:
            return None
        parent = self._agent_execution_leases[-1]
        if parent.role != "planner" or role == "planner":
            return None
        parent.pause()
        return parent

    def _resume_parent_execution_lease(
        self,
        parent: _AgentExecutionLease | None,
    ) -> None:
        if parent is None:
            return
        stage_remaining = self._remaining_stage_seconds(parent.role)
        parent.resume(maximum_seconds=stage_remaining)

    def _remaining_stage_seconds(self, role: str | None = None) -> float | None:
        """Return phase-aware time without letting design consume verification.

        Designer calls stop before the critic and fallback reserves. Once visual
        evidence is ready, critic scoring is a required quality transaction with
        its own deadline; it must not inherit a nearly exhausted design-stage
        wall clock. The bounded planner loop still controls how many such
        candidate evaluations can occur.
        """
        critic_evaluation_started_at = getattr(
            self, "_critic_evaluation_started_at", None
        )
        if role == "critic" and critic_evaluation_started_at is not None:
            evaluation_limit = float(
                self._stage_budget_value("critic_evaluation_max_seconds", 0.0)
                or 0.0
            )
            if evaluation_limit > 0:
                return evaluation_limit - (
                    time.monotonic() - critic_evaluation_started_at
                )

        wall_clock_limit = float(
            self._stage_budget_value("max_wall_clock_seconds", 0.0) or 0.0
        )
        if wall_clock_limit <= 0 or self._stage_runtime_started_at is None:
            return evaluation_remaining
        reserve_fraction = 0.0
        critic_reserve = float(
            self._stage_budget_value("critic_reserve_fraction", 0.25) or 0.0
        )
        fallback_reserve = float(
            self._stage_budget_value("fallback_reserve_fraction", 0.10) or 0.0
        )
        finalization_reserve = float(
            self._stage_budget_value("finalization_reserve_fraction", 0.05) or 0.0
        )
        if role == "designer":
            reserve_fraction = critic_reserve + finalization_reserve
            if self._stage_runtime_phase != "fallback":
                reserve_fraction += fallback_reserve
        elif role == "planner":
            # A shorter outer planner deadline cancels an in-flight designer or
            # critic before its own valid budget expires.  Only finalization is
            # reserved here; nested role deadlines preserve critic/fallback time.
            reserve_fraction = finalization_reserve
        elif role == "critic":
            reserve_fraction = finalization_reserve
            if self._stage_runtime_phase != "fallback":
                reserve_fraction += fallback_reserve
        reserve_fraction = max(0.0, min(0.9, reserve_fraction))
        elapsed = time.monotonic() - self._stage_runtime_started_at
        stage_remaining = wall_clock_limit * (1.0 - reserve_fraction) - elapsed
        return stage_remaining

    @staticmethod
    def _is_agent_budget_error(error: BaseException) -> bool:
        current: BaseException | None = error
        while current is not None:
            if type(current).__name__ in {
                "MaxTurnsExceeded",
                "ModelBehaviorError",
            } and (
                type(current).__name__ == "MaxTurnsExceeded"
                or "max turns" in str(current).lower()
            ):
                return True
            current = current.__cause__ or current.__context__
        return "max turns" in str(error).lower()

    async def _run_agent_with_stage_sla(
        self,
        *,
        starting_agent: Agent,
        input: Any,
        role: str,
        event: str,
        configured_max_turns: int | None = None,
        session: Session | None = None,
        run_config: RunConfig | None = None,
        call_timeout_seconds: float | None = None,
    ) -> RunResult | None:
        """Run one Agents SDK call under the role and phase-specific SLA.

        Planning/design observe the stage clock and their exclusive active-time
        leases. Required structured critic scoring observes its isolated quality
        transaction instead. Budget exhaustion is an execution outcome, not a
        process failure; callers preserve the current candidate for validation.
        """
        if role == "planner" and isinstance(input, str):
            input += self._planner_completion_contract()
        role_turn_key = {
            "planner": "max_planner_turns",
            "designer": "max_designer_turns",
            "critic": "max_critic_turns",
        }.get(role, "")
        max_turns = configured_max_turns
        if role_turn_key and self._stage_runtime_budget:
            stage_turns = int(self._stage_budget_value(role_turn_key, 0) or 0)
            if stage_turns > 0:
                max_turns = (
                    min(int(max_turns), stage_turns)
                    if max_turns is not None
                    else stage_turns
                )

        remaining = self._minimum_positive_seconds(
            self._remaining_stage_seconds(role),
            self._remaining_role_active_seconds(role),
            (
                call_timeout_seconds
                if call_timeout_seconds is not None and call_timeout_seconds > 0
                else None
            ),
        )
        if remaining is not None and remaining <= 0:
            self._stage_runtime_exhausted = True
            self._planner_budget_exhausted = True
            console_logger.warning(
                "Stage SLA exhausted before %s/%s; preserving current candidate",
                role,
                event,
            )
            return None

        parent_lease = self._pause_parent_execution_lease(role)
        start_time = time.time()
        try:
            run_kwargs: dict[str, Any] = {
                "starting_agent": starting_agent,
                "input": input,
            }
            if max_turns is not None:
                run_kwargs["max_turns"] = max(1, int(max_turns))
            if session is not None:
                run_kwargs["session"] = session
            if run_config is not None:
                run_kwargs["run_config"] = run_config
            if remaining is None:
                return await Runner.run(**run_kwargs)
            async with asyncio.timeout(max(0.1, remaining)) as timeout:
                lease = _AgentExecutionLease(
                    role=role,
                    timeout=timeout,
                    remaining_seconds=max(0.1, remaining),
                    resumed_at=asyncio.get_running_loop().time(),
                )
                self._agent_execution_leases.append(lease)
                try:
                    return await Runner.run(**run_kwargs)
                finally:
                    lease.pause()
                    if self._agent_execution_leases:
                        popped = self._agent_execution_leases.pop()
                        if popped is not lease:
                            self._agent_execution_leases.clear()
                            console_logger.error(
                                "SceneExpert nested execution lease stack became "
                                "inconsistent; cleared it defensively"
                            )
                    self._stage_role_active_consumed[role] = (
                        float(self._stage_role_active_consumed.get(role, 0.0))
                        + lease.active_elapsed_seconds
                    )
        except Exception as exc:
            budget_error = isinstance(exc, TimeoutError) or self._is_agent_budget_error(
                exc
            )
            if not budget_error:
                raise
            self._stage_runtime_exhausted = isinstance(exc, TimeoutError)
            if role in {"planner", "designer"}:
                self._planner_budget_exhausted = True
            self._record_module_timing(
                role,
                f"{event}_budget_exhausted",
                start_time,
                extra={
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "max_turns": max_turns,
                    "remaining_stage_seconds": remaining,
                },
            )
            self._record_llm_call_debug(
                agent_role=role,
                event=f"{event}_budget_exhausted",
                prompt=input,
                error=f"{type(exc).__name__}: {exc}",
            )
            console_logger.warning(
                "%s/%s reached its execution budget (%s: %s); preserving the "
                "current candidate for deterministic validation",
                role,
                event,
                type(exc).__name__,
                exc,
            )
            return None
        finally:
            self._resume_parent_execution_lease(parent_lease)

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
            console_logger.warning(
                "Failed to record timing %s/%s: %s", module, event, e
            )

    def _retrieve_working_memory_for_designer(self, query: str) -> str:
        """Fetch compact online memory to inject into the next designer call."""
        if self._critical_retry_compact_context:
            return ""
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
                or (
                    [self._pending_hard_repair_hint]
                    if self._pending_hard_repair_hint
                    else []
                ),
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
            max_chars = (
                min(900, self._stage_context_max_chars())
                if self._critical_retry_compact_context
                else self._stage_context_max_chars()
            )
            return bundle.to_llm_text(max_chars=max_chars)
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
        scores: CritiqueWithScores | None,
        critique: str,
        physics_context: str,
        score_source: str,
    ) -> None:
        """Attach critic scores and critique text to the current render memory."""
        if render_dir is None:
            return
        try:
            feedback = parse_critic_feedback(critique)
            self.stage_working_memory.save_render_record(
                render_dir=render_dir,
                role="critic",
                event=event,
                scene=self.scene,
                scores=scores,
                critique=critique,
                score_source=score_source,
                extra={
                    "physics_context": physics_context[:1500],
                    "critic_feedback": feedback.model_dump(),
                },
            )
        except Exception as e:
            console_logger.warning("Failed to save critic working memory: %s", e)

    def _write_score_artifacts(
        self,
        *,
        response: CritiqueWithScores,
        images_dir: Path,
        physics_context: str = "",
        event: str = "critique",
    ) -> dict[str, Any]:
        """Persist decision scores, source-specific scores, and provenance."""
        provenance = self._score_provenance_for_response(
            response=response,
            physics_context=physics_context,
            event=event,
        )
        score_source = str(provenance["score_source"])
        source_filename = str(provenance["source_scores_file"])
        hard_check_passed = provenance["hard_check_passed"]

        # A transport timeout has no numeric meaning. Keep a diagnostic marker,
        # but never emit made-up 5/10 values into scores.yaml or the memory path.
        if score_source == "critic_fallback":
            diagnostic = {
                "schema_version": "1.0",
                "status": "unscored",
                "reason": str(response.critique or "visual critic unavailable"),
                "retryable": True,
            }
            with open(images_dir / source_filename, "w") as f:
                yaml.dump(
                    data=diagnostic,
                    stream=f,
                    default_flow_style=False,
                    sort_keys=False,
                )
            with open(images_dir / "score_provenance.yaml", "w") as f:
                yaml.dump(
                    data=provenance,
                    stream=f,
                    default_flow_style=False,
                    sort_keys=False,
                )
            if hard_check_passed is not None:
                with open(images_dir / "hard_check_report.yaml", "w") as f:
                    yaml.dump(
                        data={
                            "schema_version": "1.0",
                            "passed": hard_check_passed,
                            "evidence": str(physics_context or ""),
                            "decision_scores_file": "",
                            "decision_scores_are_synthetic": False,
                        },
                        stream=f,
                        default_flow_style=False,
                        sort_keys=False,
                    )
            console_logger.warning(
                "Visual critic did not complete; wrote unscored diagnostic to %s "
                "without placeholder numeric grades",
                images_dir / source_filename,
            )
            return provenance

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

        with open(images_dir / source_filename, "w") as f:
            yaml.dump(
                data=scores_dict,
                stream=f,
                default_flow_style=False,
                sort_keys=False,
            )
        with open(images_dir / "score_provenance.yaml", "w") as f:
            yaml.dump(
                data=provenance,
                stream=f,
                default_flow_style=False,
                sort_keys=False,
            )
        if hard_check_passed is not None:
            with open(images_dir / "hard_check_report.yaml", "w") as f:
                yaml.dump(
                    data={
                        "schema_version": "1.0",
                        "passed": hard_check_passed,
                        "evidence": str(physics_context or response.critique or ""),
                        "decision_scores_file": (
                            source_filename
                            if score_source == "deterministic_hard_check"
                            else ""
                        ),
                        "decision_scores_are_synthetic": (
                            score_source == "deterministic_hard_check"
                        ),
                    },
                    stream=f,
                    default_flow_style=False,
                    sort_keys=False,
                )
        console_logger.info(
            "Score provenance saved to %s (source=%s)",
            images_dir / "score_provenance.yaml",
            score_source,
        )
        return provenance

    def _score_provenance_for_response(
        self,
        *,
        response: CritiqueWithScores,
        physics_context: str = "",
        event: str = "critique",
    ) -> dict[str, Any]:
        """Classify score evidence even when no render directory was produced."""
        critique_upper = str(response.critique or "").upper()
        if event == "deterministic_hard_fail":
            score_source = "deterministic_hard_check"
            source_filename = "hard_check_decision_scores.yaml"
            score_semantics = (
                "Synthetic repair-priority grades; not a VLM quality assessment."
            )
        elif any(
            marker in critique_upper
            for marker in (
                "CRITIC DEGRADED",
                "TRANSIENT LOCAL VLM TIMEOUT",
                "VISUAL CRITIC UNAVAILABLE",
            )
        ):
            score_source = "critic_fallback"
            source_filename = "critic_unavailable.yaml"
            score_semantics = (
                "Unscored transport failure; no numeric quality assessment exists."
            )
        else:
            score_source = "vlm_critic"
            source_filename = "vlm_scores.yaml"
            score_semantics = "VLM critic quality assessment."

        hard_check_passed: bool | None
        if score_source == "deterministic_hard_check":
            hard_check_passed = False
        elif score_source == "vlm_critic":
            hard_check_passed = True
        elif (
            "DETERMINISTIC HARD CHECKS PASSED" in critique_upper
            or (
                "LAYOUT=OK" in critique_upper
                and "CONNECTIVITY=OK" in critique_upper
            )
        ):
            hard_check_passed = True
        elif "LAYOUT=" in critique_upper or "CONNECTIVITY=" in critique_upper:
            hard_check_passed = False
        else:
            hard_check_passed = None

        provenance = {
            "schema_version": "1.0",
            "score_source": score_source,
            "vlm_scoring_performed": score_source == "vlm_critic",
            "score_scale": (
                "none" if score_source == "critic_fallback" else "0-10"
            ),
            "hard_check_passed": hard_check_passed,
            "scores_semantics": score_semantics,
            "decision_scores_file": (
                "" if score_source == "critic_fallback" else "scores.yaml"
            ),
            "source_scores_file": source_filename,
            "hard_check_evidence": str(physics_context or ""),
        }
        return provenance

    @staticmethod
    def _normalized_visual_score(scores: CritiqueWithScores | None) -> float | None:
        """Return the mean 0-1 VLM quality score for one candidate."""
        if scores is None:
            return None
        categories = list(scores.get_scores())
        if not categories:
            return None
        return sum(max(0, min(10, score.grade)) for score in categories) / (
            10.0 * len(categories)
        )

    def _write_scores_and_memory(
        self,
        *,
        response: CritiqueWithScores,
        images_dir: Path | None,
        physics_context: str,
        event: str = "critique",
    ) -> None:
        """Persist score artifacts and stage working memory for a render."""
        provenance = self._score_provenance_for_response(
            response=response,
            physics_context=physics_context,
            event=event,
        )
        # Provenance is a control-plane signal, not merely a render artifact.
        # Record it before checking images_dir so a critic timeout that happened
        # before observe_scene can never masquerade as a trusted VLM score.
        self._last_score_provenance = dict(provenance)
        if images_dir:
            provenance = self._write_score_artifacts(
                response=response,
                images_dir=images_dir,
                physics_context=physics_context,
                event=event,
            )
            if (
                provenance.get("score_source") == "vlm_critic"
                and self.scene is not None
            ):
                controller = getattr(self, "furniture_safety_controller", None)
                weighted_score = None
                if controller is not None and getattr(controller, "enabled", False):
                    weighted_score = controller.evaluate_scores(response).weighted_score
                self._last_trusted_critic_candidate = {
                    "scene_state": copy.deepcopy(self.scene.to_state_dict()),
                    "scores": copy.deepcopy(response),
                    "render_dir": images_dir,
                    "weighted_score": weighted_score,
                    "score_source": "vlm_critic",
                }
            self._save_critic_working_memory(
                render_dir=images_dir,
                event=event,
                scores=(
                    response
                    if provenance.get("score_source")
                    in {"vlm_critic", "deterministic_hard_check"}
                    else None
                ),
                critique=response.critique,
                physics_context=physics_context,
                score_source=str(provenance.get("score_source", "unknown")),
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

    def _critic_fast_path_value(self, key: str, default: Any) -> Any:
        return _cfg_get(self._critic_fast_path_cfg(), key, default)

    def _critic_fast_path_enabled(self, key: str, default: bool = True) -> bool:
        return bool(self._critic_fast_path_value(key, default))

    def _sceneexpert_critic_feedback_contract(self) -> str:
        if not self._stage_runtime_budget:
            return ""
        return critic_feedback_contract()

    def _remember_critic_feedback(self, critique: str) -> CriticFeedback:
        feedback = parse_critic_feedback(critique)
        self._last_critic_feedback = feedback
        return feedback

    def _critic_feedback_for_planner(
        self,
        critique: str,
        *,
        max_chars: int = 5000,
    ) -> str:
        feedback = self._remember_critic_feedback(critique)
        return feedback.to_designer_text(max_chars=max_chars)

    def _critic_feedback_for_designer(self, max_chars: int = 5000) -> str:
        feedback = getattr(self, "_last_critic_feedback", CriticFeedback())
        return feedback.to_designer_text(max_chars=max_chars)

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
            before_hard_state = current_state
            before_hash = self.scene.content_hash() if self.scene is not None else ""
            before_scene_state = (
                copy.deepcopy(self.scene.to_state_dict())
                if self.scene is not None
                else None
            )
            before_objective = hard_state_repair_objective(current_state)
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
            if repaired_hard_state is not None:
                after_objective = hard_state_repair_objective(repaired_hard_state)
                if (
                    after_objective > before_objective
                    and before_scene_state is not None
                ):
                    self.scene.restore_from_state_dict(before_scene_state)
                    self.rendering_manager.clear_cache()
                    self._reset_critic_candidate_cache()
                    current_state = before_hard_state
                    physics_context = None
                    console_logger.warning(
                        "Rolled back deterministic repair from %s because hard-state "
                        "objective worsened from %s to %s",
                        source,
                        before_objective,
                        after_objective,
                    )
                    all_actions.append(
                        "rolled back repair that worsened hard constraints"
                    )
                    break
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
        return (
            self._critic_candidate_cache.get("scene_hash") == self.scene.content_hash()
        )

    def _get_cached_physics_context(self) -> str:
        if (
            self._cache_valid_for_current_scene()
            and "physics_context" in self._critic_candidate_cache
        ):
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
        if self.scene is not None and self._stage_runtime_budget:
            object_type = self.agent_type.to_object_type()
            if object_type is not None:
                stage_objects = self.scene.get_objects_by_type(object_type)
                minimum = int(
                    getattr(self.scene, "scene_expert_min_output_objects", 0) or 0
                )
                maximum = int(
                    getattr(self.scene, "scene_expert_max_output_objects", 0) or 0
                )
                if len(stage_objects) < minimum:
                    hard_reasons.append(
                        "missing required stage output: "
                        f"{self.agent_type.value} produced {len(stage_objects)} "
                        f"objects but requires at least {minimum}"
                    )
                if maximum > 0 and len(stage_objects) > maximum:
                    hard_reasons.append(
                        "stage object count exceeded: "
                        f"{self.agent_type.value} produced {len(stage_objects)} "
                        f"objects but allows at most {maximum}"
                    )

                present_text = [
                    " ".join(
                        str(value or "").lower()
                        for value in (
                            getattr(obj, "object_id", ""),
                            getattr(obj, "name", ""),
                            getattr(obj, "description", ""),
                        )
                    )
                    for obj in stage_objects
                ]
                for required in list(
                    getattr(self.scene, "scene_expert_required_objects", []) or []
                ):
                    required_text = str(required).strip().lower()
                    if required_text and not any(
                        required_text in candidate or candidate in required_text
                        for candidate in present_text
                        if candidate
                    ):
                        hard_reasons.append(
                            f"missing required stage object '{required}'"
                        )
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
        if any("stage object count exceeded" in reason.lower() for reason in reasons):
            hints.append(
                "Remove the least essential stage-native extras until the configured "
                "maximum is met; preserve every explicitly required object."
            )
        if any("collision" in reason.lower() for reason in reasons):
            hints.append(
                "Resolve collisions by moving, snapping, or reducing only the involved "
                "modifiable objects; do not delete required prompt objects."
            )
        if any(
            "door" in reason.lower() or "open connection" in reason.lower()
            for reason in reasons
        ):
            hints.append(
                "Clear door/open-connection clearance first; keep a walkable path from "
                "the doorway into the room."
            )
        if any(
            "fallen" in reason.lower() or "below-floor" in reason.lower()
            for reason in reasons
        ):
            hints.append(
                "Restore fallen or below-floor objects onto a valid support surface, "
                "or remove only optional unstable small objects."
            )
        if any(
            "geometry construction" in reason.lower() or "drake/qhull" in reason.lower()
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

    def _make_category_score(
        self, name: str, grade: int, comment: str
    ) -> CategoryScore:
        return CategoryScore(
            name=name, grade=max(0, min(10, int(grade))), comment=comment
        )

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
                functionality=self._make_category_score(
                    "functionality", 2, repair_comment
                ),
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
                functionality=self._make_category_score(
                    "functionality", 2, repair_comment
                ),
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
                functionality=self._make_category_score(
                    "functionality", 2, repair_comment
                ),
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
                functionality=self._make_category_score(
                    "functionality", 2, repair_comment
                ),
                layout=self._make_category_score("layout", 3, hard_comment),
                prompt_following=self._make_category_score(
                    "prompt_following", 3, repair_comment
                ),
            )
        raise TypeError(
            "Cannot create deterministic critique for output type "
            f"{getattr(output_type, '__name__', output_type)}"
        )

    def _make_transient_critic_fallback_scores(
        self,
        *,
        error: Exception,
    ) -> CritiqueWithScores:
        """Create an in-memory control response when local vLLM times out.

        The hard-check-first path has already verified required objects and
        deterministic geometry before this fallback is reached. Numeric values
        keep the legacy planner schema satisfied but are never persisted as
        scores or admitted to SceneExpert verification/memory.
        """
        output_type = self._critic_output_type or type(self.previous_scores)
        detail = f"Visual critic unavailable: {type(error).__name__}: {error}"
        critique = (
            "TRANSIENT LOCAL VLM TIMEOUT DURING VISUAL CRITIC SCORING. "
            "Deterministic hard checks passed, but visual quality remains "
            f"unverified. {detail}"
        )
        neutral = "Conservative fallback; visual critic did not complete."
        if output_type is FurnitureCritiqueWithScores:
            return FurnitureCritiqueWithScores(
                critique=critique,
                realism=self._make_category_score("realism", 5, neutral),
                functionality=self._make_category_score("functionality", 5, neutral),
                layout=self._make_category_score("layout", 5, neutral),
                layout_plausibility=self._make_category_score(
                    "layout_plausibility", 5, neutral
                ),
                holistic_completeness=self._make_category_score(
                    "holistic_completeness", 5, neutral
                ),
                prompt_following=self._make_category_score(
                    "prompt_following",
                    8,
                    "Required-object deterministic checks passed; visual review timed out.",
                ),
                reachability=self._make_category_score("reachability", 5, neutral),
            )
        if output_type is ManipulandCritiqueWithScores:
            return ManipulandCritiqueWithScores(
                critique=critique,
                realism=self._make_category_score("realism", 5, neutral),
                functionality=self._make_category_score("functionality", 5, neutral),
                layout=self._make_category_score("layout", 5, neutral),
                holistic_completeness=self._make_category_score(
                    "holistic_completeness", 5, neutral
                ),
                prompt_following=self._make_category_score(
                    "prompt_following", 6, neutral
                ),
            )
        if output_type is WallCritiqueWithScores:
            return WallCritiqueWithScores(
                critique=critique,
                realism=self._make_category_score("realism", 5, neutral),
                functionality=self._make_category_score("functionality", 5, neutral),
                layout=self._make_category_score("layout", 5, neutral),
                holistic_completeness=self._make_category_score(
                    "holistic_completeness", 5, neutral
                ),
                prompt_following=self._make_category_score(
                    "prompt_following", 6, neutral
                ),
            )
        if output_type is CeilingCritiqueWithScores:
            return CeilingCritiqueWithScores(
                critique=critique,
                realism=self._make_category_score("realism", 5, neutral),
                functionality=self._make_category_score("functionality", 5, neutral),
                layout=self._make_category_score("layout", 5, neutral),
                prompt_following=self._make_category_score(
                    "prompt_following", 6, neutral
                ),
            )
        raise TypeError(
            "Cannot create transient critic fallback for output type "
            f"{getattr(output_type, '__name__', output_type)}"
        )

    @staticmethod
    def _is_transient_model_error(error: Exception) -> bool:
        transient_names = {
            "APITimeoutError",
            "APIConnectionError",
            "ReadTimeout",
            "ConnectTimeout",
            "ConnectError",
            "TimeoutError",
        }
        current: BaseException | None = error
        while current is not None:
            if type(current).__name__ in transient_names:
                return True
            current = current.__cause__ or current.__context__
        text = str(error).lower()
        return "timed out" in text or "timeout" in text

    def _critic_render_profile_name(self, update_checkpoint: bool) -> str:
        if update_checkpoint and self._critic_fast_path_enabled(
            "use_intermediate_render_profile", True
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
        if (
            self._cache_valid_for_current_scene()
            and "scene_state" in self._critic_candidate_cache
        ):
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
        method = (
            getattr(owner, "_observe_scene_impl", None) if owner is not None else None
        )
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

    def _collect_direct_critic_observation(
        self, render_profile: str
    ) -> tuple[Path | None, list[dict[str, str]], str]:
        """Render once in-process and return compact multimodal critic content."""
        owner = getattr(self, "_critic_vision_tools", None)
        method = (
            getattr(owner, "_observe_scene_impl", None) if owner is not None else None
        )
        outputs: list[Any] = []
        fallback_image_urls: list[str] = []
        if method is None:
            render_dir = self._observe_scene_for_synthetic_score(render_profile)
            if render_dir is not None:
                for image_path in sorted(render_dir.glob("*.png")):
                    fallback_image_urls.append(
                        "data:image/png;base64,"
                        + encode_image_to_base64(image_path)
                    )
        else:
            observe_start = time.time()
            with self.rendering_manager.use_render_profile(render_profile):
                outputs = list(method() or [])
            self._record_module_timing(
                "critic",
                "observe_scene_direct",
                observe_start,
                extra={"render_profile": render_profile},
            )
            render_dir = self.rendering_manager.last_render_dir

        max_images = max(
            1,
            int(
                _cfg_get(
                    self._critic_fast_path_cfg(),
                    "direct_multimodal_max_images",
                    6,
                )
                or 6
            ),
        )
        image_parts: list[dict[str, str]] = []
        notes: list[str] = []
        for image_url in fallback_image_urls[:max_images]:
            image_parts.append({"type": "input_image", "image_url": image_url})
        for output in outputs:
            image_url = getattr(output, "image_url", None)
            if image_url and len(image_parts) < max_images:
                image_parts.append(
                    {"type": "input_image", "image_url": str(image_url)}
                )
            text_output = getattr(output, "text", None)
            if text_output:
                notes.append(str(text_output))
        return render_dir, image_parts, "\n".join(notes)

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

        score_source = self._last_score_provenance.get("score_source")
        if score_source and score_source != "vlm_critic":
            # Transport fallbacks and deterministic repair-priority grades are
            # evidence, not visual quality judgments. They must never displace a
            # real critic-scored candidate or claim that fallback improved it.
            hard_state = self._evaluate_current_furniture_hard_state(
                physics_context=physics_context
            )
            if hard_state is None or hard_state.hard_valid:
                controller.remember_hard_valid_scene_state(
                    scene_state=self.scene.to_state_dict(),
                    source=f"{score_source}_unscored",
                )
            if controller.best_scene_state is not None:
                self._restore_furniture_scene_state(controller.best_scene_state)
            return (
                "\n\n**Safety Controller:** visual critic unavailable; "
                "preserved the best hard-valid candidate without treating "
                "fallback grades as quality scores.",
                controller.best_scores,
                controller.best_render_dir,
                controller.best_scene_state is not None,
            )

        hard_state_evaluation = self._evaluate_current_furniture_hard_state(
            physics_context=physics_context
        )
        candidate_state = copy.deepcopy(self.scene.to_state_dict())
        decision = controller.consider_candidate(
            scores=scores,
            scene_state=candidate_state,
            render_dir=images_dir,
            hard_state_evaluation=hard_state_evaluation,
            score_source="vlm_critic",
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

        # Bound local-model completions explicitly. Tool-rich agents otherwise
        # inherit the backend's large default and a single malformed response can
        # occupy the floor-plan worker for tens of minutes.
        output_limits = getattr(self.cfg.openai, "max_output_tokens", None)
        runtime_limit = None
        if settings_key and self._stage_runtime_budget:
            runtime_limit = self._stage_budget_value(
                f"{settings_key}_max_output_tokens",
                None,
            )
        if settings_key and output_limits is not None:
            max_tokens = _cfg_get(output_limits, settings_key, None)
        else:
            max_tokens = None
        if runtime_limit is not None and int(runtime_limit) > 0:
            max_tokens = runtime_limit
        if max_tokens is not None and int(max_tokens) > 0:
            kwargs["max_tokens"] = int(max_tokens)

        # Add tool_choice to force specific tool call first.
        if tool_choice:
            kwargs["tool_choice"] = tool_choice

        # Add parallel_tool_calls setting if specified.
        if parallel_tool_calls is not None:
            kwargs["parallel_tool_calls"] = parallel_tool_calls

        return ModelSettings(**kwargs) if kwargs else None

    def _get_agent_instructions(
        self, prompt_enum: Any, settings_key: str, **kwargs: Any
    ) -> str:
        """Render prompt instructions and attach the configured thinking mode."""
        instructions = self.prompt_registry.get_prompt(
            prompt_enum=prompt_enum, **kwargs
        )
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
        freeze_selected_candidate = bool(
            getattr(self, "_freeze_selected_fallback_candidate", False)
        )
        if (
            controller
            and controller.enabled
            and controller.best_scene_state is not None
            and not freeze_selected_candidate
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
            if freeze_selected_candidate:
                final_repair_actions: list[str] = []
            else:
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
                # The repair changed the canonical scene after the previous
                # render/score decision.  Never copy the stale hard-fail
                # artifacts as if they described the repaired final state.
                if final_hard_state is None or final_hard_state.hard_valid:
                    repaired_hash = self.scene.content_hash()
                    if self._last_scored_scene_hash != repaired_hash:
                        console_logger.info(
                            "Final deterministic repair produced a hard-valid "
                            "scene; rendering and scoring the repaired state"
                        )
                        self.rendering_manager.clear_cache()
                        try:
                            await self._request_critique_impl(update_checkpoint=False)
                        except Exception as exc:
                            # A transport/renderer failure must not re-label the
                            # repaired scene with an earlier hard-fail score.
                            console_logger.warning(
                                "Could not score final repaired scene; preserving "
                                "it as explicitly unscored: %s: %s",
                                type(exc).__name__,
                                exc,
                            )
                            self.previous_scores = None
                            self.final_render_dir = None
                            self.checkpoint_render_dir = None
                            try:
                                self.final_render_dir = (
                                    self.rendering_manager.render_scene(
                                        scene=self.scene,
                                        blender_server=self.blender_server,
                                        rendering_mode=self.agent_type.value,
                                        render_name="final_repaired_unscored",
                                    )
                                )
                            except Exception:
                                console_logger.warning(
                                    "Could not render final repaired scene",
                                    exc_info=True,
                                )
                        final_hard_state = self._evaluate_current_hard_state()
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
            if final_hard_state is not None and not final_hard_state.hard_valid:
                reasons = "; ".join(final_hard_state.hard_reasons)
                if self._allow_degraded_stage_completion:
                    console_logger.warning(
                        "Furniture stage exhausted local repair and stage "
                        "regeneration; preserving the diagnosed candidate so the "
                        "remaining pipeline and verifier can continue: %s",
                        reasons,
                    )
                    self._degraded_stage_reasons = list(
                        final_hard_state.hard_reasons
                    )
                else:
                    console_logger.error(
                        "Furniture stage failed with unresolved deterministic hard "
                        "constraints: %s",
                        reasons,
                    )
                    raise StageValidationError(
                        stage=self.agent_type.value,
                        reasons=final_hard_state.hard_reasons,
                    )

        enforce_sceneexpert_completion = bool(
            self._stage_runtime_budget
            and self.agent_type.is_placement_agent
            and not getattr(self, "_defer_stage_completion_contract", False)
            and not (controller and getattr(controller, "enabled", False))
            and fail_on_hard_constraints
        )
        if enforce_sceneexpert_completion:
            final_hard_state = self._evaluate_current_hard_state()
            if final_hard_state is not None and not final_hard_state.hard_valid:
                reasons = list(final_hard_state.hard_reasons)
                if self._allow_degraded_stage_completion:
                    console_logger.warning(
                        "%s stage exhausted regeneration; preserving an explicitly "
                        "degraded result: %s",
                        self.agent_type.value,
                        "; ".join(reasons),
                    )
                    self._degraded_stage_reasons = reasons
                else:
                    raise StageValidationError(
                        stage=self.agent_type.value,
                        reasons=reasons,
                    )

        if (
            self._stage_runtime_budget
            and self.agent_type.is_placement_agent
            and not getattr(self, "_defer_stage_completion_contract", False)
        ):
            trusted_score_available = (
                bool(
                    controller
                    and getattr(controller, "enabled", False)
                    and getattr(controller, "best_score_source", "") == "vlm_critic"
                )
                or self._last_score_provenance.get("score_source") == "vlm_critic"
            )
            if not trusted_score_available:
                reason = (
                    "visual critic did not produce a trustworthy score after "
                    "bounded compact retries"
                )
                if self._allow_degraded_stage_completion:
                    console_logger.warning(
                        "%s; stage remains explicitly unscored", reason
                    )
                    self._degraded_stage_reasons.append(reason)
                else:
                    raise StageValidationError(
                        stage=self.agent_type.value,
                        reasons=[reason],
                    )
            elif not (controller and getattr(controller, "enabled", False)):
                visual_score = self._normalized_visual_score(self.previous_scores)
                minimum_visual_score = float(
                    self._stage_budget_value("min_visual_score", 0.60) or 0.60
                )
                if (
                    visual_score is not None
                    and visual_score < minimum_visual_score
                ):
                    reason = (
                        "visual critic quality below stage threshold: "
                        f"{visual_score:.3f} < {minimum_visual_score:.3f}"
                    )
                    if self._allow_degraded_stage_completion:
                        console_logger.warning("%s", reason)
                        self._degraded_stage_reasons.append(reason)
                    else:
                        raise StageValidationError(
                            stage=self.agent_type.value,
                            reasons=[reason],
                        )

        # Copy final scores and renders to per-stage directory.
        # Use final_render_dir (tracks actual last render) instead of checkpoint_render_dir
        # (which may be stale when final critique uses update_checkpoint=False).
        render_dir_to_copy = self.final_render_dir or self.checkpoint_render_dir
        if render_dir_to_copy is not None:
            final_scene_dir = self._get_final_scores_directory()
            final_scene_dir.mkdir(parents=True, exist_ok=True)

            # Copy decision scores together with their source-specific payload and
            # provenance.  Consumers should inspect score_provenance.yaml before
            # treating numeric grades as VLM quality scores.
            score_artifacts = (
                "scores.yaml",
                "score_provenance.yaml",
                "vlm_scores.yaml",
                "hard_check_decision_scores.yaml",
                "hard_check_report.yaml",
                "critic_unavailable.yaml",
            )
            copied_score_artifacts: list[str] = []
            for filename in score_artifacts:
                source = render_dir_to_copy / filename
                if not source.exists():
                    continue
                shutil.copy(source, final_scene_dir / filename)
                copied_score_artifacts.append(filename)
            if copied_score_artifacts:
                console_logger.info(
                    "Saved final score artifacts to %s: %s",
                    final_scene_dir,
                    ", ".join(copied_score_artifacts),
                )
            else:
                console_logger.warning(
                    "No score artifacts found at %s, cannot copy",
                    render_dir_to_copy,
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
            f"max_critique_rounds={self._effective_critique_round_limit()} budget has "
            "been reached. Do not call request_critique(), "
            "request_design_change(), or reset_scene_to_checkpoint() again. "
            "Return your final concise workflow summary now. The framework will "
            "run the final critique automatically after the planner exits."
        )

    def _planner_budget_hint_after_critique(self) -> str:
        if self._planner_critique_tool_calls < self._effective_critique_round_limit():
            return ""
        return (
            "\n\n[Planner budget] This is the last allowed planner critique. "
            "If changes are still needed, call request_design_change() once to "
            "address the critique, then return the final summary. Do not call "
            "request_critique() again."
        )

    def _planner_budget_hint_after_design_change(self) -> str:
        if self._planner_design_change_tool_calls < self._effective_critique_round_limit():
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
        if self._planner_budget_exhausted or self._stage_runtime_exhausted:
            return (
                "\n\n[Auto scoring] Stage execution budget is exhausted; "
                "skipped the extra critic call and kept the current candidate "
                "for deterministic validation."
            )
        if not self._auto_score_after_design_attempts_enabled():
            return ""
        if self._effective_critique_round_limit() <= 0:
            return ""
        if self._planner_critique_tool_calls >= self._effective_critique_round_limit():
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
                "Critic scoring failed with " f"{type(exc).__name__}: {exc}."
            )
        self._record_module_timing(
            "planner",
            f"auto_score_after_{attempt_label.replace(' ', '_')}",
            score_start,
        )
        budget_hint = ""
        if self._planner_critique_tool_calls >= self._effective_critique_round_limit():
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
            result += await self._score_design_attempt_if_configured("initial design")
            return self._truncate_planner_tool_output(
                result,
                label="initial design",
                max_chars=self._planner_context_limit("initial_design_max_chars", 5000),
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

            if self._planner_critique_tool_calls >= self._effective_critique_round_limit():
                return self._planner_budget_stop_message("request_critique")

            self._planner_critique_tool_calls += 1
            try:
                result = await self._request_critique_impl()
            except Exception as exc:
                self._critic_failed = True
                console_logger.exception("Planner-requested critic scoring failed")
                return self._stop_planner_after_failure(
                    "Critic scoring failed with " f"{type(exc).__name__}: {exc}."
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
                >= self._effective_critique_round_limit()
                and not self._hard_repair_allowance_available()
            ):
                return self._planner_budget_stop_message("request_design_change")

            if (
                counts_as_critique_cycle
                and self._planner_design_change_tool_calls
                >= self._effective_critique_round_limit()
                and not self._hard_repair_allowance_available()
            ):
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
            result += await self._score_design_attempt_if_configured("design change")
            result = self._truncate_planner_tool_output(
                result,
                label="design change",
                max_chars=self._planner_context_limit("design_change_max_chars", 5000),
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
        if self._effective_critique_round_limit() > 0:
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

            # Synthetic hard-check grades are repair priorities, not a quality
            # checkpoint and not a commensurate score delta.
            score_change_msg = ""

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
                safety_msg = "\n\n**Hard Check:** " + self._repair_hint_from_hard_state(
                    hard_state
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

            self.final_render_dir = checkpoint_render_dir or images_dir
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

        if self._critic_fast_path_enabled("fresh_session_per_evaluation", True):
            clear_session = getattr(self.critic_session, "clear_session", None)
            if callable(clear_session):
                await clear_session()

        prompt_enum = self._get_critique_prompt_enum()
        extra_kwargs = self._get_extra_critique_kwargs()

        critique_instruction = self.prompt_registry.get_prompt(
            prompt_enum=prompt_enum,
            physics_context=physics_context,
            placement_style=self.placement_style,
            **extra_kwargs,
        )
        if hard_state is not None and hard_state.soft_reasons:
            critique_instruction += (
                "\n\nDeterministic soft diagnostics for this same candidate "
                "(verify them against the renders and exact scene state; do not "
                "treat them as automatic hard failures):\n- "
                + "\n- ".join(hard_state.soft_reasons[:8])
            )
        if self.agent_type == AgentType.FURNITURE:
            critique_instruction += (
                "\n\nOrientation evidence contract: asset preprocessing "
                "canonicalizes the visual front to local +Y. Therefore an "
                "object's world-facing direction is its local +Y transformed by "
                "the reported yaw. Evaluate sofa/bed/chair facing in this same "
                "critic pass; do not claim a separate facing-tool result."
            )
        feedback_contract = self._sceneexpert_critic_feedback_contract()
        if feedback_contract:
            critique_instruction += "\n\n" + feedback_contract
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
                ModelSettings(tool_choice="observe_scene", parallel_tool_calls=False)
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
        score_instructions = self.critic.instructions
        if self._stage_runtime_budget and isinstance(score_instructions, str):
            score_instructions = direct_critic_scoring_instructions(
                score_instructions
            )
        critic_score = self.critic.clone(
            tools=[],
            instructions=score_instructions,
            model_settings=base_settings.resolve(
                ModelSettings(tool_choice="none", parallel_tool_calls=False)
            ),
        )

        direct_multimodal = self._critic_fast_path_enabled(
            "direct_multimodal_evaluation", True
        )
        direct_image_parts: list[dict[str, str]] = []
        direct_observation_note = ""
        if direct_multimodal:
            console_logger.info(
                "[CRITIC harness] Step 1: direct framework render (no tool-call LLM)"
            )
            (
                _,
                direct_image_parts,
                direct_observation_note,
            ) = self._collect_direct_critic_observation(render_profile)
            result_observe = None
            self._record_llm_call_debug(
                agent_role="critic",
                event="observe_scene_direct",
                prompt=critique_instruction,
                output=(
                    f"framework render supplied {len(direct_image_parts)} image(s)"
                ),
            )
        else:
            # Compatibility path for providers that require tool-output images in
            # session history. Qwen/vLLM uses the direct path above.
            console_logger.info("[CRITIC harness] Step 1: observe_scene")
            observe_start = time.time()
            with self.rendering_manager.use_render_profile(render_profile):
                result_observe = await self._run_agent_with_stage_sla(
                    starting_agent=critic_observe,
                    input=critique_instruction,
                    role="critic",
                    event="observe_scene",
                    session=self.critic_session,
                    configured_max_turns=self.cfg.agents.critic_agent.max_turns,
                    run_config=run_config,
                )
            self._record_module_timing("critic", "observe_scene", observe_start)
            if result_observe is not None:
                log_agent_usage(result=result_observe, agent_name="CRITIC (observe)")
            self._record_llm_call_debug(
                agent_role="critic",
                event="observe_scene",
                prompt=critique_instruction,
                output=(result_observe.final_output or "") if result_observe else "",
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
        elif not direct_multimodal:
            console_logger.info("[CRITIC harness] Step 2: get_current_scene_state")
            scene_state_start = time.time()
            result_scene = await self._run_agent_with_stage_sla(
                starting_agent=critic_scene_state,
                input="Now retrieve exact object data with get_current_scene_state.",
                role="critic",
                event="get_current_scene_state",
                session=self.critic_session,
                configured_max_turns=self.cfg.agents.critic_agent.max_turns,
                run_config=run_config,
            )
            self._record_module_timing(
                "critic", "get_current_scene_state", scene_state_start
            )
            if result_scene is not None:
                log_agent_usage(result=result_scene, agent_name="CRITIC (scene_state)")
            self._record_llm_call_debug(
                agent_role="critic",
                event="get_current_scene_state",
                prompt="Now retrieve exact object data with get_current_scene_state.",
                output=(result_scene.final_output or "") if result_scene else "",
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
        score_input: Any = score_prompt
        score_session: Session | None = self.critic_session
        if direct_multimodal:
            direct_text = critique_instruction + "\n\n" + score_prompt
            if direct_observation_note:
                direct_text += "\n\nRender result: " + direct_observation_note
            score_input = [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": direct_text},
                        *direct_image_parts,
                    ],
                }
            ]
            # A fresh one-shot multimodal request avoids carrying old images and
            # tool traces through SQLite session history.
            score_session = None
        # Visual scoring is a required decision boundary for quality
        # regeneration, deterministic fallback selection, and positive memory.
        # Start its sole transaction deadline only after deterministic render and
        # exact scene-state evidence have been collected.
        self._begin_critic_evaluation()
        try:
            configured_max_attempts = int(
                _cfg_get(
                    self._critic_fast_path_cfg(),
                    "direct_multimodal_max_attempts",
                    2,
                )
                or 2
            )
            if self._stage_runtime_budget:
                configured_max_attempts = int(
                    self._stage_budget_value(
                        "critic_max_attempts",
                        configured_max_attempts,
                    )
                    or configured_max_attempts
                )
            max_attempts = (
                max(1, configured_max_attempts) if direct_multimodal else 1
            )
            if direct_multimodal and not direct_image_parts:
                console_logger.warning(
                    "Direct visual critic render produced no images; recording the "
                    "candidate as unscored instead of issuing a text-only score"
                )
                max_attempts = 0
            configured_attempt_timeout = float(
                _cfg_get(
                    self._critic_fast_path_cfg(),
                    "direct_multimodal_attempt_timeout_seconds",
                    120.0,
                )
                or 120.0
            )
            attempt_timeout = float(configured_attempt_timeout)
            result = None
            for attempt in range(1, max_attempts + 1):
                result = await self._run_agent_with_stage_sla(
                    starting_agent=critic_score,
                    input=score_input,
                    role="critic",
                    event=f"score_scene_attempt_{attempt}",
                    session=score_session,
                    configured_max_turns=self.cfg.agents.critic_agent.max_turns,
                    run_config=run_config,
                    call_timeout_seconds=(
                        self._critic_score_call_timeout(attempt_timeout)
                        if direct_multimodal
                        else None
                    ),
                )
                if result is not None:
                    break
                if attempt < max_attempts:
                    console_logger.warning(
                        "Compact visual critic attempt %d/%d did not complete; "
                        "retrying the same rendered candidate with fresh context",
                        attempt,
                        max_attempts,
                    )
            self._record_module_timing("critic", "score_scene", score_start)
            if result is None:
                response = self._make_transient_critic_fallback_scores(
                    error=TimeoutError("critic stage execution budget exhausted")
                )
            else:
                log_agent_usage(result=result, agent_name="CRITIC (score)")
                self._record_llm_call_debug(
                    agent_role="critic",
                    event="score_scene",
                    prompt=score_prompt,
                    output=result.final_output or "",
                    result=result,
                )
                response = result.final_output
        except Exception as exc:
            if not self._is_transient_model_error(exc):
                raise
            self._record_module_timing(
                "critic",
                "score_scene",
                score_start,
                extra={"fallback": "transient_model_error", "error": str(exc)},
            )
            console_logger.warning(
                "Local VLM critic scoring failed transiently; recording an "
                "unscored retryable diagnostic: %s: %s",
                type(exc).__name__,
                exc,
            )
            response = self._make_transient_critic_fallback_scores(error=exc)
            self._record_llm_call_debug(
                agent_role="critic",
                event="score_scene_transient_fallback",
                prompt=score_prompt,
                output=response.critique,
                error=f"{type(exc).__name__}: {exc}",
            )

        # Parse structured output or the unscored transport diagnostic.
        if not isinstance(response, CritiqueWithScores):
            raise TypeError(
                "Critic returned an unexpected final output type: "
                f"{type(response).__name__}"
            )

        self._remember_critic_feedback(response.critique)

        # Log critique text and scores to console.
        log_agent_response(response=response.critique, agent_name="CRITIC")
        response_provenance = self._score_provenance_for_response(
            response=response,
            physics_context=physics_context,
        )
        if response_provenance.get("score_source") == "critic_fallback":
            console_logger.warning(
                "Critic response is unscored; suppressing legacy placeholder grades"
            )
        else:
            log_critique_scores(response, title="CRITIQUE SCORES")

        # Save scores to YAML next to scene renders (from observe_scene call).
        images_dir = self.rendering_manager.last_render_dir
        self._write_scores_and_memory(
            response=response,
            images_dir=images_dir,
            physics_context=physics_context,
        )

        # Compute score deltas and format for planner if we have previous scores.
        trusted_visual_score = (
            response_provenance.get("score_source") == "vlm_critic"
        )
        score_change_msg = ""
        if trusted_visual_score and self.previous_scores is not None:
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
        checkpoint_accepted = bool(checkpoint_accepted and trusted_visual_score)

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

        # Only comparable VLM quality scores participate in deltas and
        # checkpoint selection. Transport placeholders and deterministic repair
        # grades remain available through their separate provenance artifacts.
        if trusted_visual_score:
            self.previous_scores = response

        # Always track the final render directory (separate from checkpoint logic).
        # This is needed because final critique uses update_checkpoint=False, but we
        # still need to know the actual last render dir for copying to final output.
        self.final_render_dir = checkpoint_render_dir or images_dir
        if trusted_visual_score:
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
        planner_feedback = self._critic_feedback_for_planner(response.critique)
        return planner_feedback + score_change_msg + safety_msg

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
        critic_feedback = self._critic_feedback_for_designer()
        if critic_feedback:
            full_instruction += (
                "\n\nThe planner selected this repair turn. Use the authoritative "
                "critic evidence below without dropping object IDs, required "
                "changes, preserve constraints, or acceptance checks.\n"
                + critic_feedback
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
            result = await self._run_agent_with_stage_sla(
                starting_agent=self.designer,
                input=full_instruction,
                role="designer",
                event="request_design_change",
                session=self.designer_session,
                configured_max_turns=self.cfg.agents.designer_agent.max_turns,
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
        if result is None:
            safety_msg = self._end_furniture_design_transaction(transaction)
            return (
                "Designer execution budget was exhausted. Preserve the current "
                "candidate and continue to deterministic validation."
                + safety_msg
            )
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
            result = await self._run_agent_with_stage_sla(
                starting_agent=self.designer,
                input=input_message,
                role="designer",
                event="request_initial_design",
                session=self.designer_session,
                configured_max_turns=self.cfg.agents.designer_agent.max_turns,
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
        if result is None:
            safety_msg = self._end_furniture_design_transaction(transaction)
            return (
                "Designer execution budget was exhausted. Preserve all objects "
                "already created and continue to deterministic validation."
                + safety_msg
            )
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
