"""Base class for stateful agents using planner/designer/critic workflow.

This module provides the shared framework for all design agents (floor plan,
furniture, wall, manipuland), extracting the common multi-agent architecture
while allowing domain-specific customization through abstract methods and
subclass-defined tools.
"""

import copy
import logging
import shutil

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
    CritiqueWithScores,
    compute_total_score,
    format_score_deltas_for_planner,
    log_agent_response,
    log_critique_scores,
    scores_to_dict,
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
        self._planner_critique_tool_calls = 0
        self._planner_design_change_tool_calls = 0
        self._planner_budget_exhausted = False

    def _configure_furniture_safety_for_scene(self, scene_description: str) -> None:
        """Reset furniture safety counters and required-object inference."""
        controller = getattr(self, "furniture_safety_controller", None)
        if controller and controller.enabled:
            controller.reset_for_scene(scene_description=scene_description)

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
    ) -> tuple[str, CritiqueWithScores, Path | None]:
        """Evaluate a critiqued scene and rollback to best if needed."""
        controller = getattr(self, "furniture_safety_controller", None)
        if not controller or not controller.enabled:
            return "", scores, images_dir

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

        checkpoint_scores = scores
        checkpoint_render_dir = images_dir
        if decision.rollback_to_best and controller.best_scene_state is not None:
            self._restore_furniture_scene_state(controller.best_scene_state)
            if controller.best_scores is not None:
                checkpoint_scores = controller.best_scores
            checkpoint_render_dir = controller.best_render_dir
            console_logger.info(
                "Safety controller restored best hard-valid checkpoint "
                f"(weighted_score={controller.best_weighted_score:.3f})."
            )

        return (
            f"\n\n**Safety Controller:** {decision.message}",
            checkpoint_scores,
            checkpoint_render_dir,
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
            instructions=self.prompt_registry.get_prompt(
                prompt_enum=prompt_enum,
                **prompt_kwargs,
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
        return Agent(
            name=critic_config.name,
            model=self.cfg.openai.model,
            tools=tools,
            instructions=self.prompt_registry.get_prompt(
                prompt_enum=prompt_enum,
                **prompt_kwargs,
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
            instructions=self.prompt_registry.get_prompt(
                prompt_enum=prompt_enum,
                **prompt_kwargs,
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
        self._planner_critique_tool_calls = 0
        self._planner_design_change_tool_calls = 0
        self._planner_budget_exhausted = False

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
            result = await self._request_initial_design_impl()
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
            if self._planner_critique_tool_calls >= int(self.cfg.max_critique_rounds):
                return self._planner_budget_stop_message("request_critique")

            self._planner_critique_tool_calls += 1
            result = await self._request_critique_impl()
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
            counts_as_critique_cycle = (
                self._planner_critique_tool_calls
                > self._planner_design_change_tool_calls
            )
            if counts_as_critique_cycle and self._planner_design_change_tool_calls >= int(
                self.cfg.max_critique_rounds
            ):
                return self._planner_budget_stop_message("request_design_change")

            safety_block = self._record_furniture_design_change_budget()
            if safety_block:
                return safety_block

            result = await self._request_design_change_impl(instruction)
            result = self._truncate_planner_tool_output(
                result,
                label="design change",
                max_chars=self._planner_context_limit(
                    "design_change_max_chars", 5000
                ),
            )
            if counts_as_critique_cycle:
                self._planner_design_change_tool_calls += 1
                result += self._planner_budget_hint_after_design_change()
            return result

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

        # Get current furniture ID for manipuland agents.
        current_furniture_id = getattr(self, "current_furniture_id", None)

        # Get physics violations using the same logic as the check_physics tool.
        # This ensures the critic sees exactly the same information as the designer.
        physics_context = check_physics_violations(
            scene=self.scene,
            cfg=self.cfg,
            current_furniture_id=current_furniture_id,
            agent_type=self.agent_type,
        )

        prompt_enum = self._get_critique_prompt_enum()
        extra_kwargs = self._get_extra_critique_kwargs()

        critique_instruction = self.prompt_registry.get_prompt(
            prompt_enum=prompt_enum,
            physics_context=physics_context,
            placement_style=self.placement_style,
            **extra_kwargs,
        )
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
        # Step 3 uses the original critic (with output_type and no forced tool).
        critic_score = self.critic.clone(
            model_settings=base_settings.resolve(ModelSettings(tool_choice=None))
        )

        # All three steps share self.critic_session so history accumulates
        # naturally; inputs are strings (lists are illegal when a session is
        # used without a custom session_input_callback).

        # Step 1: force observe_scene; stop_on_first_tool returns immediately.
        console_logger.info("[CRITIC harness] Step 1: observe_scene")
        result_observe = await Runner.run(
            starting_agent=critic_observe,
            input=critique_instruction,
            session=self.critic_session,
            run_config=run_config,
        )
        log_agent_usage(result=result_observe, agent_name="CRITIC (observe)")

        # Step 2: force get_current_scene_state; session carries Step 1 history.
        console_logger.info("[CRITIC harness] Step 2: get_current_scene_state")
        result_scene = await Runner.run(
            starting_agent=critic_scene_state,
            input="Now retrieve exact object data with get_current_scene_state.",
            session=self.critic_session,
            run_config=run_config,
        )
        log_agent_usage(result=result_scene, agent_name="CRITIC (scene_state)")

        # Step 3: free evaluation with structured output. The critic now has
        # the observation images and the scene-state JSON in its session
        # history and can run STEPS 3-6 of the YAML workflow.
        console_logger.info("[CRITIC harness] Step 3: evaluate and score")
        result = await Runner.run(
            starting_agent=critic_score,
            input=(
                "Steps 1 and 2 of the MANDATORY EVALUATION WORKFLOW are "
                "complete (scene observed, object data retrieved). Now perform "
                "STEPS 3-6 (physics review, placement evaluation, "
                "lighting/coverage analysis, synthesis) and return your final "
                "critique with scores."
            ),
            session=self.critic_session,
            max_turns=self.cfg.agents.critic_agent.max_turns,
            run_config=run_config,
        )
        log_agent_usage(result=result, agent_name="CRITIC (score)")

        # Parse structured output.
        response = result.final_output_as(CritiqueWithScores)

        # Log critique text and scores to console.
        log_agent_response(response=response.critique, agent_name="CRITIC")
        log_critique_scores(response, title="CRITIQUE SCORES")

        # Save scores to YAML next to scene renders (from observe_scene call).
        images_dir = self.rendering_manager.last_render_dir
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
        else:
            console_logger.error(
                "No render directory available - scores not saved to file"
            )

        # Compute score deltas and format for planner if we have previous scores.
        score_change_msg = ""
        if self.previous_scores is not None:
            score_change_msg = format_score_deltas_for_planner(
                current_scores=response,
                previous_scores=self.previous_scores,
                format_style="detailed",
            )

        safety_msg, checkpoint_scores, checkpoint_render_dir = (
            self._apply_furniture_safety_after_critique(
                scores=response,
                images_dir=images_dir,
                physics_context=physics_context,
            )
        )

        # Shift checkpoints only during iteration critiques, not final critique.
        # This preserves N-1 checkpoint for reset check in _finalize_scene_and_scores.
        if update_checkpoint:
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

        # Always update previous_scores for delta formatting in planner.
        self.previous_scores = response

        # Always track the final render directory (separate from checkpoint logic).
        # This is needed because final critique uses update_checkpoint=False, but we
        # still need to know the actual last render dir for copying to final output.
        self.final_render_dir = checkpoint_render_dir or images_dir

        # Return natural language critique with score deltas for planner.
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

        # Designer run with critique-based instruction.
        try:
            result = await Runner.run(
                starting_agent=self.designer,
                input=full_instruction,
                session=self.designer_session,
                max_turns=self.cfg.agents.designer_agent.max_turns,
                run_config=self._create_run_config(),
            )
        except Exception:
            self._end_furniture_design_transaction(transaction)
            raise
        log_agent_usage(result=result, agent_name="DESIGNER (CHANGE)")

        if result.final_output:
            log_agent_response(
                response=result.final_output, agent_name="DESIGNER (CHANGE)"
            )

        safety_msg = self._end_furniture_design_transaction(transaction)
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

        # Build input (may include context image if enabled).
        input_message = self._build_initial_design_input(instruction)

        # Designer runs with initial design instruction.
        try:
            result = await Runner.run(
                starting_agent=self.designer,
                input=input_message,
                session=self.designer_session,
                max_turns=self.cfg.agents.designer_agent.max_turns,
                run_config=self._create_run_config(),
            )
        except Exception:
            self._end_furniture_design_transaction(transaction)
            raise
        log_agent_usage(result=result, agent_name="DESIGNER (INITIAL)")

        if result.final_output:
            log_agent_response(
                response=result.final_output, agent_name="DESIGNER (INITIAL)"
            )

        safety_msg = self._end_furniture_design_transaction(transaction)
        return (result.final_output or "") + safety_msg
