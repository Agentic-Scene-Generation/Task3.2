"""Unit tests for finalization checkpoint reset logic."""

import asyncio
import shutil
import tempfile
import unittest

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from scenesmith.agent_utils.base_stateful_agent import BaseStatefulAgent
from scenesmith.agent_utils.furniture_safety import HardStateEvaluation
from scenesmith.agent_utils.placement_noise import PlacementNoiseMode
from scenesmith.agent_utils.room import AgentType
from scenesmith.agent_utils.scoring import CategoryScore, FurnitureCritiqueWithScores


def _make_scores(
    realism: int = 7,
    functionality: int = 7,
    layout: int = 7,
    layout_plausibility: int = 7,
    holistic: int = 7,
    prompt: int = 7,
    reachability: int = 7,
) -> FurnitureCritiqueWithScores:
    """Create FurnitureCritiqueWithScores with specified grades."""
    return FurnitureCritiqueWithScores(
        critique="Test critique",
        realism=CategoryScore(name="Realism", grade=realism, comment="test"),
        functionality=CategoryScore(
            name="Functionality", grade=functionality, comment="test"
        ),
        layout=CategoryScore(name="Layout", grade=layout, comment="test"),
        layout_plausibility=CategoryScore(
            name="Layout Plausibility", grade=layout_plausibility, comment="test"
        ),
        holistic_completeness=CategoryScore(
            name="Holistic Completeness", grade=holistic, comment="test"
        ),
        prompt_following=CategoryScore(
            name="Prompt Following", grade=prompt, comment="test"
        ),
        reachability=CategoryScore(
            name="Reachability", grade=reachability, comment="test"
        ),
    )


class TestFinalizeSceneReset(unittest.TestCase):
    """Test that finalization resets to N-1 checkpoint when scores degrade."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())

        # Create mock render directories with scores.yaml files.
        self.n2_render_dir = self.temp_dir / "renders_003"
        self.n1_render_dir = self.temp_dir / "renders_008"
        self.final_render_dir = self.temp_dir / "renders_009"

        for render_dir in [
            self.n2_render_dir,
            self.n1_render_dir,
            self.final_render_dir,
        ]:
            render_dir.mkdir(parents=True)
            (render_dir / "scores.yaml").write_text("test: scores")
            (render_dir / "view_0.png").write_text("test image")

        # Create final scores directory.
        self.final_scores_dir = self.temp_dir / "scene_states" / "furniture"
        self.final_scores_dir.mkdir(parents=True)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def _create_testable_agent(
        self,
        mock_scene: MagicMock,
        mock_rendering_manager: MagicMock,
        agent_type: AgentType = AgentType.FURNITURE,
    ):
        """Create a concrete testable subclass of BaseStatefulPlacementAgent."""
        final_scores_dir = self.final_scores_dir

        mock_cfg = MagicMock()
        mock_cfg.reset_single_category_threshold = 3  # Trigger reset on 3+ point drop.
        mock_cfg.reset_total_sum_threshold = 6
        mock_cfg.max_critique_rounds = 1
        mock_cfg.planner_context_limits = {}

        # Set up action_log_path as a real path for the action logger decorator.
        mock_scene.action_log_path = self.temp_dir / "action_log.json"

        class TestableAgent(BaseStatefulAgent):
            def __init__(self):
                # Skip parent __init__ to avoid complex setup.
                self.scene = mock_scene
                self.rendering_manager = mock_rendering_manager
                self.cfg = mock_cfg

            @property
            def agent_type(self) -> AgentType:
                return agent_type

            def _get_final_scores_directory(self) -> Path:
                return final_scores_dir

            def _get_critique_prompt_enum(self) -> Any:
                return None

            def _get_design_change_prompt_enum(self) -> Any:
                return None

            def _get_initial_design_prompt_enum(self) -> Any:
                return None

            def _get_initial_design_prompt_kwargs(self) -> dict:
                return {}

            def _set_placement_noise_profile(self, mode: PlacementNoiseMode) -> None:
                pass

        return TestableAgent()

    def test_planner_tools_include_finish_stage(self):
        """Planner should have a legal explicit stage-completion tool."""
        mock_scene = MagicMock()
        mock_rendering_manager = MagicMock()
        agent = self._create_testable_agent(mock_scene, mock_rendering_manager)

        tools = agent._create_planner_tools()
        tool_names = {getattr(tool, "name", "") for tool in tools}

        self.assertIn("finish_stage", tool_names)

    def test_initial_designer_timeout_stops_planner_and_records_root_failure(self):
        mock_scene = MagicMock()
        mock_rendering_manager = MagicMock()
        agent = self._create_testable_agent(mock_scene, mock_rendering_manager)
        agent.stage_working_memory = MagicMock()
        agent._planner_orchestration_calls = 0
        agent._request_initial_design_impl = AsyncMock(
            side_effect=TimeoutError("designer deadline exceeded")
        )

        tools = {tool.name: tool for tool in agent._create_planner_tools()}
        result = asyncio.run(
            tools["request_initial_design"].on_invoke_tool(MagicMock(), "{}")
        )

        self.assertIn("STOP: Initial designer failed with TimeoutError", result)
        self.assertTrue(agent._planner_budget_exhausted)
        self.assertEqual(
            agent._planner_terminal_failure,
            {
                "operation": "request_initial_design",
                "child_agent": "designer",
                "error_type": "TimeoutError",
                "error": "designer deadline exceeded",
                "recovered": False,
            },
        )
        statuses = [
            call.kwargs["status"]
            for call in agent.stage_working_memory.record_planner_orchestration.mock_calls
        ]
        self.assertEqual(statuses, ["started", "failed"])

        repeated = asyncio.run(
            tools["request_initial_design"].on_invoke_tool(MagicMock(), "{}")
        )
        self.assertIn("already been marked complete or failed", repeated)
        agent._request_initial_design_impl.assert_awaited_once_with()

    def test_terminal_delegation_failure_records_deterministic_recovery(self):
        mock_scene = MagicMock()
        mock_rendering_manager = MagicMock()
        agent = self._create_testable_agent(mock_scene, mock_rendering_manager)
        agent.stage_working_memory = MagicMock()
        agent._planner_terminal_failure = {
            "operation": "request_critique",
            "child_agent": "critic",
            "error_type": "TimeoutError",
            "error": "critic deadline exceeded",
            "recovered": False,
        }

        agent._mark_planner_terminal_failure_recovered()

        self.assertTrue(agent._planner_terminal_failure["recovered"])
        event = agent.stage_working_memory.record_planner_orchestration.call_args.kwargs
        self.assertEqual(event["status"], "recovered")
        self.assertEqual(event["operation"], "request_critique")
        self.assertEqual(event["child_agent"], "critic")

    def test_design_change_timeout_stops_planner_and_preserves_instruction(self):
        mock_scene = MagicMock()
        mock_rendering_manager = MagicMock()
        agent = self._create_testable_agent(mock_scene, mock_rendering_manager)
        agent.stage_working_memory = MagicMock()
        agent._planner_orchestration_calls = 0
        agent._request_design_change_impl = AsyncMock(
            side_effect=TimeoutError("designer change deadline exceeded")
        )

        tools = {tool.name: tool for tool in agent._create_planner_tools()}
        result = asyncio.run(
            tools["request_design_change"].on_invoke_tool(
                MagicMock(), '{"instruction": "move the sofa away from the door"}'
            )
        )

        self.assertIn("STOP: Designer change failed with TimeoutError", result)
        self.assertTrue(agent._planner_budget_exhausted)
        self.assertEqual(
            agent._planner_terminal_failure["operation"], "request_design_change"
        )
        dispatch = agent.stage_working_memory.record_planner_orchestration.mock_calls[
            0
        ].kwargs
        self.assertEqual(
            dispatch["detail"],
            {"instruction": "move the sofa away from the door"},
        )

    def test_finalize_resets_to_n1_not_n2_when_scores_degrade(self):
        """When final scores are worse than N-1, reset should use N-1 state (not N-2).

        This tests for the bug where:
        - Comparison correctly uses checkpoint_scores (N-1)
        - But reset incorrectly uses previous_scene_checkpoint (N-2)

        The fix should make reset use scene_checkpoint (N-1) to match comparison.
        """
        mock_scene = MagicMock()
        mock_rendering_manager = MagicMock()

        agent = self._create_testable_agent(mock_scene, mock_rendering_manager)
        agent.final_scores_dir = self.final_scores_dir

        # Set up checkpoint state simulating the bug scenario:
        # N-2 (previous_scene_checkpoint): Realism=3 (old bad state)
        # N-1 (scene_checkpoint): Realism=6 (good checkpoint we want)
        # N (previous_scores/final): Realism=3 (degraded final)

        # N-2 state (what the bug incorrectly resets to).
        agent.previous_scene_checkpoint = {"state": "N-2", "objects": {"old": "state"}}
        agent.previous_checkpoint_scores = _make_scores(realism=3)
        agent.previous_checkpoint_render_dir = self.n2_render_dir

        # N-1 state (what we SHOULD reset to).
        agent.scene_checkpoint = {"state": "N-1", "objects": {"good": "state"}}
        agent.checkpoint_scores = _make_scores(realism=6)  # Good scores.
        agent.checkpoint_render_dir = self.n1_render_dir

        # Final scores (N) - degraded compared to N-1.
        agent.previous_scores = _make_scores(realism=3)  # 3 point drop triggers reset.
        agent.final_render_dir = self.final_render_dir

        # Run finalization.
        asyncio.run(agent._finalize_scene_and_scores())

        # ASSERTION: The scene should be restored to N-1 state, not N-2.
        # The bug causes restore_from_state_dict to be called with N-2.
        mock_scene.restore_from_state_dict.assert_called_once()
        call_args = mock_scene.restore_from_state_dict.call_args[0][0]

        # This assertion will FAIL with the current buggy code.
        # Current code passes previous_scene_checkpoint (N-2).
        # Fixed code should pass scene_checkpoint (N-1).
        self.assertEqual(
            call_args["state"],
            "N-1",
            f"Expected reset to N-1 state but got {call_args.get('state', 'unknown')}. "
            "The finalization reset is using the wrong checkpoint (N-2 instead of N-1).",
        )

        # Also verify the render dir is set to N-1.
        # Current buggy code sets it to previous_checkpoint_render_dir (N-2).
        self.assertEqual(
            agent.final_render_dir,
            self.n1_render_dir,
            f"Expected final_render_dir to be N-1 ({self.n1_render_dir}) "
            f"but got {agent.final_render_dir}",
        )

    def test_finalize_no_reset_when_scores_improve(self):
        """When final scores are better than N-1, no reset should occur."""
        mock_scene = MagicMock()
        mock_rendering_manager = MagicMock()

        agent = self._create_testable_agent(mock_scene, mock_rendering_manager)

        # Set up state where final scores are BETTER than checkpoint.
        agent.previous_scene_checkpoint = {"state": "N-2"}
        agent.previous_checkpoint_scores = _make_scores(realism=5)
        agent.previous_checkpoint_render_dir = self.n2_render_dir

        agent.scene_checkpoint = {"state": "N-1"}
        agent.checkpoint_scores = _make_scores(realism=6)  # N-1 scores.
        agent.checkpoint_render_dir = self.n1_render_dir

        # Final scores improved from N-1.
        agent.previous_scores = _make_scores(realism=8)  # Better than N-1.
        agent.final_render_dir = self.final_render_dir

        asyncio.run(agent._finalize_scene_and_scores())

        # No reset should occur.
        mock_scene.restore_from_state_dict.assert_not_called()

        # Final render dir should remain as the final iteration's render.
        self.assertEqual(agent.final_render_dir, self.final_render_dir)

    def test_finalize_resets_on_total_sum_drop(self):
        """Reset should trigger when total sum drops by threshold.

        Tests the alternative reset path (total sum) vs single category.
        Same bug applies: should reset to N-1, not N-2.
        """
        mock_scene = MagicMock()
        mock_rendering_manager = MagicMock()

        agent = self._create_testable_agent(mock_scene, mock_rendering_manager)

        # N-2 state.
        agent.previous_scene_checkpoint = {"state": "N-2"}
        agent.previous_checkpoint_scores = _make_scores()
        agent.previous_checkpoint_render_dir = self.n2_render_dir

        # N-1 state with good scores (all 7s = 49 total for 7 categories).
        agent.scene_checkpoint = {"state": "N-1"}
        agent.checkpoint_scores = _make_scores()  # All 7s.
        agent.checkpoint_render_dir = self.n1_render_dir

        # Final scores: each category drops by 2 (total drop = 12 > threshold 6).
        # No single category drops by 3, so only total sum triggers reset.
        agent.previous_scores = _make_scores(
            realism=5, functionality=5, layout=5, holistic=5, prompt=5, reachability=5
        )
        agent.final_render_dir = self.final_render_dir

        asyncio.run(agent._finalize_scene_and_scores())

        # Reset should occur.
        mock_scene.restore_from_state_dict.assert_called_once()
        call_args = mock_scene.restore_from_state_dict.call_args[0][0]

        # Should reset to N-1 (same bug as single category test).
        self.assertEqual(
            call_args["state"],
            "N-1",
            "Total sum reset should also use N-1 checkpoint, not N-2.",
        )

    def test_finalize_no_reset_when_checkpoint_scores_none(self):
        """No reset should occur when checkpoint_scores is None.

        This is an edge case at the start of iteration (no previous checkpoint).
        """
        mock_scene = MagicMock()
        mock_rendering_manager = MagicMock()

        agent = self._create_testable_agent(mock_scene, mock_rendering_manager)

        # No checkpoint scores (first iteration scenario).
        agent.checkpoint_scores = None
        agent.scene_checkpoint = None
        agent.checkpoint_render_dir = None

        agent.previous_scores = _make_scores(realism=3)  # Low scores.
        agent.final_render_dir = self.final_render_dir

        asyncio.run(agent._finalize_scene_and_scores())

        # No reset should occur since there's nothing to compare against.
        mock_scene.restore_from_state_dict.assert_not_called()

    def test_non_controller_stage_keeps_legacy_non_strict_default(self):
        mock_scene = MagicMock()
        mock_rendering_manager = MagicMock()
        agent = self._create_testable_agent(mock_scene, mock_rendering_manager)
        agent._evaluate_current_hard_state = MagicMock()
        agent.previous_scores = None
        agent.checkpoint_scores = None
        agent.checkpoint_render_dir = None
        agent.final_render_dir = None

        asyncio.run(agent._finalize_scene_and_scores())

        agent._evaluate_current_hard_state.assert_not_called()

    def test_explicit_non_controller_hard_validation_rejects_invalid_scene(self):
        mock_scene = MagicMock()
        mock_rendering_manager = MagicMock()
        agent = self._create_testable_agent(
            mock_scene,
            mock_rendering_manager,
            agent_type=AgentType.MANIPULAND,
        )
        hard_state = HardStateEvaluation(
            hard_valid=False,
            hard_reasons=["physics hard violation: collisions"],
        )
        agent._final_hard_validation_enabled = MagicMock(return_value=True)
        agent._evaluate_current_hard_state = MagicMock(return_value=hard_state)
        agent._try_deterministic_repair_for_hard_state = MagicMock(
            return_value=(hard_state, None, [])
        )
        agent.previous_scores = None
        agent.checkpoint_scores = None
        agent.checkpoint_render_dir = None
        agent.final_render_dir = None

        with self.assertRaisesRegex(
            RuntimeError,
            "Manipuland stage failed with unresolved hard constraints",
        ):
            asyncio.run(agent._finalize_scene_and_scores())

        mock_scene.restore_from_state_dict.assert_not_called()

    def test_finalize_keeps_hard_valid_scene_over_unscored_snapshot(self):
        mock_scene = MagicMock()
        mock_rendering_manager = MagicMock()
        agent = self._create_testable_agent(mock_scene, mock_rendering_manager)
        agent.cfg.get.side_effect = lambda key, default=None: (
            False if key == "fail_stage_on_unresolved_hard_constraints" else default
        )
        agent.furniture_safety_controller = SimpleNamespace(
            enabled=True,
            best_scene_state={"state": "bed-only-baseline"},
            best_scores=None,
            best_render_dir=None,
            best_weighted_score=0.0,
        )
        agent._evaluate_current_hard_state = MagicMock(
            return_value=HardStateEvaluation(hard_valid=True)
        )
        agent.previous_scores = None
        agent.checkpoint_scores = None
        agent.checkpoint_render_dir = None
        agent.final_render_dir = None

        asyncio.run(agent._finalize_scene_and_scores())

        mock_scene.restore_from_state_dict.assert_not_called()
        agent._evaluate_current_hard_state.assert_called_once_with()

    def test_finalize_uses_unscored_snapshot_to_rescue_hard_invalid_scene(self):
        mock_scene = MagicMock()
        mock_rendering_manager = MagicMock()
        agent = self._create_testable_agent(mock_scene, mock_rendering_manager)
        agent.cfg.get.side_effect = lambda key, default=None: (
            False if key == "fail_stage_on_unresolved_hard_constraints" else default
        )
        agent.furniture_safety_controller = SimpleNamespace(
            enabled=True,
            best_scene_state={"state": "hard-valid-baseline"},
            best_scores=None,
            best_render_dir=None,
            best_weighted_score=0.0,
        )
        agent._evaluate_current_hard_state = MagicMock(
            return_value=HardStateEvaluation(
                hard_valid=False,
                hard_reasons=["wardrobe collision"],
            )
        )
        agent.previous_scores = None
        agent.checkpoint_scores = None
        agent.checkpoint_render_dir = None
        agent.final_render_dir = None

        asyncio.run(agent._finalize_scene_and_scores())

        mock_scene.restore_from_state_dict.assert_called_once_with(
            {"state": "hard-valid-baseline"}
        )
        mock_rendering_manager.clear_cache.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
