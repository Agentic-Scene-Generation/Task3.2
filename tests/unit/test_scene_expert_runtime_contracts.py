import asyncio
import json
import time
import unittest

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from scenesmith.scene_expert.critic_feedback import (
    critic_feedback_contract,
    feedback_repair_text,
    parse_critic_feedback,
)
from scenesmith.scene_expert.harness import Harness
from scenesmith.scene_expert.runtime_state import (
    ScenePausedError,
    is_scene_paused_error,
    mark_retryable_pause_resolved,
    persist_retryable_pause,
)

try:
    from scenesmith.agent_utils.base_stateful_agent import BaseStatefulAgent
except ModuleNotFoundError as exc:
    BaseStatefulAgent = None
    _BASE_AGENT_IMPORT_ERROR = exc
else:
    _BASE_AGENT_IMPORT_ERROR = None


class SceneExpertRuntimeContractTest(unittest.TestCase):
    def test_compact_critic_feedback_preserves_action_and_acceptance(self) -> None:
        feedback = parse_critic_feedback(
            """
STATUS: REPAIR_REQUIRED
SUMMARY: The wardrobe blocks the bedroom window.
FINDING 1
SEVERITY: BLOCKING
CATEGORY: window_clearance
OBJECTS: wardrobe_0, window_1
OBSERVATION: wardrobe_0 overlaps the visible window opening.
REASON: The window cannot provide light or be accessed.
REQUIRED_CHANGE: Move wardrobe_0 to an uninterrupted wall segment.
PRESERVE: keep bed_0 and nightstand_0 fixed; preserve the door path
ACCEPTANCE_CHECK: the full window opening is visible and unobstructed.
END_FINDING
""".strip()
        )

        self.assertTrue(feedback.structured)
        self.assertEqual("REPAIR_REQUIRED", feedback.status)
        self.assertEqual(1, len(feedback.blocking_findings))
        finding = feedback.findings[0]
        self.assertEqual(["wardrobe_0", "window_1"], finding.object_ids)
        self.assertIn("Verify:", feedback_repair_text(finding))
        designer_text = feedback.to_designer_text()
        self.assertIn("wardrobe_0", designer_text)
        self.assertIn("Preserve:", designer_text)
        self.assertIn("Accept when:", designer_text)

    def test_contract_does_not_limit_blocking_findings(self) -> None:
        contract = critic_feedback_contract()

        self.assertIn("Include EVERY blocking issue", contract)
        self.assertIn("have no count limit", contract)
        self.assertIn("at most three major/refinement", contract)

    def test_legacy_critic_text_remains_available_as_fallback(self) -> None:
        feedback = parse_critic_feedback(
            "The sofa faces the wall. Rotate it toward the conversation area."
        )

        self.assertFalse(feedback.structured)
        self.assertIn("sofa faces the wall", feedback.to_designer_text())

    def test_stage_budget_inherits_exclusive_role_limits(self) -> None:
        cfg = SimpleNamespace(
            stage_budget=SimpleNamespace(
                default=SimpleNamespace(
                    planner_active_max_seconds=120,
                    designer_active_max_seconds=360,
                    critic_active_max_seconds=300,
                    planner_max_output_tokens=768,
                    designer_max_output_tokens=1536,
                    critic_max_output_tokens=1536,
                    critic_max_attempts=1,
                    critic_attempt_timeout_seconds=150,
                ),
                wall_mounted=SimpleNamespace(
                    designer_active_max_seconds=240,
                ),
            )
        )

        budget = Harness(cfg)._get_stage_budget("wall_mounted")

        self.assertEqual(120, budget.planner_active_max_seconds)
        self.assertEqual(240, budget.designer_active_max_seconds)
        self.assertEqual(300, budget.critic_active_max_seconds)
        self.assertEqual(768, budget.planner_max_output_tokens)
        self.assertEqual(1536, budget.critic_max_output_tokens)
        self.assertEqual(1, budget.critic_max_attempts)
        self.assertEqual(150, budget.critic_attempt_timeout_seconds)

    def test_retryable_pause_persists_candidate_without_success_marker(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = persist_retryable_pause(
                scene_root_dir=root,
                stage="furniture",
                reason="visual critic unavailable",
                candidate_state={"objects": ["sofa_0"]},
                candidate_hash="candidate-hash",
                render_dir=root / "renders_001",
                attempt_count=2,
            )

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            candidate_path = Path(manifest["candidate_state_path"])
            error = ScenePausedError(
                "furniture",
                manifest["reason"],
                str(manifest_path),
            )

            self.assertEqual("PAUSED_RETRYABLE", manifest["status"])
            self.assertEqual("retry_critic_only", manifest["resume_action"])
            self.assertTrue(candidate_path.exists())
            self.assertTrue(is_scene_paused_error(error))
            self.assertFalse((root / "_SUCCESS").exists())

            resolved_path = mark_retryable_pause_resolved(root)
            self.assertIsNotNone(resolved_path)
            self.assertFalse(manifest_path.exists())
            resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
            self.assertEqual("RESOLVED", resolved["status"])


class NestedPlannerBudgetTest(unittest.IsolatedAsyncioTestCase):
    @unittest.skipIf(
        BaseStatefulAgent is None,
        f"requires OpenAI Agents SDK: {_BASE_AGENT_IMPORT_ERROR}",
    )
    async def test_planner_active_lease_pauses_for_nested_designer(self) -> None:
        class ConcreteAgent(BaseStatefulAgent):
            @property
            def agent_type(self):
                return SimpleNamespace(value="test", is_placement_agent=False)

            def _get_final_scores_directory(self):
                return Path()

            def _get_critique_prompt_enum(self):
                return None

            def _set_placement_noise_profile(self, mode):
                del mode

            def _get_design_change_prompt_enum(self):
                return None

            def _get_initial_design_prompt_enum(self):
                return None

            def _get_initial_design_prompt_kwargs(self):
                return {}

        agent = object.__new__(ConcreteAgent)
        agent._stage_runtime_budget = {
            "max_wall_clock_seconds": 1.0,
            "planner_active_max_seconds": 0.08,
            "designer_active_max_seconds": 0.20,
            "critic_active_max_seconds": 0.20,
            "critic_reserve_fraction": 0.0,
            "fallback_reserve_fraction": 0.0,
            "finalization_reserve_fraction": 0.0,
        }
        agent._stage_runtime_started_at = time.monotonic()
        agent._critic_evaluation_started_at = None
        agent._stage_runtime_phase = "agent"
        agent._stage_runtime_exhausted = False
        agent._planner_budget_exhausted = False
        agent._stage_role_active_consumed = {}
        agent._agent_execution_leases = []

        async def fake_run(**kwargs):
            if kwargs["starting_agent"] == "planner":
                child = await agent._run_agent_with_stage_sla(
                    starting_agent="designer",
                    input={},
                    role="designer",
                    event="nested_design",
                )
                self.assertIsNotNone(child)
                await asyncio.sleep(0.03)
                return SimpleNamespace(final_output="planner complete")
            await asyncio.sleep(0.10)
            return SimpleNamespace(final_output="design complete")

        with patch(
            "scenesmith.agent_utils.base_stateful_agent.Runner.run",
            side_effect=fake_run,
        ):
            result = await agent._run_agent_with_stage_sla(
                starting_agent="planner",
                input={},
                role="planner",
                event="planner",
            )

        self.assertIsNotNone(result)
        self.assertGreaterEqual(agent._stage_role_active_consumed["designer"], 0.09)
        self.assertLess(agent._stage_role_active_consumed["planner"], 0.08)


if __name__ == "__main__":
    unittest.main()
