import unittest

from pathlib import Path

from scenesmith.scene_expert.schemas import StageBudget

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class SceneExpertRuntimeBoundaryTest(unittest.TestCase):
    def _source(self, relative_path: str) -> str:
        return (_PROJECT_ROOT / relative_path).read_text(encoding="utf-8")

    def test_harness_budget_does_not_override_scenesmith_runtime(self) -> None:
        self.assertEqual(
            set(StageBudget.model_fields),
            {"max_designer_iterations", "max_repair_steps"},
        )

        hooks_source = self._source("scenesmith/scene_expert/hooks.py")
        self.assertNotIn("scene_expert_stage_budget", hooks_source)
        self.assertNotIn("scene_expert_min_output_objects", hooks_source)
        self.assertNotIn("scene_expert_max_output_objects", hooks_source)

    def test_asset_and_critic_ownership_remain_in_scenesmith(self) -> None:
        asset_source = self._source("scenesmith/agent_utils/asset_manager.py")
        scoring_source = self._source("scenesmith/agent_utils/scoring.py")
        verifier_source = self._source("scenesmith/scene_expert/verifier.py")

        self.assertNotIn("AssetRuntimeGate", asset_source)
        self.assertNotIn("scene_expert_stage_budget", asset_source)
        self.assertNotIn("scenesmith.scene_expert", scoring_source)
        self.assertNotIn("room_size_policy", verifier_source)
        self.assertNotIn("placeholder_asset", verifier_source)

    def test_sceneexpert_verifier_only_bridges_existing_critic_output(self) -> None:
        verifier_source = self._source("scenesmith/scene_expert/verifier.py")

        self.assertIn("_find_scores_yaml", verifier_source)
        self.assertIn("critic_bridge_enabled", verifier_source)
        self.assertNotIn("chat.completions.create", verifier_source)

    def test_trace_gate_avoids_constructing_trace_logger_when_disabled(self) -> None:
        hooks_source = self._source("scenesmith/scene_expert/hooks.py")

        self.assertIn("trace_logger: TraceLogger | None", hooks_source)
        self.assertNotIn("self._trace_logger = TraceLogger(", hooks_source)
        trace_guard = hooks_source.rfind(
            'if component_flags["trace"]:',
            0,
            hooks_source.index("trace_logger = TraceLogger("),
        )
        self.assertGreaterEqual(trace_guard, 0)

    def test_parallel_runner_only_applies_explicit_component_overrides(self) -> None:
        runner_source = self._source("scripts/run_parallel_critic_on.sh")

        self.assertIn('raw="${!env_name:-}"', runner_source)
        self.assertIn(
            "components.${component}.enabled=${normalized}",
            runner_source,
        )
        for component in (
            "task_compiler",
            "harness",
            "harness_budget",
            "global_planner",
            "prompt_injection",
            "fast_memory_retrieval",
            "memory_writer",
            "stage_working_memory",
            "verifier",
            "repair",
            "critic_bridge",
            "trace",
            "structured_llm",
            "slow_memory_capture",
        ):
            self.assertIn(f" {component}\n", runner_source)

    def test_full_runtime_requires_the_same_hybrid_memory_preflight(self) -> None:
        runner_source = self._source("scripts/run_parallel_critic_on.sh")

        self.assertIn(
            '|| [ "$EXPERIMENT" = "ablation_5_qwen3_full" ]',
            runner_source,
        )

    def test_parallel_runner_preserves_shared_base_recovery_failure(self) -> None:
        runner_source = self._source("scripts/run_parallel_critic_on.sh")

        self.assertIn("if run_shared_base_with_recovery; then", runner_source)
        self.assertNotIn("if ! run_shared_base_with_recovery; then", runner_source)

    def test_parallel_runner_scopes_scene_failure_policy(self) -> None:
        runner_source = self._source("scripts/run_parallel_critic_on.sh")

        self.assertIn(
            'SCENE_FAILURE_POLICY="${SCENE_FAILURE_POLICY:-record}"',
            runner_source,
        )
        self.assertIn('scene_failure_policy="strict"', runner_source)
        self.assertIn(
            '"experiment.scene_failure_policy=${scene_failure_policy}"',
            runner_source,
        )

    def test_parallel_runner_preserves_generation_exit_on_metrics_failure(
        self,
    ) -> None:
        runner_source = self._source("scripts/run_parallel_critic_on.sh")

        self.assertIn("metrics_exit_code=$?", runner_source)
        self.assertIn('if [ "$run_exit_code" -eq 0 ]; then', runner_source)
        self.assertIn('run_exit_code="$metrics_exit_code"', runner_source)

    def test_partial_pipeline_cannot_promote_long_term_memory(self) -> None:
        hooks_source = self._source("scenesmith/scene_expert/hooks.py")

        self.assertIn(
            "GENERATION_TERMINAL_STAGE = GENERATION_STAGE_ORDER[-1]",
            hooks_source,
        )
        self.assertIn(
            "allow_long_term_memory_updates=(stop_stage == GENERATION_TERMINAL_STAGE)",
            hooks_source,
        )
        self.assertNotIn(
            "allow_long_term_memory_updates=(stop_stage == CONTRACT_STAGE_ORDER[-1])",
            hooks_source,
        )
        self.assertIn("skipped_non_terminal_pipeline", hooks_source)

    def test_online_pipeline_never_starts_dpo_training(self) -> None:
        online_sources = "\n".join(
            self._source(path)
            for path in (
                "main.py",
                "scenesmith/experiments/indoor_scene_generation.py",
                "scenesmith/scene_expert/hooks.py",
            )
        )

        self.assertNotIn("train_sceneexpert_dpo", online_sources)
        self.assertNotIn("DPOTrainer", online_sources)


if __name__ == "__main__":
    unittest.main()
