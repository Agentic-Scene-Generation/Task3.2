"""Tests for complete-scene isolation and retry orchestration."""

import json
import tempfile
import unittest

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import scenesmith.experiments.indoor_scene_generation as scene_generation
from scenesmith.experiments.indoor_scene_generation import (
    IndoorSceneGenerationExperiment,
)


class TestSceneTaskIsolation(unittest.TestCase):
    """Verify retries use a clean output directory and remain selective."""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temporary_directory.name)
        self.experiment = IndoorSceneGenerationExperiment.__new__(
            IndoorSceneGenerationExperiment
        )
        self.experiment.output_dir = self.output_dir
        self.experiment.geometry_server = None
        self.experiment.hssd_server = None
        self.experiment.articulated_server = None
        self.experiment.materials_server = None
        self.cfg_dict = {"experiment": {"scene_retry_attempts": 1}}

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_native_crash_is_archived_and_retried(self) -> None:
        call_count = 0

        def fake_run(tasks, max_workers, return_values=False):
            nonlocal call_count
            call_count += 1
            task_id = tasks[0][0]
            scene_dir = self.output_dir / "scene_000"
            scene_dir.mkdir(parents=True, exist_ok=True)
            (scene_dir / "partial.txt").write_text(
                f"attempt {call_count}", encoding="utf-8"
            )
            if call_count == 1:
                return {
                    task_id: (
                        False,
                        "Process crashed (exitcode=-11 (SIGSEGV))",
                    )
                }
            return {task_id: (True, None)}

        with patch(
            "scenesmith.experiments.indoor_scene_generation." "run_parallel_isolated",
            side_effect=fake_run,
        ):
            self.experiment._run_isolated_scene_generation(
                prompts_with_ids=[(0, "A bedroom")],
                cfg_dict=self.cfg_dict,
                experiment_run_id="test-run",
                num_workers=1,
                capture_logs=False,
            )

        self.assertEqual(call_count, 2)
        archived_attempts = list(
            (self.output_dir / "failed_attempts").glob("scene_000_attempt_01_*")
        )
        self.assertEqual(len(archived_attempts), 1)
        self.assertTrue((archived_attempts[0] / "partial.txt").exists())
        self.assertTrue((self.output_dir / "scene_000" / "partial.txt").exists())

    def test_deterministic_failure_is_not_retried(self) -> None:
        call_count = 0

        def fake_run(tasks, max_workers, return_values=False):
            nonlocal call_count
            call_count += 1
            return {tasks[0][0]: (False, "Invalid start_stage 'bad_stage'")}

        with patch(
            "scenesmith.experiments.indoor_scene_generation." "run_parallel_isolated",
            side_effect=fake_run,
        ):
            with self.assertRaisesRegex(RuntimeError, "scene_000"):
                self.experiment._run_isolated_scene_generation(
                    prompts_with_ids=[(0, "A bedroom")],
                    cfg_dict=self.cfg_dict,
                    experiment_run_id="test-run",
                    num_workers=1,
                    capture_logs=False,
                )

        self.assertEqual(call_count, 1)
        self.assertFalse((self.output_dir / "failed_attempts").exists())

    def test_openrouter_upstream_rate_limit_is_retried(self) -> None:
        call_count = 0

        def fake_run(tasks, max_workers, return_values=False):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {
                    tasks[0][0]: (
                        False,
                        "openai/gpt-5.6-luna-pro is temporarily rate-limited "
                        "upstream. Please retry shortly.",
                    )
                }
            return {tasks[0][0]: (True, None)}

        with patch(
            "scenesmith.experiments.indoor_scene_generation."
            "run_parallel_isolated",
            side_effect=fake_run,
        ):
            self.experiment._run_isolated_scene_generation(
                prompts_with_ids=[(0, "A polygon bathroom")],
                cfg_dict=self.cfg_dict,
                experiment_run_id="test-run",
                num_workers=1,
                capture_logs=False,
            )

        self.assertEqual(call_count, 2)

    def test_worker_bootstrap_exit_one_is_not_retried(self) -> None:
        call_count = 0

        def fake_run(tasks, max_workers, return_values=False):
            nonlocal call_count
            call_count += 1
            return {tasks[0][0]: (False, "Process crashed (exitcode=1)")}

        with patch(
            "scenesmith.experiments.indoor_scene_generation." "run_parallel_isolated",
            side_effect=fake_run,
        ):
            with self.assertRaisesRegex(RuntimeError, "exitcode=1"):
                self.experiment._run_isolated_scene_generation(
                    prompts_with_ids=[(0, "A bedroom")],
                    cfg_dict=self.cfg_dict,
                    experiment_run_id="test-run",
                    num_workers=1,
                    capture_logs=False,
                )

        self.assertEqual(call_count, 1)
        self.assertFalse((self.output_dir / "failed_attempts").exists())

    def test_record_policy_returns_typed_scene_failure_without_batch_error(
        self,
    ) -> None:
        self.cfg_dict["experiment"]["scene_failure_policy"] = "record"

        def fake_run(tasks, max_workers, return_values=False):
            kwargs = tasks[0][2]
            error = scene_generation.PlannerWorkflowNoMutationError(
                stage="ceiling_mounted",
                workflow_calls=2,
                successful_mutations=0,
                operation="request_design_change",
                evidence={"asset_circuit_breaker": "open"},
            )
            scene_generation._write_scene_status(
                output_dir=self.output_dir,
                scene_id=kwargs["scene_id"],
                prompt=kwargs["prompt"],
                status="failed",
                attempt=kwargs["attempt"],
                run_id=kwargs["experiment_run_id"],
                error=str(error),
                failure=scene_generation._scene_failure_record(
                    error, attempt=kwargs["attempt"]
                ),
            )
            return {tasks[0][0]: (False, str(error))}

        with patch(
            "scenesmith.experiments.indoor_scene_generation." "run_parallel_isolated",
            side_effect=fake_run,
        ):
            recorded = self.experiment._run_isolated_scene_generation(
                prompts_with_ids=[(0, "A bedroom")],
                cfg_dict=self.cfg_dict,
                experiment_run_id="test-run",
                num_workers=1,
                capture_logs=False,
            )

        self.assertEqual(recorded[0]["failure_class"], "stage_unavailable")
        status = json.loads(
            (self.output_dir / "scene_000" / "scene_status.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(status["schema_version"], "scenesmith.scene_status.v3")
        self.assertEqual(status["failure"]["reason"], "no_mutation")
        self.assertFalse(status["failure"]["retryable"])
        self.assertEqual(
            status["failure"]["provenance"]["terminal_evidence"],
            {"asset_circuit_breaker": "open"},
        )

    def test_strict_policy_rejects_the_same_typed_scene_failure(self) -> None:
        error = scene_generation.IntentCompilationError(
            "compiler exhausted", trace={"attempts": [{}, {}]}
        )

        def fake_run(tasks, max_workers, return_values=False):
            kwargs = tasks[0][2]
            scene_generation._write_scene_status(
                output_dir=self.output_dir,
                scene_id=kwargs["scene_id"],
                prompt=kwargs["prompt"],
                status="failed",
                attempt=kwargs["attempt"],
                run_id=kwargs["experiment_run_id"],
                error=str(error),
                failure=scene_generation._scene_failure_record(
                    error, attempt=kwargs["attempt"]
                ),
            )
            return {tasks[0][0]: (False, str(error))}

        with patch(
            "scenesmith.experiments.indoor_scene_generation." "run_parallel_isolated",
            side_effect=fake_run,
        ):
            with self.assertRaisesRegex(RuntimeError, "intent_unavailable"):
                self.experiment._run_isolated_scene_generation(
                    prompts_with_ids=[(0, "A bedroom")],
                    cfg_dict=self.cfg_dict,
                    experiment_run_id="test-run",
                    num_workers=1,
                    capture_logs=False,
                )

    def test_record_policy_rejects_untyped_worker_failure(self) -> None:
        self.cfg_dict["experiment"]["scene_failure_policy"] = "record"

        with patch(
            "scenesmith.experiments.indoor_scene_generation." "run_parallel_isolated",
            return_value={"scene_000": (False, "Process crashed (exitcode=1)")},
        ):
            with self.assertRaisesRegex(RuntimeError, "not recordable"):
                self.experiment._run_isolated_scene_generation(
                    prompts_with_ids=[(0, "A bedroom")],
                    cfg_dict=self.cfg_dict,
                    experiment_run_id="test-run",
                    num_workers=1,
                    capture_logs=False,
                )

        status = json.loads(
            (self.output_dir / "scene_000" / "scene_status.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(status["failure"]["failure_class"], "fatal_run_failure")
        self.assertFalse(status["failure"]["recordable"])

    def test_record_policy_rejects_stale_typed_scene_failure(self) -> None:
        self.cfg_dict["experiment"]["scene_failure_policy"] = "record"

        def fake_run(tasks, max_workers, return_values=False):
            kwargs = tasks[0][2]
            error = scene_generation.PlannerWorkflowNoMutationError(
                stage="ceiling_mounted",
                workflow_calls=2,
                successful_mutations=0,
                operation="request_design_change",
            )
            scene_generation._write_scene_status(
                output_dir=self.output_dir,
                scene_id=kwargs["scene_id"],
                prompt=kwargs["prompt"],
                status="failed",
                attempt=kwargs["attempt"],
                run_id="previous-run",
                error=str(error),
                failure=scene_generation._scene_failure_record(
                    error, attempt=kwargs["attempt"]
                ),
            )
            return {tasks[0][0]: (False, str(error))}

        with patch(
            "scenesmith.experiments.indoor_scene_generation." "run_parallel_isolated",
            side_effect=fake_run,
        ):
            with self.assertRaisesRegex(RuntimeError, "run_id does not match"):
                self.experiment._run_isolated_scene_generation(
                    prompts_with_ids=[(0, "A bedroom")],
                    cfg_dict=self.cfg_dict,
                    experiment_run_id="test-run",
                    num_workers=1,
                    capture_logs=False,
                )

    def test_record_gate_requires_complete_matching_terminal_status(self) -> None:
        valid = {
            "schema_version": "scenesmith.scene_status.v3",
            "scene_id": 0,
            "status": "failed",
            "attempt": 1,
            "run_id": "test-run",
            "failure": {
                "failure_class": "stage_unavailable",
                "stage": "ceiling_mounted",
                "error_type": "PlannerWorkflowNoMutationError",
                "message": "no committed mutation",
                "reason": "no_mutation",
                "operation": "request_design_change",
                "root_error_type": "",
                "root_error_message": "",
                "retryable": False,
                "attempt": 1,
                "stage_execution_attempt": 1,
                "recordable": True,
                "provenance": {"workflow_calls": 2},
            },
        }
        trusted, reason = scene_generation._recordable_scene_failure(
            valid,
            scene_id=0,
            attempt=1,
            run_id="test-run",
        )
        self.assertTrue(trusted, reason)

        invalid_cases = {
            "schema": ("schema_version", "scenesmith.scene_status.v1"),
            "scene": ("scene_id", 9),
            "status": ("status", "running"),
            "attempt": ("attempt", 2),
            "run": ("run_id", "previous-run"),
        }
        for label, (field, value) in invalid_cases.items():
            with self.subTest(label=label):
                payload = json.loads(json.dumps(valid))
                payload[field] = value
                trusted, _ = scene_generation._recordable_scene_failure(
                    payload,
                    scene_id=0,
                    attempt=1,
                    run_id="test-run",
                )
                self.assertFalse(trusted)

        invalid_failure_cases = {
            "failure_attempt": ("attempt", 2),
            "recordable": ("recordable", False),
            "error_type": ("error_type", "RuntimeError"),
            "stage": ("stage", ""),
            "message": ("message", ""),
            "provenance": ("provenance", []),
            "reason": ("reason", "child_failure"),
            "retryable": ("retryable", True),
            "stage_execution_attempt": ("stage_execution_attempt", 0),
        }
        for label, (field, value) in invalid_failure_cases.items():
            with self.subTest(label=label):
                payload = json.loads(json.dumps(valid))
                payload["failure"][field] = value
                trusted, _ = scene_generation._recordable_scene_failure(
                    payload,
                    scene_id=0,
                    attempt=1,
                    run_id="test-run",
                )
                self.assertFalse(trusted)

    def test_child_timeout_is_retryable_but_never_recordable(self) -> None:
        error = scene_generation.PlannerStageFailure(
            reason="child_failure",
            stage="ceiling_mounted",
            workflow_calls=1,
            successful_mutations=0,
            operation="request_initial_design",
            stage_execution_attempt=2,
            retryable=True,
            root_error_type="APITimeoutError",
            root_error_message="request timed out",
        )

        failure = scene_generation._scene_failure_record(error, attempt=2)
        payload = {
            "schema_version": "scenesmith.scene_status.v3",
            "scene_id": 7,
            "status": "failed",
            "attempt": 2,
            "run_id": "current-run",
            "failure": failure,
        }

        self.assertEqual(failure["failure_class"], "scene_runtime_failure")
        self.assertEqual(failure["stage"], "ceiling_mounted")
        self.assertEqual(failure["reason"], "child_failure")
        self.assertEqual(failure["root_error_type"], "APITimeoutError")
        self.assertTrue(failure["retryable"])
        self.assertFalse(failure["recordable"])
        self.assertTrue(
            scene_generation._is_retryable_scene_failure(
                payload,
                scene_id=7,
                attempt=2,
                retry_budget=2,
                run_id="current-run",
            )
        )
        self.assertFalse(
            scene_generation._is_retryable_scene_failure(
                payload,
                scene_id=7,
                attempt=3,
                retry_budget=2,
                run_id="current-run",
            )
        )

    def test_unclassified_runtime_failure_has_valid_nonempty_reason(self) -> None:
        failure = scene_generation._scene_failure_record(
            RuntimeError("support surface extraction failed"), attempt=1
        )
        payload = {
            "schema_version": "scenesmith.scene_status.v3",
            "scene_id": 7,
            "status": "failed",
            "attempt": 1,
            "run_id": "current-run",
            "failure": failure,
        }

        validated, validation_error = scene_generation._validated_scene_failure(
            payload,
            scene_id=7,
            attempt=1,
            run_id="current-run",
        )

        self.assertEqual(validation_error, "")
        self.assertIsNotNone(validated)
        self.assertEqual(failure["reason"], "unclassified_runtime_failure")
        self.assertEqual(failure["failure_class"], "scene_runtime_failure")
        self.assertFalse(failure["recordable"])
        self.assertFalse(
            scene_generation._is_retryable_scene_failure(
                payload,
                scene_id=7,
                attempt=1,
                retry_budget=1,
                run_id="current-run",
            )
        )

    def test_retry_gate_rejects_stale_or_mismatched_structured_status(self) -> None:
        failure = scene_generation._worker_failure_record(
            "APITimeoutError: request timed out",
            attempt=1,
            status_error="missing status",
        )
        payload = {
            "schema_version": "scenesmith.scene_status.v3",
            "scene_id": 4,
            "status": "failed",
            "attempt": 1,
            "run_id": "old-run",
            "failure": failure,
        }

        self.assertFalse(
            scene_generation._is_retryable_scene_failure(
                payload,
                scene_id=4,
                attempt=1,
                retry_budget=1,
                run_id="current-run",
            )
        )
        payload["run_id"] = "current-run"
        payload["schema_version"] = "scenesmith.scene_status.v2"
        self.assertFalse(
            scene_generation._is_retryable_scene_failure(
                payload,
                scene_id=4,
                attempt=1,
                retry_budget=1,
                run_id="current-run",
            )
        )

    def test_retry_gate_rejects_malformed_or_forged_retry_disposition(self) -> None:
        error = scene_generation.PlannerStageFailure(
            reason="child_failure",
            stage="furniture",
            workflow_calls=1,
            successful_mutations=0,
            operation="request_initial_design",
            retryable=True,
            root_error_type="APITimeoutError",
            root_error_message="request timed out",
        )
        payload = {
            "schema_version": "scenesmith.scene_status.v3",
            "scene_id": 5,
            "status": "failed",
            "attempt": 1,
            "run_id": "current-run",
            "failure": scene_generation._scene_failure_record(error, attempt=1),
        }

        for field in ("reason", "root_error_type", "recordable"):
            with self.subTest(missing=field):
                malformed = json.loads(json.dumps(payload))
                malformed["failure"].pop(field)
                self.assertFalse(
                    scene_generation._is_retryable_scene_failure(
                        malformed,
                        scene_id=5,
                        attempt=1,
                        retry_budget=1,
                        run_id="current-run",
                    )
                )

        forged = json.loads(json.dumps(payload))
        forged["failure"]["root_error_type"] = "ValueError"
        self.assertFalse(
            scene_generation._is_retryable_scene_failure(
                forged,
                scene_id=5,
                attempt=1,
                retry_budget=1,
                run_id="current-run",
            )
        )

    def test_worker_retry_fallback_accepts_native_exit_only(self) -> None:
        api_failure = scene_generation._worker_failure_record(
            "APITimeoutError: request timed out",
            attempt=1,
            status_error="status write failed",
        )
        native_failure = scene_generation._worker_failure_record(
            "Process crashed (exitcode=-11 (SIGSEGV))",
            attempt=1,
            status_error="missing status",
        )

        self.assertFalse(api_failure["retryable"])
        self.assertTrue(native_failure["retryable"])

    def test_invalid_scene_failure_policy_fails_before_dispatch(self) -> None:
        self.cfg_dict["experiment"]["scene_failure_policy"] = "ignore"

        with patch(
            "scenesmith.experiments.indoor_scene_generation.run_parallel_isolated"
        ) as run:
            with self.assertRaisesRegex(ValueError, "scene_failure_policy"):
                self.experiment._run_isolated_scene_generation(
                    prompts_with_ids=[(0, "A bedroom")],
                    cfg_dict=self.cfg_dict,
                    experiment_run_id="test-run",
                    num_workers=1,
                    capture_logs=False,
                )

        run.assert_not_called()

    def test_degraded_quality_policy_disables_stage_abort_gates(self) -> None:
        cfg = {
            "experiment": {"quality_failure_policy": "degraded"},
            "furniture_agent": {
                "fail_stage_on_unresolved_hard_constraints": True,
            },
            "manipuland_agent": {
                "fail_stage_on_unresolved_hard_constraints": True,
            },
        }

        policy = scene_generation._configure_stage_quality_gates(cfg)

        self.assertEqual(policy, "degraded")
        self.assertFalse(
            cfg["furniture_agent"]["fail_stage_on_unresolved_hard_constraints"]
        )
        self.assertFalse(
            cfg["manipuland_agent"]["fail_stage_on_unresolved_hard_constraints"]
        )

    def test_degraded_completion_uses_truthful_marker(self) -> None:
        scene_generation._write_scene_completion(
            output_dir=self.output_dir,
            scene_id=0,
            prompt="A bedroom",
            attempt=1,
            degraded=True,
        )

        scene_dir = self.output_dir / "scene_000"
        self.assertTrue((scene_dir / "_DEGRADED").exists())
        self.assertFalse((scene_dir / "_SUCCESS").exists())
        status = (scene_dir / "scene_status.json").read_text(encoding="utf-8")
        self.assertIn('"schema_version": "scenesmith.scene_status.v3"', status)
        self.assertIn('"status": "completed_with_quality_issues"', status)


class TestManipulandAgentCleanup(unittest.TestCase):
    """Manipuland services must not survive a completed or failed scene."""

    def test_cleanup_runs_after_success(self) -> None:
        agent = MagicMock()
        agent.add_manipulands = AsyncMock()
        scene = MagicMock()

        scene_generation._add_manipulands_with_cleanup(agent, scene)

        agent.add_manipulands.assert_awaited_once_with(scene=scene)
        agent.cleanup.assert_called_once_with()

    def test_cleanup_runs_after_failure(self) -> None:
        agent = MagicMock()
        agent.add_manipulands = AsyncMock(side_effect=RuntimeError("stage failed"))

        with self.assertRaisesRegex(RuntimeError, "stage failed"):
            scene_generation._add_manipulands_with_cleanup(agent, MagicMock())

        agent.cleanup.assert_called_once_with()


class TestTargetedManipulandReplay(unittest.TestCase):
    """Only explicit support-surface replays may relax the full-scene gate."""

    def test_explicit_target_ids_enable_targeted_replay(self) -> None:
        cfg = {
            "manipuland_agent": {
                "target_furniture_ids": ["dining_table_0"],
            }
        }

        self.assertTrue(
            scene_generation._is_targeted_manipuland_replay(
                cfg_dict=cfg,
                start_stage="manipuland",
                stop_stage="manipuland",
            )
        )

    def test_csv_target_ids_enable_targeted_replay(self) -> None:
        cfg = {
            "manipuland_agent": {
                "target_furniture_ids": "dining_table_0, sideboard_0",
            }
        }

        self.assertTrue(
            scene_generation._is_targeted_manipuland_replay(
                cfg_dict=cfg,
                start_stage="manipuland",
                stop_stage="manipuland",
            )
        )

    def test_stage_or_target_mismatch_keeps_full_scene_gate(self) -> None:
        cases = (
            ({"manipuland_agent": {"target_furniture_ids": []}}, "manipuland"),
            (
                {"manipuland_agent": {"target_furniture_ids": ["dining_table_0"]}},
                "ceiling_mounted",
            ),
        )

        for cfg, stop_stage in cases:
            with self.subTest(cfg=cfg, stop_stage=stop_stage):
                self.assertFalse(
                    scene_generation._is_targeted_manipuland_replay(
                        cfg_dict=cfg,
                        start_stage="manipuland",
                        stop_stage=stop_stage,
                    )
                )


class TestWorkerReasoningPersistenceBootstrap(unittest.TestCase):
    """Verify every clean worker restores passive reasoning persistence."""

    def setUp(self) -> None:
        self.cfg_dict = {
            "openai": {
                "reasoning_persistence": {
                    "enabled": True,
                    "provider": "qwen",
                }
            },
            "llm": {"model_id": "Qwen/Qwen3.6-27B"},
        }

    def test_worker_configuration_is_restored_from_serialized_config(self) -> None:
        with (
            patch.object(
                scene_generation, "configure_reasoning_persistence"
            ) as configure,
            patch.dict(
                scene_generation.os.environ,
                {"OPENAI_BASE_URL": "http://127.0.0.1:8002/v1"},
            ),
        ):
            scene_generation._configure_reasoning_persistence_for_worker(self.cfg_dict)

        configure.assert_called_once_with(
            enabled=True,
            provider="qwen",
            model_id="Qwen/Qwen3.6-27B",
            base_url="http://127.0.0.1:8002/v1",
        )

    def test_all_isolated_worker_entrypoints_restore_configuration(self) -> None:
        class StopAfterBootstrap(Exception):
            pass

        worker_calls = (
            lambda: IndoorSceneGenerationExperiment._generate_single_scene(
                prompt="A bedroom",
                scene_id=0,
                output_dir=Path("/tmp/test-reasoning-bootstrap"),
                cfg_dict=self.cfg_dict,
            ),
            lambda: scene_generation._generate_floor_plan_worker(
                prompt="A bedroom",
                scene_dir="/tmp/test-reasoning-bootstrap/scene_000",
                cfg_dict=self.cfg_dict,
                experiment_run_id=None,
            ),
            lambda: scene_generation._generate_room_worker(
                room_id="bedroom",
                room_prompt="A bedroom",
                room_geometry_dict={},
                room_dir="/tmp/test-reasoning-bootstrap/scene_000/room_bedroom",
                cfg_dict=self.cfg_dict,
                start_stage="furniture",
                stop_stage="manipuland",
                scene_id=0,
            ),
        )

        for worker_call in worker_calls:
            with self.subTest(worker=worker_call):
                with (
                    patch.object(scene_generation, "_reset_inherited_sdk_state"),
                    patch.object(
                        scene_generation,
                        "_configure_reasoning_persistence_for_worker",
                    ) as configure,
                    patch.object(
                        scene_generation.faulthandler,
                        "enable",
                        side_effect=StopAfterBootstrap,
                    ),
                ):
                    with self.assertRaises(StopAfterBootstrap):
                        worker_call()

                configure.assert_called_once_with(self.cfg_dict)


if __name__ == "__main__":
    unittest.main()
