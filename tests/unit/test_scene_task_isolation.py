"""Tests for complete-scene isolation and retry orchestration."""

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
