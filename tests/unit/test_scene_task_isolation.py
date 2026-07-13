"""Tests for complete-scene isolation and retry orchestration."""

import tempfile
import unittest

from pathlib import Path
from unittest.mock import patch

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
            "scenesmith.experiments.indoor_scene_generation."
            "run_parallel_isolated",
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
            "scenesmith.experiments.indoor_scene_generation."
            "run_parallel_isolated",
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


if __name__ == "__main__":
    unittest.main()
