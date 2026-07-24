import unittest

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from scenesmith.experiments.indoor_scene_generation import (
    _is_repairable_stage_validation,
    _is_retryable_scene_failure,
    _run_sceneexpert_placement_stage,
    _score_postprocessed_candidate_or_pause,
)
from scenesmith.scene_expert.exceptions import StageValidationError
from scenesmith.scene_expert.runtime_state import ScenePausedError


class StageFailureRecoveryTest(unittest.TestCase):
    def test_collision_and_missing_content_are_repairable(self) -> None:
        error = StageValidationError(
            stage="furniture",
            reasons=[
                "physics hard violation: collisions",
                "missing required wardrobe: expected 1, found 0",
            ],
        )

        self.assertTrue(_is_repairable_stage_validation(error))

    def test_invalid_room_geometry_remains_terminal(self) -> None:
        error = StageValidationError(
            stage="furniture",
            reasons=["invalid room geometry: floor polygon is empty"],
        )

        self.assertFalse(_is_repairable_stage_validation(error))

    def test_deterministic_stage_failure_is_not_a_process_retry(self) -> None:
        self.assertFalse(
            _is_retryable_scene_failure(
                "wall_mounted stage failed deterministic validation: "
                "missing required stage output"
            )
        )
        self.assertTrue(_is_retryable_scene_failure("worker exitcode=-11"))

    def test_exhausted_stage_retries_full_planner_then_degrades(self) -> None:
        class FakeScene:
            text_description = "bedroom"
            scene_expert_stage_budget = {"max_stage_regenerations": 1}

            def __init__(self) -> None:
                self.restore_calls = 0

            def to_state_dict(self) -> dict:
                return {"objects": {}}

            def restore_from_state_dict(self, state: dict) -> None:
                self.restore_calls += 1

        calls = {
            "run": 0,
            "prepare": 0,
            "degraded": 0,
        }

        async def run_once() -> None:
            calls["run"] += 1
            raise StageValidationError(
                stage="wall_mounted",
                reasons=["missing required stage output"],
            )

        async def prepare(reasons: list[str]) -> None:
            calls["prepare"] += 1

        async def complete_degraded(reasons: list[str]) -> None:
            calls["degraded"] += 1

        scene = FakeScene()
        agent = SimpleNamespace(
            prepare_stage_regeneration=prepare,
            complete_repair_exhausted_stage=complete_degraded,
        )

        attempts = _run_sceneexpert_placement_stage(
            stage="wall_mounted",
            agent=agent,
            scene=scene,
            run_once=run_once,
        )

        self.assertEqual(attempts, 1)
        self.assertEqual(calls, {"run": 2, "prepare": 1, "degraded": 1})
        self.assertEqual(scene.restore_calls, 1)

    def test_disabled_sceneexpert_does_not_add_recovery_attempts(self) -> None:
        class FakeScene:
            text_description = "bedroom"

            @staticmethod
            def to_state_dict() -> dict:
                return {"objects": {}}

        calls = {"run": 0}

        async def run_once() -> None:
            calls["run"] += 1
            raise StageValidationError(
                stage="wall_mounted",
                reasons=["missing required stage output"],
            )

        with self.assertRaises(StageValidationError):
            _run_sceneexpert_placement_stage(
                stage="wall_mounted",
                agent=SimpleNamespace(),
                scene=FakeScene(),
                run_once=run_once,
            )

        self.assertEqual(calls, {"run": 1})

    def test_budget_exhausted_zero_output_still_gets_fresh_planner_retry(
        self,
    ) -> None:
        class FakeScene:
            text_description = "living room"
            scene_expert_stage_budget = {"max_stage_regenerations": 1}

            @staticmethod
            def to_state_dict() -> dict:
                return {"objects": {}}

            @staticmethod
            def restore_from_state_dict(state: dict) -> None:
                del state

        calls = {"run": 0, "prepare": 0, "degraded": 0}

        async def run_once() -> None:
            calls["run"] += 1
            raise StageValidationError(
                stage="wall_mounted",
                reasons=[
                    "missing required stage output: wall_mounted produced 0 "
                    "objects but requires at least 1"
                ],
            )

        async def prepare(reasons: list[str]) -> None:
            del reasons
            calls["prepare"] += 1

        async def complete_degraded(reasons: list[str]) -> None:
            del reasons
            calls["degraded"] += 1

        attempts = _run_sceneexpert_placement_stage(
            stage="wall_mounted",
            agent=SimpleNamespace(
                _stage_runtime_exhausted=True,
                _planner_budget_exhausted=True,
                prepare_stage_regeneration=prepare,
                complete_repair_exhausted_stage=complete_degraded,
            ),
            scene=FakeScene(),
            run_once=run_once,
        )

        self.assertEqual(1, attempts)
        self.assertEqual({"run": 2, "prepare": 1, "degraded": 1}, calls)

    def test_second_critic_timeout_pauses_without_redesign_loop(self) -> None:
        class FakeScene:
            text_description = "living room"
            scene_expert_stage_budget = {"max_stage_regenerations": 1}

            def __init__(self, scene_dir: Path) -> None:
                self.scene_dir = scene_dir
                self.room_id = "living_room"

            def to_state_dict(self) -> dict:
                return {"objects": {}}

            def content_hash(self) -> str:
                return "candidate-hash"

        calls = {"run": 0, "critic": 0}

        async def run_once() -> None:
            calls["run"] += 1
            raise StageValidationError(
                stage="wall_mounted",
                reasons=[
                    "visual critic did not produce a trustworthy score after "
                    "bounded compact retries"
                ],
            )

        async def retry_critic() -> None:
            calls["critic"] += 1
            raise StageValidationError(
                stage="wall_mounted",
                reasons=[
                    "visual critic did not produce a trustworthy score after "
                    "bounded compact retries"
                ],
            )

        with TemporaryDirectory() as tmp:
            scene = FakeScene(Path(tmp))
            with self.assertRaises(ScenePausedError):
                _run_sceneexpert_placement_stage(
                    stage="wall_mounted",
                    agent=SimpleNamespace(
                        retry_final_critic_evaluation=retry_critic,
                        stage_working_memory=SimpleNamespace(
                            scene_root_dir=Path(tmp)
                        ),
                        _last_score_provenance={"score_source": "unavailable"},
                    ),
                    scene=scene,
                    run_once=run_once,
                )

            self.assertEqual({"run": 1, "critic": 1}, calls)
            self.assertIn(
                "expanded_compact_critic_retry",
                scene.scene_expert_runtime_repair_events,
            )
            self.assertTrue(
                (Path(tmp) / "scene_expert" / "resume" / "pause_manifest.json").exists()
            )

    def test_postprocessed_candidate_gets_one_transport_retry(self) -> None:
        class FakeScene:
            room_id = "living_room"

            def __init__(self, scene_dir: Path) -> None:
                self.scene_dir = scene_dir

            def to_state_dict(self) -> dict:
                return {"objects": {}}

            def content_hash(self) -> str:
                return "postprocess-hash"

        calls = {"critic": 0}

        async def retry_critic() -> None:
            calls["critic"] += 1
            if calls["critic"] == 1:
                raise StageValidationError(
                    stage="manipuland",
                    reasons=[
                        "visual critic did not produce a trustworthy score "
                        "after bounded compact retries"
                    ],
                )

        with TemporaryDirectory() as tmp:
            scene = FakeScene(Path(tmp))
            events: list[str] = []
            _score_postprocessed_candidate_or_pause(
                stage="manipuland",
                agent=SimpleNamespace(
                    retry_final_critic_evaluation=retry_critic,
                    stage_working_memory=SimpleNamespace(
                        scene_root_dir=Path(tmp)
                    ),
                    _last_score_provenance={"score_source": "vlm_critic"},
                ),
                scene=scene,
                runtime_events=events,
            )

            self.assertEqual(2, calls["critic"])
            self.assertIn("postprocess_final_critic_verified", events)
            self.assertFalse(
                (Path(tmp) / "scene_expert" / "resume" / "pause_manifest.json").exists()
            )


if __name__ == "__main__":
    unittest.main()
