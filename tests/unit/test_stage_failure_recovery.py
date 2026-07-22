import unittest

from types import SimpleNamespace

from scenesmith.experiments.indoor_scene_generation import (
    _is_repairable_stage_validation,
    _is_retryable_scene_failure,
    _run_sceneexpert_placement_stage,
)
from scenesmith.scene_expert.exceptions import StageValidationError


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

    def test_exhausted_stage_uses_focused_rescue_then_degrades(self) -> None:
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
            "rescue": 0,
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

        async def rescue(reasons: list[str]) -> None:
            calls["rescue"] += 1
            raise StageValidationError(stage="wall_mounted", reasons=reasons)

        async def complete_degraded(reasons: list[str]) -> None:
            calls["degraded"] += 1

        scene = FakeScene()
        agent = SimpleNamespace(
            prepare_stage_regeneration=prepare,
            run_required_stage_completion_rescue=rescue,
            complete_repair_exhausted_stage=complete_degraded,
        )

        attempts = _run_sceneexpert_placement_stage(
            stage="wall_mounted",
            agent=agent,
            scene=scene,
            run_once=run_once,
        )

        self.assertEqual(attempts, 1)
        self.assertEqual(calls, {"run": 2, "prepare": 2, "rescue": 1, "degraded": 1})
        self.assertEqual(scene.restore_calls, 1)

    def test_disabled_sceneexpert_does_not_add_recovery_attempts(self) -> None:
        class FakeScene:
            text_description = "bedroom"

            @staticmethod
            def to_state_dict() -> dict:
                return {"objects": {}}

        calls = {"run": 0, "rescue": 0}

        async def run_once() -> None:
            calls["run"] += 1
            raise StageValidationError(
                stage="wall_mounted",
                reasons=["missing required stage output"],
            )

        async def rescue(reasons: list[str]) -> None:
            calls["rescue"] += 1

        with self.assertRaises(StageValidationError):
            _run_sceneexpert_placement_stage(
                stage="wall_mounted",
                agent=SimpleNamespace(
                    run_required_stage_completion_rescue=rescue,
                ),
                scene=FakeScene(),
                run_once=run_once,
            )

        self.assertEqual(calls, {"run": 1, "rescue": 0})


if __name__ == "__main__":
    unittest.main()
