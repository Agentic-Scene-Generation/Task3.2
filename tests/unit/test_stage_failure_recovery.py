import json
import unittest

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from scenesmith.experiments.indoor_scene_generation import (
    _is_repairable_stage_validation,
    _is_retryable_scene_failure,
    _raise_if_required_assets_unavailable,
    _run_sceneexpert_placement_stage,
    _score_postprocessed_candidate_or_pause,
    _stage_recovery_made_progress,
    _stage_validation_kind,
    _write_batch_summary,
)
from scenesmith.scene_expert.exceptions import StageValidationError
from scenesmith.scene_expert.runtime_state import (
    ScenePausedError,
    candidate_state_hash,
    split_degraded_stage_reasons,
)


class StageFailureRecoveryTest(unittest.TestCase):
    def test_candidate_hash_ignores_mutable_prompt_injection(self) -> None:
        scene = SimpleNamespace(
            text_description="injected stage brief",
            to_state_dict=lambda: {
                "text_description": "injected stage brief",
                "objects": {"bed_0": {"position": [1.0, 2.0, 0.0]}},
            },
        )

        injected_hash = candidate_state_hash(scene)
        scene.text_description = "original user prompt"
        scene.to_state_dict = lambda: {
            "text_description": "original user prompt",
            "objects": {"bed_0": {"position": [1.0, 2.0, 0.0]}},
        }

        self.assertEqual(injected_hash, candidate_state_hash(scene))

    def test_required_asset_unavailable_is_not_retried_as_layout_failure(self) -> None:
        error = StageValidationError(
            stage="furniture",
            reasons=[
                "required asset unavailable: no semantically admitted real HSSD "
                "asset for family 'plant'"
            ],
        )
        agent = SimpleNamespace(unavailable_required_asset_families=lambda: ["plant"])

        self.assertEqual("asset_unavailable", _stage_validation_kind(error, agent))

    def test_retry_progress_requires_acceptance_relevant_improvement(self) -> None:
        before = {
            "placed_count": 0,
            "admitted_asset_count": 1,
            "unavailable_required_count": 0,
            "required_missing_count": 1,
            "hard_issue_count": 2,
            "reason_signature": ("collision at x=1",),
        }
        renamed_only = dict(
            before,
            reason_signature=("collision at x=2",),
        )
        improved = dict(before, hard_issue_count=1)

        self.assertFalse(_stage_recovery_made_progress(before, renamed_only))
        self.assertTrue(_stage_recovery_made_progress(before, improved))

    def test_degraded_reasons_are_scoped_to_the_origin_stage(self) -> None:
        all_reasons, current, upstream = split_degraded_stage_reasons(
            [
                "[furniture] visual critic unavailable",
                "[wall_mounted] optional asset unavailable",
            ],
            current_stage="wall_mounted",
        )

        self.assertEqual(2, len(all_reasons))
        self.assertEqual(["[wall_mounted] optional asset unavailable"], current)
        self.assertEqual(["[furniture] visual critic unavailable"], upstream)

    def test_batch_summary_preserves_degraded_and_orphaned_outcomes(self) -> None:
        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            degraded_manifest = (
                output_dir
                / "scene_000"
                / "scene_expert"
                / "degraded"
                / "degraded_manifest.json"
            )
            degraded_manifest.parent.mkdir(parents=True)
            degraded_manifest.write_text(
                '{"status": "DEGRADED_INCOMPLETE"}',
                encoding="utf-8",
            )
            orphan_status = output_dir / "scene_001" / "scene_status.json"
            orphan_status.parent.mkdir(parents=True)
            orphan_status.write_text(
                json.dumps(
                    {
                        "status": "running",
                        "pid": 2_147_483_647,
                    }
                ),
                encoding="utf-8",
            )

            _write_batch_summary(
                output_dir=output_dir,
                experiment_run_id="run-test",
                prompts_with_ids=[(0, "bedroom"), (1, "living room")],
                results={"scene_000": ("completed", None)},
            )

            payload = json.loads(
                (output_dir / "batch_summary.json").read_text(encoding="utf-8")
            )
            statuses = {item["scene_id"]: item["status"] for item in payload["scenes"]}
            self.assertEqual("degraded_incomplete", statuses["scene_000"])
            self.assertEqual("orphaned", statuses["scene_001"])
            self.assertEqual(1, payload["degraded_incomplete_scenes"])
            self.assertEqual(1, payload["orphaned_scenes"])

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

    def test_exhausted_stage_retries_export_degraded_incomplete(self) -> None:
        class FakeScene:
            text_description = "bedroom"
            scene_expert_stage_budget = {"max_stage_regenerations": 1}

            def __init__(self, scene_dir: Path) -> None:
                self.restore_calls = 0
                self.scene_dir = scene_dir
                self.room_id = "bedroom"

            def to_state_dict(self) -> dict:
                return {"objects": {}}

            def restore_from_state_dict(self, state: dict) -> None:
                self.restore_calls += 1

            def content_hash(self) -> str:
                return "empty-stage"

        calls = {
            "run": 0,
            "prepare": 0,
        }

        async def run_once() -> None:
            calls["run"] += 1
            raise StageValidationError(
                stage="wall_mounted",
                reasons=["missing required stage output"],
            )

        async def prepare(reasons: list[str]) -> None:
            calls["prepare"] += 1

        with TemporaryDirectory() as tmp:
            scene = FakeScene(Path(tmp))
            agent = SimpleNamespace(
                prepare_stage_regeneration=prepare,
                admitted_stage_assets=lambda: [],
                stage_working_memory=SimpleNamespace(scene_root_dir=Path(tmp)),
            )

            _run_sceneexpert_placement_stage(
                stage="wall_mounted",
                agent=agent,
                scene=scene,
                run_once=run_once,
            )

            self.assertEqual(calls, {"run": 2, "prepare": 1})
            self.assertEqual(scene.restore_calls, 1)
            manifest = (
                Path(tmp) / "scene_expert" / "degraded" / "degraded_manifest.json"
            ).read_text(encoding="utf-8")
            self.assertIn('"status": "DEGRADED_INCOMPLETE"', manifest)
            self.assertIn(
                '"recommended_resume_action": "retry_stage_asset_acquisition"', manifest
            )

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

    def test_optional_zero_output_retries_then_continues_with_diagnostic(
        self,
    ) -> None:
        class FakeScene:
            text_description = "bedroom"
            scene_expert_stage_budget = {"max_stage_regenerations": 1}
            scene_expert_min_output_objects = 1
            scene_expert_required_min_output_objects = 0

            def __init__(self) -> None:
                self.restore_calls = 0

            @staticmethod
            def to_state_dict() -> dict:
                return {"objects": {}}

            def restore_from_state_dict(self, state: dict) -> None:
                del state
                self.restore_calls += 1

            @staticmethod
            def get_objects_by_type(object_type: object) -> list:
                del object_type
                return []

        calls = {"run": 0, "prepare": 0}

        async def run_once() -> None:
            calls["run"] += 1

        async def prepare(reasons: list[str]) -> None:
            self.assertTrue(reasons)
            calls["prepare"] += 1

        scene = FakeScene()
        attempts = _run_sceneexpert_placement_stage(
            stage="wall_mounted",
            agent=SimpleNamespace(
                admitted_stage_assets=lambda: [],
                prepare_stage_regeneration=prepare,
            ),
            scene=scene,
            run_once=run_once,
        )

        self.assertEqual(attempts, 1)
        self.assertEqual(calls, {"run": 2, "prepare": 1})
        self.assertEqual(scene.restore_calls, 1)
        self.assertIn(
            "optional_output_target_unmet",
            scene.scene_expert_runtime_repair_events,
        )
        self.assertIn("wall_mounted", scene.scene_expert_stage_diagnostics)
        self.assertFalse(hasattr(scene, "scene_expert_degraded_stage_reasons"))

    def test_preferred_maximum_overage_is_not_a_hard_stage_failure(self) -> None:
        class FakeScene:
            text_description = "bedroom"
            scene_expert_stage_budget = {"max_stage_regenerations": 1}
            scene_expert_min_output_objects = 1
            scene_expert_required_min_output_objects = 0
            scene_expert_max_output_objects = 3
            scene_expert_required_max_output_objects = 0

            def __init__(self) -> None:
                self.restore_calls = 0

            @staticmethod
            def to_state_dict() -> dict:
                return {"objects": {}}

            def restore_from_state_dict(self, state: dict) -> None:
                del state
                self.restore_calls += 1

            @staticmethod
            def get_objects_by_type(object_type: object) -> list:
                del object_type
                return [object(), object(), object(), object()]

        calls = {"run": 0, "prepare": 0}

        async def run_once() -> None:
            calls["run"] += 1

        async def prepare(reasons: list[str]) -> None:
            self.assertIn("preferred optional stage output exceeded", reasons[0])
            calls["prepare"] += 1

        scene = FakeScene()
        attempts = _run_sceneexpert_placement_stage(
            stage="wall_mounted",
            agent=SimpleNamespace(
                admitted_stage_assets=lambda: [],
                prepare_stage_regeneration=prepare,
            ),
            scene=scene,
            run_once=run_once,
        )

        self.assertEqual(1, attempts)
        self.assertEqual({"run": 2, "prepare": 1}, calls)
        self.assertEqual(1, scene.restore_calls)
        self.assertIn("wall_mounted", scene.scene_expert_stage_diagnostics)
        self.assertFalse(hasattr(scene, "scene_expert_degraded_stage_reasons"))

    def test_budget_exhausted_zero_output_still_gets_fresh_planner_retry(
        self,
    ) -> None:
        class FakeScene:
            text_description = "living room"
            scene_expert_stage_budget = {"max_stage_regenerations": 1}

            def __init__(self, scene_dir: Path) -> None:
                self.scene_dir = scene_dir
                self.room_id = "living_room"

            @staticmethod
            def to_state_dict() -> dict:
                return {"objects": {}}

            @staticmethod
            def restore_from_state_dict(state: dict) -> None:
                del state

            @staticmethod
            def content_hash() -> str:
                return "empty-stage"

        calls = {"run": 0, "prepare": 0}

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

        with TemporaryDirectory() as tmp:
            _run_sceneexpert_placement_stage(
                stage="wall_mounted",
                agent=SimpleNamespace(
                    _stage_runtime_exhausted=True,
                    _planner_budget_exhausted=True,
                    prepare_stage_regeneration=prepare,
                    admitted_stage_assets=lambda: [],
                    stage_working_memory=SimpleNamespace(scene_root_dir=Path(tmp)),
                ),
                scene=FakeScene(Path(tmp)),
                run_once=run_once,
            )

        self.assertEqual({"run": 2, "prepare": 1}, calls)

    def test_acquired_assets_get_placement_only_continuation_first(self) -> None:
        class FakeScene:
            text_description = "classroom"
            scene_expert_stage_budget = {"max_stage_regenerations": 1}

            def __init__(self) -> None:
                self.restore_calls = 0

            @staticmethod
            def to_state_dict() -> dict:
                return {"objects": {}}

            def restore_from_state_dict(self, state: dict) -> None:
                del state
                self.restore_calls += 1

        calls = {"run": 0, "placement": 0}

        async def run_once() -> None:
            calls["run"] += 1
            if calls["run"] == 1:
                raise StageValidationError(
                    stage="wall_mounted",
                    reasons=["missing required stage output: produced 0 objects"],
                )

        async def prepare_placement(reasons: list[str]) -> None:
            self.assertTrue(reasons)
            calls["placement"] += 1

        scene = FakeScene()
        attempts = _run_sceneexpert_placement_stage(
            stage="wall_mounted",
            agent=SimpleNamespace(
                admitted_stage_assets=lambda: [
                    SimpleNamespace(object_id="blackboard_asset")
                ],
                prepare_placement_continuation=prepare_placement,
            ),
            scene=scene,
            run_once=run_once,
        )

        self.assertEqual(0, attempts)
        self.assertEqual({"run": 2, "placement": 1}, calls)
        self.assertEqual(1, scene.restore_calls)

    def test_missing_required_asset_family_exports_degraded_scene(self) -> None:
        class FakeScene:
            text_description = "bedroom"
            scene_expert_stage_budget = {"max_stage_regenerations": 0}
            room_id = "bedroom"

            def __init__(self, scene_dir: Path) -> None:
                self.scene_dir = scene_dir

            @staticmethod
            def to_state_dict() -> dict:
                return {"objects": {"nightstand_0": {}}}

            @staticmethod
            def content_hash() -> str:
                return "partial-furniture-stage"

        agent = SimpleNamespace(
            admitted_stage_assets=lambda: [
                SimpleNamespace(object_id="nightstand_asset")
            ],
            unavailable_required_asset_families=lambda: ["bed"],
        )
        with self.assertRaises(StageValidationError):
            _raise_if_required_assets_unavailable(stage="furniture", agent=agent)

        with TemporaryDirectory() as tmp:
            scene = FakeScene(Path(tmp))

            async def run_once() -> None:
                return None

            _run_sceneexpert_placement_stage(
                stage="furniture",
                agent=SimpleNamespace(
                    **vars(agent),
                    stage_working_memory=SimpleNamespace(scene_root_dir=Path(tmp)),
                ),
                scene=scene,
                run_once=run_once,
            )

            manifest = (
                Path(tmp) / "scene_expert" / "degraded" / "degraded_manifest.json"
            ).read_text(encoding="utf-8")
            self.assertIn('"responsible_role": "asset"', manifest)
            self.assertIn(
                '"recommended_resume_action": "retry_stage_asset_acquisition"', manifest
            )
            self.assertIn('"unavailable_required_asset_families": [', manifest)
            self.assertIn('"bed"', manifest)

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
                        stage_working_memory=SimpleNamespace(scene_root_dir=Path(tmp)),
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
                    stage_working_memory=SimpleNamespace(scene_root_dir=Path(tmp)),
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
