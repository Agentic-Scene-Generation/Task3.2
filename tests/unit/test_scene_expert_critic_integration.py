import json
import unittest

from pathlib import Path
from tempfile import TemporaryDirectory

from scenesmith.scene_expert.critic_bridge import (
    SCENEBENCHMARK_REPORT_SCHEMA,
    SceneBenchmarkCriticBridge,
)
from scenesmith.scene_expert.critic_memory import build_critic_memory_records
from scenesmith.scene_expert.schemas import (
    CriticEvidence,
    CriticEvidenceResult,
    SceneTaskSpec,
)
from scenesmith.scene_expert.verifier import StageVerifier


def _payload(*, schema_version: str = SCENEBENCHMARK_REPORT_SCHEMA) -> dict:
    return {
        "schema_version": schema_version,
        "scope": "room:room_0",
        "stage": "scene_expert_post_furniture",
        "results": [
            {
                "check_id": "access__chair_0",
                "metric": "spatial_accessibility",
                "label": "fail",
                "scoring_tier": "core",
                "primary_object": "chair_0",
                "related_objects": ["desk_0"],
                "reason": "The chair access zone is blocked.",
                "repair_advice": "Move chair_0 clear of the desk access corridor.",
                "confidence": 0.97,
            },
            {
                "check_id": "shadow__desk_0",
                "metric": "functional_dependency",
                "label": "fail",
                "scoring_tier": "auxiliary",
                "primary_object": "desk_0",
                "reason": "Prompt contract shadow observation.",
                "confidence": 0.70,
            },
        ],
        "summary": {
            "scene_summary": {
                "total_checks": 1,
                "pass": 0,
                "degraded": 0,
                "fail": 1,
                "unknown": 0,
                "score": 0.0,
            },
            "metric_summary": {
                "spatial_accessibility": {"score": 0.0},
                "functional_dependency": {"score": None},
            },
        },
        "gate": {
            "enabled": True,
            "blocked": True,
            "label": "fail",
        },
    }


class SceneExpertCriticBridgeTest(unittest.TestCase):
    def test_bridge_persists_and_normalizes_versioned_report(self) -> None:
        calls = []

        def evaluator(scene, *, config, stage):
            calls.append((scene, config, stage))
            return _payload()

        with TemporaryDirectory() as tmp:
            config = {"experiment": {"scenebenchmark_critic": {"enabled": True}}}
            bridge = SceneBenchmarkCriticBridge(
                enabled=True,
                critic_config=config,
                output_dir=tmp,
                evaluator=evaluator,
            )

            evidence = bridge.collect_room_stage(object(), "furniture")

            self.assertTrue(evidence.available)
            self.assertEqual(
                SCENEBENCHMARK_REPORT_SCHEMA, evidence.provider_schema_version
            )
            self.assertEqual(1, len(evidence.core_failures))
            self.assertEqual("chair_0", evidence.core_failures[0].primary_object)
            self.assertEqual(0.0, evidence.metric_scores["spatial_accessibility"])
            self.assertEqual(
                "scene_expert_post_furniture",
                calls[0][2],
            )
            report_path = Path(evidence.report_path)
            self.assertTrue(report_path.exists())
            self.assertEqual(_payload(), json.loads(report_path.read_text("utf-8")))

    def test_bridge_rejects_unknown_provider_schema_nonfatally(self) -> None:
        with TemporaryDirectory() as tmp:
            bridge = SceneBenchmarkCriticBridge(
                enabled=True,
                critic_config={},
                output_dir=tmp,
                evaluator=lambda *args, **kwargs: _payload(
                    schema_version="scenesmith.scenebenchmark_critic.report.v2"
                ),
            )

            evidence = bridge.collect_room_stage(object(), "furniture")

            self.assertFalse(evidence.available)
            self.assertEqual("error", evidence.status)
            self.assertIn("Unsupported SceneBenchmark", evidence.error)
            self.assertTrue(Path(evidence.report_path).exists())


class SceneExpertCriticVerifierTest(unittest.TestCase):
    def test_evidence_is_advisory_when_sceneexpert_gate_is_off(self) -> None:
        bridge = SceneBenchmarkCriticBridge(
            enabled=True,
            critic_config={},
            output_dir="unused",
            persist_raw_reports=False,
            evaluator=lambda *args, **kwargs: _payload(),
        )
        evidence = bridge.collect_room_stage(object(), "furniture")

        report = StageVerifier(critic_gate_enabled=False).verify(
            stage="furniture",
            stage_output_dir="",
            task_spec=SceneTaskSpec(room_type="study", style="standard"),
            critic_evidence=evidence,
        )

        self.assertTrue(report.pass_stage)
        self.assertEqual(
            0.0,
            report.rule_scores["critic.spatial_accessibility"],
        )
        self.assertIn("Move chair_0", report.repair_suggestions[0])

    def test_stage_blocks_only_when_both_gates_authorize_it(self) -> None:
        evidence = CriticEvidence(
            status="ok",
            available=True,
            stage="furniture",
            gate_enabled=True,
            gate_blocked=True,
            core_fail_count=1,
            results=[
                CriticEvidenceResult(
                    check_id="access__chair_0",
                    metric="spatial_accessibility",
                    label="fail",
                    scoring_tier="core",
                    primary_object="chair_0",
                )
            ],
        )

        report = StageVerifier(critic_gate_enabled=True).verify(
            stage="furniture",
            stage_output_dir="",
            task_spec=SceneTaskSpec(room_type="study", style="standard"),
            critic_evidence=evidence,
        )

        self.assertFalse(report.pass_stage)
        self.assertEqual(
            "scenebenchmark_critic_gate",
            report.issues[0].issue_type,
        )

        evidence.gate_enabled = False
        advisory_report = StageVerifier(critic_gate_enabled=True).verify(
            stage="furniture",
            stage_output_dir="",
            task_spec=SceneTaskSpec(room_type="study", style="standard"),
            critic_evidence=evidence,
        )
        self.assertTrue(advisory_report.pass_stage)

    def test_required_evidence_applies_only_to_configured_room_stages(self) -> None:
        verifier = StageVerifier(
            critic_required=True,
            critic_required_stages=["furniture"],
        )

        floor_report = verifier.verify(
            stage="floor_plan",
            stage_output_dir="",
            task_spec=SceneTaskSpec(room_type="study", style="standard"),
        )
        furniture_report = verifier.verify(
            stage="furniture",
            stage_output_dir="",
            task_spec=SceneTaskSpec(room_type="study", style="standard"),
        )

        self.assertTrue(floor_report.pass_stage)
        self.assertFalse(furniture_report.pass_stage)
        self.assertEqual(
            "scenebenchmark_critic_unavailable",
            furniture_report.issues[0].issue_type,
        )


class SceneExpertCriticMemoryTest(unittest.TestCase):
    def test_only_core_failures_become_durable_negative_memory(self) -> None:
        bridge = SceneBenchmarkCriticBridge(
            enabled=True,
            critic_config={},
            output_dir="unused",
            persist_raw_reports=False,
            evaluator=lambda *args, **kwargs: _payload(),
        )
        evidence = bridge.collect_room_stage(object(), "furniture")

        failures, success = build_critic_memory_records(
            evidence=evidence,
            task_spec=SceneTaskSpec(
                room_type="study",
                style="modern",
                required_large_objects=["desk", "chair"],
            ),
            stage="furniture",
            required_objects=["desk", "chair"],
            trace_ref="trace_000001",
            created_at="2026-07-28T00:00:00Z",
        )

        self.assertIsNone(success)
        self.assertEqual(1, len(failures))
        self.assertEqual("chair_0", failures[0].object)
        self.assertTrue(failures[0].is_deterministic)
        self.assertIn("access__chair_0", failures[0].critic_check)
        self.assertNotIn("shadow", failures[0].failure_id)

    def test_clean_core_report_creates_positive_memory(self) -> None:
        evidence = CriticEvidence(
            status="ok",
            available=True,
            stage="furniture",
            scene_score=0.9,
            core_total_checks=2,
            core_pass_count=2,
            metric_scores={"spatial_accessibility": 1.0},
            results=[
                CriticEvidenceResult(
                    check_id="access__chair_0",
                    metric="spatial_accessibility",
                    label="pass",
                    scoring_tier="core",
                )
            ],
        )

        failures, success = build_critic_memory_records(
            evidence=evidence,
            task_spec=SceneTaskSpec(room_type="study", style="modern"),
            stage="furniture",
            required_objects=["desk", "chair"],
            trace_ref="trace_000002",
            created_at="2026-07-28T00:00:00Z",
        )

        self.assertEqual([], failures)
        self.assertIsNotNone(success)
        self.assertEqual(0.9, success.quality_score)
        self.assertEqual(1.0, success.scores["critic.spatial_accessibility"])
        self.assertTrue(success.embedding_text)


if __name__ == "__main__":
    unittest.main()
