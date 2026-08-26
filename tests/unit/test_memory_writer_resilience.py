import json
import unittest

from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import ValidationError

from scenesmith.scene_expert.memory.schemas import (
    FailureMemoryCandidate,
    MemoryWriterResponse,
    SuccessMemoryCandidate,
)
from scenesmith.scene_expert.memory.writer import MemoryWriter
from scenesmith.scene_expert.schemas import FullVerifyReport
from scenesmith.scene_expert.structured_llm import StructuredLLMResult
from scenesmith.scene_expert.trace_logger import TraceLogger


def _full_report(*, passed: bool = True) -> FullVerifyReport:
    return FullVerifyReport(
        overall_score=0.86 if passed else 0.4,
        pass_scene=passed,
        deterministic_pass=passed,
    )


def _evidence(
    *,
    stage_passed: bool = True,
    hard_failure: bool = False,
    repair_verified: bool = False,
) -> dict:
    return {
        "trace_id": "trace_000001",
        "scene_id": "scene_001",
        "prompt": "A modern bedroom with a bed and two nightstands.",
        "task_spec": {
            "room_type": "bedroom",
            "style": "modern",
            "required_large_objects": ["bed", "nightstand"],
            "functional_zones": ["sleeping_zone"],
        },
        "stages": [
            {
                "stage": "furniture",
                "scene_state_path": "/tmp/scene/final_furniture",
                "relation_context": {
                    "stage": "furniture",
                    "hard_constraints": [
                        {
                            "constraint_id": "face-desk",
                            "relation_type": "facing",
                            "subject": {"role": "student chair"},
                            "target": {"role": "student desk"},
                        }
                    ],
                },
                "verify_report": {
                    "stage": "furniture",
                    "pass_stage": stage_passed,
                    "visual_scores": {"semantic": 0.9, "plausibility": 0.8},
                    "score_source": "scores.yaml",
                    "critique_summary": (
                        "DETERMINISTIC HARD FAILURE: missing required bed"
                        if hard_failure
                        else "The bed and nightstands form a usable sleeping zone."
                    ),
                    "issues": (
                        [
                            {
                                "issue_type": "missing_required_object",
                                "object_name": "bed",
                                "description": "missing required bed",
                            }
                        ]
                        if hard_failure
                        else []
                    ),
                    "hard_check_report": (
                        {"hard_valid": False, "failed_checks": ["required bed"]}
                        if hard_failure
                        else {"hard_valid": True}
                    ),
                },
                "repair_actions": (
                    [
                        {
                            "repair_type": "local_repair",
                            "repair_action": "add the missing bed",
                            "repair_verified": True,
                        }
                    ]
                    if repair_verified
                    else []
                ),
            }
        ],
    }


class _FakeStructuredClient:
    def __init__(self, result: StructuredLLMResult) -> None:
        self.result = result
        self.calls: list[dict] = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


class MemoryWriterResilienceTest(unittest.TestCase):
    def test_trace_evidence_has_globally_scoped_run_locator(self) -> None:
        with TemporaryDirectory() as temp_dir:
            logger = TraceLogger(
                output_dir=temp_dir,
                scene_index=1,
                prompt="bedroom",
                experiment_name="ablation_4c",
                config_hash="config_123",
                task_spec={"room_type": "bedroom", "style": "modern"},
            )
            evidence = logger.build_memory_writer_evidence()

        self.assertEqual(str(Path(temp_dir).resolve()), evidence["run_id"])
        self.assertEqual("ablation_4c", evidence["experiment_name"])
        self.assertEqual("config_123", evidence["config_hash"])

    def test_strict_candidate_is_promoted_with_deterministic_metadata(self) -> None:
        response = MemoryWriterResponse(
            success_cases=[
                SuccessMemoryCandidate(
                    stage="furniture",
                    successful_pattern=[
                        "Anchor the bed first, then balance nightstands on both sides."
                    ],
                    positive_guidance=[
                        "Preserve a clear approach path to both sides of the bed."
                    ],
                )
            ]
        )
        client = _FakeStructuredClient(StructuredLLMResult(value=response))
        writer = MemoryWriter(model="qwen", llm_client=client)

        ops = writer.write(
            "Trace: trace_000001\nPrompt: bedroom",
            _full_report(),
            evidence_payload=_evidence(),
        )

        self.assertEqual(1, len(ops))
        content = ops[0].content
        self.assertEqual("active", content["status"])
        self.assertEqual("llm", content["source"])
        self.assertEqual("bedroom", content["room_type"])
        self.assertEqual(["bed", "nightstand"], content["required_objects"])
        self.assertTrue(content["source_run_id"].startswith("run_"))
        self.assertIn(content["source_run_id"], content["evidence_refs"])
        self.assertIn("score_source=scores.yaml", content["critic_evidence"])
        self.assertEqual("trace_000001", content["provenance"]["trace_id"])
        self.assertEqual("facing", content["spatial_relations"][0]["relation_type"])
        self.assertTrue(content["embedding_text"])
        self.assertFalse(writer.last_trace["fallback_written"])
        self.assertIs(client.calls[0]["response_model"], MemoryWriterResponse)
        self.assertEqual("json_schema", client.calls[0]["profile"].response_format)
        self.assertEqual("none", client.calls[0]["profile"].thinking_mode)

    def test_model_failure_never_writes_fallback_memory(self) -> None:
        client = _FakeStructuredClient(
            StructuredLLMResult(
                final_error_kind="schema_validation",
                final_error="missing required field",
            )
        )
        with TemporaryDirectory() as temp_dir:
            writer = MemoryWriter(
                model="qwen",
                llm_client=client,
                debug_dir=temp_dir,
            )
            ops = writer.write(
                "Trace: trace_000001\nPrompt: bedroom",
                _full_report(),
                evidence_payload=_evidence(),
            )
            debug = json.loads(
                (Path(temp_dir) / "memory_writer_debug.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual([], ops)
        self.assertEqual("model_failure_no_write", writer.last_trace["write_status"])
        self.assertEqual("no_write", writer.last_trace["source"])
        self.assertFalse(debug["fallback_written"])
        self.assertEqual([], debug["result_ops"])

    def test_llm_cannot_self_certify_a_deterministic_failure(self) -> None:
        response = MemoryWriterResponse(
            failure_cases=[
                FailureMemoryCandidate(
                    stage="furniture",
                    failure_type="collision",
                    bad_pattern="The chair may collide with the desk.",
                    failure_reason="Visual critic suggested an overlap.",
                    repair_action="Move the chair.",
                    is_deterministic=True,
                )
            ]
        )
        writer = MemoryWriter(
            model="qwen",
            llm_client=_FakeStructuredClient(StructuredLLMResult(value=response)),
        )
        evidence = _evidence(hard_failure=False)
        evidence["stages"][0]["verify_report"][
            "critique_summary"
        ] = "No deterministic hard failure was found; only a possible visual overlap."

        ops = writer.write(
            "Trace: trace_000001",
            _full_report(),
            evidence_payload=evidence,
        )

        self.assertEqual([], ops)
        self.assertEqual("no_valid_candidates", writer.last_trace["write_status"])

    def test_verified_repair_from_trace_promotes_failure(self) -> None:
        response = MemoryWriterResponse(
            failure_cases=[
                FailureMemoryCandidate(
                    stage="furniture",
                    object="bed",
                    failure_type="missing_required_object",
                    bad_pattern="The required bed was absent.",
                    failure_reason="The asset selection omitted the required bed.",
                    repair_action="Add the bed and rerun hard checks.",
                )
            ]
        )
        writer = MemoryWriter(
            model="qwen",
            llm_client=_FakeStructuredClient(StructuredLLMResult(value=response)),
        )

        ops = writer.write(
            "Trace: trace_000001",
            _full_report(passed=False),
            evidence_payload=_evidence(
                stage_passed=False,
                hard_failure=True,
                repair_verified=True,
            ),
        )

        self.assertEqual(1, len(ops))
        self.assertTrue(ops[0].content["repair_verified"])
        self.assertTrue(ops[0].content["is_deterministic"])
        self.assertIn("is_deterministic=true", ops[0].content["embedding_text"])

    def test_candidate_schema_rejects_uncontracted_fields(self) -> None:
        with self.assertRaises(ValidationError):
            SuccessMemoryCandidate.model_validate(
                {
                    "stage": "furniture",
                    "successful_pattern": ["valid lesson"],
                    "invented_score": 0.99,
                }
            )


if __name__ == "__main__":
    unittest.main()
