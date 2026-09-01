import json
import unittest

from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import ValidationError

from scenesmith.scene_expert.memory.schemas import (
    FailureMemoryCandidate,
    MemoryWriterResponse,
    SkillMemoryCandidate,
    SuccessMemoryCandidate,
)
from scenesmith.scene_expert.memory.injection import build_memory_injection_bundle
from scenesmith.scene_expert.memory.retriever import MemoryRetriever
from scenesmith.scene_expert.memory.store import FastMemoryStore
from scenesmith.scene_expert.memory.writer import MemoryWriter
from scenesmith.scene_expert.schemas import (
    FullVerifyReport,
    SceneTaskSpec,
    StageBrief,
    StageRelationContext,
)
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
    stage_agent_invoked: bool = False,
    prompt: str = "A modern bedroom with a bed and two nightstands.",
    relation_subject_count: int | None = None,
) -> dict:
    return {
        "trace_id": "trace_000001",
        "scene_id": "scene_001",
        "prompt": prompt,
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
                "execution_evidence": {
                    "stage_agent_invoked": stage_agent_invoked,
                    "required_objects": ["bed", "nightstand"],
                },
                "relation_context": {
                    "stage": "furniture",
                    "hard_constraints": [
                        {
                            "constraint_id": "face-desk",
                            "relation": "facing",
                            "subjects": {
                                "role": "student chair",
                                **(
                                    {"count": relation_subject_count}
                                    if relation_subject_count is not None
                                    else {}
                                ),
                            },
                            "targets": {"role": "student desk"},
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
        self.assertEqual("scene", content["promotion_scope"])
        self.assertTrue(content["source_scene_passed"])
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
        self.assertEqual(
            {"success_case": 1, "failure_case": 0, "skill": 0},
            writer.last_trace["promoted_counts"],
        )

    def test_degraded_scene_promotes_only_verified_stage_local_success(self) -> None:
        response = MemoryWriterResponse(
            success_cases=[
                SuccessMemoryCandidate(
                    stage="furniture",
                    successful_pattern=[
                        "Anchor the required desk before arranging nearby chairs."
                    ],
                    positive_guidance=[
                        "Keep chair orientation aligned with the desk work surface."
                    ],
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
            evidence_payload=_evidence(stage_passed=True),
        )

        self.assertEqual(1, len(ops))
        content = ops[0].content
        self.assertEqual("stage", content["promotion_scope"])
        self.assertFalse(content["source_scene_passed"])
        self.assertAlmostEqual(0.85, content["quality_score"])
        self.assertLessEqual(content["confidence"], 0.75)
        self.assertIn("promotion_scope=stage", content["embedding_text"])

    def test_degraded_scene_rejects_success_from_failed_exact_stage(self) -> None:
        response = MemoryWriterResponse(
            success_cases=[
                SuccessMemoryCandidate(
                    stage="furniture",
                    successful_pattern=["Place the bed before secondary furniture."],
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
            evidence_payload=_evidence(stage_passed=False),
        )

        self.assertEqual([], ops)

    def test_degraded_scene_persists_verified_stage_skill_as_candidate(self) -> None:
        response = MemoryWriterResponse(
            skills=[
                SkillMemoryCandidate(
                    skill_name="align_seating_to_work_surface",
                    stage="furniture",
                    preconditions=["A desk and its assigned chair are present."],
                    procedure=[
                        "Bind each chair to its intended desk before placement.",
                        "Orient the chair toward the desk and verify approach clearance.",
                    ],
                    failure_avoidance=["Do not infer alignment from proximity alone."],
                    postconditions=["Every assigned chair faces its desk."],
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
            evidence_payload=_evidence(stage_passed=True),
        )

        self.assertEqual(1, len(ops))
        skill = ops[0].content
        self.assertEqual("skill", ops[0].memory_type)
        self.assertEqual("candidate", skill["status"])
        self.assertEqual("stage", skill["promotion_scope"])
        self.assertFalse(skill["source_scene_passed"])
        self.assertTrue(
            skill["semantic_signature"].startswith("sceneexpert.skill_signature.v1:")
        )
        self.assertEqual(1, skill["independent_support_count"])
        self.assertEqual(2, skill["activation_min_independent_support"])
        self.assertEqual(1, writer.last_trace["llm_skill_candidate_count"])
        self.assertEqual(1, writer.last_trace["skill_persisted_candidate_count"])
        self.assertEqual(0, writer.last_trace["skill_promoted_active_count"])
        self.assertEqual(0, writer.last_trace["skill_rejected_count"])
        self.assertEqual("persisted_candidate", writer.last_trace["write_status"])
        self.assertEqual(1, writer.last_trace["persisted_count"])
        self.assertEqual(0, writer.last_trace["promoted_count"])

    def test_failed_exact_stage_skill_is_rejected_with_structured_reasons(self) -> None:
        response = MemoryWriterResponse(
            skills=[
                SkillMemoryCandidate(
                    skill_name="verify_support_surface_alignment",
                    stage="furniture",
                    procedure=[
                        "Check support overlap before accepting placement.",
                        "Move unsupported objects onto a verified support region.",
                    ],
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
            evidence_payload=_evidence(stage_passed=False, hard_failure=True),
        )

        self.assertEqual([], ops)
        self.assertEqual(1, writer.last_trace["llm_skill_candidate_count"])
        self.assertEqual(1, writer.last_trace["skill_rejected_count"])
        self.assertEqual(
            {
                "scene_gate_failed": 1,
                "score_gate_failed": 1,
                "stage_gate_failed": 1,
                "deterministic_stage_failure": 1,
            },
            writer.last_trace["skill_rejection_reasons"],
        )
        self.assertEqual(
            "rejected", writer.last_trace["skill_decisions"][0]["decision"]
        )

    def test_fully_verified_skill_keeps_direct_active_promotion(self) -> None:
        response = MemoryWriterResponse(
            skills=[
                SkillMemoryCandidate(
                    skill_name="align_seating_to_work_surface",
                    stage="furniture",
                    procedure=[
                        "Bind each chair to its intended desk.",
                        "Orient every chair toward the assigned work surface.",
                    ],
                )
            ]
        )
        writer = MemoryWriter(
            model="qwen",
            llm_client=_FakeStructuredClient(StructuredLLMResult(value=response)),
        )

        ops = writer.write(
            "Trace: trace_000001",
            _full_report(passed=True),
            evidence_payload=_evidence(stage_passed=True),
        )

        self.assertEqual("active", ops[0].content["status"])
        self.assertEqual("scene", ops[0].content["promotion_scope"])
        self.assertTrue(ops[0].content["source_scene_passed"])
        self.assertEqual(
            "scene_and_stage_verified", ops[0].content["activation_reason"]
        )
        self.assertEqual(1, writer.last_trace["skill_promoted_active_count"])

    def test_verified_passing_stage_bootstraps_candidate_when_llm_omits_skill(
        self,
    ) -> None:
        response = MemoryWriterResponse(noop_reason="No model-authored Skill proposed.")
        writer = MemoryWriter(
            model="qwen",
            llm_client=_FakeStructuredClient(StructuredLLMResult(value=response)),
        )

        ops = writer.write(
            "Trace: trace_000001",
            _full_report(passed=True),
            evidence_payload=_evidence(stage_agent_invoked=True),
        )

        self.assertEqual(2, len(ops))
        self.assertTrue(all(op.memory_type == "skill" for op in ops))
        skill = next(
            op.content
            for op in ops
            if op.content["applicability"]["required_relation_types"] == ["facing"]
        )
        self.assertEqual("candidate", skill["status"])
        self.assertEqual("deterministic", skill["source"])
        self.assertEqual("stage", skill["promotion_scope"])
        self.assertTrue(skill["source_scene_passed"])
        self.assertGreaterEqual(len(skill["procedure"]), 2)
        self.assertEqual("facing", skill["spatial_relations"][0]["relation_type"])
        self.assertEqual(
            "verified_stage_bootstrap_awaiting_independent_support",
            skill["activation_reason"],
        )
        self.assertEqual(0, writer.last_trace["llm_skill_candidate_count"])
        self.assertEqual(2, writer.last_trace["bootstrap_skill_candidate_count"])
        self.assertEqual(
            2, writer.last_trace["bootstrap_skill_persisted_candidate_count"]
        )
        self.assertEqual(0, writer.last_trace["bootstrap_skill_rejected_count"])
        self.assertEqual(2, writer.last_trace["skill_persisted_candidate_count"])
        self.assertEqual(0, writer.last_trace["skill_promoted_active_count"])

    def test_grounded_bootstrap_survives_writer_model_failure_without_fallback(
        self,
    ) -> None:
        writer = MemoryWriter(
            model="qwen",
            llm_client=_FakeStructuredClient(
                StructuredLLMResult(
                    final_error_kind="schema_validation",
                    final_error="missing required field",
                )
            ),
        )

        ops = writer.write(
            "Trace: trace_000001",
            _full_report(passed=True),
            evidence_payload=_evidence(stage_agent_invoked=True),
        )

        self.assertEqual(2, len(ops))
        self.assertTrue(
            all(op.content["source"] == "deterministic" for op in ops)
        )
        self.assertTrue(all(op.content["status"] == "candidate" for op in ops))
        self.assertEqual("persisted_candidate", writer.last_trace["write_status"])
        self.assertEqual(
            "deterministic_skill_bootstrap", writer.last_trace["source"]
        )
        self.assertTrue(writer.last_trace["degraded"])
        self.assertFalse(writer.last_trace["fallback_written"])

    def test_bootstrap_never_self_confirms_failed_or_planner_only_evidence(
        self,
    ) -> None:
        response = MemoryWriterResponse(noop_reason="No model-authored Skill proposed.")
        for evidence in (
            _evidence(stage_passed=False, stage_agent_invoked=True),
            _evidence(stage_passed=True, stage_agent_invoked=True),
        ):
            if evidence["stages"][0]["verify_report"]["pass_stage"]:
                evidence["task_spec"]["required_large_objects"] = []
                evidence["stages"][0]["execution_evidence"]["required_objects"] = []
                evidence["stages"][0]["relation_context"]["hard_constraints"] = []
                evidence["stages"][0]["stage_brief"] = {
                    "recommended_skills": ["unverified_planner_skill"],
                    "constraints_for_designer": ["Unverified planner-only advice."],
                }
            writer = MemoryWriter(
                model="qwen",
                llm_client=_FakeStructuredClient(
                    StructuredLLMResult(value=response)
                ),
            )

            ops = writer.write(
                "Trace: trace_000001",
                _full_report(passed=True),
                evidence_payload=evidence,
            )

            self.assertEqual([], ops)
            self.assertEqual(0, writer.last_trace["bootstrap_skill_candidate_count"])
            self.assertEqual("no_valid_candidates", writer.last_trace["write_status"])

    def test_two_independent_bootstraps_promote_retrieve_and_inject_skill(self) -> None:
        response = MemoryWriterResponse(noop_reason="No model-authored Skill proposed.")
        with TemporaryDirectory() as temp_dir:
            store = FastMemoryStore(temp_dir)
            for prompt, subject_count in (
                ("Bedroom alpha with a bed and two nightstands.", 2),
                ("Bedroom beta with a bed and two nightstands.", 4),
            ):
                writer = MemoryWriter(
                    model="qwen",
                    llm_client=_FakeStructuredClient(
                        StructuredLLMResult(value=response)
                    ),
                )
                ops = writer.write(
                    "Trace: trace_000001",
                    _full_report(passed=True),
                    evidence_payload=_evidence(
                        stage_agent_invoked=True,
                        prompt=prompt,
                        relation_subject_count=subject_count,
                    ),
                )
                store.apply_updates(ops)

            self.assertEqual(2, len(store.skills))
            self.assertTrue(all(skill.status == "active" for skill in store.skills))
            self.assertTrue(
                all(skill.independent_support_count == 2 for skill in store.skills)
            )
            self.assertTrue(
                all(
                    skill.activation_reason
                    == "independent_stage_support_threshold_met"
                    for skill in store.skills
                )
            )
            skill = next(
                item
                for item in store.skills
                if item.applicability.required_relation_types == ["facing"]
            )

            task_spec = SceneTaskSpec.model_validate(_evidence()["task_spec"])
            relation_context = StageRelationContext.model_validate(
                _evidence(relation_subject_count=6)["stages"][0][
                    "relation_context"
                ]
            )
            pack = MemoryRetriever(store, max_skills=2).retrieve(
                task_spec,
                "furniture",
                relation_context=relation_context,
            )
            self.assertIn(skill.skill_name, pack.skill_names)
            bundle = build_memory_injection_bundle(
                stage="furniture",
                stage_brief=StageBrief(
                    stage="furniture",
                    stage_objective="Build a grounded bedroom layout.",
                    recommended_skills=pack.skill_names,
                ),
                memory_pack=pack,
            )

        self.assertIn(skill.skill_name, bundle.prompt_delivered_skill_names)
        self.assertEqual(1, bundle.final_text.count(f"[Skill: {skill.skill_name}]"))
        self.assertIn(skill.procedure[0], bundle.final_text)
        self.assertIn(skill.postconditions[0], bundle.final_text)
        self.assertNotIn("subject_count=", bundle.final_text)

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
