import json
import unittest
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from pydantic import BaseModel

from scenesmith.scene_expert.global_planner import GlobalPlanner, _format_task_spec
from scenesmith.scene_expert.schemas import (
    FullVerifyReport,
    HarnessContext,
    MemoryPack,
    SceneTaskSpec,
    StageExecutionEvidence,
)
from scenesmith.scene_expert.structured_llm import (
    SceneExpertStructuredLLMClient,
    StructuredLLMProfile,
)
from scenesmith.scene_expert.trace_logger import TraceLogger


class _Payload(BaseModel):
    value: str


def _response(
    *,
    content: str | None,
    reasoning: str = "",
    finish_reason: str = "stop",
):
    message = SimpleNamespace(
        content=content,
        reasoning_content=reasoning,
        model_extra={},
    )
    usage = SimpleNamespace(
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        completion_tokens_details=None,
    )
    return SimpleNamespace(
        id="request-test",
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
        usage=usage,
    )


class _FakeOpenAI:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []
        self.chat = SimpleNamespace(completions=self)

    def with_options(self, **kwargs):
        self.options = kwargs
        return self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class SceneExpertStructuredLLMTest(unittest.TestCase):
    def _client(self, outcomes, profile=None):
        fake = _FakeOpenAI(outcomes)
        active = profile or StructuredLLMProfile(
            thinking_mode="none",
            max_tokens=128,
            retry_max_tokens=256,
            max_attempts=2,
            response_format="json_schema",
        )
        client = SceneExpertStructuredLLMClient(
            model="local-qwen",
            client=fake,
            profiles={"test": active},
        )
        return client, fake, active

    def test_no_think_is_sent_in_template_kwargs_and_prompt(self):
        client, fake, profile = self._client(
            [_response(content=json.dumps({"value": "ok"}))]
        )

        result = client.complete(
            role="test",
            stage="startup",
            event="smoke",
            messages=[{"role": "user", "content": "Return JSON."}],
            response_model=_Payload,
            profile=profile,
        )

        self.assertTrue(result.success)
        call = fake.calls[0]
        self.assertFalse(
            call["extra_body"]["chat_template_kwargs"]["enable_thinking"]
        )
        self.assertTrue(call["messages"][0]["content"].startswith("/no_think\n"))
        self.assertEqual("json_schema", call["response_format"]["type"])

    def test_reasoning_only_response_retries_without_parsing_reasoning(self):
        profile = StructuredLLMProfile(
            thinking_mode="low",
            max_tokens=128,
            retry_max_tokens=256,
            max_attempts=2,
            response_format="json_schema",
        )
        client, fake, _ = self._client(
            [
                _response(content=None, reasoning='{"value":"wrong-source"}'),
                _response(content='{"value":"final-content"}'),
            ],
            profile,
        )

        result = client.complete(
            role="test",
            stage="furniture",
            event="brief",
            messages=[{"role": "user", "content": "Return JSON."}],
            response_model=_Payload,
            profile=profile,
        )

        self.assertEqual("final-content", result.value.value)
        self.assertEqual("reasoning_only", result.attempts[0].error_kind)
        self.assertTrue(
            fake.calls[0]["extra_body"]["chat_template_kwargs"]["enable_thinking"]
        )
        self.assertFalse(
            fake.calls[1]["extra_body"]["chat_template_kwargs"]["enable_thinking"]
        )

    def test_length_retry_is_bounded_and_uses_retry_budget(self):
        client, fake, profile = self._client(
            [
                _response(content='{"value":', finish_reason="length"),
                _response(content='{"value":"ok"}'),
            ]
        )

        result = client.complete(
            role="test",
            stage="furniture",
            event="brief",
            messages=[{"role": "user", "content": "Return JSON."}],
            response_model=_Payload,
            profile=profile,
        )

        self.assertTrue(result.success)
        self.assertEqual(2, len(fake.calls))
        self.assertEqual(128, fake.calls[0]["max_tokens"])
        self.assertEqual(256, fake.calls[1]["max_tokens"])
        self.assertEqual("length", result.attempts[0].error_kind)

    def test_repeated_reasoning_only_returns_typed_failure(self):
        client, fake, profile = self._client(
            [
                _response(content=None, reasoning='{"value":"not-content"}'),
                _response(content=None, reasoning='{"value":"still-not-content"}'),
            ]
        )

        result = client.complete(
            role="test",
            stage="furniture",
            event="brief",
            messages=[{"role": "user", "content": "Return JSON."}],
            response_model=_Payload,
            profile=profile,
        )

        self.assertFalse(result.success)
        self.assertEqual(2, len(fake.calls))
        self.assertEqual("reasoning_only", result.final_error_kind)

    def test_bad_request_downgrades_json_schema_once(self):
        bad_request = type("BadRequestError", (Exception,), {})
        client, fake, profile = self._client(
            [bad_request("json_schema unsupported"), _response(content='{"value":"ok"}')]
        )

        result = client.complete(
            role="test",
            stage="startup",
            event="smoke",
            messages=[{"role": "user", "content": "Return JSON."}],
            response_model=_Payload,
            profile=profile,
        )

        self.assertTrue(result.success)
        self.assertEqual("json_schema", fake.calls[0]["response_format"]["type"])
        self.assertEqual("json_object", fake.calls[1]["response_format"]["type"])

    def test_memory_pack_deduplicates_prompt_content_and_ids(self):
        pack = MemoryPack(
            success_hints=["Keep clearance", " Keep   clearance "],
            failure_hints=["Avoid collision", "avoid collision"],
            skill_texts=["Anchor bed", "Anchor bed"],
            success_case_ids=["success_1", "success_1"],
        ).deduplicated()

        self.assertEqual(["Keep clearance"], pack.success_hints)
        self.assertEqual(["Avoid collision"], pack.failure_hints)
        self.assertEqual(["Anchor bed"], pack.skill_texts)
        self.assertEqual(["success_1"], pack.success_case_ids)

    def test_floor_plan_context_marks_furniture_as_downstream_capacity(self):
        text = _format_task_spec(
            SceneTaskSpec(
                room_type="bedroom",
                style="modern",
                required_large_objects=["bed", "wardrobe"],
            ),
            "floor_plan",
        )

        self.assertIn("Downstream furniture capacity requirements", text)
        self.assertIn("do not place these objects in floor_plan", text)
        self.assertNotIn("Required objects for this stage", text)

        fallback = GlobalPlanner.__new__(GlobalPlanner)._fallback_brief(
            HarnessContext(
                stage="floor_plan",
                task_spec=SceneTaskSpec(
                    room_type="bedroom",
                    style="modern",
                    required_large_objects=["bed", "wardrobe"],
                ),
                memory_pack=MemoryPack(),
            )
        )
        fallback_text = " ".join(fallback.constraints_for_designer)
        self.assertIn("Do not place furniture during floor_plan", fallback_text)
        self.assertNotIn("Ensure these objects are present", fallback_text)

    def test_trace_exposes_fallback_and_stage_injection_evidence(self):
        with TemporaryDirectory() as tmp:
            logger = TraceLogger(
                output_dir=tmp,
                scene_index=0,
                prompt="A bedroom with a bed.",
                task_spec={"room_type": "bedroom"},
                task_spec_status={"source": "fallback", "degraded": True},
            )
            logger.log_stage(
                stage="furniture",
                memory_pack=MemoryPack(success_case_ids=["success_1"]),
                stage_brief=None,
                scene_state_path="scene_states/furniture",
                verify_report=None,
                repair_actions=[],
                execution_evidence=StageExecutionEvidence(
                    task_spec_source="fallback",
                    stage_brief_source="llm",
                    retrieved_memory_ids=["success_1"],
                    designer_prompt_contains_brief=True,
                    degraded=True,
                ),
            )
            trace = logger.finalize(FullVerifyReport(), exports={})

        self.assertTrue(trace["degraded"])
        self.assertIn("task_compiler", trace["degraded_components"])
        self.assertEqual("bedroom", trace["task_spec"]["room_type"])
        evidence = trace["stages"][0]["execution_evidence"]
        self.assertTrue(evidence["designer_prompt_contains_brief"])
        self.assertEqual(["success_1"], evidence["retrieved_memory_ids"])


if __name__ == "__main__":
    unittest.main()
