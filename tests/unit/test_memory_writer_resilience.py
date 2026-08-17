import json
import types
import unittest

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scenesmith.scene_expert.memory.writer import MemoryWriter
from scenesmith.scene_expert.schemas import FullVerifyReport


def _response(*, content=None, reasoning=None, finish_reason="stop"):
    message = types.SimpleNamespace(
        content=content,
        reasoning_content=reasoning,
        text=None,
        refusal=None,
    )
    return types.SimpleNamespace(
        choices=[
            types.SimpleNamespace(message=message, finish_reason=finish_reason)
        ]
    )


class MemoryWriterResilienceTest(unittest.TestCase):
    def test_reasoning_is_not_treated_as_final_json(self) -> None:
        writer = MemoryWriter.__new__(MemoryWriter)
        response = _response(
            reasoning=(
                "```json\n"
                '{"updates":[{"op":"NOOP","memory_type":"success_case",'
                '"content":{}}]}\n```'
            )
        )

        content, reasoning = writer._extract_response_parts(response)

        self.assertEqual("", content)
        self.assertIn('"updates"', reasoning)
        self.assertEqual("", writer._extract_response_text(response))

    def test_fenced_json_is_found_after_an_unrelated_object(self) -> None:
        writer = MemoryWriter.__new__(MemoryWriter)
        raw = """\
<think>{"plan":"draft an update"}</think>
The final result is:
```json
{"updates":[{"op":"NOOP","memory_type":"success_case","content":{}}]}
```
"""

        data = writer._parse_json_payload(raw)

        self.assertEqual(1, len(data["updates"]))
        self.assertEqual("NOOP", data["updates"][0]["op"])

    def test_incomplete_json_is_rejected(self) -> None:
        writer = MemoryWriter.__new__(MemoryWriter)

        with self.assertRaisesRegex(ValueError, "No complete JSON object"):
            writer._parse_json_payload('{"updates":[{"op":"NOOP"}')

    def test_valid_json_is_found_after_a_malformed_leading_fragment(self) -> None:
        writer = MemoryWriter.__new__(MemoryWriter)
        raw = (
            'discarded fragment {"draft": '
            '{"updates":[{"op":"NOOP","memory_type":"success_case",'
            '"content":{}}]}'
        )

        data = writer._parse_json_payload(raw)

        self.assertEqual("NOOP", data["updates"][0]["op"])

    def test_qwen38_request_disables_thinking_and_uses_retry_budget(self) -> None:
        recorded = {}

        class _Completions:
            @staticmethod
            def create(**kwargs):
                recorded.update(kwargs)
                return _response(content='{"updates":[]}')

        writer = MemoryWriter.__new__(MemoryWriter)
        writer._model = "unsloth/Qwen3.8-27B-GGUF"
        writer._max_tokens = 2048
        writer._retry_max_tokens = 4096
        writer._thinking_mode = "none"
        writer._timeout_seconds = 90.0
        writer._temperature = 0.1
        writer._client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=_Completions())
        )

        writer._request_completion(
            "trace",
            use_response_format=False,
            max_tokens=writer._retry_max_tokens,
            thinking_effort="none",
        )

        self.assertEqual(4096, recorded["max_tokens"])
        self.assertEqual(
            {"chat_template_kwargs": {"enable_thinking": False}},
            recorded["extra_body"],
        )
        self.assertNotIn("/think", recorded["messages"][0]["content"])
        self.assertNotIn("response_format", recorded)

    def test_reasoning_only_response_recovers_and_records_attempts(self) -> None:
        writer = MemoryWriter.__new__(MemoryWriter)
        writer._model = "unsloth/Qwen3.8-27B-GGUF"
        writer._max_tokens = 2048
        writer._retry_max_tokens = 4096
        writer._thinking_mode = "none"
        writer._timeout_seconds = 90.0
        writer._temperature = 0.1
        first = _response(reasoning="draft", finish_reason="length")
        second = _response(
            content=(
                '{"updates":[{"op":"NOOP","memory_type":"success_case",'
                '"content":{}}]}'
            )
        )
        full_report = FullVerifyReport(overall_score=0.2, pass_scene=False)

        with TemporaryDirectory() as temp_dir:
            writer._debug_dir = Path(temp_dir)
            with patch.object(
                writer,
                "_request_completion",
                side_effect=[first, second],
            ) as request:
                ops = writer.write("Trace: trace_1", full_report)

            debug_payload = json.loads(
                (Path(temp_dir) / "memory_writer_debug.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual("NOOP", ops[0].op)
        self.assertEqual(2, request.call_count)
        self.assertEqual(4096, request.call_args_list[1].kwargs["max_tokens"])
        self.assertEqual("none", request.call_args_list[1].kwargs["thinking_effort"])
        self.assertEqual("recovered_after_retry", debug_payload["status"])
        self.assertIn("reasoning-only", debug_payload["attempts"][0]["error"])
        self.assertEqual("NOOP", debug_payload["result_ops"][0]["op"])


if __name__ == "__main__":
    unittest.main()
