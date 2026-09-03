import unittest

import sys
import types

sys.modules.setdefault(
    "openai", types.SimpleNamespace(OpenAI=object, AsyncOpenAI=object)
)

from scenesmith.agent_utils.thinking import (
    chat_api_reasoning_effort,
    chat_template_kwargs_from_effort,
    is_qwen38_model,
    prepend_text_thinking_directive,
    thinking_directive_from_effort,
)
from scenesmith.agent_utils.vlm_service import VLMService


class ThinkingDirectivesTest(unittest.TestCase):
    def test_online_chat_reasoning_effort_normalization(self) -> None:
        self.assertEqual("high", chat_api_reasoning_effort("HIGH"))
        self.assertEqual("none", chat_api_reasoning_effort("off"))
        self.assertIsNone(chat_api_reasoning_effort(None))
        with self.assertRaises(ValueError):
            chat_api_reasoning_effort("unsupported")

    def test_agent_instruction_directive_mapping(self) -> None:
        self.assertEqual("/no_think", thinking_directive_from_effort("none"))
        self.assertEqual("/no_think", thinking_directive_from_effort("minimal"))
        self.assertEqual("/think", thinking_directive_from_effort("high"))

    def test_chat_template_kwargs_follow_effort(self) -> None:
        self.assertEqual(
            {"chat_template_kwargs": {"enable_thinking": False}},
            chat_template_kwargs_from_effort("none"),
        )
        self.assertEqual(
            {"chat_template_kwargs": {"enable_thinking": True}},
            chat_template_kwargs_from_effort("low"),
        )

    def test_qwen38_uses_reasoning_effort_without_text_directive(self) -> None:
        model = "unsloth/Qwen3.8-27B-GGUF"
        self.assertTrue(is_qwen38_model(model))
        self.assertEqual("", thinking_directive_from_effort("medium", model))
        self.assertEqual(
            {"chat_template_kwargs": {"reasoning_effort": "medium"}},
            chat_template_kwargs_from_effort("medium", model),
        )
        self.assertEqual(
            "Place the bed.",
            prepend_text_thinking_directive(
                "/no_think\nPlace the bed.",
                thinking_directive_from_effort("medium", model),
            ),
        )
        self.assertEqual(
            {"chat_template_kwargs": {"enable_thinking": False}},
            chat_template_kwargs_from_effort("none", model),
        )

    def test_qwen36_keeps_enable_thinking_contract(self) -> None:
        model = "unsloth/Qwen3.6-27B-GGUF"
        self.assertFalse(is_qwen38_model(model))
        self.assertEqual("/think", thinking_directive_from_effort("medium", model))
        self.assertEqual(
            {"chat_template_kwargs": {"enable_thinking": True}},
            chat_template_kwargs_from_effort("medium", model),
        )

    def test_agent_instruction_directive_replaces_existing_prefix(self) -> None:
        self.assertEqual(
            "/think\nPlace the bed.",
            prepend_text_thinking_directive("/no_think\nPlace the bed.", "/think"),
        )

    def test_vlm_chat_directive_updates_first_user_text(self) -> None:
        service = VLMService.__new__(VLMService)
        messages = [
            {"role": "system", "content": "You are concise."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this mesh."},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,x"},
                    },
                ],
            },
        ]

        updated = service._prepend_thinking_directive(messages, "/no_think")

        self.assertEqual(
            "/no_think\nAnalyze this mesh.",
            updated[1]["content"][0]["text"],
        )
        self.assertEqual(
            "data:image/png;base64,x", updated[1]["content"][1]["image_url"]["url"]
        )


if __name__ == "__main__":
    unittest.main()
