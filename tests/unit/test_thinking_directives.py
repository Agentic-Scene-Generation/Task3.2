import unittest

import sys
import types

sys.modules.setdefault("openai", types.SimpleNamespace(OpenAI=object))

from scenesmith.agent_utils.thinking import (
    prepend_text_thinking_directive,
    thinking_directive_from_effort,
)
from scenesmith.agent_utils.vlm_service import VLMService


class ThinkingDirectivesTest(unittest.TestCase):
    def test_agent_instruction_directive_mapping(self) -> None:
        self.assertEqual("/no_think", thinking_directive_from_effort("none"))
        self.assertEqual("/no_think", thinking_directive_from_effort("minimal"))
        self.assertEqual("/think", thinking_directive_from_effort("high"))

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
