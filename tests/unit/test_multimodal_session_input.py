"""Tests for merging multimodal Designer input with session history."""

from scenesmith.agent_utils.base_stateful_agent import _append_session_input


def test_append_session_input_keeps_history_before_new_multimodal_message() -> None:
    history = [{"role": "assistant", "content": "previous result"}]
    new_input = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "edit this room"},
                {"type": "input_image", "image_url": "data:image/png;base64,abc"},
            ],
        }
    ]

    merged = _append_session_input(history, new_input)

    assert merged == history + new_input
    assert merged is not history
    assert merged is not new_input
