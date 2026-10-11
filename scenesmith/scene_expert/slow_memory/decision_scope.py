"""Portable classification of direct furniture policy targets."""

from typing import Any

DESIGN_TOOLS = frozenset(
    {
        "generate_assets",
        "add_furniture_to_scene",
        "add_furniture_to_scene_tool",
        "move_furniture",
        "move_furniture_tool",
        "remove_furniture",
        "remove_furniture_tool",
        "rescale_furniture",
        "rescale_furniture_tool",
    }
)


def first_turn_has_design(messages: list[dict[str, Any]]) -> bool:
    """Require a design tool in the first assistant turn, without transport IDs."""
    for message in messages:
        if message.get("role") != "assistant":
            break
        if any(
            call.get("function", {}).get("name") in DESIGN_TOOLS
            for call in message.get("tool_calls", [])
        ):
            return True
    return False
