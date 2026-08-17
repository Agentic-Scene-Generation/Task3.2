"""Tests for the critic-facing physics summary."""

from scenesmith.agent_utils.physics_tools import _build_violation_message


class _Violation:
    def __init__(self, description: str) -> None:
        self.description = description

    def to_description(self) -> str:
        return self.description


def test_window_only_physics_context_is_explicitly_advisory():
    message = _build_violation_message(
        collisions=[],
        thin_covering_overlaps=[],
        thin_covering_boundary_violations=[],
        door_violations=[],
        open_violations=[],
        height_violations=[],
        window_violations=[_Violation("monitor_0 blocks window_1")],
    )

    assert message.startswith("No hard physics violations detected.")
    assert "Window access warnings are advisory" in message
    assert "do not move required furniture" in message


def test_hard_physics_context_distinguishes_window_warnings():
    message = _build_violation_message(
        collisions=[_Violation("desk_0 collides with chair_0")],
        thin_covering_overlaps=[],
        thin_covering_boundary_violations=[],
        door_violations=[],
        open_violations=[],
        height_violations=[],
        window_violations=[_Violation("desk_0 blocks window_1")],
    )

    assert "1 hard issue(s), 1 advisory window warning(s)" in message
    assert "Resolve the hard issues" in message
