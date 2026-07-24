from scenesmith.wall_agents.prompt_constraints import (
    build_required_wall_object_constraints,
)


def test_tv_with_media_support_requires_window_repair_before_offset() -> None:
    constraints = build_required_wall_object_constraints(
        "A sofa faces a TV stand and television on the opposite wall."
    )

    assert "REQUIRED media display" in constraints
    assert "call list_windows" in constraints
    assert "Never leave the display offset" in constraints


def test_desktop_monitor_does_not_become_wall_requirement() -> None:
    constraints = build_required_wall_object_constraints(
        "A desk centered against the back wall with a computer monitor on the desk."
    )

    assert "No explicit wall-object obligations" in constraints
