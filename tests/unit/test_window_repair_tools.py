"""Focused tests for wall-stage window repair contract protection."""

from __future__ import annotations

import json

from scenesmith.agent_utils.house import HouseLayout
from scenesmith.wall_agents.tools.window_tools import WindowRepairTools


def test_remove_window_is_blocked_when_prompt_requires_explicit_windows(
    tmp_path,
) -> None:
    layout = HouseLayout(house_dir=tmp_path)
    layout.windows = [object(), object(), object()]
    (tmp_path / "floor_plan_reservation_manifest.json").write_text(
        json.dumps(
            {
                "explicit_window_required": True,
                "explicit_window_count": 2,
            }
        ),
        encoding="utf-8",
    )
    repair_tools = WindowRepairTools.__new__(WindowRepairTools)
    repair_tools.house_layout = layout
    repair_tools.room_output_dir = tmp_path / "room_office"

    result = json.loads(
        repair_tools._remove_window_impl(window_id="untracked-window-id")
    )

    assert result["success"] is False
    assert "identities are not tracked" in result["message"]
    assert len(layout.windows) == 3
