"""Focused tests for wall-stage window repair contract protection."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import Mock

from scenesmith.agent_utils.house import (
    HouseLayout,
    Opening,
    OpeningType,
    PlacedRoom,
    RoomSpec,
    Wall,
    WallDirection,
    Window,
)
from scenesmith.wall_agents.tools.window_tools import WindowRepairTools


def _migration_tools(tmp_path) -> WindowRepairTools:
    walls = [
        Wall(
            wall_id="room_north",
            room_id="room",
            direction=WallDirection.NORTH,
            start_point=(0.0, 4.0),
            end_point=(4.0, 4.0),
            length=4.0,
            openings=[
                Opening(
                    opening_id="window_0",
                    opening_type=OpeningType.WINDOW,
                    position_along_wall=1.4,
                    width=1.2,
                    height=1.1,
                    sill_height=0.85,
                )
            ],
        ),
        Wall(
            wall_id="room_east",
            room_id="room",
            direction=WallDirection.EAST,
            start_point=(4.0, 4.0),
            end_point=(4.0, 0.0),
            length=4.0,
        ),
    ]
    layout = HouseLayout(
        house_dir=tmp_path,
        room_specs=[RoomSpec(room_id="room", width=4.0, length=4.0)],
        placed_rooms=[
            PlacedRoom(
                room_id="room",
                position=(0.0, 0.0),
                width=4.0,
                depth=4.0,
                walls=walls,
            )
        ],
        windows=[
            Window(
                id="window_0",
                boundary_label="A",
                position_along_wall=1.4,
                room_id="room",
                wall_direction=WallDirection.NORTH,
                width=1.2,
                height=1.1,
                sill_height=0.85,
            )
        ],
        boundary_labels={
            "A": ("room", None, "north"),
            "B": ("room", None, "east"),
        },
    )
    tools = WindowRepairTools.__new__(WindowRepairTools)
    tools.house_layout = layout
    tools.room_output_dir = tmp_path / "room_room"
    tools.scene = SimpleNamespace(
        room_id="room",
        to_state_dict=Mock(return_value={"state": "before"}),
        restore_from_state_dict=Mock(),
    )
    tools.floor_plan_tools = SimpleNamespace(
        min_opening_separation=0.4,
        door_window_config=SimpleNamespace(window_segment_margin=0.1),
    )
    tools.refresh_wall_surfaces = Mock()
    tools.rendering_manager = SimpleNamespace(clear_cache=Mock())
    tools._rebuild_room_geometry = Mock()
    return tools


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


def test_atomic_window_migration_prefers_same_wall_and_preserves_dimensions(
    tmp_path,
) -> None:
    tools = _migration_tools(tmp_path)

    result = tools.migrate_window_atomically(
        window_id="window_0",
        accept_candidate=lambda candidate: candidate["same_wall"],
    )

    assert result.success
    assert result.old_wall_direction == "north"
    assert result.new_wall_direction == "north"
    window = tools.house_layout.windows[0]
    assert window.id == "window_0"
    assert (window.width, window.height, window.sill_height) == (1.2, 1.1, 0.85)
    assert window.position_along_wall != 1.4
    persisted = json.loads((tmp_path / "house_layout.json").read_text())
    assert persisted["windows"][0]["position_along_wall"] == (
        window.position_along_wall
    )


def test_atomic_window_migration_searches_other_exterior_walls(tmp_path) -> None:
    tools = _migration_tools(tmp_path)

    result = tools.migrate_window_atomically(
        window_id="window_0",
        accept_candidate=lambda candidate: candidate["wall_direction"] == "east",
    )

    assert result.success
    assert result.new_wall_direction == "east"
    window = tools.house_layout.windows[0]
    assert window.wall_direction == WallDirection.EAST
    assert window.boundary_label == "B"


def test_atomic_window_migration_rolls_back_layout_and_scene_on_rebuild_failure(
    tmp_path,
) -> None:
    tools = _migration_tools(tmp_path)
    tools._rebuild_room_geometry = Mock(side_effect=RuntimeError("rebuild failed"))

    result = tools.migrate_window_atomically(
        window_id="window_0",
        accept_candidate=lambda _candidate: True,
    )

    assert not result.success
    window = tools.house_layout.windows[0]
    assert window.wall_direction == WallDirection.NORTH
    assert window.position_along_wall == 1.4
    assert tools.scene.restore_from_state_dict.call_count >= 2
    persisted = json.loads((tmp_path / "house_layout.json").read_text())
    assert persisted["windows"][0]["wall_direction"] == "north"
    assert persisted["windows"][0]["position_along_wall"] == 1.4


def test_rejected_window_migration_rebuilds_original_geometry_files(tmp_path) -> None:
    tools = _migration_tools(tmp_path)
    marker = tmp_path / "room_geometry.marker"

    def rebuild(*, persist_layout=True):
        marker.write_text(
            str(tools.house_layout.windows[0].position_along_wall),
            encoding="utf-8",
        )

    tools._rebuild_room_geometry = Mock(side_effect=rebuild)

    result = tools.migrate_window_atomically(
        window_id="window_0",
        accept_candidate=lambda _candidate: (False, "still blocked"),
    )

    assert not result.success
    assert marker.read_text(encoding="utf-8") == "1.4"
    assert tools.house_layout.windows[0].position_along_wall == 1.4
