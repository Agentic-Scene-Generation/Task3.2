"""Tests for authoritative floor-plan critic context."""

from scenesmith.agent_utils.house import (
    HouseLayout,
    Opening,
    OpeningType,
    PlacedRoom,
    Wall,
    WallDirection,
)
from scenesmith.floor_plan_agents.critic_context import format_floor_plan_critic_context


def test_critic_context_reports_live_dimensions_openings_and_free_wall_spans():
    wall_north = Wall(
        wall_id="living_north",
        room_id="living",
        direction=WallDirection.NORTH,
        start_point=(0.0, 0.0),
        end_point=(4.0, 0.0),
        length=4.0,
        openings=[
            Opening("window_1", OpeningType.WINDOW, 0.5, 1.0, 1.2, 0.9),
            Opening("window_2", OpeningType.WINDOW, 2.0, 1.0, 1.2, 0.9),
        ],
    )
    wall_south = Wall(
        wall_id="living_south",
        room_id="living",
        direction=WallDirection.SOUTH,
        start_point=(0.0, 8.5),
        end_point=(4.0, 8.5),
        length=4.0,
        openings=[Opening("door_1", OpeningType.DOOR, 1.55, 0.9, 2.1)],
    )
    layout = HouseLayout(
        placement_valid=True,
        connectivity_valid=True,
        placed_rooms=[
            PlacedRoom(
                room_id="living",
                position=(0.0, 0.0),
                width=4.0,
                depth=8.5,
                walls=[wall_north, wall_south],
            )
        ],
        boundary_labels={
            "A": ("living", None, WallDirection.NORTH),
            "B": ("living", None, WallDirection.SOUTH),
        },
    )

    context = format_floor_plan_critic_context(layout)

    assert "placement_valid=True; connectivity_valid=True" in context
    assert "living: 4.00m x 8.50m (34.00m2)" in context
    assert "wall A (north, exterior): length=4.00m" in context
    assert "window_1=window [0.50-1.50m]" in context
    assert "window_2=window [2.00-3.00m]" in context
    assert "opening-free spans=0.00-0.50m, 1.50-2.00m, 3.00-4.00m" in context
    assert "door_1=door [1.55-2.45m]" in context


def test_critic_context_handles_layout_before_room_placement():
    context = format_floor_plan_critic_context(HouseLayout())

    assert "No placed room geometry is available yet." in context
