"""Window repair tools usable while resolving wall-mounted media placement."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

from agents import function_tool
from omegaconf import DictConfig, OmegaConf

from scenesmith.agent_utils.house import HouseLayout
from scenesmith.agent_utils.room import ObjectType, RoomScene
from scenesmith.floor_plan_agents.stateful_floor_plan_agent import (
    StatefulFloorPlanAgent,
)
from scenesmith.floor_plan_agents.tools.floor_plan_tools import (
    DoorWindowConfig,
    FloorPlanTools,
)
from scenesmith.floor_plan_agents.tools.geometry_cache import GeometryCache

console_logger = logging.getLogger(__name__)


class _GeometryLogger:
    """Forward SDF writes while keeping generated files at the house root."""

    def __init__(self, delegate: Any, output_dir: Path):
        self._delegate = delegate
        self.output_dir = output_dir

    def log_sdf(self, *args: Any, **kwargs: Any) -> Path:
        return self._delegate.log_sdf(*args, **kwargs)


class WindowRepairTools:
    """Expose floor-plan window edits to the wall designer.

    Window edits invalidate and regenerate the current room geometry. This is
    intentionally kept in the wall package: the edit is only offered when a
    wall-mounted object has a concrete media/window conflict.
    """

    def __init__(
        self,
        *,
        scene: RoomScene,
        house_layout: HouseLayout,
        floor_plan_cfg: DictConfig | dict[str, Any],
        room_output_dir: Path,
        refresh_wall_surfaces: Callable[[], None],
        rendering_manager: Any,
        logger: Any,
    ) -> None:
        self.scene = scene
        self.house_layout = house_layout
        self.floor_plan_cfg = (
            floor_plan_cfg
            if isinstance(floor_plan_cfg, DictConfig)
            else OmegaConf.create(floor_plan_cfg)
        )
        self.room_output_dir = Path(room_output_dir)
        self.refresh_wall_surfaces = refresh_wall_surfaces
        self.rendering_manager = rendering_manager
        self.logger = logger

        windows_cfg = self.floor_plan_cfg.get("windows", {})
        room_placement_cfg = self.floor_plan_cfg.get("room_placement", {})
        width_range = list(windows_cfg.get("width_range", [0.6, 3.0]))
        height_range = list(windows_cfg.get("height_range", [0.6, 2.0]))
        self.floor_plan_tools = FloorPlanTools(
            layout=house_layout,
            mode="room",
            min_opening_separation=float(
                room_placement_cfg.get("min_opening_separation", 0.5)
            ),
            door_window_config=DoorWindowConfig(
                window_width_min=float(width_range[0]),
                window_width_max=float(width_range[1]),
                window_height_min=float(height_range[0]),
                window_height_max=float(height_range[1]),
                window_default_width=float(windows_cfg.get("default_width", 1.2)),
                window_default_height=float(windows_cfg.get("default_height", 1.2)),
                window_default_sill_height=float(
                    windows_cfg.get("default_sill_height", 0.9)
                ),
            ),
        )
        self.tools = self._create_tool_closures()

    def _create_tool_closures(self) -> dict[str, Any]:
        @function_tool
        def list_windows() -> str:
            """List exact window IDs, walls, dimensions, and positions."""
            return self._list_windows_impl()

        @function_tool
        def resize_window(
            window_id: str,
            width: float,
            height: float | None = None,
            sill_height: float | None = None,
        ) -> str:
            """Resize a window around its current center and rebuild the room."""
            return self._resize_window_impl(
                window_id=window_id,
                width=width,
                height=height,
                sill_height=sill_height,
            )

        @function_tool
        def move_window(window_id: str, position_along_wall: float) -> str:
            """Move a window along its existing wall and rebuild the room."""
            return self._move_window_impl(
                window_id=window_id,
                position_along_wall=position_along_wall,
            )

        @function_tool
        def remove_window(window_id: str) -> str:
            """Remove a window and rebuild the room."""
            return self._remove_window_impl(window_id=window_id)

        return {
            "list_windows": list_windows,
            "resize_window": resize_window,
            "move_window": move_window,
            "remove_window": remove_window,
        }

    def _list_windows_impl(self) -> str:
        room_id = self.scene.room_id
        rows = []
        for window in self.house_layout.windows:
            if window.room_id != room_id:
                continue
            rows.append(
                {
                    "window_id": window.id,
                    "wall_surface_id": f"{room_id}_{window.wall_direction.value}"
                    if window.wall_direction
                    else window.boundary_label,
                    "wall_direction": (
                        window.wall_direction.value if window.wall_direction else None
                    ),
                    "boundary_label": window.boundary_label,
                    "position_along_wall": round(float(window.position_along_wall), 4),
                    "center_along_wall": round(
                        float(window.position_along_wall + window.width / 2), 4
                    ),
                    "width": round(float(window.width), 4),
                    "height": round(float(window.height), 4),
                    "sill_height": round(float(window.sill_height), 4),
                }
            )
        return json.dumps({"room_id": room_id, "windows": rows}, indent=2)

    def _resize_window_impl(
        self,
        *,
        window_id: str,
        width: float,
        height: float | None,
        sill_height: float | None,
    ) -> str:
        # A wall critique must be able to free the support centerline directly;
        # moving the TV instead would violate the requested support relation.
        result = self.floor_plan_tools._resize_window_impl(
            window_id=window_id,
            width=width,
            height=height,
            sill_height=sill_height,
        )
        return self._finish_edit(result, f"resized window '{window_id}'")

    def _move_window_impl(self, *, window_id: str, position_along_wall: float) -> str:
        result = self.floor_plan_tools._move_window_impl(
            window_id=window_id,
            position_along_wall=position_along_wall,
        )
        return self._finish_edit(result, f"moved window '{window_id}'")

    def _remove_window_impl(self, *, window_id: str) -> str:
        result = self.floor_plan_tools._remove_window_impl(window_id)
        return self._finish_edit(result, f"removed window '{window_id}'")

    def _finish_edit(self, result: Any, description: str) -> str:
        if not getattr(result, "success", False):
            return json.dumps(
                {"success": False, "message": getattr(result, "message", str(result))}
            )
        try:
            self._rebuild_room_geometry()
            self.refresh_wall_surfaces()
            self.rendering_manager.clear_cache()
        except Exception as exc:
            console_logger.error(
                "Window edit succeeded but room geometry refresh failed", exc_info=True
            )
            return json.dumps(
                {
                    "success": False,
                    "message": (
                        f"{description}, but geometry refresh failed: {exc}. "
                        "Do not place or move the TV until the room is refreshed."
                    ),
                }
            )
        return json.dumps(
            {
                "success": True,
                "message": (
                    f"Successfully {description}; wall openings, window visuals, "
                    "collision geometry, and wall excluded regions were refreshed."
                ),
            }
        )

    def _rebuild_room_geometry(self) -> None:
        room_id = self.scene.room_id
        room_spec = self.house_layout.get_room_spec(room_id)
        if room_spec is None:
            raise RuntimeError(f"Room spec '{room_id}' not found")
        house_dir = Path(self.house_layout.house_dir or self.room_output_dir.parent)
        floor_plans_dir = house_dir / "floor_plans"
        cache = GeometryCache(cache_dir=house_dir / ".window_repair_geometry_cache")

        # The floor-plan generator already rebuilds wall holes, window frames,
        # collision SDFs, and opening metadata. Reuse it so rendered geometry and
        # in-memory wall exclusions stay consistent.
        rebuilder = StatefulFloorPlanAgent.__new__(StatefulFloorPlanAgent)
        rebuilder.cfg = self.floor_plan_cfg
        rebuilder.layout = self.house_layout
        rebuilder.logger = _GeometryLogger(self.logger, house_dir)
        rebuilder._geometry_cache = cache
        new_geometry = rebuilder._generate_room_geometry(
            room_spec=room_spec,
            output_dir=floor_plans_dir,
        )

        self.house_layout.set_room_geometry(room_id, new_geometry)
        self.scene.room_geometry = new_geometry
        old_walls = self.scene.get_objects_by_type(ObjectType.WALL)
        for wall in old_walls:
            self.scene.remove_object(wall.object_id)
        for wall in new_geometry.walls:
            self.scene.add_object(wall)

        # Persist the repaired layout so later stages and replay checkpoints use
        # the same opening geometry instead of reloading the original window.
        layout_path = house_dir / "house_layout.json"
        layout_path.write_text(
            json.dumps(self.house_layout.to_dict(scene_dir=house_dir), indent=2),
            encoding="utf-8",
        )
