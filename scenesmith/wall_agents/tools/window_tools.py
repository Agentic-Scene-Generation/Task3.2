"""Window repair tools usable while resolving wall-mounted media placement."""

from __future__ import annotations

import copy
import json
import logging
import os

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from agents import function_tool
from omegaconf import DictConfig, OmegaConf

from scenesmith.agent_utils.house import (
    HouseLayout,
    Opening,
    OpeningType,
    Wall,
    WallDirection,
)
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


@dataclass(frozen=True)
class WindowMigrationResult:
    """Outcome of one transactional window migration search."""

    success: bool
    window_id: str
    old_wall_direction: str = ""
    new_wall_direction: str = ""
    old_position_along_wall: float = 0.0
    new_position_along_wall: float = 0.0
    reason: str = ""


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
                    "wall_surface_id": (
                        f"{room_id}_{window.wall_direction.value}"
                        if window.wall_direction
                        else window.boundary_label
                    ),
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
        manifest_path = (
            Path(self.house_layout.house_dir or self.room_output_dir.parent)
            / "floor_plan_reservation_manifest.json"
        )
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                explicit_count = int(manifest.get("explicit_window_count") or 0)
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                explicit_count = 0
            # The current layout format does not retain which generated window
            # fulfilled an explicit prompt obligation.  Removing any one of them
            # could therefore delete a required opening even when extra windows
            # remain, so use resize/move whenever the contract names windows.
            if explicit_count:
                return json.dumps(
                    {
                        "success": False,
                        "message": (
                            "Cannot remove windows: the floor-plan contract "
                            f"explicitly requires {explicit_count} window(s), and "
                            "their identities are not tracked. "
                            "Resize or move it instead."
                        ),
                    }
                )
        result = self.floor_plan_tools._remove_window_impl(window_id)
        return self._finish_edit(result, f"removed window '{window_id}'")

    def migrate_window_atomically(
        self,
        *,
        window_id: str,
        accept_candidate: Callable[[dict[str, Any]], bool | tuple[bool, str]],
    ) -> WindowMigrationResult:
        """Search legal exterior-wall positions and commit one fresh-validated pose."""
        window = next(
            (item for item in self.house_layout.windows if item.id == window_id),
            None,
        )
        if window is None:
            return WindowMigrationResult(
                success=False,
                window_id=window_id,
                reason=f"Window '{window_id}' not found",
            )
        old_direction = window.wall_direction
        old_position = float(window.position_along_wall)
        snapshot = self._window_transaction_snapshot()
        candidates = self._window_migration_candidates(window_id)
        rejection_reasons: list[str] = []
        for wall, position in candidates:
            self._restore_window_transaction(snapshot, persist=False)
            try:
                self._set_window_location(
                    window_id=window_id,
                    wall_direction=wall.direction,
                    position_along_wall=position,
                )
                self._rebuild_room_geometry(persist_layout=False)
                self.refresh_wall_surfaces()
                self.rendering_manager.clear_cache()
                verdict = accept_candidate(
                    {
                        "window_id": window_id,
                        "wall_direction": wall.direction.value,
                        "position_along_wall": position,
                        "same_wall": wall.direction == old_direction,
                    }
                )
                if isinstance(verdict, tuple):
                    accepted, reason = bool(verdict[0]), str(verdict[1])
                else:
                    accepted, reason = bool(verdict), "candidate rejected"
                if accepted:
                    self._persist_layout()
                    return WindowMigrationResult(
                        success=True,
                        window_id=window_id,
                        old_wall_direction=(
                            old_direction.value if old_direction is not None else ""
                        ),
                        new_wall_direction=wall.direction.value,
                        old_position_along_wall=old_position,
                        new_position_along_wall=position,
                    )
                rejection_reasons.append(reason)
            except Exception as exc:
                rejection_reasons.append(
                    f"{wall.direction.value}@{position:.3f}: {exc}"
                )
                console_logger.warning(
                    "Rejected window migration candidate %s %s@%.3f",
                    window_id,
                    wall.direction.value,
                    position,
                    exc_info=True,
                )

        self._restore_window_transaction(snapshot, persist=False)
        if candidates:
            try:
                # Candidate generation rewrites room SDF/mesh files in place.
                # Rebuild once from the restored layout so rollback covers the
                # on-disk wall holes as well as the in-memory scene and JSON.
                self._rebuild_room_geometry(persist_layout=False)
            except Exception:
                self._restore_window_transaction(snapshot, persist=False)
                console_logger.warning(
                    "Could not rebuild original geometry after rejecting window "
                    "migration candidates",
                    exc_info=True,
                )
        self._persist_layout()
        self.refresh_wall_surfaces()
        self.rendering_manager.clear_cache()
        return WindowMigrationResult(
            success=False,
            window_id=window_id,
            old_wall_direction=(
                old_direction.value if old_direction is not None else ""
            ),
            old_position_along_wall=old_position,
            reason=(
                "; ".join(dict.fromkeys(rejection_reasons))
                if rejection_reasons
                else "no legal exterior-wall candidate"
            ),
        )

    def _window_transaction_snapshot(self) -> dict[str, Any]:
        house_dir = Path(self.house_layout.house_dir or self.room_output_dir.parent)
        return {
            "layout": copy.deepcopy(self.house_layout.to_dict(scene_dir=house_dir)),
            "scene": copy.deepcopy(self.scene.to_state_dict()),
        }

    def _restore_window_transaction(
        self,
        snapshot: dict[str, Any],
        *,
        persist: bool,
    ) -> None:
        house_dir = Path(self.house_layout.house_dir or self.room_output_dir.parent)
        restored = HouseLayout.from_dict(
            copy.deepcopy(snapshot["layout"]),
            house_dir=house_dir,
        )
        self.house_layout.__dict__.clear()
        self.house_layout.__dict__.update(restored.__dict__)
        self.scene.restore_from_state_dict(copy.deepcopy(snapshot["scene"]))
        if persist:
            self._persist_layout()

    def _window_migration_candidates(
        self,
        window_id: str,
    ) -> list[tuple[Wall, float]]:
        window = next(
            (item for item in self.house_layout.windows if item.id == window_id),
            None,
        )
        room = self.house_layout.get_placed_room(self.scene.room_id)
        if window is None or room is None:
            return []
        old_position = float(window.position_along_wall)
        margin = max(
            0.0,
            float(
                getattr(
                    self.floor_plan_tools.door_window_config,
                    "window_segment_margin",
                    0.1,
                )
            ),
        )
        walls = [wall for wall in room.walls if wall.is_exterior]
        walls.sort(
            key=lambda wall: (
                0 if wall.direction == window.wall_direction else 1,
                wall.direction.value,
            )
        )
        candidates: list[tuple[Wall, float]] = []
        seen: set[tuple[str, float]] = set()
        for wall in walls:
            if wall.length + 1e-9 < window.width + 2.0 * margin:
                continue
            occupied: list[tuple[float, float]] = []
            for opening in wall.openings:
                if opening.opening_id == window_id:
                    continue
                occupied.append(
                    (
                        max(
                            margin,
                            float(opening.position_along_wall)
                            - self.floor_plan_tools.min_opening_separation,
                        ),
                        min(
                            float(wall.length) - margin,
                            float(opening.position_along_wall)
                            + float(opening.width)
                            + self.floor_plan_tools.min_opening_separation,
                        ),
                    )
                )
            for start, end in self._free_wall_intervals(
                start=margin,
                end=float(wall.length) - margin,
                occupied=occupied,
            ):
                latest = end - float(window.width)
                if latest < start - 1e-9:
                    continue
                for position in (start, (start + latest) / 2.0, latest):
                    if (
                        wall.direction == window.wall_direction
                        and abs(position - old_position) < 1e-4
                    ):
                        continue
                    key = (wall.direction.value, round(position, 6))
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append((wall, float(position)))
        return candidates

    @staticmethod
    def _free_wall_intervals(
        *,
        start: float,
        end: float,
        occupied: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        cursor = start
        free: list[tuple[float, float]] = []
        for lower, upper in sorted(occupied):
            if upper <= cursor:
                continue
            if lower > cursor:
                free.append((cursor, min(lower, end)))
            cursor = max(cursor, upper)
            if cursor >= end:
                break
        if cursor < end:
            free.append((cursor, end))
        return [(lower, upper) for lower, upper in free if upper > lower]

    def _set_window_location(
        self,
        *,
        window_id: str,
        wall_direction: WallDirection,
        position_along_wall: float,
    ) -> None:
        window = next(
            (item for item in self.house_layout.windows if item.id == window_id),
            None,
        )
        room = self.house_layout.get_placed_room(self.scene.room_id)
        if window is None or room is None:
            raise RuntimeError(f"Window '{window_id}' is no longer available")
        target_wall = next(
            (item for item in room.walls if item.direction == wall_direction),
            None,
        )
        if target_wall is None or not target_wall.is_exterior:
            raise RuntimeError("Window migration target is not an exterior wall")
        for room_wall in room.walls:
            room_wall.openings = [
                opening
                for opening in room_wall.openings
                if opening.opening_id != window_id
            ]
        boundary_label = self._exterior_boundary_label(target_wall.direction)
        if boundary_label is None:
            raise RuntimeError(
                f"No exterior boundary label for {target_wall.direction.value} wall"
            )
        window.boundary_label = boundary_label
        window.wall_direction = target_wall.direction
        window.position_along_wall = float(position_along_wall)
        target_wall.openings.append(
            Opening(
                opening_id=window.id,
                opening_type=OpeningType.WINDOW,
                position_along_wall=float(position_along_wall),
                width=float(window.width),
                height=float(window.height),
                sill_height=float(window.sill_height),
            )
        )
        self.house_layout.invalidate_room_geometry(window.room_id)

    def _exterior_boundary_label(
        self,
        direction: WallDirection,
    ) -> str | None:
        for label, value in self.house_layout.boundary_labels.items():
            room_a, room_b, direction_value = value
            if (
                str(room_a) == str(self.scene.room_id)
                and room_b is None
                and str(direction_value) == direction.value
            ):
                return str(label)
        return None

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

    def _rebuild_room_geometry(self, *, persist_layout: bool = True) -> None:
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

        if persist_layout:
            # Persist ordinary wall-tool edits immediately. Transactional
            # migration candidates stay memory-only until fresh validation.
            self._persist_layout()

    def _persist_layout(self) -> None:
        house_dir = Path(self.house_layout.house_dir or self.room_output_dir.parent)
        layout_path = house_dir / "house_layout.json"
        temporary = layout_path.with_suffix(layout_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.house_layout.to_dict(scene_dir=house_dir), indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, layout_path)
