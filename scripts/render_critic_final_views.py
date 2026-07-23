"""Render final, unlabeled inspection views for a critic-probe run.

Run with Blender's Python::

    blender -b --python scripts/render_critic_final_views.py -- \
      /path/to/outputs/critic_probe/<run_id>

The script finds ``critic_on/**/combined_house/house.blend`` files and writes
``critic_final_views/00_top.png`` and ``critic_final_views/01_side.png`` next
to each scene.  A side-facing wall is hidden only for the side render.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector

import render_blend_views as render


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Run directory, scene directory, or .blend file")
    parser.add_argument("--resolution", type=int, default=2048)
    parser.add_argument("--engine", choices=("auto", "eevee", "cycles", "workbench"), default="eevee")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--side", choices=("north", "east", "south", "west"), default="north")
    parser.add_argument("--margin", type=float, default=1.18)
    parser.add_argument("--include-shared-base", action="store_true")
    return parser.parse_args(argv)


def blend_files(input_path: Path, include_shared_base: bool) -> list[Path]:
    input_path = input_path.resolve()
    if input_path.is_file():
        return [input_path] if input_path.suffix == ".blend" else []
    if not input_path.is_dir():
        raise FileNotFoundError(input_path)
    files = sorted(input_path.rglob("combined_house/house.blend"))
    if include_shared_base or "shared_base" in input_path.parts:
        return files
    return [path for path in files if "shared_base" not in path.relative_to(input_path).parts]


def object_center(obj: bpy.types.Object) -> Vector:
    points = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    return sum(points, Vector((0.0, 0.0, 0.0))) / len(points)


def hide_nearest_wall(direction: Vector) -> str | None:
    """Hide the wall whose mesh center is nearest to the side camera."""
    candidates = [
        obj for obj in bpy.context.scene.objects
        if obj.type == "MESH" and "wall" in obj.name.lower() and not obj.hide_render
    ]
    if not candidates:
        return None
    # Camera looks from +direction, so the largest projection is nearest.
    nearest = max(candidates, key=lambda obj: object_center(obj).dot(direction))
    nearest.hide_render = True
    nearest.hide_viewport = True
    return nearest.name


def render_one(blend_path: Path, args: argparse.Namespace) -> Path:
    bpy.ops.wm.open_mainfile(filepath=str(blend_path))
    render.hide_matching_objects(argparse.Namespace(hide_name_contains=[], hide_ceiling=True))
    render.configure_scene(argparse.Namespace(
        resolution=args.resolution, transparent=False, engine=args.engine,
        samples=args.samples,
    ))
    output_dir = blend_path.parent.parent / "critic_final_views"
    output_dir.mkdir(parents=True, exist_ok=True)
    minimum, maximum, corners = render.scene_bbox()
    center = (minimum + maximum) / 2.0
    extent = maximum - minimum
    camera = render.ensure_camera()

    # Top view keeps all visible walls; side view removes only the wall facing it.
    rendered = []
    for filename, view_name in (("00_top.png", "top"), ("01_side.png", args.side)):
        hidden_wall = None
        view_center, view_extent, view_corners = center, extent, corners
        if view_name != "top":
            hidden_wall = hide_nearest_wall(render.view_direction(view_name))
            if hidden_wall:
                view_minimum, view_maximum, view_corners = render.scene_bbox()
                view_center = (view_minimum + view_maximum) / 2.0
                view_extent = view_maximum - view_minimum
        render.setup_camera_for_view(
            camera, view_name, view_corners, view_center, view_extent,
            argparse.Namespace(projection="ortho", camera_distance_scale=3.0, margin=args.margin),
        )
        path = output_dir / filename
        bpy.context.scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        rendered.append({"view": view_name, "path": str(path), "hidden_wall": hidden_wall})
        # Restore the wall before the next file is opened (and for manifest clarity).
        if hidden_wall:
            obj = bpy.data.objects.get(hidden_wall)
            if obj:
                obj.hide_render = False
                obj.hide_viewport = False
    manifest = {"input": str(blend_path), "resolution": args.resolution, "renders": rendered}
    (output_dir / "render_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output_dir


def main() -> None:
    args = parse_args()
    files = blend_files(args.input, args.include_shared_base)
    if not files:
        raise SystemExit(f"No combined_house/house.blend found below {args.input}")
    for blend_path in files:
        print(f"[render_critic_final_views] {blend_path}")
        print(f"[render_critic_final_views] wrote {render_one(blend_path, args)}")


if __name__ == "__main__":
    main()
