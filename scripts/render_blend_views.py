"""Render inspection PNGs from a Blender .blend scene.

Run this script with Blender's Python, not the project virtualenv Python:

    blender -b tmp/house.blend --python scripts/render_blend_views.py -- \
      --output tmp/house_blend_views

or:

    blender -b --python scripts/render_blend_views.py -- \
      --input tmp/house.blend --output tmp/house_blend_views
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


DEFAULT_VIEWS = ("top", "north", "east", "south", "west", "iso")


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    parser = argparse.ArgumentParser(
        description="Render a .blend file into multiple PNG inspection views."
    )
    parser.add_argument(
        "--input",
        type=Path,
        help=(
            "Input .blend file. Optional when Blender is launched as "
            "`blender -b scene.blend --python ...`."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=("Output directory. Defaults to <blend_parent>/<blend_stem>_blend_views."),
    )
    parser.add_argument(
        "--views",
        default=",".join(DEFAULT_VIEWS),
        help=(
            "Comma-separated views. Supported: top,north,east,south,west,iso,all. "
            "Default: top,north,east,south,west,iso."
        ),
    )
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument(
        "--engine",
        default="auto",
        choices=("auto", "eevee", "cycles", "workbench"),
        help="Render engine. Use eevee/auto for fast visual checks.",
    )
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--margin", type=float, default=1.18)
    parser.add_argument(
        "--projection",
        choices=("ortho", "perspective"),
        default="ortho",
        help="Camera projection for side/isometric views. Top is always orthographic.",
    )
    parser.add_argument(
        "--transparent",
        action="store_true",
        help="Render with transparent background.",
    )
    parser.add_argument(
        "--hide-ceiling",
        action="store_true",
        help="Hide objects whose name contains 'ceiling' before rendering.",
    )
    parser.add_argument(
        "--hide-name-contains",
        action="append",
        default=[],
        metavar="TEXT",
        help=(
            "Hide objects whose name contains TEXT, case-insensitive. "
            "Can be specified multiple times."
        ),
    )
    parser.add_argument(
        "--camera-distance-scale",
        type=float,
        default=3.0,
        help="Camera distance multiplier relative to scene radius.",
    )
    return parser.parse_args(argv)


def open_blend_file(input_path: Path | None) -> Path:
    if input_path is not None:
        input_path = input_path.resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"Input .blend file not found: {input_path}")
        bpy.ops.wm.open_mainfile(filepath=str(input_path))
        return input_path

    if bpy.data.filepath:
        return Path(bpy.data.filepath).resolve()

    raise ValueError(
        "No .blend file is loaded. Pass --input scene.blend or launch Blender as "
        "`blender -b scene.blend --python scripts/render_blend_views.py -- ...`."
    )


def hide_matching_objects(args: argparse.Namespace) -> list[str]:
    patterns = [p.lower() for p in args.hide_name_contains]
    if args.hide_ceiling:
        patterns.append("ceiling")

    hidden: list[str] = []
    if not patterns:
        return hidden

    for obj in bpy.context.scene.objects:
        name = obj.name.lower()
        if any(pattern in name for pattern in patterns):
            obj.hide_render = True
            obj.hide_viewport = True
            hidden.append(obj.name)
    return hidden


def scene_bbox() -> tuple[Vector, Vector, list[Vector]]:
    corners: list[Vector] = []
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH" or obj.hide_render:
            continue
        for corner in obj.bound_box:
            corners.append(obj.matrix_world @ Vector(corner))

    if not corners:
        raise RuntimeError("No visible mesh objects found in the scene.")

    mins = Vector(
        (
            min(corner.x for corner in corners),
            min(corner.y for corner in corners),
            min(corner.z for corner in corners),
        )
    )
    maxs = Vector(
        (
            max(corner.x for corner in corners),
            max(corner.y for corner in corners),
            max(corner.z for corner in corners),
        )
    )
    return mins, maxs, corners


def ensure_camera() -> bpy.types.Object:
    camera = bpy.data.objects.get("SceneExpertInspectionCamera")
    if camera is not None:
        return camera

    camera_data = bpy.data.cameras.new("SceneExpertInspectionCamera")
    camera = bpy.data.objects.new("SceneExpertInspectionCamera", camera_data)
    bpy.context.scene.collection.objects.link(camera)
    return camera


def look_at(camera: bpy.types.Object, target: Vector) -> None:
    direction = target - camera.location
    if direction.length == 0:
        raise ValueError("Camera location equals target; cannot orient camera.")
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def supported_render_engines() -> set[str]:
    scene = bpy.context.scene
    return {
        item.identifier for item in scene.render.bl_rna.properties["engine"].enum_items
    }


def set_render_engine(args: argparse.Namespace) -> str:
    scene = bpy.context.scene
    supported = supported_render_engines()

    candidates: list[str]
    if args.engine == "eevee":
        candidates = ["BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"]
    elif args.engine == "cycles":
        candidates = ["CYCLES"]
    elif args.engine == "workbench":
        candidates = ["BLENDER_WORKBENCH"]
    else:
        candidates = [
            "BLENDER_EEVEE_NEXT",
            "BLENDER_EEVEE",
            "CYCLES",
            "BLENDER_WORKBENCH",
        ]

    for engine in candidates:
        if engine in supported:
            scene.render.engine = engine
            break
    else:
        raise RuntimeError(f"No supported render engine found from: {candidates}")

    if scene.render.engine == "CYCLES":
        scene.cycles.samples = args.samples
        scene.cycles.use_denoising = True
    elif hasattr(scene, "eevee"):
        eevee = scene.eevee
        if hasattr(eevee, "taa_render_samples"):
            eevee.taa_render_samples = args.samples

    return scene.render.engine


def configure_scene(args: argparse.Namespace) -> str:
    scene = bpy.context.scene
    scene.render.resolution_x = args.resolution
    scene.render.resolution_y = args.resolution
    scene.render.film_transparent = args.transparent
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA" if args.transparent else "RGB"

    if scene.world is None:
        scene.world = bpy.data.worlds.new("SceneExpertWorld")
    scene.world.color = (0.78, 0.78, 0.78)

    if not any(obj.type == "LIGHT" and not obj.hide_render for obj in scene.objects):
        light_data = bpy.data.lights.new("SceneExpertInspectionLight", type="AREA")
        light_obj = bpy.data.objects.new("SceneExpertInspectionLight", light_data)
        scene.collection.objects.link(light_obj)
        light_data.energy = 500.0
        light_data.size = 5.0

    return set_render_engine(args)


def view_direction(view_name: str) -> Vector:
    elevation = math.radians(28.0)
    horizontal = math.cos(elevation)
    vertical = math.sin(elevation)
    directions = {
        "top": Vector((0.0, 0.0, 1.0)),
        "north": Vector((0.0, horizontal, vertical)),
        "east": Vector((horizontal, 0.0, vertical)),
        "south": Vector((0.0, -horizontal, vertical)),
        "west": Vector((-horizontal, 0.0, vertical)),
        "iso": Vector((1.0, -1.0, 0.72)),
    }
    if view_name not in directions:
        raise ValueError(f"Unsupported view: {view_name}")
    return directions[view_name].normalized()


def projected_ortho_scale(
    scene: bpy.types.Scene,
    camera: bpy.types.Object,
    corners: list[Vector],
    margin: float,
) -> float:
    bpy.context.view_layer.update()
    inv_camera = camera.matrix_world.inverted()
    camera_points = [inv_camera @ corner for corner in corners]
    width = max(point.x for point in camera_points) - min(
        point.x for point in camera_points
    )
    height = max(point.y for point in camera_points) - min(
        point.y for point in camera_points
    )
    aspect = scene.render.resolution_x / scene.render.resolution_y
    return max(height, width / aspect, 0.1) * margin


def setup_camera_for_view(
    camera: bpy.types.Object,
    view_name: str,
    corners: list[Vector],
    center: Vector,
    extent: Vector,
    args: argparse.Namespace,
) -> None:
    scene = bpy.context.scene
    direction = view_direction(view_name)
    radius = max(extent.length / 2.0, 1.0)
    distance = radius * args.camera_distance_scale
    camera.location = center + direction * distance
    look_at(camera, center)

    camera.data.clip_start = 0.01
    camera.data.clip_end = max(distance * 4.0, radius * 8.0, 100.0)

    if view_name == "top" or args.projection == "ortho":
        camera.data.type = "ORTHO"
        camera.data.ortho_scale = projected_ortho_scale(
            scene, camera, corners, args.margin
        )
    else:
        camera.data.type = "PERSP"
        camera.data.lens = 35

    scene.camera = camera


def parse_views(value: str) -> list[str]:
    raw_views = [view.strip().lower() for view in value.split(",") if view.strip()]
    if not raw_views or "all" in raw_views:
        return list(DEFAULT_VIEWS)
    valid = set(DEFAULT_VIEWS)
    invalid = [view for view in raw_views if view not in valid]
    if invalid:
        raise ValueError(f"Unsupported views: {invalid}; valid views: {sorted(valid)}")
    return raw_views


def render_views(args: argparse.Namespace, blend_path: Path) -> list[dict]:
    output_dir = args.output
    if output_dir is None:
        output_dir = blend_path.parent / f"{blend_path.stem}_blend_views"
    output_dir.mkdir(parents=True, exist_ok=True)

    min_corner, max_corner, corners = scene_bbox()
    center = (min_corner + max_corner) / 2.0
    extent = max_corner - min_corner
    camera = ensure_camera()
    views = parse_views(args.views)

    rendered: list[dict] = []
    for index, view_name in enumerate(views):
        setup_camera_for_view(camera, view_name, corners, center, extent, args)
        output_path = output_dir / f"{index:02d}_{view_name}.png"
        bpy.context.scene.render.filepath = str(output_path)
        bpy.ops.render.render(write_still=True)
        rendered.append(
            {
                "view": view_name,
                "path": str(output_path),
                "projection": camera.data.type.lower(),
                "ortho_scale": (
                    float(camera.data.ortho_scale)
                    if camera.data.type == "ORTHO"
                    else None
                ),
                "camera_location": [float(v) for v in camera.location],
            }
        )
        print(f"[render_blend_views] rendered {view_name}: {output_path}")

    return rendered


def main() -> None:
    args = parse_args()
    blend_path = open_blend_file(args.input)
    hidden_objects = hide_matching_objects(args)
    engine = configure_scene(args)
    rendered = render_views(args, blend_path)

    output_dir = args.output or blend_path.parent / f"{blend_path.stem}_blend_views"
    manifest_path = output_dir / "render_manifest.json"
    min_corner, max_corner, _ = scene_bbox()
    manifest = {
        "input": str(blend_path),
        "output_dir": str(output_dir),
        "engine": engine,
        "resolution": args.resolution,
        "hidden_objects": hidden_objects,
        "bbox_min": [float(v) for v in min_corner],
        "bbox_max": [float(v) for v in max_corner],
        "renders": rendered,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[render_blend_views] wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
