"""Render unlabeled final views with the furniture-stage renderer.

Run with the project's Blender-enabled Python::

    .venv/bin/python scripts/render_critic_final_views.py -- \
      /path/to/outputs/critic_probe/<run_id>

The script loads each complete ``combined_house/house.blend`` and reuses
``BlenderRenderer.render_agent_observation_views`` for camera placement,
lighting, EEVEE settings, and partial-wall handling.  Unlike the furniture
stage, the loaded scene contains every generated stage object.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import MethodType, SimpleNamespace

import bpy
from mathutils import Vector
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scenesmith.agent_utils.blender.camera_utils import (
    calculate_camera_distance,
    configure_metric_camera,
)
from scenesmith.agent_utils.blender.params import RenderParams
from scenesmith.agent_utils.blender.render_dataclasses import OverlayRenderingSetup
from scenesmith.agent_utils.blender.render_settings import (
    apply_render_settings,
    setup_metric_world,
)
from scenesmith.agent_utils.blender.renderer import BlenderRenderer
from scenesmith.agent_utils.blender.scene_utils import compute_scene_bounds
from scenesmith.agent_utils.blender.wall_utils import looks_like_wall
from scenesmith.agent_utils.blender import renderer as renderer_module


def parse_args() -> argparse.Namespace:
    raw_argv = sys.argv[1:]
    if "--" in raw_argv:
        separator = raw_argv.index("--")
        # Blender places one separator before the script arguments; direct
        # venv execution places it after the options and input path.
        argv = raw_argv[1:] if separator == 0 else raw_argv[:separator] + raw_argv[separator + 1 :]
    else:
        argv = raw_argv
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", type=Path, help="Run directory, scene directory, or .blend file"
    )
    parser.add_argument("--resolution", type=int, default=1536)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--side", choices=("north", "east", "south", "west"), default=None)
    parser.add_argument("--include-shared-base", action="store_true")
    parser.add_argument(
        "--parallelism",
        type=int,
        default=8,
        help="Number of independent render processes for a run directory (default: 8)",
    )
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_params(blend_path: Path, resolution: int) -> RenderParams:
    fov_y = math.radians(45.0)
    return RenderParams(
        scene=blend_path,
        scene_sha256=sha256(blend_path),
        image_type="color",
        width=resolution,
        height=resolution,
        near=0.01,
        far=100000.0,
        focal_x=resolution / 2.0,
        focal_y=resolution / 2.0,
        fov_x=fov_y,
        fov_y=fov_y,
        center_x=resolution / 2.0,
        center_y=resolution / 2.0,
    )


def install_loaded_blend_setup(renderer: BlenderRenderer, blend_path: Path) -> None:
    """Adapt production setup to a loaded .blend instead of an incoming GLTF.

    The remainder of ``render_agent_observation_views`` is intentionally used
    unchanged.  This keeps its camera/view generation, wall visibility, world,
    and post-processing behavior aligned with furniture-stage renders.
    """

    def setup_loaded_blend(
        self: BlenderRenderer,
        params: RenderParams,
        view_size: int | None,
        margin_scale: float = 1.8,
    ) -> OverlayRenderingSetup:
        mesh_objects = [
            obj
            for obj in bpy.context.scene.objects
            if obj.type == "MESH" and not obj.hide_render
        ]
        if not mesh_objects:
            raise RuntimeError(f"No visible mesh objects in {blend_path}")

        # Production renderer expects a collection-like object for scene bounds
        # and floor bounds. A lightweight proxy avoids moving objects between
        # collections in the user's loaded .blend.
        self._client_objects = SimpleNamespace(objects=mesh_objects)
        bbox_center, max_dim = compute_scene_bounds(self._client_objects)
        camera_obj = configure_metric_camera(params=params)
        apply_render_settings(params=params, view_size=view_size)
        setup_metric_world()

        scene = bpy.context.scene
        scene.render.engine = "BLENDER_EEVEE_NEXT"
        scene.render.film_transparent = True
        scene.render.image_settings.color_mode = "RGBA"
        scene.render.image_settings.color_depth = "8"
        scene.render.resolution_percentage = 100

        # Match _setup_overlay_rendering in the furniture-stage renderer.
        try:
            scene.eevee.taa_render_samples = self._taa_samples
        except AttributeError:
            pass
        for attribute, value in (
            ("use_gtao", False),
            ("use_bloom", False),
            ("use_ssr", False),
            ("use_volumetric_shadows", False),
            ("use_shadows", False),
        ):
            try:
                setattr(scene.eevee, attribute, value)
            except AttributeError:
                pass
        for light in bpy.data.lights:
            try:
                light.use_shadow = False
            except AttributeError:
                pass

        camera_distance = calculate_camera_distance(
            camera_obj=camera_obj, max_dim=max_dim, margin_scale=margin_scale
        )
        return OverlayRenderingSetup(
            camera_obj=camera_obj,
            bbox_center=bbox_center,
            max_dim=max_dim,
            camera_distance=camera_distance,
        )

    renderer._setup_overlay_rendering = MethodType(setup_loaded_blend, renderer)


def install_geometry_wall_visibility() -> object:
    """Hide only the two walls between the side camera and room.

    Production GLTF renders receive wall normals from RoomScene metadata. A
    combined .blend can have an arbitrary world-space room origin and generic
    GLTF object names, so derive the inward normal from the actual wall centers.
    This preserves the production ``camera_direction · inward_normal`` rule.
    """
    wall_objects = [
        obj
        for obj in bpy.context.scene.objects
        if obj.type == "MESH" and looks_like_wall(obj)
    ]
    if not wall_objects:
        return None
    centers = []
    for obj in wall_objects:
        points = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
        centers.append(sum(points, Vector((0.0, 0.0, 0.0))) / len(points))
    room_center = sum(centers, Vector((0.0, 0.0, 0.0))) / len(centers)
    original = renderer_module.should_hide_wall

    def hide_wall(obj, camera_direction, is_top_view, wall_normals):
        if is_top_view or obj not in wall_objects:
            return False
        points = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
        center = sum(points, Vector((0.0, 0.0, 0.0))) / len(points)
        inward = Vector((room_center.x - center.x, room_center.y - center.y, 0.0))
        if inward.length == 0:
            return False
        return camera_direction.dot(inward.normalized()) > 0.1

    renderer_module.should_hide_wall = hide_wall
    return original


def render_one(blend_path: Path, args: argparse.Namespace) -> Path:
    bpy.ops.wm.open_mainfile(filepath=str(blend_path))
    for obj in bpy.context.scene.objects:
        if "ceiling" in obj.name.lower():
            obj.hide_render = True
            obj.hide_viewport = True

    renderer = BlenderRenderer()
    renderer._taa_samples = args.samples
    install_loaded_blend_setup(renderer, blend_path)
    original_wall_visibility = install_geometry_wall_visibility()

    resolution = args.resolution
    params = make_params(blend_path, resolution)
    annotations = OmegaConf.create(
        {
            "enable_set_of_mark_labels": False,
            "enable_bounding_boxes": False,
            "enable_direction_arrows": False,
            "enable_partial_walls": True,
            "enable_coordinate_grid": False,
            "show_coordinate_frame": False,
            "enable_support_surface_debug": False,
            "enable_convex_hull_debug": False,
            "rendering_mode": "furniture_selection",
            "annotate_object_types": [],
        }
    )
    side_start_azimuth = None
    if args.side:
        side_start_azimuth = {"north": 90.0, "east": 0.0, "south": 270.0, "west": 180.0}[args.side]

    temp_dir = Path(tempfile.mkdtemp(prefix="critic_final_furniture_renderer_"))
    try:
        rendered = renderer.render_agent_observation_views(
            params=params,
            output_dir=temp_dir,
            layout="top_plus_sides",
            top_view_width=resolution,
            top_view_height=resolution,
            side_view_count=1,
            side_view_width=resolution,
            side_view_height=resolution,
            scene_objects=[],
            annotations=annotations,
            wall_normals={},
            include_vertical_views=True,
            side_view_start_azimuth_degrees=side_start_azimuth,
        )
        output_dir = blend_path.parent.parent / "critic_final_views"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_paths = {"0_top": output_dir / "00_top.png", "0_side": output_dir / "01_side.png"}
        for source in rendered:
            stem = source.stem
            if stem in output_paths:
                shutil.copy2(source, output_paths[stem])
        manifest = {
            "input": str(blend_path),
            "renderer": "BlenderRenderer.render_agent_observation_views",
            "resolution": resolution,
            "taa_samples": args.samples,
            "side": args.side or "production_default_corner",
            "renders": [str(path) for path in output_paths.values()],
        }
        (output_dir / "render_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        return output_dir
    finally:
        if original_wall_visibility is not None:
            renderer_module.should_hide_wall = original_wall_visibility
        shutil.rmtree(temp_dir, ignore_errors=True)


def main() -> None:
    args = parse_args()
    files = blend_files(args.input, args.include_shared_base)
    if not files:
        raise SystemExit(f"No combined_house/house.blend found below {args.input}")

    if len(files) > 1 and args.parallelism > 1:
        process_count = min(args.parallelism, len(files))
        print(
            f"[render_critic_final_views] rendering {len(files)} scenes with "
            f"{process_count} processes"
        )
        child_processes = []
        return_codes = []
        for blend_path in files:
            child_args = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--resolution",
                str(args.resolution),
                "--samples",
                str(args.samples),
                "--parallelism",
                "1",
            ]
            if args.side:
                child_args.extend(("--side", args.side))
            if args.include_shared_base:
                child_args.append("--include-shared-base")
            child_args.extend(("--", str(blend_path)))
            child_processes.append(subprocess.Popen(child_args, env=os.environ.copy()))
            if len(child_processes) >= process_count:
                return_codes.append(child_processes.pop(0).wait())

        return_codes.extend(process.wait() for process in child_processes)
        if any(code != 0 for code in return_codes):
            raise SystemExit(max(code for code in return_codes))
        return

    for blend_path in files:
        print(f"[render_critic_final_views] {blend_path}")
        print(f"[render_critic_final_views] wrote {render_one(blend_path, args)}")


if __name__ == "__main__":
    main()
