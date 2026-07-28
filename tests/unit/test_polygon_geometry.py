"""Unit tests for the polygon room geometry contract."""

import math
import json
import lxml.etree as ET
from pathlib import Path

import numpy as np
import pytest
import trimesh

from pydrake.all import RigidTransform

from scenesmith.agent_utils.room import SupportSurface, UniqueID
from scenesmith.agent_utils.clearance_zones import compute_openings_data
from scenesmith.agent_utils.house import HouseLayout, RoomGeometry
from scenesmith.floor_plan_agents.tools.floor_plan_tools import FloorPlanTools
from scenesmith.floor_plan_agents.stateful_floor_plan_agent import (
    StatefulFloorPlanAgent,
)
from scenesmith.floor_plan_agents.tools.geometry_cache import polygon_floor_cache_key
from scenesmith.floor_plan_agents.tools.wall_geometry import WallSpec
from scenesmith.utils.gltf_generation import create_polygon_floor_gltf
from scenesmith.utils.material import Material
from scenesmith.wall_agents.tools.wall_surface import (
    extract_wall_surfaces,
    extract_wall_surfaces_from_room_geometry,
)
from scenesmith.floor_plan_agents.tools.polygon_geometry import (
    PolygonValidationConfig,
    PolygonValidationError,
    canonicalize_polygon,
    polygon_edges,
    to_room_local_vertices,
    triangulate_polygon,
)


L_SHAPE = [(0, 0), (6, 0), (6, 2), (3, 2), (3, 5), (0, 5)]


def test_canonicalization_is_ccw_stable_and_aabb_min_normalized() -> None:
    clockwise_shifted = [(8, 6), (11, 6), (11, 3), (14, 3), (14, 1), (8, 1)]

    canonical = canonicalize_polygon(clockwise_shifted)

    assert canonical == L_SHAPE


@pytest.mark.parametrize(
    "vertices, message",
    [
        ([(0, 0), (4, 4), (0, 4), (4, 0)], "self-intersecting"),
        ([(0, 0), (1, 0), (2, 0)], "collinear"),
        ([(0, 0), (4, 0), (4, float("nan")), (0, 4)], "finite"),
    ],
)
def test_invalid_polygon_is_rejected_without_repair(vertices, message) -> None:
    with pytest.raises(PolygonValidationError, match=message):
        canonicalize_polygon(vertices)


def test_edge_frames_point_into_concave_polygon() -> None:
    edges = polygon_edges(L_SHAPE)

    assert edges[0].tangent == (1.0, 0.0)
    assert edges[0].inward_normal == (0.0, 1.0)
    assert edges[2].inward_normal == (0.0, -1.0)
    assert edges[3].inward_normal == (-1.0, 0.0)
    assert math.isclose(edges[1].yaw, math.pi / 2)


def test_triangulation_preserves_concave_area() -> None:
    triangles = triangulate_polygon(L_SHAPE)

    area = 0.0
    for triangle in triangles:
        a, b, c = (np.asarray(L_SHAPE[index], dtype=float) for index in triangle)
        area += abs(np.cross(b - a, c - a)) / 2

    assert len(triangles) == len(L_SHAPE) - 2
    assert area == pytest.approx(21.0)


def test_room_local_coordinates_use_aabb_center_not_centroid() -> None:
    assert to_room_local_vertices(L_SHAPE)[0] == (-3.0, -2.5)


def test_exact_support_surface_rejects_aabb_concavity() -> None:
    local_boundary = to_room_local_vertices(L_SHAPE)
    surface = SupportSurface(
        surface_id=UniqueID("floor"),
        bounding_box_min=np.array([-3.0, -2.5, 0.0]),
        bounding_box_max=np.array([3.0, 2.5, 0.0]),
        transform=RigidTransform(),
        exact_boundary_vertices=local_boundary,
    )

    assert surface.contains_point_2d(np.array([-2.0, 2.0]))
    assert not surface.contains_point_2d(np.array([2.0, 2.0]))
    assert surface.area == pytest.approx(21.0)


def test_validation_limits_are_configurable() -> None:
    config = PolygonValidationConfig(min_area_m2=40.0)
    with pytest.raises(PolygonValidationError, match="at least 40"):
        canonicalize_polygon(L_SHAPE, config)


def test_polygon_tool_set_is_isolated_from_rectangle_modes() -> None:
    rectangle_names = set(FloorPlanTools(HouseLayout(), mode="room").tools)
    polygon_names = set(FloorPlanTools(HouseLayout(), mode="polygon").tools)

    assert "generate_room_specs" in rectangle_names
    assert "generate_polygon_room" not in rectangle_names
    assert "generate_polygon_room" in polygon_names
    assert "set_room_polygon" in polygon_names
    assert (
        not {
            "generate_room_specs",
            "resize_room",
            "add_adjacency",
            "remove_adjacency",
            "add_open_connection",
            "remove_open_connection",
        }
        & polygon_names
    )


def test_polygon_tool_creates_stable_edge_walls_and_openings() -> None:
    layout = HouseLayout(house_prompt="An L-shaped living room")
    tools = FloorPlanTools(layout, mode="polygon")

    result = tools._generate_polygon_room_impl(
        json.dumps({"type": "living_room", "vertices": L_SHAPE})
    )

    assert result.success, result.message
    assert list(layout.boundary_labels) == [f"W{i:02d}" for i in range(6)]
    assert layout.boundary_wall_ids["W03"] == "living_room_edge_003"
    room = layout.placed_rooms[0]
    assert room.footprint_vertices == L_SHAPE
    assert len(room.walls) == 6
    assert all(wall.direction is None for wall in room.walls)
    assert room.walls[3].inward_normal == (-1.0, 0.0)

    door_result = tools._add_door_impl("W00", "center", width=0.9, height=2.1)
    window_result = tools._add_window_impl(
        "W05", "center", width=1.2, height=1.2, sill_height=0.9
    )
    assert door_result.success, door_result.message
    assert window_result.success, window_result.message
    assert room.walls[0].openings[0].opening_id == layout.doors[0].id
    assert room.walls[5].openings[0].opening_id == layout.windows[0].id


def test_polygon_layout_round_trip_preserves_stable_wall_ids() -> None:
    layout = HouseLayout()
    tools = FloorPlanTools(layout, mode="polygon")
    assert tools._generate_polygon_room_impl(
        json.dumps({"type": "room", "vertices": L_SHAPE})
    ).success

    restored = HouseLayout.from_dict(layout.to_dict())

    assert restored.room_specs[0].footprint_vertices == L_SHAPE
    assert restored.placed_rooms[0].footprint_vertices == L_SHAPE
    assert restored.placed_rooms[0].walls[2].direction is None
    assert restored.boundary_wall_ids == layout.boundary_wall_ids


def test_polygon_floor_cache_key_includes_exact_shape() -> None:
    t_shape = [(0, 0), (6, 0), (6, 2), (4, 2), (4, 5), (2, 5), (2, 2), (0, 2)]
    kwargs = {"thickness": 0.1, "material": None, "texture_scale": 0.5}
    assert polygon_floor_cache_key(L_SHAPE, **kwargs) != polygon_floor_cache_key(
        t_shape, **kwargs
    )


def test_room_geometry_hash_combines_sdf_content_and_footprint(tmp_path) -> None:
    sdf_path = tmp_path / "room.sdf"
    sdf_path.write_text("<sdf version='1.7'/>", encoding="utf-8")
    tree = ET.parse(str(sdf_path))
    l_geometry = RoomGeometry(
        sdf_tree=tree,
        sdf_path=sdf_path,
        footprint_vertices=to_room_local_vertices(L_SHAPE),
    )
    rectangle_geometry = RoomGeometry(
        sdf_tree=tree,
        sdf_path=sdf_path,
        footprint_vertices=[(-3, -2.5), (3, -2.5), (3, 2.5), (-3, 2.5)],
    )

    assert l_geometry.content_hash() != rectangle_geometry.content_hash()


def test_polygon_floor_visual_is_watertight_and_keeps_concavity(tmp_path) -> None:
    local = to_room_local_vertices(L_SHAPE)
    output = tmp_path / "floor.gltf"
    create_polygon_floor_gltf(
        vertices=local,
        triangles=triangulate_polygon(local),
        thickness=0.1,
        material=Material.from_path("materials/Wood094_1K-JPG"),
        output_path=output,
    )

    loaded = trimesh.load(output, force="mesh")
    assert loaded.is_watertight
    # A point in the L-shaped AABB notch has no top-face triangle beneath it.
    top_faces = loaded.triangles[np.isclose(loaded.triangles[:, :, 1].max(axis=1), 0.0)]
    assert top_faces.size > 0


def test_polygon_floor_collision_uses_one_convex_prism_per_triangle(tmp_path) -> None:
    local = to_room_local_vertices(L_SHAPE)
    triangles = triangulate_polygon(local)
    link = ET.Element("link")
    StatefulFloorPlanAgent._add_polygon_floor_collisions(
        link_element=link,
        vertices=local,
        triangles=triangles,
        floor_thickness=0.1,
        room_id="room",
        floors_dir=tmp_path,
    )

    collisions = link.findall("collision")
    assert len(collisions) == len(triangles)
    assert all(
        collision.find("geometry/mesh/uri") is not None for collision in collisions
    )
    assert len(list((tmp_path / "collisions").glob("*.obj"))) == len(triangles)


def test_polygon_wall_scene_object_keeps_local_bbox_yaw_and_normal() -> None:
    spec = WallSpec(
        name="room_edge_001",
        center_x=3.0,
        center_y=0.0,
        bbox_width=4.0,
        bbox_depth=0.05,
        thickness=0.05,
        yaw=math.pi / 4,
        inward_normal=(-math.sqrt(0.5), math.sqrt(0.5)),
        wall_id="room_edge_001",
    )
    agent = object.__new__(StatefulFloorPlanAgent)
    wall = agent._create_wall_objects([spec], wall_height=2.5)[0]

    assert wall.bbox_max[0] - wall.bbox_min[0] == pytest.approx(4.0)
    assert wall.metadata["wall_id"] == "room_edge_001"
    assert np.allclose(wall.metadata["inward_normal"], spec.inward_normal)
    assert wall.transform.rotation().matrix()[0, 0] == pytest.approx(math.sqrt(0.5))


def test_diagonal_polygon_wall_surfaces_and_replay_keep_edge_ids() -> None:
    vertices = [(0, 0), (5, 0), (6, 3), (3, 5), (0, 3)]
    layout = HouseLayout()
    tools = FloorPlanTools(layout, mode="polygon")
    assert tools._generate_polygon_room_impl(
        json.dumps({"type": "studio", "vertices": vertices})
    ).success
    assert tools._add_door_impl("W01", "center", width=0.9, height=2.1).success

    placed_room = layout.placed_rooms[0]
    surfaces = extract_wall_surfaces(layout, "studio", ceiling_height=2.5)
    assert len(surfaces) == len(vertices)
    assert [surface.wall_label for surface in surfaces] == [
        f"W{index:02d}" for index in range(len(vertices))
    ]
    assert all(surface.wall_direction is None for surface in surfaces)
    assert all(
        np.linalg.det(surface.transform.rotation().matrix()) == pytest.approx(1.0)
        for surface in surfaces
    )

    footprint_local = to_room_local_vertices(placed_room.footprint_vertices)
    room_geometry = RoomGeometry(
        sdf_tree=ET.ElementTree(ET.Element("sdf")),
        sdf_path=Path("/tmp/not-read-for-this-test.sdf"),
        width=placed_room.depth,
        length=placed_room.width,
        wall_height=2.5,
        wall_thickness=0.05,
        openings=compute_openings_data(
            placed_room,
            wall_height=2.5,
            door_clearance_distance=1.0,
            window_clearance_distance=0.5,
        ),
        footprint_vertices=footprint_local,
    )
    replayed = extract_wall_surfaces_from_room_geometry(room_geometry, room_id="studio")
    assert [surface.wall_id for surface in replayed] == [
        wall.wall_id for wall in placed_room.walls
    ]
    assert len(replayed[1].excluded_regions) == 1
