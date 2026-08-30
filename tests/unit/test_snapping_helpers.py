from pathlib import Path

import numpy as np
import trimesh

from omegaconf import OmegaConf
from pydrake.math import RigidTransform

from scenesmith.agent_utils.room import ObjectType, SceneObject, UniqueID
from scenesmith.furniture_agents.tools import snapping_helpers
from scenesmith.utils.mesh_loading import load_object_collision_geometry


class _NoCollisionManager:
    """Exercise the footprint guard without an earlier mesh collision."""

    def __init__(self) -> None:
        self._objs = {}

    def add_object(self, name, mesh, transform=None) -> None:
        self._objs[name] = (mesh, transform)

    def min_distance_other(self, other) -> float:
        return float("inf")


class _PenetratingCollisionManager(_NoCollisionManager):
    """Force the push-out path without relying on a local FCL installation."""

    def min_distance_other(self, other) -> float:
        return -0.1


def _north_wall() -> SceneObject:
    return SceneObject(
        object_id=UniqueID("north_wall"),
        object_type=ObjectType.WALL,
        name="north_wall",
        description="north wall",
        transform=RigidTransform(p=[0.0, 2.2, 1.35]),
        bbox_min=np.array([-2.5, -0.05, -1.35]),
        bbox_max=np.array([2.5, 0.05, 1.35]),
        immutable=True,
    )


def _object(object_id: str, position: tuple[float, float, float]) -> SceneObject:
    return SceneObject(
        object_id=UniqueID(object_id),
        object_type=ObjectType.FURNITURE,
        name=object_id,
        description=object_id,
        transform=RigidTransform(p=np.asarray(position, dtype=float)),
        geometry_path=Path(f"/{object_id}.gltf"),
        sdf_path=Path(f"/{object_id}.sdf"),
    )


def _write_collision_sdf(directory: Path, mesh: trimesh.Trimesh) -> Path:
    """Write a minimal SDF whose collision mesh is already physics-scaled."""
    mesh_path = directory / "collision.obj"
    mesh.export(mesh_path)
    sdf_path = directory / "asset.sdf"
    sdf_path.write_text(
        """<sdf version=\"1.7\">
  <model name=\"asset\">
    <link name=\"base_link\">
      <collision name=\"collision\">
        <geometry><mesh><uri>collision.obj</uri></mesh></geometry>
      </collision>
    </link>
  </model>
</sdf>
"""
    )
    return sdf_path


def test_sdf_collision_geometry_is_not_scaled_again_by_hssd_metadata(tmp_path) -> None:
    """Snap geometry must agree with the requested dimensions baked into SDF."""
    sdf_path = _write_collision_sdf(
        tmp_path,
        trimesh.creation.box(extents=(1.2, 0.6, 0.7)),
    )
    desk = _object("desk", (0.0, 0.0, 0.0))
    desk.sdf_path = sdf_path
    desk.scale_factor = 0.9230769

    meshes = load_object_collision_geometry(desk)
    bounds = np.vstack([mesh.bounds for mesh in meshes])
    extents = bounds.max(axis=0) - bounds.min(axis=0)

    np.testing.assert_allclose(extents, [1.2, 0.6, 0.7], atol=1e-6)


def test_axis_wall_snap_uses_sdf_collision_extent_not_hssd_metadata(tmp_path) -> None:
    """A furniture-to-wall snap retains the configured clearance in physics."""
    sdf_path = _write_collision_sdf(
        tmp_path,
        trimesh.creation.box(extents=(1.2, 0.6, 0.7)),
    )
    desk = _object("desk", (0.0, -0.277, 0.0))
    desk.sdf_path = sdf_path
    desk.scale_factor = 0.9230769
    wall = SceneObject(
        object_id=UniqueID("south_wall"),
        object_type=ObjectType.WALL,
        name="south_wall",
        description="south wall",
        transform=RigidTransform(p=[0.0, -1.725, 1.25]),
        bbox_min=np.array([-2.0, -0.025, -1.25]),
        bbox_max=np.array([2.0, 0.025, 1.25]),
        immutable=True,
    )
    cfg = OmegaConf.create({"snap_to_object": {"snap_margin_m": 0.01}})

    movement, _ = snapping_helpers.snap_mesh_to_aabb_along_axis(
        desk,
        wall,
        np.array([0.0, -1.0, 0.0]),
        cfg,
    )

    final_y = desk.transform.translation()[1] + movement[1]
    # The south wall's interior face is y=-1.7. With a 0.6m desk and a 1cm
    # margin, the center must stop at y=-1.39, not at the false smaller-mesh
    # result near -1.413 that penetrates Drake's wall collision.
    assert abs(final_y - (-1.39)) < 1e-6


def test_axis_snap_stops_chair_front_at_table_footprint(monkeypatch) -> None:
    chair = _object("dining_chair", (0.0, -1.0, 0.25))
    table = _object("dining_table", (0.0, 0.0, 0.75))
    chair_mesh = trimesh.creation.box(extents=(0.5, 0.5, 0.5))
    tabletop_mesh = trimesh.creation.box(extents=(2.0, 1.0, 0.1))

    def collision_geometry(obj):
        return [chair_mesh if obj is chair else tabletop_mesh]

    monkeypatch.setattr(
        snapping_helpers, "load_object_collision_geometry", collision_geometry
    )
    monkeypatch.setattr(
        snapping_helpers.trimesh.collision,
        "CollisionManager",
        _NoCollisionManager,
    )
    cfg = OmegaConf.create(
        {
            "snap_to_object": {
                "iterative_snap_step_m": 0.01,
                "max_snap_distance_m": 2.0,
            }
        }
    )

    movement, distance = snapping_helpers.snap_with_iterative_collision_check(
        obj=chair,
        target=table,
        direction=np.array([0.0, 1.0, 0.0]),
        cfg=cfg,
    )

    np.testing.assert_allclose(movement, [0.0, 0.24, 0.0], atol=1e-8)
    assert abs(distance - 0.24) < 1e-8
    chair_front_y = chair.transform.translation()[1] + movement[1] + 0.25
    assert chair_front_y < -0.5


def test_floor_furniture_wall_snap_never_changes_elevation(monkeypatch) -> None:
    """Wall AABB mid-height must not lift a floor-standing furniture object."""
    wardrobe = _object("wardrobe", (0.0, 1.5, 0.0))
    wall = _north_wall()
    wardrobe_mesh = trimesh.creation.box(extents=(1.0, 0.5, 2.0))
    monkeypatch.setattr(
        snapping_helpers, "load_object_collision_geometry", lambda _: [wardrobe_mesh]
    )
    monkeypatch.setattr(
        snapping_helpers.trimesh.collision,
        "CollisionManager",
        _PenetratingCollisionManager,
    )
    cfg = OmegaConf.create(
        {
            "snap_to_object": {
                "iterative_snap_step_m": 0.01,
                "snap_margin_m": 0.01,
            }
        }
    )

    movement, _ = snapping_helpers.snap_mesh_to_aabb(wardrobe, wall, cfg)

    assert movement[2] == 0.0
    assert abs(movement[1]) > 1e-6


def test_floor_furniture_wall_closest_point_snap_never_changes_elevation(
    monkeypatch,
) -> None:
    """The separated closest-point path must also remain in the floor plane."""
    wardrobe = _object("wardrobe", (0.0, 1.5, 0.0))
    wall = _north_wall()
    wardrobe_mesh = trimesh.creation.box(extents=(1.0, 0.5, 2.0))
    monkeypatch.setattr(
        snapping_helpers, "load_object_collision_geometry", lambda _: [wardrobe_mesh]
    )
    monkeypatch.setattr(
        snapping_helpers.trimesh.collision,
        "CollisionManager",
        _NoCollisionManager,
    )
    cfg = OmegaConf.create(
        {
            "snap_to_object": {
                "iterative_snap_step_m": 0.01,
                "snap_margin_m": 0.01,
            }
        }
    )

    movement, _ = snapping_helpers.snap_mesh_to_aabb(wardrobe, wall, cfg)

    assert movement[2] == 0.0
    assert abs(movement[1]) > 1e-6


def test_high_poly_target_uses_bounded_vertex_pair_for_snap_direction(
    monkeypatch,
) -> None:
    """A small source must not query every face on a high-poly target."""
    source = _object("nightstand", (-2.0, 0.0, 0.0))
    target = _object("bed", (2.0, 0.0, 0.0))
    source_mesh = trimesh.creation.box(extents=(0.5, 0.5, 0.5))
    target_mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    meshes = iter((source_mesh, target_mesh))

    monkeypatch.setattr(
        snapping_helpers,
        "load_object_collision_geometry",
        lambda _obj: [next(meshes).copy()],
    )
    monkeypatch.setattr(
        snapping_helpers.trimesh.proximity,
        "closest_point",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("high-poly target should not use full mesh proximity")
        ),
    )
    cfg = OmegaConf.create({"snap_to_object": {"max_sample_vertices": 64}})

    direction = snapping_helpers.compute_snap_direction_mesh_to_mesh(
        source, target, cfg
    )

    assert direction[0] > 0.9
    assert abs(direction[1]) < 0.1
    assert abs(direction[2]) < 0.1


def test_mesh_to_mesh_direction_uses_collision_geometry_not_visual_mesh(
    monkeypatch,
) -> None:
    """The direction source must match the physical snap/collision geometry."""
    source = _object("source", (0.0, 0.0, 0.0))
    target = _object("target", (3.0, 0.0, 0.0))
    source.scale_factor = 0.5
    target.scale_factor = 0.5
    source_mesh = trimesh.creation.box(extents=(1.2, 0.8, 0.8))
    target_mesh = trimesh.creation.box(extents=(1.2, 0.8, 0.8))
    meshes = iter((source_mesh, target_mesh))
    monkeypatch.setattr(
        snapping_helpers,
        "load_object_collision_geometry",
        lambda _obj: [next(meshes).copy()],
    )
    cfg = OmegaConf.create({"snap_to_object": {"max_sample_vertices": 64}})

    direction = snapping_helpers.compute_snap_direction_mesh_to_mesh(
        source, target, cfg
    )

    np.testing.assert_allclose(direction, [1.0, 0.0, 0.0], atol=1e-7)
