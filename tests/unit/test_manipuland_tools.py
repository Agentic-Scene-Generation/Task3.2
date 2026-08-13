import json
import math
import shutil
import tempfile
import unittest

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import trimesh

from omegaconf import OmegaConf
from pydrake.all import RigidTransform, RollPitchYaw

from scenesmith.agent_utils.asset_manager import AssetManager
from scenesmith.agent_utils.placement_noise import PlacementNoiseMode
from scenesmith.agent_utils.room import (
    ObjectType,
    RoomScene,
    SceneObject,
    SupportSurface,
    UniqueID,
)
from scenesmith.manipuland_agents.tools.manipuland_tools import ManipulandTools


class TestManipulandTools(unittest.TestCase):
    """Test ManipulandTools class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())

        # Load configuration from actual config file.
        config_path = (
            Path(__file__).parent.parent.parent
            / "configurations/manipuland_agent/base_manipuland_agent.yaml"
        )
        self.cfg = OmegaConf.load(config_path)

        # Create mock scene.
        self.mock_scene = Mock(spec=RoomScene)
        self.mock_scene.objects = {}
        self.mock_scene.action_log_path = None

        # Create mock asset manager.
        self.mock_asset_manager = Mock(spec=AssetManager)

        # Create mock support surface.
        self.mock_surface = Mock(spec=SupportSurface)
        self.mock_surface.surface_id = UniqueID("test_surface_001")
        self.mock_surface.bounding_box_min = np.array([-0.5, -0.5, 0.0])
        self.mock_surface.bounding_box_max = np.array(
            [0.5, 0.5, 0.5]
        )  # 0.5m clearance.
        self.mock_surface.contains_point_2d = Mock(return_value=True)
        self.mock_surface.to_world_pose = Mock(
            return_value=RigidTransform(p=[0.0, 0.0, 1.0])
        )

        # Create support surfaces dict (multi-surface API).
        self.support_surfaces = {str(self.mock_surface.surface_id): self.mock_surface}

        # Create manipuland tools instance.
        self.manipuland_tools = ManipulandTools(
            scene=self.mock_scene,
            asset_manager=self.mock_asset_manager,
            cfg=self.cfg,
            current_furniture_id=UniqueID("furniture_001"),
            support_surfaces=self.support_surfaces,
        )

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_set_noise_profile_natural(self):
        """Test set_noise_profile with natural mode."""
        self.manipuland_tools.set_noise_profile(PlacementNoiseMode.NATURAL)
        self.assertEqual(
            self.manipuland_tools.active_noise_profile,
            self.cfg.placement_noise.natural_profile,
        )

    def test_set_noise_profile_perfect(self):
        """Test set_noise_profile with perfect mode."""
        self.manipuland_tools.set_noise_profile(PlacementNoiseMode.PERFECT)
        self.assertEqual(
            self.manipuland_tools.active_noise_profile,
            self.cfg.placement_noise.perfect_profile,
        )

    def test_dining_place_setting_alignment_tool_is_available(self):
        # 2026-07-23 修改原因：餐位对齐 critic 必须能调用实际修复工具，不能只输出
        # 文字建议，否则盘子虽齐全仍会随机散落在餐桌上。
        self.assertIn("align_dining_place_settings", self.manipuland_tools.tools)

    @patch(
        "scenesmith.manipuland_agents.tools.manipuland_tools."
        "evaluate_dining_place_setting_alignment"
    )
    @patch(
        "scenesmith.manipuland_agents.tools.manipuland_tools.room_scene_to_case_pack"
    )
    def test_dining_alignment_builds_object_index_before_overlap_baseline(
        self, mock_case_pack, mock_evaluate
    ):
        """Assignments must reach transactional overlap checks without scope errors."""
        table = SceneObject(
            object_id=UniqueID("furniture_001"),
            object_type=ObjectType.FURNITURE,
            name="dining_table",
            description="Test dining table",
            transform=RigidTransform(),
            support_surfaces=[self.mock_surface],
        )
        self.mock_scene.objects = {table.object_id: table}
        self.mock_scene.get_object.return_value = table
        self.mock_scene.to_state_dict.return_value = {}
        alignment = {
            "primary_object": "furniture_001",
            "label": "fail",
            "diagnostics": {
                "assignments": [
                    {
                        "anchor_id": "missing_plate",
                        "companion_ids": [],
                        "recommended_anchor_center_xy_m": [0.0, 0.0],
                    }
                ]
            },
        }
        mock_case_pack.return_value = {}
        mock_evaluate.return_value = [alignment]

        result = json.loads(
            self.manipuland_tools._align_dining_place_settings_impl(
                table_id="furniture_001"
            )
        )

        self.assertIn("restored", result)
        self.assertTrue(result["restored"])

    def test_adjacent_dining_table_strips_are_coalesced(self):
        # 2026-07-23 修改原因：桌面内部的 HSSD seam 不应把长边中点餐位推到
        # S8 或 S9 的一侧；相邻条带合并后仍使用真实外轮廓做边界校验。
        def surface(surface_id: str, x: float) -> SupportSurface:
            return SupportSurface(
                surface_id=UniqueID(surface_id),
                bounding_box_min=np.array([-0.5, -0.4, 0.01]),
                bounding_box_max=np.array([0.5, 0.4, 0.5]),
                transform=RigidTransform(p=[x, 0.0, 0.8]),
            )

        merged = ManipulandTools._coalesce_adjacent_support_surfaces(
            [surface("S_8", 0.0), surface("S_9", 1.0)]
        )

        self.assertEqual(len(merged), 1)
        logical_surface = merged[0]
        self.assertTrue(logical_surface.contains_point_2d(np.array([0.0, 0.0])))
        self.assertTrue(logical_surface.contains_point_2d(np.array([0.99, 0.0])))

    def test_disconnected_support_surfaces_are_not_coalesced(self):
        def surface(surface_id: str, x: float) -> SupportSurface:
            return SupportSurface(
                surface_id=UniqueID(surface_id),
                bounding_box_min=np.array([-0.3, -0.3, 0.01]),
                bounding_box_max=np.array([0.3, 0.3, 0.5]),
                transform=RigidTransform(p=[x, 0.0, 0.8]),
            )

        result = ManipulandTools._coalesce_adjacent_support_surfaces(
            [surface("S_left", 0.0), surface("S_right", 1.0)]
        )

        self.assertEqual(
            [str(surface.surface_id) for surface in result], ["S_left", "S_right"]
        )

    @patch.object(ManipulandTools, "_dining_position_is_valid", return_value=True)
    def test_dining_target_uses_critic_selected_segmented_surface(self, _mock_valid):
        # 2026-07-23 修改原因：真实 HSSD 餐桌可能由多个连续 surface strip 组成。
        # critic 已指定餐位 strip 时，工具必须保留该指定面，不能折叠到中央条带。
        def surface(surface_id: str, y: float) -> SupportSurface:
            return SupportSurface(
                surface_id=UniqueID(surface_id),
                bounding_box_min=np.array([-0.8, -0.1, 0.0]),
                bounding_box_max=np.array([0.8, 0.1, 0.5]),
                transform=RigidTransform(p=[0.0, y, 0.8]),
            )

        north = surface("S_north", 0.22)
        center = surface("S_center", 0.0)
        plate = Mock()
        plate.bbox_min = np.array([-0.12, -0.12, 0.0])
        plate.bbox_max = np.array([0.12, 0.12, 0.03])
        plate.placement_info = Mock(rotation_2d=0.0)

        selected = self.manipuland_tools._select_dining_surface_position(
            surface_map={"S_center": center, "S_north": north},
            scene_object=plate,
            target_xy=(0.0, 0.24),
            preferred_surface_id="S_north",
        )

        self.assertIsNotNone(selected)
        selected_surface, selected_position = selected
        self.assertEqual(str(selected_surface.surface_id), "S_north")
        self.assertLess(abs(float(selected_position[1])), 0.1)

    @patch.object(ManipulandTools, "_select_dining_surface_position")
    def test_dining_target_shifts_clear_of_settled_setting(self, mock_select):
        surface = SupportSurface(
            surface_id=UniqueID("S_table"),
            bounding_box_min=np.array([-1.0, -1.0, 0.0]),
            bounding_box_max=np.array([1.0, 1.0, 0.5]),
            transform=RigidTransform(),
        )
        candidate = Mock()
        candidate.bbox_min = np.array([-0.1, -0.1, 0.0])
        candidate.bbox_max = np.array([0.1, 0.1, 0.03])
        candidate.scale_factor = 1.0
        candidate.transform = RigidTransform()
        candidate.placement_info = None
        occupied = Mock()
        occupied.bbox_min = np.array([-0.1, -0.1, 0.0])
        occupied.bbox_max = np.array([0.1, 0.1, 0.03])
        occupied.scale_factor = 1.0
        occupied.transform = RigidTransform(p=[0.0, 0.0, 0.8])

        def select_position(**kwargs):
            return surface, np.asarray(kwargs["target_xy"], dtype=float)

        mock_select.side_effect = select_position
        selected = self.manipuland_tools._select_clear_dining_surface_position(
            surface_map={"S_table": surface},
            scene_object=candidate,
            target_xy=(0.0, 0.0),
            occupied_objects=[occupied],
        )

        self.assertIsNotNone(selected)
        _surface, position = selected
        self.assertGreater(np.linalg.norm(position), 0.0)
        self.assertFalse(
            self.manipuland_tools._dining_oriented_footprints_overlap(
                candidate,
                position,
                occupied,
                np.asarray(occupied.transform.translation()[:2]),
            )
        )

    @patch.object(ManipulandTools, "_select_dining_surface_position")
    def test_dining_target_avoids_overlap_with_current_setting(self, mock_select):
        surface = SupportSurface(
            surface_id=UniqueID("S_table"),
            bounding_box_min=np.array([-1.0, -1.0, 0.0]),
            bounding_box_max=np.array([1.0, 1.0, 0.5]),
            transform=RigidTransform(),
        )
        candidate = Mock()
        candidate.object_id = "cutlery_0"
        candidate.bbox_min = np.array([-0.1, -0.1, 0.0])
        candidate.bbox_max = np.array([0.1, 0.1, 0.03])
        candidate.scale_factor = 1.0
        candidate.transform = RigidTransform()
        candidate.placement_info = None
        anchor = Mock()
        anchor.object_id = "dinner_plate_0"
        anchor.bbox_min = np.array([-0.1, -0.1, 0.0])
        anchor.bbox_max = np.array([0.1, 0.1, 0.03])
        anchor.scale_factor = 1.0
        anchor.transform = RigidTransform(p=[0.0, 0.0, 0.8])
        mock_select.side_effect = lambda **kwargs: (
            surface,
            np.asarray(kwargs["target_xy"], dtype=float),
        )

        selected = self.manipuland_tools._select_clear_dining_surface_position(
            surface_map={"S_table": surface},
            scene_object=candidate,
            target_xy=(0.0, 0.0),
            occupied_objects=[anchor],
        )

        self.assertIsNotNone(selected)
        _surface, position = selected
        self.assertFalse(
            self.manipuland_tools._dining_oriented_footprints_overlap(
                candidate,
                position,
                anchor,
                np.asarray(anchor.transform.translation()[:2]),
            )
        )

    @patch.object(ManipulandTools, "_select_dining_surface_position")
    def test_thin_dining_companion_search_uses_long_side_within_lane(self, mock_select):
        surface = SupportSurface(
            surface_id=UniqueID("S_table"),
            bounding_box_min=np.array([-1.0, -1.0, 0.0]),
            bounding_box_max=np.array([1.0, 1.0, 0.5]),
            transform=RigidTransform(),
        )
        cutlery = Mock()
        cutlery.object_id = "cutlery_0"
        cutlery.bbox_min = np.array([-0.015, -0.075, 0.0])
        cutlery.bbox_max = np.array([0.015, 0.075, 0.02])
        cutlery.scale_factor = 1.0
        cutlery.transform = RigidTransform()
        cutlery.placement_info = None
        plate = Mock()
        plate.object_id = "plate_0"
        plate.bbox_min = np.array([-0.10, -0.10, 0.0])
        plate.bbox_max = np.array([0.10, 0.10, 0.03])
        plate.scale_factor = 1.0
        plate.transform = RigidTransform()
        mock_select.side_effect = lambda **kwargs: (
            surface,
            np.asarray(kwargs["target_xy"], dtype=float),
        )

        selected = self.manipuland_tools._select_clear_dining_surface_position(
            surface_map={"S_table": surface},
            scene_object=cutlery,
            target_xy=(0.0, 0.0),
            occupied_objects=[plate],
            search_axes=((1.0, 0.0), (0.0, 1.0)),
            lane_constraint=((0.0, 0.0), (1.0, 0.0), 0.16),
        )

        self.assertIsNotNone(selected)
        _surface, position = selected
        self.assertGreater(abs(float(position[0])), 0.08)
        self.assertLessEqual(abs(float(position[0])), 0.16)
        self.assertAlmostEqual(float(position[1]), 0.0)

    def test_dining_oriented_clearance_does_not_treat_cutlery_as_circle(self):
        plate = Mock()
        plate.bbox_min = np.array([-0.135, -0.135, 0.0])
        plate.bbox_max = np.array([0.135, 0.135, 0.03])
        plate.scale_factor = 1.0
        plate.transform = RigidTransform()
        cutlery = Mock()
        cutlery.bbox_min = np.array([-0.015, -0.11, 0.0])
        cutlery.bbox_max = np.array([0.015, 0.11, 0.02])
        cutlery.scale_factor = 1.0
        cutlery.transform = RigidTransform()

        self.assertFalse(
            self.manipuland_tools._dining_oriented_footprints_overlap(
                cutlery,
                np.array([0.17, 0.0]),
                plate,
                np.array([0.0, 0.0]),
            )
        )
        self.assertTrue(
            self.manipuland_tools._dining_oriented_footprints_overlap(
                cutlery,
                np.array([0.14, 0.0]),
                plate,
                np.array([0.0, 0.0]),
            )
        )

    def test_dining_candidate_clearance_uses_final_target_rotation(self):
        plate = Mock()
        plate.bbox_min = np.array([-0.135, -0.135, 0.0])
        plate.bbox_max = np.array([0.135, 0.135, 0.03])
        plate.scale_factor = 1.0
        plate.transform = RigidTransform()
        cutlery = Mock()
        cutlery.bbox_min = np.array([-0.015, -0.11, 0.0])
        cutlery.bbox_max = np.array([0.015, 0.11, 0.02])
        cutlery.scale_factor = 1.0
        cutlery.transform = RigidTransform()

        candidate_xy = np.array([0.17, 0.0])
        self.assertFalse(
            self.manipuland_tools._dining_oriented_footprints_overlap(
                cutlery,
                candidate_xy,
                plate,
                np.array([0.0, 0.0]),
            )
        )
        self.assertTrue(
            self.manipuland_tools._dining_oriented_footprints_overlap(
                cutlery,
                candidate_xy,
                plate,
                np.array([0.0, 0.0]),
                first_transform=RigidTransform(
                    rpy=RollPitchYaw(0.0, 0.0, math.pi / 2.0),
                    p=[candidate_xy[0], candidate_xy[1], 0.0],
                ),
            )
        )

    def test_move_manipuland_out_of_bounds_fails(self):
        """Test that move_manipuland fails when position is out of bounds."""
        # Create a mock object.
        object_id = UniqueID("manipuland_001")
        mock_object = Mock()
        mock_object.name = "Test Object"
        mock_object.placement_info = Mock()
        mock_object.placement_info.position_2d = np.array([0.0, 0.0])
        mock_object.placement_info.rotation_2d = 0.0
        mock_object.placement_info.parent_surface_id = self.mock_surface.surface_id

        self.mock_scene.get_object = Mock(return_value=mock_object)

        # Set surface to reject the position.
        self.mock_surface.contains_point_2d = Mock(return_value=False)

        # Try to move to out-of-bounds position.
        result_json = self.manipuland_tools._move_manipuland_impl(
            object_id=str(object_id),
            surface_id=str(self.mock_surface.surface_id),
            position_x=10.0,
            position_z=10.0,
        )

        # Parse JSON to check error.
        result_dict = json.loads(result_json)
        self.assertFalse(result_dict["success"])
        self.assertEqual(result_dict["error_type"], "position_out_of_bounds")

    def test_move_manipuland_nonexistent_object_fails(self):
        """Test that move_manipuland fails when object doesn't exist."""
        self.mock_scene.get_object = Mock(return_value=None)

        object_id = "nonexistent_object"
        result_json = self.manipuland_tools._move_manipuland_impl(
            object_id=object_id,
            surface_id=str(self.mock_surface.surface_id),
            position_x=0.1,
            position_z=0.1,
        )

        # Parse JSON to check error.
        result_dict = json.loads(result_json)
        self.assertFalse(result_dict["success"])
        self.assertEqual(result_dict["error_type"], "object_not_found")

    @patch(
        "scenesmith.manipuland_agents.tools.manipuland_tools.intent_contract_constraints_for_scene"
    )
    @patch(
        "scenesmith.manipuland_agents.tools.manipuland_tools.room_scene_to_case_pack"
    )
    def test_remove_required_manipuland_is_rejected(
        self, mock_case_pack, mock_constraints
    ):
        required = Mock()
        required.object_type = ObjectType.MANIPULAND
        required.name = "plate"
        self.mock_scene.get_object.return_value = required
        mock_case_pack.return_value = {
            "scene_geometry": {"objects": [{"id": "plate_0", "category": "plate"}]}
        }
        mock_constraints.return_value = [
            {
                "relation": "required_count",
                "strength": "hard",
                "subjects": {"category": "plate", "count": 1},
            }
        ]

        result = json.loads(self.manipuland_tools._remove_manipuland_impl("plate_0"))

        self.assertFalse(result["success"])
        self.assertEqual(result["error_type"], "invalid_operation")
        self.mock_scene.remove_object.assert_not_called()

    @patch.object(
        ManipulandTools, "_validate_convex_hull_footprint", return_value=(True, None)
    )
    @patch.object(ManipulandTools, "_is_top_surface", return_value=True)
    def test_move_manipuland_no_movement_fails(self, mock_is_top, mock_validate):
        """Test that move_manipuland fails when trying to move to same position."""
        # Create a mock object at position (0.1, 0.1).
        object_id = UniqueID("manipuland_001")
        current_position = np.array([0.1, 0.1])
        current_rotation = math.radians(45.0)

        mock_object = Mock()
        mock_object.name = "Test Object"
        mock_object.bbox_min = np.array([0.0, 0.0, 0.0])
        mock_object.bbox_max = np.array(
            [0.1, 0.1, 0.1]
        )  # 0.1m height, fits in 0.5m clearance.
        mock_object.placement_info = Mock()
        mock_object.placement_info.position_2d = current_position
        mock_object.placement_info.rotation_2d = current_rotation
        mock_object.placement_info.parent_surface_id = self.mock_surface.surface_id

        self.mock_scene.get_object = Mock(return_value=mock_object)

        # Try to move to same position.
        result_json = self.manipuland_tools._move_manipuland_impl(
            object_id=str(object_id),
            surface_id=str(self.mock_surface.surface_id),
            position_x=0.1,
            position_z=0.1,
            rotation_degrees=45.0,
        )

        # Parse JSON to check error.
        result_dict = json.loads(result_json)
        self.assertFalse(result_dict["success"])
        self.assertEqual(result_dict["error_type"], "no_movement")

    @patch.object(
        ManipulandTools, "_validate_convex_hull_footprint", return_value=(True, None)
    )
    @patch.object(ManipulandTools, "_is_top_surface", return_value=True)
    def test_move_manipuland_no_placement_info_fails(self, mock_is_top, mock_validate):
        """Test that move_manipuland fails when object has no placement_info."""
        # Create a mock object without placement_info.
        object_id = UniqueID("manipuland_001")
        mock_object = Mock()
        mock_object.name = "Test Object"
        mock_object.bbox_min = np.array([0.0, 0.0, 0.0])
        mock_object.bbox_max = np.array(
            [0.1, 0.1, 0.1]
        )  # 0.1m height, fits in 0.5m clearance.
        mock_object.placement_info = None

        self.mock_scene.get_object = Mock(return_value=mock_object)

        # Try to move object.
        result_json = self.manipuland_tools._move_manipuland_impl(
            object_id=str(object_id),
            surface_id=str(self.mock_surface.surface_id),
            position_x=0.1,
            position_z=0.1,
        )

        # Parse JSON to check error.
        result_dict = json.loads(result_json)
        self.assertFalse(result_dict["success"])
        self.assertEqual(result_dict["error_type"], "invalid_operation")
        self.assertIn("no placement info", result_dict["message"].lower())

    def test_move_manipuland_invalid_surface_fails(self):
        """Test that move_manipuland fails when surface_id doesn't exist."""
        # Create a mock object on a known surface.
        object_id = UniqueID("manipuland_001")
        mock_object = Mock()
        mock_object.name = "Test Object"
        mock_object.placement_info = Mock()
        mock_object.placement_info.position_2d = np.array([0.0, 0.0])
        mock_object.placement_info.rotation_2d = 0.0
        mock_object.placement_info.parent_surface_id = self.mock_surface.surface_id

        self.mock_scene.get_object = Mock(return_value=mock_object)

        # Try to move object to non-existent surface.
        result_json = self.manipuland_tools._move_manipuland_impl(
            object_id=str(object_id),
            surface_id="nonexistent_surface_id",
            position_x=0.1,
            position_z=0.1,
        )

        # Parse JSON to check error.
        result_dict = json.loads(result_json)
        self.assertFalse(result_dict["success"])
        self.assertEqual(result_dict["error_type"], "invalid_surface")

    @patch.object(
        ManipulandTools, "_validate_convex_hull_footprint", return_value=(True, None)
    )
    @patch.object(ManipulandTools, "_is_top_surface", return_value=True)
    @patch("scenesmith.manipuland_agents.tools.manipuland_tools.apply_placement_noise")
    def test_move_manipuland_applies_noise(
        self, mock_apply_noise, mock_is_top, mock_validate
    ):
        """Test that move_manipuland applies placement noise."""
        # Create a mock object.
        object_id = UniqueID("manipuland_001")
        mock_object = Mock()
        mock_object.name = "Test Object"
        mock_object.bbox_min = np.array([0.0, 0.0, 0.0])
        mock_object.bbox_max = np.array(
            [0.1, 0.1, 0.1]
        )  # 0.1m height, fits in 0.5m clearance.
        mock_object.placement_info = Mock()
        mock_object.placement_info.position_2d = np.array([0.0, 0.0])
        mock_object.placement_info.rotation_2d = 0.0
        mock_object.placement_info.parent_surface_id = self.mock_surface.surface_id

        self.mock_scene.get_object = Mock(return_value=mock_object)

        # Set up noise mock to return a slightly modified transform.
        def noise_side_effect(transform, **kwargs):
            return RigidTransform(p=[0.1, 0.1, 1.0])

        mock_apply_noise.side_effect = noise_side_effect

        # Move to new position.
        result_json = self.manipuland_tools._move_manipuland_impl(
            object_id=str(object_id),
            surface_id=str(self.mock_surface.surface_id),
            position_x=0.2,
            position_z=0.2,
        )

        # Verify noise was applied.
        mock_apply_noise.assert_called_once()
        call_kwargs = mock_apply_noise.call_args[1]
        self.assertEqual(
            call_kwargs["position_xy_std_meters"],
            self.cfg.placement_noise.natural_profile.position_xy_std_meters,
        )
        self.assertEqual(
            call_kwargs["rotation_yaw_std_degrees"],
            self.cfg.placement_noise.natural_profile.rotation_yaw_std_degrees,
        )

    @patch.object(
        ManipulandTools, "_validate_convex_hull_footprint", return_value=(True, None)
    )
    @patch.object(ManipulandTools, "_is_top_surface", return_value=True)
    @patch("scenesmith.manipuland_agents.tools.manipuland_tools.apply_placement_noise")
    def test_placement_applies_noise(
        self, mock_apply_noise, mock_is_top, mock_validate
    ):
        """Test that place_manipuland_on_surface applies placement noise."""
        # Create a mock asset.
        asset_id = UniqueID("asset_001")
        mock_asset = Mock()
        mock_asset.object_id = asset_id
        mock_asset.name = "Test Manipuland"
        mock_asset.description = "A test object"
        mock_asset.geometry_path = Path("/tmp/test.obj")
        mock_asset.sdf_path = Path("/tmp/test.sdf")
        mock_asset.image_path = None
        mock_asset.metadata = {}
        mock_asset.bbox_min = np.array([0.0, 0.0, 0.0])
        mock_asset.bbox_max = np.array([0.1, 0.1, 0.1])

        self.mock_asset_manager.get_asset_by_id = Mock(return_value=mock_asset)
        self.mock_scene.generate_unique_id = Mock(
            return_value=UniqueID("manipuland_001")
        )

        # Set up noise mock to return a slightly modified transform.
        def noise_side_effect(transform, **kwargs):
            return RigidTransform(p=[0.1, 0.1, 1.0])

        mock_apply_noise.side_effect = noise_side_effect

        # Place manipuland.
        result_json = self.manipuland_tools._place_manipuland_on_surface_impl(
            asset_id=str(asset_id),
            surface_id=str(self.mock_surface.surface_id),
            position_x=0.1,
            position_z=0.1,
        )

        # Verify noise was applied.
        mock_apply_noise.assert_called_once()
        call_kwargs = mock_apply_noise.call_args[1]
        self.assertEqual(
            call_kwargs["position_xy_std_meters"],
            self.cfg.placement_noise.natural_profile.position_xy_std_meters,
        )
        self.assertEqual(
            call_kwargs["rotation_yaw_std_degrees"],
            self.cfg.placement_noise.natural_profile.rotation_yaw_std_degrees,
        )

    def test_validate_convex_hull_strict_containment(self):
        """Test convex hull validation with strict containment (0% overlap)."""
        # Create a simple square mesh (1m x 1m).
        vertices = np.array(
            [
                [-0.5, -0.5, 0.0],
                [0.5, -0.5, 0.0],
                [0.5, 0.5, 0.0],
                [-0.5, 0.5, 0.0],
            ]
        )
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        # Save mesh to temp file.
        mesh_path = self.temp_dir / "test_square.obj"
        mesh.export(mesh_path)

        # Create a circular surface (radius 1.0m).
        circle_surface = Mock(spec=SupportSurface)
        circle_surface.surface_id = UniqueID("circle_surface_001")

        def contains_point(point_2d):
            # Circle of radius 1.0 centered at origin.
            return np.linalg.norm(point_2d) <= 1.0

        circle_surface.contains_point_2d = Mock(side_effect=contains_point)

        # Test: Centered square corners at distance 0.707m should fit in 1.0m radius.
        is_valid, error_msg = self.manipuland_tools._validate_convex_hull_footprint(
            target_surface=circle_surface,
            geometry_path=mesh_path,
            position_2d=np.array([0.0, 0.0]),
            rotation_degrees=0.0,
            allow_overlap_ratio=0.0,
        )
        self.assertTrue(is_valid)
        self.assertIsNone(error_msg)

        # Test: Offset square should fail.
        is_valid, error_msg = self.manipuland_tools._validate_convex_hull_footprint(
            target_surface=circle_surface,
            geometry_path=mesh_path,
            position_2d=np.array([0.5, 0.5]),
            rotation_degrees=0.0,
            allow_overlap_ratio=0.0,
        )
        self.assertFalse(is_valid)
        self.assertIsNotNone(error_msg)

    def test_validate_convex_hull_with_overlap_tolerance(self):
        """Test convex hull validation with 15% overlap tolerance."""
        # Create a simple square mesh (1m x 1m).
        vertices = np.array(
            [
                [-0.5, -0.5, 0.0],
                [0.5, -0.5, 0.0],
                [0.5, 0.5, 0.0],
                [-0.5, 0.5, 0.0],
            ]
        )
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        # Save mesh to temp file.
        mesh_path = self.temp_dir / "test_square.obj"
        mesh.export(mesh_path)

        # Create a circular surface (radius 0.65m).
        circle_surface = Mock(spec=SupportSurface)
        circle_surface.surface_id = UniqueID("circle_surface_001")

        def contains_point(point_2d):
            return np.linalg.norm(point_2d) <= 0.65

        circle_surface.contains_point_2d = Mock(side_effect=contains_point)

        # With 15% shrink, corners at 0.707 * 0.85 = 0.601m should fit in 0.65m radius.
        is_valid, error_msg = self.manipuland_tools._validate_convex_hull_footprint(
            target_surface=circle_surface,
            geometry_path=mesh_path,
            position_2d=np.array([0.0, 0.0]),
            rotation_degrees=0.0,
            allow_overlap_ratio=0.15,
        )
        self.assertTrue(is_valid)
        self.assertIsNone(error_msg)

    def test_collision_frame_bbox_overrides_misaligned_visual_mesh(self):
        """SDF-frame bounds must prevent placements a rotated GLTF would accept."""
        visual_mesh = trimesh.creation.box(extents=[0.16, 0.02, 0.20])
        mesh_path = self.temp_dir / "misaligned_book.obj"
        visual_mesh.export(mesh_path)
        surface = SupportSurface(
            surface_id=UniqueID("shelf_surface"),
            bounding_box_min=np.array([-0.5, -0.5, 0.0]),
            bounding_box_max=np.array([0.5, 0.5, 0.5]),
            transform=RigidTransform(p=[0.0, 0.0, 1.0]),
        )

        is_valid, error_msg = self.manipuland_tools._validate_convex_hull_footprint(
            target_surface=surface,
            geometry_path=mesh_path,
            position_2d=np.array([0.0, 0.45]),
            rotation_degrees=0.0,
            bounding_box_min=np.array([-0.08, -0.10, 0.0]),
            bounding_box_max=np.array([0.08, 0.10, 0.03]),
        )

        self.assertFalse(is_valid)
        self.assertIn("surface boundary", error_msg)

    def test_wall_side_of_support_surface_requires_clearance(self):
        """A valid support point cannot put the full object footprint in a wall."""
        wall = SceneObject(
            object_id=UniqueID("east_wall"),
            object_type=ObjectType.WALL,
            name="east_wall",
            description="Room east wall",
            transform=RigidTransform(p=[2.475, 0.0, 1.35]),
            bbox_min=np.array([-0.025, -2.0, -1.35]),
            bbox_max=np.array([0.025, 2.0, 1.35]),
        )
        self.mock_scene.objects = {wall.object_id: wall}
        shelf_surface = SupportSurface(
            surface_id=UniqueID("S_2"),
            bounding_box_min=np.array([-0.40, -0.155, 0.0]),
            bounding_box_max=np.array([0.40, 0.155, 0.5]),
            transform=RigidTransform(
                rpy=RollPitchYaw(0.0, 0.0, math.pi / 2),
                p=np.array([2.295, 0.0, 1.52]),
            ),
        )
        bbox_min = np.array([-0.08, -0.10, 0.0])
        bbox_max = np.array([0.08, 0.10, 0.03])

        blocked, error_msg = self.manipuland_tools._validate_convex_hull_footprint(
            target_surface=shelf_surface,
            geometry_path=self.temp_dir / "unused.obj",
            position_2d=np.array([0.0, -0.05]),
            rotation_degrees=0.0,
            bounding_box_min=bbox_min,
            bounding_box_max=bbox_max,
        )
        clear, clear_error = self.manipuland_tools._validate_convex_hull_footprint(
            target_surface=shelf_surface,
            geometry_path=self.temp_dir / "unused.obj",
            position_2d=np.array([0.0, 0.0]),
            rotation_degrees=0.0,
            bounding_box_min=bbox_min,
            bounding_box_max=bbox_max,
        )

        self.assertFalse(blocked)
        self.assertIn("east_wall", error_msg)
        self.assertTrue(clear)
        self.assertIsNone(clear_error)

    def test_top_surface_uses_overlap_tolerance(self):
        """Test that top surfaces use configured overlap tolerance."""
        # Load base configuration.
        base_config_path = (
            Path(__file__).parent.parent.parent
            / "configurations/manipuland_agent/base_manipuland_agent.yaml"
        )
        base_cfg = OmegaConf.load(base_config_path)

        # Load stateful configuration.
        stateful_config_path = (
            Path(__file__).parent.parent.parent
            / "configurations/manipuland_agent/stateful_manipuland_agent.yaml"
        )
        stateful_cfg = OmegaConf.load(stateful_config_path)

        # Merge configurations (simulating Hydra defaults resolution).
        cfg_with_validation = OmegaConf.merge(base_cfg, stateful_cfg)

        # Create new manipuland tools with updated config.
        tools = ManipulandTools(
            scene=self.mock_scene,
            asset_manager=self.mock_asset_manager,
            cfg=cfg_with_validation,
            current_furniture_id=UniqueID("furniture_001"),
            support_surfaces=self.support_surfaces,
        )

        # Verify tolerance is loaded from config.
        expected_tolerance = self.cfg.placement_validation.top_surface_overlap_tolerance
        self.assertEqual(tools.top_surface_overlap_tolerance, expected_tolerance)

    def test_non_top_surface_uses_strict_containment(self):
        """Test that non-top surfaces use strict containment."""
        # Create two surfaces at different heights.
        top_surface = Mock(spec=SupportSurface)
        top_surface.surface_id = UniqueID("top_surface_001")
        top_surface.bounding_box_min = np.array([-0.5, -0.5, 1.0])
        top_surface.bounding_box_max = np.array([0.5, 0.5, 1.5])
        top_surface.transform = RigidTransform(p=[0.0, 0.0, 1.5])  # Z=1.5m (top)

        shelf_surface = Mock(spec=SupportSurface)
        shelf_surface.surface_id = UniqueID("shelf_surface_001")
        shelf_surface.bounding_box_min = np.array([-0.5, -0.5, 0.5])
        shelf_surface.bounding_box_max = np.array([0.5, 0.5, 1.0])
        shelf_surface.transform = RigidTransform(p=[0.0, 0.0, 0.75])  # Z=0.75m (shelf)

        # Create a mock furniture object with these surfaces.
        mock_furniture = Mock()
        mock_furniture.support_surfaces = [top_surface, shelf_surface]

        # Add furniture to scene.
        furniture_id = UniqueID("furniture_001")
        self.mock_scene.objects = {furniture_id: mock_furniture}

        # Update support surfaces.
        multi_surfaces = {
            str(top_surface.surface_id): top_surface,
            str(shelf_surface.surface_id): shelf_surface,
        }
        tools = ManipulandTools(
            scene=self.mock_scene,
            asset_manager=self.mock_asset_manager,
            cfg=self.cfg,
            current_furniture_id=furniture_id,
            support_surfaces=multi_surfaces,
        )

        # Top surface should be identified correctly.
        self.assertTrue(tools._is_top_surface(str(top_surface.surface_id)))
        self.assertFalse(tools._is_top_surface(str(shelf_surface.surface_id)))


if __name__ == "__main__":
    unittest.main()
