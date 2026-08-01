"""Unit tests for furniture checkpoint branching workflow."""

import asyncio
import json
import shutil
import tempfile
import unittest

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from scenesmith.experiments.indoor_scene_generation import (
    _apply_and_rescore_final_furniture_state,
    _apply_final_wall_functional_guards,
    _copy_checkpoint_for_stage,
    _fix_paths_in_json_file,
    _raise_for_unresolved_furniture_relations,
    _rescore_furniture_after_postprocessing,
)


class TestFinalFurnitureRescore(unittest.TestCase):
    """Verify relation guards run before the canonical furniture render."""

    def test_rescore_explicitly_renders_canonical_state_before_critique(self):
        events: list[str] = []
        scene = object()
        canonical_render_dir = Path("/tmp/canonical-furniture-render")
        rendering_manager = MagicMock()
        rendering_manager.use_render_profile.return_value = nullcontext()
        rendering_manager.render_scene.side_effect = lambda **_kwargs: (
            events.append("render") or canonical_render_dir
        )

        critic_vision_tools = SimpleNamespace(
            _get_room_bounds=MagicMock(return_value=(-2.5, -2.25, 2.5, 2.25))
        )
        safety_controller = SimpleNamespace(
            enabled=True,
            reset_best_checkpoint=MagicMock(
                side_effect=lambda: events.append("reset_checkpoint")
            ),
        )
        agent = SimpleNamespace(
            scene=None,
            critic=None,
            furniture_safety_controller=safety_controller,
            rendering_manager=rendering_manager,
            blender_server=object(),
            _critic_vision_tools=critic_vision_tools,
            _create_critic_tools=MagicMock(return_value=["observe_scene"]),
            _create_critic_agent=MagicMock(return_value="critic"),
            _critic_render_profile_name=MagicMock(return_value="final"),
            _request_critique_impl=AsyncMock(
                side_effect=lambda **_kwargs: events.append("critique")
            ),
            _finalize_scene_and_scores=AsyncMock(
                side_effect=lambda: events.append("finalize")
            ),
        )

        asyncio.run(
            _rescore_furniture_after_postprocessing(
                furniture_agent=agent,
                scene=scene,
            )
        )

        self.assertIs(agent.scene, scene)
        self.assertEqual(events, ["reset_checkpoint", "render", "critique", "finalize"])
        safety_controller.reset_best_checkpoint.assert_called_once_with()
        rendering_manager.clear_cache.assert_called_once_with()
        rendering_manager.use_render_profile.assert_called_once_with("final")
        rendering_manager.render_scene.assert_called_once_with(
            scene=scene,
            blender_server=agent.blender_server,
            room_bounds=(-2.5, -2.25, 2.5, 2.25),
        )
        agent._request_critique_impl.assert_awaited_once_with(update_checkpoint=False)
        agent._finalize_scene_and_scores.assert_awaited_once_with()

    def test_post_wall_contract_guard_is_noop_when_critic_disabled(self):
        module = "scenesmith.experiments.indoor_scene_generation"
        with (
            patch(
                f"{module}.critic_config_from_any",
                return_value=SimpleNamespace(enabled=False),
            ),
            patch(f"{module}.improve_furniture_relations") as improve,
            patch(f"{module}._raise_for_unresolved_furniture_relations") as validate,
        ):
            _apply_final_wall_functional_guards(scene=object(), cfg_dict={})

        improve.assert_not_called()
        validate.assert_not_called()

    def test_post_wall_contract_guard_repairs_only_delayed_alignment(self):
        module = "scenesmith.experiments.indoor_scene_generation"
        scene = object()
        cfg_dict = {"experiment": "config"}
        fix = SimpleNamespace(
            object_id="chalkboard_0",
            relation_type="instructional_surface_alignment",
        )
        with (
            patch(
                f"{module}.critic_config_from_any",
                return_value=SimpleNamespace(enabled=True),
            ),
            patch(
                f"{module}.improve_furniture_relations",
                return_value=[fix],
            ) as improve,
            patch(f"{module}._raise_for_unresolved_furniture_relations") as validate,
        ):
            _apply_final_wall_functional_guards(scene=scene, cfg_dict=cfg_dict)

        improve.assert_called_once_with(
            scene,
            config=cfg_dict,
            allowed_relation_types={"instructional_surface_alignment"},
        )
        validate.assert_called_once_with(scene=scene, cfg_dict=cfg_dict)

    def test_rescore_observes_state_after_all_final_guards(self):
        events: list[tuple[str, str]] = []

        class FakeScene:
            version = 0

            def content_hash(self) -> str:
                return str(self.version)

        scene = FakeScene()

        def mutate(label: str):
            def apply(target, **_kwargs):
                target.version += 1
                events.append((label, target.content_hash()))

            return apply

        async def record_rescore(*, furniture_agent, scene):
            self.assertIs(furniture_agent, agent)
            events.append(("rescore", scene.content_hash()))

        agent = object()
        rescore = AsyncMock(side_effect=record_rescore)
        module = "scenesmith.experiments.indoor_scene_generation"
        with (
            patch(f"{module}.align_seating_to_nearest_surface", mutate("align")),
            patch(f"{module}.improve_storage_front_access", mutate("storage")),
            patch(f"{module}.improve_furniture_relations", mutate("relations")),
            patch(
                f"{module}._raise_for_unresolved_furniture_relations",
                side_effect=lambda **_kwargs: events.append(
                    ("validate_relations", scene.content_hash())
                ),
            ),
            patch(
                f"{module}.critic_config_from_any",
                return_value=SimpleNamespace(enabled=True),
            ),
            patch(
                f"{module}.seating_orientation_targets",
                return_value=None,
            ),
            patch(f"{module}._rescore_furniture_after_postprocessing", rescore),
        ):
            changed = asyncio.run(
                _apply_and_rescore_final_furniture_state(
                    furniture_agent=agent,
                    scene=scene,
                    cfg_dict={},
                    previous_scene_hash="0",
                )
            )

        self.assertTrue(changed)
        self.assertEqual(
            events,
            [
                ("align", "1"),
                ("storage", "2"),
                ("relations", "3"),
                ("align", "4"),
                ("validate_relations", "4"),
                ("rescore", "4"),
                ("align", "5"),
                ("storage", "6"),
                ("relations", "7"),
                ("align", "8"),
                ("validate_relations", "8"),
            ],
        )
        rescore.assert_awaited_once()

    def test_inventory_added_by_rescore_receives_one_physical_revalidation(self):
        furniture_type = SimpleNamespace(value="furniture")

        class FakeScene:
            def __init__(self):
                self.objects = {"sofa_0": SimpleNamespace(object_type=furniture_type)}

            def content_hash(self) -> str:
                return ",".join(sorted(self.objects))

        scene = FakeScene()
        rescore_calls = 0

        async def rescore(*, furniture_agent, scene):
            nonlocal rescore_calls
            rescore_calls += 1
            furniture_agent.scene = scene
            if rescore_calls == 1:
                scene.objects["plant_repair_0"] = SimpleNamespace(
                    object_type=furniture_type
                )

        module = "scenesmith.experiments.indoor_scene_generation"
        physical = MagicMock(return_value=(scene, True, []))
        agent = SimpleNamespace(scene=scene)
        with (
            patch(f"{module}._apply_final_furniture_guards"),
            patch(f"{module}._raise_for_unresolved_furniture_relations"),
            patch(
                f"{module}._rescore_furniture_after_postprocessing",
                new=AsyncMock(side_effect=rescore),
            ) as rescore_mock,
            patch(f"{module}._run_furniture_physical_postprocessing", physical),
        ):
            changed = asyncio.run(
                _apply_and_rescore_final_furniture_state(
                    furniture_agent=agent,
                    scene=scene,
                    cfg_dict={},
                    previous_scene_hash="pre-simulation",
                    physically_validated_furniture_ids=frozenset({"sofa_0"}),
                )
            )

        self.assertTrue(changed)
        self.assertEqual(rescore_mock.await_count, 2)
        physical.assert_called_once_with(scene=scene, cfg_dict={})
        self.assertEqual(set(scene.objects), {"sofa_0", "plant_repair_0"})

    def test_repeated_inventory_repair_after_revalidation_fails_stage(self):
        furniture_type = SimpleNamespace(value="furniture")

        class FakeScene:
            def __init__(self):
                self.objects = {"sofa_0": SimpleNamespace(object_type=furniture_type)}

            def content_hash(self) -> str:
                return ",".join(sorted(self.objects))

        scene = FakeScene()
        rescore_calls = 0

        async def rescore(*, furniture_agent, scene):
            nonlocal rescore_calls
            rescore_calls += 1
            furniture_agent.scene = scene
            scene.objects[f"plant_repair_{rescore_calls}"] = SimpleNamespace(
                object_type=furniture_type
            )

        def remove_unstable_repair(*, scene, cfg_dict):
            del scene.objects["plant_repair_1"]
            return scene, True, ["plant_repair_1"]

        module = "scenesmith.experiments.indoor_scene_generation"
        agent = SimpleNamespace(scene=scene)
        with (
            patch(f"{module}._apply_final_furniture_guards"),
            patch(f"{module}._raise_for_unresolved_furniture_relations"),
            patch(
                f"{module}._rescore_furniture_after_postprocessing",
                new=AsyncMock(side_effect=rescore),
            ),
            patch(
                f"{module}._run_furniture_physical_postprocessing",
                side_effect=remove_unstable_repair,
            ) as physical,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "refusing to persist unvalidated furniture: plant_repair_2",
            ):
                asyncio.run(
                    _apply_and_rescore_final_furniture_state(
                        furniture_agent=agent,
                        scene=scene,
                        cfg_dict={},
                        previous_scene_hash="pre-simulation",
                        physically_validated_furniture_ids=frozenset({"sofa_0"}),
                    )
                )

        physical.assert_called_once_with(scene=scene, cfg_dict={})
        self.assertEqual(rescore_calls, 2)

    def test_unresolved_core_relation_blocks_furniture_stage(self):
        module = "scenesmith.experiments.indoor_scene_generation"
        failures = [
            {
                "label": "fail",
                "primary_object": "dining_table_0",
                "relation_type": "room_center_alignment",
            },
            {
                "label": "fail",
                "primary_object": "dining_table_0",
                "relation_type": "dining_seat_distribution",
            },
        ]
        with patch(
            f"{module}.unresolved_furniture_relation_failures",
            return_value=failures,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "room_center_alignment:dining_table_0, "
                "dining_seat_distribution:dining_table_0",
            ):
                _raise_for_unresolved_furniture_relations(
                    scene=object(),
                    cfg_dict={
                        "furniture_agent": {
                            "fail_stage_on_unresolved_hard_constraints": True
                        }
                    },
                )

    def test_relation_gate_can_be_disabled_for_failed_trajectory_collection(self):
        module = "scenesmith.experiments.indoor_scene_generation"
        with patch(
            f"{module}.unresolved_furniture_relation_failures"
        ) as evaluate_relations:
            _raise_for_unresolved_furniture_relations(
                scene=object(),
                cfg_dict={
                    "furniture_agent": {
                        "fail_stage_on_unresolved_hard_constraints": False
                    }
                },
            )
        evaluate_relations.assert_not_called()


class TestPathFixing(unittest.TestCase):
    """Test path fixing in JSON files."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_fix_paths_in_json_file_simple(self):
        """Test fixing simple absolute paths."""
        # Create a JSON file with absolute paths.
        json_path = self.temp_path / "test.json"
        old_base = "/old/path/scene_000/room_main"
        data = {
            "sdf_path": f"{old_base}/generated_assets/furniture/table.sdf",
            "geometry_path": f"{old_base}/generated_assets/furniture/table.glb",
        }
        with open(json_path, "w") as f:
            json.dump(data, f)

        # Fix paths to new base.
        new_base = self.temp_path / "room_main"
        new_base.mkdir(parents=True)
        _fix_paths_in_json_file(json_path=json_path, new_room_dir=new_base)

        # Verify paths were fixed.
        with open(json_path) as f:
            fixed_data = json.load(f)

        self.assertEqual(
            fixed_data["sdf_path"],
            str(new_base / "generated_assets/furniture/table.sdf"),
        )
        self.assertEqual(
            fixed_data["geometry_path"],
            str(new_base / "generated_assets/furniture/table.glb"),
        )

    def test_fix_paths_in_json_file_nested(self):
        """Test fixing paths in nested structures."""
        json_path = self.temp_path / "test.json"
        old_base = "/old/experiment/scene_000/room_main"
        data = {
            "objects": [
                {
                    "name": "table",
                    "sdf_path": f"{old_base}/generated_assets/furniture/table.sdf",
                },
                {
                    "name": "chair",
                    "sdf_path": f"{old_base}/generated_assets/furniture/chair.sdf",
                },
            ],
            "metadata": {
                "nested": {
                    "path": f"{old_base}/generated_assets/textures/wood.png",
                }
            },
        }
        with open(json_path, "w") as f:
            json.dump(data, f)

        new_base = self.temp_path / "room_main"
        new_base.mkdir(parents=True)
        _fix_paths_in_json_file(json_path=json_path, new_room_dir=new_base)

        with open(json_path) as f:
            fixed_data = json.load(f)

        self.assertEqual(
            fixed_data["objects"][0]["sdf_path"],
            str(new_base / "generated_assets/furniture/table.sdf"),
        )
        self.assertEqual(
            fixed_data["objects"][1]["sdf_path"],
            str(new_base / "generated_assets/furniture/chair.sdf"),
        )
        self.assertEqual(
            fixed_data["metadata"]["nested"]["path"],
            str(new_base / "generated_assets/textures/wood.png"),
        )

    def test_fix_paths_preserves_relative_paths(self):
        """Test that relative paths are preserved."""
        json_path = self.temp_path / "test.json"
        data = {
            "relative_path": "generated_assets/furniture/table.sdf",
            "other_path": "../floor_plans/floor.sdf",
            "name": "test object",
            "count": 42,
        }
        with open(json_path, "w") as f:
            json.dump(data, f)

        new_base = self.temp_path / "room_main"
        new_base.mkdir(parents=True)
        _fix_paths_in_json_file(json_path=json_path, new_room_dir=new_base)

        with open(json_path) as f:
            fixed_data = json.load(f)

        # Relative paths should be unchanged.
        self.assertEqual(
            fixed_data["relative_path"], "generated_assets/furniture/table.sdf"
        )
        self.assertEqual(fixed_data["other_path"], "../floor_plans/floor.sdf")
        self.assertEqual(fixed_data["name"], "test object")
        self.assertEqual(fixed_data["count"], 42)

    def test_fix_paths_scene_level(self):
        """Test fixing scene-level paths (room_geometry/, floor_plans/)."""
        # Create directory structure: scene_dir/room_dir.
        scene_dir = self.temp_path / "scene_000"
        room_dir = scene_dir / "room_main"
        room_dir.mkdir(parents=True)

        json_path = room_dir / "test.json"
        old_scene = "/old/experiment/scene_000"
        data = {
            # Room-level path.
            "furniture_sdf": f"{old_scene}/room_main/generated_assets/furniture/table.sdf",
            # Scene-level paths.
            "wall_sdf": f"{old_scene}/room_geometry/room_geometry_main.sdf",
            "floor_sdf": f"{old_scene}/floor_plans/floor.sdf",
        }
        with open(json_path, "w") as f:
            json.dump(data, f)

        _fix_paths_in_json_file(
            json_path=json_path,
            new_room_dir=room_dir,
            new_scene_dir=scene_dir,
        )

        with open(json_path) as f:
            fixed_data = json.load(f)

        # Room-level path should point to room_dir.
        self.assertEqual(
            fixed_data["furniture_sdf"],
            str(room_dir / "generated_assets/furniture/table.sdf"),
        )
        # Scene-level paths should point to scene_dir.
        self.assertEqual(
            fixed_data["wall_sdf"],
            str(scene_dir / "room_geometry/room_geometry_main.sdf"),
        )
        self.assertEqual(
            fixed_data["floor_sdf"],
            str(scene_dir / "floor_plans/floor.sdf"),
        )

    def test_fix_paths_nonexistent_file(self):
        """Test that nonexistent files are handled gracefully."""
        json_path = self.temp_path / "does_not_exist.json"
        new_base = self.temp_path / "room_main"

        # Should not raise an error.
        _fix_paths_in_json_file(json_path=json_path, new_room_dir=new_base)


class TestCopyCheckpointForStage(unittest.TestCase):
    """Test selective checkpoint copy for stage resumption."""

    def setUp(self):
        """Set up test fixtures."""
        self.source_dir = Path(tempfile.mkdtemp())
        self.target_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.source_dir)
        shutil.rmtree(self.target_dir)

    def _create_source_scene(self, with_furniture_assets: bool = True):
        """Create a source scene directory structure for testing."""
        scene_dir = self.source_dir / "scene_000"
        room_dir = scene_dir / "room_main"
        room_dir.mkdir(parents=True)

        # Create house_layout.json (no paths to fix).
        house_layout = scene_dir / "house_layout.json"
        house_layout.write_text(json.dumps({"rooms": ["room_main"], "width": 5.0}))

        # Create floor_plans directory.
        floor_plans = scene_dir / "floor_plans"
        floor_plans.mkdir()
        (floor_plans / "floor.sdf").write_text("<sdf>floor</sdf>")

        # Create room_geometry at scene level.
        room_geometry = scene_dir / "room_geometry"
        room_geometry.mkdir()
        (room_geometry / "room_geometry_main.sdf").write_text("<sdf>room</sdf>")

        if with_furniture_assets:
            # Create generated_assets with absolute paths.
            assets_dir = room_dir / "generated_assets" / "furniture"
            assets_dir.mkdir(parents=True)
            (assets_dir / "table.sdf").write_text("<sdf>table</sdf>")
            (assets_dir / "table.glb").write_text("glb content")

            # Create asset_registry.json with absolute paths.
            registry = {
                "table_01": {
                    "sdf_path": str(room_dir / "generated_assets/furniture/table.sdf"),
                    "geometry_path": str(
                        room_dir / "generated_assets/furniture/table.glb"
                    ),
                }
            }
            registry_path = assets_dir / "asset_registry.json"
            registry_path.write_text(json.dumps(registry))

            # Create scene_states with absolute paths.
            scene_states = room_dir / "scene_states" / "scene_after_furniture"
            scene_states.mkdir(parents=True)
            scene_state = {
                "objects": [
                    {
                        "id": "table_01",
                        "sdf_path": str(
                            room_dir / "generated_assets/furniture/table.sdf"
                        ),
                    }
                ]
            }
            (scene_states / "scene_state.json").write_text(json.dumps(scene_state))

            # Create scene.dmd.yaml with file:// URIs.
            dmd_content = f"""directives:
- add_model:
    name: table_01
    file: file://{room_dir}/generated_assets/furniture/table.sdf
"""
            (scene_states / "scene.dmd.yaml").write_text(dmd_content)

            # Create session DB (should NOT be copied).
            (room_dir / "furniture_agent.db").write_text("session data")

            # Create render directory (should NOT be copied).
            renders_dir = room_dir / "scene_renders" / "furniture" / "renders_001"
            renders_dir.mkdir(parents=True)
            (renders_dir / "render.png").write_text("render content")

        return scene_dir

    def test_copy_scene_for_furniture_stage(self):
        """Test copying scene for furniture stage (no checkpoint needed)."""
        source_scene = self._create_source_scene(with_furniture_assets=False)
        target_scene = self.target_dir / "scene_000"
        target_scene.mkdir(parents=True)

        _copy_checkpoint_for_stage(
            source_scene_dir=source_scene,
            target_scene_dir=target_scene,
            start_stage="furniture",
        )

        # Verify scene-level directories were copied.
        self.assertTrue((target_scene / "house_layout.json").exists())
        self.assertTrue((target_scene / "floor_plans" / "floor.sdf").exists())
        self.assertTrue(
            (target_scene / "room_geometry" / "room_geometry_main.sdf").exists()
        )

        # Verify room directory was created but no checkpoint copied.
        target_room = target_scene / "room_main"
        self.assertTrue(target_room.exists())
        self.assertFalse((target_room / "scene_states").exists())

    def test_copy_scene_for_wall_mounted_stage(self):
        """Test copying scene for wall_mounted stage copies furniture checkpoint."""
        source_scene = self._create_source_scene(with_furniture_assets=True)
        source_room = source_scene / "room_main"
        target_scene = self.target_dir / "scene_000"
        target_scene.mkdir(parents=True)

        _copy_checkpoint_for_stage(
            source_scene_dir=source_scene,
            target_scene_dir=target_scene,
            start_stage="wall_mounted",
        )

        target_room = target_scene / "room_main"

        # Verify checkpoint was copied.
        checkpoint_dir = target_room / "scene_states" / "scene_after_furniture"
        self.assertTrue(checkpoint_dir.exists())
        self.assertTrue((checkpoint_dir / "scene_state.json").exists())
        self.assertTrue((checkpoint_dir / "scene.dmd.yaml").exists())

        # Verify furniture assets were copied.
        self.assertTrue(
            (target_room / "generated_assets" / "furniture" / "table.sdf").exists()
        )

        # Verify paths were fixed in scene_state.json.
        with open(checkpoint_dir / "scene_state.json") as f:
            state = json.load(f)
        table_sdf = state["objects"][0]["sdf_path"]
        self.assertIn(str(target_room), table_sdf)
        self.assertNotIn(str(source_room), table_sdf)

        # Verify paths were fixed in scene.dmd.yaml.
        dmd_content = (checkpoint_dir / "scene.dmd.yaml").read_text()
        self.assertIn(str(target_room), dmd_content)
        self.assertNotIn(str(source_room), dmd_content)

        # Verify session DB was NOT copied.
        self.assertFalse((target_room / "furniture_agent.db").exists())

        # Verify render directories were NOT copied.
        self.assertFalse((target_room / "scene_renders").exists())

    def test_copy_scene_for_manipuland_stage(self):
        """Test copying scene for manipuland stage copies all previous assets."""
        source_scene = self._create_source_scene(with_furniture_assets=True)

        # Add wall_mounted and ceiling_mounted assets.
        source_room = source_scene / "room_main"
        wall_dir = source_room / "generated_assets" / "wall_mounted"
        wall_dir.mkdir(parents=True)
        (wall_dir / "picture.sdf").write_text("<sdf>picture</sdf>")

        ceiling_dir = source_room / "generated_assets" / "ceiling_mounted"
        ceiling_dir.mkdir(parents=True)
        (ceiling_dir / "light.sdf").write_text("<sdf>light</sdf>")

        # Create ceiling checkpoint.
        ceiling_state = source_room / "scene_states" / "scene_after_ceiling_objects"
        ceiling_state.mkdir(parents=True)
        (ceiling_state / "scene_state.json").write_text("{}")

        target_scene = self.target_dir / "scene_000"
        target_scene.mkdir(parents=True)

        _copy_checkpoint_for_stage(
            source_scene_dir=source_scene,
            target_scene_dir=target_scene,
            start_stage="manipuland",
        )

        target_room = target_scene / "room_main"

        # Verify all asset directories were copied.
        self.assertTrue(
            (target_room / "generated_assets" / "furniture" / "table.sdf").exists()
        )
        self.assertTrue(
            (target_room / "generated_assets" / "wall_mounted" / "picture.sdf").exists()
        )
        self.assertTrue(
            (
                target_room / "generated_assets" / "ceiling_mounted" / "light.sdf"
            ).exists()
        )

        # Verify ceiling checkpoint was copied (not furniture checkpoint).
        self.assertTrue(
            (target_room / "scene_states" / "scene_after_ceiling_objects").exists()
        )
        self.assertFalse(
            (target_room / "scene_states" / "scene_after_furniture").exists()
        )

    def test_copy_scene_nonexistent_source(self):
        """Test error when source scene doesn't exist."""
        target_scene = self.target_dir / "scene_000"
        target_scene.mkdir(parents=True)
        nonexistent_source = self.source_dir / "scene_000"

        with self.assertRaises(FileNotFoundError):
            _copy_checkpoint_for_stage(
                source_scene_dir=nonexistent_source,
                target_scene_dir=target_scene,
                start_stage="furniture",
            )


if __name__ == "__main__":
    unittest.main()
