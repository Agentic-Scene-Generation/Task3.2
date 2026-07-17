import shutil
import tempfile
import unittest

from pathlib import Path
from unittest.mock import Mock

from omegaconf import OmegaConf

from scenesmith.agent_utils.room import ObjectType, RoomScene, UniqueID
from scenesmith.agent_utils.scene_analyzer import SceneAnalyzer
from scenesmith.utils.llm_json import parse_llm_json, parse_llm_json_object


class TestSceneAnalyzer(unittest.TestCase):
    """Test SceneAnalyzer class contracts."""

    # Test configuration constants.
    TEST_MODEL = "gpt-4o-mini"
    TEST_REASONING_EFFORT = "low"

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.mock_vlm_service = Mock()
        self.mock_rendering_manager = Mock()
        self.mock_scene = Mock(spec=RoomScene)

        # Create mock BlenderServer.
        self.mock_blender_server = Mock()
        self.mock_blender_server.is_running.return_value = True

        # Create test config (only OpenAI settings needed).
        test_config_dict = {
            "openai": {
                "model": self.TEST_MODEL,
                "vision_detail": "low",
                "furniture_analysis_max_retries": 2,
                "context_selection_max_retries": 2,
                "reasoning_effort": {
                    "scene_critique": self.TEST_REASONING_EFFORT,
                    "furniture_analysis": self.TEST_REASONING_EFFORT,
                },
                "verbosity": {
                    "scene_critique": "low",
                    "furniture_analysis": "low",
                },
            },
        }
        # Convert to OmegaConf to match expected structure.
        self.test_config = OmegaConf.create(test_config_dict)

        self.scene_analyzer = SceneAnalyzer(
            vlm_service=self.mock_vlm_service,
            rendering_manager=self.mock_rendering_manager,
            cfg=self.test_config,
            blender_server=self.mock_blender_server,
        )

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_scene_analyzer_initialization(self):
        """Test that SceneAnalyzer initializes properly."""
        self.assertIsNotNone(self.scene_analyzer)
        self.assertEqual(self.scene_analyzer.vlm_service, self.mock_vlm_service)
        self.assertEqual(
            self.scene_analyzer.rendering_manager, self.mock_rendering_manager
        )
        self.assertEqual(self.scene_analyzer.cfg, self.test_config)
        self.assertEqual(self.scene_analyzer.blender_server, self.mock_blender_server)

    def test_configuration_access(self):
        """Test that SceneAnalyzer can access configuration values."""
        # Verify configuration was stored and accessible.
        self.assertEqual(self.scene_analyzer.cfg["openai"]["model"], self.TEST_MODEL)
        self.assertEqual(
            self.scene_analyzer.cfg["openai"]["reasoning_effort"]["scene_critique"],
            self.TEST_REASONING_EFFORT,
        )

    def test_parse_llm_json_strips_markdown_fences(self):
        """Fenced JSON from local/open models should still parse."""
        payload = """```json
        {
          "furniture_selections": [{"furniture_id": "desk_0"}]
        }
        ```"""

        parsed = parse_llm_json(payload)

        self.assertEqual(parsed["furniture_selections"][0]["furniture_id"], "desk_0")

    def test_parse_llm_json_repairs_trailing_comma(self):
        """Lightly malformed JSON should be repaired for local-model output."""
        payload = '{"furniture_selections": [{"furniture_id": "desk_0",}]}'

        parsed = parse_llm_json(payload)

        self.assertEqual(parsed["furniture_selections"][0]["furniture_id"], "desk_0")

    def test_parse_llm_json_object_rejects_non_object_payloads(self):
        """A scalar response should be retried instead of treated as a schema."""
        with self.assertRaisesRegex(
            ValueError, "Expected top-level JSON object but got str"
        ):
            parse_llm_json_object('"not enough visual context"')

    def test_analyze_furniture_accepts_fenced_json(self):
        """Furniture analysis should accept fenced output from local models."""
        furniture = Mock()
        furniture.object_id = UniqueID("desk_0")
        furniture.name = "desk"
        furniture.description = "wood desk"
        furniture.object_type = ObjectType.FURNITURE
        furniture.immutable = False
        furniture.bbox_min = [0, 0, 0]
        furniture.bbox_max = [1, 1, 1]

        self.mock_scene.objects = {"desk_0": furniture}
        self.mock_scene.room_geometry = None
        self.mock_scene.text_description = "A simple study."
        self.scene_analyzer.analyze_scene = Mock(
            return_value='''```json
            {
              "furniture_selections": [
                {
                  "furniture_id": "desk_0",
                  "suggested_items": "Optional: lamp",
                  "prompt_constraints": "No specific requirements",
                  "style_notes": "minimal"
                }
              ]
            }
            ```'''
        )

        selections = self.scene_analyzer.analyze_furniture_for_manipulands(
            scene=self.mock_scene,
            prompt_enum=Mock(),
        )

        self.assertEqual(len(selections), 1)
        self.assertEqual(selections[0].furniture_id, UniqueID("desk_0"))


if __name__ == "__main__":
    unittest.main()
