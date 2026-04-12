import shutil
import tempfile
import unittest

from pathlib import Path
from unittest.mock import MagicMock, patch

from scenesmith.agent_utils.geometry_generation_server.geometry_generation import (
    generate_geometry_from_image,
)
from scenesmith.agent_utils.image_generation import (
    OpenAIImageGenerator,
    _extract_and_save_openai_image,
)

# Mock hy3dgen modules for CI where Hunyuan3D-2 is not installed.
# This allows the @patch decorators to work even when the real modules don't exist.
try:
    import hy3dgen.rembg  # noqa: F401
    import hy3dgen.shapegen  # noqa: F401
    import hy3dgen.texgen  # noqa: F401
except ImportError:
    # Only mock if real modules don't exist (CI environment).
    import sys

    # Create structured mocks with proper hierarchy.
    hy3dgen = MagicMock()
    hy3dgen.rembg = MagicMock()
    hy3dgen.shapegen = MagicMock()
    hy3dgen.shapegen.pipelines = MagicMock()
    hy3dgen.texgen = MagicMock()

    # Create the classes on the submodules.
    hy3dgen.rembg.BackgroundRemover = MagicMock()
    hy3dgen.shapegen.Hunyuan3DDiTFlowMatchingPipeline = MagicMock()
    hy3dgen.shapegen.FaceReducer = MagicMock()
    hy3dgen.shapegen.pipelines.export_to_trimesh = MagicMock()
    hy3dgen.texgen.Hunyuan3DPaintPipeline = MagicMock()

    # Add to sys.modules with proper structure.
    sys.modules["hy3dgen"] = hy3dgen
    sys.modules["hy3dgen.rembg"] = hy3dgen.rembg
    sys.modules["hy3dgen.shapegen"] = hy3dgen.shapegen
    sys.modules["hy3dgen.shapegen.pipelines"] = hy3dgen.shapegen.pipelines
    sys.modules["hy3dgen.texgen"] = hy3dgen.texgen


class TestOpenAIImageGenerator(unittest.TestCase):
    """Test the OpenAIImageGenerator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

        # Mock OpenAI client.
        self.mock_client = MagicMock()
        self.generator = OpenAIImageGenerator(client=self.mock_client)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """Test generator initialization."""
        self.assertEqual(self.generator.client, self.mock_client)
        self.assertEqual(self.generator.image_quality, "low")

    def test_initialization_with_quality(self):
        """Test generator initialization with custom quality."""
        generator = OpenAIImageGenerator(client=self.mock_client, quality="high")
        self.assertEqual(generator.image_quality, "high")

    @patch("scenesmith.agent_utils.image_generation._extract_and_save_openai_image")
    def test_generate_single_image(self, mock_extract):
        """Test single image generation."""
        # Mock response.
        mock_response = MagicMock()
        mock_response.data = [MagicMock(b64_json="base64data")]
        self.mock_client.images.generate.return_value = mock_response

        # Generate image.
        output_path = self.temp_path / "test_image.png"
        self.generator.generate_images(
            style_prompt="Modern minimalist",
            object_descriptions=["A modern chair"],
            output_paths=[output_path],
        )

        # Verify OpenAI API was called correctly.
        self.mock_client.images.generate.assert_called_once()
        call_kwargs = self.mock_client.images.generate.call_args[1]
        self.assertEqual(call_kwargs["model"], "gpt-image-1.5")
        self.assertEqual(call_kwargs["size"], "1024x1024")
        self.assertEqual(call_kwargs["quality"], "low")
        self.assertEqual(call_kwargs["background"], "opaque")

        # Verify image extraction happened.
        mock_extract.assert_called_once()

    @patch("scenesmith.agent_utils.image_generation._extract_and_save_openai_image")
    def test_generate_multiple_images(self, mock_extract):
        """Test generating multiple images in parallel."""
        # Mock responses.
        mock_response1 = MagicMock()
        mock_response1.data = [MagicMock(b64_json="base64data1")]
        mock_response2 = MagicMock()
        mock_response2.data = [MagicMock(b64_json="base64data2")]
        self.mock_client.images.generate.side_effect = [mock_response1, mock_response2]

        # Generate images.
        descriptions = ["A modern chair", "A modern table"]
        output_paths = [self.temp_path / "chair.png", self.temp_path / "table.png"]

        self.generator.generate_images(
            style_prompt="Modern minimalist",
            object_descriptions=descriptions,
            output_paths=output_paths,
        )

        # Verify API was called correct number of times.
        self.assertEqual(self.mock_client.images.generate.call_count, 2)
        self.assertEqual(mock_extract.call_count, 2)

    def test_generate_images_mismatched_lengths(self):
        """Test error when descriptions and output paths have different lengths."""
        with self.assertRaises(ValueError) as context:
            self.generator.generate_images(
                style_prompt="Modern",
                object_descriptions=["chair", "table"],
                output_paths=[self.temp_path / "chair.png"],  # Only one path.
            )
        self.assertIn("Number of descriptions must match", str(context.exception))

    def test_generate_images_failure_propagates(self):
        """Test that exceptions from image generation propagate (fail-fast)."""
        # Mock API to raise an error.
        self.mock_client.images.generate.side_effect = RuntimeError(
            "Image generation API error"
        )

        descriptions = ["Item 1", "Item 2"]
        output_paths = [self.temp_path / "item1.png", self.temp_path / "item2.png"]

        # Verify that the original exception propagates.
        with self.assertRaises(RuntimeError) as context:
            self.generator.generate_images(
                style_prompt="Test style",
                object_descriptions=descriptions,
                output_paths=output_paths,
            )

        self.assertIn("Image generation API error", str(context.exception))


class TestGeometryGeneration(unittest.TestCase):
    """Test the geometry generation functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    @patch("hy3dgen.rembg.BackgroundRemover")
    @patch(
        "scenesmith.agent_utils.geometry_generation_server.geometry_generation.Hunyuan3DPipelineManager"
    )
    @patch("hy3dgen.shapegen.pipelines.export_to_trimesh")
    @patch(
        "scenesmith.agent_utils.geometry_generation_server.geometry_generation.Image"
    )
    def test_generate_geometry_from_image_success(
        self,
        mock_image,
        mock_export_to_trimesh,
        mock_pipeline_manager_class,
        mock_bg_remover_class,
    ):
        """Test successful geometry generation from image with pipeline caching."""
        # Mock image loading and processing.
        mock_img = MagicMock()
        mock_img.convert.return_value = mock_img
        mock_image.open.return_value = mock_img

        # Mock pipeline manager and pipelines.
        mock_shape_pipeline = MagicMock()
        mock_pipeline_output = MagicMock()
        mock_shape_pipeline.return_value = mock_pipeline_output

        mock_texture_pipeline = MagicMock()
        mock_face_reducer = MagicMock()
        mock_background_remover = MagicMock()

        mock_mesh = MagicMock()
        mock_face_reducer.return_value = mock_mesh
        mock_texture_pipeline.return_value = mock_mesh
        mock_background_remover.return_value = mock_img
        mock_bg_remover_class.return_value = mock_background_remover

        # Configure pipeline manager to return mocked pipelines.
        mock_pipeline_manager_class.get_pipelines.return_value = (
            mock_shape_pipeline,
            mock_texture_pipeline,
            mock_face_reducer,
            mock_background_remover,
        )

        # Mock export_to_trimesh.
        mock_export_to_trimesh.return_value = [mock_mesh]

        # Test paths.
        image_path = self.temp_path / "test_image.png"
        output_path = self.temp_path / "test_output.glb"
        debug_folder = self.temp_path / "debug"
        debug_folder.mkdir()

        # Call function with pipeline caching enabled.
        generate_geometry_from_image(
            image_path=image_path,
            output_path=output_path,
            debug_folder=debug_folder,
            use_pipeline_caching=True,
        )

        # Verify pipeline manager was used with correct config.
        mock_pipeline_manager_class.get_pipelines.assert_called_once_with(
            use_mini=False
        )

        # Verify image processing.
        mock_image.open.assert_called_once_with(image_path)
        mock_background_remover.assert_called_once_with(mock_img)

        # Verify shape generation pipeline.
        mock_shape_pipeline.assert_called_once()

        # Verify mesh processing pipeline.
        mock_export_to_trimesh.assert_called_once_with(mock_pipeline_output)
        mock_face_reducer.assert_called_once_with(mock_mesh)

        # Verify texture generation.
        mock_texture_pipeline.assert_called_once_with(mock_mesh, image=mock_img)

        # Verify final export.
        mock_mesh.export.assert_called_once_with(output_path)

        # Verify debug image was saved.
        mock_img.save.assert_called_once()

    @patch("hy3dgen.shapegen.FaceReducer")
    @patch("hy3dgen.shapegen.Hunyuan3DDiTFlowMatchingPipeline")
    @patch("hy3dgen.texgen.Hunyuan3DPaintPipeline")
    @patch("hy3dgen.rembg.BackgroundRemover")
    @patch("hy3dgen.shapegen.pipelines.export_to_trimesh")
    @patch(
        "scenesmith.agent_utils.geometry_generation_server.geometry_generation.Image"
    )
    def test_generate_geometry_from_image_without_debug(
        self,
        mock_image,
        mock_export_to_trimesh,
        mock_bg_remover_class,
        mock_tex_pipeline_class,
        mock_shape_pipeline_class,
        mock_face_reducer_class,
    ):
        """Test geometry generation without debug folder (no caching)."""
        # Mock image loading and processing.
        mock_img = MagicMock()
        mock_img.convert.return_value = mock_img
        mock_image.open.return_value = mock_img

        # Mock background remover.
        mock_bg_remover = MagicMock()
        mock_bg_remover.return_value = mock_img
        mock_bg_remover_class.return_value = mock_bg_remover

        # Mock shape pipeline.
        mock_shape_pipeline = MagicMock()
        mock_shape_pipeline.enable_flashvdm = MagicMock()
        mock_pipeline_output = MagicMock()
        mock_shape_pipeline.return_value = mock_pipeline_output
        mock_shape_pipeline_class.from_pretrained.return_value = mock_shape_pipeline

        # Mock export_to_trimesh.
        mock_mesh = MagicMock()
        mock_export_to_trimesh.return_value = [mock_mesh]

        # Mock face reducer.
        mock_face_reducer = MagicMock()
        mock_face_reducer.return_value = mock_mesh
        mock_face_reducer_class.return_value = mock_face_reducer

        # Mock texture pipeline.
        mock_tex_pipeline = MagicMock()
        mock_tex_pipeline.return_value = mock_mesh
        mock_tex_pipeline_class.from_pretrained.return_value = mock_tex_pipeline

        # Test paths.
        image_path = self.temp_path / "test_image.png"
        output_path = self.temp_path / "test_output.glb"

        # Call function without debug folder and without caching.
        generate_geometry_from_image(
            image_path=image_path,
            output_path=output_path,
            debug_folder=None,
            use_pipeline_caching=False,
        )

        # Verify debug image was not saved.
        mock_img.save.assert_not_called()

        # Verify pipelines were initialized fresh (not cached).
        mock_shape_pipeline_class.from_pretrained.assert_called_once_with(
            "tencent/Hunyuan3D-2", subfolder="hunyuan3d-dit-v2-0-turbo"
        )
        mock_tex_pipeline_class.from_pretrained.assert_called_once_with(
            "tencent/Hunyuan3D-2"
        )

    @patch("hy3dgen.rembg.BackgroundRemover")
    @patch(
        "scenesmith.agent_utils.geometry_generation_server.geometry_generation.Hunyuan3DPipelineManager"
    )
    @patch("hy3dgen.shapegen.pipelines.export_to_trimesh")
    @patch(
        "scenesmith.agent_utils.geometry_generation_server.geometry_generation.Image"
    )
    def test_generate_geometry_from_image_with_mini_model(
        self,
        mock_image,
        mock_export_to_trimesh,
        mock_pipeline_manager_class,
        mock_bg_remover_class,
    ):
        """Test geometry generation with mini model and pipeline caching."""
        # Mock image loading and processing.
        mock_img = MagicMock()
        mock_img.convert.return_value = mock_img
        mock_image.open.return_value = mock_img

        # Mock pipeline manager and pipelines.
        mock_shape_pipeline = MagicMock()
        mock_pipeline_output = MagicMock()
        mock_shape_pipeline.return_value = mock_pipeline_output

        mock_texture_pipeline = MagicMock()
        mock_face_reducer = MagicMock()
        mock_background_remover = MagicMock()

        mock_mesh = MagicMock()
        mock_face_reducer.return_value = mock_mesh
        mock_texture_pipeline.return_value = mock_mesh
        mock_background_remover.return_value = mock_img
        mock_bg_remover_class.return_value = mock_background_remover

        # Configure pipeline manager to return mocked pipelines.
        mock_pipeline_manager_class.get_pipelines.return_value = (
            mock_shape_pipeline,
            mock_texture_pipeline,
            mock_face_reducer,
            mock_background_remover,
        )

        # Mock export_to_trimesh.
        mock_export_to_trimesh.return_value = [mock_mesh]

        # Test paths.
        image_path = self.temp_path / "test_image.png"
        output_path = self.temp_path / "test_output.glb"

        # Call function with mini model and caching enabled.
        generate_geometry_from_image(
            image_path=image_path,
            output_path=output_path,
            use_mini=True,
            use_pipeline_caching=True,
        )

        # Verify pipeline manager was called with mini model config.
        mock_pipeline_manager_class.get_pipelines.assert_called_once_with(use_mini=True)

        # Verify mesh processing pipeline.
        mock_export_to_trimesh.assert_called_once_with(mock_pipeline_output)
        mock_face_reducer.assert_called_once_with(mock_mesh)

        # Verify final export.
        mock_mesh.export.assert_called_once_with(output_path)


class TestAssetConversion(unittest.TestCase):
    """Test the asset conversion functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)


class TestExtractAndSaveImage(unittest.TestCase):
    """Test the _extract_and_save_openai_image utility function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    @patch("builtins.open", create=True)
    @patch("base64.b64decode")
    def test_extract_and_save_image_success(self, mock_b64decode, mock_open):
        """Test successful image extraction and saving."""
        # Mock response with image data (Images API format).
        mock_data = MagicMock()
        mock_data.b64_json = "base64_image_data"

        mock_response = MagicMock()
        mock_response.data = [mock_data]

        # Mock base64 decode.
        mock_b64decode.return_value = b"decoded_image_data"

        # Mock file operations.
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file

        # Call function.
        output_path = self.temp_path / "test.png"
        _extract_and_save_openai_image(mock_response, output_path, "test chair")

        # Check base64 decode was called.
        mock_b64decode.assert_called_once_with("base64_image_data")

        # Check file was opened and written.
        mock_open.assert_called_once_with(output_path, "wb")
        mock_file.write.assert_called_once_with(b"decoded_image_data")

    def test_extract_and_save_image_no_data(self):
        """Test error when no image data in response."""
        # Mock response with no data.
        mock_response = MagicMock()
        mock_response.data = []

        # Call function and expect error.
        output_path = self.temp_path / "test.png"
        with self.assertRaises(ValueError) as context:
            _extract_and_save_openai_image(mock_response, output_path, "test chair")

        self.assertIn(
            "No image data returned from OpenAI for test chair", str(context.exception)
        )

    def test_extract_and_save_image_no_b64_json(self):
        """Test error when b64_json is None."""
        # Mock response with None b64_json.
        mock_data = MagicMock()
        mock_data.b64_json = None

        mock_response = MagicMock()
        mock_response.data = [mock_data]

        # Call function and expect error.
        output_path = self.temp_path / "test.png"
        with self.assertRaises(ValueError) as context:
            _extract_and_save_openai_image(mock_response, output_path, "test chair")

        self.assertIn(
            "No base64 image data in response for test chair", str(context.exception)
        )


if __name__ == "__main__":
    unittest.main()
