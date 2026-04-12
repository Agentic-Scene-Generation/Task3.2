import shutil
import tempfile
import unittest

from pathlib import Path
from unittest.mock import Mock, patch

from scenesmith.agent_utils.vlm_service import VLMService


class TestVLMService(unittest.TestCase):
    """Test VLMService class contracts."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.mock_openai_client = Mock()

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("scenesmith.agent_utils.vlm_service.OpenAI")
    def test_vlm_initialization(self, mock_openai_class):
        """Test VLMService initializes OpenAI client properly."""
        mock_openai_class.return_value = self.mock_openai_client

        vlm_service = VLMService()

        # Verify VLMService was initialized.
        self.assertIsNotNone(vlm_service)
        self.assertEqual(vlm_service.client, self.mock_openai_client)

        # Verify OpenAI client was created.
        mock_openai_class.assert_called_once()

    @patch("scenesmith.agent_utils.vlm_service.OpenAI")
    def test_create_completion_basic(self, mock_openai_class):
        """Test create_completion with basic parameters for standard models."""
        mock_openai_client = Mock()
        mock_openai_class.return_value = mock_openai_client

        # Mock the chat completions API for standard models.
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "Test response content"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_openai_client.chat.completions.create.return_value = mock_response

        vlm_service = VLMService()

        messages = [{"role": "user", "content": "Test message"}]
        model = "gpt-4o"
        # Verbosity is ignored for non-reasoning models (Chat Completions API).
        result = vlm_service.create_completion(
            model=model,
            messages=messages,
            reasoning_effort="medium",
            verbosity="low",
        )

        # Verify result.
        self.assertEqual(result, "Test response content")

        # Verify OpenAI chat API was called for standard model.
        mock_openai_client.chat.completions.create.assert_called_once()
        call_args = mock_openai_client.chat.completions.create.call_args
        self.assertEqual(call_args[1]["model"], model)

    @patch("scenesmith.agent_utils.vlm_service.OpenAI")
    def test_create_completion_with_reasoning_effort_and_verbosity(
        self, mock_openai_class
    ):
        """Test create_completion with reasoning effort and verbosity for reasoning models."""
        mock_openai_client = Mock()
        mock_openai_class.return_value = mock_openai_client

        # Mock the responses API.
        mock_response = Mock()
        mock_response.output_text = "Reasoning response"
        mock_openai_client.responses.create.return_value = mock_response

        vlm_service = VLMService()

        messages = [{"role": "user", "content": "Complex reasoning task"}]
        model = "gpt-5"
        reasoning_effort = "high"
        verbosity = "low"
        result = vlm_service.create_completion(
            model=model,
            messages=messages,
            reasoning_effort=reasoning_effort,
            verbosity=verbosity,
        )

        # Verify result.
        self.assertEqual(result, "Reasoning response")

        # Verify reasoning effort was passed.
        call_args = mock_openai_client.responses.create.call_args
        self.assertIn("reasoning", call_args[1])
        self.assertEqual(call_args[1]["reasoning"]["effort"], reasoning_effort)

        # Verify verbosity was passed.
        self.assertIn("text", call_args[1])
        self.assertEqual(call_args[1]["text"]["verbosity"], verbosity)

    @patch("scenesmith.agent_utils.vlm_service.OpenAI")
    def test_create_completion_with_json_format(self, mock_openai_class):
        """Test create_completion with JSON response format."""
        mock_openai_client = Mock()
        mock_openai_class.return_value = mock_openai_client

        # Mock the chat completions API for standard models.
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = '{"result": "json_response"}'
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_openai_client.chat.completions.create.return_value = mock_response

        vlm_service = VLMService()

        model = "gpt-4o"
        messages = [{"role": "user", "content": "Return JSON"}]
        result = vlm_service.create_completion(
            model=model,
            messages=messages,
            reasoning_effort="medium",
            verbosity="low",
            response_format={"type": "json_object"},
        )

        # Verify result.
        self.assertEqual(result, '{"result": "json_response"}')

        # Verify chat API was called with JSON format for standard model.
        mock_openai_client.chat.completions.create.assert_called_once()
        call_args = mock_openai_client.chat.completions.create.call_args
        self.assertEqual(call_args[1]["model"], model)
        self.assertEqual(call_args[1]["response_format"], {"type": "json_object"})

    @patch("scenesmith.agent_utils.vlm_service.OpenAI")
    def test_error_handling_for_api_failures(self, mock_openai_class):
        """Test handling of OpenAI API errors."""
        mock_openai_client = Mock()
        mock_openai_class.return_value = mock_openai_client

        # Mock chat API to raise an error for standard models.
        mock_openai_client.chat.completions.create.side_effect = Exception(
            "API rate limit exceeded"
        )

        vlm_service = VLMService()

        # Test that API errors are propagated.
        messages = [{"role": "user", "content": "Test"}]
        with self.assertRaises(Exception) as context:
            vlm_service.create_completion(
                model="gpt-4o",
                messages=messages,
                reasoning_effort="medium",
                verbosity="low",
            )

        self.assertIn("API rate limit exceeded", str(context.exception))

    @patch("scenesmith.agent_utils.vlm_service.OpenAI")
    def test_message_conversion_to_responses_format(self, mock_openai_class):
        """Test that messages work correctly for reasoning models with images."""
        mock_openai_client = Mock()
        mock_openai_class.return_value = mock_openai_client

        # Mock the responses API for reasoning models.
        mock_response = Mock()
        mock_response.output_text = "Converted response"
        mock_openai_client.responses.create.return_value = mock_response

        vlm_service = VLMService()

        # Test with image content in messages for reasoning model.
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this image"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,..."},
                    },
                ],
            }
        ]

        # Test with reasoning model (gpt-5) for image conversion.
        result = vlm_service.create_completion(
            model="gpt-5",
            messages=messages,
            reasoning_effort="medium",
            verbosity="low",
        )

        # Verify conversion worked and responses API was called.
        self.assertEqual(result, "Converted response")
        mock_openai_client.responses.create.assert_called_once()

        # Verify input was converted for responses API.
        call_args = mock_openai_client.responses.create.call_args
        self.assertIn("input", call_args[1])

    @patch("scenesmith.agent_utils.vlm_service.OpenAI")
    def test_vision_detail_parameter_chat_completions(self, mock_openai_class):
        """Test that vision_detail parameter is added to image_url objects for Chat
        API."""
        mock_openai_client = Mock()
        mock_openai_class.return_value = mock_openai_client

        # Mock the chat completions API for standard models.
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "Vision response"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_openai_client.chat.completions.create.return_value = mock_response

        vlm_service = VLMService()

        # Test with image content in messages for standard model.
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this image"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,..."},
                    },
                ],
            }
        ]

        result = vlm_service.create_completion(
            model="gpt-4o",
            messages=messages,
            reasoning_effort="medium",
            verbosity="low",
            vision_detail="high",
        )

        # Verify result.
        self.assertEqual(result, "Vision response")

        # Verify chat API was called.
        mock_openai_client.chat.completions.create.assert_called_once()
        call_args = mock_openai_client.chat.completions.create.call_args

        # Verify that detail parameter was added to image_url.
        messages_sent = call_args[1]["messages"]
        image_content = None
        for item in messages_sent[0]["content"]:
            if item["type"] == "image_url":
                image_content = item
                break

        self.assertIsNotNone(image_content)
        self.assertEqual(image_content["image_url"]["detail"], "high")

    @patch("scenesmith.agent_utils.vlm_service.OpenAI")
    def test_vision_detail_parameter_responses_api(self, mock_openai_class):
        """Test vision_detail parameter handling for Responses API (reasoning models)."""
        mock_openai_client = Mock()
        mock_openai_class.return_value = mock_openai_client

        # Mock the responses API for reasoning models.
        mock_response = Mock()
        mock_response.output_text = "Reasoning vision response"
        mock_openai_client.responses.create.return_value = mock_response

        vlm_service = VLMService()

        # Test with image content in messages for reasoning model.
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this image"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,..."},
                    },
                ],
            }
        ]

        result = vlm_service.create_completion(
            model="gpt-5",
            messages=messages,
            reasoning_effort="medium",
            verbosity="low",
            vision_detail="high",
        )

        # Verify result.
        self.assertEqual(result, "Reasoning vision response")

        # Verify responses API was called.
        mock_openai_client.responses.create.assert_called_once()

        # Verify the detail parameter was included in the Responses API format.
        call_args = mock_openai_client.responses.create.call_args
        input_messages = call_args[1]["input"]
        self.assertIsNotNone(input_messages)

        # Find the image content in the converted format.
        image_content = None
        for msg in input_messages:
            if "content" in msg and isinstance(msg["content"], list):
                for item in msg["content"]:
                    if item.get("type") == "input_image":
                        image_content = item
                        break

        self.assertIsNotNone(image_content)
        self.assertEqual(image_content["detail"], "high")


if __name__ == "__main__":
    unittest.main()
