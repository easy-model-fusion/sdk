import unittest
from unittest.mock import MagicMock, patch
from sdk.models import ModelsTextConversation
from transformers import Conversation

import uuid


class TestModelsTextConversation(unittest.TestCase):
    def setUp(self):
        self.model_name = "test_model"
        self.model_path = "/path/to/model"
        self.tokenizer_path = "/path/to/tokenizer"
        self.model_class_mock = MagicMock()
        self.tokenizer_class_mock = MagicMock()
        self.device = "gpu"
        self.kwargs = {"param1": "value1", "param2": "value2"}
        self.transformers_pipeline = MagicMock()

    @patch("sdk.ModelTransformers.create_pipeline")
    def test_init(self, create_pipeline_mock):
        instance = ModelsTextConversation(
            model_name=self.model_name, model_path=self.model_path,
            tokenizer_path=self.tokenizer_path,
            tokenizer_class=self.tokenizer_class_mock,
            model_class=self.model_class_mock, device=self.device
        )
        instance.transformers_pipeline = MagicMock()

        # Assert
        self.assertEqual(instance.model_name, self.model_name)
        self.assertEqual(instance.model_path, self.model_path)
        self.assertEqual(instance.model_class, self.model_class_mock)
        self.assertEqual(instance.tokenizer_class, self.tokenizer_class_mock)
        self.assertEqual(instance.device, self.device)
        create_pipeline_mock.assert_called_once()

    @patch("sdk.ModelsTextConversation.create_pipeline")
    @patch("sdk.ModelsTextConversation.write_input", return_value=None)
    def test_generate_prompt(self, write_input_mock,
                             pipeline_mock):
        # Arrange
        prompt = "test_prompt"

        instance = ModelsTextConversation(
            model_name=self.model_name, model_path=self.model_path,
            tokenizer_path=self.tokenizer_path,
            tokenizer_class=self.tokenizer_class_mock,
            model_class=self.model_class_mock, device=self.device
        )
        instance.transformers_pipeline = MagicMock()
        instance.conversation = MagicMock()

        # Act
        result = instance.generate_prompt(prompt)

        # Assert
        write_input_mock.assert_called_once_with(prompt)
        pipeline_mock.assert_called_once_with()
        self.assertIsNotNone(result)  # Assert that result is not None

    @patch("sdk.ModelsTextConversation.create_pipeline")
    def test_write_input(self, pipeline_mock):
        # Arrange
        prompt = "test_prompt"
        instance = ModelsTextConversation(
            model_name=self.model_name, model_path=self.model_path,
            tokenizer_path=self.tokenizer_path,
            tokenizer_class=self.tokenizer_class_mock,
            model_class=self.model_class_mock, device=self.device
        )
        instance.transformers_pipeline = MagicMock()
        conversation_mock = MagicMock()

        instance.conversation = conversation_mock

        # Act
        instance.write_input(prompt)

        # Assert
        pipeline_mock.assert_called_once()
        conversation_mock.add_message.assert_called_once_with(
            {"role": "user", "content": prompt}
        )

    @patch("sdk.ModelsTextConversation.create_pipeline")
    def test_create_new_conversation(self, pipeline_mock):
        # Arrange
        instance = ModelsTextConversation(
            model_name=self.model_name, model_path=self.model_path,
            tokenizer_path=self.tokenizer_path,
            tokenizer_class=self.tokenizer_class_mock,
            model_class=self.model_class_mock, device=self.device
        )
        instance.transformers_pipeline = MagicMock()
        instance.conversation = Conversation()

        # Act
        instance.create_new_conversation()

        # Assert
        pipeline_mock.assert_called_once()
        self.assertIsInstance(instance.conversation, Conversation)
        self.assertIn(instance.conversation.uuid,
                      instance.conversation_dict)

    @patch("sdk.ModelsTextConversation.create_pipeline")
    def test_change_conversation_valid_id(self, pipeline_mock):
        # Arrange

        # Arrange
        instance = ModelsTextConversation(
            model_name=self.model_name, model_path=self.model_path,
            tokenizer_path=self.tokenizer_path,
            tokenizer_class=self.tokenizer_class_mock,
            model_class=self.model_class_mock, device=self.device
        )
        instance.transformers_pipeline = MagicMock()
        instance.conversation = Conversation()
        instance.create_new_conversation()

        conversation_one = instance.conversation  # Conversation to fetch
        conv_id = conversation_one.uuid

        instance.create_new_conversation()
        instance.create_new_conversation()

        # Act
        result = instance.change_conversation(conv_id)

        # Assert
        pipeline_mock.assert_called_once()
        self.assertTrue(result)
        self.assertEqual(instance.conversation, conversation_one)

    @patch("sdk.ModelsTextConversation.create_pipeline")
    def test_change_conversation_invalid_id(self, pipeline_mock):
        # Arrange

        # Arrange
        instance = ModelsTextConversation(
            model_name=self.model_name, model_path=self.model_path,
            tokenizer_path=self.tokenizer_path,
            tokenizer_class=self.tokenizer_class_mock,
            model_class=self.model_class_mock, device=self.device
        )
        instance.transformers_pipeline = MagicMock()
        instance.conversation = Conversation()
        instance.create_new_conversation()

        # Act
        # Random UUID value
        uuid_value = '123e4567-e89b-12d3-a456-426614174000'

        # Create a UUID object from the specified value
        custom_uuid = uuid.UUID(uuid_value)

        result = instance.change_conversation(custom_uuid)

        # Assert
        pipeline_mock.assert_called_once()
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
