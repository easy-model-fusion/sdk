import unittest
from typing import Optional, Dict
from unittest.mock import MagicMock, patch
from sdk.models import Model
from sdk.models import ModelsManagement


class TestModelsManagement(unittest.TestCase):
    def setUp(self):
        # Set up mock objects and parameters
        self.model_name = "test_model"
        self.model_path = "test_model"
        self.model_instance_mock = MagicMock(spec=Model)
        self.loaded_model: Optional[Model] = MagicMock()
        self.loaded_models_cache: Dict[str, Model] = {}

    def test_add_model(self):
        instance = ModelsManagement()
        # Arrange
        instance.loaded_models_cache = {}  # Clear existing cache
        # Create a mock object with model_name attribute
        self.model_instance_mock = MagicMock(model_name=self.model_name)
        instance.loaded_models_cache[self.model_name] = self.model_instance_mock

        # Act
        result = instance.add_model(self.model_instance_mock)

        # Assert
        self.assertFalse(result)  # Model already exists in cache
        self.assertEqual(len(instance.loaded_models_cache), 1)

    def test_add_model_success(self):
        instance = ModelsManagement()
        # Arrange
        instance.loaded_models_cache = {}  # Clear existing cache
        # Create a mock object with model_name attribute
        self.model_instance_mock = MagicMock(model_name=self.model_name)

        # Act
        result = instance.add_model(self.model_instance_mock)

        # Assert
        self.assertTrue(result)  # Model already exists in cache
        self.assertEqual(len(instance.loaded_models_cache), 1)

    @patch("sdk.models.ModelsManagement.load_model")
    def test_generate_prompt_without_model_name(self, load_model_mock):
        # Arrange
        prompt = "test_prompt"
        instance = ModelsManagement()

        # Create a mock object with model_name attribute
        self.model_instance_mock = MagicMock(model_name="")
        instance.add_model(self.model_instance_mock)
        instance.loaded_model = self.model_instance_mock
        # Act
        instance.generate_prompt(prompt)

        # Assert
        load_model_mock.assert_not_called()

    def test_unload_model_success(self):
        # Arrange
        instance = ModelsManagement()
        self.model_instance_mock = MagicMock(model_name=self.model_name,
                                             name=self.model_path,
                                             class_name=MagicMock(),
                                             module=MagicMock())
        instance.add_model(self.model_instance_mock)
        instance.loaded_model = self.model_instance_mock

        # Act
        result = instance.unload_model()

        # Assert
        self.assertTrue(result)  # Ensure that unload_model returns True when model is successfully unloaded
        self.assertIsNone(instance.loaded_model)  # Ensure that loaded_model is set to None after unloading

    def test_unload_model_failure(self):
        # Arrange
        instance = ModelsManagement()
        self.model_instance_mock = MagicMock(model_name=self.model_name,
                                             name=self.model_path,
                                             class_name=MagicMock(),
                                             module=MagicMock())
        self.model_instance_mock.unload_model.return_value = False  # Simulate failure during unload_model
        instance.add_model(self.model_instance_mock)
        instance.loaded_model = self.model_instance_mock

        # Act
        result = instance.unload_model()

        # Assert
        self.assertFalse(result)  # Ensure that unload_model returns False when model is unsuccessfully unloaded
        self.assertEqual(len(instance.loaded_models_cache), 1)  # Ensure that loaded_model is set and not unloaded

    def test_print_models(self):
        # Arrange
        instance = ModelsManagement()
        instance.loaded_models_cache = {
            "model1": MagicMock(spec=Model),
            "model2": MagicMock(spec=Model)
        }
        instance.loaded_model = instance.loaded_models_cache["model1"]

        # Act
        with patch("builtins.print") as mocked_print:
            instance.print_models()

        # Assert
        expected_output = (
            "- model2 "
        )
        mocked_print.assert_called_with(expected_output)


if __name__ == "__main__":
    unittest.main()
