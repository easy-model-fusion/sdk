import unittest
from unittest.mock import patch, MagicMock

from diffusers import DiffusionPipeline

from sdk import ModelDiffusers, DIFFUSERS
from sdk.options import Devices


class TestModelDiffusers(unittest.TestCase):
    def setUp(self):
        # Set up mock objects and parameters
        self.model_name = "test_model"
        self.model_path = "/path/to/model"
        self.model_class_mock = MagicMock()
        self.device = "cpu"
        self.kwargs = {"param1": "value1", "param2": "value2"}

    def test_unload_model_when_loaded(self):
        # Arrange
        pipeline_mock = MagicMock()
        instance = ModelDiffusers(self.model_name, self.model_path, self.model_class_mock, self.device,
                                  **self.kwargs)
        instance.loaded = True
        instance.pipeline = pipeline_mock

        # Act
        result = instance.unload_model()

        # Assert
        self.assertTrue(result)
        self.assertFalse(instance.loaded)
        pipeline_mock.to.assert_called_once_with(device=Devices.RESET.value)

    def test_unload_model_when_not_loaded(self):
        # Arrange
        instance = ModelDiffusers(self.model_name, self.model_path, self.model_class_mock, self.device, **self.kwargs)
        instance.loaded = False

        # Act
        result = instance.unload_model()

        # Assert
        self.assertFalse(result)
        self.assertFalse(instance.loaded)

    def test_load_model_when_loaded(self):
        # Arrange
        instance = ModelDiffusers(self.model_name, self.model_path, self.model_class_mock, Devices.CPU,
                                  **self.kwargs)
        instance.loaded = True

        # Act
        result = instance.load_model()

        # Assert
        self.assertTrue(result)
        self.assertTrue(instance.loaded)

    def test_load_model_on_reset_device(self):
        # Arrange
        instance = ModelDiffusers(self.model_name, self.model_path, self.model_class_mock, Devices.RESET,
                                  **self.kwargs)

        # Act
        result = instance.load_model()

        # Assert
        self.assertFalse(result)
        self.assertFalse(instance.loaded)

    def test_load_model_successfully(self):
        # Arrange
        pipeline_mock = MagicMock()
        instance = ModelDiffusers(self.model_name, self.model_path, self.model_class_mock, Devices.CPU,
                                  **self.kwargs)
        instance.pipeline = pipeline_mock

        # Act
        result = instance.load_model()

        # Assert
        self.assertTrue(result)
        self.assertTrue(instance.loaded)
        pipeline_mock.to.assert_called_once_with(device=Devices.CPU.value)

    @patch("sdk.models.ModelDiffusers.unload_model", return_value=True)
    def test_unload_model(self, mock_unload_model):
        # Create an instance of ModelDiffusers
        model = ModelDiffusers(
            model_name="TestModel",
            model_path="test/path",
            device=Devices.GPU,
            model_class=MagicMock()
        )

        # Call unload_model method
        result = model.unload_model()

        # Assert that the model is unloaded
        self.assertTrue(result)
        mock_unload_model.assert_called_once()

    def test_generate_prompt(self):
        # Arrange
        prompt = "test_prompt"
        pipeline_mock = MagicMock()

        # Create an instance of ModelsTextToImage and set the pipeline attribute
        instance = ModelDiffusers(self.model_name, self.model_path, self.model_class_mock, self.device, **self.kwargs)
        instance.pipeline = pipeline_mock

        # Act
        instance.generate_prompt(prompt)

        # Assert
        pipeline_mock.assert_called_once()

    def test_create_pipeline(self):
        # Act
        instance = ModelDiffusers(
            self.model_name, self.model_path, self.model_class_mock, self.device, **self.kwargs
        )
        # Assert
        self.assertIsNotNone(instance.pipeline)


if __name__ == "__main__":
    unittest.main()
