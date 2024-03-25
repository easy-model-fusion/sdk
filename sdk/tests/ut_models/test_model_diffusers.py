import unittest
from unittest.mock import patch, MagicMock

from diffusers import DiffusionPipeline

from sdk import ModelDiffusers, DIFFUSERS
from sdk.options import Devices


class TestModelDiffusers(unittest.TestCase):

    @patch("sdk.models.ModelDiffusers.load_model", return_value=True)
    def test_load_model(self, mock_load_model):
        # Create an instance of ModelDiffusers
        model = ModelDiffusers(
            model_name="TestModel",
            model_path="test/path",
            device=Devices.GPU,
            model_class=MagicMock()
        )
        model.pipeline.to = MagicMock

        # Call load_model method
        result = model.load_model()

        # Assert that the model is loaded
        self.assertTrue(result)
        mock_load_model.assert_called_once()

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

    @patch("sdk.models.ModelDiffusers.generate_prompt",
           return_value="Generated image")
    def test_generate_prompt(self, mock_generate):
        # Create an instance of ModelDiffusers
        model = ModelDiffusers(
            model_name="TestModel",
            model_path="test/path",
            device=Devices.GPU,
            model_class=MagicMock()
        )

        # Set up mock pipeline behavior
        mock_generate.return_value = "Generated image"

        # Call generate_prompt method
        prompt = "Generate image from text"
        result = model.generate_prompt(prompt)

        # Assert that the pipeline method was called with the prompt
        mock_generate.assert_called_once_with(prompt)
        # Assert that the result is "Generated image"
        self.assertEqual(result, "Generated image")


if __name__ == "__main__":
    unittest.main()
