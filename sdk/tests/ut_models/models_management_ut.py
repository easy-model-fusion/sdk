import unittest
from sdk.sdk.models.model import Model
from sdk.options import Options
from sdk.sdk.models.models_management import ModelsManagement


class TestModelsManagement(unittest.TestCase):
    def setUp(self):
        # Initialize the ModelsManagement instance for testing
        self.models_management = ModelsManagement()

    def test_add_model(self):
        # Create a mock Model and Options objects for testing
        mock_model = Model("MockModel", "/path/to/mock/model")
        mock_options = Options()

        # Add the mock model to the ModelsManagement instance
        result = self.models_management.add_model(mock_model, mock_options)

        # Assert that the model was successfully added
        self.assertTrue(result)
        # Assert that the model is in the cache
        self.assertIn("MockModel", self.models_management.loaded_models_cache)

    def test_load_model(self):
        # Add a mock model to the cache
        mock_model = Model("MockModel", "/path/to/mock/model")
        mock_options = Options()
        self.models_management.add_model(mock_model, mock_options)

        # Load the mock model
        result = self.models_management.load_model("MockModel")

        # Assert that the model was successfully loaded
        self.assertTrue(result)
        # Assert that the loaded model matches the one in the cache
        self.assertEqual(self.models_management.loaded_model, mock_model)

    def test_unload_model(self):
        # Add a mock model to the cache and load it
        mock_model = Model("MockModel", "/path/to/mock/model")
        mock_options = Options()
        self.models_management.add_model(mock_model, mock_options)
        self.models_management.load_model("MockModel")

        # Unload the loaded model
        result = self.models_management.unload_model()

        # Assert that the model was successfully unloaded
        self.assertTrue(result)
        # Assert that no model is currently loaded
        self.assertIsNone(self.models_management.loaded_model)

    # Add more test methods for other functionalities as needed


if __name__ == '__main__':
    unittest.main()
