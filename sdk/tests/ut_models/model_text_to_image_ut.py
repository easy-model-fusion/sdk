import unittest
from sdk.options import OptionsTextToImage, Devices
from sdk.sdk.models.model_text_to_image import ModelTextToImage


class TestModelTextToImage(unittest.TestCase):
    def setUp(self):
        self.model_name = "test_model"
        self.model_path = "/path/to/test_model"
        self.model = ModelTextToImage(self.model_name, self.model_path)

    def test_load_model(self):
        options = OptionsTextToImage(
            device=Devices.CPU,
            prompt="Test prompt"
        )
        loaded = self.model.load_model(options)
        self.assertTrue(loaded)

    def test_unload_model(self):
        options = OptionsTextToImage(
            device=Devices.CPU,
            prompt="Test prompt"
        )
        self.model.load_model(options)
        unloaded = self.model.unload_model()
        self.assertTrue(unloaded)

    def test_generate_prompt(self):
        options = OptionsTextToImage(
            device=Devices.CPU,
            prompt="Test prompt"
        )
        image = self.model.generate_prompt(None, options)
        self.assertIsNotNone(image)


if __name__ == '__main__':
    unittest.main()
