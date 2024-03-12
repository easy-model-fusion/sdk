import unittest
from sdk.options import Options
from sdk.sdk.models.model import Model


class TestModel(unittest.TestCase):
    def setUp(self):
        self.model_name = "test_model"
        self.model_path = "/path/to/test_model"
        self.model = Model(self.model_name, self.model_path)

    def test_load_model(self):
        options = Options(device="cpu")  # Modify this based on your Options class
        loaded = self.model.load_model(options)
        self.assertTrue(loaded)

    def test_unload_model(self):
        unloaded = self.model.unload_model()
        self.assertTrue(unloaded)

    def test_generate_prompt(self):
        options = Options(device="cpu")  # Modify this based on your Options class
        prompt = "Test prompt"
        generated = self.model.generate_prompt(prompt, options)
        self.assertIsNotNone(generated)


if __name__ == '__main__':
    unittest.main()
