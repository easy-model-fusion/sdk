import unittest
from sdk.options import OptionsTextConversation, Devices
from sdk.sdk.models.model_text_conversation import ModelsTextConversation


class TestModelsTextConversation(unittest.TestCase):
    def setUp(self):
        self.model_name = "test_model"
        self.model_path = "/path/to/test_model"
        self.model = ModelsTextConversation(self.model_name, self.model_path)

    def test_load_model(self):
        options = OptionsTextConversation(
            device=Devices.CPU,
            prompt="Test prompt"
        )
        loaded = self.model.load_model(options)
        self.assertTrue(loaded)

    def test_unload_model(self):
        options = OptionsTextConversation(
            device=Devices.CPU,
            prompt="Test prompt"
        )
        self.model.load_model(options)
        unloaded = self.model.unload_model()
        self.assertTrue(unloaded)

    def test_generate_prompt(self):
        options = OptionsTextConversation(
            device=Devices.CPU,
            prompt="Test prompt"
        )
        generated_prompt = self.model.generate_prompt(None, options)
        self.assertIsNotNone(generated_prompt)


if __name__ == '__main__':
    unittest.main()
