import unittest
from abc import ABC
from sdk.models import Model


class TestModel(unittest.TestCase):
    def test_init(self):
        # Arrange
        model_name = "test_model"
        model_path = "/path/to/model"

        # Act
        model = Model(model_name, model_path)

        # Assert
        self.assertEqual(model.model_name, model_name)
        self.assertEqual(model.model_path, model_path)

    def test_load_model_abstract_method(self):
        # Arrange
        model = Model("test_model", "/path/to/model")

        # Act & Assert
        with self.assertRaises(NotImplementedError):
            model.load_model()

    def test_unload_model_abstract_method(self):
        # Arrange
        model = Model("test_model", "/path/to/model")

        # Act & Assert
        with self.assertRaises(NotImplementedError):
            model.unload_model()

    def test_generate_prompt_abstract_method(self):
        # Arrange
        model = Model("test_model", "/path/to/model")

        # Act & Assert
        with self.assertRaises(NotImplementedError):
            model.generate_prompt("test_prompt")


if __name__ == "__main__":
    unittest.main()
