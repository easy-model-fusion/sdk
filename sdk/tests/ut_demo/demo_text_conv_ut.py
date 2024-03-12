import unittest
from unittest.mock import MagicMock
from sdk.sdk.demo import DemoTextConv

class TestDemoTextConv(unittest.TestCase):
    def test_demo(self):
        # Mocking the ModelsTextConversation object
        model_mock = MagicMock()
        model_mock.model_name = "Test Model"
        model_mock.tokenizer.decode.return_value = "Mocked response"

        demo = DemoTextConv()
        demo.demo(model_mock)

        # Ensure that tokenizer.decode() is called with the correct arguments
        model_mock.tokenizer.decode.assert_called_once()

if __name__ == "__main__":
    unittest.main()
