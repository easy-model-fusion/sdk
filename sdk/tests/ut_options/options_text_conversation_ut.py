import unittest
from sdk.options import Devices
from sdk.sdk.options.options_text_conversation import OptionsTextConversation


class TestOptionsTextConversation(unittest.TestCase):
    def setUp(self):
        # Initialize common attributes for tests
        self.prompt = "Test prompt"
        self.model = "test_model"
        self.tokenizer = "test_tokenizer"

    def test_init(self):
        # Test initialization of the class with default arguments
        options = OptionsTextConversation(prompt=self.prompt)
        self.assertEqual(options.prompt, self.prompt)
        self.assertEqual(options.model, None)
        self.assertEqual(options.tokenizer, None)
        self.assertEqual(options.framework, None)
        self.assertEqual(options.task, "")
        self.assertEqual(options.num_workers, 8)
        self.assertEqual(options.batch_size, 1)
        self.assertEqual(options.device, -1)
        self.assertEqual(options.arg_parser, None)
        self.assertEqual(options.torch_dtype, None)
        self.assertEqual(options.binary_output, False)
        self.assertEqual(options.min_length_for_response, 32)
        self.assertEqual(options.minimum_tokens, 10)
        self.assertEqual(options.max_steps, 50)

    def test_init_with_args(self):
        # Test initialization of the class with specific arguments
        options = OptionsTextConversation(prompt=self.prompt,
                                          model=self.model,
                                          tokenizer=self.tokenizer,
                                          framework="test_framework",
                                          task="test_task",
                                          num_workers=4,
                                          batch_size=2,
                                          device=Devices.GPU,
                                          arg_parser={"arg1": "value1"},
                                          torch_dtype="float32",
                                          binary_output=True,
                                          min_length_for_response=16,
                                          minimum_tokens=5,
                                          max_steps=100)
        self.assertEqual(options.prompt, self.prompt)
        self.assertEqual(options.model, self.model)
        self.assertEqual(options.tokenizer, self.tokenizer)
        self.assertEqual(options.framework, "test_framework")
        self.assertEqual(options.task, "test_task")
        self.assertEqual(options.num_workers, 4)
        self.assertEqual(options.batch_size, 2)
        self.assertEqual(options.device, Devices.GPU)
        self.assertEqual(options.arg_parser, {"arg1": "value1"})
        self.assertEqual(options.torch_dtype, "float32")
        self.assertEqual(options.binary_output, True)
        self.assertEqual(options.min_length_for_response, 16)
        self.assertEqual(options.minimum_tokens, 5)
        self.assertEqual(options.max_steps, 100)



if __name__ == '__main__':
    unittest.main()
