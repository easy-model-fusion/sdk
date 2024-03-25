import unittest

from sdk import Devices
from sdk.options.options_tokenizer import OptionsTokenizer


class TestOptionsTokenizer(unittest.TestCase):
    def test_init_defaults(self):
        options = OptionsTokenizer(device="cpu", padding_side=None, return_tensors=None)
        self.assertEqual(options.device, "cpu")
        self.assertEqual(options.padding_side, "left")
        self.assertIsNone(options.return_tensors)

    def test_init_custom(self):
        options = OptionsTokenizer(device=Devices.CPU, padding_side="right", return_tensors="tf")
        self.assertEqual(options.device, Devices.CPU)
        self.assertEqual(options.padding_side, "right")
        self.assertEqual(options.return_tensors, "tf")

    def test_invalid_padding_side(self):
        with self.assertRaises(ValueError):
            OptionsTokenizer(device="cuda", padding_side="invalid",  return_tensors=None)

    def test_padding_side_default(self):
        options = OptionsTokenizer(device="cuda", padding_side=None, return_tensors=None)
        self.assertEqual(options.padding_side, "left")


if __name__ == "__main__":
    unittest.main()