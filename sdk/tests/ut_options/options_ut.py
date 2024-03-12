import unittest
from sdk.options import Devices, Options


class TestOptions(unittest.TestCase):
    def test_init(self):
        device = Devices.GPU
        options = Options(device)
        self.assertEqual(options.device, device)


if __name__ == '__main__':
    unittest.main()
