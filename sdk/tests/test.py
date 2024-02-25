import unittest


def add(a, b):
    return a + b


class TestAddFunction(unittest.TestCase):

    def test_add_positive_numbers(self):
        result = add(3, 5)
        self.assertEqual(result, 8)

    def test_add_negative_numbers(self):
        result = add(-2, -4)
        self.assertEqual(result, -6)

    def test_add_mixed_numbers(self):
        result = add(10, -7)
        self.assertEqual(result, 3)


if __name__ == '__main__':
    unittest.main()
