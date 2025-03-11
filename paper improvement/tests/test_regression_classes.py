import unittest
from regression_classes import YourTransformerClass

class TestTransformer(unittest.TestCase):
    def setUp(self):
        self.transformer = YourTransformerClass()

    def test_transform_without_fit(self):
        with self.assertRaises(AttributeError):
            self.transformer.transform(X)

    def test_fit_and_transform(self):
        self.transformer.fit(X_train)
        transformed = self.transformer.transform(X_train)
        self.assertIsNotNone(transformed)

if __name__ == '__main__':
    unittest.main()