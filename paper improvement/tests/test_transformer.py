import unittest
import sys
import os
import numpy as np

# Add the parent directory to the Python path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from regression_classes import ChoquetTransformer

class TestTransformer(unittest.TestCase):
    
    def test_fit_transform(self):
        """Test that the transformer's fit and transform methods work correctly"""
        # Create sample data
        X = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        
        # Create a transformer with default parameters
        transformer = ChoquetTransformer(method="choquet_2add")
        
        # Test fit method
        self.assertIs(transformer.fit(X), transformer, "fit() should return self")
        self.assertTrue(hasattr(transformer, "n_features_in_"), 
                       "After fitting, transformer should have n_features_in_ attribute")
        
        # Test transform method
        X_transformed = transformer.transform(X)
        self.assertIsNotNone(X_transformed, "transform() should return a non-None value")
        
        # Test fit_transform method
        transformer2 = ChoquetTransformer(method="choquet_2add")
        X_fit_transform = transformer2.fit_transform(X)
        self.assertIsNotNone(X_fit_transform, "fit_transform() should return a non-None value")
        
        # Test for error with mismatched dimensions
        wrong_X = np.array([[0.1, 0.2, 0.3]])  # 3 features instead of 2
        with self.assertRaises(ValueError):
            transformer.transform(wrong_X)

if __name__ == '__main__':
    unittest.main()