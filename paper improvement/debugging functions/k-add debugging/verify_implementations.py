import numpy as np
from choquet_kadd_test import (
    choquet_matrix_2add, 
    choquet_matrix_unified,
    convert_between_representations,
    explain_transformation
)

def verify_implementation_equivalence():
    """
    Demonstrate that both implementations of the 2-additive Choquet integral
    (equation 22 with k=2 and equation 23) are mathematically equivalent.
    """
    # Create a simple dataset for testing
    np.random.seed(42)
    X = np.random.rand(5, 3)
    
    print("===== TESTING IMPLEMENTATION EQUIVALENCE =====")
    print("\nTest Data:")
    print(X[:2])  # Show just first two samples
    
    # 1. Get outputs from both implementations
    eq23_output = choquet_matrix_2add(X)
    eq22_output, eq22_coalitions = choquet_matrix_unified(X, k_add=2)
    
    print("\n1. Direct outputs from both implementations:")
    print(f"Equation (23) [first row]: {eq23_output[0]}")
    print(f"Equation (22) [first row]: {eq22_output[0]}")
    print(f"Coalitions in Equation (22): {eq22_coalitions}")
    
    # 2. Use the conversion function to transform between representations
    converted_to_23, transform_matrix, _ = convert_between_representations(X, from_eq22=True)
    
    print("\n2. Conversion from Equation (22) to Equation (23):")
    print(f"Original Eq(23) [first row]: {eq23_output[0]}")
    print(f"Converted from Eq(22) [first row]: {converted_to_23[0]}")
    
    # Verify accuracy of conversion
    conversion_error = np.linalg.norm(converted_to_23 - eq23_output) / np.linalg.norm(eq23_output)
    print(f"Relative conversion error: {conversion_error:.8f}")
    
    if conversion_error < 1e-6:
        print("✓ VERIFIED: Equation (22) can be accurately converted to Equation (23)")
    else:
        print("❌ FAILED: Significant error in conversion between representations")
    
    # 3. Show the detailed explanation of the transformation
    print("\n3. Mathematical Explanation of the Transformation:")
    explain_transformation(X)
    
    # 4. Practical demonstration: Prediction equivalence
    # Generate some random coefficients
    np.random.seed(100)
    coeffs_23 = np.random.rand(eq23_output.shape[1])
    
    # Make predictions with equation (23) model
    predictions_23 = eq23_output @ coeffs_23
    
    # The correct coefficient transformation is:
    # If eq23_output = eq22_output @ transform_matrix
    # Then for predictions to be equal: eq22_output @ coeffs_22 = eq23_output @ coeffs_23
    # This means: coeffs_22 = transform_matrix @ coeffs_23
    coeffs_22 = transform_matrix @ coeffs_23
    
    # Make predictions with equation (22) model using transformed coefficients
    predictions_22 = eq22_output @ coeffs_22
    
    print("\n4. Prediction Equivalence Demonstration:")
    print(f"Model coefficients in Eq(23) space: {coeffs_23}")
    print(f"Transformed coefficients for Eq(22) space: {coeffs_22}")
    
    print("\nPredictions:")
    print(f"Using Equation (23): {predictions_23[:3]}")  # Show first few predictions
    print(f"Using Equation (22): {predictions_22[:3]}")
    
    prediction_error = np.linalg.norm(predictions_23 - predictions_22) / np.linalg.norm(predictions_23)
    print(f"Relative prediction error: {prediction_error:.8f}")
    
    if prediction_error < 1e-6:
        print("✓ VERIFIED: Both implementations produce identical predictions")
    else:
        print("❌ FAILED: Models produce different predictions")
    
    # 5. Summary
    print("\n===== SUMMARY =====")
    print("The analysis confirms that:")
    print("1. Both implementations represent the same 2-additive model family")
    print("2. They differ only in their parameterization")
    print("3. There exists a perfect linear transformation between them")
    print("4. Both yield identical predictions when properly calibrated")
    print("5. The relationship is: Equation (23) = Equation (22) @ Transformation_Matrix")
    print("6. Coefficient relationship: coeffs_22 = transform_matrix @ coeffs_23")

if __name__ == "__main__":
    verify_implementation_equivalence()
