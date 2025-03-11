import numpy as np
from choquet_kadd_test import choquet_matrix, choquet_matrix_new, choquet_matrix_2add, powerset

def test_choquet_equivalence():
    """
    Test if choquet_matrix_new produces equivalent results to:
    1. choquet_matrix when k_add=None
    2. choquet_matrix_2add when k_add=2
    """
    # Create test data
    np.random.seed(42)  # For reproducibility
    X = np.random.rand(5, 3)  # 5 samples, 3 features
    
    print("Test data:")
    print(X)
    
    # Test 1: Compare with choquet_matrix when k_add=None
    print("\nTest 1: choquet_matrix_new(k_add=None) vs choquet_matrix")
    transformed_orig, coalitions_orig = choquet_matrix(X)
    transformed_new, coalitions_new = choquet_matrix_new(X, k_add=None)
    
    # Create mappings to align coalitions (they might be in different order)
    orig_to_index = {coal: i for i, coal in enumerate(coalitions_orig)}
    new_to_index = {coal: i for i, coal in enumerate(coalitions_new)}
    
    # Check if they have the same coalitions
    if set(coalitions_orig) != set(coalitions_new):
        print("❌ TEST 1 FAILED: Different coalition sets")
        print(f"Original coalitions: {coalitions_orig}")
        print(f"New coalitions: {coalitions_new}")
        return
    
    # Create matrices with aligned coalitions
    aligned_orig = np.zeros_like(transformed_orig)
    aligned_new = np.zeros_like(transformed_new)
    
    for coal in set(coalitions_orig):
        orig_idx = orig_to_index[coal]
        new_idx = new_to_index[coal]
        aligned_orig[:, orig_idx] = transformed_orig[:, orig_idx]
        aligned_new[:, orig_idx] = transformed_new[:, new_idx]
    
    # Check if aligned matrices match
    is_equal = np.allclose(aligned_orig, aligned_new, rtol=1e-5, atol=1e-8)
    
    if is_equal:
        print("✓ TEST 1 PASSED: choquet_matrix_new(k_add=None) matches choquet_matrix")
    else:
        print("❌ TEST 1 FAILED: Matrices don't match")
        print("Max difference:", np.max(np.abs(aligned_orig - aligned_new)))
    
    # Test 2: Analyze the differences between choquet_matrix_new and choquet_matrix_2add
    print("\nTest 2: Analyzing choquet_matrix_new(k_add=2) vs choquet_matrix_2add")
    
    # Get transformation from both functions
    transformed_2add = choquet_matrix_2add(X)
    transformed_new_2add, coalitions_new_2add = choquet_matrix_new(X, k_add=2)
    
    print(f"choquet_matrix_2add output shape: {transformed_2add.shape}")
    print(f"choquet_matrix_new output shape: {transformed_new_2add.shape}")
    
    # Print coalitions from choquet_matrix_new
    print("\nCoalitions in choquet_matrix_new:")
    for i, coal in enumerate(coalitions_new_2add):
        print(f"{i}: {coal}")
    
    # Understanding the structure of choquet_matrix_2add
    print("\nExpected coalitions structure in choquet_matrix_2add:")
    for i, s in enumerate(powerset(range(X.shape[1]), 2)):
        print(f"{i}: {s}")
        
    # Create a simple linear transformation test
    # Try to find a linear transformation matrix that converts from one to the other
    print("\n===== TRANSFORMATION ANALYSIS =====")
    print("Testing if there's a linear transformation between representations...")
    
    # Use pseudoinverse to find a transformation matrix
    # We want to solve: transformed_2add = transformed_new_2add @ transform_matrix
    try:
        transform_matrix = np.linalg.pinv(transformed_new_2add) @ transformed_2add
        predicted_2add = transformed_new_2add @ transform_matrix
        
        # Check if this transformation works
        transformation_error = np.linalg.norm(predicted_2add - transformed_2add) / np.linalg.norm(transformed_2add)
        print(f"Relative transformation error: {transformation_error:.6f}")
        
        if transformation_error < 1e-5:
            print("✓ Found a linear transformation between the implementations!")
            print("\nTransformation matrix:")
            print(transform_matrix)
        else:
            print("❌ No simple linear transformation exists between implementations.")
    except np.linalg.LinAlgError:
        print("❌ Could not compute a transformation matrix (singular values issue).")
    
    # Test direct values for a better understanding
    print("\n===== VALUE INSPECTION =====")
    print("Inspecting first data point values for each coalition:")
    
    # For the first data point:
    i = 0  # first sample
    x = X[i]
    print(f"Original features: {x}")
    
    print("\nchoquet_matrix_2add values:")
    print(transformed_2add[i])
    
    print("\nchoquet_matrix_new values:")
    for j, coal in enumerate(coalitions_new_2add):
        print(f"{coal}: {transformed_new_2add[i, j]}")
        
    # Compute min values for each pair to analyze 2-additive structure
    print("\nMin values for feature pairs:")
    for j in range(X.shape[1]):
        for k in range(j+1, X.shape[1]):
            print(f"min({j},{k}) = min({x[j]:.6f}, {x[k]:.6f}) = {min(x[j], x[k]):.6f}")
    
    # Test if a weighted sum of transformed_new_2add can equal transformed_2add
    print("\nTrying to find coefficients to match implementations for this sample...")
    try:
        # For first sample only
        sample_coeffs = np.linalg.lstsq(transformed_new_2add[i:i+1].T, 
                                        transformed_2add[i:i+1].T, rcond=None)[0]
        
        print("Coefficients to transform choquet_matrix_new to choquet_matrix_2add:")
        print(sample_coeffs.flatten())
    except np.linalg.LinAlgError:
        print("❌ Could not find coefficients (singular values issue).")

if __name__ == "__main__":
    test_choquet_equivalence()
