import numpy as np
from choquet_kadd_test import (
    choquet_matrix, 
    choquet_matrix_new, 
    choquet_matrix_2add, 
    choquet_matrix_unified
)

def test_unified_choquet_implementations():
    """
    Comprehensive test for the unified Choquet integral implementation.
    Tests against all existing implementations with various parameter settings.
    """
    # Generate test data
    np.random.seed(42)
    X = np.random.rand(5, 3)
    
    print("Test data:")
    print(X)
    
    # Test 1: Full Choquet integral
    print("\n===== Test 1: Full Choquet integral =====")
    print("Comparing choquet_matrix_unified(k_add=None) with choquet_matrix")
    
    # Get matrices from both implementations
    orig_matrix, orig_coalitions = choquet_matrix(X)
    unified_matrix, unified_coalitions = choquet_matrix_unified(X, k_add=None)
    
    # Check if they have the same coalitions
    coalitions_match = set(orig_coalitions) == set(unified_coalitions)
    print(f"Same coalitions: {'✓' if coalitions_match else '❌'}")
    
    if coalitions_match:
        # Create mappings to align matrices
        orig_to_idx = {coal: i for i, coal in enumerate(orig_coalitions)}
        uni_to_idx = {coal: i for i, coal in enumerate(unified_coalitions)}
        
        # Create aligned matrices
        aligned_orig = np.zeros_like(orig_matrix)
        aligned_uni = np.zeros_like(unified_matrix)
        
        for coal in set(orig_coalitions):
            orig_idx = orig_to_idx[coal]
            uni_idx = uni_to_idx[coal]
            aligned_orig[:, orig_idx] = orig_matrix[:, orig_idx]
            aligned_uni[:, orig_idx] = unified_matrix[:, uni_idx]
        
        # Compare values
        values_match = np.allclose(aligned_orig, aligned_uni)
        print(f"Values match: {'✓' if values_match else '❌'}")
        
        if not values_match:
            print(f"Max difference: {np.max(np.abs(aligned_orig - aligned_uni))}")
    
    # Test 2: Comparison between 2-additive implementation of eq(22) and eq(23)
    print("\n===== Test 2: Mathematical equivalence analysis =====")
    print("Comparing k-additive eq(22) with k=2 vs direct eq(23) implementation")
    
    # Get matrices from both implementations
    eq23_matrix = choquet_matrix_2add(X)
    eq22_matrix, eq22_coalitions = choquet_matrix_unified(X, k_add=2)
    
    # Print dimensions
    print(f"Equation (23) output shape: {eq23_matrix.shape}")
    print(f"Equation (22) with k=2 output shape: {eq22_matrix.shape}")
    print(f"Coalitions in eq(22): {eq22_coalitions}")
    
    # Analyze if there's a linear transformation between them
    print("\nAnalyzing if there's a linear transformation between implementations...")
    try:
        # Try to find a transformation matrix M such that eq23_matrix = eq22_matrix @ M
        M = np.linalg.lstsq(eq22_matrix, eq23_matrix, rcond=None)[0]
        predicted = eq22_matrix @ M
        
        # Check how close the prediction is
        error = np.linalg.norm(predicted - eq23_matrix) / np.linalg.norm(eq23_matrix)
        print(f"Relative transformation error: {error:.6f}")
        
        if error < 1e-5:
            print("✓ Found a linear transformation between implementations!")
            
            # Detailed analysis of the transformation matrix
            print("\nTransformation Matrix Analysis:")
            print("--------------------------------")
            print("This matrix maps from equation (22) coalitions to equation (23) form:")
            for i, coal in enumerate(eq22_coalitions):
                print(f"Coalition {coal} contributions:")
                for j in range(eq23_matrix.shape[1]):
                    if abs(M[i, j]) > 1e-5:  # Only show non-zero contributions
                        print(f"  → Output {j+1}: {M[i, j]:.4f}")
            
            # Calculate and verify the relationship with specific examples
            print("\nVerification with first data point:")
            first_sample_22 = eq22_matrix[0]
            first_sample_23 = eq23_matrix[0]
            print(f"eq(22) outputs: {first_sample_22}")
            print(f"eq(23) outputs: {first_sample_23}")
            print(f"Transformed: {first_sample_22 @ M}")
            
            # Attempt to express the transformation in terms of mathematical operations
            print("\nInterpretation of Transformation:")
            nAttr = X.shape[1]
            
            # Analyze transformation for singleton coalitions
            print("Singleton mappings:")
            for i in range(nAttr):
                singleton_idx = eq22_coalitions.index((i,))
                singleton_row = M[singleton_idx]
                print(f"Feature {i}: {singleton_row}")
            
            # Analyze transformation for pair coalitions
            print("\nPair mappings:")
            for i in range(nAttr):
                for j in range(i+1, nAttr):
                    try:
                        pair_idx = eq22_coalitions.index((i, j))
                        pair_row = M[pair_idx]
                        print(f"Pair ({i},{j}): {pair_row}")
                    except ValueError:
                        print(f"Pair ({i},{j}) not found in coalitions")
        else:
            print("❌ Linear transformation has significant error.")
            print("The implementations likely represent different mathematical formulations.")
    
    except np.linalg.LinAlgError:
        print("❌ Could not compute transformation (singular matrix).")
    
    # Test 3: 2-additive general approach matching
    print("\n===== Test 3: 2-additive general approach =====")
    print("Comparing choquet_matrix_unified(k_add=2) with choquet_matrix_new(k_add=2)")
    
    # Get matrices from both implementations
    new_2add, new_coalitions = choquet_matrix_new(X, k_add=2)
    unified_gen, unified_gen_coalitions = choquet_matrix_unified(X, k_add=2)
    
    # Check if they have the same coalitions
    coalitions_match_gen = set(new_coalitions) == set(unified_gen_coalitions)
    print(f"Same coalitions: {'✓' if coalitions_match_gen else '❌'}")
    
    if coalitions_match_gen:
        # Create mappings to align matrices
        new_to_idx = {coal: i for i, coal in enumerate(new_coalitions)}
        uni_to_idx = {coal: i for i, coal in enumerate(unified_gen_coalitions)}
        
        # Create aligned matrices
        aligned_new = np.zeros_like(new_2add)
        aligned_uni = np.zeros_like(unified_gen)
        
        for coal in set(new_coalitions):
            new_idx = new_to_idx[coal]
            uni_idx = uni_to_idx[coal]
            aligned_new[:, new_idx] = new_2add[:, new_idx]
            aligned_uni[:, new_idx] = unified_gen[:, uni_idx]
        
        # Compare values
        values_match_gen = np.allclose(aligned_new, aligned_uni)
        print(f"Values match: {'✓' if values_match_gen else '❌'}")
        
        if not values_match_gen:
            print(f"Max difference: {np.max(np.abs(aligned_new - aligned_uni))}")
    
    # Overall summary
    print("\n===== SUMMARY =====")
    print(f"Test 1 (Full Choquet): {'✓ PASSED' if coalitions_match and values_match else '❌ FAILED'}")
    
    # For Test 2, we focus on the linear relationship rather than exact equality
    if 'error' in locals() and error < 1e-5:
        print("Test 2 (Mathematical equivalence): ✓ PASSED - Found linear transformation")
    else:
        print("Test 2 (Mathematical equivalence): ❌ FAILED - Different mathematical formulations")
        
    print(f"Test 3 (2-additive implementation): {'✓ PASSED' if coalitions_match_gen and values_match_gen else '❌ FAILED'}")

if __name__ == "__main__":
    test_unified_choquet_implementations()
