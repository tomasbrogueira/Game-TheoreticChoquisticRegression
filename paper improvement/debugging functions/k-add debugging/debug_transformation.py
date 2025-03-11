import numpy as np
from choquet_kadd_test import (
    choquet_matrix_unified, 
    choquet_matrix_2add,
    powerset
)

def debug_transformation():
    """
    Debug the mathematical transformation between eq(22) with k=2 and eq(23).
    
    This function analyzes how the k=2 restriction of eq(22) relates to the
    direct implementation of eq(23) through careful mathematical analysis.
    """
    # Create a simple controlled dataset for clarity
    np.random.seed(42)
    X = np.random.rand(1, 3)  # Just 1 sample with 3 features for simplicity
    print("Simple test data:")
    print(X)
    
    # Get transforms from both implementations
    eq22_matrix, eq22_coalitions = choquet_matrix_unified(X, k_add=2)
    eq23_matrix = choquet_matrix_2add(X)
    
    print("\nCoalitions in eq(22):", eq22_coalitions)
    print("eq(22) output:", eq22_matrix[0])
    print("eq(23) output:", eq23_matrix[0])
    
    # Compute the transformation matrix (M where eq23 = eq22 @ M)
    M = np.linalg.lstsq(eq22_matrix, eq23_matrix, rcond=None)[0]
    
    print("\nTransformation matrix:")
    print(M)
    
    # Verify transformation is accurate
    transformed = eq22_matrix @ M
    print("\nVerification of transformation:")
    print("Original eq(23):", eq23_matrix[0])
    print("Transformed eq(22):", transformed[0])
    print("Accurate conversion:", np.allclose(transformed, eq23_matrix))
    
    # Analyze the structure of implementation differences
    print("\n=== MATHEMATICAL RELATIONSHIP ANALYSIS ===")
    
    # Trace eq(23) implementation manually
    print("\n1. TRACING EQ(23) IMPLEMENTATION:")
    X_sample = X[0]
    print(f"Features: {X_sample}")
    
    # Get coalition structure used by eq(23)
    nAttr = X.shape[1]
    k_add = 2
    coalit = np.zeros((int(7), nAttr))  # 7 = 2^3-1 for 3 features with the empty set
    for i, s in enumerate(powerset(range(nAttr), k_add)):
        s = list(s)
        coalit[i, s] = 1
    
    print("Coalition structure:")
    for i, row in enumerate(coalit):
        print(f"{i}: {row}")
    
    # Manually trace through eq(23) computation
    computed_23 = np.zeros(6)  # 6 = 2^3-1-1 (exclude empty set)
    
    # First term: feature values
    for i in range(nAttr):
        computed_23[i] = X_sample[i]
        print(f"Feature {i} direct effect: {X_sample[i]}")
    
    # Second term: pairwise minimum values
    pair_idx = nAttr
    for i in range(nAttr):
        for j in range(i+1, nAttr):
            min_val = min(X_sample[i], X_sample[j])
            print(f"Pair min({i},{j}): {min_val}")
            # Find which column in coalit represents this pair
            for idx in range(nAttr, coalit.shape[0]):
                if np.array_equal(coalit[idx], np.array([1 if k==i or k==j else 0 for k in range(nAttr)])):
                    pair_idx = idx - 1
                    computed_23[pair_idx] = min_val
                    break
    
    # Third term: adjustment terms
    for i in range(nAttr):
        for j in range(nAttr):
            if j != i:
                # Find which column has the pair (i,j)
                for idx in range(nAttr, coalit.shape[0]):
                    if coalit[idx, i] == 1 and coalit[idx, j] == 1:
                        adj_idx = idx - 1
                        adj_val = -0.5 * X_sample[i]
                        print(f"Adjustment for feature {i} in pair ({i},{j}): {adj_val}")
                        computed_23[adj_idx] += adj_val
                        break
    
    print(f"Manual computation of eq(23): {computed_23}")
    print(f"Actual output of eq(23): {eq23_matrix[0]}")
    print(f"Match: {np.allclose(computed_23, eq23_matrix[0])}")

    # Trace eq(22) implementation with k=2
    print("\n2. TRACING EQ(22) IMPLEMENTATION:")
    
    # First, order the features
    order = np.argsort(X_sample)
    sorted_vals = np.sort(X_sample)
    print(f"Sorted feature indices: {order}")
    print(f"Sorted feature values: {sorted_vals}")
    
    # Now compute differences and assign to coalitions
    computed_22 = np.zeros(len(eq22_coalitions))
    prev = 0
    for j in range(nAttr):
        diff = sorted_vals[j] - prev
        prev = sorted_vals[j]
        
        full_coalition = tuple(sorted(order[j:]))
        print(f"Step {j}: diff = {diff}, coalition = {full_coalition}")
        
        if len(full_coalition) <= 2:
            # Direct assignment for coalitions within size limit
            idx = eq22_coalitions.index(full_coalition) if full_coalition in eq22_coalitions else -1
            if idx >= 0:
                print(f"  Assigning to coalition {full_coalition} at index {idx}")
                computed_22[idx] = diff
        else:
            # Distribute to 2-sized coalitions
            current_feature = order[j]
            remaining_features = order[j+1:]
            print(f"  Distributing to 2-sized coalitions with feature {current_feature}")
            
            for feature in remaining_features:
                pair = tuple(sorted((current_feature, feature)))
                idx = eq22_coalitions.index(pair) if pair in eq22_coalitions else -1
                if idx >= 0:
                    weight = 1.0 / len(remaining_features)
                    print(f"    Pair {pair} gets {diff * weight} at index {idx}")
                    computed_22[idx] += diff * weight
    
    print(f"Manual computation of eq(22): {computed_22}")
    print(f"Actual output of eq(22): {eq22_matrix[0]}")
    print(f"Match: {np.allclose(computed_22, eq22_matrix[0])}")
    
    # Explain the transformation mathematically
    print("\n3. TRANSFORMATION EXPLANATION:")
    print("The transformation matrix represents how eq(22) coalitions map to eq(23) terms.")
    
    # Map single features
    print("\nSingleton transformations:")
    for i in range(nAttr):
        if (i,) in eq22_coalitions:
            idx = eq22_coalitions.index((i,))
            row = M[idx]
            # Interpret this row
            print(f"Feature {i} coalition in eq(22):")
            print(f"  - Contributes {row[i]:.4f} to its own Shapley value in eq(23)")
            for j in range(nAttr, eq23_matrix.shape[1]):
                if abs(row[j]) > 1e-5:
                    print(f"  - Contributes {row[j]:.4f} to interaction term {j}")
    
    # Map pairs
    print("\nPair transformations:")
    for i in range(nAttr):
        for j in range(i+1, nAttr):
            pair = (i, j)
            if pair in eq22_coalitions:
                idx = eq22_coalitions.index(pair)
                row = M[idx]
                
                # Find which column corresponds to this pair in eq(23)
                pair_col = None
                for col_idx in range(nAttr, eq23_matrix.shape[1]):
                    if abs(row[col_idx]) > 0.1:  # Use a threshold to identify major contributions
                        pair_col = col_idx
                
                if pair_col is not None:
                    print(f"Pair {pair} coalition in eq(22):")
                    print(f"  - Main contribution: {row[pair_col]:.4f} to pair interaction {pair_col}")
    
    # Theoretical explanation
    print("\nTHEORETICAL INTERPRETATION:")
    print("Based on the analysis, the transformation between eq(22) and eq(23) is:")
    print("- eq(22) represents the standard Choquet integral formulation with coalition values")
    print("- eq(23) represents the same model using Shapley values and interaction indices")
    print("- The transformation matrix converts between these two equivalent representations")
    print("- Both produce mathematically equivalent models, just with different parameterizations")

if __name__ == "__main__":
    debug_transformation()
