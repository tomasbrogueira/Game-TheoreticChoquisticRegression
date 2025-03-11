import numpy as np
from regression_classes import compute_shapley_values, compute_choquet_interaction_matrix

def verify_shapley_decomposition_relationship(v, all_coalitions, feature_names=None):
    """
    Verify the mathematical relationship: φᵢ = v({i}) + 0.5 * Σⱼ≠ᵢ I({i,j})
    
    This function performs a detailed analysis of the calculations to determine 
    why the matrix method and Shapley-marginal method might give different results.
    
    Parameters:
    -----------
    v : array-like
        Game/capacity values (with v[0] representing empty set)
    all_coalitions : list of tuples
        List of all coalitions in the model
    feature_names : list or None
        Names of features for more readable output
        
    Returns:
    --------
    dict : Contains comparison results and details for analysis
    """
    # Get number of features
    m = max(max(c) for c in all_coalitions if c) + 1
    
    # 1. Compute Shapley values
    shapley_values = compute_shapley_values(v, m, all_coalitions)
    
    # 2. Extract singleton/marginal values
    marginal_values = np.zeros(m)
    for i in range(m):
        singleton_tuple = (i,)
        if singleton_tuple in all_coalitions:
            idx = all_coalitions.index(singleton_tuple)
            marginal_values[i] = v[idx + 1]  # +1 for empty set at v[0]
    
    # 3. Compute interaction matrix
    interaction_matrix = compute_choquet_interaction_matrix(v, m, all_coalitions)
    
    # 4. Calculate overall interaction by both methods
    method1 = shapley_values - marginal_values  # Shapley - Marginal
    method2 = 0.5 * np.sum(interaction_matrix, axis=1)  # Matrix Method
    
    # 5. Compare results
    diff = method1 - method2
    max_diff = np.max(np.abs(diff))
    avg_diff = np.mean(np.abs(diff))
    
    print("\n=== VERIFICATION OF SHAPLEY DECOMPOSITION ===")
    print(f"Maximum absolute difference: {max_diff:.8f}")
    print(f"Average absolute difference: {avg_diff:.8f}")
    
    # 6. Examine specific cases in detail
    worst_idx = np.argmax(np.abs(diff))
    feature_name = feature_names[worst_idx] if feature_names else f"Feature {worst_idx}"
    
    print(f"\nDetailed analysis for {feature_name} (worst case):")
    print(f"  Shapley value: {shapley_values[worst_idx]:.6f}")
    print(f"  Marginal value: {marginal_values[worst_idx]:.6f}")
    print(f"  Shapley - Marginal = {method1[worst_idx]:.6f}")
    print(f"  0.5 * Sum of interactions = {method2[worst_idx]:.6f}")
    print(f"  Difference: {diff[worst_idx]:.6f}")
    
    # 7. Raw matrix analysis for worst case
    row = interaction_matrix[worst_idx]
    print(f"\nInteraction matrix row sum (excluding diagonal): {np.sum(row) - row[worst_idx]:.6f}")
    
    # 8. Check for implementation issues
    if np.allclose(diff, 0, atol=1e-8):
        print("\nCONCLUSION: Mathematical relationship holds (within numerical precision)")
    else:
        print("\nCONCLUSION: Discrepancy detected - mathematical relationship does not hold")
        print("Possible causes:")
        print("  1. Different coalition value retrieval between calculations")
        print("  2. Inconsistent handling of the empty set")
        print("  3. Error in interaction indices calculation")
        print("  4. Normalization factor inconsistency")
    
    # Try different interpretation of the relationship
    print("\nTrying alternative interpretations of the relationship:")
    
    # Alternative 1: Check if interaction signs might be reversed
    method2_alt1 = -0.5 * np.sum(interaction_matrix, axis=1)
    diff_alt1 = method1 - method2_alt1
    print(f"1. Using reversed signs: max diff = {np.max(np.abs(diff_alt1)):.8f}")
    
    # Alternative 2: Check if missing normalization factor
    # Try common normalization factors
    for factor in [2.0, m-1, factorial(m-1), factorial(m)/factorial(m-2)]:
        method2_alt2 = 0.5 * factor * np.sum(interaction_matrix, axis=1)
        diff_alt2 = method1 - method2_alt2
        print(f"2. With normalization factor {factor}: max diff = {np.max(np.abs(diff_alt2)):.8f}")
        
        # If this factor fixes the issue, show details
        if np.max(np.abs(diff_alt2)) < 1e-4:
            print(f"   *** Solution found! Normalization factor {factor} fixes the discrepancy ***")
            print(f"   For feature {worst_idx} ({feature_name}):")
            print(f"     Shapley - Marginal = {method1[worst_idx]:.6f}")
            print(f"     0.5 * {factor} * Sum of interactions = {method2_alt2[worst_idx]:.6f}")
            print(f"     Difference: {diff_alt2[worst_idx]:.6f}")
    
    return {
        "shapley_values": shapley_values,
        "marginal_values": marginal_values,
        "interaction_matrix": interaction_matrix,
        "method1": method1,
        "method2": method2,
        "diff": diff
    }

def recompute_interaction_matrix(v, m, all_coalitions):
    """
    Recompute the interaction matrix with extensive validation.
    This functions walks through the calculation step by step to identify issues.
    
    Parameters:
    -----------
    v : array-like
        Game/capacity values (with v[0] representing empty set)
    m : int
        Number of features
    all_coalitions : list of tuples
        List of all coalitions in the model
        
    Returns:
    --------
    np.ndarray : Recomputed interaction matrix
    """
    interaction_matrix = np.zeros((m, m))
    coalition_to_index = {coal: idx for idx, coal in enumerate(all_coalitions)}
    coalition_values = {}  # For memoization
    
    def get_coalition_value(coalition):
        """Get coalition value with explicit validation"""
        if not coalition:
            return 0.0  # Empty coalition
        
        # Use memoization
        if coalition in coalition_values:
            return coalition_values[coalition]
        
        # Direct lookup with validation
        if coalition in coalition_to_index:
            idx = coalition_to_index[coalition]
            if 0 <= idx + 1 < len(v):  # Check bounds: +1 for empty set
                value = v[idx + 1]
                coalition_values[coalition] = value
                return value
            else:
                print(f"WARNING: Index out of bounds for {coalition} -> {idx + 1}")
                coalition_values[coalition] = 0.0
                return 0.0
        else:
            coalition_values[coalition] = 0.0
            return 0.0
    
    # Compute for a specific pair to trace the calculation
    sample_i, sample_j = 0, 1  # First two features
    sample_others = [k for k in range(m) if k not in (sample_i, sample_j)]
    sample_total = 0.0
    print(f"\nTracing interaction calculation for features {sample_i} and {sample_j}:")
    
    # Loop over all pairs
    for i in range(m):
        for j in range(i+1, m):
            others = [feat for feat in range(m) if feat not in (i, j)]
            total = 0.0
            
            for r in range(len(others) + 1):
                for B in itertools.combinations(others, r):
                    B_tuple = tuple(sorted(B))
                    Bi_tuple = tuple(sorted(B + (i,)))
                    Bj_tuple = tuple(sorted(B + (j,)))
                    Bij_tuple = tuple(sorted(B + (i, j)))
                    
                    # Get coalition values
                    vB = get_coalition_value(B_tuple)
                    vBi = get_coalition_value(Bi_tuple)
                    vBj = get_coalition_value(Bj_tuple)
                    vBij = get_coalition_value(Bij_tuple)
                    
                    # Calculate weight according to Shapley interaction formula
                    weight = factorial(m - r - 2) * factorial(r) / factorial(m - 1)
                    
                    # For sample pair, print the calculation
                    if (i, j) == (sample_i, sample_j):
                        if len(B) <= 2:  # Only print a subset to avoid clutter
                            term = weight * (vBij - vBi - vBj + vB)
                            print(f"  B={B_tuple}: weight={weight:.5f}, values={vBij:.5f}-{vBi:.5f}-{vBj:.5f}+{vB:.5f} = {term:.5f}")
                            sample_total += term
                    
                    total += weight * (vBij - vBi - vBj + vB)
            
            interaction_matrix[i, j] = total
            interaction_matrix[j, i] = total  # Ensure symmetry
    
    if (sample_i, sample_j) == (0, 1):
        print(f"  Final interaction I({sample_i},{sample_j}) = {sample_total:.6f}")
        print(f"  Matrix value: {interaction_matrix[sample_i, sample_j]:.6f}")
    
    # After recomputation, try verification with different factorials
    print("\nVerifying with different normalizations:")
    
    # Get shapley values and marginal values
    shapley_values = compute_shapley_values(v, m, all_coalitions)
    marginal_values = np.zeros(m)
    for i in range(m):
        if (i,) in all_coalitions:
            idx = all_coalitions.index((i,))
            marginal_values[i] = v[idx + 1]
    
    # Method 1: Direct computation
    method1 = shapley_values - marginal_values
    
    # Try different normalizations of the interaction matrix
    for factor_name, factor in [("None", 1.0), ("2", 2.0), ("m-1", m-1), 
                            ("(m-1)!", factorial(m-1)), ("m!/(m-2)!", factorial(m)/factorial(m-2))]:
        method2 = 0.5 * factor * np.sum(interaction_matrix, axis=1)
        diff = method1 - method2
        print(f"  Factor {factor_name}: max diff = {np.max(np.abs(diff)):.8f}")
        
        # If we find a good match, provide details
        if np.max(np.abs(diff)) < 0.01:
            print(f"  *** Good match with factor {factor_name} = {factor} ***")
            print(f"  Average absolute difference with this factor: {np.mean(np.abs(diff)):.8f}")
    
    return interaction_matrix

if __name__ == "__main__":
    import itertools
    import os
    from math import factorial
    
    # Set the correct path to the debug file
    debug_file_paths = [
        'model_debug.pkl',  # Try current directory first
        os.path.join('plots', 'dados_covid_sbpo_atual', 'model_debug.pkl'),  # Then try dataset-specific folder
    ]
    
    # Try to load the model data from any of the possible locations
    debug_data = None
    for path in debug_file_paths:
        try:
            import pickle
            print(f"Trying to load debug data from: {path}")
            with open(path, 'rb') as f:
                debug_data = pickle.load(f)
            print(f"Successfully loaded data from {path}")
            break
        except (FileNotFoundError, Exception) as e:
            print(f"Could not load debug data from {path}: {e}")
    
    if debug_data:
        v = debug_data.get('v')
        all_coalitions = debug_data.get('all_coalitions')
        feature_names = debug_data.get('feature_names')
        
        # If successful, run full verification
        if v is not None and all_coalitions is not None:
            results = verify_shapley_decomposition_relationship(v, all_coalitions, feature_names)
            
            # If discrepancy found, recompute with detailed tracing
            if np.max(np.abs(results['diff'])) > 1e-6:
                print("\nRecomputing interaction matrix with detailed tracing...")
                m = len(results['shapley_values'])
                new_matrix = recompute_interaction_matrix(v, m, all_coalitions)
                
                # Compare with original matrix
                orig_matrix = results['interaction_matrix']
                matrix_diff = new_matrix - orig_matrix
                print(f"\nMatrix recomputation max difference: {np.max(np.abs(matrix_diff)):.8f}")
                
                # Check if recomputation fixes the relationship
                method2_fixed = 0.5 * np.sum(new_matrix, axis=1)
                diff_fixed = results['method1'] - method2_fixed
                print(f"Relationship discrepancy after fix: {np.max(np.abs(diff_fixed)):.8f}")
    else:
        print("Creating simple test case instead...")
        
        # Create a simple test case
        m = 3  # 3 features
        all_coalitions = []
        for r in range(1, m + 1):
            all_coalitions.extend(list(itertools.combinations(range(m), r)))
        
        # Simple capacity function (values increase with coalition size)
        v = np.zeros(len(all_coalitions) + 1)  # +1 for empty set
        for i, coal in enumerate(all_coalitions):
            v[i+1] = len(coal) / m  # Normalized by number of features
            
        verify_shapley_decomposition_relationship(v, all_coalitions)
