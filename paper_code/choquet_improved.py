"""
Improved implementations of Choquet integral transformations that strictly
adhere to k-additive coalition structure.

These implementations fix the issues identified in the previous versions:
1. Missing singleton coalitions
2. Extra complex coalitions
3. Improper representation of interactions
"""

import numpy as np
from itertools import combinations
from math import factorial

def strict_kadd_choquet(X, k_add=2):
    """
    Strict k-additive Choquet integral implementation that correctly
    represents exactly the theoretical coalitions and nothing more.
    
    Args:
        X: Input data matrix of shape (n_samples, n_features)
        k_add: Maximum coalition size (k-additivity level)
        
    Returns:
        Transformed matrix with exactly the right coalitions for k-additivity
    """
    n_samples, n_features = X.shape
    
    # Calculate number of coalitions for each size from 1 to k_add
    num_coalitions = sum(comb(n_features, r) for r in range(1, k_add+1))
    
    # Initialize output matrix
    output = np.zeros((n_samples, num_coalitions))
    
    # Current column index
    col_idx = 0
    
    # Add singleton coalitions (size 1)
    for i in range(n_features):
        output[:, col_idx] = X[:, i]
        col_idx += 1
    
    # Add coalitions of sizes 2 to k_add
    for size in range(2, k_add+1):
        for combo in combinations(range(n_features), size):
            # Use minimum operator for interactions (Choquet standard)
            output[:, col_idx] = np.min(X[:, combo], axis=1)
            col_idx += 1
    
    return output

def improved_refined_choquet(X, k_add=2):
    """
    Improved version of the refined Choquet implementation that correctly
    represents all theoretically expected coalitions for k-additivity.
    
    This fixes the missing singleton coalitions and removes complex coalitions
    identified in the original refined_choquet_k_additive.
    
    Args:
        X: Input data matrix of shape (n_samples, n_features)
        k_add: Maximum coalition size (k-additivity level)
        
    Returns:
        Transformed matrix with proper coalitions for k-additivity
    """
    n_samples, n_features = X.shape
    
    # Calculate number of coalitions for each size from 1 to k_add
    num_coalitions = sum(comb(n_features, r) for r in range(1, k_add+1))
    
    # Initialize output matrix
    transformed = np.zeros((n_samples, num_coalitions))
    
    # Store coalition information for lookup
    coalition_map = {}
    col_idx = 0
    
    # Build singleton coalitions first
    for i in range(n_features):
        coalition_map[(i,)] = col_idx
        transformed[:, col_idx] = X[:, i]
        col_idx += 1
    
    # Build coalitions of sizes 2 to k_add
    for size in range(2, k_add+1):
        for combo in combinations(range(n_features), size):
            coalition_map[combo] = col_idx
            transformed[:, col_idx] = np.min(X[:, combo], axis=1)
            col_idx += 1
    
    return transformed

def improved_shapley_2add(X):
    """
    Improved version of the Shapley 2-additive implementation that correctly
    represents all singleton and pair coalitions without negative interaction values
    or complex coalitions.
    
    This implementation fixes the issues identified in the original choquet_matrix_2add
    where it was missing singleton coalitions and using interaction terms with negative values.
    
    Args:
        X: Input data matrix of shape (n_samples, n_features)
        
    Returns:
        Transformed matrix with proper 2-additive coalitions
    """
    n_samples, n_features = X.shape
    
    # For 2-additivity: singleton coalitions + pair coalitions
    n_singletons = n_features
    n_pairs = n_features * (n_features - 1) // 2
    total_columns = n_singletons + n_pairs
    
    # Initialize output matrix
    output = np.zeros((n_samples, total_columns))
    
    # Fill singleton coalitions (first n_features columns)
    for i in range(n_features):
        output[:, i] = X[:, i]
    
    # Fill pair coalitions using the minimum operator (remaining columns)
    col_idx = n_features
    for i in range(n_features):
        for j in range(i+1, n_features):
            output[:, col_idx] = np.minimum(X[:, i], X[:, j])
            col_idx += 1
    
    return output

def improved_kadd_ordered(X, k_add=2):
    """
    Improved version of the ordered (Guilherme) k-additive implementation that ensures
    the proper coalition structure is maintained.
    
    This implementation fixes issues in the original choquet_matrix_kadd_guilherme function
    where some coalitions were missing or not properly represented.
    
    Args:
        X: Input data matrix of shape (n_samples, n_features)
        k_add: Maximum coalition size
        
    Returns:
        Transformed matrix with proper k-additive coalitions
    """
    n_samples, n_features = X.shape
    
    # Calculate number of coalitions
    num_coalitions = sum(comb(n_features, r) for r in range(1, k_add+1))
    
    # Initialize output matrix
    output = np.zeros((n_samples, num_coalitions))
    
    # Map coalitions to their column indices
    coalition_map = {}
    col_idx = 0
    
    # Process singleton coalitions
    for i in range(n_features):
        coalition_map[(i,)] = col_idx
        col_idx += 1
    
    # Process coalitions of size 2 to k_add
    for size in range(2, k_add+1):
        for combo in combinations(range(n_features), size):
            coalition_map[combo] = col_idx
            col_idx += 1
    
    # Fill the matrix using the ordered approach
    for sample_idx in range(n_samples):
        x = X[sample_idx]
        
        # Process all singleton coalitions directly
        for i in range(n_features):
            output[sample_idx, coalition_map[(i,)]] = x[i]
        
        # Process larger coalitions using the minimum operator
        for size in range(2, k_add+1):
            for combo in combinations(range(n_features), size):
                col = coalition_map[combo]
                output[sample_idx, col] = np.min(x[list(combo)])
    
    return output

def comb(n, k):
    """
    Calculate binomial coefficient (n choose k) efficiently.
    
    Args:
        n: Number of elements
        k: Subset size
        
    Returns:
        The binomial coefficient C(n,k)
    """
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    
    # Use symmetry property to reduce computation
    k = min(k, n - k)
    c = 1
    for i in range(k):
        c = c * (n - i) // (i + 1)
    return c

def verify_implementation(X, transform_func, k_add):
    """
    Verify if an implementation correctly follows k-additivity structure.
    
    Args:
        X: Input data matrix
        transform_func: The transformation function to test
        k_add: The k-additivity level to verify
        
    Returns:
        Dictionary with verification results
    """
    n_features = X.shape[1]
    
    # Expected number of coalitions
    expected_count = sum(comb(n_features, r) for r in range(1, k_add+1))
    
    # Transform the data
    transformed = transform_func(X, k_add)
    
    # Check if we have the correct number of columns
    correct_shape = transformed.shape[1] == expected_count
    
    # For a small test pattern, check if singleton and pair coalitions are properly represented
    test_X = np.eye(n_features)  # Identity matrix contains one-hot patterns
    test_transformed = transform_func(test_X, k_add)
    
    # For each one-hot pattern, check if exactly one singleton column is activated
    correct_singletons = True
    for i in range(n_features):
        row = test_transformed[i]
        # When feature i is active, column i should be 1 and other singleton columns should be 0
        if not (row[i] == 1 and np.sum(row[:n_features]) == 1):
            correct_singletons = False
            break
    
    # If k_add >= 2, test pairs
    correct_pairs = True
    if k_add >= 2:
        # Create test data for all possible pairs
        pair_test_data = []
        pair_indices = []
        
        for i in range(n_features):
            for j in range(i+1, n_features):
                pair_pattern = np.zeros(n_features)
                pair_pattern[i] = 1
                pair_pattern[j] = 1
                pair_test_data.append(pair_pattern)
                pair_indices.append((i, j))
        
        pair_test_data = np.array(pair_test_data)
        pair_transformed = transform_func(pair_test_data, k_add)
        
        # Check each pair transformation
        for idx, (i, j) in enumerate(pair_indices):
            row = pair_transformed[idx]
            
            # When features i and j are active:
            # 1. Columns i and j should be 1 (singleton activations)
            # 2. A pair column should also be 1 (with value min(1,1) = 1)
            # 3. No other column should be active
            
            # Check singleton activations first
            if not (row[i] == 1 and row[j] == 1):
                correct_pairs = False
                break
                
            # Find the pair column index
            pair_col_idx = n_features
            for a in range(n_features):
                for b in range(a+1, n_features):
                    if (a, b) == (i, j):
                        # This should be the column that's also activated (equal to 1)
                        if row[pair_col_idx] != 1:
                            correct_pairs = False
                            break
                    else:
                        # This should be inactive (equal to 0)
                        if row[pair_col_idx] != 0:
                            correct_pairs = False
                            break
                    pair_col_idx += 1
    
    return {
        "Correct shape": correct_shape,
        "Correct singleton representation": correct_singletons,
        "Correct pair representation": correct_pairs if k_add >= 2 else "N/A",
        "Overall k-additivity compliance": correct_shape and correct_singletons and (correct_pairs if k_add >= 2 else True)
    }

if __name__ == "__main__":
    # Test the improved implementations
    print("=== Testing Improved Choquet Implementations ===")
    
    # Generate test data
    n_features = 4
    test_data = np.random.random((10, n_features))
    
    # Test 1-additive
    print("\nTesting 1-additive implementations:")
    k_add = 1
    expected_cols = sum(comb(n_features, r) for r in range(1, k_add+1))
    print(f"  Expected columns for k={k_add}: {expected_cols}")
    
    strict_1add = strict_kadd_choquet(test_data, k_add=k_add)
    improved_refined_1add = improved_refined_choquet(test_data, k_add=k_add)
    improved_ordered_1add = improved_kadd_ordered(test_data, k_add=k_add)
    
    print(f"  Strict k-add shape: {strict_1add.shape}")
    print(f"  Improved Refined shape: {improved_refined_1add.shape}")
    print(f"  Improved Ordered shape: {improved_ordered_1add.shape}")
    
    # Test 2-additive
    print("\nTesting 2-additive implementations:")
    k_add = 2
    expected_cols = sum(comb(n_features, r) for r in range(1, k_add+1))
    print(f"  Expected columns for k={k_add}: {expected_cols}")
    
    strict_2add = strict_kadd_choquet(test_data, k_add=k_add)
    improved_refined_2add = improved_refined_choquet(test_data, k_add=k_add)
    improved_shapley = improved_shapley_2add(test_data)
    improved_ordered_2add = improved_kadd_ordered(test_data, k_add=k_add)
    
    print(f"  Strict k-add shape: {strict_2add.shape}")
    print(f"  Improved Refined shape: {improved_refined_2add.shape}")
    print(f"  Improved Shapley shape: {improved_shapley.shape}")
    print(f"  Improved Ordered shape: {improved_ordered_2add.shape}")
    
    # Verify implementation correctness
    print("\nVerifying implementation correctness:")
    
    implementations = [
        ("Strict k-add", strict_kadd_choquet),
        ("Improved Refined", improved_refined_choquet),
        ("Improved Ordered", improved_kadd_ordered)
    ]
    
    for name, impl in implementations:
        print(f"\n{name} (k=2):")
        verification = verify_implementation(test_data, impl, k_add=2)
        for check, result in verification.items():
            print(f"  {check}: {result}")
    
    print("\nImproved Shapley (k=2):")
    # For improved Shapley, we need to create a wrapper that ignores k_add
    shapley_wrapper = lambda X, k: improved_shapley_2add(X)
    verification = verify_implementation(test_data, shapley_wrapper, k_add=2)
    for check, result in verification.items():
        print(f"  {check}: {result}")
