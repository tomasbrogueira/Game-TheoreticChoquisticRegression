"""
Analysis and fix for the Shapley 2-additive implementation.

The current Shapley implementation doesn't correctly model 2-additive coalitions.
This file provides analysis and an improved implementation.
"""

import numpy as np
from itertools import combinations
from choquet_function import choquet_matrix_2add
from paper_code.k_add_test import refined_choquet_k_additive
from coalition_analysis_test import create_simple_patterns, identify_coalitions

def analyze_shapley_representation():
    """Analyze how the Shapley implementation represents coalitions"""
    n_features = 4
    patterns, pattern_names = create_simple_patterns(n_features)
    
    print("=== Shapley Implementation Representation Analysis ===")
    
    # Get transformations
    shapley_matrix = choquet_matrix_2add(patterns)
    refined_matrix = refined_choquet_k_additive(patterns, k_add=2)
    
    print(f"Shapley matrix shape: {shapley_matrix.shape}")
    print(f"Refined (k=2) matrix shape: {refined_matrix.shape}")
    
    # Identify coalitions
    shapley_coalitions = identify_coalitions(shapley_matrix, 2, n_features, patterns)
    
    # Look at the first few columns of the Shapley matrix
    print("\nShapley matrix first 5 columns for important patterns:")
    important_patterns = ["Feature 0 only", "Feature 1 only", "Features 0 & 1", "All ones"]
    
    for pattern_name in important_patterns:
        i = pattern_names.index(pattern_name)
        print(f"\n  {pattern_name}:")
        for col in range(min(5, shapley_matrix.shape[1])):
            val = shapley_matrix[i, col]
            if abs(val) > 1e-10:
                print(f"    Column {col} ({shapley_coalitions.get(col, 'Unknown')}): {val:.4f}")
    
    # Check if columns in Shapley implementation represent interaction indices
    print("\nPossible Shapley representation interpretation:")
    
    # For each column, check how it responds to different patterns
    for col in range(shapley_matrix.shape[1]):
        # Find all patterns that activate this column
        active_patterns = []
        for i, pattern_name in enumerate(pattern_names):
            if abs(shapley_matrix[i, col]) > 1e-10:
                active_patterns.append((pattern_name, shapley_matrix[i, col]))
        
        if len(active_patterns) > 0:
            desc = shapley_coalitions.get(col, "Unknown")
            print(f"\n  Column {col} ({desc}):")
            print(f"    Activated by {len(active_patterns)} patterns")
            for name, val in active_patterns[:3]:  # Show first 3 for brevity
                print(f"      {name}: {val:.4f}")
            if len(active_patterns) > 3:
                print(f"      ... and {len(active_patterns) - 3} more")

def create_fixed_shapley_implementation(X, correct_2add=True):
    """
    Create a fixed Shapley implementation that properly models 2-additive coalitions
    
    Args:
        X: Input matrix of shape (n_samples, n_features)
        correct_2add: If True, use correct 2-additive coalition structure
                     If False, use the original implementation
    
    Returns:
        Transformed matrix using Shapley-based features with proper coalitions
    """
    if not correct_2add:
        # Return the original implementation
        return choquet_matrix_2add(X)
    
    # Get shape info
    n_samples, n_features = X.shape
    
    # Calculate the number of expected columns for 2-additivity
    n_singletons = n_features  # All size-1 subsets
    n_pairs = n_features * (n_features - 1) // 2  # All size-2 subsets
    total_columns = n_singletons + n_pairs
    
    # Create output matrix
    output = np.zeros((n_samples, total_columns))
    
    # Fill in singleton (size 1) coalitions
    for i in range(n_features):
        output[:, i] = X[:, i]
    
    # Fill in pair (size 2) coalitions
    col_idx = n_features
    for i in range(n_features):
        for j in range(i+1, n_features):
            # Create interaction term similar to Shapley interaction index
            # We use min(x_i, x_j) for 2-additive Choquet integrals
            output[:, col_idx] = np.minimum(X[:, i], X[:, j])
            col_idx += 1
    
    return output

def compare_implementations():
    """Compare original Shapley, fixed Shapley, and Refined implementations"""
    n_features = 4
    patterns, pattern_names = create_simple_patterns(n_features)
    
    print("\n=== Implementation Comparison ===")
    
    # Get transformations
    original_shapley = choquet_matrix_2add(patterns)
    fixed_shapley = create_fixed_shapley_implementation(patterns, correct_2add=True)
    refined_k2 = refined_choquet_k_additive(patterns, k_add=2)
    
    # Print matrix shapes
    print(f"Original Shapley: {original_shapley.shape}")
    print(f"Fixed Shapley: {fixed_shapley.shape}")
    print(f"Refined (k=2): {refined_k2.shape}")
    
    # Identify coalitions
    fixed_coalitions = identify_coalitions(fixed_shapley, 2, n_features, patterns)
    
    # Look at singular patterns to see how each implementation handles them
    singular_patterns = ["Feature 0 only", "Features 0 & 1", "All ones"]
    
    for pattern_name in singular_patterns:
        i = pattern_names.index(pattern_name)
        print(f"\n  Pattern: {pattern_name}")
        
        # Check which columns are activated in each implementation
        original_cols = np.where(np.abs(original_shapley[i]) > 1e-10)[0]
        fixed_cols = np.where(np.abs(fixed_shapley[i]) > 1e-10)[0]
        refined_cols = np.where(np.abs(refined_k2[i]) > 1e-10)[0]
        
        print(f"    Original Shapley: {len(original_cols)} columns")
        print(f"    Fixed Shapley: {len(fixed_cols)} columns")
        print(f"    Refined (k=2): {len(refined_cols)} columns")
        
        # Show activated columns for fixed Shapley
        print(f"    Fixed Shapley activated columns:")
        for col in fixed_cols[:min(3, len(fixed_cols))]:
            print(f"      Column {col} ({fixed_coalitions.get(col, 'Unknown')}): {fixed_shapley[i, col]:.4f}")
    
    # Test if fixed Shapley works properly on theoretical expectations
    single_activation_count = 0
    pair_activation_count = 0
    
    # Singleton patterns should activate exactly 1 singleton column
    for i in range(2, 2 + n_features):  # Indices for singleton patterns
        pattern_name = pattern_names[i]
        fixed_cols = np.where(np.abs(fixed_shapley[i]) > 1e-10)[0]
        
        # Should activate exactly the corresponding singleton column
        expected_col = i - 2  # Map pattern index to column index
        if expected_col in fixed_cols and len(fixed_cols) == 1:
            single_activation_count += 1
    
    # Pair patterns should activate exactly 2 singleton columns and 1 pair column
    pair_pattern_start = 2 + n_features  # After all singletons
    for i in range(pair_pattern_start, pair_pattern_start + n_features * (n_features - 1) // 2):
        pattern_name = pattern_names[i]
        fixed_cols = np.where(np.abs(fixed_shapley[i]) > 1e-10)[0]
        
        # Should activate 3 columns (2 singletons and 1 pair)
        if len(fixed_cols) == 3:
            # Check if at least one column is beyond the singleton range
            has_pair_col = any(col >= n_features for col in fixed_cols)
            if has_pair_col:
                pair_activation_count += 1
    
    print("\nFixed Shapley implementation theoretical checks:")
    print(f"  Correctly handles singleton patterns: {single_activation_count}/{n_features}")
    print(f"  Correctly handles pair patterns: {pair_activation_count}/{(n_features * (n_features - 1) // 2)}")
    
    # Final recommendation
    print("\nRecommendation:")
    if single_activation_count == n_features and pair_activation_count == (n_features * (n_features - 1) // 2):
        print("  The fixed Shapley implementation correctly represents 2-additive coalitions.")
        print("  Consider using this implementation instead of the original Shapley.")
    else:
        print("  The fixed Shapley implementation still has issues.")
        print("  The Refined implementation with k=2 is currently the most reliable option.")

if __name__ == "__main__":
    # Analyze the current Shapley implementation
    analyze_shapley_representation()
    
    # Compare original Shapley, fixed Shapley, and Refined implementations
    compare_implementations()
