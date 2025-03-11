import numpy as np
import itertools
from choquet_kadd_test import (
    choquet_matrix_unified, 
    choquet_matrix_2add
)

def theoretical_transformation_matrix(n_features):
    """
    Generate the theoretical transformation matrix for simple cases.
    
    This implements the identified pattern for singleton features:
    - T[i,i] = 1.0  (diagonal elements mapping feature to its Shapley value)
    - T[i,n+k] = -0.5 (mapping to interaction terms involving the feature)
    
    Parameters:
    -----------
    n_features : int
        Number of features
        
    Returns:
    --------
    numpy.ndarray : Theoretical transformation matrix
    """
    # Calculate size based on number of features
    n_pairs = n_features * (n_features - 1) // 2
    size = n_features + n_pairs
    
    # Create empty matrix
    T = np.zeros((size, size))
    
    # Fill diagonal for singleton features
    for i in range(n_features):
        T[i, i] = 1.0
    
    # Fill interaction terms for singleton features
    pair_idx = 0
    for i in range(n_features):
        for j in range(i+1, n_features):
            interaction_idx = n_features + pair_idx
            # Feature i contributes -0.5 to interaction (i,j)
            T[i, interaction_idx] = -0.5
            # Feature j contributes -0.5 to interaction (i,j)
            T[j, interaction_idx] = -0.5
            pair_idx += 1
    
    return T

def analyze_ordering_based_formula(X):
    """
    Analyze how feature orderings affect the transformation matrix.
    
    Parameters:
    -----------
    X : array-like
        Input data matrix
    """
    n_samples, n_features = X.shape
    
    # Get actual transformation matrix
    eq22_matrix, coalitions = choquet_matrix_unified(X, k_add=2)
    eq23_matrix = choquet_matrix_2add(X)
    actual_T = np.linalg.lstsq(eq22_matrix, eq23_matrix, rcond=None)[0]
    
    # Get theoretical matrix for comparison
    theory_T = theoretical_transformation_matrix(n_features)
    
    print("Actual transformation matrix:")
    print(actual_T)
    print("\nTheoretical transformation matrix:")
    print(theory_T)
    
    # Analyze feature orderings
    print("\nAnalyzing feature orderings:")
    ordering_counts = {}
    
    for i in range(n_samples):
        order = tuple(np.argsort(X[i]))
        ordering_counts[order] = ordering_counts.get(order, 0) + 1
    
    print("\nFeature ordering frequencies:")
    for order, count in ordering_counts.items():
        print(f"Order {order}: {count}/{n_samples} samples ({count/n_samples:.2%})")
    
    # Analyze differences between actual and theoretical matrices
    diff = actual_T - theory_T
    
    print("\nDifference between actual and theoretical matrices:")
    print(diff)
    
    # Try to derive formula based on ordering frequencies
    print("\nAttempting to derive ordering-based correction formula...")
    
    # Calculate weights based on ordering frequencies
    weights = {}
    for order, count in ordering_counts.items():
        weights[order] = count / n_samples
    
    # Create ordering-based adjustment matrix
    adj_matrix = np.zeros_like(theory_T)
    
    # For each ordering, calculate its contribution
    for order, weight in weights.items():
        # For pairs of features, adjust based on their relative ordering
        for i in range(n_features):
            pos_i = order.index(i)
            for j in range(i+1, n_features):
                pos_j = order.index(j)
                # Calculate pair index correctly
                pair_idx = i * (2*n_features - i - 1) // 2 + (j - i - 1)
                
                # If i comes before j in the ordering
                if pos_i < pos_j:
                    adj_matrix[i, j] += weight  # i contributes to j's Shapley
                    adj_matrix[n_features+pair_idx, j] += weight  # Pair contributes to j's Shapley
                # If j comes before i in the ordering
                else:
                    adj_matrix[j, i] += weight  # j contributes to i's Shapley
                    adj_matrix[n_features+pair_idx, i] += weight  # Pair contributes to i's Shapley
    
    print("Ordering-based adjustment matrix:")
    print(adj_matrix)
    
    # Try creating an improved theoretical matrix
    improved_T = theory_T.copy()
    
    # Apply data-dependent corrections to pair rows
    n_pairs = n_features * (n_features - 1) // 2
    pair_idx = 0
    for i in range(n_features):
        for j in range(i+1, n_features):
            # For the row representing pair (i,j)
            row_idx = n_features + pair_idx
            
            # Calculate average value of feature i vs j
            i_gt_j = np.mean(X[:, i] > X[:, j])
            j_gt_i = 1 - i_gt_j
            
            # Adjust pair's contribution to Shapley values
            # When i > j more often, pair contributes more to i's Shapley
            improved_T[row_idx, i] = i_gt_j
            improved_T[row_idx, j] = j_gt_i
            
            pair_idx += 1
    
    print("\nImproved theoretical matrix with ordering-based corrections:")
    print(improved_T)
    
    # Calculate errors
    basic_error = np.linalg.norm(actual_T - theory_T) / np.linalg.norm(actual_T)
    improved_error = np.linalg.norm(actual_T - improved_T) / np.linalg.norm(actual_T)
    
    print(f"\nError with basic theoretical matrix: {basic_error:.6f}")
    print(f"Error with improved theoretical matrix: {improved_error:.6f}")
    
    return actual_T, theory_T, improved_T, ordering_counts

def propose_final_formula():
    """
    Based on all analyses, propose a final formula for the transformation matrix.
    """
    print("\n==== PROPOSED FINAL FORMULA ====\n")
    print("For a transformation matrix T that maps from eq(22) to eq(23):")
    
    print("\n1. For singleton features (i):")
    print("   T[i,i] = 1.0")
    print("   T[i,n+idx(i,j)] = -0.5 for each interaction involving feature i")
    
    print("\n2. For pair rows (i,j) where row_idx = n_features + pair_idx:")
    print("   T[row_idx, i] = P(X_i > X_j)")
    print("   T[row_idx, j] = P(X_j > X_i) = 1 - P(X_i > X_j)")
    print("   T[row_idx, n+pair_idx] ≈ 0")
    
    print("\nWhere:")
    print("- P(X_i > X_j) is the probability (frequency) that feature i is greater than feature j")
    print("- n is the number of features")
    print("- idx(i,j) is the index of the interaction term for features i and j")
    
    print("\nThis formula accounts for the data-dependent nature of the transformation,")
    print("particularly how the relative ordering of features affects the mapping.")
    
    print("\n==== IMPLEMENTATION ====\n")
    print("def compute_transformation_matrix(X, k_add=2):")
    print("    n_samples, n_features = X.shape")
    print("    n_pairs = n_features * (n_features - 1) // 2")
    print("    T = np.zeros((n_features + n_pairs, n_features + n_pairs))")
    print("    ")
    print("    # 1. Set singleton rows")
    print("    for i in range(n_features):")
    print("        # Diagonal element (feature to its own Shapley value)")
    print("        T[i, i] = 1.0")
    print("        ")
    print("        # Contribution to interaction terms")
    print("        pair_idx = 0")
    print("        for j in range(n_features):")
    print("            for k in range(j+1, n_features):")
    print("                if i == j or i == k:")
    print("                    T[i, n_features + pair_idx] = -0.5")
    print("                pair_idx += 1")
    print("    ")
    print("    # 2. Set pair rows based on ordering frequencies")
    print("    pair_idx = 0")
    print("    for i in range(n_features):")
    print("        for j in range(i+1, n_features):")
    print("            row_idx = n_features + pair_idx")
    print("            # Calculate P(X_i > X_j) and P(X_j > X_i)")
    print("            p_i_gt_j = np.mean(X[:, i] > X[:, j])")
    print("            p_j_gt_i = 1 - p_i_gt_j")
    print("            ")
    print("            # Set contributions to Shapley values")
    print("            T[row_idx, i] = p_i_gt_j")
    print("            T[row_idx, j] = p_j_gt_i")
    print("            pair_idx += 1")
    print("    ")
    print("    return T")

def main():
    """Run a comprehensive analysis of the transformation formula."""
    print("===== ANALYZING TRANSFORMATION MATRIX FORMULA =====")
    
    # Test with various datasets
    datasets = [
        ("Unit feature values", np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])),
        ("Balanced values", np.array([[0.1, 0.5, 0.9], [0.9, 0.5, 0.1], [0.5, 0.9, 0.1]])),
        ("Random values", np.random.rand(10, 3)),
        ("Ordered values", np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]))
    ]
    
    for name, X in datasets:
        print(f"\n----- Dataset: {name} -----")
        actual, theory, improved, _ = analyze_ordering_based_formula(X)
    
    # Propose final formula
    propose_final_formula()

if __name__ == "__main__":
    main()
