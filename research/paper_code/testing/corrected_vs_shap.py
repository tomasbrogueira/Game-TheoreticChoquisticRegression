import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
from choquet_function import choquet_matrix_2add, nParam_kAdd
from math import comb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

def refined_choquet_k_additive(X_orig, k_add=2):
    """
    Refined implementation of k-additive Choquet integral transformation.
    
    Properly implements the k-additive Choquet integral with correct coalition handling.
    
    Parameters:
    -----------
    X_orig : array-like of shape (n_samples, n_features)
        Original feature matrix
    k_add : int
        Maximum coalition size to consider
        
    Returns:
    --------
    transformed_matrix : array-like
        Transformed feature matrix for use in linear models
    """
    X_orig = np.asarray(X_orig)
    nSamp, nAttr = X_orig.shape
    
    # Generate all coalitions up to size k_add (excluding empty set)
    coalitions = []
    for r in range(1, k_add+1):
        coalitions.extend(list(combinations(range(nAttr), r)))
    
    # Initialize output matrix
    n_coalitions = len(coalitions)
    transformed = np.zeros((nSamp, n_coalitions))
    
    # For each sample
    for i in range(nSamp):
        x = X_orig[i]
        
        # Sort feature indices by their values (ascending)
        sorted_indices = np.argsort(x)
        sorted_values = x[sorted_indices]
        
        # Add a zero value at the start for calculating differences
        sorted_values = np.concatenate([[0], sorted_values])
        
        # Process each permutation position
        for j in range(nAttr):
            # Difference between adjacent sorted values
            diff = sorted_values[j+1] - sorted_values[j]
            if abs(diff) < 1e-10:  # Skip if difference is negligible
                continue
                
            # Current feature and all features with higher or equal values
            current_feat = sorted_indices[j]
            remaining_feats = set(sorted_indices[j:])
            
            # Find all coalitions that contain the current feature and are subsets of remaining features
            for coal_idx, coalition in enumerate(coalitions):
                if (current_feat in coalition) and all(f in remaining_feats for f in coalition):
                    # Properly weight each coalition by the value difference
                    transformed[i, coal_idx] += diff
    
    return transformed, coalitions

def validate_coalition_structure(X, method, k_add=2):
    """Test if coalitions are properly structured"""
    n_features = X.shape[1]
    
    # Create test data with special prime values
    test_matrix = np.zeros((1, n_features))
    primes = [2, 3, 5, 7, 11, 13, 17, 19][:n_features]
    for i in range(n_features):
        test_matrix[0, i] = primes[i]
    
    # Apply the transformation
    if method == "refined":
        result, coalitions = refined_choquet_k_additive(test_matrix, k_add)
        # Print the actual coalitions being used
        print("\nRefined implementation coalitions:")
        for i, coal in enumerate(coalitions):
            print(f"  {i}: {coal}")
    elif method == "shapley":
        result = choquet_matrix_2add(test_matrix)
    else:
        raise ValueError("Unknown method")
    
    # Analyze which features contribute to each output value
    print(f"\n{method.capitalize()} implementation output:")
    non_zero = np.where(np.abs(result[0]) > 1e-10)[0]
    for idx in non_zero:
        val = result[0, idx]
        print(f"  Feature {idx}: {val:.6f}")
    
    return result

def compare_implementations():
    """Compare refined game domain and Shapley domain implementations"""
    print("=== Comparing Refined Game Domain vs Shapley Domain Implementations ===")
    
    # Create test datasets with varying complexity
    test_cases = [
        ("All ones", np.ones((1, 4))),
        ("Identity", np.eye(4)),
        ("Increasing values", np.array([[0.2, 0.4, 0.6, 0.8]])),
        ("Random values", np.random.rand(1, 4))
    ]
    
    for name, X in test_cases:
        print(f"\n\n--- Test Case: {name} ---")
        print("Input data:")
        print(X)
        
        # Apply both transformations
        refined_result, coalitions = refined_choquet_k_additive(X, k_add=2)
        shapley_result = choquet_matrix_2add(X)
        
        print(f"\nRefined Game Domain output (shape: {refined_result.shape}):")
        for i, val in enumerate(refined_result[0]):
            if abs(val) > 1e-10:
                coalition_str = str(coalitions[i]) if i < len(coalitions) else "unknown"
                print(f"  Feature {i} {coalition_str}: {val:.6f}")
        
        print(f"\nShapley Domain output (shape: {shapley_result.shape}):")
        for i, val in enumerate(shapley_result[0]):
            if abs(val) > 1e-10:
                print(f"  Feature {i}: {val:.6f}")
        
        # Compare sparsity
        refined_nonzero = np.count_nonzero(refined_result)
        shapley_nonzero = np.count_nonzero(shapley_result)
        
        print(f"\nSparsity comparison:")
        print(f"  Refined Game Domain: {refined_nonzero}/{refined_result.size} non-zero elements")
        print(f"  Shapley Domain: {shapley_nonzero}/{shapley_result.size} non-zero elements")
    
    # Validate the coalition structure of each implementation
    print("\n\n=== Validating Coalition Structures ===")
    validate_coalition_structure(np.random.rand(1, 4), "refined", k_add=2)
    validate_coalition_structure(np.random.rand(1, 4), "shapley", k_add=2)
    
def predictive_performance_comparison():
    """Compare predictive performance of both implementations"""
    print("\n\n=== Predictive Performance Comparison ===")
    
    # Generate a simple binary classification dataset
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=500, n_features=5, n_informative=3, 
                             n_redundant=1, random_state=42)
    
    # Scale features to [0,1]
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Apply transformations
    refined_train, _ = refined_choquet_k_additive(X_train, k_add=2)
    refined_test, _ = refined_choquet_k_additive(X_test, k_add=2)
    
    shapley_train = choquet_matrix_2add(X_train)
    shapley_test = choquet_matrix_2add(X_test)
    
    # Train classifiers
    refined_model = LogisticRegression(max_iter=1000)
    refined_model.fit(refined_train, y_train)
    
    shapley_model = LogisticRegression(max_iter=1000)
    shapley_model.fit(shapley_train, y_train)
    
    # Evaluate
    refined_preds = refined_model.predict(refined_test)
    shapley_preds = shapley_model.predict(shapley_test)
    
    refined_proba = refined_model.predict_proba(refined_test)[:, 1]
    shapley_proba = shapley_model.predict_proba(shapley_test)[:, 1]
    
    # Calculate metrics
    refined_acc = accuracy_score(y_test, refined_preds)
    shapley_acc = accuracy_score(y_test, shapley_preds)
    
    refined_auc = roc_auc_score(y_test, refined_proba)
    shapley_auc = roc_auc_score(y_test, shapley_proba)
    
    print("\nPerformance Metrics:")
    print(f"  Refined Game Domain: Accuracy={refined_acc:.4f}, AUC={refined_auc:.4f}")
    print(f"  Shapley Domain: Accuracy={shapley_acc:.4f}, AUC={shapley_auc:.4f}")
    
    # Model agreement
    agreement = np.mean(refined_preds == shapley_preds)
    correlation = np.corrcoef(refined_proba, shapley_proba)[0, 1]
    
    print("\nModel Agreement:")
    print(f"  Prediction agreement: {agreement:.4f}")
    print(f"  Probability correlation: {correlation:.4f}")
    
    # Visualize probability comparison
    plt.figure(figsize=(10, 6))
    plt.scatter(refined_proba, shapley_proba, alpha=0.5)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('Refined Game Domain Probabilities')
    plt.ylabel('Shapley Domain Probabilities')
    plt.title('Comparison of Model Probabilities')
    plt.grid(True)
    plt.savefig('refined_vs_shapley.png')
    print("\nPlot saved as 'refined_vs_shapley.png'")

if __name__ == "__main__":
    compare_implementations()
    predictive_performance_comparison()