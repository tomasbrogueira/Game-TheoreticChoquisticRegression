import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
from choquet_function import powerset, nParam_kAdd

from math import comb

def correct_choquet_k_additive(X_orig, k_add=2):
    """
    Correct implementation of k-additive Choquet integral transformation.
    
    This implements the proper Choquet integral with k-additive constraint:
    C_μ(x) = Σ (x_σ(i) - x_σ(i-1)) * μ({σ(i), ..., σ(n)})
    where only coalitions up to size k are considered.
    
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
    
    # Generate all valid coalitions up to size k_add
    all_coalitions = []
    for r in range(1, k_add+1):
        all_coalitions.extend(list(combinations(range(nAttr), r)))
    
    # Calculate number of features in the transformed space
    n_transformed = len(all_coalitions)
    
    # Initialize output matrix
    transformed = np.zeros((nSamp, n_transformed))
    
    # For each sample
    for i in range(nSamp):
        x = X_orig[i]
        
        # Sort feature indices by their values
        sorted_indices = np.argsort(x)
        sorted_values = x[sorted_indices]
        
        # Add a sentinel value for the first difference
        sorted_values_ext = np.concatenate([[0], sorted_values])
        
        # For each position in the sorted feature list
        for j in range(nAttr):
            # Current feature index and value
            feat_idx = sorted_indices[j]
            # Difference with previous value
            diff = sorted_values_ext[j+1] - sorted_values_ext[j]
            
            # All features from this position onward
            higher_features = sorted_indices[j:]
            
            # Find all valid coalitions containing this feature and higher features
            for coal_idx, coalition in enumerate(all_coalitions):
                # Check if coalition is valid: contains current feature and only higher features
                if feat_idx in coalition and all(f in higher_features for f in coalition):
                    transformed[i, coal_idx] += diff
    
    return transformed

def test_corrected_implementation():
    """Test the corrected implementation"""
    print("=== Testing Corrected k-additive Choquet Implementation ===")
    
    # Test different coalition sizes
    for k_add in range(1, 5):
        print(f"\nTesting k_add = {k_add}")
        
        # Create test data with 4 features
        n_features = 4
        
        # Test with special case of increasing values
        X_test = np.array([[0.2, 0.4, 0.6, 0.8]])
        
        # Apply corrected transformation
        result = correct_choquet_k_additive(X_test, k_add)
        
        # Count active coalitions
        active_coalition_count = np.count_nonzero(result)
        expected_coalition_count = sum(comb(n_features, r) for r in range(1, k_add+1))
        
        print(f"Matrix dimensions: {result.shape}")
        print(f"Non-zero elements: {active_coalition_count} of {result.shape[1]}")
        print(f"Expected coalitions: {expected_coalition_count}")
        
        # Check coalition sizes by using prime-number detection
        prime_test = np.zeros((1, n_features))
        primes = [2, 3, 5, 7]
        for i in range(n_features):
            prime_test[0, i] = primes[i]
        
        prime_result = correct_choquet_k_additive(prime_test, k_add)
        
        # Analyze coalition sizes
        coalition_sizes = []
        for val in prime_result[0]:
            if abs(val) > 1e-10:
                coalition = []
                # Process positive and negative values differently
                if val > 0:
                    # Look for factors in the prime value
                    test_val = val
                    for i, p in enumerate(primes):
                        if abs(test_val % p) < 1e-10:
                            coalition.append(i)
                else:
                    # For negative values, look at which interaction terms might be present
                    for i, p in enumerate(primes):
                        coalition.append(i)
                
                coalition_sizes.append(len(coalition))
        
        max_coalition_size = max(coalition_sizes) if coalition_sizes else 0
        print(f"Detected coalition sizes: {sorted(set(coalition_sizes))}")
        print(f"Maximum coalition size: {max_coalition_size}")
        
        # Compare with original implementation to show differences
        from choquet_function import choquet_matrix_kadd_guilherme
        orig_result = choquet_matrix_kadd_guilherme(X_test, kadd=k_add)
        
        print("\nComparison with original implementation:")
        print(f"Original shape: {orig_result.shape}, Corrected shape: {result.shape}")
        print(f"Original non-zeros: {np.count_nonzero(orig_result)}, Corrected non-zeros: {np.count_nonzero(result)}")
        
        # Show the actual values for comparison
        print("\nCorrected implementation output:")
        for j, val in enumerate(result[0]):
            if abs(val) > 1e-10:
                print(f"  Feature {j}: {val:.6f}")

if __name__ == "__main__":
    test_corrected_implementation()