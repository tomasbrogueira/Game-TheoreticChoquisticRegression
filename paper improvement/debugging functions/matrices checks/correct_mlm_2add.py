"""
This file implements the correct 2-additive multilinear model (MLM) according to 
equation (17) in the paper:

f_ML(μ, x_i) = ∑_j x_i,j(φ_j^B - (1/2)∑_{j'≠j} I_{j,j'}^B) + ∑_{j≠j'} x_i,j x_i,j' I_{j,j'}^B

The transformation matrix should produce features that directly correspond to this equation.
"""

import numpy as np
from math import comb
from itertools import chain, combinations

def mlm_matrix_2add_correct(X_orig):
    """
    Correct implementation of the 2-additive multilinear model transformation.
    
    This implementation provides the transformation matrix for the 2-additive MLM 
    according to equation (17) in the paper. It returns:
    
    - Original features (singletons): x_i,j
    - Pairwise product features: x_i,j * x_i,j'
    
    Note: The adjustment terms in equation (17) are applied through the model 
    coefficients, not in the feature transformation matrix.
    
    Parameters:
    -----------
    X_orig : array-like
        Original feature matrix (should be scaled to [0,1])
        
    Returns:
    --------
    numpy.ndarray : 2-additive MLM basis transformation matrix
    """
    X_orig = np.array(X_orig)
    nSamp, nAttr = X_orig.shape
    n_singletons = nAttr
    n_pairs = comb(nAttr, 2)
    data_opt = np.zeros((nSamp, n_singletons + n_pairs))
    
    # Use original features for singletons
    data_opt[:, :n_singletons] = X_orig
    
    # Calculate pairwise terms as simple products
    idx = n_singletons
    for i in range(nAttr):
        for j in range(i + 1, nAttr):
            data_opt[:, idx] = X_orig[:, i] * X_orig[:, j]
            idx += 1
    
    return data_opt

def compare_mlm_implementations(X):
    """
    Compare the original and new MLM 2-additive implementations.
    
    Parameters:
    -----------
    X : array-like
        Input data to transform
        
    Returns:
    --------
    tuple : Results for each implementation
    """
    from itertools import chain, combinations
    import time
    
    # Helper functions for original implementation
    def powerset(iterable, k_add):
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(k_add+1))
    
    def nParam_kAdd(kAdd, nAttr):
        aux_numb = 1
        for ii in range(kAdd):
            aux_numb += comb(nAttr, ii+1)
        return aux_numb
    
    # Original implementation from main_choquistic_new2.py
    def mlm_matrix_2add_original(X_orig):
        X_orig = np.array(X_orig)
        nSamp, nAttr = X_orig.shape
        k_add = 2
        k_add_numb = nParam_kAdd(k_add, nAttr)
        coalit = np.zeros((k_add_numb, nAttr))
        for i, s in enumerate(powerset(range(nAttr), k_add)):
            s = list(s)
            coalit[i, s] = 1
        data_opt = np.zeros((nSamp, k_add_numb))
        for i in range(nAttr):
            data_opt[:, i+1] = data_opt[:, i+1] + X_orig[:, i]
            for i2 in range(i+1, nAttr):
                data_opt[:, (coalit[:, [i, i2]]==1).all(axis=1)] = (X_orig[:, i]* X_orig[:, i2]).reshape(nSamp, 1)
            for ii in range(nAttr+1, len(coalit)):
                if coalit[ii, i] == 1:
                    data_opt[:, ii] = data_opt[:, ii] + (-1/2)*X_orig[:, i]
        return data_opt[:, 1:]

    # Time and run each implementation
    print(f"Input shape: {X.shape}")
    
    start = time.time()
    result_orig = mlm_matrix_2add_original(X)
    time_orig = time.time() - start
    
    start = time.time()
    result_correct = mlm_matrix_2add_correct(X)
    time_correct = time.time() - start
    
    # Check equivalence and print results
    print(f"Original implementation: {result_orig.shape}, time: {time_orig:.6f}s")
    print(f"Correct implementation: {result_correct.shape}, time: {time_correct:.6f}s")
    
    if result_orig.shape == result_correct.shape:
        # Check only relevant columns (skip any adjustment columns)
        n_attr = X.shape[1]
        n_pairs = comb(n_attr, 2)
        equal = np.allclose(result_correct, result_orig[:, :n_attr+n_pairs], rtol=1e-5)
        if equal:
            print("✓ Both implementations produce equivalent results for the core features")
        else:
            print("✗ Implementations produce different results")
            diff = np.abs(result_correct - result_orig[:, :n_attr+n_pairs]).max()
            print(f"  Maximum absolute difference: {diff:.8f}")
            
            # Show first row differences
            print("\nFirst row comparison:")
            for j in range(min(result_correct.shape[1], result_orig.shape[1])):
                if j < n_attr:
                    feature_type = f"Singleton {j}"
                else:
                    idx = j - n_attr
                    pair_idx1, pair_idx2 = -1, -1
                    count = 0
                    for i1 in range(n_attr):
                        for i2 in range(i1+1, n_attr):
                            if count == idx:
                                pair_idx1, pair_idx2 = i1, i2
                                break
                            count += 1
                        if pair_idx1 >= 0:
                            break
                    feature_type = f"Pair ({pair_idx1},{pair_idx2})"
                    
                print(f"{feature_type:>15}: orig={result_orig[0,j]:.6f}, correct={result_correct[0,j]:.6f}, " + 
                      f"diff={result_orig[0,j]-result_correct[0,j]:.6f}")
    else:
        print("✗ Shape mismatch between implementations")
        print(f"  Original shape: {result_orig.shape}")
        print(f"  Correct shape: {result_correct.shape}")
        
    return result_orig, result_correct

if __name__ == "__main__":
    # Test with a small random dataset
    np.random.seed(42)
    X_test = np.random.rand(5, 3)
    print("Test data:")
    print(X_test)
    print("\nComparing implementations:")
    result_orig, result_correct = compare_mlm_implementations(X_test)
    
    print("\nConclusion:")
    print("The correct MLM 2-additive implementation follows equation (17) from the paper:")
    print("f_ML(μ, x_i) = ∑_j x_i,j(φ_j^B - (1/2)∑_{j'≠j} I_{j,j'}^B) + ∑_{j≠j'} x_i,j x_i,j' I_{j,j'}^B")
    print("\nThis implementation correctly provides:")
    print("1. The original features (singletons): x_i,j")
    print("2. The pairwise products (interactions): x_i,j * x_i,j'")
    print("\nThe adjustment terms should be handled by the model coefficients, not")
    print("by modifying the feature transformation itself.")
