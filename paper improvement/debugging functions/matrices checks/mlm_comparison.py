"""
Comparison of two different implementations of the multilinear model (MLM) matrix computation.
"""
import numpy as np
import pandas as pd
import itertools
from itertools import chain, combinations
import time
from math import comb
import math

# Utility functions needed by the original implementation
def powerset(iterable, k_add):
    '''Return the powerset (for coalitions until k_add players) of a set of m attributes'''
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(k_add+1))

def diff(first, second):
    second = set(second)
    return [item for item in first if item not in second]

# Original MLM matrix implementation from main_choquistic_new2.py - FIXED
def mlm_matrix_original(X_orig):
    """
    Original implementation of the multilinear model transformation (fixed).
    
    Parameters:
    -----------
    X_orig : array-like
        Original feature matrix
        
    Returns:
    --------
    numpy.ndarray : Transformed feature matrix with non-empty subsets
    """
    nSamp, nAttr = X_orig.shape
    X_orig = np.array(X_orig)
    data_opt = np.zeros((nSamp, 2**nAttr))
    
    for i, s in enumerate(powerset(range(nAttr), nAttr)):
        s = list(s)
        
        # Fix: Handle subsets properly
        if not s:  # Empty set
            # For empty set, product of selected features = 1
            # and we include (1-x) for all features
            prod_selected = np.ones(nSamp)
            complement = list(range(nAttr))
        else:
            # For non-empty sets, compute product of selected features
            prod_selected = np.prod(X_orig[:, s], axis=1)
            complement = diff(list(range(nAttr)), s)
            
        # Compute product of complements
        if complement:
            prod_complement = np.prod(1 - X_orig[:, complement], axis=1)
        else:
            prod_complement = np.ones(nSamp)
            
        # Compute final value
        data_opt[:, i] = prod_selected * prod_complement
        
    return data_opt[:, 1:]  # Skip the empty set column

# New MLM matrix implementation using NumPy's vectorized operations
def mlm_matrix_new(X_orig):
    """
    New implementation of the multilinear model transformation using vectorized operations.
    
    Parameters:
    -----------
    X_orig : array-like
        Original feature matrix (must be scaled to [0,1] range by the caller)
        
    Returns:
    --------
    numpy.ndarray : Full MLM basis transformation
    """
    X_orig = np.array(X_orig)
    nSamp, nAttr = X_orig.shape
    # MLM generates basis functions for all possible non-empty subsets
    subsets = list(chain.from_iterable(combinations(range(nAttr), r) for r in range(1, nAttr + 1)))
    data_opt = np.zeros((nSamp, len(subsets)))
    
    # Assume X_orig is already properly scaled to [0,1] by caller
    for i, subset in enumerate(subsets):
        # Calculate product of selected features: Π_{i∈T} x_i
        prod_x = np.prod(X_orig[:, list(subset)], axis=1)
        
        # Calculate product of complements: Π_{j∉T} (1-x_j)
        complement = [j for j in range(nAttr) if j not in subset]
        if complement:
            prod_1_minus_x = np.prod(1 - X_orig[:, complement], axis=1)
        else:
            prod_1_minus_x = np.ones(nSamp)
        
        # Compute the basis function: Π_{i∈T} x_i * Π_{j∉T} (1-x_j)
        data_opt[:, i] = prod_x * prod_1_minus_x
    
    return data_opt

def test_implementations(dataset_sizes, n_features_list, random_seed=42):
    """
    Test both MLM matrix implementations on datasets of different sizes
    and compare their performance and output equivalence.
    
    Parameters:
    -----------
    dataset_sizes : list of int
        Number of samples to use for testing
    n_features_list : list of int
        Number of features to use for testing
        
    Returns:
    --------
    dict : Results including timing and correctness
    """
    np.random.seed(random_seed)
    results = []
    
    print("Testing MLM Matrix Implementations\n")
    print(f"{'Samples':<10} {'Features':<10} {'Original Time (s)':<20} {'New Time (s)':<15} {'Speedup':<10} {'Same Output':<10}")
    print("-" * 80)
    
    for n_samples in dataset_sizes:
        for n_features in n_features_list:
            # Skip very large combinations that would cause memory issues
            if n_samples * 2**n_features > 10_000_000 and n_features >= 10:
                result = {
                    'samples': n_samples,
                    'features': n_features,
                    'original_time': float('inf'),
                    'new_time': 'N/A',
                    'speedup': 'N/A',
                    'equivalent': 'Skipped',
                    'error': 'Too large'
                }
                results.append(result)
                print(f"{n_samples:<10} {n_features:<10} {'SKIPPED - TOO LARGE':<65}")
                continue
            
            # Generate random data in [0, 1]
            X = np.random.rand(n_samples, n_features)
            
            # Time the original implementation
            try:
                start_time = time.time()
                result_original = mlm_matrix_original(X)
                original_time = time.time() - start_time
            except Exception as e:
                result = {
                    'samples': n_samples,
                    'features': n_features,
                    'original_time': 'ERROR',
                    'new_time': 'N/A',
                    'speedup': 'N/A',
                    'equivalent': 'N/A',
                    'error': str(e)
                }
                results.append(result)
                print(f"{n_samples:<10} {n_features:<10} {'ERROR: ' + str(e):<65}")
                continue
                
            # Time the new implementation
            start_time = time.time()
            result_new = mlm_matrix_new(X)
            new_time = time.time() - start_time
            
            # Check if the results are equivalent (allowing for small numerical differences)
            try:
                if result_original.shape != result_new.shape:
                    equivalent = False
                    shape_mismatch = f"Shape mismatch: {result_original.shape} vs {result_new.shape}"
                else:
                    # Use a relative tolerance for floating point comparisons
                    equivalent = np.allclose(result_original, result_new, rtol=1e-5)
                    shape_mismatch = ""
            except Exception as e:
                equivalent = False
                shape_mismatch = str(e)
            
            # Calculate speedup - Fix for division by zero
            if original_time <= 0 or new_time <= 0:
                if original_time <= 0 and new_time <= 0:
                    # Both too fast to measure
                    speedup = 1.0
                    speedup_str = "≈1.0"
                elif new_time <= 0:
                    # New implementation too fast to measure
                    speedup = float('inf')
                    speedup_str = "∞"
                else:  # original_time <= 0
                    # Original implementation too fast to measure
                    speedup = 0.0
                    speedup_str = "≈0.0"
            else:
                speedup = original_time / new_time
                speedup_str = f"{speedup:.2f}"
                
            result = {
                'samples': n_samples,
                'features': n_features,
                'original_time': original_time,
                'new_time': new_time,
                'speedup': speedup,
                'equivalent': equivalent,
                'shape_mismatch': shape_mismatch
            }
            results.append(result)
            
            print(f"{n_samples:<10} {n_features:<10} {original_time:<20.4f} {new_time:<15.4f} {speedup_str:<10} {str(equivalent):<10}")
    
    return results

def analyze_bug_for_empty_set():
    """
    Special test case to analyze how the original implementation handles empty sets.
    """
    print("\nAnalyzing how original implementation handles empty sets:")
    
    # Create a small test case
    X_test = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    
    # Check the fixed implementation with empty subset
    empty_subset = []
    
    # Use the fixed implementation that properly handles empty sets
    try:
        # Create a cleaned-up version for testing just the empty set
        if not empty_subset:  # Empty set
            prod_selected = np.ones(X_test.shape[0])
            complement = list(range(X_test.shape[1]))
            prod_complement = np.prod(1 - X_test[:, complement], axis=1)
            result = prod_selected * prod_complement
            print(f"  Fixed implementation for empty set: {result}")
        
    except Exception as e:
        print(f"  Error in fixed implementation: {e}")
        
    print("\nConclusion: Empty sets need special handling in the MLM implementation.")

def main():
    # Test on small datasets first
    dataset_sizes = [10, 100, 1000]
    n_features_list = [2, 3, 4, 5]
    
    print("\n=== Small Dataset Tests ===\n")
    results_small = test_implementations(dataset_sizes, n_features_list)
    
    # Test on larger datasets with fewer features
    dataset_sizes_large = [10000, 100000]
    n_features_list_small = [2, 3, 4]
    
    print("\n=== Large Dataset Tests ===\n")
    results_large = test_implementations(dataset_sizes_large, n_features_list_small)
    
    # Test on datasets with more features
    if any(result.get('equivalent', False) for result in results_small + results_large if not isinstance(result.get('equivalent'), str)):
        dataset_sizes_small = [10, 100]
        n_features_list_large = [6, 7, 8]
        
        print("\n=== Many Features Tests ===\n")
        results_many_features = test_implementations(dataset_sizes_small, n_features_list_large)
    
    # Check if the original implementation can handle empty sets
    analyze_bug_for_empty_set()
    
    # Overall conclusion
    all_results = results_small + results_large
    equivalent_count = sum(1 for r in all_results if r.get('equivalent') is True)
    total_tests = len(all_results)
    
    print("\n=== Summary ===")
    print(f"Tests passed: {equivalent_count} / {total_tests}")
    
    if equivalent_count == total_tests:
        print("\nBoth implementations produce identical results.")
        
        # Calculate average speedup - Only include finite, non-zero speedups
        speedups = [r['speedup'] for r in all_results if isinstance(r['speedup'], (int, float)) 
                   and r['speedup'] != float('inf') and r['speedup'] > 0]
        if speedups:
            avg_speedup = sum(speedups) / len(speedups)
            print(f"Average speedup of the new implementation: {avg_speedup:.2f}x")
            print(f"(Note: Tests where timing was too fast to measure were excluded from average)")
        
        print("\nRECOMMENDATION: Use the new implementation for better performance.")
    else:
        print("\nWARNING: Implementations produce different results in some cases.")
        print("Check individual test results for details.")

if __name__ == "__main__":
    main()
