"""
Script to compare the two implementations of choquet_matrix_2add function.
"""
import numpy as np
import matplotlib.pyplot as plt
from math import comb
from itertools import chain, combinations
import time

# Helper functions
def powerset(iterable, k_add):
    '''Return the powerset (for coalitions until k_add players) of a set of m attributes'''
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(k_add+1))

def nParam_kAdd(kAdd, nAttr):
    '''Return the number of parameters in a k-additive model'''
    aux_numb = 1
    for ii in range(kAdd):
        aux_numb += comb(nAttr, ii+1)
    return aux_numb

# Original implementation
def choquet_matrix_2add_original(X_orig):
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
            data_opt[:, (coalit[:, [i, i2]]==1).all(axis=1)] = (np.min([X_orig[:, i], X_orig[:, i2]], axis=0)).reshape(nSamp, 1)
        for ii in range(nAttr+1, len(coalit)):
            if coalit[ii, i] == 1:
                data_opt[:, ii] = data_opt[:, ii] + (-1/2)*X_orig[:, i]
    return data_opt[:, 1:]

# New implementation
def choquet_matrix_2add_new(X_orig):
    X_orig = np.array(X_orig)
    nSamp, nAttr = X_orig.shape
    n_singletons = nAttr
    n_pairs = comb(nAttr, 2)
    data_opt = np.zeros((nSamp, n_singletons + n_pairs))
    data_opt[:, :n_singletons] = X_orig
    idx = n_singletons
    for i in range(nAttr):
        for j in range(i + 1, nAttr):
            data_opt[:, idx] = np.minimum(X_orig[:, i], X_orig[:, j])
            idx += 1
    return data_opt

def analyze_coalition_matrix(nAttr):
    """Analyze the coalition matrix created by the original implementation"""
    k_add = 2
    k_add_numb = nParam_kAdd(k_add, nAttr)
    coalit = np.zeros((k_add_numb, nAttr))
    
    print(f"Coalition matrix shape for {nAttr} attributes: {coalit.shape}")
    
    for i, s in enumerate(powerset(range(nAttr), k_add)):
        s = list(s)
        coalit[i, s] = 1
        
    print("Coalition matrix rows (representing subsets):")
    for i, row in enumerate(coalit):
        # Print subset represented by each row
        subset = [j for j, val in enumerate(row) if val == 1]
        print(f"Row {i}: {subset} -> {row}")
    
    return coalit

def compare_implementations(X):
    """Compare the outputs of the two implementations"""
    print(f"\nComparing implementations for input shape: {X.shape}")
    
    # Get nAttr from input shape
    nSamp, nAttr = X.shape
    
    # Time and run original implementation
    start_time = time.time()
    result_orig = choquet_matrix_2add_original(X)
    orig_time = time.time() - start_time
    
    # Time and run new implementation
    start_time = time.time()
    result_new = choquet_matrix_2add_new(X)
    new_time = time.time() - start_time
    
    print(f"Original implementation output shape: {result_orig.shape}")
    print(f"New implementation output shape: {result_new.shape}")
    print(f"Original execution time: {orig_time:.6f} seconds")
    print(f"New execution time: {new_time:.6f} seconds")
    
    # Handle zero division case
    if orig_time == 0 or new_time == 0:
        print("Execution times too small to measure accurately")
    else:
        print(f"Speedup: {orig_time/new_time:.2f}x")
    
    # Check if outputs are equivalent
    if result_orig.shape == result_new.shape:
        equal = np.allclose(result_orig, result_new)
        if equal:
            print("✓ Outputs are IDENTICAL")
        else:
            diff = np.abs(result_orig - result_new).max()
            print(f"✗ Outputs DIFFER (max abs difference: {diff:.8f})")
            
            # Analyze where differences occur
            diff_mask = ~np.isclose(result_orig, result_new)
            diff_indices = np.where(diff_mask)
            
            if len(diff_indices[0]) > 0:
                print("Sample differences (up to 5):")
                for idx in range(min(5, len(diff_indices[0]))):
                    i, j = diff_indices[0][idx], diff_indices[1][idx]
                    print(f"  Position [{i},{j}]: orig={result_orig[i,j]:.6f}, new={result_new[i,j]:.6f}")
                    
                # Analyze differences by column (feature)
                col_diff_means = np.mean(np.abs(result_orig - result_new), axis=0)
                print("\nAverage absolute difference per column:")
                for j, diff_val in enumerate(col_diff_means):
                    if j < nAttr:
                        print(f"  Column {j} (singleton): {diff_val:.6f}")
                    else:
                        # Calculate which pair this represents
                        pair_idx = j - nAttr
                        idx1, idx2 = 0, 0
                        count = 0
                        for i1 in range(nAttr):
                            for i2 in range(i1+1, nAttr):
                                if count == pair_idx:
                                    idx1, idx2 = i1, i2
                                    break
                                count += 1
                            if idx1 != 0 or idx2 != 0:
                                break
                        print(f"  Column {j} (pair {idx1},{idx2}): {diff_val:.6f}")
    else:
        print(f"✗ Shape mismatch: {result_orig.shape} vs {result_new.shape}")
    
    return result_orig, result_new

def main():
    print("Analyzing choquet_matrix_2add implementations\n")
    
    # Examine coalition matrix structure
    nAttr = 3  # Small number for clarity
    coalit = analyze_coalition_matrix(nAttr)
    
    # Test with a small 2D array
    print("\n\n=== Test with small dataset ===")
    np.random.seed(42)
    X_small = np.random.rand(4, 3)
    print("\nTest data:")
    print(X_small)
    
    result_orig, result_new = compare_implementations(X_small)
    
    # Test with a larger dataset
    print("\n\n=== Test with larger dataset ===")
    X_large = np.random.rand(1000, 5)
    result_orig_large, result_new_large = compare_implementations(X_large)
    
    # Visualize the differences
    if not np.allclose(result_orig, result_new):
        # Create visual comparison of first row values
        plt.figure(figsize=(10, 6))
        x = np.arange(result_orig.shape[1])
        
        plt.subplot(2, 1, 1)
        plt.title("Values comparison (first sample)")
        plt.plot(x, result_orig[0], 'b-o', label='Original')
        plt.plot(x, result_new[0], 'r--x', label='New')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.title("Absolute differences")
        diff = np.abs(result_orig - result_new).mean(axis=0)
        plt.bar(x, diff)
        plt.xlabel("Column index")
        plt.ylabel("Mean absolute difference")
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig("choquet_2add_comparison.png")
        print("\nComparison plot saved as 'choquet_2add_comparison.png'")

    # Analyze implementation differences in detail
    print("\n=== Implementation Differences Analysis ===")
    print("1. Original implementation:")
    print("   - Creates a coalition matrix with all subsets including empty set")
    print("   - Sets singles values directly: `data_opt[:, i+1] = data_opt[:, i+1] + X_orig[:, i]`")
    print("   - Sets pair values using vectorized boolean indexing: `data_opt[:, (coalit[:, [i, i2]]==1).all(axis=1)]`")
    print("   - Has an adjustment term: `data_opt[:, ii] = data_opt[:, ii] + (-1/2)*X_orig[:, i]`")
    print("   - Includes empty set in intermediate calculations and removes at end: `return data_opt[:, 1:]`")
    
    print("\n2. New implementation:")
    print("   - Directly calculates output size based on number of singletons and pairs")
    print("   - Sets singles values by direct assignment: `data_opt[:, :n_singletons] = X_orig`")
    print("   - Sets pair values using incremental indexing: `data_opt[:, idx] = np.minimum(X_orig[:, i], X_orig[:, j])`")
    print("   - Does not include any adjustment term")
    print("   - No empty set handling required")
    
    # Analyze the adjustment term in the original implementation
    print("\n=== Key Difference: Adjustment Term ===")
    print("The original implementation adds an adjustment term: (-1/2)*X_orig[:, i] for certain columns")
    print("This adjustment term appears to modify the pairwise minimum values in a systematic way")
    print("This is the primary reason the outputs differ between implementations")
    
    # If we have a pair mapping from the original model, show exactly which columns are adjusted
    nAttr = min(5, X_small.shape[1])  # Use a reasonable size for demonstration
    k_add = 2
    k_add_numb = nParam_kAdd(k_add, nAttr)
    coalit = np.zeros((k_add_numb, nAttr))
    for i, s in enumerate(powerset(range(nAttr), k_add)):
        s = list(s)
        coalit[i, s] = 1
        
    print("\nAdjustment analysis for a model with", nAttr, "attributes:")
    adjustment_applied = []
    for i in range(nAttr):
        for ii in range(nAttr+1, len(coalit)):
            if coalit[ii, i] == 1:
                subset = [j for j, val in enumerate(coalit[ii]) if val == 1]
                if len(subset) > 1:  # Only show pairs or larger
                    adjustment_applied.append((i, subset))
                    
    for i, subset in adjustment_applied:
        print(f"  Feature {i} contributes adjustment of -1/2*x_{i} to coalition {subset}")
    
    # Final recommendation
    print("\n=== Recommendation ===")
    if np.allclose(result_orig, result_new):
        print("Both implementations produce identical results, but the new implementation is cleaner and more efficient.")
    else:
        print("The two implementations produce different results due to the adjustment term in the original implementation.")
        print("The choice between implementations depends on whether this adjustment is mathematically necessary for your use case.")
        print("The new implementation is simpler and follows standard min(x_i, x_j) calculation without adjustments.")

if __name__ == "__main__":
    main()
