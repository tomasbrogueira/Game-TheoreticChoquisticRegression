import numpy as np
import pandas as pd
import itertools
from math import comb
from scipy.special import bernoulli
import matplotlib.pyplot as plt

# Original implementation
def choquet_matrix_original(X_orig):
    X_orig_sort = np.sort(X_orig)
    X_orig_sort_ind = np.array(np.argsort(X_orig))
    nSamp, nAttr = X_orig.shape  # Number of samples (train) and attributes
    X_orig_sort_ext = np.concatenate((np.zeros((nSamp, 1)), X_orig_sort), axis=1)
    
    sequence = np.arange(nAttr)
    
    combin = (99) * np.ones((2**nAttr - 1, nAttr))
    count = 0
    for ii in range(nAttr):
        combin[count:count + comb(nAttr, ii + 1), 0:ii + 1] = np.array(list(itertools.combinations(sequence, ii + 1)))
        count += comb(nAttr, ii + 1)
    
    data_opt = np.zeros((nSamp, 2**nAttr - 1))
    for ii in range(nAttr):
        for jj in range(nSamp):
            list1 = combin.tolist()
            aux = list1.index(np.concatenate((np.sort(X_orig_sort_ind[jj, ii:]), 99 * np.ones((ii,))), axis=0).tolist())
            data_opt[jj, aux] = X_orig_sort_ext[jj, ii + 1] - X_orig_sort_ext[jj, ii]
            
    return data_opt

# New implementation
def choquet_matrix_new(X_orig, all_coalitions=None):
    """
    Compute the full Choquet integral transformation matrix.
    
    This implements the Choquet integral formulation:
    C_μ(x) = Σ_{i=1}^n (x_σ(i) - x_σ(i-1)) * μ({σ(i), ..., σ(n)})
    where σ is a permutation that orders features in ascending order.
    
    Parameters:
    -----------
    X_orig : array-like of shape (n_samples, n_features)
        Original feature matrix
    all_coalitions : list of tuples, optional
        Pre-computed list of all nonempty coalitions
        
    Returns:
    --------
    tuple : (transformed feature matrix, list of all coalitions)
    """
    X_orig = np.array(X_orig)
    nSamp, nAttr = X_orig.shape
    if all_coalitions is None:
        all_coalitions = []
        for r in range(1, nAttr + 1):
            all_coalitions.extend(list(itertools.combinations(range(nAttr), r)))
    coalition_to_index = {coalition: idx for idx, coalition in enumerate(all_coalitions)}
    data_opt = np.zeros((nSamp, len(all_coalitions)))
    for i in range(nSamp):
        order = np.argsort(X_orig[i])
        sorted_vals = np.sort(X_orig[i])
        prev = 0.0
        for j in range(nAttr):
            coalition = tuple(sorted(order[j:]))
            idx = coalition_to_index.get(coalition)
            if idx is None:
                continue
            diff = sorted_vals[j] - prev
            prev = sorted_vals[j]
            data_opt[i, idx] = diff
    return data_opt, all_coalitions

# Test with a small dataset
np.random.seed(42)
X_test = np.random.rand(5, 3)
print("Test dataset (5 samples, 3 features):")
print(X_test)
print("-" * 50)

# Run both implementations
result_original = choquet_matrix_original(X_test)
result_new, coalitions_new = choquet_matrix_new(X_test)

print("Original implementation output shape:", result_original.shape)
print("New implementation output shape:", result_new.shape)
print("-" * 50)

# Generate all coalitions for comparison
all_coalitions = []
nAttr = X_test.shape[1]
for r in range(1, nAttr + 1):
    all_coalitions.extend(list(itertools.combinations(range(nAttr), r)))

print("All coalitions:", all_coalitions)
print("-" * 50)

# Check if the results are equivalent by comparing corresponding coalition values
print("Comparing outputs for each coalition:")
equivalent = True
for idx, coalition in enumerate(all_coalitions):
    # Find the corresponding index in the original implementation
    orig_idx = None
    combin = (99) * np.ones((2**nAttr - 1, nAttr))
    count = 0
    for ii in range(nAttr):
        combin[count:count + comb(nAttr, ii + 1), 0:ii + 1] = np.array(list(itertools.combinations(range(nAttr), ii + 1)))
        count += comb(nAttr, ii + 1)
    
    list1 = combin.tolist()
    padded_coalition = list(coalition) + [99] * (nAttr - len(coalition))
    if padded_coalition in list1:
        orig_idx = list1.index(padded_coalition)
        
        # Compare values for the first sample
        print(f"Coalition {coalition}:")
        print(f"  Original: {result_original[0, orig_idx]}")
        print(f"  New:      {result_new[0, idx]}")
        if not np.isclose(result_original[0, orig_idx], result_new[0, idx]):
            equivalent = False
            print("  ❌ Different values")
        else:
            print("  ✓ Same values")
    else:
        print(f"Coalition {coalition} not found in original implementation")
        equivalent = False

print("-" * 50)
if equivalent:
    print("✅ The implementations produce equivalent results!")
else:
    print("❌ The implementations produce different results.")

# Optional visualization for detailed comparison
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.imshow(result_original, aspect='auto')
plt.title("Original Implementation")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(result_new, aspect='auto')
plt.title("New Implementation")
plt.colorbar()

plt.tight_layout()
plt.savefig('choquet_comparison.png')
print("Comparison visualization saved as 'choquet_comparison.png'")
