import numpy as np
import pandas as pd
import itertools
from math import comb
from itertools import chain, combinations
import matplotlib.pyplot as plt

def nParam_kAdd(kAdd, nAttr):
    '''Return the number of parameters in a k-additive model'''
    aux_numb = 1
    for ii in range(kAdd):
        aux_numb += comb(nAttr,ii+1)
    return aux_numb

def powerset(iterable, k_add):
    '''Return the powerset (for coalitions until k_add players) of a set of m attributes'''
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(k_add+1))

# The choquet_matrix_mobius function
def choquet_matrix_mobius(X_orig, kadd):
    """
    Create a feature matrix based on the Möbius representation for k-additive models.
    
    For each subset of features up to size k, computes the minimum value among 
    those features as a new derived feature.
    """
    nSamp, nAttr = X_orig.shape
    # Only count non-empty subsets (exclude the empty set)
    k_add_numb = nParam_kAdd(kadd, nAttr) - 1
    data_opt = np.zeros((nSamp, k_add_numb))
    
    idx = 0
    # Skip empty set by filtering for non-empty subsets
    for s in [subset for subset in powerset(range(nAttr), kadd) if subset]:
        s = list(s)
        data_opt[:, idx] = np.min(X_orig.iloc[:, s], axis=1)
        idx += 1
    
    return data_opt

# The new choquet_matrix function
def choquet_matrix(X_orig, all_coalitions=None):
    """
    Compute the full Choquet integral transformation matrix.
    
    This implements the Choquet integral formulation:
    C_μ(x) = Σ_{i=1}^n (x_σ(i) - x_σ(i-1)) * μ({σ(i), ..., σ(n)})
    where σ is a permutation that orders features in ascending order.
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
# Convert to DataFrame for choquet_matrix_mobius
X_test_df = pd.DataFrame(X_test)

print("Test dataset (5 samples, 3 features):")
print(X_test)
print("-" * 50)

# Run both implementations for full model (k=n)
kadd = 3  # Full model for our 3-feature dataset
result_mobius = choquet_matrix_mobius(X_test_df, kadd)
result_choquet, coalitions = choquet_matrix(X_test)

print("Comparison of outputs:")
print(f"choquet_matrix output shape: {result_choquet.shape}")
print(f"choquet_matrix_mobius output shape: {result_mobius.shape}")
print("-" * 50)

print("All coalitions:", coalitions)
print("-" * 50)

# Show some of the values
print("Sample values from both matrices:")
print("\nchoquet_matrix (first sample):")
for i, coal in enumerate(coalitions):
    print(f"  Coalition {coal}: {result_choquet[0, i]}")

print("\nchoquet_matrix_mobius (first sample):")
idx = 0
for s in [subset for subset in powerset(range(X_test.shape[1]), kadd) if subset]:
    print(f"  Subset {s}: {result_mobius[0, idx]}")
    idx += 1

print("-" * 50)
print("Mathematical explanation of differences:")
print("1. choquet_matrix: Computes differences between consecutively ordered feature values")
print("   (used for the Choquet integral formula directly)")
print("2. choquet_matrix_mobius: Computes minimums of feature subsets")
print("   (used for the Möbius transform representation)")
print("\nThese functions compute different values because they represent different mathematical concepts,")
print("even though both are related to Choquet integrals.")

# Visualization for comparison
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(result_choquet, aspect='auto')
plt.title("choquet_matrix\n(diff between sorted values)")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(result_mobius, aspect='auto')
plt.title("choquet_matrix_mobius\n(min of feature subsets)")
plt.colorbar()

plt.tight_layout()
plt.savefig('choquet_mobius_comparison.png')
print("\nComparison visualization saved as 'choquet_mobius_comparison.png'")
