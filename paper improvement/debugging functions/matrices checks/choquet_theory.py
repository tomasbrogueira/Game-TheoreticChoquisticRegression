"""
This file provides a theoretical explanation of the 2-additive Choquet integral
and analyzes which implementation is mathematically correct.
"""

# Standard mathematical definition of 2-additive Choquet integral:
"""
For a 2-additive Choquet integral, the formula is:

f_CI(μ, x_i) = ∑_j x_i,j(φ_j^S - (1/2)∑_{j'≠j} I_{j,j'}^S) + ∑_{j≠j'} (x_i,j ∧ x_i,j') I_{j,j'}^S

Where:
- φ_j^S are the Shapley values (singleton importance)
- I_{j,j'}^S are the interaction indices between features j and j'
- x_i,j ∧ x_i,j' is the minimum operation between values
- The term (1/2)∑_{j'≠j} I_{j,j'}^S represents adjustment to account for double-counting

The ORIGINAL implementation correctly implements this formula:
1. Includes original features as singletons: x_i,j
2. Computes min(x_i,j, x_i,j') for each pair
3. Includes the adjustment term: -0.5*x_i,j for each pair involving feature j

The NEW implementation is incomplete:
1. Includes original features correctly
2. Computes min(x_i,j, x_i,j') for each pair 
3. BUT misses the adjustment term: -0.5*∑_{j'≠j} I_{j,j'}^S * x_i,j
"""

import numpy as np

def demonstrate_2additive_choquet(x, shapley_values, interaction_indices):
    """
    Demonstrate a proper 2-additive Choquet integral calculation
    using both original and new implementation approaches.
    
    Parameters:
    -----------
    x : array-like
        Feature values (assumed to be 1D array)
    shapley_values : array-like
        Shapley values for each feature
    interaction_indices : array-like
        Interaction indices for each pair (flattened upper triangular)
    """
    m = len(x)
    
    # Build interaction matrix from flattened indices
    interaction_matrix = np.zeros((m, m))
    idx = 0
    for i in range(m):
        for j in range(i+1, m):
            interaction_matrix[i, j] = interaction_indices[idx]
            interaction_matrix[j, i] = interaction_indices[idx]  # Symmetric
            idx += 1
    
    # 1. Original implementation (correct full formula)
    result_orig = 0
    
    # Add singleton terms with adjustments
    for j in range(m):
        # Shapley value term
        result_orig += x[j] * shapley_values[j]
        # Adjustment term
        adjustment = 0
        for j_prime in range(m):
            if j_prime != j:
                adjustment += interaction_matrix[j, j_prime]
        result_orig -= 0.5 * x[j] * adjustment
    
    # Add interaction terms
    for j in range(m):
        for j_prime in range(j+1, m):
            result_orig += min(x[j], x[j_prime]) * interaction_matrix[j, j_prime]
    
    # 2. New implementation (incomplete formula)
    result_new = 0
    
    # Add singleton terms (but missing adjustments)
    for j in range(m):
        result_new += x[j] * shapley_values[j]
    
    # Add interaction terms
    for j in range(m):
        for j_prime in range(j+1, m):
            result_new += min(x[j], x[j_prime]) * interaction_matrix[j, j_prime]
    
    return result_orig, result_new

# Example demonstration
"""
THEORETICAL CONCLUSION:

After examining the mathematical definition in the paper, the ORIGINAL implementation 
is theoretically correct according to the standard formula for 2-additive Choquet integral.

The formula includes:
1. Singleton terms: ∑_j x_i,j(φ_j^S)
2. Adjustment terms: -∑_j x_i,j((1/2)∑_{j'≠j} I_{j,j'}^S)
3. Interaction terms: ∑_{j≠j'} (x_i,j ∧ x_i,j') I_{j,j'}^S

The original implementation correctly includes the adjustment terms that account
for the way interactions are handled in 2-additive capacity models.

The new implementation is missing these adjustment terms.

For accurate implementation of 2-additive Choquet integral models,
the original implementation should be preferred.
"""
