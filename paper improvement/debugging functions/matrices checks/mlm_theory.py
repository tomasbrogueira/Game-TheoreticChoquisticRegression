"""
This file provides a theoretical explanation of the 2-additive multilinear model (MLM)
and analyzes which implementation is mathematically correct.
"""

# Standard mathematical definition of 2-additive multilinear model:
"""
For a 2-additive MLM, according to equation (17) in the paper, the formula is:

f_ML(μ, x_i) = ∑_j x_i,j(φ_j^B - (1/2)∑_{j'≠j} I_{j,j'}^B) + ∑_{j≠j'} x_i,j x_i,j' I_{j,j'}^B

Where:
- φ_j^B are the Banzhaf values (singleton importance)
- I_{j,j'}^B are the interaction indices between features j and j'
- x_i,j x_i,j' is the product operation between values (unlike min() in Choquet)
- The term (1/2)∑_{j'≠j} I_{j,j'}^B represents an adjustment similar to the Choquet case

However, in standard MLM implementations (outside this paper), the MLM is often 
expressed as a simpler polynomial without the adjustment terms:

f_ML(μ, x) = ∑_j w_j x_j + ∑_{j<k} w_{jk} x_j x_k + ...

The NEW implementation follows this simpler form, while the ORIGINAL implementation
appears to be using the full formula from equation (17) that includes the adjustments.
"""

import numpy as np

def demonstrate_2additive_mlm(x, banzhaf_values, interaction_indices):
    """
    Demonstrate a proper 2-additive MLM calculation using both implementation approaches.
    
    Parameters:
    -----------
    x : array-like
        Feature values (assumed to be 1D array)
    banzhaf_values : array-like
        Banzhaf values for each feature
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
    
    # 1. Full formula implementation (similar to original implementation)
    result_orig = 0
    
    # Add singleton terms with adjustments
    for j in range(m):
        # Banzhaf value term
        result_orig += x[j] * banzhaf_values[j]
        # Adjustment term
        adjustment = 0
        for j_prime in range(m):
            if j_prime != j:
                adjustment += interaction_matrix[j, j_prime]
        result_orig -= 0.5 * x[j] * adjustment
    
    # Add interaction terms
    for j in range(m):
        for j_prime in range(j+1, m):
            result_orig += x[j] * x[j_prime] * interaction_matrix[j, j_prime]
    
    # 2. Simple polynomial implementation (similar to new implementation)
    result_new = 0
    
    # Add singleton terms without adjustments
    for j in range(m):
        result_new += x[j] * banzhaf_values[j]
    
    # Add interaction terms
    for j in range(m):
        for j_prime in range(j+1, m):
            result_new += x[j] * x[j_prime] * interaction_matrix[j, j_prime]
    
    return result_orig, result_new

"""
CONCLUSION:

1. The new implementation follows the standard MLM formulation as a polynomial:
   f(x) = ∑_j w_j x_j + ∑_{j<k} w_{jk} x_j x_k

2. The original implementation attempts to apply the full equation (17) from the paper,
   but does so using the same approach as for Choquet integrals, which leads to 
   inconsistencies in the implementation.

3. When looking at the paper's equation (17), the adjustments are theoretical adjustments
   in the model parameters (φ_j^B and I_{j,j'}^B), not computational adjustments that 
   need to be applied during feature transformation.

RECOMMENDATION:
The new implementation is more appropriate for standard MLM use cases, as it
implements the standard polynomial form of MLM without unnecessary adjustments.
"""
