"""
This file explains the mathematical theory behind the 2-additive Choquet integral
and 2-additive multilinear model (MLM) as described in the paper.
"""

import numpy as np

# Mathematical definitions
"""
MATHEMATICAL DEFINITIONS FROM THE PAPER:

1. 2-additive Choquet Integral (equation 23):
   f_CI(v, x_i) = ∑_j x_i,j(φ_j^S - (1/2)∑_{j'≠j} I_{j,j'}^S) + ∑_{j≠j'} (x_i,j ∧ x_i,j') I_{j,j'}^S

2. 2-additive Multilinear Model (equation 25):
   f_ML(v, x_i) = ∑_j x_i,j(φ_j^B - (1/2)∑_{j'≠j} I_{j,j'}^B) + ∑_{j≠j'} x_i,j x_i,j' I_{j,j'}^B

In both equations:
- φ_j^S (or φ_j^B) are the Shapley (or Banzhaf) values for feature j
- I_{j,j'}^S (or I_{j,j'}^B) are interaction indices between features j and j'
- x_i,j represents the value of feature j for instance i
- ∧ represents the minimum operator

IMPLEMENTATION DETAILS:

These equations can be rewritten to separate the feature transformation from coefficients:

For Choquet integral:
f_CI(v, x_i) = ∑_j [φ_j^S - (1/2)∑_{j'≠j} I_{j,j'}^S] * x_i,j + ∑_{j≠j'} I_{j,j'}^S * (x_i,j ∧ x_i,j')

For MLM:
f_ML(v, x_i) = ∑_j [φ_j^B - (1/2)∑_{j'≠j} I_{j,j'}^B] * x_i,j + ∑_{j≠j'} I_{j,j'}^B * x_i,j * x_i,j'

Our implementations follow these equations by creating feature matrices with:
1. Original features for singleton terms: x_i,j
2. Interaction features:
   - min(x_i,j, x_i,j') for Choquet
   - x_i,j * x_i,j' for MLM
3. The adjustment term -0.5*x_i,j for each feature j is included to match the equations.

This implementation choice correctly represents the mathematical models from the paper.
"""

def demonstrate_models(x, shapley_values, banzhaf_values, interaction_S, interaction_B):
    """
    Demonstrate how the 2-additive models are calculated correctly.
    
    Parameters:
    -----------
    x : array-like
        Feature values (1D array)
    shapley_values : array-like
        Shapley values for each feature
    banzhaf_values : array-like
        Banzhaf values for each feature
    interaction_S : array 2D
        Shapley interaction indices (matrix)
    interaction_B : array 2D
        Banzhaf interaction indices (matrix)
    """
    m = len(x)
    
    # 1. Choquet integral calculation (equation 23)
    choquet_result = 0
    
    # Singleton terms with adjustments
    for j in range(m):
        # Calculate adjustment term for feature j
        adjustment = 0
        for j_prime in range(m):
            if j_prime != j:
                adjustment += interaction_S[j, j_prime]
        
        # Apply singleton coefficient (Shapley value minus adjustment)
        term = shapley_values[j] - 0.5 * adjustment
        choquet_result += x[j] * term
    
    # Interaction terms
    for j in range(m):
        for j_prime in range(j+1, m):  # avoid double counting
            choquet_result += min(x[j], x[j_prime]) * interaction_S[j, j_prime]
    
    print(f"2-additive Choquet integral result: {choquet_result:.6f}")
    
    # 2. MLM calculation (equation 25)
    mlm_result = 0
    
    # Singleton terms with adjustments
    for j in range(m):
        # Calculate adjustment term for feature j
        adjustment = 0
        for j_prime in range(m):
            if j_prime != j:
                adjustment += interaction_B[j, j_prime]
        
        # Apply singleton coefficient (Banzhaf value minus adjustment)
        term = banzhaf_values[j] - 0.5 * adjustment
        mlm_result += x[j] * term
    
    # Interaction terms
    for j in range(m):
        for j_prime in range(j+1, m):  # avoid double counting
            mlm_result += x[j] * x[j_prime] * interaction_B[j, j_prime]
    
    print(f"2-additive MLM result: {mlm_result:.6f}")
    
    return choquet_result, mlm_result
