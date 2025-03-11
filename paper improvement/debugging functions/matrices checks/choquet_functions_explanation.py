"""
Explanation of different Choquet integral-related functions

This file explains the key differences between choquet_matrix and choquet_matrix_mobius
with mathematical interpretations and practical use cases.
"""

"""
1. choquet_matrix
----------------
Purpose: Calculates the coefficients in the Choquet integral decomposition formula
Mathematical form: C_μ(x) = Σ_{i=1}^n (x_σ(i) - x_σ(i-1)) * μ({σ(i), ..., σ(n)})

- For each sample, features are sorted in ascending order
- The matrix stores differences between consecutive sorted values (x_σ(i) - x_σ(i-1))
- These differences will be multiplied by capacity values for each coalition
- Used for direct evaluation of the Choquet integral

2. choquet_matrix_mobius
----------------------
Purpose: Calculates the minimum values for each subset (coalition) of features
Mathematical form: Möbius representation: C_m(x) = Σ_{T⊆N} m(T) * min_{i∈T}(x_i)

- For each subset of features up to size k, compute minimum value among those features
- The matrix stores these minimum values for each coalition
- Used with Möbius transform coefficients of a capacity
- Represents an alternative way to compute the Choquet integral via the Möbius transform

Usage Context:
- choquet_matrix: 
  * When you have a capacity function and want to evaluate the Choquet integral directly
  * For full model representation with all possible feature interactions
  * When analyzing the importance of feature orderings (permutations)
  * In models where different orderings of feature values are significant
  * When the focus is on the marginal contributions of features in different contexts
  * Often used in complex decision models where all feature interactions matter

- choquet_matrix_mobius: 
  * When working with Möbius representations in k-additive models
  * For reducing model complexity by limiting interaction order to k features
  * When computational efficiency is important (fewer parameters)
  * In feature importance analysis focused on specific coalition contributions
  * When interpretability of feature interactions is the primary goal
  * Commonly used in simplified models where higher-order interactions can be ignored
  * Particularly useful for high-dimensional data where full models are intractable

Computational Complexity:
- choquet_matrix: O(2^n) complexity, where n is the number of features
- choquet_matrix_mobius with k-additivity: O(n^k) complexity, much more efficient when k << n

Example results from test case (first sample):
Coalition (0,):
  choquet_matrix: 0.0 (no difference contribution)
  choquet_mobius: 0.3745 (minimum value of feature 0)

Coalition (0,1,2):
  choquet_matrix: 0.3745 (difference contribution)
  choquet_mobius: 0.3745 (minimum of all three features)
"""

# Note: This file is documentation-only, containing no executable code
