# Choquet Integral Implementations: Mathematical Equivalence Documentation

This document explains the mathematical relationship between the two implementations of the Choquet integral in the codebase: the standard formulation (equation 22) and the Shapley/interaction formulation (equation 23).

## Summary of Findings

Our analysis confirms that:

1. Both implementations represent the same 2-additive model family
2. They differ only in their parameterization
3. There exists a perfect linear transformation between them
4. Both yield identical predictions when properly calibrated
5. The relationship is: `Equation (23) = Equation (22) @ Transformation_Matrix`
6. Coefficient relationship: `coeffs_22 = transformation_matrix @ coeffs_23`

## Data Preprocessing Requirements

The Choquet integral is defined for capacities in the range [0,1], so all implementations automatically apply MinMax scaling to ensure:
- All feature values fall within the [0,1] range
- The relative ordering of feature values is preserved
- Mathematical properties of the Choquet integral are maintained

This preprocessing step happens internally within each implementation function, so users don't need to scale data beforehand.

## Mathematical Formulations

### Equation (22) - Standard Choquet Integral

```
f_CI(v, x_i) = Σ_{j=1}^{m} (x_{i,(j)} - x_{i,(j-1)}) * v({(j), ..., (m)})
```

This formulation represents the classic Choquet integral, where:
- features are ordered in ascending order
- differences between consecutive sorted values are multiplied by coalition values
- coalitions represent sets of features from the current feature to the end

### Equation (23) - Shapley/Interaction Formulation

```
f_CI(v, x_i) = Σ_j x_i,j(φ_j^S - (1/2)Σ_{j'≠j} I_{j,j'}^S) + Σ_{j≠j'} (x_i,j ∧ x_i,j') I_{j,j'}^S
```

This formulation uses:
- Shapley values (φ) for feature importance
- Interaction indices (I) for feature interactions
- Original feature values and minimum of feature pairs

## Transformation Between Representations

For a given dataset, the transformation matrix can be calculated that perfectly maps between the two representations:

```python
# Get both representations
eq22_matrix, eq22_coalitions = choquet_matrix_unified(X, k_add=2)
eq23_matrix = choquet_matrix_2add(X)

# Calculate transformation matrix
transform_matrix = np.linalg.lstsq(eq22_matrix, eq23_matrix, rcond=None)[0]

# Convert eq(22) to eq(23)
eq23_predicted = eq22_matrix @ transform_matrix

# Convert coefficients from eq(23) space to eq(22) space
coeffs_22 = transform_matrix @ coeffs_23
```

## Prediction Equivalence

When proper coefficient transformation is applied:
- `predictions_23 = eq23_matrix @ coeffs_23`
- `predictions_22 = eq22_matrix @ coeffs_22`
- Given that `coeffs_22 = transform_matrix @ coeffs_23`
- Then `predictions_22 = predictions_23` with zero error

## Benefits of Each Representation

### Equation (22) - Standard Choquet
- More directly implements the mathematical definition
- Easier to generalize to any k-additive model
- Parameter interpretation follows classic Choquet integral theory

### Equation (23) - Shapley/Interaction
- Parameters have more interpretable meaning in terms of feature importance
- Directly exposes Shapley values and interaction terms
- Closer to the interaction interpretation in the paper

## Conclusion

Both implementations are mathematically equivalent and will produce identical predictions when properly calibrated. The choice between them depends on specific needs for interpretation and generalizability.

For most uses, the unified implementation (`choquet_matrix_unified`) is recommended as it provides flexibility to handle any k-additive model while maintaining mathematical equivalence.
