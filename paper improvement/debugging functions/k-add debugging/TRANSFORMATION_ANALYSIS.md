# Analysis of Transformation Between Choquet Integral Representations

This document summarizes our findings about the transformation matrix between the standard Choquet integral representation (Equation 22) and the Shapley/interaction representation (Equation 23).

## Key Properties of the Transformation Matrix

1. **Scale Invariant**: The transformation matrix remains identical when the data is scaled uniformly (e.g., multiplying all values by a constant).

2. **Distribution-Dependent**: The transformation varies significantly between different data distributions (uniform, normal, exponential).

3. **Feature-Order Related**: The transformation structure is intimately tied to the ordering of feature values within each sample.

4. **Partially Structured**: Some transformation elements show consistent patterns, while others vary widely across datasets.

5. **Not Universal**: There is no single, fixed transformation matrix that works for all datasets.

## Patterns Observed

### Unit Vector Analysis

When using unit vectors (only one feature has value 1, others are 0):

```
Unit vector with X[0]=1:
[[ 1.   0.   0.  -0.5 -0.5  0. ]
 [ 0.   0.   0.   0.   0.   0. ]
 ...
```

```
Unit vector with X[1]=1:
[[-0.   0.  -0.   0.  -0.   0. ]
 [ 0.   1.   0.  -0.5  0.  -0.5]
 ...
```

This reveals that each feature activates a specific pattern in the transformation matrix, with:
- Direct mapping to its own Shapley value (1.0 on diagonal)
- -0.5 contributions to interaction terms involving that feature

### Statistical Analysis

Elements with lowest variation (most stable across datasets):
- Diagonal elements mapping each feature to its own Shapley value
- Core interaction mappings between commonly ordered pairs

Elements with highest variation (most data-dependent):
- Elements connecting pairs of features with variable ordering across samples
- Cross-terms that depend on the specific ordering patterns in the dataset

## Mathematical Interpretation

The transformation matrix T maps from coalition values v̂ (eq. 22) to Shapley/interaction values v (eq. 23):

v = T · v̂

For 2-additive models with n features, the structure follows a pattern where:

- First n rows map to the n Shapley values
- Remaining rows map to interaction indices
- The mapping depends on the ordering frequency of features in the dataset

## Practical Recommendations

1. **Compute Per Dataset**: Calculate a transformation matrix for each specific dataset rather than using a fixed transformation.

2. **Check for Scale Invariance**: Take advantage of the scale invariance property - you can normalize your data without affecting the transformation.

3. **Prediction Equivalence**: Remember that predictions will be identical when using the appropriate transformation between coefficients:
   - If eq22_matrix @ transform_matrix = eq23_matrix
   - Then model coefficients relate as: coeffs_22 = transform_matrix @ coeffs_23

4. **Choose Representation Based on Need**:
   - Use Equation (22) when you need flexibility for any k-additive model
   - Use Equation (23) when you specifically want Shapley values and interaction indices

## Conclusion

While there are clear patterns in the transformation matrix, its specific values depend on the distribution of the dataset and particularly on the ordering relationships between features. This explains why the transformation varies across different datasets despite representing the same mathematical model family.

For practical applications, compute the transformation matrix for each dataset rather than attempting to use a universal transformation.
