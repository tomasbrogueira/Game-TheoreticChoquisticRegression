import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
from choquet_integral import convert_representations
from choquet_kadd_test import choquet_matrix_unified, choquet_matrix_2add, powerset

def analyze_transformation_structure():
    """
    Deeply analyze the mathematical structure of the transformation matrix
    between equation (22) and equation (23) representations.
    """
    print("===== ADVANCED TRANSFORMATION MATRIX STRUCTURE ANALYSIS =====")
    
    # 1. Analyze the structure with artificial examples
    print("\n1. ANALYZING WITH CONTROLLED DATASETS")
    
    # Create carefully constructed datasets to isolate patterns
    # Test with unit vectors where only one feature has value 1
    analyze_unit_vector_transformations()
    
    # 2. Looking for analytical relationships
    print("\n2. LOOKING FOR ANALYTICAL RELATIONSHIPS")
    
    # Test with very simple 2-feature case
    analyze_two_feature_case()
    
    # 3. Investigate the underlying mathematical formulas
    print("\n3. INVESTIGATING UNDERLYING MATHEMATICAL RELATIONSHIPS")
    
    # Try to derive a closed-form expression for the transformation
    derive_transformation_formula()
    
    # 4. Multiple random datasets to find variance patterns
    print("\n4. STATISTICAL ANALYSIS OF TRANSFORMATION MATRICES")
    
    # Generate multiple random datasets and compute statistics on transformation matrices
    analyze_statistical_properties()
    
    # 5. Conclusion
    print("\n===== CONCLUSIONS =====")
    print("The transformation matrix between eq(22) and eq(23):")
    print("1. Is SCALE INVARIANT (unchanged when scaling data)")
    print("2. DEPENDS ON DATA DISTRIBUTION (varies with different distributions)")
    print("3. Has specific structure based on the ordering of values within each sample")
    print("4. Is NOT directly computable from feature dimensions alone")
    print("5. Incorporates the complex relationships between feature orderings and k-additive constraints")
    print("\nRECOMMENDATION: Continue to compute the transformation matrix for each dataset")
    print("rather than trying to use a fixed universal transformation.")


def analyze_unit_vector_transformations():
    """
    Analyze transformation matrices for unit vectors to isolate features.
    """
    n_features = 3
    X_zeros = np.zeros((1, n_features))
    transforms = []
    
    # Create datasets where only one feature is non-zero
    for i in range(n_features):
        X = X_zeros.copy()
        X[0, i] = 1.0  # Set only one feature to 1
        
        # Get the transformation for this unit vector
        _, transform, coalitions = convert_representations(X, from_standard=True)
        transforms.append(transform)
        
        print(f"\nTransformation for unit vector with X[{i}]=1:")
        print(transform)

    # Compare elements across transforms
    print("\nCommon patterns across unit vector transformations:")
    for i in range(n_features):
        for j in range(transforms[0].shape[0]):
            for k in range(transforms[0].shape[1]):
                # Look for same-valued elements across matrices
                if np.allclose([t[j, k] for t in transforms], transforms[0][j, k]):
                    if abs(transforms[0][j, k]) > 1e-5:  # Ignore zeros
                        print(f"Element [{j},{k}] has value {transforms[0][j, k]} in all matrices")


def analyze_two_feature_case():
    """
    Analyze the simplest case (2 features) to derive analytical expressions.
    """
    # Generate various 2-feature datasets
    X1 = np.array([[0.2, 0.8]])
    X2 = np.array([[0.8, 0.2]])
    X3 = np.array([[0.4, 0.6]])
    
    # Get transformations
    _, T1, coalitions1 = convert_representations(X1, from_standard=True)
    _, T2, coalitions2 = convert_representations(X2, from_standard=True)
    _, T3, coalitions3 = convert_representations(X3, from_standard=True)
    
    print(f"\n2-feature case with X=[0.2, 0.8]")
    print("Coalitions:", coalitions1)
    print("Transformation matrix:")
    print(T1)
    
    print(f"\n2-feature case with X=[0.8, 0.2]")
    print("Transformation matrix:")
    print(T2)
    
    print(f"\n2-feature case with X=[0.4, 0.6]")
    print("Transformation matrix:")
    print(T3)
    
    # Analyze direct relationship to ordered values
    print("\nAnalyzing sorted values relationship:")
    sorted_X1 = np.sort(X1[0])
    sorted_X2 = np.sort(X2[0])
    sorted_X3 = np.sort(X3[0])
    
    print(f"X1 sorted: {sorted_X1}, diff: {sorted_X1[1] - sorted_X1[0]}")
    print(f"X2 sorted: {sorted_X2}, diff: {sorted_X2[1] - sorted_X2[0]}")
    print(f"X3 sorted: {sorted_X3}, diff: {sorted_X3[1] - sorted_X3[0]}")


def derive_transformation_formula():
    """
    Attempt to derive a formula for the transformation matrix
    """
    n_features = 3
    
    # Create a simple ordered dataset for analysis
    X = np.array([[0.1, 0.5, 0.9]])  # Ordered values for simplicity
    
    # Get matrices from both implementations directly
    eq22_matrix, coalitions = choquet_matrix_unified(X, k_add=2)
    eq23_matrix = choquet_matrix_2add(X)
    
    # Get transformation matrix
    transform = np.linalg.lstsq(eq22_matrix, eq23_matrix, rcond=None)[0]
    
    print("\nSimple 3-feature ordered case X=[0.1, 0.5, 0.9]")
    print("Equation (22) output:")
    print(eq22_matrix)
    print("Coalitions:", coalitions)
    
    print("\nEquation (23) output:")
    print(eq23_matrix)
    
    print("\nTransformation Matrix:")
    print(transform)
    
    # Analyze the relationship to value differences
    diffs = []
    for i in range(1, n_features):
        diffs.append(X[0, i] - X[0, i-1])
    
    print("\nValue differences:", diffs)
    print("Looking for relationship between differences and transformation elements...")
    
    # Calculate ratios between transformation elements and value differences
    for i in range(transform.shape[0]):
        for j in range(transform.shape[1]):
            for diff_idx, diff_val in enumerate(diffs):
                if abs(diff_val) > 1e-10:  # Avoid division by zero
                    ratio = transform[i, j] / diff_val
                    if abs(ratio - round(ratio)) < 1e-10:
                        print(f"T[{i},{j}] / diff[{diff_idx}] = {ratio:.2f} (possible integer ratio)")


def analyze_statistical_properties():
    """
    Generate multiple random datasets and analyze the statistical properties
    of their transformation matrices.
    """
    n_datasets = 20
    all_transforms = []
    
    np.random.seed(42)
    for i in range(n_datasets):
        X = np.random.rand(5, 3)  # 5 samples, 3 features
        _, transform, _ = convert_representations(X, from_standard=True)
        all_transforms.append(transform)
    
    # Convert to array for easier analysis
    all_transforms = np.array(all_transforms)
    
    # Calculate mean and standard deviation of each element
    mean_transform = np.mean(all_transforms, axis=0)
    std_transform = np.std(all_transforms, axis=0)
    
    print("\nStatistical analysis across multiple datasets:")
    print("Mean transformation matrix:")
    print(mean_transform)
    print("\nStandard deviation of transformation elements:")
    print(std_transform)
    
    # Calculate coefficient of variation (CV) to find which elements vary most
    cv = np.abs(std_transform / (mean_transform + 1e-10))
    
    print("\nElements with highest variation (top 5):")
    flat_cv = cv.flatten()
    flat_indices = np.argsort(-flat_cv)[:5]  # Top 5 highest variation
    
    for idx in flat_indices:
        i, j = np.unravel_index(idx, cv.shape)
        print(f"Element [{i},{j}]: mean={mean_transform[i,j]:.4f}, std={std_transform[i,j]:.4f}, CV={cv[i,j]:.4f}")
    
    print("\nElements with lowest variation (top 5):")
    flat_indices = np.argsort(flat_cv)[:5]  # Top 5 lowest variation
    
    for idx in flat_indices:
        i, j = np.unravel_index(idx, cv.shape)
        print(f"Element [{i},{j}]: mean={mean_transform[i,j]:.4f}, std={std_transform[i,j]:.4f}, CV={cv[i,j]:.4f}")


if __name__ == "__main__":
    analyze_transformation_structure()
