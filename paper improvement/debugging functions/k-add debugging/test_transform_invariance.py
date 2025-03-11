import numpy as np
import matplotlib.pyplot as plt
from choquet_integral import convert_representations

def analyze_transformation_matrix_invariance():
    """
    Analyze whether the transformation matrix between equation (22) and equation (23)
    representations is constant across different datasets, or how it varies.
    """
    print("===== TESTING TRANSFORMATION MATRIX INVARIANCE =====")
    
    # 1. Test with different feature counts
    print("\n1. TESTING WITH DIFFERENT FEATURE COUNTS")
    for n_features in [2, 3, 4]:
        np.random.seed(42)  # Keep seed consistent for reproducibility
        X = np.random.rand(100, n_features)
        _, transform_matrix, _ = convert_representations(X, from_standard=True)
        
        print(f"\n{n_features} features transformation matrix:")
        print(transform_matrix)
    
    # 2. Test with different data distributions
    print("\n\n2. TESTING WITH DIFFERENT DISTRIBUTIONS")
    # Uniform distribution [0,1]
    np.random.seed(42)
    X_uniform = np.random.rand(100, 3)
    _, transform_uniform, _ = convert_representations(X_uniform, from_standard=True)
    
    # Normal distribution
    np.random.seed(42)
    X_normal = np.random.randn(100, 3)
    _, transform_normal, _ = convert_representations(X_normal, from_standard=True)
    
    # Exponential distribution
    np.random.seed(42)
    X_exp = np.random.exponential(size=(100, 3))
    _, transform_exp, _ = convert_representations(X_exp, from_standard=True)
    
    print("\nUniform distribution [0,1]:")
    print(transform_uniform)
    print("\nNormal distribution:")
    print(transform_normal)
    print("\nExponential distribution:")
    print(transform_exp)
    
    # Check if they're the same
    uniform_vs_normal = np.allclose(transform_uniform, transform_normal, rtol=1e-5)
    uniform_vs_exp = np.allclose(transform_uniform, transform_exp, rtol=1e-5)
    
    print(f"\nUniform vs Normal matrices match: {'✓' if uniform_vs_normal else '❌'}")
    print(f"Uniform vs Exponential matrices match: {'✓' if uniform_vs_exp else '❌'}")
    
    # 3. Test with different scales
    print("\n\n3. TESTING WITH DIFFERENT SCALES")
    np.random.seed(42)
    X_base = np.random.rand(100, 3)
    X_scaled = X_base * 10
    
    _, transform_base, _ = convert_representations(X_base, from_standard=True)
    _, transform_scaled, _ = convert_representations(X_scaled, from_standard=True)
    
    print("\nBase scale:")
    print(transform_base)
    print("\nScaled by 10:")
    print(transform_scaled)
    
    scales_match = np.allclose(transform_base, transform_scaled, rtol=1e-5)
    print(f"\nDifferent scales match: {'✓' if scales_match else '❌'}")
    
    # 4. Analyze a specific case in detail for mathematical relationship
    print("\n\n4. DETAILED ANALYSIS OF A SPECIFIC CASE")
    np.random.seed(42)
    X_small = np.random.rand(1, 3)  # Just one sample, 3 features
    _, transform_small, coalitions = convert_representations(X_small, from_standard=True)
    
    print("\nCoalitions:", coalitions)
    print("Transformation matrix for 3 features:")
    print(transform_small)
    
    # Try to identify mathematical pattern in the matrix
    # For 3 features with coalitions [(0,), (1,), (2,), (0,1), (0,2), (1,2)]
    # The matrix should map from coalition values to Shapley/interaction values
    
    # Calculate value range in each column to check for patterns
    print("\nValue ranges in transformation matrix columns:")
    for j in range(transform_small.shape[1]):
        col_min = np.min(transform_small[:, j])
        col_max = np.max(transform_small[:, j])
        col_mean = np.mean(transform_small[:, j])
        print(f"Column {j}: min={col_min:.4f}, max={col_max:.4f}, mean={col_mean:.4f}")
    
    # 5. Analyze mathematical structure
    print("\n\n5. MATHEMATICAL STRUCTURE ANALYSIS")
    # Test if the transformation matrix has a fixed structure related to 
    # coalition sizes or feature counts
    
    # For 3 features, 2-additive model:
    n_features = 3
    
    # Calculate theoretical values for a transformation matrix based on combinatorial structure
    print("\nTheoretical analysis:")
    print(f"For {n_features} features:")
    print(f"- {n_features} singleton coalitions")
    print(f"- {n_features*(n_features-1)//2} pair coalitions")
    
    for i in range(len(coalitions)):
        coalition = coalitions[i]
        print(f"\nCoalition {coalition} of size {len(coalition)}:")
        print(f"Row {i} in transformation matrix: {transform_small[i]}")
        
        # Analyze row patterns based on coalition size
        if len(coalition) == 1:
            # Singleton coalition
            feat = coalition[0]
            print(f"Singleton feature {feat}:")
            # Look for Identity mapping for same feature
            identity_val = transform_small[i, feat]
            print(f"- Maps to itself with value: {identity_val:.4f}")
            # Look for contributions to pair interactions
            for j in range(n_features, transform_small.shape[1]):
                if abs(transform_small[i, j]) > 1e-5:
                    print(f"- Contributes {transform_small[i, j]:.4f} to column {j}")
    
    # 6. Summary and conclusion
    print("\n\n6. SUMMARY AND CONCLUSION")
    
    if all([uniform_vs_normal, uniform_vs_exp, scales_match]):
        print("✓ TRANSFORMATION MATRIX IS CONSTANT across different datasets")
        print("This means the relationship between eq(22) and eq(23) is purely mathematical")
        print("and doesn't depend on the specific data values")
    else:
        print("❌ TRANSFORMATION MATRIX VARIES based on dataset properties:")
        if not uniform_vs_normal:
            print("- Varies with data distribution")
        if not uniform_vs_exp:
            print("- Varies with data distribution")
        if not scales_match:
            print("- Varies with data scaling")
        print("This means the relationship between eq(22) and eq(23) is data-dependent")
    
    # 7. Visualize structure of transformation matrices
    plt.figure(figsize=(15, 5))
    
    # Plot matrix for 2 features
    plt.subplot(131)
    np.random.seed(42)
    X2 = np.random.rand(100, 2)
    _, transform2, _ = convert_representations(X2, from_standard=True)
    plt.imshow(transform2, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title(f"2 features ({transform2.shape[0]}×{transform2.shape[1]})")
    plt.xlabel("Target columns")
    plt.ylabel("Source rows")
    
    # Plot matrix for 3 features
    plt.subplot(132)
    plt.imshow(transform_uniform, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title(f"3 features ({transform_uniform.shape[0]}×{transform_uniform.shape[1]})")
    plt.xlabel("Target columns")
    plt.ylabel("Source rows")
    
    # Plot matrix for 4 features
    plt.subplot(133)
    np.random.seed(42)
    X4 = np.random.rand(100, 4)
    _, transform4, _ = convert_representations(X4, from_standard=True)
    plt.imshow(transform4, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title(f"4 features ({transform4.shape[0]}×{transform4.shape[1]})")
    plt.xlabel("Target columns")
    plt.ylabel("Source rows")
    
    plt.tight_layout()
    plt.savefig("transformation_matrices.png")
    plt.close()
    print("\nVisualization of transformation matrices saved to transformation_matrices.png")

if __name__ == "__main__":
    analyze_transformation_matrix_invariance()
