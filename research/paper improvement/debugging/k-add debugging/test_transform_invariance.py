import numpy as np
import matplotlib.pyplot as plt
from choquet_integral import convert_representations

def analyze_transformation_matrix_invariance():
    """
    Analyze invariance properties of the transformation matrix between 
    equation (22) and (23) representations across different datasets.
    """
    print("Testing transformation matrix invariance")
    
    # Test with different feature counts
    for n_features in [2, 3, 4]:
        np.random.seed(42)
        X = np.random.rand(100, n_features)
        _, transform_matrix, _ = convert_representations(X, from_standard=True)
        print(f"\nMatrix for {n_features} features:")
        print(transform_matrix)
    
    # Test with different data distributions
    np.random.seed(42)
    X_uniform = np.random.rand(100, 3)
    X_normal = np.random.randn(100, 3)
    X_exp = np.random.exponential(size=(100, 3))
    
    _, transform_uniform, _ = convert_representations(X_uniform, from_standard=True)
    _, transform_normal, _ = convert_representations(X_normal, from_standard=True)
    _, transform_exp, _ = convert_representations(X_exp, from_standard=True)
    
    uniform_vs_normal = np.allclose(transform_uniform, transform_normal, rtol=1e-5)
    uniform_vs_exp = np.allclose(transform_uniform, transform_exp, rtol=1e-5)
    
    print(f"\nDistribution invariance check:")
    print(f"Uniform vs Normal: {'equal' if uniform_vs_normal else 'different'}")
    print(f"Uniform vs Exponential: {'equal' if uniform_vs_exp else 'different'}")
    
    # Test with different scales
    np.random.seed(42)
    X_base = np.random.rand(100, 3)
    X_scaled = X_base * 10
    
    _, transform_base, _ = convert_representations(X_base, from_standard=True)
    _, transform_scaled, _ = convert_representations(X_scaled, from_standard=True)
    
    scales_match = np.allclose(transform_base, transform_scaled, rtol=1e-5)
    print(f"\nScale invariance: {'equal' if scales_match else 'different'}")
    
    # Analyze specific case
    np.random.seed(42)
    X_small = np.random.rand(1, 3)
    _, transform_small, coalitions = convert_representations(X_small, from_standard=True)
    
    print(f"\nCoalitions: {coalitions}")
    print("Matrix structure for 3 features:")
    print(transform_small)
    
    # Column statistics
    for j in range(transform_small.shape[1]):
        col_stats = (np.min(transform_small[:, j]), 
                     np.max(transform_small[:, j]), 
                     np.mean(transform_small[:, j]))
        print(f"Col {j}: min={col_stats[0]:.3f}, max={col_stats[1]:.3f}, mean={col_stats[2]:.3f}")
    
    # Analyze singleton coalition patterns
    n_features = 3
    for i, coalition in enumerate(coalitions):
        if len(coalition) == 1:
            feat = coalition[0]
            identity_val = transform_small[i, feat]
            interactions = [(j, transform_small[i, j]) for j in range(n_features, transform_small.shape[1]) 
                           if abs(transform_small[i, j]) > 1e-5]
            
            if interactions:
                print(f"\nFeature {feat} contributions: {interactions}")
    
    # Results summary
    invariant = all([uniform_vs_normal, uniform_vs_exp, scales_match])
    print(f"\nMatrix is {'invariant' if invariant else 'variant'} across datasets")
    
    # Visualize matrices
    plt.figure(figsize=(15, 5))

    # 2 features
    plt.subplot(131)
    np.random.seed(42)
    X2 = np.random.rand(100, 2)
    _, transform2, coalitions2 = convert_representations(X2, from_standard=True)

    # Create coalition labels for 2 features
    coalition_labels2 = [f"{{{','.join(map(str, c))}}}" for c in coalitions2]

    plt.imshow(transform2, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title(f"2 features")
    plt.xticks(range(len(coalition_labels2)), coalition_labels2, rotation=90, fontsize=8)
    plt.yticks(range(len(coalition_labels2)), coalition_labels2, fontsize=8)

    # 3 features
    plt.subplot(132)
    _, transform_uniform, coalitions3 = convert_representations(X_uniform, from_standard=True)

    # Create coalition labels for 3 features
    coalition_labels3 = [f"{{{','.join(map(str, c))}}}" for c in coalitions3]

    plt.imshow(transform_uniform, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title(f"3 features")
    plt.xticks(range(len(coalition_labels3)), coalition_labels3, rotation=90, fontsize=8)
    plt.yticks(range(len(coalition_labels3)), coalition_labels3, fontsize=8)

    # 4 features
    plt.subplot(133)
    np.random.seed(42)
    X4 = np.random.rand(100, 4)
    _, transform4, coalitions4 = convert_representations(X4, from_standard=True)

    # Create coalition labels for 4 features
    coalition_labels4 = [f"{{{','.join(map(str, c))}}}" for c in coalitions4]

    plt.imshow(transform4, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title(f"4 features")
    plt.xticks(range(len(coalition_labels4)), coalition_labels4, rotation=90, fontsize=8)
    plt.yticks(range(len(coalition_labels4)), coalition_labels4, fontsize=8)

    plt.tight_layout()
    plt.savefig("transformation_matrices.png")
    plt.close()


analyze_transformation_matrix_invariance()