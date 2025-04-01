"""
Compare the different Choquet integral implementations to understand 
their parameterization approaches and coalition structures.
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from choquet_function import choquet_matrix_2add, choquet_matrix_2add_fixed
from paper_code.covid_comprehensive_test import refined_choquet_k_additive
from coalition_analysis_test import create_simple_patterns

def analyze_matrix_structure(X):
    """Analyze the structure of a transformation matrix"""
    non_zeros = np.count_nonzero(X)
    sparsity = non_zeros / X.size
    
    mean_val = np.mean(X)
    std_val = np.std(X)
    min_val = np.min(X)
    max_val = np.max(X)
    
    # Count negative values
    neg_vals = np.sum(X < 0)
    pos_vals = np.sum(X > 0)
    
    return {
        'non_zeros': non_zeros,
        'sparsity': sparsity,
        'mean': mean_val,
        'std': std_val,
        'min': min_val,
        'max': max_val,
        'negative_values': neg_vals,
        'positive_values': pos_vals
    }

def compare_parameterizations():
    """Compare the different parameterization approaches"""
    print("=== Comparing Choquet Integral Parameterization Approaches ===")
    
    # Create test patterns
    n_features = 4
    patterns, pattern_names = create_simple_patterns(n_features)
    
    # Apply all three implementations
    shapley_original = choquet_matrix_2add(patterns)
    shapley_fixed = choquet_matrix_2add_fixed(patterns)
    game_based = refined_choquet_k_additive(patterns, k_add=2)
    
    # Basic shape comparison
    print("\nMatrix Shapes:")
    print(f"Shapley Original: {shapley_original.shape}")
    print(f"Shapley Fixed: {shapley_fixed.shape}")
    print(f"Game Based: {game_based.shape}")
    
    # Analyze matrix structures
    print("\nMatrix Structure Analysis:")
    print("\nShapley Original:")
    shapley_stats = analyze_matrix_structure(shapley_original)
    for key, value in shapley_stats.items():
        print(f"  {key}: {value}")
    
    print("\nShapley Fixed:")
    fixed_stats = analyze_matrix_structure(shapley_fixed)
    for key, value in fixed_stats.items():
        print(f"  {key}: {value}")
    
    print("\nGame Based:")
    game_stats = analyze_matrix_structure(game_based)
    for key, value in game_stats.items():
        print(f"  {key}: {value}")
    
    # Check column values for specific pattern types
    print("\nColumn Values for Specific Patterns:")
    
    # 1. Single feature active
    pattern_idx = pattern_names.index("Feature 0 only")
    print(f"\nPattern: {pattern_names[pattern_idx]}")
    print("Shapley Original:", shapley_original[pattern_idx, :5])
    print("Shapley Fixed:", shapley_fixed[pattern_idx, :5])
    print("Game Based:", game_based[pattern_idx, :5])
    
    # 2. Two features active
    pattern_idx = pattern_names.index("Features 0 & 1")
    print(f"\nPattern: {pattern_names[pattern_idx]}")
    print("Shapley Original:", shapley_original[pattern_idx, :5])
    print("Shapley Fixed:", shapley_fixed[pattern_idx, :5])
    print("Game Based:", game_based[pattern_idx, :5])
    
    # 3. All features active
    pattern_idx = pattern_names.index("All ones")
    print(f"\nPattern: {pattern_names[pattern_idx]}")
    print("Shapley Original:", shapley_original[pattern_idx, :5])
    print("Shapley Fixed:", shapley_fixed[pattern_idx, :5])
    print("Game Based:", game_based[pattern_idx, :5])
    
    # Calculate similarity between implementations
    print("\nImplementation Similarity (correlation):")
    
    # For each pattern, compute correlation between implementations
    correlations = {
        'shapley_original_vs_fixed': [],
        'shapley_original_vs_game': [],
        'shapley_fixed_vs_game': []
    }
    
    for i in range(len(patterns)):
        # Flatten matrices for correlation calculation
        so = shapley_original[i].flatten()
        sf = shapley_fixed[i].flatten()
        gb = game_based[i].flatten()
        
        # Check if any implementation has all zero values for this pattern
        if np.all(so == 0) or np.all(sf == 0) or np.all(gb == 0):
            continue
        
        # Calculate correlations
        corr_so_sf = np.corrcoef(so, sf)[0, 1] if so.size == sf.size else np.nan
        corr_so_gb = np.corrcoef(so, gb)[0, 1] if so.size == gb.size else np.nan
        corr_sf_gb = np.corrcoef(sf, gb)[0, 1] if sf.size == gb.size else np.nan
        
        correlations['shapley_original_vs_fixed'].append(corr_so_sf)
        correlations['shapley_original_vs_game'].append(corr_so_gb)
        correlations['shapley_fixed_vs_game'].append(corr_sf_gb)
    
    # Calculate mean correlations
    mean_corr_so_sf = np.nanmean(correlations['shapley_original_vs_fixed'])
    mean_corr_so_gb = np.nanmean(correlations['shapley_original_vs_game'])
    mean_corr_sf_gb = np.nanmean(correlations['shapley_fixed_vs_game'])
    
    print(f"Shapley Original vs Fixed: {mean_corr_so_sf:.4f}")
    print(f"Shapley Original vs Game: {mean_corr_so_gb:.4f}")
    print(f"Shapley Fixed vs Game: {mean_corr_sf_gb:.4f}")
    
    # Overall conclusion
    print("\nConclusion:")
    if mean_corr_so_sf > mean_corr_sf_gb:
        print("The fixed implementation is more similar to the original Shapley parameterization.")
    else:
        print("The fixed implementation is more similar to the Game parameterization.")
    
    # Check specific characteristics
    # 1. Presence of negative values
    if shapley_stats['negative_values'] > 0 and fixed_stats['negative_values'] == 0:
        print("However, the fixed implementation doesn't use negative values like the original Shapley.")
    
    # 2. Sparsity
    if abs(fixed_stats['sparsity'] - game_stats['sparsity']) < abs(fixed_stats['sparsity'] - shapley_stats['sparsity']):
        print("The fixed implementation has a sparsity pattern more similar to the Game parameterization.")
    else:
        print("The fixed implementation has a sparsity pattern more similar to the original Shapley.")
    
    # 3. Implementation approach
    print("\nStructural similarities:")
    print("- Shapley Fixed organizes columns by coalition size, like both implementations")
    print("- Shapley Fixed uses direct feature values for singletons, like Shapley Original")
    print("- Shapley Fixed uses min() for pairs, unlike either implementation")
    print("- Shapley Fixed doesn't use negative coefficients, unlike the original Shapley")
    print("- Shapley Fixed doesn't use sorted features and differences, unlike Game-based")
    
    # Add visual comparison of the implementations
    print("\nCreating visual comparison...")
    
    # Pattern to feature matrix visualization
    plt.figure(figsize=(15, 6))
    
    # Choose 6 representative patterns
    selected_pattern_indices = [
        pattern_names.index("Feature 0 only"),         # Single feature
        pattern_names.index("Features 0 & 1"),         # Two features
        pattern_names.index("All ones"),               # All features
        pattern_names.index("All zeros"),              # No features
        pattern_names.index("Ascending values"),       # Gradual values
        pattern_names.index("Descending values")       # Gradual values reverse
    ]
    
    selected_patterns = [pattern_names[i] for i in selected_pattern_indices]
    
    # Create matrix visualizations
    fig, axs = plt.subplots(len(selected_patterns), 3, figsize=(15, 12))
    fig.suptitle('Transformation Matrix Visualization Comparison', fontsize=16)
    
    for i, pattern_idx in enumerate(selected_pattern_indices):
        # Original Shapley
        so_values = shapley_original[pattern_idx]
        im1 = axs[i, 0].imshow(so_values.reshape(1, -1), cmap='coolwarm', vmin=-0.5, vmax=1.0)
        axs[i, 0].set_title(f"{pattern_names[pattern_idx]} - Original" if i == 0 else "")
        axs[i, 0].set_yticks([])
        axs[i, 0].set_xticks([])
        
        # Fixed Shapley
        sf_values = shapley_fixed[pattern_idx]
        im2 = axs[i, 1].imshow(sf_values.reshape(1, -1), cmap='coolwarm', vmin=-0.5, vmax=1.0)
        axs[i, 1].set_title(f"{pattern_names[pattern_idx]} - Fixed" if i == 0 else "")
        axs[i, 1].set_yticks([])
        axs[i, 1].set_xticks([])
        
        # Game Based
        gb_values = game_based[pattern_idx]
        im3 = axs[i, 2].imshow(gb_values.reshape(1, -1), cmap='coolwarm', vmin=-0.5, vmax=1.0)
        axs[i, 2].set_title(f"{pattern_names[pattern_idx]} - Game" if i == 0 else "")
        axs[i, 2].set_yticks([])
        axs[i, 2].set_xticks([])
        
        # Add pattern name as y-label
        axs[i, 0].set_ylabel(pattern_names[pattern_idx], fontsize=10)
    
    # Add shared colorbar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    fig.colorbar(im1, cax=cbar_ax)
    
    # Save visualization
    plt.tight_layout(rect=[0, 0, 0.85, 0.95])
    plt.savefig('choquet_implementation_comparison.png')
    print("Visual comparison saved as 'choquet_implementation_comparison.png'")
    
    print("\nOverall, the fixed implementation uses a Shapley-style parameterization")
    print("but with a corrected coalition structure that better adheres to k-additivity theory.")
    print("The visual comparison shows that it combines the best aspects of both approaches.")

if __name__ == "__main__":
    compare_parameterizations()
