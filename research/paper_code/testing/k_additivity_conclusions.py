"""
Analysis of how each Choquet model implementation handles k-additivity.
This analysis is based on the output of the coalition_analysis_test.py script.
"""
import matplotlib.pyplot as plt
import numpy as np
import os

def create_summary_visualizations():
    """Create visualizations to summarize the k-additivity findings"""
    # Directory for plots
    os.makedirs("plots", exist_ok=True)
    
    # Visualize coalition activation patterns
    plt.figure(figsize=(15, 8))
    
    # Define test patterns
    patterns = ["Feature 0 only", "Features 0 & 1", "All ones", "Ascending values"]
    
    # Coalition activations by implementation and k-value
    # Format: [Game-k1, Game-k2, Game-k3, Game-k4, Refined-k1, Refined-k2, Refined-k3, Refined-k4, Shapley-k2]
    activations = np.array([
        # Feature 0 only
        [1, 1, 1, 1, 1, 1, 1, 1, 4],
        # Features 0 & 1
        [0, 1, 1, 1, 1, 2, 2, 2, 6],
        # All ones
        [0, 0, 0, 1, 1, 4, 7, 8, 4],
        # Ascending values
        [1, 2, 3, 4, 4, 10, 14, 15, 10]
    ])
    
    # Total possible columns for each k
    k_columns = [4, 10, 14, 15, 4, 10, 14, 15, 10]
    
    # Normalized activations (percentage of possible columns)
    normalized = activations / np.array(k_columns) * 100
    
    # Plot the data with a heatmap
    plt.imshow(normalized, aspect='auto', cmap='viridis')
    
    # Add labels
    plt.colorbar(label='Percentage of Columns Activated')
    plt.ylabel('Input Pattern')
    plt.yticks(range(len(patterns)), patterns)
    
    implementations = ['Game k=1', 'Game k=2', 'Game k=3', 'Game k=4', 
                      'Refined k=1', 'Refined k=2', 'Refined k=3', 'Refined k=4',
                      'Shapley k=2']
    plt.xlabel('Implementation and k-value')
    plt.xticks(range(len(implementations)), implementations, rotation=45, ha='right')
    
    plt.title('Coalition Activation Patterns Across Implementations', fontsize=16)
    plt.tight_layout()
    plt.savefig('plots/coalition_activation_heatmap.png')
    plt.close()
    
    # Create a summary of missing coalitions
    plt.figure(figsize=(10, 6))
    
    # Expected coalitions vs. active coalitions
    # [Game k=1, Game k=2, Game k=3, Game k=4, Refined k=1, Refined k=2, Refined k=3, Refined k=4, Shapley k=2]
    expected = [4, 10, 14, 15, 4, 10, 14, 15, 10]
    active = [4, 10, 12, 13, 4, 10, 14, 15, 10]
    
    # Calculate missing coalitions
    missing = [e - a for e, a in zip(expected, active)]
    
    # Create bar chart
    x = range(len(implementations))
    plt.bar(x, expected, label='Expected Coalitions', alpha=0.7, color='blue')
    plt.bar(x, active, label='Active Coalitions', alpha=0.7, color='green')
    
    # Add missing coalition counts
    for i, m in enumerate(missing):
        if m > 0:
            plt.text(i, active[i] + 0.5, f"Missing: {m}", ha='center', fontweight='bold', color='red')
    
    plt.xlabel('Implementation and k-value')
    plt.xticks(x, implementations, rotation=45, ha='right')
    plt.ylabel('Number of Coalitions')
    plt.title('Expected vs. Actual Coalitions by Implementation', fontsize=16)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('plots/missing_coalitions.png')
    plt.close()

def print_k_additivity_findings():
    """Print a comprehensive analysis of k-additivity across implementations"""
    print("\n=== K-ADDITIVITY ANALYSIS ACROSS IMPLEMENTATIONS ===\n")
    
    print("1. GAME-BASED (ORDERED) IMPLEMENTATION")
    print("--------------------------------------")
    print("• K-additivity behavior:")
    print("  - Creates extremely sparse transformation matrices")
    print("  - Only activates columns when there are differences between sorted values")
    print("  - For k=1: Single features activate exactly one column")
    print("  - For k=2: Two-feature patterns activate exactly one column")
    print("  - For higher k: Each k value adds one column for patterns with enough variability")
    print("  - Uniform patterns ('All zeros', 'All ones') activate few or no columns")
    print("  - Missing 2 expected coalitions at k=3 and k=4")
    print("• Coalition selection principle:")
    print("  - Uses permutation-dependent coalitions based on sorted feature ordering")
    print("  - Only considers specific coalitions determined by ordering sequence")
    print("  - Implementation follows classic Choquet integral with sorting constraints")
    print("• Key insight: Information loss at low k values requires higher k for good performance\n")
    
    print("2. REFINED (UNORDERED) IMPLEMENTATION")
    print("------------------------------------")
    print("• K-additivity behavior:")
    print("  - Creates dense transformation matrices")
    print("  - Activates all possible columns for patterns with variability")
    print("  - For k=1: Uses all singleton coalitions")
    print("  - For k=2: Uses all singleton and pair coalitions") 
    print("  - No missing coalitions at any k value")
    print("  - Perfect consistency with k-additivity theory")
    print("• Coalition selection principle:")
    print("  - Considers all valid coalitions containing each feature")
    print("  - For each feature, activates all coalitions that include it and higher-ranked features")
    print("  - Preserves more information through comprehensive coalition evaluation")
    print("• Key insight: Better performance at lower k values but computationally more expensive\n")
    
    print("3. SHAPLEY (2-ADD) IMPLEMENTATION")
    print("--------------------------------")
    print("• K-additivity behavior:")
    print("  - Fixed at k=2 with a specialized structure")
    print("  - Creates moderately dense transformation matrices")
    print("  - For single features: Activates 4 columns (1 positive, 3 negative)")
    print("  - For two-feature patterns: Activates 6 columns")
    print("  - For continuous patterns: Activates all 10 columns")
    print("• Coalition selection principle:")
    print("  - Direct parametrization of main effects (singleton features)")
    print("  - Explicit modeling of all pairwise interactions using min operations")
    print("  - Uses negative coefficients on interaction terms to adjust")
    print("  - Mathematically optimized specifically for 2-additive models")
    print("• Key insight: Most efficient at k=2, mathematically specialized for interactions\n")
    
    print("CONCLUSIONS")
    print("-----------")
    print("1. The implementations differ fundamentally in how they select and activate coalitions:")
    print("   - Game-based: Selective activation based on feature ordering")
    print("   - Refined: Comprehensive activation of all valid coalitions")
    print("   - Shapley: Specialized structure for main effects and interactions")
    
    print("2. These differences explain the performance variations:")
    print("   - Game-based needs higher k (≥4) to achieve good performance")
    print("   - Refined achieves good performance at lower k values")
    print("   - Shapley achieves excellent performance at fixed k=2")
    
    print("3. Coalition selection strategies directly impact:")
    print("   - Information preservation (how much of the original data is retained)")
    print("   - Model complexity (number of parameters/features)")
    print("   - Computational efficiency (transformation time and memory usage)")
    
    print("4. The Shapley implementation's superiority at k=2 comes from its:")
    print("   - Mathematical optimization for capturing pairwise interactions")
    print("   - Direct modeling of both main effects and interactions")
    print("   - Efficient representation with minimal information loss")
    
    print("5. The k-additivity parameter functions as expected in all implementations")
    print("   but with different information preservation characteristics")

if __name__ == "__main__":
    create_summary_visualizations()
    print_k_additivity_findings()
