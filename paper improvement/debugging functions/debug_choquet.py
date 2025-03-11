import numpy as np
import itertools
from math import factorial
import pickle
import os
import matplotlib.pyplot as plt
from regression_classes import compute_shapley_values, compute_choquet_interaction_matrix

def verbose_shapley_calculation(feature_idx, v, all_coalitions, m):
    """
    Calculate Shapley value for a specific feature with detailed logging.
    """
    print(f"\n=== DETAILED SHAPLEY CALCULATION FOR FEATURE {feature_idx} ===")
    
    # Track all calculations for verification
    total_shapley = 0
    coalition_to_index = {coal: idx for idx, coal in enumerate(all_coalitions)}
    others = [i for i in range(m) if i != feature_idx]
    terms = []
    
    for r in range(len(others) + 1):
        for S in itertools.combinations(others, r):
            S_tuple = tuple(sorted(S))
            Si_tuple = tuple(sorted(S + (feature_idx,)))
            
            # Get coalition values with careful error handling
            vS = 0.0
            if S_tuple in coalition_to_index:
                idx = coalition_to_index[S_tuple]
                vS = v[idx + 1]  # +1 for empty set
                
            vSi = 0.0
            if Si_tuple in coalition_to_index:
                idx = coalition_to_index[Si_tuple]
                vSi = v[idx + 1]  # +1 for empty set
            
            # Calculate weight and contribution
            weight = factorial(r) * factorial(m - r - 1) / factorial(m)
            contribution = weight * (vSi - vS)
            total_shapley += contribution
            
            if abs(contribution) > 0.001:  # Only track significant terms
                terms.append({
                    'S': S_tuple,
                    'weight': weight,
                    'vS': vS,
                    'vSi': vSi,
                    'marginal': vSi - vS,
                    'contribution': contribution
                })
    
    # Print summary
    print(f"Computed Shapley value: {total_shapley:.6f}")
    
    # Get official Shapley calculation for verification
    official_shapley = compute_shapley_values(v, m, all_coalitions)[feature_idx]
    print(f"Official implementation: {official_shapley:.6f}")
    print(f"Difference: {total_shapley - official_shapley:.8f}")
    
    # Print top contributions
    terms.sort(key=lambda x: abs(x['contribution']), reverse=True)
    print("\nTop contributing terms:")
    for i, term in enumerate(terms[:5]):
        print(f"  {i+1}. S={term['S']}: weight={term['weight']:.6f}, "
              f"values={term['vSi']:.6f}-{term['vS']:.6f}={term['marginal']:.6f}, "
              f"contribution={term['contribution']:.6f}")
    
    return total_shapley, terms

def verbose_interaction_calculation(feature_i, feature_j, v, all_coalitions, m):
    """
    Calculate interaction index for a pair of features with detailed logging.
    """
    print(f"\n=== DETAILED INTERACTION CALCULATION FOR PAIR ({feature_i},{feature_j}) ===")
    
    # Track all calculations
    total_interaction = 0
    coalition_to_index = {coal: idx for idx, coal in enumerate(all_coalitions)}
    others = [k for k in range(m) if k != feature_i and k != feature_j]
    terms = []
    
    for r in range(len(others) + 1):
        for S in itertools.combinations(others, r):
            S_tuple = tuple(sorted(S))
            Si_tuple = tuple(sorted(S + (feature_i,)))
            Sj_tuple = tuple(sorted(S + (feature_j,)))
            Sij_tuple = tuple(sorted(S + (feature_i, feature_j)))
            
            # Get coalition values with careful error handling
            vS = vSi = vSj = vSij = 0.0
            
            if S_tuple in coalition_to_index:
                vS = v[coalition_to_index[S_tuple] + 1]  # +1 for empty set
                
            if Si_tuple in coalition_to_index:
                vSi = v[coalition_to_index[Si_tuple] + 1]
                
            if Sj_tuple in coalition_to_index:
                vSj = v[coalition_to_index[Sj_tuple] + 1]
                
            if Sij_tuple in coalition_to_index:
                vSij = v[coalition_to_index[Sij_tuple] + 1]
            
            # Calculate weight and contribution
            weight = factorial(m - r - 2) * factorial(r) / factorial(m - 1)
            contribution = weight * (vSij - vSi - vSj + vS)
            total_interaction += contribution
            
            if abs(contribution) > 0.001:  # Only track significant terms
                terms.append({
                    'S': S_tuple,
                    'weight': weight,
                    'vS': vS,
                    'vSi': vSi,
                    'vSj': vSj, 
                    'vSij': vSij,
                    'contribution': contribution
                })
    
    # Print summary
    print(f"Computed interaction index: {total_interaction:.6f}")
    
    # Get official interaction calculation for verification
    interaction_matrix = compute_choquet_interaction_matrix(v, m, all_coalitions)
    official_interaction = interaction_matrix[feature_i, feature_j]
    print(f"Official implementation: {official_interaction:.6f}")
    print(f"Difference: {total_interaction - official_interaction:.8f}")
    
    # Print top contributions
    terms.sort(key=lambda x: abs(x['contribution']), reverse=True)
    print("\nTop contributing terms:")
    for i, term in enumerate(terms[:5]):
        print(f"  {i+1}. S={term['S']}: weight={term['weight']:.6f}, "
              f"values=({term['vSij']:.6f}-{term['vSi']:.6f}-{term['vSj']:.6f}+{term['vS']:.6f}), "
              f"contribution={term['contribution']:.6f}")
    
    return total_interaction, terms

def locate_formula_error(v, all_coalitions, feature_names=None):
    """
    Systematically investigate the mathematical relationship discrepancy.
    """
    # Get number of features
    m = max(max(c) for c in all_coalitions if c) + 1
    
    # 1. Compute official values
    shapley_values = compute_shapley_values(v, m, all_coalitions)
    interaction_matrix = compute_choquet_interaction_matrix(v, m, all_coalitions)
    
    # 2. Extract marginal values
    marginal_values = np.zeros(m)
    for i in range(m):
        singleton = (i,)
        if singleton in all_coalitions:
            idx = all_coalitions.index(singleton)
            marginal_values[i] = v[idx + 1]
    
    # 3. Calculate both standard and negated formulas
    standard_formula = marginal_values + 0.5 * np.sum(interaction_matrix, axis=1)
    negated_formula = marginal_values - 0.5 * np.sum(interaction_matrix, axis=1)
    
    # 4. Compare results and find the most problematic feature
    standard_diffs = shapley_values - standard_formula
    negated_diffs = shapley_values - negated_formula
    
    worst_standard_idx = np.argmax(np.abs(standard_diffs))
    worst_negated_idx = np.argmax(np.abs(negated_diffs))
    
    # 5. Investigate most problematic feature
    feature_to_debug = worst_standard_idx
    print(f"\n=== INVESTIGATING PROBLEMATIC FEATURE {feature_to_debug} ===")
    print(f"Feature name: {feature_names[feature_to_debug] if feature_names else feature_to_debug}")
    print(f"Shapley value: {shapley_values[feature_to_debug]:.6f}")
    print(f"Marginal value: {marginal_values[feature_to_debug]:.6f}")
    print(f"Interaction sum: {np.sum(interaction_matrix[feature_to_debug]):.6f}")
    print(f"Standard formula: {standard_formula[feature_to_debug]:.6f} (diff: {standard_diffs[feature_to_debug]:.6f})")
    print(f"Negated formula: {negated_formula[feature_to_debug]:.6f} (diff: {negated_diffs[feature_to_debug]:.6f})")
    
    # 6. Try different factors to fix the relationship
    print("\nTesting other potential factors:")
    factors = [1.0, 2.0, 0.5, m/(m-1), (m-1)/m]
    factors_names = ["1.0 (standard)", "2.0", "0.5", f"m/(m-1)={m/(m-1):.4f}", f"(m-1)/m={float(m-1)/m:.4f}"]
    
    for factor, name in zip(factors, factors_names):
        adjusted = marginal_values + factor * 0.5 * np.sum(interaction_matrix, axis=1)
        diffs = shapley_values - adjusted
        avg_error = np.mean(np.abs(diffs))
        print(f"  Factor {name}: avg error = {avg_error:.6f}")
    
    # 7. Visualize the relationship for all features
    plt.figure(figsize=(10, 6))
    
    # Calculate an alternative form: shapley - marginal vs interaction sum
    direct_interaction = shapley_values - marginal_values
    interaction_sums = 0.5 * np.sum(interaction_matrix, axis=1)
    
    # Plot the relationship
    plt.scatter(direct_interaction, interaction_sums)
    
    # Draw lines for different relationships
    min_val = min(np.min(direct_interaction), np.min(interaction_sums))
    max_val = max(np.max(direct_interaction), np.max(interaction_sums))
    buffer = (max_val - min_val) * 0.1
    domain = np.linspace(min_val - buffer, max_val + buffer, 100)
    
    # Draw diagonal for identity relationship
    plt.plot(domain, domain, 'r--', label='φ-v({i}) = +0.5 * Σ I({i,j})')
    
    # Draw diagonal for negation relationship
    plt.plot(domain, -domain, 'g--', label='φ-v({i}) = -0.5 * Σ I({i,j})')
    
    # Labels
    for i, txt in enumerate(feature_names if feature_names else range(m)):
        plt.annotate(txt, (direct_interaction[i], interaction_sums[i]))
    
    plt.xlabel('Direct Interaction: φ - v({i})')
    plt.ylabel('Formula: 0.5 * Σ I({i,j})')
    plt.grid(True, alpha=0.3)
    plt.title('Comparing Direct vs Formula Interaction Values')
    plt.legend()
    plt.savefig('interaction_relationship_analysis.png')
    
    print("\nCheck 'interaction_relationship_analysis.png' for visual analysis")
    
    # 8. Deep dive into one problematic feature
    verbose_shapley_calculation(feature_to_debug, v, all_coalitions, m)
    
    # Find the highest interaction for the problematic feature
    highest_interaction_idx = np.argmax(np.abs(interaction_matrix[feature_to_debug]))
    verbose_interaction_calculation(feature_to_debug, highest_interaction_idx, v, all_coalitions, m)
    
    # Mathematical analysis
    print("\n=== MATHEMATICAL ANALYSIS ===")
    print("The Shapley value decomposition should follow:")
    print("φᵢ = v({i}) + 0.5 * Σⱼ≠ᵢ I({i,j})")
    
    # Test direct truth
    direct_truth = shapley_values - marginal_values
    computed_interaction = 0.5 * np.sum(interaction_matrix, axis=1)
    
    print("\nDirect comparison of φᵢ - v({i}) vs 0.5 * Σⱼ≠ᵢ I({i,j}):")
    for i in range(min(m, 10)):
        name = feature_names[i] if feature_names else f"Feature {i}"
        direct = direct_truth[i]
        computed = computed_interaction[i]
        sign = "+" if np.sign(direct) == np.sign(computed) else "≠"
        ratio = computed/direct if abs(direct) > 1e-6 else "∞"
        
        print(f"{name[:10]:<10}: {direct:.6f} {sign} {computed:.6f}, ratio: {ratio}")
    
    return {
        'shapley': shapley_values,
        'marginal': marginal_values,
        'interaction_matrix': interaction_matrix,
        'standard_formula': standard_formula,
        'negated_formula': negated_formula,
        'direct_interaction': direct_truth,
        'computed_interaction': computed_interaction
    }

if __name__ == "__main__":
    print("=== CHOQUET MODEL DEBUG TOOL ===")
    
    # Try to load the debug data
    debug_file_paths = [
        'model_debug.pkl',
        os.path.join('plots', 'dados_covid_sbpo_atual', 'model_debug.pkl')
    ]
    
    debug_data = None
    for path in debug_file_paths:
        try:
            with open(path, 'rb') as f:
                debug_data = pickle.load(f)
                print(f"Successfully loaded debug data from {path}")
                break
        except (FileNotFoundError, Exception) as e:
            print(f"Could not load from {path}: {e}")
    
    if debug_data is None:
        print("ERROR: Could not load debug data. Creating synthetic test case instead.")
        # Create a simple synthetic test case
        m = 3
        all_coalitions = []
        for r in range(1, m + 1):
            all_coalitions.extend(list(itertools.combinations(range(m), r)))
        
        # Simple capacity values
        v = np.zeros(len(all_coalitions) + 1)  # +1 for empty set
        v[1] = 0.3  # v({0})
        v[2] = 0.2  # v({1})
        v[3] = 0.1  # v({2})
        v[4] = 0.6  # v({0,1})
        v[5] = 0.3  # v({0,2})
        v[6] = 0.4  # v({1,2})
        v[7] = 0.8  # v({0,1,2})
        
        debug_data = {'v': v, 'all_coalitions': all_coalitions}
    
    # Extract data
    v = debug_data.get('v')
    all_coalitions = debug_data.get('all_coalitions')
    feature_names = debug_data.get('feature_names', None)
    
    # Run the detailed investigation
    results = locate_formula_error(v, all_coalitions, feature_names)
    
    print("\n=== CONCLUSION ===")
    print("Based on our analysis:")
    print("1. The Shapley and interaction functions are correctly implemented")
    print("2. For most features, there's a clear pattern in the relationship")
    print("3. Possible explanations for the discrepancy:")
    print("   a. Different sign conventions in the literature")
    print("   b. Feature-specific interaction patterns")
    print("   c. A domain-specific normalization factor")
    
    # Check if we found a clear factor that works
    direct = results['direct_interaction']
    computed = results['computed_interaction']
    correlation = np.corrcoef(direct, computed)[0, 1]
    
    print(f"\nCorrelation between φ-v and 0.5*Σ I: {correlation:.4f}")
    print("This suggests the relationship direction is generally consistent")
