import numpy as np
import itertools
from math import factorial
import matplotlib.pyplot as plt
import pickle
import os
from regression_classes import compute_shapley_values, compute_choquet_interaction_matrix

def trace_shapley_calculation(feature_idx, v, all_coalitions, m):
    """
    Trace the full calculation of the Shapley value for a specific feature.
    
    The Shapley value formula is:
    φᵢ = ∑_{S⊆N\{i}} [|S|!(n-|S|-1)!/n!] * [v(S∪{i}) - v(S)]
    
    Parameters:
    -----------
    feature_idx : int
        Index of the feature to trace
    v : array-like
        Capacity values with v[0] for empty set
    all_coalitions : list of tuples
        List of all coalitions
    m : int
        Number of features
        
    Returns:
    --------
    dict : Detailed breakdown of the calculation
    """
    print(f"\n=== TRACING SHAPLEY VALUE CALCULATION FOR FEATURE {feature_idx} ===")
    
    coalition_to_index = {coal: idx for idx, coal in enumerate(all_coalitions)}
    shapley_value = 0
    terms = []
    
    # Others = all features except feature_idx
    others = [i for i in range(m) if i != feature_idx]
    
    # Track running sum for verification
    running_sum = 0
    
    # Iterate through all subsets of others
    for r in range(len(others) + 1):
        for S in itertools.combinations(others, r):
            S = tuple(sorted(S))
            Si = tuple(sorted(S + (feature_idx,)))
            
            # Get coalition values
            try:
                vS_idx = coalition_to_index[S] if S in coalition_to_index else -1
                vS = v[vS_idx + 1] if vS_idx >= 0 else 0
            except (IndexError, ValueError):
                vS = 0
                
            try:
                vSi_idx = coalition_to_index[Si] if Si in coalition_to_index else -1
                vSi = v[vSi_idx + 1] if vSi_idx >= 0 else 0
            except (IndexError, ValueError):
                vSi = 0
            
            # Calculate weight
            weight = factorial(r) * factorial(m - r - 1) / factorial(m)
            term = weight * (vSi - vS)
            
            # Store detailed information about this term
            terms.append({
                "S": S,
                "S∪{i}": Si,
                "v(S)": vS,
                "v(S∪{i})": vSi,
                "weight": weight,
                "contribution": term
            })
            
            # Update running sum
            running_sum += term
    
    print(f"Final Shapley value: {running_sum:.6f}")
    
    # Sort terms by absolute contribution
    sorted_terms = sorted(terms, key=lambda x: abs(x["contribution"]), reverse=True)
    
    # Show top contributing terms
    print("\nTop 5 contributing terms:")
    for i, term in enumerate(sorted_terms[:5]):
        print(f"  {i+1}. S={term['S']}: weight={term['weight']:.6f}, v(S∪{i})={term['v(S∪{i})']:.6f}, v(S)={term['v(S)']:.6f}, contribution={term['contribution']:.6f}")
    
    # Also return the marginal value
    singleton = (feature_idx,)
    marginal_value = 0
    if singleton in coalition_to_index:
        marginal_value = v[coalition_to_index[singleton] + 1]
    
    print(f"\nMarginal value μ({feature_idx}): {marginal_value:.6f}")
    print(f"Overall interaction (φᵢ - μ(i)): {running_sum - marginal_value:.6f}")
    
    return {
        "shapley": running_sum,
        "marginal": marginal_value,
        "terms": terms,
        "overall_interaction": running_sum - marginal_value
    }

def trace_interaction_row(feature_idx, v, all_coalitions, m):
    """
    Trace the calculation of all interaction indices for a specific feature.
    
    The Shapley interaction index formula is:
    I({i,j}) = ∑_{S⊆N\{i,j}} [(n-|S|-2)!|S|!/(n-1)!] * [v(S∪{i,j}) - v(S∪{i}) - v(S∪{j}) + v(S)]
    
    Parameters:
    -----------
    feature_idx : int
        Index of the feature to trace
    v : array-like
        Capacity values with v[0] for empty set
    all_coalitions : list of tuples
        List of all coalitions
    m : int
        Number of features
        
    Returns:
    --------
    dict : Detailed breakdown of the calculation
    """
    print(f"\n=== TRACING INTERACTION CALCULATIONS FOR FEATURE {feature_idx} ===")
    
    interaction_row = np.zeros(m)
    coalition_to_index = {coal: idx for idx, coal in enumerate(all_coalitions)}
    
    def get_value(coalition):
        """Get coalition value with validation"""
        if not coalition:
            return 0.0  # Empty set
        
        coalition = tuple(sorted(coalition))
        if coalition in coalition_to_index:
            idx = coalition_to_index[coalition]
            if 0 <= idx < len(all_coalitions):
                return v[idx + 1]  # +1 for empty set
        return 0.0
    
    detailed_results = []
    
    # Calculate interaction with each other feature
    for other_idx in range(m):
        if other_idx == feature_idx:
            continue
        
        total_interaction = 0
        interaction_terms = []
        
        # Others = all features except feature_idx and other_idx
        others = [i for i in range(m) if i != feature_idx and i != other_idx]
        
        # Iterate through all subsets of others
        for r in range(len(others) + 1):
            for S in itertools.combinations(others, r):
                S = tuple(sorted(S))
                
                # Calculate coalitions
                S_tuple = tuple(sorted(S))
                Si_tuple = tuple(sorted(S + (feature_idx,)))
                Sj_tuple = tuple(sorted(S + (other_idx,)))
                Sij_tuple = tuple(sorted(S + (feature_idx, other_idx)))
                
                # Get coalition values
                vS = get_value(S_tuple)
                vSi = get_value(Si_tuple)
                vSj = get_value(Sj_tuple)
                vSij = get_value(Sij_tuple)
                
                # Calculate weight according to Shapley interaction formula
                weight = factorial(m - r - 2) * factorial(r) / factorial(m - 1)
                
                # Calculate term
                term = weight * (vSij - vSi - vSj + vS)
                
                # Store detailed information
                interaction_terms.append({
                    "S": S,
                    "v(S)": vS,
                    "v(S∪{i})": vSi,
                    "v(S∪{j})": vSj,
                    "v(S∪{i,j})": vSij,
                    "weight": weight,
                    "contribution": term
                })
                
                # Update running sum
                total_interaction += term
        
        interaction_row[other_idx] = total_interaction
        
        # Store detailed results for this pair
        detailed_results.append({
            "feature_j": other_idx,
            "interaction": total_interaction,
            "terms": interaction_terms
        })
    
    # Sum and report results
    row_sum = np.sum(interaction_row)
    matrix_method = 0.5 * row_sum
    
    print(f"Sum of interaction row: {row_sum:.6f}")
    print(f"Matrix method (0.5 * sum): {matrix_method:.6f}")
    
    # Show top interactions
    sorted_idx = np.argsort(np.abs(interaction_row))[::-1]
    print("\nTop interactions:")
    for i in range(min(5, m-1)):
        idx = sorted_idx[i]
        if idx != feature_idx:
            print(f"  With feature {idx}: {interaction_row[idx]:.6f}")
    
    # Show top contributing terms for the largest interaction
    largest_idx = sorted_idx[0] if sorted_idx[0] != feature_idx else sorted_idx[1]
    largest_detail = next(detail for detail in detailed_results if detail["feature_j"] == largest_idx)
    sorted_terms = sorted(largest_detail["terms"], key=lambda x: abs(x["contribution"]), reverse=True)
    
    print(f"\nDetailed breakdown of largest interaction (with feature {largest_idx}):")
    for i, term in enumerate(sorted_terms[:3]):
        print(f"  {i+1}. S={term['S']}: weight={term['weight']:.6f}, values={term['v(S∪{i,j})']:.6f}-{term['v(S∪{i})']:.6f}-{term['v(S∪{j})']:.6f}+{term['v(S)']:.6f}, contribution={term['contribution']:.6f}")
    
    return {
        "interaction_row": interaction_row,
        "matrix_method_value": matrix_method,
        "detailed_results": detailed_results
    }

def verify_mathematical_relationship(feature_idx, shapley_result, interaction_result, marginal_value):
    """
    Verify the mathematical relationship φᵢ = μ({i}) + 0.5 * ∑ⱼ≠ᵢ I({i,j})
    """
    shapley = shapley_result["shapley"]
    matrix_method = interaction_result["matrix_method_value"]
    
    left_side = shapley
    right_side = marginal_value + matrix_method
    difference = left_side - right_side
    
    print("\n=== MATHEMATICAL RELATIONSHIP VERIFICATION ===")
    print(f"Left side (φᵢ): {left_side:.6f}")
    print(f"Right side (μ({feature_idx}) + 0.5 * ∑ⱼ≠ᵢ I({{i,j}})): {right_side:.6f}")
    print(f"  = {marginal_value:.6f} + {matrix_method:.6f}")
    print(f"Difference: {difference:.6f}")
    
    # Try with negation
    right_side_neg = marginal_value - matrix_method
    print(f"\nWith negated interaction: μ({feature_idx}) - 0.5 * ∑ⱼ≠ᵢ I({{i,j}}) = {right_side_neg:.6f}")
    print(f"Difference with negated interaction: {left_side - right_side_neg:.6f}")
    
    return {
        "left_side": left_side,
        "right_side": right_side,
        "right_side_neg": right_side_neg,
        "difference": difference,
        "difference_neg": left_side - right_side_neg
    }

def inspect_all_features(v, all_coalitions, feature_names=None):
    """
    Inspect Shapley values, marginal values, and matrix method for all features
    """
    m = max(max(c) for c in all_coalitions if c) + 1
    
    # Calculate once for all features
    shapley_values = compute_shapley_values(v, m, all_coalitions)
    interaction_matrix = compute_choquet_interaction_matrix(v, m, all_coalitions)
    
    # Extract marginal values
    marginal_values = np.zeros(m)
    coalition_to_index = {coal: idx for idx, coal in enumerate(all_coalitions)}
    for i in range(m):
        singleton = (i,)
        if singleton in coalition_to_index:
            marginal_values[i] = v[coalition_to_index[singleton] + 1]
    
    # Calculate matrix method values
    matrix_method = 0.5 * np.sum(interaction_matrix, axis=1)
    matrix_method_neg = -0.5 * np.sum(interaction_matrix, axis=1)
    
    # Compute differences for all features
    diff = shapley_values - (marginal_values + matrix_method)
    diff_neg = shapley_values - (marginal_values - matrix_method)
    
    # Prepare results
    results = []
    for i in range(m):
        name = feature_names[i] if feature_names and i < len(feature_names) else f"Feature {i}"
        results.append({
            "index": i,
            "name": name,
            "shapley": shapley_values[i],
            "marginal": marginal_values[i],
            "matrix_method": matrix_method[i],
            "matrix_method_neg": -matrix_method[i],
            "diff": diff[i],
            "diff_neg": diff_neg[i],
            "overall_interaction": shapley_values[i] - marginal_values[i]
        })
    
    # Sort by absolute difference
    results.sort(key=lambda x: abs(x["diff"]), reverse=True)
    
    # Print summary
    print("\n=== MATHEMATICAL RELATIONSHIP FOR ALL FEATURES ===")
    print(f"{'Feature':<15} {'Shapley':<10} {'Marginal':<10} {'Matrix':<10} {'Diff':<10} {'NegMatrix':<10} {'NegDiff':<10}")
    print("-" * 75)
    
    for r in results[:10]:  # Show top 10
        print(f"{r['name'][:15]:<15} {r['shapley']:<10.6f} {r['marginal']:<10.6f} {r['matrix_method']:<10.6f} {r['diff']:<10.6f} {r['matrix_method_neg']:<10.6f} {r['diff_neg']:<10.6f}")
    
    # Plot two comparisons in a clear way
    plt.figure(figsize=(15, 6))
    
    # Create x-tick labels
    if feature_names:
        labels = [name[:15] for name in feature_names]  # Truncate long names
    else:
        labels = [f"F{i}" for i in range(m)]
    
    # Plot overall interaction (ground truth)
    overall_interaction = shapley_values - marginal_values
    
    # Compare Matrix Methods vs Ground Truth
    x = np.arange(m)
    width = 0.25
    
    plt.subplot(1, 2, 1)
    plt.bar(x - width, overall_interaction, width, label='Ground Truth (φ-μ)', color='black')
    plt.bar(x, matrix_method, width, label='Matrix Method', color='blue', alpha=0.7)
    plt.bar(x + width, matrix_method_neg, width, label='Negated Matrix Method', color='red', alpha=0.7)
    plt.xticks(x, labels, rotation=90, fontsize=8)
    plt.ylabel('Interaction Value')
    plt.title('Comparison of Interaction Methods')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot absolute differences
    plt.subplot(1, 2, 2)
    plt.bar(x - width/2, np.abs(diff), width, label='|Normal Diff|', color='blue', alpha=0.7)
    plt.bar(x + width/2, np.abs(diff_neg), width, label='|Negated Diff|', color='red', alpha=0.7)
    plt.xticks(x, labels, rotation=90, fontsize=8)
    plt.ylabel('Absolute Difference')
    plt.yscale('log')  # Log scale for better visibility
    plt.title('Absolute Differences')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('interaction_comparison.png')
    plt.close()
    
    print(f"\nPlot saved to 'interaction_comparison.png'")
    
    # Return sorted results
    return results

def examine_shapley_values_directly(v, all_coalitions, test_coalition=(0, 1)):
    """
    Directly examine and compare Shapley value computation approaches to verify 
    if we're comparing apples to oranges.
    
    Parameters:
    -----------
    v : array-like
        Capacity values with v[0] for empty set
    all_coalitions : list of tuples
        List of all coalitions
    test_coalition : tuple
        Test coalition to verify interaction value calculation
    """
    m = max(max(c) for c in all_coalitions if c) + 1
    print(f"\n=== DIRECT INSPECTION OF SHAPLEY VALUES AND INTERACTIONS (m={m}) ===")
    
    # Verify Shapley values using both our existing function and a manual calculation
    shapley_values = compute_shapley_values(v, m, all_coalitions)
    
    # Select a test feature and calculate manually
    test_feature = 0
    print(f"Testing Shapley calculation for feature {test_feature}")
    
    # Calculate manually for the test feature
    coalition_to_index = {coal: idx for idx, coal in enumerate(all_coalitions)}
    manual_shapley = 0
    others = [i for i in range(m) if i != test_feature]
    
    for r in range(len(others) + 1):
        for S in itertools.combinations(others, r):
            S = tuple(sorted(S))
            Si = tuple(sorted(S + (test_feature,)))
            
            # Try to get coalition values
            vS = 0
            if S in coalition_to_index:
                vS = v[coalition_to_index[S] + 1]
                
            vSi = 0
            if Si in coalition_to_index:
                vSi = v[coalition_to_index[Si] + 1]
            
            weight = factorial(r) * factorial(m - r - 1) / factorial(m)
            manual_shapley += weight * (vSi - vS)
    
    print(f"  Computed Shapley value: {shapley_values[test_feature]:.6f}")
    print(f"  Manual Shapley value: {manual_shapley:.6f}")
    print(f"  Difference: {shapley_values[test_feature] - manual_shapley:.6f}")
    
    # Test interaction calculation
    i, j = test_coalition
    print(f"\nTesting interaction calculation for pair ({i}, {j})")
    
    # Compute interaction matrix
    interaction_matrix = compute_choquet_interaction_matrix(v, m, all_coalitions)
    
    # Calculate manually
    manual_interaction = 0
    others = [k for k in range(m) if k != i and k != j]
    
    for r in range(len(others) + 1):
        for S in itertools.combinations(others, r):
            S = tuple(sorted(S))
            Si = tuple(sorted(S + (i,)))
            Sj = tuple(sorted(S + (j,)))
            Sij = tuple(sorted(S + (i, j)))
            
            # Try to get coalition values
            vS = vSi = vSj = vSij = 0
            if S in coalition_to_index:
                vS = v[coalition_to_index[S] + 1]
            if Si in coalition_to_index:
                vSi = v[coalition_to_index[Si] + 1]
            if Sj in coalition_to_index:
                vSj = v[coalition_to_index[Sj] + 1]
            if Sij in coalition_to_index:
                vSij = v[coalition_to_index[Sij] + 1]
            
            weight = factorial(m - r - 2) * factorial(r) / factorial(m - 1)
            manual_interaction += weight * (vSij - vSi - vSj + vS)
    
    print(f"  Computed interaction: {interaction_matrix[i, j]:.6f}")
    print(f"  Manual interaction: {manual_interaction:.6f}")
    print(f"  Difference: {interaction_matrix[i, j] - manual_interaction:.6f}")
    
    # Check a specific feature's overall interaction contribution
    feature_idx = 0
    shapley = shapley_values[feature_idx]
    
    # Get marginal value
    singleton = (feature_idx,)
    marginal = 0
    if singleton in coalition_to_index:
        marginal = v[coalition_to_index[singleton] + 1]
    
    # Get matrix method value (both regular and negated)
    matrix_method = 0.5 * np.sum(interaction_matrix[feature_idx])
    matrix_method_neg = -0.5 * np.sum(interaction_matrix[feature_idx])
    
    print(f"\nOverall interaction verification for feature {feature_idx}:")
    print(f"  Shapley value: {shapley:.6f}")
    print(f"  Marginal value: {marginal:.6f}")
    print(f"  Shapley - Marginal (ground truth): {shapley - marginal:.6f}")
    print(f"  0.5 * ∑ I(i,j): {matrix_method:.6f}")
    print(f"  -0.5 * ∑ I(i,j): {matrix_method_neg:.6f}")
    
    # Check which version is closer
    diff1 = abs((shapley - marginal) - matrix_method)
    diff2 = abs((shapley - marginal) - matrix_method_neg)
    
    print(f"  Difference with regular: {diff1:.6f}")
    print(f"  Difference with negated: {diff2:.6f}")
    print(f"  Better match: {'NEGATED' if diff2 < diff1 else 'REGULAR'}")
    
    return {
        "shapley_test": {
            "computed": shapley_values[test_feature],
            "manual": manual_shapley,
            "diff": shapley_values[test_feature] - manual_shapley
        },
        "interaction_test": {
            "computed": interaction_matrix[i, j],
            "manual": manual_interaction,
            "diff": interaction_matrix[i, j] - manual_interaction
        },
        "overall_test": {
            "shapley": shapley,
            "marginal": marginal,
            "ground_truth": shapley - marginal,
            "matrix_method": matrix_method,
            "matrix_method_neg": matrix_method_neg,
            "diff1": diff1,
            "diff2": diff2,
            "better_match": "NEGATED" if diff2 < diff1 else "REGULAR"
        }
    }

if __name__ == "__main__":
    print("Advanced Choquet Model Diagnostics")
    
    # Load debug data from most likely locations
    debug_file_paths = [
        'model_debug.pkl',  # Current directory
        os.path.join('plots', 'dados_covid_sbpo_atual', 'model_debug.pkl')  # Dataset folder
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
        print("No debug data found. Exiting.")
        exit(1)
    
    # Extract components
    v = debug_data.get('v')
    all_coalitions = debug_data.get('all_coalitions')
    feature_names = debug_data.get('feature_names', None)
    
    # First, check all features to identify problematic ones
    print("\n=========================================")
    print("EXAMINING ALL FEATURES")
    print("=========================================")
    results = inspect_all_features(v, all_coalitions, feature_names)
    
    # Find the most problematic feature (largest difference)
    worst_feature = results[0]["index"]
    worst_feature_name = results[0]["name"]
    
    print(f"\nWorst feature is {worst_feature_name} (index {worst_feature})")
    
    # Then do deep analysis on the most problematic feature
    print("\n=========================================")
    print(f"DETAILED ANALYSIS FOR FEATURE {worst_feature}")
    print("=========================================")
    
    # First, directly examine Shapley values and interaction formula implementation
    print("\n---------------------------------------")
    print("VERIFYING IMPLEMENTATION CORRECTNESS")
    print("---------------------------------------")
    implementation_check = examine_shapley_values_directly(v, all_coalitions)
    
    # Trace Shapley value calculation for the worst feature
    print("\n---------------------------------------")
    print("TRACING SHAPLEY CALCULATION")
    print("---------------------------------------")
    shapley_result = trace_shapley_calculation(worst_feature, v, all_coalitions, len(feature_names))
    
    # Trace interaction calculations for the worst feature
    print("\n---------------------------------------")
    print("TRACING INTERACTION CALCULATIONS")
    print("---------------------------------------")
    interaction_result = trace_interaction_row(worst_feature, v, all_coalitions, len(feature_names))
    
    # Verify mathematical relationship
    print("\n---------------------------------------")
    print("VERIFYING MATHEMATICAL RELATIONSHIP")
    print("---------------------------------------")
    verify_result = verify_mathematical_relationship(
        worst_feature, 
        shapley_result, 
        interaction_result,
        shapley_result["marginal"]
    )
    
    # Provide a conclusion based on all tests
    print("\n=========================================")
    print("DIAGNOSTIC CONCLUSION")
    print("=========================================")
    
    if abs(verify_result["difference"]) < 0.001:
        print("✅ The mathematical relationship holds with standard formula!")
    elif abs(verify_result["difference_neg"]) < 0.001:
        print("✅ The mathematical relationship holds with NEGATED interaction values!")
        print("To fix: Use negative_matrix = -interaction_matrix when computing overall interaction")
    else:
        print("❌ Neither the standard nor negated formulation matches the expected relationship.")
        print("There may be a fundamental issue with the calculation methods.")
    
    # Check if both formulas are consistently implemented
    if abs(implementation_check["shapley_test"]["diff"]) < 0.001 and abs(implementation_check["interaction_test"]["diff"]) < 0.001:
        print("✅ Implementation check: Both Shapley and Interaction formulas are correctly implemented!")
    else:
        print("❌ Implementation check: There are discrepancies in the implementation of formulas!")
        if abs(implementation_check["shapley_test"]["diff"]) >= 0.001:
            print("   - Issue detected in Shapley value calculation")
        if abs(implementation_check["interaction_test"]["diff"]) >= 0.001:
            print("   - Issue detected in Interaction index calculation")
    
    # Print the correct formula based on our analysis
    if implementation_check["overall_test"]["better_match"] == "NEGATED":
        print("\nCorrect formula appears to be:")
        print("φⱼ = μ({j}) - 0.5 * Σᵢ≠ⱼ I({i,j})")
    else:
        print("\nCorrect formula appears to be:")
        print("φⱼ = μ({j}) + 0.5 * Σᵢ≠ⱼ I({i,j})")
