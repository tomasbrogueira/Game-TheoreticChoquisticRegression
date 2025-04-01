"""
Simple test to analyze which coalitions are considered by different Choquet model implementations
and whether they follow k-additivity theory.
"""
import numpy as np
from itertools import combinations
import pandas as pd
from math import comb
from choquet_function import choquet_matrix_2add
from paper_code.covid_comprehensive_test import refined_choquet_k_additive

def create_simple_patterns(n_features=4):
    """Create simple input patterns to test coalition behavior"""
    # Create basic test patterns that will help identify which coalitions are activated
    patterns = []
    pattern_names = []
    
    # Pattern 1: All zeros (baseline)
    patterns.append(np.zeros(n_features))
    pattern_names.append("All zeros")
    
    # Pattern 2: All ones
    patterns.append(np.ones(n_features))
    pattern_names.append("All ones")
    
    # Pattern 3-6: One-hot patterns (one feature active at a time)
    for i in range(n_features):
        pattern = np.zeros(n_features)
        pattern[i] = 1.0
        patterns.append(pattern)
        pattern_names.append(f"Feature {i} only")
    
    # Pattern 7-12: Two-feature patterns
    for i in range(n_features):
        for j in range(i+1, n_features):
            pattern = np.zeros(n_features)
            pattern[i] = 1.0
            pattern[j] = 1.0
            patterns.append(pattern)
            pattern_names.append(f"Features {i} & {j}")
    
    # Pattern 13: Ascending values
    patterns.append(np.linspace(0.1, 0.9, n_features))
    pattern_names.append("Ascending values")
    
    # Pattern 14: Descending values
    patterns.append(np.linspace(0.9, 0.1, n_features))
    pattern_names.append("Descending values")
    
    return np.array(patterns), pattern_names

def identify_coalitions(transformation_matrix, k_max, n_features, patterns):
    """
    Identify which coalitions correspond to which columns in the transformed matrix.
    This is done by analyzing the pattern of activations across different input patterns.
    """
    coalitions = {}
    
    # Create a pattern where only one feature is active at a time to identify singleton coalitions
    singleton_patterns = np.zeros((n_features, n_features))
    for i in range(n_features):
        singleton_patterns[i, i] = 1.0
    
    # For each column in the transformation matrix
    for col in range(transformation_matrix.shape[1]):
        # Check which patterns activate this column
        activations = []
        for i, pattern in enumerate(patterns):
            if abs(transformation_matrix[i, col]) > 1e-10:
                activations.append(i)
        
        # Try to identify the coalition based on activation patterns
        if len(activations) == 0:
            coalitions[col] = "Empty (never activated)"
        elif all(np.all(patterns[i] == 1.0) for i in activations):
            coalitions[col] = "Grand coalition (all features)"
        else:
            # Check if this could be a singleton coalition
            is_singleton = False
            for i in range(n_features):
                pattern_idx = patterns.tolist().index(singleton_patterns[i].tolist()) if singleton_patterns[i].tolist() in patterns.tolist() else -1
                if pattern_idx != -1 and pattern_idx in activations:
                    coalitions[col] = f"Singleton: Feature {i}"
                    is_singleton = True
                    break
            
            if not is_singleton:
                # Attempt to identify based on pattern of activations
                features_list = []
                for feat_idx in range(n_features):
                    # If this column activates whenever feature feat_idx is active
                    if all(patterns[act][feat_idx] > 0 for act in activations):
                        features_list.append(feat_idx)
                
                if features_list:
                    coalitions[col] = f"Coalition: Features {features_list}"
                else:
                    coalitions[col] = f"Unidentified coalition"
    
    return coalitions

def analyze_coalitions(k_values=[1, 2, 3, 4], verbose=False):
    """Analyze which coalitions are considered by each implementation with different k values"""
    n_features = 4  # Use 4 features for simplicity
    patterns, pattern_names = create_simple_patterns(n_features)
    
    print(f"=== Coalition Analysis with {n_features} features ===")
    print(f"Created {len(patterns)} test patterns")
    
    # For each k, show how many coalitions are theoretically expected
    for k in k_values:
        expected_count = sum(comb(n_features, i) for i in range(1, k+1))
        print(f"k={k}: Expect {expected_count} coalitions theoretically")
        if verbose:
            # List the expected coalition sizes
            for size in range(1, k+1):
                num_coalitions = comb(n_features, size)
                print(f"   - Size {size}: {num_coalitions} coalitions")
    
    # Define the implementations to test
    implementations = [
        ("Refined", lambda X, k: refined_choquet_k_additive(X, k_add=k)),
        ("Shapley", lambda X, k: choquet_matrix_2add(X))
    ]
    
    results = []
    
    for name, implementation in implementations:
        print(f"\nAnalyzing {name} implementation:")
        
        for k in k_values:
            # Skip higher k values for Shapley since it's fixed at k=2
            if name == "Shapley" and k != 2:
                continue
                
            print(f"  With k={k}:")
            
            # Transform all patterns at once
            transformed = implementation(patterns, k)
            
            # Identify which columns correspond to which coalitions
            coalitions = identify_coalitions(transformed, k, n_features, patterns)
            
            # Count coalition sizes
            coalition_sizes = {}
            for col, coalition_desc in coalitions.items():
                size = get_coalition_size(coalition_desc, n_features)
                if size not in coalition_sizes:
                    coalition_sizes[size] = 0
                coalition_sizes[size] += 1
            
            # Print coalition size distribution (simplified)
            print(f"    Coalition sizes: ", end="")
            sizes = []
            for size in sorted(coalition_sizes.keys(), key=lambda x: (isinstance(x, str), x)):
                count = coalition_sizes[size]
                sizes.append(f"Size {size}: {count}")
            print(", ".join(sizes))
            
            if verbose:
                # For a sample of columns, print the identified coalition
                print(f"    Sample of identified coalitions:")
                sample_size = min(3, transformed.shape[1])
                for col in range(sample_size):
                    print(f"      Column {col}: {coalitions[col]}")
            
            # For each pattern, store results but only print summaries
            active_patterns_count = 0
            for i, pattern in enumerate(patterns):
                # Find non-zero indices (activated coalitions)
                non_zero = np.where(np.abs(transformed[i]) > 1e-10)[0]
                
                if len(non_zero) > 0:
                    active_patterns_count += 1
                
                # Create a result entry
                result = {
                    'Implementation': name,
                    'k': k,
                    'Pattern': pattern_names[i],
                    'Input': pattern,
                    'Activated Columns': non_zero.tolist(),
                    'Values': transformed[i, non_zero].tolist() if len(non_zero) > 0 else [],
                    'Num Activated': len(non_zero),
                    'Matrix Shape': transformed.shape,
                    'Coalitions': {col: coalitions[col] for col in non_zero}
                }
                
                results.append(result)
                
                # Only print details for a few interesting patterns if verbose
                if verbose and (i < 2 or i == pattern_idx_for_feature_0_only):
                    print(f"    {pattern_names[i]}: {len(non_zero)} columns activated")
                    if len(non_zero) > 0 and len(non_zero) <= 3:
                        # If fewer than 3 columns activated, show details
                        for idx, val in zip(non_zero, transformed[i, non_zero]):
                            print(f"      Column {idx} ({coalitions[idx]}): {val:.4f}")
            
            # Print a summary of how many patterns activated coalitions
            print(f"    {active_patterns_count}/{len(patterns)} patterns activated coalitions")
            
            # Overall statistics for this implementation and k value
            active_cols = set()
            for i in range(len(patterns)):
                active_cols.update(np.where(np.abs(transformed[i]) > 1e-10)[0])
            
            # Check if active columns make sense for k-additivity
            expected_count = sum(comb(n_features, i) for i in range(1, k+1))
            consistency = "✓" if len(active_cols) <= expected_count else "✗"
            
            print(f"  Summary for k={k}:")
            print(f"    Matrix shape: {transformed.shape}")
            print(f"    Active columns: {len(active_cols)}/{transformed.shape[1]} (Expected: {expected_count})")
            print(f"    Consistent with k-additivity: {consistency}")
    
    return results

def check_k_additivity_consistency(results, verbose=False):
    """Check if the implementations follow k-additivity theory"""
    print("\n=== K-Additivity Consistency Check ===")
    
    # Group by implementation and k value
    grouped = {}
    for r in results:
        key = (r['Implementation'], r['k'])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(r)
    
    # For each implementation and k combination
    for (impl, k), group in sorted(grouped.items()):
        # Skip Shapley with k!=2
        if impl == "Shapley" and k != 2:
            continue
            
        # Get the matrix shape (should be same for all patterns)
        matrix_shape = group[0]['Matrix Shape']
        n_features = len(group[0]['Input'])
        
        # Expected number of coalitions based on k-additivity theory
        expected_cols = sum(comb(n_features, i) for i in range(1, k+1))
        
        # Find unique activated columns across all patterns
        all_activated = set()
        coalitions_used = {}
        for r in group:
            all_activated.update(r['Activated Columns'])
            # Collect coalition information
            coalitions_used.update(r.get('Coalitions', {}))
        
        print(f"\n{impl} with k={k}:")
        print(f"  Matrix: {matrix_shape[1]} columns, {len(all_activated)} activated (Expected: {expected_cols})")
        
        # Count coalition sizes (simplified output)
        coalition_sizes = {}
        allowed_coalitions = 0
        disallowed_coalitions = 0
        
        for col, coalition_desc in coalitions_used.items():
            size = get_coalition_size(coalition_desc, n_features)
            if size not in coalition_sizes:
                coalition_sizes[size] = 0
            coalition_sizes[size] += 1
            
            # Count if allowed by k-additivity
            if size != "Unknown":
                if size <= k:
                    allowed_coalitions += 1
                else:
                    disallowed_coalitions += 1
        
        # Print simplified coalition size distribution
        print(f"  Coalition sizes: ", end="")
        sizes = []
        for size in sorted(coalition_sizes.keys(), key=lambda x: (isinstance(x, str), x)):
            count = coalition_sizes[size]
            sizes.append(f"Size {size}: {count}")
        print(", ".join(sizes))
        
        # Print allowed vs disallowed coalitions
        if disallowed_coalitions > 0:
            print(f"  Warning: {disallowed_coalitions} coalitions exceed k-additivity limit")
        else:
            print(f"  All identified coalitions respect k-additivity limit")
        
        # Basic consistency check
        if len(all_activated) <= expected_cols and matrix_shape[1] >= expected_cols:
            consistency = "Consistent"
        else:
            consistency = "Not consistent"
            
        print(f"  Consistency with k-additivity: {consistency}")
        
        # Special analysis for detecting missing or extra coalitions
        if len(all_activated) < expected_cols:
            print(f"  Missing {expected_cols - len(all_activated)} expected coalitions")
        elif len(all_activated) > expected_cols:
            print(f"  Has {len(all_activated) - expected_cols} extra coalitions")
        
        if verbose:    
            # Sample of identified coalitions (reduced)
            print(f"  Sample of identified coalitions:")
            sample_cols = list(all_activated)[:min(3, len(all_activated))]
            for col in sample_cols:
                print(f"    Column {col}: {coalitions_used.get(col, 'Unknown')}")
            
            # Patterns with most and least activations
            activations = [(r['Pattern'], len(r['Activated Columns'])) for r in group]
            max_pattern = max(activations, key=lambda x: x[1])
            min_pattern = min(activations, key=lambda x: x[1])
            
            print(f"  Pattern with most activations: {max_pattern[0]} ({max_pattern[1]} columns)")
            print(f"  Pattern with least activations: {min_pattern[0]} ({min_pattern[1]} columns)")

def analyze_specific_pattern(pattern_name="Feature 0 only", k_values=[1, 2, 3], verbose=True):
    """Analyze a specific pattern across implementations and k values"""
    n_features = 4
    patterns, pattern_names = create_simple_patterns(n_features)
    
    # Find the target pattern
    pattern_idx = pattern_names.index(pattern_name) if pattern_name in pattern_names else 0
    pattern = patterns[pattern_idx]
    
    print(f"\n=== Analysis of Pattern: {pattern_name} ===")
    
    implementations = [
        ("Refined", lambda X, k: refined_choquet_k_additive(X, k_add=k)),
        ("Shapley", lambda X, k: choquet_matrix_2add(X))
    ]
    
    for name, implementation in implementations:
        print(f"\n{name} Implementation:")
        
        for k in k_values:
            # Skip higher k values for Shapley since it's fixed at k=2
            if name == "Shapley" and k != 2:
                continue
                
            # Transform the single pattern
            transformed = implementation(pattern.reshape(1, -1), k)
            
            # Find non-zero values
            non_zero = np.where(np.abs(transformed[0]) > 1e-10)[0]
            
            # Use a simplified approach to identify coalitions
            # Create a dictionary to store column info
            coalitions = {}
            for col in non_zero:
                # Try to identify the coalition based on the activation
                if np.array_equal(pattern, np.ones(n_features)):
                    coalitions[col] = "Grand coalition (all features)"
                elif sum(pattern > 0) == 1:
                    # This is a one-hot pattern - identify which feature
                    feature_idx = np.where(pattern > 0)[0][0]
                    coalitions[col] = f"Singleton: Feature {feature_idx}"
                elif sum(pattern > 0) == 2:
                    # This is a two-feature pattern
                    feature_idxs = np.where(pattern > 0)[0]
                    coalitions[col] = f"Coalition: Features {feature_idxs.tolist()}"
                else:
                    # For more complex patterns, just note the active features
                    active_feats = np.where(pattern > 0)[0]
                    coalitions[col] = f"Coalition with active features: {active_feats.tolist()}"
            
            print(f"  k={k}: {len(non_zero)}/{transformed.shape[1]} columns activated")
            
            if len(non_zero) > 0:
                if verbose:
                    print("    Values and coalitions:")
                    # Show at most 5 columns
                    for idx, val in zip(non_zero[:5], transformed[0, non_zero[:5]]):
                        print(f"      Column {idx} ({coalitions[idx]}): {val:.4f}")
                    if len(non_zero) > 5:
                        print(f"      ... and {len(non_zero) - 5} more columns")
            else:
                print("    No columns activated")

def compare_k2_implementations():
    """
    Specifically compare the coalitions used in the 2-additive implementations:
    1. Refined with k=2
    2. Direct Shapley 2-additive matrix
    """
    n_features = 4
    patterns, pattern_names = create_simple_patterns(n_features)
    
    print("\n=== Detailed Comparison of 2-Additive Implementations ===")
    
    # Get transformations for all patterns
    refined_k2 = refined_choquet_k_additive(patterns, k_add=2)
    shapley_2add = choquet_matrix_2add(patterns)
    
    # Identify coalitions for each implementation
    refined_coalitions = identify_coalitions(refined_k2, 2, n_features, patterns)
    shapley_coalitions = identify_coalitions(shapley_2add, 2, n_features, patterns)
    
    # Print matrix shapes
    print(f"Refined (k=2) matrix shape: {refined_k2.shape}")
    print(f"Shapley 2-additive matrix shape: {shapley_2add.shape}")
    
    # Compare number of expected coalitions for k=2
    expected_count = sum(comb(n_features, i) for i in range(1, 2+1))
    print(f"Expected coalitions for k=2: {expected_count}")
    
    # Count coalition sizes for each implementation
    refined_sizes = {}
    shapley_sizes = {}
    
    for col, desc in refined_coalitions.items():
        size = get_coalition_size(desc, n_features)
        if size not in refined_sizes:
            refined_sizes[size] = 0
        refined_sizes[size] += 1
    
    for col, desc in shapley_coalitions.items():
        size = get_coalition_size(desc, n_features)
        if size not in shapley_sizes:
            shapley_sizes[size] = 0
        shapley_sizes[size] += 1
    
    # Print size distributions
    print("\nCoalition size distribution:")
    print("  Refined (k=2):", end=" ")
    sizes = []
    for size in sorted(refined_sizes.keys(), key=lambda x: (isinstance(x, str), x)):
        count = refined_sizes[size]
        sizes.append(f"Size {size}: {count}")
    print(", ".join(sizes))
    
    print("  Shapley:", end=" ")
    sizes = []
    for size in sorted(shapley_sizes.keys(), key=lambda x: (isinstance(x, str), x)):
        count = shapley_sizes[size]
        sizes.append(f"Size {size}: {count}")
    print(", ".join(sizes))
    
    # Compare which patterns activate which columns
    print("\nPattern activation comparison:")
    for i, pattern_name in enumerate(pattern_names):
        # Which columns are activated by this pattern in each implementation
        refined_cols = np.where(np.abs(refined_k2[i]) > 1e-10)[0]
        shapley_cols = np.where(np.abs(shapley_2add[i]) > 1e-10)[0]
        
        print(f"\n  Pattern: {pattern_name}")
        print(f"    Refined activates {len(refined_cols)} columns, Shapley activates {len(shapley_cols)} columns")
        
        # If important pattern, show more details
        if "Feature" in pattern_name or pattern_name == "All ones":
            print("    Refined coalitions activated:")
            for col in refined_cols[:3]:  # Show first 3 for brevity
                value = refined_k2[i, col]
                print(f"      Column {col} ({refined_coalitions[col]}): {value:.4f}")
            if len(refined_cols) > 3:
                print(f"      ... and {len(refined_cols) - 3} more")
                
            print("    Shapley coalitions activated:")
            for col in shapley_cols[:3]:  # Show first 3 for brevity
                value = shapley_2add[i, col]
                print(f"      Column {col} ({shapley_coalitions[col]}): {value:.4f}")
            if len(shapley_cols) > 3:
                print(f"      ... and {len(shapley_cols) - 3} more")
    
    # Check if the Shapley implementation is correctly using 2-additive coalitions
    print("\nCoalition consistency check:")
    print("  Theoretical expectation for k=2:")
    print(f"    Size 1 (singletons): {comb(n_features, 1)}")
    print(f"    Size 2 (pairs): {comb(n_features, 2)}")
    
    # Check singleton coalitions
    singleton_pattern_indices = [pattern_names.index(f"Feature {i} only") for i in range(n_features)]
    
    print("\n  Testing singleton patterns:")
    for i in singleton_pattern_indices:
        feature_idx = int(pattern_names[i].split(" ")[1])
        pattern_name = pattern_names[i]
        
        refined_cols = np.where(np.abs(refined_k2[i]) > 1e-10)[0]
        shapley_cols = np.where(np.abs(shapley_2add[i]) > 1e-10)[0]
        
        print(f"    {pattern_name}:")
        print(f"      Refined: {len(refined_cols)} columns activated")
        print(f"      Shapley: {len(shapley_cols)} columns activated")
        
        # Check if the right singleton is included
        refined_has_singleton = any("Singleton: Feature " + str(feature_idx) in refined_coalitions.get(col, "") for col in refined_cols)
        shapley_has_singleton = any("Singleton: Feature " + str(feature_idx) in shapley_coalitions.get(col, "") for col in shapley_cols)
        
        print(f"      Refined correctly includes singleton: {'Yes' if refined_has_singleton else 'No'}")
        print(f"      Shapley correctly includes singleton: {'Yes' if shapley_has_singleton else 'No'}")
    
    # Check pair coalitions
    print("\n  Testing pair patterns:")
    pair_patterns = [i for i, name in enumerate(pattern_names) if " & " in name]
    
    for i in pair_patterns[:2]:  # Just check a couple for brevity
        pattern_name = pattern_names[i]
        feature_indices = [int(idx) for idx in pattern_name.split("Features ")[1].split(" & ")]
        
        refined_cols = np.where(np.abs(refined_k2[i]) > 1e-10)[0]
        shapley_cols = np.where(np.abs(shapley_2add[i]) > 1e-10)[0]
        
        print(f"    {pattern_name}:")
        print(f"      Refined: {len(refined_cols)} columns activated")
        print(f"      Shapley: {len(shapley_cols)} columns activated")
        
        # Check if the right pair coalition is included
        pair_coalition_desc = str(feature_indices)
        refined_has_pair = any(pair_coalition_desc in refined_coalitions.get(col, "") for col in refined_cols)
        shapley_has_pair = any(pair_coalition_desc in shapley_coalitions.get(col, "") for col in shapley_cols)
        
        print(f"      Refined correctly includes pair coalition: {'Yes' if refined_has_pair else 'No'}")
        print(f"      Shapley correctly includes pair coalition: {'Yes' if shapley_has_pair else 'No'}")
    
    # Final analysis on which implementation appears to be using the right coalitions
    print("\nFinal coalition analysis:")
    refined_ok = len(refined_cols) <= expected_count and refined_sizes.get(1, 0) == comb(n_features, 1) and refined_sizes.get(2, 0) == comb(n_features, 2)
    shapley_ok = len(shapley_cols) <= expected_count and shapley_sizes.get(1, 0) == comb(n_features, 1) and shapley_sizes.get(2, 0) == comb(n_features, 2)
    
    print(f"  Refined k=2 appears to use correct 2-additive coalitions: {'Yes' if refined_ok else 'No'}")
    print(f"  Shapley implementation appears to use correct 2-additive coalitions: {'Yes' if shapley_ok else 'No'}")

def get_coalition_size(coalition_desc, n_features):
    """Helper function to extract coalition size from description"""
    if "Singleton" in coalition_desc:
        return 1
    elif "Features" in coalition_desc:
        try:
            return len(eval(coalition_desc.split("Features ")[1]))
        except:
            return "Unknown"
    elif "Grand coalition" in coalition_desc:
        return n_features
    else:
        return "Unknown"

if __name__ == "__main__":
    # Find index for "Feature 0 only" pattern
    _, pattern_names = create_simple_patterns(4)
    pattern_idx_for_feature_0_only = pattern_names.index("Feature 0 only")
    
    # Run the analysis with varying k values (reduced verbosity)
    results = analyze_coalitions(k_values=[1, 2, 3, 4], verbose=False)
    
    # Check consistency with k-additivity theory (reduced verbosity)
    check_k_additivity_consistency(results, verbose=False)
    
    # Analyze specific interesting patterns
    analyze_specific_pattern("Feature 0 only", k_values=[1, 2, 3, 4], verbose=True)
    analyze_specific_pattern("Features 0 & 1", k_values=[1, 2, 3, 4], verbose=False)
    analyze_specific_pattern("All ones", k_values=[1, 2, 3, 4], verbose=False)
    analyze_specific_pattern("Ascending values", k_values=[1, 2, 3, 4], verbose=False)
    
    # Compare the 2-additive implementations in detail
    compare_k2_implementations()
