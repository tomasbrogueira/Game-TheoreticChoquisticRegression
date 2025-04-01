"""
Script to precisely identify which coalitions (feature combinations) are used in each implementation,
how they are activated, and how they correspond to the k-additive theory.
"""

import numpy as np
import pandas as pd
from itertools import combinations
from math import comb
from choquet_function import choquet_matrix_2add, choquet_matrix_2add_fixed
from paper_code.covid_comprehensive_test import refined_choquet_k_additive
from coalition_analysis_test import create_simple_patterns

def theoretical_coalitions(n_features, k_value):
    """Generate all theoretical coalitions up to size k for n_features"""
    coalitions = []
    # Generate all coalitions of sizes 1 to k
    for size in range(1, k_value + 1):
        for combo in combinations(range(n_features), size):
            coalitions.append({
                'features': combo,
                'size': len(combo),
                'description': f"Coalition {combo}"
            })
    return coalitions

def identify_column_coalitions(implementation_name, transform_func, k, patterns, pattern_names):
    """Identify which coalitions correspond to which columns in an implementation"""
    # Transform all patterns
    transformed = transform_func(patterns, k)
    print(f"\n{implementation_name} - Matrix shape: {transformed.shape}")
    
    # Identify columns by their activation patterns
    n_features = patterns.shape[1]
    n_columns = transformed.shape[1]
    
    # Step 1: Create feature patterns for identifying columns
    singleton_patterns = []  # Index of patterns with just one feature active
    pair_patterns = []       # Index of patterns with specific pairs active
    all_ones_pattern = np.where(np.all(patterns == 1, axis=1))[0][0]
    
    for i in range(n_features):
        # Find pattern with only feature i active
        for j, pattern in enumerate(patterns):
            if sum(pattern > 0) == 1 and pattern[i] > 0:
                singleton_patterns.append((i, j))
                break
    
    for i in range(n_features):
        for j in range(i+1, n_features):
            # Find pattern with only features i and j active
            for p, pattern in enumerate(patterns):
                if sum(pattern > 0) == 2 and pattern[i] > 0 and pattern[j] > 0:
                    pair_patterns.append(((i, j), p))
                    break
    
    # Step 2: Analyze each column based on its activation pattern
    column_coalitions = []
    
    for col in range(n_columns):
        # Check which patterns activate this column
        activations = np.where(np.abs(transformed[:, col]) > 1e-10)[0]
        
        # Check if this column is activated by any singleton pattern
        singleton_activations = []
        for feat, pat_idx in singleton_patterns:
            if pat_idx in activations:
                singleton_activations.append(feat)
        
        # Check if this column is activated by any pair pattern
        pair_activations = []
        for (feat1, feat2), pat_idx in pair_patterns:
            if pat_idx in activations:
                pair_activations.append((feat1, feat2))
        
        # Determine coalition type based on activation pattern
        if len(singleton_activations) == 1 and len(pair_activations) == 0:
            # Pure singleton coalition
            coalition_type = "Singleton"
            features = (singleton_activations[0],)
            size = 1
        elif len(pair_activations) == 1 and all_ones_pattern in activations:
            # Pure pair coalition
            coalition_type = "Pair"
            features = pair_activations[0]
            size = 2
        elif len(singleton_activations) > 1:
            # Interaction between singletons
            coalition_type = "Interaction"
            features = tuple(sorted(singleton_activations))
            size = len(features)
        else:
            # Complex or unidentified coalition
            coalition_type = "Complex"
            # Try to identify which features are involved
            features = tuple()
            for i, pattern in enumerate(patterns):
                if i in activations:
                    # This pattern activates the column
                    # Add all active features in this pattern
                    active_feats = np.where(pattern > 0)[0]
                    features = tuple(sorted(set(features).union(set(active_feats))))
            size = len(features)
        
        # Sample activation values to understand the column behavior
        activation_info = []
        for i in activations[:3]:  # First 3 activations only
            activation_info.append({
                'pattern': pattern_names[i],
                'value': transformed[i, col]
            })
        
        column_coalitions.append({
            'column': col,
            'type': coalition_type,
            'features': features,
            'size': size,
            'description': f"{coalition_type}: {features}",
            'num_activations': len(activations),
            'activation_samples': activation_info
        })
    
    return column_coalitions

def analyze_coalitions_usage():
    """Analyze which coalitions are used by each implementation"""
    # Create test patterns
    n_features = 4
    patterns, pattern_names = create_simple_patterns(n_features)
    
    # Generate all theoretical coalitions for k=2
    print(f"Theoretical coalitions for k=2 with {n_features} features:")
    theoretical_k2 = theoretical_coalitions(n_features, 2)
    print(f"Total: {len(theoretical_k2)} coalitions")
    
    # Print singleton coalitions
    singleton_coalitions = [c for c in theoretical_k2 if c['size'] == 1]
    print(f"\nSingleton coalitions ({len(singleton_coalitions)}):")
    for i, c in enumerate(singleton_coalitions):
        print(f"  {i+1}. {c['description']}")
    
    # Print pair coalitions
    pair_coalitions = [c for c in theoretical_k2 if c['size'] == 2]
    print(f"\nPair coalitions ({len(pair_coalitions)}):")
    for i, c in enumerate(pair_coalitions):
        print(f"  {i+1}. {c['description']}")
    
    # Define implementations to analyze
    implementations = [
        ("Refined", lambda X, k: refined_choquet_k_additive(X, k_add=k), 2),
        ("Shapley Original", lambda X, k: choquet_matrix_2add(X), 2),
        ("Shapley Fixed", lambda X, k: choquet_matrix_2add_fixed(X), 2)
    ]
    
    all_column_info = {}
    
    # Analyze each implementation
    for name, transform_func, k in implementations:
        column_coalitions = identify_column_coalitions(name, transform_func, k, patterns, pattern_names)
        all_column_info[name] = column_coalitions
        
        # Summarize the coalitions used in this implementation
        print(f"\n{name} Implementation - Total columns: {len(column_coalitions)}")
        
        # Group by coalition type and size
        by_type = {}
        for col_info in column_coalitions:
            col_type = col_info['type']
            if col_type not in by_type:
                by_type[col_type] = []
            by_type[col_type].append(col_info)
        
        # Print summary by type
        for coalition_type, cols in sorted(by_type.items()):
            print(f"  {coalition_type} coalitions: {len(cols)}")
            # Show first few of each type
            for i, col in enumerate(cols[:min(3, len(cols))]):
                activations = ', '.join([f"{a['pattern']}: {a['value']:.4f}" for a in col['activation_samples']])
                print(f"    Column {col['column']}: {col['description']} - [{activations}]")
            
            if len(cols) > 3:
                print(f"    ... and {len(cols) - 3} more {coalition_type} coalitions")
    
    # Create a table comparing implementations
    print("\n=== Coalition Usage Comparison ===")
    
    # Organize by theoretical coalition to see how each implementation represents it
    comparison_table = []
    for theory_coal in theoretical_k2:
        comparison_row = {
            'coalition': theory_coal['description'],
            'size': theory_coal['size']
        }
        theory_features = set(theory_coal['features'])
        
        # Check how each implementation represents this coalition
        for name in [impl[0] for impl in implementations]:
            columns = []
            for col_info in all_column_info[name]:
                if set(col_info['features']) == theory_features:
                    columns.append(f"Column {col_info['column']} ({col_info['type']})")
            
            comparison_row[name] = ', '.join(columns) if columns else 'Not represented'
        
        comparison_table.append(comparison_row)
    
    # Display the table
    df = pd.DataFrame(comparison_table)
    print("\nCoalition representation across implementations:")
    pd.set_option('display.max_colwidth', None)
    print(df)
    
    # Final analysis: which coalitions are utilized, which are missing
    print("\n=== Coalition Coverage Analysis ===")
    for name in [impl[0] for impl in implementations]:
        impl_columns = all_column_info[name]
        
        # Count how many theoretical coalitions are covered
        covered_count = 0
        missing_coalitions = []
        
        for theory_coal in theoretical_k2:
            theory_features = set(theory_coal['features'])
            is_covered = any(set(col_info['features']) == theory_features for col_info in impl_columns)
            
            if is_covered:
                covered_count += 1
            else:
                missing_coalitions.append(theory_coal['description'])
        
        print(f"\n{name} Implementation:")
        print(f"  Covers {covered_count}/{len(theoretical_k2)} theoretical coalitions")
        
        if missing_coalitions:
            print(f"  Missing coalitions:")
            for coal in missing_coalitions:
                print(f"    - {coal}")
        else:
            print("  All theoretical coalitions are represented")
        
        # Check for extra coalitions beyond the theoretical ones
        extra_coalitions = []
        for col_info in impl_columns:
            col_features = set(col_info['features'])
            if not any(set(theory['features']) == col_features for theory in theoretical_k2):
                extra_coalitions.append(f"Column {col_info['column']}: {col_info['description']}")
        
        if extra_coalitions:
            print(f"  Extra coalitions beyond theoretical k=2:")
            for coal in extra_coalitions:
                print(f"    - {coal}")

if __name__ == "__main__":
    analyze_coalitions_usage()
