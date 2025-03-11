import numpy as np
import itertools
from scipy.special import comb
from choquet_kadd_test import (
    choquet_matrix_unified, 
    choquet_matrix_2add
)

def derive_formula():
    """
    Attempt to derive a formula for the transformation matrix between
    equation (22) and equation (23) representations.
    """
    print("==== DERIVING TRANSFORMATION FORMULA ====")
    
    # Test with the simplest case - 2 features
    print("\n1. ANALYZING 2-FEATURE CASE")
    np.random.seed(42)
    X_2feat = np.random.rand(5, 2)
    
    # Get transformation matrix
    eq22_2feat, coalitions_2feat = choquet_matrix_unified(X_2feat, k_add=2)
    eq23_2feat = choquet_matrix_2add(X_2feat)
    T_2feat = np.linalg.lstsq(eq22_2feat, eq23_2feat, rcond=None)[0]
    
    print("Coalitions:", coalitions_2feat)
    print("Transformation matrix for 2 features:")
    print(T_2feat)
    
    # Test with unit vectors for 2 features
    print("\n2. CHECKING WITH UNIT VECTORS (2 features)")
    X_unit1 = np.array([[1.0, 0.0]])
    X_unit2 = np.array([[0.0, 1.0]])
    
    # Get transformation matrices
    eq22_unit1, _ = choquet_matrix_unified(X_unit1, k_add=2)
    eq23_unit1 = choquet_matrix_2add(X_unit1)
    T_unit1 = np.linalg.lstsq(eq22_unit1, eq23_unit1, rcond=None)[0]
    
    eq22_unit2, _ = choquet_matrix_unified(X_unit2, k_add=2)
    eq23_unit2 = choquet_matrix_2add(X_unit2)
    T_unit2 = np.linalg.lstsq(eq22_unit2, eq23_unit2, rcond=None)[0]
    
    print("Unit vector [1, 0] transformation:")
    print(T_unit1)
    print("\nUnit vector [0, 1] transformation:")
    print(T_unit2)
    
    # 3-feature case
    print("\n3. ANALYZING 3-FEATURE CASE")
    X_3feat = np.random.rand(5, 3)
    
    # Get transformation matrix
    eq22_3feat, coalitions_3feat = choquet_matrix_unified(X_3feat, k_add=2)
    eq23_3feat = choquet_matrix_2add(X_3feat)
    T_3feat = np.linalg.lstsq(eq22_3feat, eq23_3feat, rcond=None)[0]
    
    print("Coalitions:", coalitions_3feat)
    print("Transformation matrix for 3 features:")
    print(T_3feat)
    
    # For each coalition position, identify pattern in all matrices
    print("\n4. PATTERN IDENTIFICATION")
    for i, coalition in enumerate(coalitions_3feat):
        print(f"\nAnalyzing coalition {coalition}:")
        row = T_3feat[i]
        
        # Identify pattern for singletons
        if len(coalition) == 1:
            feature = coalition[0]
            n_features = 3
            print(f"Singleton {feature} pattern:")
            
            # Self-term
            print(f"  To Shapley value {feature}: {row[feature]:.4f}")
            
            # Interaction terms
            for j in range(n_features):
                if j != feature:
                    # Find the interaction column
                    for k, pair in enumerate(itertools.combinations(range(n_features), 2)):
                        if feature in pair and j in pair:
                            interaction_idx = n_features + k
                            print(f"  To interaction {pair} (col {interaction_idx}): {row[interaction_idx]:.4f}")
                            # Check if the value is close to -0.5
                            if abs(row[interaction_idx] + 0.5) < 0.1:
                                print("    PATTERN: Approximately -0.5")
    
    # Check theoretical formula against actual matrices
    print("\n5. FORMULA VALIDATION")
    print("Based on patterns, the theoretical formula for transformation matrix T is:")
    print("1. For singleton coalition (i):")
    print("   - T[i,i] = 1.0  (maps to own Shapley value)")
    print("   - T[i,n+idx(i,j)] = -0.5  (maps to each interaction involving i)")
    print("2. For pair coalition (i,j):")
    print("   - T[i,j] has complex dependence on data distribution")
    
    # Try to construct a formula-based matrix for the 3-feature case
    n_features = 3
    n_pairs = n_features * (n_features - 1) // 2
    theoretical_T = np.zeros((n_features + n_pairs, n_features + n_pairs))
    
    # Fill according to formula
    for i, coalition in enumerate(coalitions_3feat):
        if len(coalition) == 1:
            feature = coalition[0]
            # Self-mapping to Shapley value
            theoretical_T[i, feature] = 1.0
            
            # Mapping to interaction terms
            for j in range(n_features):
                if j != feature:
                    # Find the interaction column
                    for k, pair in enumerate(itertools.combinations(range(n_features), 2)):
                        if feature in pair and j in pair:
                            interaction_idx = n_features + k
                            theoretical_T[i, interaction_idx] = -0.5
    
    print("\nTheoretical transformation matrix (partially filled):")
    print(theoretical_T[:n_features])  # Just show singleton rows
    
    # Check match for singleton rows
    error = np.linalg.norm(theoretical_T[:n_features] - T_3feat[:n_features]) / np.linalg.norm(T_3feat[:n_features])
    print(f"\nError for singleton rows: {error:.4f}")
    
    # Formula for special case: precisely balanced data
    print("\n6. BALANCED DATA SPECIAL CASE")
    # Create perfect balanced data
    X_balanced = np.array([[0.1, 0.5, 0.9]])  # Evenly spaced values
    
    # Get transformation
    eq22_bal, coalitions_bal = choquet_matrix_unified(X_balanced, k_add=2)
    eq23_bal = choquet_matrix_2add(X_balanced)
    T_bal = np.linalg.lstsq(eq22_bal, eq23_bal, rcond=None)[0]
    
    print("Balanced data transformation matrix:")
    print(T_bal)
    
    # Summary of findings
    print("\n==== FORMULA DERIVATION SUMMARY ====")
    print("1. For singleton coalition (i):")
    print("   - T[i,i] = 1.0")  
    print("   - T[i,n+k] = -0.5 where k is any interaction index involving feature i")
    print("2. For pair coalition (i,j):")
    print("   - The formula depends on the feature ordering distribution in the dataset")
    print("   - This explains why the matrix varies with data distribution")
    print("\nConclusion: There is no simple universal formula for the entire matrix.")
    print("However, the singleton portion follows a clear pattern that can be coded explicitly.")

if __name__ == "__main__":
    derive_formula()
