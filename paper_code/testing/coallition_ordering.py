import numpy as np
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt
from choquet_function import (
    choquet_matrix_kadd_guilherme, choquet_matrix_2add, 
    powerset, nParam_kAdd
)
import mod_GenFuzzyRegression as mGFR

def extract_coalition_ordering(X, method="game", k_add=2):
    """
    Extract the actual coalition ordering used in different implementations.
    
    Parameters:
    -----------
    X : array-like
        Sample data matrix
    method : str
        "game" or "shapley" to indicate which implementation to check
    k_add : int
        Maximum coalition size
        
    Returns:
    --------
    coalitions : list
        List of coalitions in the order they appear in the transformation
    """
    n_features = X.shape[1]
    
    # Create a special test matrix where each feature has a unique prime number
    # This allows us to identify which features are used in each transformed column
    test_matrix = np.zeros((1, n_features))
    prime_values = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29][:n_features]
    for i in range(n_features):
        test_matrix[0, i] = prime_values[i]
    
    # Apply the transformation
    if method == "game":
        transformed = choquet_matrix_kadd_guilherme(test_matrix, kadd=k_add)
    else:
        transformed = choquet_matrix_2add(test_matrix)
    
    # For each column in the transformation, determine which features contributed
    coalitions = []
    for col_idx in range(transformed.shape[1]):
        col_value = transformed[0, col_idx]
        
        # Skip columns with zero contribution (might be intercept or empty set)
        if abs(col_value) < 1e-10:
            coalitions.append(())
            continue
            
        # Identify contributing features by testing divisibility by prime numbers
        coalition = []
        for feat_idx, prime in enumerate(prime_values):
            # For shapley domain, we need to check min operations differently
            if method == "shapley" and col_value == prime:
                coalition.append(feat_idx)
            elif method == "game" and col_value > 0:
                coalition.append(feat_idx)
                
        coalitions.append(tuple(sorted(coalition)))
    
    return coalitions

def compare_coalition_orderings():
    """Compare coalition orderings between game and Shapley domain implementations."""
    # Create sample data
    n_features = 4  # Small number for clarity
    X_sample = np.random.rand(5, n_features)
    
    # Extract coalition orderings
    game_coalitions = extract_coalition_ordering(X_sample, "game", k_add=2)
    shapley_coalitions = extract_coalition_ordering(X_sample, "shapley", k_add=2)
    
    # Generate expected coalitions in standard order
    expected_coalitions = []
    for r in range(1, 3):  # k_add=2
        expected_coalitions.extend(combinations(range(n_features), r))
    
    # Print results
    print(f"Testing with {n_features} features, k_add=2")
    print("\nExpected coalition ordering:")
    for i, coal in enumerate(expected_coalitions):
        print(f"{i}: {coal}")
    
    print("\nGame domain coalition ordering:")
    for i, coal in enumerate(game_coalitions):
        print(f"{i}: {coal}")
    
    print("\nShapley domain coalition ordering:")
    for i, coal in enumerate(shapley_coalitions):
        print(f"{i}: {coal}")
    
    # Create mapping between orderings
    game_to_expected = {i: expected_coalitions.index(coal) 
                        for i, coal in enumerate(game_coalitions) 
                        if coal in expected_coalitions}
    
    shapley_to_expected = {i: expected_coalitions.index(coal) 
                           for i, coal in enumerate(shapley_coalitions) 
                           if coal in expected_coalitions}
    
    # Verify if mappings are consistent
    print("\nCoalition mapping consistency check:")
    print(f"Game domain coalitions match expected: {len(game_to_expected) == len(expected_coalitions)}")
    print(f"Shapley domain coalitions match expected: {len(shapley_to_expected) == len(expected_coalitions)}")
    
    # Create corrected mobius transform function
    print("\nCreating corrected Möbius transform with consistent coalition ordering...")
    
def corrected_mobius_transform(game_coeffs, shapley_coeffs, X_sample, n_features, k_add=2):
    """
    Create mappings to correct the Möbius transform between domains.
    
    Parameters:
    -----------
    game_coeffs : array-like
        Coefficients from game domain
    shapley_coeffs : array-like
        Coefficients from Shapley domain
    X_sample : array-like
        Sample data to extract coalition orderings
    n_features : int
        Number of features
    k_add : int
        Maximum coalition size
        
    Returns:
    --------
    corrected_game_to_shapley : array-like
        Game coefficients converted to Shapley domain with correct ordering
    corrected_shapley_to_game : array-like
        Shapley coefficients converted to game domain with correct ordering
    """
    # Extract coalition orderings
    game_coalitions = extract_coalition_ordering(X_sample, "game", k_add)
    shapley_coalitions = extract_coalition_ordering(X_sample, "shapley", k_add)
    
    # Create standard ordering
    expected_coalitions = []
    for r in range(1, k_add+1):
        expected_coalitions.extend(combinations(range(n_features), r))
    
    # Create mappings
    game_to_std = {i: expected_coalitions.index(coal) 
                   for i, coal in enumerate(game_coalitions) 
                   if coal in expected_coalitions}
    
    std_to_shapley = {expected_coalitions.index(coal): i 
                       for i, coal in enumerate(shapley_coalitions) 
                       if coal in expected_coalitions}
    
    # Apply mappings
    corrected_game_to_shapley = np.zeros_like(shapley_coeffs)
    corrected_shapley_to_game = np.zeros_like(game_coeffs)
    
    for game_idx, std_idx in game_to_std.items():
        if std_idx in std_to_shapley:
            shapley_idx = std_to_shapley[std_idx]
            corrected_game_to_shapley[shapley_idx] = game_coeffs[game_idx]
    
    for std_idx, shapley_idx in std_to_shapley.items():
        if std_idx in game_to_std:
            game_idx = game_to_std[std_idx]
            corrected_shapley_to_game[game_idx] = shapley_coeffs[shapley_idx]
    
    return corrected_game_to_shapley, corrected_shapley_to_game

if __name__ == "__main__":
    compare_coalition_orderings()