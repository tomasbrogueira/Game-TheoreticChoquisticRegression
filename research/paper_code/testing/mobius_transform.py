import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import chain, combinations
from math import comb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import sys
import os


from choquet_function import (
    powerset, choquet_matrix_kadd_guilherme, choquet_matrix_2add, 
    nParam_kAdd
)
import mod_GenFuzzyRegression as mGFR

def powerset_list(iterable, max_size=None):
    """Return the powerset of a set as a list of tuples, up to max_size."""
    s = list(iterable)
    if max_size is None:
        max_size = len(s)
    return list(chain.from_iterable(combinations(s, r) for r in range(max_size+1)))

def mobius_transform(game_coeffs, n_features, k_add=2):
    """
    Convert game domain coefficients to Shapley domain (Möbius transform).
    
    Parameters:
    -----------
    game_coeffs : array-like
        Coefficients from the game domain model
    n_features : int
        Number of original features
    k_add : int
        Maximum size of interactions to consider
        
    Returns:
    --------
    shapley_coeffs : array-like
        Coefficients in the Shapley domain
    """
    # Create coalition mapping for game domain
    game_coalitions = []
    for r in range(1, k_add + 1):
        game_coalitions.extend(combinations(range(n_features), r))
    
    # Initialize Shapley coefficients
    shapley_coeffs = np.zeros(len(game_coalitions))
    
    # Apply Möbius transform formula
    for i, S in enumerate(game_coalitions):
        S_set = set(S)
        for j, T in enumerate(game_coalitions):
            T_set = set(T)
            if T_set.issubset(S_set):
                shapley_coeffs[i] += ((-1) ** (len(S_set) - len(T_set))) * game_coeffs[j]
    
    return shapley_coeffs

def inverse_mobius_transform(shapley_coeffs, n_features, k_add=2):
    """
    Convert Shapley domain coefficients back to game domain.
    
    Parameters:
    -----------
    shapley_coeffs : array-like
        Coefficients from the Shapley domain model
    n_features : int
        Number of original features
    k_add : int
        Maximum size of interactions to consider
        
    Returns:
    --------
    game_coeffs : array-like
        Coefficients in the game domain
    """
    # Create coalition mapping for Shapley domain
    shapley_coalitions = []
    for r in range(1, k_add + 1):
        shapley_coalitions.extend(combinations(range(n_features), r))
    
    # Initialize game coefficients
    game_coeffs = np.zeros(len(shapley_coalitions))
    
    # Apply inverse Möbius transform formula
    for i, S in enumerate(shapley_coalitions):
        S_set = set(S)
        for j, T in enumerate(shapley_coalitions):
            T_set = set(T)
            if T_set.issubset(S_set):
                game_coeffs[i] += shapley_coeffs[j]
    
    return game_coeffs

def predict_with_transformed_coeffs(X, coeffs, intercept, n_features, k_add=2, domain="game"):
    """
    Make predictions using transformed coefficients.
    
    Parameters:
    -----------
    X : array-like
        Original feature matrix
    coeffs : array-like
        Model coefficients (before or after transformation)
    intercept : float
        Model intercept
    n_features : int
        Number of original features
    k_add : int
        Maximum coalition size
    domain : str
        "game" or "shapley" indicating which transformation to apply
        
    Returns:
    --------
    probabilities : array-like
        Predicted probabilities
    """
    if domain == "game":
        X_transformed = choquet_matrix_kadd_guilherme(X, kadd=k_add)
    else:
        X_transformed = choquet_matrix_2add(X)
    
    # Compute logits
    logits = X_transformed @ coeffs + intercept
    
    # Convert to probabilities
    probs = 1 / (1 + np.exp(-logits))
    
    return probs

def verify_mobius_transform():
    """Verify that the Möbius transform correctly converts between domains."""
    # Load data
    data_name = 'banknotes'
    X, y = mGFR.func_read_data(data_name)
    
    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    n_features = X.shape[1]
    k_add = 2
    
    print(f"Testing Möbius transform with {data_name} dataset")
    print(f"Number of features: {n_features}, k_add: {k_add}")
    print("=" * 70)
    
    # Create transformation matrices
    X_game_train = choquet_matrix_kadd_guilherme(X_train_scaled, kadd=k_add)
    X_game_test = choquet_matrix_kadd_guilherme(X_test_scaled, kadd=k_add)
    
    X_shapley_train = choquet_matrix_2add(X_train_scaled)
    X_shapley_test = choquet_matrix_2add(X_test_scaled)
    
    # Train models
    game_model = LogisticRegression(max_iter=1000)
    game_model.fit(X_game_train, y_train)
    
    shapley_model = LogisticRegression(max_iter=1000)
    shapley_model.fit(X_shapley_train, y_train)
    
    # Get coefficients
    game_coeffs = game_model.coef_[0]
    shapley_coeffs = shapley_model.coef_[0]
    
    game_intercept = game_model.intercept_[0]
    shapley_intercept = shapley_model.intercept_[0]
    
    # 1. Direct predictions (baseline)
    game_proba = game_model.predict_proba(X_game_test)[:, 1]
    shapley_proba = shapley_model.predict_proba(X_shapley_test)[:, 1]
    
    # 2. Transform game coefficients to Shapley domain
    shapley_from_game = mobius_transform(game_coeffs, n_features, k_add)
    
    # 3. Transform Shapley coefficients to game domain
    game_from_shapley = inverse_mobius_transform(shapley_coeffs, n_features, k_add)
    
    # 4. Make predictions with transformed coefficients
    shapley_from_game_proba = predict_with_transformed_coeffs(
        X_test_scaled, shapley_from_game, game_intercept, n_features, k_add, "shapley"
    )
    
    game_from_shapley_proba = predict_with_transformed_coeffs(
        X_test_scaled, game_from_shapley, shapley_intercept, n_features, k_add, "game"
    )
    
    # 5. Compare predictions
    print("\nBaseline Model Correlation:")
    baseline_corr = np.corrcoef(game_proba, shapley_proba)[0, 1]
    print(f"Correlation between original game and Shapley models: {baseline_corr:.6f}")
    
    print("\nTransformed Coefficients Correlation:")
    game_to_shapley_corr = np.corrcoef(shapley_proba, shapley_from_game_proba)[0, 1]
    print(f"Game→Shapley transformation accuracy: {game_to_shapley_corr:.6f}")
    
    shapley_to_game_corr = np.corrcoef(game_proba, game_from_shapley_proba)[0, 1]
    print(f"Shapley→Game transformation accuracy: {shapley_to_game_corr:.6f}")
    
    # 6. Visualize results
    plt.figure(figsize=(18, 6))
    
    # Original model comparison
    plt.subplot(1, 3, 1)
    plt.scatter(game_proba, shapley_proba, alpha=0.5, label='Original')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('Game Domain Probabilities')
    plt.ylabel('Shapley Domain Probabilities')
    plt.title('Original Models')
    plt.grid(True)
    
    # Game → Shapley transformation
    plt.subplot(1, 3, 2)
    plt.scatter(shapley_proba, shapley_from_game_proba, alpha=0.5, 
                label='Shapley vs Transformed Game')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('Original Shapley Probabilities')
    plt.ylabel('Game→Shapley Transformed')
    plt.title(f'Game→Shapley Transform\nCorrelation: {game_to_shapley_corr:.4f}')
    plt.grid(True)
    
    # Shapley → Game transformation
    plt.subplot(1, 3, 3)
    plt.scatter(game_proba, game_from_shapley_proba, alpha=0.5,
                label='Game vs Transformed Shapley')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('Original Game Probabilities')
    plt.ylabel('Shapley→Game Transformed')
    plt.title(f'Shapley→Game Transform\nCorrelation: {shapley_to_game_corr:.4f}')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('mobius_transform_verification.png')
    print("\nPlot saved as 'mobius_transform_verification.png'")

if __name__ == "__main__":
    verify_mobius_transform()