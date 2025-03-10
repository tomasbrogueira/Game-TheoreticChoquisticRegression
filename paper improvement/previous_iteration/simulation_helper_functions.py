import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def get_feature_names(X):
    """Extract feature names from DataFrame or create default names."""
    if isinstance(X, pd.DataFrame):
        return X.columns.tolist()
    else:
        return [f"F{i}" for i in range(X.shape[1])]

def plot_horizontal_bar(names, values, std=None, title="", xlabel="", filename="", color="steelblue"):
    """
    Plots a horizontal bar chart.
    
    Parameters:
      - names: list of feature names.
      - values: values to plot.
      - std: error bars (optional).
      - title: plot title.
      - xlabel: label for x-axis.
      - filename: path to save the figure.
      - color: bar color.
    """
    ordered_indices = np.argsort(values)[::-1]
    ordered_names = np.array(names)[ordered_indices]
    ordered_values = values[ordered_indices]
    ordered_std = std[ordered_indices] if std is not None else None

    plt.figure(figsize=(10, 8))
    plt.barh(ordered_names, ordered_values, xerr=ordered_std, color=color, edgecolor="black")
    plt.xlabel(xlabel, fontsize=16)
    plt.title(title, fontsize=18)
    plt.gca().invert_yaxis()
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print("Saved plot to:", filename)




# Functions for overall interaction feature importance

def weighted_pairwise_interaction_effect(v, interaction_matrix, m):
    """
    Compute the weighted interaction effect for each feature in 2-additive models.
    
    Args:
        v (np.array): Coefficients for singletons (first m entries) and pairs (remaining entries).
        interaction_matrix (np.array): Symmetric matrix of pairwise interaction indices.
        m (int): Number of features.
    
    Returns:
        np.array: Weighted interaction effect for each feature.
    """
    # Extract pairwise coefficients (assuming v is ordered as [singletons, pairs])
    pair_coefs = v[m:]
    
    # Map pairwise coefficients to interaction matrix
    weighted_matrix = np.zeros((m, m))
    idx = 0
    for i in range(m):
        for j in range(i+1, m):
            weighted_matrix[i, j] = pair_coefs[idx] * interaction_matrix[i, j]
            weighted_matrix[j, i] = weighted_matrix[i, j]  # Symmetry
            idx += 1
    
    # Sum absolute values for overall effect
    return np.sum(np.abs(weighted_matrix), axis=1)


def weighted_full_interaction_effect(v, all_coalitions, m, index_type='shapley'):
    """
    Compute the weighted interaction effect for each feature in full Choquet models.
    
    Args:
        v (np.array): Coefficients for all nonempty coalitions.
        all_coalitions (list): List of all nonempty coalitions (tuples).
        m (int): Number of features.
        index_type (str): 'shapley' or 'banzhaf'.
    
    Returns:
        np.array: Weighted interaction effect for each feature.
    """
    weighted_effect = np.zeros(m)
    coalition_to_index = {coal: idx for idx, coal in enumerate(all_coalitions)}

    from regression_classes import shapley_interaction_index, banzhaf_interaction_index
    
    for A in all_coalitions:
        if len(A) < 2:
            continue  # Skip singletons (no interaction)
        
        # Compute interaction index for coalition A
        if index_type == 'shapley':
            I_A = shapley_interaction_index(A, all_coalitions, v, m)
        else:
            I_A = banzhaf_interaction_index(A, all_coalitions, v, m)
        
        # Weight by the coalition's coefficient (absolute value)
        weight = np.abs(v[coalition_to_index[A]])
        
        # Distribute the weighted interaction effect to all features in A
        for i in A:
            weighted_effect[i] += weight * np.abs(I_A)
    
    return weighted_effect



# Interaction to Shapely Ratio (ISR) functions

import itertools
import numpy as np

def total_interaction_contribution(interaction_matrix, all_coalitions, m):
    """
    Compute normalized interaction contribution per feature, adjusted for combinatorial scaling.

    Args:
        interaction_matrix (np.array): Interaction indices matrix (Shapley/Banzhaf); assumed 2D.
        all_coalitions (list): List of all coalitions (tuples) used in the model.
        m (int): Number of features.
    
    Returns:
        np.array: Normalized interaction contribution per feature ∈ [0, 1].
    """
    interaction_matrix = np.array(interaction_matrix)
    # If a list (or 3D array) of matrices is passed, average over the first axis.
    if interaction_matrix.ndim > 2:
        interaction_matrix = np.mean(interaction_matrix, axis=0)
    
    tic = np.zeros(m)
    coalition_counts = np.zeros(m)
    
    for A in all_coalitions:
        if len(A) < 2:
            continue  # Skip singletons
        for i in A:
            if len(A) == 2:
                elem = np.abs(interaction_matrix[int(A[0]), int(A[1])])
                if np.size(elem) == 1:
                    val = float(elem)
                else:
                    val = float(np.sum(elem))
                tic[i] += val
            else:
                # For higher-order coalitions, sum over all unique pairs.
                pair_sum = 0.0
                for u, v in itertools.combinations(A, 2):
                    pair_sum += np.abs(interaction_matrix[int(u), int(v)])
                tic[i] += pair_sum
            coalition_counts[i] += 1

    tic_normalized = np.divide(tic, coalition_counts, where=(coalition_counts != 0))
    return tic_normalized / np.sum(tic_normalized)


def interaction_shapley_ratio(shapley_values, interaction_matrix, all_coalitions, m):
    """
    Compute Interaction-to-Shapley Ratio (ISR) for each feature.
    
    Args:
        shapley_values (np.array): Shapley values (all coalition contributions).
        interaction_matrix (np.array): Interaction indices matrix.
        all_coalitions (list): List of all coalitions.
        m (int): Number of features.
    
    Returns:
        np.array: ISR per feature ∈ [0, 1], where 0 = no interaction, 1 = all interaction.
    """
    # In our Choquet model the coefficient vector contains both singleton and interaction values.
    # Use only singleton Shapley contributions (first m elements) for computing the ratio.
    if shapley_values.shape[0] != m:
        tsc = np.abs(shapley_values[:m])
    else:
        tsc = np.abs(shapley_values)
    
    tic = total_interaction_contribution(interaction_matrix, all_coalitions, m)
    
    # Avoid division by zero
    total_contribution = tsc + tic
    isr = np.divide(tic, total_contribution, where=(total_contribution != 0))
    return isr




def overall_interaction_index(interaction_matrix):
    """
    Compute the overall interaction effect for each feature from the pairwise interaction matrix.
    
    For feature j, the overall interaction is defined as 0.5 times the sum of the interactions 
    between feature j and every other feature.
    
    Parameters:
        interaction_matrix (np.ndarray): An m x m symmetric matrix of pairwise interaction indices.
    
    Returns:
        np.ndarray: A 1D array of length m with the overall interaction index for each feature.
    """
    overall = 0.5 * np.sum(interaction_matrix, axis=1)
    return overall



def overall_interaction_index_abs(interaction_matrix):
    """
    Compute the overall interaction magnitude for each feature, ignoring sign.
    
    For feature j, this is defined as 0.5 times the sum of the absolute interaction 
    indices with every other feature.
    
    Parameters:
        interaction_matrix (np.ndarray): An m x m symmetric matrix of pairwise interaction indices.
    
    Returns:
        np.ndarray: A 1D array of length m with the overall interaction magnitude for each feature.
    """
    overall = 0.5 * np.sum(np.abs(interaction_matrix), axis=1)
    return overall




def overall_interaction_from_shapley(shapley_vals, marginal_vals):
    """
    Compute the overall interaction effect for each feature from the Shapley and marginal values.
    
    In a 2-additive Choquet model:
      φ_S(j) = marginal_j + (1/2) * Σ_{k≠j} I_S(j,k)
    
    Therefore, the overall interaction effect is:
      overall_interaction(j) = φ_S(j) - marginal_j
    
    Parameters:
        shapley_vals (np.ndarray): 1D array of Shapley values for each feature.
        marginal_vals (np.ndarray): 1D array of marginal (singleton) contributions.
    
    Returns:
        np.ndarray: A 1D array of overall interaction effects for each feature.
    """
    return shapley_vals - marginal_vals




# debugging functions

def compare_transformations(X_sample):
    from sklearn.preprocessing import MinMaxScaler
    from regression_classes import choquet_matrix_2add, mlm_matrix_2add
    """Compare all transformation methods using a sample."""
    # Ensure data is properly scaled
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_sample)
    
    # Apply transformations
    choq_trans = choquet_matrix_2add(X_scaled)
    mlm_trans = mlm_matrix_2add(X_scaled)
    
    # Create synthetic data with non-binary values
    X_synthetic = np.random.random((3, X_sample.shape[1]))
    choq_synth = choquet_matrix_2add(X_synthetic)
    mlm_synth = mlm_matrix_2add(X_synthetic)
    
    # Print differences
    print("Real data difference:", np.sum(np.abs(choq_trans - mlm_trans)))
    print("Synthetic data difference:", np.sum(np.abs(choq_synth - mlm_synth)))
    
    # Print sample values to compare min vs product
    i, j = 0, 1  # First two features
    print(f"\nComparison for features {i} and {j}:")
    print(f"X_synthetic values: {X_synthetic[0, i]:.4f}, {X_synthetic[0, j]:.4f}")
    print(f"min: {min(X_synthetic[0, i], X_synthetic[0, j]):.4f}")
    print(f"product: {X_synthetic[0, i] * X_synthetic[0, j]:.4f}")




def verify_shapley_decomposition(feature_idx, v, all_coalitions, m):
    """Verify Shapley decomposition for a specific feature"""
    from regression_classes import compute_shapley_values, compute_choquet_interaction_matrix
    # Get Shapley value
    shapley = compute_shapley_values(v, m, all_coalitions)[feature_idx]
    
    # Get singleton value
    try:
        singleton_idx = all_coalitions.index((feature_idx,))
        singleton = v[singleton_idx+1]
    except ValueError:
        singleton = 0.0
        
    # Get interaction matrix
    int_matrix = compute_choquet_interaction_matrix(v, m, all_coalitions)
    int_sum = 0.5 * np.sum(int_matrix[feature_idx, :])
    
    # Verify the relationship
    print(f"Shapley: {shapley}")
    print(f"Singleton: {singleton}")
    print(f"0.5 * Sum Interactions: {int_sum}")
    print(f"Singleton + Interactions: {singleton + int_sum}")
    print(f"Difference: {shapley - (singleton + int_sum)}")





def verify_matrix_shapley_equivalence(v, m, all_coalitions):
    """Verify the mathematical relationship: φᵢ = v({i}) + 0.5 * Σⱼ≠ᵢ I({i,j})"""
    from regression_classes import compute_shapley_values, compute_choquet_interaction_matrix
    shapley = compute_shapley_values(v, m, all_coalitions)
    int_matrix = compute_choquet_interaction_matrix(v, m, all_coalitions)
    
    # Get singleton values
    singletons = np.zeros(m)
    for i in range(m):
        try:
            idx = all_coalitions.index((i,))
            singletons[i] = v[idx + 1]  # +1 for empty set
        except ValueError:
            pass
            
    # Calculate method 1: from interaction matrix
    method1 = singletons + 0.5 * np.sum(int_matrix, axis=1)
    
    # Compare with shapley values
    diff = np.abs(method1 - shapley)
    print("Matrix vs Shapley max difference:", np.max(diff))
    print("Matrix vs Shapley average difference:", np.mean(diff))
    return diff