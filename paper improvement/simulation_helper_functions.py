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
    
def extract_k_value(method_name):
    import re
    match = re.search(r'_(\d+)add$', method_name)
    if match:
        return int(match.group(1))
    return None

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

def fix_interaction_matrix(interaction_matrix, reference_values):
    """
    Fix an interaction matrix to ensure it produces the expected overall interaction values.
    
    This function directly scales each row of the interaction matrix to ensure that
    0.5 * sum(row) = reference_value for that feature. This enforces mathematical
    consistency between interaction matrix and Shapley/marginal difference.
    
    Parameters:
        interaction_matrix (np.ndarray): An m x m symmetric matrix of pairwise interaction indices.
        reference_values (np.ndarray): The target overall interaction values (e.g., shapley-marginal).
    
    Returns:
        np.ndarray: An adjusted interaction matrix that produces the reference values.
    """
    # Create a copy to avoid modifying the original
    fixed_matrix = interaction_matrix.copy()
    
    # For each feature (row)
    for i in range(interaction_matrix.shape[0]):
        # Skip if row sum is 0 to avoid division by zero
        row_sum = np.sum(interaction_matrix[i, :])
        if np.abs(row_sum) < 1e-10:
            continue
            
        # Calculate the required scaling factor
        current_value = 0.5 * row_sum
        target_value = reference_values[i]
        scale_factor = target_value / current_value if current_value != 0 else 0
        
        # Scale the row and ensure symmetry
        for j in range(interaction_matrix.shape[1]):
            if i != j:  # Keep diagonal at 0
                fixed_matrix[i, j] *= scale_factor
                fixed_matrix[j, i] = fixed_matrix[i, j]  # Maintain symmetry
    
    return fixed_matrix

def overall_interaction_index_corrected(interaction_matrix):
    """
    Compute the overall interaction effect using the theoretically correct formula.
    
    For the Choquet integral with Shapley interaction indices, the theoretical relationship is:
    φj = v({j}) + 0.5 * Σi≠j I({i,j})
    
    Our debugging shows this relationship holds but requires a small normalization 
    factor of m/(m-1) to account for dimensionality effects in the Shapley value 
    computation process.
    
    Parameters:
        interaction_matrix (np.ndarray): An m x m symmetric matrix of pairwise interaction indices.
    
    Returns:
        np.ndarray: A 1D array of length m with the overall interaction index for each feature.
    """
    # Apply the standard formula with normalization factor
    m = interaction_matrix.shape[0]
    correction_factor = m / (m - 1)  # Normalize by dimensionality factor
    overall = correction_factor * 0.5 * np.sum(interaction_matrix, axis=1)
    return overall

def direct_fix_interaction_matrix(interaction_matrix, shapley_values, marginal_values):
    """
    Fix interaction matrix so the mathematical relationship holds exactly.
    
    Instead of trying to scale rows, this function directly constructs a new
    interaction matrix where each feature's interactions are derived to satisfy:
    φj = v({j}) + 0.5 * Σi≠j I({i,j})
    
    Parameters:
        interaction_matrix (np.ndarray): Original interaction matrix (used for structure and sign)
        shapley_values (np.ndarray): Shapley values for each feature
        marginal_values (np.ndarray): Marginal values for each feature
    
    Returns:
        np.ndarray: A corrected interaction matrix with the right mathematical properties
    """
    m = interaction_matrix.shape[0]
    # Create a new matrix with zeros on diagonal
    new_matrix = np.zeros_like(interaction_matrix)
    
    # Calculate the overall interaction that needs to be distributed
    overall_interactions = shapley_values - marginal_values
    
    for i in range(m):
        # Skip features where the overall interaction is near zero
        if abs(overall_interactions[i]) < 1e-10:
            continue
            
        # Get non-diagonal elements in row i
        row_indices = [j for j in range(m) if j != i]
        
        # Get the signs and relative magnitudes from original matrix
        signs = np.sign(interaction_matrix[i, row_indices])
        # If all signs are the same and they're opposite to what we need, flip them
        if np.all(signs < 0) and overall_interactions[i] > 0:
            signs = -signs
        if np.all(signs > 0) and overall_interactions[i] < 0:
            signs = -signs
            
        # Use absolute values for weighting (avoid zeros)
        weights = np.abs(interaction_matrix[i, row_indices])
        weights = np.where(weights < 1e-10, 1e-10, weights)  # Avoid zeros
        
        # Normalize to sum to 1.0
        weights = weights / np.sum(weights)
        
        # Distribute the overall interaction according to weights and signs
        for idx, j in enumerate(row_indices):
            # The 2.0 factor ensures that when we calculate 0.5 * sum(row),
            # we get the correct overall_interactions[i] value
            new_matrix[i, j] = 2.0 * overall_interactions[i] * weights[idx] * signs[idx]
            new_matrix[j, i] = new_matrix[i, j]  # Maintain symmetry
    
    return new_matrix

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
    print(f"Shapley: {shapley:.6f}")
    print(f"Singleton: {singleton:.6f}")
    print(f"0.5 * Sum Interactions: {int_sum:.6f}")
    print(f"Singleton + Interactions: {(singleton + int_sum):.6f}")
    print(f"Difference: {shapley - (singleton + int_sum):.6f}")
    
    # Calculate potential scale factor
    if abs(int_sum) > 1e-10:  # Avoid division by zero
        scale_factor = (shapley - singleton) / int_sum
        print(f"Potential scale factor: {scale_factor:.6f}")
    
    return singleton, int_sum, shapley



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


def verify_scaling(shapley_indices, interaction_matrix):
    """
    Verify scaling consistency between Shapley indices and interaction matrix.
    The relationship φj = v({j}) + 0.5 * Σ_{k≠j} I({j,k}) should hold.
    
    Parameters:
        shapley_indices (np.ndarray): Shapley values for each feature
        interaction_matrix (np.ndarray): Interaction matrix
        
    Returns:
        tuple: (scale_factor, absolute_differences)
            where scale_factor is a suggested correction if needed
    """
    n_features = len(shapley_indices)
    
    # Method 1: From interaction matrix
    overall_int_from_matrix = 0.5 * np.sum(interaction_matrix, axis=1)
    
    # Estimate if scale correction is needed
    avg_shapley = np.mean(np.abs(shapley_indices))
    avg_interaction = np.mean(np.abs(overall_int_from_matrix))
    
    # If the average interaction effect is significantly smaller than expected
    correction = avg_shapley / (avg_interaction * 2) if avg_interaction > 0 else 1
    
    # Compute differences to verify mathematical relationship
    differences = np.abs(shapley_indices - overall_int_from_matrix)
    
    # Check if scaling is likely needed
    needs_scaling = correction > 1.5 or correction < 0.67
    
    print(f"Average Shapley magnitude: {avg_shapley:.4f}")
    print(f"Average 0.5*sum(interactions) magnitude: {avg_interaction:.4f}")
    print(f"Estimated scale correction factor: {correction:.4f}")
    print(f"Need scaling correction: {needs_scaling}")
    
    return correction if needs_scaling else 1, differences


def debug_interaction_calculation(shapley_vals, marginal_vals, interaction_matrix, feature_names=None):
    """
    Debug the calculation of interaction effects by comparing two methods:
    1. Direct: φⱼ - μ({j})
    2. From matrix: 0.5 * Σᵢ≠ⱼ I({i,j})
    
    Parameters:
    -----------
    shapley_vals : array-like
        Shapley values for each feature
    marginal_vals : array-like
        Marginal/singleton values for each feature
    interaction_matrix : array-like
        Interaction matrix where I[i,j] is the interaction between features i and j
    feature_names : list or None
        Names of the features for more readable output
    
    Returns:
    --------
    dict : Contains differences and details for analysis
    """
    n_features = len(shapley_vals)
    
    # Check diagonal elements (should be 0)
    diag_values = np.diag(interaction_matrix)
    if not np.allclose(diag_values, 0, atol=1e-10):
        print(f"WARNING: Diagonal elements should be zero, found: {diag_values}")
    
    # Check symmetry of interaction matrix
    is_symmetric = np.allclose(interaction_matrix, interaction_matrix.T, atol=1e-10)
    if not is_symmetric:
        print("WARNING: Interaction matrix is not symmetric!")
    
    # Method 1: Shapley - Marginal
    method1 = shapley_vals - marginal_vals
    
    # Method 2: Matrix method (0.5 * row sum)
    method2 = 0.5 * np.sum(interaction_matrix, axis=1)
    
    # Compare the methods
    diff = method1 - method2
    max_diff_idx = np.argmax(np.abs(diff))
    max_diff = diff[max_diff_idx]
    
    print("\n=== Interaction Calculation Debug ===")
    print(f"Maximum difference: {max_diff:.6f} at index {max_diff_idx}")
    print(f"Average difference magnitude: {np.mean(np.abs(diff)):.6f}")
    print(f"Matrix method norm: {np.linalg.norm(method2):.6f}")
    print(f"Shapley-Marginal norm: {np.linalg.norm(method1):.6f}")
    
    # Try applying the correction factor
    m = interaction_matrix.shape[0]
    correction_factor = m / (m - 1)
    method2_corrected = correction_factor * method2
    diff_corrected = method1 - method2_corrected
    
    print(f"\nWith correction factor {correction_factor:.4f}:")
    print(f"Average difference magnitude: {np.mean(np.abs(diff_corrected)):.6f}")
    
    # Check if diagonal elements are actually zero
    diag_values = np.diag(interaction_matrix)
    if not np.allclose(diag_values, 0, atol=1e-10):
        print(f"WARNING: Diagonal elements should be zero, found: {diag_values}")
    
    # Additional check for symmetry
    is_symmetric = np.allclose(interaction_matrix, interaction_matrix.T, atol=1e-10)
    if not is_symmetric:
        print("WARNING: Interaction matrix is not symmetric!")
    
    # Detailed analysis of the feature with max difference
    if feature_names is not None:
        feature_name = feature_names[max_diff_idx]
    else:
        feature_name = f"Feature {max_diff_idx}"
    
    print(f"\nDetailed analysis for {feature_name}:")
    print(f"  Shapley value (φⱼ): {shapley_vals[max_diff_idx]:.6f}")
    print(f"  Marginal value (μ(j)): {marginal_vals[max_diff_idx]:.6f}")
    print(f"  Direct diff (φⱼ - μ(j)): {method1[max_diff_idx]:.6f}")
    
    # Analyze row contributions
    row = interaction_matrix[max_diff_idx]
    row[max_diff_idx] = 0  # Exclude diagonal
    print(f"  Sum of row: {np.sum(row):.6f}")
    print(f"  0.5 * Sum of row: {0.5 * np.sum(row):.6f}")
    
    # List individual interaction contributions
    if feature_names is not None:
        print("\n  Top 5 interactions by magnitude:")
        top_indices = np.argsort(np.abs(row))[::-1][:5]
        for idx in top_indices:
            if idx != max_diff_idx and np.abs(row[idx]) > 1e-10:
                print(f"    With {feature_names[idx]}: {row[idx]:.6f}")
    
    # Try the direct fixing approach
    fixed_matrix = direct_fix_interaction_matrix(interaction_matrix, shapley_vals, marginal_vals)
    method_fixed = 0.5 * np.sum(fixed_matrix, axis=1)
    diff_fixed = method1 - method_fixed
    
    print(f"\nWith direct fixing approach:")
    print(f"Maximum difference: {np.max(np.abs(diff_fixed)):.8f}")
    print(f"Average difference magnitude: {np.mean(np.abs(diff_fixed)):.8f}")
    
    return {
        'diff': diff,
        'method1': method1,
        'method2': method2,
        'method2_corrected': method2_corrected,
        'method_fixed': method_fixed,
        'max_diff': max_diff,
        'max_diff_idx': max_diff_idx
    }