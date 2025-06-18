import os
os.environ["SCIPY_ARRAY_API"] = "1"

RANDOM_STATE = 42

import numpy as np
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.cm import get_cmap
import time
from tqdm import tqdm
import scipy.special
import itertools
import warnings
warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

# Import required modules
from regression_classes import (
    nParam_kAdd,
    choquet_k_additive_shapley
)
from mod_GenFuzzyRegression import func_read_data
from dataset_operations import (
    add_gaussian_noise,
    add_bias,
    scale_features,
    log_transform,
    power_transform,
    tanh_transform,
    threshold_features,
    clip_features
)
from plotting_functions import plot_model_coefficients

def indices_from_shapley_coefficients(coeffs, n_attr, k_add=None):
    """
    Returns for each sample x:
      - phi (length n):   the n singleton Shapley values I({i})
      - interactions:     the vector of I(A) for all 2≤|A|≤k_add
    """
    if k_add is None:
        k_add = n_attr

    phi = coeffs[:n_attr]
    inter_values = coeffs[n_attr:]

    return phi, inter_values

def get_interaction_features(idx, n_attr):
    """
    Maps an interaction index to its corresponding feature combination.
    Assumes standard indexing order: all pairs, then all triplets, etc.
    
    Parameters:
    -----------
    idx: int
        Index in the interaction vector
    n_attr: int
        Number of attributes/features
        
    Returns:
    -----------
    tuple: Feature indices involved in the interaction
    """
    remaining_idx = idx
    size = 2  # Start with pairs
    
    # Skip through larger sizes until we find the right one
    while True:
        comb_count = scipy.special.comb(n_attr, size, exact=True)
        if remaining_idx < comb_count:
            break
        remaining_idx -= comb_count
        size += 1
        if size > n_attr:  # Safety check
            return tuple(range(n_attr))
    
    # Generate all combinations of the current size and get the one at remaining_idx
    all_combinations = list(itertools.combinations(range(n_attr), size))
    return all_combinations[remaining_idx]

def get_interaction_ranges(n_attr, k_add):
    """
    Calculates index ranges for each interaction order in the interaction vector.
    
    Parameters:
    -----------
    n_attr: int
        Number of attributes/features
    k_add: int
        Maximum interaction order (k-additivity)
        
    Returns:
    -----------
    dict: Dictionary mapping interaction order to (start, end) index ranges
    """
    ranges = {}
    start_idx = 0
    
    for order in range(2, k_add + 1):
        comb_count = scipy.special.comb(n_attr, order, exact=True)
        end_idx = start_idx + comb_count
        ranges[order] = (start_idx, end_idx)
        start_idx = end_idx
        
    return ranges

def compute_indices_robustness(
    dataset,
    output_dir=None,
    test_size=0.3,
    random_state=42,
    n_repetitions=10
):
    """
    Analyze the robustness of interaction indices across different k-additivity levels.
    
    Parameters:
    -----------
    dataset : str or tuple
        Either dataset name to load or tuple containing (dataset_name, X, y)
    output_dir : str, optional
        Directory to save results
    test_size : float, default=0.3
        Proportion of data to use for testing
    random_state : int, default=42
        Random seed for reproducibility
    n_repetitions : int, default=10
        Number of repetitions for noise and perturbation tests
    """
    # Determine name, X, y
    if isinstance(dataset, (list, tuple)) and len(dataset) == 3:
        dataset_name, X, y = dataset
    else:
        dataset_name = dataset
        X, y = func_read_data(dataset_name)
    
    print(f"\n{'='*80}\nAnalyzing indices robustness for: {dataset_name}...\n{'='*80}")
    
    # Create output directory if not specified
    if output_dir is None:
        output_dir = os.path.join("indices_robustness", dataset_name)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get dataset dimensions
    nSamp, nAttr = X.shape
    
    # Split data into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Convert to numpy arrays
    X_train_values = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
    X_test_values = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
    y_train_values = y_train.values if isinstance(y_train, pd.Series) else y_train
    y_test_values = y_test.values if isinstance(y_test, pd.Series) else y_test
    
    print(f"Dataset: {dataset_name}")
    print(f"- Number of samples: {nSamp} (train: {len(X_train)}, test: {len(X_test)})")
    print(f"- Number of attributes: {nAttr}")
    print(f"- Will test k-additivity from 1 to {nAttr}")
    
    # Save dataset info
    with open(os.path.join(output_dir, "dataset_info.txt"), 'w') as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Number of samples: {nSamp}\n")
        f.write(f"Number of attributes: {nAttr}\n")
        f.write(f"Train samples: {len(X_train)}\n")
        f.write(f"Test samples: {len(X_test)}\n")
    
    # Create columns for stability metrics of different orders
    columns = ['k_value', 'n_params', 'train_time', 'baseline_accuracy',
              'noise_0.05', 'noise_0.1', 'noise_0.2', 'noise_0.3', 
              'shapley_stability']  # Renamed from 'indices_stability' to be clearer
    
    # Add columns for interaction order stability up to the max attributes
    for order in range(2, nAttr + 1):
        columns.append(f'order{order}_stability')
    
    # Results dataframe for all k values
    results_df = pd.DataFrame(index=range(1, nAttr + 1), columns=columns)
    
    # Initialize dictionary to store indices for each k
    all_indices = {}
    
    try:
        # Scale the data
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train_values)
        X_test_scaled = scaler.transform(X_test_values)
        
        # Train and evaluate a model for each k value
        for k in range(1, nAttr + 1):
            print(f"\nAnalyzing indices with k = {k}/{nAttr}...")
            
            # Record number of parameters
            n_params = nParam_kAdd(k, nAttr)
            results_df.loc[k, 'k_value'] = k
            results_df.loc[k, 'n_params'] = n_params
            
            # Apply Shapley transformation and compute indices
            start_time = time.time()
            try:
                # Transform training data using Shapley representation
                X_train_shapley = choquet_k_additive_shapley(X_train_scaled, k_add=k)
                
                # Train logistic regression model
                model = LogisticRegression(
                    max_iter=1000,
                    random_state=random_state,
                    solver="newton-cg"
                )
                model.fit(X_train_shapley, y_train_values)
                train_time = time.time() - start_time
                results_df.loc[k, 'train_time'] = train_time
                
                # Generate coefficient plot for the model
                plot_model_coefficients(model, k, nAttr, "shapley", dataset_name, output_dir)
                
                # Extract model coefficients
                coeffs = model.coef_[0] if model.coef_.shape[0] == 1 else model.coef_
                
                # Compute interaction indices from Shapley values
                phi, inter_values = indices_from_shapley_coefficients(coeffs, nAttr, k)
                all_indices[k] = (phi, inter_values)
                
                # Transform test data and evaluate baseline performance
                X_test_shapley = choquet_k_additive_shapley(X_test_scaled, k_add=k)
                y_pred = model.predict(X_test_shapley)
                baseline_acc = np.mean(y_pred == y_test_values)
                results_df.loc[k, 'baseline_accuracy'] = baseline_acc
                
                print(f"- Indices computed with {n_params} parameters in {train_time:.2f} seconds")
                print(f"- Baseline accuracy: {baseline_acc:.4f}")
                
                # Get the ranges for each interaction order
                interaction_ranges = get_interaction_ranges(nAttr, k)
                
                # Noise robustness testing
                print("- Testing Shapley values robustness...")
                noise_levels = [0.05, 0.1, 0.2, 0.3]
                
                noise_indices = {}
                # Store phi and interaction correlations separately for detailed analysis
                phi_correlations = {level: [] for level in noise_levels}
                
                # Store correlations for each interaction order
                order_correlations = {order: {level: [] for level in noise_levels} 
                                     for order in range(2, k+1) if order in interaction_ranges}
                
                # Get original indices
                original_phi, original_inter = phi, inter_values
                
                for noise_level in noise_levels:
                    noise_indices_list = []
                    
                    # Run multiple noise tests
                    for n in range(n_repetitions):
                        # Apply Gaussian noise to training set
                        feature_stds = np.std(X_train_scaled, axis=0, keepdims=True)
                        noise = np.random.normal(0, 1, X_train_scaled.shape) * noise_level * feature_stds
                        X_train_noisy = X_train_scaled.copy() + noise
                        
                        # Transform noisy training data
                        X_train_noisy_shapley = choquet_k_additive_shapley(X_train_noisy, k)
                        
                        # Train new model on noisy training data
                        noise_model = LogisticRegression(
                            max_iter=1000,
                            random_state=random_state + n,
                            solver="newton-cg"
                        )
                        noise_model.fit(X_train_noisy_shapley, y_train_values)
                        
                        # Extract coefficients and compute indices
                        noise_coeffs = noise_model.coef_[0] if noise_model.coef_.shape[0] == 1 else noise_model.coef_
                        noisy_phi, noisy_inter = indices_from_shapley_coefficients(noise_coeffs, nAttr, k)
                        noise_indices_list.append((noisy_phi, noisy_inter))
                        
                        # Calculate correlation for singleton Shapley values (phi)
                        phi_corr = np.corrcoef(original_phi, noisy_phi)[0, 1]
                        phi_correlations[noise_level].append(phi_corr)
                        
                        # Calculate correlation for each order of interaction indices
                        for order in range(2, k+1):
                            if order in interaction_ranges:
                                start_idx, end_idx = interaction_ranges[order]
                                
                                if start_idx < end_idx and start_idx < len(original_inter) and end_idx <= len(original_inter):
                                    order_original = original_inter[start_idx:end_idx]
                                    
                                    if start_idx < len(noisy_inter) and end_idx <= len(noisy_inter):
                                        order_noisy = noisy_inter[start_idx:end_idx]
                                        
                                        if len(order_original) > 0 and len(order_noisy) > 0:
                                            try:
                                                order_corr = np.corrcoef(order_original, order_noisy)[0, 1]
                                                order_correlations[order][noise_level].append(order_corr)
                                            except:
                                                # Handle edge case where correlation cannot be computed
                                                pass
                    
                    noise_indices[noise_level] = noise_indices_list
                    
                    # Report Shapley value stability for this noise level
                    avg_phi_corr = np.mean(phi_correlations[noise_level])
                    print(f"  Noise level {noise_level}: Shapley value stability = {avg_phi_corr:.4f}")
                    
                    # Report stability for each interaction order
                    for order in range(2, k+1):
                        if order in order_correlations and order_correlations[order][noise_level]:
                            avg_order_corr = np.mean(order_correlations[order][noise_level])
                            print(f"  Noise level {noise_level}: Order-{order} interaction stability = {avg_order_corr:.4f}")
                
                # Calculate overall stability scores for Shapley values
                avg_phi_stability = np.mean([np.mean(phi_correlations[level]) for level in noise_levels])
                results_df.loc[k, 'shapley_stability'] = avg_phi_stability
                print(f"- Overall Shapley values stability score: {avg_phi_stability:.4f}")
                
                # Calculate overall stability scores for each interaction order
                for order in range(2, k+1):
                    if order in order_correlations:
                        # Only include noise levels that have data
                        valid_levels = [level for level in noise_levels if order_correlations[order][level]]
                        
                        if valid_levels:
                            avg_order_stability = np.mean([np.mean(order_correlations[order][level]) for level in valid_levels])
                            results_df.loc[k, f'order{order}_stability'] = avg_order_stability
                            print(f"- Overall Order-{order} interaction stability score: {avg_order_stability:.4f}")
                
                # Save detailed stability metrics for this k value
                stability_detail = pd.DataFrame({
                    'noise_level': np.repeat(noise_levels, n_repetitions),
                    'phi_correlation': [c for level in noise_levels for c in phi_correlations[level]]
                })
                
                # Add interaction correlations for each order
                for order in range(2, k+1):
                    if order in order_correlations:
                        order_data = []
                        for level in noise_levels:
                            # Use NaN if no data for this level
                            level_data = order_correlations[order][level] if order_correlations[order][level] else [np.nan] * n_repetitions
                            order_data.extend(level_data[:n_repetitions])  # Ensure consistent size
                        
                        # Pad with NaNs if needed to match the DataFrame size
                        while len(order_data) < len(stability_detail):
                            order_data.append(np.nan)
                        
                        stability_detail[f'order{order}_correlation'] = order_data[:len(stability_detail)]
                    
                stability_detail.to_csv(os.path.join(output_dir, f"stability_detail_k{k}.csv"), index=False)
                
            except Exception as e:
                print(f"Error processing k={k}: {str(e)}")
                # Fill with NaN to indicate missing data
                for col in results_df.columns[2:]:  # Skip k_value and n_params
                    results_df.loc[k, col] = np.nan
    
    except Exception as e:
        print(f"Error in compute_indices_robustness for dataset {dataset_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        # Return whatever we have so far
        return results_df, all_indices

    # Save complete results
    results_df.to_csv(os.path.join(output_dir, "indices_robustness_results.csv"))
    
    # Save indices for all k values
    for k, indices_k in all_indices.items():
        # Convert indices tuple to DataFrame for easy saving
        phi, inter_values = indices_k
        phi_df = pd.DataFrame({'Feature': range(len(phi)), 'Phi_Value': phi})
        phi_df.to_csv(os.path.join(output_dir, f"phi_values_k{k}.csv"), index=False)
        
        if len(inter_values) > 0:
            inter_df = pd.DataFrame({'Index': range(len(inter_values)), 'Interaction_Value': inter_values})
            inter_df.to_csv(os.path.join(output_dir, f"interaction_values_k{k}.csv"), index=False)
    
    # Create plots
    print("\nGenerating plots...")
    
    # 1. Number of parameters vs k plot (keep this unchanged)
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['k_value'], results_df['n_params'], 'o-', linewidth=2)
    plt.title(f'Number of Parameters vs k-additivity ({dataset_name})')
    plt.xlabel('k-additivity')
    plt.ylabel('Number of Parameters')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.savefig(os.path.join(output_dir, "parameter_counts.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Shapley values stability plot
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['k_value'], results_df['shapley_stability'], 'o-', linewidth=2, color='blue')
    plt.title(f'Shapley Values Stability vs k-additivity ({dataset_name})')
    plt.xlabel('k-additivity')
    plt.ylabel('Stability Score (Mean Correlation)')
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])  # Correlations are between 0 and 1
    plt.savefig(os.path.join(output_dir, "shapley_stability_vs_k.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Generate separate plots for each interaction order
    cmap = get_cmap('viridis')
    
    # First, find out which orders have data
    available_orders = []
    for order in range(2, nAttr + 1):
        col_name = f'order{order}_stability'
        if col_name in results_df.columns and not results_df[col_name].isnull().all():
            available_orders.append(order)
    
    # Plot each interaction order separately
    for order in available_orders:
        col_name = f'order{order}_stability'
        plt.figure(figsize=(10, 6))
        plt.plot(results_df['k_value'], results_df[col_name], 'o-', 
                linewidth=2, color=cmap(0.8 * order/nAttr))
        plt.title(f'Order-{order} Interaction Stability vs k-additivity ({dataset_name})')
        plt.xlabel('k-additivity')
        plt.ylabel('Stability Score (Mean Correlation)')
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1])
        plt.savefig(os.path.join(output_dir, f"order{order}_stability_vs_k.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Create a combined plot with all interaction orders
    plt.figure(figsize=(12, 8))
    
    # Plot Shapley values first
    plt.plot(results_df['k_value'], results_df['shapley_stability'], 'o-', 
            linewidth=2, color='black', label='Shapley Values (Order 1)')
    
    # Plot each interaction order
    for i, order in enumerate(available_orders):
        col_name = f'order{order}_stability'
        color = cmap(i / max(1, len(available_orders) - 1))
        plt.plot(results_df['k_value'], results_df[col_name], 'o-', 
                linewidth=2, color=color, label=f'Order {order} Interactions')
    
    plt.title(f'Stability by Interaction Order vs k-additivity ({dataset_name})')
    plt.xlabel('k-additivity')
    plt.ylabel('Stability Score (Mean Correlation)')
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])
    plt.legend(loc='best')
    plt.savefig(os.path.join(output_dir, "all_orders_stability_vs_k.png"), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Noise level comparison plot for Shapley values
    plt.figure(figsize=(12, 8))
    cmap = get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, 4))  # 4 noise levels
    
    noise_levels = [0.05, 0.1, 0.2, 0.3]
    
    # For each k value, get the average phi correlation at each noise level
    noise_stability_data = {}
    for k in range(1, nAttr + 1):
        if k not in all_indices:
            continue
        
        # Load the detailed stability data for this k
        try:
            detail_file = os.path.join(output_dir, f"stability_detail_k{k}.csv")
            if os.path.exists(detail_file):
                detail_data = pd.read_csv(detail_file)
                
                # Calculate average stability by noise level
                for i, noise_level in enumerate(noise_levels):
                    noise_data = detail_data[detail_data['noise_level'] == noise_level]
                    if not noise_data.empty:
                        avg_stability = noise_data['phi_correlation'].mean()
                        if k not in noise_stability_data:
                            noise_stability_data[k] = {}
                        noise_stability_data[k][noise_level] = avg_stability
        except Exception as e:
            print(f"Error loading stability data for k={k}: {e}")
    
    # Plot stability vs noise level for each k for Shapley values
    for i, noise_level in enumerate(noise_levels):
        k_values = []
        stability_values = []
        
        for k, stability_dict in noise_stability_data.items():
            if noise_level in stability_dict:
                k_values.append(k)
                stability_values.append(stability_dict[noise_level])
        
        if k_values:
            plt.plot(k_values, stability_values, 'o-', 
                    color=colors[i], linewidth=2, label=f'Noise level: {noise_level}')
    
    plt.title(f'Shapley Values Stability at Different Noise Levels ({dataset_name})')
    plt.xlabel('k-additivity')
    plt.ylabel('Shapley Values Stability (Correlation)')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])
    plt.savefig(os.path.join(output_dir, "noise_level_shapley_stability.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Create noise level plots for each interaction order
    for order in available_orders:
        # Calculate noise level stability for this interaction order
        order_noise_data = {}
        
        for k in range(order, nAttr + 1):  # Order-n interactions only exist for k ≥ n
            try:
                detail_file = os.path.join(output_dir, f"stability_detail_k{k}.csv")
                if os.path.exists(detail_file):
                    detail_data = pd.read_csv(detail_file)
                    
                    # Check if data for this order exists
                    col_name = f'order{order}_correlation'
                    if col_name in detail_data.columns:
                        for noise_level in noise_levels:
                            noise_data = detail_data[detail_data['noise_level'] == noise_level]
                            if not noise_data.empty and not noise_data[col_name].isnull().all():
                                avg_stability = noise_data[col_name].mean()
                                if not np.isnan(avg_stability):
                                    if k not in order_noise_data:
                                        order_noise_data[k] = {}
                                    order_noise_data[k][noise_level] = avg_stability
            except Exception as e:
                print(f"Error processing order-{order} data for k={k}: {e}")
        
        if order_noise_data:
            plt.figure(figsize=(12, 8))
            
            for i, noise_level in enumerate(noise_levels):
                k_values = []
                stability_values = []
                
                for k, stability_dict in order_noise_data.items():
                    if noise_level in stability_dict:
                        k_values.append(k)
                        stability_values.append(stability_dict[noise_level])
                
                if k_values:
                    plt.plot(k_values, stability_values, 'o-', 
                            color=colors[i], linewidth=2, label=f'Noise level: {noise_level}')
            
            plt.title(f'Order-{order} Interaction Stability at Different Noise Levels ({dataset_name})')
            plt.xlabel('k-additivity')
            plt.ylabel(f'Order-{order} Interaction Stability (Correlation)')
            plt.legend(loc='best')
            plt.grid(True, alpha=0.3)
            plt.ylim([0, 1])
            plt.savefig(os.path.join(output_dir, f"noise_level_order{order}_stability.png"), 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    # 7. Visualize top Shapley values (phi) for different k values
    for k in [1, min(2, nAttr), min(3, nAttr), nAttr]:
        if k not in all_indices:
            continue
        
        phi, inter_values = all_indices[k]
        
        # Sort phi values by absolute magnitude
        phi_indices = np.argsort(np.abs(phi))[::-1]
        top_n = min(10, len(phi))
        
        if top_n > 0:
            plt.figure(figsize=(12, 6))
            
            # Extract top indices and their values
            feature_names = [f"X{i}" for i in phi_indices[:top_n]]
            values = phi[phi_indices[:top_n]]
            
            # Create bar chart
            bars = plt.bar(range(top_n), [abs(val) for val in values], 
                          color=['blue' if val >= 0 else 'red' for val in values])
            plt.xticks(range(top_n), feature_names, rotation=45, ha='right')
            plt.title(f'Top {top_n} Shapley Values (k={k})')
            plt.xlabel('Feature')
            plt.ylabel('Absolute Shapley Value')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"top_shapley_k{k}.png"), dpi=300, bbox_inches='tight')
            plt.close()
    
    # 8. Visualize top interaction indices for each order separately
    print("- Generating histograms for top interaction indices by order...")
    
    # For each k value that has interactions
    for k in range(2, nAttr + 1):
        if k not in all_indices:
            continue
        
        _, inter_values = all_indices[k]
        if len(inter_values) == 0:
            continue
        
        # Get ranges for each interaction order
        interaction_ranges = get_interaction_ranges(nAttr, k)
        
        # For each interaction order (from 2 to k)
        for order in range(2, k + 1):
            if order not in interaction_ranges:
                continue
                
            start_idx, end_idx = interaction_ranges[order]
            
            # Extract interactions of this order
            if start_idx < end_idx and start_idx < len(inter_values) and end_idx <= len(inter_values):
                order_interactions = inter_values[start_idx:end_idx]
                
                # If we have interactions of this order
                if len(order_interactions) > 0:
                    # Sort by absolute magnitude
                    order_inter_indices = np.argsort(np.abs(order_interactions))[::-1]
                    top_n = min(10, len(order_interactions))
                    
                    if top_n > 0:
                        plt.figure(figsize=(12, 6))
                        
                        # Create meaningful labels for interactions
                        interaction_labels = []
                        for rel_idx in order_inter_indices[:top_n]:
                            idx = start_idx + rel_idx  # Convert to global index
                            try:
                                features = get_interaction_features(idx, nAttr)
                                label = f"I{''.join(str(f) for f in features)}"
                            except Exception:
                                label = f"I{idx}"
                            interaction_labels.append(label)
                        
                        values = order_interactions[order_inter_indices[:top_n]]
                        
                        # Create bar chart
                        bars = plt.bar(range(top_n), [abs(val) for val in values], 
                                      color=['blue' if val >= 0 else 'red' for val in values])
                        plt.xticks(range(top_n), interaction_labels, rotation=45, ha='right')
                        plt.title(f'Top {top_n} Order-{order} Interaction Indices (k={k})')
                        plt.xlabel('Interaction')
                        plt.ylabel('Absolute Interaction Value')
                        plt.grid(True, alpha=0.3)
                        plt.tight_layout()
                        plt.savefig(os.path.join(output_dir, f"top_order{order}_interactions_k{k}.png"), 
                                  dpi=300, bbox_inches='tight')
                        plt.close()
    
    return results_df, all_indices

def run_batch_indices_analysis(datasets):
    """
    Run analysis on multiple datasets and return summary statistics.
    
    Parameters:
    datasets - Either a dictionary {name: (X, y)} or a list of (name, X, y) tuples
    
    Returns:
    summary_df - DataFrame with summary metrics
    all_indices - Dictionary with detailed indices for each dataset
    """
    results = []
    all_indices = {}
    
    # Handle datasets whether it's a dictionary or a list
    if isinstance(datasets, dict):
        dataset_items = [(name, data) for name, data in datasets.items()]
    else:
        dataset_items = [(name, (X, y)) for name, X, y in datasets]
    
    for dataset_name, (X, y) in dataset_items:
        try:
            # Create dataset-specific subdirectory
            dataset_dir = os.path.join("indices_robustness", dataset_name)
            
            # Compute indices robustness for this dataset
            result_df, indices = compute_indices_robustness(
                (dataset_name, X, y), output_dir=dataset_dir
            )
            
            if result_df is not None and not result_df.empty:
                all_indices[dataset_name] = indices
                
                # Append results to summary DataFrame
                for k in result_df.index:
                    row = result_df.loc[k]
                    new_row = {
                        'dataset': dataset_name,
                        'k_value': k,
                        'n_params': row.get('n_params', np.nan),
                        'train_time': row.get('train_time', np.nan),
                        'baseline_accuracy': row.get('baseline_accuracy', np.nan),
                    }
                    
                    # Handle renamed column
                    if 'shapley_stability' in row:
                        new_row['shapley_stability'] = row['shapley_stability']
                    elif 'indices_stability' in row:
                        new_row['shapley_stability'] = row['indices_stability']
                    else:
                        new_row['shapley_stability'] = np.nan
                        
                    results.append(new_row)
            else:
                print(f"Warning: No results returned for dataset {dataset_name}")
        
        except Exception as e:
            print(f"Error processing dataset {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Convert results to DataFrame
    summary_df = pd.DataFrame(results)
    
    # Save overall summary
    if not summary_df.empty:
        summary_df.to_csv(os.path.join("indices_robustness", "overall_summary.csv"), index=False)
    
    # Create cross-dataset comparison plots
    try:
        if not summary_df.empty and 'shapley_stability' in summary_df.columns:
            # Stability vs k for all datasets
            plt.figure(figsize=(12, 8))
            for name in summary_df['dataset'].unique():
                dataset_summary = summary_df[summary_df['dataset'] == name]
                if 'shapley_stability' in dataset_summary.columns and not dataset_summary['shapley_stability'].isnull().all():
                    plt.plot(dataset_summary['k_value'], dataset_summary['shapley_stability'], 'o-', 
                            linewidth=2, label=name)
            
            plt.title('Shapley Values Stability Across Datasets')
            plt.xlabel('k-additivity')
            plt.ylabel('Stability Score (Mean Correlation)')
            plt.grid(True, alpha=0.3)
            plt.legend(loc='best')
            plt.ylim([0, 1])
            plt.savefig(os.path.join("indices_robustness", "cross_dataset_stability.png"), dpi=300, bbox_inches='tight')
            plt.close()
    except Exception as e:
        print(f"Error creating cross-dataset plots: {e}")
    
    return summary_df, all_indices

if __name__ == "__main__":
    # Define datasets first
    datasets = []
    
    # Load your datasets here
    # For example:
    # dataset_names = ["dados_covid_sbpo_atual", "banknotes", "transfusion", ...]
    # for dataset in dataset_names:
    #     base_x, base_y = load_dataset(dataset)
    #     datasets.append((dataset, base_x, base_y))
    
    # Then check its type and process it
    if isinstance(datasets, list):
        datasets_dict = {name: (X, y) for name, X, y in datasets}
        summary, all_indices = run_batch_indices_analysis(datasets_dict)
    else:
        summary, all_indices = run_batch_indices_analysis(datasets)
    dataset_list = [ "pure_pairwise_interaction","triplet_interaction",
    'dados_covid_sbpo_atual','banknotes','transfusion','mammographic','raisin','rice','diabetes','skin']
    datasets = {}
    for dataset in dataset_list:
        base_x, base_y = func_read_data(dataset)
        datasets[dataset] = (base_x, base_y)

    # 3) Run batch analysis - use the batch function instead of the individual function
    summary, all_indices = run_batch_indices_analysis(datasets)
    print("\nBatch analysis completed. Summary of results:")
    print(summary)
    dataset_list = ['dados_covid_sbpo_atual','banknotes','transfusion','mammographic','raisin','rice','diabetes','skin']
    datasets = []
    for dataset in dataset_list:
        base_x, base_y = func_read_data(dataset)
        datasets.append((dataset, base_x, base_y))

    # 3) Run batch analysis - use the batch function instead of the individual function
    summary, all_indices = run_batch_indices_analysis(datasets)
    print("\nBatch analysis completed. Summary of results:")
    print(summary)
    datasets.append((dataset, base_x, base_y))

    # 3) Run batch analysis - use the batch function instead of the individual function
    summary, all_indices = run_batch_indices_analysis(datasets)
    print("\nBatch analysis completed. Summary of results:")
    print(summary)
