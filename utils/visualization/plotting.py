"""
Visualization utilities for PIC-I models.

This module provides functions for visualizing model results, including:
- Shapley values for different representations (game, Mobius, Shapley)
- Interaction matrices
- Coefficient plots
- Performance metrics
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from os.path import join
import itertools


def ensure_folder(folder_path):
    """
    Ensure that a folder exists, creating it if necessary.
    
    Parameters
    ----------
    folder_path : str
        Path to the folder to ensure exists
    """
    os.makedirs(folder_path, exist_ok=True)


def plot_horizontal_bar(names, values, title, xlabel, filename=None, color="steelblue", std=None):
    """
    Create a horizontal bar plot.
    
    Parameters
    ----------
    names : list
        Names for the bars
    values : array-like
        Values for the bars
    title : str
        Plot title
    xlabel : str
        X-axis label
    filename : str, optional
        Path to save the plot
    color : str, default="steelblue"
        Color for the bars
    std : array-like, optional
        Standard deviations for error bars
    """
    # Sort by absolute value
    idx = np.argsort(np.abs(values))[::-1]
    sorted_names = np.array(names)[idx]
    sorted_values = np.array(values)[idx]
    
    # Create plot
    fig_height = max(6, len(names) * 0.3)
    plt.figure(figsize=(10, fig_height))
    
    # Plot bars
    if std is not None:
        sorted_std = np.array(std)[idx]
        plt.barh(sorted_names, sorted_values, xerr=sorted_std, color=color, edgecolor="black", error_kw={"ecolor": "black", "capsize": 5})
    else:
        plt.barh(sorted_names, sorted_values, color=color, edgecolor="black")
    
    # Add labels and title
    plt.xlabel(xlabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    # Save if filename provided
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# ============================================================================
# Shapley representation visualization functions
# ============================================================================


def plot_coefficients(feature_names, all_coefficients, plot_folder, k_add):
    """
    Plot model coefficients.

    Parameters
    ----------
    feature_names : list
        Names of features
    all_coefficients : list of arrays
        Coefficients from multiple runs
    plot_folder : str
        Folder to save the plot
    k_add : int
        k-additivity level
    """
    if not all_coefficients:
        print("No coefficients computed; skipping plot.")
        return
    
    all_coeffs_arr = np.vstack(all_coefficients)
    mean_coeffs = np.mean(all_coeffs_arr, axis=0)
    std_coeffs = np.std(all_coeffs_arr, axis=0)
    
    # Print value range for verification
    print(f"Coefficients values range: {np.min(mean_coeffs):.4f} to {np.max(mean_coeffs):.4f}")
    print(f"Coefficients average magnitude: {np.mean(np.abs(mean_coeffs)):.4f}")
    
    filename = join(plot_folder, f"coefficients_k{k_add}.png")
    plot_horizontal_bar(
        names=feature_names,
        values=mean_coeffs,
        std=std_coeffs,
        title=f"Average Model Coefficients (k={k_add})",
        xlabel="Average Coefficient Value",
        filename=filename,
        color="seagreen"
    )
    print("Saved coefficients plot to:", filename)


def plot_interaction_matrix_2add(feature_names, all_interaction_matrices, plot_folder):
    """
    Plot interaction matrix for 2-additive Shapley representation.
    
    Parameters
    ----------
    feature_names : list
        Names of features
    all_interaction_matrices : list of arrays
        Interaction matrices from multiple runs
    plot_folder : str
        Folder to save the plot
    """
    if not all_interaction_matrices:
        print("No 2-additive Shapley interaction matrices computed; skipping plot.")
        return
    
    mean_interaction_matrix = np.mean(np.array(all_interaction_matrices), axis=0)
    
    # Print range information
    print(f"2-additive Shapley interaction matrix range: {np.min(mean_interaction_matrix):.4f} to {np.max(mean_interaction_matrix):.4f}")
    print(f"2-additive Shapley interaction average magnitude: {np.mean(np.abs(mean_interaction_matrix)):.4f}")
    
    # Create plot
    plt.figure(figsize=(8, 6))
    plt.imshow(mean_interaction_matrix, cmap="viridis", interpolation="nearest")
    plt.colorbar(orientation="vertical", label="Interaction Value")
    plt.xticks(range(len(feature_names)), feature_names, rotation=90, fontsize=12)
    plt.yticks(range(len(feature_names)), feature_names, fontsize=12)
    plt.title("Average Interaction Effects Matrix (2-additive Shapley)", fontsize=16)
    plt.tight_layout()
    
    # Save plot
    plot_path = join(plot_folder, "interaction_matrix_2add.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved 2-additive Shapley interaction matrix plot to:", plot_path)

# ============================================================================
# General visualization functions
# ============================================================================

def plot_model_performance_comparison(results, plot_folder, title="Model Performance Comparison"):
    """
    Plot performance comparison of multiple models.
    
    Parameters
    ----------
    results : dict
        Dictionary of model results
    plot_folder : str
        Folder to save the plot
    title : str, default="Model Performance Comparison"
        Plot title
    """
    if not results or "baseline" not in results:
        print("No performance results to plot; skipping.")
        return
    
    # Extract model names and metrics
    model_names = list(results["baseline"].keys())
    metrics = results.get("metrics", ["accuracy", "roc_auc", "f1"])
    
    # Create figure with subplots for each metric
    fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 6))
    if len(metrics) == 1:
        axes = [axes]
    
    # Plot each metric
    for i, metric in enumerate(metrics):
        values = [results["baseline"][model].get(metric, 0) for model in model_names]
        axes[i].bar(model_names, values, color="cornflowerblue", edgecolor="black")
        axes[i].set_title(f"{metric.replace('_', ' ').title()}", fontsize=14)
        axes[i].set_ylim(0, 1.05)
        axes[i].grid(axis="y", linestyle="--", alpha=0.3)
        axes[i].tick_params(axis="x", rotation=45)
    
    # Add overall title and adjust layout
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    # Save plot
    plot_path = join(plot_folder, "model_performance_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved model performance comparison plot to:", plot_path)


def plot_noise_robustness(results, plot_folder):
    """
    Plot noise robustness of multiple models.
    
    Parameters
    ----------
    results : dict
        Dictionary of model results
    plot_folder : str
        Folder to save the plot
    """
    if not results or "noise_robustness" not in results:
        print("No noise robustness results to plot; skipping.")
        return
    
    # Extract model names and noise levels
    model_names = list(results["noise_robustness"].keys())
    if not model_names:
        print("No models with noise robustness results; skipping.")
        return
    
    # Get noise levels from first model
    noise_levels = sorted([float(level) for level in results["noise_robustness"][model_names[0]].keys()])
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot each model
    for model in model_names:
        noise_values = []
        for level in noise_levels:
            level_str = str(level)
            if level_str in results["noise_robustness"][model]:
                noise_values.append(results["noise_robustness"][model][level_str]["accuracy"])
            else:
                # Try with different precision
                level_str = f"{level:.1f}"
                if level_str in results["noise_robustness"][model]:
                    noise_values.append(results["noise_robustness"][model][level_str]["accuracy"])
                else:
                    noise_values.append(np.nan)
        
        plt.plot(noise_levels, noise_values, 'o-', linewidth=2, markersize=8, label=model)
    
    # Add labels and legend
    plt.xlabel("Noise Level", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.title("Model Robustness to Noise", fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=12)
    
    # Save plot
    plot_path = join(plot_folder, "noise_robustness.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved noise robustness plot to:", plot_path)


def plot_k_additivity_results(results_df, plot_folder, dataset_name, representation):
    """
    Plot k-additivity analysis results.
    
    Parameters
    ----------
    results_df : pandas.DataFrame
        Results dataframe from k-additivity analysis
    plot_folder : str
        Folder to save the plots
    dataset_name : str
        Name of the dataset
    representation : str
        Representation used (game or mobius)
    """
    # Ensure folder exists
    ensure_folder(plot_folder)
    
    # Plot accuracy vs k
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['k_value'], results_df['baseline_accuracy'], 'o-', linewidth=2, markersize=8, color="royalblue")
    plt.xlabel("k-additivity", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.title(f"Accuracy vs k-additivity ({dataset_name}, {representation} representation)", fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xticks(results_df['k_value'])
    
    # Save plot
    plot_path = join(plot_folder, f"accuracy_vs_k_{dataset_name}_{representation}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved accuracy vs k-additivity plot to: {plot_path}")
    
    # Plot training time vs k
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['k_value'], results_df['train_time'], 'o-', linewidth=2, markersize=8, color="firebrick")
    plt.xlabel("k-additivity", fontsize=14)
    plt.ylabel("Training Time (seconds)", fontsize=14)
    plt.title(f"Training Time vs k-additivity ({dataset_name}, {representation} representation)", fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xticks(results_df['k_value'])
    
    # Save plot
    plot_path = join(plot_folder, f"time_vs_k_{dataset_name}_{representation}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved training time vs k-additivity plot to: {plot_path}")
    
    # Plot parameter count vs k
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['k_value'], results_df['n_params'], 'o-', linewidth=2, markersize=8, color="darkgreen")
    plt.xlabel("k-additivity", fontsize=14)
    plt.ylabel("Number of Parameters", fontsize=14)
    plt.title(f"Parameter Count vs k-additivity ({dataset_name}, {representation} representation)", fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xticks(results_df['k_value'])
    
    # Save plot
    plot_path = join(plot_folder, f"params_vs_k_{dataset_name}_{representation}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved parameter count vs k-additivity plot to: {plot_path}")
    
    # Plot noise robustness vs k
    if 'noise_0.1' in results_df.columns:
        plt.figure(figsize=(10, 6))
        for noise_level in [0.05, 0.1, 0.2, 0.3]:
            if f'noise_{noise_level}' in results_df.columns:
                plt.plot(results_df['k_value'], results_df[f'noise_{noise_level}'], 'o-', 
                         linewidth=2, markersize=8, label=f"Noise {noise_level}")
        
        plt.xlabel("k-additivity", fontsize=14)
        plt.ylabel("Accuracy under Noise", fontsize=14)
        plt.title(f"Noise Robustness vs k-additivity ({dataset_name}, {representation} representation)", fontsize=16)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.xticks(results_df['k_value'])
        plt.legend(fontsize=12)
        
        # Save plot
        plot_path = join(plot_folder, f"noise_vs_k_{dataset_name}_{representation}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved noise robustness vs k-additivity plot to: {plot_path}")
