import numpy as np
import matplotlib.pyplot as plt
import itertools
import os
from os.path import join
from simulation_helper_functions import plot_horizontal_bar, ensure_folder
from itertools import combinations

def plot_shapley_full(feature_names, all_shapley_full, plot_folder):
    if not all_shapley_full:
        print("No Full Choquet Shapley values computed; skipping plot.")
        return
    all_shapley_full_arr = np.vstack(all_shapley_full)
    mean_shapley_full = np.mean(all_shapley_full_arr, axis=0)
    
    # Print value range for verification
    print(f"Full Shapley values range: {np.min(mean_shapley_full):.4f} to {np.max(mean_shapley_full):.4f}")
    print(f"Full Shapley average magnitude: {np.mean(np.abs(mean_shapley_full)):.4f}")
    
    filename = join(plot_folder, "shapley_full.png")
    plot_horizontal_bar(
        names=feature_names,
        values=mean_shapley_full,
        title="Average Shapley Values (Full Choquet Model)",
        xlabel="Average Shapley Value",
        filename=filename,
        color="steelblue"
    )
    print("Saved plot to:", filename)

def plot_shapley_2add(feature_names, all_shapley_2add, plot_folder):
    if not all_shapley_2add:
        print("No Choquet 2-add Shapley values computed; skipping plot.")
        return
    all_shapley_2add_arr = np.vstack(all_shapley_2add)
    mean_shapley_2add = np.mean(all_shapley_2add_arr, axis=0)
    std_shapley_2add = np.std(all_shapley_2add_arr, axis=0)
    
    # Print value range for verification  
    print(f"2-add Shapley values range: {np.min(mean_shapley_2add):.4f} to {np.max(mean_shapley_2add):.4f}")
    print(f"2-add Shapley average magnitude: {np.mean(np.abs(mean_shapley_2add)):.4f}")
    
    filename = join(plot_folder, "shapley_2add.png")
    plot_horizontal_bar(
        names=feature_names,
        values=mean_shapley_2add,
        std=std_shapley_2add,
        title="Average Shapley Values (Choquet 2-add Model)",
        xlabel="Average Shapley Value",
        filename=filename,
        color="seagreen"
    )
    print("Saved Choquet 2-add Shapley values plot to:", filename)

def plot_marginal_2add(feature_names, all_marginal_2add, plot_folder):
    if not all_marginal_2add:
        print("No marginal contributions computed; skipping plot.")
        return
    all_marginal_2add_arr = np.vstack(all_marginal_2add)
    mean_marginal_2add = np.mean(all_marginal_2add_arr, axis=0)
    std_marginal_2add = np.std(all_marginal_2add_arr, axis=0)
    filename = join(plot_folder, "marginal_2add.png")
    plot_horizontal_bar(
        names=feature_names,
        values=mean_marginal_2add,
        std=std_marginal_2add,
        title="Average Marginal Contributions (Direct Main Effects - Choquet 2-add)",
        xlabel="Average Marginal Contribution",
        filename=filename,
        color="darkorange"
    )
    print("Saved marginal contributions plot to:", filename)

def plot_coef(X_or_feature_names, mean_coef, plot_folder, model_type="choquet", k_add=None, intercept=None):
    """
    Plot coefficients for any k-additive model (Choquet or MLM).
    
    Parameters:
    -----------
    X_or_feature_names : array-like or list
        Either the feature matrix (for full models) or feature names list (for k-additive)
    mean_coef : array-like
        Average coefficient values across simulations
    plot_folder : str
        Directory to save the plot
    model_type : str, default="choquet"
        Type of model: "choquet" or "mlm"
    k_add : int, default=None
        If None, plot full model coefficients. Otherwise, plot k-additive model coefficients.
    intercept : float, default=None
        The bias/intercept term to include in the plot
    """
    model_desc = f"{k_add}-add" if k_add else "full"
    if mean_coef is None or mean_coef.size == 0:
        print(f"No {model_type} {model_desc} regression coefficients computed; skipping plot.")
        return

    if isinstance(X_or_feature_names, list) or (hasattr(X_or_feature_names, 'ndim') and X_or_feature_names.ndim == 1):
        feature_names = X_or_feature_names
        n_features = len(feature_names)
    else:
        X = X_or_feature_names
        n_features = X.shape[1]
        feature_names = [f"Feature {i+1}" for i in range(n_features)]
    
    # Set style based on model type and k-additivity
    if model_type.lower() == "choquet":
        color = "gray"
        if k_add is None:
            title = "Average Regression Coefficients (Choquet Model)"
            filename = "coef_full.png"
        else:
            title = f"Average Regression Coefficients (Choquet {k_add}-add Model)"
            filename = f"coef_{k_add}add.png"
    else:  # MLM
        color = "cornflowerblue"
        if k_add is None:
            title = "Average Regression Coefficients (MLM Model)"
            filename = "coef_mlm_full.png"
        else:
            title = f"Average Regression Coefficients (MLM {k_add}-add Model)"
            filename = f"coef_mlm_{k_add}add.png"
    
    # Generate all coalition labels up to size k (or all if k is None)
    all_labels = []
    if k_add:
        singleton_labels = feature_names
        interaction_labels = []
        
        # Add singletons
        all_labels.extend(singleton_labels)
        
        # Add interaction terms for k > 1
        if k_add > 1:
            for r in range(2, k_add + 1):
                for combo in itertools.combinations(range(n_features), r):
                    if model_type.lower() == "choquet":
                        interaction_labels.append(",".join(feature_names[i] for i in combo))
                    else:  # MLM
                        interaction_labels.append("×".join(feature_names[i] for i in combo))
            all_labels.extend(interaction_labels)
    else:
        all_coalitions = []
        for r in range(1, n_features + 1):
            all_coalitions.extend(list(itertools.combinations(range(n_features), r)))
        
        if model_type.lower() == "choquet":
            all_labels = [",".join(feature_names[i] for i in coalition) for coalition in all_coalitions]
        else:
            for coalition in all_coalitions:
                label_parts = []
                # Add active features
                active_parts = [feature_names[i] for i in coalition]
                label_parts.append(",".join(active_parts))
                # Add complement features if not full set
                if len(coalition) < n_features:
                    complement_parts = [f"¬{feature_names[i]}" for i in range(n_features) if i not in coalition]
                    if complement_parts:
                        label_parts.append(",".join(complement_parts))
                all_labels.append(" × ".join(label_parts))
    
    # Add bias
    coef_values = mean_coef.copy()
    if intercept is not None:
        all_labels = ["Bias"] + list(all_labels)
        coef_values = np.insert(coef_values, 0, intercept)
    
    # Sort by absolute value
    all_labels = np.array(all_labels)
    indices_sorted = np.argsort(np.abs(coef_values))[::-1]
    sorted_labels = all_labels[indices_sorted]
    sorted_values = coef_values[indices_sorted]
    
    # Create plot
    fig_height = max(6, len(sorted_labels) * 0.15)
    plt.figure(figsize=(10, fig_height))
    plt.barh(sorted_labels, sorted_values, color=color, edgecolor="black")
    plt.xlabel("Regression Coefficient")
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plot_path = join(plot_folder, filename)
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Saved {model_type} {model_desc} regression coefficients plot to:", plot_path)
    
    print(f"Saved {model_type} {model_desc} regression coefficients plot to:", plot_path)

def plot_interaction_matrix(X, feature_names, all_interaction_matrices, plot_folder, method):
    if not all_interaction_matrices:
        print("No interaction effects computed; skipping plot.")
        return
    mean_interaction_matrix = np.mean(np.array(all_interaction_matrices), axis=0)
    
    # Print range information but don't let it affect plot scaling
    print(f"{method} interaction matrix range: {np.min(mean_interaction_matrix):.4f} to {np.max(mean_interaction_matrix):.4f}")
    print(f"{method} interaction average magnitude: {np.mean(np.abs(mean_interaction_matrix)):.4f}")
    
    # Use consistent visualization settings to match previous version
    plt.figure(figsize=(8, 6))  # Keep original size
    plt.imshow(mean_interaction_matrix, cmap="viridis", interpolation="nearest")  # Keep original colormap
    plt.colorbar(orientation="vertical", label="Interaction Value")
    plt.xticks(range(X.shape[1]), feature_names, rotation=90, fontsize=12)
    plt.yticks(range(X.shape[1]), feature_names, fontsize=12)
    plt.title(f"Average Interaction Effects Matrix ({method})", fontsize=16)
    plt.tight_layout()
    plot_path = join(plot_folder, f"interaction_matrix_{method}.png")
    plt.savefig(plot_path)
    plt.close()
    print("Saved interaction effects plot to:", plot_path)

def plot_log_odds_hist(all_log_odds, log_odds_bins, plot_folder):
    if not all_log_odds:
        print("No log-odds computed; skipping plot.")
        return
    all_log_odds_concat = np.concatenate(all_log_odds)
    plt.figure(figsize=(10, 6))
    plt.hist(all_log_odds_concat, bins=log_odds_bins, color="mediumseagreen", edgecolor="black")
    plt.xlabel("Log-Odds", fontsize=16)
    plt.ylabel("Frequency", fontsize=16)
    plt.title("Log-Odds Distribution (Choquet 2-add)", fontsize=18)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plot_path = join(plot_folder, "logodds_hist.png")
    plt.savefig(plot_path)
    plt.close()
    print("Saved log-odds histogram to:", plot_path)

def plot_log_odds_vs_prob(all_log_odds, all_probs, plot_folder):
    if not all_log_odds or not all_probs:
        print("No log-odds or predicted probabilities computed; skipping plot.")
        return
    all_log_odds_concat = np.concatenate(all_log_odds)
    all_probs_concat = np.concatenate(all_probs, axis=0)
    plt.figure(figsize=(10, 6))
    plt.scatter(all_log_odds_concat, all_probs_concat[:, 1], alpha=0.7, color="darkorange", edgecolor="k")
    plt.xlabel("Log-Odds", fontsize=16)
    plt.ylabel("Predicted Probability (Positive)", fontsize=16)
    plt.title("Log-Odds vs. Predicted Probability", fontsize=18)
    plt.grid(axis="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plot_path = join(plot_folder, "logodds_vs_prob.png")
    plt.savefig(plot_path)
    plt.close()
    print("Saved log-odds vs. probability plot to:", plot_path)

def plot_shapley_vs_interaction(feature_names, shapleys, interaction_values, plot_folder, method):
    if shapleys is None or (hasattr(shapleys, '__len__') and len(shapleys) == 0):
        print("No Shapley values computed for Choquet method; skipping comparison plot.")
        return
    mean_shapley = np.mean(shapleys, axis=0)
    mean_banzhaf = np.mean(interaction_values, axis=0)
    m = len(feature_names)
    indices = np.arange(m)
    width = 0.4
    plt.figure(figsize=(12, 6))
    plt.bar(indices - width/2, mean_shapley, width, color="dodgerblue", label="Shapley")
    plt.bar(indices + width/2, mean_banzhaf, width, color="crimson", label="Interaction")
    plt.xticks(indices, feature_names, rotation=45, fontsize=12)
    plt.xlabel("Features", fontsize=14)
    plt.ylabel("Value", fontsize=14)
    plt.title(f"Shapley vs. Interaction Comparison ({method})", fontsize=16)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plot_path = join(plot_folder, f"shapley_vs_interaction_{method}.png")
    plt.savefig(plot_path)
    plt.close()
    print("Saved Shapley vs. Banzhaf comparison plot to:", plot_path)

def plot_test_accuracy(model_names, all_sim_results, plot_folder):
    if not model_names:
        print("No model names available for test accuracy plot.")
        return
    # all_sim_results is a dict with keys as model names and values as lists of simulation results
    acc_data = {model: [result.get("test_acc") for result in all_sim_results.get(model, [])]
                for model in model_names}
    avg_acc = {model: np.mean([acc for acc in acc_data[model] if acc is not None])
               for model in model_names}
    colors = plt.get_cmap("Set2").colors
    plt.figure(figsize=(10, 6))
    x = np.arange(len(model_names))
    width = 0.8
    for i, model in enumerate(model_names):
        plt.bar(x[i], avg_acc[model], width=width,
                color=colors[i % len(colors)], edgecolor="black", label=model)
    plt.xlabel("Model", fontsize=14)
    plt.ylabel("Test Accuracy", fontsize=14)
    plt.title("Average Test Accuracy Across Models", fontsize=16)
    plt.xticks(x, model_names, fontsize=12)
    plt.ylim(0, 1)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plot_path = join(plot_folder, "test_accuracy.png")
    plt.savefig(plot_path)
    plt.close()
    print("Saved test accuracy plot to:", plot_path)

def plot_decision_boundary(X, y, model, filename):
    from matplotlib.colors import ListedColormap
    X = np.array(X)
    y = np.array(y)
    cmap_light = ListedColormap(["#FFCCCC", "#CCFFCC"])
    cmap_bold = ListedColormap(["#FF0000", "#00AA00"])
    if X.shape[1] != 2:
        print("Decision boundary plot only works for 2D data.")
        return
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=cmap_light)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor="k", s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Decision Boundary for 2D Data")
    plt.savefig(filename)
    plt.close()
    print("Saved decision boundary plot to:", filename)

def plot_overall_interaction(feature_names, method_dict, title, plot_folder):
    """Plot overall interaction indices using different methods."""
    n_methods = len(method_dict)
    
    # Create a wider figure for better spacing
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set up bar positions
    bar_width = 0.8 / n_methods
    opacity = 0.8
    
    # Sort features by the average across methods
    avg_values = np.mean([values for values in method_dict.values()], axis=0)
    sorted_indices = np.argsort(avg_values)
    sorted_features = [feature_names[i] for i in sorted_indices]
    
    # Plot each method's bars - CHANGE FROM bar TO barh
    for i, (method_name, values) in enumerate(method_dict.items()):
        sorted_values = values[sorted_indices]
        # For horizontal bars, we flip the coordinates
        pos = np.arange(len(sorted_features))
        # Position bars along y-axis with proper spacing
        y_pos = pos - (n_methods-1)/2 * bar_width + i * bar_width
        ax.barh(y_pos, sorted_values, bar_width, alpha=opacity, label=method_name)
    
    # Set up axes and labels - X and Y are now correct for horizontal bars
    ax.set_xlabel('Interaction Index Value', fontsize=12)
    ax.set_yticks(range(len(sorted_features)))
    
    # Adjust label size based on number of features
    label_fontsize = max(6, min(12, 200 / len(sorted_features)))
    ax.set_yticklabels(sorted_features, fontsize=label_fontsize)
    
    ax.set_title(title, fontsize=14)
    ax.legend(loc='best')
    
    # Use explicit padding instead of tight_layout
    plt.subplots_adjust(left=0.25, right=0.95, top=0.95, bottom=0.1)
    
    plt.savefig(join(plot_folder, 'Overall_Interaction_Comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_interaction_comparison(feature_names, choquet_interaction, mlm_interaction, plot_folder, method_suffix="2add"):
    """
    Compare interaction matrices between Choquet and MLM models side by side.
    
    Parameters:
    -----------
    feature_names : list
        Names of the features
    choquet_interaction : list of numpy.ndarray
        Interaction matrices for Choquet model
    mlm_interaction : list of numpy.ndarray
        Interaction matrices for MLM model
    plot_folder : str
        Directory to save the plot
    method_suffix : str, default="2add"
        Suffix for the filename, indicating if it's "full" or "2add"
    """
    # Average over simulations
    avg_choquet = np.mean(np.array(choquet_interaction), axis=0)
    avg_mlm = np.mean(np.array(mlm_interaction), axis=0)
    
    # Create a figure with two subplots - use consistent sizing from previous version
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Plot Choquet interaction - use original "viridis" colormap without custom scaling
    im1 = axes[0].imshow(avg_choquet, cmap='viridis')
    axes[0].set_title(f'Choquet {method_suffix} Interaction', fontsize=14)
    axes[0].set_xticks(np.arange(len(feature_names)))
    axes[0].set_yticks(np.arange(len(feature_names)))
    # Adjust label size and padding
    axes[0].set_xticklabels(feature_names, rotation=90, fontsize=10)
    axes[0].set_yticklabels(feature_names, fontsize=10)
    
    # Plot MLM interaction - use original "viridis" colormap without custom scaling
    im2 = axes[1].imshow(avg_mlm, cmap='viridis')
    axes[1].set_title(f'MLM {method_suffix} Interaction', fontsize=14)
    axes[1].set_xticks(np.arange(len(feature_names)))
    axes[1].set_yticks(np.arange(len(feature_names)))
    # Adjust label size and padding
    axes[1].set_xticklabels(feature_names, rotation=90, fontsize=10)
    axes[1].set_yticklabels(feature_names, fontsize=10)
    
    # Add colorbar with adjusted size
    cbar = fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.6)
    cbar.ax.tick_params(labelsize=10)
    
    # Use original spacing settings
    fig.subplots_adjust(wspace=0.3, bottom=0.2)
    
    plt.savefig(join(plot_folder, f'interaction_comparison_{method_suffix}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_banzhaf_indices(feature_names, banzhaf_indices, plot_folder, method):
    """
    Plot Banzhaf power indices for features.
    
    Parameters:
    -----------
    feature_names : list
        Names of the features
    banzhaf_indices : list of numpy.ndarray
        List of Banzhaf indices for each simulation
    plot_folder : str
        Directory to save the plot
    method : str
        Method name for title and filename
    """
    if not banzhaf_indices:
        print(f"No Banzhaf indices computed for {method}; skipping plot.")
        return
        
    banzhaf_array = np.vstack(banzhaf_indices)
    mean_banzhaf = np.mean(banzhaf_array, axis=0)
    std_banzhaf = np.std(banzhaf_array, axis=0)
    
    filename = join(plot_folder, f"banzhaf_indices_{method}.png")
    
    plot_horizontal_bar(
        names=feature_names,
        values=mean_banzhaf,
        std=std_banzhaf,
        title=f"Average Banzhaf Power Indices ({method})",
        xlabel="Average Banzhaf Power Index",
        filename=filename,
        color="indianred"
    )
    
    print(f"Saved Banzhaf indices plot for {method} to: {filename}")

def plot_model_coefficients(model, k, nAttr, representation, dataset_name, output_dir):
    """
    Create a horizontal bar plot of model coefficients ordered by absolute value.
    
    Parameters:
    -----------
    model : LogisticRegression model
        The trained model containing coefficients
    k : int
        k-additivity value of the model
    nAttr : int
        Number of attributes in the dataset
    representation : str
        Choquet representation used ("game", "mobius", or "shapley")
    dataset_name : str
        Name of the dataset being analyzed
    output_dir : str
        Directory to save the plot
    """
    # Create directory for coefficient plots
    coef_plots_dir = os.path.join(output_dir, "coefficient_plots")
    os.makedirs(coef_plots_dir, exist_ok=True)
    
    # Extract coefficients from model
    coefficients = model.coef_[0]  # For binary classification, first row contains coefficients
    intercept = model.intercept_[0]
    
    # Generate appropriate labels based on k-additivity
    labels = []
    if representation == "mobius" or representation == "shapley":
        # For Mobius and Shapley, coefficients represent singletons and combinations
        # Start with singletons
        labels = [f"X{i+1}" for i in range(nAttr)]
        
        # Add interaction terms for k > 1
        if k > 1:
            for r in range(2, k + 1):
                for combo in combinations(range(nAttr), r):
                    labels.append(",".join(f"X{i+1}" for i in combo))
    else:  # game representation
        # For game representation, coefficients typically correspond to coalitional values
        for r in range(1, k + 1):
            for combo in combinations(range(nAttr), r):
                labels.append(",".join(f"X{i+1}" for i in combo))
    
    # Verify label count matches coefficient count
    if len(labels) != len(coefficients):
        print(f"Warning: Label count ({len(labels)}) doesn't match coefficient count ({len(coefficients)})")
        # Fallback to generic labels
        labels = [f"Coef {i+1}" for i in range(len(coefficients))]
    
    # Sort by absolute value (largest to smallest)
    sorted_indices = np.argsort(np.abs(coefficients))[::-1]
    sorted_coefs = coefficients[sorted_indices]
    sorted_labels = [labels[i] for i in sorted_indices]
    
    # Create horizontal bar plot
    plt.figure(figsize=(10, max(6, len(sorted_coefs) * 0.3)))
    bars = plt.barh(range(len(sorted_coefs)), sorted_coefs)
    
    # Color code based on sign
    for i, bar in enumerate(bars):
        if sorted_coefs[i] < 0:
            bar.set_color('indianred')
        else:
            bar.set_color('steelblue')
            
    plt.yticks(range(len(sorted_coefs)), sorted_labels)
    plt.xlabel('Coefficient Value')
    plt.title(f'Model Coefficients for k={k} ({dataset_name}, {representation})')
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Invert y-axis to show largest coefficients at the top
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    
    # Add intercept value as text annotation
    plt.figtext(0.01, 0.01, f"Intercept: {intercept:.4f}", fontsize=10)
    
    # Save plot
    plt.savefig(os.path.join(coef_plots_dir, f"coefficients_k{k}.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"- Saved coefficient plot for k={k} to {coef_plots_dir}")