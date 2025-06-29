import os
os.environ["SCIPY_ARRAY_API"] = "1"

RANDOM_STATE = 42
import sys
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from itertools import combinations

np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

# Add the parent directory ('paper improvement') to the Python path to allow module imports
# This ensures that the script can find 'regression_classes' and 'mod_GenFuzzyRegression'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from regression_classes import choquet_k_additive_shapley, choquet_k_additive_mobius, choquet_k_additive_game
from mod_GenFuzzyRegression import func_read_data


def analyze_pairwise_coefficients(
    dataset,
    representation="shapley",
    k=2,
    regularization='l2',
    output_dir="pairwise_coefficients_results",
    test_size=0.3,
    random_state=42
):
    """
    Trains a logistic regression model on k-additive Choquet-transformed data,
    extracts, saves, and plots the model coefficients as a horizontal bar plot.
    """
    # Skip if k is not specified, as k-additive transform requires an integer.
    if k is None:
        print(f"Skipping analysis for k=None for dataset {dataset}.")
        return

    if isinstance(dataset, (list, tuple)) and len(dataset) == 3:
        dataset_name, X, y = dataset
    else:
        dataset_name = dataset
        X, y = func_read_data(dataset_name)

    # Handle case where regularization is None for file naming
    reg_str = regularization if regularization is not None else 'none'

    print(f"\nAnalyzing coefficients for: {dataset_name} (k={k}, repr={representation}, reg={reg_str})")

    # Create a specific directory for the dataset and k-value
    k_output_dir = os.path.join(output_dir, dataset_name, f"k_{k}")
    os.makedirs(k_output_dir, exist_ok=True)

    nSamp, nAttr = X.shape
    
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist()
    else:
        feature_names = [f'F{i+1}' for i in range(nAttr)]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Scale data
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Select transformation function
    if representation == "game":
        choquet_transform = choquet_k_additive_game
    elif representation == "mobius":
        choquet_transform = choquet_k_additive_mobius
    elif representation == "shapley":
        choquet_transform = choquet_k_additive_shapley
    else:
        raise ValueError(f"Unknown representation: {representation}")

    # Transform data
    print(f"Applying {k}-additive Choquet transformation...")
    X_train_choquet = choquet_transform(X_train_scaled, k_add=k)
    X_test_choquet = choquet_transform(X_test_scaled, k_add=k)

    # Train model
    print("Training Logistic Regression model...")
    solver = "newton-cg" if reg_str in ['l2', 'none'] else "saga"
    
    model = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
        solver=solver,
        penalty=regularization  # Use original value here (None, 'l1', or 'l2')
    )
    model.fit(X_train_choquet, y_train)

    # --- Coefficient Extraction and Naming ---
    coefficients = model.coef_[0]
    intercept = model.intercept_[0]

    # Generate coefficient names for all coalitions up to size k
    labels = []
    for r in range(1, k + 1):
        for combo in combinations(range(nAttr), r):
            labels.append(",".join(feature_names[i] for i in combo))

    if len(labels) != len(coefficients):
        print(f"Warning: Label count ({len(labels)}) doesn't match coefficient count ({len(coefficients)}). Using generic labels.")
        labels = [f'Coeff_{i}' for i in range(len(coefficients))]

    # --- Save Coefficients ---
    coeffs_df = pd.DataFrame({'name': labels, 'value': coefficients})
    coeffs_df.loc['intercept'] = {'name': 'intercept', 'value': intercept}
    
    coeffs_filename = f"{dataset_name}_{representation}_k{k}_reg_{reg_str}_coeffs.csv"
    coeffs_path = os.path.join(k_output_dir, coeffs_filename)
    coeffs_df.to_csv(coeffs_path, index=False)
    print(f"Coefficients saved to {coeffs_path}")

    # --- Plotting Logic (Horizontal Bar Plot) ---
    
    # Sort by absolute value
    sorted_indices = np.argsort(np.abs(coefficients))[::-1]
    sorted_coefs = coefficients[sorted_indices]
    sorted_labels = [labels[i] for i in sorted_indices]

    # Limit to the top 10 largest absolute value coefficients
    top_n = 10
    if len(sorted_coefs) > top_n:
        sorted_coefs = sorted_coefs[:top_n]
        sorted_labels = sorted_labels[:top_n]

    plt.figure(figsize=(12, max(8, len(sorted_coefs) * 0.3)))
    bars = plt.barh(range(len(sorted_coefs)), sorted_coefs, align='center')
    
    # Color bars
    for i, bar in enumerate(bars):
        if sorted_coefs[i] < 0:
            bar.set_color('indianred')
        else:
            bar.set_color('steelblue')

    plt.yticks(range(len(sorted_coefs)), sorted_labels)
    plt.gca().invert_yaxis()  # Show largest on top
    plt.xlabel("Coefficient Value")
    plt.title(f'Model Coefficients for {dataset_name}\n(k={k}, {representation}, reg={reg_str})')
    plt.axvline(x=0, color='grey', linestyle='--')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add intercept value as text
    plt.figtext(0.01, 0.01, f"Intercept: {intercept:.4f}", ha="left", fontsize=10,
                bbox={"facecolor":"white", "alpha":0.5, "pad":5})

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make space for figtext

    plot_filename = f"coeffs_{representation}_reg_{reg_str}.png"
    plot_path = os.path.join(k_output_dir, plot_filename)
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Coefficient plot saved to {plot_path}")


if __name__ == "__main__":
    # Change working directory to the parent directory ('paper improvement')
    # so that 'func_read_data' can find the dataset files in their relative paths.
    paper_improvement_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    os.chdir(paper_improvement_dir)

    datasets_to_analyze = ['pure_pairwise_interaction']
    k_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] # Changed to include k=1
    regularizations = [None, 'l2', 'l1']
    for dataset in datasets_to_analyze:
        for representation in ['shapley']:
            for k in k_values:
                for regularization in regularizations:
                    analyze_pairwise_coefficients(
                        dataset=dataset,
                        representation=representation,
                        k=k,
                        regularization=regularization
                    )
