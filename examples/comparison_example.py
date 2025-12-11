"""
Example script demonstrating the use of ChoquisticRegression with different representations.

This script shows how to use the ChoquisticRegression model with different
mathematical bases (Game, Mobius, and Shapley) on the banknotes dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import pandas as pd

from core.models.regression import ChoquisticRegression
from utils.data_loader import func_read_data
from utils.visualization.plotting import (
    plot_coefficients,
    plot_interaction_matrix_2add_shapley,
    plot_model_performance_comparison,
    plot_noise_robustness,
    plot_k_additivity_results
)


def main():
    # Load data
    print("Loading banknotes dataset...")
    X, y = func_read_data("banknotes")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Create and train models with different representations
    print("\nTraining models with different representations...")
    
    # Game representation (k=2)
    model_game = ChoquisticRegression(
        representation="game",
        k_add=2,
        scale_data=True,
        random_state=42
    )
    model_game.fit(X_train, y_train)
    
    # Mobius representation (k=2)
    model_mobius = ChoquisticRegression(
        representation="mobius",
        k_add=2,
        scale_data=True,
        random_state=42
    )
    model_mobius.fit(X_train, y_train)
    
    # Shapley representation (k=2)
    model_shapley = ChoquisticRegression(
        representation="shapley",
        k_add=2,
        scale_data=True,
        random_state=42
    )
    model_shapley.fit(X_train, y_train)
    
    # Evaluate models
    print("\nEvaluating models...")
    
    # Game representation
    y_pred_game = model_game.predict(X_test)
    y_proba_game = model_game.predict_proba(X_test)[:, 1]
    acc_game = accuracy_score(y_test, y_pred_game)
    auc_game = roc_auc_score(y_test, y_proba_game)
    
    # Mobius representation
    y_pred_mobius = model_mobius.predict(X_test)
    y_proba_mobius = model_mobius.predict_proba(X_test)[:, 1]
    acc_mobius = accuracy_score(y_test, y_pred_mobius)
    auc_mobius = roc_auc_score(y_test, y_proba_mobius)
    
    # Shapley representation
    y_pred_shapley = model_shapley.predict(X_test)
    y_proba_shapley = model_shapley.predict_proba(X_test)[:, 1]
    acc_shapley = accuracy_score(y_test, y_pred_shapley)
    auc_shapley = roc_auc_score(y_test, y_proba_shapley)
    
    # Print results
    print("\nModel Performance:")
    print(f"Game representation (k=2):")
    print(f"  - Accuracy: {acc_game:.4f}")
    print(f"  - AUC: {auc_game:.4f}")
    print(f"  - Classification Report:")
    print(classification_report(y_test, y_pred_game))
    
    print(f"Mobius representation (k=2):")
    print(f"  - Accuracy: {acc_mobius:.4f}")
    print(f"  - AUC: {auc_mobius:.4f}")
    print(f"  - Classification Report:")
    print(classification_report(y_test, y_pred_mobius))
    
    print(f"Shapley representation (k=2):")
    print(f"  - Accuracy: {acc_shapley:.4f}")
    print(f"  - AUC: {auc_shapley:.4f}")
    print(f"  - Classification Report:")
    print(classification_report(y_test, y_pred_shapley))

    # Create output directory
    import os
    output_folder = "results/example"
    os.makedirs(output_folder, exist_ok=True)

    # --- Plotting Examples ---
    print("\nGenerating plots...")

    # Example for plot_coefficients
    if hasattr(model_shapley, 'coef_'):
        feature_names_out = model_shapley.get_feature_names_out()
        plot_coefficients(
            feature_names=feature_names_out,
            all_coefficients=[model_shapley.coef_],
            plot_folder=output_folder,
            k_add=2 
        )

    # Example for plot_interaction_matrix_2add_shapley
    if hasattr(model_shapley, 'interaction_matrix_'):
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        plot_interaction_matrix_2add_shapley(
            feature_names=feature_names,
            coefs=model_shapley.coef_.flatten(),
            plot_folder=output_folder
        )
    
    # Create results dictionary for visualization
    results = {
        "baseline": {
            "Game (k=2)": {"accuracy": acc_game, "roc_auc": auc_game},
            "Mobius (k=2)": {"accuracy": acc_mobius, "roc_auc": auc_mobius},
            "Shapley (2-add)": {"accuracy": acc_shapley, "roc_auc": auc_shapley}
        },
        "metrics": ["accuracy", "roc_auc"]
    }
    
    # Plot model performance comparison
    plot_model_performance_comparison(
        results,
        plot_folder=output_folder,
        title="Model Performance Comparison (Banknotes Dataset)"
    )

    # Example for plot_noise_robustness
    noise_levels = [0.0, 0.1, 0.2, 0.3]
    noise_results = {"Shapley (2-add)": {}}
    for noise in noise_levels:
        X_test_noisy = X_test + np.random.normal(0, noise, X_test.shape)
        y_pred_noisy = model_shapley.predict(X_test_noisy)
        acc_noisy = accuracy_score(y_test, y_pred_noisy)
        noise_results["Shapley (2-add)"][str(noise)] = {"accuracy": acc_noisy}
    
    plot_noise_robustness(
        {"noise_robustness": noise_results},
        plot_folder=output_folder
    )

    # Example for plot_k_additivity_results
    k_values = [1, 2, 3]
    k_add_results = []
    for k in k_values:
        model = ChoquisticRegression(
            representation="game", # Using 'game' for k-additivity example
            k_add=k,
            scale_data=True,
            random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        k_add_results.append({
            'k_value': k,
            'baseline_accuracy': acc,
            'train_time': 0, # Placeholder
            'n_params': len(model.coef_)
        })
    results_df = pd.DataFrame(k_add_results)
    plot_k_additivity_results(
        results_df,
        plot_folder=output_folder,
        dataset_name="banknotes",
        representation="game"
    )
    
    print(f"\nExample completed successfully. Results saved to '{output_folder}' directory.")


if __name__ == "__main__":
    main()
