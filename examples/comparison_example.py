"""
Example script demonstrating the use of ChoquisticRegression with different representations.

This script shows how to use the ChoquisticRegression model with different
mathematical bases (Game, Mobius, and Shapley) on the banknotes dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

from core.models.regression import ChoquisticRegression
from utils.data_loader import func_read_data
from utils.visualization.plotting import (
    plot_shapley_game, plot_interaction_matrix_game,
    plot_shapley_mobius, plot_interaction_matrix_mobius,
    plot_shapley_2add, plot_interaction_matrix_2add,
    plot_model_performance_comparison
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
        method="choquet",
        representation="game",
        k_add=2,
        scale_data=True,
        random_state=42
    )
    model_game.fit(X_train, y_train)
    
    # Mobius representation (k=2)
    model_mobius = ChoquisticRegression(
        method="choquet",
        representation="mobius",
        k_add=2,
        scale_data=True,
        random_state=42
    )
    model_mobius.fit(X_train, y_train)
    
    # Shapley representation (2-additive)
    model_shapley = ChoquisticRegression(
        method="choquet_2add",
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
    
    print(f"Shapley representation (2-additive):")
    print(f"  - Accuracy: {acc_shapley:.4f}")
    print(f"  - AUC: {auc_shapley:.4f}")
    print(f"  - Classification Report:")
    print(classification_report(y_test, y_pred_shapley))
    
    # Create results dictionary for visualization
    results = {
        "baseline": {
            "Game (k=2)": {"accuracy": acc_game, "roc_auc": auc_game},
            "Mobius (k=2)": {"accuracy": acc_mobius, "roc_auc": auc_mobius},
            "Shapley (2-add)": {"accuracy": acc_shapley, "roc_auc": auc_shapley}
        },
        "metrics": ["accuracy", "roc_auc"]
    }
    
    # Create output directory
    import os
    os.makedirs("results/example", exist_ok=True)
    
    # Plot model performance comparison
    plot_model_performance_comparison(
        results,
        plot_folder="results/example",
        title="Model Performance Comparison (Banknotes Dataset)"
    )
    
    print("\nExample completed successfully. Results saved to 'results/example' directory.")


if __name__ == "__main__":
    main()
