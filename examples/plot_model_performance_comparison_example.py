"""
Example script for plotting model performance comparison.
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import os

from core.models.regression import ChoquisticRegression
from utils.data_loader import func_read_data
from utils.visualization.plotting import plot_model_performance_comparison

def main():
    # Load data
    print("Loading banknotes dataset...")
    X, y = func_read_data("banknotes")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Train models
    print("\nTraining models...")
    model_game = ChoquisticRegression(representation="game", k_add=2, scale_data=True, random_state=42)
    model_game.fit(X_train, y_train)
    
    model_mobius = ChoquisticRegression(representation="mobius", k_add=2, scale_data=True, random_state=42)
    model_mobius.fit(X_train, y_train)

    model_shapley = ChoquisticRegression(representation="shapley", k_add=2, scale_data=True, random_state=42)
    model_shapley.fit(X_train, y_train)

    # Evaluate models
    print("\nEvaluating models...")
    acc_game = accuracy_score(y_test, model_game.predict(X_test))
    auc_game = roc_auc_score(y_test, model_game.predict_proba(X_test)[:, 1])
    acc_mobius = accuracy_score(y_test, model_mobius.predict(X_test))
    auc_mobius = roc_auc_score(y_test, model_mobius.predict_proba(X_test)[:, 1])
    acc_shapley = accuracy_score(y_test, model_shapley.predict(X_test))
    auc_shapley = roc_auc_score(y_test, model_shapley.predict_proba(X_test)[:, 1])

    # Create output directory
    output_folder = "results/example_performance_comparison"
    os.makedirs(output_folder, exist_ok=True)

    # Plot model performance
    print("\nGenerating model performance plot...")
    results = {
        "baseline": {
            "Game (k=2)": {"accuracy": acc_game, "roc_auc": auc_game},
            "Mobius (k=2)": {"accuracy": acc_mobius, "roc_auc": auc_mobius},
            "Shapley (2-add)": {"accuracy": acc_shapley, "roc_auc": auc_shapley}
        },
        "metrics": ["accuracy", "roc_auc"]
    }
    plot_model_performance_comparison(
        results,
        plot_folder=output_folder,
        title="Model Performance Comparison (Banknotes Dataset)"
    )
    print(f"Plot saved to '{output_folder}' directory.")

if __name__ == "__main__":
    main()
