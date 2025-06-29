"""
Example script for plotting k-additivity results.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

from core.models.regression import ChoquisticRegression
from utils.data_loader import func_read_data
from utils.visualization.plotting import plot_k_additivity_results

def main():
    # Load data
    print("Loading banknotes dataset...")
    X, y = func_read_data("banknotes")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Create output directory
    output_folder = "results/example_k_additivity"
    os.makedirs(output_folder, exist_ok=True)

    # --- Plotting k-additivity results ---
    print("\nGenerating k-additivity plot...")
    k_values = [1, 2, 3, 4]
    k_add_results = []
    for k in k_values:
        print(f"  - Training model with k={k}...")
        model = ChoquisticRegression(
            representation="game",
            k_add=k,
            scale_data=True,
            random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        # Get number of parameters
        n_params = len(model.model_.coef_[0])

        k_add_results.append({
            'k_value': k,
            'baseline_accuracy': acc,
            'train_time': 0,  # Placeholder for train time
            'n_params': n_params
        })
        
    results_df = pd.DataFrame(k_add_results)
    
    plot_k_additivity_results(
        results_df,
        plot_folder=output_folder,
        dataset_name="banknotes",
        representation="game"
    )
    print(f"Plot saved to '{output_folder}' directory.")

if __name__ == "__main__":
    main()
