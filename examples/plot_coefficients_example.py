"""
Example script for plotting model coefficients.
"""
import numpy as np
from sklearn.model_selection import train_test_split
import os

from core.models.regression import ChoquisticRegression
from utils.data_loader import func_read_data
from utils.visualization.plotting import plot_coefficients

def main():
    # Load data
    print("Loading banknotes dataset...")
    X, y = func_read_data("banknotes")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Train Shapley representation model
    print("\nTraining Shapley representation model...")
    model_shapley = ChoquisticRegression(
        representation="shapley",
        k_add=2,
        scale_data=True,
        random_state=42
    )
    model_shapley.fit(X_train, y_train)

    # Create output directory
    output_folder = "results/example_coefficients"
    os.makedirs(output_folder, exist_ok=True)

    # Plot coefficients
    print("\nGenerating coefficients plot...")
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    plot_coefficients(
        feature_names=feature_names,
        all_coefficients=[model_shapley.coef_[0]],
        plot_folder=output_folder,
        k_add=2 
    )
    print(f"Plot saved to '{output_folder}' directory.")

if __name__ == "__main__":
    main()
