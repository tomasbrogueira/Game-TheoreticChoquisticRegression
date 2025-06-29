"""
Example script for plotting noise robustness.
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

from core.models.regression import ChoquisticRegression
from utils.data_loader import func_read_data
from utils.visualization.plotting import plot_noise_robustness

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
    output_folder = "results/example_noise_robustness"
    os.makedirs(output_folder, exist_ok=True)

    # Plot noise robustness
    print("\nGenerating noise robustness plot...")
    noise_levels = [0.0, 0.1, 0.2, 0.3]
    noise_results = {"Shapley (k=2)": {}}
    for noise in noise_levels:
        X_test_noisy = X_test + np.random.normal(0, noise, X_test.shape)
        y_pred_noisy = model_shapley.predict(X_test_noisy)
        acc_noisy = accuracy_score(y_test, y_pred_noisy)
        noise_results["Shapley (k=2)"][str(noise)] = {"accuracy": acc_noisy}
    
    plot_noise_robustness(
        {"noise_robustness": noise_results},
        plot_folder=output_folder
    )
    print(f"Plot saved to '{output_folder}' directory.")

if __name__ == "__main__":
    main()
