from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from core.models.regression import ChoquisticRegression
from utils.visualization.plotting import (
	plot_coefficients,
	plot_interaction_matrix_2add_shapely,
	plot_horizontal_bar,
	plot_model_performance_comparison,
	plot_noise_robustness,
	plot_k_additivity_results,
	ensure_folder
)

def choquistic_regression_test():
	# Setup paths
	data_path = Path(__file__).resolve().parents[2] / "data" / "diabetes.csv"
	plot_folder = Path(__file__).resolve().parent / "test_plots"
	ensure_folder(plot_folder)
	
	# Load data
	df = pd.read_csv(data_path)

	X = df.drop(columns="Outcome")
	y = df["Outcome"].to_numpy()
	feature_names = X.columns.tolist()

	# split data into train and test sets to evaluate generalization
	x_train, x_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size=0.2,
		random_state=1,
		stratify=y,
	)

	print("=" * 80)
	print("Testing ChoquisticRegression with visualizations")
	print("=" * 80)
	
	# Train model
	model = ChoquisticRegression(
		representation="shapley",
		k_add=8,
		C=10.0,
		max_iter=500,
		random_state=0,
	)
	model.fit(x_train, y_train)

	# Get coefficients
	coefficients = model.model_.coef_[0]
	print(f"\nModel has {len(coefficients)} coefficients")

	# Get test score
	test_score = model.score(x_test, y_test)
	train_score = model.score(x_train, y_train)
	print(f"Train Accuracy: {train_score:.4f}")
	print(f"Test Accuracy: {test_score:.4f}")
	
	# Generate coefficient names for all interactions up to k_add
	from itertools import combinations
	coefficient_names = []
	for size in range(1, min(9, len(feature_names) + 1)):  # k_add=8
		for combo in combinations(feature_names, size):
			coefficient_names.append(" × ".join(combo))
	
	# Ensure we have the right number of names
	if len(coefficient_names) != len(coefficients):
		print(f"Warning: Expected {len(coefficients)} names, got {len(coefficient_names)}")
		# Generate generic names if mismatch
		coefficient_names = [f"Coef_{i}" for i in range(len(coefficients))]
	
	# ========================================================================
	# Test 1: Plot coefficients
	# ========================================================================
	print("\n" + "=" * 80)
	print("Test 1: Plotting coefficients")
	print("=" * 80)
	
	# Simulate multiple runs for averaging
	all_coefficients = [coefficients + np.random.normal(0, 0.01, len(coefficients)) for _ in range(5)]
	all_coefficients[0] = coefficients  # Include actual coefficients
	
	plot_coefficients(
		feature_names=coefficient_names,
		all_coefficients=all_coefficients,
		plot_folder=str(plot_folder),
		k_add=8
	)
	
	# ========================================================================
	# Test 2: Plot horizontal bar (direct test)
	# ========================================================================
	print("\n" + "=" * 80)
	print("Test 2: Plotting horizontal bar chart")
	print("=" * 80)
	
	plot_horizontal_bar(
		names=coefficient_names,
		values=coefficients,
		title="Direct Coefficients Visualization",
		xlabel="Coefficient Value",
		filename=str(plot_folder / "horizontal_bar_test.png"),
		color="coral"
	)
	print(f"Saved horizontal bar plot to: {plot_folder / 'horizontal_bar_test.png'}")
	
	# ========================================================================
	# Test 3: Plot interaction matrix (for 2-additive)
	# ========================================================================
	print("\n" + "=" * 80)
	print("Test 3: Plotting interaction matrix (2-additive)")
	print("=" * 80)
	
	# Train a 2-additive model to get interaction matrix
	model_2add = ChoquisticRegression(
		representation="shapley",
		k_add=2,
		C=10.0,
		max_iter=500,
		random_state=0,
	)
	model_2add.fit(x_train, y_train)
	
	plot_interaction_matrix_2add_shapely(
		feature_names=feature_names,
		coefs=model_2add.model_.coef_[0],
		plot_folder=str(plot_folder),
	)
	
	# ========================================================================
	# Test 4: Plot model performance comparison
	# ========================================================================
	print("\n" + "=" * 80)
	print("Test 4: Plotting model performance comparison")
	print("=" * 80)
	
	# Create mock results for different models
	results = {
		"baseline": {
			"ChoquisticRegression": {
				"accuracy": test_score,
				"roc_auc": 0.82,
				"f1": 0.75
			},
			"LogisticRegression": {
				"accuracy": 0.76,
				"roc_auc": 0.80,
				"f1": 0.72
			},
			"RandomForest": {
				"accuracy": 0.78,
				"roc_auc": 0.83,
				"f1": 0.74
			}
		},
		"metrics": ["accuracy", "roc_auc", "f1"]
	}
	
	plot_model_performance_comparison(
		results=results,
		plot_folder=str(plot_folder),
		title="Model Performance Comparison - Diabetes Dataset"
	)
	
	# ========================================================================
	# Test 5: Plot noise robustness
	# ========================================================================
	print("\n" + "=" * 80)
	print("Test 5: Plotting noise robustness")
	print("=" * 80)
	
	# Create mock noise robustness results
	noise_results = {
		"noise_robustness": {
			"ChoquisticRegression": {
				"0.0": {"accuracy": test_score},
				"0.05": {"accuracy": test_score - 0.02},
				"0.1": {"accuracy": test_score - 0.05},
				"0.2": {"accuracy": test_score - 0.10},
			},
			"LogisticRegression": {
				"0.0": {"accuracy": 0.76},
				"0.05": {"accuracy": 0.73},
				"0.1": {"accuracy": 0.69},
				"0.2": {"accuracy": 0.62},
			}
		}
	}
	
	plot_noise_robustness(
		results=noise_results,
		plot_folder=str(plot_folder)
	)
	
	# ========================================================================
	# Test 6: Plot k-additivity results
	# ========================================================================
	print("\n" + "=" * 80)
	print("Test 6: Plotting k-additivity analysis")
	print("=" * 80)
	
	# Create mock k-additivity results
	k_add_data = {
		'k_value': [1, 2, 3, 4, 5, 6, 7, 8],
		'baseline_accuracy': [0.72, 0.75, 0.77, 0.78, test_score - 0.01, test_score, test_score, test_score],
		'train_time': [0.5, 0.8, 1.2, 1.8, 2.5, 3.2, 4.0, 5.0],
		'n_params': [8, 36, 84, 126, 210, 252, 294, 336],
		'noise_0.05': [0.70, 0.73, 0.75, 0.76, 0.77, 0.77, 0.77, 0.77],
		'noise_0.1': [0.68, 0.71, 0.73, 0.74, 0.75, 0.75, 0.75, 0.75],
		'noise_0.2': [0.64, 0.67, 0.69, 0.70, 0.71, 0.71, 0.71, 0.71],
		'noise_0.3': [0.60, 0.63, 0.65, 0.66, 0.67, 0.67, 0.67, 0.67],
	}
	results_df = pd.DataFrame(k_add_data)
	
	plot_k_additivity_results(
		results_df=results_df,
		plot_folder=str(plot_folder),
		dataset_name="diabetes",
		representation="shapley"
	)
	
	# ========================================================================
	# Summary
	# ========================================================================
	print("\n" + "=" * 80)
	print("All visualization tests completed!")
	print(f"Plots saved to: {plot_folder}")
	print("=" * 80)
	print("\nGenerated plots:")
	for plot_file in sorted(plot_folder.glob("*.png")):
		print(f"  - {plot_file.name}")
	print("=" * 80)

if __name__ == "__main__":
	choquistic_regression_test()