# PIC-I: Game-theoretic Extensions of Logistic Regression

This repository contains the implementation of game-theoretic extensions to logistic regression, focusing on the Choquet integral and other aggregation functions. The project aims to capture non-linear interactions between parameters while maintaining interpretability and efficiency of logistic regression models.

## Project Overview

The PIC-I is my end of Bachelor's project. This project extends traditional logistic regression by incorporating aggregation functions like the Choquet integral and Multi-Linear Model (MLM). These extensions allow the model to capture complex non-linear interactions between features while preserving the interpretability advantages of logistic regression.

### Key Features

- **Multiple Representation Bases**: Implementation of different mathematical bases (Game, Mobius, and Shapley) that are linearly related but have distinct interpretability properties
- **K-additivity Analysis**: Tools for analyzing the impact of k-additivity on model performance, complexity, and interpretability
- **Robustness Testing**: Comprehensive framework for testing model robustness under various perturbations
- **Visualization Tools**: Specialized visualization functions for each representation basis

## Repository Structure

```
PIC-I/
├── core/                # Core implementation
│   ├── models/          # Model implementations
│   │   ├── choquet.py   # Choquet integral implementations
│   │   ├── mlm.py       # Multi-Linear Model implementations
│   │   └── regression.py # ChoquisticRegression model
│   └── __init__.py
├── utils/               # Utility functions
│   ├── visualization/   # Visualization utilities
│   │   └── plotting.py  # Plotting functions for different bases
│   ├── data_loader.py   # Data loading utilities
│   └── __init__.py
├── tests/               # Testing code
│   ├── robustness/      # Robustness testing
│   │   └── robustness.py # Robustness testing framework
│   ├── k_additivity/    # K-additivity analysis
│   │   └── k_additivity.py # K-additivity analysis tools
│   └── __init__.py
├── examples/            # Example scripts
│   └── __init__.py
├── data/                # Data directory
├── research/            # Development of the project
├── requirements.txt     # Project dependencies
├── setup.py             # Package installation script
└── README.md            # This file
```

## Mathematical Background

The project is based on three different mathematical bases that are linearly related:

1. **Game Representation**: The traditional representation of fuzzy measures with the caviat of having less restriction such as monotonicity.
2. **Mobius Representation**: An alternative representation that directly captures the interaction between features.
3. **Shapley Representation**: A representation that uses the shapely value and the pairwise interaction indices between features.

Each representation has its own interpretability properties and is suitable for different types of analysis.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/tomasbrogueira/PIC-I.git
cd PIC-I
```
```

2. Install the package in development mode:
```bash
pip install -e .
```

3. Install additional dependencies if needed:
```bash
pip install -r requirements.txt
```

### Option 2: Use Without Installation

If you prefer not to install the package, you can still run the scripts by adding the project root to your Python path:

1. Clone the repository:
```bash
git clone https://github.com/tomasbrogueira/PIC-I.git
cd PIC-I-Organized
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Run scripts from the project root:
```bash
python -m examples.comparison_example
```

## Dataset Preparation

Before running the examples, you need to place your datasets in the `data/` directory. The data loader expects the following files:

- `data_banknotes.csv`: Banknote authentication dataset
- `transfusion.csv`: Blood Transfusion Service Center Data Set
- `data_mammographic.data`: Mammographic mass dataset
- `data_raisin.xlsx`: Raisin dataset
- `data_rice.xlsx`: Rice (Commeo and Osmancik) dataset
- `diabetes.csv`: Diabetes (PIMA) dataset
- `data_skin.csv`: Skin segmentation dataset

Copy these files from your original project to the `data/` directory.

## Usage

### Basic Example

```python
from core.models.regression import ChoquisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils.data_loader import func_read_data

# Load data
X, y = func_read_data("banknotes")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train model with game representation
model_game = ChoquisticRegression(
    method="choquet",
    representation="game",
    k_add=2,
    scale_data=True
)
model_game.fit(X_train, y_train)

# Create and train model with Mobius representation
model_mobius = ChoquisticRegression(
    method="choquet",
    representation="mobius",
    k_add=2,
    scale_data=True
)
model_mobius.fit(X_train, y_train)

# Create and train model with Shapley representation (2-additive)
model_shapley = ChoquisticRegression(
    method="choquet_2add",
    scale_data=True
)
model_shapley.fit(X_train, y_train)

# Evaluate models
y_pred_game = model_game.predict(X_test)
y_pred_mobius = model_mobius.predict(X_test)
y_pred_shapley = model_shapley.predict(X_test)

print(f"Game representation accuracy: {accuracy_score(y_test, y_pred_game):.4f}")
print(f"Mobius representation accuracy: {accuracy_score(y_test, y_pred_mobius):.4f}")
print(f"Shapley representation accuracy: {accuracy_score(y_test, y_pred_shapley):.4f}")
```

### Running the Example Script

The repository includes an example script that demonstrates the use of different representations:

```bash
# If installed as a package (Option 1):
python examples/comparison_example.py

# If using without installation (Option 2):
python -m examples.comparison_example
```

### K-additivity Analysis

```python
from tests.k_additivity.k_additivity import direct_k_additivity_analysis, find_optimal_k
from utils.visualization.plotting import plot_k_additivity_results

# Analyze k-additivity impact with game representation
results_game = direct_k_additivity_analysis(
    dataset_name="banknotes",
    representation="game",
    output_dir="results/k_additivity/game"
)

# Analyze k-additivity impact with Mobius representation
results_mobius = direct_k_additivity_analysis(
    dataset_name="banknotes",
    representation="mobius",
    output_dir="results/k_additivity/mobius"
)

# Find optimal k value
optimal_k_game, scores_game = find_optimal_k(results_game)
optimal_k_mobius, scores_mobius = find_optimal_k(results_mobius)

print(f"Optimal k for game representation: {optimal_k_game}")
print(f"Optimal k for Mobius representation: {optimal_k_mobius}")

# Plot results
plot_k_additivity_results(
    results_game,
    plot_folder="results/k_additivity/plots",
    dataset_name="banknotes",
    representation="game"
)
```

### Robustness Testing

```python
from tests.robustness.robustness import test_model_robustness
from utils.visualization.plotting import plot_model_performance_comparison, plot_noise_robustness

# Create models dictionary
models = {
    "Game (k=2)": model_game,
    "Mobius (k=2)": model_mobius,
    "Shapley (2-add)": model_shapley
}

# Test robustness
robustness_results = test_model_robustness(
    models=models,
    X_test=X_test,
    y_test=y_test,
    feature_names=X.columns,
    output_folder="results/robustness"
)

# Plot results
plot_model_performance_comparison(
    robustness_results,
    plot_folder="results/robustness",
    title="Model Performance Comparison"
)

plot_noise_robustness(
    robustness_results,
    plot_folder="results/robustness"
)
```

## Visualization (not yet functional for all basis)

The project includes specialized visualization functions for each representation basis:

### Game Representation

```python
from utils.visualization.plotting import plot_shapley_game, plot_interaction_matrix_game

# Plot Shapley values for game representation
plot_shapley_game(
    feature_names=X.columns,
    all_shapley_values=[shapley_values],  # List of Shapley values from multiple runs
    plot_folder="results/visualization"
)

# Plot interaction matrix for game representation
plot_interaction_matrix_game(
    feature_names=X.columns,
    all_interaction_matrices=[interaction_matrix],  # List of interaction matrices from multiple runs
    plot_folder="results/visualization"
)
```

### Mobius Representation

```python
from utils.visualization.plotting import plot_shapley_mobius, plot_interaction_matrix_mobius

# Plot Shapley values for Mobius representation
plot_shapley_mobius(
    feature_names=X.columns,
    all_shapley_values=[shapley_values],  # List of Shapley values from multiple runs
    plot_folder="results/visualization"
)

# Plot interaction matrix for Mobius representation
plot_interaction_matrix_mobius(
    feature_names=X.columns,
    all_interaction_matrices=[interaction_matrix],  # List of interaction matrices from multiple runs
    plot_folder="results/visualization"
)
```

### Shapley Representation

```python
from utils.visualization.plotting import plot_shapley_2add, plot_interaction_matrix_2add

# Plot Shapley values for 2-additive Shapley representation
plot_shapley_2add(
    feature_names=X.columns,
    all_shapley_values=[shapley_values],  # List of Shapley values from multiple runs
    plot_folder="results/visualization"
)

# Plot interaction matrix for 2-additive Shapley representation
plot_interaction_matrix_2add(
    feature_names=X.columns,
    all_interaction_matrices=[interaction_matrix],  # List of interaction matrices from multiple runs
    plot_folder="results/visualization"
)
```

## Troubleshooting

### Import Errors

If you encounter import errors like `ModuleNotFoundError: No module named 'core'`, try one of these solutions:

1. **Install the package** (recommended):
   ```bash
   pip install -e .
   ```

2. **Run from the project root**:
   ```bash
   python -m examples.comparison_example
   ```

3. **Add the project root to your Python path** in your script:
   ```python
   import sys
   import os
   sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
   ```

### Missing Datasets

If you encounter errors related to missing datasets, make sure you've copied all required dataset files to the `data/` directory.
