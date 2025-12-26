"""
Robustness testing utilities for Choquet integral models.

This module provides functions for testing the robustness of models
under various perturbations such as noise, feature permutation,
feature dropout, and bootstrap sampling.
"""

import numpy as np
import pandas as pd
import os
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.utils import resample


def test_model_robustness(
    models,
    X_test,
    y_test,
    feature_names=None,
    noise_levels=[0.05, 0.1, 0.2, 0.3, 0.5],
    n_permutations=10,
    feature_dropout_count=None,
    n_bootstrap_samples=100,
    bootstrap_size=0.8,
    output_folder="robustness_results",
    random_state=0
):
    """
    Test the robustness of multiple models under various perturbations.
    
    Parameters
    ----------
    models : dict
        Dictionary of trained models with model names as keys
    X_test : array-like
        Test data features
    y_test : array-like
        Test data labels
    feature_names : list, optional
        Names of features for better visualization
    noise_levels : list, default=[0.05, 0.1, 0.2, 0.3, 0.5]
        Levels of Gaussian noise to add (as proportion of feature std dev)
    n_permutations : int, default=10
        Number of random permutations for feature permutation test
    feature_dropout_count : int, default=None
        Number of features to drop (default: 1/3 of features)
    n_bootstrap_samples : int, default=100
        Number of bootstrap samples for stability testing
    bootstrap_size : float, default=0.8
        Size of each bootstrap sample as proportion of test set
    output_folder : str, default="robustness_results"
        Folder to save result plots
    random_state : int, default=0
        Random seed for reproducibility
    
    Returns
    -------
    dict
        Results of robustness tests for each model and perturbation type
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # Check if input is a DataFrame to preserve column names
    is_dataframe = isinstance(X_test, pd.DataFrame)
    original_columns = X_test.columns if is_dataframe else None
    
    if feature_names is None:
        if is_dataframe:
            feature_names = list(original_columns)
        else:
            feature_names = [f"Feature_{i}" for i in range(X_test.shape[1])]
    
    # Set default feature dropout count if not specified
    if feature_dropout_count is None:
        feature_dropout_count = max(1, X_test.shape[1] // 3)
    
    results = {
        "baseline": {},
        "noise_robustness": {},
        "feature_permutation": {},
        "feature_dropout": {},
        "bootstrap_stability": {},
        "metrics": ["accuracy", "roc_auc", "f1"]
    }

    # Set random seed for reproducibility
    np.random.seed(random_state)
    
    # Test each model
    for model_name, model in models.items():
        try:
            # Baseline performance
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan
            f1 = f1_score(y_test, y_pred)
            
            results["baseline"][model_name] = {
                "accuracy": acc,
                "roc_auc": auc,
                "f1": f1
            }
            
            # Noise robustness test
            noise_results = {}
            for noise_level in noise_levels:
                noise_accs = []
                noise_aucs = []
                noise_f1s = []
                
                for _ in range(5):  # Multiple runs for stability
                    # Add Gaussian noise scaled by feature standard deviation
                    feature_stds = np.std(X_test, axis=0, keepdims=True)
                    noise = np.random.normal(0, noise_level * feature_stds, X_test.shape)
                    X_noisy = X_test.copy() + noise
                    
                    # Evaluate on noisy data
                    y_pred_noisy = model.predict(X_noisy)
                    y_proba_noisy = model.predict_proba(X_noisy)[:, 1] if hasattr(model, 'predict_proba') else None
                    
                    noise_accs.append(accuracy_score(y_test, y_pred_noisy))
                    if y_proba_noisy is not None:
                        noise_aucs.append(roc_auc_score(y_test, y_proba_noisy))
                    noise_f1s.append(f1_score(y_test, y_pred_noisy))
                
                noise_results[noise_level] = {
                    "accuracy": np.mean(noise_accs),
                    "accuracy_std": np.std(noise_accs),
                    "roc_auc": np.mean(noise_aucs) if y_proba is not None else np.nan,
                    "roc_auc_std": np.std(noise_aucs) if y_proba is not None else np.nan,
                    "f1": np.mean(noise_f1s),
                    "f1_std": np.std(noise_f1s)
                }
            
            results["noise_robustness"][model_name] = noise_results
            
            # Feature permutation test
            permutation_accs = []
            permutation_aucs = []
            permutation_f1s = []
            
            for _ in range(n_permutations):
                # Create copy of test data
                X_perm = X_test.copy()
                
                # Randomly select a feature to permute
                feature_idx = np.random.randint(0, X_test.shape[1])
                
                # Permute the selected feature
                X_perm[:, feature_idx] = np.random.permutation(X_perm[:, feature_idx])
                
                # Evaluate on permuted data
                y_pred_perm = model.predict(X_perm)
                y_proba_perm = model.predict_proba(X_perm)[:, 1] if hasattr(model, 'predict_proba') else None
                
                permutation_accs.append(accuracy_score(y_test, y_pred_perm))
                if y_proba_perm is not None:
                    permutation_aucs.append(roc_auc_score(y_test, y_proba_perm))
                permutation_f1s.append(f1_score(y_test, y_pred_perm))
            
            results["feature_permutation"][model_name] = {
                "accuracy": np.mean(permutation_accs),
                "accuracy_std": np.std(permutation_accs),
                "roc_auc": np.mean(permutation_aucs) if y_proba is not None else np.nan,
                "roc_auc_std": np.std(permutation_aucs) if y_proba is not None else np.nan,
                "f1": np.mean(permutation_f1s),
                "f1_std": np.std(permutation_f1s)
            }
            
            # Feature dropout test
            dropout_results = {}
            
            # Test dropping each feature
            for feature_idx in range(X_test.shape[1]):
                # Create mask for all features except the current one
                mask = np.ones(X_test.shape[1], dtype=bool)
                mask[feature_idx] = False
                
                # Create dataset with feature dropped
                X_dropped = X_test[:, mask]
                
                # Skip if model can't handle reduced feature set
                try:
                    y_pred_dropped = model.predict(X_dropped)
                    y_proba_dropped = model.predict_proba(X_dropped)[:, 1] if hasattr(model, 'predict_proba') else None
                    
                    dropout_results[feature_names[feature_idx]] = {
                        "accuracy": accuracy_score(y_test, y_pred_dropped),
                        "roc_auc": roc_auc_score(y_test, y_proba_dropped) if y_proba_dropped is not None else np.nan,
                        "f1": f1_score(y_test, y_pred_dropped)
                    }
                except Exception as e:
                    dropout_results[feature_names[feature_idx]] = {
                        "accuracy": np.nan,
                        "roc_auc": np.nan,
                        "f1": np.nan,
                        "error": str(e)
                    }
            
            results["feature_dropout"][model_name] = dropout_results
            
            # Bootstrap stability test
            bootstrap_accs = []
            bootstrap_aucs = []
            bootstrap_f1s = []
            
            for _ in range(n_bootstrap_samples):
                # Create bootstrap sample
                n_samples = int(len(X_test) * bootstrap_size)
                indices = np.random.choice(len(X_test), size=n_samples, replace=True)
                X_boot = X_test[indices]
                y_boot = y_test[indices]
                
                # Evaluate on bootstrap sample
                y_pred_boot = model.predict(X_boot)
                y_proba_boot = model.predict_proba(X_boot)[:, 1] if hasattr(model, 'predict_proba') else None
                
                bootstrap_accs.append(accuracy_score(y_boot, y_pred_boot))
                if y_proba_boot is not None:
                    bootstrap_aucs.append(roc_auc_score(y_boot, y_proba_boot))
                bootstrap_f1s.append(f1_score(y_boot, y_pred_boot))
            
            results["bootstrap_stability"][model_name] = {
                "accuracy": np.mean(bootstrap_accs),
                "accuracy_std": np.std(bootstrap_accs),
                "roc_auc": np.mean(bootstrap_aucs) if y_proba is not None else np.nan,
                "roc_auc_std": np.std(bootstrap_aucs) if y_proba is not None else np.nan,
                "f1": np.mean(bootstrap_f1s),
                "f1_std": np.std(bootstrap_f1s)
            }
            
        except Exception as e:
            print(f"Error testing model {model_name}: {e}")
            results["errors"] = results.get("errors", {})
            results["errors"][model_name] = str(e)
    
    return results
