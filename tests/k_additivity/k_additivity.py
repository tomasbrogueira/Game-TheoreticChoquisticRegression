"""
K-additivity analysis utilities for Choquet integral models.

This module provides functions for analyzing the impact of k-additivity
on model performance, computational efficiency, and interpretability.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import time

from core.models.choquet import nParam_kAdd, choquet_k_additive_game, choquet_k_additive_mobius
from utils.data_loader import func_read_data


def direct_k_additivity_analysis(
    dataset_name, 
    representation="game", 
    output_dir=None, 
    test_size=0.3, 
    random_state=42, 
    regularization='l2'
):
    """
    Complete analysis of k-additivity impact using direct implementation.
    
    Parameters
    ----------
    dataset_name : str
        Name of the dataset to analyze
    representation : str, default="game"
        Choquet representation to use, either "game" or "mobius"
    output_dir : str, optional
        Directory to save results (default: creates folder based on dataset and representation)
    test_size : float, default=0.3
        Proportion of data to use for testing
    random_state : int, default=42
        Random seed for reproducibility
    regularization : str, default='l2'
        Regularization type for logistic regression ('l1', 'l2', 'elasticnet', or 'none')
        
    Returns
    -------
    pandas.DataFrame
        Results dataframe with performance metrics for each k value
    """
    # Validate representation parameter
    if representation not in ["game", "mobius"]:
        raise ValueError(f"Invalid representation '{representation}'. Use 'game' or 'mobius'.")
        
    print(f"\n{'='*80}\nAnalyzing dataset: {dataset_name} with {representation} representation\n{'='*80}")
    
    # Create output directory if not specified
    if output_dir is None:
        output_dir = f"k_additivity_analysis_{dataset_name}_{representation}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and prepare data
    X, y = func_read_data(dataset_name)
    nSamp, nAttr = X.shape
    
    # Split data into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Convert to numpy arrays
    X_train_values = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
    X_test_values = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
    y_train_values = y_train.values if isinstance(y_train, pd.Series) else y_train
    y_test_values = y_test.values if isinstance(y_test, pd.Series) else y_test
    
    print(f"Dataset: {dataset_name}")
    print(f"- Representation: {representation}")
    print(f"- Number of samples: {nSamp} (train: {len(X_train)}, test: {len(X_test)})")
    print(f"- Number of attributes: {nAttr}")
    print(f"- Will test k-additivity from 1 to {nAttr}")
    
    # Save dataset info
    with open(os.path.join(output_dir, "dataset_info.txt"), 'w') as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Representation: {representation}\n")
        f.write(f"Number of samples: {nSamp}\n")
        f.write(f"Number of attributes: {nAttr}\n")
        f.write(f"Train samples: {len(X_train)}\n")
        f.write(f"Test samples: {len(X_test)}\n")
    
    # Results dataframe for all k values
    results_df = pd.DataFrame(
        index=range(1, nAttr + 1),
        columns=[
            'k_value', 
            'n_params', 
            'train_time', 
            'baseline_accuracy',
            'noise_0.05', 'noise_0.1', 'noise_0.2', 'noise_0.3',
            'bootstrap_mean', 'bootstrap_std'
        ]
    )
    
    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_values)
    X_test_scaled = scaler.transform(X_test_values)
    
    # Select the appropriate transformation function based on representation
    choquet_transform = choquet_k_additive_game if representation == "game" else choquet_k_additive_mobius
    
    # Train and evaluate a model for each k value
    for k in range(1, nAttr + 1):
        print(f"\nTraining model with k = {k}/{nAttr}...")
        
        # Record number of parameters
        n_params = nParam_kAdd(k, nAttr)
        results_df.loc[k, 'k_value'] = k
        results_df.loc[k, 'n_params'] = n_params
        
        # Apply Choquet transformation with the selected representation
        start_time = time.time()
        try:
            X_train_choquet = choquet_transform(X_train_scaled, k_add=k)
            
            # Train logistic regression on transformed data
            model = LogisticRegression(
                max_iter=1000, 
                random_state=random_state,
                solver="newton-cg" if regularization in ['l2', 'none'] else "saga",
                penalty=None if regularization == 'none' else regularization
            )
            model.fit(X_train_choquet, y_train_values)
            train_time = time.time() - start_time
            results_df.loc[k, 'train_time'] = train_time
            
            print(f"- Trained with {n_params} parameters in {train_time:.2f} seconds")
            
            # Transform test data and evaluate baseline performance
            X_test_choquet = choquet_transform(X_test_scaled, k_add=k)
            y_pred = model.predict(X_test_choquet)
            baseline_acc = accuracy_score(y_test_values, y_pred)
            results_df.loc[k, 'baseline_accuracy'] = baseline_acc
            print(f"- Baseline accuracy: {baseline_acc:.4f}")
            
            # Noise robustness testing
            print("- Testing noise robustness...")
            noise_levels = [0.05, 0.1, 0.2, 0.3]
            
            for noise_level in noise_levels:
                noise_accuracies = []
                
                # Run multiple noise tests and average results
                for _ in range(5):  # 5 repetitions for each noise level
                    # Apply Gaussian noise to test set (scale by data std)
                    feature_stds = np.std(X_test_scaled, axis=0, keepdims=True)
                    noise = np.random.normal(0, 1, X_test_scaled.shape) * noise_level * feature_stds
                    X_test_noisy = X_test_scaled.copy() + noise
                    
                    # Transform noisy data and evaluate
                    X_test_noisy_choquet = choquet_transform(X_test_noisy, k_add=k)
                    y_pred_noisy = model.predict(X_test_noisy_choquet)
                    noise_acc = accuracy_score(y_test_values, y_pred_noisy)
                    noise_accuracies.append(noise_acc)
                
                # Record average noise performance
                avg_noise_acc = np.mean(noise_accuracies)
                results_df.loc[k, f'noise_{noise_level}'] = avg_noise_acc
                print(f"  - Noise level {noise_level}: {avg_noise_acc:.4f}")
            
            # Bootstrap stability testing
            print("- Testing bootstrap stability...")
            bootstrap_accuracies = []
            
            # Run multiple bootstrap tests
            for _ in range(30):  # 30 bootstrap samples
                # Create bootstrap sample (with replacement)
                n_samples = len(X_test_scaled)
                indices = np.random.choice(n_samples, size=n_samples, replace=True)
                X_bootstrap = X_test_scaled[indices]
                y_bootstrap = y_test_values[indices]
                
                # Transform bootstrap data and evaluate
                X_bootstrap_choquet = choquet_transform(X_bootstrap, k_add=k)
                y_pred_bootstrap = model.predict(X_bootstrap_choquet)
                bootstrap_acc = accuracy_score(y_bootstrap, y_pred_bootstrap)
                bootstrap_accuracies.append(bootstrap_acc)
            
            # Record bootstrap statistics
            bootstrap_mean = np.mean(bootstrap_accuracies)
            bootstrap_std = np.std(bootstrap_accuracies)
            results_df.loc[k, 'bootstrap_mean'] = bootstrap_mean
            results_df.loc[k, 'bootstrap_std'] = bootstrap_std
            print(f"  - Bootstrap accuracy: {bootstrap_mean:.4f} ± {bootstrap_std:.4f}")
            
        except Exception as e:
            print(f"Error processing k={k}: {e}")
            results_df.loc[k, 'error'] = str(e)
    
    # Save results to CSV
    results_path = os.path.join(output_dir, f"k_additivity_results_{dataset_name}_{representation}.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")
    
    return results_df


def find_optimal_k(results_df, accuracy_weight=0.7, complexity_weight=0.2, robustness_weight=0.1):
    """
    Find the optimal k value based on a weighted combination of accuracy, complexity, and robustness.
    
    Parameters
    ----------
    results_df : pandas.DataFrame
        Results dataframe from direct_k_additivity_analysis
    accuracy_weight : float, default=0.7
        Weight for accuracy in the optimization
    complexity_weight : float, default=0.2
        Weight for model complexity (inverse of parameter count) in the optimization
    robustness_weight : float, default=0.1
        Weight for robustness (noise and bootstrap stability) in the optimization
        
    Returns
    -------
    int
        Optimal k value
    dict
        Scores for each k value
    """
    # Create a copy to avoid modifying the original
    df = results_df.copy()
    
    # Normalize parameter count (inverse, fewer is better)
    max_params = df['n_params'].max()
    df['complexity_score'] = 1 - (df['n_params'] / max_params)
    
    # Normalize accuracy
    min_acc = df['baseline_accuracy'].min()
    max_acc = df['baseline_accuracy'].max()
    acc_range = max_acc - min_acc
    if acc_range > 0:
        df['accuracy_score'] = (df['baseline_accuracy'] - min_acc) / acc_range
    else:
        df['accuracy_score'] = 1.0  # All accuracies are the same
    
    # Calculate robustness score (average of noise robustness and bootstrap stability)
    noise_cols = [col for col in df.columns if col.startswith('noise_')]
    if noise_cols:
        # Normalize noise robustness
        for col in noise_cols:
            min_noise = df[col].min()
            max_noise = df[col].max()
            noise_range = max_noise - min_noise
            if noise_range > 0:
                df[f'{col}_norm'] = (df[col] - min_noise) / noise_range
            else:
                df[f'{col}_norm'] = 1.0
        
        # Average normalized noise scores
        df['noise_score'] = df[[f'{col}_norm' for col in noise_cols]].mean(axis=1)
    else:
        df['noise_score'] = 0.5  # Default if no noise tests
    
    # Normalize bootstrap stability (higher mean and lower std is better)
    if 'bootstrap_mean' in df.columns and 'bootstrap_std' in df.columns:
        # Normalize bootstrap mean
        min_boot_mean = df['bootstrap_mean'].min()
        max_boot_mean = df['bootstrap_mean'].max()
        boot_mean_range = max_boot_mean - min_boot_mean
        if boot_mean_range > 0:
            df['bootstrap_mean_norm'] = (df['bootstrap_mean'] - min_boot_mean) / boot_mean_range
        else:
            df['bootstrap_mean_norm'] = 1.0
        
        # Normalize bootstrap std (inverse, lower is better)
        max_boot_std = df['bootstrap_std'].max()
        if max_boot_std > 0:
            df['bootstrap_std_norm'] = 1 - (df['bootstrap_std'] / max_boot_std)
        else:
            df['bootstrap_std_norm'] = 1.0
        
        # Combine bootstrap metrics
        df['bootstrap_score'] = 0.7 * df['bootstrap_mean_norm'] + 0.3 * df['bootstrap_std_norm']
    else:
        df['bootstrap_score'] = 0.5  # Default if no bootstrap tests
    
    # Calculate overall robustness score
    df['robustness_score'] = 0.6 * df['noise_score'] + 0.4 * df['bootstrap_score']
    
    # Calculate final weighted score
    df['final_score'] = (
        accuracy_weight * df['accuracy_score'] +
        complexity_weight * df['complexity_score'] +
        robustness_weight * df['robustness_score']
    )
    
    # Find optimal k
    optimal_k = df['final_score'].idxmax()
    
    # Create scores dictionary
    scores = {
        k: {
            'accuracy': df.loc[k, 'accuracy_score'],
            'complexity': df.loc[k, 'complexity_score'],
            'robustness': df.loc[k, 'robustness_score'],
            'final_score': df.loc[k, 'final_score']
        }
        for k in df.index
    }
    
    return optimal_k, scores
