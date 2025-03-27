import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.utils import resample
from sklearn.preprocessing import MinMaxScaler
import os
from typing import List, Dict, Any, Tuple, Union
import seaborn as sns
from tqdm import tqdm

def test_model_robustness(
    models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str] = None,
    noise_levels: List[float] = [0.05, 0.1, 0.2, 0.3, 0.5],
    n_permutations: int = 10,
    feature_dropout_count: int = None,
    n_bootstrap_samples: int = 100,
    bootstrap_size: float = 0.8,
    output_folder: str = "robustness_results",
    random_state: int = 0
) -> Dict[str, Any]:
    """
    Test the robustness of multiple models under various perturbations.
    
    Parameters:
    -----------
    models : Dict[str, Any]
        Dictionary of trained models with model names as keys
    X_test : np.ndarray
        Test data features
    y_test : np.ndarray
        Test data labels
    feature_names : List[str], optional
        Names of features for better visualization
    noise_levels : List[float], default=[0.05, 0.1, 0.2, 0.3, 0.5]
        Levels of Gaussian noise to add (as proportion of feature std dev)
    n_permutations : int, default=10
        Number of random permutations for feature permutation test
    feature_dropout_count : int, default=None
        Number of features to drop (default: 1/3 of features)
    n_bootstrap_samples : int, default=100
        Number of bootstrap samples for stability testing
    bootstrap_size : float, default=0.8
        Size of each bootstrap sample as proportion of test set
    scale_data : bool, default=True
        Whether to scale data after perturbation (required for some methods)
    output_folder : str, default="robustness_results"
        Folder to save result plots
    random_state : int, default=42
        Random seed for reproducibility
    
    Returns:
    --------
    Dict[str, Any]
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

    
    for model_name, model in models.items():
        try:
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
        except Exception as e:
            print(f"Error calculating baseline for {model_name}: {str(e)}")
            results["baseline"][model_name] = {
                "accuracy": np.nan,
                "roc_auc": np.nan,
                "f1": np.nan
            }
    
    # 1. noise robustness test
    print("Testing noise robustness...")
    noise_results = {model_name: {level: {} for level in noise_levels} for model_name in models}

    for noise_level in noise_levels:
        np.random.seed(random_state)
        
        # Check if data is binary (all values are 0 or 1)
        is_binary = np.all(np.isin(X_test, [0, 1, True, False]))
        
        if is_binary:
            # Create a copy of the test data
            X_test_noisy = X_test.copy()
            
            # Generate the flip mask once (same for all models)
            flip_probability = noise_level
            flip_mask = np.random.random(X_test.shape) < flip_probability
            
            if isinstance(X_test_noisy, pd.DataFrame):
                # Handle DataFrame case more cleanly
                for col in X_test_noisy.columns:
                    # Get the mask for this column
                    col_idx = X_test_noisy.columns.get_loc(col)
                    col_flip = flip_mask[:, col_idx]
                    
                    # Access only rows that need flipping
                    X_test_noisy.loc[col_flip, col] = 1 - X_test_noisy.loc[col_flip, col]
            else:
                # Simple version for numpy arrays
                X_test_noisy[flip_mask] = 1 - X_test_noisy[flip_mask]
        else:
            # Original Gaussian noise approach for continuous data
            noise = np.random.normal(0, 1, X_test.shape) * noise_level
            if is_dataframe:
                feature_stds = np.std(X_test.values, axis=0, keepdims=True)
            else:
                feature_stds = np.std(X_test, axis=0, keepdims=True)
            X_test_noisy = X_test + noise * feature_stds
        
        for model_name, model in models.items():
            try:
                y_pred = model.predict(X_test_noisy)
                y_proba = model.predict_proba(X_test_noisy)[:, 1] if hasattr(model, 'predict_proba') else None
                
                acc = accuracy_score(y_test, y_pred)
                auc = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan
                f1 = f1_score(y_test, y_pred)
                
                noise_results[model_name][noise_level] = {
                    "accuracy": acc,
                    "roc_auc": auc,
                    "f1": f1
                }
            except Exception as e:
                print(f"Error in noise test for {model_name} at level {noise_level}: {str(e)}")
                noise_results[model_name][noise_level] = {
                    "accuracy": np.nan,
                    "roc_auc": np.nan,
                    "f1": np.nan
                }
    
    results["noise_robustness"] = noise_results
    
    # 2 feature permutation test
    print("Testing feature permutation robustness...")
    permutation_results = {model_name: {f: {} for f in range(X_test.shape[1])} for model_name in models}

    for feat_idx in range(X_test.shape[1]):
        feature_name = feature_names[feat_idx]
        print(f"  Permuting feature: {feature_name}")
        
        acc_drops = {model_name: [] for model_name in models}
        auc_drops = {model_name: [] for model_name in models}
        f1_drops = {model_name: [] for model_name in models}
        
        for perm in range(n_permutations):
            np.random.seed(random_state + perm)
            
            # Create a copy with one permuted feature
            X_test_perm = X_test.copy()
            perm_idx = np.random.permutation(len(X_test))
            if isinstance(X_test_perm, pd.DataFrame):
                # For DataFrame, permute using column name
                col_name = X_test_perm.columns[feat_idx]
                X_test_perm[col_name] = X_test_perm[col_name].values[perm_idx]
            else:
                # For numpy array, permute using index
                X_test_perm[:, feat_idx] = X_test_perm[perm_idx, feat_idx]
            
            for model_name, model in models.items():
                try:
                    # Get baseline metrics
                    baseline_acc = results["baseline"][model_name]["accuracy"]
                    baseline_auc = results["baseline"][model_name]["roc_auc"]
                    baseline_f1 = results["baseline"][model_name]["f1"]
                    
                    # Get metrics with permuted feature
                    y_pred = model.predict(X_test_perm)
                    y_proba = model.predict_proba(X_test_perm)[:, 1] if hasattr(model, 'predict_proba') else None
                    
                    perm_acc = accuracy_score(y_test, y_pred)
                    perm_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan
                    perm_f1 = f1_score(y_test, y_pred)
                    
                    # Calculate drops
                    acc_drops[model_name].append(baseline_acc - perm_acc)
                    auc_drops[model_name].append(baseline_auc - perm_auc if not np.isnan(baseline_auc) and not np.isnan(perm_auc) else np.nan)
                    f1_drops[model_name].append(baseline_f1 - perm_f1)
                    
                except Exception as e:
                    print(f"Error in permutation test for {model_name}, feature {feature_name}: {str(e)}")
                    acc_drops[model_name].append(np.nan)
                    auc_drops[model_name].append(np.nan)
                    f1_drops[model_name].append(np.nan)
        
        # Store average drops
        for model_name in models:
            permutation_results[model_name][feat_idx] = {
                "feature_name": feature_name,
                "accuracy_drop": np.nanmean(acc_drops[model_name]),
                "roc_auc_drop": np.nanmean(auc_drops[model_name]),
                "f1_drop": np.nanmean(f1_drops[model_name])
            }
    
    results["feature_permutation"] = permutation_results
    
    # 3. FEATURE DROPOUT TEST
    # -----------------------
    print("Testing feature dropout robustness...")
    dropout_results = {model_name: {} for model_name in models}
    
    # Test dropping subsets of features
    np.random.seed(random_state)
    n_features = X_test.shape[1]
    
    # Try multiple random subsets
    for trial in range(min(n_permutations, 10)):
        features_to_drop = np.random.choice(
            range(n_features), 
            size=feature_dropout_count, 
            replace=False
        )
        
        # Create a mask for features to keep
        keep_mask = np.ones(n_features, dtype=bool)
        keep_mask[features_to_drop] = False
        
        
        # For each model, retrain on the reduced feature set and evaluate
        for model_name, model in models.items():
            try:
                X_test_zeros = X_test.copy()
                
                if isinstance(X_test_zeros, pd.DataFrame):
                    for idx in features_to_drop:
                        col_name = X_test_zeros.columns[idx]
                        X_test_zeros[col_name] = 0
                else:
                    X_test_zeros[:, features_to_drop] = 0
                
                y_pred = model.predict(X_test_zeros)
                y_proba = model.predict_proba(X_test_zeros)[:, 1] if hasattr(model, 'predict_proba') else None
                
                acc = accuracy_score(y_test, y_pred)
                auc = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan
                f1 = f1_score(y_test, y_pred)
                
                trial_key = f"trial_{trial}"
                if trial_key not in dropout_results[model_name]:
                    dropout_results[model_name][trial_key] = {}
                
                dropout_results[model_name][trial_key] = {
                    "dropped_features": [feature_names[i] for i in features_to_drop],
                    "accuracy": acc,
                    "roc_auc": auc,
                    "f1": f1
                }
            except Exception as e:
                print(f"Error in dropout test for {model_name}, trial {trial}: {str(e)}")
                trial_key = f"trial_{trial}"
                if trial_key not in dropout_results[model_name]:
                    dropout_results[model_name][trial_key] = {}
                
                dropout_results[model_name][trial_key] = {
                    "dropped_features": [feature_names[i] for i in features_to_drop],
                    "accuracy": np.nan,
                    "roc_auc": np.nan,
                    "f1": np.nan
                }
    
    # Compute average performance across trials
    for model_name in models:
        acc_values = [dropout_results[model_name][f"trial_{t}"]["accuracy"] 
                     for t in range(min(n_permutations, 10))]
        auc_values = [dropout_results[model_name][f"trial_{t}"]["roc_auc"] 
                     for t in range(min(n_permutations, 10))]
        f1_values = [dropout_results[model_name][f"trial_{t}"]["f1"] 
                    for t in range(min(n_permutations, 10))]
        
        dropout_results[model_name]["average"] = {
            "accuracy": np.nanmean(acc_values),
            "roc_auc": np.nanmean(auc_values),
            "f1": np.nanmean(f1_values)
        }
    
    results["feature_dropout"] = dropout_results
    
    # 4. BOOTSTRAP STABILITY TEST
    # ---------------------------
    print("Testing bootstrap stability...")
    bootstrap_results = {model_name: [] for model_name in models}
    
    np.random.seed(random_state)
    n_samples = len(X_test)
    sample_size = int(bootstrap_size * n_samples)
    
    for i in tqdm(range(n_bootstrap_samples)):
        # Create bootstrap sample
        indices = np.random.choice(n_samples, size=sample_size, replace=True)
        
        if is_dataframe:
            X_boot = X_test.iloc[indices].copy()
        else:
            X_boot = X_test[indices]
            
        if isinstance(y_test, pd.Series):
            y_boot = y_test.iloc[indices]
        else:
            y_boot = y_test[indices]
        
        for model_name, model in models.items():
            try:
                y_pred = model.predict(X_boot)
                y_proba = model.predict_proba(X_boot)[:, 1] if hasattr(model, 'predict_proba') else None
                
                acc = accuracy_score(y_boot, y_pred)
                auc = roc_auc_score(y_boot, y_proba) if y_proba is not None else np.nan
                f1 = f1_score(y_boot, y_pred)
                
                bootstrap_results[model_name].append({
                    "accuracy": acc,
                    "roc_auc": auc,
                    "f1": f1
                })
            except Exception as e:
                print(f"Error in bootstrap test for {model_name}, sample {i}: {str(e)}")
                bootstrap_results[model_name].append({
                    "accuracy": np.nan,
                    "roc_auc": np.nan,
                    "f1": np.nan
                })
    
    # Calculate mean and standard deviation for each metric
    for model_name in models:
        try:
            acc_values = [bs["accuracy"] for bs in bootstrap_results[model_name]]
            auc_values = [bs["roc_auc"] for bs in bootstrap_results[model_name]]
            f1_values = [bs["f1"] for bs in bootstrap_results[model_name]]
            
            results["bootstrap_stability"][model_name] = {
                "accuracy_mean": np.nanmean(acc_values),
                "accuracy_std": np.nanstd(acc_values),
                "roc_auc_mean": np.nanmean(auc_values),
                "roc_auc_std": np.nanstd(auc_values),
                "f1_mean": np.nanmean(f1_values),
                "f1_std": np.nanstd(f1_values)
            }
        except Exception as e:
            print(f"Error calculating bootstrap stats for {model_name}: {str(e)}")
            results["bootstrap_stability"][model_name] = {
                "accuracy_mean": np.nan,
                "accuracy_std": np.nan,
                "roc_auc_mean": np.nan,
                "roc_auc_std": np.nan,
                "f1_mean": np.nan,
                "f1_std": np.nan
            }
    
    # PLOT RESULTS
    # ------------
    # 1. Noise robustness plot
    plt.figure(figsize=(12, 8))
    for model_name in models:
        noise_acc = [noise_results[model_name][level]["accuracy"] for level in noise_levels]
        plt.plot(noise_levels, noise_acc, 'o-', label=model_name)
    
    plt.xlabel('Noise Level (× feature std)')
    plt.ylabel('Accuracy')
    plt.title('Model Robustness to Noise')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_folder, 'noise_robustness.png'), dpi=300, bbox_inches='tight')
    
    # 2. Feature permutation importance
    avg_perm_importance = {model_name: {} for model_name in models}
    
    for model_name in models:
        feat_importance = []
        for feat_idx in range(X_test.shape[1]):
            feat_importance.append(permutation_results[model_name][feat_idx]["accuracy_drop"])
        avg_perm_importance[model_name] = feat_importance
    
    # Plot permutation importance
    plt.figure(figsize=(14, 8))
    
    x = np.arange(len(feature_names))
    width = 0.8 / len(models)
    
    for i, (model_name, importances) in enumerate(avg_perm_importance.items()):
        plt.bar(x + i*width - 0.4 + width/2, importances, width, label=model_name)
    
    plt.xlabel('Features')
    plt.ylabel('Accuracy Drop (Importance)')
    plt.title('Feature Importance via Permutation')
    plt.xticks(x, feature_names, rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'feature_importance.png'), dpi=300, bbox_inches='tight')
    
    # 3. Feature dropout robustness
    plt.figure(figsize=(10, 6))
    model_names = list(models.keys())
    
    baseline_acc = [results["baseline"][m]["accuracy"] for m in model_names]
    dropout_acc = [results["feature_dropout"][m]["average"]["accuracy"] for m in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    plt.bar(x - width/2, baseline_acc, width, label='Baseline')
    plt.bar(x + width/2, dropout_acc, width, label=f'With {feature_dropout_count} Features Dropped')
    
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title(f'Model Robustness to Feature Dropout ({feature_dropout_count} features)')
    plt.xticks(x, model_names)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'feature_dropout.png'), dpi=300, bbox_inches='tight')
    
    # 4. Bootstrap stability
    plt.figure(figsize=(10, 6))
    
    acc_means = [results["bootstrap_stability"][m]["accuracy_mean"] for m in model_names]
    acc_stds = [results["bootstrap_stability"][m]["accuracy_std"] for m in model_names]
    
    plt.bar(model_names, acc_means, yerr=acc_stds, capsize=5)
    plt.xlabel('Models')
    plt.ylabel('Bootstrap Accuracy')
    plt.title(f'Model Stability (Bootstrap n={n_bootstrap_samples})')
    plt.ylim(min(min(acc_means) - max(acc_stds) - 0.05, 0.5), 1.0)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'bootstrap_stability.png'), dpi=300, bbox_inches='tight')
    
    # Summary heatmap for model comparison
    plt.figure(figsize=(12, 8))
    
    # Prepare data for heatmap
    model_summary = {
        "Baseline Accuracy": [results["baseline"][m]["accuracy"] for m in model_names],
        "Noise Robustness": [noise_results[m][noise_levels[-1]]["accuracy"] for m in model_names],
        "Feature Dropout": [results["feature_dropout"][m]["average"]["accuracy"] for m in model_names],
        "Bootstrap Stability": [results["bootstrap_stability"][m]["accuracy_std"] * -1 for m in model_names]  # Negative because lower std is better
    }
    
    # Normalize values
    for metric in model_summary:
        values = model_summary[metric]
        if metric == "Baseline Accuracy":
            continue
        elif metric == "Bootstrap Stability":  
            min_val, max_val = min(values), max(values)
            model_summary[metric] = [(v - min_val) / (max_val - min_val) if max_val > min_val else 0.5 for v in values]
        else:
            min_val, max_val = min(values), max(values)
            model_summary[metric] = [(v - min_val) / (max_val - min_val) if max_val > min_val else 0.5 for v in values]
    
    summary_df = pd.DataFrame(model_summary, index=model_names)
    
    # Create heatmap
    sns.heatmap(summary_df, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Normalized Score'})
    plt.title('Model Robustness Summary (Normalized Scores)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'robustness_summary.png'), dpi=300, bbox_inches='tight')
    
    # Save full results
    with open(os.path.join(output_folder, 'robustness_results.txt'), 'w') as f:
        f.write("MODEL ROBUSTNESS TEST RESULTS\n")
        f.write("=============================\n\n")
        
        # Baseline results
        f.write("Baseline Performance:\n")
        f.write("-" * 50 + "\n")
        for model_name in models:
            f.write(f"{model_name}:\n")
            f.write(f"  - Accuracy: {results['baseline'][model_name]['accuracy']:.4f}\n")
            f.write(f"  - ROC AUC:  {results['baseline'][model_name]['roc_auc']:.4f}\n")
            f.write(f"  - F1 Score: {results['baseline'][model_name]['f1']:.4f}\n")
        f.write("\n")
        
        # Noise robustness
        f.write("Noise Robustness (Accuracy):\n")
        f.write("-" * 50 + "\n")
        f.write("Noise Level | " + " | ".join(model_names) + "\n")
        for level in noise_levels:
            f.write(f"{level:.2f}        | ")
            for model_name in model_names:
                f.write(f"{noise_results[model_name][level]['accuracy']:.4f} | ")
            f.write("\n")
        f.write("\n")
        
        # Feature permutation (top 5 most important)
        f.write("Feature Importance (Top 5 via Permutation):\n")
        f.write("-" * 50 + "\n")
        for model_name in model_names:
            f.write(f"{model_name}:\n")
            importance_scores = []
            for feat_idx in range(X_test.shape[1]):
                score = permutation_results[model_name][feat_idx]["accuracy_drop"]
                importance_scores.append((feature_names[feat_idx], score))
            
            importance_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Display top 5
            for i, (feat, score) in enumerate(importance_scores[:5]):
                f.write(f"  {i+1}. {feat}: {score:.4f}\n")
            f.write("\n")
        
        # Feature dropout summary
        f.write(f"Feature Dropout Summary ({feature_dropout_count} features):\n")
        f.write("-" * 50 + "\n")
        for model_name in model_names:
            baseline = results["baseline"][model_name]["accuracy"]
            dropout = results["feature_dropout"][model_name]["average"]["accuracy"]
            drop_pct = 100 * (baseline - dropout) / baseline if baseline > 0 else float('nan')
            
            f.write(f"{model_name}:\n")
            f.write(f"  - Baseline:   {baseline:.4f}\n")
            f.write(f"  - With drops: {dropout:.4f}\n")
            f.write(f"  - % Change:   {drop_pct:.2f}%\n")
        f.write("\n")
        
        # Bootstrap stability
        f.write("Bootstrap Stability:\n")
        f.write("-" * 50 + "\n")
        for model_name in model_names:
            bs = results["bootstrap_stability"][model_name]
            f.write(f"{model_name}:\n")
            f.write(f"  - Accuracy: {bs['accuracy_mean']:.4f} ± {bs['accuracy_std']:.4f}\n")
            f.write(f"  - ROC AUC:  {bs['roc_auc_mean']:.4f} ± {bs['roc_auc_std']:.4f}\n")
            f.write(f"  - F1 Score: {bs['f1_mean']:.4f} ± {bs['f1_std']:.4f}\n")
        
    return results


def compare_regularization_robustness(none_results, l2_results, output_folder):
    """
    Creates comparison plots between models with no regularization and with L2 regularization.
    
    Parameters:
    -----------
    none_results : Dict
        Results from test_model_robustness for models without regularization
    l2_results : Dict
        Results from test_model_robustness for models with L2 regularization
    output_folder : str
        Folder to save comparison plots
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Get model names (should be the same in both result sets)
    model_names = list(none_results["baseline"].keys())
    
    # 1. Baseline Accuracy Comparison
    plt.figure(figsize=(12, 6))
    x = np.arange(len(model_names))
    width = 0.35
    
    none_acc = [none_results["baseline"][m]["accuracy"] for m in model_names]
    l2_acc = [l2_results["baseline"][m]["accuracy"] for m in model_names]
    
    plt.bar(x - width/2, none_acc, width, label='No Regularization')
    plt.bar(x + width/2, l2_acc, width, label='L2 Regularization')
    
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title('Baseline Accuracy: No Regularization vs L2')
    plt.xticks(x, model_names)
    plt.legend()
    plt.ylim(0.5, 1.0)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'baseline_accuracy_comparison.png'), dpi=300, bbox_inches='tight')
    
    # 2. Noise Robustness Comparison
    noise_levels = list(none_results["noise_robustness"][model_names[0]].keys())
    
    for model in model_names:
        plt.figure(figsize=(10, 6))
        
        none_noise_acc = [none_results["noise_robustness"][model][level]["accuracy"] for level in noise_levels]
        l2_noise_acc = [l2_results["noise_robustness"][model][level]["accuracy"] for level in noise_levels]
        
        plt.plot(noise_levels, none_noise_acc, 'o-', label='No Regularization')
        plt.plot(noise_levels, l2_noise_acc, 's--', label='L2 Regularization')
        
        plt.xlabel('Noise Level (× feature std)')
        plt.ylabel('Accuracy')
        plt.title(f'{model}: Noise Robustness Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_folder, f'{model}_noise_robustness.png'), dpi=300, bbox_inches='tight')
    
    # 3. Bootstrap Stability Comparison
    plt.figure(figsize=(12, 6))
    
    # Accuracy means
    none_acc_means = [none_results["bootstrap_stability"][m]["accuracy_mean"] for m in model_names]
    l2_acc_means = [l2_results["bootstrap_stability"][m]["accuracy_mean"] for m in model_names]
    
    # Accuracy standard deviations
    none_acc_stds = [none_results["bootstrap_stability"][m]["accuracy_std"] for m in model_names]
    l2_acc_stds = [l2_results["bootstrap_stability"][m]["accuracy_std"] for m in model_names]
    
    # Create bar positions
    x = np.arange(len(model_names))
    width = 0.35
    
    # Create grouped bars with error bars
    plt.bar(x - width/2, none_acc_means, width, yerr=none_acc_stds, capsize=5, label='No Regularization')
    plt.bar(x + width/2, l2_acc_means, width, yerr=l2_acc_stds, capsize=5, label='L2 Regularization')
    
    plt.xlabel('Models')
    plt.ylabel('Bootstrap Accuracy')
    plt.title('Bootstrap Stability: No Regularization vs L2')
    plt.xticks(x, model_names)
    plt.legend()
    plt.ylim(0.5, 1.0)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'bootstrap_stability_comparison.png'), dpi=300, bbox_inches='tight')
    
    # 4. Feature Dropout Resilience Comparison
    plt.figure(figsize=(12, 6))
    
    none_baseline = [none_results["baseline"][m]["accuracy"] for m in model_names]
    none_dropout = [none_results["feature_dropout"][m]["average"]["accuracy"] for m in model_names]
    none_drop_pct = [(b - d) / b if b > 0 else 0 for b, d in zip(none_baseline, none_dropout)]
    
    l2_baseline = [l2_results["baseline"][m]["accuracy"] for m in model_names]
    l2_dropout = [l2_results["feature_dropout"][m]["average"]["accuracy"] for m in model_names]
    l2_drop_pct = [(b - d) / b if b > 0 else 0 for b, d in zip(l2_baseline, l2_dropout)]
    
    plt.bar(x - width/2, none_drop_pct, width, label='No Regularization')
    plt.bar(x + width/2, l2_drop_pct, width, label='L2 Regularization')
    
    plt.xlabel('Models')
    plt.ylabel('Performance Drop (%)')
    plt.title('Feature Dropout Impact: No Regularization vs L2')
    plt.xticks(x, model_names)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'feature_dropout_comparison.png'), dpi=300, bbox_inches='tight')
    
    # 5. Summary heatmap
    # Prepare data for heatmap
    summary_data = []
    
    for model in model_names:
        # Non-regularized metrics
        none_baseline_acc = none_results["baseline"][model]["accuracy"]
        none_noise_robustness = none_results["noise_robustness"][model][noise_levels[-1]]["accuracy"]
        none_bootstrap_std = none_results["bootstrap_stability"][model]["accuracy_std"] 
        none_dropout_impact = (none_baseline[model_names.index(model)] - none_dropout[model_names.index(model)]) / none_baseline[model_names.index(model)] if none_baseline[model_names.index(model)] > 0 else 0

        
        # L2-regularized metrics
        l2_baseline_acc = l2_results["baseline"][model]["accuracy"]
        l2_noise_robustness = l2_results["noise_robustness"][model][noise_levels[-1]]["accuracy"]
        l2_bootstrap_std = l2_results["bootstrap_stability"][model]["accuracy_std"]
        l2_dropout_impact = (l2_baseline[model_names.index(model)] - l2_dropout[model_names.index(model)]) / l2_baseline[model_names.index(model)] if l2_baseline[model_names.index(model)] > 0 else 0
        
        row = {
            'Model': model,
            'Reg': 'None',
            'Baseline Acc': none_baseline_acc,
            'Noise Robustness': none_noise_robustness,
            'Bootstrap Stability': -none_bootstrap_std,  # Negative since lower std is better
            'Feature Dropout Resilience': -none_dropout_impact  # Negative since lower impact is better
        }
        summary_data.append(row)
        
        row = {
            'Model': model,
            'Reg': 'L2',
            'Baseline Acc': l2_baseline_acc,
            'Noise Robustness': l2_noise_robustness, 
            'Bootstrap Stability': -l2_bootstrap_std,
            'Feature Dropout Resilience': -l2_dropout_impact
        }
        summary_data.append(row)
    
    # Create DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Normalize metrics for fair comparison
    metrics = ['Noise Robustness', 'Bootstrap Stability', 'Feature Dropout Resilience']
    for metric in metrics:
        min_val = summary_df[metric].min()
        max_val = summary_df[metric].max()
        if max_val > min_val:
            summary_df[metric] = (summary_df[metric] - min_val) / (max_val - min_val)
    
    # Create a pivot table for the heatmap
    pivot_df = summary_df.pivot(index=['Model', 'Reg'], columns=[], values=metrics)
    
    # Plot the heatmap
    plt.figure(figsize=(10, len(model_names)*1.5))
    sns.heatmap(pivot_df, annot=True, cmap='YlGnBu', fmt='.2f', cbar_kws={'label': 'Normalized Score (higher is better)'})
    plt.title('Regularization Impact on Model Robustness')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'regularization_comparison_heatmap.png'), dpi=300, bbox_inches='tight')
    
    # Save summary to text file
    with open(os.path.join(output_folder, 'regularization_comparison.txt'), 'w') as f:
        f.write("REGULARIZATION IMPACT ON MODEL ROBUSTNESS\n")
        f.write("=======================================\n\n")
        
        for model in model_names:
            f.write(f"MODEL: {model}\n")
            f.write("-" * 50 + "\n")
            
            # Get data from results
            none_data = {
                "Baseline Accuracy": none_results["baseline"][model]["accuracy"],
                "Noise Robustness (max noise)": none_results["noise_robustness"][model][noise_levels[-1]]["accuracy"],
                "Bootstrap Stability (std)": none_results["bootstrap_stability"][model]["accuracy_std"],
                "Feature Dropout Impact": none_drop_pct[model_names.index(model)]
            }
            
            l2_data = {
                "Baseline Accuracy": l2_results["baseline"][model]["accuracy"],
                "Noise Robustness (max noise)": l2_results["noise_robustness"][model][noise_levels[-1]]["accuracy"],
                "Bootstrap Stability (std)": l2_results["bootstrap_stability"][model]["accuracy_std"],
                "Feature Dropout Impact": l2_drop_pct[model_names.index(model)]
            }
            
            f.write("Metric               | No Regularization | L2 Regularization | Difference (L2 - None)\n")
            f.write("-" * 80 + "\n")
            
            for metric, none_val in none_data.items():
                l2_val = l2_data[metric]
                diff = l2_val - none_val
                diff_str = f"{diff:+.4f}"
                
                # For std and dropout, negative difference is better
                if "Stability" in metric or "Dropout" in metric:
                    better = "BETTER" if diff < 0 else "WORSE"
                else:  # For accuracy metrics, positive difference is better
                    better = "BETTER" if diff > 0 else "WORSE"
                
                f.write(f"{metric.ljust(20)} | {none_val:.4f}           | {l2_val:.4f}          | {diff_str} {better}\n")
            
            f.write("\n\n")
    
    return