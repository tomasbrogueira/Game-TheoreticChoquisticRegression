import os
os.environ["SCIPY_ARRAY_API"] = "1"

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import time
from datetime import datetime
import seaborn as sns
from tqdm import tqdm
import warnings
from itertools import combinations
warnings.filterwarnings("ignore")

from regression_classes import nParam_kAdd, powerset, choquet_k_additive_game, choquet_k_additive_mobius
from mod_GenFuzzyRegression import func_read_data
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression


def direct_k_additivity_analysis(dataset_name, representation="game", output_dir=None, test_size=0.3, random_state=42, regularization='l2'):
    """
    Complete analysis of k-additivity impact using a direct implementation without 
    relying on the problematic ChoquisticRegression class.
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset to analyze
    representation : str, default="game"
        Choquet representation to use, either "game" or "mobius"
    output_dir : str, optional
        Directory to save results (default: creates timestamped folder)
    test_size : float, default=0.3
        Proportion of data to use for testing
    random_state : int, default=42
        Random seed for reproducibility
    regularization : str, default='l2'
        Regularization type for logistic regression ('l1', 'l2', 'elasticnet', or 'none')
    """
    # Validate representation parameter
    if representation not in ["game", "mobius"]:
        raise ValueError(f"Invalid representation '{representation}'. Use 'game' or 'mobius'.")
        
    print(f"\n{'='*80}\nAnalyzing dataset: {dataset_name} with {representation} representation\n{'='*80}")
    
    # Create output directory if not specified - simplified naming without timestamp
    if output_dir is None:
        output_dir = f"k_analysis_{dataset_name}_{representation}"
    
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
    scaler = MinMaxScaler()
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
                print(f"  Noise level {noise_level}: accuracy = {avg_noise_acc:.4f}")
            
            # Bootstrap stability testing
            print("- Testing bootstrap stability...")
            bootstrap_accuracies = []
            n_bootstrap = 50  # Number of bootstrap samples
            
            for _ in range(n_bootstrap):
                # Create bootstrap sample
                indices = np.random.choice(len(X_test_scaled), size=int(0.8*len(X_test_scaled)), replace=True)
                X_boot = X_test_scaled[indices]
                y_boot = y_test_values[indices]
                
                # Transform bootstrap data and evaluate
                X_boot_choquet = choquet_transform(X_boot, k_add=k)
                y_pred_boot = model.predict(X_boot_choquet)
                boot_acc = accuracy_score(y_boot, y_pred_boot)
                bootstrap_accuracies.append(boot_acc)
            
            # Record bootstrap results
            bootstrap_mean = np.mean(bootstrap_accuracies)
            bootstrap_std = np.std(bootstrap_accuracies)
            results_df.loc[k, 'bootstrap_mean'] = bootstrap_mean
            results_df.loc[k, 'bootstrap_std'] = bootstrap_std
            print(f"  Bootstrap: mean = {bootstrap_mean:.4f}, std = {bootstrap_std:.4f}")
            
        except Exception as e:
            print(f"Error processing k={k}: {str(e)}")
            # Fill with NaN to indicate missing data
            for col in results_df.columns[2:]:  # Skip k_value and n_params
                results_df.loc[k, col] = np.nan
    
    # Save complete results
    results_df.to_csv(os.path.join(output_dir, "k_comparison_results.csv"))
    
    # Create plots
    print("\nGenerating plots...")
    
    # 1. Number of parameters vs k plot
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['k_value'], results_df['n_params'], 'o-', linewidth=2)
    plt.title(f'Number of Parameters vs k-additivity ({dataset_name}, {representation})')
    plt.xlabel('k-additivity')
    plt.ylabel('Number of Parameters')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.savefig(os.path.join(output_dir, "parameter_counts.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Baseline accuracy vs k
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['k_value'], results_df['baseline_accuracy'], 'o-', linewidth=2)
    plt.title(f'Baseline Accuracy vs k-additivity ({dataset_name}, {representation})')
    plt.xlabel('k-additivity')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "baseline_accuracy_vs_k.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Noise robustness comparison
    plt.figure(figsize=(12, 8))
    cmap = get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, 4))  # 4 noise levels
    
    noise_levels = [0.05, 0.1, 0.2, 0.3]
    for i, noise_level in enumerate(noise_levels):
        plt.plot(results_df['k_value'], results_df[f'noise_{noise_level}'], 'o-', 
                 color=colors[i], linewidth=2, label=f'Noise level: {noise_level}')
    
    plt.plot(results_df['k_value'], results_df['baseline_accuracy'], 'k--', 
             linewidth=2, label='Baseline (no noise)')
    
    plt.title(f'Noise Robustness vs k-additivity ({dataset_name}, {representation})')
    plt.xlabel('k-additivity')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "noise_robustness.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Bootstrap stability
    plt.figure(figsize=(10, 6))
    plt.errorbar(results_df['k_value'], results_df['bootstrap_mean'], 
                 yerr=results_df['bootstrap_std'], fmt='o-', capsize=5)
    plt.title(f'Bootstrap Stability vs k-additivity ({dataset_name}, {representation})')
    plt.xlabel('k-additivity')
    plt.ylabel('Bootstrap Accuracy (mean ± std)')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "bootstrap_stability.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Accuracy vs parameters trade-off
    plt.figure(figsize=(10, 6))
    valid_k = results_df.dropna(subset=['baseline_accuracy']).index
    plt.scatter(results_df.loc[valid_k, 'n_params'], results_df.loc[valid_k, 'baseline_accuracy'], s=100)
    
    # Add k-value annotations only for valid entries
    for k in valid_k:
        plt.annotate(f'k={k}', 
                    (results_df.loc[k, 'n_params'], results_df.loc[k, 'baseline_accuracy']),
                    textcoords="offset points", xytext=(0, 10), ha='center')
    
    plt.title(f'Accuracy vs Model Complexity ({dataset_name}, {representation})')
    plt.xlabel('Number of Parameters (log scale)')
    plt.ylabel('Accuracy')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "accuracy_vs_parameters.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Heatmap showing all metrics across k values - only for valid k values
    valid_results = results_df.dropna(subset=['baseline_accuracy']).copy()
    if len(valid_results) > 0:
        try:
            # Select only numeric columns and make sure they're floating point values
            plot_data = valid_results[['baseline_accuracy', 'noise_0.05', 'noise_0.1', 
                                   'noise_0.2', 'noise_0.3', 'bootstrap_mean']].astype(float)
            
            # Check if data has any NaN values and fill them
            if plot_data.isna().any().any():
                print("Warning: Filling NaN values in heatmap data")
                plot_data = plot_data.fillna(0)
            
            # Rename columns for better display
            plot_data.columns = ['Baseline', 'Noise 0.05', 'Noise 0.10', 
                                 'Noise 0.20', 'Noise 0.30', 'Bootstrap']
            
            plt.figure(figsize=(12, 8))
            # Make sure the data is properly transposed and contains only floats
            heatmap_data = plot_data.T
            sns.heatmap(heatmap_data, annot=True, cmap='viridis', 
                        linewidths=.5, cbar_kws={'label': 'Accuracy'})
            plt.title(f'Performance Metrics vs k-additivity ({dataset_name}, {representation})')
            plt.xlabel('k-additivity')
            plt.savefig(os.path.join(output_dir, "metrics_heatmap.png"), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error creating heatmap: {str(e)}")
    
    # Find optimal k values for different criteria (only among valid k values)
    if len(valid_results) > 0:
        best_k_accuracy = valid_results['baseline_accuracy'].idxmax()
        best_k_noise03 = valid_results['noise_0.3'].idxmax()
        best_k_stability = valid_results['bootstrap_std'].idxmin()
        
        # Create summary file
        with open(os.path.join(output_dir, "summary.txt"), 'w') as f:
            f.write(f"K-ADDITIVITY ANALYSIS SUMMARY FOR {dataset_name} ({representation})\n")
            f.write("="*60 + "\n\n")
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Representation: {representation}\n")
            f.write(f"Number of attributes: {nAttr}\n")
            f.write(f"Number of samples: {nSamp}\n\n")
            
            f.write("OPTIMAL K VALUES:\n")
            f.write(f"- Best k for accuracy: {best_k_accuracy} (accuracy: {valid_results.loc[best_k_accuracy, 'baseline_accuracy']:.4f})\n")
            f.write(f"- Best k for noise robustness: {best_k_noise03} (accuracy at noise 0.3: {valid_results.loc[best_k_noise03, 'noise_0.3']:.4f})\n")
            f.write(f"- Best k for stability: {best_k_stability} (std dev: {valid_results.loc[best_k_stability, 'bootstrap_std']:.4f})\n\n")
            
            f.write("FULL RESULTS TABLE:\n")
            f.write(results_df.to_string())
    else:
        with open(os.path.join(output_dir, "summary.txt"), 'w') as f:
            f.write(f"K-ADDITIVITY ANALYSIS SUMMARY FOR {dataset_name} ({representation})\n")
            f.write("="*60 + "\n\n")
            f.write("No valid results found for any k value.\n")
    
    # Calculate efficiency metrics (performance/complexity tradeoff)
    if len(valid_results) > 0:
        try:
            # Ensure all necessary columns are numeric
            for col in ['baseline_accuracy', 'noise_0.3', 'bootstrap_std', 'n_params']:
                valid_results[col] = pd.to_numeric(valid_results[col], errors='coerce')
            
            # Fill any NaN values that might have appeared
            valid_results = valid_results.fillna({
                'baseline_accuracy': 0.0,
                'noise_0.3': 0.0,
                'bootstrap_std': 1.0,
                'n_params': 10.0
            })
            
            # Create columns for efficiency metrics using NumPy calculations on numeric data
            valid_results['acc_efficiency'] = valid_results['baseline_accuracy'] / np.log10(valid_results['n_params'] + 10)
            valid_results['noise_efficiency'] = valid_results['noise_0.3'] / np.log10(valid_results['n_params'] + 10)
            valid_results['stability_efficiency'] = valid_results['stability_efficiency'] = (1 - valid_results['bootstrap_std']) / np.log10(valid_results['n_params'] + 10)
            
            # Find k values with best efficiency
            best_k_eff_acc = valid_results['acc_efficiency'].idxmax()
            best_k_eff_noise = valid_results['noise_efficiency'].idxmax()
            best_k_eff_stability = valid_results['stability_efficiency'].idxmax()
            
            # Add to summary file
            with open(os.path.join(output_dir, "efficiency_summary.txt"), 'w') as f:
                f.write(f"COMPLEXITY-EFFICIENCY ANALYSIS FOR {dataset_name} ({representation})\n")
                f.write("="*60 + "\n\n")
                f.write("OPTIMAL K VALUES FOR EFFICIENCY (performance/complexity tradeoff):\n")
                f.write(f"- Best k for accuracy efficiency: {best_k_eff_acc} (eff: {valid_results.loc[best_k_eff_acc, 'acc_efficiency']:.4f}, params: {valid_results.loc[best_k_eff_acc, 'n_params']:.0f})\n")
                f.write(f"- Best k for noise robustness efficiency: {best_k_eff_noise} (eff: {valid_results.loc[best_k_eff_noise, 'noise_efficiency']:.4f}, params: {valid_results.loc[best_k_eff_noise, 'n_params']:.0f})\n")
                f.write(f"- Best k for stability efficiency: {best_k_eff_stability} (eff: {valid_results.loc[best_k_eff_stability, 'stability_efficiency']:.4f}, params: {valid_results.loc[best_k_eff_stability, 'n_params']:.0f})\n")
            
            # Create efficiency plot
            plt.figure(figsize=(12, 8))
            plt.plot(valid_results.index, valid_results['acc_efficiency'], 'o-', label='Accuracy Efficiency')
            plt.plot(valid_results.index, valid_results['noise_efficiency'], 's-', label='Noise Robustness Efficiency')
            plt.plot(valid_results.index, valid_results['stability_efficiency'], '^-', label='Stability Efficiency')
            
            plt.title(f'Efficiency Metrics vs k-additivity ({dataset_name}, {representation})')
            plt.xlabel('k-additivity')
            plt.ylabel('Efficiency (Performance/log(Parameters))')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, "efficiency_vs_k.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save efficiency values
            results_df['acc_efficiency'] = np.nan
            results_df['noise_efficiency'] = np.nan
            results_df['stability_efficiency'] = np.nan
            
            # Only fill values for valid k values
            for k in valid_results.index:
                results_df.loc[k, 'acc_efficiency'] = valid_results.loc[k, 'acc_efficiency']
                results_df.loc[k, 'noise_efficiency'] = valid_results.loc[k, 'noise_efficiency']
                results_df.loc[k, 'stability_efficiency'] = valid_results.loc[k, 'stability_efficiency']
                
            # Add most efficient k values to the returned DataFrame for cross-dataset comparison
            results_df.attrs['best_k_eff_acc'] = best_k_eff_acc
            results_df.attrs['best_k_eff_noise'] = best_k_eff_noise
            results_df.attrs['best_k_eff_stability'] = best_k_eff_stability
            
        except Exception as e:
            print(f"Error calculating efficiency metrics: {e}")
    
    print(f"Analysis completed. Results saved to: {output_dir}")
    return results_df

def run_batch_analysis(datasets_list, representation="game", regularization='l2'):
    """
    Run k-additivity analysis on multiple datasets.
    
    Parameters:
    -----------
    datasets_list : list
        List of dataset names to analyze
    representation : str, default="game"
        Choquet representation to use, either "game" or "mobius"
    regularization : str, default='l2'
        Regularization type for logistic regression ('l1', 'l2', 'elasticnet', or 'none')
    """
    # Create regularization string for folder naming
    reg_str = "none" if regularization is None else regularization
    
    # Create main directory with both representation and regularization
    main_dir = f"k_additivity_analysis_{representation}_{reg_str}"
    os.makedirs(main_dir, exist_ok=True)
    
    # Datasets to analyze
    datasets = datasets_list
    
    # Summary data for cross-dataset comparison
    summary_data = []
    
    # Process each dataset
    for dataset in datasets:
        # Create dataset-specific directory
        dataset_dir = os.path.join(main_dir, dataset)
        os.makedirs(dataset_dir, exist_ok=True)
        
        try:
            # Run analysis with our direct implementation
            results = direct_k_additivity_analysis(
                dataset, 
                representation=representation,
                output_dir=dataset_dir,
                regularization=regularization
            )
            
            # Only extract metrics if there are valid results
            valid_results = results.dropna(subset=['baseline_accuracy'])
            if len(valid_results) > 0:
                try:
                    # Extract key metrics for summary with error handling
                    dataset_summary = {'dataset': dataset, 'n_attr': len(results)}
                    
                    # Try to get best k for accuracy
                    try:
                        best_k_acc = valid_results['baseline_accuracy'].idxmax()
                        dataset_summary['best_k_accuracy'] = int(best_k_acc)
                        dataset_summary['max_accuracy'] = float(valid_results.loc[best_k_acc, 'baseline_accuracy'])
                    except Exception as e:
                        print(f"Warning: Could not determine best k for accuracy for {dataset}: {e}")
                        dataset_summary['best_k_accuracy'] = 1
                        dataset_summary['max_accuracy'] = 0.0
                    
                    # Try to get best k for noise robustness
                    try:
                        best_k_noise = valid_results['noise_0.3'].idxmax()
                        dataset_summary['best_k_noise'] = int(best_k_noise)
                        dataset_summary['noise_robustness'] = float(valid_results.loc[best_k_noise, 'noise_0.3'])
                    except Exception as e:
                        print(f"Warning: Could not determine best k for noise robustness for {dataset}: {e}")
                        dataset_summary['best_k_noise'] = 1
                        dataset_summary['noise_robustness'] = 0.0
                    
                    # Try to get best k for stability
                    try:
                        best_k_stability = valid_results['bootstrap_std'].idxmin()
                        dataset_summary['best_k_stability'] = int(best_k_stability)
                        dataset_summary['stability'] = float(valid_results.loc[best_k_stability, 'bootstrap_std'])
                    except Exception as e:
                        print(f"Warning: Could not determine best k for stability for {dataset}: {e}")
                        dataset_summary['best_k_stability'] = 1
                        dataset_summary['stability'] = 1.0  # Higher is worse for stability
                    
                    # Add to summary data
                    summary_data.append(dataset_summary)
                    
                except Exception as e:
                    print(f"Warning: Could not extract summary metrics for {dataset}: {e}")
            else:
                print(f"Warning: No valid results for dataset {dataset}")
            
        except Exception as e:
            print(f"Error processing dataset {dataset}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Create cross-dataset comparison plots
    if summary_data:
        # Convert to DataFrame, ensuring all values are proper types
        summary_df = pd.DataFrame(summary_data)
        
        # Save raw summary data
        summary_df.to_csv(os.path.join(main_dir, f"datasets_summary_{representation}.csv"), index=False)
        
        # Create bar chart comparing optimal k values across datasets
        try:
            plt.figure(figsize=(14, 8))
            datasets = summary_df['dataset'].tolist()
            x = np.arange(len(datasets))
            width = 0.25
            
            # Ensure numeric columns are properly converted for plotting
            k_acc = summary_df['best_k_accuracy'].astype(int).tolist()
            k_noise = summary_df['best_k_noise'].astype(int).tolist()
            k_stability = summary_df['best_k_stability'].astype(int).tolist()
            
            plt.bar(x - width, k_acc, width, label='Best k (Accuracy)')
            plt.bar(x, k_noise, width, label='Best k (Noise Robustness)')
            plt.bar(x + width, k_stability, width, label='Best k (Stability)')
            
            plt.xlabel('Dataset')
            plt.ylabel('Optimal k-additivity')
            plt.title(f'Optimal k-additivity Values by Dataset and Criterion ({representation})')
            plt.xticks(x, datasets, rotation=45)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(main_dir, f"optimal_k_comparison_{representation}.png"), dpi=300)
            plt.close()
        except Exception as e:
            print(f"Error creating k comparison plot: {e}")
        
        # Create performance heatmap
        try:
            # Create a copy to avoid modifying the original
            perf_df = summary_df[['dataset', 'max_accuracy', 'noise_robustness', 'stability']].copy()
            
            # Force conversion to numeric types
            for col in ['max_accuracy', 'noise_robustness', 'stability']:
                perf_df[col] = pd.to_numeric(perf_df[col], errors='coerce')
            
            # Fill NaN values with zeros
            perf_df = perf_df.fillna(0)
            
            # Set dataset as index and rename columns for clarity
            perf_df = perf_df.set_index('dataset')
            perf_df.columns = ['Accuracy', 'Noise Robustness', 'Stability']
            
            # Invert stability so higher is better (for consistent coloring)
            # But only if it contains valid numeric data
            if pd.to_numeric(perf_df['Stability'], errors='coerce').notna().any():
                perf_df['Stability'] = -perf_df['Stability']
            
            # Create the heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(perf_df, annot=True, cmap='YlGnBu', linewidths=.5, fmt='.4f')
            plt.title(f'Performance Metrics by Dataset ({representation})')
            plt.tight_layout()
            plt.savefig(os.path.join(main_dir, f"dataset_performance_comparison_{representation}.png"), dpi=300)
            plt.close()
        except Exception as e:
            print(f"Error creating performance heatmap: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"Cross-dataset comparison completed. Results saved to: {main_dir}")
    else:
        print("No summary data collected, cannot create comparison plots.")
    
    # Create efficiency metrics comparison across datasets
    try:
        # Gather efficiency data
        efficiency_data = []
        for dataset in datasets:
            try:
                # Get the results file
                results_path = os.path.join(main_dir, dataset, "k_comparison_results.csv")
                if os.path.exists(results_path):
                    df = pd.read_csv(results_path, index_col=0)
                    
                    # Calculate efficiency for each dataset and their optimal k values
                    for row in summary_data:
                        if row['dataset'] == dataset:
                            n_attr = row['n_attr']
                            
                            # Get k values
                            best_k_acc = row['best_k_accuracy']
                            best_k_noise = row['best_k_noise']
                            best_k_stability = row['best_k_stability']
                            
                            # Calculate parameters
                            params_acc = nParam_kAdd(best_k_acc, n_attr)
                            params_noise = nParam_kAdd(best_k_noise, n_attr)
                            params_stability = nParam_kAdd(best_k_stability, n_attr)
                            
                            # Calculate efficiency metrics
                            acc_eff = row['max_accuracy'] / np.log10(params_acc + 10)
                            robustness_eff = row['noise_robustness'] / np.log10(params_noise + 10)
                            stability_eff = (1 - row['stability']) / np.log10(params_stability + 10)
                            
                            efficiency_data.append({
                                'dataset': dataset,
                                'accuracy_efficiency': acc_eff,
                                'robustness_efficiency': robustness_eff,
                                'stability_efficiency': stability_eff
                            })
                            break
            except Exception as e:
                print(f"Error processing efficiency for {dataset}: {e}")
        
        if efficiency_data:
            # Create efficiency DataFrame
            eff_df = pd.DataFrame(efficiency_data)
            
            # Save efficiency summary
            eff_df.to_csv(os.path.join(main_dir, f"efficiency_summary_{representation}.csv"), index=False)
            
            # Create bar chart of efficiency metrics across datasets
            plt.figure(figsize=(14, 8))
            datasets = eff_df['dataset'].tolist()
            x = np.arange(len(datasets))
            width = 0.25
            
            plt.bar(x - width, eff_df['accuracy_efficiency'], width, label='Accuracy Efficiency')
            plt.bar(x, eff_df['robustness_efficiency'], width, label='Robustness Efficiency')
            plt.bar(x + width, eff_df['stability_efficiency'], width, label='Stability Efficiency')
            
            plt.xlabel('Dataset')
            plt.ylabel('Efficiency (Performance/log(Parameters))')
            plt.title(f'Performance-Complexity Tradeoff by Dataset ({representation})')
            plt.xticks(x, datasets, rotation=45)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(main_dir, f"efficiency_comparison_{representation}.png"), dpi=300)
            plt.close()
            
            # Create efficiency heatmap
            eff_heatmap = eff_df.set_index('dataset')
            plt.figure(figsize=(10, 8))
            sns.heatmap(eff_heatmap, annot=True, cmap='YlGnBu', linewidths=.5, fmt='.4f')
            plt.title(f'Performance-Complexity Efficiency by Dataset ({representation})')
            plt.tight_layout()
            plt.savefig(os.path.join(main_dir, f"efficiency_heatmap_{representation}.png"), dpi=300)
            plt.close()
            
            print(f"Efficiency metrics analysis completed for {representation}")
    except Exception as e:
        print(f"Error creating efficiency comparison: {e}")
        import traceback
        traceback.print_exc()

def feature_dropout_analysis(dataset_name, representation="game", output_dir=None, test_size=0.3, random_state=0, max_features_to_drop=None, regularization='l2'):
    """
    Perform feature dropout analysis for different k-additivity values.
    For each k, test model performance when dropping features systematically.
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset to analyze
    representation : str, default="game"
        Choquet representation to use, either "game" or "mobius"
    output_dir : str, optional
        Directory to save results
    test_size : float, default=0.3
        Proportion of data to use for testing
    random_state : int, default=0
        Random seed for reproducibility
    max_features_to_drop : int, optional
        Maximum number of features to drop (to limit combinatorial explosion)
    regularization : str, default='l2'
        Regularization type for logistic regression ('l1', 'l2', 'elasticnet', or 'none')
    """
    from itertools import combinations
    
    # Validate representation parameter
    if representation not in ["game", "mobius"]:
        raise ValueError(f"Invalid representation '{representation}'. Use 'game' or 'mobius'.")
        
    print(f"\n{'='*80}\nPerforming feature dropout analysis: {dataset_name} with {representation} representation\n{'='*80}")
    
    # Create output directory - now within the k-additivity structure
    if output_dir is None:
        # Create within the same folder structure as k-additivity analysis
        base_dir = f"k_additivity_analysis_{representation}/{dataset_name}"
        output_dir = os.path.join(base_dir, f"feature_dropout{dataset_name}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and prepare data
    X, y = func_read_data(dataset_name)
    nSamp, nAttr = X.shape
    
    # Limit the maximum features to drop to prevent combinatorial explosion
    if max_features_to_drop is None:
        max_features_to_drop = min(3, nAttr - 1)  # Default: drop at most 3 features or leave at least 1
    else:
        max_features_to_drop = min(max_features_to_drop, nAttr - 1)  # Ensure at least 1 feature remains
    
    print(f"Dataset: {dataset_name}")
    print(f"- Representation: {representation}")
    print(f"- Number of samples: {nSamp}")
    print(f"- Number of attributes: {nAttr}")
    print(f"- Will drop up to {max_features_to_drop} features")
    
    # Split data into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Convert to numpy arrays
    X_train_values = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
    X_test_values = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
    y_train_values = y_train.values if isinstance(y_train, pd.Series) else y_train
    y_test_values = y_test.values if isinstance(y_test, pd.Series) else y_test
    
    # Scale the data
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_values)
    X_test_scaled = scaler.transform(X_test_values)
    
    # Select the appropriate transformation function
    choquet_transform = choquet_k_additive_game if representation == "game" else choquet_k_additive_mobius
    
    # Dictionary to store results for each k value
    all_results = {}
    
    # For each k value
    for k in range(1, nAttr + 1):
        print(f"\nAnalyzing k = {k}/{nAttr} with feature dropout...")
        
        # Only test feature combinations that make sense for this k
        # For k-additivity, we need at least k features
        min_features_needed = k
        max_features_to_drop_k = min(max_features_to_drop, nAttr - min_features_needed)
        
        if max_features_to_drop_k < 0:
            print(f"- Skipping k={k}: requires at least {min_features_needed} features")
            continue
        
        # Generate all feature dropout combinations to test
        all_feature_sets = []
        
        # Start with full feature set
        all_feature_indices = list(range(nAttr))
        all_feature_sets.append(all_feature_indices)
        
        # Then add combinations with dropped features
        for n_drop in range(1, max_features_to_drop_k + 1):
            for dropped_indices in combinations(range(nAttr), n_drop):
                kept_indices = [i for i in range(nAttr) if i not in dropped_indices]
                if len(kept_indices) >= min_features_needed:
                    all_feature_sets.append(kept_indices)
        
        n_combos = len(all_feature_sets)
        print(f"- Testing {n_combos} feature combinations")
        
        # Results for this k value
        k_results = []
        
        # Test each feature combination
        for i, feature_indices in enumerate(all_feature_sets):
            if i % max(1, n_combos // 10) == 0:  # Progress update
                print(f"  Progress: {i}/{n_combos} combinations")
            
            # Create dataset with only selected features
            X_train_subset = X_train_scaled[:, feature_indices]
            X_test_subset = X_test_scaled[:, feature_indices]
            
            try:
                # Apply Choquet transformation
                X_train_choquet = choquet_transform(X_train_subset, k_add=k)
                
                # Train logistic regression on transformed data
                model = LogisticRegression(
                    max_iter=1000, 
                    random_state=random_state,
                    solver="newton-cg" if regularization in ['l2', 'none'] else "saga",
                    penalty=None if regularization == 'none' else regularization
                )
                model.fit(X_train_choquet, y_train_values)
                
                # Transform test data and evaluate
                X_test_choquet = choquet_transform(X_test_subset, k_add=k)
                y_pred = model.predict(X_test_choquet)
                accuracy = accuracy_score(y_test_values, y_pred)
                
                # Store results
                k_results.append({
                    'k_value': k,
                    'feature_combination': str(feature_indices),
                    'features_kept': len(feature_indices),
                    'features_dropped': nAttr - len(feature_indices),
                    'kept_indices': ','.join(map(str, feature_indices)),
                    'accuracy': accuracy,
                    'n_params': nParam_kAdd(k, len(feature_indices))
                })
                
            except Exception as e:
                print(f"  Error with features {feature_indices}: {str(e)}")
        
        # Convert results to DataFrame and sort by accuracy
        if k_results:
            results_df = pd.DataFrame(k_results)
            results_df = results_df.sort_values('accuracy', ascending=False)
            
            # Save results for this k
            all_results[k] = results_df
            
            # Save to CSV
            results_df.to_csv(os.path.join(output_dir, f"dropout_k{k}.csv"), index=False)
            
            # Create visualization of top and bottom performing combinations
            top_n = min(10, len(results_df))
            if top_n > 0:
                # Top combinations
                plt.figure(figsize=(12, 8))
                top_df = results_df.head(top_n)
                plt.bar(range(len(top_df)), top_df['accuracy'])
                plt.xticks(range(len(top_df)), top_df['kept_indices'], rotation=90)
                plt.title(f'Top {top_n} Feature Combinations (k={k}, {representation} representation)')
                plt.xlabel('Kept Features')
                plt.ylabel('Accuracy')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"top_combinations_k{k}_{representation}.png"), dpi=300)
                plt.close()
                
                # Distribution of accuracies
                plt.figure(figsize=(10, 6))
                # Use standard vertical orientation with accuracy on x-axis
                sns.histplot(x=results_df['accuracy'], kde=True, bins=15, color='skyblue')
                # Use vertical reference lines
                plt.axvline(results_df['accuracy'].mean(), color='r', linestyle='--', linewidth=2,
                        label=f'Mean: {results_df["accuracy"].mean():.4f}')
                plt.axvline(results_df['accuracy'].max(), color='g', linestyle='--', linewidth=2,
                        label=f'Max: {results_df["accuracy"].max():.4f}')
                plt.title(f'Distribution of Accuracy Scores for k={k} ({dataset_name}, {representation} repr.)', fontsize=14)
                # Set labels correctly for vertical histogram
                plt.xlabel('Accuracy Value', fontsize=12)
                plt.ylabel('Count of Feature Combinations', fontsize=12)
                plt.grid(True, alpha=0.3)
                # Set x-axis limits for accuracy range
                min_acc = max(0, results_df['accuracy'].min() - 0.05)
                max_acc = min(1, results_df['accuracy'].max() + 0.05)
                plt.xlim(min_acc, max_acc)
                plt.legend(loc='best', frameon=True, framealpha=0.9)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"accuracy_distribution_k{k}_{representation}.png"), dpi=300, bbox_inches='tight')
                plt.close()
                
                # Feature importance based on dropout impact
                if len(results_df) > 1:
                    # Create feature impact visualization
                    # Baseline is the accuracy with all features
                    baseline = results_df[results_df['features_dropped'] == 0]['accuracy'].values[0]
                    feature_impact = np.zeros(nAttr)
                    
                    # For each feature, calculate average impact when it's removed
                    for feature_idx in range(nAttr):
                        # Find combinations where this feature is dropped
                        dropped_combinations = [row for _, row in results_df.iterrows() 
                                             if feature_idx not in map(int, row['kept_indices'].split(','))]
                        
                        if dropped_combinations:
                            # Average accuracy when this feature is dropped
                            avg_acc_without = np.mean([row['accuracy'] for row in dropped_combinations])
                            # Impact is difference from baseline
                            feature_impact[feature_idx] = baseline - avg_acc_without
                    
                    # Plot feature importance
                    plt.figure(figsize=(10, 6))
                    plt.bar(range(nAttr), feature_impact)
                    plt.xticks(range(nAttr), [f'Feature {i}' for i in range(nAttr)])
                    plt.title(f'Feature Impact Analysis (k={k}, {representation} representation)')
                    plt.xlabel('Feature')
                    plt.ylabel('Impact on Accuracy')
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f"feature_impact_k{k}_{representation}.png"), dpi=300)
                    plt.close()
    
    # Create summary across all k values
    if all_results:
        summary_rows = []
        for k, df in all_results.items():
            best_row = df.iloc[0]
            full_features_row = df[df['features_dropped'] == 0].iloc[0] if any(df['features_dropped'] == 0) else None
            
            summary_row = {
                'k_value': k,
                'best_accuracy': best_row['accuracy'],
                'best_feature_set': best_row['kept_indices'],
                'best_n_features': best_row['features_kept'],
            }
            
            if full_features_row is not None:
                summary_row['full_accuracy'] = full_features_row['accuracy']
                summary_row['accuracy_improvement'] = best_row['accuracy'] - full_features_row['accuracy']
            
            summary_rows.append(summary_row)
        
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(os.path.join(output_dir, f"dropout_summary_{representation}.csv"), index=False)
        
        # Summary plot
        plt.figure(figsize=(12, 8))
        plt.plot(summary_df['k_value'], summary_df['best_accuracy'], 'o-', label='Best Feature Subset')
        if 'full_accuracy' in summary_df.columns:
            plt.plot(summary_df['k_value'], summary_df['full_accuracy'], 's--', label='All Features')
        
        plt.title(f'Feature Dropout Analysis Summary ({dataset_name}, {representation} representation)')
        plt.xlabel('k-additivity')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, f"dropout_summary_{representation}.png"), dpi=300)
        plt.close()
        
        # Optimal feature count plot
        plt.figure(figsize=(10, 6))
        plt.bar(summary_df['k_value'], summary_df['best_n_features'])
        plt.title(f'Optimal Number of Features vs k-additivity ({dataset_name}, {representation} representation)')
        plt.xlabel('k-additivity')
        plt.ylabel('Optimal Feature Count')
        plt.savefig(os.path.join(output_dir, f"optimal_feature_count_{representation}.png"), dpi=300)
        plt.close()
    
    print(f"Feature dropout analysis completed. Results saved to: {output_dir}")
    return all_results

if __name__ == "__main__":
    datasets = [        "pairwise_interaction",
        "exponentially_weighted_interaction",
        "extreme_coalition",
        "hierarchical_interaction",
        "high_dimensional_complex",
        "higher_order_5_interaction",
        "higher_order_6_interaction",
        "mixed_higher_order_interaction",
        "mixed_interaction",
        "nested_interaction",
        "nonlinear_transformation",]
    
    # Choose the representation type - can be "game" or "mobius"
    representations = ["game", "mobius"]
    
    # Choose regularization - options: 'l1', 'l2', 'elasticnet', 'none'
    regularizations = [None,'l2']

    run_k_additivity = True
    run_feature_dropout = True
    max_features_to_drop = 2  

    # Loop through both representations and regularizations
    for representation in representations:
        for regularization in regularizations:
            # Create regularization string for folder naming
            reg_str = "none" if regularization is None else regularization
            
            # Create main directory with both representation and regularization
            main_dir = f"k_additivity_analysis_{representation}_{reg_str}"
            os.makedirs(main_dir, exist_ok=True)
            
            print(f"\n{'-'*80}\nAnalyzing with {representation} representation and {reg_str} regularization\n{'-'*80}")
        
            # Run the regular k-additivity analysis
            if run_k_additivity:
                print("\nRunning k-additivity analysis...")
                run_batch_analysis(datasets_list=datasets, representation=representation, regularization=regularization)
            
            # Run feature dropout analysis for each dataset
            if run_feature_dropout:
                print("\nRunning feature dropout analysis...")
                for dataset in datasets:
                    # Create dataset directory if it doesn't exist
                    dataset_dir = os.path.join(main_dir, dataset)
                    os.makedirs(dataset_dir, exist_ok=True)
                    
                    # Run feature dropout with path to store in dataset subdirectory
                    feature_dropout_output_dir = os.path.join(dataset_dir, f"featuredropout{dataset}")
                    
                    feature_dropout_analysis(
                        dataset, 
                        representation=representation,
                        output_dir=feature_dropout_output_dir,
                        max_features_to_drop=max_features_to_drop,
                        regularization=regularization
                    )