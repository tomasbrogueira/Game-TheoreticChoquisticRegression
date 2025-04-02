"""
Comprehensive comparison of improved Choquet integral implementations
on the COVID-19 dataset. This analyzes how the strict k-additive implementations
perform compared to the original implementations.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import mod_GenFuzzyRegression as mGFR

# Import original implementations for comparison
from choquet_function import choquet_matrix_kadd_guilherme, choquet_matrix_2add
from paper_code.k_add_test import refined_choquet_k_additive

# Import improved implementations
from paper_code.choquet_improved import (
    strict_kadd_choquet,
    improved_refined_choquet,
    improved_shapley_2add,
    improved_kadd_ordered,
    comb
)

def generate_coalition_labels(feature_names, max_k):
    """Generate labels for all possible feature coalitions up to size max_k."""
    coalition_labels = []
    for k in range(1, max_k+1):
        for combo in combinations(range(len(feature_names)), k):
            label = ", ".join([feature_names[i] for i in combo])
            coalition_labels.append(label)
    return coalition_labels

def plot_model_coefficients(model, feature_names, coalition_labels, plot_title, filename, max_features=50):
    """Plot the coefficients of a model."""
    coef = model.coef_[0]
    
    # If we have too many features, just show the top ones by magnitude
    if len(coef) > max_features:
        top_indices = np.argsort(np.abs(coef))[::-1][:max_features]
        coef = coef[top_indices]
        labels = [coalition_labels[i] for i in top_indices]
    else:
        labels = coalition_labels
    
    # Sort by coefficient absolute value
    sorted_idx = np.argsort(np.abs(coef))[::-1]
    sorted_coef = coef[sorted_idx]
    sorted_labels = [labels[i] for i in sorted_idx]
    
    plt.figure(figsize=(14, max(8, len(sorted_coef) * 0.3)))
    bars = plt.barh(range(len(sorted_coef)), sorted_coef, color='blue', edgecolor='black')
    
    # Color negative coefficients red
    for i, bar in enumerate(bars):
        if sorted_coef[i] < 0:
            bar.set_color('red')
    
    plt.yticks(range(len(sorted_coef)), sorted_labels, fontsize=10)
    plt.xlabel('Coefficient Value', fontsize=14)
    plt.title(plot_title, fontsize=16)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Ensure plots directory exists
    os.makedirs("plots", exist_ok=True)
    plt.savefig(os.path.join("plots", filename))
    plt.close()
    print(f"Saved coefficient plot to: plots/{filename}")

def compare_choquet_implementations():
    """Compare the original and improved Choquet implementations on the COVID dataset"""
    print("=== Comparison of Original vs. Improved Choquet Implementations ===\n")
    
    # Load COVID dataset
    X, y = mGFR.func_read_data('dados_covid_sbpo_atual')
    feature_names = ['Feature ' + str(i) for i in range(X.shape[1])]
    print(f"Dataset: COVID-19, shape: {X.shape}\n")
    
    # Maximum k value (can't exceed number of features)
    nAttr = X.shape[1]
    max_k = min(4, nAttr)  # Limit to 4 to keep it manageable
    
    # Print theoretical coalition counts
    print("Number of coalitions grows exponentially with k:")
    for k in range(1, max_k+1):
        coalition_count = sum(comb(nAttr, r) for r in range(1, k+1))
        print(f"  k={k}: {coalition_count} coalitions")
    print("")
    
    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define all implementations to test (original and improved)
    implementations = [
        # Original implementations
        ("Original Ordered", lambda X, k: choquet_matrix_kadd_guilherme(X, kadd=k)),
        ("Original Refined", lambda X, k: refined_choquet_k_additive(X, k_add=k)),
        ("Original Shapley", lambda X, k: choquet_matrix_2add(X)),
        
        # Improved implementations
        ("Strict k-add", lambda X, k: strict_kadd_choquet(X, k_add=k)),
        ("Improved Refined", lambda X, k: improved_refined_choquet(X, k_add=k)),
        ("Improved Ordered", lambda X, k: improved_kadd_ordered(X, k_add=k)),
        ("Improved Shapley", lambda X, k: improved_shapley_2add(X))
    ]
    
    # Test with different k values
    k_values = list(range(1, max_k+1))
    results = []
    
    # Store all models for later coefficient plotting
    all_models = {}
    
    # Generate coalition labels for all possible coalitions up to max_k
    all_coalition_labels = generate_coalition_labels(feature_names, max_k)
    
    # Add timing information
    import time
    
    for name, implementation in implementations:
        print(f"\nTesting {name} implementation:")
        
        if "Shapley" in name:
            # Shapley is fixed at k=2
            k_values_to_test = [2]
        else:
            k_values_to_test = k_values
            
        for k in k_values_to_test:
            print(f"\n  With k={k}:")
            
            # Transform data with timing
            try:
                start_time = time.time()
                X_train_trans = implementation(X_train_scaled, k)
                transform_time = time.time() - start_time
                
                start_time = time.time()
                X_test_trans = implementation(X_test_scaled, k)
                transform_time += time.time() - start_time
                
                # Report matrix statistics and timing
                print(f"  Transformed shape: {X_train_trans.shape}")
                print(f"  Non-zero elements: {np.count_nonzero(X_train_trans)}/{X_train_trans.size} ({100*np.count_nonzero(X_train_trans)/X_train_trans.size:.2f}%)")
                print(f"  Mean: {np.mean(X_train_trans):.6f}, Std: {np.std(X_train_trans):.6f}")
                print(f"  Transform time: {transform_time:.2f} seconds")
                
                # Train model
                model = LogisticRegression(max_iter=10000, penalty=None)
                model.fit(X_train_trans, y_train)
                
                # Store model for coefficient plotting
                model_key = f"{name}_k{k}"
                all_models[model_key] = {
                    'model': model,
                    'n_features': X_train_trans.shape[1],
                    'k': k
                }
                
                # Test performance
                train_preds = model.predict(X_train_trans)
                test_preds = model.predict(X_test_trans)
                train_proba = model.predict_proba(X_train_trans)[:, 1]
                test_proba = model.predict_proba(X_test_trans)[:, 1]
                
                train_accuracy = accuracy_score(y_train, train_preds)
                test_accuracy = accuracy_score(y_test, test_preds)
                test_auc = roc_auc_score(y_test, test_proba)
                
                # Cross-validation
                cv_scores = cross_val_score(LogisticRegression(max_iter=10000, penalty=None), 
                                          X_train_trans, y_train, cv=5)
                
                # Store results
                results.append({
                    'Implementation': name,
                    'Type': 'Original' if 'Original' in name else 'Improved',
                    'k': k,
                    'Train Accuracy': train_accuracy,
                    'Test Accuracy': test_accuracy,
                    'AUC': test_auc,
                    'CV Accuracy': np.mean(cv_scores),
                    'CV Std': np.std(cv_scores),
                    'Features': X_train_trans.shape[1],
                    'Sparsity': np.count_nonzero(X_train_trans)/X_train_trans.size
                })
                
                print(f"  Train Accuracy: {train_accuracy:.4f}")
                print(f"  Test Accuracy: {test_accuracy:.4f}, AUC: {test_auc:.4f}")
                print(f"  Cross-val Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
                
                # Feature importance
                coef = model.coef_[0]
                top_indices = np.argsort(np.abs(coef))[::-1][:5]
                print("  Top 5 coefficients by magnitude:")
                for i in top_indices:
                    print(f"    Feature {i}: {coef[i]:.6f}")
                    
            except Exception as e:
                print(f"  ERROR: {str(e)}")
    
    # Summary table by implementation and k value
    print("\n=== Summary of Results ===")
    print(f"{'Implementation':<20} {'Type':<10} {'k':<3} {'Train Acc':<10} {'Test Acc':<10} {'AUC':<10} {'CV Accuracy':<12} {'Features':<10}")
    print("-" * 100)
    for r in results:
        print(f"{r['Implementation']:<20} {r['Type']:<10} {r['k']:<3} {r['Train Accuracy']:<10.4f} {r['Test Accuracy']:<10.4f} {r['AUC']:<10.4f} {r['CV Accuracy']:<8.4f} ± {r['CV Std']:<5.4f} {r['Features']:<10}")
    
    # Group results by original vs improved
    orig_results = [r for r in results if r['Type'] == 'Original']
    impr_results = [r for r in results if r['Type'] == 'Improved']
    
    # Compare average performance
    print("\n=== Average Performance Comparison ===")
    print(f"{'Group':<15} {'Test Accuracy':<15} {'AUC':<15}")
    print("-" * 50)
    print(f"{'Original':<15} {np.mean([r['Test Accuracy'] for r in orig_results]):<15.4f} {np.mean([r['AUC'] for r in orig_results]):<15.4f}")
    print(f"{'Improved':<15} {np.mean([r['Test Accuracy'] for r in impr_results]):<15.4f} {np.mean([r['AUC'] for r in impr_results]):<15.4f}")
    
    # Create visualization comparing original vs improved implementations
    plt.figure(figsize=(14, 10))
    
    # Group implementations by type and k value
    orig_by_k = {}
    impr_by_k = {}
    
    for r in results:
        if r['Type'] == 'Original':
            if r['k'] not in orig_by_k:
                orig_by_k[r['k']] = []
            orig_by_k[r['k']].append(r)
        else:
            if r['k'] not in impr_by_k:
                impr_by_k[r['k']] = []
            impr_by_k[r['k']].append(r)
    
    # Plot accuracy comparison
    plt.subplot(2, 1, 1)
    
    bar_width = 0.35
    x = np.arange(len(k_values))
    
    # Calculate average accuracies by k value
    orig_accs = []
    impr_accs = []
    
    for k in k_values:
        if k in orig_by_k:
            orig_accs.append(np.mean([r['Test Accuracy'] for r in orig_by_k[k]]))
        else:
            orig_accs.append(0)
            
        if k in impr_by_k:
            impr_accs.append(np.mean([r['Test Accuracy'] for r in impr_by_k[k]]))
        else:
            impr_accs.append(0)
    
    plt.bar(x - bar_width/2, orig_accs, bar_width, label='Original Implementations')
    plt.bar(x + bar_width/2, impr_accs, bar_width, label='Improved Implementations')
    
    plt.xlabel('k value', fontsize=14)
    plt.ylabel('Average Test Accuracy', fontsize=14)
    plt.title('Original vs. Improved Implementations: Accuracy by k-value', fontsize=16)
    plt.xticks(x, k_values)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    
    # Plot AUC comparison
    plt.subplot(2, 1, 2)
    
    # Calculate average AUCs by k value
    orig_aucs = []
    impr_aucs = []
    
    for k in k_values:
        if k in orig_by_k:
            orig_aucs.append(np.mean([r['AUC'] for r in orig_by_k[k]]))
        else:
            orig_aucs.append(0)
            
        if k in impr_by_k:
            impr_aucs.append(np.mean([r['AUC'] for r in impr_by_k[k]]))
        else:
            impr_aucs.append(0)
    
    plt.bar(x - bar_width/2, orig_aucs, bar_width, label='Original Implementations')
    plt.bar(x + bar_width/2, impr_aucs, bar_width, label='Improved Implementations')
    
    plt.xlabel('k value', fontsize=14)
    plt.ylabel('Average AUC', fontsize=14)
    plt.title('Original vs. Improved Implementations: AUC by k-value', fontsize=16)
    plt.xticks(x, k_values)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join("plots", "original_vs_improved_by_k.png"))
    plt.close()
    print("\nPlot saved as 'plots/original_vs_improved_by_k.png'")
    
    # Plot comparison by implementation type
    plt.figure(figsize=(16, 8))
    
    # Group by implementation name
    impl_names = sorted(set([r['Implementation'] for r in results]))
    impl_accs = []
    impl_aucs = []
    impl_types = []
    
    for impl in impl_names:
        impl_results = [r for r in results if r['Implementation'] == impl]
        impl_accs.append(np.mean([r['Test Accuracy'] for r in impl_results]))
        impl_aucs.append(np.mean([r['AUC'] for r in impl_results]))
        impl_types.append('Original' if 'Original' in impl else 'Improved')
    
    # Sort by implementation type then accuracy
    sort_idx = np.lexsort((impl_accs, [1 if t == 'Original' else 0 for t in impl_types]))
    impl_names = [impl_names[i] for i in sort_idx]
    impl_accs = [impl_accs[i] for i in sort_idx]
    impl_aucs = [impl_aucs[i] for i in sort_idx]
    impl_types = [impl_types[i] for i in sort_idx]
    
    # Set colors by implementation type
    colors = ['#1f77b4' if t == 'Original' else '#ff7f0e' for t in impl_types]
    
    plt.bar(np.arange(len(impl_names)), impl_accs, color=colors, edgecolor='black')
    
    # Add AUC as line
    plt.plot(np.arange(len(impl_names)), impl_aucs, 'ro-', linewidth=2, markersize=8, label='AUC')
    
    plt.xlabel('Implementation', fontsize=14)
    plt.ylabel('Performance', fontsize=14)
    plt.title('Performance by Implementation Type', fontsize=16)
    plt.xticks(np.arange(len(impl_names)), impl_names, rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1f77b4', edgecolor='black', label='Original'),
        Patch(facecolor='#ff7f0e', edgecolor='black', label='Improved'),
        plt.Line2D([0], [0], color='red', marker='o', linestyle='-', markersize=8, label='AUC')
    ]
    plt.legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.savefig(os.path.join("plots", "performance_by_implementation.png"))
    plt.close()
    print("Plot saved as 'plots/performance_by_implementation.png'")
    
    # Find best overall performance
    best_result = max(results, key=lambda r: r['Test Accuracy'])
    print(f"\nBest overall performance: {best_result['Implementation']} with k={best_result['k']}")
    print(f"  Train Accuracy: {best_result['Train Accuracy']:.4f}")
    print(f"  Test Accuracy: {best_result['Test Accuracy']:.4f}, AUC: {best_result['AUC']:.4f}")
    print(f"  Features: {best_result['Features']}")
    
    # Find best improved implementation
    best_improved = max(impr_results, key=lambda r: r['Test Accuracy'])
    print(f"\nBest improved implementation: {best_improved['Implementation']} with k={best_improved['k']}")
    print(f"  Train Accuracy: {best_improved['Train Accuracy']:.4f}")
    print(f"  Test Accuracy: {best_improved['Test Accuracy']:.4f}, AUC: {best_improved['AUC']:.4f}")
    print(f"  Features: {best_improved['Features']}")
    
    # Optional: Plot model coefficients for the best models
    best_original_model_key = f"{best_result['Implementation']}_k{best_result['k']}"
    best_improved_model_key = f"{best_improved['Implementation']}_k{best_improved['k']}"
    
    # Plot coefficients for the best models if they're in our stored models
    for model_key in [best_original_model_key, best_improved_model_key]:
        if model_key in all_models:
            model_info = all_models[model_key]
            model = model_info['model']
            k = model_info['k']
            n_features = model_info['n_features']
            
            # Generate coalition labels based on k
            coalition_labels = generate_coalition_labels(feature_names, k)
            
            # Ensure we have the right number of labels
            if len(coalition_labels) != n_features:
                # Pad with generic labels if needed
                if len(coalition_labels) < n_features:
                    coalition_labels.extend([f"Feature_{i}" for i in range(len(coalition_labels), n_features)])
                else:
                    coalition_labels = coalition_labels[:n_features]
            
            plot_title = f"Best {model_key.split('_')[0]} Model: {model_key}"
            filename = f"best_{model_key.lower().replace(' ', '_').replace('(', '').replace(')', '')}_coefficients.png"
            plot_model_coefficients(model, feature_names, coalition_labels, plot_title, filename)

if __name__ == "__main__":
    compare_choquet_implementations()
