import os
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import mod_GenFuzzyRegression as mGFR
from choquet_function import choquet_matrix_kadd_guilherme, choquet_matrix_2add

def refined_choquet_k_additive(X_orig, k_add=2):
    """Refined implementation of k-additive Choquet integral transformation."""
    X_orig = np.asarray(X_orig)
    nSamp, nAttr = X_orig.shape
    
    # Generate all valid coalitions up to size k_add
    all_coalitions = []
    for r in range(1, k_add+1):
        all_coalitions.extend(list(combinations(range(nAttr), r)))
    
    # Calculate number of features in the transformed space
    n_transformed = len(all_coalitions)
    
    # Initialize output matrix
    transformed = np.zeros((nSamp, n_transformed))
    
    # Performance optimization: Pre-compute coalition membership tests
    # This significantly reduces the computational burden of the inner loop
    coalition_members = {}
    for coal_idx, coalition in enumerate(all_coalitions):
        coalition_set = set(coalition)
        coalition_members[coal_idx] = coalition_set
    
    # For each sample
    for i in range(nSamp):
        x = X_orig[i]
        
        # Sort feature indices by their values
        sorted_indices = np.argsort(x)
        sorted_values = x[sorted_indices]
        
        # Add a sentinel value for the first difference
        sorted_values_ext = np.concatenate([[0], sorted_values])
        
        # For each position in the sorted feature list
        for j in range(nAttr):
            # Current feature index and value
            feat_idx = sorted_indices[j]
            # Difference with previous value
            diff = sorted_values_ext[j+1] - sorted_values_ext[j]
            
            # All features from this position onward
            higher_features = set(sorted_indices[j:])
            
            # Find all valid coalitions containing this feature and higher features
            for coal_idx, coalition_set in coalition_members.items():
                # Check if coalition is valid (optimized set operations)
                if feat_idx in coalition_set and coalition_set.issubset(higher_features):
                    transformed[i, coal_idx] += diff
    
    return transformed

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

def generate_coalition_labels(feature_names, max_k):
    """Generate labels for all possible feature coalitions up to size max_k."""
    coalition_labels = []
    for k in range(1, max_k+1):
        for combo in combinations(range(len(feature_names)), k):
            label = ", ".join([feature_names[i] for i in combo])
            coalition_labels.append(label)
    return coalition_labels

def comprehensive_covid_test():
    """Comprehensive test of all implementations on the COVID dataset"""
    print("=== Comprehensive COVID-19 Dataset Test ===\n")
    
    # Load COVID dataset
    X, y = mGFR.func_read_data('dados_covid_sbpo_atual')
    print(f"Dataset: COVID-19, shape: {X.shape}\n")
    
    # Maximum k value (can't exceed number of features)
    nAttr = X.shape[1]
    max_k = nAttr  # Set to number of features (9 for COVID dataset)
    
    # Explain computational complexity and k-value limitation
    print(f"Using maximum k value of {max_k} (number of features in dataset)")
    print("Note: The number of coalitions grows exponentially with k:")
    for k in range(1, nAttr+1):
        coalition_count = sum(comb(nAttr, i) for i in range(1, k+1))
        print(f"  k={k}: {coalition_count} coalitions")
    
    print("\nComparison of implementation sparsity:")
    print("  Ordered implementation: More efficient, produces sparser matrices")
    print("  Unordered implementation: More comprehensive, produces denser matrices")
    print("  Shapley 2-add: Fixed structure optimized for 2-additive models\n")
    
    # Add timing information
    import time
    
    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define implementations to test
    implementations = [
        ("Game-based (Ordered)", lambda X, k: choquet_matrix_kadd_guilherme(X, kadd=k)),
        ("Refined (Unordered)", lambda X, k: refined_choquet_k_additive(X, k_add=k)),
        ("Shapley (2-add)", lambda X, k: choquet_matrix_2add(X))
    ]
    
    # Test with different k values
    k_values = list(range(1, max_k+1))
    results = []
    
    # Store all models for later coefficient plotting
    all_models = {}
    
    # Generate coalition labels for all possible coalitions up to max_k
    all_coalition_labels = generate_coalition_labels(feature_names, max_k)
    
    for name, implementation in implementations:
        print(f"\nTesting {name} implementation:")
        
        if name == "Shapley (2-add)":
            # Shapley is fixed
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
                
                # For Shapley 2-add model, create more interpretable labels
                if name == "Shapley (2-add)":
                    # First nAttr features are individual features
                    # Next ones are interactions
                    feature_col_names = feature_names.copy()
                    for i in range(nAttr):
                        for j in range(i+1, nAttr):
                            feature_col_names.append(f"{feature_names[i]} & {feature_names[j]}")
                            
                    for i in top_indices:
                        if i < len(feature_col_names):
                            print(f"    {feature_col_names[i]}: {coef[i]:.6f}")
                        else:
                            print(f"    Feature {i}: {coef[i]:.6f}")
                else:
                    # For other models, just use feature indices
                    for i in top_indices:
                        print(f"    Feature {i}: {coef[i]:.6f}")
                    
            except Exception as e:
                print(f"  ERROR: {str(e)}")
    
    # Plot coefficients for each model
    for model_key, model_info in all_models.items():
        model = model_info['model']
        k = model_info['k']
        n_features = model_info['n_features']
        
        # Generate appropriate coalition labels
        if "Shapley" in model_key:
            # For Shapley 2-add model (specific structure)
            coalition_labels = feature_names.copy()
            for i in range(nAttr):
                for j in range(i+1, nAttr):
                    coalition_labels.append(f"{feature_names[i]} & {feature_names[j]}")
            
            # Pad with generic labels if needed
            if len(coalition_labels) < n_features:
                coalition_labels.extend([f"Feature_{i}" for i in range(len(coalition_labels), n_features)])
        else:
            # For other models, generate based on k
            coalition_labels = []
            for r in range(1, k+1):
                for combo in combinations(range(nAttr), r):
                    label = ", ".join([feature_names[i] for i in combo])
                    coalition_labels.append(label)
        
        # Check if we have correct number of labels
        if len(coalition_labels) != n_features:
            print(f"Warning: {len(coalition_labels)} labels for {n_features} features in model {model_key}")
            # Fill in missing labels
            if len(coalition_labels) < n_features:
                coalition_labels.extend([f"Feature_{i}" for i in range(len(coalition_labels), n_features)])
            else:
                coalition_labels = coalition_labels[:n_features]
        
        plot_title = f"{model_key} Coefficients"
        filename = f"{model_key.replace(' ', '_').replace('(', '').replace(')', '').lower()}_coefficients.png"
        plot_model_coefficients(model, feature_names, coalition_labels, plot_title, filename)
    
    # Summary table
    print("\n=== Summary of Results ===")
    print(f"{'Implementation':<20} {'k':<3} {'Train Acc':<10} {'Test Acc':<10} {'AUC':<10} {'CV Accuracy':<12} {'Features':<10}")
    print("-" * 90)
    for r in results:
        print(f"{r['Implementation']:<20} {r['k']:<3} {r['Train Accuracy']:<10.4f} {r['Test Accuracy']:<10.4f} {r['AUC']:<10.4f} {r['CV Accuracy']:<8.4f} ± {r['CV Std']:<5.4f} {r['Features']:<10}")
    
    # Visualize results
    plt.figure(figsize=(14, 10))
    
    # Plot test accuracy by k value for each implementation
    plt.subplot(2, 1, 1)
    
    implementations = sorted(set(r['Implementation'] for r in results))
    markers = ['o', 's', '^']
    colors = ['blue', 'red', 'green']
    
    for i, impl in enumerate(implementations):
        impl_results = [r for r in results if r['Implementation'] == impl]
        impl_results.sort(key=lambda x: x['k'])
        
        k_vals = [r['k'] for r in impl_results]
        accuracy = [r['Test Accuracy'] for r in impl_results]
        
        plt.plot(k_vals, accuracy, marker=markers[i], color=colors[i], label=impl, linewidth=2)
    
    plt.xlabel('k value', fontsize=14)
    plt.ylabel('Test Accuracy', fontsize=14)
    plt.title('COVID-19 Dataset: Accuracy by k value', fontsize=16)
    plt.xticks(k_values)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Plot AUC by k value
    plt.subplot(2, 1, 2)
    
    for i, impl in enumerate(implementations):
        impl_results = [r for r in results if r['Implementation'] == impl]
        impl_results.sort(key=lambda x: x['k'])
        
        k_vals = [r['k'] for r in impl_results]
        auc = [r['AUC'] for r in impl_results]
        
        plt.plot(k_vals, auc, marker=markers[i], color=colors[i], label=impl, linewidth=2)
    
    plt.xlabel('k value', fontsize=14)
    plt.ylabel('AUC', fontsize=14)
    plt.title('COVID-19 Dataset: AUC by k value', fontsize=16)
    plt.xticks(k_values)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join("plots", "covid_results_by_k.png"))
    plt.close()
    print("\nPlot saved as 'plots/covid_results_by_k.png'")
    
    # Plot showing number of features vs. performance
    plt.figure(figsize=(14, 7))
    
    # Group by implementation
    for i, impl in enumerate(implementations):
        impl_results = [r for r in results if r['Implementation'] == impl]
        impl_results.sort(key=lambda x: x['Features'])
        
        features = [r['Features'] for r in impl_results]
        accuracy = [r['Test Accuracy'] for r in impl_results]
        auc = [r['AUC'] for r in impl_results]
        
        plt.scatter(features, accuracy, marker=markers[i], color=colors[i], s=100, 
                   label=f"{impl} (Accuracy)")
        plt.scatter(features, auc, marker=markers[i], edgecolors=colors[i], facecolors='none', s=100,
                   label=f"{impl} (AUC)")
        
        # Add k value annotations
        for j, r in enumerate(impl_results):
            plt.annotate(f"k={r['k']}", (features[j], accuracy[j]), 
                        textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.xlabel('Number of Features', fontsize=14)
    plt.ylabel('Performance Metric', fontsize=14)
    plt.title('COVID-19 Dataset: Performance vs. Number of Features', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join("plots", "covid_performance_vs_features.png"))
    plt.close()
    print("Plot saved as 'plots/covid_performance_vs_features.png'")
    
    # Find best performance
    best_result = max(results, key=lambda r: r['Test Accuracy'])
    print(f"\nBest performance: {best_result['Implementation']} with k={best_result['k']}")
    print(f"  Train Accuracy: {best_result['Train Accuracy']:.4f}")
    print(f"  Test Accuracy: {best_result['Test Accuracy']:.4f}, AUC: {best_result['AUC']:.4f}")
    print(f"  Features: {best_result['Features']}")

if __name__ == "__main__":
    comprehensive_covid_test()
