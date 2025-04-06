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
            higher_features = sorted_indices[j:]
            
            # Find all valid coalitions containing this feature and higher features
            for coal_idx, coalition in enumerate(all_coalitions):
                # Check if coalition is valid
                if feat_idx in coalition and all(f in higher_features for f in coalition):
                    transformed[i, coal_idx] += diff
    
    return transformed

def comprehensive_banknotes_test():
    """Comprehensive test of all implementations on the banknotes dataset"""
    print("=== Comprehensive Banknotes Dataset Test ===\n")
    
    # Load banknotes dataset
    X, y = mGFR.func_read_data('dados_covid_sbpo_atual')
    print(f"Dataset: Banknotes, shape: {X.shape}\n")
    
    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define implementations to test
    implementations = [
        ("Original Game", lambda X, k: choquet_matrix_kadd_guilherme(X, kadd=k)),
        ("Refined Game", lambda X, k: refined_choquet_k_additive(X, k_add=k)),
        ("Shapley", lambda X, k: choquet_matrix_2add(X))
    ]
    
    # Test with different k values
    k_values = [1, 2, 3, 4]
    results = []
    
    for name, implementation in implementations:
        print(f"\nTesting {name} implementation:")
        
        if name == "Shapley":
            # Shapley is fixed at k=2
            k_values_to_test = [2]
        else:
            k_values_to_test = k_values
            
        for k in k_values_to_test:
            print(f"\n  With k={k}:")
            
            # Transform data
            try:
                X_train_trans = implementation(X_train_scaled, k)
                X_test_trans = implementation(X_test_scaled, k)
                
                # Report matrix statistics
                print(f"  Transformed shape: {X_train_trans.shape}")
                print(f"  Sparsity: {np.count_nonzero(X_train_trans)}/{X_train_trans.size} non-zero elements")
                print(f"  Mean: {np.mean(X_train_trans):.6f}, Std: {np.std(X_train_trans):.6f}")
                
                # Train model
                model = LogisticRegression(max_iter=1000)
                model.fit(X_train_trans, y_train)
                
                # Test performance
                preds = model.predict(X_test_trans)
                proba = model.predict_proba(X_test_trans)[:, 1]
                
                accuracy = accuracy_score(y_test, preds)
                auc = roc_auc_score(y_test, proba)
                
                # Cross-validation
                cv_scores = cross_val_score(LogisticRegression(max_iter=1000), 
                                           X_train_trans, y_train, cv=5)
                
                # Store results
                results.append({
                    'Implementation': name,
                    'k': k,
                    'Accuracy': accuracy,
                    'AUC': auc,
                    'CV Accuracy': np.mean(cv_scores),
                    'CV Std': np.std(cv_scores),
                    'Features': X_train_trans.shape[1],
                    'Sparsity': np.count_nonzero(X_train_trans)/X_train_trans.size
                })
                
                print(f"  Test Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
                print(f"  Cross-val Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
                
                # Feature importance
                coef = model.coef_[0]
                top_indices = np.argsort(np.abs(coef))[::-1][:5]
                print("  Top 5 coefficients:")
                for i in top_indices:
                    print(f"    Feature {i}: {coef[i]:.6f}")
                    
            except Exception as e:
                print(f"  ERROR: {str(e)}")
    
    # Summary table
    print("\n=== Summary of Results ===")
    print(f"{'Implementation':<15} {'k':<3} {'Accuracy':<10} {'AUC':<10} {'CV Accuracy':<12} {'Features':<10} {'Sparsity':<10}")
    print("-" * 80)
    for r in results:
        print(f"{r['Implementation']:<15} {r['k']:<3} {r['Accuracy']:<10.4f} {r['AUC']:<10.4f} {r['CV Accuracy']:<10.4f} ± {r['CV Std']:<5.4f} {r['Features']:<10} {r['Sparsity']:<10.4f}")
    
    # Visualize results
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    
    # Group by implementation and k value
    implementations = sorted(set(r['Implementation'] for r in results))
    k_values = sorted(set(r['k'] for r in results))
    
    # Create a bar chart for accuracy
    x = np.arange(len(implementations))
    width = 0.2
    
    for i, k in enumerate(k_values):
        accuracies = [next((r['Accuracy'] for r in results if r['Implementation'] == impl and r['k'] == k), 0) 
                     for impl in implementations]
        
        plt.bar(x + (i-1)*width, accuracies, width, label=f'k={k}')
    
    plt.xlabel('Implementation')
    plt.ylabel('Accuracy')
    plt.title('Banknotes Dataset: Accuracy Comparison')
    plt.xticks(x, implementations)
    plt.ylim(0.8, 1.0)  # Focus on relevant range
    plt.legend()
    plt.grid(axis='y')
    
    # AUC comparison
    plt.subplot(2, 1, 2)
    
    for i, k in enumerate(k_values):
        aucs = [next((r['AUC'] for r in results if r['Implementation'] == impl and r['k'] == k), 0) 
               for impl in implementations]
        
        plt.bar(x + (i-1)*width, aucs, width, label=f'k={k}')
    
    plt.xlabel('Implementation')
    plt.ylabel('AUC')
    plt.title('Banknotes Dataset: AUC Comparison')
    plt.xticks(x, implementations)
    plt.ylim(0.8, 1.0)  # Focus on relevant range
    plt.legend()
    plt.grid(axis='y')
    
    plt.tight_layout()
    plt.savefig('banknotes_comprehensive_test.png')
    print("\nPlot saved as 'banknotes_comprehensive_test.png'")
    
    # Find best performance
    best_result = max(results, key=lambda r: r['Accuracy'])
    print(f"\nBest performance: {best_result['Implementation']} with k={best_result['k']}")
    print(f"  Accuracy: {best_result['Accuracy']:.4f}, AUC: {best_result['AUC']:.4f}")

if __name__ == "__main__":
    comprehensive_banknotes_test()