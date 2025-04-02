import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from math import comb
from itertools import combinations
import time
import os
from choquet_function import choquet_matrix_kadd_guilherme, choquet_matrix_2add
from paper_code.k_add_test import refined_choquet_k_additive

def analyze_coalition_structure(X, k_values=[1, 2, 3]):
    """Analyze which coalitions are active in each implementation for different input patterns"""
    n_features = X.shape[1]
    
    print("\n=== Analyzing Coalition Structures ===")
    print(f"Testing with {n_features} features")
    
    # Create special test vectors with one feature active at a time
    test_vectors = np.zeros((n_features + 3, n_features))
    
    # Single feature patterns (one-hot vectors)
    for i in range(n_features):
        test_vectors[i, i] = 1.0
    
    # Add a few multi-feature patterns
    test_vectors[n_features, :2] = 1.0  # First two features
    test_vectors[n_features+1, :3] = 1.0  # First three features
    test_vectors[n_features+2, :] = 1.0  # All features
    
    implementations = [
        ("Game-based", lambda X, k: choquet_matrix_kadd_guilherme(X, kadd=k)),
        ("Refined", lambda X, k: refined_choquet_k_additive(X, k_add=k)),
        ("Shapley", lambda X, k: choquet_matrix_2add(X))
    ]
    
    pattern_names = []
    for i in range(n_features):
        pattern_names.append(f"Feature {i} only")
    pattern_names.append("Features 0 & 1")
    pattern_names.append("Features 0, 1 & 2")
    pattern_names.append("All features")
    
    # For each implementation and k value
    for name, implementation in implementations:
        print(f"\n{name} Implementation:")
        
        if name == "Shapley":
            k_to_test = [2]  # Shapley is fixed at k=2
        else:
            k_to_test = k_values
            
        for k in k_to_test:
            print(f"  k = {k}:")
            
            try:
                # Get transformed vectors
                transformed = implementation(test_vectors, k)
                
                # For each input pattern
                for i, pattern in enumerate(test_vectors):
                    non_zero = np.where(transformed[i] != 0)[0]
                    
                    # Only show results with at least one non-zero value
                    if len(non_zero) > 0:
                        print(f"    {pattern_names[i]}:")
                        for idx in non_zero:
                            print(f"      Column {idx}: {transformed[i, idx]:.4f}")
                        
                # Create heatmap visualization
                plt.figure(figsize=(12, 8))
                plt.imshow(transformed, aspect='auto', cmap='viridis')
                plt.colorbar(label='Value')
                plt.title(f"{name} (k={k}): Activated Coalitions by Pattern")
                plt.xlabel("Coalition Index")
                plt.ylabel("Input Pattern")
                plt.yticks(range(len(pattern_names)), pattern_names)
                
                # Ensure plots directory exists
                os.makedirs("plots", exist_ok=True)
                plt.savefig(f"plots/{name.lower().replace('-', '_').replace(' ', '_')}_k{k}_coalitions.png")
                plt.close()
                
            except Exception as e:
                print(f"    ERROR: {str(e)}")

def test_implementation_properties(k_values=[1, 2, 3, 4]):
    """Test implementation properties with controlled test cases for all three implementations"""
    print("\n=== Testing Implementation Properties ===")
    
    # Create test cases with specific properties
    test_cases = [
        {"name": "Equal values", "data": np.ones((1, 5))},
        {"name": "Ascending values", "data": np.array([[0.1, 0.3, 0.5, 0.7, 0.9]])},
        {"name": "Descending values", "data": np.array([[0.9, 0.7, 0.5, 0.3, 0.1]])},
        {"name": "Binary pattern 1", "data": np.array([[1, 1, 0, 0, 0]])},
        {"name": "Binary pattern 2", "data": np.array([[0, 0, 1, 1, 1]])},
        {"name": "Binary pattern 3", "data": np.array([[1, 0, 1, 0, 1]])}
    ]
    
    implementations = [
        ("Game-based", lambda X, k: choquet_matrix_kadd_guilherme(X, kadd=k)),
        ("Refined", lambda X, k: refined_choquet_k_additive(X, k_add=k)),
        ("Shapley", lambda X, k: choquet_matrix_2add(X))
    ]
    
    for case in test_cases:
        print(f"\nTest case: {case['name']}")
        print(f"Input: {case['data']}")
        
        for name, implementation in implementations:
            if name == "Shapley":
                k_to_test = [2]  # Shapley is fixed at k=2
            else:
                k_to_test = k_values
                
            for k in k_to_test:
                try:
                    start_time = time.time()
                    result = implementation(case['data'], k)
                    elapsed = time.time() - start_time
                    
                    print(f"  {name} (k={k}):")
                    print(f"    Shape: {result.shape}")
                    print(f"    Non-zeros: {np.count_nonzero(result)}/{result.size} ({np.count_nonzero(result)/result.size:.2%})")
                    print(f"    Processing time: {elapsed:.6f} sec")
                    
                    # Show non-zero outputs
                    if result.size > 0:
                        non_zero_idx = np.where(result[0] != 0)[0]
                        if len(non_zero_idx) > 0:
                            print(f"    Non-zero outputs:")
                            for idx in non_zero_idx[:5]:  # Show at most 5
                                print(f"      Column {idx}: {result[0, idx]:.4f}")
                            if len(non_zero_idx) > 5:
                                print(f"      ...and {len(non_zero_idx) - 5} more")
                        else:
                            print("    All outputs are zero")
                except Exception as e:
                    print(f"  {name} (k={k}): ERROR: {str(e)}")

def analyze_k_effect_on_performance(max_k=4):
    """Analyze how k affects model performance with different implementations"""
    print("\n=== Analyzing Effect of k on Model Performance ===")
    
    # Create a dataset with interactions of different complexity
    X, y = make_classification(n_samples=500, n_features=5, n_informative=5, 
                              n_redundant=0, random_state=42)
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))  # Scale to [0,1]
    
    # Add 2-way interaction
    interact_2way = (X[:, 0] > 0.7) & (X[:, 1] > 0.7)
    y[interact_2way] = 1 - y[interact_2way]
    
    # Add 3-way interaction
    interact_3way = (X[:, 2] > 0.6) & (X[:, 3] > 0.6) & (X[:, 4] > 0.6)
    y[interact_3way] = 1 - y[interact_3way]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Define implementations to test
    implementations = [
        ("Game-based", lambda X, k: choquet_matrix_kadd_guilherme(X, kadd=k)),
        ("Refined", lambda X, k: refined_choquet_k_additive(X, k_add=k)),
        ("Shapley", lambda X, k: choquet_matrix_2add(X))
    ]
    
    k_values = list(range(1, max_k + 1))
    results = []
    
    for name, implementation in implementations:
        print(f"\nTesting {name} implementation:")
        
        if name == "Shapley":
            k_to_test = [2]  # Shapley is fixed at k=2
        else:
            k_to_test = k_values
            
        for k in k_to_test:
            try:
                # Transform data
                X_train_trans = implementation(X_train, k)
                X_test_trans = implementation(X_test, k)
                
                # Get matrix properties
                sparsity = np.count_nonzero(X_train_trans) / X_train_trans.size
                
                # Train model
                model = LogisticRegression(max_iter=1000, random_state=42)
                model.fit(X_train_trans, y_train)
                
                # Test performance
                train_acc = model.score(X_train_trans, y_train)
                test_acc = model.score(X_test_trans, y_test)
                
                # Get coefficient properties
                coef = model.coef_[0]
                active_coef = np.count_nonzero(coef)
                
                results.append({
                    'Implementation': name,
                    'k': k,
                    'Features': X_train_trans.shape[1],
                    'Sparsity': sparsity,
                    'Train Accuracy': train_acc,
                    'Test Accuracy': test_acc,
                    'Active Coefficients': active_coef,
                    'Coefficient Efficiency': active_coef / X_train_trans.shape[1]
                })
                
                # Analyze active coefficients
                top_coef_idx = np.argsort(np.abs(coef))[::-1][:5]
                
                print(f"  k={k}:")
                print(f"    Features: {X_train_trans.shape[1]}")
                print(f"    Matrix sparsity: {sparsity:.2%}")
                print(f"    Test accuracy: {test_acc:.4f}")
                print(f"    Active coefficients: {active_coef}/{len(coef)} ({active_coef/len(coef):.2%})")
                print(f"    Top 5 coefficients:")
                for idx in top_coef_idx:
                    print(f"      Column {idx}: {coef[idx]:.4f}")
                
            except Exception as e:
                print(f"  ERROR with k={k}: {str(e)}")
    
    # Plot results
    plt.figure(figsize=(12, 10))
    
    # Plot 1: Accuracy vs k
    plt.subplot(2, 1, 1)
    
    for name in set(r['Implementation'] for r in results):
        name_results = [r for r in results if r['Implementation'] == name]
        name_results.sort(key=lambda x: x['k'])
        
        k_vals = [r['k'] for r in name_results]
        accuracy = [r['Test Accuracy'] for r in name_results]
        
        plt.plot(k_vals, accuracy, 'o-', label=name)
    
    plt.xlabel('k value')
    plt.ylabel('Test Accuracy')
    plt.title('Effect of k on Model Accuracy')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Plot 2: Accuracy vs Features
    plt.subplot(2, 1, 2)
    
    for name in set(r['Implementation'] for r in results):
        name_results = [r for r in results if r['Implementation'] == name]
        
        features = [r['Features'] for r in name_results]
        accuracy = [r['Test Accuracy'] for r in name_results]
        
        plt.plot(features, accuracy, 'o-', label=name)
        
        # Annotate points with k value
        for r in name_results:
            plt.annotate(f"k={r['k']}", 
                        (r['Features'], r['Test Accuracy']),
                        textcoords="offset points", 
                        xytext=(0,10), 
                        ha='center')
    
    plt.xlabel('Number of Features')
    plt.ylabel('Test Accuracy')
    plt.title('Accuracy vs Model Complexity')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/k_additive_performance.png")
    plt.close()
    
    print("\nPerformance analysis plot saved to plots/k_additive_performance.png")

if __name__ == "__main__":
    # Create plots directory
    os.makedirs("plots", exist_ok=True)
    
    # Test with controlled patterns to analyze coalition structures
    analyze_coalition_structure(np.zeros((1, 5)), k_values=[1, 2, 3, 4])
    
    # Test implementation properties with specific test cases
    test_implementation_properties(k_values=[1, 2, 3, 4])
    
    # Analyze effect of k on model performance
    analyze_k_effect_on_performance(max_k=4)
